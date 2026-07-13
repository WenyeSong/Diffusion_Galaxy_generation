"""
ResCNN morphology predictor — CSD3 GPU training script.

Usage (via SLURM, see submit_rescnn.sh):
    python train_csd3.py

Outputs written to the same directory as this script:
    checkpoints/epoch_XX.pt   every 2 epochs
    checkpoints/best.pt       best val-loss checkpoint
    loss_curve.png
"""

import os, sys, json, time, glob
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")   # no display on HPC
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MorphCNN

# ── paths ─────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/rds/user/ws452/hpc-work/lizarraga_2024/data"

HDF5_FILES = {
    "train": os.path.join(DATA_DIR, "5x64x64_training_with_morphology.hdf5"),
    "val"  : os.path.join(DATA_DIR, "5x64x64_validation_with_morphology.hdf5"),
}

CKPT_DIR = os.path.join(HERE, "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── hyperparameters ────────────────────────────────────────────────────────
TARGET_COLS = ["g_ellipticity", "g_major_axis", "g_sersic_index", "g_isophotal_area"]
EPOCHS      = 40
BATCH_SIZE  = 512          # larger batch on GPU
LR          = 3e-4
NUM_WORKERS = 4
CKPT_EVERY  = 2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ── normaliser ────────────────────────────────────────────────────────────
with open(os.path.join(HERE, "..", "normaliser.json")) as fp:
    NORM = json.load(fp)

def normalise_labels(vals):
    out = []
    for i, col in enumerate(TARGET_COLS):
        v = vals[:, i].astype(np.float32)
        t = NORM[col]["transform"]
        if t == "log1p": v = np.log1p(v)
        elif t == "log":  v = np.log(v)
        v = (v - NORM[col]["mean"]) / NORM[col]["std"]
        out.append(v)
    return np.stack(out, axis=1)

def denormalise(arr):
    out = []
    for i, col in enumerate(TARGET_COLS):
        v = arr[:, i].copy()
        v = v * NORM[col]["std"] + NORM[col]["mean"]
        t = NORM[col]["transform"]
        if t == "log1p": v = np.expm1(v)
        elif t == "log":  v = np.exp(v)
        out.append(v)
    return np.stack(out, axis=1)


# ── image stats ────────────────────────────────────────────────────────────
STATS_FILE = os.path.join(HERE, "..", "image_stats.json")

def load_split(hdf5_path, indices, desc=""):
    n = len(indices)
    imgs   = np.empty((n, 5, 64, 64), dtype=np.float32)
    labels = np.empty((n, 4),          dtype=np.float32)
    print(f"Loading {desc} ({n:,} samples)...", flush=True)
    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        for s in range(0, n, 2000):
            sl = slice(s, min(s + 2000, n))
            imgs[sl]   = f["image"][indices[sl]]
            for j, col in enumerate(TARGET_COLS):
                labels[sl, j] = f[col][indices[sl]]
    print(f"  done in {time.time()-t0:.0f}s  ({imgs.nbytes/1e9:.2f} GB)", flush=True)
    return imgs, labels


# ── load data ──────────────────────────────────────────────────────────────
train_idx = np.load(os.path.join(HERE, "..", "data/indices_train.npy"))
val_idx   = np.load(os.path.join(HERE, "..", "data/indices_val.npy"))

train_imgs, train_lbls = load_split(HDF5_FILES["train"], train_idx, "train")
val_imgs,   val_lbls   = load_split(HDF5_FILES["val"],   val_idx,   "val")

train_imgs = np.arcsinh(train_imgs)
val_imgs   = np.arcsinh(val_imgs)

if os.path.exists(STATS_FILE):
    with open(STATS_FILE) as fp: img_stats = json.load(fp)
    print("Loaded image_stats.json")
else:
    print("Computing image stats from training set...")
    stats = {"mean": train_imgs.mean(axis=(0,2,3)).tolist(),
             "std" : train_imgs.std(axis=(0,2,3)).tolist()}
    with open(STATS_FILE, "w") as fp: json.dump(stats, fp, indent=2)
    img_stats = stats
    print("Saved image_stats.json")

IMG_MEAN = np.array(img_stats["mean"], dtype=np.float32).reshape(1, 5, 1, 1)
IMG_STD  = np.array(img_stats["std"],  dtype=np.float32).reshape(1, 5, 1, 1)

train_imgs = (train_imgs - IMG_MEAN) / IMG_STD
val_imgs   = (val_imgs   - IMG_MEAN) / IMG_STD

train_lbls = normalise_labels(train_lbls)
val_lbls   = normalise_labels(val_lbls)

train_ds = TensorDataset(torch.from_numpy(train_imgs), torch.from_numpy(train_lbls))
val_ds   = TensorDataset(torch.from_numpy(val_imgs),   torch.from_numpy(val_lbls))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)


# ── model ─────────────────────────────────────────────────────────────────
model     = MorphCNN(n_outputs=4).to(DEVICE)
criterion = nn.HuberLoss(delta=1.0)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

# ── auto-resume ────────────────────────────────────────────────────────────
existing = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch_*.pt")))
train_losses, val_losses = [], []
start_epoch = 1
best_val    = float("inf")

if existing:
    ckpt = torch.load(existing[-1], map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    scheduler.load_state_dict(ckpt["scheduler"])
    train_losses = ckpt.get("train_losses", [])
    val_losses   = ckpt.get("val_losses",   [])
    start_epoch  = ckpt["epoch"] + 1
    best_val     = min(val_losses)
    print(f"Resumed from {existing[-1]}  (epoch {ckpt['epoch']})", flush=True)
else:
    print(f"Starting from scratch  ({sum(p.numel() for p in model.parameters()):,} params)",
          flush=True)


# ── training loop ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimiser, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            pred = model(imgs)
            loss = criterion(pred, labels)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * imgs.size(0)
            n     += imgs.size(0)
    return total / n


for epoch in range(start_epoch, EPOCHS + 1):
    t0      = time.time()
    tr_loss = run_epoch(model, train_loader, criterion, optimiser, train=True)
    va_loss = run_epoch(model, val_loader,   criterion, optimiser, train=False)
    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    elapsed = time.time() - t0

    marker = " ★" if va_loss < best_val else ""
    print(f"Epoch {epoch:02d}/{EPOCHS}  train={tr_loss:.5f}  val={va_loss:.5f}  "
          f"({elapsed:.1f}s){marker}", flush=True)

    if epoch % CKPT_EVERY == 0:
        path = os.path.join(CKPT_DIR, f"epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_losses":   val_losses}, path)
        print(f"  -> saved {path}", flush=True)

    if va_loss < best_val:
        best_val = va_loss
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "val_loss": va_loss},
                   os.path.join(CKPT_DIR, "best.pt"))

# ── loss curve ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(train_losses)+1), train_losses, label="train", marker="o", ms=3)
ax.plot(range(1, len(val_losses)+1),   val_losses,   label="val",   marker="s", ms=3)
ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
ax.set_title("ResCNN MorphPredictor — training curve (g-band, CSD3)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "figures/loss_curve.png"), dpi=110)
print(f"\nDone. Best val loss: {best_val:.5f}")
print("Saved loss_curve.png")
