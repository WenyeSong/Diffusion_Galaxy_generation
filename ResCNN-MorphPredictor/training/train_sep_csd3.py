"""
ResCNN morphology predictor — retrain with SEP-computed labels.

Replaces the GalaxiesML catalog labels (g_ellipticity etc.) with the
SEP metrics produced by evaluate_train_sep.py so that training labels
and generated-image evaluation are fully self-consistent.

Prerequisites:
    sep_labels/train_sep_chunk*.csv   (from evaluate_train_sep.py)
    sep_labels/val_sep_chunk*.csv

Usage (via SLURM, see submit_rescnn.sh — point it at this file):
    python train_sep_csd3.py
"""

import os, sys, json, time, glob
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MorphCNN

# ── paths ─────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = "/rds/user/ws452/hpc-work/lizarraga_2024/data"
SEP_DIR   = os.path.join(HERE, "..", "sep_labels")
CKPT_DIR  = os.path.join(HERE, "..", "checkpoints_sep")
os.makedirs(CKPT_DIR, exist_ok=True)

HDF5 = {
    "train": os.path.join(DATA_DIR, "5x64x64_training_with_morphology.hdf5"),
    "val"  : os.path.join(DATA_DIR, "5x64x64_validation_with_morphology.hdf5"),
}

# Must match the column order used by the CNN output and evaluate_wy.py
SEP_COLS = ["Ellipticity", "Semi-major Axis", "Sersic Index", "Isophotal Area"]

# Transforms that linearise each metric before z-scoring.
# Keep log/log1p consistent with evaluate_wy.py column ranges.
SEP_TRANSFORMS = {
    "Ellipticity":    "none",
    "Semi-major Axis": "log1p",
    "Sersic Index":   "log",
    "Isophotal Area": "log",
}

EPOCHS     = 40
BATCH_SIZE = 512
LR         = 3e-4
NUM_WORKERS = 4
CKPT_EVERY  = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ── load and merge SEP CSVs ───────────────────────────────────────────────────
def load_sep_csv(split):
    pattern = os.path.join(SEP_DIR, f"{split}_sep_chunk*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No SEP CSV files found at {pattern}\n"
            "Run evaluate_train_sep.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="hdf5_index")
    # drop rows where any label is NaN (SEP found object but Sersic fitting failed)
    n_before = len(df)
    df = df.dropna(subset=SEP_COLS)
    n_dropped = n_before - len(df)
    print(f"  {split}: {len(df):,} valid  ({n_dropped:,} dropped for NaN labels)")
    df = df.sort_values("hdf5_index").reset_index(drop=True)
    return df


print("Loading SEP labels...")
train_df = load_sep_csv("train")
val_df   = load_sep_csv("val")


# ── build normaliser from training SEP labels ─────────────────────────────────
NORM_FILE = os.path.join(HERE, "..", "stats/normaliser_sep.json")

def build_normaliser(df):
    norm = {}
    for col in SEP_COLS:
        v = df[col].values.astype(np.float64)
        t = SEP_TRANSFORMS[col]
        if t == "log1p": v = np.log1p(v)
        elif t == "log":  v = np.log(v)
        norm[col] = {"transform": t, "mean": float(v.mean()), "std": float(v.std())}
    return norm

if os.path.exists(NORM_FILE):
    with open(NORM_FILE) as f:
        NORM = json.load(f)
    print(f"Loaded {NORM_FILE}")
else:
    NORM = build_normaliser(train_df)
    with open(NORM_FILE, "w") as f:
        json.dump(NORM, f, indent=2)
    print(f"Computed and saved {NORM_FILE}")
    for col, v in NORM.items():
        print(f"  {col}: transform={v['transform']}  mean={v['mean']:.4f}  std={v['std']:.4f}")


def normalise_labels(arr):
    """arr: (N, 4) in SEP_COLS order → normalised (N, 4) float32."""
    out = []
    for i, col in enumerate(SEP_COLS):
        v = arr[:, i].astype(np.float32)
        t = NORM[col]["transform"]
        if t == "log1p": v = np.log1p(v)
        elif t == "log":  v = np.log(v)
        v = (v - NORM[col]["mean"]) / NORM[col]["std"]
        out.append(v)
    return np.stack(out, axis=1)


def denormalise(arr):
    """(N, 4) normalised → (N, 4) original scale numpy float64."""
    out = []
    for i, col in enumerate(SEP_COLS):
        v = arr[:, i].copy().astype(np.float64)
        v = v * NORM[col]["std"] + NORM[col]["mean"]
        t = NORM[col]["transform"]
        if t == "log1p": v = np.expm1(v)
        elif t == "log":  v = np.exp(v)
        out.append(v)
    return np.stack(out, axis=1)


# ── load images from HDF5 ─────────────────────────────────────────────────────
STATS_FILE = os.path.join(HERE, "..", "stats/image_stats.json")

def load_images_and_labels(hdf5_path, df, desc=""):
    hdf5_idx = df["hdf5_index"].values.astype(int)
    labels   = df[SEP_COLS].values.astype(np.float32)
    n = len(hdf5_idx)
    imgs = np.empty((n, 5, 64, 64), dtype=np.float32)
    print(f"Loading {desc} images ({n:,})...", flush=True)
    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        for s in range(0, n, 2000):
            sl = slice(s, min(s + 2000, n))
            imgs[sl] = f["image"][hdf5_idx[sl]]
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return imgs, labels


train_imgs, train_lbls_raw = load_images_and_labels(HDF5["train"], train_df, "train")
val_imgs,   val_lbls_raw   = load_images_and_labels(HDF5["val"],   val_df,   "val")

# asinh stretch
train_imgs = np.arcsinh(train_imgs)
val_imgs   = np.arcsinh(val_imgs)

# image standardisation
if os.path.exists(STATS_FILE):
    with open(STATS_FILE) as f:
        img_stats = json.load(f)
    print("Loaded image_stats.json")
else:
    print("Computing image stats from training set...")
    img_stats = {"mean": train_imgs.mean(axis=(0,2,3)).tolist(),
                 "std" : train_imgs.std(axis=(0,2,3)).tolist()}
    with open(STATS_FILE, "w") as f:
        json.dump(img_stats, f, indent=2)
    print("Saved image_stats.json")

IMG_MEAN = np.array(img_stats["mean"], dtype=np.float32).reshape(1, 5, 1, 1)
IMG_STD  = np.array(img_stats["std"],  dtype=np.float32).reshape(1, 5, 1, 1)
train_imgs = (train_imgs - IMG_MEAN) / IMG_STD
val_imgs   = (val_imgs   - IMG_MEAN) / IMG_STD

train_lbls = normalise_labels(train_lbls_raw)
val_lbls   = normalise_labels(val_lbls_raw)

train_ds = TensorDataset(torch.from_numpy(train_imgs), torch.from_numpy(train_lbls))
val_ds   = TensorDataset(torch.from_numpy(val_imgs),   torch.from_numpy(val_lbls))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)


# ── model ─────────────────────────────────────────────────────────────────────
model     = MorphCNN(n_outputs=4).to(DEVICE)
criterion = nn.HuberLoss(delta=1.0)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Starting from scratch  ({n_params:,} params)", flush=True)


# ── training loop ─────────────────────────────────────────────────────────────
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
    marker = " ★" if va_loss < best_val else ""
    print(f"Epoch {epoch:02d}/{EPOCHS}  train={tr_loss:.5f}  val={va_loss:.5f}  "
          f"({time.time()-t0:.1f}s){marker}", flush=True)

    if epoch % CKPT_EVERY == 0:
        path = os.path.join(CKPT_DIR, f"epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_losses":   val_losses}, path)
        print(f"  -> {path}", flush=True)

    if va_loss < best_val:
        best_val = va_loss
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "val_loss": va_loss, "normaliser": NORM, "sep_cols": SEP_COLS},
                   os.path.join(CKPT_DIR, "best.pt"))


# ── loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(train_losses)+1), train_losses, label="train", marker="o", ms=3)
ax.plot(range(1, len(val_losses)+1),   val_losses,   label="val",   marker="s", ms=3)
ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
ax.set_title("ResCNN SEP-label training curve")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "figures/loss_curve_sep.png"), dpi=110)
print(f"\nDone. Best val loss: {best_val:.5f}")
print("Saved loss_curve_sep.png")
