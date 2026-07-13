"""
Train the MorphCNN regressor on GalaxiesML HDF5 data.

Usage:
    python train.py

Outputs (all written to the same directory as this script):
    checkpoints/epoch_XX.pt   — model weights every 5 epochs
    checkpoints/best.pt       — best val-loss checkpoint
    loss_curve.png            — train/val loss plot
"""

import os, sys, json, time, glob
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MorphCNN

# ── Config ─────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = "/Users/wen/Desktop/project/archive"

HDF5_FILES = {
    "train": os.path.join(DATA_DIR, "5x64x64_training_with_morphology.hdf5"),
    "val"  : os.path.join(DATA_DIR, "5x64x64_validation_with_morphology.hdf5"),
}

TARGET_COLS = ["g_ellipticity", "g_major_axis", "g_sersic_index", "g_isophotal_area"]

EPOCHS     = 20
BATCH_SIZE = 128
LR         = 3e-4
NUM_WORKERS = 0      # HDF5 doesn't support multiprocessing reads safely on macOS
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

CKPT_DIR   = os.path.join(HERE, "..", "checkpoints")
CKPT_EVERY = 2
os.makedirs(CKPT_DIR, exist_ok=True)


# ── Normaliser ─────────────────────────────────────────────────────────────
with open(os.path.join(HERE, "..", "normaliser.json")) as fp:
    NORM = json.load(fp)

def normalise(arr_dict):
    """Dict of col -> np.array  =>  (N, 4) normalised float32 array."""
    out = []
    for col in TARGET_COLS:
        v = arr_dict[col].astype(np.float32)
        t = NORM[col]["transform"]
        if t == "log1p":
            v = np.log1p(v)
        elif t == "log":
            v = np.log(v)
        v = (v - NORM[col]["mean"]) / NORM[col]["std"]
        out.append(v)
    return np.stack(out, axis=1)   # (N, 4)

def denormalise(tensor):
    """(N, 4) normalised tensor -> (N, 4) original-scale numpy array."""
    arr = tensor.cpu().numpy().astype(np.float64)
    out = []
    for i, col in enumerate(TARGET_COLS):
        v = arr[:, i] * NORM[col]["std"] + NORM[col]["mean"]
        t = NORM[col]["transform"]
        if t == "log1p":
            v = np.expm1(v)
        elif t == "log":
            v = np.exp(v)
        out.append(v)
    return np.stack(out, axis=1)


# ── Dataset ────────────────────────────────────────────────────────────────
class GalaxyDataset(Dataset):
    def __init__(self, hdf5_path, indices):
        self.path    = hdf5_path
        self.indices = indices   # sorted int array
        self._file   = None      # opened lazily per worker

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._open()
        idx = int(self.indices[i])
        img = self._file["image"][idx].astype(np.float32)        # (5,64,64)
        lbl = {col: float(self._file[col][idx]) for col in TARGET_COLS}
        y   = normalise({c: np.array([v]) for c, v in lbl.items()})[0]
        return torch.from_numpy(img), torch.from_numpy(y)


# ── Image preprocessing ────────────────────────────────────────────────────
# Per-channel asinh stretch + standardise using training-set statistics.
# We compute channel stats lazily from a small sample if not cached.
STATS_FILE = os.path.join(HERE, "..", "image_stats.json")

def compute_image_stats(hdf5_path, indices, n_sample=2000):
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(indices, size=min(n_sample, len(indices)), replace=False)
    sample_idx.sort()
    imgs = []
    with h5py.File(hdf5_path, "r") as f:
        for idx in sample_idx:
            imgs.append(f["image"][int(idx)])
    imgs = np.stack(imgs)   # (N, 5, 64, 64)
    # asinh stretch per channel
    imgs = np.arcsinh(imgs)
    mean = imgs.mean(axis=(0, 2, 3)).tolist()   # (5,)
    std  = imgs.std(axis=(0, 2, 3)).tolist()
    stats = {"mean": mean, "std": std}
    with open(STATS_FILE, "w") as fp:
        json.dump(stats, fp, indent=2)
    print(f"  image stats computed and saved to {STATS_FILE}")
    return stats

def load_image_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as fp:
            return json.load(fp)
    return None


class NormalisedGalaxyDataset(GalaxyDataset):
    """Applies asinh + channel standardisation on top of GalaxyDataset."""
    def __init__(self, hdf5_path, indices, img_stats):
        super().__init__(hdf5_path, indices)
        self.mean = np.array(img_stats["mean"], dtype=np.float32).reshape(5, 1, 1)
        self.std  = np.array(img_stats["std"],  dtype=np.float32).reshape(5, 1, 1)

    def __getitem__(self, i):
        img, y = super().__getitem__(i)
        img = torch.arcsinh(img)
        img = (img - torch.from_numpy(self.mean)) / torch.from_numpy(self.std)
        return img, y


# ── Training loop ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimiser, device, train=True):
    model.train(train)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss  = criterion(preds, labels)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return total_loss / n


def main():
    print(f"Device: {DEVICE}")

    # --- load / compute image stats ---
    train_idx = np.load(os.path.join(HERE, "..", "data/indices_train.npy"))
    val_idx   = np.load(os.path.join(HERE, "..", "data/indices_val.npy"))

    img_stats = load_image_stats()
    if img_stats is None:
        print("Computing image stats from training sample...")
        img_stats = compute_image_stats(HDF5_FILES["train"], train_idx)

    # --- datasets ---
    train_ds = NormalisedGalaxyDataset(HDF5_FILES["train"], train_idx, img_stats)
    val_ds   = NormalisedGalaxyDataset(HDF5_FILES["val"],   val_idx,   img_stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    # --- model ---
    model     = MorphCNN(n_outputs=4).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── auto-resume ────────────────────────────────────────────────────────
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
        print(f"Resumed from {existing[-1]}  (epoch {ckpt['epoch']})")
    else:
        print("Starting from scratch")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, criterion, optimiser, DEVICE, train=True)
        va_loss = run_epoch(model, val_loader,   criterion, optimiser, DEVICE, train=False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train={tr_loss:.5f}  val={va_loss:.5f}  ({elapsed:.1f}s)")

        if epoch % CKPT_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"epoch_{epoch:02d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_losses": train_losses,
                        "val_losses":   val_losses}, path)
            print(f"  -> saved {path}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": va_loss}, os.path.join(CKPT_DIR, "best.pt"))

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), train_losses, label="train")
    ax.plot(range(1, EPOCHS + 1), val_losses,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Training curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "..", "figures/loss_curve.png"), dpi=110)
    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Loss curve saved to loss_curve.png")


if __name__ == "__main__":
    main()
