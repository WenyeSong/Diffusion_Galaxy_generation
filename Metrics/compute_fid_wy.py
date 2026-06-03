"""
FID (Fréchet Inception Distance) computation.
Follows Heusel et al. 2017 using InceptionV3 pool3 features (2048-dim).

Paper approach: sub-sample g,r,i channels as pseudo-RGB, resize to 299x299.
Reference: Lizarraga et al. 2024, Table 1.
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import h5py
import glob
import os
from tqdm import tqdm
from scipy.linalg import sqrtm

# ─── Sigma argument ────────────────────────────────────────────────────────────
import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument('--sigma', default='', help='sigma tag: 01 / 05 / 10')
args = parser.parse_args()
SIGMA_TAG = args.sigma
suffix = f'_s{SIGMA_TAG}' if SIGMA_TAG else ''

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/rds/user/ws452/hpc-work/lizarraga_2024'
TEST_HDF5 = os.path.join(BASE, 'data/5x64x64_testing_with_morphology.hdf5')
GEN_DIR   = os.path.join(BASE, f'generated_images{suffix}')
FID_DIR   = os.path.join(BASE, f'fid_output{suffix}')
print(f"sigma_tag={SIGMA_TAG!r}  GEN_DIR={GEN_DIR}")


NUM_REAL_IMAGES = 40914  # full test set (matches paper)
NUM_GEN_IMAGES  = 10000  # all generated images
BATCH_SIZE = 64
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# ─── Load InceptionV3 (pool3 features, 2048-dim) ──────────────────────────────
inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
inception.fc = torch.nn.Identity()   # remove final classification layer
inception.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_features(images_tensor):
    """Extract InceptionV3 pool3 features. Input: (N, 3, 299, 299) float [0,1]"""
    images_tensor = transform(images_tensor).to(DEVICE)
    with torch.no_grad():
        feats = inception(images_tensor)
    return feats.cpu().numpy()


def images_to_tensor(images_np):
    """
    Convert (N, 5, 64, 64) numpy array to (N, 3, 299, 299) tensor.
    Uses g, r, i channels (indices 0, 1, 2) as pseudo-RGB per paper convention.
    """
    gri = images_np[:, [0, 1, 2], :, :]   # g→R, r→G, i→B (paper convention)
    # per-image min-max normalise to [0, 1]
    mins = gri.min(axis=(1, 2, 3), keepdims=True)
    maxs = gri.max(axis=(1, 2, 3), keepdims=True)
    gri  = (gri - mins) / (maxs - mins + 1e-8)
    t    = torch.tensor(gri, dtype=torch.float32)
    # resize 64→299
    t    = torch.nn.functional.interpolate(t, size=(299, 299), mode='bilinear',
                                            align_corners=False)
    return t


def compute_statistics(features):
    """Compute mean and covariance of feature matrix (N, 2048)."""
    mu    = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    """
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


FID_DIR          = './fid'
os.makedirs(FID_DIR, exist_ok=True)
REAL_FEATS_CACHE = os.path.join(FID_DIR, 'real_features_40914.npy')
GEN_FEATS_CACHE  = os.path.join(FID_DIR, 'gen_features.npy')  # reuse existing

# ─── Extract features: Real test images ───────────────────────────────────────
if os.path.exists(REAL_FEATS_CACHE):
    print(f"\n=== Loading cached real features from {REAL_FEATS_CACHE} ===")
    real_feats = np.load(REAL_FEATS_CACHE)
else:
    print("\n=== Extracting features from real test images ===")
    real_feats = []
    with h5py.File(TEST_HDF5, 'r') as f:
        n = min(f['image'].shape[0], NUM_REAL_IMAGES)
        for i in tqdm(range(0, n, BATCH_SIZE), desc='Real images'):
            batch = f['image'][i:i+BATCH_SIZE].astype('float32')
            t     = images_to_tensor(batch)
            real_feats.append(get_features(t))
    real_feats = np.concatenate(real_feats, axis=0)
    np.save(REAL_FEATS_CACHE, real_feats)
    print(f"Saved to {REAL_FEATS_CACHE}")
print(f"Real features shape: {real_feats.shape}")

# ─── Extract features: Generated images ───────────────────────────────────────
if os.path.exists(GEN_FEATS_CACHE):
    print(f"\n=== Loading cached generated features from {GEN_FEATS_CACHE} ===")
    gen_feats = np.load(GEN_FEATS_CACHE)
else:
    print("\n=== Extracting features from generated images ===")
    pt_files  = sorted(glob.glob(os.path.join(GEN_DIR, 'generated_image_*.pt')))[:NUM_GEN_IMAGES]
    gen_feats = []
    for i in tqdm(range(0, len(pt_files), BATCH_SIZE), desc='Generated images'):
        batch_files = pt_files[i:i+BATCH_SIZE]
        batch = np.stack([torch.load(f, weights_only=False).numpy()
                          for f in batch_files]).astype('float32')
        t     = images_to_tensor(batch)
        gen_feats.append(get_features(t))
    gen_feats = np.concatenate(gen_feats, axis=0)
    np.save(GEN_FEATS_CACHE, gen_feats)
    print(f"Saved to {GEN_FEATS_CACHE}")
print(f"Generated features shape: {gen_feats.shape}")

# ─── Compute FID ──────────────────────────────────────────────────────────────
print("\n=== Computing FID ===")
mu_r, sigma_r = compute_statistics(real_feats)
mu_g, sigma_g = compute_statistics(gen_feats)

fid = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

print(f"\nFID score: {fid:.2f}")
print(f"(Paper best result: 14.2 for continuous-cond DDPM sigma=0.1)")
print(f"Lower is better.")

# save result
with open(os.path.join(FID_DIR, 'fid_result_40914_wy.txt'), 'w') as f:
    f.write(f"FID = {fid:.4f}\n")
    f.write(f"Real images: {len(real_feats)}\n")
    f.write(f"Generated images: {len(gen_feats)}\n")
    f.write(f"Channels: g, r, i (pseudo-RGB, per Lizarraga et al. 2024)\n")
print("Saved: fid_result_wy.txt")
