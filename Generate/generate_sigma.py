#!/usr/bin/env python3
"""
Generate 10,000 images from a trained sigma model.
Usage: python generate_sigma.py <SIGMA_TAG>
  SIGMA_TAG: "01" for σ=0.1,  "05" for σ=0.5,  "10" for σ=1.0
"""
import sys, os, glob
sys.path.append('/rds/user/ws452/hpc-work/lizarraga_2024/code')

import torch
import numpy as np
from tqdm import tqdm
from modules import EMA, UNet_conditional_conv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('sigma',   type=str)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end',   type=int, default=10000)
args_cli = parser.parse_args()
SIGMA_TAG  = args_cli.sigma
IDX_START  = args_cli.start
IDX_END    = args_cli.end

BASE_DIR   = '/rds/user/ws452/hpc-work/lizarraga_2024'
CKPT_PATH  = os.path.join(BASE_DIR, f'code/Model_Checkpoints_s{SIGMA_TAG}/best_model.pth')
OUT_DIR    = os.path.join(BASE_DIR, f'generated_images_s{SIGMA_TAG}')
N_IMAGES   = 10000
BATCH_SIZE = 100

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | sigma_tag: {SIGMA_TAG} | ckpt: {CKPT_PATH}")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta      = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha     = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size  = img_size
        self.device    = device

    def sample(self, model, n, labels):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 5, self.img_size, self.img_size)).to(self.device)
            labels = labels.view(-1, 1) if labels.dim() == 1 else labels
            for i in tqdm(reversed(range(1, self.noise_steps)),
                          total=self.noise_steps - 1, leave=False):
                t    = (torch.ones(n) * i).long().to(self.device)
                pred = model(x, t, labels)
                a    = self.alpha[t][:, None, None, None]
                ah   = self.alpha_hat[t][:, None, None, None]
                b    = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1/torch.sqrt(a) * (x - (1-a)/torch.sqrt(1-ah)*pred) + torch.sqrt(b)*noise
        model.train()
        return x

# ── Load model ─────────────────────────────────────────────────────────────────
model = UNet_conditional_conv(c_in=5, c_out=5, time_dim=256, y_dim=1).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
ema = EMA(model, beta=0.995)
if 'ema_state_dict' in ckpt:
    ema.load_state_dict(ckpt['ema_state_dict'])
ema_model = ema.get_ema_model()
diffusion  = Diffusion(device=device)

# ── Redshifts (resume-safe) ────────────────────────────────────────────────────
z_file = os.path.join(OUT_DIR, 'generated_redshifts.npy')
if os.path.exists(z_file):
    redshifts = np.load(z_file)
    print(f"Loaded existing redshifts ({len(redshifts)})")
else:
    redshifts = np.random.uniform(0.0, 4.0, N_IMAGES).astype(np.float32)
    np.save(z_file, redshifts)
    print("Saved new redshifts")

print(f"Chunk: images {IDX_START} – {IDX_END}")

for start in tqdm(range(IDX_START, IDX_END, BATCH_SIZE),
                  desc=f"Generating s{SIGMA_TAG} [{IDX_START}:{IDX_END}]"):
    end     = min(start + BATCH_SIZE, N_IMAGES)
    batch_z = torch.tensor(redshifts[start:end]).unsqueeze(1).to(device)
    imgs    = diffusion.sample(ema_model, len(batch_z), batch_z).cpu()
    for i, img in enumerate(imgs):
        torch.save(img, os.path.join(OUT_DIR, f'generated_image_{start+i:05d}.pt'))

print(f"Done. Images {IDX_START}–{IDX_END} saved to {OUT_DIR}")
