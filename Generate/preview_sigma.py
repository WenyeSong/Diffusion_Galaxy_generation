#!/usr/bin/env python3
"""
Generate 10 preview images for a sigma model and save as PNG.
Usage: python preview_sigma.py <SIGMA_TAG>
"""
import sys, os
sys.path.append('/rds/user/ws452/hpc-work/lizarraga_2024/code')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed on CSD3
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import EMA, UNet_conditional_conv

SIGMA_TAG = sys.argv[1]
BASE      = '/rds/user/ws452/hpc-work/lizarraga_2024'
CKPT      = os.path.join(BASE, f'code/Model_Checkpoints_s{SIGMA_TAG}/best_model.pth')
OUT_PNG   = os.path.join(BASE, f'preview_s{SIGMA_TAG}.png')
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"sigma={SIGMA_TAG}  device={device}  ckpt={CKPT}")

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
            for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t    = (torch.ones(n) * i).long().to(self.device)
                pred = model(x, t, labels)
                a    = self.alpha[t][:, None, None, None]
                ah   = self.alpha_hat[t][:, None, None, None]
                b    = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1/torch.sqrt(a) * (x - (1-a)/torch.sqrt(1-ah)*pred) + torch.sqrt(b)*noise
        model.train()
        return x

model = UNet_conditional_conv(c_in=5, c_out=5, time_dim=256, y_dim=1).to(device)
ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
ema   = EMA(model, beta=0.995)
if 'ema_state_dict' in ckpt:
    ema.load_state_dict(ckpt['ema_state_dict'])
diff  = Diffusion(noise_steps=200, device=str(device))  # 200 steps for fast preview

N = 6   # fewer images for speed
zs   = torch.linspace(0.1, 3.5, N).unsqueeze(1).to(device)
imgs = diff.sample(ema.get_ema_model(), N, zs)

def to_rgb(img):
    rgb = img[[2,1,0]].cpu().numpy().astype(float)
    rgb = np.transpose(rgb, (1,2,0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.clip(rgb, 0, 1)

fig, axes = plt.subplots(1, N, figsize=(N*2, 2.5))
fig.patch.set_facecolor('black')
for i in range(N):
    axes[i].imshow(to_rgb(imgs[i]))
    axes[i].set_title(f'z={zs[i,0].item():.2f}', color='white', fontsize=8)
    axes[i].axis('off')
plt.suptitle(f'sigma={SIGMA_TAG} preview', color='white', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='black')
print(f"Saved: {OUT_PNG}")
