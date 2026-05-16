import os
import sys
import torch
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append('/rds/user/ws452/hpc-work/lizarraga_2024/code')

from modules import EMA, UNet_conditional_conv

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = '/rds/user/ws452/hpc-work/lizarraga_2024/code/Model_Checkpoints_wy/best_model.pth'
TEST_HDF5   = '/rds/user/ws452/hpc-work/lizarraga_2024/data/5x64x64_testing_with_morphology.hdf5'
OUTPUT_DIR  = '/rds/user/ws452/hpc-work/lizarraga_2024/generated_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Settings ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 50        # images per generation batch (increase if GPU memory allows)
NUM_IMAGES  = 10000     # total images to generate (matches test set size for FID)
SEED        = 42
DEVICE      = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ─── Diffusion (linear schedule, matches training) ────────────────────────────
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta        = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha       = 1. - self.beta
        self.alpha_hat   = torch.cumprod(self.alpha, dim=0)
        self.img_size    = img_size
        self.device      = device

    def sample(self, model, n, labels):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 5, self.img_size, self.img_size)).to(self.device)
            labels = labels.view(-1, 1) if labels.dim() == 1 else labels
            for i in tqdm(reversed(range(1, self.noise_steps)),
                          desc=f'Denoising', total=self.noise_steps - 1, leave=False):
                t          = (torch.ones(n) * i).long().to(self.device)
                pred_noise = model(x, t, labels)
                alpha      = self.alpha[t][:, None, None, None]
                alpha_hat  = self.alpha_hat[t][:, None, None, None]
                beta       = self.beta[t][:, None, None, None]
                noise      = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
                ) + torch.sqrt(beta) * noise
        model.train()
        return x


# ─── Load model ───────────────────────────────────────────────────────────────
print(f'Using device: {DEVICE}')
model = UNet_conditional_conv(c_in=5, c_out=5, time_dim=256, y_dim=1).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

ema = EMA(model, beta=0.995)
if 'ema_state_dict' in ckpt:
    ema.load_state_dict(ckpt['ema_state_dict'])
ema_model = ema.get_ema_model()

print(f"Loaded model: epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.4f}")

# ─── Load test redshifts ──────────────────────────────────────────────────────
with h5py.File(TEST_HDF5, 'r') as f:
    all_redshifts = f['specz_redshift'][:]

print(f'Test set size: {len(all_redshifts)}')

np.random.seed(SEED)
n_generate = min(NUM_IMAGES, len(all_redshifts))
indices    = np.random.choice(len(all_redshifts), size=n_generate, replace=False)
redshifts  = all_redshifts[indices]

# Save redshifts so Evaluate can match them to images
np.save(os.path.join(OUTPUT_DIR, 'generated_redshifts.npy'), redshifts)

# ─── Generate in batches ──────────────────────────────────────────────────────
diffusion = Diffusion(device=DEVICE)
count = 0

print(f'Generating {n_generate} images in batches of {BATCH_SIZE}...')
for start in range(0, n_generate, BATCH_SIZE):
    batch_z = redshifts[start:start + BATCH_SIZE]
    n       = len(batch_z)
    labels  = torch.tensor(batch_z, dtype=torch.float32).to(DEVICE).unsqueeze(1)

    imgs = diffusion.sample(ema_model, n, labels)  # (n, 5, 64, 64)

    for j in range(n):
        path = os.path.join(OUTPUT_DIR, f'generated_image_{count:05d}.pt')
        torch.save(imgs[j].cpu(), path)
        count += 1

    print(f'  {count}/{n_generate} done')

print(f'Finished. {count} images saved to {OUTPUT_DIR}')
