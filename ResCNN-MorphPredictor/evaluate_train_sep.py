"""
Compute SEP morphology labels for GalaxiesML train/val images.
Uses the identical analysis function as evaluate_wy.py for full consistency.

Run as SLURM array job (see submit_train_sep.sh):
    python evaluate_train_sep.py --split train --chunk 0 --n_chunks 20
    python evaluate_train_sep.py --split val   --chunk 0 --n_chunks 4

Outputs to sep_labels/{split}_sep_chunk{chunk:03d}.csv
Columns: hdf5_index, Ellipticity, Semi-major Axis, Isophotal Area, Sersic Index
"""

import os, argparse
import numpy as np
import pandas as pd
import h5py
import sep
import scipy.ndimage as ndimage
import logging
from tqdm import tqdm
from scipy.optimize import curve_fit
from photutils.isophote import EllipseGeometry, Ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--split',    choices=['train', 'val'], default='train')
parser.add_argument('--chunk',    type=int, default=0)
parser.add_argument('--n_chunks', type=int, default=1)
args = parser.parse_args()

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/rds/user/ws452/hpc-work/lizarraga_2024/data"
OUTPUT_DIR = os.path.join(HERE, "sep_labels")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HDF5 = {
    "train": os.path.join(DATA_DIR, "5x64x64_training_with_morphology.hdf5"),
    "val"  : os.path.join(DATA_DIR, "5x64x64_validation_with_morphology.hdf5"),
}
IDX_FILES = {
    "train": os.path.join(HERE, "indices_train.npy"),
    "val"  : os.path.join(HERE, "indices_val.npy"),
}

# Band order confirmed from step1_data_exploration.ipynb: g=0, r=1, i=2, z=3, y=4
# train_csd3.py uses g-band catalog labels; evaluate_wy.py takes image[0] from generated images.
# Both are g-band, so BAND_IDX = 0.
BAND_IDX   = 0
BATCH_SIZE = 50

logging.basicConfig(level=logging.WARNING)
logging.getLogger('photutils').setLevel(logging.WARNING)
logging.getLogger('astropy').setLevel(logging.WARNING)


def analyze_image_with_sep(image, apply_smoothing=True, sigma=1.5, noise_threshold=0.50):
    """Identical logic to evaluate_wy.py — do not diverge from this."""
    if image.ndim == 3:
        image = image[BAND_IDX]

    image = image.astype(np.float64)
    image = np.nan_to_num(image)

    if apply_smoothing:
        image = ndimage.gaussian_filter(image, sigma=sigma)
    image = np.nan_to_num(image)

    try:
        bkg = sep.Background(image)
        image_sub = image - bkg
    except Exception:
        return None

    if np.isnan(image_sub).all() or image_sub.size == 0:
        return None

    epsilon  = 1e-10
    mean_snr = np.mean(image_sub / (bkg.globalrms + epsilon))
    if np.isnan(mean_snr) or mean_snr < noise_threshold:
        return None

    try:
        objects = sep.extract(image_sub, 1.5, err=bkg.globalrms)
    except Exception:
        return None

    if len(objects) == 0:
        return None

    obj         = objects[0]
    semi_major  = obj['a']
    semi_minor  = obj['b']
    ellipticity = 1 - (semi_minor / semi_major)

    result = {
        'Semi-major Axis': float(semi_major),
        'Ellipticity':     float(ellipticity),
        'Isophotal Area':  float(obj['npix']),
        'Sersic Index':    np.nan,
    }

    if semi_major > 0 and 0 < ellipticity < 1:
        try:
            geometry = EllipseGeometry(x0=obj['x'], y0=obj['y'],
                                       sma=semi_major, eps=ellipticity,
                                       pa=obj['theta'])
            ellipse  = Ellipse(image_sub, geometry)
            isolist  = ellipse.fit_image()

            if len(isolist) >= 5:
                sma    = np.array(isolist.sma)
                intens = np.array(isolist.intens)
                valid  = (isolist.stop_code == 0) & (intens > 0)
                sma, intens = sma[valid], intens[valid]

                if len(sma) >= 5:
                    def sersic_log(r, A, B, n):
                        return A - B * r ** (1.0 / n)
                    try:
                        p0   = [np.log(intens[0]), 1.0, 1.0]
                        popt, _ = curve_fit(
                            sersic_log, sma, np.log(intens), p0=p0,
                            maxfev=2000,
                            bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 10]))
                        result['Sersic Index'] = float(popt[2])
                    except Exception:
                        pass
        except Exception:
            pass

    return result


# ── slice this chunk ───────────────────────────────────────────────────────────
all_hdf5_indices = np.load(IDX_FILES[args.split])
n_total    = len(all_hdf5_indices)
chunk_size = (n_total + args.n_chunks - 1) // args.n_chunks
start      = args.chunk * chunk_size
end        = min(start + chunk_size, n_total)
chunk_hdf5_indices = all_hdf5_indices[start:end]   # still sorted

out_csv = os.path.join(OUTPUT_DIR, f"{args.split}_sep_chunk{args.chunk:03d}.csv")
if os.path.exists(out_csv):
    os.remove(out_csv)

print(f"split={args.split}  chunk={args.chunk}/{args.n_chunks}  "
      f"images {start}–{end-1}  total={len(chunk_hdf5_indices)}")
print(f"Output: {out_csv}")

# ── main loop ─────────────────────────────────────────────────────────────────
n_ok, n_fail = 0, 0
buffer = []

with h5py.File(HDF5[args.split], 'r') as f:
    images_ds = f['image']
    with tqdm(total=len(chunk_hdf5_indices), unit='img') as pbar:
        for b in range(0, len(chunk_hdf5_indices), BATCH_SIZE):
            batch_idx = chunk_hdf5_indices[b:b + BATCH_SIZE]
            imgs      = images_ds[batch_idx]           # (B, 5, 64, 64)

            for j, img in enumerate(imgs):
                res = analyze_image_with_sep(img)
                if res is not None:
                    res['hdf5_index'] = int(batch_idx[j])
                    buffer.append(res)
                    n_ok += 1
                else:
                    n_fail += 1
                pbar.update(1)

            if len(buffer) >= BATCH_SIZE:
                df = pd.DataFrame(buffer)
                df.to_csv(out_csv, index=False, mode='a',
                          header=not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0)
                buffer = []

if buffer:
    df = pd.DataFrame(buffer)
    df.to_csv(out_csv, index=False, mode='a',
              header=not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0)

print(f"\nDone — OK={n_ok}  failed={n_fail}  "
      f"({100*n_fail/(n_ok+n_fail+1e-9):.1f}% dropped)")
print(f"Saved to {out_csv}")
