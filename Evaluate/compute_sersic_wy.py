"""
Standalone Sersic Index computation script.
Runs on ALL test images (no noise threshold filter) and ALL generated images.
Compares:
- Ellipticity Proxy: log(2) / log((1+e)/(1-e))  [original heuristic]
- Sersic Index (fitted): proper 1D profile fit via curve_fit

Checkpoint supported: resumes from where it left off if job is cancelled.
"""

import sys, os
sys.path.append('/rds/user/ws452/hpc-work/lizarraga_2024/code')

import numpy as np
import pandas as pd
import h5py
import torch
import sep
import scipy.ndimage as ndimage
import logging
from tqdm import tqdm
from scipy.optimize import curve_fit
from photutils.isophote import EllipseGeometry, Ellipse

# ─── Paths ─────────────────────────────────────────────────────────────────────
TESTING_HDF5   = '/rds/user/ws452/hpc-work/lizarraga_2024/data/5x64x64_testing_with_morphology.hdf5'
GENERATED_DIR  = '/rds/user/ws452/hpc-work/lizarraga_2024/generated_images'
GENERATED_Z    = os.path.join(GENERATED_DIR, 'generated_redshifts.npy')
OUTPUT_DIR     = '/rds/user/ws452/hpc-work/lizarraga_2024/eval_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CSV  = os.path.join(OUTPUT_DIR, 'sersic_testing_all.csv')
GEN_CSV   = os.path.join(OUTPUT_DIR, 'sersic_generated_all.csv')
TEST_CKPT = os.path.join(OUTPUT_DIR, 'sersic_test_ckpt.txt')
GEN_CKPT  = os.path.join(OUTPUT_DIR, 'sersic_gen_ckpt.txt')

BATCH_SIZE     = 100
MAX_GEN_IMAGES = 10000

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logging.getLogger('photutils').setLevel(logging.WARNING)
logging.getLogger('astropy').setLevel(logging.WARNING)


# ─── Core function: Sersic fitting only ────────────────────────────────────────
def compute_sersic(image, apply_smoothing=True, sigma=1.5):
    """
    Returns dict with Ellipticity Proxy and Sersic Index (fitted).
    No noise threshold filter — runs on all images.
    """
    result = {}

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        image = image[0]

    image = image.astype(np.float64)
    image = np.nan_to_num(image)

    if apply_smoothing:
        image = ndimage.gaussian_filter(image, sigma=sigma)
    image = np.nan_to_num(image)

    try:
        bkg       = sep.Background(image)
        image_sub = image - bkg
    except Exception:
        return None

    if np.isnan(image_sub).all() or image_sub.size == 0:
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

    result['Ellipticity'] = float(ellipticity)

    if semi_major <= 0 or not (0 <= ellipticity <= 1):
        result['Ellipticity Proxy']    = np.nan
        result['Sersic Index (fitted)'] = np.nan
        return result

    # Ellipticity Proxy (original heuristic, kept for reference)
    try:
        ep = np.log(2) / np.log((1 + ellipticity) / (1 - ellipticity))
        result['Ellipticity Proxy'] = float(ep)
    except Exception:
        result['Ellipticity Proxy'] = np.nan

    # Proper 1D Sersic fit from isophote radial profile
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
            sma    = sma[valid]
            intens = intens[valid]

            if len(sma) >= 5:
                def sersic_log(r, A, B, n):
                    return A - B * r ** (1.0 / n)

                p0   = [np.log(intens[0]), 1.0, 1.0]
                popt, _ = curve_fit(sersic_log, sma, np.log(intens),
                                    p0=p0, maxfev=2000,
                                    bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 10]))
                result['Sersic Index (fitted)'] = float(popt[2])
            else:
                result['Sersic Index (fitted)'] = np.nan
        else:
            result['Sersic Index (fitted)'] = np.nan

    except Exception:
        result['Sersic Index (fitted)'] = np.nan

    return result


def save_csv(results, csv_path):
    if not results:
        return
    df = pd.DataFrame(results)
    header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, index=False, mode='a', header=header)


# ─── Testing images ────────────────────────────────────────────────────────────
print('\n=== Sersic fitting: real testing images (ALL, no noise filter) ===')

start_i = 0
if os.path.exists(TEST_CKPT):
    with open(TEST_CKPT) as f:
        start_i = int(f.read().strip())
    print(f'Resuming from index {start_i}')
else:
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)

with h5py.File(TESTING_HDF5, 'r') as f:
    n = f['image'].shape[0]
    with tqdm(total=n - start_i, desc='Testing') as pbar:
        for i in range(start_i, n, BATCH_SIZE):
            img_batch = f['image'][i:i+BATCH_SIZE]
            batch_results = []
            for j, image in enumerate(img_batch):
                res = compute_sersic(image)
                if res is not None:
                    res['Image Index'] = i + j
                    batch_results.append(res)
                pbar.update(1)
            save_csv(batch_results, TEST_CSV)
            with open(TEST_CKPT, 'w') as ck:
                ck.write(str(i + BATCH_SIZE))

if os.path.exists(TEST_CKPT):
    os.remove(TEST_CKPT)
print(f'Saved: {TEST_CSV}')


# ─── Generated images ──────────────────────────────────────────────────────────
print('\n=== Sersic fitting: generated images (ALL, no noise filter) ===')

pt_files  = sorted([
    os.path.join(GENERATED_DIR, f)
    for f in os.listdir(GENERATED_DIR)
    if f.endswith('.pt') and f.startswith('generated_image_')
])[:MAX_GEN_IMAGES]
redshifts = np.load(GENERATED_Z) if os.path.exists(GENERATED_Z) else [None]*len(pt_files)

start_gen = 0
if os.path.exists(GEN_CKPT):
    with open(GEN_CKPT) as f:
        start_gen = int(f.read().strip())
    print(f'Resuming from index {start_gen}')
else:
    if os.path.exists(GEN_CSV):
        os.remove(GEN_CSV)

with tqdm(total=len(pt_files) - start_gen, desc='Generated') as pbar:
    for i in range(start_gen, len(pt_files), BATCH_SIZE):
        batch_files = pt_files[i:i+BATCH_SIZE]
        batch_z     = redshifts[i:i+BATCH_SIZE]
        batch_results = []
        for j, fpath in enumerate(batch_files):
            try:
                image = torch.load(fpath, weights_only=False)
                res   = compute_sersic(image)
                if res is not None:
                    res['Image File'] = os.path.basename(fpath)
                    res['Redshift']   = float(batch_z[j]) if batch_z[j] is not None else np.nan
                    batch_results.append(res)
            except Exception as e:
                logging.error(f"Error {fpath}: {e}")
            pbar.update(1)
        save_csv(batch_results, GEN_CSV)
        with open(GEN_CKPT, 'w') as ck:
            ck.write(str(i + BATCH_SIZE))

if os.path.exists(GEN_CKPT):
    os.remove(GEN_CKPT)
print(f'Saved: {GEN_CSV}')
print('\nDone.')
