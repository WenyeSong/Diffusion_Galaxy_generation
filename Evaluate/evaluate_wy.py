import sys
import os
import argparse
sys.path.append('/rds/user/ws452/hpc-work/lizarraga_2024/code')

import torch
import h5py
import numpy as np
import pandas as pd
import sep
import scipy.ndimage as ndimage
import logging
from tqdm import tqdm
from scipy.stats import circmean
from scipy.optimize import curve_fit
from photutils.isophote import EllipseGeometry, Ellipse

# ─── Sigma argument ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--sigma',     default='',  help='sigma tag: 01 / 05 / 10')
parser.add_argument('--chunk',     type=int, default=0,     help='chunk index (0-based)')
parser.add_argument('--n_chunks',  type=int, default=1,     help='total number of chunks')
parser.add_argument('--skip_test', action='store_true',     help='skip test-set evaluation')
args = parser.parse_args()
SIGMA_TAG = args.sigma
CHUNK     = args.chunk
N_CHUNKS  = args.n_chunks
suffix    = f'_s{SIGMA_TAG}' if SIGMA_TAG else ''

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE             = '/rds/user/ws452/hpc-work/lizarraga_2024'
TESTING_HDF5     = os.path.join(BASE, 'data/5x64x64_testing_with_morphology.hdf5')
GENERATED_DIR    = os.path.join(BASE, f'generated_images{suffix}')
GENERATED_Z_FILE = os.path.join(GENERATED_DIR, 'generated_redshifts.npy')
OUTPUT_DIR       = os.path.join(BASE, f'eval_output{suffix}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# chunk-specific output CSVs; merged later by merge_eval.py
TEST_CSV  = os.path.join(BASE, 'eval_output', 'testing_images_metrics.csv')
GEN_CSV   = os.path.join(OUTPUT_DIR, f'generated_metrics_chunk{CHUNK}.csv')
print(f"sigma={SIGMA_TAG!r}  chunk={CHUNK}/{N_CHUNKS}  GEN_CSV={GEN_CSV}")

BATCH_SIZE        = 100
MAX_TEST_IMAGES   = 40914
MAX_GEN_IMAGES    = 10000
# chunk range for generated images
GEN_CHUNK_SIZE = MAX_GEN_IMAGES // N_CHUNKS
GEN_START      = CHUNK * GEN_CHUNK_SIZE
GEN_END        = GEN_START + GEN_CHUNK_SIZE if CHUNK < N_CHUNKS - 1 else MAX_GEN_IMAGES
print(f"Generated images range: {GEN_START} – {GEN_END}")

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'evaluate.log'), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.handlers[1].setLevel(logging.ERROR)
logging.getLogger('photutils').setLevel(logging.WARNING)
logging.getLogger('astropy').setLevel(logging.WARNING)


# ─── SEP analysis ──────────────────────────────────────────────────────────────
def analyze_image_with_sep(image, redshift=None, apply_smoothing=True,
                            sigma=1.5, noise_threshold=0.50):
    result = {}

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        image = image[0]

    image = image.astype(np.float64)
    image = np.nan_to_num(image)

    if apply_smoothing:
        image = ndimage.gaussian_filter(image, sigma=sigma)   # smooth small noise in bg, place each pixel value to its neighbours weighted average, radius = sigma
    image = np.nan_to_num(image)

    try:
        bkg = sep.Background(image)  # bg subtraction
        image_sub = image - bkg
    except Exception as e:
        logging.warning(f"Background error: {e}")
        return None

    if np.isnan(image_sub).all() or image_sub.size == 0:
        return None

    epsilon = 1e-10
    mean_snr = np.mean(image_sub / (bkg.globalrms + epsilon))  # check signal-noise-rate
    if np.isnan(mean_snr) or mean_snr < noise_threshold:
        return None

    try:
        objects = sep.extract(image_sub, 1.5, err=bkg.globalrms)
    except Exception as e:
        logging.warning(f"SEP extract error: {e}")
        return None

    if len(objects) == 0:
        return None

    obj = objects[0]
    semi_major = obj['a']
    semi_minor = obj['b']
    ellipticity = 1 - (semi_minor / semi_major)

    result['Semi-major Axis']    = float(semi_major)
    result['Semi-minor Axis']    = float(semi_minor)
    result['Ellipticity']        = float(ellipticity)
    result['Orientation Angle']  = float(obj['theta'])
    result['Isophotal Area']     = float(obj['npix'])

    if redshift is not None:
        result['Redshift'] = float(redshift)

    if semi_major > 0 and 0 <= ellipticity <= 1:
        try:
            geometry = EllipseGeometry(x0=obj['x'], y0=obj['y'],
                                       sma=semi_major, eps=ellipticity,
                                       pa=obj['theta'])
            ellipse = Ellipse(image_sub, geometry)

            # ── ORIGINAL (kept for reference) ─────────────────────────────────
            # NOTE: NOT a formal Sersic index — ellipticity-derived heuristic.
            # Formula: n = ln(2) / ln((1+e)/(1-e))
            # Supervisor recommendation: replace with proper 1D profile fitting.
            ellipse.fit_image()
            ellipticity_proxy = np.log(2) / np.log((1 + ellipticity) / (1 - ellipticity))
            result['Ellipticity Proxy'] = float(ellipticity_proxy)

            # ── IMPROVED: proper 1D Sersic fit from isophote radial profile ───
            # Following supervisor recommendation (see feedback).
            # Quality cuts: >= 5 converged isophotes, positive intensity.
            isolist = ellipse.fit_image()

            if len(isolist) >= 5:
                sma    = np.array(isolist.sma)
                intens = np.array(isolist.intens)

                # keep only converged isophotes with positive intensity
                valid  = (isolist.stop_code == 0) & (intens > 0)
                sma    = sma[valid]
                intens = intens[valid]

                if len(sma) >= 5:
                    # Sersic profile: I(r) = I_e * exp(-b_n * ((r/r_e)^(1/n) - 1))
                    # Linearise, use ln: ln I = ln I_e - b_n * ((r/r_e)^(1/n) - 1)
                    # Fit simplified form: ln I(r) = A - B * r^(1/n)  (we need A,B n)
                    def sersic_log(r, A, B, n):
                        return A - B * r ** (1.0 / n)

                    try:
                        p0 = [np.log(intens[0]), 1.0, 1.0]
                        popt, _ = curve_fit(sersic_log, sma, np.log(intens),
                                            p0=p0, maxfev=2000, 
                                            bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 10]))   # fit
                        sersic_n_fitted = float(popt[2])
                        result['Sersic Index (fitted)'] = sersic_n_fitted
                    except Exception:
                        result['Sersic Index (fitted)'] = np.nan
                else:
                    result['Sersic Index (fitted)'] = np.nan
            else:
                result['Sersic Index (fitted)'] = np.nan

        except Exception as e:
            logging.warning(f"Ellipse fit error: {e}")
            result['Ellipticity Proxy']    = np.nan
            result['Sersic Index (fitted)'] = np.nan

    return result


# ─── CSV saving ────────────────────────────────────────────────────────────────
def save_results_to_csv(results, csv_path):
    if not results:
        return
    cleaned = []
    for r in results:
        cleaned.append({k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in r.items()})
    df = pd.DataFrame(cleaned)
    header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, index=False, mode='a', header=header)


# ─── Process testing images (with checkpoint resume) ──────────────────────────
TEST_CKPT = os.path.join(OUTPUT_DIR, 'test_checkpoint.txt')

print('\n=== Processing real testing images ===')

# resume from checkpoint if exists
start_i = 0
if os.path.exists(TEST_CKPT):
    with open(TEST_CKPT) as f:
        start_i = int(f.read().strip())
    print(f'Resuming from image index {start_i}')
else:
    # fresh start: clear CSV
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)

with h5py.File(TESTING_HDF5, 'r') as f:
    images_ds   = f['image']
    redshift_ds = f['specz_redshift']
    num_images  = min(images_ds.shape[0], MAX_TEST_IMAGES)

    with tqdm(total=num_images - start_i, desc='Testing images') as pbar:
        for i in range(start_i, num_images, BATCH_SIZE):
            img_batch = images_ds[i:i + BATCH_SIZE]
            z_batch   = redshift_ds[i:i + BATCH_SIZE]
            batch_results = []
            for j, image in enumerate(img_batch):
                try:
                    result = analyze_image_with_sep(image, redshift=z_batch[j])
                    if result:
                        result['Image Index'] = i + j
                        batch_results.append(result)
                except Exception as e:
                    logging.error(f"Error image {i+j}: {e}")
                pbar.update(1)
            save_results_to_csv(batch_results, TEST_CSV)
            # save checkpoint after each batch
            with open(TEST_CKPT, 'w') as f:
                f.write(str(i + BATCH_SIZE))

# done: remove checkpoint
if os.path.exists(TEST_CKPT):
    os.remove(TEST_CKPT)
print(f'Testing metrics saved to {TEST_CSV}')


# ─── Process generated images (with checkpoint resume) ────────────────────────
GEN_CKPT = os.path.join(OUTPUT_DIR, 'gen_checkpoint.txt')

print('\n=== Processing generated images ===')

start_gen = 0
if os.path.exists(GEN_CKPT):
    with open(GEN_CKPT) as f:
        start_gen = int(f.read().strip())
    print(f'Resuming generated images from index {start_gen}')
else:
    if os.path.exists(GEN_CSV):
        os.remove(GEN_CSV)

pt_files = sorted([
    os.path.join(GENERATED_DIR, f)
    for f in os.listdir(GENERATED_DIR)
    if f.endswith('.pt') and f.startswith('generated_image_')
])[:MAX_GEN_IMAGES]

redshifts = np.load(GENERATED_Z_FILE) if os.path.exists(GENERATED_Z_FILE) else [None] * len(pt_files)

with tqdm(total=len(pt_files) - start_gen, desc='Generated images') as pbar:
    for i in range(start_gen, len(pt_files), BATCH_SIZE):
        batch_files = pt_files[i:i + BATCH_SIZE]
        batch_z     = redshifts[i:i + BATCH_SIZE]
        batch_results = []
        for j, fpath in enumerate(batch_files):
            try:
                image  = torch.load(fpath, weights_only=False)
                result = analyze_image_with_sep(image, redshift=batch_z[j])
                if result:
                    result['Image File'] = os.path.basename(fpath)
                    batch_results.append(result)
            except Exception as e:
                logging.error(f"Error file {fpath}: {e}")
            pbar.update(1)
        save_results_to_csv(batch_results, GEN_CSV)
        with open(GEN_CKPT, 'w') as f:
            f.write(str(i + BATCH_SIZE))

if os.path.exists(GEN_CKPT):
    os.remove(GEN_CKPT)
print(f'Generated metrics saved to {GEN_CSV}')
print('\nDone.')
