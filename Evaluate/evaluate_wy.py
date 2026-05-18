import sys
import os
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
from photutils.isophote import EllipseGeometry, Ellipse

# ─── Paths ─────────────────────────────────────────────────────────────────────
TESTING_HDF5       = '/rds/user/ws452/hpc-work/lizarraga_2024/data/5x64x64_testing_with_morphology.hdf5'
GENERATED_DIR      = '/rds/user/ws452/hpc-work/lizarraga_2024/generated_images'
GENERATED_Z_FILE   = os.path.join(GENERATED_DIR, 'generated_redshifts.npy')
OUTPUT_DIR         = '/rds/user/ws452/hpc-work/lizarraga_2024/eval_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_CSV      = os.path.join(OUTPUT_DIR, 'testing_images_metrics.csv')
GEN_CSV       = os.path.join(OUTPUT_DIR, 'generated_images_metrics.csv')

BATCH_SIZE        = 100
MAX_TEST_IMAGES   = 40914   # full test set (matches paper)
MAX_GEN_IMAGES    = 10000   # number of generated images

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
            ellipse.fit_image()
            sersic_n = np.log(2) / np.log((1 + ellipticity) / (1 - ellipticity))
            result['Sersic Index'] = float(sersic_n)
        except Exception as e:
            logging.warning(f"Ellipse fit error: {e}")
            result['Sersic Index'] = np.nan

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


# ─── Process testing images ────────────────────────────────────────────────────
print('\n=== Processing real testing images ===')
if os.path.exists(TEST_CSV):
    os.remove(TEST_CSV)

with h5py.File(TESTING_HDF5, 'r') as f:
    images_ds   = f['image']
    redshift_ds = f['specz_redshift']
    num_images  = min(images_ds.shape[0], MAX_TEST_IMAGES)

    with tqdm(total=num_images, desc='Testing images') as pbar:
        for i in range(0, num_images, BATCH_SIZE):
            img_batch = images_ds[i:i + BATCH_SIZE]  # load pictures by batch
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

print(f'Testing metrics saved to {TEST_CSV}')


# ─── Process generated images ──────────────────────────────────────────────────
print('\n=== Processing generated images ===')
if os.path.exists(GEN_CSV):
    os.remove(GEN_CSV)

pt_files = sorted([
    os.path.join(GENERATED_DIR, f)
    for f in os.listdir(GENERATED_DIR)
    if f.endswith('.pt') and f.startswith('generated_image_')
])[:MAX_GEN_IMAGES]

redshifts = np.load(GENERATED_Z_FILE) if os.path.exists(GENERATED_Z_FILE) else [None] * len(pt_files)

with tqdm(total=len(pt_files), desc='Generated images') as pbar:
    for i in range(0, len(pt_files), BATCH_SIZE):
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

print(f'Generated metrics saved to {GEN_CSV}')
print('\nDone.')
