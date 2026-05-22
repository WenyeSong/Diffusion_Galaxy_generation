"""
Compare SEP-extracted metrics vs HDF5 pre-computed metrics for the test set.
Uses Image Index in testing_images_metrics_wy.csv to match same galaxies.
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

SEP_CSV   = './testing_images_metrics_wy.csv'
HDF5_PATH = '/Users/wen/Desktop/project/archive/5x64x64_testing_with_morphology.hdf5'

# ─── Load SEP results ──────────────────────────────────────────────────────────
sep_df = pd.read_csv(SEP_CSV)
print(f"SEP results: {len(sep_df)} images")

# ─── Load matching HDF5 pre-computed values ────────────────────────────────────
indices = sep_df['Image Index'].values.astype(int)

with h5py.File(HDF5_PATH, 'r') as f:
    hdf5_ellipticity  = f['g_ellipticity'][indices]
    hdf5_major_axis   = f['g_major_axis'][indices]
    hdf5_isophotal    = f['g_isophotal_area'][indices]
    hdf5_sersic       = f['g_sersic_index'][indices]

# ─── Compare ───────────────────────────────────────────────────────────────────
metrics = [
    ('Ellipticity',    sep_df['Ellipticity'].values,   hdf5_ellipticity,  'g_ellipticity'),
    ('Semi-major Axis',sep_df['Semi-major Axis'].values,hdf5_major_axis,  'g_major_axis'),
    ('Isophotal Area', sep_df['Isophotal Area'].values, hdf5_isophotal,   'g_isophotal_area'),
    ('Sersic Index',   sep_df['Sersic Index'].values,   hdf5_sersic,      'g_sersic_index'),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, (name, sep_vals, hdf5_vals, hdf5_key) in zip(axes, metrics):
    # clip outliers for display
    sep_clean   = pd.Series(sep_vals).dropna()
    hdf5_clean  = pd.Series(hdf5_vals)[sep_clean.index]

    p95_sep  = np.percentile(sep_clean, 95)
    p95_hdf5 = np.percentile(hdf5_clean, 95)
    mask = (sep_clean < p95_sep) & (hdf5_clean < p95_hdf5)

    r, _ = pearsonr(sep_clean[mask], hdf5_clean[mask])

    ax.scatter(hdf5_clean[mask], sep_clean[mask], s=1, alpha=0.3, color='steelblue')
    lim = max(p95_sep, p95_hdf5)
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1)
    ax.set_xlabel(f'HDF5 ({hdf5_key})', fontsize=9)
    ax.set_ylabel(f'SEP ({name})', fontsize=9)
    ax.set_title(f'{name}\nr = {r:.3f}', fontsize=10)
    ax.set_xlim(0, p95_hdf5 * 1.05)
    ax.set_ylim(0, p95_sep  * 1.05)

plt.suptitle('SEP-extracted vs HDF5 pre-computed metrics (test set, g-band)', fontsize=12)
plt.tight_layout()
plt.savefig('./compare_sep_vs_hdf5.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: compare_sep_vs_hdf5.png")

# ─── Print mean comparison ────────────────────────────────────────────────────
print("\nMean value comparison:")
print(f"{'Metric':<20} {'SEP mean':>12} {'HDF5 mean':>12} {'Ratio SEP/HDF5':>15}")
print("-" * 62)
for name, sep_vals, hdf5_vals, _ in metrics:
    s = np.nanmean(sep_vals)
    h = np.nanmean(hdf5_vals)
    print(f"{name:<20} {s:>12.4f} {h:>12.4f} {s/h:>15.4f}")
