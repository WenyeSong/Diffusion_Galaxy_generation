"""
Analysis using testing_metrics.csv (original paper's Real data) as the Real baseline.

Difference from analyse_wy.py:
- Figure 3 Real = testing_metrics.csv (40,914 images, possibly from HSC official catalog)
- Figure 4 Real = testing_images_metrics_wy.csv (has Redshift column, needed for binning)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

# ─── Load CSVs ─────────────────────────────────────────────────────────────────
PAPER_TEST_CSV = './testing_metrics.csv'                        # paper Real for Figure 3
SEP_TEST_CSV   = './testing_images_metrics_wy_fit_sersic.csv'   # our SEP Real for Figure 4
GEN_CSV        = './generated_images_metrics_wy_fit_sersic.csv' # our generated images
ANDREW_CSV     = './AndrewMetrics.csv'                           # paper's generated images

paper_test = pd.read_csv(PAPER_TEST_CSV)
sep_test   = pd.read_csv(SEP_TEST_CSV)
gen_df     = pd.read_csv(GEN_CSV)
paper_gen  = pd.read_csv(ANDREW_CSV)

print(f"Paper Real (testing_metrics.csv):  {len(paper_test)} images")
print(f"SEP Real (testing_images_wy.csv):  {len(sep_test)} images")
print(f"Our Generated:                     {len(gen_df)} images")
print(f"Paper Generated (AndrewMetrics):   {len(paper_gen)} images")

METRICS = ['Ellipticity', 'Semi-major Axis', 'Isophotal Area',
           'Ellipticity Proxy', 'Sersic Index (fitted)']

# AndrewMetrics uses 'Sersic Index' for what we call 'Ellipticity Proxy'
ANDREW_MAP = {
    'Ellipticity':          'Ellipticity',
    'Semi-major Axis':      'Semi-major Axis',
    'Isophotal Area':       'Isophotal Area',
    'Ellipticity Proxy':    'Sersic Index',
    'Sersic Index (fitted)': None,
}

XLIMS = {
    'Ellipticity':          (0, 0.9),
    'Semi-major Axis':      (0, 20),
    'Isophotal Area':       (0, 2500),
    'Ellipticity Proxy':    (0, 6),
    'Sersic Index (fitted)': (0, 6),
    'Sersic Index':         (0, 6),
}

import os
OUT_DIR = './figures_paper_real'
os.makedirs(OUT_DIR, exist_ok=True)

REAL_COLOR   = '#4575b4'   # blue
GEN_COLOR    = '#d73027'   # red (our reproduce)
WY_COLOR     = '#e07b00'   # orange (comparison)
PAPER_COLOR  = '#1a9641'   # green (paper's generated)


# ─── Figure 3A: Paper Real vs Our Reproduce ────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.patch.set_facecolor('#f0f0f0')

for ax, metric in zip(axes, METRICS):
    real_data = paper_test[metric].dropna()
    gen_data  = gen_df[metric].dropna()

    xmin, xmax = XLIMS[metric]
    bins = np.linspace(xmin, xmax, 50)

    ax.hist(real_data, bins=bins, alpha=0.7, color=REAL_COLOR,
            label='Real (paper test)', density=True)
    ax.hist(gen_data,  bins=bins, alpha=0.7, color=GEN_COLOR,
            label='Reproduce', density=True)

    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 3 (Paper Real): Reproduce vs Paper Real test set',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure3_paper_real.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_paper_real.png")


# ─── Figure 3B: Paper Real vs Paper Generated vs Our Reproduce ─────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.patch.set_facecolor('#f0f0f0')

for ax, metric in zip(axes, METRICS):
    real_data  = paper_test[metric].dropna()
    paper_data = paper_gen[metric].dropna()
    gen_data   = gen_df[metric].dropna()

    xmin, xmax = XLIMS[metric]
    bins = np.linspace(xmin, xmax, 50)

    ax.hist(real_data,  bins=bins, alpha=0.5, color=REAL_COLOR,
            label='Real (paper test)', density=True)
    ax.hist(paper_data, bins=bins, alpha=0.6, color=PAPER_COLOR,
            label='Paper Generated', density=True)
    ax.hist(gen_data,   bins=bins, alpha=0.6, color=GEN_COLOR,
            label='Reproduce', density=True)

    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 3 Comparison (Paper Real): Paper Real vs Paper Generated vs Reproduce',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure3_paper_real_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_paper_real_comparison.png")


# ─── Table 1: ratios using Paper Real as baseline ──────────────────────────────
print("\n" + "="*75)
print("Table 1 (Paper Real baseline): Generated / Paper Real")
print("  Closer to 1.0 = better")
print("="*75)

PAPER_RATIOS = {
    'Ellipticity':     0.98,
    'Semi-major Axis': 0.96,
    'Sersic Index':    0.93,
    'Isophotal Area':  0.90
}

rows = []
for metric in METRICS:
    real_mean  = paper_test[metric].dropna().mean()
    paper_mean = paper_gen[metric].dropna().mean()
    wy_mean    = gen_df[metric].dropna().mean()

    paper_ratio = paper_mean / real_mean if real_mean != 0 else np.nan
    wy_ratio    = wy_mean    / real_mean if real_mean != 0 else np.nan

    rows.append({
        'Metric':            metric,
        'Real mean':         f'{real_mean:.4f}',
        'Paper Gen mean':    f'{paper_mean:.4f}',
        'Reproduce mean':    f'{wy_mean:.4f}',
        'Paper Gen ratio':   f'{paper_ratio:.4f}',
        'Reproduce ratio':   f'{wy_ratio:.4f}',
        'Paper (σ=0.1)':    f'{PAPER_RATIOS[metric]:.2f}'
    })

table_df = pd.DataFrame(rows)
print(table_df.to_string(index=False))
table_df.to_csv(os.path.join(OUT_DIR, 'table1_paper_real.csv'), index=False)
print("Saved: table1_paper_real.csv")


# ─── Figure 4: Mean metrics vs redshift (std error bars, paper style) ──────────
# Uses SEP test data for Real (has Redshift), generated for Reproduce
REDSHIFT_BINS = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
BIN_CENTERS   = 0.5 * (REDSHIFT_BINS[:-1] + REDSHIFT_BINS[1:])

YLIMS_FIG4 = {
    'Ellipticity':     (-0.05, 0.70),
    'Semi-major Axis': (-0.5,  10),
    'Sersic Index':    (-0.3,  6.0),
    'Isophotal Area':  (-50,   2000),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.patch.set_facecolor('white')

for ax, metric in zip(axes, METRICS):
    real_means, real_stds = [], []
    gen_means,  gen_stds  = [], []

    for lo, hi in zip(REDSHIFT_BINS[:-1], REDSHIFT_BINS[1:]):
        r_bin = sep_test[(sep_test['Redshift'] >= lo) & (sep_test['Redshift'] < hi)][metric].dropna()
        g_bin = gen_df[(gen_df['Redshift']     >= lo) & (gen_df['Redshift']   < hi)][metric].dropna()

        # clip outliers to 5-95 percentile before computing std
        def clip_bin(b):
            if len(b) < 2:
                return b
            lo_p, hi_p = b.quantile(0.05), b.quantile(0.95)
            return b[(b >= lo_p) & (b <= hi_p)]

        r_clip = clip_bin(r_bin)
        g_clip = clip_bin(g_bin)

        real_means.append(r_bin.mean() if len(r_bin) > 0 else np.nan)
        gen_means.append( g_bin.mean() if len(g_bin) > 0 else np.nan)
        real_stds.append(r_clip.std()  if len(r_clip) > 1 else np.nan)
        gen_stds.append( g_clip.std()  if len(g_clip) > 1 else np.nan)

    real_means_arr = np.array(real_means)
    gen_means_arr  = np.array(gen_means)
    real_yerr = np.array(real_stds)
    gen_yerr  = np.array(gen_stds)

    gen_means_arr = np.array(gen_means)
    ax.errorbar(BIN_CENTERS, gen_means_arr,  yerr=gen_yerr,
                fmt='o', color=GEN_COLOR,  label='Reproduce',
                capsize=3, capthick=1.2, elinewidth=1.2, markersize=5)
    ax.errorbar(BIN_CENTERS, real_means_arr, yerr=real_yerr,
                fmt='o', color=REAL_COLOR, label='Real (SEP)',
                capsize=3, capthick=1.2, elinewidth=1.2, markersize=5)

    ax.set_xlabel('Redshift Bins', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlim(-0.1, 4.0)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_ylim(YLIMS_FIG4[metric])
    ax.legend(fontsize=9, loc='upper right')
    ax.set_facecolor('#eef2f7')
    ax.grid(True, alpha=0.4, color='white', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Figure 4 (Paper Real): Mean morphological metrics vs redshift (±1 std)',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure4_paper_real.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure4_paper_real.png")
print("\nDone.")
