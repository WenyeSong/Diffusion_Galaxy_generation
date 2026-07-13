"""
Analysis script - reproduces paper Figure 3, Figure 4, and Table 1.
Also compares reproduced results with paper results.
Updated: uses fit_sersic CSVs with Ellipticity Proxy + Sersic Index (fitted).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# ─── Output folder ─────────────────────────────────────────────────────────────
OUT_DIR = './figures_wy'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load CSVs ─────────────────────────────────────────────────────────────────
TEST_CSV   = './results/testing_images_metrics_wy_fit_sersic.csv'
GEN_CSV    = './results/generated_images_metrics_wy_fit_sersic.csv'
ANDREW_CSV = './results/AndrewMetrics.csv'
ORIG_TEST  = './results/testing_metrics.csv'

test_df   = pd.read_csv(TEST_CSV)
gen_df    = pd.read_csv(GEN_CSV)
andrew_df = pd.read_csv(ANDREW_CSV)
orig_test = pd.read_csv(ORIG_TEST)

print(f"Reproduced real images:      {len(test_df)}")
print(f"Reproduced generated images: {len(gen_df)}")
print(f"Paper generated:             {len(andrew_df)}")
print(f"Original test set:           {len(orig_test)}")

# 5 metrics: 3 original + Ellipticity Proxy + Sersic Index (fitted)
METRICS = ['Ellipticity', 'Semi-major Axis', 'Isophotal Area',
           'Ellipticity Proxy', 'Sersic Index (fitted)']

# AndrewMetrics uses 'Sersic Index' (same formula as our Ellipticity Proxy)
ANDREW_MAP = {
    'Ellipticity':          'Ellipticity',
    'Semi-major Axis':      'Semi-major Axis',
    'Isophotal Area':       'Isophotal Area',
    'Ellipticity Proxy':    'Sersic Index',   # same formula
    'Sersic Index (fitted)': None,            # not available in Andrew's data
}

PAPER_RATIOS = {
    'Ellipticity':     0.98,
    'Semi-major Axis': 0.96,
    'Isophotal Area':  0.90,
    'Ellipticity Proxy': 0.93,   # paper's Sersic Index
    'Sersic Index (fitted)': None,
}

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION: re-compute paper's Table 1 from their own CSVs
# score = mean(AndrewMetrics) / mean(testing_metrics)  ← paper's formula
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERIFICATION: Paper Table 1 — recomputed from AndrewMetrics.csv / testing_metrics.csv")
print("  (score = mean_generated / mean_real_test, closer to 1 = better)")
print("="*70)

# Column mapping: our metric name → column in testing_metrics.csv / AndrewMetrics.csv
PAPER_COL_MAP = {
    'Ellipticity':     ('Ellipticity',   'Ellipticity'),
    'Semi-major Axis': ('Semi-major Axis','Semi-major Axis'),
    'Isophotal Area':  ('Isophotal Area', 'Isophotal Area'),
    'Sersic Index':    ('Sersic Index',   'Sersic Index'),   # both CSVs use this name
}

print(f"\n{'Metric':<20} {'Real mean':>12} {'Gen mean':>12} {'Computed ratio':>16} {'Paper Table1':>13} {'Match?':>8}")
print("-" * 75)
for metric, (test_col, gen_col) in PAPER_COL_MAP.items():
    real_mean = orig_test[test_col].dropna().mean()  if test_col in orig_test.columns  else float('nan')
    gen_mean  = andrew_df[gen_col].dropna().mean()   if gen_col  in andrew_df.columns  else float('nan')
    ratio     = gen_mean / real_mean if real_mean > 0 else float('nan')
    paper_val = PAPER_RATIOS.get(metric if metric != 'Sersic Index' else 'Ellipticity Proxy', float('nan'))
    match     = '✓' if paper_val and abs(ratio - paper_val) < 0.02 else '✗'
    print(f"{metric:<20} {real_mean:>12.4f} {gen_mean:>12.4f} {ratio:>16.4f} {str(paper_val):>13} {match:>8}")

print("\nNote: 'Match?' uses ±0.02 tolerance. Mismatch may mean different CSV versions or rounding.")

XLIMS = {
    'Ellipticity':          (0, 0.9),
    'Semi-major Axis':      (0, 20),
    'Isophotal Area':       (0, 2500),
    'Ellipticity Proxy':    (0, 6),
    'Sersic Index (fitted)': (0, 6),
}

REAL_COLOR  = "#3717ec"   # blue
GEN_COLOR   = "#fa4b54" 
WY_COLOR    = "#fa4b54"  # orange (comparison figure, distinct from green)
PAPER_COLOR = "#F1D682" # red 


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: WY RESULTS (Figure 3 & 4)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Figure 3: Distribution histograms ────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.patch.set_facecolor('#f0f0f0')

for ax, metric in zip(axes, METRICS):
    real_data = test_df[metric].dropna()
    gen_data  = gen_df[metric].dropna()
    xmin, xmax = XLIMS[metric]
    bins = np.linspace(xmin, xmax, 50)
    ax.hist(real_data, bins=bins, alpha=0.7, color=REAL_COLOR, label='Real', density=True)
    ax.hist(gen_data,  bins=bins, alpha=0.7, color=GEN_COLOR,  label='Reproduce', density=True)
    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(metric, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 3 (Reproduced): Frequency distribution of morphological metrics',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure3_reproduce.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_reproduce.png")


# ─── Figure 4: Mean metrics vs redshift bins ───────────────────────────────────
REDSHIFT_BINS = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
BIN_CENTERS   = 0.5 * (REDSHIFT_BINS[:-1] + REDSHIFT_BINS[1:])

YLIMS_FIG4 = {
    'Ellipticity':          (-0.05, 0.70),
    'Semi-major Axis':      (-0.5,  10),
    'Isophotal Area':       (-50,   2000),
    'Ellipticity Proxy':    (-0.3,  6.0),
    'Sersic Index (fitted)': (-0.3,  6.0),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.patch.set_facecolor('white')

for ax, metric in zip(axes, METRICS):
    real_means, real_stds = [], []
    gen_means,  gen_stds  = [], []

    for lo, hi in zip(REDSHIFT_BINS[:-1], REDSHIFT_BINS[1:]):
        r_bin = test_df[(test_df['Redshift'] >= lo) & (test_df['Redshift'] < hi)][metric].dropna()
        g_bin = gen_df[(gen_df['Redshift']   >= lo) & (gen_df['Redshift']  < hi)][metric].dropna()

        def clip_bin(b):
            if len(b) < 2: return b
            lo_p, hi_p = b.quantile(0.05), b.quantile(0.95)
            return b[(b >= lo_p) & (b <= hi_p)]

        real_means.append(r_bin.mean() if len(r_bin) > 0 else np.nan)
        gen_means.append( g_bin.mean() if len(g_bin) > 0 else np.nan)
        real_stds.append(clip_bin(r_bin).std() if len(r_bin) > 1 else np.nan)
        gen_stds.append( clip_bin(g_bin).std() if len(g_bin) > 1 else np.nan)

    ax.errorbar(BIN_CENTERS, gen_means,  yerr=gen_stds,
                fmt='o', color=GEN_COLOR,  label='Reproduce',
                capsize=3, capthick=1.2, elinewidth=1.2, markersize=5)
    ax.errorbar(BIN_CENTERS, real_means, yerr=real_stds,
                fmt='o', color=REAL_COLOR, label='Real',
                capsize=3, capthick=1.2, elinewidth=1.2, markersize=5)
    ax.set_xlabel('Redshift Bins', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xlim(-0.1, 4.0)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_ylim(YLIMS_FIG4[metric])
    ax.legend(fontsize=8, loc='upper right')
    ax.set_facecolor('#eef2f7')
    ax.grid(True, alpha=0.4, color='white', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Figure 4 (Reproduced): Mean morphological metrics vs redshift', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure4_reproduce.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure4_reproduce.png")


# ─── Table 1 ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Table 1 (Reproduced): Metric ratios — Generated / Real")
print("="*70)

rows = []
for metric in METRICS:
    real_mean = test_df[metric].dropna().mean()
    gen_mean  = gen_df[metric].dropna().mean()
    ratio     = gen_mean / real_mean if real_mean != 0 else np.nan
    paper_ref = PAPER_RATIOS.get(metric)
    rows.append({
        'Metric':           metric,
        'Real mean':        f'{real_mean:.4f}',
        'Reproduce mean':   f'{gen_mean:.4f}',
        'Reproduce Ratio':  f'{ratio:.4f}',
        'Paper (σ=0.1)':   f'{paper_ref:.2f}' if paper_ref else 'N/A'
    })

table_df = pd.DataFrame(rows)
print(table_df.to_string(index=False))
table_df.to_csv(os.path.join(OUT_DIR, 'table1_reproduce.csv'), index=False)
print("Saved: table1_reproduce.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: COMPARISON WITH PAPER
# ══════════════════════════════════════════════════════════════════════════════

# ─── Figure 3 comparison ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.patch.set_facecolor('#f0f0f0')

for ax, metric in zip(axes, METRICS):
    real_data = test_df[metric].dropna()
    wy_data   = gen_df[metric].dropna()
    andrew_col = ANDREW_MAP.get(metric)

    xmin, xmax = XLIMS[metric]
    bins = np.linspace(xmin, xmax, 50)

    ax.hist(real_data, bins=bins, alpha=0.5, color=REAL_COLOR, label='Real', density=True)
    ax.hist(wy_data,   bins=bins, alpha=0.6, color=WY_COLOR,   label='Reproduce', density=True)
    if andrew_col and andrew_col in andrew_df.columns:
        and_data = andrew_df[andrew_col].dropna()
        ax.hist(and_data, bins=bins, alpha=0.5, color=PAPER_COLOR, label='Paper', density=True)

    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(metric, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 3 Comparison: Reproduce vs Paper vs Real', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure3_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_comparison.png")


# ─── Comparison Table ─────────────────────────────────────────────────────────
print("\n" + "="*75)
print("Comparison Table: Reproduce vs Paper")
print("="*75)

rows2 = []
for metric in METRICS:
    orig_mean  = test_df[metric].dropna().mean()
    wy_mean    = gen_df[metric].dropna().mean()
    wy_ratio   = wy_mean / orig_mean if orig_mean != 0 else np.nan
    andrew_col = ANDREW_MAP.get(metric)
    if andrew_col and andrew_col in andrew_df.columns:
        paper_mean  = andrew_df[andrew_col].dropna().mean()
        paper_ratio = paper_mean / orig_mean if orig_mean != 0 else np.nan
    else:
        paper_mean, paper_ratio = np.nan, np.nan
    rows2.append({
        'Metric':          metric,
        'Real mean':       f'{orig_mean:.4f}',
        'Paper ratio':     f'{paper_ratio:.4f}' if not np.isnan(paper_ratio) else 'N/A',
        'Reproduce ratio': f'{wy_ratio:.4f}',
        'Paper (σ=0.1)':  f'{PAPER_RATIOS[metric]:.2f}' if PAPER_RATIOS.get(metric) else 'N/A'
    })

table2_df = pd.DataFrame(rows2)
print(table2_df.to_string(index=False))
table2_df.to_csv(os.path.join(OUT_DIR, 'table_comparison.csv'), index=False)
print("Saved: table_comparison.csv")
print("\nDone. All figures saved to:", OUT_DIR)
