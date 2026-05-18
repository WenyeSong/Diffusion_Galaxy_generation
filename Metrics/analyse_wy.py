"""
Analysis script - reproduces paper Figure 3, Figure 4, and Table 1.
Also compares reproduced results with paper results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import sem

# ─── Load CSVs ─────────────────────────────────────────────────────────────────
TEST_CSV   = './testing_images_metrics_wy.csv'
GEN_CSV    = './generated_images_metrics_wy.csv'
ANDREW_CSV = './AndrewMetrics.csv'
ORIG_TEST  = './testing_metrics.csv'

test_df   = pd.read_csv(TEST_CSV)
gen_df    = pd.read_csv(GEN_CSV)
andrew_df = pd.read_csv(ANDREW_CSV)
orig_test = pd.read_csv(ORIG_TEST)

print(f"Reproduced real images:      {len(test_df)}")
print(f"Reproduced generated images: {len(gen_df)}")
print(f"Paper generated:             {len(andrew_df)}")
print(f"Original test set:           {len(orig_test)}")

METRICS = ['Ellipticity', 'Semi-major Axis', 'Sersic Index', 'Isophotal Area']

# Paper-style axis limits (matching Figure 3)
XLIMS = {
    'Ellipticity':    (0, 0.9),
    'Semi-major Axis': (0, 20),
    'Sersic Index':   (0, 6),
    'Isophotal Area': (0, 2500),
}

# Paper colors: blue = Real, red = DDPM
REAL_COLOR = '#4575b4'   # blue
GEN_COLOR  = '#d73027'   # red

# Comparison colors (wy vs Paper)
WY_COLOR     = '#d73027'  # same red as GEN_COLOR
PAPER_COLOR  = '#1a9641'  # green


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: YOUR RESULTS (Figure 3 & 4 style matching paper)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Figure 3: Distribution histograms (log-scale y-axis, paper style) ─────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.patch.set_facecolor('#f0f0f0')

for ax, metric in zip(axes, METRICS):
    real_data = test_df[metric].dropna()  # use SEP-extracted test metrics
    gen_data  = gen_df[metric].dropna()

    xmin, xmax = XLIMS[metric]
    bins = np.linspace(xmin, xmax, 50)

    ax.hist(real_data, bins=bins, alpha=0.7, color=REAL_COLOR,
            label='Real', density=True)
    ax.hist(gen_data,  bins=bins, alpha=0.7, color=GEN_COLOR,
            label='Reproduce', density=True)

    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 3 (Reproduced): Frequency distribution of morphological metrics',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('./figure3_reproduce.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_reproduce.png")


# ─── Figure 4: Mean metrics vs redshift bins (paper style) ─────────────────────
REDSHIFT_BINS = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
BIN_CENTERS   = 0.5 * (REDSHIFT_BINS[:-1] + REDSHIFT_BINS[1:])

# Y-axis limits matching paper
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
        r_bin = test_df[(test_df['Redshift'] >= lo) & (test_df['Redshift'] < hi)][metric].dropna()
        g_bin = gen_df[(gen_df['Redshift']   >= lo) & (gen_df['Redshift']  < hi)][metric].dropna()

        def clip_bin(b):
            if len(b) < 2:
                return b
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

plt.suptitle('Figure 4 (Reproduced): Mean morphological metrics vs redshift',
             fontsize=12)
plt.tight_layout()
plt.savefig('./figure4_reproduce.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure4_reproduce.png")


# ─── Table 1: metric ratios ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("Table 1 (Reproduced): Metric ratios — Generated / Real")
print("  Closer to 1.0 = better match")
print("="*65)

PAPER_RATIOS = {
    'Ellipticity':     0.98,
    'Semi-major Axis': 0.96,
    'Sersic Index':    0.93,
    'Isophotal Area':  0.90
}

rows = []
for metric in METRICS:
    real_mean = test_df[metric].dropna().mean()  # use SEP-extracted test metrics
    gen_mean  = gen_df[metric].dropna().mean()
    ratio     = gen_mean / real_mean if real_mean != 0 else np.nan
    rows.append({
        'Metric':           metric,
        'Real mean':        f'{real_mean:.4f}',
        'Reproduce mean':   f'{gen_mean:.4f}',
        'Reproduce Ratio':  f'{ratio:.4f}',
        'Paper (σ=0.1)':   f'{PAPER_RATIOS[metric]:.2f}'
    })

table_df = pd.DataFrame(rows)
print(table_df.to_string(index=False))
table_df.to_csv('./table1_reproduce.csv', index=False)
print("Saved: table1_reproduce.csv")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: COMPARISON WITH PAPER
# ══════════════════════════════════════════════════════════════════════════════

# ─── Figure 3 comparison: reproduced vs Andrew vs Real ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.patch.set_facecolor('#f0f0f0')

for ax, metric in zip(axes, METRICS):
    real_data = test_df[metric].dropna()  # SEP-extracted, same pipeline as generated
    wy_data   = gen_df[metric].dropna()
    and_data  = andrew_df[metric].dropna()

    xmin, xmax = XLIMS[metric]
    bins = np.linspace(xmin, xmax, 50)

    ax.hist(real_data, bins=bins, alpha=0.5, color=REAL_COLOR,
            label='Real', density=True)
    ax.hist(and_data,  bins=bins, alpha=0.6, color=PAPER_COLOR,
            label='Paper', density=True)
    ax.hist(wy_data,   bins=bins, alpha=0.6, color=WY_COLOR,
            label='Reproduce', density=True)

    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 3 Comparison: Reproduce vs Paper vs Real',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('./figure3_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_comparison.png")


# ─── Comparison Table: reproduce vs Andrew (both vs original test set) ──────────
print("\n" + "="*75)
print("Comparison Table: Reproduce vs Paper")
print("  Ratio = Generated / Real test set  |  Closer to 1.0 = better")
print("="*75)

rows2 = []
for metric in METRICS:
    orig_mean    = test_df[metric].dropna().mean()  # SEP-extracted
    paper_mean   = andrew_df[metric].dropna().mean()
    wy_mean      = gen_df[metric].dropna().mean()

    paper_ratio  = paper_mean  / orig_mean if orig_mean != 0 else np.nan
    wy_ratio     = wy_mean     / orig_mean if orig_mean != 0 else np.nan

    rows2.append({
        'Metric':          metric,
        'Real mean':       f'{orig_mean:.4f}',
        'Paper ratio':     f'{paper_ratio:.4f}',
        'Reproduce ratio': f'{wy_ratio:.4f}',
        'Paper (σ=0.1)':  f'{PAPER_RATIOS[metric]:.2f}'
    })

table2_df = pd.DataFrame(rows2)
print(table2_df.to_string(index=False))
table2_df.to_csv('./table_comparison.csv', index=False)
print("Saved: table_comparison.csv")
print("\nDone.")
