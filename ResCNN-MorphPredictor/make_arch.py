import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CI='#aec6cf'; CS='#b5ead7'; CR='#d4f5d4'
CG='#ffd700'; CF='#ffb347'; CO='#ff6b6b'

XC=1.75; BW=2.4
LG=0.07    # gap between layers (visible arrow space)
SG=0.11    # gap between stages
ADD_R=0.09 # add-circle radius

BH_IO  = 0.25  # input / output box
BH_FC  = 0.25  # fc / gap box
BH_STR = 0.24  # strided conv box
BH_RES = 0.22  # residual conv box

def draw_box(ax, y_bot, h, text, color, fs=5.4):
    ax.add_patch(mpatches.FancyBboxPatch(
        (XC-BW/2, y_bot), BW, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="#555", lw=0.7, zorder=2))
    ax.text(XC, y_bot+h/2, text, ha='center', va='center',
            fontsize=fs, color='#222', zorder=3, linespacing=1.25)

def arrow_down(ax, y_top, y_bot, x=XC):
    """Arrow pointing downward: from y_top (high) to y_bot (low)."""
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.9), zorder=1)

def draw_add(ax, y_cen):
    ax.add_patch(plt.Circle((XC, y_cen), ADD_R,
        facecolor='#fffde7', edgecolor='#555', lw=0.8, zorder=2))
    ax.text(XC, y_cen, '+', ha='center', va='center',
            fontsize=7.5, color='#333', fontweight='bold', zorder=3)

def draw_skip(ax, y_sc_top, y_add_cen):
    """Red skip: vertical line on right from strided-conv top DOWN to add circle,
       then horizontal arrow left into the circle."""
    sx = XC + BW/2 + 0.20
    # vertical line going down the right side
    ax.plot([sx, sx], [y_sc_top, y_add_cen], color='#c0392b', lw=1.8, zorder=1)
    # downward arrow tip at the add-circle level, going left into circle
    ax.annotate("", xy=(XC + ADD_R + 0.02, y_add_cen),
                xytext=(sx, y_add_cen),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8), zorder=1)
    # small downward arrowhead at top of skip line to show direction
    ax.annotate("", xy=(sx, y_sc_top - 0.06),
                xytext=(sx, y_sc_top),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8), zorder=1)

def stage_label(ax, y_mid, text):
    ax.text(0.05, y_mid, text, ha='left', va='center',
            fontsize=5.5, color='#555', style='italic')

# ── pass 1: compute total height ─────────────────────────────────
def compute_height():
    y = 0.20
    y += BH_IO + LG    # output → FC2
    y += BH_FC + LG    # FC2 → FC1
    y += BH_FC + LG    # FC1 → GAP
    y += BH_FC + SG    # GAP → stage 5 add circle
    for _ in range(5):
        y += ADD_R*2 + LG   # add → res2
        y += BH_RES + LG    # res2 → res1
        y += BH_RES + LG    # res1 → strided
        y += BH_STR + SG    # strided → next stage
    y += BH_IO + 0.20  # input box + top margin
    return y

YMAX = compute_height()
print(f"Total height: {YMAX:.2f}")

fig, ax = plt.subplots(figsize=(3.5, YMAX * 1.04))
ax.set_xlim(0, 3.5)
ax.set_ylim(0, YMAX)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── pass 2: build bottom → top, draw arrows pointing DOWN ────────
y = 0.20

# Output (bottom)
draw_box(ax, y, BH_IO, "Output  (4 params)", CO, fs=6.0)
prev_top = y + BH_IO
y = prev_top + LG

# FC 2
arrow_down(ax, y + BH_FC, prev_top)   # arrow from FC2 bottom → output top
draw_box(ax, y, BH_FC, "FC  128 → 4", CF)
prev_top = y + BH_FC
y = prev_top + LG

# FC 1
arrow_down(ax, y + BH_FC, prev_top)
draw_box(ax, y, BH_FC, "FC  256 → 128\nReLU  +  Dropout(0.3)", CF)
prev_top = y + BH_FC
y = prev_top + LG

# GAP
arrow_down(ax, y + BH_FC, prev_top)
draw_box(ax, y, BH_FC, "Global Average Pooling\n256 × 1 × 1", CG, fs=5.8)
prev_top = y + BH_FC
y = prev_top + SG

# 5 stages (bottom→top: stage 5 first)
for sn, ch in [(5,256),(4,256),(3,128),(2,64),(1,32)]:

    # Add circle
    y_add_cen = y + ADD_R
    arrow_down(ax, y_add_cen - ADD_R, prev_top)   # arrow from add-bottom → previous top
    draw_add(ax, y_add_cen)
    prev_top = y_add_cen + ADD_R
    y = prev_top + LG

    # Res conv 2
    arrow_down(ax, y + BH_RES, prev_top)
    draw_box(ax, y, BH_RES, f"conv 3×3, {ch}  |  BN", CR)
    prev_top = y + BH_RES
    y = prev_top + LG

    # Res conv 1
    arrow_down(ax, y + BH_RES, prev_top)
    draw_box(ax, y, BH_RES, f"conv 3×3, {ch}  |  BN + ReLU", CR)
    prev_top = y + BH_RES
    y = prev_top + LG

    # Strided conv (top of stage)
    y_sc_bot = y
    arrow_down(ax, y + BH_STR, prev_top)
    draw_box(ax, y, BH_STR, f"conv 3×3, {ch}, /2  |  BN + ReLU", CS, fs=5.6)
    y_sc_top = y + BH_STR
    prev_top = y_sc_top
    y = y_sc_top + SG

    # Skip: from strided conv top down to add circle
    draw_skip(ax, y_sc_top, y_add_cen)

    # Stage label on left (centered on this stage)
    stage_label(ax, (y_sc_bot + y_add_cen) / 2, f"Stage {sn}")

# Input (top)
arrow_down(ax, y + BH_IO, prev_top)
draw_box(ax, y, BH_IO, "Input   5 × 64 × 64", CI, fs=6.0)

# ── legend ───────────────────────────────────────────────────────
legend_els = [
    mpatches.Patch(facecolor=CI, edgecolor='#555', label='Input / Output'),
    mpatches.Patch(facecolor=CS, edgecolor='#555', label='Conv(stride=2) + BN + ReLU'),
    mpatches.Patch(facecolor=CR, edgecolor='#555', label='Conv + BN (+ ReLU)'),
    mpatches.Patch(facecolor=CG, edgecolor='#555', label='Global Avg Pooling'),
    mpatches.Patch(facecolor=CF, edgecolor='#555', label='FC + Dropout(0.3)'),
    Line2D([0],[0], color='#c0392b', lw=2, label='Skip connection'),
]
ax.legend(handles=legend_els, loc='lower center', fontsize=5.0,
          framealpha=0.9, ncol=2, bbox_to_anchor=(0.5, -0.01))

ax.set_title("MorphCNN Architecture", fontsize=8.5, fontweight='bold', pad=4)
plt.tight_layout()
plt.savefig("figures/architecture_vertical.png", dpi=160, bbox_inches='tight')
print("Saved.")
