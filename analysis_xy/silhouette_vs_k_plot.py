# -*- coding: utf-8 -*-
"""Silhouette-vs-k line plot (k-means K sweep, k = 2..8, n = 39 beagles, 9 interaction features).

Style: journal white background, no top/right spines, outward thin ticks,
Liberation Sans (~Arial) 8 pt text / 9 pt labels, Okabe-Ito palette
(the k = 2 peak is highlighted in blue, the rest in grey), wide-low layout,
300 dpi PNG + vector PDF.
Usage: python3 analysis_xy/silhouette_vs_k_plot.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# Average silhouette for k = 2..8 (9-feature space after RobustScaler, KMeans n_init=20, seed=42)
K = list(range(2, 9))
SIL = [0.551, 0.321, 0.317, 0.324, 0.332, 0.318, 0.244]

# ---- Global style ----
rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

BLUE, GREY = "#0072B2", "#999999"   # Okabe-Ito blue + neutral grey

fig, ax = plt.subplots(figsize=(2.9, 2.0))   # wide-low layout, suits a Figure 4A inset

ax.plot(K, SIL, "-", color=GREY, lw=0.9, zorder=2)
ax.plot(K[1:], SIL[1:], "o", ms=3.5, mfc="white", mec=GREY, mew=0.8, zorder=3)
ax.plot(K[0], SIL[0], "o", ms=4.5, mfc=BLUE, mec=BLUE, zorder=4)
ax.annotate("k = 2\n0.55", xy=(K[0], SIL[0]), xytext=(2.55, 0.47),
            fontsize=8, color=BLUE, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE))

ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Average silhouette")
ax.set_xticks(K)
ax.set_ylim(0.0, 0.65)
ax.set_xlim(1.6, 8.4)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="both", which="major", pad=2)

fig.tight_layout(pad=0.4)
png = os.path.join(FIG, "fig_silhouette_vs_k.png")
pdf = os.path.join(FIG, "fig_silhouette_vs_k.pdf")
fig.savefig(png, dpi=300)
fig.savefig(pdf)
print(f"-> {png}\n-> {pdf}")
