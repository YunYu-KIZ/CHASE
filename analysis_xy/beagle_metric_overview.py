#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""beagle_metric_overview.py — Distribution overview of ALL 12 dog-side XY metrics
for 39 beagle dogs across the three experimental conditions (S1/S2/S3).

Purpose: one overview figure showing how each parameter is estimated per
condition (distribution shape + median/IQR + individual dogs).

  Row 1 (A-D): general activity metrics, defined in all three conditions
  Rows 2-3 (E-L): dog-human interaction metrics, defined only in S2/S3
                  (no human present in S1)

Style: violin (kernel density, cut=0, bounded at data range) + box
(median/IQR) + white jitter dots, consistent with the fig_box_* journal
style (vis_common.setup_journal_style, condition colors S1/S2/S3).

Inputs:
  data/xy_rec_metrics.csv                     (recording-level metrics)
  data/xy_frame_metrics.csv                   (frame-level gaze/approach, S2/S3)

Outputs:
  figures/fig_metric_overview.png/.pdf        (4 x 3 panel overview)
  tables/beagle_metric_distributions.csv      (n/mean/SD/median/Q1/Q3/min/max)

Usage: python3 beagle_metric_overview.py
"""
from __future__ import annotations

import os
import string
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from vis_common import setup_journal_style, save_fig  # noqa: E402

D_DATA = os.path.join(HERE, "data")
D_FIG = os.path.join(HERE, "figures")
D_TAB = os.path.join(HERE, "tables")

setup_journal_style()
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

# ---- All 12 dog-side XY metrics (no human-side, no depth-derived) ----
# scope="all": defined in S1/S2/S3; scope="int": defined only in S2/S3
METRICS = [
    ("dog_mean_speed_mps", "Mean speed (m/s)", "all"),
    ("dog_dist_rate_mpm", "Distance rate (m/min)", "all"),
    ("tail_wag_freq_hz", "Wag frequency (Hz)", "all"),
    ("tail_wag_amp_mean_mm", "Wag amplitude (mm)", "all"),
    ("d_head_toe_mean_m", "Dog\u2013human distance (m)", "int"),
    ("d_head_toe_min_m", "Min distance (m)", "int"),
    ("d_head_toe_within1m_ratio", "Time within 1 m", "int"),
    ("gaze_angle_mean_deg", "Head angle to human (\u00b0)", "int"),
    ("gaze_ratio", "Gaze-at-human ratio", "int"),
    ("approach_vel_mean", "Approach velocity (m/s)", "int"),
    ("follow_ratio", "Following ratio", "int"),
    ("vel_align_mean", "Velocity alignment (cos)", "int"),
]
COND_COLOR = {"S1": "#8c8c8c", "S2": "#1f77b4", "S3": "#ff7f0e"}
XTICK = {"all": ["S1\nno human", "S2\ngaze away", "S3\ngaze at dog"],
         "int": ["S2\ngaze away", "S3\ngaze at dog"]}
CONDS = {"all": ["S1", "S2", "S3"], "int": ["S2", "S3"]}


def _panel(ax, wide_m, scope, rng):
    """One metric panel: violin + box + white jitter dots. Returns n per condition."""
    conds = CONDS[scope]
    colors = {c: COND_COLOR[c] for c in conds}
    groups = [wide_m[c].dropna().to_numpy(float) for c in conds]
    ns = [len(g) for g in groups]

    # Violin (kernel density, cut=0 -> bounded at observed data range)
    df = pd.DataFrame({
        "cond": np.concatenate([[c] * len(g) for c, g in zip(conds, groups)]),
        "value": np.concatenate(groups) if groups else np.array([]),
    })
    sns.violinplot(x="cond", y="value", data=df, order=conds, hue="cond",
                   hue_order=conds, palette=colors, legend=False, cut=0,
                   density_norm="width", inner=None, linewidth=0.8,
                   saturation=1, ax=ax)
    for coll in ax.collections:
        coll.set_alpha(0.35)
    ax.set_ylabel("")

    # Box (median + IQR) on top of the violin
    bp = ax.boxplot(groups, positions=range(len(conds)), widths=0.40,
                    patch_artist=True,
                    medianprops={"color": "0.15", "lw": 1.1},
                    whiskerprops={"color": "0.25", "lw": 0.8},
                    capprops={"color": "0.25", "lw": 0.8},
                    flierprops={"marker": "", "linewidth": 0})
    for pi, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[conds[pi]])
        patch.set_alpha(0.85)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.8)

    # Individual dogs (white jitter dots)
    for xi, g in enumerate(groups):
        jit = rng.uniform(-0.09, 0.09, len(g))
        ax.scatter(xi + jit, g, s=10, c="white", edgecolors="0.15",
                   linewidths=0.45, zorder=10, alpha=0.95)

    vals_all = np.concatenate([g for g in groups if len(g)])
    vmax, vmin = float(vals_all.max()), float(vals_all.min())
    sp = max(vmax - vmin, 1e-9)
    ax.set_ylim(vmin - 0.05 * sp, vmax + 0.10 * sp)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(XTICK[scope])
    ax.set_xlabel("")
    return ns


def main():
    print("=" * 78)
    print("Beagle metric distribution overview (12 dog-side XY metrics, 3 conditions)")
    print(f"  Input : {os.path.join(D_DATA, 'xy_rec_metrics.csv')}")
    print(f"  Input : {os.path.join(D_DATA, 'xy_frame_metrics.csv')}")

    rm = pd.read_csv(os.path.join(D_DATA, "xy_rec_metrics.csv"),
                     dtype={"batch": str, "rec": str})
    fm = pd.read_csv(os.path.join(D_DATA, "xy_frame_metrics.csv"),
                     dtype={"batch": str, "rec": str},
                     usecols=["batch", "rec", "gaze_angle_deg", "gaze_at_human",
                              "approach_vel_mps"])

    # Frame-level aggregated columns (defined only in S2/S3, no human in S1)
    gz = (fm.groupby(["batch", "rec"], as_index=False)
          .agg(gaze_angle_mean_deg=("gaze_angle_deg", "mean"),
               gaze_ratio=("gaze_at_human", "mean"),
               approach_vel_mean=("approach_vel_mps", "mean")))
    rm = rm.merge(gz, on=["batch", "rec"], how="left")
    bg = rm[(rm.breed == "beagle") & (rm.cond.isin(["S1", "S2", "S3"]))]
    wide = bg.pivot_table(index="dog_id", columns="cond",
                          values=[m for m, _, _ in METRICS])
    want = pd.MultiIndex.from_product(
        [[m for m, _, _ in METRICS], ["S1", "S2", "S3"]])
    wide = wide.reindex(columns=want)
    n_dogs = len(wide)
    print(f"  Beagle dogs = {n_dogs} (S1/S2/S3 recordings = {len(bg)})")

    rng = np.random.default_rng(7)

    # ---- Overview figure: 4 columns x 3 rows (A-L) ----
    fig, axes = plt.subplots(3, 4, figsize=(11.8, 9.0))
    letters = list(string.ascii_uppercase[:len(METRICS)])
    rows = []
    for ax, (m, label, scope), letter in zip(axes.ravel(), METRICS, letters):
        wide_m = wide[m]
        ns = _panel(ax, wide_m, scope, rng)
        ax.set_title(label, loc="left")
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
        # Companion table rows: n / mean / SD / median / Q1 / Q3 / min / max
        for c, n in zip(CONDS[scope], ns):
            v = wide_m[c].dropna().to_numpy(float)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rows.append({"metric": m, "label": label, "cond": c, "n": n,
                         "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                         "median": float(med), "q1": float(q1), "q3": float(q3),
                         "iqr": float(q3 - q1),
                         "min": float(v.min()), "max": float(v.max())})

    handles = [mpatches.Patch(facecolor=COND_COLOR[c], alpha=0.65,
                              edgecolor="0.25", linewidth=0.8,
                              label={"S1": "S1  no human",
                                     "S2": "S2  human, gaze away",
                                     "S3": "S3  human, gaze at dog"}[c])
               for c in ["S1", "S2", "S3"]]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Beagle dogs (n = 39): distributions of all dog-side XY metrics "
                 "across the three experimental conditions",
                 fontsize=13, y=0.985)
    fig.text(0.01, 0.005,
             "Violins show kernel density estimates bounded at the observed data range; "
             "boxes show median and interquartile range; white dots show individual dogs "
             "(n = 39 per condition).\nDog\u2013human interaction metrics (E\u2013L) are "
             "undefined in S1 because no human was present.",
             fontsize=8.5, color="0.30", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 0.925))
    fig.savefig(os.path.join(D_FIG, "fig_metric_overview.png"), dpi=300)
    fig.savefig(os.path.join(D_FIG, "fig_metric_overview.pdf"))
    plt.close(fig)
    print(f"  Output: {os.path.join(D_FIG, 'fig_metric_overview.png')} / .pdf")

    dist = pd.DataFrame(rows)
    dist.to_csv(os.path.join(D_TAB, "beagle_metric_distributions.csv"), index=False)
    print(f"  Output: {os.path.join(D_TAB, 'beagle_metric_distributions.csv')} "
          f"({len(dist)} rows)")
    for _, r in dist.iterrows():
        print(f"    {r['metric']:28s} {r['cond']:2s} n={r['n']:2d} "
              f"median={r['median']:8.3f} IQR=[{r['q1']:.3f}, {r['q3']:.3f}] "
              f"range=[{r['min']:.3f}, {r['max']:.3f}]")
    print("Done.")


if __name__ == "__main__":
    main()
