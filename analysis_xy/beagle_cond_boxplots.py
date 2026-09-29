#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""beagle_cond_boxplots.py — Paired three-condition comparisons for beagle dogs (boxplot style, all dog-side XY metrics).

Metric scope: all dog-side XY metrics (12); excludes any human-side metric
             (human speed / distance travelled, etc.), median speed, and any
             depth-derived metric (tail_elev / head_pitch / tail_up, etc.).

  Defined in S1 (4): dog mean speed / distance rate / wag frequency / wag amplitude
  Defined only in S2/S3 (8): dog–human mean distance / min distance / within-1m ratio /
                       head angle / gaze-at-human ratio / approach velocity /
                       following ratio / velocity alignment

Statistics: within-dog paired Wilcoxon signed-rank + Cohen dz; all evaluable
tests across the three-group comparison (34) jointly BH-corrected.
Outputs:
  figures/fig_box_s1vshuman.png/.pdf     (4 metrics, S1/S2/S3 boxplots)
  figures/fig_box_s2vs3.png/.pdf         (12 metrics, S2/S3 boxplots)
  figures/fig_sig_heatmap.png            (9 metrics x 3 contrasts, overwrites old figure)
  tables/beagle_s1vshuman_tests.csv      (overwrite: 4 metrics x 2 contrasts)
  tables/beagle_s2s3_tests.csv           (overwrite: 12 metrics)
  tables/beagle_scenario_sig_full.csv    (full table: 12 metrics x 3 contrasts)

Style: consistent with fig_social_feature_profiles_xy.png (boxplot + white-dot
     jitter scatter + stars at the top + q/dz annotation + white journal
     background), with larger font sizes.

Usage: python3 beagle_cond_boxplots.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from vis_common import setup_journal_style, save_fig  # noqa: E402

D_DATA = os.path.join(HERE, "data")
D_FIG = os.path.join(HERE, "figures")
D_TAB = os.path.join(HERE, "tables")

setup_journal_style()
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 10.5,
    "legend.fontsize": 11,
})

# ---- Metrics (all dog-side XY, no human-side / no depth-derived) ----
MET_S1 = [
    ("dog_mean_speed_mps", "Mean speed (m/s)"),
    ("dog_dist_rate_mpm", "Distance rate (m/min)"),
    ("tail_wag_freq_hz", "Wag frequency (Hz)"),
    ("tail_wag_amp_mean_mm", "Wag amplitude (mm)"),
]
MET_INT = [
    ("d_head_toe_mean_m", "Dog\u2013human distance (m)"),
    ("d_head_toe_min_m", "Min distance (m)"),
    ("d_head_toe_within1m_ratio", "Time within 1 m"),
    ("gaze_angle_mean_deg", "Head angle to human (\u00b0)"),
    ("gaze_ratio", "Gaze-at-human ratio"),
    ("approach_vel_mean", "Approach velocity (m/s)"),
    ("follow_ratio", "Following ratio"),
    ("vel_align_mean", "Velocity alignment (cos)"),
]
MET_ALL = MET_S1 + MET_INT
# Heatmap display subset (9 metrics): min distance / within-1m removed at user
# request (highly redundant with mean speed / mean distance);
# q values still come from joint BH over the full 12-metric set
# (consistent with manuscript Table 5 / the statistics tables).
MET_HEAT = [mm for mm in MET_ALL if mm[0] not in (
    "d_head_toe_min_m", "d_head_toe_within1m_ratio")]
# S2-vs-S3 boxplot display subset (8 metrics): at user request, wag frequency /
# wag amplitude are additionally removed from the heatmap display set, so the
# 2x4 grid has no empty panel; q values still come from joint BH over the full
# 12-metric set (consistent with manuscript Table 5 / the statistics tables).
MET_S23 = [mm for mm in MET_HEAT if mm[0] not in (
    "tail_wag_freq_hz", "tail_wag_amp_mean_mm")]
COND_COLOR = {"S1": "#8c8c8c", "S2": "#1f77b4", "S3": "#ff7f0e"}
XTICK3 = ["S1\nno human", "S2\ngaze away", "S3\ngaze at dog"]
XTICK2 = ["S2\ngaze away", "S3\ngaze at dog"]


def stars_q(q):
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "n.s."


def bh_q(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    out = np.empty(n)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def _fin(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _boxpanel(ax, groups, colors, rng, head=0.46):
    """Single panel in feature_profiles style: boxplot + white-dot jitter scatter. Returns data range."""
    vals_all = np.concatenate([_fin(g) for g in groups])
    bp = ax.boxplot(groups, positions=range(len(groups)), widths=0.55,
                    patch_artist=True,
                    medianprops={"color": "0.15", "lw": 1.1},
                    flierprops={"marker": "", "linewidth": 0})
    for pi, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[pi])
        patch.set_alpha(0.65)
    for xi, g in enumerate(groups):
        g = _fin(g)
        jit = rng.uniform(-0.10, 0.10, len(g))
        ax.scatter(xi + jit, g, s=15, c="white", edgecolors="0.15",
                   linewidths=0.5, zorder=10, alpha=0.95)
    vmax, vmin = float(vals_all.max()), float(vals_all.min())
    sp = max(vmax - vmin, 1e-9)
    ax.set_ylim(vmin - 0.03 * sp, vmax + head * sp)
    return vmax, vmin, sp


def _annot(ax, x, y, qv, dzv, sp, big=False):
    mark = stars_q(qv)
    ax.text(x, y + 0.035 * sp, mark, ha="center", va="bottom",
            fontsize=11 if mark != "n.s." else 9,
            fontweight="bold" if mark != "n.s." else "normal",
            color="0.1" if mark != "n.s." else "0.55")
    ax.text(x, y + 0.15 * sp,
            (f"q = {qv:.3f}" if qv >= 0.001 else "q < 0.001")
            + f", dz = {dzv:+.2f}",
            ha="center", va="bottom", fontsize=8.5, color="0.30")


def main():
    print("=" * 78)
    print("Beagle three-condition paired comparison (12 dog-side XY metrics, Wilcoxon + joint BH, boxplot style)")
    rm = pd.read_csv(os.path.join(D_DATA, "xy_rec_metrics.csv"),
                     dtype={"batch": str, "rec": str})
    fm = pd.read_csv(os.path.join(D_DATA, "xy_frame_metrics.csv"),
                     dtype={"batch": str, "rec": str},
                     usecols=["batch", "rec", "gaze_angle_deg", "gaze_at_human",
                              "approach_vel_mps"])

    # Frame-level aggregated extra columns: head angle / gaze ratio / approach velocity (defined only in S2/S3)
    gz = (fm.groupby(["batch", "rec"], as_index=False)
          .agg(gaze_angle_mean_deg=("gaze_angle_deg", "mean"),
               gaze_ratio=("gaze_at_human", "mean"),
               approach_vel_mean=("approach_vel_mps", "mean")))
    rm = rm.merge(gz, on=["batch", "rec"], how="left")
    bg = rm[(rm.breed == "beagle") & (rm.cond.isin(["S1", "S2", "S3"]))]
    wide = bg.pivot_table(index="dog_id", columns="cond",
                          values=[m for m, _ in MET_ALL])
    # Interaction metric columns that are all-NaN in S1 are dropped by pivot -> restore them as NaN columns
    want = pd.MultiIndex.from_product(
        [[m for m, _ in MET_ALL], ["S1", "S2", "S3"]])
    wide = wide.reindex(columns=want)
    n_dogs = len(wide)
    print(f"  Beagle dogs = {n_dogs}")

    # ---- Statistics: 12 metrics x 3 contrasts, paired Wilcoxon, 34 p-values jointly BH-corrected ----
    pairs = [("S1", "S2"), ("S1", "S3"), ("S2", "S3")]
    rows = []
    for m, _ in MET_ALL:
        for a, b in pairs:
            x = wide[(m, a)].to_numpy(float)
            y = wide[(m, b)].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            row = {"metric": m, "pair": f"{a} vs {b}",
                   "n_pairs": int(ok.sum()),
                   "median_S1": float(np.nanmedian(wide[(m, "S1")])),
                   "median_S2": float(np.nanmedian(wide[(m, "S2")])),
                   "median_S3": float(np.nanmedian(wide[(m, "S3")]))}
            if ok.sum() >= 5:
                d = x[ok] - y[ok]
                res = st.wilcoxon(x[ok], y[ok], zero_method="wilcox")
                sd = d.std(ddof=1)
                row["p_wilcoxon"] = float(res.pvalue)
                row["dz"] = float(d.mean() / sd) if sd > 1e-12 else np.nan
            else:
                row["p_wilcoxon"] = np.nan
                row["dz"] = np.nan
            rows.append(row)
    tst = pd.DataFrame(rows)
    mask = tst.p_wilcoxon.notna()
    tst.loc[mask, "q_bh"] = bh_q(tst.loc[mask, "p_wilcoxon"].to_numpy())
    tst.to_csv(os.path.join(D_TAB, "beagle_scenario_sig_full.csv"), index=False)
    print(f"  -> {os.path.join(D_TAB, 'beagle_scenario_sig_full.csv')}")
    for _, r in tst.iterrows():
        qv = r.q_bh if np.isfinite(r.get("q_bh", np.nan)) else float("nan")
        print(f"    {r['metric']:28s} {r['pair']:9s} n={r['n_pairs']:2d} "
              f"p={r['p_wilcoxon']:.4f} q={qv:.4f} dz={r['dz']:+.2f}"
              if np.isfinite(r.p_wilcoxon)
              else f"    {r['metric']:28s} {r['pair']:9s} n/a")

    # Split into the legacy table schemas (referenced by the manuscript): S1vshuman / S2vs3
    t1 = tst[tst.pair.isin(["S1 vs S2", "S1 vs S3"])].reset_index(drop=True)
    t1.to_csv(os.path.join(D_TAB, "beagle_s1vshuman_tests.csv"), index=False)
    t2 = tst[tst.pair == "S2 vs S3"].reset_index(drop=True)
    t2.to_csv(os.path.join(D_TAB, "beagle_s2s3_tests.csv"), index=False)
    print(f"  -> beagle_s1vshuman_tests.csv ({len(t1)} rows) / "
          f"beagle_s2s3_tests.csv ({len(t2)} rows)")

    rng = np.random.default_rng(7)

    # ---- Figure 1: S1 (no human) vs S2/S3 (human present), 4 metrics x 3 groups ----
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    for ax, (m, label) in zip(axes.ravel(), MET_S1):
        groups = [wide[(m, c)].dropna().to_numpy(float)
                  for c in ["S1", "S2", "S3"]]
        vmax, vmin, sp = _boxpanel(ax, groups,
                                   [COND_COLOR[c] for c in ["S1", "S2", "S3"]],
                                   rng, head=0.80)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(XTICK3)
        ax.set_title(label, loc="left")
        for lev, (a, b), (xa, xb) in [(0, ("S1", "S2"), (0, 1)),
                                      (1, ("S1", "S3"), (0, 2))]:
            rr = tst[(tst.metric == m) & (tst.pair == f"{a} vs {b}")].iloc[0]
            y_line = vmax + (0.10 + 0.32 * lev) * sp
            ax.plot([xa, xb], [y_line, y_line], color="0.25", lw=0.8)
            _annot(ax, (xa + xb) / 2, y_line, float(rr.q_bh), float(rr.dz), sp)
    for ax in axes.ravel()[len(MET_S1):]:
        ax.axis("off")
    fig.suptitle("Beagle dogs: S1 (no human) vs S2/S3 (human present) \u2014 "
                 "paired Wilcoxon signed-rank, BH-corrected",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_fig(fig, os.path.join(D_FIG, "fig_box_s1vshuman"))

    # ---- Figure 2: S2 vs S3, 8 metrics x 2 groups (wag frequency/amplitude removed) ----
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.6))
    for ax, (m, label) in zip(axes.ravel(), MET_S23):
        groups = [wide[(m, c)].dropna().to_numpy(float) for c in ["S2", "S3"]]
        vmax, vmin, sp = _boxpanel(ax, groups,
                                   [COND_COLOR[c] for c in ["S2", "S3"]], rng)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(XTICK2)
        ax.set_title(label, loc="left")
        rr = tst[(tst.metric == m) & (tst.pair == "S2 vs S3")].iloc[0]
        _annot(ax, 0.5, vmax + 0.06 * sp, float(rr.q_bh), float(rr.dz), sp)
    for ax in axes.ravel()[len(MET_S23):]:
        ax.axis("off")
    fig.suptitle("Beagle dogs: S2 (human gaze away) vs S3 (human gaze at dog) "
                 "\u2014 paired Wilcoxon signed-rank, BH-corrected",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_fig(fig, os.path.join(D_FIG, "fig_box_s2vs3"))

    # ---- Figure 3: significance heatmap (10 metrics x 3 contrasts, overwrites fig_sig_heatmap) ----
    qmat = np.full((len(MET_HEAT), 3), np.nan)
    for i, (m, _) in enumerate(MET_HEAT):
        for j, p_ in enumerate(["S1 vs S2", "S1 vs S3", "S2 vs S3"]):
            r = tst[(tst.metric == m) & (tst.pair == p_)]
            if len(r) and np.isfinite(r.q_bh.iloc[0]):
                qmat[i, j] = r.q_bh.iloc[0]
    fig, ax = plt.subplots(figsize=(0.98 * 3 + 3.6, 0.46 * len(MET_HEAT) + 1.8))
    with np.errstate(all="ignore"):
        neglog = -np.log10(qmat)
    neglog = np.clip(neglog, 0, 4)
    im = ax.imshow(neglog, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4)
    for i in range(len(MET_HEAT)):
        for j in range(3):
            q = qmat[i, j]
            if not np.isfinite(q):
                ax.text(j, i, "\u2013", ha="center", va="center",
                        fontsize=10, color="0.55")
            elif q < 0.05:
                ax.text(j, i, stars_q(q), ha="center", va="center",
                        fontsize=12, color="black", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"S1 vs S2", "S1 vs S3", "S2 vs S3"], fontsize=11)
    ax.set_yticks(range(len(MET_HEAT)))
    ax.set_yticklabels([lab for _, lab in MET_HEAT], fontsize=11)
    ax.set_title("Beagle scenario effects (paired Wilcoxon, BH-corrected)\n"
                 "cell = BH-corrected q over the full 12-metric set; "
                 "*, q<0.05; **, q<0.01; ***, q<0.001; "
                 "\u2013 = undefined (no human in S1)", fontsize=11.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MET_HEAT), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="both", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("\u2212log$_{10}$(q)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    save_fig(fig, os.path.join(D_FIG, "fig_sig_heatmap"))
    print("Done: fig_box_s1vshuman / fig_box_s2vs3 / fig_sig_heatmap (+pdf)")


if __name__ == "__main__":
    main()
