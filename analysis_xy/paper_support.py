#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_support.py — SCI paper support analyses (beagle-only, XY 2D-only convention).

This script produces the following content for the XY paper (all restricted to
39 beagles / 117 recordings, excluding the pet-dog batch 20260910xjx; no
depth-derived metrics; depth back-projection is used only as an external
reference for fence-scale calibration):

  1) Per-keypoint DLC likelihood statistics (beagle) -> tables/paper_keypoint_likelihood.csv
     + figures/fig_likelihood_beagle.png (8 histograms: 4 dog + 4 human)
  2) Fence-scale calibration accuracy (beagle) -> tables/paper_scale_accuracy_beagle.csv
     + figures/fig_scale_accuracy_beagle.png (Bland-Altman, 4 panels)
  3) Beagle per-condition descriptive statistics (XY metrics, median [IQR])
     -> tables/paper_beagle_descriptives.csv
  4) UMAP sociability typing (9 XY features, depth-derived tail_up_ratio removed)
     -> umap/beagle_social_clusters_xy.csv, cluster_kmw_xy.csv,
        external_validity_s1_xy.csv, social_classification_stats_xy.json,
        fig_social_umap_xy.png, fig_social_feature_profiles_xy.png
     incl.: silhouette k selection / bootstrap ARI / per-condition S2, S3 ARI /
        S1 external validity / between-group MWU
  5) Representative XY timeseries figure (8 panels, no tail-pitch depth panel;
     panel 8 = dog-human movement-direction consistency)
     -> figures/fig_timeseries_example.png
  6) Representative beagle trajectory figures (one per S1/S2/S3)
     -> figures/fig_traj_examples_beagle.png
  7) Fence annotation check figure (4 beagle batches) -> figures/fig_fence_video_check_beagle.png

Usage: python3 paper_support.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Ellipse

import umap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from vis_common import setup_journal_style, save_fig, DOG_COLOR, HUMAN_COLOR  # noqa: E402

D_DATA = os.path.join(HERE, "data")
D_FIG = os.path.join(HERE, "figures")
D_TAB = os.path.join(HERE, "tables")
D_UMAP = os.path.join(HERE, "umap")
S0_ALL = os.path.join(os.path.dirname(HERE), "stage0_data/stage0_keypoints_xyz_all.csv")
FENCE_CSV = os.path.join(os.path.dirname(HERE), "stage0_ground/fence_corners.csv")
VIDEO_ROOT = "/home/yy/data/1-Circular-Fence-Test/depth/videos"

FPS = 30.0
DT = 1.0 / FPS
SEED = 42
PET_BATCH = "20260910xjx"
RGB_W, RGB_H = 1280, 720
GAZE_DEG = 30.0

setup_journal_style()
plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
})

COND_FULL = {"S1": "S1 (no human)", "S2": "S2 (human, gaze away)",
             "S3": "S3 (human, gaze at dog)"}
SOC_COLOR = {"human-oriented": "#ff7f0e", "less human-oriented": "#AA3377"}


def log(msg=""):
    print(msg, flush=True)


def _fin(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    out = np.empty(n)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out[order] = adj
    return out


def cliff_delta(x, y):
    x, y = _fin(x), _fin(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(int(np.sum(y < xv)) for xv in x)
    lt = sum(int(np.sum(y > xv)) for xv in x)
    return (gt - lt) / (len(x) * len(y))


def mwu_row(feat, label, xa, xb, in_clustering):
    u = st.mannwhitneyu(xa, xb, alternative="two-sided")
    return {"feature": feat, "label": label,
            "median_human_oriented": float(np.median(_fin(xa))),
            "median_less": float(np.median(_fin(xb))),
            "p_mwu": float(u.pvalue),
            "cliffs_delta": cliff_delta(xa, xb),
            "in_clustering": in_clustering}


def group_ellipse(ax, x, y, color, n_std=2.448):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return
    cx, cy = float(x.mean()), float(y.mean())
    cov = np.cov(np.stack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, fill=True,
                         facecolor=color, alpha=0.10, edgecolor=color,
                         lw=1.4, zorder=2))
    ax.scatter([cx], [cy], marker="o", s=70, c=color, edgecolors="white",
               linewidths=1.1, zorder=6)


# ======================================================================
# 1) Per-keypoint DLC likelihood statistics (beagle)
# ======================================================================
def task1_keypoint_likelihood():
    log("=" * 78)
    log("[1/7] Per-keypoint DLC likelihood statistics (beagle batches only)")
    log(f"  Input: {S0_ALL}")
    df = pd.read_csv(S0_ALL, dtype={"batch": str, "rec": str, "subject": str})
    df = df[df.batch != PET_BATCH]
    log(f"  Beagle rows = {len(df):,} "
        f"({df[['batch','rec']].drop_duplicates().shape[0]} recordings)")
    nice = {"occipital_protuberance": "Head (occiput)", "withers": "Withers",
            "tail_base": "Tail base", "tail_tip": "Tail tip",
            "left_toe_tip": "Left toe tip", "right_toe_tip": "Right toe tip",
            "left_shoulder": "Left shoulder", "right_shoulder": "Right shoulder"}
    rows = []
    for (subj, part), g in df.groupby(["subject", "body_part"]):
        c = g["confidence"].to_numpy(float)
        rows.append({
            "subject": subj, "body_part": part, "label": nice.get(part, part),
            "n_frames": int(len(c)),
            "mean": float(np.mean(c)), "median": float(np.median(c)),
            "sd": float(np.std(c)),
            "pct_ge_0.6": float((c >= 0.6).mean() * 100),
            "pct_ge_0.3": float((c >= 0.3).mean() * 100),
            "pct_lt_0.1": float((c < 0.1).mean() * 100),
            "pct_interpolated": float(g["interpolated"].mean() * 100)
            if "interpolated" in g else np.nan,
            "pct_jump_removed": float(g["jump_removed"].mean() * 100)
            if "jump_removed" in g else np.nan,
        })
    t = pd.DataFrame(rows)
    t = t.sort_values(["subject", "body_part"])
    out = os.path.join(D_TAB, "paper_keypoint_likelihood.csv")
    t.to_csv(out, index=False)
    log(f"  -> {out}")
    for _, r in t.iterrows():
        log(f"    {r['subject']:6s} {r['label']:16s} n={r['n_frames']:>7,} "
            f"mean={r['mean']:.3f} median={r['median']:.3f} "
            f">=0.6: {r['pct_ge_0.6']:5.1f}%  >=0.3: {r['pct_ge_0.3']:5.1f}%")

    # Figure: 2x4 histograms (top row dog, bottom row human)
    order = {"dog": ["occipital_protuberance", "withers", "tail_base", "tail_tip"],
             "human": ["left_toe_tip", "right_toe_tip",
                       "left_shoulder", "right_shoulder"]}
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.2))
    for i, subj in enumerate(["dog", "human"]):
        for j, part in enumerate(order[subj]):
            ax = axes[i, j]
            c = df[(df.subject == subj) & (df.body_part == part)][
                "confidence"].to_numpy(float)
            ax.hist(c, bins=40, range=(0, 1), color=DOG_COLOR if subj == "dog"
                    else HUMAN_COLOR, alpha=0.85, edgecolor="white",
                    linewidth=0.2)
            ax.axvline(0.6, color="0.2", lw=0.9, ls="--")
            med = float(np.median(c))
            ax.axvline(med, color="#AA3377", lw=0.9)
            ax.set_title(nice.get(part, part), fontsize=9, loc="left")
            ax.text(0.97, 0.60, f"median {med:.2f}\n≥0.6: {(c >= 0.6).mean()*100:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                    color="0.25")
            ax.text(0.97, 0.97, f"n = {len(c):,}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="0.4")
            if j == 0:
                ax.set_ylabel("Frames" if i == 0 else "")
            ax.set_xlabel("DLC likelihood")
    fig.suptitle("Per-keypoint DeepLabCut likelihood, beagle recordings "
                 f"(n = {len(df):,} keypoint-frames)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, os.path.join(D_FIG, "fig_likelihood_beagle"))
    return t


# ======================================================================
# 2) Fence-scale calibration accuracy (beagle)
# ======================================================================
def task2_scale_accuracy(rm):
    log("=" * 78)
    log("[2/7] Fence-scale calibration accuracy (beagle only, primary fence scale vs depth back-projection reference)")
    keep = rm[(~rm.excluded) & (rm.breed == "beagle")].copy()

    def rel_bias(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
        out = np.full(len(a), np.nan)
        out[m] = (a[m] - b[m]) / np.abs(b[m]) * 100.0
        return out

    pairs = [
        ("dog_mean_speed_mps", "dog_mean_speed_mps_depth", "Dog mean speed (m/s)"),
        ("dog_cum_dist_m", "dog_cum_dist_m_depth", "Dog cumulative distance (m)"),
        ("d_head_toe_mean_m", "d_head_toe_mean_m_depth", "Head-toe distance (m)"),
        ("tail_wag_amp_mean_mm", "tail_wag_amp_mean_mm_depth", "Tail wag amplitude (mm)"),
        ("human_toe_mean_speed_mps", "human_toe_mean_speed_mps_depth",
         "Human toe speed (m/s)"),
    ]
    rows, figdata = [], {}
    for main_c, ref_c, label in pairs:
        b = rel_bias(keep[main_c].to_numpy(), keep[ref_c].to_numpy())
        bf = _fin(b)
        a = keep[main_c].to_numpy(float)
        r_ = keep[ref_c].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(r_)
        r_p = float(st.pearsonr(a[m], r_[m])[0])
        rows.append({"metric": label, "n": len(bf),
                     "bias_median_pct": round(float(np.median(bf)), 2),
                     "bias_iqr_pct": round(float(np.subtract(
                         *np.percentile(bf, [75, 25]))), 2),
                     "mape_pct": round(float(np.mean(np.abs(bf))), 2),
                     "max_abs_bias_pct": round(float(np.max(np.abs(bf))), 2),
                     "pearson_r": round(r_p, 4)})
        figdata[label] = (a, r_, b)
    t = pd.DataFrame(rows)
    out = os.path.join(D_TAB, "paper_scale_accuracy_beagle.csv")
    t.to_csv(out, index=False)
    log(f"  -> {out}")
    log(t.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0))
    for ax, (label, (a, r_, b)) in zip(axes.ravel(), list(figdata.items())[:4]):
        m = np.isfinite(a) & np.isfinite(r_) & np.isfinite(b)
        ax.scatter((a[m] + r_[m]) / 2, b[m], s=7, c=DOG_COLOR, alpha=0.6,
                   edgecolor="none")
        ax.axhline(0, color="0.3", lw=0.7)
        med, sd = float(np.median(b[m])), float(np.std(b[m]))
        ax.axhline(med, color="#EE6677", lw=0.9, ls="--",
                   label=f"median bias {med:+.1f}%")
        ax.axhline(med + 1.96 * sd, color="0.6", lw=0.6, ls=":")
        ax.axhline(med - 1.96 * sd, color="0.6", lw=0.6, ls=":",
                   label=f"95% LoA ±{1.96 * sd:.1f}%")
        ax.set_xlabel(f"{label} (two-scale mean)")
        ax.set_ylabel("Fence vs reference bias (%)")
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Fence-scale calibration (6 panels × 0.90 m) vs metric "
                 "back-projection reference, beagle recordings (Bland–Altman)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, os.path.join(D_FIG, "fig_scale_accuracy_beagle"))
    return t


# ======================================================================
# 3) Beagle per-condition descriptive statistics (XY metrics)
# ======================================================================
def task3_descriptives(rm, rm2):
    log("=" * 78)
    log("[3/7] Beagle per-condition descriptive statistics (XY metrics, median [IQR])")
    bg = rm2[(rm2.breed == "beagle")].copy()
    dog_metrics = [
        ("dog_mean_speed_mps", "Dog mean speed (m/s)"),
        ("dog_median_speed_mps", "Dog median speed (m/s)"),
        ("dog_dist_rate_mpm", "Dog distance rate (m/min)"),
        ("tail_wag_amp_mean_mm", "Tail wag amplitude (mm)"),
        ("tail_wag_freq_hz", "Tail wag frequency (Hz)"),
    ]
    human_metrics = [
        ("human_toe_mean_speed_mps", "Human toe mean speed (m/s)"),
        ("human_toe_dist_rate_mpm", "Human toe distance rate (m/min)"),
        ("d_head_toe_mean_m", "Dog–human distance, mean (m)"),
        ("d_head_toe_median_m", "Dog–human distance, median (m)"),
        ("d_head_toe_min_m", "Dog–human distance, min (m)"),
        ("d_head_toe_within1m_ratio", "Time within 1 m of human"),
        ("gaze_ratio", "Gaze-at-human ratio (<30°)"),
        ("gaze_angle_mean_deg", "Head angle to human, mean (°)"),
        ("follow_ratio", "Following ratio"),
        ("vel_align_mean", "Velocity alignment (cos)"),
    ]
    rows = []
    for scope, metrics, conds in [
        ("dog", dog_metrics, ["S1", "S2", "S3"]),
        ("human", human_metrics, ["S2", "S3"]),
    ]:
        for m, label in metrics:
            row = {"scope": scope, "metric": m, "label": label}
            for c in conds:
                v = _fin(bg[bg.cond == c][m])
                row[f"n_{c}"] = len(v)
                row[f"median_{c}"] = float(np.median(v)) if len(v) else np.nan
                row[f"q1_{c}"] = float(np.percentile(v, 25)) if len(v) else np.nan
                row[f"q3_{c}"] = float(np.percentile(v, 75)) if len(v) else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    out = os.path.join(D_TAB, "paper_beagle_descriptives.csv")
    t.to_csv(out, index=False)
    log(f"  -> {out}")
    for _, r in t.iterrows():
        parts = []
        for c in (["S1", "S2", "S3"] if r["scope"] == "dog" else ["S2", "S3"]):
            if np.isfinite(r.get(f"median_{c}", np.nan)):
                parts.append(f"{c}: {r[f'median_{c}']:.3f} "
                             f"[{r[f'q1_{c}']:.3f}, {r[f'q3_{c}']:.3f}]")
        log(f"    {r['label']:32s} " + " | ".join(parts))
    return t


# ======================================================================
# 4) UMAP sociability typing (9 XY features)
# ======================================================================
SOC_FEATS_XY = [
    "d_head_toe_mean_m", "d_head_toe_min_m", "d_head_toe_within1m_ratio",
    "follow_ratio", "vel_align_mean", "gaze_ratio", "approach_vel_mean",
    "tail_wag_amp_mean_mm", "dog_dist_rate_mpm",
]
SOC_FEAT_LABEL = {
    "d_head_toe_mean_m": "Dog–human distance (m)",
    "d_head_toe_min_m": "Minimum distance (m)",
    "d_head_toe_within1m_ratio": "Proximity (<1 m) ratio",
    "follow_ratio": "Following ratio",
    "vel_align_mean": "Velocity alignment",
    "gaze_ratio": "Gaze-at-human ratio",
    "approach_vel_mean": "Approach velocity (m/s)",
    "tail_wag_amp_mean_mm": "Tail wag amplitude (mm)",
    "dog_dist_rate_mpm": "Dog distance rate (m/min)",
}
S1_FEATS_XY = ["dog_mean_speed_mps", "dog_dist_rate_mpm",
               "tail_wag_amp_mean_mm", "tail_wag_freq_hz"]


def task4_social_classification(rm2):
    log("=" * 78)
    log("[4/7] UMAP sociability typing (39 beagles, S2/S3 means, 9 XY features)")
    bg = rm2[(rm2.breed == "beagle") & (rm2.cond.isin(["S2", "S3"]))]
    dog_lvl = bg.groupby("dog_id", as_index=False)[SOC_FEATS_XY].mean()
    log(f"  Beagle dogs = {len(dog_lvl)}, features = {len(SOC_FEATS_XY)}")
    Xd = dog_lvl[SOC_FEATS_XY].to_numpy(float)
    Xds = RobustScaler().fit_transform(Xd)

    # UMAP embedding (visualization) + KMeans (in scaled space, k chosen by silhouette)
    Ed = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=SEED
                   ).fit_transform(Xds)
    sils = {}
    for k in range(2, 9):   # k = 2..8 (n=39 dogs, 9 features)
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(Xds)
        sils[k] = float(silhouette_score(Xds, lab))
        log(f"    k={k}: silhouette={sils[k]:.3f}")
    best_k = max(sils, key=sils.get)
    lab = KMeans(best_k, n_init=20, random_state=SEED).fit_predict(Xds)
    dog_lvl["cluster"] = lab
    dog_lvl["umap_x"], dog_lvl["umap_y"] = Ed[:, 0], Ed[:, 1]

    prof = dog_lvl.groupby("cluster")[SOC_FEATS_XY].mean()
    score = (prof["d_head_toe_within1m_ratio"] + prof["follow_ratio"]
             + prof["gaze_ratio"] - prof["d_head_toe_mean_m"])
    friendly = score.idxmax()
    dog_lvl["social_type"] = np.where(dog_lvl.cluster == friendly,
                                     "human-oriented", "less human-oriented")
    n_ho = int((dog_lvl.social_type == "human-oriented").sum())
    log(f"  Selected k={best_k} (silhouette={sils[best_k]:.3f}); "
        f"human-oriented {n_ho} / less {len(dog_lvl) - n_ho}")
    out_c = os.path.join(D_UMAP, "beagle_social_clusters_xy.csv")
    dog_lvl.sort_values(["social_type", "dog_id"]).to_csv(out_c, index=False)
    log(f"  -> {out_c}")

    # ---- Check 1: bootstrap stability (ARI, 200 resamples at 80%) ----
    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(200):
        idx = rng.choice(len(Xds), size=int(0.8 * len(Xds)), replace=True)
        lab_b = KMeans(best_k, n_init=20, random_state=SEED).fit_predict(Xds[idx])
        aris.append(adjusted_rand_score(lab[idx], lab_b))
    ari_boot_mean, ari_boot_sd = float(np.mean(aris)), float(np.std(aris))
    log(f"  [Check 1] bootstrap ARI = {ari_boot_mean:.3f} ± {ari_boot_sd:.3f}")

    # ---- Check 2: per-condition (S2/S3) clustering consistency ----
    ari_cond = {}
    for cond in ["S2", "S3"]:
        sub = rm2[(rm2.breed == "beagle") & (rm2.cond == cond)]
        dl = sub.groupby("dog_id", as_index=False)[SOC_FEATS_XY].mean()
        dl = dl.merge(dog_lvl[["dog_id", "social_type"]], on="dog_id")
        Xc = RobustScaler().fit_transform(dl[SOC_FEATS_XY].to_numpy(float))
        lab_c = KMeans(2, n_init=20, random_state=SEED).fit_predict(Xc)
        sc = (pd.DataFrame({"c": lab_c, "t": dl.social_type})
              .groupby("c")["t"].apply(lambda s: (s == "human-oriented").mean()))
        flip = sc.idxmax()
        ari_cond[cond] = float(adjusted_rand_score(
            dl.social_type == "human-oriented", lab_c == flip))
        log(f"  [Check 2] {cond}-only clustering vs primary typing: ARI = {ari_cond[cond]:.3f}")

    # ---- Check 3: S1 (no-human) baseline external validity (4 XY features, not used for typing) ----
    s1 = (rm2[(rm2.breed == "beagle") & (rm2.cond == "S1")]
          .groupby("dog_id", as_index=False)[S1_FEATS_XY].mean())
    s1 = s1.merge(dog_lvl[["dog_id", "social_type"]], on="dog_id")
    a1 = s1[s1.social_type == "human-oriented"]
    b1 = s1[s1.social_type != "human-oriented"]
    rows = []
    for f in S1_FEATS_XY:
        r = mwu_row(f, {"dog_mean_speed_mps": "Dog mean speed (m/s)",
                        "dog_dist_rate_mpm": "Dog distance rate (m/min)",
                        "tail_wag_amp_mean_mm": "Tail wag amplitude (mm)",
                        "tail_wag_freq_hz": "Tail wag frequency (Hz)"}[f],
                    a1[f], b1[f], False)
        rows.append(r)
    ext = pd.DataFrame(rows)
    ext["q_bh"] = bh(ext.p_mwu.to_numpy())
    out_e = os.path.join(D_UMAP, "external_validity_s1_xy.csv")
    ext.to_csv(out_e, index=False)
    log(f"  [Check 3] S1 external validity (XY features):")
    for _, r in ext.iterrows():
        log(f"    {r['label']:26s} p={r['p_mwu']:.4f} q={r['q_bh']:.4f} "
            f"δ={r['cliffs_delta']:+.2f}")
    log(f"  -> {out_e}")

    # ---- Between-group comparison (9 typing features, constructive differences, for the feature profiles) ----
    a = dog_lvl[dog_lvl.social_type == "human-oriented"]
    b = dog_lvl[dog_lvl.social_type != "human-oriented"]
    rows = [mwu_row(f, SOC_FEAT_LABEL[f], a[f], b[f], True)
            for f in SOC_FEATS_XY]
    kmw = pd.DataFrame(rows)
    kmw["q_bh"] = bh(kmw.p_mwu.to_numpy())
    out_k = os.path.join(D_UMAP, "cluster_kmw_xy.csv")
    kmw.to_csv(out_k, index=False)
    log(f"  -> {out_k}")
    for _, r in kmw.iterrows():
        log(f"    {r['label']:26s} p={r['p_mwu']:.2e} q={r['q_bh']:.4f} "
            f"δ={r['cliffs_delta']:+.2f}")

    stats_json = {
        "n_dogs": int(len(dog_lvl)), "n_features": len(SOC_FEATS_XY),
        "silhouette_by_k": {str(k): v for k, v in sils.items()},
        "best_k": int(best_k), "best_silhouette": sils[best_k],
        "n_human_oriented": n_ho,
        "n_less_human_oriented": int(len(dog_lvl) - n_ho),
        "bootstrap_ari_mean": ari_boot_mean, "bootstrap_ari_sd": ari_boot_sd,
        "ari_s2_only": ari_cond["S2"], "ari_s3_only": ari_cond["S3"],
        "umap_params": "n_neighbors=10, min_dist=0.2, seed=42",
    }
    out_j = os.path.join(D_UMAP, "social_classification_stats_xy.json")
    with open(out_j, "w") as f:
        json.dump(stats_json, f, indent=2)
    log(f"  -> {out_j}")

    # ---- Figure: UMAP scatter ----
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for t_ in ["human-oriented", "less human-oriented"]:
        m = dog_lvl[dog_lvl.social_type == t_]
        ax.scatter(m.umap_x, m.umap_y, s=30, c=SOC_COLOR[t_], label=t_,
                   linewidths=0.5, edgecolors="white", zorder=4)
        group_ellipse(ax, m.umap_x, m.umap_y, SOC_COLOR[t_])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Beagle sociability types (UMAP of 9 XY features, "
                 f"S2/S3 means; k-means k={best_k})", fontsize=9, loc="left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    save_fig(fig, os.path.join(D_FIG, "fig_social_umap_xy"))

    # ---- Figure: between-group feature comparison, 3x3 small multiples ----
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.0))
    rngp = np.random.default_rng(7)
    for ax, f in zip(axes.ravel(), SOC_FEATS_XY):
        kmr = kmw[kmw.feature == f].iloc[0]
        bp = ax.boxplot([_fin(a[f]), _fin(b[f])], positions=[0, 1],
                        widths=0.55, patch_artist=True,
                        medianprops={"color": "0.15", "lw": 1.0},
                        flierprops={"marker": "", "linewidth": 0})
        for pi, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(SOC_COLOR["human-oriented" if pi == 0
                                           else "less human-oriented"])
            patch.set_alpha(0.65)
        for xi, vals in [(0, _fin(a[f])), (1, _fin(b[f]))]:
            jit = rngp.uniform(-0.10, 0.10, len(vals))
            ax.scatter(xi + jit, vals, s=13, c="white", edgecolors="0.15",
                       linewidths=0.5, zorder=10, alpha=0.95)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HO", "LHO"], fontsize=8.5)
        qv, dv = float(kmr.q_bh), float(kmr.cliffs_delta)
        mark = ("***" if qv < 0.001 else "**" if qv < 0.01
                else "*" if qv < 0.05 else "n.s.")
        vals_all = np.concatenate([_fin(a[f]), _fin(b[f])])
        vmax, vmin = float(vals_all.max()), float(vals_all.min())
        sp = max(vmax - vmin, 1e-9)
        ax.set_ylim(vmin - 0.03 * sp, vmax + 0.30 * sp)
        ax.text(0.5, vmax + 0.06 * sp, mark, ha="center", va="bottom",
                fontsize=10 if mark != "n.s." else 7.5,
                fontweight="bold" if mark != "n.s." else "normal",
                color="0.1" if mark != "n.s." else "0.55")
        ax.text(0.5, vmax + 0.17 * sp, f"\u03b4 = {dv:+.2f}", ha="center",
                va="bottom", fontsize=7.5, color="0.30")
        ax.set_title(SOC_FEAT_LABEL[f], fontsize=8.5, loc="left")
        ax.tick_params(axis="y", labelsize=7.5)
    fig.suptitle("Sociability groups: feature profiles "
                 "(HO = human-oriented, LHO = less human-oriented; "
                 "Mann–Whitney + BH; \u03b4 = Cliff's delta)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, os.path.join(D_FIG, "fig_social_feature_profiles_xy"))
    return stats_json


# ======================================================================
# 5) Representative XY timeseries figure (8 panels, no tail-pitch depth panel)
# ======================================================================
def task6_timeseries_example(fm, rm):
    log("=" * 78)
    log("[5/7] Representative XY timeseries figure (beagle S3, 8 panels)")
    cand = rm[(rm.breed == "beagle") & (rm.cond == "S3")
              & (rm.tail_vis_ratio >= 0.80) & (~rm.excluded)].copy()
    med = cand.follow_ratio.median()
    cand["d_follow"] = (cand.follow_ratio - med).abs()
    cand = cand.sort_values("d_follow")
    r = cand.iloc[0]
    log(f"  Selected: {r['batch']} / {r['rec']} (follow_ratio={r['follow_ratio']:.2f}, "
        f"duration {r['duration_s']:.0f}s)")
    g = fm[fm.rec == r["rec"]].sort_values("frame").copy()
    twin = (r["win_start"], r["win_end"])
    if np.isfinite(twin[0]) and np.isfinite(twin[1]):
        g = g[(g.frame >= twin[0]) & (g.frame <= twin[1])]
    t = (g["frame"].to_numpy(float) - float(g["frame"].iloc[0])) / FPS

    fig, axes = plt.subplots(8, 1, figsize=(6.8, 11.6), sharex=True)
    fig.subplots_adjust(hspace=0.75, top=0.95)

    ax = axes[0]
    ax.plot(t, g["dog_speed_mps"], lw=0.7, color=DOG_COLOR, label="dog")
    ax.plot(t, g["toe_speed_mps"], lw=0.7, color=HUMAN_COLOR,
            label="human (left toe)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Instantaneous speed", fontsize=8.5, loc="left")

    ax = axes[1]
    ax.plot(t, g["dog_cum_m"], lw=0.9, color=DOG_COLOR, label="dog")
    ax.plot(t, g["toe_cum_m"], lw=0.9, color=HUMAN_COLOR,
            label="human (left toe)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylabel("Cum. dist (m)")
    ax.set_title("Cumulative distance", fontsize=8.5, loc="left")

    ax = axes[2]
    ax.plot(t, g["d_head_toe_m"], lw=0.7, color="#AA3377")
    for y, ls in [(0.5, "--"), (1.0, "-."), (2.0, ":")]:
        ax.axhline(y, color="0.75", lw=0.6, ls=ls)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Dist (m)")
    ax.set_title("Dog head–human toe distance", fontsize=8.5, loc="left")

    ax = axes[3]
    ax.plot(t, g["tail_wag_freq_inst_hz"], lw=0.7, color="#228833")
    ax.set_ylim(0, 8)
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("Tail wag frequency (zero-crossing, pp \u2265 15 mm gate)",
                 fontsize=8.5, loc="left")

    ax = axes[4]
    off = g["tail_lat_dev_mm"].to_numpy(float)
    ax.plot(t, off, lw=0.5, color="#CC6677", alpha=0.9)
    ax.axhline(0, color="0.75", lw=0.6, ls=":")
    lim = float(np.nanmax(np.abs(off))) if np.any(np.isfinite(off)) else 1.0
    lim = max(lim, 5.0) * 1.08
    ax.set_ylim(-lim, lim)
    ax.text(0.99, 0.96, "left (+)", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color="#CC6677")
    ax.text(0.99, 0.04, "right (\u2212)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5, color="#CC6677")
    ax.set_ylabel("Offset (mm)")
    ax.set_title("Signed lateral tail swing (baseline-removed)",
                 fontsize=8.5, loc="left")

    ax = axes[5]
    ax.plot(t, g["gaze_angle_deg"], lw=0.6, color="#4477AA")
    ax.axhline(GAZE_DEG, color="#EE6677", lw=0.8, ls="--",
               label=f"looking at human (< {GAZE_DEG:.0f}\u00b0)")
    gz = g["gaze_at_human"].to_numpy(float)
    ax.fill_between(t, 0, 1, where=(gz == 1),
                    transform=ax.get_xaxis_transform(),
                    color="#4477AA", alpha=0.18, linewidth=0)
    ax.set_ylim(0, 180)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Head angle (deg)")
    ax.set_title("Dog head orientation to human (shaded = looking)",
                 fontsize=8.5, loc="left")

    ax = axes[6]
    av = g["approach_vel_mps"].to_numpy(float)
    ax.plot(t, av, lw=0.6, color=DOG_COLOR)
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    lim = float(np.nanmax(np.abs(av))) if np.any(np.isfinite(av)) else 0.2
    ax.set_ylim(-max(lim, 0.1) * 1.15, max(lim, 0.1) * 1.15)
    hm = (g["human_moving"].to_numpy(float) == 1)
    ax.fill_between(t, 0.95, 1.0, where=hm,
                    transform=ax.get_xaxis_transform(),
                    color=HUMAN_COLOR, alpha=0.55, linewidth=0)
    fs = (g["follow_state"].to_numpy(float) == 1)
    ax.fill_between(t, 0, 1, where=fs,
                    transform=ax.get_xaxis_transform(),
                    color=DOG_COLOR, alpha=0.15, linewidth=0)
    ax.text(0.01, 0.93, "top bar = human moving; shaded = following",
            transform=ax.transAxes, fontsize=6, color="0.35")
    ax.set_ylabel("V toward human (m/s)")
    ax.set_title("Following: dog velocity component toward human",
                 fontsize=8.5, loc="left")

    # Panel 8: dog-human movement-direction consistency (cosine of the angle between
    # both velocity vectors, defined on frames where both speeds > 0.05 m/s)
    ax = axes[7]
    va = g["vel_align_cos"].to_numpy(float)
    va_sm = (pd.Series(va).rolling(31, center=True, min_periods=8)
             .median().to_numpy(float))
    ax.plot(t, va, lw=0.4, color="#882255", alpha=0.30)
    ax.plot(t, va_sm, lw=1.1, color="#882255")
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_ylim(-1.05, 1.05)
    bm = np.isfinite(va)
    ax.fill_between(t, 0.95, 1.0, where=bm,
                    transform=ax.get_xaxis_transform(),
                    color="#882255", alpha=0.50, linewidth=0)
    ax.text(0.01, 0.93, "top bar = both moving (> 0.05 m/s)",
            transform=ax.transAxes, fontsize=6, color="0.35")
    ax.text(0.99, 0.96, "same dir (+1)", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color="#882255")
    ax.text(0.99, 0.04, "opposite (\u22121)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5, color="#882255")
    ax.set_ylabel("Alignment (cos)")
    ax.set_title("Dog\u2013human movement direction consistency "
                 "(thin = per-frame; bold = 1 s rolling median)",
                 fontsize=8.5, loc="left")
    ax.set_xlabel("Time (s)")

    fig.suptitle(f"Representative beagle recording ({r['batch']} / "
                 f"{r['rec'].replace('录制', 'rec')}, S3 human gaze at dog)",
                 fontsize=9.5)
    fig.savefig(os.path.join(D_FIG, "fig_timeseries_example.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {os.path.join(D_FIG, 'fig_timeseries_example.png')}")
    return r


# ======================================================================
# 6) Representative beagle trajectory figures (S1/S2/S3)
# ======================================================================
def _fence_patch(fence_px, ax):
    if fence_px is None or len(fence_px) < 3:
        return
    poly = MplPolygon(fence_px, closed=True, facecolor="0.92",
                      edgecolor="0.35", lw=0.9, alpha=0.9, zorder=0,
                      label="Hexagonal fence")
    ax.add_patch(poly)
    ax.scatter(fence_px[:, 0], fence_px[:, 1], s=6, c="0.25", zorder=3)


def task7_trajectories(fm, rm):
    log("=" * 78)
    log("[6/7] Representative beagle trajectory figures (S1/S2/S3, median-speed recording per condition)")
    fc = pd.read_csv(os.path.join(D_DATA, "fence_corners_per_rec.csv"),
                     dtype={"batch": str, "rec": str})
    per_rec = {}
    for (b, r_), sub in fc.groupby(["batch", "rec"]):
        per_rec[(b, r_)] = sub.sort_values("corner_id")[
            ["color_x_720", "color_y_720"]].to_numpy(float)
    keep = rm[(rm.breed == "beagle") & (~rm.excluded)]
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.3))
    for j, c in enumerate(["S1", "S2", "S3"]):
        sub = keep[keep.cond == c]
        med = sub.dog_mean_speed_mps.median()
        r = sub.iloc[(sub.dog_mean_speed_mps - med).abs().argsort().iloc[0]]
        g = fm[fm.rec == r.rec]
        ax = axes[j]
        _fence_patch(per_rec.get((r.batch, r.rec)), ax)
        dog = g[["dog_cx_px", "dog_cy_px"]].dropna().to_numpy()
        hum = g[["toe_x_px", "toe_y_px"]].dropna().to_numpy()
        if len(hum):
            ax.plot(hum[:, 0], hum[:, 1], color=HUMAN_COLOR, lw=0.6,
                    alpha=0.8, zorder=2, label="Human (left toe)")
        if len(dog):
            ax.plot(dog[:, 0], dog[:, 1], color=DOG_COLOR, lw=0.6,
                    alpha=0.85, zorder=2, label="Dog (body centroid)")
            ax.scatter(dog[0, 0], dog[0, 1], s=10, marker="o", color=DOG_COLOR,
                      zorder=4, edgecolor="white", linewidth=0.3)
            ax.scatter(dog[-1, 0], dog[-1, 1], s=10, marker="x",
                      color=DOG_COLOR, zorder=4, linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(0, RGB_W)
        ax.set_ylim(RGB_H, 0)
        ax.set_xlabel("x (px, 1280\u00d7720)")
        if j == 0:
            ax.set_ylabel("y (px)")
        ax.set_title(f"{COND_FULL[c]}\ndog {r.dog_id.split('_')[-1]} "
                     f"({r.duration_s:.0f} s)", fontsize=9)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7, handletextpad=0.4,
                      borderaxespad=0.2, markerscale=0.7)
    fig.suptitle("Representative beagle trajectories "
                 "(median-speed recording per condition)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, os.path.join(D_FIG, "fig_traj_examples_beagle"))
    picks = keep[keep.cond.isin(["S1", "S2", "S3"])].sort_values("cond")
    rows = []
    for c in ["S1", "S2", "S3"]:
        sub = keep[keep.cond == c]
        med = sub.dog_mean_speed_mps.median()
        r = sub.iloc[(sub.dog_mean_speed_mps - med).abs().argsort().iloc[0]]
        rows.append({"cond": c, "batch": r.batch, "rec": r.rec,
                     "dog_id": r.dog_id, "duration_s": r.duration_s,
                     "dog_mean_speed_mps": r.dog_mean_speed_mps,
                     "follow_ratio": r.follow_ratio})
    pd.DataFrame(rows).to_csv(
        os.path.join(D_TAB, "paper_traj_example_picks.csv"), index=False)


# ======================================================================
# 7) Fence annotation check figure (beagle batches)
# ======================================================================
def task8_fence_check():
    log("=" * 78)
    log("[7/7] Fence annotation check figure (beagle batches only)")
    import cv2
    fc = pd.read_csv(FENCE_CSV, dtype={"batch": str, "rec": str})
    fc = fc[fc.batch != PET_BATCH]
    picks = fc.groupby("batch", sort=True)["rec"].first().reset_index()

    def video_path(batch, rec):
        base = os.path.join(VIDEO_ROOT, batch, rec)
        if os.path.isdir(os.path.join(base, "color")):
            vids = sorted(glob.glob(os.path.join(base, "color", "*.mp4")))
        else:
            vids = sorted(glob.glob(os.path.join(base, "*.mp4")))
        return vids[0] if vids else None

    picks = [p for p in picks.itertuples(index=False) if video_path(p.batch, p.rec)]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 2.2))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, picks):
        fsub = fc[(fc.batch == p.batch) & (fc.rec == p.rec)].sort_values(
            "corner_id")
        cap = cv2.VideoCapture(video_path(p.batch, p.rec))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(600, n - 1))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            ax.axis("off")
            continue
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts = fsub[["video_x", "video_y"]].to_numpy()
        poly = MplPolygon(pts, closed=True, fill=False, edgecolor="#EE6677",
                          lw=1.4)
        ax.add_patch(poly)
        ax.scatter(pts[:, 0], pts[:, 1], s=20, c="#EE6677",
                   edgecolor="white", linewidth=0.5, zorder=5)
        ax.set_title(f"{p.batch}", fontsize=9)
        ax.set_xlabel("video x (px, 854\u00d7480)", fontsize=8)
        ax.set_ylabel("video y (px)", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fence corner annotation check (one beagle recording per "
                 "batch)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, os.path.join(D_FIG, "fig_fence_video_check_beagle"))


# ======================================================================
def main():
    os.makedirs(D_UMAP, exist_ok=True)
    log("paper_support.py — paper support analyses (beagle-only, XY-only)")
    log(f"  Output directory: {HERE}")

    # ---- Load shared data ----
    rm = pd.read_csv(os.path.join(D_DATA, "xy_rec_metrics.csv"),
                     dtype={"batch": str, "rec": str})
    fm = pd.read_csv(os.path.join(D_DATA, "xy_frame_metrics.csv"),
                     dtype={"batch": str, "rec": str})
    # Per-recording gaze/approach aggregation (within effective time windows)
    fm_w = fm.merge(rm[["batch", "rec", "win_start", "win_end"]],
                    on=["batch", "rec"], how="left")
    fm_w["win_start"] = fm_w["win_start"].fillna(-1)
    fm_w["win_end"] = fm_w["win_end"].fillna(1e12)
    fm_w = fm_w[(fm_w.frame >= fm_w.win_start)
                & (fm_w.frame <= fm_w.win_end)]
    gz = (fm_w.groupby(["batch", "rec"], as_index=False)
          .agg(gaze_ratio=("gaze_at_human", "mean"),
               gaze_angle_mean_deg=("gaze_angle_deg", "mean"),
               approach_vel_mean=("approach_vel_mps", "mean")))
    rm2 = rm.merge(gz, on=["batch", "rec"], how="left")
    log(f"  Recording-level metrics: {len(rm)} rows; "
        f"beagle: {int((rm.breed == 'beagle').sum())} rows")

    task1_keypoint_likelihood()
    task2_scale_accuracy(rm)
    task3_descriptives(rm, rm2)
    task4_social_classification(rm2)
    task6_timeseries_example(fm, rm)
    task7_trajectories(fm, rm)
    task8_fence_check()
    log("=" * 78)
    log("All tasks completed")


if __name__ == "__main__":
    main()
