#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""traj_group_overlay.py — Dog–human trajectory overlays for the two UMAP sociability groups (high/low human orientation).

Content (all S2 / S3 conditions, beagle dogs, non-excluded recordings):
  1) Multi-recording overlay figure (2x2): rows = S2/S3, columns =
     human-oriented / less human-oriented. Each panel overlays the dog
     trajectories (orange, withers point) and the human trajectory (blue, left
     toe tip) of all selected recordings of that group + the hexagonal fence
     background. The high-orientation group has many dogs (32), so dogs are
     randomly sampled (seed=42) down to the size of the low-orientation group (7).
  2) Single-recording trajectory figures: for every selected recording, the dog
     + human trajectory overlay is drawn separately (+ fence / start-end
     markers), saved under figures/traj_group_overlay/single/.

Coordinate registration: camera placement / fence position differ between
  recordings, so pixel coordinates cannot be overlaid directly.
  All trajectories are transformed into a common "fence coordinate system"
  (metric): origin = mean of the recording's six fence corners (centre),
  scale = 0.90 m / mean hexagon side length in pixels, y axis flipped
  (image y points down -> mathematical y points up).
  Depth back-projection coordinates (dog_cx_m etc.) are a dropped reference
  convention and are not used. Dog trajectories use the withers keypoint
  (from stage0, valid and non-jump_removed frames), not the body centroid;
  human trajectories use the left toe tip.

Inputs:
  data/xy_rec_metrics.csv                       (recording-level: cond/breed/excluded/win_*)
  data/xy_frame_metrics.csv                     (frame-level: toe_x_px/toe_y_px)
  data/fence_corners_per_rec.csv                (per-recording six fence corners, 720p)
  stage0_data/stage0_keypoints_xyz_all.csv     (dog withers pixel coordinates color_x/color_y)
  umap/beagle_social_clusters_xy.csv            (UMAP dog-level typing: dog_id -> social_type)
Outputs:
  figures/traj_group_overlay/fig_traj_overlay_S2S3_groups.png
  figures/traj_group_overlay/single/{cond}_{social_type}_{rec}.png

Usage: python3 traj_group_overlay.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from vis_common import setup_journal_style, save_fig, DOG_COLOR, HUMAN_COLOR  # noqa: E402

D_DATA = os.path.join(HERE, "data")
S0_ALL = os.path.join(os.path.dirname(HERE), "stage0_data",
                      "stage0_keypoints_xyz_all.csv")
D_UMAP = os.path.join(HERE, "umap")
D_FIG = os.path.join(HERE, "figures", "traj_group_overlay")
D_SINGLE = os.path.join(D_FIG, "single")
FENCE_SIDE_M = 0.90          # physical width of each hexagon panel
SEED = 42
SOC_COLOR = {"human-oriented": "#ff7f0e", "less human-oriented": "#AA3377"}
AX_LIM = 1.35                # coordinate range ±1.35 m (fence circumradius 0.9 m)


def log(msg=""):
    print(msg, flush=True)


def fence_transformer(fc: pd.DataFrame):
    """Per-recording fence registration: returns {rec: (center_px, scale_m_per_px, fence_px_6x2)}."""
    per_rec = {}
    for rec, sub in fc.groupby("rec"):
        p = sub.sort_values("corner_id")[["color_x_720", "color_y_720"]].to_numpy(float)
        if len(p) < 3:
            continue
        ctr = p.mean(axis=0)
        # mean hexagon side length in pixels: distances between adjacent corners (incl. closing edge)
        side_px = float(np.mean(np.hypot(*(np.roll(p, -1, axis=0) - p).T)))
        scale = FENCE_SIDE_M / side_px
        per_rec[rec] = (ctr, scale, p)
    return per_rec


def to_fence_xy(px, py, ctr, scale):
    """Pixel coordinates -> fence coordinate system (metres, centre origin, y up)."""
    return (np.asarray(px, float) - ctr[0]) * scale, -(np.asarray(py, float) - ctr[1]) * scale


def fence_patch_m(fence_px, ctr, scale, ax, label=None, lw=0.9):
    """Draw the fence polygon (metric units) onto ax."""
    fx, fy = to_fence_xy(fence_px[:, 0], fence_px[:, 1], ctr, scale)
    poly = MplPolygon(np.column_stack([fx, fy]), closed=True, facecolor="0.92",
                      edgecolor="0.35", lw=lw, alpha=0.9, zorder=0,
                      label=label)
    ax.add_patch(poly)


def load_traj(fm: pd.DataFrame, wsub: pd.DataFrame, rec: str, win: tuple):
    """Frame-level trajectory of one recording (time-window filtered; NaN bounds = all frames).

    Dog = withers pixel coordinates (stage0, pre-filtered to valid and
    non-jump_removed); human = left toe tip pixel coordinates
    (xy_frame_metrics). Returns (dog_xy, human_xy) pixel arrays.
    """
    w0, w1 = win
    g = fm[fm.rec == rec]
    if np.isfinite(w0) and np.isfinite(w1):
        g = g[(g.frame >= w0) & (g.frame <= w1)]
        m = (wsub.frame >= w0) & (wsub.frame <= w1)
    else:
        m = np.ones(len(wsub), dtype=bool)
    dog = wsub.loc[m, ["color_x", "color_y"]].dropna().to_numpy(float)
    hum = g[["toe_x_px", "toe_y_px"]].dropna().to_numpy(float)
    return dog, hum


def style_axis(ax):
    ax.set_xlim(-AX_LIM, AX_LIM)
    ax.set_ylim(-AX_LIM, AX_LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


def main():
    os.makedirs(D_SINGLE, exist_ok=True)
    log("traj_group_overlay.py — dog–human trajectory overlays for the two groups (S2/S3)")
    rm_path = os.path.join(D_DATA, "xy_rec_metrics.csv")
    fm_path = os.path.join(D_DATA, "xy_frame_metrics.csv")
    fc_path = os.path.join(D_DATA, "fence_corners_per_rec.csv")
    cl_path = os.path.join(D_UMAP, "beagle_social_clusters_xy.csv")
    for p in (rm_path, fm_path, fc_path, cl_path):
        log(f"Input: {p}")

    rm = pd.read_csv(rm_path, dtype={"batch": str, "rec": str})
    fc = pd.read_csv(fc_path, dtype={"batch": str, "rec": str})
    cl = pd.read_csv(cl_path)

    keep = rm[(rm.breed == "beagle") & (~rm.excluded)
              & rm.cond.isin(["S2", "S3"])].copy()
    keep = keep.merge(cl[["dog_id", "social_type"]], on="dog_id", how="left")
    assert keep.social_type.notna().all(), "found S2/S3 beagle recordings without a group label"
    log(f"  S2/S3 beagle non-excluded recordings: {len(keep)} "
        f"({keep.groupby(['cond','social_type']).size().to_dict()})")

    # ---- Dog-level balanced sampling: random sample of n_less dogs from the high-orientation group ----
    dogs_ho = sorted(keep.loc[keep.social_type == "human-oriented", "dog_id"].unique())
    dogs_less = sorted(keep.loc[keep.social_type == "less human-oriented", "dog_id"].unique())
    rng = np.random.default_rng(SEED)
    picked_ho = sorted(rng.choice(dogs_ho, size=len(dogs_less), replace=False))
    log(f"  Human-oriented group: {len(dogs_ho)} dogs -> random sample of {len(picked_ho)} dogs "
        f"(seed={SEED}, matched to the {len(dogs_less)} dogs of the less-human-oriented group):")
    log(f"    human-oriented sampled dogs: {picked_ho}")
    log(f"    less human-oriented dogs: {dogs_less}")
    sel = keep[keep.dog_id.isin(picked_ho + dogs_less)].copy()
    log(f"  Selected recordings: {len(sel)} "
        f"({sel.groupby(['cond','social_type']).size().to_dict()})")

    # ---- Frame-level trajectories (only the columns needed for selected recordings) ----
    recs_need = set(sel.rec)
    fm = pd.read_csv(fm_path, dtype={"batch": str, "rec": str},
                     usecols=["batch", "rec", "frame", "toe_x_px", "toe_y_px"])
    fm = fm[fm.rec.isin(recs_need)]
    ft = fence_transformer(fc)
    missing = recs_need - set(ft)
    if missing:
        raise SystemExit(f"Selected recordings missing fence annotation: {missing}")

    # ---- Dog withers-point trajectories: extracted from stage0, QC = valid and non-jump_removed ----
    log(f"Input (withers): {S0_ALL}")
    sd = pd.read_csv(S0_ALL, dtype={"batch": str, "rec": str},
                     usecols=["rec", "frame", "subject", "body_part",
                              "color_x", "color_y", "valid", "jump_removed"])
    wd = sd[(sd.subject == "dog") & (sd.body_part == "withers")
            & sd.rec.isin(recs_need) & (sd.valid == True)          # noqa: E712
            & (sd.jump_removed == False)]                            # noqa: E712
    wpx = {r_: g.sort_values("frame")[["frame", "color_x", "color_y"]]
           for r_, g in wd.groupby("rec")}
    del sd
    miss_w = recs_need - set(wpx)
    if miss_w:
        raise SystemExit(f"Selected recordings missing withers data: {miss_w}")
    log(f"  Valid withers frames in total {sum(len(v) for v in wpx.values()):,} "
        f"({len(wpx)} recordings, valid and non-jump)")

    # Pre-transform each recording into the fence coordinate system (metres)
    traj = {}
    for _, r in sel.iterrows():
        dog, hum = load_traj(fm, wpx[r.rec], r.rec, (r.win_start, r.win_end))
        ctr, scale, fpx = ft[r.rec]
        dog_m = np.column_stack(to_fence_xy(dog[:, 0], dog[:, 1], ctr, scale)) if len(dog) else np.empty((0, 2))
        hum_m = np.column_stack(to_fence_xy(hum[:, 0], hum[:, 1], ctr, scale)) if len(hum) else np.empty((0, 2))
        traj[(r.cond, r.social_type, r.rec)] = (dog_m, hum_m, (ctr, scale, fpx), r)

    # ================= 1) Multi-recording overlay figure (2x2) =================
    setup_journal_style()
    # Enlarged for embedding into the composite figure: larger fonts / line widths
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                         "axes.titlesize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 11.5})
    conds = ["S2", "S3"]
    groups = ["human-oriented", "less human-oriented"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), sharex=True, sharey=True)
    leg_ax = axes[0, 1]
    for i, cond in enumerate(conds):
        for j, grp in enumerate(groups):
            ax = axes[i, j]
            sub = [(k, v) for k, v in traj.items() if k[0] == cond and k[1] == grp]
            is_leg = ax is leg_ax
            for (dog_m, hum_m, ftr, _r) in [v for _, v in sub]:
                if len(hum_m):
                    ax.plot(hum_m[:, 0], hum_m[:, 1], color=HUMAN_COLOR,
                            lw=1.8, alpha=0.30, zorder=2,
                            label="Human (left toe)" if is_leg else None)
                if len(dog_m):
                    ax.plot(dog_m[:, 0], dog_m[:, 1], color=DOG_COLOR,
                            lw=2.2, alpha=0.36, zorder=3,
                            label="Dog (withers)" if is_leg else None)
                fence_patch_m(ftr[2], ftr[0], ftr[1], ax, lw=1.6,
                              label="Hexagonal fence" if is_leg else None)
            style_axis(ax)
            n_rec = len(sub)
            ax.set_title(f"{cond} — {grp} (n = {n_rec} recordings)",
                         fontsize=12.5, loc="left")
            if i == 1:
                ax.set_xlabel("x (m, fence-centred)")
            if j == 0:
                ax.set_ylabel("y (m)")
    # De-duplicate: all 7 recording trajectories carry labels; keep one handle per
    # class only; place the legend outside on the right of the whole figure (no overlap)
    h, l = leg_ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), fontsize=12, loc="center left",
               bbox_to_anchor=(1.04, 0.5), frameon=True, framealpha=0.95,
               borderaxespad=0)
    fig.suptitle("Dog–human trajectory overlays by human-orientation group "
                 f"(fence-centred; {len(dogs_less)} dogs per group)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_all = os.path.join(D_FIG, "fig_traj_overlay_S2S3_groups")
    save_fig(fig, out_all)
    log(f"Output (overlay figure): {out_all}.png / .pdf")

    # ================= 2) Single-recording trajectory figures =================
    # Restore regular font sizes (the large fonts above are for the overlay only)
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10,
                         "axes.titlesize": 10, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 8.5})
    n_single = 0
    for (cond, grp, rec), (dog_m, hum_m, ftr, r) in sorted(traj.items()):
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        fence_patch_m(ftr[2], ftr[0], ftr[1], ax, label="Hexagonal fence")
        if len(hum_m):
            ax.plot(hum_m[:, 0], hum_m[:, 1], color=HUMAN_COLOR, lw=0.8,
                    alpha=0.85, zorder=2, label="Human (left toe)")
            ax.scatter(*hum_m[0], s=14, marker="o", color=HUMAN_COLOR,
                       edgecolors="white", linewidths=0.5, zorder=5)
            ax.scatter(*hum_m[-1], s=14, marker="s", color=HUMAN_COLOR,
                       edgecolors="white", linewidths=0.5, zorder=5)
        if len(dog_m):
            ax.plot(dog_m[:, 0], dog_m[:, 1], color=DOG_COLOR, lw=0.8,
                    alpha=0.9, zorder=3, label="Dog (withers)")
            ax.scatter(*dog_m[0], s=14, marker="o", color=DOG_COLOR,
                       edgecolors="white", linewidths=0.5, zorder=5)
            ax.scatter(*dog_m[-1], s=14, marker="s", color=DOG_COLOR,
                       edgecolors="white", linewidths=0.5, zorder=5)
        style_axis(ax)
        ax.set_xlabel("x (m, fence-centred)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{rec.replace('录制', 'rec')} — {cond}, {grp}\n"
                     f"dog {r.dog_id}, follow ratio = {r.follow_ratio:.2f}",
                     fontsize=9, loc="left")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        out_single = os.path.join(D_SINGLE, f"{cond}_{grp}_{rec}")
        save_fig(fig, out_single)
        plt.close(fig)
        n_single += 1
    log(f"Output (single-recording figures): {D_SINGLE}/ {n_single} figures in total "
        f"({{cond}}_{{social_type}}_{{rec}}.png/.pdf)")
    log("Done.")


if __name__ == "__main__":
    main()
