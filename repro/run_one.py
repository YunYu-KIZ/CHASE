#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_one.py — Analyse ONE experiment from its own data folder.

The data folder must contain (column schemas in README.md):
    dog_keypoints.csv      dog keypoints per frame (required)
    human_keypoints.csv    human keypoints per frame (optional; a missing or
                           empty file is treated as "no human present")
    fence_corners.csv      6 annotated hexagonal fence corners (required)

Usage:
    python3 run_one.py /abs/path/to/experiment_folder
    python3 run_one.py /abs/path/to/experiment_folder --out /abs/path/to/out_folder
    python3 run_one.py /abs/path/to/exp --win 120 1704   # manual time window (frames)

Outputs (into the out folder; default results/single/<folder name>/):
    metrics_frame.csv    frame-level metrics
    metrics_rec.csv      per-recording summary (1 row)
    timeseries.png       8-panel overview (paper Fig. 2 style)
    trajectory.png       trajectory + fence hexagon
Key summary metrics are also printed to the console.

Metadata (batch/rec/cond/breed/dog_id) is inferred from the folder name; the
analysis time window defaults to the full recording. If the folder happens to
be one of the package's own recordings (data/recordings/...), its metadata and
time window are taken from the package index so results match run.py exactly.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R  # noqa: E402  (reuses all metric/figure functions)

COND_S = {"01": "S1", "02": "S2", "03": "S3"}


def infer_meta(exp_dir):
    """Metadata for an arbitrary experiment folder: inferred from the folder
    name, or inherited from the package index when the folder is one of the
    package's own recordings (keeps results identical to run.py)."""
    name = os.path.basename(os.path.normpath(exp_dir))
    meta = {"batch": name, "rec": name, "dir": name, "cond": "",
            "scenario": "", "breed": "", "dog_id": name,
            "win_start": np.nan, "win_end": np.nan}
    idx_path = os.path.join(HERE, "data", "recordings.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path, dtype=str, keep_default_na=False)
        hit = idx[(idx.dir == name) | (idx.rec == name)]
        if len(hit):
            m = hit.iloc[0].to_dict()
            m["win_start"] = float(m["win_start"]) if m["win_start"] else np.nan
            m["win_end"] = float(m["win_end"]) if m["win_end"] else np.nan
            return m, "package index"
    tail = name.split("_")[-1]
    if tail in COND_S:
        meta["cond"] = tail
        meta["scenario"] = COND_S[tail]
    return meta, "folder name (full-recording analysis)"


def main():
    ap = argparse.ArgumentParser(
        description="Analyse one experiment from its own data folder")
    ap.add_argument("data_folder",
                    help="folder containing dog_keypoints.csv, "
                         "human_keypoints.csv (optional) and fence_corners.csv")
    ap.add_argument("--out", default=None,
                    help="output folder (default: results/single/<folder name>)")
    ap.add_argument("--win", nargs=2, type=int, metavar=("START", "END"),
                    default=None,
                    help="analysis time window in frames (default: full recording)")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.data_folder)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"Not a folder: {exp_dir}")
    for f in ["dog_keypoints.csv", "fence_corners.csv"]:
        if not os.path.exists(os.path.join(exp_dir, f)):
            raise SystemExit(f"Missing required file: {os.path.join(exp_dir, f)}")

    meta, meta_src = infer_meta(exp_dir)
    print("=" * 78)
    print(f"Single-experiment analysis: {exp_dir}")
    print(f"  metadata from: {meta_src}"
          + (f", scenario {meta['scenario']}" if meta["scenario"] else ""))

    # ---- time window: --win > package index > full recording ----
    if args.win is not None:
        w = (int(args.win[0]), int(args.win[1]))
    elif np.isfinite(meta["win_start"]) and np.isfinite(meta["win_end"]):
        w = (int(meta["win_start"]), int(meta["win_end"]))
        print(f"  time window from index: frames {w[0]}–{w[1]}")
    else:
        w = None
        print("  time window: full recording (annotate none)")

    # ---- load + fence scale ----
    g = R.load_recording(exp_dir)
    g["batch"], g["rec"] = meta["batch"], meta["rec"]
    fence_px, scale = R.load_fence(exp_dir)
    if fence_px is None or len(fence_px) != 6 or not np.isfinite(scale):
        raise SystemExit("fence_corners.csv must contain exactly 6 corners "
                         "(scale calibration needs the full hexagon)")

    out_dir = (os.path.abspath(args.out) if args.out
               else os.path.join(HERE, "results", "single", meta["dir"]))
    os.makedirs(out_dir, exist_ok=True)

    # ---- compute (same pipeline as run.py process_one) ----
    R.setup_journal_style()
    fm, qc = R.process_recording(g, w, scale)
    qc["win_start"] = float(w[0]) if w is not None else np.nan
    qc["win_end"] = float(w[1]) if w is not None else np.nan
    if fence_px is not None and len(fence_px) == 6:
        poly = MplPath(fence_px)
        pts = fm[["dog_cx_px", "dog_cy_px"]].to_numpy(float)
        okm = np.isfinite(pts).all(axis=1)
        qc["dog_inside_fence_ratio"] = (float(poly.contains_points(pts[okm]).mean())
                                        if okm.sum() else np.nan)
    else:
        qc["dog_inside_fence_ratio"] = np.nan
    rec_row = R.summarize_recording(fm, qc, meta)

    fig_meta = {"rec": meta["rec"], "batch": meta["batch"], "breed": meta["breed"],
                "scenario": meta["scenario"], "dog_id": meta["dog_id"]}
    R.plot_timeseries(fm, fig_meta, os.path.join(out_dir, "timeseries.png"))
    R.plot_trajectory(fm, fence_px, fig_meta, os.path.join(out_dir, "trajectory.png"))

    fm.to_csv(os.path.join(out_dir, "metrics_frame.csv"), index=False)
    pd.DataFrame([rec_row]).to_csv(os.path.join(out_dir, "metrics_rec.csv"),
                                   index=False)

    # ---- console summary ----
    print("-" * 78)
    print(f"  Frames = {rec_row['n_frames']} ({rec_row['duration_s']:.0f} s), "
          f"dog valid = {rec_row['dog_valid_ratio']:.0%}, "
          f"tail visible = {rec_row['tail_vis_ratio']:.0%}, "
          f"scale = {scale*1000:.2f} mm/px")
    m = rec_row
    for k, lab in [
        ("dog_mean_speed_mps", "Dog mean speed (m/s)"),
        ("dog_median_speed_mps", "Dog median speed (m/s)"),
        ("dog_cum_dist_m", "Dog cumulative distance (m)"),
        ("dog_dist_rate_mpm", "Dog distance rate (m/min)"),
        ("human_toe_mean_speed_mps", "Human toe mean speed (m/s)"),
        ("human_toe_cum_dist_m", "Human cumulative distance (m)"),
        ("d_head_toe_mean_m", "Head-toe distance, mean (m)"),
        ("d_head_toe_median_m", "Head-toe distance, median (m)"),
        ("d_head_toe_min_m", "Head-toe distance, min (m)"),
        ("d_head_toe_within1m_ratio", "Time within 1 m of human"),
        ("tail_wag_amp_mean_mm", "Tail wag amplitude (mm)"),
        ("tail_wag_freq_hz", "Tail wag frequency (Hz)"),
        ("tail_elev_mean_deg", "Tail elevation mean (deg)"),
        ("head_pitch_mean_deg", "Head pitch mean (deg)"),
        ("follow_ratio", "Following ratio (while human moving)"),
        ("vel_align_mean", "Velocity alignment dog-human (cos)"),
    ]:
        v = m.get(k, float("nan"))
        print(f"  {lab:38s} {v:.4f}" if isinstance(v, float) and v == v
              else f"  {lab:38s} {v}")
    print("-" * 78)
    print(f"  Output directory: {out_dir}")
    print(f"    metrics_frame.csv, metrics_rec.csv, timeseries.png, trajectory.png")


if __name__ == "__main__":
    main()
