#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_one.py — Analyse ONE experiment.

Two input modes:

A) File mode — give the keypoint CSVs directly (any absolute paths):
    python3 run_one.py --dog dog.csv --fence fence.csv --out /path/out_dir
    python3 run_one.py --dog dog.csv --fence fence.csv --human human.csv \
        --out /path/out_dir --win 120 1704 --rec my_experiment

B) Folder mode — a folder holding the package layout
   (dog_keypoints.csv / human_keypoints.csv / fence_corners.csv):
    python3 run_one.py /path/to/experiment_folder [--out /path/out_dir]
    python3 run_one.py data/recordings/08__录制_08_201802_03

Column schemas are documented in README.md (pixel coordinates refer to the
1280x720 colour frames; dog = occipital_protuberance / withers / tail_base /
tail_tip; human = left_shoulder / right_shoulder / left_toe_tip /
right_toe_tip; fence = corner_id 1-6 + color_x_720 / color_y_720).

Outputs (into the out folder; default results/single/<name>/):
    metrics_frame.csv    frame-level metrics
    metrics_rec.csv      per-recording summary (1 row)
    timeseries.png       8-panel overview (paper Fig. 2 style)
    trajectory.png       trajectory + fence hexagon
Key summary metrics are also printed to the console.

Notes:
- `--human` is optional: omitting it (or an empty file) means "no human
  present" and all human-related metrics become NaN.
- `--fence` is optional: without it (or a fence_corners.csv in folder mode)
  the metre-per-pixel scale cannot be calibrated, so all metre-based metrics
  become NaN; pixel trajectories and figures are still produced. A provided
  fence file must contain exactly 6 corners.
- Recording name: `--rec` if given; otherwise the dog-CSV stem, or the parent
  folder name when the stem is a generic name like "dog_keypoints". If the
  name matches one of the package's own recordings, its metadata and time
  window are inherited from the package index so results match run.py.
- Analysis time window: `--win START END` (frames) > package index > full
  recording.
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


def load_from_files(dog_csv, human_csv=None):
    """dog + human CSV files -> single long table (same as run.load_recording)."""
    dog = pd.read_csv(dog_csv)
    dog["subject"] = "dog"
    if human_csv:
        if not os.path.exists(human_csv):
            raise SystemExit(f"Human keypoints file not found: {human_csv}")
        hum = pd.read_csv(human_csv)
    else:
        hum = pd.DataFrame(columns=dog.columns)
    hum["subject"] = "human"
    return pd.concat([dog, hum], ignore_index=True)


def load_fence_file(fence_csv):
    """fence corners CSV -> (points, m/px scale); identical to run.load_fence."""
    fc = pd.read_csv(fence_csv).sort_values("corner_id")
    pts = fc[["color_x_720", "color_y_720"]].to_numpy(float)
    if len(pts) == 6:
        sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        scale = R.FENCE_SIDE_M / float(sides.mean())
    else:
        scale = np.nan
    return pts, scale


def infer_meta(name):
    """Metadata for an arbitrary experiment name: inferred from the name, or
    inherited from the package index when it matches one of the package's own
    recordings (keeps results identical to run.py)."""
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
    return meta, "inferred from name (full-recording analysis)"


def main():
    ap = argparse.ArgumentParser(
        description="Analyse one experiment from keypoint CSVs "
                    "(file mode or package-folder mode)")
    ap.add_argument("data_folder", nargs="?", default=None,
                    help="folder mode: folder containing dog_keypoints.csv, "
                         "human_keypoints.csv (optional) and fence_corners.csv")
    ap.add_argument("--dog", help="file mode: dog keypoints CSV file")
    ap.add_argument("--human", default=None,
                    help="file mode: human keypoints CSV file "
                         "(optional; omit = no human present)")
    ap.add_argument("--fence",
                    help="file mode: fence corners CSV file (6 corners). "
                         "Optional; without it the scale cannot be calibrated "
                         "and all metre-based metrics will be NaN")
    ap.add_argument("--rec", default=None,
                    help="recording name used for outputs/metadata "
                         "(default: dog-CSV stem, or its parent folder name "
                         "for generic names like dog_keypoints.csv)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: results/single/<name>)")
    ap.add_argument("--win", nargs=2, type=int, metavar=("START", "END"),
                    default=None,
                    help="analysis time window in frames (default: full recording)")
    args = ap.parse_args()

    file_mode = args.dog is not None or args.human is not None or args.fence is not None
    if file_mode and args.data_folder:
        raise SystemExit("Give either a data folder OR --dog/--human/--fence "
                         "files, not both.")

    if file_mode:
        # ---------- file mode ----------
        if not args.dog:
            raise SystemExit("--dog is required in file mode.")
        if not os.path.isfile(args.dog):
            raise SystemExit(f"File not found: {args.dog}")
        if args.fence and not os.path.isfile(args.fence):
            raise SystemExit(f"File not found: {args.fence}")
        stem = os.path.splitext(os.path.basename(args.dog))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.dog)))
        default_name = parent if stem in ("dog_keypoints", "dog") else stem
        name = args.rec or default_name
        print("=" * 78)
        print(f"Single-experiment analysis (file mode), recording name: {name}")
        print(f"Input (dog)   : {os.path.abspath(args.dog)}")
        print(f"Input (human) : {os.path.abspath(args.human) if args.human else '(none — no human)'}")
        print(f"Input (fence) : {os.path.abspath(args.fence) if args.fence else '(none — no fence)'}")
        g = load_from_files(args.dog, args.human)
        if args.fence:
            fence_px, scale = load_fence_file(args.fence)
        else:
            fence_px, scale = None, np.nan
            print("  WARNING: no --fence given -> scale cannot be calibrated; "
                  "all metre-based metrics will be NaN")
    else:
        # ---------- folder mode ----------
        if not args.data_folder:
            ap.error("give a data folder, or --dog/--fence CSV files "
                     "(see --help)")
        exp_dir = os.path.abspath(args.data_folder)
        if not os.path.isdir(exp_dir):
            raise SystemExit(f"Not a folder: {exp_dir}")
        if not os.path.exists(os.path.join(exp_dir, "dog_keypoints.csv")):
            raise SystemExit(f"Missing required file: "
                             f"{os.path.join(exp_dir, 'dog_keypoints.csv')}")
        name = os.path.basename(os.path.normpath(exp_dir))
        print("=" * 78)
        print(f"Single-experiment analysis (folder mode): {exp_dir}")
        g = R.load_recording(exp_dir)
        if os.path.exists(os.path.join(exp_dir, "fence_corners.csv")):
            fence_px, scale = R.load_fence(exp_dir)
        else:
            fence_px, scale = None, np.nan
            print("  WARNING: no fence_corners.csv -> scale cannot be "
                  "calibrated; all metre-based metrics will be NaN")

    # a provided fence must be a valid hexagon (a missing fence is allowed)
    if fence_px is not None and (len(fence_px) != 6 or not np.isfinite(scale)):
        raise SystemExit("fence corners file must contain exactly 6 corners "
                         "(scale calibration needs the full hexagon)")

    meta, meta_src = infer_meta(name)
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

    g["batch"], g["rec"] = meta["batch"], meta["rec"]

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
    scale_txt = (f"{scale*1000:.2f} mm/px" if np.isfinite(scale)
                 else "N/A (no fence)")
    print(f"  Frames = {rec_row['n_frames']} ({rec_row['duration_s']:.0f} s), "
          f"dog valid = {rec_row['dog_valid_ratio']:.0%}, "
          f"tail visible = {rec_row['tail_vis_ratio']:.0%}, "
          f"scale = {scale_txt}")
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
