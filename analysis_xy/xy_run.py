#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean XY-plane behavioural workflow (circular fence test · all 132 recordings)

Uses XY coordinates (horizontal plane) + per-recording video-annotated hexagonal
fence corners only; depth is used only for head up/down pitch and tail elevation.

Fence convention (important):
  - Keypoint XY coordinates come from inference on 720p (1280x720) video; fence
    corners were annotated on 480p proportionally compressed video and converted
    by the annotation tool to 720p coordinates (color_x_720/color_y_720), i.e.
    the same frame as the dog/human keypoints.
  - Trajectory plots are drawn in the 720p image pixel coordinate system:
    human/dog trajectories + that recording's own fence hexagon (per recording,
    positions differ).
  - Scale calibration: the fence consists of 6 panels, each physically 0.90 m
    wide -> per-recording scale_m_per_px = 0.9 / mean hexagon side length in
    pixels; all XY-plane metrics (speed/distance/dog-human distance/wag
    amplitude) are computed from the 720p pixel trajectory x scale (primary
    convention).
  - The depth back-projection convention (x_m/y_m, stage0) is kept as reference
    metrics (_depth suffix) to assess the accuracy and bias of the fence-scale
    approximation; head/tail vertical angles use each point's depth (h_m)
    directly, with no further depth analysis.

Metrics (per-recording summary):
  Dog:    mean/median speed, cumulative distance, distance rate (m/min,
          duration-normalized)
  Human:  left toe tip mean speed, cumulative distance (only scenarios 02/03
          with human present)
  Distance: dog head point (H) - human left toe tip XY distance:
          mean/median/min/time-share within 1 m
  Tail:   left-right wag amplitude (dynamic deviation relative to a 1 s sliding
          median baseline, mm), dominant wag frequency (Welch PSD 0.5-8 Hz)
  Depth:  head pitch head_pitch=atan2(hH-hW, |H-W|xy)  (+head up/-head down),
          tail elevation=atan2(hT-hB, |T-B|xy)       (+tail up/-tail down)

Quality control (QC):
  stage0 already applied: low-confidence (conf<0.3) removal, physical jump
  (0.5 m) removal, short gap (<=5 frames) interpolation, 5-frame smoothing
  This workflow adds: secondary jump removal (0.25 m/frame, centroid and human
  toe tip) + speed 5-frame sliding median
  Exclusion rules: dog valid-frame ratio < 0.5 -> whole recording excluded;
  tail tip visibility < 0.2 -> tail metrics set to NaN

Statistics:
  Distributions: overall / by scenario (S1=01 no human, S2=02 human not looking
          at dog, S3=03 human looking at dog) / by breed (beagle vs pet) /
          breed x scenario
  Tests: scenario effect Friedman (repeated measures, by breed and all dogs) +
          within-group pairwise Wilcoxon (Holm);
          human-related metrics only S2 vs S3 paired Wilcoxon
          breed effect (per scenario) Mann-Whitney U + Cliff's delta
          (cross-metric BH-FDR)
  Effect sizes: Kendall's W (Friedman), Cohen's dz (paired), Cliff's delta

Inputs:
  pipeline/stage0_data/stage0_keypoints_xyz_all.csv   (XY+depth, passed stage0 QC)
  pipeline/stage0_ground/fence_corners.csv            (per-recording video-marked
                                                       fence 6 corners, 480p clicks
                                                       converted to 720p)
  depth/圆形围栏统计开始结束表格.xlsx                    (per-recording effective time
                                                       window: start frame/end frame;
                                                       unannotated recordings use the
                                                       full video duration)
  depth/videos/{batch}/{rec}/color/*.mp4              (only for fence corner visual
                                                       verification)
Outputs (analysis_xy/):
  data/fence_corners_per_rec.csv, data/xy_frame_metrics.csv, data/xy_rec_metrics.csv,
  data/xy_qc_report.csv, trajectories/*.png(132), figures/*.png,
  tables/dist_*.csv, tables/test_scenario.csv, tables/test_breed.csv
"""

import os
import sys
import glob
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

PIPE = "/home/yy/data/1-Circular-Fence-Test/pipeline"
sys.path.insert(0, os.path.join(PIPE, "visualization"))
from vis_common import setup_journal_style, save_fig, DOG_COLOR, HUMAN_COLOR  # noqa: E402

# ===================== Configuration =====================
S0_ALL = os.path.join(PIPE, "stage0_data/stage0_keypoints_xyz_all.csv")
FENCE_CSV = os.path.join(PIPE, "stage0_ground/fence_corners.csv")
XLSX_TIME = "/home/yy/data/1-Circular-Fence-Test/depth/圆形围栏统计开始结束表格.xlsx"
VIDEO_ROOT = "/home/yy/data/1-Circular-Fence-Test/depth/videos"

OUT = os.path.join(PIPE, "analysis_xy")
D_DATA = os.path.join(OUT, "data")
D_TRAJ = os.path.join(OUT, "trajectories")
D_TS = os.path.join(OUT, "timeseries")
D_FIG = os.path.join(OUT, "figures")
D_TAB = os.path.join(OUT, "tables")

FPS = 30.0
DT = 1.0 / FPS

# Camera intrinsics (same convention as stage0/fence annotation, @1280x720)
FX, CX, CY = 920.0, 640.0, 360.0
RGB_W, RGB_H = 1280, 720

PART = {"occipital_protuberance": "H", "withers": "W",
        "tail_base": "B", "tail_tip": "T"}
PET_BATCH = "20260910xjx"

# QC parameters
CONF_TAIL = 0.5          # tail tip visibility confidence threshold
JUMP2_M = 0.25           # secondary jump removal threshold (m/frame)
GAP_INTERP = 5           # short gap interpolation limit (frames)
SPEED_MED_WIN = 5        # speed sliding median window
DOG_VALID_MIN = 0.5      # minimum dog valid-frame ratio (below -> whole recording excluded)
TAIL_VIS_MIN = 0.2       # minimum visibility ratio for tail metrics

# Tail parameters
WAG_BASE_WIN = 31        # wag baseline window (frames, ~1 s), sliding median
PSD_BAND = (0.5, 8.0)    # wag frequency search band (Hz)
PSD_SEG_MIN = 60         # minimum usable continuous segment length for PSD (frames, 2 s)

# Fence scale calibration: 6 panels, each physically 0.90 m wide
FENCE_SIDE_M = 0.90

# Head posture thresholds
HEAD_UP_DEG = 20.0
GAZE_DEG = 30.0            # dog head-to-human angle < 30 deg counts as "looking at human"
HUMAN_MOVE_MIN = 0.10      # human movement threshold (m/s, toe speed) for "human moving" frames
FOLLOW_MIN_VEL = 0.02      # minimum dog velocity component toward human (m/s) for "following"
HEAD_DOWN_DEG = -20.0

COND_EN = {"01": "S1\nno human", "02": "S2\ngaze away", "03": "S3\ngaze at dog"}
COND_FULL = {"01": "S1 (no human)", "02": "S2 (human, gaze away)",
             "03": "S3 (human, gaze at dog)"}
COND_S = {"01": "S1", "02": "S2", "03": "S3"}   # CSV output labels (avoid leading-zero loss)
BREED_EN = {"beagle": "Beagle", "pet": "Pet dog"}
BREED_COLOR = {"beagle": "#4477AA", "pet": "#EE6677"}   # Paul Tol, colour-blind friendly

METRIC_COLS = [
    "dog_mean_speed_mps", "dog_median_speed_mps", "dog_cum_dist_m", "dog_dist_rate_mpm",
    "human_toe_mean_speed_mps", "human_toe_cum_dist_m", "human_toe_dist_rate_mpm",
    "d_head_toe_mean_m", "d_head_toe_median_m", "d_head_toe_min_m",
    "d_head_toe_within1m_ratio",
    "tail_wag_amp_mean_mm", "tail_wag_freq_hz", "tail_elev_mean_deg", "tail_up_ratio",
    "head_pitch_mean_deg", "head_up_ratio", "head_down_ratio",
    "follow_ratio", "vel_align_mean",
]
METRIC_NICE = {
    "dog_mean_speed_mps": "Dog mean speed (m/s)",
    "dog_median_speed_mps": "Dog median speed (m/s)",
    "dog_cum_dist_m": "Dog cumulative distance (m)",
    "dog_dist_rate_mpm": "Dog distance rate (m/min)",
    "human_toe_mean_speed_mps": "Human toe mean speed (m/s)",
    "human_toe_cum_dist_m": "Human toe cumulative distance (m)",
    "human_toe_dist_rate_mpm": "Human toe distance rate (m/min)",
    "d_head_toe_mean_m": "Head\u2013toe distance, mean (m)",
    "d_head_toe_median_m": "Head\u2013toe distance, median (m)",
    "d_head_toe_min_m": "Head\u2013toe distance, min (m)",
    "d_head_toe_within1m_ratio": "Time within 1 m of human",
    "tail_wag_amp_mean_mm": "Tail wag amplitude (mm)",
    "tail_wag_freq_hz": "Tail wag frequency (Hz)",
    "tail_elev_mean_deg": "Tail elevation (deg)",
    "tail_up_ratio": "Tail-up time ratio",
    "head_pitch_mean_deg": "Head pitch (deg)",
    "head_up_ratio": "Head-up time ratio",
    "head_down_ratio": "Head-down time ratio",
    "follow_ratio": "Following ratio (while human moving)",
    "vel_align_mean": "Velocity alignment dog\u2013human (cos)",
}
HUMAN_ONLY = {"human_toe_mean_speed_mps", "human_toe_cum_dist_m",
              "human_toe_dist_rate_mpm",
              "d_head_toe_mean_m", "d_head_toe_median_m", "d_head_toe_min_m",
              "d_head_toe_within1m_ratio", "follow_ratio", "vel_align_mean"}

# Raw cumulative metrics affected by scenario duration (S1~138 s vs S2/S3~47-56 s,
# not directly comparable across scenarios):
# statistical tests and main figures always use the duration-corrected rate version
# (m/min); raw cumulative values are kept only in data/descriptive tables
CUM_RAW = {"dog_cum_dist_m", "human_toe_cum_dist_m"}
TEST_COLS = [m for m in METRIC_COLS if m not in CUM_RAW]

# Depth back-projection reference convention (same-name metrics with _depth
# suffix), only for assessing the accuracy of the fence-scale approximation
DEPTH_REF_COLS = [
    "dog_mean_speed_mps_depth", "dog_median_speed_mps_depth",
    "dog_cum_dist_m_depth", "dog_dist_rate_mpm_depth",
    "human_toe_mean_speed_mps_depth", "human_toe_cum_dist_m_depth",
    "human_toe_dist_rate_mpm_depth",
    "d_head_toe_mean_m_depth", "d_head_toe_min_m_depth",
    "tail_wag_amp_mean_mm_depth",
]


def _fin(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def stars_p(q):
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"


def holm(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = np.maximum.accumulate((n - np.arange(n)) * ranked)
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out


# ===================== Step 0: per-recording fence corners (720p pixels) =====================
def step0_fence():
    """Per-recording fence corners (annotated on 480p video -> converted to 720p
    pixels, same frame as dog/human keypoints) + scale calibration.

    Scale: the fence consists of 6 panels, each physically FENCE_SIDE_M=0.90 m
    wide -> scale_m_per_px = 0.90 / mean hexagon side length in pixels.
    Returns (per_rec, per_scale, qc):
      per_rec   = {(batch, rec): ndarray(6,2) 720p pixel corners}
      per_scale = {(batch, rec): m/px}
      qc        = DataFrame per-recording fence QC (incl. scale/side lengths)
    Outputs: data/fence_corners_per_rec.csv, data/fence_qc_per_rec.csv.
    """
    print("=" * 78)
    print(f"Step 0  Per-recording fence corners + scale calibration "
          f"(6 panels x {FENCE_SIDE_M:.2f} m/panel)")
    print(f"  Input: {FENCE_CSV}")
    fc = pd.read_csv(FENCE_CSV, dtype={"batch": str, "rec": str})
    per_rec, per_scale, qc_rows = {}, {}, []
    for (b, r), sub in fc.groupby(["batch", "rec"]):
        sub = sub.sort_values("corner_id")
        pts = sub[["color_x_720", "color_y_720"]].to_numpy(float)
        per_rec[(b, r)] = pts
        row = {"batch": b, "rec": r, "n_corners": len(pts),
               "cx_px": pts[:, 0].mean() if len(pts) else np.nan,
               "cy_px": pts[:, 1].mean() if len(pts) else np.nan}
        if len(pts) == 6:
            sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            side_mean = float(sides.mean())
            # Scale calibration: 6 panels x 0.90 m -> scale = 0.90 / mean side length (px)
            scale = FENCE_SIDE_M / side_mean
            per_scale[(b, r)] = scale
            sides_m = sides * scale            # per-panel converted length (should be ~0.90 m)
            row.update(side_mean_px=side_mean,
                       scale_m_per_px=scale,
                       scale_mm_per_px=scale * 1000.0,
                       side_m_min=float(sides_m.min()),
                       side_m_max=float(sides_m.max()),
                       side_m_std=float(sides_m.std()),
                       perimeter_m=float(sides_m.sum()))
        else:
            row.update(side_mean_px=np.nan,
                       scale_m_per_px=np.nan, scale_mm_per_px=np.nan,
                       side_m_min=np.nan, side_m_max=np.nan,
                       side_m_std=np.nan, perimeter_m=np.nan)
        qc_rows.append(row)
    out_csv = os.path.join(D_DATA, "fence_corners_per_rec.csv")
    fc[["batch", "rec", "corner_id", "color_x_720", "color_y_720",
        "video_x", "video_y"]].sort_values(["batch", "rec", "corner_id"]).to_csv(
            out_csv, index=False)
    qc = pd.DataFrame(qc_rows)
    qc_csv = os.path.join(D_DATA, "fence_qc_per_rec.csv")
    qc.to_csv(qc_csv, index=False)
    if len(qc):
        print(f"  Annotated recordings = {len(qc)} "
              f"(with full 6 corners = {int((qc.n_corners == 6).sum())})")
        ok = qc[qc.n_corners == 6]
        if len(ok):
            print(f"  Scale = 0.90 m / mean side length (px): median "
                  f"{ok.scale_mm_per_px.median():.3f} mm/px "
                  f"(range {ok.scale_mm_per_px.min():.3f}-"
                  f"{ok.scale_mm_per_px.max():.3f})")
            print(f"  Side length (px): mean {ok.side_mean_px.mean():.1f}; "
                  f"median per-panel SD {ok.side_m_std.median()*1000:.1f} mm, "
                  f"range {ok.side_m_min.min():.3f}-{ok.side_m_max.max():.3f} m")
            print(f"  Fence centre (px): x {ok.cx_px.mean():.1f} \u00b1 {ok.cx_px.std():.1f}, "
                  f"y {ok.cy_px.mean():.1f} \u00b1 {ok.cy_px.std():.1f}")
    print(f"  Output: {out_csv}")
    print(f"  Output: {qc_csv}")
    return per_rec, per_scale, qc


def load_time_windows():
    """Per-recording effective time window (start frame/end frame, same frame
    convention as the stage0 keypoint 'frame' column).

    Input: depth/圆形围栏统计开始结束表格.xlsx (sheet '圆形围栏统计',
          columns: 视频文件夹/开始帧/结束帧).
    Unannotated (NaN) or recordings missing from the table -> None (use the
    full video duration).
    Returns: {rec: (start, end) | None}.
    """
    print("=" * 78)
    print("Step 0b  Per-recording effective time window (start/end frames)")
    print(f"  Input: {XLSX_TIME}")
    tw = pd.read_excel(XLSX_TIME, sheet_name=0)
    tw.columns = ["rec", "start", "end"]
    win = {}
    for r in tw.itertuples(index=False):
        win[str(r.rec)] = (None if (pd.isna(r.start) or pd.isna(r.end))
                           else (int(r.start), int(r.end)))
    n_win = sum(v is not None for v in win.values())
    print(f"  Recordings in table = {len(win)}, annotated windows = {n_win}, "
          f"using full duration = {len(win) - n_win}")
    return win


# ===================== Metric computation helpers =====================
def jump_remove_interp(x, y, jump=JUMP2_M, gap=GAP_INTERP):
    """Secondary jump removal (>jump m/frame, remove the latter frame) + short
    gap linear interpolation (<=gap frames, interior only)."""
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.hypot(np.diff(x), np.diff(y))
    bad = np.where(np.nan_to_num(d, nan=0.0) > jump)[0] + 1
    x[bad] = np.nan
    y[bad] = np.nan
    xs = pd.Series(x).interpolate("linear", limit=gap, limit_area="inside")
    ys = pd.Series(y).interpolate("linear", limit=gap, limit_area="inside")
    return xs.to_numpy(), ys.to_numpy()


def inst_wag_freq(off, fps=FPS, pp_win=31, min_pp_mm=15.0, max_hz=8.0, smooth_win=15):
    """Estimate frame-wise tail wag frequency (Hz) with the zero-crossing
    half-period method.

    off: signed lateral offset series (mm, NaN = tail tip invisible), uniformly
    sampled.
    Steps: linear interpolation of zero-crossing times within continuous valid
    segments -> half periods -> instantaneous frequency;
    positions with < 15 mm peak-to-peak amplitude within a 1 s window are set
    to NaN (stationary-noise gating);
    0.5 s sliding median smoothing. Returns a frequency array (Hz) the same
    length as off.
    """
    n = len(off)
    freq = np.full(n, np.nan)
    if n < 4:
        return freq
    off = np.asarray(off, float)
    valid = np.isfinite(off)
    idx = np.where(valid)[0]
    if len(idx) < 4:
        return freq
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for seg in segs:
        if len(seg) < 4:
            continue
        y = off[seg]
        s = np.sign(y)
        cross = np.where(s[:-1] * s[1:] < 0)[0]
        if len(cross) < 2:
            continue
        f_idx = seg.astype(float)
        zc = f_idx[cross] - y[cross] * (f_idx[cross + 1] - f_idx[cross]) / \
             (y[cross + 1] - y[cross])
        half_p = np.diff(zc) / fps
        fz = 1.0 / (2.0 * np.maximum(half_p, 1e-9))
        fz[~np.isfinite(fz) | (fz > max_hz)] = np.nan
        if not np.isfinite(fz).any():
            continue
        pos = np.clip(np.searchsorted(zc, f_idx, side="right") - 1, 0, len(fz) - 1)
        f_seg = fz[pos]
        f_seg[~np.isfinite(f_seg)] = np.nan
        freq[seg] = f_seg
    ser = pd.Series(off)
    rmax = ser.rolling(pp_win, center=True, min_periods=pp_win // 3).max()
    rmin = ser.rolling(pp_win, center=True, min_periods=pp_win // 3).min()
    pp = (rmax - rmin).to_numpy(float)
    freq[pp < min_pp_mm] = np.nan
    freq = (pd.Series(freq).rolling(smooth_win, center=True, min_periods=3)
            .median().to_numpy(float))
    return freq


def speed_from_xy(x, y):
    """Central-difference speed + 5-frame sliding median (spike suppression)."""
    vx = np.gradient(x, DT)
    vy = np.gradient(y, DT)
    sp = np.hypot(vx, vy)
    sp = pd.Series(sp).rolling(SPEED_MED_WIN, center=True, min_periods=3).median()
    return sp.to_numpy()


def cum_from_xy(x, y):
    step = np.hypot(np.diff(x), np.diff(y))
    step = np.nan_to_num(step, nan=0.0)
    return np.concatenate([[0.0], np.cumsum(step)])


def wag_frequency(lat_interp):
    """Per-segment Welch PSD over visible segments (interpolated continuous
    segments >= PSD_SEG_MIN frames), length-weighted average, then dominant
    frequency."""
    valid = np.isfinite(lat_interp)
    if valid.sum() < PSD_SEG_MIN:
        return np.nan
    segs = []
    start = None
    for i in range(len(valid)):
        if valid[i] and start is None:
            start = i
        elif not valid[i] and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(valid)))
    usable = [(a, b) for a, b in segs if b - a >= PSD_SEG_MIN]
    if not usable:
        return np.nan
    # Unified nperseg (<=128 and not exceeding the shortest segment) so all
    # segment PSDs share the same frequency grid
    nper = min(128, min(b - a for a, b in usable))
    psd_sum, w_sum, fs = None, 0.0, None
    for a, b in usable:
        seg = lat_interp[a:b]
        f, p = signal.welch(seg, fs=FPS, nperseg=nper, detrend="constant")
        if psd_sum is None:
            psd_sum, fs = p * len(seg), f
        else:
            psd_sum += p * len(seg)
        w_sum += len(seg)
    pxx = psd_sum / w_sum
    band = (fs >= PSD_BAND[0]) & (fs <= PSD_BAND[1])
    if not band.any():
        return np.nan
    return float(fs[band][np.argmax(pxx[band])])


def process_recording(g, twin=None, scale=np.nan):
    """Compute frame-wise metrics for a single recording. g = that recording's
    stage0 long table (dog + human);
    twin=(start frame, end frame) effective time window, None = use all time;
    scale = that recording's fence scale calibration (m/px, primary
    convention). Returns (frame-metric df, QC dict).

    Two conventions:
      primary  = 720p pixel trajectory x scale (fence 6 panels x 0.90 m)
                 -> speed/distance/wag amplitude
      reference = stage0 depth back-projection x_m/y_m (_depth suffix), for
                 approximation-accuracy assessment
    """
    rec = g["rec"].iloc[0]
    batch = g["batch"].iloc[0]

    # ---- Effective time-window filter (keep only in-window frames;
    #      recordings unannotated in the table are not trimmed) ----
    if twin is not None:
        g = g[(g["frame"] >= twin[0]) & (g["frame"] <= twin[1])]

    # ---- Pivot to wide table ----
    dog = g[g.subject == "dog"]
    hum = g[g.subject == "human"]

    w = dog.pivot(index="frame", columns="body_part",
                  values=["x_m", "y_m", "h_m", "color_x", "color_y",
                          "confidence", "valid"])
    w.columns = w.columns.map(lambda t: (t[0], PART.get(t[1], t[1])))
    w = w.sort_index()
    frames = w.index.to_numpy()
    xH, yH = w[("x_m", "H")].to_numpy(float), w[("y_m", "H")].to_numpy(float)
    xW, yW = w[("x_m", "W")].to_numpy(float), w[("y_m", "W")].to_numpy(float)
    xB, yB = w[("x_m", "B")].to_numpy(float), w[("y_m", "B")].to_numpy(float)
    xT, yT = w[("x_m", "T")].to_numpy(float), w[("y_m", "T")].to_numpy(float)
    pH = w[("color_x", "H")].to_numpy(float)
    pW = w[("color_x", "W")].to_numpy(float)
    pB = w[("color_x", "B")].to_numpy(float)
    pT = w[("color_x", "T")].to_numpy(float)
    qH = w[("color_y", "H")].to_numpy(float)
    qW = w[("color_y", "W")].to_numpy(float)
    qB = w[("color_y", "B")].to_numpy(float)
    qT = w[("color_y", "T")].to_numpy(float)
    hH = w[("h_m", "H")].to_numpy(float)
    hW = w[("h_m", "W")].to_numpy(float)
    hB = w[("h_m", "B")].to_numpy(float)
    hT = w[("h_m", "T")].to_numpy(float)
    confT = w[("confidence", "T")].to_numpy(float)
    validT = w[("valid", "T")].to_numpy(bool)

    # ---- Dog centroid: primary convention (720p pixels x scale) + reference
    #      convention (depth back-projection) ----
    # body_ok: H/W/B pixel coordinates all valid (shared by both conventions)
    body_ok = (np.isfinite(pH) & np.isfinite(qH) & np.isfinite(pW) & np.isfinite(qW)
               & np.isfinite(pB) & np.isfinite(qB))
    cen_x = np.where(body_ok, (xH + xW + xB) / 3.0, np.nan)
    cen_y = np.where(body_ok, (yH + yW + yB) / 3.0, np.nan)
    cx_q, cy_q = jump_remove_interp(cen_x, cen_y)
    dog_speed_d = speed_from_xy(cx_q, cy_q)          # reference (depth)
    dog_cum_d = cum_from_xy(cx_q, cy_q)

    cen_px_x = np.where(body_ok, (pH + pW + pB) / 3.0, np.nan)
    cen_px_y = np.where(body_ok, (qH + qW + qB) / 3.0, np.nan)
    jump_px = JUMP2_M / scale if np.isfinite(scale) else np.inf
    cpx_q, cpy_q = jump_remove_interp(cen_px_x, cen_px_y, jump=jump_px)
    dog_speed = speed_from_xy(cpx_q, cpy_q) * scale  # primary (fence scale, m/s)
    dog_cum = cum_from_xy(cpx_q, cpy_q) * scale

    # ---- Human left toe tip (both conventions) ----
    if len(hum):
        wh = hum.pivot(index="frame", columns="body_part",
                       values=["x_m", "y_m", "color_x", "color_y"])
        wh = wh.sort_index().reindex(frames)
        toe_x = wh[("x_m", "left_toe_tip")].to_numpy(float)
        toe_y = wh[("y_m", "left_toe_tip")].to_numpy(float)
        toe_px = wh[("color_x", "left_toe_tip")].to_numpy(float)
        toe_py = wh[("color_y", "left_toe_tip")].to_numpy(float)
    else:
        toe_x = np.full(len(frames), np.nan)
        toe_y = np.full(len(frames), np.nan)
        toe_px = np.full(len(frames), np.nan)
        toe_py = np.full(len(frames), np.nan)
    tx_q, ty_q = jump_remove_interp(toe_x, toe_y)
    toe_speed_d = speed_from_xy(tx_q, ty_q)
    toe_cum_d = cum_from_xy(tx_q, ty_q)
    tpx_q, tpy_q = jump_remove_interp(toe_px, toe_py, jump=jump_px)
    toe_speed = speed_from_xy(tpx_q, tpy_q) * scale
    toe_cum = cum_from_xy(tpx_q, tpy_q) * scale

    # ---- Dog head - human left toe tip XY distance (both conventions) ----
    d_ht = np.hypot(pH - tpx_q, qH - tpy_q) * scale   # primary
    d_ht_d = np.hypot(xH - tx_q, yH - ty_q)           # reference (depth)

    # ---- Tail left-right wagging (XY plane, both conventions) ----
    # Pixel domain (primary): lateral offset of tail tip relative to the body
    # axis (W->B) in px x scale
    axp, ayp = pB - pW, qB - qW
    Lp = np.hypot(axp, ayp)
    axis_ok = Lp > 5.0
    upx = np.where(axis_ok, axp / np.maximum(Lp, 1e-9), np.nan)
    upy = np.where(axis_ok, ayp / np.maximum(Lp, 1e-9), np.nan)
    lat_px = upx * (qT - qB) - upy * (pT - pB)
    # Metre domain (reference): depth back-projection coordinates
    ax, ay = xB - xW, yB - yW
    L_axis = np.hypot(ax, ay)
    axis_ok_d = L_axis > 0.05
    ux = np.where(axis_ok_d, ax / np.maximum(L_axis, 1e-9), np.nan)
    uy = np.where(axis_ok_d, ay / np.maximum(L_axis, 1e-9), np.nan)
    lat_m_d = ux * (yT - yB) - uy * (xT - xB)
    tail_vis = (np.isfinite(pT) & np.isfinite(qT) & np.isfinite(pW) & np.isfinite(qW)
                & np.isfinite(pB) & np.isfinite(qB) & (confT >= CONF_TAIL) & validT
                & axis_ok & axis_ok_d)
    lat_mm = np.where(tail_vis, lat_px, np.nan) * scale * 1000.0    # primary (mm)
    lat_mm_d = np.where(tail_vis, lat_m_d, np.nan) * 1000.0         # reference (mm)
    # Interpolate short gaps, then take a 1 s sliding median as the slow-drift
    # baseline; dynamic deviation = wagging
    def _lat_dev(v):
        vi = pd.Series(v).interpolate("linear", limit=GAP_INTERP,
                                      limit_area="inside").to_numpy()
        b = (pd.Series(vi).rolling(WAG_BASE_WIN, center=True, min_periods=15)
             .median().to_numpy())
        return vi, vi - b
    lat_i, lat_dev = _lat_dev(lat_mm)
    _, lat_dev_d = _lat_dev(lat_mm_d)

    # ---- Depth-specific: head pitch / tail elevation ----
    dxy_HW = np.hypot(xH - xW, yH - yW)
    head_pitch = np.degrees(np.arctan2(hH - hW, np.maximum(dxy_HW, 0.02)))
    dxy_BT = np.hypot(xT - xB, yT - yB)
    tail_elev = np.degrees(np.arctan2(hT - hB, np.maximum(dxy_BT, 0.02)))
    tail_elev = np.where(tail_vis & np.isfinite(tail_elev), tail_elev, np.nan)
    head_pitch = np.where(np.isfinite(head_pitch) & np.isfinite(hH) & np.isfinite(hW),
                          head_pitch, np.nan)

    # ---- Whether the dog head faces the human ("looking at human"): angle
    #      between W->H forward direction and dog-head->human-left-toe direction ----
    fxh, fyh = pH - pW, qH - qW
    vxh, vyh = tpx_q - pH, tpy_q - qH
    fn_ = np.hypot(fxh, fyh)
    vn_ = np.hypot(vxh, vyh)
    cosang = (fxh * vxh + fyh * vyh) / np.maximum(fn_ * vn_, 1e-9)
    gaze_ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    gaze_ok = np.isfinite(fn_) & (fn_ > 5.0) & np.isfinite(vn_) & (vn_ > 1.0)
    gaze_ang = np.where(gaze_ok, gaze_ang, np.nan)
    gaze_at_human = np.where(gaze_ok, (gaze_ang < GAZE_DEG).astype(float), np.nan)

    # ---- Dog following the moving human (following behaviour,
    #      human-present scenarios only) ----
    # Dog/human velocity vectors (same QC as primary convention: 5-frame
    # sliding median smoothing)
    dvx = (pd.Series(np.gradient(cpx_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    dvy = (pd.Series(np.gradient(cpy_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    hvx = (pd.Series(np.gradient(tpx_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    hvy = (pd.Series(np.gradient(tpy_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    dv, hv = np.hypot(dvx, dvy), np.hypot(hvx, hvy)
    # Dog centroid -> human left toe tip direction
    dx_ = (tpx_q - cpx_q) * scale
    dy_ = (tpy_q - cpy_q) * scale
    d_cen = np.hypot(dx_, dy_)
    ux_ = np.where(d_cen > 1e-6, dx_ / np.maximum(d_cen, 1e-9), np.nan)
    uy_ = np.where(d_cen > 1e-6, dy_ / np.maximum(d_cen, 1e-9), np.nan)
    approach_vel = dvx * ux_ + dvy * uy_     # dog velocity component toward human (+ = approaching, m/s)
    ok_h = np.isfinite(hv)
    human_moving = np.where(ok_h, (hv >= HUMAN_MOVE_MIN).astype(float), np.nan)
    follow_state = np.where(
        ok_h & np.isfinite(approach_vel),
        ((hv >= HUMAN_MOVE_MIN) & (approach_vel >= FOLLOW_MIN_VEL)).astype(float),
        np.nan)                              # human moving AND dog moving toward human = following
    vel_align = np.where((dv > 0.05) & (hv > 0.05),
                         (dvx * hvx + dvy * hvy) / np.maximum(dv * hv, 1e-9), np.nan)

    # ---- Instantaneous wag frequency (zero-crossing half-period method,
    #      1 s peak-to-peak >= 15 mm gating) ----
    wag_freq_inst = inst_wag_freq(lat_mm)

    out = pd.DataFrame({
        "batch": batch, "rec": rec, "frame": frames,
        # Primary convention (fence scale): 720p pixels x scale
        "dog_speed_mps": dog_speed, "dog_cum_m": dog_cum,
        "toe_speed_mps": toe_speed, "toe_cum_m": toe_cum,
        "d_head_toe_m": d_ht,
        "tail_lat_mm": lat_mm, "tail_lat_dev_mm": lat_dev,
        # Reference convention (depth back-projection)
        "dog_cx_m": cx_q, "dog_cy_m": cy_q,
        "dog_speed_mps_depth": dog_speed_d, "dog_cum_m_depth": dog_cum_d,
        "toe_x_m": tx_q, "toe_y_m": ty_q,
        "toe_speed_mps_depth": toe_speed_d, "toe_cum_m_depth": toe_cum_d,
        "d_head_toe_m_depth": d_ht_d,
        "tail_lat_mm_depth": lat_mm_d, "tail_lat_dev_mm_depth": lat_dev_d,
        # 720p pixel trajectories (after QC, for trajectory plots / fence
        # containment ratio)
        "dog_cx_px": cpx_q, "dog_cy_px": cpy_q,
        "toe_x_px": tpx_q, "toe_y_px": tpy_q,
        "tail_visible": tail_vis,
        "head_pitch_deg": head_pitch, "tail_elev_deg": tail_elev,
        "gaze_angle_deg": gaze_ang, "gaze_at_human": gaze_at_human,
        "tail_wag_freq_inst_hz": wag_freq_inst,
        "approach_vel_mps": approach_vel, "human_moving": human_moving,
        "follow_state": follow_state, "vel_align_cos": vel_align,
    })

    qc = {
        "n_frames": int(len(frames)),
        "dog_valid_ratio": float(body_ok.mean()),
        "tail_vis_ratio": float(tail_vis.mean()),
        "human_toe_valid_ratio": float(np.isfinite(toe_x).mean()) if len(hum) else np.nan,
    }
    return out, qc


def summarize_recording(fm, qc, meta):
    """Per-recording summary metrics. fm = frame-wise metrics, qc = QC stats,
    meta = batch/rec/cond/breed/dog_id."""
    n = qc["n_frames"]
    dur = n / FPS
    row = {**meta, **qc, "duration_s": dur}

    dog_ok = fm["dog_speed_mps"].to_numpy(float)
    dog_ok = dog_ok[np.isfinite(dog_ok)]
    row["dog_mean_speed_mps"] = float(dog_ok.mean()) if len(dog_ok) else np.nan
    row["dog_median_speed_mps"] = float(np.median(dog_ok)) if len(dog_ok) else np.nan
    cum = fm["dog_cum_m"].to_numpy(float)
    row["dog_cum_dist_m"] = float(np.nanmax(cum)) if np.isfinite(cum).any() else np.nan
    row["dog_dist_rate_mpm"] = (row["dog_cum_dist_m"] / dur * 60.0
                                if np.isfinite(row["dog_cum_dist_m"]) else np.nan)

    toe_sp = fm["toe_speed_mps"].to_numpy(float)
    toe_sp = toe_sp[np.isfinite(toe_sp)]
    row["human_toe_mean_speed_mps"] = float(toe_sp.mean()) if len(toe_sp) else np.nan
    tcum = fm["toe_cum_m"].to_numpy(float)
    row["human_toe_cum_dist_m"] = float(np.nanmax(tcum)) if np.isfinite(tcum).any() else np.nan
    row["human_toe_dist_rate_mpm"] = (row["human_toe_cum_dist_m"] / dur * 60.0
                                      if np.isfinite(row["human_toe_cum_dist_m"]) else np.nan)

    d = fm["d_head_toe_m"].to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d):
        row["d_head_toe_mean_m"] = float(d.mean())
        row["d_head_toe_median_m"] = float(np.median(d))
        row["d_head_toe_min_m"] = float(d.min())
        row["d_head_toe_within1m_ratio"] = float((d < 1.0).mean())
    else:
        for k in ["d_head_toe_mean_m", "d_head_toe_median_m",
                  "d_head_toe_min_m", "d_head_toe_within1m_ratio"]:
            row[k] = np.nan

    vis = fm["tail_visible"].to_numpy(bool)
    lat_dev = fm["tail_lat_dev_mm"].to_numpy(float)
    dev_ok = lat_dev[vis & np.isfinite(lat_dev)]
    row["tail_wag_amp_mean_mm"] = float(np.mean(np.abs(dev_ok))) if len(dev_ok) else np.nan
    lat_series = fm["tail_lat_mm"].to_numpy(float)
    lat_i = pd.Series(lat_series).interpolate("linear", limit=GAP_INTERP,
                                              limit_area="inside").to_numpy()
    row["tail_wag_freq_hz"] = wag_frequency(lat_i) if qc["tail_vis_ratio"] >= TAIL_VIS_MIN else np.nan

    elev = fm["tail_elev_deg"].to_numpy(float)
    elev = elev[np.isfinite(elev)]
    row["tail_elev_mean_deg"] = float(elev.mean()) if len(elev) else np.nan
    row["tail_up_ratio"] = float((elev > 0).mean()) if len(elev) else np.nan

    # ---- Following behaviour (human-present scenarios only) ----
    hm_ = fm["human_moving"].to_numpy(float)
    fs_ = fm["follow_state"].to_numpy(float)
    m = (hm_ == 1) & np.isfinite(fs_)
    row["follow_ratio"] = float(fs_[m].mean()) if m.sum() else np.nan
    va_ = fm["vel_align_cos"].to_numpy(float)
    va_ = va_[np.isfinite(va_)]
    row["vel_align_mean"] = float(va_.mean()) if len(va_) else np.nan

    pitch = fm["head_pitch_deg"].to_numpy(float)
    pitch = pitch[np.isfinite(pitch)]
    row["head_pitch_mean_deg"] = float(pitch.mean()) if len(pitch) else np.nan
    row["head_up_ratio"] = float((pitch > HEAD_UP_DEG).mean()) if len(pitch) else np.nan
    row["head_down_ratio"] = float((pitch < HEAD_DOWN_DEG).mean()) if len(pitch) else np.nan

    # ---- Reference convention (depth back-projection) summary, for
    #      fence-scale approximation-accuracy assessment ----
    dsp = fm["dog_speed_mps_depth"].to_numpy(float)
    dsp = dsp[np.isfinite(dsp)]
    row["dog_mean_speed_mps_depth"] = float(dsp.mean()) if len(dsp) else np.nan
    row["dog_median_speed_mps_depth"] = float(np.median(dsp)) if len(dsp) else np.nan
    dcum = fm["dog_cum_m_depth"].to_numpy(float)
    row["dog_cum_dist_m_depth"] = (float(np.nanmax(dcum))
                                   if np.isfinite(dcum).any() else np.nan)
    row["dog_dist_rate_mpm_depth"] = (row["dog_cum_dist_m_depth"] / dur * 60.0
                                      if np.isfinite(row["dog_cum_dist_m_depth"]) else np.nan)
    tsp = fm["toe_speed_mps_depth"].to_numpy(float)
    tsp = tsp[np.isfinite(tsp)]
    row["human_toe_mean_speed_mps_depth"] = float(tsp.mean()) if len(tsp) else np.nan
    tcum = fm["toe_cum_m_depth"].to_numpy(float)
    row["human_toe_cum_dist_m_depth"] = (float(np.nanmax(tcum))
                                         if np.isfinite(tcum).any() else np.nan)
    row["human_toe_dist_rate_mpm_depth"] = (row["human_toe_cum_dist_m_depth"] / dur * 60.0
                                            if np.isfinite(row["human_toe_cum_dist_m_depth"])
                                            else np.nan)
    dd = fm["d_head_toe_m_depth"].to_numpy(float)
    dd = dd[np.isfinite(dd)]
    row["d_head_toe_mean_m_depth"] = float(dd.mean()) if len(dd) else np.nan
    row["d_head_toe_min_m_depth"] = float(dd.min()) if len(dd) else np.nan
    dev_d = fm["tail_lat_dev_mm_depth"].to_numpy(float)
    dev_d = dev_d[vis & np.isfinite(dev_d)]
    row["tail_wag_amp_mean_mm_depth"] = (float(np.mean(np.abs(dev_d)))
                                         if len(dev_d) else np.nan)

    # Exclusion rules
    if qc["dog_valid_ratio"] < DOG_VALID_MIN:
        row["excluded"] = True
        row["exclude_reason"] = f"dog_valid_ratio<{DOG_VALID_MIN}"
        for k in METRIC_COLS + DEPTH_REF_COLS:
            row[k] = np.nan
    else:
        row["excluded"] = False
        row["exclude_reason"] = ""
        if qc["tail_vis_ratio"] < TAIL_VIS_MIN:
            row["exclude_reason"] = f"tail_vis_ratio<{TAIL_VIS_MIN} (tail metrics NaN)"
            for k in ["tail_wag_amp_mean_mm", "tail_wag_freq_hz",
                      "tail_elev_mean_deg", "tail_up_ratio"]:
                row[k] = np.nan
    return row


# ===================== Steps 1-3: QC + frame-wise metrics + per-recording summary =====================
def step1to3_metrics(per_rec, per_scale, twin):
    print("=" * 78)
    print("Steps 1-3  QC + frame-wise metrics + per-recording summary "
          "(within effective time windows, fence-scale primary convention)")
    print(f"  Input: {S0_ALL}")
    t0 = time.time()
    df = pd.read_csv(S0_ALL, dtype={"batch": str, "rec": str, "subject": str})
    print(f"  Loaded {len(df)} rows, "
          f"{df[['batch','rec']].drop_duplicates().shape[0]} recordings")

    frame_parts, rec_rows, no_fence, no_table = [], [], [], []
    for (batch, rec), g in df.groupby(["batch", "rec"], sort=True):
        g = g.copy()
        cond = rec.split("_")[-1]
        breed = "pet" if batch == PET_BATCH else "beagle"
        meta = {"batch": batch, "rec": rec, "cond": cond, "breed": breed,
                "dog_id": f"{batch}_{rec.split('_')[-2]}"}
        if rec not in twin:
            no_table.append(rec)
        w = twin.get(rec)
        fm, qc = process_recording(g, w, per_scale.get((batch, rec), np.nan))
        qc["win_start"] = float(w[0]) if w is not None else np.nan
        qc["win_end"] = float(w[1]) if w is not None else np.nan
        # Time share of the dog centroid inside that recording's fence
        # (720p pixels); unannotated fence -> NaN
        fpx = per_rec.get((batch, rec))
        if fpx is not None and len(fpx) == 6:
            poly = MplPath(fpx)
            pts = fm[["dog_cx_px", "dog_cy_px"]].to_numpy(float)
            okm = np.isfinite(pts).all(axis=1)
            qc["dog_inside_fence_ratio"] = (float(poly.contains_points(pts[okm]).mean())
                                            if okm.sum() else np.nan)
        else:
            qc["dog_inside_fence_ratio"] = np.nan
            no_fence.append((batch, rec))
        frame_parts.append(fm)
        rec_rows.append(summarize_recording(fm, qc, meta))

    fm_all = pd.concat(frame_parts, ignore_index=True)
    recs = pd.DataFrame(rec_rows)
    cols = (["batch", "rec", "dog_id", "cond", "breed", "duration_s", "n_frames",
             "win_start", "win_end",
             "dog_valid_ratio", "tail_vis_ratio", "human_toe_valid_ratio",
             "dog_inside_fence_ratio", "excluded", "exclude_reason"]
            + METRIC_COLS + DEPTH_REF_COLS)
    recs = recs[cols]

    p_fm = os.path.join(D_DATA, "xy_frame_metrics.csv")
    p_rec = os.path.join(D_DATA, "xy_rec_metrics.csv")
    p_qc = os.path.join(D_DATA, "xy_qc_report.csv")
    fm_all.to_csv(p_fm, index=False)
    recs_out = recs.copy()
    recs_out["cond"] = recs_out["cond"].map(COND_S)
    recs_out.to_csv(p_rec, index=False)
    recs_out[cols[:15]].to_csv(p_qc, index=False)

    n_ex = int(recs.excluded.sum())
    n_tail = int((recs.exclude_reason.str.contains("tail", na=False)).sum())
    n_fence = int(recs.dog_inside_fence_ratio.notna().sum())
    n_w = int(recs.win_start.notna().sum())
    print(f"  Recordings = {len(recs)}, whole-recording excluded = {n_ex}, "
          f"tail metrics set to NaN = {n_tail}")
    print(f"  Effective time windows: table-annotated = {n_w}, "
          f"using full duration = {len(recs) - n_w}"
          + (f", missing from table = {len(no_table)}" if no_table else ""))
    for c in ["01", "02", "03"]:
        d = recs[recs.cond == c].duration_s
        print(f"    {COND_S[c]} effective duration (s): median {d.median():.0f}, "
              f"range {d.min():.0f}-{d.max():.0f}")
    print(f"  Recordings with annotated fence = {n_fence}/{len(recs)} "
          f"({len(no_fence)} unannotated; those trajectory plots omit the fence)")
    if n_fence:
        ins = recs.dog_inside_fence_ratio.dropna()
        print(f"  Dog centroid inside-fence time share: median {ins.median():.1%} "
              f"({int((ins < 0.80).sum())} recordings < 80%)")
    print(f"  Dog valid-frame ratio: median {recs.dog_valid_ratio.median():.1%}")
    print(f"  Tail tip visibility: median {recs.tail_vis_ratio.median():.1%}")
    print(f"  Output: {p_fm}")
    print(f"  Output: {p_rec}")
    print(f"  Output: {p_qc}")
    print(f"  Elapsed {time.time()-t0:.1f}s")
    return recs


# ===================== Step 4: descriptive statistics =====================
def desc_table(sub, metrics):
    rows = []
    for m in metrics:
        v = _fin(sub[m])
        if len(v) == 0:
            rows.append({"metric": m, "n": 0, "mean": np.nan, "sd": np.nan,
                         "median": np.nan, "q1": np.nan, "q3": np.nan,
                         "min": np.nan, "max": np.nan})
        else:
            rows.append({"metric": m, "n": len(v),
                         "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "median": np.median(v),
                         "q1": np.percentile(v, 25), "q3": np.percentile(v, 75),
                         "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def step4_describe(recs):
    print("=" * 78)
    print("Step 4  Descriptive statistics "
          "(overall/scenario/breed/breed x scenario)")
    keep = recs[~recs.excluded]
    outs = {}

    t = desc_table(keep, METRIC_COLS)
    t.insert(0, "scope", "overall")
    p = os.path.join(D_TAB, "dist_overall.csv")
    t.to_csv(p, index=False)
    outs["dist_overall.csv"] = p
    print(f"  Overall n={len(keep)} -> {p}")

    t = pd.concat([desc_table(keep[keep.cond == c], METRIC_COLS)
                   .assign(cond=COND_S[c], scope=f"cond={COND_S[c]}")
                   for c in ["01", "02", "03"]])
    t = t[["scope", "cond", "metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max"]]
    p = os.path.join(D_TAB, "dist_by_cond.csv")
    t.to_csv(p, index=False)
    outs["dist_by_cond.csv"] = p
    print(f"  By scenario 01/02/03 -> {p}")

    t = pd.concat([desc_table(keep[keep.breed == b], METRIC_COLS).assign(breed=b, scope=f"breed={b}")
                   for b in ["beagle", "pet"]])
    t = t[["scope", "breed", "metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max"]]
    p = os.path.join(D_TAB, "dist_by_breed.csv")
    t.to_csv(p, index=False)
    outs["dist_by_breed.csv"] = p
    print(f"  By breed beagle/pet -> {p}")

    t = pd.concat([desc_table(keep[(keep.breed == b) & (keep.cond == c)], METRIC_COLS)
                   .assign(breed=b, cond=COND_S[c], scope=f"breed={b},cond={COND_S[c]}")
                   for b in ["beagle", "pet"] for c in ["01", "02", "03"]])
    t = t[["scope", "breed", "cond", "metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max"]]
    p = os.path.join(D_TAB, "dist_by_breed_cond.csv")
    t.to_csv(p, index=False)
    outs["dist_by_breed_cond.csv"] = p
    print(f"  By breed x scenario -> {p}")
    return outs


# ===================== Step 5: statistical tests =====================
def friedman_w(x1, x2, x3):
    """Returns (statistic, p, Kendall's W, n). Requires all three paired samples
    to be finite."""
    m = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(x3)
    a, b, c = x1[m], x2[m], x3[m]
    n = len(a)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    try:
        stat, p = st.friedmanchisquare(a, b, c)
    except ValueError:
        return np.nan, np.nan, np.nan, n
    w = stat / (n * 2) if n > 0 else np.nan
    return float(stat), float(p), float(w), n


def wilcoxon_dz(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    n = len(d)
    if n < 4 or np.allclose(d, 0):
        return np.nan, np.nan, np.nan, n
    try:
        res = st.wilcoxon(x[m], y[m], zero_method="wilcox", alternative="two-sided",
                          method="auto")
        p = float(res.pvalue)
    except ValueError:
        return np.nan, np.nan, np.nan, n
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 1e-12 else np.nan
    return float(res.statistic), p, dz, n


def cliff_delta(x, y):
    x, y = _fin(x), _fin(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(int(np.sum(y < xv)) for xv in x)
    lt = sum(int(np.sum(y > xv)) for xv in x)
    return (gt - lt) / (len(x) * len(y))


def step5_tests(recs):
    print("=" * 78)
    print("Step 5  Statistical tests")
    keep = recs[~recs.excluded].copy()

    # ---- Scenario effect: Friedman + pairwise Wilcoxon (Holm), by breed and
    #      all dogs ----
    # Raw cumulative distances (CUM_RAW) are duration-confounded and excluded
    # from tests; the rate version (m/min) is used instead
    rows = []
    for scope in ["beagle", "pet", "all"]:
        sub = keep if scope == "all" else keep[keep.breed == scope]
        for m in TEST_COLS:
            if m in HUMAN_ONLY:
                continue
            piv = sub.pivot_table(index="dog_id", columns="cond", values=m)
            if not all(c in piv.columns for c in ["01", "02", "03"]):
                continue
            x1 = piv.get("01", pd.Series(dtype=float)).to_numpy(float)
            x2 = piv.get("02", pd.Series(dtype=float)).to_numpy(float)
            x3 = piv.get("03", pd.Series(dtype=float)).to_numpy(float)
            stat, p, w, n = friedman_w(x1, x2, x3)
            rows.append({"scope": scope, "metric": m, "test": "Friedman (S1/S2/S3)",
                         "pair": "", "n_dogs": n, "statistic": stat, "p": p,
                         "p_holm": p, "effect": "Kendall W", "effect_size": w})
            # Pairwise
            pairs = [("S1 vs S2", x1, x2), ("S1 vs S3", x1, x3), ("S2 vs S3", x2, x3)]
            raw = []
            for name, a, b in pairs:
                s, p_, dz, n_ = wilcoxon_dz(a, b)
                raw.append([name, s, p_, dz, n_])
            corr = holm([r[2] for r in raw])
            for (name, s, p_, dz, n_), q in zip(raw, corr):
                rows.append({"scope": scope, "metric": m, "test": "Wilcoxon signed-rank",
                             "pair": name, "n_dogs": n_, "statistic": s, "p": p_,
                             "p_holm": q, "effect": "Cohen dz", "effect_size": dz})
    # Human-related metrics: S2 vs S3 paired
    for scope in ["beagle", "pet", "all"]:
        sub = keep if scope == "all" else keep[keep.breed == scope]
        for m in [c for c in TEST_COLS if c in HUMAN_ONLY]:
            piv = sub.pivot_table(index="dog_id", columns="cond", values=m)
            if not all(c in piv.columns for c in ["02", "03"]):
                continue
            x2 = piv["02"].to_numpy(float)
            x3 = piv["03"].to_numpy(float)
            s, p_, dz, n_ = wilcoxon_dz(x2, x3)
            rows.append({"scope": scope, "metric": m, "test": "Wilcoxon signed-rank",
                         "pair": "S2 vs S3", "n_dogs": n_, "statistic": s, "p": p_,
                         "p_holm": p_, "effect": "Cohen dz", "effect_size": dz})
    ts = pd.DataFrame(rows)
    p = os.path.join(D_TAB, "test_scenario.csv")
    ts.to_csv(p, index=False)
    print(f"  Scenario tests {len(ts)} rows -> {p}")

    # ---- Breed effect: per-scenario Mann-Whitney U + Cliff's delta,
    #      cross-metric BH ----
    rows = []
    for c in ["01", "02", "03"]:
        sub = keep[keep.cond == c]
        for m in TEST_COLS:
            a = sub[sub.breed == "beagle"][m].to_numpy(float)
            b = sub[sub.breed == "pet"][m].to_numpy(float)
            a, b = _fin(a), _fin(b)
            if len(a) < 2 or len(b) < 2:
                rows.append({"cond": COND_S[c], "metric": m, "n_beagle": len(a),
                             "n_pet": len(b), "U": np.nan, "p": np.nan, "q_bh": np.nan,
                             "effect": "Cliff delta", "effect_size": np.nan})
                continue
            res = st.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"cond": COND_S[c], "metric": m, "n_beagle": len(a),
                         "n_pet": len(b), "U": float(res.statistic),
                         "p": float(res.pvalue), "q_bh": np.nan,
                         "effect": "Cliff delta", "effect_size": cliff_delta(a, b)})
    tb = pd.DataFrame(rows)
    for c in ["S1", "S2", "S3"]:
        msk = (tb.cond == c) & tb.p.notna()
        tb.loc[msk, "q_bh"] = bh(tb.loc[msk, "p"].to_numpy(float))
    p = os.path.join(D_TAB, "test_breed.csv")
    tb.to_csv(p, index=False)
    print(f"  Breed tests {len(tb)} rows -> {p}")
    return ts, tb


# ===================== Step 5b: fence-scale approximation accuracy and bias =====================
def step5b_scale_accuracy(recs, fence_qc):
    """Accuracy and bias of the fence-scale (6 panels x 0.90 m) approximation
    vs the depth back-projection reference.

    Primary convention = fence scale (720p pixels x scale); reference = depth
    back-projection (x_m/y_m).
    Outputs:
      tables/scale_accuracy_per_rec.csv   per recording: scale/per-panel
                                           bias/relative bias of each metric
      tables/scale_accuracy_summary.csv   overall: median bias/IQR/MAPE/max
                                           bias/Pearson r
      figures/fig_scale_accuracy.png      Bland-Altman plots (speed/cumulative
                                           distance/dog-human distance/wag
                                           amplitude)
    """
    print("=" * 78)
    print("Step 5b  Fence-scale approximation accuracy and bias "
          "(primary convention vs depth back-projection reference)")
    keep = recs[~recs.excluded].copy()
    fq = fence_qc[["batch", "rec", "scale_mm_per_px",
                   "side_m_min", "side_m_max", "side_m_std"]]
    keep = keep.merge(fq, on=["batch", "rec"], how="left")

    def rel_bias(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
        out = np.full(len(a), np.nan)
        out[m] = (a[m] - b[m]) / np.abs(b[m]) * 100.0
        return out

    pairs = [
        ("dog_mean_speed_mps", "dog_mean_speed_mps_depth", "speed_bias_pct",
         "Dog mean speed (m/s)"),
        ("dog_cum_dist_m", "dog_cum_dist_m_depth", "cum_dist_bias_pct",
         "Dog cumulative distance (m)"),
        ("d_head_toe_mean_m", "d_head_toe_mean_m_depth", "d_mean_bias_pct",
         "Head-toe distance (m)"),
        ("tail_wag_amp_mean_mm", "tail_wag_amp_mean_mm_depth", "wag_amp_bias_pct",
         "Tail wag amplitude (mm)"),
        ("human_toe_mean_speed_mps", "human_toe_mean_speed_mps_depth",
         "human_speed_bias_pct", "Human toe speed (m/s)"),
    ]
    for main_c, ref_c, bias_c, _ in pairs:
        keep[bias_c] = rel_bias(keep[main_c].to_numpy(), keep[ref_c].to_numpy())

    out_cols = (["batch", "rec", "cond", "breed", "scale_mm_per_px",
                 "side_m_min", "side_m_max", "side_m_std"]
                + [p[0] for p in pairs] + [p[1] for p in pairs]
                + [p[2] for p in pairs])
    p1 = os.path.join(D_TAB, "scale_accuracy_per_rec.csv")
    keep[out_cols].to_csv(p1, index=False)

    rows = []
    for main_c, ref_c, bias_c, label in pairs:
        b = keep[bias_c].to_numpy(float)
        b = b[np.isfinite(b)]
        a = keep[main_c].to_numpy(float)
        r_ = keep[ref_c].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(r_)
        r_p = (float(st.pearsonr(a[m], r_[m])[0])
               if (m.sum() > 2 and np.std(a[m]) > 0 and np.std(r_[m]) > 0) else np.nan)
        rows.append({
            "metric": label, "n": len(b),
            "bias_median_pct": round(float(np.median(b)), 2) if len(b) else np.nan,
            "bias_iqr_pct": (round(float(np.subtract(*np.percentile(b, [75, 25]))), 2)
                             if len(b) else np.nan),
            "mape_pct": round(float(np.mean(np.abs(b))), 2) if len(b) else np.nan,
            "max_abs_bias_pct": round(float(np.max(np.abs(b))), 2) if len(b) else np.nan,
            "pearson_r": round(r_p, 4) if np.isfinite(r_p) else np.nan,
        })
    summ = pd.DataFrame(rows)
    p2 = os.path.join(D_TAB, "scale_accuracy_summary.csv")
    summ.to_csv(p2, index=False)
    print(summ.to_string(index=False))
    print(f"  Output: {p1}")
    print(f"  Output: {p2}")

    # Bland-Altman: x = mean of the two conventions, y = relative bias (%)
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.8))
    for ax, (main_c, ref_c, bias_c, label) in zip(axes.ravel(), pairs[:4]):
        a = keep[main_c].to_numpy(float)
        r_ = keep[ref_c].to_numpy(float)
        b = keep[bias_c].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(r_) & np.isfinite(b)
        if m.sum() == 0:
            ax.axis("off")
            continue
        ax.scatter((a[m] + r_[m]) / 2, b[m], s=6, c=DOG_COLOR, alpha=0.6,
                   edgecolor="none")
        ax.axhline(0, color="0.3", lw=0.7)
        med, sd = float(np.median(b[m])), float(np.std(b[m]))
        ax.axhline(med, color="#EE6677", lw=0.8, ls="--",
                   label=f"median bias {med:+.1f}%")
        ax.axhline(med + 1.96 * sd, color="0.6", lw=0.6, ls=":")
        ax.axhline(med - 1.96 * sd, color="0.6", lw=0.6, ls=":",
                   label=f"95% LoA \u00b1{1.96 * sd:.1f}%")
        ax.set_xlabel(f"{label} (two-scale mean)")
        ax.set_ylabel("Fence vs depth bias (%)")
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Fence-scale calibration (6 panels \u00d7 0.90 m) vs depth "
                 "back-projection: Bland\u2013Altman", fontsize=11)
    fig.tight_layout()
    save_fig(fig, os.path.join(D_FIG, "fig_scale_accuracy"))
    return keep


# ===================== Step 6f: per-recording timeseries overview (unified time axis) =====================
def step6f_timeseries(fm_all):
    """One 8-panel timeseries overview figure per recording, all metrics on a
    unified time X-axis.

    Input: data/xy_frame_metrics.csv (in memory)
    Output: timeseries/ts_{batch}_{rec}.png
    Panels: 1 human/dog instantaneous speed | 2 human/dog cumulative distance |
            3 dog-human instantaneous distance |
            4 wag frequency (zero-crossing) | 5 signed tail lateral amplitude
            (1 s sliding RMS) | 6 tail elevation angle |
            7 head orientation to human (<30 deg = looking, shaded at bottom) |
            8 following behaviour
    """
    print("=" * 78)
    print("Step 6f  Per-recording timeseries overview (unified time axis)")
    print(f"  Input: {os.path.join(D_DATA, 'xy_frame_metrics.csv')} (in memory)")
    os.makedirs(D_TS, exist_ok=True)
    n = 0
    for (batch, rec), g in fm_all.groupby(["batch", "rec"], sort=True):
        g = g.sort_values("frame")
        t = (g["frame"].to_numpy(float) - float(g["frame"].iloc[0])) / FPS
        cond = rec.split("_")[-1]
        has_human = bool(g["d_head_toe_m"].notna().any())
        rec_short = rec.replace("录制", "rec")

        fig, axes = plt.subplots(8, 1, figsize=(7.2, 11.8), sharex=True)
        fig.subplots_adjust(hspace=0.85, top=0.94)

        # 1 Instantaneous speed: dog + human
        ax = axes[0]
        ax.plot(t, g["dog_speed_mps"], lw=0.7, color=DOG_COLOR, label="dog")
        if has_human:
            ax.plot(t, g["toe_speed_mps"], lw=0.7, color=HUMAN_COLOR,
                    label="human (toe)")
            ax.legend(fontsize=6.5, loc="upper right", ncol=2)
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Instantaneous speed", fontsize=8, loc="left")

        # 2 Cumulative distance
        ax = axes[1]
        ax.plot(t, g["dog_cum_m"], lw=0.9, color=DOG_COLOR, label="dog")
        if has_human:
            ax.plot(t, g["toe_cum_m"], lw=0.9, color=HUMAN_COLOR,
                    label="human (toe)")
            ax.legend(fontsize=6.5, loc="upper left")
        ax.set_ylabel("Cum. dist (m)")
        ax.set_title("Cumulative distance", fontsize=8, loc="left")

        # 3 Dog-human instantaneous distance
        ax = axes[2]
        if has_human:
            ax.plot(t, g["d_head_toe_m"], lw=0.7, color="#AA3377")
            for y, ls in [(0.5, "--"), (1.0, "-."), (2.0, ":")]:
                ax.axhline(y, color="0.75", lw=0.6, ls=ls)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "no human recorded", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.6")
        ax.set_ylabel("Dist (m)")
        ax.set_title("Dog head\u2013human toe distance", fontsize=8, loc="left")

        # 4 Wag frequency
        ax = axes[3]
        ax.plot(t, g["tail_wag_freq_inst_hz"], lw=0.7, color="#228833")
        ax.set_ylim(0, 8)
        ax.set_ylabel("Freq (Hz)")
        ax.set_title("Tail wag frequency (zero-crossing, pp\u2009\u2265\u200915 mm gate)",
                     fontsize=8, loc="left")

        # 5 Signed tail lateral amplitude (positive = left of body axis,
        #   negative = right, no absolute value)
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
                     fontsize=8, loc="left")

        # 6 Tail elevation angle
        ax = axes[5]
        ax.plot(t, g["tail_elev_deg"], lw=0.7, color="#EE6677")
        ax.axhline(0, color="0.75", lw=0.6, ls=":")
        ax.set_ylabel("Elev (deg)")
        ax.set_title("Tail elevation (up = positive)", fontsize=8, loc="left")

        # 7 Head orientation to human (<30 deg = looking, bottom shaded)
        ax = axes[6]
        if has_human:
            ax.plot(t, g["gaze_angle_deg"], lw=0.6, color="#4477AA")
            ax.axhline(GAZE_DEG, color="#EE6677", lw=0.8, ls="--",
                       label=f"looking at human (< {GAZE_DEG:.0f}\u00b0)")
            gz = g["gaze_at_human"].to_numpy(float)
            ax.fill_between(t, 0, 1, where=(gz == 1),
                            transform=ax.get_xaxis_transform(),
                            color="#4477AA", alpha=0.18, linewidth=0)
            ax.set_ylim(0, 180)
            ax.legend(fontsize=6.5, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no human recorded", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.6")
        ax.set_ylabel("Head angle (deg)")
        ax.set_title("Dog head orientation to human (shaded = looking)",
                     fontsize=8, loc="left")

        # 8 Dog following the moving human (following behaviour)
        ax = axes[7]
        if has_human:
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
        else:
            ax.text(0.5, 0.5, "no human recorded", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.6")
        ax.set_ylabel("V toward human (m/s)")
        ax.set_title("Following: dog velocity component toward human",
                     fontsize=8, loc="left")
        ax.set_xlabel("Time (s)")

        fig.suptitle(f"{batch} / {rec_short}  ({COND_S[cond]})", fontsize=10)
        fig.savefig(os.path.join(D_TS, f"ts_{batch}_{rec_short}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        n += 1
    print(f"  Output: {D_TS}/ts_*.png ({n} figures)")


# ===================== Step 6: visualization =====================
def _fence_patch(fence_px, ax, label="Hexagonal fence"):
    if fence_px is None or len(fence_px) < 3:
        return
    poly = MplPolygon(fence_px, closed=True, facecolor="0.92", edgecolor="0.35",
                      lw=0.9, alpha=0.9, zorder=0, label=label)
    ax.add_patch(poly)
    ax.scatter(fence_px[:, 0], fence_px[:, 1], s=6, c="0.25", zorder=3)


def plot_trajectory(ax, fm, fence_px, title=""):
    """Trajectory plot in the 720p image pixel coordinate system (same frame
    as dog/human keypoints; fence = that recording's own 6 corners)."""
    _fence_patch(fence_px, ax)
    dog = fm[["dog_cx_px", "dog_cy_px"]].dropna().to_numpy()
    hum = fm[["toe_x_px", "toe_y_px"]].dropna().to_numpy()
    if len(hum):
        ax.plot(hum[:, 0], hum[:, 1], color=HUMAN_COLOR, lw=0.6, alpha=0.8,
                zorder=2, label="Human (left toe)")
        ax.scatter(hum[0, 0], hum[0, 1], s=10, marker="o", color=HUMAN_COLOR,
                   zorder=4, edgecolor="white", linewidth=0.3)
    if len(dog):
        ax.plot(dog[:, 0], dog[:, 1], color=DOG_COLOR, lw=0.6, alpha=0.85,
                zorder=2, label="Dog (body centroid)")
        ax.scatter(dog[0, 0], dog[0, 1], s=10, marker="o", color=DOG_COLOR,
                   zorder=4, edgecolor="white", linewidth=0.3)
        ax.scatter(dog[-1, 0], dog[-1, 1], s=10, marker="x", color=DOG_COLOR,
                   zorder=4, linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(0, RGB_W)
    ax.set_ylim(RGB_H, 0)          # image y-axis points down, matching the video
    ax.set_xlabel("x (px, 1280\u00d7720)")
    ax.set_ylabel("y (px)")
    if title:
        ax.set_title(title, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.4,
              borderaxespad=0.2, markerscale=0.7)


def step6_trajectories(fm_all, recs, per_rec):
    print("=" * 78)
    print("Step 6a  Human/dog trajectory plots (720p pixel coordinates, "
          "per-recording fence)")
    print(f"  Input: {os.path.join(D_DATA, 'xy_frame_metrics.csv')} (in memory)")
    print(f"  Output directory: {D_TRAJ}/")
    n_no_fence = 0
    for (batch, rec), fm in fm_all.groupby(["batch", "rec"], sort=True):
        r = recs[recs.rec == rec].iloc[0]
        if r.excluded:
            continue
        fence_px = per_rec.get((batch, rec))
        if fence_px is None or len(fence_px) != 6:
            n_no_fence += 1
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        rec_short = rec.replace("录制_", "rec ")
        title = (f"{BREED_EN[r.breed]} dog {r.dog_id.split('_')[-1]} \u00b7 "
                 f"{COND_FULL[r.cond]}\n{batch} / {rec_short} ({r.duration_s:.0f} s)")
        plot_trajectory(ax, fm, fence_px, title)
        fig.savefig(os.path.join(D_TRAJ, f"traj_{batch}_{rec}.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
    n_traj = len(fm_all[["batch", "rec"]].drop_duplicates())
    print(f"  Output: {D_TRAJ}/traj_*.png ({n_traj} figures, of which "
          f"{n_no_fence} without fence annotation)")

    # ---- Example composite figure: breed x scenario, the median-speed
    #      recording of each group ----
    keep = recs[~recs.excluded]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.8))
    for i, breed in enumerate(["beagle", "pet"]):
        for j, c in enumerate(["01", "02", "03"]):
            sub = keep[(keep.breed == breed) & (keep.cond == c)]
            if not len(sub):
                continue
            med = sub.dog_mean_speed_mps.median()
            r = sub.iloc[(sub.dog_mean_speed_mps - med).abs().argsort().iloc[0]]
            fm = fm_all[fm_all.rec == r.rec]
            ax = axes[i, j]
            plot_trajectory(ax, fm, per_rec.get((r.batch, r.rec)))
            ax.set_title(f"{BREED_EN[breed]} \u00b7 {COND_FULL[c]}\n"
                         f"dog {r.dog_id.split('_')[-1]} ({r.duration_s:.0f} s)",
                         fontsize=9.5)
            if j > 0:
                ax.set_ylabel("")
            if i == 0:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Representative trajectories (median-speed recording per group)",
                 fontsize=11.5)
    save_fig(fig, os.path.join(D_FIG, "fig_traj_examples"))


def _video_path(batch, rec):
    base = os.path.join(VIDEO_ROOT, batch, rec)
    if os.path.isdir(os.path.join(base, "color")):
        vids = sorted(glob.glob(os.path.join(base, "color", "*.mp4")))
    else:
        vids = sorted(glob.glob(os.path.join(base, "*.mp4")))
    return vids[0] if vids else None


def step6_fence_check(per_rec):
    """Fence corner vs video frame verification: the first annotated recording
    of each batch, video frame + hexagon (original 480p click coordinates)."""
    print("=" * 78)
    print("Step 6b  Fence corner vs video frame verification "
          "(one example per batch)")
    fc = pd.read_csv(FENCE_CSV, dtype={"batch": str, "rec": str})
    if not len(fc):
        print("  [Skip] no fence annotations")
        return
    import cv2
    picks = fc.groupby("batch", sort=True)["rec"].first().reset_index()
    picks = [{"batch": b, "rec": r} for b, r in picks.itertuples(index=False)]
    picks = [p for p in picks if _video_path(p["batch"], p["rec"])]
    if not picks:
        print("  [Skip] no videos found")
        return
    fig, axes = plt.subplots(1, len(picks),
                             figsize=(3.3 * len(picks), 2.3))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, picks):
        fsub = fc[(fc.batch == p["batch"]) & (fc.rec == p["rec"])].sort_values(
            "corner_id")
        cap = cv2.VideoCapture(_video_path(p["batch"], p["rec"]))
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
                          lw=1.4, label="Annotated hexagon fence")
        ax.add_patch(poly)
        ax.scatter(pts[:, 0], pts[:, 1], s=22, c="#EE6677", edgecolor="white",
                   linewidth=0.5, zorder=5)
        rec_short = p["rec"].replace("录制_", "rec ")
        ax.set_title(f"{p['batch']} / {rec_short}", fontsize=9.5)
        ax.set_xlabel("video x (px, 854\u00d7480)", fontsize=9)
        ax.set_ylabel("video y (px)", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Fence corner annotation check (video frame, one example per batch)",
                 fontsize=11)
    save_fig(fig, os.path.join(D_FIG, "fig_fence_video_check"))


def violin_panel(ax, keep, tb_breed, m, conds):
    import seaborn as sns
    sub = keep[keep[m].notna()]
    if not len(sub):
        ax.axis("off")
        return
    order = [c for c in ["01", "02", "03"] if c in conds]
    sns.violinplot(data=sub, x="cond", y=m, hue="breed", order=order,
                   hue_order=["beagle", "pet"], split=True, inner="box", cut=0,
                   palette=BREED_COLOR, bw_method="scott",
                   linewidth=0.6, ax=ax)
    sns.stripplot(data=sub, x="cond", y=m, hue="breed", order=order,
                  hue_order=["beagle", "pet"], dodge=True, size=1.4, alpha=0.6,
                  palette={"beagle": "white", "pet": "white"},
                  edgecolor="0.3", linewidth=0.25, ax=ax, legend=False)
    ax.set_xlabel("")
    ax.set_ylabel(METRIC_NICE[m])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([COND_EN[c] for c in order])
    # Breed significance stars
    ymax = np.nanmax(sub[m].to_numpy(float))
    yspan = np.nanmax(sub[m].to_numpy(float)) - np.nanmin(sub[m].to_numpy(float))
    for k, c in enumerate(order):
        qrow = tb_breed[(tb_breed.cond == c) & (tb_breed.metric == m)]
        if len(qrow) and np.isfinite(qrow.q_bh.iloc[0]):
            ax.text(k, ymax + 0.03 * (yspan + 1e-9), stars_p(qrow.q_bh.iloc[0]),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(top=ymax + 0.18 * (yspan + 1e-9))
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h[:2], ["Beagle", "Pet dog"], loc="upper left",
                  fontsize=9, borderaxespad=0.2)


def step6_violins(recs, tb_breed):
    print("=" * 78)
    print("Step 6c  Metric distribution violin plots "
          "(scenario x breed + breed-test stars)")
    keep = recs[~recs.excluded]
    groups = [
        ("movement", ["dog_mean_speed_mps", "dog_median_speed_mps",
                      "dog_dist_rate_mpm",
                      "human_toe_mean_speed_mps", "human_toe_dist_rate_mpm"], (3, 2)),
        ("distance", ["d_head_toe_mean_m", "d_head_toe_median_m",
                      "d_head_toe_min_m", "d_head_toe_within1m_ratio"], (4, 1)),
        ("tail_head", ["tail_wag_amp_mean_mm", "tail_wag_freq_hz",
                       "tail_elev_mean_deg", "tail_up_ratio",
                       "head_pitch_mean_deg", "head_up_ratio", "head_down_ratio"], (4, 2)),
    ]
    for name, metrics, (nc, nr) in groups:
        fig, axes = plt.subplots(nr, nc, figsize=(2.8 * nc, 2.5 * nr))
        axes = np.atleast_1d(axes).ravel()
        for ax, m in zip(axes, metrics):
            conds = ["02", "03"] if m in HUMAN_ONLY else ["01", "02", "03"]
            violin_panel(ax, keep, tb_breed, m, conds)
        for ax in axes[len(metrics):]:
            ax.axis("off")
        fig.suptitle("Behavioral metrics by scenario and breed "
                     "(stars: Beagle vs Pet, BH-corrected)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_fig(fig, os.path.join(D_FIG, f"fig_violin_{name}"))


def step6_overall_hist(recs):
    print("=" * 78)
    print("Step 6d  Overall distributions (all valid recordings)")
    keep = recs[~recs.excluded]
    ncol = 5
    nrow = int(np.ceil(len(METRIC_COLS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, METRIC_COLS):
        v = _fin(keep[m])
        if len(v) == 0:
            ax.axis("off")
            continue
        ax.hist(v, bins=18, color="#4477AA", alpha=0.85, edgecolor="white",
                linewidth=0.3)
        ax.axvline(np.median(v), color="#EE6677", lw=1.0, ls="--")
        ax.set_title(METRIC_NICE[m], fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
        ax.text(0.97, 0.95, f"n={len(v)}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color="0.3")
        if m in HUMAN_ONLY:
            ax.text(0.97, 0.84, "S2/S3 only", transform=ax.transAxes, ha="right",
                    va="top", fontsize=7, color="0.4")
    for ax in axes[len(METRIC_COLS):]:
        ax.axis("off")
    fig.suptitle("Overall metric distributions (all valid recordings)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, os.path.join(D_FIG, "fig_dist_overall"))


def step6_sig_heatmap(ts, tb):
    """Beagle three-scenario significance heatmap: 8 core metrics x
    (S1vsS2, S1vsS3, S2vsS3).

    tb is unused (only beagle scenario effects are plotted; breed effects are
    not plotted), kept in the signature for call compatibility.
    """
    print("=" * 78)
    print("Step 6e  Beagle three-scenario significance heatmap "
          "(8 core metrics)")
    metrics = [
        ("dog_mean_speed_mps", "Dog mean speed (m/s)"),
        ("dog_dist_rate_mpm", "Dog distance rate (m/min)"),
        ("d_head_toe_mean_m", "Dog\u2013human mean distance (m)"),
        ("d_head_toe_min_m", "Dog\u2013human min distance (m)"),
        ("tail_wag_freq_hz", "Tail wag frequency (Hz)"),
        ("tail_wag_amp_mean_mm", "Tail wag amplitude (mm)"),
        ("follow_ratio", "Following ratio"),
        ("vel_align_mean", "Velocity alignment (cos)"),
    ]
    pairs = ["S1 vs S2", "S1 vs S3", "S2 vs S3"]
    col_labels = [f"Beagle\n{p}" for p in pairs]

    qmat = np.full((len(metrics), len(pairs)), np.nan)
    for i, (m, _) in enumerate(metrics):
        for j, p in enumerate(pairs):
            r = ts[(ts.scope == "beagle") & (ts.metric == m) & (ts.pair == p)]
            if len(r) and np.isfinite(r.p_holm.iloc[0]):
                qmat[i, j] = r.p_holm.iloc[0]

    ncol = qmat.shape[1]
    fig, ax = plt.subplots(figsize=(0.98 * ncol + 3.2, 0.42 * len(metrics) + 1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        neglog = -np.log10(qmat)
    neglog = np.clip(neglog, 0, 4)
    im = ax.imshow(neglog, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4)
    for i in range(qmat.shape[0]):
        for j in range(ncol):
            q = qmat[i, j]
            if not np.isfinite(q):
                ax.text(j, i, "\u2013", ha="center", va="center", fontsize=8.5,
                        color="0.55")
            elif q < 0.05:
                ax.text(j, i, stars_p(q), ha="center", va="center",
                        fontsize=11, color="black", fontweight="bold")
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([lab for _, lab in metrics], fontsize=10)
    ax.set_title("Beagle scenario effects (Friedman + Wilcoxon signed-rank, "
                 "Holm-corrected)\n"
                 "cell = corrected p; *, q<0.05; **, q<0.01; ***, q<0.001; "
                 "\u2013 = n/a (human metrics, no human in S1)",
                 fontsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="both", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("\u2212log$_{10}$(corrected p)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(D_FIG, "fig_sig_heatmap"))


# ===================== Main pipeline =====================
def main():
    t0 = time.time()
    for d in [D_DATA, D_TRAJ, D_TS, D_FIG, D_TAB]:
        os.makedirs(d, exist_ok=True)
    setup_journal_style()
    # Unified larger font sizes for analysis_xy figures (text 10/labels
    # 11.5/ticks 9.5), tighter canvas
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11.5, "axes.titlesize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
    })

    per_rec, per_scale, fence_qc = step0_fence()
    twin = load_time_windows()
    recs = step1to3_metrics(per_rec, per_scale, twin)
    step4_describe(recs)
    ts, tb = step5_tests(recs)
    step5b_scale_accuracy(recs, fence_qc)

    # Trajectory plots need frame-wise data: reload from the output CSV
    # (keeps the pipeline steps independent)
    print("=" * 78)
    print("Step 6  Loading frame-wise metrics for plotting")
    fm_csv = os.path.join(D_DATA, "xy_frame_metrics.csv")
    print(f"  Input: {fm_csv}")
    fm_all = pd.read_csv(fm_csv, dtype={"batch": str, "rec": str})
    step6_trajectories(fm_all, recs, per_rec)
    step6f_timeseries(fm_all)
    step6_fence_check(per_rec)
    step6_violins(recs, tb)
    step6_overall_hist(recs)
    step6_sig_heatmap(ts, tb)

    print("=" * 78    )
    print(f"Pipeline finished, total elapsed {time.time()-t0:.1f}s")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
