#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py — One-command reproduction of the circular-fence test per-recording
metrics and figures from the released keypoint data.

Computes, for every recording (or a selected one), the paper's frame-level
behavioural metrics and per-recording summary:
  dog/human speed, cumulative distance, dog head-human toe distance,
  tail wag amplitude/frequency, head pitch, tail elevation, head orientation
  to human (looking), following behaviour, dog-human velocity alignment
  (fence-scale primary convention + depth back-projection reference).

Figures per recording (paper Fig. 2 style):
  timeseries   8-panel unified-time-axis overview
  trajectory   dog/human trajectory + that recording's hexagonal fence

Usage:
    python3 run.py                              # all recordings
    python3 run.py --rec 录制_08_201802_03       # single recording
    python3 run.py --rec 08__                   # substring filter
    python3 run.py --jobs 4                      # parallel workers

Inputs (data/):
  recordings.csv                       index (batch, rec, cond, breed, ...)
  recordings/{dir}/dog_keypoints.csv, human_keypoints.csv, fence_corners.csv
                                       (dir names carry no year-month, e.g. 08__录制_08_201802_03)
Outputs (results/):
  xy_frame_metrics.csv                frame-level metrics, all recordings
  xy_rec_metrics.csv                  per-recording summary metrics
  xy_qc_report.csv                    per-recording QC
  fence_scale_qc.csv                  per-recording fence scale calibration
  timeseries/{dir}.png                8-panel overview (one per recording)
  trajectories/{dir}.png              trajectory + fence hexagon
"""

import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

HERE = os.path.dirname(os.path.abspath(__file__))
D_DATA = os.path.join(HERE, "data")
D_RES = os.path.join(HERE, "results")

# ===================== Configuration =====================
FPS = 30.0
DT = 1.0 / FPS
RGB_W, RGB_H = 1280, 720

PART = {"occipital_protuberance": "H", "withers": "W",
        "tail_base": "B", "tail_tip": "T"}

# QC parameters
CONF_TAIL = 0.5          # tail tip visibility confidence threshold
JUMP2_M = 0.25           # secondary jump removal threshold (m/frame)
GAP_INTERP = 5           # short gap interpolation limit (frames)
SPEED_MED_WIN = 5        # speed sliding median window
DOG_VALID_MIN = 0.5      # minimum dog valid-frame ratio
TAIL_VIS_MIN = 0.2       # minimum visibility ratio for tail metrics

# Tail parameters
WAG_BASE_WIN = 31        # wag baseline window (frames, ~1 s), sliding median
PSD_BAND = (0.5, 8.0)    # wag frequency search band (Hz)
PSD_SEG_MIN = 60         # minimum usable continuous segment length for PSD

# Fence scale calibration: 6 panels, each physically 0.90 m wide
FENCE_SIDE_M = 0.90

# Head posture / interaction thresholds
HEAD_UP_DEG = 20.0
GAZE_DEG = 30.0            # head-to-human angle < 30 deg = looking at human
HUMAN_MOVE_MIN = 0.10      # human moving threshold (toe speed, m/s)
FOLLOW_MIN_VEL = 0.02      # dog velocity toward human for "following" (m/s)
HEAD_DOWN_DEG = -20.0

COND_FULL = {"S1": "S1 (no human)", "S2": "S2 (human, gaze away)",
             "S3": "S3 (human, gaze at dog)"}
BREED_EN = {"beagle": "Beagle", "pet": "Pet dog"}

DOG_COLOR = "#ff7f0e"
HUMAN_COLOR = "#1f77b4"

METRIC_COLS = [
    "dog_mean_speed_mps", "dog_median_speed_mps", "dog_cum_dist_m", "dog_dist_rate_mpm",
    "human_toe_mean_speed_mps", "human_toe_cum_dist_m", "human_toe_dist_rate_mpm",
    "d_head_toe_mean_m", "d_head_toe_median_m", "d_head_toe_min_m",
    "d_head_toe_within1m_ratio",
    "tail_wag_amp_mean_mm", "tail_wag_freq_hz", "tail_elev_mean_deg", "tail_up_ratio",
    "head_pitch_mean_deg", "head_up_ratio", "head_down_ratio",
    "follow_ratio", "vel_align_mean",
]
DEPTH_REF_COLS = [
    "dog_mean_speed_mps_depth", "dog_median_speed_mps_depth",
    "dog_cum_dist_m_depth", "dog_dist_rate_mpm_depth",
    "human_toe_mean_speed_mps_depth", "human_toe_cum_dist_m_depth",
    "human_toe_dist_rate_mpm_depth",
    "d_head_toe_mean_m_depth", "d_head_toe_min_m_depth",
    "tail_wag_amp_mean_mm_depth",
]

TS_FONTS = {"font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
            "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5}


def strip_ym(s):
    """Remove year+month from 8-digit dates in names (20260908 -> 08);
    normalizes user input so legacy full-date names still match."""
    return re.sub(r"\d{8}", lambda m: m.group(0)[6:], str(s))


def setup_journal_style():
    """Top-journal figure style (no seaborn dependency):
    white background, no top/right spines, outward thin ticks,
    Liberation Sans (~Arial) 8pt text / 9pt labels, 300 dpi."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.axisbelow": True,
    })


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


def inst_wag_freq(off, fps=FPS, pp_win=31, min_pp_mm=15.0, max_hz=8.0, smooth_win=15):
    """Frame-wise tail wag frequency (Hz), zero-crossing half-period method,
    1 s peak-to-peak >= 15 mm gating, 0.5 s sliding median smoothing."""
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


def wag_frequency(lat_interp):
    """Per-segment Welch PSD over visible segments (>= PSD_SEG_MIN frames),
    length-weighted average, dominant frequency in PSD_BAND."""
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


# ===================== Per-recording processing =====================
def load_recording(rec_dir):
    """dog + human long tables -> single long table with a subject column.
    A missing human_keypoints.csv is treated as 'no human recorded'."""
    dog = pd.read_csv(os.path.join(rec_dir, "dog_keypoints.csv"))
    dog["subject"] = "dog"
    hp = os.path.join(rec_dir, "human_keypoints.csv")
    if os.path.exists(hp):
        hum = pd.read_csv(hp)
    else:
        hum = pd.DataFrame(columns=dog.columns)
    hum["subject"] = "human"
    return pd.concat([dog, hum], ignore_index=True)


def load_fence(rec_dir):
    """6 fence corners (720p px) + fence scale (m/px) = 0.90 / mean side length."""
    fc = pd.read_csv(os.path.join(rec_dir, "fence_corners.csv")).sort_values("corner_id")
    pts = fc[["color_x_720", "color_y_720"]].to_numpy(float)
    if len(pts) == 6:
        sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        scale = FENCE_SIDE_M / float(sides.mean())
    else:
        scale = np.nan
    return pts, scale


def process_recording(g, twin=None, scale=np.nan):
    """Frame-wise metrics for one recording (same conventions as the paper).

    g            long table (dog + human) with columns subject/frame/body_part/
                 color_x/color_y/x_m/y_m/h_m/confidence/valid
    twin         (start frame, end frame) effective time window; None = all
    scale        fence scale calibration (m/px, primary convention)
    Returns (frame-metric DataFrame, QC dict).

    primary  = 720p pixel trajectory x fence scale -> speed/distance/amplitude
    reference = stage0 depth back-projection x_m/y_m (_depth suffix)
    """
    rec = g["rec"].iloc[0] if "rec" in g else ""
    batch = g["batch"].iloc[0] if "batch" in g else ""

    if twin is not None:
        g = g[(g["frame"] >= twin[0]) & (g["frame"] <= twin[1])]

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

    # ---- Dog centroid: primary (720p px x scale) + reference (depth) ----
    body_ok = (np.isfinite(pH) & np.isfinite(qH) & np.isfinite(pW) & np.isfinite(qW)
               & np.isfinite(pB) & np.isfinite(qB))
    cen_x = np.where(body_ok, (xH + xW + xB) / 3.0, np.nan)
    cen_y = np.where(body_ok, (yH + yW + yB) / 3.0, np.nan)
    cx_q, cy_q = jump_remove_interp(cen_x, cen_y)
    dog_speed_d = speed_from_xy(cx_q, cy_q)
    dog_cum_d = cum_from_xy(cx_q, cy_q)

    cen_px_x = np.where(body_ok, (pH + pW + pB) / 3.0, np.nan)
    cen_px_y = np.where(body_ok, (qH + qW + qB) / 3.0, np.nan)
    jump_px = JUMP2_M / scale if np.isfinite(scale) else np.inf
    cpx_q, cpy_q = jump_remove_interp(cen_px_x, cen_px_y, jump=jump_px)
    dog_speed = speed_from_xy(cpx_q, cpy_q) * scale
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

    # ---- Dog head - human left toe tip XY distance ----
    d_ht = np.hypot(pH - tpx_q, qH - tpy_q) * scale
    d_ht_d = np.hypot(xH - tx_q, yH - ty_q)

    # ---- Tail left-right wagging (XY plane) ----
    axp, ayp = pB - pW, qB - qW
    Lp = np.hypot(axp, ayp)
    axis_ok = Lp > 5.0
    upx = np.where(axis_ok, axp / np.maximum(Lp, 1e-9), np.nan)
    upy = np.where(axis_ok, ayp / np.maximum(Lp, 1e-9), np.nan)
    lat_px = upx * (qT - qB) - upy * (pT - pB)
    ax, ay = xB - xW, yB - yW
    L_axis = np.hypot(ax, ay)
    axis_ok_d = L_axis > 0.05
    ux = np.where(axis_ok_d, ax / np.maximum(L_axis, 1e-9), np.nan)
    uy = np.where(axis_ok_d, ay / np.maximum(L_axis, 1e-9), np.nan)
    lat_m_d = ux * (yT - yB) - uy * (xT - xB)
    tail_vis = (np.isfinite(pT) & np.isfinite(qT) & np.isfinite(pW) & np.isfinite(qW)
                & np.isfinite(pB) & np.isfinite(qB) & (confT >= CONF_TAIL) & validT
                & axis_ok & axis_ok_d)
    lat_mm = np.where(tail_vis, lat_px, np.nan) * scale * 1000.0
    lat_mm_d = np.where(tail_vis, lat_m_d, np.nan) * 1000.0

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

    # ---- Dog head facing the human ("looking at human") ----
    fxh, fyh = pH - pW, qH - qW
    vxh, vyh = tpx_q - pH, tpy_q - qH
    fn_ = np.hypot(fxh, fyh)
    vn_ = np.hypot(vxh, vyh)
    cosang = (fxh * vxh + fyh * vyh) / np.maximum(fn_ * vn_, 1e-9)
    gaze_ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    gaze_ok = np.isfinite(fn_) & (fn_ > 5.0) & np.isfinite(vn_) & (vn_ > 1.0)
    gaze_ang = np.where(gaze_ok, gaze_ang, np.nan)
    gaze_at_human = np.where(gaze_ok, (gaze_ang < GAZE_DEG).astype(float), np.nan)

    # ---- Following / velocity alignment ----
    dvx = (pd.Series(np.gradient(cpx_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    dvy = (pd.Series(np.gradient(cpy_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    hvx = (pd.Series(np.gradient(tpx_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    hvy = (pd.Series(np.gradient(tpy_q, DT) * scale)
           .rolling(SPEED_MED_WIN, center=True, min_periods=3).median().to_numpy())
    dv, hv = np.hypot(dvx, dvy), np.hypot(hvx, hvy)
    dx_ = (tpx_q - cpx_q) * scale
    dy_ = (tpy_q - cpy_q) * scale
    d_cen = np.hypot(dx_, dy_)
    ux_ = np.where(d_cen > 1e-6, dx_ / np.maximum(d_cen, 1e-9), np.nan)
    uy_ = np.where(d_cen > 1e-6, dy_ / np.maximum(d_cen, 1e-9), np.nan)
    approach_vel = dvx * ux_ + dvy * uy_
    ok_h = np.isfinite(hv)
    human_moving = np.where(ok_h, (hv >= HUMAN_MOVE_MIN).astype(float), np.nan)
    follow_state = np.where(
        ok_h & np.isfinite(approach_vel),
        ((hv >= HUMAN_MOVE_MIN) & (approach_vel >= FOLLOW_MIN_VEL)).astype(float),
        np.nan)
    vel_align = np.where((dv > 0.05) & (hv > 0.05),
                         (dvx * hvx + dvy * hvy) / np.maximum(dv * hv, 1e-9), np.nan)

    wag_freq_inst = inst_wag_freq(lat_mm)

    out = pd.DataFrame({
        "batch": batch, "rec": rec, "frame": frames,
        "dog_speed_mps": dog_speed, "dog_cum_m": dog_cum,
        "toe_speed_mps": toe_speed, "toe_cum_m": toe_cum,
        "d_head_toe_m": d_ht,
        "tail_lat_mm": lat_mm, "tail_lat_dev_mm": lat_dev,
        "dog_cx_m": cx_q, "dog_cy_m": cy_q,
        "dog_speed_mps_depth": dog_speed_d, "dog_cum_m_depth": dog_cum_d,
        "toe_x_m": tx_q, "toe_y_m": ty_q,
        "toe_speed_mps_depth": toe_speed_d, "toe_cum_m_depth": toe_cum_d,
        "d_head_toe_m_depth": d_ht_d,
        "tail_lat_mm_depth": lat_mm_d, "tail_lat_dev_mm_depth": lat_dev_d,
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
    """Per-recording summary metrics."""
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


# ===================== Figures =====================
def _no_human_note(ax):
    ax.text(0.5, 0.5, "no human recorded", ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="0.6")


def plot_timeseries(fm, meta, out_png):
    """8-panel unified-time-axis overview, paper Fig. 2 style.

    Panels: 1 dog/human speed | 2 cumulative distance | 3 head-toe distance
            | 4 tail wag frequency | 5 signed lateral tail swing
            | 6 head orientation to human (looking shaded)
            | 7 following (velocity toward human)
            | 8 dog-human movement-direction consistency
    """
    g = fm.sort_values("frame")
    t = (g["frame"].to_numpy(float) - float(g["frame"].iloc[0])) / FPS
    has_human = bool(g["d_head_toe_m"].notna().any())
    rec_short = meta["rec"].replace("录制", "rec")

    plt.rcParams.update(TS_FONTS)
    fig, axes = plt.subplots(8, 1, figsize=(6.8, 11.6), sharex=True)
    fig.subplots_adjust(hspace=0.75, top=0.95)

    ax = axes[0]
    ax.plot(t, g["dog_speed_mps"], lw=0.7, color=DOG_COLOR, label="dog")
    if has_human:
        ax.plot(t, g["toe_speed_mps"], lw=0.7, color=HUMAN_COLOR,
                label="human (left toe)")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Instantaneous speed", fontsize=8.5, loc="left")

    ax = axes[1]
    ax.plot(t, g["dog_cum_m"], lw=0.9, color=DOG_COLOR, label="dog")
    if has_human:
        ax.plot(t, g["toe_cum_m"], lw=0.9, color=HUMAN_COLOR,
                label="human (left toe)")
        ax.legend(fontsize=7, loc="upper left")
    ax.set_ylabel("Cum. dist (m)")
    ax.set_title("Cumulative distance", fontsize=8.5, loc="left")

    ax = axes[2]
    if has_human:
        ax.plot(t, g["d_head_toe_m"], lw=0.7, color="#AA3377")
        for y, ls in [(0.5, "--"), (1.0, "-."), (2.0, ":")]:
            ax.axhline(y, color="0.75", lw=0.6, ls=ls)
        ax.set_ylim(bottom=0)
    else:
        _no_human_note(ax)
    ax.set_ylabel("Dist (m)")
    ax.set_title("Dog head–human toe distance", fontsize=8.5, loc="left")

    ax = axes[3]
    ax.plot(t, g["tail_wag_freq_inst_hz"], lw=0.7, color="#228833")
    ax.set_ylim(0, 8)
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("Tail wag frequency (zero-crossing, pp ≥ 15 mm gate)",
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
    ax.text(0.99, 0.04, "right (−)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5, color="#CC6677")
    ax.set_ylabel("Offset (mm)")
    ax.set_title("Signed lateral tail swing (baseline-removed)",
                 fontsize=8.5, loc="left")

    ax = axes[5]
    if has_human:
        ax.plot(t, g["gaze_angle_deg"], lw=0.6, color="#4477AA")
        ax.axhline(GAZE_DEG, color="#EE6677", lw=0.8, ls="--",
                   label=f"looking at human (< {GAZE_DEG:.0f}°)")
        gz = g["gaze_at_human"].to_numpy(float)
        ax.fill_between(t, 0, 1, where=(gz == 1),
                        transform=ax.get_xaxis_transform(),
                        color="#4477AA", alpha=0.18, linewidth=0)
        ax.set_ylim(0, 180)
        ax.legend(fontsize=7, loc="upper right")
    else:
        _no_human_note(ax)
    ax.set_ylabel("Head angle (deg)")
    ax.set_title("Dog head orientation to human (shaded = looking)",
                 fontsize=8.5, loc="left")

    ax = axes[6]
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
        _no_human_note(ax)
    ax.set_ylabel("V toward human (m/s)")
    ax.set_title("Following: dog velocity component toward human",
                 fontsize=8.5, loc="left")

    ax = axes[7]
    if has_human:
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
        ax.text(0.99, 0.04, "opposite (−1)", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5, color="#882255")
    else:
        _no_human_note(ax)
    ax.set_ylabel("Alignment (cos)")
    ax.set_title("Dog–human movement direction consistency "
                 "(thin = per-frame; bold = 1 s rolling median)",
                 fontsize=8.5, loc="left")
    ax.set_xlabel("Time (s)")

    b = meta.get("breed") or ""
    breed_label = (BREED_EN.get(b, b) + " dog") if b else "Dog"
    sc = meta.get("scenario") or ""
    cond_full = COND_FULL.get(sc, sc) if sc else "condition unspecified"
    fig.suptitle(f"{breed_label} · {cond_full} — "
                 f"{meta['batch']} / {rec_short}", fontsize=9.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(fm, fence_px, meta, out_png):
    """Dog/human trajectory in 720p image pixels + that recording's fence."""
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    if fence_px is not None and len(fence_px) >= 3:
        poly = MplPolygon(fence_px, closed=True, facecolor="0.92",
                          edgecolor="0.35", lw=0.9, alpha=0.9, zorder=0)
        ax.add_patch(poly)
        ax.scatter(fence_px[:, 0], fence_px[:, 1], s=6, c="0.25", zorder=3)
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
    ax.set_ylim(RGB_H, 0)
    ax.set_xlabel("x (px, 1280×720)")
    ax.set_ylabel("y (px)")
    rec_short = meta["rec"].replace("录制_", "rec ")
    b = meta.get("breed") or ""
    breed_label = (BREED_EN.get(b, b) + " dog") if b else "Dog"
    sc = meta.get("scenario") or ""
    cond_full = COND_FULL.get(sc, sc) if sc else "condition unspecified"
    ax.set_title(f"{breed_label} "
                 f"{meta['dog_id'].split('_')[-1]} · {cond_full}\n"
                 f"{meta['batch']} / {rec_short} "
                 f"({fm['frame'].nunique() / FPS:.0f} s)", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.4,
              borderaxespad=0.2, markerscale=0.7)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================== Worker =====================
def process_one(row, data_dir, d_res_ts, d_res_traj):
    """Full per-recording pipeline: load -> metrics -> summary -> 2 figures."""
    rec_dir = os.path.join(data_dir, "recordings", row["dir"])
    g = load_recording(rec_dir)
    g["batch"], g["rec"] = row["batch"], row["rec"]
    fence_px, scale = load_fence(rec_dir)

    w = (None if (pd.isna(row["win_start"]) or pd.isna(row["win_end"]))
         else (int(row["win_start"]), int(row["win_end"])))
    fm, qc = process_recording(g, w, scale)
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

    meta = {"batch": row["batch"], "rec": row["rec"], "dog_id": row["dog_id"],
            "cond": row["cond"], "breed": row["breed"]}
    rec_row = summarize_recording(fm, qc, meta)

    plot_timeseries(fm, {"rec": row["rec"], "batch": row["batch"],
                         "breed": row["breed"], "scenario": row["scenario"],
                         "dog_id": row["dog_id"]},
                    os.path.join(d_res_ts, f"{row['dir']}.png"))
    plot_trajectory(fm, fence_px,
                    {"rec": row["rec"], "batch": row["batch"],
                     "breed": row["breed"], "scenario": row["scenario"],
                     "dog_id": row["dog_id"]},
                    os.path.join(d_res_traj, f"{row['dir']}.png"))
    return fm, rec_row, scale, fence_px


def _worker(args):
    row, data_dir, d_res_ts, d_res_traj = args
    setup_journal_style()
    fm, rec_row, scale, fence_px = process_one(row, data_dir, d_res_ts, d_res_traj)
    return fm, rec_row, scale, fence_px


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(
        description="One-command reproduction of per-recording metrics + figures")
    ap.add_argument("--data", default=D_DATA, help="data directory (default: ./data)")
    ap.add_argument("--out", default=D_RES, help="output directory (default: ./results)")
    ap.add_argument("--rec", default=None,
                    help="process only recordings whose name contains this substring")
    ap.add_argument("--jobs", type=int, default=4, help="parallel workers (default 4)")
    args = ap.parse_args()

    data_dir = args.data
    d_res = args.out
    d_res_ts = os.path.join(d_res, "timeseries")
    d_res_traj = os.path.join(d_res, "trajectories")
    for d in [d_res, d_res_ts, d_res_traj]:
        os.makedirs(d, exist_ok=True)

    setup_journal_style()

    idx_path = os.path.join(data_dir, "recordings.csv")
    print("=" * 78)
    print("One-command reproduction: circular-fence test per-recording analysis")
    print(f"  Input : {idx_path}")
    idx = pd.read_csv(idx_path, dtype={"batch": str, "rec": str, "cond": str,
                                       "dir": str, "dog_id": str})
    if args.rec:
        key = strip_ym(args.rec)
        m = idx.rec.str.contains(key, regex=False) | \
            idx.dir.str.contains(key, regex=False)
        idx = idx[m].reset_index(drop=True)
        if idx.empty:
            raise SystemExit(f"No recording matches: {args.rec}")
    print(f"  Recordings to process = {len(idx)}")

    t0 = time.time()
    rows = idx.to_dict("records")
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_worker,
                                  [(r, data_dir, d_res_ts, d_res_traj) for r in rows]))
    else:
        results = [_worker((r, data_dir, d_res_ts, d_res_traj)) for r in rows]

    fm_all = pd.concat([r[0] for r in results], ignore_index=True)
    recs = pd.DataFrame([r[1] for r in results])
    scale_rows = []
    for row, r in zip(rows, results):
        scale_rows.append({"batch": row["batch"], "rec": row["rec"],
                           "scale_m_per_px": r[2],
                           "scale_mm_per_px": r[2] * 1000.0 if np.isfinite(r[2]) else np.nan,
                           "n_corners": 0 if r[3] is None else len(r[3])})
    scale_qc = pd.DataFrame(scale_rows)

    cols = (["batch", "rec", "dog_id", "cond", "breed", "duration_s", "n_frames",
             "win_start", "win_end",
             "dog_valid_ratio", "tail_vis_ratio", "human_toe_valid_ratio",
             "dog_inside_fence_ratio", "excluded", "exclude_reason"]
            + METRIC_COLS + DEPTH_REF_COLS)
    recs = recs[cols]
    recs_out = recs.copy()
    recs_out["cond"] = recs_out["cond"].map({"01": "S1", "02": "S2", "03": "S3"})

    p_fm = os.path.join(d_res, "xy_frame_metrics.csv")
    p_rec = os.path.join(d_res, "xy_rec_metrics.csv")
    p_qc = os.path.join(d_res, "xy_qc_report.csv")
    p_sc = os.path.join(d_res, "fence_scale_qc.csv")
    fm_all.to_csv(p_fm, index=False)
    recs_out.to_csv(p_rec, index=False)
    recs_out[cols[:15]].to_csv(p_qc, index=False)
    scale_qc.to_csv(p_sc, index=False)

    n_ex = int(recs.excluded.sum())
    print(f"  Recordings = {len(recs)}, whole-recording excluded = {n_ex}")
    print(f"  Figures: {d_res_ts}/ (8-panel timeseries), "
          f"{d_res_traj}/ (trajectories), {len(recs)} each")
    print(f"  Output: {p_fm}")
    print(f"  Output: {p_rec}")
    print(f"  Output: {p_qc}")
    print(f"  Output: {p_sc}")
    print(f"  Elapsed {time.time()-t0:.1f}s")
    print(f"  Output directory: {d_res}")


if __name__ == "__main__":
    main()
