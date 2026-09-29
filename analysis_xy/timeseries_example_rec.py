#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""timeseries_example_rec.py — Generate the 8-panel XY timeseries example figure for a given recording.

Reuses the 8-panel plotting style of paper_support.py task6, but the recording
is given as a command-line argument (instead of auto-picking the S3 recording
with median follow_ratio).

Panels: 1 dog/human instantaneous speed | 2 cumulative distance | 3 head-to-toe
        distance | 4 tail wag frequency | 5 signed lateral tail swing offset
        | 6 head angle to human (looking shaded) | 7 dog velocity component
        toward human (following shaded) | 8 dog–human movement-direction consistency

Usage:
    python3 timeseries_example_rec.py 录制_20260908_201802_02
    python3 timeseries_example_rec.py                # defaults to the above
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from vis_common import setup_journal_style, DOG_COLOR, HUMAN_COLOR  # noqa: E402

D_DATA = os.path.join(HERE, "data")
D_FIG = os.path.join(HERE, "figures")
FPS = 30.0
GAZE_DEG = 30.0
DEFAULT_REC = "录制_20260908_201802_02"

setup_journal_style()
plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
})


def main():
    rec_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REC
    rm_path = os.path.join(D_DATA, "xy_rec_metrics.csv")
    fm_path = os.path.join(D_DATA, "xy_frame_metrics.csv")
    print(f"Input (recording-level): {rm_path}")
    print(f"Input (frame-level)  : {fm_path}")
    rm = pd.read_csv(rm_path, dtype={"batch": str, "rec": str})
    fm = pd.read_csv(fm_path, dtype={"batch": str, "rec": str})

    hit = rm[rm.rec == rec_name]
    if hit.empty:
        raise SystemExit(f"Recording not found: {rec_name} (examples: {sorted(rm.rec.unique())[:5]} ...)")
    r = hit.iloc[0]
    print(f"Selected: {r['batch']} / {r['rec']} ({r['cond']}, "
          f"follow_ratio={r['follow_ratio']:.2f}, duration {r['duration_s']:.0f}s)")

    g = fm[fm.rec == rec_name].sort_values("frame").copy()
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
    ax.plot(t, g["gaze_angle_deg"], lw=0.6, color="#4477AA")
    ax.axhline(GAZE_DEG, color="#EE6677", lw=0.8, ls="--",
               label=f"looking at human (< {GAZE_DEG:.0f}°)")
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
    ax.text(0.99, 0.04, "opposite (−1)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.5, color="#882255")
    ax.set_ylabel("Alignment (cos)")
    ax.set_title("Dog–human movement direction consistency "
                 "(thin = per-frame; bold = 1 s rolling median)",
                 fontsize=8.5, loc="left")
    ax.set_xlabel("Time (s)")

    cond_full = {"S1": "S1 no human", "S2": "S2 human, gaze away",
                 "S3": "S3 human, gaze at dog"}.get(r["cond"], r["cond"])
    fig.suptitle(f"Representative beagle recording ({r['batch']} / "
                 f"{rec_name.replace('录制', 'rec')}, {cond_full})",
                 fontsize=9.5)
    rec_slug = rec_name.replace("录制", "rec")   # ASCII-safe output filename
    out = os.path.join(D_FIG, f"fig_timeseries_example_{rec_slug}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    out_svg = os.path.join(D_FIG, f"fig_timeseries_example_{rec_slug}.svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Output: {out}")
    print(f"Output: {out_svg}")


if __name__ == "__main__":
    main()
