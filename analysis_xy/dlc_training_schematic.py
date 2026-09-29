#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dlc_training_schematic.py — Full DeepLabCut workflow schematic for the dog
(beagle4) and human (human4) models (vertical layout, 2:1).

Four-row structure (width:height = 2:1, figsize 14x7):
  Row 1: dog/human annotation data boxes (skeleton + number of keypoints +
         labelled frames/split)
  Row 2: DLC training schematic (augmentation -> ResNet-101 -> deconv ->
         heatmap/locref dual heads -> loss(pred, GT) -> AdamW, backprop back
         to the backbone)
  Row 3: DLC inference for dog/human separately (test accuracy)
  Row 4: dog-human keypoint merging -> joint XY behavioural analysis
         (real-time metric list)

Values match paper Table 1 / Section 2.4:
  dog beagle4: 4 keypoints (head/withers/tail base/tail tip), 1,620 frames,
               90/10, RMSE 1.47 px, mAP 98.6%
  human human4: 4 keypoints (L/R shoulder/L/R toe tip), 4,746 frames, 95/5,
               RMSE 3.44 px, mAP 94.1%

Output: figures/fig_dlc_training_schematic.png (300 dpi) + .pdf (vector)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "visualization"))
from vis_common import setup_journal_style, save_fig

setup_journal_style()
plt.rcParams.update({"font.size": 9.0})

D_FIG = os.path.join(HERE, "figures")

# ---- Colour scheme (colour-blind friendly) ----
DOG = "#ff7f0e"          # dog (orange)
HUM = "#1f77b4"          # human (blue)
BACKBONE = "#4a6fa5"     # backbone primary colour
BACKBONE_LIGHT = "#8aadd4"
DECONV = "#2d8659"
HEAD_HM = "#0072B2"      # heatmap head - blue
HEAD_OFF = "#D55E00"     # locref head - orange-red
LOSS = "#CC0000"
OPT = "#E69F00"
EDGE = "#2c2c2c"
FILL = "#f7f7f7"
PANEL = "#ffffff"
GT_C = "#666666"
PRED_C = "#CC0000"


# ---------------------------------------------------------------------------
# Basic drawing helpers
# ---------------------------------------------------------------------------
def rbox(ax, x, y, w, h, label, fc, ec=EDGE, lw=1.2, fs=8.5,
         bold=False, sub=None, sub_fs=6.2, tc="#111111", pad=0.45):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={pad},rounding_size=0.5",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    cx, cy = x + w / 2.0, y + h / 2.0
    if sub is not None:
        ax.text(cx, cy + h * 0.18, label, ha="center", va="center",
                fontsize=fs, color=tc, zorder=5,
                fontweight="bold" if bold else "normal")
        ax.text(cx, cy - h * 0.20, sub, ha="center", va="center",
                fontsize=sub_fs, color="#333333", zorder=5)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
                color=tc, zorder=5,
                fontweight="bold" if bold else "normal")
    return (cx, cy)


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.6, dashed=False,
          style="-|>", ms=13, rad=0.0, zorder=3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, color=color, lw=lw,
                        linestyle=(0, (4, 2)) if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}", zorder=zorder)
    ax.add_patch(a)


def dline(ax, x1, y1, x2, y2, color="#333333", lw=1.4, zorder=3):
    ax.plot([x1, x2], [y1, y2], ls=(0, (3, 2)), color=color, lw=lw, zorder=zorder)


def make_heatmap(ax, x, y, w, h, n_peaks=3):
    cmap = LinearSegmentedColormap.from_list(
        "hm", ["#ffffff", "#a8d8f0", "#0072B2", "#003366"])
    nx, ny = 100, 40
    XX, YY = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    Z = np.zeros_like(XX)
    centers = [(0.28, 0.45), (0.66, 0.60), (0.48, 0.25)]
    rng = np.random.RandomState(42)
    for i in range(n_peaks):
        cx, cy = centers[i % len(centers)]
        sigma = 0.09 + 0.02 * rng.rand()
        Z += np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma ** 2))
    Z /= Z.max()
    ax.imshow(Z, extent=[x, x + w, y, y + h], origin="lower", cmap=cmap,
              aspect="auto", zorder=1, interpolation="gaussian")


def draw_conv_block(ax, x, y, w, h, n=4):
    gap = 0.2
    bh = (h - gap * (n - 1)) / n
    for i in range(n):
        fc = BACKBONE if i % 2 == 0 else BACKBONE_LIGHT
        ax.add_patch(Rectangle((x, y + i * (bh + gap)), w, bh,
                               linewidth=0.8, edgecolor=EDGE,
                               facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, "ResNet-101\nstride 16",
            ha="center", va="center", fontsize=6.4, color="white",
            fontweight="bold", zorder=5)


def draw_deconv(ax, x, y, w, h, n=3):
    gap = 0.18
    bh = (h - gap * (n - 1)) / n
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * (bh + gap)), w, bh,
                               linewidth=0.7, edgecolor=EDGE,
                               facecolor=DECONV, alpha=0.88, zorder=2))
    ax.text(x + w / 2, y + h / 2, "deconv\n×2",
            ha="center", va="center", fontsize=6.0, color="white",
            fontweight="bold", zorder=5)


def draw_dog(ax, ox, oy, s=1.0, color=DOG, lw=2.0):
    pts = [(0.10, 0.60), (0.40, 0.50), (0.66, 0.40), (0.90, 0.28)]
    pts = [(ox + x * s, oy + y * s) for x, y in pts]
    ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts],
                       color=color, lw=lw, zorder=4, solid_capstyle="round"))
    for (px, py) in pts:
        ax.add_patch(Circle((px, py), 0.055 * s, facecolor=color,
                            edgecolor="white", lw=0.8, zorder=6))


def draw_human(ax, ox, oy, s=1.0, color=HUM, lw=2.0):
    sh = [(ox - 0.30 * s, oy + 0.30 * s), (ox + 0.30 * s, oy + 0.30 * s)]
    toe = [(ox - 0.20 * s, oy - 0.28 * s), (ox + 0.20 * s, oy - 0.28 * s)]
    ax.add_line(Line2D([sh[0][0], sh[1][0]], [sh[0][1], sh[1][1]],
                       color=color, lw=lw, zorder=4))
    ax.add_line(Line2D([toe[0][0], toe[1][0]], [toe[0][1], toe[1][1]],
                       color=color, lw=lw * 0.8, zorder=4))
    ax.add_line(Line2D([ox, ox], [sh[0][1] - 0.02, toe[0][1] + 0.02],
                       color=color, lw=lw * 0.6, ls=(0, (3, 3)), zorder=4))
    for p in sh + toe:
        ax.add_patch(Circle(p, 0.055 * s, facecolor=color, edgecolor="white",
                            lw=0.8, zorder=6))


def draw_loss_curve(ax, x, y, w, h):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.2",
                          linewidth=0.7, edgecolor=EDGE, facecolor="white", zorder=4)
    ax.add_patch(rect)
    xs = np.linspace(0, 1, 50)
    ys = np.exp(-xs * 5.0) * 0.92 + 0.05 + np.random.RandomState(7).randn(50) * 0.01
    ys = np.clip(ys, 0, 1)
    ax.plot(x + 0.06 + xs * (w - 0.12), y + 0.12 + ys * (h - 0.5),
            color=LOSS, lw=1.1, zorder=5)
    ax.text(x + w / 2, y + h - 0.14, "loss", fontsize=5.4,
            ha="center", va="top", color="#333", zorder=6)


def panel_title(ax, x, y, text, color="#111"):
    ax.text(x, y, text, fontsize=7.6, ha="left", va="center",
            fontweight="bold", color=color, zorder=5)


# ---------------------------------------------------------------------------
# Main figure: four-row layout (xlim 0-100, ylim 0-50, width:height = 2:1)
# ---------------------------------------------------------------------------
def main():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    ax.text(50, 48.8, "DeepLabCut pipeline \u2014 from manual annotation to joint "
                     "dog\u2013human analysis",
            fontsize=11.5, ha="center", va="center", fontweight="bold", color="#111")

    # ==================================================================
    # Row 1: dog / human annotation data boxes  (y 38.5-46.5)
    # ==================================================================
    y1, h1 = 38.5, 8.0
    # -- Left box: dog --
    rbox(ax, 4, y1, 42, h1, "", PANEL, lw=1.1)
    panel_title(ax, 5.2, y1 + h1 - 1.0, "Dog \u2014 beagle4 model", DOG)
    draw_dog(ax, 7.5, y1 + 1.2, s=6.5)
    ax.text(19.5, y1 + 5.6, "4 keypoints\nhead \u00b7 withers \u00b7 tail base \u00b7 tail tip",
            fontsize=6.8, ha="left", va="center", color="#222")
    ax.text(19.5, y1 + 1.8, "1,620 labelled frames \u00b7 90/10 split",
            fontsize=6.8, ha="left", va="center", color="#222")

    # -- Right box: human --
    rbox(ax, 54, y1, 42, h1, "", PANEL, lw=1.1)
    panel_title(ax, 55.2, y1 + h1 - 1.0, "Human \u2014 human4 model", HUM)
    draw_human(ax, 58.0, y1 + 1.4, s=6.2)
    ax.text(69.5, y1 + 5.6, "4 keypoints\nL/R shoulder \u00b7 L/R toe tip",
            fontsize=6.8, ha="left", va="center", color="#222")
    ax.text(69.5, y1 + 1.8, "4,746 labelled frames \u00b7 95/5 split",
            fontsize=6.8, ha="left", va="center", color="#222")

    # ---- Row 1 -> Row 2 arrows ----
    arrow(ax, 25, y1 - 0.1, 25, 36.0, lw=1.8)
    arrow(ax, 75, y1 - 0.1, 75, 36.0, lw=1.8)
    ax.text(50, 37.3, "labelled frames (GT)", fontsize=7.8, ha="center",
            va="center", color="#333", fontweight="bold")

    # ==================================================================
    # Row 2: DLC training schematic  (y 26-35.5)
    # ==================================================================
    y2, h2 = 26.0, 9.5
    rbox(ax, 4, y2, 92, h2, "", FILL, lw=1.1)
    panel_title(ax, 5.2, y2 + h2 - 0.9,
                "DLC training \u2014 shared architecture (PyTorch)")
    yc2 = 30.5

    # 2.1 Augmentation
    rbox(ax, 6.5, 27.8, 10.0, 5.4, "Augmentation", PANEL,
         sub="\u00b130\u00b0 \u00b7 scale 0.5\u20131.25\nnoise \u03c3=12.75 \u00b7 blur\n448\u00d7448",
         fs=7.0, sub_fs=5.6)
    # 2.2 backbone conv block
    draw_conv_block(ax, 19.5, 27.8, 9.5, 5.4, n=4)
    # 2.3 deconv
    draw_deconv(ax, 31.0, 27.8, 6.0, 5.4, n=3)
    # 2.4 heatmap head (glow background)
    make_heatmap(ax, 40.0, 30.7, 11.5, 2.5, n_peaks=3)
    ax.add_patch(Rectangle((40.0, 30.7), 11.5, 2.5, linewidth=1.0,
                          edgecolor=EDGE, facecolor="none", zorder=4))
    ax.text(45.75, 31.95, "Heatmap head (w = 1.0)", fontsize=6.0,
            ha="center", va="center", color="#0a2a4a", fontweight="bold", zorder=6)
    # 2.5 locref head
    rbox(ax, 40.0, 27.8, 11.5, 2.5, "LocRef head (Huber, w = 0.05)", HEAD_OFF,
         tc="white", bold=True, fs=6.0)
    # 2.6 loss
    rbox(ax, 54.0, 28.4, 11.0, 4.6, "Loss \u2112(pred, GT)", LOSS, tc="white",
         bold=True, sub="heatmap + Huber", fs=6.8, sub_fs=5.8)
    # 2.7 AdamW + loss curve
    rbox(ax, 68.0, 28.4, 12.5, 4.6, "AdamW", OPT, bold=True, fs=7.0)
    ax.text(71.5, 29.6, "lr 5\u00d710\u207b\u2074 \u00b7 batch 8", fontsize=5.8,
            ha="center", va="center", color="#222", zorder=5)
    draw_loss_curve(ax, 74.2, 28.9, 5.6, 3.6)

    # -- Forward arrows inside the box --
    arrow(ax, 16.6, yc2, 19.4, yc2, lw=1.3, ms=11)
    arrow(ax, 29.1, yc2, 31.0, yc2, lw=1.3, ms=11)
    arrow(ax, 37.1, 31.4, 39.9, 31.95, lw=1.3, ms=11, rad=-0.1)
    arrow(ax, 37.1, 29.2, 39.9, 29.05, lw=1.3, ms=11, rad=0.1)
    # heads → loss (pred)
    arrow(ax, 51.6, 31.95, 53.9, 31.2, lw=1.4, ms=11, color=PRED_C, rad=0.1)
    arrow(ax, 51.6, 29.05, 53.9, 29.9, lw=1.4, ms=11, color=PRED_C, rad=-0.1)
    ax.text(52.6, 32.7, "pred", fontsize=6.2, color=PRED_C, ha="left",
            va="center", fontweight="bold", zorder=6)
    # loss → AdamW
    arrow(ax, 65.1, 30.7, 67.9, 30.7, lw=1.4, color=LOSS, ms=11)
    # GT dashed line into loss
    dline(ax, 59.5, 35.4, 59.5, 33.1, color=GT_C, lw=1.3)
    arrow(ax, 59.5, 33.4, 59.5, 33.1, color=GT_C, lw=1.3, ms=10, dashed=True)
    ax.text(60.3, 34.2, "GT", fontsize=6.2, color=GT_C, ha="left",
            va="center", fontweight="bold", zorder=6)
    # backprop three-segment polyline: AdamW bottom -> box bottom horizontal
    # -> backbone bottom
    y_bp = 26.55
    dline(ax, 74.0, 28.3, 74.0, y_bp, color=PRED_C, lw=1.3)
    dline(ax, 74.0, y_bp, 24.2, y_bp, color=PRED_C, lw=1.3)
    arrow(ax, 24.2, y_bp, 24.2, 27.7, color=PRED_C, lw=1.3, dashed=True, ms=10)
    ax.text(49.0, 26.85, "backpropagation", fontsize=6.0, color=PRED_C,
            ha="center", va="bottom", fontweight="bold", zorder=6)

    # ---- Row 2 -> Row 3 arrows ----
    arrow(ax, 25, y2 - 0.1, 25, 23.4, lw=1.8)
    arrow(ax, 75, y2 - 0.1, 75, 23.4, lw=1.8)
    ax.text(50, 24.6, "trained models", fontsize=7.8, ha="center",
            va="center", color="#333", fontweight="bold")

    # ==================================================================
    # Row 3: DLC inference for dog / human  (y 13.5-23)
    # ==================================================================
    y3, h3 = 13.5, 9.0
    # -- Left box: dog inference --
    rbox(ax, 4, y3, 42, h3, "", PANEL, lw=1.1)
    panel_title(ax, 5.2, y3 + h3 - 1.0, "Dog inference \u2014 beagle4", DOG)
    # Video frame icon
    ax.add_patch(Rectangle((6.5, 14.6), 8.5, 5.6, linewidth=1.0,
                           edgecolor="#555", facecolor="#e9e9e9", zorder=3))
    draw_dog(ax, 7.5, 15.6, s=4.6)
    ax.text(10.75, 14.0, "video frames", fontsize=5.8, ha="center",
            va="top", color="#444")
    # DLC arrow
    arrow(ax, 15.2, 17.4, 19.2, 17.4, lw=1.5)
    ax.text(17.2, 18.3, "DLC", fontsize=6.6, ha="center", va="bottom",
            color="#333", fontweight="bold")
    # Output skeleton (prediction)
    draw_dog(ax, 21.5, 15.6, s=4.6, color=PRED_C)
    ax.text(27.5, 19.3, "keypoints + likelihood", fontsize=6.6, ha="left",
            va="center", color="#222")
    ax.text(27.5, 17.4, "(x, y, \u2113) per frame", fontsize=6.6, ha="left",
            va="center", color="#222")
    ax.text(27.5, 15.2, "test RMSE 1.47 px \u00b7 mAP 98.6%", fontsize=6.6,
            ha="left", va="center", color="#222", fontweight="bold")

    # -- Right box: human inference --
    rbox(ax, 54, y3, 42, h3, "", PANEL, lw=1.1)
    panel_title(ax, 55.2, y3 + h3 - 1.0, "Human inference \u2014 human4", HUM)
    ax.add_patch(Rectangle((56.5, 14.6), 8.5, 5.6, linewidth=1.0,
                           edgecolor="#555", facecolor="#e9e9e9", zorder=3))
    draw_human(ax, 60.5, 15.8, s=4.3)
    ax.text(60.75, 14.0, "video frames", fontsize=5.8, ha="center",
            va="top", color="#444")
    arrow(ax, 65.2, 17.4, 69.2, 17.4, lw=1.5)
    ax.text(67.2, 18.3, "DLC", fontsize=6.6, ha="center", va="bottom",
            color="#333", fontweight="bold")
    draw_human(ax, 72.0, 15.8, s=4.3, color=PRED_C)
    ax.text(78.0, 19.3, "keypoints + likelihood", fontsize=6.6, ha="left",
            va="center", color="#222")
    ax.text(78.0, 17.4, "(x, y, \u2113) per frame", fontsize=6.6, ha="left",
            va="center", color="#222")
    ax.text(78.0, 15.2, "test RMSE 3.44 px \u00b7 mAP 94.1%", fontsize=6.6,
            ha="left", va="center", color="#222", fontweight="bold")

    # ---- Row 3 -> Row 4 arrows ----
    arrow(ax, 25, y3 - 0.1, 25, 11.0, lw=1.8)
    arrow(ax, 75, y3 - 0.1, 75, 11.0, lw=1.8)
    ax.text(50, 12.2, "merged keypoints", fontsize=7.8, ha="center",
            va="center", color="#333", fontweight="bold")

    # ==================================================================
    # Row 4: dog-human keypoint merging -> joint XY behavioural analysis
    # (y 1.5-10.5)
    # ==================================================================
    y4, h4 = 1.5, 9.0
    rbox(ax, 4, y4, 92, h4, "", PANEL, lw=1.1)
    panel_title(ax, 5.2, y4 + h4 - 1.0,
                "Joint dog\u2013human behavioural analysis (XY, per frame)")
    # Left: merged scene (dog orange + human blue + distance line)
    draw_dog(ax, 7.0, 2.6, s=5.2)
    draw_human(ax, 17.5, 2.9, s=4.8)
    # Dog head <-> human toe tip distance dashed line
    ax.add_line(Line2D([11.3, 16.7], [5.7, 2.5], color="#777",
                       lw=1.1, ls=(0, (3, 2)), zorder=5))
    ax.text(15.4, 5.0, "d", fontsize=6.4, color="#444", ha="left",
            va="center", style="italic")
    ax.text(11.5, 1.55, "dog + human in shared frame", fontsize=5.8,
            ha="center", va="top", color="#444")
    # Middle arrow
    arrow(ax, 24.5, 5.6, 28.5, 5.6, lw=1.6)
    # Right: metric list in 3 columns
    cols = [
        ["Dog speed", "Distance rate", "Dog\u2013human distance", "Time within 1 m"],
        ["Wag frequency", "Wag amplitude", "Head angle to human",
         "Gaze-at-human ratio"],
        ["Approach velocity", "Following ratio", "Velocity alignment", ""],
    ]
    xs = [31.5, 52.0, 72.5]
    for ci, col in enumerate(cols):
        for ri, item in enumerate(col):
            if not item:
                continue
            ax.text(xs[ci], 7.3 - ri * 1.5, "\u2022 " + item, fontsize=6.6,
                    ha="left", va="center", color="#222")
    ax.text(88.5, 2.8, "per frame \u00b7\nper recording", fontsize=5.8,
            ha="right", va="bottom", color="#666", style="italic")

    save_fig(fig, os.path.join(D_FIG, "fig_dlc_training_schematic"))


if __name__ == "__main__":
    main()
