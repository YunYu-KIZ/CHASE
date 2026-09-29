#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段0 数据工程：地面重建 + 离地高度(倾斜校正) + QC + 3D重建 + 统一长表

只对 camera_1_20260908_150631 一个录制跑通流程（该录制有原始深度帧）。
其他录制无原始深度帧（只有 keypoints_depth.csv），后续真实分析时直接复用本脚本
产出的背景深度图 bg_depth.npy（相机固定，背景不变）。

离地高度计算：不用平面拟合（会被围栏/四周地面污染），而是对每个关键点，
在其投影像素邻域取背景(时间中位数)深度作为"正下方地面深度"，则
    h_mm = ground_depth - keypoint_depth_mm
该值沿光轴方向，俯视相机近似等于垂直离地高度；本方法天然完成倾斜校正、
不依赖内参、不假设地面为平面。

流程：
  1. 读取深度帧(frame_*.png, uint16, mm, 0=无效) 采样 -> 时间中位数 = 静态背景深度
  2. 保存 bg_depth.npy（后续录制复用）与 meta json
  3. 读取狗关键点 long CSV -> QC(低置信/无效/物理跳变剔除, 短缺失插值<=5帧, 轨迹平滑窗5)
  4. 3D重建：X=(color_x-cx)*Z/fx, Y=(color_y-cy)*Z/fy, Z=depth_mm
  5. 离地高度 h = 背景深度采样(投影邻域中位数) - depth_mm
  6. 输出统一长表 stage0_keypoints_xyz.csv 与诊断图

输出文件：
  stage0_data/ground/camera_1_20260908_150631_bg_depth.npy     (背景中位数深度, 可推广)
  stage0_data/ground/camera_1_20260908_150631_ground_meta.json (内参/方法/统计)
  stage0_data/ground/camera_1_20260908_150631_ground_diag.png  (诊断图)
  stage0_data/stage0_keypoints_xyz.csv                          (统一长表)
"""

import os
import json
import glob
import time

import numpy as np
import pandas as pd
import cv2

# ===================== 配置 =====================
CONFIG = {
    "batch": "20260908",
    "rec": "录制_20260908_201802_01",
    # 原始深度帧目录（仅此录制有）
    "depth_dir": "/home/yy/data/1-Circular-Fence-Test/depth/camera_1_20260908_150631",
    # 狗关键点长表（RGB坐标 + depth_mm）
    "keypoints_csv": "/home/yy/data/1-Circular-Fence-Test/depth/20260908/录制_20260908_201802_01/keypoints_depth.csv",
    # 输出目录
    "out_dir": "/home/yy/data/1-Circular-Fence-Test/pipeline/stage0_data",

    # RGB相机内参（d455 RGB, @1280x720；用于3D重建的X/Y，不影响离地高度h）
    "fx": 920.0, "fy": 920.0, "cx": 640.0, "cy": 360.0,
    "rgb_w": 1280, "rgb_h": 720,
    "depth_w": 848, "depth_h": 480,
    "fps": 30.0,

    # 背景深度帧采样步长（4167帧 -> ~139帧）
    "bg_stride": 30,
    # 地面采样邻域（深度图像素，取该邻域内有效值中位数作为地面深度）
    "ground_kernel": 7,

    # QC 参数
    "conf_thresh": 0.3,        # 关键点置信度阈值
    "interpolate_limit": 5,    # 短缺失插值上限帧数
    "smooth_window": 5,        # 轨迹平滑窗口
    "jump_thresh_m": 0.5,      # 相邻帧物理跳变阈值(米)
}


# ===================== 地面重建 =====================
def load_depth_frame(path):
    """读取单张深度帧，返回 uint16 (mm)，0=无效像素。"""
    d = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise IOError(f"无法读取深度帧: {path}")
    return d


def compute_background_depth(depth_dir, stride):
    """采样多帧深度图，逐像素取时间中位数 -> 静态背景深度(mm)。
    无效像素(0 或饱和 65535)置 NaN 后取中位数，避免污染背景。"""
    frame_paths = sorted(glob.glob(os.path.join(depth_dir, "frame_*.png")))
    sampled = frame_paths[::stride]
    print(f"  [地面] 读取 {len(sampled)}/{len(frame_paths)} 帧深度图做背景中位数...")
    stacks = []
    for p in sampled:
        d = load_depth_frame(p).astype(np.float32)
        d[(d == 0) | (d >= 65535)] = np.nan
        stacks.append(d)
    bg = np.nanmedian(np.stack(stacks, axis=0), axis=0)
    bg = np.nan_to_num(bg, nan=0.0)
    return bg, len(frame_paths)


def sample_ground_depth(bg_depth, depth_x, depth_y, kernel):
    """
    在背景深度图上，对每个投影点 (depth_y, depth_x) 取 kernel x kernel 邻域内
    有效像素(>0)的中位数作为地面深度(mm)。depth_x/depth_y 为深度图坐标(浮点)。
    """
    h, w = bg_depth.shape
    k = kernel
    pad = k // 2
    padded = np.pad(bg_depth, pad, mode="edge")

    # 浮点坐标 -> 取整偏移到 padded 空间
    xi = np.round(depth_x).astype(int) + pad  # 注意 bg_depth[y, x]
    yi = np.round(depth_y).astype(int) + pad

    out = np.full(len(depth_x), np.nan, dtype=np.float32)
    for i in range(len(depth_x)):
        y0, y1 = yi[i] - pad, yi[i] + pad + 1
        x0, x1 = xi[i] - pad, xi[i] + pad + 1
        block = padded[y0:y1, x0:x1]
        v = block[block > 0]
        if len(v):
            out[i] = np.median(v)
    return out  # mm


# ===================== 关键点 QC + 3D重建 =====================
def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def _process_series(vals, limit, smooth_win):
    """缺省插值(<=limit帧,只中段) + 居中移动平均平滑。"""
    s = vals.copy().astype(float)
    s = s.interpolate(method="linear", limit=limit, limit_area="inside")
    if smooth_win > 1 and len(s) >= 3:
        s = s.rolling(window=smooth_win, center=True, min_periods=1).mean()
    return s


def qc_and_reconstruct(df, cfg, bg_depth=None):
    """
    QC + 3D重建 + 离地高度（bg_depth 为 None 时跳过地面, h_m/ground_mm 记 NaN）。
    返回统一长表：frame, body_part, color_x, color_y, depth_mm, confidence, valid,
                  x_m, y_m, z_m, h_m, interpolated, jump_removed
    """
    fx, fy, cx, cy = cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"]
    conf_th = cfg["conf_thresh"]
    jump = cfg["jump_thresh_m"]
    limit = cfg["interpolate_limit"]
    win = cfg["smooth_window"]

    df = df.sort_values(["body_part", "frame"]).reset_index(drop=True)

    # 1) 原始值备份 + 低置信/无效标记
    df["color_x_q"] = df["color_x"].astype(float)
    df["color_y_q"] = df["color_y"].astype(float)
    df["depth_mm_q"] = df["depth_mm"].astype(float)
    df["low_conf"] = (df["confidence"] < conf_th) | (~df["valid"].astype(bool))
    df.loc[df["low_conf"], ["color_x_q", "color_y_q", "depth_mm_q"]] = np.nan

    # 2) 物理跳变检测（每个 body_part，3D 相邻位移 > 阈值则剔除后一帧）
    df["jump_removed"] = False
    for bp, g in df.groupby("body_part"):
        g = g.sort_values("frame")
        X = (g["color_x_q"] - cx) * g["depth_mm_q"] / fx
        Y = (g["color_y_q"] - cy) * g["depth_mm_q"] / fy
        Z = g["depth_mm_q"]
        disp = np.sqrt(X.diff() ** 2 + Y.diff() ** 2 + Z.diff() ** 2) / 1000.0
        bad_idx = g.index[(disp > jump)].tolist()
        df.loc[bad_idx, ["color_x_q", "color_y_q", "depth_mm_q"]] = np.nan
        df.loc[bad_idx, "jump_removed"] = True

    # 3) 短缺失插值 + 平滑（按 body_part 分维度）
    df["interpolated"] = False
    for bp, g in df.groupby("body_part"):
        g = g.sort_values("frame")
        was_nan = g[["color_x_q", "color_y_q", "depth_mm_q"]].isna().any(axis=1)
        for col in ["color_x_q", "color_y_q", "depth_mm_q"]:
            df.loc[g.index, col] = _process_series(g[col].reset_index(drop=True), limit, win).values
        filled = was_nan & ~(df.loc[g.index, ["color_x_q", "color_y_q", "depth_mm_q"]].isna().any(axis=1))
        df.loc[g.index[filled.values], "interpolated"] = True

    # 4) 3D 重建（米）
    Z = df["depth_mm_q"] / 1000.0
    X = (df["color_x_q"] - cx) * Z / fx
    Y = (df["color_y_q"] - cy) * Z / fy
    df["x_m"] = X
    df["y_m"] = Y
    df["z_m"] = Z

    # 5) 离地高度：背景深度局部采样 - keypoint深度（无背景深度时跳过, h_m 记为 NaN）
    if bg_depth is not None:
        depth_x = df["color_x_q"] * cfg["depth_w"] / cfg["rgb_w"]
        depth_y = df["color_y_q"] * cfg["depth_h"] / cfg["rgb_h"]
        ground_mm = sample_ground_depth(bg_depth, depth_x.values, depth_y.values, cfg["ground_kernel"])
        df["ground_mm"] = ground_mm
        df["h_m"] = (ground_mm - df["depth_mm_q"]) / 1000.0
    else:
        df["ground_mm"] = np.nan
        df["h_m"] = np.nan

    out = df[["frame", "body_part", "color_x_q", "color_y_q", "depth_mm_q",
              "confidence", "valid", "x_m", "y_m", "z_m", "h_m",
              "interpolated", "jump_removed"]].copy()
    out = out.rename(columns={
        "color_x_q": "color_x", "color_y_q": "color_y", "depth_mm_q": "depth_mm",
    })
    out = out.sort_values(["body_part", "frame"]).reset_index(drop=True)
    return out


# ===================== 主流程 =====================
def main():
    cfg = CONFIG
    t0 = time.time()
    out_dir = cfg["out_dir"]
    ground_dir = os.path.join(out_dir, "ground")
    os.makedirs(ground_dir, exist_ok=True)

    print("=" * 70)
    print("阶段0：地面重建 + QC + 3D重建  (单录制跑通)")
    print(f"  录制: {cfg['rec']}  batch={cfg['batch']}")
    print(f"  深度帧目录: {cfg['depth_dir']}")
    print(f"  关键点CSV: {cfg['keypoints_csv']}")
    print("=" * 70)

    # ---- 1. 背景深度 ----
    print("\n[1/4] 背景地面重建")
    bg_depth, n_frames = compute_background_depth(cfg["depth_dir"], cfg["bg_stride"])

    bg_path = os.path.join(ground_dir, "camera_1_20260908_150631_bg_depth.npy")
    np.save(bg_path, bg_depth.astype(np.float32))
    print(f"  -> 输出背景深度图: {bg_path}")

    meta = {
        "rec": cfg["rec"], "batch": cfg["batch"],
        "source_depth_dir": cfg["depth_dir"],
        "n_total_frames": n_frames, "bg_stride": cfg["bg_stride"],
        "ground_kernel": cfg["ground_kernel"],
        "intrinsics": {"fx": cfg["fx"], "fy": cfg["fy"], "cx": cfg["cx"], "cy": cfg["cy"],
                       "rgb_w": cfg["rgb_w"], "rgb_h": cfg["rgb_h"],
                       "depth_w": cfg["depth_w"], "depth_h": cfg["depth_h"]},
        "method": "background-median + local-neighborhood median sampling",
        "note": "h_m = ground_depth(projection_neighborhood_median) - keypoint_depth_mm; 单位 mm->m",
    }
    meta_path = os.path.join(ground_dir, "camera_1_20260908_150631_ground_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  -> 输出地面元信息: {meta_path}")

    # ---- 2. 关键点 QC + 3D重建 ----
    print("\n[2/4] 关键点 QC + 3D重建")
    kp = load_keypoints(cfg["keypoints_csv"])
    n_body = kp["body_part"].unique()
    n_frame = kp["frame"].max() + 1
    print(f"  [关键点] 原始 {len(kp)} 行, body_part={list(n_body)}, 帧数={n_frame}")

    out = qc_and_reconstruct(kp, cfg, bg_depth)
    out.insert(0, "batch", cfg["batch"])
    out.insert(1, "rec", cfg["rec"])

    n_low_conf = (kp["confidence"] < cfg["conf_thresh"]).sum()
    n_invalid = (~kp["valid"].astype(bool)).sum()
    print(f"  [QC] 低置信剔除: {n_low_conf} 行 (conf<{cfg['conf_thresh']}), invalid: {n_invalid} 行")
    print(f"  [QC] 物理跳变剔除: {out['jump_removed'].sum()} 行")
    print(f"  [QC] 插值填补: {out['interpolated'].sum()} 行")
    print(f"  [重建] x_m [{out['x_m'].min():.3f},{out['x_m'].max():.3f}], "
          f"y_m [{out['y_m'].min():.3f},{out['y_m'].max():.3f}], "
          f"z_m [{out['z_m'].min():.3f},{out['z_m'].max():.3f}]")
    print(f"  [离地高度] h_m [{out['h_m'].min():.3f},{out['h_m'].max():.3f}] m")

    # ---- 3. 输出统一长表 ----
    csv_path = os.path.join(out_dir, "stage0_keypoints_xyz.csv")
    out.to_csv(csv_path, index=False)
    print(f"\n[3/4] -> 输出统一长表: {csv_path}  ({len(out)} 行)")

    # ---- 4. 诊断图 + 汇总 ----
    _save_diag(ground_dir, bg_depth, out, cfg)
    print("\n[4/4] 完成")
    print("  h_m 按 body_part 分布(米):")
    for bp in n_body:
        sub = out[out["body_part"] == bp]["h_m"].dropna()
        print(f"    {bp:24s} min={sub.min():.3f} med={sub.median():.3f} max={sub.max():.3f} "
              f"负值占比={(sub < 0).mean()*100:.1f}%")
    print(f"\n总耗时 {time.time()-t0:.1f}s")


def _save_diag(ground_dir, bg_depth, out, cfg):
    """诊断图：背景深度热图 + h 分布 + h 时间序列(中位数)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) 背景深度热图
    im = axes[0].imshow(bg_depth, cmap="turbo", aspect="auto",
                        vmin=np.percentile(bg_depth[bg_depth > 0], 2),
                        vmax=np.percentile(bg_depth[bg_depth > 0], 98))
    axes[0].set_title("Background median depth (mm)", fontsize=12)
    fig.colorbar(im, ax=axes[0], label="depth mm")

    # 2) h 分布（分 body_part 直方图）
    colors = {"occipital_protuberance": "tab:orange", "withers": "tab:red",
              "tail_base": "tab:blue", "tail_tip": "tab:green"}
    for bp in ["occipital_protuberance", "withers", "tail_base", "tail_tip"]:
        sub = out[out["body_part"] == bp]["h_m"].dropna()
        axes[1].hist(sub, bins=60, alpha=0.5, label=bp.split("_")[0] if "_" in bp else bp,
                     color=colors[bp])
    axes[1].set_xlabel("height above ground h_m (m)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Height distribution by body_part")
    axes[1].legend(loc="upper right", fontsize=8)

    # 3) h 时间序列（每帧 4 点中位数）
    ts = out.groupby("frame")["h_m"].median()
    axes[2].plot(ts.index / cfg["fps"], ts.values, lw=0.6, color="steelblue")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("median height (m)")
    axes[2].set_title("Median height over time")

    plt.tight_layout()
    diag_path = os.path.join(ground_dir, "camera_1_20260908_150631_ground_diag.png")
    plt.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> 输出诊断图: {diag_path}")


if __name__ == "__main__":
    main()