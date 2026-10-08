#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_data.py — Export the release data package for one-command reproduction.

Provenance (pipeline-internal inputs; not needed by end users):
  pipeline/stage0_data/stage0_keypoints_xyz_all.csv   dog+human keypoints
                                                      (XY pixels + depth
                                                      back-projection, stage0 QC)
  pipeline/stage0_ground/fence_corners.csv            per-recording hexagonal
                                                      fence 6 corners (720p px)
  depth/圆形围栏统计开始结束表格.xlsx                    per-recording effective
                                                      time window (start/end frame)

Output (repro/data/):
  recordings.csv                          index: batch,rec,cond,scenario,breed,dog_id
  time_windows.csv                        rec,start,end (empty = full duration)
  recordings/{batch}__{rec}/
    dog_keypoints.csv                     frame,body_part,color_x,color_y,
                                          x_m,y_m,h_m,confidence,valid
    human_keypoints.csv                   same schema (header only if no human)
    fence_corners.csv                     corner_id,color_x_720,color_y_720

Usage:
    python3 export_data.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)

S0_ALL = os.path.join(PIPE, "stage0_data/stage0_keypoints_xyz_all.csv")
FENCE_CSV = os.path.join(PIPE, "stage0_ground/fence_corners.csv")
XLSX_TIME = os.path.join(os.path.dirname(PIPE), "depth",
                         "圆形围栏统计开始结束表格.xlsx")

D_OUT = os.path.join(HERE, "data")
D_REC = os.path.join(D_OUT, "recordings")

PET_BATCH = "20260910xjx"
COND_S = {"01": "S1", "02": "S2", "03": "S3"}
KEY_COLS = ["frame", "body_part", "color_x", "color_y",
            "x_m", "y_m", "h_m", "confidence", "valid"]


def strip_ym(s):
    """Remove year+month from 8-digit dates in names (20260908 -> 08),
    so directory/file names carry no year-month."""
    return re.sub(r"\d{8}", lambda m: m.group(0)[6:], s)


def main():
    os.makedirs(D_REC, exist_ok=True)

    print(f"Input : {S0_ALL}")
    df = pd.read_csv(S0_ALL, dtype={"batch": str, "rec": str, "subject": str})
    # Paper analysis set: 117 recordings from 39 beagle dogs
    # (pet-dog batch 20260910xjx excluded from the article)
    df = df[df.batch != PET_BATCH]
    print(f"  {len(df)} rows, {df[['batch','rec']].drop_duplicates().shape[0]} recordings "
          f"(beagle only, pet batch '{PET_BATCH}' excluded)")

    print(f"Input : {FENCE_CSV}")
    fc = pd.read_csv(FENCE_CSV, dtype={"batch": str, "rec": str})

    print(f"Input : {XLSX_TIME}")
    tw = pd.read_excel(XLSX_TIME, sheet_name=0)
    tw.columns = ["rec", "start", "end"]
    tw["rec"] = tw["rec"].astype(str)
    win = {str(r.rec): (None if (pd.isna(r.start) or pd.isna(r.end))
                        else (int(r.start), int(r.end)))
           for r in tw.itertuples(index=False)}

    idx_rows = []
    for (batch, rec), g in df.groupby(["batch", "rec"], sort=True):
        d = os.path.join(D_REC, strip_ym(f"{batch}__{rec}"))
        os.makedirs(d, exist_ok=True)
        cond = rec.split("_")[-1]
        gd = g[g.subject == "dog"][KEY_COLS].sort_values(["frame", "body_part"])
        gh = g[g.subject == "human"][KEY_COLS].sort_values(["frame", "body_part"])
        gd.to_csv(os.path.join(d, "dog_keypoints.csv"), index=False)
        gh.to_csv(os.path.join(d, "human_keypoints.csv"), index=False)
        fcsub = fc[(fc.batch == batch) & (fc.rec == rec)]
        fcsub[["corner_id", "color_x_720", "color_y_720"]].sort_values(
            "corner_id").to_csv(os.path.join(d, "fence_corners.csv"), index=False)
        w = win.get(rec)
        idx_rows.append({
            "batch": strip_ym(batch), "rec": strip_ym(rec),
            "dir": strip_ym(f"{batch}__{rec}"),
            "cond": cond, "scenario": COND_S.get(cond, cond),
            "breed": "pet" if batch == PET_BATCH else "beagle",
            "dog_id": strip_ym(f"{batch}_{rec.split('_')[-2]}"),
            "has_human": int(len(gh) > 0),
            "n_frames": int(gd.frame.nunique()),
            "win_start": np.nan if w is None else w[0],
            "win_end": np.nan if w is None else w[1],
        })

    idx = pd.DataFrame(idx_rows)
    p_idx = os.path.join(D_OUT, "recordings.csv")
    idx.to_csv(p_idx, index=False)
    p_tw = os.path.join(D_OUT, "time_windows.csv")
    idx[["rec", "win_start", "win_end"]].to_csv(p_tw, index=False)
    n_h = int(idx.has_human.sum())
    print(f"  Recordings = {len(idx)} (with human = {n_h}, S1 no-human = {len(idx)-n_h})")
    print(f"  Annotated time windows = {int(idx.win_start.notna().sum())}")
    print(f"Output: {p_idx}")
    print(f"Output: {p_tw}")
    print(f"Output: {D_REC}/{{batch}}__{{rec}}/dog_keypoints.csv, "
          f"human_keypoints.csv, fence_corners.csv")


if __name__ == "__main__":
    main()
