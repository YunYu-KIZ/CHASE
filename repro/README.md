# CHASE circular-fence test — one-command reproduction package

Reproduces, from the released keypoint data, every per-recording behavioural
metric and figure of the circular-fence (hexagonal enclosure) dog–human
experiment: **117 recordings from 39 beagle dogs** (39 × S1 no human /
S2 human gaze away / S3 human gaze at dog), matching the paper's analysis set
(pet-dog recordings excluded).

The package is self-contained: `run.py` and `run_one.py` depend only on
Python + numpy/pandas/scipy/matplotlib (no other repository code), and
`data/` carries all released inputs.

## Get the package

```bash
git clone https://github.com/YunYu-KIZ/CHASE.git
# then unpack the released data archive so that CHASE/repro/data/ exists
# (download data.tar.gz from the GitHub "Releases" page, ~165 MB)
tar -xzf data.tar.gz -C CHASE/repro/
```

Folder contents: `run.py`, `run_one.py`, `README.md` (this file),
`export_data.py` (internal provenance tool, see below) and `data/`
(distributed via the release archive).

## Requirements

- Python 3.9+
- `pip install numpy pandas scipy matplotlib`
- Fonts: figures request Liberation Sans and fall back to
  DejaVu Sans / Arial / Helvetica — no font installation is required.

---

## Usage — `run.py` (all recordings)

```bash
python3 run.py                    # all 117 recordings, default data/ -> results/
```

| Option | Default | Meaning |
|---|---|---|
| `--data DIR` | `./data` | data package root (must contain `recordings.csv` + `recordings/`) |
| `--out DIR` | `./results` | output root (created if missing; re-running overwrites) |
| `--rec NAME` | all | process only recordings whose `rec` or `dir` name contains NAME (substring); input names are year-month-normalised, so legacy full-date names also match |
| `--jobs N` | 4 | parallel workers |

Typical run time: ~1 min for all 117 recordings (4 workers).

## Usage — `run_one.py` (one experiment, standalone)

File mode — give the keypoint CSVs directly (any absolute paths):

```bash
python3 run_one.py --dog /path/dog.csv --fence /path/fence.csv \
    --human /path/human.csv --out /path/output_folder
python3 run_one.py --dog /path/dog.csv --fence /path/fence.csv --out OUT  # no human
```

Folder mode — a folder holding the package layout:

```bash
python3 run_one.py /path/to/experiment_folder --out /path/to/output_folder
python3 run_one.py data/recordings/08__录制_08_201802_03
```

| Argument | Required | Meaning |
|---|---|---|
| `--dog FILE` | file mode | dog keypoints CSV |
| `--fence FILE` | no | fence corners CSV (exactly 6 corners). Optional: without it the scale cannot be calibrated and all metre-based metrics become NaN; pixel trajectories and figures are still produced |
| `--human FILE` | no | human keypoints CSV; omitting it (or an empty file) = "no human present" |
| `--rec NAME` | no | recording name for outputs/metadata; default = dog-CSV file stem, or its parent folder name for generic names like `dog_keypoints.csv` |
| `data_folder` | folder mode | folder containing `dog_keypoints.csv` and `fence_corners.csv` (and optionally `human_keypoints.csv`) |
| `--out DIR` | no | output folder; default `results/single/<name>/` (created if missing) |
| `--win START END` | no | analysis time window in **frame numbers**, inclusive |

Behaviour details:

- **Required files** — `dog_keypoints.csv` must exist. `fence_corners.csv` /
  `--fence` is optional: a missing fence disables scale calibration (all
  metre-based metrics NaN, pixel trajectories and figures still produced);
  a fence file that is provided but does not contain exactly 6 corners is an
  error. `human_keypoints.csv` is optional: a missing or empty file
  is treated as "no human present" and all human-related metrics become NaN.
- **Metadata inference** — recording name = folder name; scenario is inferred
  when the folder name ends in `_01/_02/_03` (S1/S2/S3); otherwise it is
  reported as unspecified. If the folder happens to be one of the package's
  own recordings, its metadata and time window are taken from
  `data/recordings.csv` so the results match `run.py` exactly.
- **Time-window priority** — `--win` (manual) > package index > full
  recording. A full-recording analysis of a package recording therefore
  differs from `run.py` (which uses the annotated windows) — pass `--win` to
  replicate a window manually.
- **Output** — `metrics_frame.csv`, `metrics_rec.csv` (1-row summary),
  `timeseries.png` (8-panel, paper Fig. 2 style), `trajectory.png`, plus a
  console summary of the key metrics.

---

## Input data format

All CSVs are UTF-8, comma-separated. Pixel coordinates refer to the
1280×720 colour frames in standard image convention (**origin top-left,
y axis pointing down**).

### `data/recordings.csv` (index, only used by `run.py` and for package folders)

| Column | Type | Meaning |
|---|---|---|
| `batch`, `rec` | str | recording identifiers (names carry no year-month, e.g. batch `08`, rec `录制_08_201802_03`) |
| `dir` | str | folder name under `recordings/` = `{batch}__{rec}` |
| `cond` | str | `01`/`02`/`03` = S1/S2/S3 |
| `scenario` | str | `S1` no human · `S2` human, gaze away · `S3` human, gaze at dog |
| `breed` | str | `beagle` (analysis set) |
| `dog_id` | str | dog identity, shared across the 3 scenarios per dog |
| `win_start`, `win_end` | int/empty | annotated analysis window in frames; empty = full recording |
| `n_frames` | int | total frames of the source video |

### `recordings/{dir}/dog_keypoints.csv` (required, long format)

| Column | Type | Meaning |
|---|---|---|
| `frame` | int | frame number (video at 30 fps) |
| `body_part` | str | `occipital_protuberance`, `withers`, `tail_base`, `tail_tip` |
| `color_x`, `color_y` | float | keypoint position in 720p colour-frame pixels (primary coordinates for all XY metrics) |
| `x_m`, `y_m`, `h_m` | float/NaN | depth-camera back-projection in metres (reference only; feeds the `_depth` metric variants) |
| `confidence` | float | tracker confidence 0–1 |
| `valid` | bool | stage0 QC validity flag |

### `recordings/{dir}/human_keypoints.csv` (optional, same schema)

`body_part` ∈ `left_shoulder`, `right_shoulder`, `left_toe_tip`,
`right_toe_tip`. All human metrics use the **left toe tip**. Missing file,
header-only file, or all-NaN rows ⇒ "no human present" (S1-like analysis).

### `recordings/{dir}/fence_corners.csv` (required)

| Column | Type | Meaning |
|---|---|---|
| `corner_id` | int 1–6 | hexagonal fence corner |
| `color_x_720`, `color_y_720` | float | corner position in 720p colour-frame pixels |

The metre-per-pixel scale is calibrated per recording as
`0.90 m / mean hexagon side length (px)` — the released fence has 6 panels of
0.90 m each. **Using data recorded with a different fence requires changing
`FENCE_SIDE_M` in `run.py`.**

---

## Output data format

### Frame-level table (`results/xy_frame_metrics.csv`, one row per frame; `metrics_frame.csv` from `run_one.py`)

| Group | Columns |
|---|---|
| identity | `batch`, `rec`, `frame` |
| dog movement | `dog_speed_mps`, `dog_cum_m`, `dog_cx_m`, `dog_cy_m`, `dog_cx_px`, `dog_cy_px` |
| human movement | `toe_speed_mps`, `toe_cum_m`, `toe_x_m`, `toe_y_m`, `toe_x_px`, `toe_y_px` |
| dog–human interaction | `d_head_toe_m` (dog head–human left-toe distance), `gaze_angle_deg` (0–180°), `gaze_at_human` (1 = < 30°), `approach_vel_mps` (dog velocity component toward human), `human_moving` (toe speed ≥ 0.10 m/s), `follow_state` (human moving AND approach ≥ 0.02 m/s), `vel_align_cos` (movement-direction cosine, both ≥ 0.05 m/s) |
| tail / head | `tail_lat_mm`, `tail_lat_dev_mm` (baseline-removed lateral swing), `tail_visible`, `tail_elev_deg`, `head_pitch_deg`, `tail_wag_freq_inst_hz` (zero-crossing, pp ≥ 15 mm gate) |
| depth reference | `dog_speed_mps_depth`, `dog_cum_m_depth`, `toe_speed_mps_depth`, `toe_cum_m_depth`, `d_head_toe_m_depth`, `tail_lat_mm_depth`, `tail_lat_dev_mm_depth` |

NaN = not computable in that frame (e.g. no human, tail not visible).

### Recording-level table (`results/xy_rec_metrics.csv`, one row per recording; `metrics_rec.csv`)

- identity: `batch`, `rec`, `cond` (S1/S2/S3), `breed`, `dog_id`
- QC: `n_frames`, `duration_s`, `win_start`, `win_end`, `dog_valid_ratio`,
  `tail_vis_ratio`, `human_toe_valid_ratio`, `dog_inside_fence_ratio`
  (share of dog positions inside the annotated hexagon)
- primary metrics (20): dog mean/median speed, cumulative distance, distance
  rate; human-toe speed/cumulative distance/distance rate; head–toe distance
  mean/median/min and share < 1 m; tail wag amplitude/frequency, tail
  elevation mean, tail-up ratio; head pitch mean, head-up/head-down ratio;
  following ratio; dog–human velocity alignment (mean cos)
- depth-reference variants (10): same names with a `_depth` suffix

### Figures (per recording)

- `timeseries.png` — 8 panels (paper Fig. 2 style): dog/human speed;
  cumulative distance; head–toe distance with 0.5/1/2 m guides; tail wag
  frequency; lateral tail swing; head orientation to human (looking shaded);
  dog velocity toward human (human-moving bar + following shading);
  dog–human direction consistency (per-frame + 1 s rolling median).
  For no-human recordings the human-dependent panels show an explanatory note.
- `trajectory.png` — dog (and human, if present) trajectory in 720p pixels
  with that recording's fence hexagon.

Additional batch outputs (`run.py` only): `xy_qc_report.csv` (per-recording
QC + fence containment), `fence_scale_qc.csv` (per-recording m/px
calibration).

---

## Notes & caveats

1. **Coordinates & scale** — all primary metrics use 720p colour-pixel
   trajectories converted to metres with the per-recording fence scale
   (panel width 0.90 m). Depth back-projection variants are kept only as
   `_depth` reference columns.
2. **Frame rate is fixed at 30 fps** (`FPS` in `run.py`).
3. **Built-in QC / thresholds** (edit the constants at the top of `run.py`
   to change): secondary jump removal 0.25 m/frame; ≤ 5-frame gap
   interpolation; 5-frame sliding-median speed; gaze < 30° = looking at
   human; human moving ≥ 0.10 m/s; following ≥ 0.02 m/s toward human;
   head-up/down ± 20°; tail wag gated at peak-to-peak ≥ 15 mm.
4. **S1 / no-human recordings** — every human-dependent metric is NaN by
   design (not zero).
5. **`run_one.py` full-recording default** — when analysing a package
   recording through a non-indexed path without `--win`, the annotated
   window is not applied and the whole video is analysed.
6. **Re-running is idempotent** — output folders are created as needed and
   existing files are overwritten.
7. **Reproducibility** — reproduced metrics were verified to match the
   paper's analysis exactly (117 recordings × 38 summary columns;
   frame-level table identical to floating-point machine precision,
   max |diff| ≈ 1e-13).

## Methods summary (as implemented)

- **Scale calibration** — `scale_m_per_px = 0.90 / mean hexagon side length
  (px)` per recording.
- **QC** — stage0 pre-QC (conf < 0.3 removal, 0.5 m jump removal, ≤5-frame
  gap interpolation); this workflow adds secondary 0.25 m/frame jump removal,
  ≤5-frame gap interpolation, 5-frame sliding-median speed.
- **Metrics** (per recording) — dog mean/median speed, cumulative distance,
  distance rate; human left-toe speed/cumulative distance; dog head–human
  toe distance (mean/median/min, share < 1 m); tail wag amplitude (1 s
  baseline-removed deviation) and dominant frequency (Welch PSD 0.5–8 Hz);
  tail elevation & head pitch (from depth); head orientation to human
  (<30° = looking); following ratio; dog–human velocity direction alignment
  (cosine).

## Provenance

`export_data.py` regenerates `data/` from the internal pipeline
(stage0 keypoint table, fence corner annotations, annotated time windows).
It is an internal provenance tool and requires the non-released raw data —
**end users can ignore it**: reproduction needs only `data/` + `run.py` /
`run_one.py`.
