#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serve_fence_annotator.py — Hexagonal fence corner annotation tool (Flask).

Purpose: for each recording, click the six upper corners of the hexagonal
fence (clockwise) on a video frame. The tool stores the corners in the same
720p keypoint convention used downstream and writes them to
stage0_ground/fence_corners.csv for fence-based scale calibration
(six panels x 0.90 m) and trajectory plotting.

Coordinate conversion (automatic):
  video click (854x480) --x(1280/854, 720/480)--> 720p colour coordinates
  (1280x720, the keypoint inference resolution).

Real-time QC (image plane): the six side lengths in 720p pixels are
computed on the fly, together with their mean, coefficient of variation and
the circumradius, so the regularity of the hexagon can be checked while
annotating.

Output: stage0_ground/fence_corners.csv
  columns: batch, rec, frame, corner_id, color_x_720, color_y_720,
           video_x, video_y, depth_px_x, depth_px_y, depth_mm, saved_at
  (the depth_* columns are legacy fields kept for schema compatibility and
   are written as empty values by this tool)

Start:
  batch mode : python3 stage0_ground/serve_fence_annotator.py [--port 8130]
  single file: python3 stage0_ground/serve_fence_annotator.py \
                   --video /abs/path/camera_1_xxx.mp4 \
                   [--out /abs/path/fence_corners.csv] [--port 8130]

Single-file mode annotates one video given by absolute path and saves the six
corners to a fence_corners.csv next to the video (or to --out), in the format
accepted by repro/run_one.py --fence (corner_id, color_x_720, color_y_720, ...).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "stage0_data"))

from flask import Flask, request, Response  # noqa: E402
from stage0_run import CONFIG as S0CFG       # noqa: E402  (single source of intrinsics)

STAGE0_ALL = ROOT / "stage0_data" / "stage0_keypoints_xyz_all.csv"
VIDEO_ROOT = Path("/home/yy/data/1-Circular-Fence-Test/depth/videos")
CORNER_CSV = _HERE / "fence_corners.csv"

# single-file mode (set by --video): the one video to annotate + output CSV
SINGLE_VIDEO: Path | None = None
SINGLE_OUT: Path | None = None

RGB_W, RGB_H = S0CFG["rgb_w"], S0CFG["rgb_h"]  # 1280x720 keypoint convention

CORNER_COLS = ["batch", "rec", "frame", "corner_id", "color_x_720",
               "color_y_720", "video_x", "video_y", "depth_px_x",
               "depth_px_y", "depth_mm", "saved_at"]


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------
def _video_path(batch: str, rec: str) -> Path | None:
    if SINGLE_VIDEO is not None:
        return SINGLE_VIDEO
    base = VIDEO_ROOT / batch / rec
    if (base / "color").is_dir():
        mp4s = sorted((base / "color").glob("camera_1_*.mp4"))
    else:
        mp4s = sorted(base.glob("camera_1_*.mp4"))
    return mp4s[0] if mp4s else None


def _recordings():
    """[(batch, rec)] from the video directory (single video in file mode)."""
    out = []
    if SINGLE_VIDEO is not None:
        return [(SINGLE_VIDEO.parent.name, SINGLE_VIDEO.stem)]
    if not VIDEO_ROOT.is_dir():
        return out
    for bdir in sorted(VIDEO_ROOT.iterdir()):
        if not bdir.is_dir():
            continue
        for rdir in sorted(bdir.iterdir()):
            if not rdir.is_dir() or _video_path(bdir.name, rdir.name) is None:
                continue
            out.append((bdir.name, rdir.name))
    return out


def _load_corners() -> pd.DataFrame:
    if CORNER_CSV.exists():
        return pd.read_csv(CORNER_CSV, dtype={"batch": str, "rec": str})
    return pd.DataFrame(columns=CORNER_COLS)


_FRAME_JPG_CACHE = OrderedDict()  # (batch,rec,frame) -> jpeg bytes
_FRAME_JPG_MAX = 300
_CAP_CACHE = {}                  # (batch,rec) -> (VideoCapture, n_frames, w, h)
_DOG_DF = None                   # all dog keypoints (720p coordinates)


def _cap(batch: str, rec: str):
    key = (batch, rec)
    if key in _CAP_CACHE:
        return _CAP_CACHE[key]
    vp = _video_path(batch, rec)
    if vp is None:
        return None
    cap = cv2.VideoCapture(str(vp))
    info = (cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if len(_CAP_CACHE) > 12:
        for k, (c, *_r) in list(_CAP_CACHE.items())[:4]:
            c.release()
            _CAP_CACHE.pop(k, None)
    _CAP_CACHE[key] = info
    return info


def _warm_dog_cache():
    """Background preload of dog keypoints (720p coordinates) for overlay."""
    global _DOG_DF
    if not STAGE0_ALL.exists():
        return
    df = pd.read_csv(STAGE0_ALL, dtype={"batch": str, "rec": str, "subject": str},
                     usecols=["batch", "rec", "subject", "frame", "body_part",
                              "color_x", "color_y", "confidence"])
    _DOG_DF = df[df["subject"] == "dog"].reset_index(drop=True)


def _dog_kp(batch: str, rec: str, frame: int):
    if _DOG_DF is None:
        _warm_dog_cache()
    if _DOG_DF is None:
        return []
    m = _DOG_DF
    m = m[(m["batch"] == batch) & (m["rec"] == rec) & (m["frame"] == frame)]
    return [{"name": r.body_part, "x": float(r.color_x), "y": float(r.color_y),
             "conf": float(r.confidence)} for r in m.itertuples()]


def _jresp(obj):
    return Response(json.dumps(obj, ensure_ascii=False), mimetype="application/json")


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE_HTML

    @app.route("/api/recordings")
    def api_recordings():
        corners = _load_corners()
        done = corners.groupby(["batch", "rec"]).size().to_dict() if len(corners) else {}
        recs = [{"batch": b, "rec": r, "done": int(done.get((b, r), 0))}
                for b, r in _recordings()]
        return _jresp({"recs": recs, "rgb": [RGB_W, RGB_H]})

    @app.route("/api/frame")
    def api_frame():
        batch = request.args.get("batch", "")
        rec = request.args.get("rec", "")
        frame = int(request.args.get("frame", 0))
        key = (batch, rec, frame)
        if key not in _FRAME_JPG_CACHE:
            info = _cap(batch, rec)
            if info is None:
                return _jresp({"error": "no video"}), 404
            cap, n_frames, w, h = info
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                return _jresp({"error": "read fail"}), 404
            _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            _FRAME_JPG_CACHE[key] = jpg.tobytes()
            while len(_FRAME_JPG_CACHE) > _FRAME_JPG_MAX:
                _FRAME_JPG_CACHE.popitem(last=False)
        return Response(_FRAME_JPG_CACHE[key], mimetype="image/jpeg")

    @app.route("/api/meta")
    def api_meta():
        batch = request.args.get("batch", "")
        rec = request.args.get("rec", "")
        info = _cap(batch, rec)
        if info is None:
            return _jresp({"error": "no video"}), 404
        _, n_frames, w, h = info
        return _jresp({"n_frames": n_frames, "w": w, "h": h})

    @app.route("/api/dogkp")
    def api_dogkp():
        batch = request.args.get("batch", "")
        rec = request.args.get("rec", "")
        frame = int(request.args.get("frame", 0))
        return _jresp({"pts": _dog_kp(batch, rec, frame)})

    @app.route("/api/annotations")
    def api_annotations():
        batch = request.args.get("batch", "")
        rec = request.args.get("rec", "")
        if SINGLE_OUT is not None:
            if SINGLE_OUT.exists():
                df = pd.read_csv(SINGLE_OUT)
                return _jresp({"rows": df.sort_values("corner_id").to_dict("records")})
            return _jresp({"rows": []})
        corners = _load_corners()
        if not len(corners):
            return _jresp({"rows": []})
        m = corners[(corners["batch"] == batch) & (corners["rec"] == rec)]
        rows = m.sort_values("corner_id").to_dict("records")
        return _jresp({"rows": rows})

    @app.route("/api/save", methods=["POST"])
    def api_save():
        d = request.get_json(force=True)
        batch, rec, frame = d["batch"], d["rec"], int(d["frame"])
        pts = d.get("points", [])
        if len(pts) == 0:
            return _jresp({"error": "no points to save"})
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if SINGLE_OUT is not None:
            # server-side conversion fallback: if the browser could not
            # precompute the 720p coordinates, convert video clicks using the
            # actual capture resolution read by OpenCV
            cap_info = _cap(batch, rec)
            vw, vh = (cap_info[2], cap_info[3]) if cap_info else (None, None)
            def _fin(v):
                try:
                    return np.isfinite(float(v))
                except (TypeError, ValueError):
                    return False
            rows = []
            for i, p in enumerate(pts, 1):
                x720, y720 = p.get("x720"), p.get("y720")
                if not (_fin(x720) and _fin(y720)):
                    if not vw or not vh:
                        return _jresp({"error":
                                       "720p conversion failed (no video "
                                       "resolution); reload the page and "
                                       "re-annotate"})
                    x720 = float(p["video_x"]) * RGB_W / vw
                    y720 = float(p["video_y"]) * RGB_H / vh
                rows.append({
                    "frame": frame, "corner_id": i,
                    "color_x_720": x720, "color_y_720": y720,
                    "video_x": p["video_x"], "video_y": p["video_y"],
                    "saved_at": now})
            pd.DataFrame(rows).to_csv(SINGLE_OUT, index=False)
            return _jresp({"saved": len(pts), "path": str(SINGLE_OUT)})
        corners = _load_corners()
        corners = corners[~((corners["batch"] == batch) & (corners["rec"] == rec))]
        for i, p in enumerate(pts, 1):
            corners.loc[len(corners)] = {
                "batch": batch, "rec": rec, "frame": frame, "corner_id": i,
                "color_x_720": p["x720"], "color_y_720": p["y720"],
                "video_x": p["video_x"], "video_y": p["video_y"],
                "depth_px_x": None, "depth_px_y": None,
                "depth_mm": None, "saved_at": now}
        corners = corners[CORNER_COLS]
        corners.to_csv(CORNER_CSV, index=False)
        return _jresp({"saved": len(pts), "path": str(CORNER_CSV)})

    return app


# ----------------------------------------------------------------------
# Frontend page
# ----------------------------------------------------------------------
PAGE_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hexagonal fence corner annotation</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f0f2f5;color:#222}
 .bar{padding:8px 12px;background:#fff;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:14px}
 #cv{display:block;margin:12px auto;max-width:92vw;background:#000;cursor:crosshair;border-radius:4px}
 #qc{max-width:900px;margin:0 auto 30px;background:#fff;padding:12px;border-radius:6px;font-size:13px}
 table{border-collapse:collapse;margin-top:6px}
 td,th{border:1px solid #ddd;padding:3px 10px;text-align:right}
 th:first-child,td:first-child{text-align:center}
 button{padding:4px 12px;cursor:pointer}
 input,select{font-size:13px}
 .dim{color:#888}
 .tag{background:#e8f0fe;padding:1px 8px;border-radius:10px}
 option.full{background-color:#c8e6c9} option.part{background-color:#ffe0b2}
</style></head><body>
<div class="bar"><b>Hexagonal fence upper-corner annotation</b>
 <span class="dim">Click the six corners of the fence <b>upper edge</b> in one direction (clockwise). The fence is static, any frame can be annotated.</span></div>
<div class="bar">Recording <input id="recFilter" placeholder="filter..." size="10">
 <select id="recSel" size="1"></select>
 <span id="progress" class="tag"></span>
 <span class="dim" id="infoFlag"></span></div>
<div class="bar">Frame <input type="number" id="frameInp" style="width:80px">
 <input type="range" id="frameSlider" style="width:260px;vertical-align:middle">
 <span class="dim" id="frameInfo"></span>
 <button onclick="loadFrame()">Load frame</button>
 <button onclick="undoPt()">Undo</button>
 <button onclick="clearPts()">Clear</button>
 <label><input type="checkbox" id="dogOv"> Dog keypoints</label>
 <button onclick="savePts()"><b>Save</b></button>
 <span id="status" class="tag"></span></div>
<canvas id="cv"></canvas>
<div id="qc"><i>Click on the frame to add corners. Coordinates are converted automatically from the 480p video to the 720p keypoint convention. Side lengths below are in 720p pixels for hexagon regularity checks.</i><div id="qcBody"></div></div>
<script>
const S = {batch:null, rec:null, nFrames:0, vw:0, vh:0, frame:600,
           img:null, points:[], dog:null};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');

function jget(u){return fetch(u).then(r=>r.json())}
function jpost(u,d){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(r=>r.json())}

function fmt(v,n=1){return v==null?'\u2014':(+v).toFixed(n)}

async function init(){
  const R = await jget('/api/recordings');
  S.recs = R.recs; S.rgb = R.rgb;
  fillRecSel('');
  // auto-select and load the first recording (single-video mode has exactly
  // one) so the canvas is sized to the video BEFORE any frame is shown or
  // clicked — otherwise the canvas stays at the browser default 300x150 and
  // click coordinates are recorded in the wrong pixel space
  if(R.recs.length){
    const r0 = R.recs[0];
    const sel = document.getElementById('recSel');
    sel.value = JSON.stringify([r0.batch, r0.rec]);
    loadRec(r0.batch, r0.rec);
  }
  const done = R.recs.filter(r=>r.done>0).length;
  document.getElementById('progress').textContent = `Annotated ${done}/${R.recs.length} recordings`;
  setInterval(async()=>{const R2=await jget('/api/recordings');
    const d2=R2.recs.filter(r=>r.done>0).length;
    document.getElementById('progress').textContent=`Annotated ${d2}/${R2.recs.length} recordings`;}, 30000);
}
function fillRecSel(filter){
  const sel = document.getElementById('recSel');
  const f = filter.toLowerCase();
  sel.innerHTML = '';
  for(const r of S.recs){
    if(f && !(r.batch+' '+r.rec).toLowerCase().includes(f)) continue;
    const o = document.createElement('option');
    o.value = JSON.stringify([r.batch,r.rec]);
    o.className = r.done>=6 ? 'full' : (r.done>0 ? 'part' : '');
    o.textContent = `${r.batch}/${r.rec}` + (r.done>0?`  \u2713${r.done}`:'');
    sel.appendChild(o);
  }
}
document.getElementById('recFilter').addEventListener('input',e=>fillRecSel(e.target.value));
document.getElementById('recSel').addEventListener('change',e=>{
  const [b,r] = JSON.parse(e.target.value); loadRec(b,r);
});
document.getElementById('frameSlider').addEventListener('change',()=>{document.getElementById('frameInp').value=document.getElementById('frameSlider').value});
document.getElementById('frameInp').addEventListener('change',loadFrame);

async function loadRec(b,r){
  S.batch=b; S.rec=r; S.points=[]; S.dog=null;
  const M = await jget(`/api/meta?batch=${encodeURIComponent(b)}&rec=${encodeURIComponent(r)}`);
  S.nFrames = M.n_frames; S.vw = M.w; S.vh = M.h;
  cv.width = M.w; cv.height = M.h;
  document.getElementById('frameSlider').max = M.n_frames-1;
  document.getElementById('frameSlider').value = Math.min(600,M.n_frames-1);
  document.getElementById('frameInp').value = Math.min(600,M.n_frames-1);
  document.getElementById('infoFlag').textContent = `Video ${M.w}x${M.h}, keypoint convention ${S.rgb[0]}x${S.rgb[1]}`;
  const A = await jget(`/api/annotations?batch=${encodeURIComponent(b)}&rec=${encodeURIComponent(r)}`);
  if(A.rows && A.rows.length){
    S.frame = A.rows[0].frame;
    document.getElementById('frameInp').value = S.frame;
    document.getElementById('frameSlider').value = S.frame;
    S.points = A.rows.map(x=>({x720:x.color_x_720, y720:x.color_y_720,
                               video_x:x.video_x, video_y:x.video_y}));
    document.getElementById('status').textContent = `Loaded ${A.rows.length} saved corners`;
  } else {
    S.frame = Math.min(600,M.n_frames-1);
    document.getElementById('status').textContent = 'New recording';
  }
  await loadFrame();
}
async function loadFrame(){
  if(S.batch===undefined){   // no recording loaded yet (canvas still 300x150)
    const sel = document.getElementById('recSel');
    if(sel.value){ const [b,r]=JSON.parse(sel.value); await loadRec(b,r); return; }
    document.getElementById('status').textContent='Select a recording first';
    return;
  }
  S.frame = Math.max(0, Math.min(S.nFrames-1, +document.getElementById('frameInp').value||0));
  document.getElementById('frameInp').value = S.frame;
  const url = `/api/frame?batch=${encodeURIComponent(S.batch)}&rec=${encodeURIComponent(S.rec)}&frame=${S.frame}`;
  const img = new Image();
  await new Promise(res=>{img.onload=res; img.src=url;});
  S.img = img;
  document.getElementById('frameInfo').textContent = `Frame ${S.frame}/${S.nFrames-1}`;
  if(document.getElementById('dogOv').checked){
    S.dog = await jget(`/api/dogkp?batch=${encodeURIComponent(S.batch)}&rec=${encodeURIComponent(S.rec)}&frame=${S.frame}`);
  } else S.dog = null;
  redraw();
}
cv.addEventListener('click', ev=>{
  if(S.frame===0 && !S.img) return;
  const rect = cv.getBoundingClientRect();
  const vx = (ev.clientX-rect.left) * cv.width/rect.width;
  const vy = (ev.clientY-rect.top) * cv.height/rect.height;
  if(S.points.length>=6){ document.getElementById('status').textContent='Already 6 corners, undo or clear first'; return; }
  S.points.push({x720: vx*S.rgb[0]/S.vw, y720: vy*S.rgb[1]/S.vh, video_x:vx, video_y:vy});
  renderQC(); redraw();
});
function undoPt(){ S.points.pop(); renderQC(); redraw(); }
function clearPts(){ S.points=[]; renderQC(); redraw(); }

function redraw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(S.img) ctx.drawImage(S.img,0,0,cv.width,cv.height);
  if(S.dog && S.dog.pts){
    for(const p of S.dog.pts){
      const x = p.x*S.vw/S.rgb[0], y = p.y*S.vh/S.rgb[1];
      ctx.fillStyle = p.conf>0.3?'rgba(0,255,255,.9)':'rgba(255,255,0,.6)';
      ctx.beginPath(); ctx.arc(x,y,4,0,7); ctx.fill();
    }
  }
  if(S.points.length){
    const px = S.points.map(p=>[p.video_x,p.video_y]);
    if(px.length>=2){
      ctx.strokeStyle='rgba(255,80,0,.85)'; ctx.lineWidth=2; ctx.setLineDash([7,5]);
      ctx.beginPath(); ctx.moveTo(px[0][0],px[0][1]);
      for(let i=1;i<px.length;i++) ctx.lineTo(px[i][0],px[i][1]);
      if(px.length>=3) ctx.closePath();
      ctx.stroke(); ctx.setLineDash([]);
    }
    S.points.forEach((p,i)=>{
      ctx.fillStyle='#0a0';
      ctx.beginPath(); ctx.arc(p.video_x,p.video_y,9,0,7); ctx.fill();
      ctx.fillStyle='#fff'; ctx.font='bold 13px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(String(i+1), p.video_x, p.video_y);
    });
  }
}
function renderQC(){
  const el = document.getElementById('qcBody');
  if(!S.points.length){ el.innerHTML=''; return; }
  let h = '<table><tr><th>#</th><th>720p x</th><th>720p y</th><th>video x</th><th>video y</th></tr>';
  S.points.forEach((p,i)=>{
    h += `<tr><td>${i+1}</td><td>${fmt(p.x720,0)}</td><td>${fmt(p.y720,0)}</td><td>${fmt(p.video_x,0)}</td><td>${fmt(p.video_y,0)}</td></tr>`;
  });
  h += '</table>';
  const n = S.points.length;
  if(n>=3){
    const P = S.points.map(p=>[p.x720,p.y720]);
    const sides = [];
    for(let i=0;i<n;i++){ sides.push(Math.hypot(P[(i+1)%n][0]-P[i][0], P[(i+1)%n][1]-P[i][1])); }
    const mean = sides.reduce((a,b)=>a+b,0)/n;
    const sd = Math.sqrt(sides.reduce((a,b)=>a+(b-mean)*(b-mean),0)/n);
    const c = P.reduce((a,p)=>[a[0]+p[0]/n, a[1]+p[1]/n], [0,0]);
    const cr = P.reduce((a,p)=>a+Math.hypot(p[0]-c[0], p[1]-c[1]),0)/n;
    h += `<p>Side length mean = <b>${fmt(mean,1)} px</b> | CV = ${fmt(sd/mean*100,1)}% | circumradius = ${fmt(cr,1)} px</p>` +
         `<p class="dim">Sides (px): ${sides.map(s=>fmt(s,0)).join(' / ')}</p>` +
         `<p class="dim">With the known panel width of 0.90 m, the scale is 0.90 m / side-mean, about ${fmt(0.90/(mean/1000),2)} mm/px for this recording.</p>`;
  }
  el.innerHTML = h;
}
async function savePts(){
  if(!S.points.length){ document.getElementById('status').textContent='No points to save'; return; }
  const pts = S.points.map(p=>({x720:p.x720, y720:p.y720, video_x:p.video_x, video_y:p.video_y}));
  const R = await jpost('/api/save', {batch:S.batch, rec:S.rec, frame:S.frame, points:pts});
  document.getElementById('status').textContent = R.saved?`\u2713 Saved ${S.batch}/${S.rec} ${R.saved} corners (${R.path})`:'Save failed';
  const R2 = await jget('/api/recordings');
  fillRecSel(document.getElementById('recFilter').value);
  const done = R2.recs.filter(r=>r.done>0).length;
  document.getElementById('progress').textContent = `Annotated ${done}/${R2.recs.length} recordings`;
}
document.getElementById('dogOv').addEventListener('change', loadFrame);
init();
</script></body></html>"""


def main():
    global SINGLE_VIDEO, SINGLE_OUT
    ap = argparse.ArgumentParser(
        description="Hexagonal fence corner annotation tool")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--video", default=None,
                    help="single-file mode: absolute path of the video to "
                         "annotate (bypasses the batch/video-root layout)")
    ap.add_argument("--out", default=None,
                    help="single-file mode: output fence_corners.csv "
                         "(default: fence_corners.csv next to the video; "
                         "format accepted by repro/run_one.py --fence)")
    args = ap.parse_args()
    if args.video:
        SINGLE_VIDEO = Path(args.video).expanduser().resolve()
        if not SINGLE_VIDEO.is_file():
            raise SystemExit(f"Video not found: {SINGLE_VIDEO}")
        SINGLE_OUT = (Path(args.out).expanduser().resolve() if args.out
                      else SINGLE_VIDEO.parent / "fence_corners.csv")
        print(f"[fence annotator] http://127.0.0.1:{args.port}  "
              f"(single video: {SINGLE_VIDEO} | output: {SINGLE_OUT})")
    else:
        threading.Thread(target=_warm_dog_cache, daemon=True).start()
        print(f"[fence annotator] http://127.0.0.1:{args.port}  "
              f"(videos: {VIDEO_ROOT} | output: {CORNER_CSV})")
    app = create_app()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
