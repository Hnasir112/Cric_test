# Cric_test — Cricket Net-Session Analyzer

Computer-vision system that turns net-session footage into data: it tracks the
**batsman's body** (pose skeleton) and **objects** (ball, bat) frame by frame,
and exports both an annotated video and per-frame coordinates for analytics.

> **Status: v0.1 — working prototype.** Uses off-the-shelf YOLOv8 models so it
> runs today with no training. Cricket-specific models, multi-camera 3D, speed
> estimation, and shot classification are on the roadmap below.

## How it works

```
video clip ──▶ pose (YOLOv8-pose) ──┐
           └─▶ objects (YOLOv8) ─────┼─▶ annotated video  (data/output/*.mp4)
                                     └─▶ tracking data     (data/output/*.json)
```

- **Pose** — 17-point body skeleton of the batsman (largest person in frame).
- **Objects** — ball and bat, currently via COCO's `sports ball` / `baseball
  bat` classes as stand-ins until we train a cricket-specific detector.

## Setup (Windows, PowerShell)

```powershell
# from the project root
.\.venv\Scripts\Activate.ps1          # activate the virtual environment
pip install -r requirements.txt        # one-time install
```

## Run it

```powershell
# put a clip in data/raw/ first, then:
python scripts/track.py --input data/raw/net.mp4

# quick test on the first 100 frames:
python scripts/track.py --input data/raw/net.mp4 --max-frames 100

# use the GPU (needs a CUDA build of torch):
python scripts/track.py --input data/raw/net.mp4 --device cuda
```

Outputs land in `data/output/`:
- `net_annotated.mp4` — video with the skeleton and ball/bat boxes drawn
- `net_tracking.json` — `{ "frames": [ { frame, time, pose, objects } ... ] }`

## Project layout

```
config.yaml          tweakable settings (device, models, confidence, classes)
scripts/track.py     command-line entry point
src/cric/
  video.py           read/write video
  pose.py            batsman skeleton (YOLOv8-pose)
  objects.py         ball/bat detection (YOLOv8)
  annotate.py        draw overlays
  pipeline.py        orchestrates everything
data/raw/            your input footage (git-ignored)
data/output/         generated videos + json (git-ignored)
```

## Roadmap

- [x] v0.1 — pose + ball/bat overlay and JSON export (single camera)
- [ ] Train a cricket-specific detector (cricket ball, bat, stumps)
- [ ] Object tracking / smoothing across frames (stable IDs, fill gaps)
- [ ] Multi-camera sync + calibration → 3D ball trajectory
- [ ] Ball speed, line & length, bounce point
- [ ] Shot detection & classification (drive / cut / pull / defense …)
- [ ] Batting analytics dashboard
```
