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

## Custom cricket detector (v0.2)

We trained a YOLOv8 detector on a labelled cricket dataset (Ball / Bat / Batsman).
Final validation results (`runs/detect/cricket`):

| Class   | Precision | Recall | mAP@50 |
|---------|-----------|--------|--------|
| Batsman | 1.00      | 0.98   | 0.98   |
| Bat     | 0.80      | 0.69   | 0.75   |
| Ball    | 0.83      | 0.50   | 0.55   |

The ball is the hard class (small object, low-res test footage). The path to
improving it: higher-resolution footage, fine-tuning on real net clips, and a
temporal tracking layer. Weights are attached to the GitHub Release, or
reproduce with:

```powershell
python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_KEY
python scripts/train_detector.py --data data/datasets/cricket-balls-l1us5/data.yaml --device cpu
```

To use the trained model in the tracker: `track.py --config config_custom.yaml`.

## Fine-tuning on YOUR own footage

The biggest accuracy gain comes from adapting the model to your actual setup
(your cameras, ball, lighting). When you have net footage:

```powershell
# 1. sample frames to label
python scripts/extract_frames.py --input data/raw/my_session.mp4

# 2. label them (Ball/Bat) in Roboflow / Label Studio, export YOLOv8 to
#    data/datasets/mine/ , then:

# 3. fine-tune from our trained model (not from scratch)
python scripts/finetune.py --data data/datasets/mine/data.yaml
```

## Roadmap

- [x] v0.1 — pose + ball/bat overlay and JSON export (single camera)
- [x] v0.2 — trained cricket-specific detector (Ball / Bat / Batsman)
- [x] Fine-tune workflow (extract_frames + finetune) for your own footage
- [ ] Object tracking / smoothing across frames (stable IDs, fill gaps)
- [ ] Multi-camera sync + calibration → 3D ball trajectory
- [ ] Ball speed, line & length, bounce point
- [ ] Shot detection & classification (drive / cut / pull / defense …)
- [ ] Batting analytics dashboard
```
