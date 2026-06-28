"""Pull frames out of a video so you can label them for fine-tuning.

When your own net footage is ready, run this to sample frames (you don't label
every frame -- every ~15th is plenty since neighbours look almost identical).
Then upload the folder to a labelling tool (Roboflow, Label Studio, LabelImg),
draw boxes around Ball / Bat, and export in YOLOv8 format.

    python scripts/extract_frames.py --input data/raw/my_session.mp4
    python scripts/extract_frames.py --input clip.mp4 --every 10 --max 300

Frames land in:  data/label_queue/<video name>/frame_000123.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames for labelling")
    p.add_argument("--input", required=True, help="video file")
    p.add_argument("--every", type=int, default=15,
                   help="save 1 frame out of every N (default 15)")
    p.add_argument("--max", type=int, default=None,
                   help="stop after saving this many frames")
    p.add_argument("--output", default=None, help="output folder")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else ROOT / "data" / "label_queue" / inp.stem
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {inp}")

    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            cv2.imwrite(str(out / f"frame_{idx:06d}.jpg"), frame)
            saved += 1
            if args.max and saved >= args.max:
                break
        idx += 1
    cap.release()

    print(f"Saved {saved} frames to {out}")
    print("Next: label these (Ball/Bat) in Roboflow or Label Studio,")
    print("export as YOLOv8, then run scripts/finetune.py on the export.")


if __name__ == "__main__":
    main()
