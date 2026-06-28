"""Fine-tune the existing cricket detector on YOUR own labelled footage.

This is the "specialist" step. Instead of training from scratch, it CONTINUES
from the model we already trained (models/cricket_best.pt) and adapts it to your
specific nets / camera angles / ball. Because your setup is consistent, a few
hundred labelled frames can lift accuracy a lot -- especially on the ball.

Workflow:
    1. python scripts/extract_frames.py --input data/raw/my_session.mp4
    2. label the frames (Ball/Bat) and export YOLOv8 -> data/datasets/mine/
    3. python scripts/finetune.py --data data/datasets/mine/data.yaml

The fine-tuned weights land in: runs/detect/cricket_finetune/weights/best.pt
Point config.yaml's object_model at them to use the sharper model.

Tip: keep the SAME class order as the base model (Ball, Bat, [Batsman]) so the
fine-tune builds on what it already knows instead of relearning from zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "cricket_best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune cricket detector on your data")
    p.add_argument("--data", required=True, help="path to YOUR dataset data.yaml")
    p.add_argument("--base", default=str(BASE),
                   help="weights to fine-tune from (default: our trained model)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640,
                   help="bump to 640+ if footage is HD -- helps the small ball")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu", help="'0' for GPU, or 'cpu'")
    p.add_argument("--name", default="cricket_finetune")
    args = p.parse_args()

    if not Path(args.base).exists():
        raise SystemExit(
            f"Base weights not found: {args.base}\n"
            "Train the base model first (scripts/train_detector.py) or pass --base."
        )

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=str(ROOT / "runs" / "detect"),
        patience=20,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print("\nFine-tuning done.")
    print(f"Sharper weights: {best}")
    print("Point config.yaml object_model at this file to use it.")


if __name__ == "__main__":
    main()
