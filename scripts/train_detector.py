"""Train a custom cricket object detector (ball / bat / stumps) with YOLOv8.

This is a thin, well-documented wrapper around Ultralytics training so you don't
have to remember all the flags. It expects a dataset in YOLO format described by
a `data.yaml` file:

    data/datasets/cricket/
        data.yaml          <- class names + paths to train/val images
        train/images/*.jpg
        train/labels/*.txt  <- one .txt per image: "class cx cy w h" (normalized)
        val/images/*.jpg
        val/labels/*.txt

Get such a dataset from Roboflow Universe (export format: "YOLOv8"), or build
your own with a labelling tool, then point --data at its data.yaml.

Examples:
    # GPU is only 2 GB here, so we default to small image size + batch.
    python scripts/train_detector.py --data data/datasets/cricket/data.yaml

    # train on CPU instead (slow but works without a capable GPU):
    python scripts/train_detector.py --data .../data.yaml --device cpu

After training, the best weights land in:
    runs/detect/cricket/weights/best.pt
Point config.yaml's `object_model:` at that file to use it in the tracker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Train cricket object detector")
    p.add_argument("--data", required=True, help="path to dataset data.yaml")
    p.add_argument("--base", default="yolov8n.pt",
                   help="starting weights to fine-tune from")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=416,
                   help="image size; keep small (416) for a 2 GB GPU")
    p.add_argument("--batch", type=int, default=8,
                   help="batch size; lower it if you hit out-of-memory")
    p.add_argument("--device", default="0",
                   help="'0' for first GPU, or 'cpu'")
    p.add_argument("--name", default="cricket", help="run name under runs/detect/")
    args = p.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=str(ROOT / "runs" / "detect"),
        patience=20,          # early-stop if no improvement for 20 epochs
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print("\nTraining done.")
    print(f"Best weights: {best}")
    print("Next: set  object_model:  in config.yaml to this path, then run track.py")


if __name__ == "__main__":
    main()
