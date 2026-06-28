"""Plot live training progress from a YOLO run's results.csv.

Run this anytime during/after training to get an up-to-date chart of the loss
curves and accuracy (mAP). Re-run it to refresh.

    python scripts/plot_training.py
    python scripts/plot_training.py --run runs/detect/cricket

Saves: <run>/live_progress.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just write a PNG
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load(csv_path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                key = k.strip()
                try:
                    rows.setdefault(key, []).append(float(v))
                except (ValueError, TypeError):
                    pass
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "runs" / "detect" / "cricket"))
    args = p.parse_args()

    run = Path(args.run)
    data = load(run / "results.csv")
    epochs = data.get("epoch", [])
    if not epochs:
        raise SystemExit("No epochs logged yet — check back after epoch 1.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: losses.
    for col, label in [("train/box_loss", "train box"),
                       ("val/box_loss", "val box"),
                       ("train/cls_loss", "train cls"),
                       ("val/cls_loss", "val cls")]:
        if col in data:
            ax1.plot(epochs, data[col], label=label)
    ax1.set_title("Losses (lower = better)")
    ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    # Right: accuracy metrics.
    for col, label in [("metrics/mAP50(B)", "mAP@50"),
                       ("metrics/mAP50-95(B)", "mAP@50-95"),
                       ("metrics/precision(B)", "precision"),
                       ("metrics/recall(B)", "recall")]:
        if col in data:
            ax2.plot(epochs, data[col], label=label, marker="o", ms=3)
    ax2.set_title("Accuracy (higher = better)")
    ax2.set_xlabel("epoch"); ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=0.3)

    last = int(epochs[-1])
    fig.suptitle(f"Cricket detector training — through epoch {last}")
    fig.tight_layout()
    out = run / "live_progress.png"
    fig.savefig(out, dpi=110)
    print("saved", out, f"(epochs logged: {last})")


if __name__ == "__main__":
    main()
