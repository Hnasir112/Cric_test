"""Command-line entry point for the cricket net-session analyzer.

Usage (from the project root, with the venv active):

    python scripts/track.py --input data/raw/net.mp4

It writes:
    data/output/<name>_annotated.mp4   (video with skeleton + ball/bat drawn)
    data/output/<name>_tracking.json   (per-frame coordinates)

Handy flags:
    --device cuda     run on the GPU
    --max-frames 100  only process the first 100 frames (quick test)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make "src/" importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cric.pipeline import PipelineConfig, analyze  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cricket net-session analyzer")
    parser.add_argument("--input", required=True, help="input video file")
    parser.add_argument("--output-video", help="annotated video path")
    parser.add_argument("--output-data", help="tracking JSON path")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--device", help="override device: cpu or cuda")
    parser.add_argument("--max-frames", type=int, help="process only N frames")
    parser.add_argument("--show", action="store_true",
                        help="open a live preview window while processing")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    input_path = Path(args.input)
    stem = input_path.stem
    out_dir = ROOT / "data" / "output"
    output_video = args.output_video or out_dir / f"{stem}_annotated.mp4"
    output_data = args.output_data or out_dir / f"{stem}_tracking.json"

    config = PipelineConfig(
        device=args.device or cfg.get("device", "cpu"),
        pose_model=cfg.get("pose_model", "yolov8n-pose.pt"),
        object_model=cfg.get("object_model", "yolov8n.pt"),
        confidence=cfg.get("confidence", 0.25),
        object_classes=cfg.get("object_classes"),
        max_frames=args.max_frames,
        show=args.show,
    )

    print(f"Analyzing {input_path} on device={config.device} ...")
    summary = analyze(input_path, output_video, output_data, config)

    print("\nDone.")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
