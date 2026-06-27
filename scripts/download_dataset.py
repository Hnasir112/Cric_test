"""Download a labelled cricket dataset from Roboflow Universe (YOLOv8 format).

You need a FREE Roboflow account and its API key:
    1. Sign up / log in at https://roboflow.com
    2. Go to Settings -> API Keys (or https://app.roboflow.com/settings/api)
    3. Copy your "Private API Key"

Then run (the key can be passed as a flag or the ROBOFLOW_API_KEY env var):

    python scripts/download_dataset.py --api-key YOUR_KEY

By default it grabs the "cricket-balls" dataset (classes: Ball, Bat, Batsman).
You can point it at any other Universe dataset with --workspace/--project, e.g.
copy those from the dataset's "Download Dataset" snippet on Roboflow.

The data lands in:  data/datasets/<project>/  with a data.yaml ready for training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "datasets"


def main() -> None:
    p = argparse.ArgumentParser(description="Download a Roboflow cricket dataset")
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                   help="Roboflow private API key (or set ROBOFLOW_API_KEY)")
    p.add_argument("--workspace", default="cricket-rfyd8")
    p.add_argument("--project", default="cricket-balls-l1us5")
    p.add_argument("--version", type=int, default=None,
                   help="dataset version; default = latest available")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No API key. Pass --api-key YOUR_KEY or set ROBOFLOW_API_KEY.\n"
            "Get one at https://app.roboflow.com/settings/api"
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)

    # Work out which version to download (latest if not specified).
    version_num = args.version
    if version_num is None:
        versions = project.versions()
        if not versions:
            raise SystemExit("This project has no exported versions to download.")
        version_num = max(int(v.version.split("/")[-1]) for v in versions)
        print(f"Using latest version: {version_num}")

    DEST.mkdir(parents=True, exist_ok=True)
    location = str(DEST / args.project)
    dataset = project.version(version_num).download("yolov8", location=location)

    print("\nDownloaded to:", dataset.location)
    print("data.yaml at: ", Path(dataset.location) / "data.yaml")
    print("Next: python scripts/train_detector.py --data",
          Path(dataset.location) / "data.yaml")


if __name__ == "__main__":
    main()
