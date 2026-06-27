"""The end-to-end pipeline: a video clip in, an annotated clip + data out.

For every frame we:
  1. detect the batsman skeleton (pose)
  2. detect the ball and bat (objects)
  3. draw both onto the frame and write it to the output video
  4. record the numbers into a list we save as JSON at the end
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import annotate
from .objects import ObjectDetector
from .pose import PoseEstimator
from .video import VideoReader, VideoWriter


@dataclass
class PipelineConfig:
    device: str = "cpu"
    pose_model: str = "yolov8n-pose.pt"
    object_model: str = "yolov8n.pt"
    confidence: float = 0.25
    object_classes: dict[str, int] | None = None
    max_frames: int | None = None   # stop early (handy for quick tests)
    show: bool = False              # open a live preview window while processing


def analyze(
    input_path: str | Path,
    output_video: str | Path,
    output_data: str | Path,
    config: PipelineConfig,
) -> dict:
    """Run the full analysis. Returns a summary dict."""
    pose = PoseEstimator(config.pose_model, config.device, config.confidence)
    objects = ObjectDetector(
        config.object_model, config.device, config.confidence,
        class_map=config.object_classes,
    )

    records: list[dict] = []

    with VideoReader(input_path) as reader:
        size = (reader.width, reader.height)
        with VideoWriter(output_video, reader.fps, size) as writer:
            for frame in reader:
                if config.max_frames is not None and frame.index >= config.max_frames:
                    break

                pose_result = pose.estimate(frame.image)
                detections = objects.detect(frame.image)

                # Draw overlays onto the frame (in place) and write it out.
                annotate.draw_pose(frame.image, pose_result)
                annotate.draw_detections(frame.image, detections)
                annotate.draw_hud(frame.image, frame.index, frame.timestamp)
                writer.write(frame.image)

                # Live preview: show the annotated frame in a window. Press 'q'
                # (or Esc) to stop early; the video/JSON saved so far are kept.
                if config.show:
                    cv2.imshow("cric tracking - press q to quit", frame.image)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                # Save the raw numbers for later analytics.
                records.append({
                    "frame": frame.index,
                    "time": round(frame.timestamp, 4),
                    "pose": pose_result.named() if pose_result.found else None,
                    "objects": [
                        {"label": d.label, "box": d.box,
                         "center": d.center, "conf": d.confidence}
                        for d in detections
                    ],
                })

                if frame.index % 25 == 0:
                    print(f"  processed frame {frame.index}...")

    if config.show:
        cv2.destroyAllWindows()

    Path(output_data).parent.mkdir(parents=True, exist_ok=True)
    with open(output_data, "w") as f:
        json.dump({"frames": records}, f, indent=2)

    summary = {
        "frames_processed": len(records),
        "frames_with_pose": sum(1 for r in records if r["pose"]),
        "ball_detections": sum(
            1 for r in records for o in r["objects"] if o["label"] == "ball"
        ),
        "bat_detections": sum(
            1 for r in records for o in r["objects"] if o["label"] == "bat"
        ),
        "output_video": str(output_video),
        "output_data": str(output_data),
    }
    return summary
