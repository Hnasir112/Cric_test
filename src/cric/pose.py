"""Batsman body-skeleton detection using YOLOv8-pose.

YOLOv8-pose finds people and, for each one, 17 body keypoints in the COCO
format (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles). In a net
session there can be more than one person in frame (batsman, thrower, net), so
we pick the *batsman* as the largest detected person, which is usually the one
closest to the camera and fully in shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

# The 17 COCO keypoint names, in the order YOLOv8-pose returns them.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Which keypoints to connect with lines when we draw the skeleton.
SKELETON_EDGES = [
    (5, 7), (7, 9),      # left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 10),     # right arm
    (11, 13), (13, 15),  # left leg: hip -> knee -> ankle
    (12, 14), (14, 16),  # right leg
    (5, 6), (11, 12),    # shoulders, hips
    (5, 11), (6, 12),    # torso sides
    (0, 5), (0, 6),      # nose -> shoulders
]


@dataclass
class PoseResult:
    """The batsman's skeleton for one frame."""
    # (17, 2) array of x,y pixel coordinates for each keypoint.
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((17, 2)))
    # (17,) array of per-keypoint confidence 0..1.
    confidence: np.ndarray = field(default_factory=lambda: np.zeros(17))
    found: bool = False

    def named(self) -> dict[str, dict[str, float]]:
        """Return keypoints keyed by body-part name, for the JSON output."""
        out: dict[str, dict[str, float]] = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            out[name] = {
                "x": float(self.keypoints[i, 0]),
                "y": float(self.keypoints[i, 1]),
                "conf": float(self.confidence[i]),
            }
        return out


class PoseEstimator:
    def __init__(self, model_path: str = "yolov8n-pose.pt", device: str = "cpu",
                 confidence: float = 0.25):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence

    def estimate(self, image: np.ndarray) -> PoseResult:
        """Run pose detection on one frame and return the batsman's skeleton."""
        results = self.model.predict(
            image, device=self.device, conf=self.confidence, verbose=False
        )
        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return PoseResult()

        # Pick the batsman = the person with the largest bounding-box area.
        boxes = result.boxes.xywh.cpu().numpy()          # (n, 4): x,y,w,h
        areas = boxes[:, 2] * boxes[:, 3]
        best = int(np.argmax(areas))

        xy = result.keypoints.xy.cpu().numpy()[best]      # (17, 2)
        conf = result.keypoints.conf
        conf = conf.cpu().numpy()[best] if conf is not None else np.ones(17)

        return PoseResult(keypoints=xy, confidence=conf, found=True)
