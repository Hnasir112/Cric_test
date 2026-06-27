"""Drawing overlays onto frames: the batsman skeleton and ball/bat boxes."""

from __future__ import annotations

import cv2
import numpy as np

from .objects import Detection
from .pose import SKELETON_EDGES, PoseResult

# BGR colours (OpenCV uses Blue-Green-Red order, not RGB).
SKELETON_COLOR = (0, 255, 0)     # green
JOINT_COLOR = (0, 200, 255)      # amber
BOX_COLORS = {
    "ball": (0, 0, 255),         # red
    "bat": (255, 128, 0),        # blue-ish
}

# Don't draw a keypoint/limb we're not confident about (avoids junk lines).
MIN_KEYPOINT_CONF = 0.30


def draw_pose(image: np.ndarray, pose: PoseResult) -> None:
    """Draw the batsman skeleton onto the frame, in place."""
    if not pose.found:
        return

    # Limbs.
    for a, b in SKELETON_EDGES:
        if pose.confidence[a] < MIN_KEYPOINT_CONF or pose.confidence[b] < MIN_KEYPOINT_CONF:
            continue
        pa = tuple(pose.keypoints[a].astype(int))
        pb = tuple(pose.keypoints[b].astype(int))
        cv2.line(image, pa, pb, SKELETON_COLOR, 2)

    # Joints.
    for i in range(len(pose.keypoints)):
        if pose.confidence[i] < MIN_KEYPOINT_CONF:
            continue
        p = tuple(pose.keypoints[i].astype(int))
        cv2.circle(image, p, 4, JOINT_COLOR, -1)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> None:
    """Draw ball/bat bounding boxes with labels, in place."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        color = BOX_COLORS.get(det.label, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(image, text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_hud(image: np.ndarray, frame_index: int, timestamp: float) -> None:
    """Draw a small frame/time readout in the top-left corner."""
    text = f"frame {frame_index}  t={timestamp:.2f}s"
    cv2.putText(image, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
