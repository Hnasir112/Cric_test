"""Ball and bat detection using a general YOLOv8 detector.

We don't have a cricket-specific model yet, so we lean on the standard COCO
classes that look closest to our objects:
    "sports ball"  -> cricket ball
    "baseball bat" -> cricket bat
This is deliberately a stepping stone. Once we collect and label net footage we
swap the weights for a model that knows "cricket_ball", "stumps", etc., and the
rest of the pipeline stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """One detected object in a frame."""
    label: str                 # our name: "ball" or "bat"
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu",
                 confidence: float = 0.25,
                 class_map: dict[str, int] | None = None):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence

        if class_map:
            # Explicit mapping (e.g. COCO stand-ins {"ball": 32, "bat": 34}):
            # only keep these class ids and rename them to our friendly labels.
            self.class_map = class_map
            self._id_to_label = {v: k.lower() for k, v in class_map.items()}
            self._filter_ids = list(class_map.values())
        else:
            # No mapping given -> this is a custom-trained model. Trust every
            # class it knows and use the model's own names (lower-cased).
            self.class_map = None
            self._id_to_label = {i: n.lower() for i, n in self.model.names.items()}
            self._filter_ids = None  # don't filter; keep all classes

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return all ball/bat detections in one frame."""
        results = self.model.predict(
            image,
            device=self.device,
            conf=self.confidence,
            classes=self._filter_ids,  # None = keep every class the model knows
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            label = self._id_to_label.get(int(cls_id))
            if label is None:
                continue
            detections.append(
                Detection(label=label, box=tuple(map(float, box)),
                          confidence=float(conf))
            )
        return detections
