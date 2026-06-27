"""Reading frames from a video and writing an annotated video back out.

Thin, friendly wrappers around OpenCV's VideoCapture / VideoWriter so the rest
of the code never has to touch OpenCV's quirky API directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    """A single video frame plus where it sits in time."""
    index: int          # 0-based frame number
    timestamp: float    # seconds from the start of the clip
    image: np.ndarray   # the pixels, as a BGR numpy array (OpenCV's format)


class VideoReader:
    """Iterate over the frames of a video file.

    Example:
        with VideoReader("net.mp4") as reader:
            print(reader.fps, reader.width, reader.height)
            for frame in reader:
                ...  # frame.image is a numpy array
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            timestamp = index / self.fps if self.fps else 0.0
            yield Frame(index=index, timestamp=timestamp, image=image)
            index += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    """Write frames to an .mp4 file.

    Example:
        with VideoWriter("out.mp4", fps=30, size=(1920, 1080)) as writer:
            writer.write(image)
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {self.path}")

    def write(self, image: np.ndarray) -> None:
        self._writer.write(image)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
