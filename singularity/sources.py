"""Frame sources — where pixels come from.

Every source is an iterable of RGB ``numpy`` frames, so the app loop does not
care whether the pixels come from a webcam, a recorded clip, a folder of stills,
or nowhere at all (the synthetic source). Phase 2's overhead/wide-angle camera
slots in here as just another source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


class FrameSource(Protocol):
    """Anything that yields RGB frames and can be released."""

    def __iter__(self) -> Iterator[np.ndarray]: ...

    def release(self) -> None: ...


class WebcamSource:
    """Live camera via OpenCV. Converts BGR -> RGB for the rest of the pipeline."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. On a headless machine use "
                f"`--source synthetic` instead."
            )
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self.cap.release()


class VideoFileSource:
    """A recorded video file, read frame by frame via OpenCV."""

    def __init__(self, path: str | Path, loop: bool = False):
        import cv2

        self._cv2 = cv2
        self.path = str(path)
        self.loop = loop
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self.cap.release()


class ImageDirSource:
    """A folder of still images, yielded in sorted filename order.

    Handy for reproducible tests with real faces (drop a few photos in a folder)
    without needing a live camera.
    """

    _EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, directory: str | Path, repeat: int = 1):
        import cv2

        self._cv2 = cv2
        self.directory = Path(directory)
        self.repeat = repeat
        self.paths = sorted(
            p for p in self.directory.iterdir() if p.suffix.lower() in self._EXTS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {self.directory}")

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.repeat):
            for path in self.paths:
                img = self._cv2.imread(str(path))
                if img is None:
                    continue
                yield self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        pass


class SyntheticSource:
    """Blank frames of a fixed size — pixels nobody looks at.

    Pairs with :class:`~singularity.identity.detector.SyntheticDetector`, which
    fabricates moving faces on its own clock. Together they let the whole
    pipeline run, render, and be recorded with no camera present.
    """

    def __init__(self, width: int = 1280, height: int = 720, num_frames: int | None = None):
        self.width = width
        self.height = height
        self.num_frames = num_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        count = 0
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while self.num_frames is None or count < self.num_frames:
            yield blank
            count += 1

    def release(self) -> None:
        pass
