"""Face detection + embedding, behind a swappable interface.

The rest of the pipeline depends only on :class:`Detector` — "give me a frame,
I'll give you :class:`FaceObservation` objects". That indirection is what lets
us:

* run the real ``face_recognition`` backend when a camera and dlib are present,
* swap in InsightFace later (Phase 2) for better multi-face separation, and
* run a deterministic :class:`SyntheticDetector` in headless CI / on this cloud
  box where there is no camera at all, so the whole pipeline stays testable.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..types import FaceObservation


def _scale_boxes_up(
    boxes: list[tuple[int, int, int, int]],
    inv_scale: float,
    height: int,
    width: int,
) -> list[tuple[int, int, int, int]]:
    """Map ``(top, right, bottom, left)`` boxes found on a downscaled frame back to
    full-resolution coordinates, clamped to the frame so a rounded-up edge can't
    fall outside the image (``face_encodings`` requires in-bounds boxes)."""

    scaled: list[tuple[int, int, int, int]] = []
    for top, right, bottom, left in boxes:
        scaled.append(
            (
                min(height, max(0, round(top * inv_scale))),
                min(width, max(0, round(right * inv_scale))),
                min(height, max(0, round(bottom * inv_scale))),
                min(width, max(0, round(left * inv_scale))),
            )
        )
    return scaled


class Detector(Protocol):
    """Anything that can turn a frame into face observations."""

    def detect(self, frame: np.ndarray) -> list[FaceObservation]: ...

    @property
    def embedding_dim(self) -> int: ...


class FaceRecognitionDetector:
    """Real backend built on ``face_recognition`` (dlib).

    Imports ``face_recognition`` lazily so the rest of the package — and the
    whole synthetic/test path — works on machines where dlib is not installed.

    ``model`` is ``"hog"`` (fast, CPU, best on near frontal faces) or ``"cnn"``
    (slower, GPU-friendly, better at oblique angles). Phase 2's wider camera will
    likely want ``"cnn"``.
    """

    embedding_dim = 128

    def __init__(self, model: str = "hog", upsample: int = 1, detect_scale: float = 1.0):
        try:
            import face_recognition  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "FaceRecognitionDetector needs the 'face_recognition' package "
                "(which builds on dlib). Install it with "
                "`pip install face_recognition`, or use SyntheticDetector for a "
                "camera-free run. See the README for dlib build notes."
            ) from exc
        if not 0.0 < detect_scale <= 1.0:
            raise ValueError(f"detect_scale must be in (0, 1], got {detect_scale}")
        self._fr = face_recognition
        self.model = model
        self.upsample = upsample
        # Fraction to shrink the frame by *for detection only*. Detection cost is
        # ~linear in pixels, so 0.5 ≈ 4x faster face_locations; boxes are scaled
        # back up and embeddings are still computed at full resolution (§4.2 A3).
        self.detect_scale = detect_scale

    def _locate(self, frame: np.ndarray, height: int, width: int):
        """Find face boxes in full-resolution coordinates, optionally detecting on
        a downscaled copy of the frame for speed."""

        if self.detect_scale >= 1.0:
            return self._fr.face_locations(
                frame, number_of_times_to_upsample=self.upsample, model=self.model
            )
        import cv2

        small = cv2.resize(
            frame,
            (max(1, round(width * self.detect_scale)), max(1, round(height * self.detect_scale))),
            interpolation=cv2.INTER_AREA,
        )
        small = np.ascontiguousarray(small)
        boxes_small = self._fr.face_locations(
            small, number_of_times_to_upsample=self.upsample, model=self.model
        )
        if not boxes_small:
            return []
        return _scale_boxes_up(boxes_small, 1.0 / self.detect_scale, height, width)

    def detect(self, frame: np.ndarray) -> list[FaceObservation]:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        boxes = self._locate(frame, height, width)
        if not boxes:
            return []
        encodings = self._fr.face_encodings(frame, boxes)
        return [
            FaceObservation(
                embedding=np.asarray(enc, dtype=np.float64),
                box=box,
                frame_shape=(height, width),
            )
            for box, enc in zip(boxes, encodings)
        ]


class SyntheticDetector:
    """Camera-free detector that fabricates deterministic moving faces.

    Each "persona" has a fixed embedding (so it resolves to a stable identity and
    a stable singularity) and a smooth Lissajous path across the frame (so the
    tracking and rendering stages have something to follow). This is how the
    pipeline is exercised end-to-end here, where ``/dev/video*`` does not exist.

    The embeddings are intentionally well separated in vector space so the
    registry never confuses two personas — this stands in for "different people
    have different faces", not for the messiness of real detection.
    """

    embedding_dim = 128

    def __init__(self, num_personas: int = 3, dim: int = 128, jitter: float = 0.02, seed: int = 7):
        self.dim = dim
        self.jitter = jitter
        self._rng = np.random.default_rng(seed)
        self._personas = [self._make_persona(i) for i in range(num_personas)]
        self._frame_index = 0

    def _make_persona(self, index: int) -> dict:
        # A base vector plus a strong, unique offset on a handful of dims keeps
        # personas far apart regardless of per-frame jitter.
        base = self._rng.normal(0, 0.1, size=self.dim)
        for d in self._rng.choice(self.dim, size=8, replace=False):
            base[d] += self._rng.uniform(-1.5, 1.5)
        path_rng = np.random.default_rng(1000 + index)
        return {
            "embedding": base,
            "freq": path_rng.uniform(0.05, 0.15, size=2),
            "phase": path_rng.uniform(0, 2 * np.pi, size=2),
            "size": path_rng.uniform(120, 200),
        }

    def detect(self, frame: np.ndarray) -> list[FaceObservation]:
        """``frame`` is used only for its shape; an internal counter drives motion."""

        height, width = frame.shape[:2]
        t = float(self._frame_index)
        self._frame_index += 1

        observations: list[FaceObservation] = []
        for p in self._personas:
            cx = (0.5 + 0.35 * np.sin(p["freq"][0] * t + p["phase"][0])) * width
            cy = (0.5 + 0.30 * np.sin(p["freq"][1] * t + p["phase"][1])) * height
            half = p["size"] / 2
            box = (
                int(cy - half),  # top
                int(cx + half),  # right
                int(cy + half),  # bottom
                int(cx - half),  # left
            )
            noise = self._rng.normal(0, self.jitter, size=self.dim)
            observations.append(
                FaceObservation(
                    embedding=p["embedding"] + noise,
                    box=box,
                    frame_shape=(height, width),
                )
            )
        return observations
