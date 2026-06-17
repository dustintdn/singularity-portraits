"""Shared data types passed between pipeline stages.

Keeping these in one place (rather than letting each module invent its own
tuples) is what keeps the stage boundaries honest: a detector returns
``FaceObservation`` objects and nothing downstream needs to know which backend
produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceObservation:
    """A single face found in a single frame.

    Attributes
    ----------
    embedding:
        Fixed-length identity vector. 128-d for ``face_recognition``, 512-d for
        InsightFace. The rest of the pipeline never assumes a particular length.
    box:
        Bounding box in pixel coordinates as ``(top, right, bottom, left)`` to
        match ``face_recognition``'s convention.
    frame_shape:
        ``(height, width)`` of the frame the box was measured in, so downstream
        stages can normalise positions independently of capture resolution.
    """

    embedding: np.ndarray
    box: tuple[int, int, int, int]
    frame_shape: tuple[int, int]

    @property
    def center(self) -> tuple[float, float]:
        top, right, bottom, left = self.box
        return ((left + right) / 2.0, (top + bottom) / 2.0)

    @property
    def normalized_center(self) -> tuple[float, float]:
        """Face centre as fractions of frame width/height, in ``[0, 1]``.

        Renderers work in their own output resolution, which is usually not the
        capture resolution, so they consume the normalised centre and scale it
        themselves.
        """

        height, width = self.frame_shape
        cx, cy = self.center
        return (cx / max(width, 1), cy / max(height, 1))

    @property
    def size(self) -> float:
        """Geometric mean of box width and height, in pixels."""

        top, right, bottom, left = self.box
        return float(np.sqrt(max(right - left, 1) * max(bottom - top, 1)))


@dataclass
class VisualParams:
    """The deterministic visual fingerprint derived from an identity.

    Everything here is a pure function of the identity's seed. Nothing in this
    object changes frame to frame — the *behaviour* it describes (pulse, drift)
    is animated by the renderer using a clock, but the parameters themselves are
    fixed for a given face.
    """

    seed: int
    hue_base: float
    hue_spread: float
    num_colors: int
    saturation: float
    angularity: float
    pulse_speed: float
    pulse_depth: float
    drift_speed: float
    motion_style: str
    noise_octaves: int
    noise_amplitude: float
    base_radius: float
    rotation_speed: float
    palette: list[tuple[int, int, int]] = field(default_factory=list)
