"""The orchestrator: wire every stage into one loop.

``App`` owns nothing clever — it just moves data along the pipeline:

    source -> detector -> registry -> seed/params (cached per identity)
           -> track manager -> renderer

The cleverness lives in the stages. Keeping the loop this thin is what makes the
single-face Phase 1 and the many-face Phase 2 literally the same code path: the
loop already handles "for each face in the frame".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .identity.registry import IdentityRegistry
from .types import VisualParams
from .visuals.render import SingularityRenderer
from .visuals.seed import embedding_to_visual_params
from .visuals.tracking import TrackManager


@dataclass
class AppConfig:
    width: int = 1280
    height: int = 720
    headless: bool = False
    fps: float = 30.0
    threshold: float = 0.6
    registry_path: str | None = None  # persist biometric data across runs if set
    max_frames: int | None = None
    record_path: str | None = None  # write an MP4 of the output if set


class App:
    def __init__(self, source, detector, config: AppConfig):
        self.source = source
        self.detector = detector
        self.config = config

        if config.registry_path:
            self.registry = IdentityRegistry.load(config.registry_path, threshold=config.threshold)
        else:
            self.registry = IdentityRegistry(threshold=config.threshold)

        self.renderer = SingularityRenderer(
            width=config.width, height=config.height, headless=config.headless
        )
        self.tracks = TrackManager()
        self._params_cache: dict[int, VisualParams] = {}
        self._writer = None
        if config.record_path:
            self._writer = _VideoWriter(config.record_path, config.width, config.height, config.fps)

    def params_for(self, identity_id: int, embedding: np.ndarray) -> VisualParams:
        """Derive (once) and cache the visual fingerprint for an identity.

        Caching by ``identity_id`` is purely a performance choice — recomputing
        from the embedding would give the same result, by design.
        """

        params = self._params_cache.get(identity_id)
        if params is None:
            params = embedding_to_visual_params(embedding)
            self._params_cache[identity_id] = params
        return params

    def run(self) -> None:
        start = time.time()
        frame_interval = 1.0 / self.config.fps if self.config.fps else 0.0
        frame_count = 0
        try:
            for frame in self.source:
                if self.renderer.should_quit():
                    break

                t = time.time() - start
                self._step(frame, t)

                self.renderer.present()
                if self._writer is not None:
                    self._writer.write(self.renderer.frame_array())

                frame_count += 1
                if self.config.max_frames and frame_count >= self.config.max_frames:
                    break
                if frame_interval and not self.config.headless:
                    self._sleep_to_pace(frame_interval)
        finally:
            self._shutdown()

    def _step(self, frame: np.ndarray, t: float) -> None:
        observations = self.detector.detect(frame)
        self.tracks.begin_frame()
        for obs in observations:
            identity_id = self.registry.resolve(obs.embedding)
            self.params_for(identity_id, obs.embedding)
            nx, ny = obs.normalized_center
            pixel_pos = (nx * self.config.width, ny * self.config.height)
            self.tracks.observe(identity_id, pixel_pos)

        visible = self.tracks.end_frame()
        self.renderer.begin()
        for track in visible:
            params = self._params_cache.get(track.identity_id)
            if params is not None:
                self.renderer.draw(track, params, t)

    def _sleep_to_pace(self, interval: float) -> None:
        # Coarse pacing for live windows; headless runs go flat out.
        time.sleep(min(interval, 0.05))

    def _shutdown(self) -> None:
        if self.config.registry_path:
            self.registry.save(self.config.registry_path)
        if self._writer is not None:
            self._writer.release()
        self.renderer.close()
        self.source.release()


class _VideoWriter:
    """Thin OpenCV MP4 writer (RGB in, BGR on disk)."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        import cv2

        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps or 30.0, (width, height))

    def write(self, rgb_frame: np.ndarray) -> None:
        self.writer.write(self._cv2.cvtColor(rgb_frame, self._cv2.COLOR_RGB2BGR))

    def release(self) -> None:
        self.writer.release()
