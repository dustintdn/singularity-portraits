"""The orchestrator: wire every stage into one loop.

``App`` owns nothing clever — it just moves data along the pipeline:

    source -> detector -> registry -> seed/params (cached per identity)
           -> track manager -> renderer

The cleverness lives in the stages. Keeping the loop this thin is what makes the
single-face Phase 1 and the many-face Phase 2 literally the same code path: the
loop already handles "for each face in the frame".
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .identity.association import associate_boxes_to_tracks
from .identity.registry import IdentityRegistry
from .types import FaceObservation, VisualParams
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
    side_by_side: bool = False
    style: str = "classic"


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
            width=config.width, height=config.height, headless=config.headless,
            side_by_side=config.side_by_side, style=config.style,
        )
        self.tracks = TrackManager()
        self._params_cache: dict[int, VisualParams] = {}
        self._face_labels: list[tuple[tuple, int]] = []
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

        self._detect_lock = threading.Lock()
        self._latest_observations: list = []
        # Published by the render thread, read by the detect thread: last-known
        # (identity_id, box) per live track, so detection can skip re-embedding a
        # face that's already tracked (§4.2 A4).
        self._track_boxes_snapshot: list = []
        self._detect_frame: np.ndarray | None = None
        self._detect_ready = threading.Event()
        self._stop_detect = threading.Event()
        detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        detect_thread.start()

        try:
            for frame in self.source:
                if self.renderer.should_quit():
                    break

                t = time.time() - start

                with self._detect_lock:
                    self._detect_frame = frame
                self._detect_ready.set()

                self._step_render(t)

                if self.config.side_by_side:
                    self.renderer.present(frame, self._face_labels)
                else:
                    self.renderer.present()
                if self._writer is not None:
                    self._writer.write(self.renderer.frame_array())

                frame_count += 1
                if self.config.max_frames and frame_count >= self.config.max_frames:
                    break
                if frame_interval and not self.config.headless:
                    self._sleep_to_pace(frame_interval)
        finally:
            self._stop_detect.set()
            self._detect_ready.set()
            detect_thread.join(timeout=1.0)
            self._shutdown()

    def _detect_loop(self) -> None:
        while not self._stop_detect.is_set():
            self._detect_ready.wait()
            self._detect_ready.clear()
            if self._stop_detect.is_set():
                break
            with self._detect_lock:
                frame = self._detect_frame
                track_boxes = list(self._track_boxes_snapshot)
            if frame is None:
                continue
            if getattr(self.detector, "incremental", False):
                observations = self._detect_incremental(frame, track_boxes)
            else:
                observations = self.detector.detect(frame)
            with self._detect_lock:
                self._latest_observations = observations

    def _detect_incremental(self, frame, track_boxes) -> list:
        """Locate faces, reuse identities for boxes that overlap live tracks, and
        embed only the genuinely new ones (§4.2 A4)."""

        boxes = self.detector.locate(frame)
        if not boxes:
            return []
        height, width = frame.shape[:2]
        matches, unmatched = associate_boxes_to_tracks(boxes, track_boxes)
        new_embeddings = self.detector.embed(frame, [boxes[i] for i in unmatched])
        emb_by_box = dict(zip(unmatched, new_embeddings))

        observations = []
        for i, box in enumerate(boxes):
            if i in matches:
                observations.append(
                    FaceObservation(
                        embedding=None,
                        box=box,
                        frame_shape=(height, width),
                        identity_id=matches[i],
                    )
                )
            else:
                observations.append(
                    FaceObservation(
                        embedding=emb_by_box[i],
                        box=box,
                        frame_shape=(height, width),
                    )
                )
        return observations

    def _step_render(self, t: float) -> None:
        with self._detect_lock:
            observations = list(self._latest_observations)

        # Observations arrive either already tied to a tracked identity (detection
        # reused it and skipped the embedding) or carrying a fresh embedding to
        # resolve. Resolve the latter in one batch, preserving order.
        new_ids = iter(
            self.registry.resolve_many(
                [obs.embedding for obs in observations if obs.identity_id is None]
            )
        )

        labels = []
        self.tracks.begin_frame()
        for obs in observations:
            if obs.identity_id is not None:
                identity_id = obs.identity_id
            else:
                identity_id = next(new_ids)
                # Populate the params cache for genuinely new identities; already
                # tracked ones were cached when they first appeared.
                self.params_for(identity_id, obs.embedding)
            labels.append((obs.box, identity_id))
            nx, ny = obs.normalized_center
            if self.config.side_by_side:
                nx = 1.0 - nx
            pixel_pos = (nx * self.config.width, ny * self.config.height)
            self.tracks.observe(identity_id, pixel_pos, obs.box)

        self._face_labels = labels
        # Publish where each identity currently is, so the detect thread can match
        # a returning face by overlap and skip re-embedding it.
        with self._detect_lock:
            self._track_boxes_snapshot = self.tracks.active_boxes()
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
