"""Motion smoothing and per-identity lifecycle.

Face bounding boxes jitter frame to frame, and faces blink in and out as
detection misses a frame or two. Two small classes handle both:

``SmoothedPosition``
    An exponential moving average so a singularity glides instead of snapping.

``TrackManager``
    One ``SmoothedPosition`` per identity plus presence bookkeeping, so the app
    can fade a singularity in when its face appears and out when it leaves —
    and, in Phase 2, do this for many faces at once with no new logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SmoothedPosition:
    """Exponential moving average of a 2-D point.

    ``alpha`` is the weight given to each new observation: higher snaps faster,
    lower glides more. The first observation is taken as-is so the singularity
    does not slingshot in from a corner.
    """

    def __init__(self, alpha: float = 0.2, start: tuple[float, float] | None = None):
        self.alpha = alpha
        self.pos = start

    def update(self, new_pos: tuple[float, float]) -> tuple[float, float]:
        if self.pos is None:
            self.pos = new_pos
        else:
            self.pos = (
                self.alpha * new_pos[0] + (1 - self.alpha) * self.pos[0],
                self.alpha * new_pos[1] + (1 - self.alpha) * self.pos[1],
            )
        return self.pos


@dataclass
class Track:
    """Live state for one identity currently (or recently) on screen."""

    identity_id: int
    position: SmoothedPosition
    presence: float = 0.0  # 0 = invisible, 1 = fully present; eased each frame
    seen_this_frame: bool = False
    misses: int = 0


@dataclass
class TrackManager:
    """Keeps a :class:`Track` per identity and eases presence in and out.

    Designed for N faces from the start: ``begin_frame``/``observe``/``end_frame``
    is the same whether one face or twenty are in view.
    """

    fade_in_rate: float = 0.12
    fade_out_rate: float = 0.06
    # Frames an identity may go unseen before its track is forgotten entirely.
    # A couple of seconds at 30fps keeps a brief occlusion from dropping someone.
    max_misses: int = 60
    smoothing_alpha: float = 0.2
    tracks: dict[int, Track] = field(default_factory=dict)

    def begin_frame(self) -> None:
        for track in self.tracks.values():
            track.seen_this_frame = False

    def observe(self, identity_id: int, position: tuple[float, float]) -> Track:
        track = self.tracks.get(identity_id)
        if track is None:
            track = Track(
                identity_id=identity_id,
                position=SmoothedPosition(self.smoothing_alpha, start=position),
            )
            self.tracks[identity_id] = track
        track.position.update(position)
        track.seen_this_frame = True
        track.misses = 0
        return track

    def end_frame(self) -> list[Track]:
        """Ease presence, drop long-absent tracks, return what to draw.

        Returns tracks with any visible presence, most-present last so callers
        that paint in order get sensible layering for overlapping singularities.
        """

        for track in self.tracks.values():
            if track.seen_this_frame:
                track.presence = min(1.0, track.presence + self.fade_in_rate)
            else:
                track.misses += 1
                track.presence = max(0.0, track.presence - self.fade_out_rate)

        self.tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if not (t.misses > self.max_misses and t.presence <= 0.0)
        }

        visible = [t for t in self.tracks.values() if t.presence > 0.0]
        visible.sort(key=lambda t: t.presence)
        return visible
