"""Rendering — turning visual parameters into moving light.

A singularity is drawn as a small stack of additively-blended glows around a
soft, noise-deformed core. Brightness, palette, jaggedness and motion all come
from the identity's :class:`VisualParams`; the only per-frame input is a clock,
so the same face always looks like itself while still breathing and drifting.

The renderer works the same whether it owns a real window or an offscreen
surface (headless), which is what lets it run on a camera-less cloud box and
emit PNGs/MP4s for review. Set ``headless=True`` to avoid opening a display.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ..types import VisualParams
from .tracking import Track

# A glow sprite is expensive to build, so cache one per (colour, radius bucket).
_GLOW_CACHE: dict[tuple, "object"] = {}


def _import_pygame(headless: bool):
    if headless:
        # Must be set before pygame initialises its subsystems. The dummy audio
        # driver keeps a server with no sound card from spewing ALSA warnings.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    return pygame


STYLES = ("classic", "vortex")


class SingularityRenderer:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        headless: bool = False,
        trail: float = 0.18,
        background: tuple[int, int, int] = (6, 6, 10),
        side_by_side: bool = False,
        style: str = "classic",
    ):
        self.width = width
        self.height = height
        self.headless = headless
        self.trail = trail  # 0 = long ghosting trails, 1 = none
        self.background = background
        self.side_by_side = side_by_side
        self.style = style

        self.pygame = _import_pygame(headless)
        self.pygame.init()
        screen_width = width * 2 if side_by_side else width
        if headless:
            self.canvas = self.pygame.Surface((width, height))
            self.screen = None
        else:
            self.screen = self.pygame.display.set_mode(
                (screen_width, height), self.pygame.RESIZABLE
            )
            self.pygame.display.set_caption("Singularity Portraits")
            self.canvas = self.pygame.Surface((width, height))
        self.canvas.fill(background)

        # Translucent veil reused each frame to fade the previous frame toward
        # the background colour, leaving motion trails.
        self._veil = self.pygame.Surface((width, height))
        self._veil.fill(background)
        self._veil.set_alpha(int(self.trail * 255))

    # -- public API -----------------------------------------------------------

    def begin(self) -> None:
        """Fade the previous frame instead of clearing, to leave light trails."""

        self.canvas.blit(self._veil, (0, 0))

    def draw(self, track: Track, params: VisualParams, t: float) -> None:
        """Composite one singularity for ``track`` at time ``t`` (seconds)."""

        if self.style == "vortex":
            self._draw_vortex(track, params, t)
        else:
            self._draw_classic(track, params, t)

    def _draw_classic(self, track: Track, params: VisualParams, t: float) -> None:
        cx = track.position.pos[0]
        cy = track.position.pos[1]
        cx, cy = self._apply_motion(cx, cy, params, t)

        pulse = 1.0 + params.pulse_depth * math.sin(t * params.pulse_speed)
        radius = params.base_radius * pulse * (0.4 + 0.6 * track.presence)
        rotation = t * params.rotation_speed

        for i, color in enumerate(reversed(params.palette)):
            layer = len(params.palette) - 1 - i
            layer_radius = radius * (1.0 + layer * 0.55)
            glow = self._glow(color, layer_radius, track.presence)
            rect = glow.get_rect(center=(int(cx), int(cy)))
            self.canvas.blit(glow, rect, special_flags=self.pygame.BLEND_RGB_ADD)

        self._draw_core(cx, cy, radius, rotation, params, track.presence)

    def _fit(self, surface, box_w, box_h):
        src_w, src_h = surface.get_size()
        scale = min(box_w / src_w, box_h / src_h)
        fit_w, fit_h = int(src_w * scale), int(src_h * scale)
        scaled = self.pygame.transform.smoothscale(surface, (fit_w, fit_h))
        x = (box_w - fit_w) // 2
        y = (box_h - fit_h) // 2
        return scaled, x, y

    def present(self, camera_frame: np.ndarray | None = None) -> None:
        """Push the canvas to the window (no-op semantics in headless mode)."""

        if self.screen is not None:
            self.screen.fill((0, 0, 0))
            win_w, win_h = self.screen.get_size()
            if self.side_by_side and camera_frame is not None:
                panel_w = win_w // 2
                scaled, x, y = self._fit(self.canvas, panel_w, win_h)
                self.screen.blit(scaled, (x, y))
                cam_surf = self.pygame.surfarray.make_surface(
                    np.transpose(camera_frame, (1, 0, 2))
                )
                cam_surf = self.pygame.transform.flip(cam_surf, True, False)
                scaled, x, y = self._fit(cam_surf, panel_w, win_h)
                self.screen.blit(scaled, (panel_w + x, y))
            else:
                scaled, x, y = self._fit(self.canvas, win_w, win_h)
                self.screen.blit(scaled, (x, y))
            self.pygame.display.flip()

    def should_quit(self) -> bool:
        """True if the window was closed or Esc pressed (always False headless)."""

        if self.screen is None:
            return False
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return True
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                return True
            if event.type == self.pygame.KEYDOWN:
                idx = event.key - self.pygame.K_1
                if 0 <= idx < len(STYLES):
                    self.style = STYLES[idx]
            if event.type == self.pygame.VIDEORESIZE:
                self.screen = self.pygame.display.set_mode(
                    event.size, self.pygame.RESIZABLE
                )
        return False

    def frame_array(self) -> np.ndarray:
        """Current canvas as an ``(H, W, 3)`` RGB array (for video/PNG export)."""

        arr = self.pygame.surfarray.array3d(self.canvas)
        return np.transpose(arr, (1, 0, 2))  # pygame is (W, H, 3)

    def save(self, path: str) -> None:
        self.pygame.image.save(self.canvas, path)

    def close(self) -> None:
        self.pygame.quit()

    # -- internals ------------------------------------------------------------

    def _apply_motion(
        self, cx: float, cy: float, params: VisualParams, t: float
    ) -> tuple[float, float]:
        """Offset the tracked position by the identity's motion archetype."""

        style = params.motion_style
        amp = params.base_radius * 0.35
        s = params.drift_speed
        if style == "orbit":
            cx += amp * math.cos(t * s)
            cy += amp * math.sin(t * s)
        elif style == "jitter":
            # Deterministic high-frequency wobble (no RNG: stays reproducible).
            cx += 0.4 * amp * math.sin(t * s * 9.0)
            cy += 0.4 * amp * math.cos(t * s * 11.0)
        elif style == "drift":
            cx += amp * math.sin(t * s * 0.5)
            cy += 0.5 * amp * math.sin(t * s * 0.37 + 1.3)
        # "pulse" expresses itself through radius, not position.
        return cx, cy

    def _glow(self, color: tuple[int, int, int], radius: float, presence: float):
        """Radial-gradient sprite, cached by colour and radius bucket."""

        bucket = max(2, int(radius / 4) * 4)
        key = (color, bucket)
        glow = _GLOW_CACHE.get(key)
        if glow is None:
            glow = self._build_glow(color, bucket)
            _GLOW_CACHE[key] = glow
        if presence < 1.0:
            glow = glow.copy()
            glow.set_alpha(int(255 * presence))
        return glow

    def _build_glow(self, color: tuple[int, int, int], radius: int):
        """Build a soft additive glow: brightness falls off with distance^2."""

        size = radius * 2
        surf = self.pygame.Surface((size, size))
        surf.fill((0, 0, 0))
        cx = cy = radius
        # Vectorised falloff is far faster than per-pixel Python loops.
        yy, xx = np.ogrid[0:size, 0:size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1)
        falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2.2
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            arr[:, :, c] = (falloff * color[c]).astype(np.uint8)
        self.pygame.surfarray.blit_array(surf, np.transpose(arr, (1, 0, 2)))
        return surf

    def _draw_core(
        self,
        cx: float,
        cy: float,
        radius: float,
        rotation: float,
        params: VisualParams,
        presence: float,
    ) -> None:
        """Filled, noise-deformed polygon for the bright core of the form."""

        n_points = 36
        core_color = params.palette[0]
        points = []
        for i in range(n_points):
            ang = rotation + 2 * math.pi * i / n_points
            # Sum of a few sinusoids stands in for smooth noise; angularity and
            # noise_amplitude decide how far the outline departs from a circle.
            deform = 1.0
            for octave in range(1, params.noise_octaves + 1):
                deform += (params.noise_amplitude / octave) * math.sin(
                    octave * (ang * (1 + params.angularity * 3) + params.seed % 7)
                )
            r = radius * 0.55 * deform
            points.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))

        if presence >= 1.0:
            self.pygame.draw.polygon(self.canvas, core_color, points)
        else:
            tmp = self.pygame.Surface((self.width, self.height))
            tmp.fill((0, 0, 0))
            self.pygame.draw.polygon(tmp, core_color, points)
            tmp.set_alpha(int(255 * presence))
            self.canvas.blit(tmp, (0, 0), special_flags=self.pygame.BLEND_RGB_ADD)

    # -- vortex style ---------------------------------------------------------

    def _draw_vortex(self, track: Track, params: VisualParams, t: float) -> None:
        cx = track.position.pos[0]
        cy = track.position.pos[1]
        cx, cy = self._apply_motion(cx, cy, params, t)

        pulse = 1.0 + params.pulse_depth * 1.5 * math.sin(t * params.pulse_speed * 3.0)
        radius = params.base_radius * 0.55 * pulse * (0.4 + 0.6 * track.presence)
        rotation = t * params.rotation_speed * 5.0
        seed_offset = params.seed % 1000

        self._draw_vortex_tendrils(cx, cy, radius, rotation, params, t, track.presence, seed_offset)
        self._draw_vortex_particles(cx, cy, radius, rotation, params, t, track.presence, seed_offset)
        self._draw_vortex_core(cx, cy, radius, rotation, params, t, track.presence, seed_offset)

    def _draw_vortex_tendrils(
        self, cx, cy, radius, rotation, params, t, presence, seed_offset
    ):
        num_arms = 3 + (params.seed % 4)
        arm_length = radius * 3.5
        for arm in range(num_arms):
            base_angle = rotation + (2 * math.pi * arm / num_arms)
            color = params.palette[arm % len(params.palette)]
            dim = max(1, int(color[0] * 0.4)), max(1, int(color[1] * 0.4)), max(1, int(color[2] * 0.4))
            segments = 24
            points = []
            for s in range(segments):
                frac = s / segments
                curl = 2.5 + params.angularity * 2.0
                angle = base_angle + frac * curl
                wobble = math.sin(t * 8.0 + seed_offset + arm * 7.0 + s * 0.5) * radius * 0.15
                r = radius * 0.3 + frac * arm_length + wobble
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                points.append((px, py))
            if len(points) > 1:
                width = max(1, int(3 * presence * (1.0 - 0.5 * (len(points) / segments))))
                self.pygame.draw.lines(self.canvas, dim, False, points, width)

    def _draw_vortex_particles(
        self, cx, cy, radius, rotation, params, t, presence, seed_offset
    ):
        num_particles = 16 + (params.seed % 12)
        for i in range(num_particles):
            phase = seed_offset * 0.1 + i * 2.39996
            orbit_speed = 3.0 + math.sin(phase * 3.7) * 1.5
            orbit_r = radius * (0.4 + 1.8 * ((i * 0.618034) % 1.0))
            jitter = math.sin(t * 12.0 + i * 4.1) * radius * 0.12
            orbit_r += jitter
            angle = rotation * orbit_speed + phase + t * (1.5 + (i % 5) * 0.8)
            px = cx + orbit_r * math.cos(angle)
            py = cy + orbit_r * math.sin(angle)
            color = params.palette[i % len(params.palette)]
            particle_r = max(2, int(radius * 0.08 * (0.5 + 0.5 * math.sin(t * 10.0 + i))))
            glow = self._glow(color, particle_r * 2, presence)
            rect = glow.get_rect(center=(int(px), int(py)))
            self.canvas.blit(glow, rect, special_flags=self.pygame.BLEND_RGB_ADD)

    def _draw_vortex_core(
        self, cx, cy, radius, rotation, params, t, presence, seed_offset
    ):
        for i, color in enumerate(reversed(params.palette)):
            layer = len(params.palette) - 1 - i
            layer_radius = radius * (0.5 + layer * 0.3)
            glow = self._glow(color, layer_radius, presence)
            rect = glow.get_rect(center=(int(cx), int(cy)))
            self.canvas.blit(glow, rect, special_flags=self.pygame.BLEND_RGB_ADD)

        n_points = 48
        core_color = params.palette[0]
        points = []
        for i in range(n_points):
            ang = rotation * 1.5 + 2 * math.pi * i / n_points
            deform = 1.0
            for octave in range(1, 6):
                freq = octave * (1 + params.angularity * 4)
                deform += (0.3 / octave) * math.sin(
                    freq * ang + t * (5.0 + octave * 1.5) + seed_offset
                )
            r = radius * 0.4 * deform
            points.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))

        if presence >= 1.0:
            self.pygame.draw.polygon(self.canvas, core_color, points)
        else:
            tmp = self.pygame.Surface((self.width, self.height))
            tmp.fill((0, 0, 0))
            self.pygame.draw.polygon(tmp, core_color, points)
            tmp.set_alpha(int(255 * presence))
            self.canvas.blit(tmp, (0, 0), special_flags=self.pygame.BLEND_RGB_ADD)
