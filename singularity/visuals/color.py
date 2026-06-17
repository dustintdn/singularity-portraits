"""Colour helpers.

The pipeline thinks in HSV (it is the natural space for "spread two or three
hues a little apart") but draws in RGB, so the conversions live here in one
place rather than being re-derived at every call site.
"""

from __future__ import annotations

import colorsys


def hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    """Convert HSV to an 8-bit RGB tuple.

    ``hue`` is given in degrees (``0-360``) because that is how the visual
    parameters describe it; ``saturation`` and ``value`` are ``0-1``.
    """

    r, g, b = colorsys.hsv_to_rgb((hue % 360) / 360.0, _clamp01(saturation), _clamp01(value))
    return (round(r * 255), round(g * 255), round(b * 255))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
