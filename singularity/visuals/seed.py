"""Embedding -> deterministic visual fingerprint.

This is the single most important design surface in the project: it is where
"the structure of your face" becomes "the structure of your light". Two
properties matter above all else and are tested in ``tests/``:

1. **Determinism.** The same embedding must always produce the same seed and the
   same :class:`VisualParams`. No wall-clock, no unseeded RNG, no set ordering.
2. **Stability.** Embeddings for one person wobble frame to frame. The
   quantisation step below *reduces* the seed's sensitivity to that wobble, but
   cannot eliminate it: any fixed grid has boundaries, and a value sitting on
   one will still flip when noise nudges it across. The actual no-flicker
   guarantee comes from upstream — the app computes a face's seed once, from the
   first (or running-averaged canonical) embedding the registry resolves, and
   caches it per identity. So a given identity's singularity never changes even
   though its raw per-frame embedding keeps jiggling. Quantisation just buys a
   little extra robustness for that first read.

Randomness still has a place — but only in the *behaviour* of a singularity
(how it drifts and pulses over time), never in *which* singularity a face gets.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from ..types import VisualParams
from .color import hsv_to_rgb

# Decimal places kept when quantising an embedding before hashing. Coarser =
# more stable against per-frame noise but more likely to collide two genuinely
# different faces. Two places is a good starting point for face_recognition's
# 128-d vectors; revisit if the registry's running-average smoothing is changed.
_QUANTIZE_DECIMALS = 2

# Motion archetypes named in the concept doc. The seed picks one; the renderer
# interprets it. Kept as plain strings so the renderer and tests share a
# vocabulary without importing each other.
MOTION_STYLES = ("drift", "pulse", "orbit", "jitter")


def embedding_to_seed(embedding: np.ndarray) -> int:
    """Reduce an identity vector to a single reproducible 64-bit integer.

    The embedding is quantised (to swallow floating-point noise), serialised in
    a fixed byte order, and hashed with SHA-256. Hashing rather than, say,
    summing means visually adjacent embeddings still scatter to unrelated seeds,
    so two similar-but-distinct faces get visibly different singularities.
    """

    quantized = np.round(np.asarray(embedding, dtype=np.float64), decimals=_QUANTIZE_DECIMALS)
    # ``+ 0.0`` normalises a possible negative zero so ``-0.0`` and ``0.0`` hash
    # alike; ``ascontiguousarray`` fixes the byte layout regardless of slicing.
    byte_repr = np.ascontiguousarray(quantized + 0.0).tobytes()
    digest = hashlib.sha256(byte_repr).hexdigest()
    return int(digest[:16], 16)


def seed_to_visual_params(seed: int) -> VisualParams:
    """Expand a seed into the full deterministic visual fingerprint.

    Uses ``random.Random(seed)`` so the mapping is reproducible across machines
    and Python versions (unlike ``hash()``), while still feeling organic: small
    changes in seed produce unrelated-looking singularities.
    """

    rng = random.Random(seed)

    hue_base = rng.uniform(0, 360)
    hue_spread = rng.uniform(15, 70)
    num_colors = rng.choice([2, 2, 3])  # bias toward two-colour palettes
    saturation = rng.uniform(0.55, 0.95)
    angularity = rng.uniform(0.0, 1.0)
    pulse_speed = rng.uniform(0.5, 3.0)
    pulse_depth = rng.uniform(0.08, 0.28)
    drift_speed = rng.uniform(0.2, 1.5)
    motion_style = rng.choice(MOTION_STYLES)
    noise_octaves = rng.randint(1, 4)
    noise_amplitude = rng.uniform(0.05, 0.35) * angularity
    base_radius = rng.uniform(46, 96)
    rotation_speed = rng.uniform(-0.8, 0.8)

    palette = _build_palette(hue_base, hue_spread, num_colors, saturation)

    return VisualParams(
        seed=seed,
        hue_base=hue_base,
        hue_spread=hue_spread,
        num_colors=num_colors,
        saturation=saturation,
        angularity=angularity,
        pulse_speed=pulse_speed,
        pulse_depth=pulse_depth,
        drift_speed=drift_speed,
        motion_style=motion_style,
        noise_octaves=noise_octaves,
        noise_amplitude=noise_amplitude,
        base_radius=base_radius,
        rotation_speed=rotation_speed,
        palette=palette,
    )


def embedding_to_visual_params(embedding: np.ndarray) -> VisualParams:
    """Convenience: the whole embedding -> params hop in one call."""

    return seed_to_visual_params(embedding_to_seed(embedding))


def _build_palette(
    hue_base: float, hue_spread: float, num_colors: int, saturation: float
) -> list[tuple[int, int, int]]:
    """Two or three hues fanned out from ``hue_base``, brightest at the core."""

    palette: list[tuple[int, int, int]] = []
    for i in range(num_colors):
        hue = hue_base + i * hue_spread
        value = 1.0 - i * 0.18  # outer layers a touch darker, for depth
        palette.append(hsv_to_rgb(hue, saturation, value))
    return palette
