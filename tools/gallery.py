"""Render a contact sheet of singularities for many synthetic identities.

This is the fastest way to judge the most important design surface in the
project — "does embedding -> visual produce inevitable-feeling variety, or
arbitrary noise?" — without a camera. It draws one singularity per cell on a
grid and writes a single PNG.

    python tools/gallery.py --count 12 --out assets/gallery.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singularity.identity.detector import SyntheticDetector  # noqa: E402
from singularity.visuals.render import SingularityRenderer  # noqa: E402
from singularity.visuals.seed import embedding_to_visual_params  # noqa: E402
from singularity.visuals.tracking import Track, SmoothedPosition  # noqa: E402


def make_identities(count: int) -> list:
    """Distinct, deterministic embeddings standing in for distinct faces."""

    detector = SyntheticDetector(num_personas=count)
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    return [obs.embedding for obs in detector.detect(blank)]


def render_gallery(count: int, cell: int, cols: int, t: float, out: str) -> str:
    rows = (count + cols - 1) // cols
    width, height = cols * cell, rows * cell
    renderer = SingularityRenderer(width=width, height=height, headless=True, trail=1.0)
    renderer.begin()

    for i, embedding in enumerate(make_identities(count)):
        params = embedding_to_visual_params(embedding)
        cx = (i % cols) * cell + cell // 2
        cy = (i // cols) * cell + cell // 2
        track = Track(
            identity_id=i,
            position=SmoothedPosition(start=(cx, cy)),
            presence=1.0,
        )
        # Scale each form to sit comfortably inside its cell.
        params.base_radius = min(params.base_radius, cell * 0.28)
        renderer.draw(track, params, t)

    renderer.save(out)
    renderer.close()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Render a singularity contact sheet.")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--cell", type=int, default=320)
    p.add_argument("--t", type=float, default=2.0, help="Animation time (seconds) to freeze at.")
    p.add_argument("--out", default="assets/gallery.png")
    args = p.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    path = render_gallery(args.count, args.cell, args.cols, args.t, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
