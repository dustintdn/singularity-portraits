"""Singularity Portraits — a live installation on identity, visibility, and light.

The package is organised as a one-directional pipeline:

    frame source -> detector -> identity registry -> seed -> visual params -> render

Each stage is deliberately ignorant of the stages around it. The renderer knows
nothing about face embeddings; the detector knows nothing about colour. That
separation is what lets Phase 2 (many faces, a better camera) be a scaling
exercise rather than a rewrite.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
