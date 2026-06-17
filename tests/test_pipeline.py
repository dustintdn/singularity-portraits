"""Tests for the correctness-critical, deterministic half of the pipeline.

These cover the properties the whole concept rests on — "same face -> same
singularity" and "different faces -> different identities" — which are exactly
the parts that do not need a camera to verify.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from singularity.identity.detector import SyntheticDetector
from singularity.identity.registry import IdentityRegistry
from singularity.visuals.seed import (
    embedding_to_seed,
    embedding_to_visual_params,
    seed_to_visual_params,
)
from singularity.visuals.tracking import SmoothedPosition, TrackManager


# -- seed determinism ---------------------------------------------------------


def test_same_embedding_same_seed():
    emb = np.linspace(-1, 1, 128)
    assert embedding_to_seed(emb) == embedding_to_seed(emb.copy())


def test_quantisation_absorbs_subgrid_noise_away_from_boundaries():
    # When values sit mid-cell, sub-grid noise rounds away and the seed holds.
    # (On a cell boundary it can still flip — which is why the *real* no-flicker
    # guarantee is per-identity caching in the app, exercised below.)
    emb = np.full(128, 0.120)  # exactly mid-cell for the 2-decimal grid
    noisy = emb + np.full(128, 0.003)  # 0.123 -> still rounds to 0.12
    assert embedding_to_seed(emb) == embedding_to_seed(noisy)


def test_different_embeddings_differ():
    a = np.zeros(128)
    b = np.zeros(128)
    b[0] = 5.0
    assert embedding_to_seed(a) != embedding_to_seed(b)


def test_params_are_deterministic():
    p1 = seed_to_visual_params(123456789)
    p2 = seed_to_visual_params(123456789)
    assert p1 == p2


def test_params_within_bounds():
    for seed in range(50):
        p = seed_to_visual_params(seed * 99991)
        assert 0 <= p.hue_base < 360
        assert p.num_colors in (2, 3)
        assert len(p.palette) == p.num_colors
        assert 0.0 <= p.angularity <= 1.0
        assert p.motion_style in ("drift", "pulse", "orbit", "jitter")
        assert p.base_radius > 0


def test_embedding_to_params_roundtrip_matches_two_step():
    emb = np.linspace(-2, 2, 128)
    assert embedding_to_visual_params(emb) == seed_to_visual_params(embedding_to_seed(emb))


# -- identity registry --------------------------------------------------------


def test_same_face_resolves_to_same_id():
    reg = IdentityRegistry(threshold=0.6)
    emb = np.zeros(128)
    first = reg.resolve(emb)
    # A near-identical observation (camera noise) should land on the same id.
    again = reg.resolve(emb + np.full(128, 0.01))
    assert first == again
    assert len(reg) == 1


def test_distinct_faces_get_distinct_ids():
    reg = IdentityRegistry(threshold=0.6)
    a = np.zeros(128)
    b = np.zeros(128)
    b[:] = 3.0  # far apart in vector space
    assert reg.resolve(a) != reg.resolve(b)
    assert len(reg) == 2


def test_registry_persistence_roundtrip(tmp_path):
    reg = IdentityRegistry(threshold=0.6)
    a = np.zeros(128)
    b = np.full(128, 3.0)
    id_a = reg.resolve(a)
    id_b = reg.resolve(b)
    path = tmp_path / "registry.json"
    reg.save(path)

    reloaded = IdentityRegistry.load(path, threshold=0.6)
    assert len(reloaded) == 2
    # The same faces must still resolve to the same ids after a reload.
    assert reloaded.resolve(a) == id_a
    assert reloaded.resolve(b) == id_b


def test_load_missing_file_returns_empty(tmp_path):
    reg = IdentityRegistry.load(tmp_path / "nope.json")
    assert len(reg) == 0


# -- synthetic detector + end-to-end identity stability -----------------------


def test_app_caches_one_stable_singularity_per_identity():
    # The real "same face -> same singularity, every frame" guarantee: across a
    # run of noisy frames, each identity yields exactly one, unchanging params.
    from singularity.app import App, AppConfig
    from singularity.sources import SyntheticSource

    detector = SyntheticDetector(num_personas=3)
    source = SyntheticSource(width=640, height=360, num_frames=30)
    app = App(source, detector, AppConfig(width=640, height=360, headless=True, max_frames=30))

    seeds_seen = {}
    for frame in source:
        for obs in detector.detect(frame):
            ident = app.registry.resolve(obs.embedding)
            params = app.params_for(ident, obs.embedding)
            seeds_seen.setdefault(ident, params.seed)
            # Whatever the per-frame embedding, the cached seed never moves.
            assert params.seed == seeds_seen[ident]
    app.renderer.close()
    assert len(seeds_seen) == 3
    assert len(set(seeds_seen.values())) == 3  # three identities, three distinct forms


def test_synthetic_detector_stable_identities_over_time():
    detector = SyntheticDetector(num_personas=3)
    reg = IdentityRegistry(threshold=0.6)
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)

    seen_per_frame = []
    for _ in range(40):
        ids = {reg.resolve(o.embedding) for o in detector.detect(blank)}
        seen_per_frame.append(ids)

    # Exactly three identities ever appear, and all three are present each frame.
    assert len(reg) == 3
    for ids in seen_per_frame:
        assert ids == {0, 1, 2}


# -- tracking -----------------------------------------------------------------


def test_smoothed_position_takes_first_sample_raw():
    sp = SmoothedPosition(alpha=0.2)
    assert sp.update((100, 200)) == (100, 200)


def test_smoothed_position_eases_toward_target():
    sp = SmoothedPosition(alpha=0.5, start=(0, 0))
    x, y = sp.update((10, 0))
    assert 0 < x < 10  # moved partway, not all the way


def test_track_manager_fades_in_and_out():
    tm = TrackManager(fade_in_rate=0.5, fade_out_rate=0.5, max_misses=2)
    for _ in range(3):
        tm.begin_frame()
        tm.observe(0, (100, 100))
        visible = tm.end_frame()
    assert visible[0].presence == pytest.approx(1.0)

    # Stop observing: presence should decay and the track eventually drop.
    for _ in range(10):
        tm.begin_frame()
        visible = tm.end_frame()
    assert all(t.identity_id != 0 for t in visible)
