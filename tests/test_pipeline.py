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

from singularity.identity.association import associate_boxes_to_tracks, iou
from singularity.identity.detector import SyntheticDetector, _scale_boxes_up
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


def test_resolve_many_matches_per_face_resolve():
    # For well-separated faces, batch resolution must agree with sequential resolve,
    # frame by frame (including running-average updates and re-entry).
    rng = np.random.default_rng(0)
    faces = [rng.normal(0, 0.01, 128) + np.eye(128)[k] * 5.0 for k in range(4)]

    seq = IdentityRegistry(threshold=0.6)
    bat = IdentityRegistry(threshold=0.6)
    for _ in range(5):  # several frames, all four faces present each time
        seq_ids = [seq.resolve(f) for f in faces]
        bat_ids = bat.resolve_many(faces)
        assert bat_ids == seq_ids
    assert len(bat) == 4


def test_resolve_many_dedupes_new_siblings_within_a_frame():
    # Two identical brand-new faces in the same frame must collapse to one identity,
    # matching sequential resolve's behaviour.
    reg = IdentityRegistry(threshold=0.6)
    same = np.full(128, 2.0)
    ids = reg.resolve_many([same, same.copy()])
    assert ids[0] == ids[1]
    assert len(reg) == 1


def test_resolve_many_empty_returns_empty():
    reg = IdentityRegistry(threshold=0.6)
    assert reg.resolve_many([]) == []


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


# -- detection downscale (A3) -------------------------------------------------


def test_scale_boxes_up_recovers_full_resolution():
    # A box found on a half-size frame maps back to ~2x coordinates.
    boxes = [(50, 200, 150, 100)]  # top, right, bottom, left on the small frame
    out = _scale_boxes_up(boxes, inv_scale=2.0, height=720, width=1280)
    assert out == [(100, 400, 300, 200)]


def test_scale_boxes_up_clamps_to_frame_bounds():
    # A box whose scaled edges exceed the frame must clamp, never go out of bounds
    # (face_encodings requires in-bounds boxes).
    boxes = [(-5, 700, 400, 10)]
    out = _scale_boxes_up(boxes, inv_scale=2.0, height=720, width=1280)
    top, right, bottom, left = out[0]
    assert top == 0  # clamped up from -10
    assert right == 1280  # clamped down from 1400
    assert 0 <= left <= 1280 and 0 <= bottom <= 720


def test_detect_scale_out_of_range_rejected():
    pytest.importorskip("face_recognition")
    from singularity.identity.detector import FaceRecognitionDetector

    with pytest.raises(ValueError):
        FaceRecognitionDetector(detect_scale=0.0)
    with pytest.raises(ValueError):
        FaceRecognitionDetector(detect_scale=1.5)


# -- detection<->track association (A4) ---------------------------------------

# Boxes are (top, right, bottom, left).
_BOX_A = (0, 100, 100, 0)  # 100x100 at the origin


def test_iou_identical_and_disjoint():
    assert iou(_BOX_A, _BOX_A) == 1.0
    assert iou(_BOX_A, (0, 300, 100, 200)) == 0.0  # far to the right, no overlap


def test_iou_partial_overlap():
    # Shifted 50px right: intersection 50x100=5000, union 20000-5000=15000.
    shifted = (0, 150, 100, 50)
    assert iou(_BOX_A, shifted) == pytest.approx(5000 / 15000)


def test_associate_matches_by_overlap_and_flags_new_faces():
    boxes = [
        (0, 105, 100, 5),  # ~overlaps track 11
        (0, 305, 100, 205),  # ~overlaps track 22
        (500, 600, 600, 500),  # overlaps nobody -> new face
    ]
    track_boxes = [(11, (0, 100, 100, 0)), (22, (0, 300, 100, 200))]
    matches, unmatched = associate_boxes_to_tracks(boxes, track_boxes)
    assert matches == {0: 11, 1: 22}
    assert unmatched == [2]


def test_associate_uses_each_track_at_most_once():
    # Two boxes both overlap the single track; only the better one may claim it.
    boxes = [(0, 100, 100, 0), (0, 120, 100, 20)]
    track_boxes = [(7, (0, 100, 100, 0))]
    matches, unmatched = associate_boxes_to_tracks(boxes, track_boxes)
    assert list(matches.values()) == [7]
    assert len(matches) == 1
    assert len(unmatched) == 1  # the loser becomes a "new face"


def test_associate_no_tracks_means_all_new():
    boxes = [_BOX_A, (0, 300, 100, 200)]
    matches, unmatched = associate_boxes_to_tracks(boxes, [])
    assert matches == {}
    assert unmatched == [0, 1]


def test_associate_empty_boxes():
    matches, unmatched = associate_boxes_to_tracks([], [(1, _BOX_A)])
    assert matches == {}
    assert unmatched == []


def test_incremental_path_skips_reembedding_tracked_faces():
    # The core A4 promise: once a face is tracked, the next detection cycle reuses
    # its identity and does NOT call embed() again for it.
    import threading

    from singularity.app import App, AppConfig
    from singularity.sources import SyntheticSource

    box_a = (0, 100, 100, 0)
    box_b = (0, 300, 100, 200)

    class FakeIncrementalDetector:
        embedding_dim = 128
        incremental = True

        def __init__(self):
            self.embed_calls: list[list] = []
            self._emb = {box_a: np.eye(128)[0] * 5.0, box_b: np.eye(128)[1] * 5.0}

        def locate(self, frame):
            return [box_a, box_b]

        def embed(self, frame, boxes):
            boxes = list(boxes)
            self.embed_calls.append(boxes)
            return [self._emb[b] for b in boxes]

        def detect(self, frame):  # pragma: no cover - not used on the incremental path
            return []

    det = FakeIncrementalDetector()
    source = SyntheticSource(width=640, height=360, num_frames=3)
    app = App(source, det, AppConfig(width=640, height=360, headless=True))
    app._detect_lock = threading.Lock()
    app._track_boxes_snapshot = []
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Cycle 1: nothing tracked yet -> both faces embedded.
    obs1 = app._detect_incremental(frame, app._track_boxes_snapshot)
    assert [len(c) for c in det.embed_calls] == [2]
    app._latest_observations = obs1
    app._step_render(0.0)
    assert len(app._track_boxes_snapshot) == 2  # both identities now have boxes

    # Cycle 2: same boxes overlap the two tracks -> embed() gets ZERO boxes.
    obs2 = app._detect_incremental(frame, app._track_boxes_snapshot)
    assert det.embed_calls[-1] == []  # nothing re-embedded
    assert all(o.identity_id is not None for o in obs2)  # identities reused
    assert all(o.embedding is None for o in obs2)
    app._latest_observations = obs2
    app._step_render(0.1)  # must not choke on None embeddings

    app.renderer.close()


def test_synthetic_detector_stable_identities_over_time():
    detector = SyntheticDetector(num_personas=3)
    reg = IdentityRegistry(threshold=0.6)
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)

    seen_per_frame = []
    for _ in range(40):
        ids = {reg.resolve(o.embedding) for o in detector.detect(blank)}
        seen_per_frame.append(ids)

    # Exactly three identities ever appear, and all three are present every frame.
    # Ids are hash-derived (not sequential), so assert on the structural property
    # rather than literal id values: the same three ids recur in every frame.
    assert len(reg) == 3
    expected_ids = seen_per_frame[0]
    assert len(expected_ids) == 3
    for ids in seen_per_frame:
        assert ids == expected_ids


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
