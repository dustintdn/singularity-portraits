"""Associate this frame's detection boxes with already-tracked identities.

The expensive part of the real detector is computing an embedding per face. Once
a face is being tracked, we don't need to re-embed it every frame — if a freshly
detected box overlaps a known track's last-known box enough, it's almost certainly
the same person, and we can reuse their identity and skip the embedding (§4.2 A4).

This module holds the pure geometry that decides "same face as an existing track?"
— deliberately free of any detector, thread, or registry state so it can be tested
without a camera. The wiring that acts on these decisions lives in the app loop.
"""

from __future__ import annotations

Box = tuple[int, int, int, int]  # (top, right, bottom, left), face_recognition order


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(top, right, bottom, left)`` boxes.

    Returns 0.0 for non-overlapping or degenerate boxes, 1.0 for identical ones.
    """

    at, ar, ab, al = a
    bt, br, bb, bl = b

    inter_w = max(0, min(ar, br) - max(al, bl))
    inter_h = max(0, min(ab, bb) - max(at, bt))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0, ar - al) * max(0, ab - at)
    area_b = max(0, br - bl) * max(0, bb - bt)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def associate_boxes_to_tracks(
    boxes: list[Box],
    track_boxes: list[tuple[int, Box]],
    iou_threshold: float = 0.3,
) -> tuple[dict[int, int], list[int]]:
    """Greedily match detection boxes to existing tracks by IoU.

    Parameters
    ----------
    boxes:
        This frame's detected boxes.
    track_boxes:
        ``(identity_id, last_box)`` for each currently active track.
    iou_threshold:
        Minimum overlap to count a box and a track as the same face.

    Returns
    -------
    (matches, unmatched)
        ``matches`` maps a box index to the identity id it was tied to; each track
        is used at most once. ``unmatched`` lists the indices of boxes that matched
        no track — these are the faces that still need embedding + resolution.

    Matching is greedy on descending IoU, which is stable and good enough here:
    boxes are sparse (a handful of faces) and well separated, so the rare ambiguous
    overlap doesn't warrant a full assignment solver.
    """

    candidates: list[tuple[float, int, int]] = []
    for bi, box in enumerate(boxes):
        for identity_id, tbox in track_boxes:
            score = iou(box, tbox)
            if score >= iou_threshold:
                candidates.append((score, bi, identity_id))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matches: dict[int, int] = {}
    used_tracks: set[int] = set()
    for _score, bi, identity_id in candidates:
        if bi in matches or identity_id in used_tracks:
            continue
        matches[bi] = identity_id
        used_tracks.add(identity_id)

    unmatched = [bi for bi in range(len(boxes)) if bi not in matches]
    return matches, unmatched
