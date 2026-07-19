"""Identity resolution: the thing that makes "same face -> same singularity".

Embeddings for one person are never bit-identical across frames — they are only
*close* in vector space. So we keep a registry of known identities and match
each new embedding to the nearest one within a distance threshold, registering a
new identity only when nothing is close enough.

A running-average update nudges each identity's stored embedding toward what we
keep seeing, so its "canonical" vector stabilises rather than chasing the most
recent noisy frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class IdentityRegistry:
    """Resolve embeddings to stable integer identity ids.

    Parameters
    ----------
    threshold:
        Maximum Euclidean distance for a match. ``0.6`` is a sensible default
        for ``face_recognition``'s 128-d embeddings; tune empirically and lower
        it if two different people ever merge into one identity.
    update_rate:
        Weight given to a new observation when updating an identity's canonical
        embedding. ``0.1`` keeps things stable but slowly adaptive.
    """

    def __init__(self, threshold: float = 0.6, update_rate: float = 0.1):
        self.threshold = threshold
        self.update_rate = update_rate
        self.identities: list[dict] = []  # {"id": int, "embedding": np.ndarray, "count": int}
        self.next_id = 0
        # Distance-search cache: an (N, dim) matrix whose row i mirrors
        # ``identities[i]["embedding"]``. Kept in sync on register / running-average
        # update so ``resolve`` is a single vectorised norm instead of a Python loop
        # over N. ``None`` when empty. Rebuilt wholesale by ``_rebuild_matrix`` (used
        # after ``load``); mutated incrementally elsewhere.
        self._matrix: np.ndarray | None = None

    def _rebuild_matrix(self) -> None:
        """Rebuild ``_matrix`` from ``identities`` (single source of truth)."""

        if self.identities:
            self._matrix = np.array(
                [e["embedding"] for e in self.identities], dtype=np.float64
            )
        else:
            self._matrix = None

    def resolve(self, embedding: np.ndarray) -> int:
        """Return the identity id for ``embedding``, registering if new."""

        embedding = np.asarray(embedding, dtype=np.float64)
        if not self.identities:
            return self._register(embedding)

        distances = np.linalg.norm(self._matrix - embedding, axis=1)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] < self.threshold:
            existing = self.identities[best_idx]
            r = self.update_rate
            existing["embedding"] = (1 - r) * existing["embedding"] + r * embedding
            self._matrix[best_idx] = existing["embedding"]
            existing["count"] += 1
            return existing["id"]

        return self._register(embedding)

    def resolve_many(self, embeddings) -> list[int]:
        """Resolve a whole frame's embeddings at once.

        Equivalent to calling :meth:`resolve` on each embedding in order, but the
        dominant work — the distance of every face to every known identity — is
        done as a single ``(F, N)`` matrix op instead of F separate passes.

        Sequential semantics are preserved where they matter: a face that matches
        no *existing* identity falls back to :meth:`resolve`, so it can still match
        (and thus dedupe against) a sibling registered earlier in the same frame.
        Faces that match an existing identity use the batched distances directly.
        """

        if len(embeddings) == 0:
            return []
        embs = np.asarray(embeddings, dtype=np.float64)
        if embs.ndim == 1:
            embs = embs[None, :]
        if not self.identities:
            return [self.resolve(e) for e in embs]

        # (F, N) Euclidean distances via |a|^2 + |b|^2 - 2 a.b, clamped for the
        # tiny negatives floating-point can produce on near-identical vectors.
        snap = self._matrix
        a2 = np.einsum("fd,fd->f", embs, embs)[:, None]
        b2 = np.einsum("nd,nd->n", snap, snap)[None, :]
        dmat = np.sqrt(np.maximum(a2 + b2 - 2.0 * (embs @ snap.T), 0.0))

        ids: list[int] = []
        for i in range(embs.shape[0]):
            best_idx = int(np.argmin(dmat[i]))
            if dmat[i, best_idx] < self.threshold:
                existing = self.identities[best_idx]
                r = self.update_rate
                existing["embedding"] = (1 - r) * existing["embedding"] + r * embs[i]
                self._matrix[best_idx] = existing["embedding"]
                existing["count"] += 1
                ids.append(existing["id"])
            else:
                # No match among the frame-start identities; defer to resolve so a
                # just-registered sibling from this same frame can still match.
                ids.append(self.resolve(embs[i]))
        return ids

    def _register(self, embedding: np.ndarray) -> int:
        digest = hashlib.sha256(np.ascontiguousarray(embedding).tobytes()).hexdigest()
        new_id = int(digest[:8], 16)
        self.identities.append({"id": new_id, "embedding": embedding, "count": 1})
        row = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        self._matrix = row if self._matrix is None else np.vstack([self._matrix, row])
        return new_id

    def __len__(self) -> int:
        return len(self.identities)

    # -- Cross-session persistence --------------------------------------------
    # This stores biometric data (face embeddings) to disk between runs. That is
    # a deliberate, ethically loaded choice for this piece — see decisions.md and
    # the consent note in the README. It is opt-in: the app only persists when a
    # path is supplied on the command line.

    def save(self, path: str | Path) -> None:
        """Serialise the registry to JSON (embeddings stored as plain lists)."""

        path = Path(path)
        payload = {
            "version": 1,
            "next_id": self.next_id,
            "threshold": self.threshold,
            "identities": [
                {"id": e["id"], "embedding": e["embedding"].tolist(), "count": e["count"]}
                for e in self.identities
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "IdentityRegistry":
        """Load a registry from JSON, or return a fresh one if the file is absent."""

        path = Path(path)
        registry = cls(**kwargs)
        if not path.exists():
            return registry

        payload = json.loads(path.read_text())
        registry.next_id = payload.get("next_id", 0)
        registry.identities = [
            {
                "id": e["id"],
                "embedding": np.asarray(e["embedding"], dtype=np.float64),
                "count": e.get("count", 1),
            }
            for e in payload.get("identities", [])
        ]
        if registry.identities and registry.next_id <= max(e["id"] for e in registry.identities):
            registry.next_id = max(e["id"] for e in registry.identities) + 1
        registry._rebuild_matrix()
        return registry
