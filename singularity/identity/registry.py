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

    def resolve(self, embedding: np.ndarray) -> int:
        """Return the identity id for ``embedding``, registering if new."""

        embedding = np.asarray(embedding, dtype=np.float64)
        if not self.identities:
            return self._register(embedding)

        distances = [np.linalg.norm(embedding - e["embedding"]) for e in self.identities]
        best_idx = int(np.argmin(distances))
        if distances[best_idx] < self.threshold:
            existing = self.identities[best_idx]
            r = self.update_rate
            existing["embedding"] = (1 - r) * existing["embedding"] + r * embedding
            existing["count"] += 1
            return existing["id"]

        return self._register(embedding)

    def _register(self, embedding: np.ndarray) -> int:
        new_id = self.next_id
        self.identities.append({"id": new_id, "embedding": embedding, "count": 1})
        self.next_id += 1
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
        return registry
