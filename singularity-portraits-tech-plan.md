# Singularity Portraits — Technical Implementation Plan

Target audience: Claude Code, implementing in Python.
Goal: working webcam prototype first (Phase 1), architected so Phase 2 (multi-face, nicer camera) is a scaling exercise, not a rewrite.

-----

## 1. High-Level Pipeline

```
Webcam frame
  -> Face detection (find bounding box / landmarks)
  -> Face embedding (convert face to a fixed-length vector — the "identity vector")
  -> Identity resolution (match embedding against known identities, or register new one)
  -> Seed derivation (turn identity vector into deterministic visual parameters)
  -> Visual parameter mapping (seed -> color palette, shape, motion behavior)
  -> Render (draw the singularity, update its position to track the face)
  -> Display (fullscreen visual output, likely separate from any debug/camera view)
```

Each stage should be its own module. The render/display stage should know nothing about embeddings; the embedding stage should know nothing about rendering. This separation matters a lot once you go from 1 face to N faces in Phase 2.

-----

## 2. Phase 1 Scope (Single Face, Webcam)

### 2.1 Face detection + embedding

Use a face recognition library that gives you both detection and an embedding in one go, since you confirmed embedding-based seeds (more stable across lighting/angle than landmark-distance ratios).

**Recommended: `face_recognition`** (built on dlib) — simplest API, well-documented, produces a 128-dimension embedding per face out of the box.

```python
import face_recognition

frame = ...  # numpy array, RGB
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)
# face_encodings[i] is a 128-d numpy float vector — this is your identity vector
```

Alternative if you want higher accuracy / more modern embeddings later: **InsightFace** (ArcFace embeddings, 512-d, much better separation between distinct identities, GPU-friendly). Worth swapping in for Phase 2 if `face_recognition` proves too noisy with multiple faces or non-frontal angles. Don’t start here — start simple, swap later if needed.

Install:

```bash
pip install face_recognition opencv-python --break-system-packages
```

(`face_recognition` depends on `dlib`, which needs CMake + a C++ compiler on the system — flag this to the user if the install fails; on some systems `pip install dlib` needs `cmake` installed first via the OS package manager.)

### 2.2 Identity resolution & stability

This is the part that makes “same face -> same singularity, every time” actually work. Embeddings for the *same* person are never bit-identical across frames (lighting, angle, expression all introduce noise) — they’re just *close* in vector space. So:

1. Maintain an in-memory (then later, persisted) registry: a list of `{identity_id, embedding}` pairs.
1. For each detected face this frame, compute its embedding, then compare it (Euclidean distance) against all known embeddings in the registry.
1. If the closest match is under a threshold (a good starting point with `face_recognition` is `0.6`, tune empirically), it’s the same person — reuse that `identity_id`.
1. If no match is close enough, register a new identity.
1. Optionally: maintain a running average embedding per identity (update it slightly each time you see that person again) so the “canonical” embedding for each identity stabilizes and drifts less over time.

```python
import numpy as np

class IdentityRegistry:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.identities = []  # list of dicts: {"id": int, "embedding": np.array}
        self.next_id = 0

    def resolve(self, embedding):
        if not self.identities:
            return self._register(embedding)
        distances = [np.linalg.norm(embedding - e["embedding"]) for e in self.identities]
        best_idx = int(np.argmin(distances))
        if distances[best_idx] < self.threshold:
            # running average update, keeps things stable but adaptive
            existing = self.identities[best_idx]
            existing["embedding"] = 0.9 * existing["embedding"] + 0.1 * embedding
            return existing["id"]
        return self._register(embedding)

    def _register(self, embedding):
        new_id = self.next_id
        self.identities.append({"id": new_id, "embedding": embedding})
        self.next_id += 1
        return new_id
```

For persistence **across sessions** (same person, different day, still gets their singularity) — persist `self.identities` to disk (pickle, or json with the embedding as a list) on shutdown, reload on startup. Flag clearly in code/UI that this means the system is storing biometric data between runs; that’s a meaningful design decision worth the user being deliberate about, even just for an art piece.

### 2.3 Seed derivation: embedding -> deterministic visual seed

Once you have a stable identity (and ideally its averaged embedding), reduce the 128-d vector to a small set of deterministic numbers that drive the visuals. The key property: **same embedding -> same seed, always** (no randomness here — randomness belongs in the *behavior* of the singularity, not in *which* singularity it is).

```python
import hashlib

def embedding_to_seed(embedding: np.ndarray) -> int:
    # Quantize to stabilize against tiny float noise, then hash deterministically
    quantized = np.round(embedding, decimals=2)
    byte_repr = quantized.tobytes()
    digest = hashlib.sha256(byte_repr).hexdigest()
    return int(digest[:16], 16)  # 64-bit int seed
```

From this single integer seed, derive all visual parameters using a seeded random generator (`random.Random(seed)` or `np.random.default_rng(seed)`) so the *mapping* from seed to params is reproducible, but you still get organic-feeling variety across different seeds.

```python
import random

def seed_to_visual_params(seed: int) -> dict:
    rng = random.Random(seed)
    hue_base = rng.uniform(0, 360)
    return {
        "hue_base": hue_base,
        "hue_spread": rng.uniform(15, 60),       # how far the 2-3 colors spread from hue_base
        "num_colors": rng.choice([2, 3]),
        "angularity": rng.uniform(0.0, 1.0),     # 0 = smooth blob, 1 = jagged/angular
        "pulse_speed": rng.uniform(0.5, 3.0),
        "drift_speed": rng.uniform(0.2, 1.5),
        "noise_octaves": rng.randint(1, 4),      # for organic surface distortion
        "base_radius": rng.uniform(40, 90),
    }
```

This is the single most important design surface in the whole project — it’s where “face structure” becomes “visual personality.” Expect to iterate on this function a lot once you can see real output; treat the first version as a placeholder to get the pipeline running end-to-end, not a final design.

### 2.4 Rendering

Two reasonable paths depending on how much visual sophistication you want in Phase 1:

**Option A — fast prototype: Pygame.** Easiest to get a moving, colored, pulsing blob on screen quickly. Good for validating the pipeline (detection -> seed -> visual -> tracking) before investing in nicer visuals.

**Option B — nicer visuals sooner: Processing-style shaders / OpenGL via `vispy` or a Pygame + custom GLSL combo, or just push frames out of Python into a TouchDesigner / Unity front-end via OSC.** Recommended path if visual quality matters a lot for an “art installation” feel — keep Python responsible for detection/identity/seed, and hand off *only* the visual parameters (color, shape params, position) to a dedicated real-time visual engine over OSC or a local WebSocket. This also decouples “how good the visuals look” from “how good the face tracking is,” letting you improve either independently.

Given you’re a data scientist comfortable in Python, and this is explicitly Phase 1 / proof-of-concept, **start with Option A (Pygame)** to validate the whole pipeline cheaply, then consider migrating the render layer to TouchDesigner/Unity/shader-based rendering once the concept is validated and you want production-quality visuals for Phase 2.

```python
import pygame

def draw_singularity(surface, position, params, t):
    x, y = position
    pulse = 1 + 0.15 * np.sin(t * params["pulse_speed"])
    radius = params["base_radius"] * pulse
    for i in range(params["num_colors"]):
        hue = (params["hue_base"] + i * params["hue_spread"]) % 360
        color = hsv_to_rgb(hue, 0.8, 1.0)
        layer_radius = radius * (1 - i * 0.25)
        pygame.draw.circle(surface, color, (int(x), int(y)), int(layer_radius))
```

(`angularity` and `noise_octaves` would drive a more advanced shape — e.g., a deformed polygon with vertex noise instead of a perfect circle — that’s a refinement once the basic version works.)

### 2.5 Tracking (position smoothing)

Face bounding boxes jitter frame to frame. Smooth the position so the singularity glides rather than jumps:

```python
class SmoothedPosition:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.pos = None

    def update(self, new_pos):
        if self.pos is None:
            self.pos = new_pos
        else:
            self.pos = (
                self.alpha * new_pos[0] + (1 - self.alpha) * self.pos[0],
                self.alpha * new_pos[1] + (1 - self.alpha) * self.pos[1],
            )
        return self.pos
```

-----

## 3. Phase 1 Build Order (suggested milestones for Claude Code)

1. **Webcam capture loop** — OpenCV `VideoCapture(0)`, display raw feed, confirm camera works.
1. **Face detection only** — draw a bounding box around detected face(s) each frame, no embeddings yet.
1. **Embedding extraction** — print the embedding vector to console, confirm it’s stable-ish across frames for the same person (sanity check before building identity logic on top).
1. **Identity registry** — confirm that walking away and back resolves to the *same* `identity_id`, and that a second person gets a *different* `identity_id`.
1. **Seed derivation** — confirm the same `identity_id` always produces the same seed and same visual params dict.
1. **Static render** — draw a static (non-moving, non-pulsing) colored circle using the derived params, positioned at face center.
1. **Add motion/behavior** — pulsing, drift, angularity-driven shape distortion.
1. **Add tracking smoothing** — singularity glides with the face instead of jumping.
1. **Polish loop** — fullscreen output mode, hide/minimize debug overlays, tune visual param ranges based on what actually looks good.

Steps 1-5 are about correctness (does identity work at all); 6-9 are about feel (does it look like art). Don’t skip ahead to 6-9 with a broken or unstable identity pipeline underneath — you’ll end up debugging visuals when the actual bug is in identity resolution.

-----

## 4. Phase 2 Considerations (multi-face, scaling up)

Not needed for the first build, but worth architecting Phase 1 with these in mind so it’s not a rewrite:

- **Multiple simultaneous identities**: the `IdentityRegistry` and seed/params derivation already generalize to N faces with no changes — this was designed in from the start. The main new work is in rendering N singularities at once and giving each its own `SmoothedPosition` tracker, keyed by `identity_id`.
- **Better camera**: if moving to a wide-angle or overhead camera, re-validate detection accuracy at the new distances/angles — `face_recognition`’s default detector is HOG-based and works best on relatively close, front-facing faces. For wider rooms or more oblique angles, switching the detector to a CNN-based model (`face_recognition` supports this via `model="cnn"`, or move to InsightFace/MediaPipe) will likely be necessary, ideally GPU-accelerated for real-time performance with multiple faces.
- **Face re-entry across occlusion/crowding**: with multiple people, faces will temporarily occlude each other or leave frame and re-enter. The distance-threshold matching in `IdentityRegistry` already handles re-entry; just confirm the threshold doesn’t cause two different people to merge into one identity (the embedding model’s separation quality matters a lot here — this is the main reason to consider upgrading to ArcFace/InsightFace embeddings for Phase 2, since they have better inter-identity separation than `face_recognition`’s default model).
- **Performance**: face detection + embedding per frame for N faces gets expensive. Options: run detection on a downscaled frame, run embedding extraction at a lower frequency than rendering (e.g., re-confirm identity every 5-10 frames rather than every frame, while position tracking updates every frame from a lighter-weight detector).
- **Persistence across sessions/days**: decide explicitly whether the identity registry should reset per session (each “showing” of the piece starts fresh) or persist indefinitely (the installation “remembers” repeat visitors across days/weeks). This is a conceptual decision as much as a technical one, given the surveillance themes — worth deciding deliberately rather than defaulting into it.

-----

## 5. Suggested Repo Structure

```
singularity-portraits/
├── main.py                  # capture loop, orchestrates everything
├── identity/
│   ├── detector.py          # face detection + embedding extraction
│   └── registry.py          # IdentityRegistry class
├── visuals/
│   ├── seed.py              # embedding_to_seed, seed_to_visual_params
│   ├── render.py            # drawing/rendering logic
│   └── tracking.py          # SmoothedPosition / motion smoothing
├── requirements.txt
└── README.md
```

-----

## 6. Dependencies (Phase 1)

```
opencv-python
face_recognition
numpy
pygame
```

Flag to the user: `face_recognition` requires `dlib`, which needs a C++ compiler and CMake available on the system to build — this is the most likely install friction point. If it fails, `pip install cmake` first, then retry, or fall back to a conda environment where dlib has prebuilt binaries.
