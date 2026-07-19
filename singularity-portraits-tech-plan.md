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

## 2. Phase 1 Scope (Single Face, Webcam) — ✅ IMPLEMENTED

> **Status:** Phase 1 is built and matches this section. The pipeline runs end-to-end
> (`face_recognition` HOG detector → 128-d embeddings → `IdentityRegistry` → seed/params →
> Pygame renderer → smoothed tracking), with a camera-free `SyntheticDetector` for headless/CI.
> The registry persists to JSON across sessions (opt-in via `--registry-path`). The code below
> is retained as the design record; the live implementation lives in the `singularity/` package
> (see §5). **Active work is now Phase 2 — see §4.**

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

## 4. Phase 2 — Scaling Architecture (multi-face, crowd-ready)

**Goal:** go from a handful of near-frontal faces to a room full of them (target hardware:
**Apple Silicon M3 Pro, 36 GB**), without a rewrite. The Phase 1 stage separation holds — the
render/tracking layers already generalize to N faces — so Phase 2 is concentrated in two places:
the **detection/embedding backend** and the **per-frame identity cost**.

### 4.0 What already generalizes (no work needed)

- **`IdentityRegistry`** resolves N embeddings to N stable ids with no changes — designed in from the start.
- **`TrackManager`** ([`visuals/tracking.py`](singularity/visuals/tracking.py)) already keeps a `Track` per
  identity, eases presence in/out, and prunes long-absent tracks. Re-entry across occlusion is handled by
  the registry's distance-threshold match; multi-face rendering is already a loop over visible tracks.
- Detection **already runs on its own thread** ([`app.py`](singularity/app.py) `_detect_loop`), decoupled from
  render. So a heavier detector lowers detection *cadence*, not frame rate — `TrackManager` interpolates between detections.

This means Phase 2 is a **scaling exercise, not new architecture** — exactly as intended.

### 4.1 Correcting the Phase 1 assumption about GPU acceleration

Section 2.1 and the old Phase 2 notes assumed `model="cnn"` (dlib) as the accuracy/scale upgrade.
**On Apple Silicon this is a dead end:** dlib has no Metal / Neural Engine backend, so `model="cnn"`
runs on CPU and is *slower* than HOG, not faster (`python -c "import dlib; print(dlib.DLIB_USE_CUDA)"`
prints `False` on an M3). The real fast path on M3 is **onnxruntime's CoreML execution provider**, which
offloads to the M3 GPU / Neural Engine — reachable via **InsightFace**, not dlib. This reframes the
"upgrade to InsightFace" note from *optional accuracy tweak* to *the actual scaling mechanism*.

### 4.2 Two development tracks

Track A is low-risk hardening on the current dlib backend (no new deps, ships value immediately).
Track B is the backend swap that unlocks crowd scale. They compose — Track A's optimizations apply to Track B too.

#### Track A — Cheap wins on the current backend (no new dependencies)

- **A1. Vectorize registry matching.** `resolve()` ([`registry.py`](singularity/identity/registry.py)) does a
  Python-level list comprehension of N `np.linalg.norm` calls **per face, per frame**. Store identity
  embeddings in one `(N, 128)` matrix and compute `np.linalg.norm(matrix - embedding, axis=1)` in a single
  C-level call. Covered by existing registry tests.
- **A2. Batch-resolve the frame.** `app.py` `_step_render` loops `registry.resolve()` per observation. Resolve
  all F observations against the matrix at once (`scipy.spatial.distance.cdist` or a broadcast). Turns F×N Python
  iterations into one vectorized op.
- **A3. Downscale before detection.** Add a `--detect-scale` flag; run `face_locations` on a shrunk frame and
  scale boxes back up. Detection cost is ~linear in pixels, so 0.5× ≈ 4× throughput at negligible cost to
  real (large) faces.
- **A4. Skip re-embedding tracked faces.** *(Highest-leverage optimization.)* Today every face is re-embedded
  every frame. Once a face is an established track, reuse its identity and only run embedding on **new /
  unmatched detections** (or on a low cadence, e.g. re-confirm every 5–10 frames). This decouples steady-state
  cost from face count and is the single change that most moves the ceiling — needs a lightweight
  position/IoU association between detections and existing tracks so a detection can be tied to a track without
  an embedding.
- **A5. Optional face cap.** A `--max-faces` guard that keeps the N largest boxes (by area) protects frame rate
  in an unexpectedly dense crowd. Off by default.

#### Track B — InsightFace + ONNX Runtime (CoreML) backend

Add a new detector class behind the **existing `Detector` protocol** ([`detector.py`](singularity/identity/detector.py)) —
no downstream changes to registry/seed/render beyond the embedding-dim/threshold retune below.

- **B1. New `InsightFaceDetector`.** Wrap InsightFace's `FaceAnalysis` (SCRFD detector + ArcFace embeddings) on
  `onnxruntime`, configured with `providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']` so it uses the
  M3 GPU/Neural Engine and falls back to CPU. SCRFD is single-shot: it finds 1 or 100 faces for ~the same cost,
  and handles small/oblique faces far better than HOG — the main win for a wide/overhead camera.
- **B2. Embedding dimension change: 128 → 512.** `embedding_dim` becomes 512 (ArcFace). Verify seed derivation
  (`embedding_to_seed`) is dimension-agnostic (it hashes quantized bytes, so it is). The same person gets a
  *different* singularity under the new model (different embedding space → different seed), but per the §4.4
  session-scoped decision this is invisible to visitors — the swap happens between sessions, and within a session
  one model is used consistently. **No seed-preservation work needed.**
- **B3. Threshold + metric retune.** The `0.6` threshold is a dlib-Euclidean number. ArcFace conventionally uses
  **cosine similarity** on normalized embeddings; either normalize + switch the registry metric to cosine, or
  re-derive an equivalent Euclidean threshold empirically. Must be tuned to avoid merging distinct people (the
  inter-identity separation is *better* with ArcFace, which helps).
- **B4. Registry compatibility guard (migration not required).** Per §4.4 the registry is session-scoped, so
  there is no long-lived file to migrate — each session starts fresh. The only work here is a **loud-fail guard**:
  if a persisted registry file is ever loaded whose embedding dimension / model doesn't match the active detector,
  raise rather than silently mismatch. Bump the schema `version` when the model changes so this check is trivial.
- **B5. Dependencies + install friction.** Adds `insightface` + `onnxruntime` (and drops the hard `dlib`
  requirement once B is the default). Document the CoreML provider setup; verify at startup which provider is
  actually active and log it (CoreML vs CPU fallback changes the performance story entirely).

### 4.3 Realistic capacity on M3 Pro / 36 GB

RAM is **not** the constraint — the SCRFD + ArcFace models are well under ~1 GB loaded. The limit is
**per-frame compute vs. target detection cadence.** Detection (SCRFD) is ~fixed per frame; embedding (ArcFace,
~one forward pass per face) is what scales linearly. Because detection runs async and `TrackManager` interpolates,
**8–15 Hz detection is plenty** for the installation. Working estimates (to be replaced by B6 benchmarks):

- **Naive (embed every face every frame):** ~**20–40 simultaneous faces** at a usable cadence.
- **With A4 (skip re-embedding tracked faces) + batched embeddings:** realistically **50–100 faces**,
  detection-bound rather than embedding-bound.

- **B6. Benchmark harness.** Before committing to numbers, add a standalone script that loads InsightFace with
  the CoreML provider and times SCRFD + batched ArcFace at 1 / 8 / 16 / 32 / 64 faces on the actual M3. Replaces
  all estimates above with measured figures and validates the CoreML provider is engaged.

### 4.4 Persistence horizon — DECIDED: session-scoped

**Decision:** the registry remembers people **within a single session only** and **resets between sessions.**
No long-term / cross-day registry. (This is the lighter surveillance stance and the simpler build.)

Consequences — this decision removes work rather than adding it:

- **Cross-session persistence is not needed.** The existing opt-in JSON `save`/`load` on `IdentityRegistry`
  ([`registry.py`](singularity/identity/registry.py)) is simply not used in the installation run — start each
  session with a fresh registry (no `--registry-path`). Persistence code can stay as-is (harmless, dev-only) or
  be dropped; it is not part of the Phase 2 critical path.
- **B2 (seed changes under the new model) stops being a visitor-facing problem.** Nobody is remembered across
  sessions, and the model swap happens *between* sessions, so no returning visitor ever sees their singularity
  change. Within a session, one model is used consistently → same face, same singularity, always. **Chosen path
  for B2: accept the reset; do not build seed-preservation.**
- **B4 (registry migration) collapses to almost nothing.** There is no long-lived registry file to migrate —
  each session is fresh. Keep only a loud-fail guard so an incompatible/old registry file can never be loaded
  silently under a different model; no migration path is required.
- **Ethical framing:** session-scoped memory means the piece recognizes you while you're in the room but forgets
  you when the session ends. Worth stating plainly in `README.md` / `decisions.md` as the deliberate stance.

### 4.5 Suggested Phase 2 build order

1. **A1 + A2** — vectorize/batch registry matching (safe, test-covered, immediate).
2. **A3** — `--detect-scale` downscaled detection.
3. **A4** — detection↔track association + skip re-embedding tracked faces (biggest ceiling move; do on dlib first, it carries to InsightFace).
4. **B6** — benchmark harness (measure before swapping).
5. **B1–B5** — InsightFace/CoreML backend, dim/threshold retune, registry compatibility guard.
6. **A5** — optional face cap as a safety valve for live installation.

Track A (1–3) is shippable on the current backend and de-risks Track B by proving the scaling wins independent of the model swap.

-----

## 5. Suggested Repo Structure

Actual current layout (Phase 1 implemented):

```
singularity-portraits/
├── main.py                  # CLI entry: arg parsing, builds source/detector/config, runs App
├── singularity/
│   ├── app.py               # App: capture loop, async detect thread, orchestrates pipeline
│   ├── sources.py           # frame sources (webcam / synthetic)
│   ├── types.py             # FaceObservation, AppConfig, shared dataclasses
│   ├── identity/
│   │   ├── detector.py      # Detector protocol; FaceRecognitionDetector, SyntheticDetector
│   │   │                    #   → Phase 2 B1: add InsightFaceDetector here (same protocol)
│   │   └── registry.py      # IdentityRegistry (resolve/persist) → Phase 2 A1/A2, B3/B4
│   └── visuals/
│       ├── seed.py          # embedding_to_seed, seed_to_visual_params
│       ├── color.py         # palette / colour helpers
│       ├── render.py        # renderer (draws N singularities)
│       └── tracking.py      # Track + TrackManager (per-identity presence/easing)
├── tests/
├── requirements.txt
└── README.md
```

-----

## 6. Dependencies

**Phase 1 (current — see `requirements.txt`):**

```
numpy<2
pygame
opencv-python-headless<4.11
dlib
face_recognition
```

Flag to the user: `face_recognition` requires `dlib`, which needs a C++ compiler and CMake available on the system to build — this is the most likely install friction point. If it fails, `pip install cmake` first, then retry, or fall back to a conda environment where dlib has prebuilt binaries.

**Phase 2 additions (Track B):**

```
insightface
onnxruntime           # CoreML execution provider ships in the standard wheel on macOS
```

Once InsightFace is the default detector, the hard `dlib` / `face_recognition` requirement can be
dropped (or kept as an optional fallback backend). Verify the CoreML provider is actually active at
startup — a silent CPU fallback changes the performance characteristics entirely (see §4.2 B5).
