# Singularity Portraits

*A live installation on identity, visibility, and light.*

A camera watches a room. For every face it finds, it reads the underlying
geometry of that face, turns it into a stable identity signature, and renders a
small, self-contained **singularity** — a glowing, moving ball of light that
belongs to that face and no other. The same face always summons the same
singularity; a different face summons a different one. No names, no labels, no
match scores — just light that is unmistakably, persistently *yours*.

See [`singularity-portraits-description.md`](singularity-portraits-description.md)
for the concept and [`singularity-portraits-tech-plan.md`](singularity-portraits-tech-plan.md)
for the technical plan this implements. Autonomous choices made during the build
are logged in [`decisions.md`](decisions.md).

---

## What's here (Phase 1)

A complete, modular pipeline:

```
frame source -> detector (face -> embedding) -> identity registry
             -> seed -> visual params -> tracking -> render
```

Each stage is its own module and ignorant of the others, so going from one face
(Phase 1) to a roomful (Phase 2) is a scaling exercise, not a rewrite — the loop
already handles "for each face in the frame."

| Stage | Module |
|-------|--------|
| Frame sources (webcam / video / images / synthetic) | `singularity/sources.py` |
| Detection + embedding (real + synthetic backends) | `singularity/identity/detector.py` |
| Identity resolution + persistence | `singularity/identity/registry.py` |
| Embedding → deterministic visual fingerprint | `singularity/visuals/seed.py` |
| Rendering (additive glow, trails, headless-capable) | `singularity/visuals/render.py` |
| Motion smoothing + per-identity lifecycle | `singularity/visuals/tracking.py` |
| Orchestration loop | `singularity/app.py`, `main.py` |

### Preview the output without a camera

These two artifacts are committed so you can see what it does at a glance:

- **`assets/gallery.png`** — twelve distinct identities, one singularity each.
- **`assets/walkthrough.mp4`** — three synthetic people moving through a room,
  each trailing their own light.

![gallery of singularities](assets/gallery.png)

---

## Running it

```bash
pip install -r requirements.txt
```

**Camera-free demo (works anywhere, no camera or display needed):**

```bash
# Render a contact sheet of distinct singularities
python tools/gallery.py --count 12 --out assets/gallery.png

# Record a moving multi-face walkthrough to MP4
python main.py --source synthetic --headless --record assets/walkthrough.mp4 \
    --max-frames 240 --width 960 --height 540
```

**The real thing (Phase 1 proof of concept — needs a webcam + `face_recognition`):**

```bash
python main.py --source webcam            # one face in, one singularity out
```

Other sources:

```bash
python main.py --source video  --video clip.mp4 --loop
python main.py --source images --images-dir ./faces
```

Useful flags: `--headless` (no window), `--record out.mp4`, `--max-frames N`,
`--threshold 0.6` (identity match distance), `--personas 3` (synthetic only),
`--fr-model {hog,cnn}`. Run `python main.py --help` for the full list.

### Persistence & a note on biometric data

By default the identity registry lives **in memory only** and is gone when the
process exits. Passing `--registry path.json` makes the installation *remember
faces across runs* — which means writing biometric embeddings to disk. That is a
deliberate, ethically loaded choice for a piece about surveillance, so it is
opt-in rather than the default. Decide it on purpose.

---

## Tests

The correctness-critical, deterministic half of the pipeline — "same face → same
singularity", "different faces → different identities", persistence, tracking —
is covered and runs with no camera and no display:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

---

## Installing the real face backend

The live-camera path needs `face_recognition`, which builds on **dlib** (needs
CMake + a C++ compiler) and is the usual install snag.

On modern systems `dlib` / `face-recognition-models` can fail to build against
recent setuptools (`AttributeError: install_layout`). If that happens, in a
fresh virtualenv:

```bash
pip install "setuptools<60" wheel
pip install dlib face_recognition
```

or use a conda environment, where dlib ships prebuilt. None of this is needed
for the synthetic/demo path.

---

## Status

Phase 1 is implemented and validated on the synthetic path (the only path a
camera-less environment can exercise). The live-webcam path is implemented but
has not been run here — there is no camera or display in this environment. Phase
2 (multiple real faces, wide/overhead camera, possibly InsightFace embeddings)
is architected for but not built; see the tech plan and `decisions.md`.
