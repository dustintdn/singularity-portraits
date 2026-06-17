# Decisions Log

Choices I made autonomously while implementing the tech plan, with enough
reasoning that you can overrule any of them on review. Grouped roughly by how
much they'd cost to change later.

> Context that shaped almost everything below: **this build ran on a headless
> cloud container with no camera (`/dev/video*` absent) and no display.** So I
> could not test the literal Phase-1 path (webcam → `face_recognition` → window).
> Several decisions exist to make the pipeline fully runnable and *reviewable*
> without a camera, while leaving the real path intact for when you run it on
> your own machine.

---

## Architectural decisions

### 1. Wrapped everything in a `singularity/` package instead of loose top-level dirs
The plan sketched `identity/` and `visuals/` as top-level folders next to
`main.py`. I put them under a single importable package, `singularity/`, with
`__init__.py` files. **Why:** clean imports for tests and the gallery tool, no
risk of name collisions with other top-level files, and it keeps `main.py` as a
thin CLI at the root exactly as the plan wanted. The module split itself
(`detector`, `registry`, `seed`, `render`, `tracking`) matches the plan.

### 2. Put a `Detector` interface in front of the embedding backend
The plan named `face_recognition` as the Phase-1 backend and InsightFace as a
possible Phase-2 swap. I made that swap a first-class seam: a `Detector`
protocol with two implementations —
- `FaceRecognitionDetector` (the real thing; imports `face_recognition` lazily
  so the package still works where dlib isn't installed), and
- `SyntheticDetector` (fabricates deterministic, well-separated moving faces).

**Why:** the synthetic backend is what let me run and validate the *entire*
pipeline — identity → seed → visuals → tracking → render — on a camera-less box,
and it doubles as the substrate for fast, deterministic tests. Swapping in
InsightFace for Phase 2 is now a one-class change, as the plan hoped.

### 3. Added a `FrameSource` abstraction (webcam / video / image-folder / synthetic)
The plan assumed a webcam loop. I generalised "where pixels come from" into four
interchangeable sources. **Why:** lets you reproduce bugs from a recorded clip
or a folder of photos, and lets the synthetic source drive everything with no
hardware. Phase 2's overhead/wide-angle camera is just another `FrameSource`.

### 4. The seed is computed **once per identity and cached**, not per frame
The plan's quantise-then-hash seed is in `visuals/seed.py` as specified. But
while testing I confirmed what the plan only implied: fixed-grid quantisation
*reduces* but cannot *eliminate* seed flicker — a value sitting on a rounding
boundary still flips when noise crosses it. So the real "same face → same
singularity, every frame" guarantee comes from the app computing each identity's
seed **once** (from the first/running-averaged embedding the registry resolves)
and caching it by `identity_id`. I corrected the comment in `seed.py` to say this
honestly, and there's a test (`test_app_caches_one_stable_singularity_per_identity`)
that nails the behaviour down. **This is the one place I'd most want your eyes.**

---

## Visual / aesthetic decisions

### 5. Expanded the visual vocabulary beyond the plan's placeholder
The plan's `seed_to_visual_params` was explicitly a placeholder. I grew it into
a fuller fingerprint: a precomputed 2–3 colour palette, `angularity` +
`noise_octaves` + `noise_amplitude` driving a noise-deformed core outline (not a
plain circle), `pulse_speed`/`pulse_depth`, `rotation_speed`, and — picking up
the concept doc's language — a discrete **motion archetype** per identity
(`drift` / `pulse` / `orbit` / `jitter`). All still pure functions of the seed.
**Why:** the plan said to expect heavy iteration here and that this mapping is
"where face structure becomes visual personality." See `assets/gallery.png` for
12 identities at a glance. Treat the specific ranges as a first pass, not gospel.

### 6. Renderer uses additive glow + motion trails, and is headless-capable
Went with the plan's **Option A (Pygame)** for Phase 1. Beyond filled circles I
added cached radial-gradient glow sprites composited with `BLEND_RGB_ADD` and a
per-frame translucent veil that leaves light trails (the `trail` parameter).
**Why:** the additive glow + trails are what make it read as "a ball of light"
rather than "a coloured disc" — see `assets/walkthrough.mp4`. The renderer draws
to an offscreen surface so it runs windowless and can export PNG/MP4, which is
how anything is reviewable without a display. The live-window path
(`--source webcam`, no `--headless`) is implemented but **untested here** — no
display to open one.

### 7. Default `trail = 0.18` (fairly long ghosting)
Subjective. Long trails look striking in the walkthrough but may feel too smeary
in a real room with people moving fast. Easy knob to turn in
`SingularityRenderer`. Flagging it as a taste call you should make in situ.

---

## Identity / ethics decisions

### 8. Cross-session persistence is **off by default, opt-in via `--registry`**
The plan flagged persisting biometric data between runs as "a meaningful design
decision." I made the safe default *not* to persist: the registry lives in
memory and evaporates when the process exits, unless you pass `--registry
path.json`. **Why:** storing face embeddings to disk shouldn't be something that
happens by accident, especially for a piece whose whole subject is surveillance.
The save path is documented in code as writing biometric data. If you
want the installation to "remember repeat visitors across days," that's the
flag — but I left the choice as an explicit, conceptual one for you, per the
concept doc's open question about consent.

### 9. Synthetic personas are deliberately *well-separated* in vector space
The `SyntheticDetector`'s fake faces are easy to tell apart on purpose. **Why:**
it stands in for "different people have different faces," not for the genuine
messiness of real detection (occlusion, lookalikes, oblique angles). Real
embeddings will be noisier and the `0.6` match threshold will need empirical
tuning — and this is exactly the case for upgrading to ArcFace/InsightFace in
Phase 2, as the plan notes. Don't read the clean synthetic separation as
evidence the threshold is right for real faces.

---

## Process / tooling decisions

### 10. `opencv-python-headless` instead of `opencv-python`
On a server the headless OpenCV wheel avoids pulling GUI/X11 deps. For a real
installation machine with a display you may prefer plain `opencv-python`;
`requirements.txt` notes both.

### 11. Added `tools/gallery.py` and committed rendered artifacts
The contact-sheet tool and the two committed artifacts (`assets/gallery.png`,
`assets/walkthrough.mp4`) exist so you can judge the actual output in the PR
without running anything. If you'd rather not version binaries, delete them and
`.gitignore assets/` — they regenerate from the two commands in the README.

### 12. Did **not** wire up `face_recognition` in CI/tests
dlib takes minutes to compile and needs a C++ toolchain; the tests deliberately
run entirely on the synthetic path so they're fast and hardware-free. The real
backend is covered only by its import-guard. I attempted to install
`face_recognition` in this environment to smoke-test instantiation; note whether
that succeeded in the PR description.

---

## Open questions I did **not** decide (left for you)

These map to the concept doc's "Open Questions Worth Sitting With" and are
product/artistic calls, not code calls:

- **Consent boundary** — no visible "you are now becoming a singularity" marker
  exists. Architecturally trivial to add (it's just another render layer).
- **Inter-singularity interaction** — currently each form is strictly solitary;
  they pass through each other with no merge/repel/resonance. Hooking physics
  between tracks would live in `TrackManager` + `render`.
- **The reveal** — there's no on-screen explanation of how your colours derive
  from your face. The gallery tool is a *debugging* reveal, not an installed one.
