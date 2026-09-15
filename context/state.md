# Current state — 2026-09-14

## GPU/threaded viewer display transform

`nodebased/color.py::display_rgb` no longer runs the ACES 2.0 view transform
single-threaded on a freshly built processor. `nodebased/gpudisplay.py` (new) renders it
through an OCIO-generated GLSL shader on a `QOpenGLContext` owned by the existing single
preview worker thread, auto-selected whenever a working GL context is available;
`nodebased/color.py::apply_threaded` runs a thread-chunked CPU path (exact, not
approximate) as the fallback and as the path always used for the cheaper `sRGB` view,
since measurement showed GPU is a net regression there. `NODEBASED_DISPLAY_GPU=0` forces
CPU; any GPU failure falls back to CPU per call without crashing. Measured ~100x (HD) /
~46x (4K) speedup for ACES 2.0 vs. the pre-existing baseline; full numbers, accuracy
tolerance, and the GPU-vs-CPU routing rationale are in
`docs/BENCHMARKS-v0.16-display.md`. Focused `tests.test_display_transform` passed
**12 tests in ~0.2s** on real GPU/display; full offscreen discovery passed **514 tests
in 127.9s**. Native-display interaction beyond the automated GL/EXR checks, and a true
GPU-less machine for the CI fallback path, remain unverified.

## In-app agent/reference-image bridge

Schema v10 adds ordered unique `references` node IDs. Dispatcher `reference` toggles are
validated, atomic, undoable, and guarded requests may carry an exact `if_revision` integer.
The GUI inspector exposes `Reference for agent` for pixel-producing nodes. GUI-local
`reference_context` captures up to eight current-frame display PNGs (view first, then tags) into
an existing caller directory with prompt, revision, frame, roles, dimensions, and absolute paths.
The bridge is deliberately model/network-free; PNGs are display previews and do not enter graph
evaluation provenance. Parent review added collision-resistant per-response directories and
committed UI/context regression coverage. Canonical focused coverage passed **24 tests in 0.439s**;
full offscreen discovery passed **502 tests in 147.834s**. Native-display interaction remains
unverified.

## Tracker pixel analysis implementation

Tracker analysis is implemented on `main` from Luna's `112806c`, followed by parent review
hardening. The pure API is `nodebased.tracker.analyse`; the desktop inspector picks a reference
point and analyzes forward asynchronously. It commits only after success through one `set_tracks`
command, so cancel/error leaves the document unchanged. Completion is pinned to the original
Tracker and rejects concurrent track edits; weak NCC peaks are reported as possible occlusion
instead of writing arbitrary motion. Float32 premultiplied scene-linear pixels, negative data-window
origins, proxy/full display mapping, and the existing solve are preserved. Tracker and Roto remain
on the reference full-frame path; no TileKey or schema change was made.

Focused Tracker/Roto/UI verification passed **56 tests in 1.036s**. Full offscreen discovery
passed **492 tests in 146.035s**. Exact-head Desktop conformance run `34926562383` passed
**492/492** on Ubuntu in 180.515s and Windows in 310.711s at `2f5c85b`. Native-display pointer
QA remains unverified.

## Closeout at exact main commit

The repository was audited at `52bb039f20b6871179ba02081a520bf0ddd8257e`
(`main`, `origin/main`). This is the artist-facing Roto UI follow-up on the
tagged `v0.15.0` release (`eaf99f4ca9e771705e17c7bd1137e9e75a7c2913`). The
release tag remains at its intended release commit; the follow-up is pushed on
`main` and has not been tagged or released separately.

`__version__` is `0.15.0`; `SCHEMA_VERSION` is **9**. Schema v9 includes the
expression section and numeric-knob formula UI shipped in the release. Schema
v8 is the `node_data` payload step for Roto/Tracker. The post-release Roto UI adds a
foreground overlay for drawing closed polygons and dragging existing points;
edits use the validated, undoable `set_shapes` command and preserve animated
point envelopes.

## Verification

- Focused: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest
  tests.test_roto tests.test_roto_ui` — **48 tests passed in 0.844s**.
- Full: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s
  tests` — **484 tests passed in 146.851s**.
- Exact-head Desktop conformance run `34924508493` passed on Ubuntu and Windows
  at `52bb039` (3m43s and 5m52s respectively).
- Real-X-server EXR QA passed its decoder check at frames 1, 3, 7, 11, and 12.
  Playback showed all 12 frames in order: 512px/no-blur sustained 24.0 fps with
  a 0.08s longest update gap; 1600px/Blur sustained 18.2 fps with a 0.34s gap.
- Native-display visual QA, real input-device feel, and GPU/display claims
  remain unverified by the offscreen suite.

## Remaining blockers

- Tracker analysis is forward-only point tracking with a fixed reference pattern;
  it has no planar/perspective solve, automatic occlusion recovery, backward pass,
  or on-viewer track-marker editing after creation.
- Roto has no transform handles or track markers, no open/stroked splines,
  planar tracking, or ROI-limited/tiered execution. The evaluator path remains
  full-frame for these node kinds.
- Native hardware and installer-shell QA still require a real desktop.
- GPU acceleration for the viewer display transform is implemented (see above); a true
  GPU-less machine for the CI no-context fallback path, and Windows GL context creation,
  remain unverified.

## Branch and TODO audit

Branch cleanup on 2026-09-14 at `f7b52a4` left only `main` locally and on origin.
Divergent historical experiments that main superseded are kept as annotated tags on
origin: `archive/spike-roto-tracker` (`da01882`), `archive/nodebased-roto2`
(`8d9d584`), `archive/roto-rebase-onto-main` (`c1a88c5`, which also holds the
formerly uncommitted agent-worktree roto WIP), and `archive/nodebased-closeout`
(`46ebad0`). Every other removed branch or worktree was already contained in `main`.
The alternate rotate/scale similarity solver and 619-line roto test file in
`archive/roto-rebase-onto-main` are the only material worth mining later.

The tracked product and test sources contain no unresolved `TODO` or `FIXME`
items. Remaining future-work language concerns the bounded limits of point tracking,
the external agent that consumes reference contexts, and unverified native/GPU behavior.

The formerly untracked `.claude/` tree held only the agent worktree now archived above
and was removed during cleanup.

## Next owner

Use `docs/AGENT_PROTOCOL.md` and `tests/test_reference_bridge.py` for the current
agent-reference contract. The next integration owner should connect an external
vision-capable agent to `reference_context`, then submit graph edits as a guarded
atomic `batch` using the returned revision.
