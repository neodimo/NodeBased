# Handoff checkpoint

## GPU/threaded display-transform handoff — 2026-09-14

Implemented on `v016/gpu-display`, based on `main` `6b3eee1`. Goal: the viewer's ACES 2.0
display transform was the 4K playback bottleneck (~2.2s/frame CPU). Two independent fixes,
each measured separately, per `docs/BENCHMARKS-v0.16-display.md`:

- **Threaded CPU** (`nodebased/color.py::apply_threaded`): OCIO's `CPUProcessor.applyRGB`
  releases the GIL, so row-chunking across a persistent 16-worker `ThreadPoolExecutor` gives
  an exact (not approximate) ~11-12x speedup at both HD and 4K. This is the CPU fallback and
  is also what serves the `sRGB` view even when GPU is available (see below).
- **GPU** (`nodebased/gpudisplay.py`, new module): builds each view's OCIO `GpuShaderDesc`
  (GLSL 4.0) once, renders through a `QOpenGLContext` owned by the single preview worker
  thread, reads back via `GL_RGB`/`GL_FLOAT`. ~100x over baseline at HD ACES 2.0, ~46x at 4K
  ACES 2.0. Auto-selected only for the `ACES 2.0` view — measured GPU overhead (fixed
  texture upload/readback cost) makes it a *regression* for the cheap `sRGB` view, so
  `sRGB` always uses the threaded CPU path. `NODEBASED_DISPLAY_GPU=0` forces CPU
  everywhere; any GPU failure (no context, no QApplication yet, mid-session fault) falls
  back to CPU per call, never crashes, and the fallback is exercised by real tests, not
  just code review.
- **Correctness:** GPU vs. CPU on a wide-gamut/HDR test image (saturated ACEScg primaries,
  values to 16.0, negatives, near-zero) — max 8-bit code-value error is exactly the `<=1`
  tolerance boundary on this machine (see the doc for the float-space numbers).
- **Real-app evidence:** a live `Window` loaded a real 4K EXR under
  `QT_QPA_PLATFORM=xcb DISPLAY=:0`; `viewer_info.text()` showed `display GPU` after the
  first frame, confirming the GPU path serves real playback, not just the isolated
  benchmark.
- **Verification:** focused `tests.test_display_transform` — **12 tests passed in ~0.2s**
  under both `xcb`/real-display and `offscreen` (this machine's `offscreen` QPA platform
  also gets a real GL context, see the doc's "Machine" section for why that is not a
  guarantee elsewhere). Full offscreen discovery — **514 tests passed in 127.9s**
  (502 pre-existing + 12 new).
- **Unverified:** no GPU-less machine was available to reproduce the CI no-context
  fallback path end-to-end (the exception-handling code path was verified, and
  `NODEBASED_DISPLAY_GPU=0` exercises the same `display_rgb` fallback branch); Windows GL
  context creation is untested; the GPU/CPU crossover resolution below HD is unmeasured.

## Reference bridge handoff — 2026-09-14

The bounded v1 bridge is implemented on `openclaw/reference-bridge`: schema v10 reference tags,
atomic `reference` edits with delete cleanup and undo/redo, exact revision preconditions, inspector
checkbox wiring, and GUI-local `reference_context` PNG capture. The context path evaluates through
the existing float32 premultiplied evaluator and display conversion, caps at eight captures, and
writes every response into a fresh collision-resistant child directory. There is no model/network
call and no TileKey/tiled change. Canonical focused coverage passed **24 tests in 0.439s** and full
offscreen discovery passed **502 tests in 147.834s**. Native-display QA remains unverified.

## Tracker pixel analysis handoff — 2026-09-14

Implemented from exact base `5958490d50c5315ac7308ac5fe85614b61be0198`: `tracker.analyse` performs
deterministic zero-mean NCC with bounded windows and sub-pixel parabola refinement, then the Tracker
UI picks a reference point and analyzes forward. The sole document mutation after success is one
validated atomic/undoable `set_tracks`; failure and cancel preserve the prior document. Parent
review pinned completion to the original Tracker, rejects concurrent track edits, made cancellation
win even if clicked just after computation, and rejects weak/occluded matches. Focused Tracker/Roto/UI
verification passed **56 tests in 1.036s**; full offscreen discovery passed **492 tests in 146.035s**.
Exact-head Desktop conformance run `34926562383` passed **492/492** on Ubuntu
(180.515s) and Windows (310.711s) at `2f5c85b`. Native-display interaction remains pending.
No `.claude/`, tag, release, schema, TileKey, or tiled claim was changed.

Updated 2026-09-14 at exact commit
`52bb039f20b6871179ba02081a520bf0ddd8257e`.

## Shipped state

- `v0.15.0` is tagged at `eaf99f4ca9e771705e17c7bd1137e9e75a7c2913`.
- The follow-up Roto UI commit is pushed on `main` and `origin/main`; it is
  intentionally not a new tag or release.
- Product version is `0.15.0`; current document schema is v10. Roto/Tracker payloads
  use the v8 `node_data` step, expressions use the v9 section, and agent reference
  tags use the v10 `references` section.
- Roto drawing and point dragging are implemented in `nodebased/app.py` and
  covered by `tests/test_roto_ui.py`. Gestures route through `set_shapes`, with
  validation, undo, save, agent semantics, and animated-point key retention.
- `docs/ROTO_TRACKING.md` describes the post-release Roto UI and bounded pixel Tracker
  analysis. `docs/RELEASE_NOTES.md` remains scoped to the exact `v0.15.0` tag and does
  not claim either follow-up shipped in that release.

## Verification and limits

The closeout focused on `tests.test_roto` plus `tests.test_roto_ui`: **48 tests
passed in 0.844s**. The full offscreen discovery suite then passed **484 tests
in 146.851s**. Exact-head Desktop conformance run `34924508493` passed Ubuntu
and Windows at `52bb039`. Real-X-server EXR QA also passed: decoder 5/5; 512px
playback 24.0 fps with a 0.08s longest gap; 1600px + Blur 18.2 fps with a 0.34s
gap. Native-display QA, real pointer feel, installer shell integration, GPU
behavior, and native Tracker pointer interaction remain unverified.

## Branch hygiene

As of 2026-09-14 the only branch is `main`, and no extra worktrees remain. Historical
roto/tracker experiments are preserved as `archive/*` tags on origin (see
`context/state.md`). They are provenance only, not merge instructions.

## Next owner

This historical Roto checkpoint is superseded by the Tracker and reference-bridge
handoffs above. Use their newest TASKLOG entries and focused tests for current work.
