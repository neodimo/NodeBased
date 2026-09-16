# Handoff checkpoint

## v0.17 playback-perf handoff — 2026-09-15

Branch `v017/playback-perf`, based on the released `v0.16.0` commit `204b455`. Goal was
real-time 24fps 4K EXR playback through ACES 2.0; v0.16 already fixed the display transform
(2.17 fps / 468ms/frame native baseline, `display GPU`), so this pass profiled and optimized
what was left: EXR decode, color ingest, and proxy decimation. Full numbers, method, and the
before/after playback table: `docs/BENCHMARKS-v0.17-playback.md`. Summary in the top
`TASKLOG.md` entry.

Four fixes, each tied to a measured cost: `color.to_working`'s unpremult/premult round trip
moved off a non-contiguous `result[..., :3]` view onto the full contiguous array with a 4-wide
factor (144ms → 60ms/4K frame); `media.py`'s channel selection takes a slice instead of fancy
indexing when the channel indices are contiguous (48ms → 30ms); `Evaluator._decimate`'s proxy
downscale replaced `reshape(...).mean(axis=(1,3))` with strided-slice accumulation (157ms →
18.7ms at tier 2, ~8x — the single biggest win, since the pre-existing "proxy while playing"
auto-switch already routes 4K playback through this exact function and decode is
tier-independent); and new `nodebased/decodepool.py::DecodeAheadPool` decodes upcoming
Read-node source frames on a small bounded thread pool ahead of the playhead, sharing no state
with the single-owner `Evaluator`/`TileCache` (`docs/PLAYBACK.md`'s single-owner invariant is
unchanged). Cancellation is an `epoch` counter bumped only on real content-invalidating events,
never a plain playback tick, mirroring the same distinction `PlaybackQueue.replace` already
draws.

**A bug caught by the QA sweep before landing:** the first version of the decode-ahead wiring
prefetched unconditionally. Tier 1 (full resolution) playback uses a bounded-region read that
never consults the pool, so unconditional prefetching there just burned CPU/GIL on decodes
nobody reads back — full-res playback measured *worse* (1.31 fps) than a clean baseline (1.18
fps) until `request_preview` was gated to skip prefetching when `self.proxy.currentData() ==
1`. After the gate, full-res is a modest but real improvement (1.18 → 1.45 fps, from fixes 1-2
only, since decode-ahead and the decimation fix don't apply at tier 1).

**Real-hardware result** (`QT_QPA_PLATFORM=xcb DISPLAY=:0`, `tools/playback_qa.py` — promoted
from the gitignored scratch harness, new `--plate`/`--full` arguments — against a clean
same-methodology baseline of unmodified `204b455` via `NODEBASED_QA_SOURCE`): ACES 2.0
auto-proxy (tier 2) playback **2.10 fps → 7.61-8.02 fps (~3.6-3.8x)**. sRGB 8.02 fps; forced-CPU
(`NODEBASED_DISPLAY_GPU=0`) 6.30 fps. All runs drew distinct frames in order, no render errors.

**24 fps at native 4K ACES 2.0 was not reached.** Best measured: ~8 fps at the standard
auto-proxy tier. An isolated warm-path microbenchmark (decimate + graph eval + GPU display
transform + `to_qimage`) measures ~76-83ms/frame (~12-13 fps ceiling), but real playback
measures ~125-130ms/frame — a ~45-50ms/frame gap the per-stage benchmarks do not explain.
Candidates (decode-pool/preview-worker GIL contention — a worker-count sweep via the new
`NODEBASED_DECODE_AHEAD_WORKERS` env var was suggestive but inconclusive; `DisplayCache`'s
per-request digest hash; Qt scene-rebuild cost per frame) and the recommended next step
(instrument the real `Window.start_preview`/`preview_ready` path directly rather than guessing
among them) are in the benchmark doc's "Remaining gap and next steps" section.

**Verification:** new `tests/test_decodepool.py` (6 tests, pure Python, no Qt). New
`DecodeAheadIntegrationTests` in `tests/test_tileexec.py` (4 tests) prove a warmed decode
avoids a redundant `source_decodes` count while staying bit-identical to the cold path. New
`DecodeAheadPlaybackTests` in `tests/test_desktop.py` (4 tests, real `Window`) cover bounded
memory, the tick-vs-scrub epoch distinction, and clean teardown with a warm pool. Full offscreen
discovery (`python -m unittest discover -s tests`): **562 tests passed in ~140-161s** across
several runs. `SlowPlaybackTests.test_slow_playback_drops_frames_rather_than_queueing_them`
reproduced its documented pre-existing wall-clock flake (failed 1 of 3 isolated runs) —
untouched by this work, consistent with the `204b455` baseline note in `context/state.md`.
`git diff --check` passed.

**Unverified:** the exact cause of the ~45-50ms real-vs-isolated gap; any Windows behaviour for
the new decode-ahead threading or benchmarks (Linux/xcb only was tested); the env-var-tunable
worker count and memory budget were exercised but not swept for an optimal default beyond the
one data point recorded in the benchmark doc.

## Reference loop client handoff — 2026-09-14

Implemented `nodebased/agentloop.py` (console script `nodebased-agent-loop`) on
`v016/agent-loop`, based on `main` `6b3eee1`. It is an external client, not a NodeBased
change: connects to a running GUI's `--agent` endpoint, loops `inspect` -> `reference_context`
-> `Provider.propose` -> client-side validation -> one guarded `batch` -> `reference_context`
again, with `describe` fetched once per run. `ScriptedProvider` is deterministic for tests/demos;
`AnthropicProvider` uses stdlib `urllib` only, reads `ANTHROPIC_API_KEY` from the environment
only, and never logs or writes it to disk. Client-side validation restricts commands to
`create/set/connect/move/rename/disable/delete/reference/view/time`, checked against `describe`'s
node/param/limit/choice schema, with hard caps on iterations, commands per batch, images, image
bytes, and provider-response size. A stale-revision rejection gets exactly one
re-inspect/re-capture/re-propose retry, then a clear error; the loop never retries blindly.
`--dry-run`/interactive-confirm/`--yes` gate application, and each applied iteration is exactly
one GUI undo step (one atomic `batch`). NodeBased itself still makes no model or network call.
Along the way, fixed an offscreen-test-only deadlock: a same-thread GUI-plus-client integration
test needs `Connection.request()` to interleave short `waitForReadyRead` polls with
`QCoreApplication.processEvents()`, since a plain blocking wait never pumps the Qt event queue
the GUI-side `LocalBridge` needs to answer. Focused `tests.test_agentloop` passed **33 tests in
1.454s**; full offscreen discovery passed **535 tests in 146.752s**. `git diff --check` passed.
Unverified: a live Anthropic API call (only `urllib.urlopen` is mocked) and native
(non-offscreen) desktop interaction with the CLI.
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
