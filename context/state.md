# Current state — 2026-09-19

## 0.21.1 released

v0.21.1 published from `8fc11d6` (run 35422320319, both platforms green). 3D work continues on
`feature/3d-foundation`, rebased onto that commit.

## 3D scene graph — 0.22.0

`feature/3d-foundation` carries the 3D system described in `docs/3D_FOUNDATION.md`: typed
card/cube/sphere/OBJ geometry with image textures, lights, nested scenes, camera, `Render3D`
(AA, depth/normal passes, cached) and the navigable viewport. CPU reference rasterizer only.
Splats, ray tracing, particles, fluids, USD and projection are roadmap items, not code. Newest
`TASKLOG.md` entry has the evidence and the parameter-rename note.

## 0.23.0 (2026-09-19)

The Astra lane (`openclaw/nb-3d-astra-lane`) was merged to `main` through `eb38a45` (fast-forward,
16 commits): optional wgpu backend, USD import/export, in-house Alembic reader, `Project3D`,
`WriteGeo3D`, shadows, Blinn-Phong materials and named AOVs. `72d3857` bundles `usd-core` and
`wgpu` in release builds with a frozen-app smoke check. The first Ubuntu packaging dry run
failed because the runner's downlevel GL adapter was treated as a usable GPU; `9b016b1` makes
adapters that cannot render to `rgba32float` report unavailable. Pre-tag evidence on `9b016b1`:
Desktop conformance 35461128952 and packaging dry run 35461133645 green on Linux and Windows,
the first Windows runs of the GPU backend, USD, Alembic, shadows, materials and AOVs. Windows
behaviour is proven by CI only. Limits are listed in `docs/RELEASE_NOTES.md`. Of the hard
requirements below, USD and Alembic are partially delivered; splats and splat relighting are
not started. The lane continues with ray tracing (BVH and a CPU ray-traced mode exist on the
lane branch, unmerged), then splats, relighting, particles and volumes.

## 3D hard requirements (from DiMo, 2026-09-19 01:18 PDT)

These are required deliverables of the 3D system, not optional roadmap ideas. Each needs real
tests and an acceptance scene before it is called supported.

- **Gaussian splats:** import (3DGS `.ply` at minimum), camera-correct anisotropic rendering,
  depth interaction with meshes.
- **Splat relighting, in both the 3D viewport and the final render.** Reference point DiMo
  named: V-Ray in Houdini with splats, i.e. splats that take scene lights, cast and receive
  shadows and sit inside a ray-traced render with ordinary geometry. Baked-SH-only display does
  not satisfy this. Nuke 17.1 also ships splat relighting and shadows.
- **Alembic (`.abc`) files:** geometry, cameras, animated/time-sampled data. There is no
  PyAlembic wheel on PyPI (checked 2026-09-19; the PyPI package named `alembic` is the
  SQLAlchemy migration tool and must never be added). Needs a packaging spike: Linux+Windows
  route for an Ogawa reader before any node is promised.
- **USD files:** `usd-core` 26.8 has cp312 wheels for Linux and Windows (license
  LicenseRef-TOST-1.0, to be recorded). Stage load, meshes, cameras, xforms, time samples,
  layer composition; export later. `usd-core` does not include the usdAbc Alembic plugin, so
  USD does not solve Alembic for free.

Owner: Astra lane (`openclaw/nb-3d-astra-lane`), reviewed by Gonzo before merge.

## Release policy (standing, from DiMo 2026-09-18)

Do not park a finished release waiting for approval. When `main` is green and the work is
merged, cut the release: bump `pyproject.toml` and `nodebased/__init__.py`, prepend the notes
to `docs/RELEASE_NOTES.md`, sync `nodebased/data/docs/RELEASE_NOTES.md` (a test now enforces
that the bundled copies match `docs/`), run the full suite, commit, tag `vX.Y.Z`, push the tag,
then watch both tag workflows and confirm the published assets. Report what shipped afterwards.

## 0.20.0

Merged `lane/agent` (Agent panel, `nodebased-mcp`, bundled knowledge, GitHub issue filing),
`lane/viewer` (pixel readout, Nuke hotkeys, viewer zoom/shuttle, shortcuts dialog) and
`lane/knobs` (Nuke knob types, expressions via `=` or the context menu, User/Node tabs,
stacked properties panels) into `main`, then tagged v0.20.0 at `5caf12f`.

Verified by hand on this machine (Linux/X11): properties tabs and knob widgets, the Node tab's
label/enable/postage-stamp switches, the Agent panel, and Claude Code genuinely launching in
its embedded terminal as far as its trust-folder prompt. The pixel readout was **dead in the
running app** until `c6d7f91`: `Viewer` had two `mouseMoveEvent` definitions and the later
roto/tracker one shadowed the readout's handler, while every readout test called
`_update_pixel_readout` directly and stayed green. Lesson: after merging UI work, drive the
real app, and check merged classes for duplicate method names.

Still unproven: signing an agent in and running a full session through the Agent panel; the
MCP server under a real agent session; anything on Windows beyond CI; and the 0.17 playback
limits are unchanged (24 fps at 4K ACES 2.0 still not reached).

# Current state — 2026-09-15

## Graph node readability and port layout

The current local UI pass in `nodebased/app.py` centers and enlarges node titles, dims disabled
cards and paints a large X, keeps disabled Merge `B` visible, places `A` on the left edge and
`mask`/`B` on the right edge, and raises Dot items above noodles. New focused assertions live in
`tests/test_desktop.py`. Six focused tests pass. The full desktop run had one known timing flake
(`test_slow_playback_drops_frames_rather_than_queueing_them`) and otherwise passed; changes are
uncommitted and not pushed.

## v0.17 4K playback performance (branch `v017/playback-perf`)

Goal: real-time 24fps 4K EXR playback through ACES 2.0 (v0.16 already made the display
transform fast — see below — this pass targeted EXR decode, color ingest and proxy decimation).
Full evidence: `docs/BENCHMARKS-v0.17-playback.md`; summary: `TASKLOG.md` top entry.

Four measured fixes: `color.to_working`'s premult round trip moved off a non-contiguous array
view (144ms → 60ms per 4K frame); `media.py` channel selection takes a slice instead of fancy
indexing when channels are contiguous (48ms → 30ms); `Evaluator._decimate`'s proxy downscale
replaced a reshape+multi-axis-mean with strided accumulation (157ms → 18.7ms at tier 2, ~8x);
and a new `nodebased/decodepool.py::DecodeAheadPool` runs Read-node source decode on a small
bounded, separately-threaded LRU ahead of the playhead, gated to the tier != 1 path only (tier
1 uses a bounded-region read that never consults it — prefetching there was measured to make
full-resolution playback *worse* and is explicitly avoided).

Real-hardware result (`QT_QPA_PLATFORM=xcb DISPLAY=:0`, same `tools/playback_qa.py` harness
against a clean baseline of unmodified `204b455`): ACES 2.0 auto-proxy playback **2.10 fps →
7.61-8.02 fps (~3.6-3.8x)**. **24fps is not reached.** An isolated warm-path microbenchmark
suggests a ~12-13fps ceiling for the current per-frame compute, but real playback lands at
~7.6-8fps — a ~45-50ms/frame gap not yet attributed to a specific stage (candidates: decode-pool
GIL contention with the single preview worker, `DisplayCache` digest hashing, Qt scene-rebuild
cost). Recommended next step: instrument the real `Window.start_preview`/`preview_ready` path
directly instead of guessing among those candidates.

Full offscreen suite: 562 tests passing (one known pre-existing flake below, reproduced
independently of this work, unaffected). Unverified: the exact cause of the real-vs-isolated
gap above; Windows behaviour for any of this; `NODEBASED_DECODE_AHEAD_WORKERS`/`_MB` env
overrides are exercised by code path but only lightly swept for tuning.

## Reference loop client (external agent)

`nodebased/agentloop.py` (console script `nodebased-agent-loop`) closes the v10 reference-image
loop from outside the process: `inspect` -> `reference_context` -> `Provider.propose` -> client
validation -> one guarded `batch` -> `reference_context` again, per iteration, `describe` fetched
once. `ScriptedProvider` is deterministic (tests/demos); `AnthropicProvider` is stdlib `urllib`
only, reads `ANTHROPIC_API_KEY` from the environment only, and is never logged/written to disk.
Client-side validation allows only `create/set/connect/move/rename/disable/delete/reference/
view/time`, checked against `describe`'s node/param/limit/choice schema; hard caps on iterations
(3 default, 10 max), commands per batch (64), images (8), image bytes, and provider-response
size. A stale-revision batch rejection gets exactly one re-inspect/re-propose retry, then a clear
error. `--dry-run`/interactive-confirm/`--yes` gate application; each applied iteration is one
GUI undo step. See `docs/AGENT_PROTOCOL.md#reference-loop-client` for the full contract.
Focused coverage passed **33 tests in 1.454s**; full offscreen discovery passed **535 tests in
146.752s**. Unverified: a live Anthropic API call and native (non-offscreen) CLI interaction.
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

Use `docs/AGENT_PROTOCOL.md` (including its "Reference loop client" section),
`tests/test_reference_bridge.py`, and `tests/test_agentloop.py` for the current agent-reference
contract and its external client. The reference loop from prompt to applied `batch` is now
implemented end to end via `nodebased-agent-loop`; a live Anthropic API call has not been
exercised (only `urllib` is mocked in tests). Candidate follow-ups: an in-app panel wrapping the
loop instead of a separate CLI process, an additional `Provider` for another vision-capable
model, and native (non-offscreen) desktop QA of the CLI against a real running GUI.

## 0.20.0 release (2026-09-18)

Published at https://github.com/neodimo/NodeBased/releases/tag/v0.20.0 from 468292e (the first
tag build, run 35318619631, failed on Windows in the fixed-2.5 s slow-playback test; fixed and
re-cut). Linux AppImage smoke test on this machine (X11 :0, Strix Halo + RTX 3080): launches,
pixel readout appears on hover. **Open finding:** the packaged AppImage falls back to CPU display
with `GPU unavailable: QOffscreenSurface is not valid on this platform`
(`qglx_findConfig: Failed to finding matching FBConfig`), while the venv build on the same
display reports `display GPU`. The GLX/EGL xcbglintegration plugins are bundled, so the cause is
not yet known. Not checked whether 0.19.0's AppImage had the same fallback. Screenshots:
`workspace/media/nb-qa/v020-*.png`.
