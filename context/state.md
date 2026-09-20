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

**Published 2026-09-19 12:29 PDT:** <https://github.com/neodimo/NodeBased/releases/tag/v0.23.0>, tag
`v0.23.0` on `68652cd`. Tag runs green on Linux and Windows: Build release packages 35463195721,
Desktop conformance 35463195732 (and 35463194074 on `main`). Assets verified: AppImage, Windows
setup exe, portable zip, `SHA256SUMS`; the three sums match the GitHub asset digests.

## Ray tracing on main after 0.23.0 (2026-09-19 15:25 PDT)

`main` was fast-forwarded to `96a458f` (7 lane commits, unreleased): BVH and CPU ray queries,
CPU ray-traced `Render3D` mode with raster parity, GPU BVH traversal for shadow rays, depth
peeling (`PEEL_BATCH=8`) so hidden opaque surfaces do not count toward `MAX_HITS_PER_RAY=64`,
a `(t, primitive)` peel cursor so exact-t ties are never dropped, and adapters with fewer than
2 storage buffers reporting unavailable. Two review defects were found and fixed before the
merge (hit cap counting hidden surfaces; t-only cursor dropping ties beyond one batch).
Evidence at `96a458f`: clean-checkout full discovery 955 tests OK with GPU and USD extras on
the RTX 3080 Ti; reviewer repros (12/19/40 coincident cards, 70 opaque stacked, ties at two
depths over hidden opaque cards) match the rasterizer. Known limits: GPU shadow budgets refuse
1080p renders above roughly 19k triangles (tiled submissions planned); the BVH work estimate is
an average; at exact-t ties between differently coloured surfaces raytrace and raster composite
in opposite tie order, so RGBA can differ. Windows behaviour for these commits is proven by CI
only. Lane queue from here: Gaussian splats, splat relighting, GPU ray-traced mode on wgpu,
tiled GPU submissions, particles, volumes.

## Gaussian splats on main, steps 1-2 (2026-09-19 4:45 PM PDT)

`main` was fast-forwarded to `df8c034` (3 lane commits, unreleased): `nodebased/splats.py`
(`SplatCloud`, SH evaluation and rotation, 3DGS `.ply` reader/writer, COLMAP orientation),
`nodebased/splatraster.py` (CPU EWA renderer, 0.3 px dilation, stable front-to-back
compositing, 400M splat-pixel budget refusal, cancel points), `Scene.splats` with `Scene3D`
nesting, and the relighting design paragraph in `docs/3D_ROADMAP.md`. One review defect was
found and fixed before the merge: `read_ply` hung on an ASCII file declaring a huge
property-free element. Evidence at `df8c034`: clean-checkout full discovery 980 tests OK with
GPU and USD extras; reviewer repros pass (single-splat footprint against the analytic Gaussian
at 2.5e-8, input-order independence, opaque-mesh occlusion at the centre and the image corner
in raster and raytrace with identical output, `samples=2`, cancel, budget refusal, mirrored and
singular matrices, 10 hostile PLY files all rejected or read in under 0.1 s).

This does NOT satisfy the splat hard requirement yet. Missing: `ReadSplat3D` node (lane is on
it), splat relighting in viewport and final render, mesh/splat mutual shadows, GPU splat path
(wgpu reports `Unsupported`, `auto` falls back to CPU), viewport splat display (owned by
`gonzo/3d-ux`). Known limits: splats appear in `rgba` only, every AOV and `return_depth` ignores
them; visibility is by splat centre depth, so a mesh cutting through a splat does not slice it
and transparent meshes are not sorted against splats; no real captured 3DGS file has been read,
only files from our own writer; `.splat` and compressed formats are unread; Windows is unrun
until CI on `df8c034` finishes.

## ReadSplat3D on main, step 3 (2026-09-19 5:25 PM PDT)

`main` was fast-forwarded to `859e126` (lane commit, was `5186b07` before the rebase onto the
state commit; unreleased): `ReadSplat3D` node producing a scene with one `SplatInstance`,
Nuke-style knobs (Translate / Rotate / Scale / Pivot as XYZ numeric fields, uniform scale,
rotation order), new `xyz` and `float` knob kinds plus a Browse button in `app.py` (small hunks;
no `viewportgpu.py`, `viewport3d.py` or graph UI files), a size/mtime-keyed LRU cloud cache
(4 entries / 1 GiB, read-only arrays), and render-time SH degree, opacity and scale
multipliers. `Transform3D` gained `order`, `pivot` and `uniform`; defaults are bit-identical to
the old matrix. Windows CI on `df8c034` (first Windows run of the splat code) finished green.
Evidence at `859e126`: clean-checkout full discovery 990 tests OK with GPU and USD extras;
reviewer repro 35/35 (node matrix against an independent ZXY + pivot + uniform composition at
3.8e-8, node render equals a pre-transformed cloud render at 9e-7, each multiplier equals the
equivalently edited cloud exactly, SH clamp commutes with rotation, raytrace equals raster
with non-default controls, nested `Scene3D` keeps the controls, 13 invalid parameter values
rejected with the document unchanged, directory / missing / mesh `.ply` / `/dev/zero` /
`/dev/null` paths all fail in under 0.01 s with a `ValueError`, same-size rewrite is picked up).

Known debt from this commit: the splat limits were also inserted into `core.TIME_LIMITS`
(harmless, `validate_time` iterates `DEFAULT_TIME`, but they leak into the state snapshot's
`time_limits`; sent to the lane as cleanup); `depth` and the other AOVs still ignore splats
(lane's current step); the only non-self-written 3DGS file read so far is a synthetic
1,161-splat SharpSplat output, no photogrammetry capture; Windows unrun for `859e126` until its
CI finishes. The splat hard requirement is still NOT met: no relighting, no mutual shadows, no
GPU path, no viewport display.

## Splat depth, AOVs, banded buffers and raster fast path on main (2026-09-19 8:26 PM PDT)

`main` was fast-forwarded from `aa947ba` to `d0dbf86` (5 lane commits, unreleased): `877a997`
per-pixel mesh/splat depth ordering with opaque and transparent meshes in both render modes;
`f440874` splat data passes (`depth`, `normals`, `position`, `uv`, `object_id` select the first
mesh hit or accumulated splat opacity >= 0.5), a new `splats` layer output, and the splat keys
removed from `core.TIME_LIMITS` with a guard test; `9950cc3` banded mesh-layer buffers and centre
depth for near-isotropic splats; `5fbcc7c` TASKLOG only; `d0dbf86` raster fast path restored. No
`viewportgpu.py`, `viewport3d.py` or graph UI files touched.

Two review defects were found and fixed before the merge. Layer buffers were full-frame at
384 B/pixel (HD 934 MiB, HD samples=2 3.0 GiB); banded they peak at 154 MiB and 272 MiB, and
banded output equals single-band output bit-for-bit (3 scenes x 7 outputs x 2 modes x samples
1/2 x 3 band sizes). `f440874` forced primary rays for every render with splats, so raster rgba
with a 20,000-triangle opaque grid + 1 splat at 1920x1080 went 2.1 s -> 23.9 s and fell under
`RAYTRACE_WORK_BUDGET`; at `d0dbf86` it is 1.92 s (samples=2: 2.52 s; mesh only 1.89 / 2.43 s).

Evidence at `d0dbf86` (clean temp worktree, shared project venv with GPU and USD extras): full
discovery 1010 tests OK in 701.8 s; 42 targeted splat/AOV tests OK; reviewer repro all pass:
raster rgba, `splats` and all five data passes call neither primary rays nor the ray budget with
opaque meshes; rgba == splats + (1-a)*mesh exactly in both modes; raster == raytrace exactly for
rgba and `splats` with 300 splats; data passes differ from the mesh-only render only inside
splat coverage; splats behind the opaque mesh leave every output identical to mesh-only; the
transparent-mesh layered path equals raytrace. Earlier per-commit evidence is in the workspace
`memory/2026-09-19.md` (5:35 PM, 6:00 PM, 6:58 PM entries).

Known debt: `render_splats` preparation (projection, SH, sort, Python binning loop) reruns once
per band, 12 bands at 1080p (200k splats HD layered 34.7 s banded vs 28.9 s single band), lane's
next step; no Jacobian clamp, so the first real capture (Nelson Ghost Town, 3,409,742 splats,
local only under `assets/splats/`, never committed, CC BY 4.0 Paolo Tosolini) renders a
full-frame veil from about 31.6k large off-frustum splats, and the default `SPLAT_WORK_BUDGET`
refuses it at 640x360; albedo/diffuse/specular/emission ignore splats; Windows unrun for these
five commits until CI 35486575225 finishes. The splat hard requirement is still NOT met: no
relighting, no mutual shadows, no GPU path, no viewport display.

## Splat preparation hoist on main (2026-09-19 9:07 PM PDT)

`main` was fast-forwarded from `46be8ff` to `0428551` (1 lane commit, unreleased):
`prepare_splats` runs once per render and `accumulate_splats` runs per band; tile bins are CSR
NumPy arrays instead of per-tile Python lists. This closes the per-band preparation debt named
above. Evidence (clean temp worktrees at `0428551` and `d0dbf86`, shared project venv): 252 of
252 arrays byte-identical between the two commits (3 scenes x 7 outputs x raster/raytrace x
samples 1/2 x band sizes 1-row / 20-row / single, two instances of random anisotropic rotated
splats); 200k splats + transparent card at 1080p banded 25.5 s -> 21.3 s (single band 20.9 s),
banded peak 299 -> 241 MiB; `tests.test_3d_splat_render` 32 OK; full discovery 1014 tests OK in
744.8 s. CI for `d0dbf86` and `46be8ff` is green on Linux and Windows; CI 35488358648
(`0428551`) was still running when this was written.

Still open at `0428551`: the Jacobian clamp for the Nelson Ghost Town veil is committed on the
lane branch only and is unreviewed; albedo/diffuse/specular/emission ignore splats. The splat
hard requirement is still NOT met: no relighting, no mutual shadows, no GPU path, no viewport
display.

## Jacobian clamp, splat relighting step A and splat shadows B1 on main (2026-09-19 10:27 PM PDT)

`main` moved `401ef85` -> `4d41524` -> `834fcbc` -> `2fdd8a1` (3 lane commits, unreleased), each
reviewed in a clean temp worktree with the shared project venv before the fast-forward.

- `4d41524` Jacobian clamp (reference 3DGS: x/z and y/z limited to 1.3*tan(fov/2), centres
  unclamped). Nelson Ghost Town capture at 640x360: the blue veil is gone (pre-clamp alpha mean
  1.0; clamped render matches a 1.3x frustum-culled render to mean abs 0.0014, differences only
  at frame borders); 140,731,576 pairs, accepted by the 400M `SPLAT_WORK_BUDGET`, 44.8 s. Fully
  in-frustum scenes are byte-identical to `0428551`. Full discovery 1018 OK (1 skipped). Pair
  counts depend on the near plane: near 0.1 gives 689M pre-clamp, near 3 gives 270M.
- `834fcbc` relighting step A: per-splat Lambert from shortest-axis normals and SH DC albedo,
  `Relight` 0..1 knob on ReadSplat3D, reusable `nodebased/splatshade.py::shade_splats`. Relight 0
  is byte-identical to `4d41524`; mix .5 equals the exact blend; point lights have no falloff,
  same as the mesh shader. Full discovery 1027 OK (1 skipped).
- `2fdd8a1` splat shadows B1: meshes and splats shadow RELIT splats through a `SplatSet` BVH
  (`nodebased/raytrace.py`), one shadow ray per splat centre per shadow-casting light, emitter
  excluded and closest approach clamped to 2.5x its largest scale so a surface does not shadow
  itself; refusal above `SPLAT_SHADOW_BUDGET`. Review evidence: 24 outputs (empty scene, mesh
  only, relight 0 with a shadow light, relight 1 with a non-shadow light, splats only; raster and
  raytrace; rgba and depth) byte-identical to `834fcbc`; a flat splat surface keeps visibility
  1.0; an off-axis card shadows only the splats under it, never brightens, leaves alpha
  untouched, and raster equals raytrace exactly; a splats-only scene (no meshes) shadows
  correctly in both modes; a point light ignores a blocker beyond the light. Full discovery
  1034 OK (1 skipped), 751.7 s. The commit changed an existing assertion:
  `test_transparent_shadow_overdraw_respects_running_budget` now expects zero `Bvh.build` calls
  for an EMPTY scene. That is a real code change (the build is skipped when there are no
  triangles; ray mode gets a hand-made empty BVH) and empty-scene output is byte-identical, so
  the relaxed assertion is accepted. `RealSplatTests` now takes its capture path from
  `NODEBASED_REAL_SPLAT`; no machine path is left in the repo.

## Splats cull near-camera and viewport instance helpers on main (2026-09-19 10:35 PM PDT)

`main` moved `e158f95` -> `723ebc0` (lane commit `affde2d`, merged by Gonzo): splats with view depth < 0.2 are culled (3DGS reference); `splatshade.instance_colors` and `splatshade.instance_geometry` exposed for the viewport; `prepare_splats` shares them; 1,293 existing outputs byte-identical to `e158f95`. Full suite full39 passed 1041 tests in 735 s. Lane supervisor died mid-report after full39; Gonzo confirmed nothing was lost.

`main` moved `22348f8` -> `9f8e9e8` (2026-09-20 12:07 AM, fast-forward): splat work budget stage 1. The budget counts tile work
(splats binned to each 16x16 tile x that tile's pixels, edge tiles smaller), `SPLAT_WORK_BUDGET = 2_000_000_000`,
`SPLAT_REFERENCE_EVALS_PER_SECOND = 16_500_000`, bounding-box pairs remain only as an 8x pre-bin guard, and the refusal message
gives tile work, the budget, a rough seconds estimate and remedies. Written by GPT-6 Astra; Gonzo carved it out of the lane's mixed
snapshot `3b58964` (which also held half-finished shadows B2 and was never suite-validated; it stays unmerged). Review in a clean
worktree: `tile_work` equals an independent per-pixel brute-force count on five sizes including 1x1 and non-multiples of 16;
budget == tile_work is accepted and tile_work - 1 refused inside prepare; 24 renders (relight 0/1 with a shadowed light, raster and
raytrace, rgba/splats/depth) SHA-identical to `22348f8`; 89 targeted tests OK; full suite 1045 OK (1 skipped, the real-capture test)
in 742 s. Real 3,409,742-splat capture with the default budget: 640x360 accepted (1,286,943,104 evaluations, estimate 78 s),
1280x720 accepted (1,726,333,184, estimate 105 s), **1920x1080 refused** (2,338,982,656, estimate 142 s). The app still has no
within-frame progress, so refusal stays for 0.24.0; the progress callback (stage 2) is planned, not built.

Known limits at `2fdd8a1`: splats do NOT yet shadow meshes (step B2); no viewport relighting
(the viewport belongs to `gonzo/3d-ux`); shadowed relighting is slow on the CPU (200k splats,
one shadow light, 64x36: 34.7 s, nearly all of it shadow rays); a shadow ray that lands exactly
on the shared diagonal of a transparent card counts both triangles (0.25 instead of 0.5,
`TriangleSet` edge handling, predates this work); ray mode builds the mesh in float64 and
raster mode in float32, so a splat centre exactly on a shadow edge can flip between modes. The
splat hard requirement is still NOT met: relighting exists in the CPU final render only, no GPU
path. (Superseded 2026-09-20: the budget question is decided, tile work with a 2e9 default, see
the paragraph above; the viewport now shows splats as a proxy, see the next section.)

## GPU viewport, 3D node shapes, Nuke-style knobs and viewport splat proxy on main (2026-09-20 12:36 AM PDT)

`main` moved `7d52456` -> `591db7c` (12:22 AM, fast-forward) -> `d876f57` (12:36 AM, fast-forward). This is
`gonzo/3d-ux` plus `gonzo/viewport-splats`, written by Gonzo on Fable 5.1 while the Astra lane sat on a Codex usage limit.
Release-gate item "gonzo/3d-ux merged" is done.

- `314c40b` interactive wgpu 3D viewport with cached meshes, one draw per object, CPU fallback; `82aa350` contiguous frame at
  any width; `4e21a53` test that the GPU viewport never reaches `scene3d.render`.
- `785704d` 3D nodes have rounded ends; Scene3D, Light3D and cameras are circles with rim sockets.
- `627623d`, `4d96ea2` 3D panels follow Nuke: XYZ rows with a keyframe diamond per axis, typed fields for sizes and open-ended
  magnitudes, sliders only for bounded scalars.
- `591db7c` splats in the 3D viewport as a LAYOUT PROXY: opaque SH-DC colour discs on the GPU, relit per splat in the vertex
  shader, at most 1M per cloud by stride, radius clamped to 1..2.5 px. The CPU fallback draws the meshes and marks up to 200k
  splat centres as 2x2 depth-tested points, ignores `Relight`, and never runs the splat rasterizer.
- `d876f57` F frames the 10th..90th percentile box widened by a quarter (the 2nd..98th box was 16x the subject on the Nelson
  capture), HUD text sits on a dark strip over splats, the million-disc frame-time test skips on software adapters. Its
  TASKLOG entry says 12:25 AM; the work was committed at 12:23 AM.

Evidence: full suite at the stacked tip `591db7c`, 1084 OK (1 skipped); full suite again at `d876f57`, 1084 OK (1 skipped) in
752 s, tree clean and HEAD unchanged when it finished. Real `Window` driven offscreen with real wgpu on the 3080 Ti, ReadSplat3D
on the 3.4M-splat capture plus cube, light and camera: GPU 2.7 ms per orbit frame, CPU fallback 63 ms, occlusion correct both
ways, `splatraster.prepare_splats` never called by the viewport.

Limits, stated plainly: the viewport splats are a proxy, not the render. No Gaussian footprint, no alpha blending, no
view-dependent SH, no shadows on splats; above 1M splats a cloud is strided. Nobody has driven it on a physical display, and it
has not run on Windows beyond CI, where GPU tests skip without an adapter. The final render of splats is still CPU-only and
refuses the real capture at 1920x1080 with the default budget. Shadows B2 (splats shadowing meshes) is still unmerged: the
lane's `a9acaf3` is a half-finished snapshot with a wrong commit message and must not be merged as is.

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

## 3D UX requirements (from DiMo, 2026-09-19 13:37 PDT)

Standing direction for every 3D node, present and future. Reference: how Nuke does it.

- **Knobs are not all sliders.** A geometry node such as Sphere3D carries general transform
  information (translate / rotate / scale as XYZ numeric fields, uniform scale, rotation and
  transform order, pivot), polygon amount (rows and columns), how the poles are treated, and
  local and world matrix values. Sliders stay only where a bounded scalar is natural.
- **Node shapes:** 3D nodes have rounded edges; Scene3D, Light3D and Camera3D are full circles.
- **Performance:** the 3D system must feel extremely lightweight and snappy. Measured
  2026-09-19 at 960x600 on the 3080 Ti box: the viewport paints through the CPU reference
  rasterizer on the UI thread, 29 ms for a card and a cube, 122 ms with one 32-segment sphere,
  344 ms with a 64-segment sphere. The wgpu final-render backend is no substitute (40 ms for
  the same sphere scene, 1.8 s at 65k triangles) because it loops over triangles in Python and
  issues one draw call per triangle for CPU parity. The viewport needs its own GPU path: cached
  per-geometry buffers, one draw per object, depth buffer, camera uniform only while orbiting.

Owner: Gonzo (GPU/display work), branch `gonzo/3d-ux`.

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
