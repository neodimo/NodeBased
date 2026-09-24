# Current state — 2026-09-22

## v0.26.0 published and verified (7:50 PM on 2026-09-23 PDT)

Release commit `816c16e` (bump to 0.26.0, notes prepended, bundled copy byte-identical), full suite at
that exact commit: Ran 1504 tests in 1102.312 s, OK (skipped=1), exit 0. Annotated tag `v0.26.0` pushed
6:53 PM with the lanes tick paused. Tag runs all green: Build release packages 35944886591 (publish
7:50 PM), Desktop conformance 35944886544 on the tag and 35944886592 on main, Linux and Windows both.
The Windows red on `f7cf948` (one `subprocess.TimeoutExpired` in
`test_time.SequenceTests.test_headless_agent_renders_requested_frame`, 30 s limit on a slow runner)
did not recur on the tag or on `816c16e`.

Published, not draft, not prerelease: https://github.com/neodimo/NodeBased/releases/tag/v0.26.0.
Assets: `NodeBased-0.26.0-linux-x86_64.AppImage` 139,385,336 B, `NodeBased-0.26.0-windows-x64-setup.exe`
68,839,562 B, `NodeBased-0.26.0-windows-x64-portable.zip` 99,918,027 B, `SHA256SUMS` 318 B; all four
answer HTTP 200. No media, per the slow-burn rule; the 3D gizmo clip is the one candidate.

Merged after the tag, so they are 0.27 as the notes say: L1 step 4 (camera and light handles,
`8680feb`) and L2 step 2c4 (TimeOffset, FrameHold, Retime, `903a99c`), docs commit `a691c2e`.

Not verified: packages not downloaded or launched; SHA256SUMS not checked against downloaded
binaries; no run on a real display or a real Windows GPU.

## Continuous mode merge: Lane 1 (viewer handles), Lane 2 (2D parity) (7:35 PM on 2026-09-23 PDT)

`main` moved `816c16e` -> `903a99c` (lane commits cherry-picked onto main in lane order) and then to this
docs commit, by the continuous-lane integrator tick (`scratch/nb-lanes/auto/tick.py` in Gonzo's
workspace; mode approved by DiMo on 2026-09-23 at 2:39 PM PDT).

**Evidence.** Integrator's independent targeted rerun on the stacked tree: Ran 50 tests in 30.008 s, OK. Full suite on
the stacked tip `903a99c` (`/tmp/nb-auto/integ-auto-0923-1905.log`, started 7:05 PM): **Ran 1531 tests in 1206.951 s, OK (skipped=1), exit 0**.

**What landed.**

- **Lane 1 (viewer handles), step 4 of 4: camera and light handles.** Commits:
  - `407a7e4` L1 step 4: camera and light position/target handles
  Diff: 4 files changed, 490 insertions(+), 27 deletions(-).
  Plain description: the lane's report file under /tmp/nb-auto and issue #1.
- **Lane 2 (2D parity), step 2c4 of 4: three time nodes.** Commits:
  - `3e5e243` L2 step 2c4: TimeOffset, FrameHold, Retime for Nuke parity
  Diff: 10 files changed, 430 insertions(+), 19 deletions(-).
  Plain description: the lane's report file under /tmp/nb-auto and issue #2.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA on the
real display owed by Gonzo. Lane-reported limits are in each lane's report file and issue.

## Continuous mode merge: Lane 1 (viewer handles) (6:25 PM on 2026-09-23 PDT)

`main` moved `38e718d` -> `367739c` (lane commits cherry-picked onto main in lane order) and then to this
docs commit, by the continuous-lane integrator tick (`scratch/nb-lanes/auto/tick.py` in Gonzo's
workspace; mode approved by DiMo on 2026-09-23 at 2:39 PM PDT).

**Evidence.** Integrator's independent targeted rerun on the stacked tree: Ran 37 tests in 21.242 s, OK. Full suite on
the stacked tip `367739c` (`/tmp/nb-auto/integ-auto-0923-1805.log`, started 6:05 PM): **Ran 1504 tests in 1078.888 s, OK (skipped=1), exit 0**.

**What landed.**

- **Lane 1 (viewer handles), step 3b of 4: rotate and scale gizmos (fix in progress).** Commits:
  - `ab7d72f` L1 step 3b fix: drop edge-on rings from the rotate gizmo hit test
  - `846745a` L1 step 3b: 3D rotate and scale gizmos
  Diff: 5 files changed, 525 insertions(+), 41 deletions(-).
  Plain description: the lane's report file under /tmp/nb-auto and issue #1.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA on the
real display owed by Gonzo. Lane-reported limits are in each lane's report file and issue.

## Continuous mode merge: Lane 2 (2D parity) (5:55 PM on 2026-09-23 PDT)

`main` moved `a2feaa3` -> `fe3074c` (lane commits cherry-picked onto main in lane order) and then to this
docs commit, by the continuous-lane integrator tick (`scratch/nb-lanes/auto/tick.py` in Gonzo's
workspace; mode approved by DiMo on 2026-09-23 at 2:39 PM PDT).

**Evidence.** Integrator's independent targeted rerun on the stacked tree: Ran 39 tests in 13.468 s, OK. Full suite on
the stacked tip `fe3074c` (`/tmp/nb-auto/integ-auto-0923-1735.log`, started 5:35 PM): **Ran 1483 tests in 1035.056 s, OK (skipped=1), exit 0**.

**What landed.**

- **Lane 2 (2D parity), step 2c3 of 4: three keyer nodes.** Commits:
  - `02fd755` L2 step 2c3: Difference node for Nuke parity
  - `aba7573` L2 step 2c3: HueKeyer node for Nuke parity
  - `38703f4` L2 step 2c3: Keyer node for Nuke parity
  Diff: 10 files changed, 530 insertions(+), 17 deletions(-).
  Plain description: the lane's report file under /tmp/nb-auto and issue #2.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA on the
real display owed by Gonzo. Lane-reported limits are in each lane's report file and issue.

## Continuous mode merge: Lane 2 (2D parity) (4:25 PM on 2026-09-23 PDT)

`main` moved `8306454` -> `8b9609d` (lane commits cherry-picked onto main in lane order) and then to this
docs commit, by the continuous-lane integrator tick (`scratch/nb-lanes/auto/tick.py` in Gonzo's
workspace; mode approved by DiMo on 2026-09-23 at 2:39 PM PDT).

**Evidence.** Integrator's independent targeted rerun on the stacked tree: Ran 78 tests in 11.376 s, OK. Full suite on
the stacked tip `8b9609d` (`/tmp/nb-auto/integ-auto-0923-1605.log`, started 4:05 PM): **Ran 1460 tests in 1003.491 s, OK (skipped=1), exit 0**.

**What landed.**

- **Lane 2 (2D parity), step 2c2 of 4: five draw nodes.** Commits:
  - `27099b2` L2 step 2c2: five draw nodes for Nuke parity (Ramp, Radial, Rectangle, Noise, Text)
  Diff: 12 files changed, 709 insertions(+), 21 deletions(-).
  Plain description: see the lane's report in #nodebased and issue #2.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA on the
real display owed by Gonzo. Lane-reported limits are in each lane's #nodebased report and issue.

## Continuous mode merge: Lane 1 (viewer handles) (3:55 PM on 2026-09-23 PDT)

`main` moved `f593ab5` -> `d0c1a2a` (lane commits cherry-picked onto main in lane order) and then to this
docs commit, by the continuous-lane integrator tick (`scratch/nb-lanes/auto/tick.py` in Gonzo's
workspace; mode approved by DiMo on 2026-09-23 at 2:39 PM PDT).

**Evidence.** Integrator's independent targeted rerun on the stacked tree: Ran 58 tests in 21.815 s, OK. Full suite on
the stacked tip `d0c1a2a` (`/tmp/nb-auto/integ-auto-0923-1535.log`, started 3:35 PM): **Ran 1441 tests in 997.159 s, OK (skipped=1), exit 0**.

**What landed.**

- **Lane 1 (viewer handles), step 3a of 4: translate gizmo and pivot mode.** Commits:
  - `109e8e8` L1 step 3a: 3D translate gizmo and pivot mode
  Diff: 5 files changed, 761 insertions(+), 4 deletions(-).
  Plain description: see the lane's report in #nodebased and issue #1.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA on the
real display owed by Gonzo. Lane-reported limits are in each lane's #nodebased report and issue.

## Continuous mode merge: Lane 2 (2D parity) (3:25 PM on 2026-09-23 PDT)

`main` moved `ccacd22` -> `a3ee5eb` (lane commits cherry-picked onto main in lane order) and then to this
docs commit, by the continuous-lane integrator tick (`scratch/nb-lanes/auto/tick.py` in Gonzo's
workspace; mode approved by DiMo on 2026-09-23 at 2:39 PM PDT).

**Evidence.** Integrator's independent targeted rerun on the stacked tree: Ran 79 tests in 14.024 s, OK. Full suite on
the stacked tip `a3ee5eb` (`/tmp/nb-auto/integ-auto-0923-1505.log`, started 3:05 PM): **Ran 1410 tests in 974.024 s, OK (skipped=1), exit 0**.

**What landed.**

- **Lane 2 (2D parity), step 2c1 of 4: six filter nodes.** Commits:
  - `e3719ed` L2 step 2c1: six filter nodes for Nuke parity (Erode, Dilate, Median, Sharpen, Glow, Mirror)
  Diff: 11 files changed, 576 insertions(+), 17 deletions(-).
  Plain description: see the lane's report in #nodebased and issue #2.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA on the
real display owed by Gonzo. Lane-reported limits are in each lane's #nodebased report and issue.

## L1 step 2 and L2 step 2b merged (2026-09-22/23, 10:31 AM on 2026-09-23 PDT)

`main` moved `cd37094` -> `0ee9198` (the stacked code tip: `openclaw/nb-viewer-handles` `ba3d940` then
`openclaw/nb-2d-parity` `a1c3667`, cherry-picked onto `main`) and then to this docs commit. First two
lane steps under the 2026-09-22 cadence: each lane was one Sonnet 5 session writing the code itself in
its worktree, one bounded run, targeted tests only, started 11:26 PM and finished by 11:41 PM.

**Evidence.**
- Lane targeted runs (their own claims, reported on the board): L1 26 tests OK at 11:41 PM; L2 126 tests
  OK at 11:39 PM.
- Integrator's independent rerun on the stacked tree (`tests.test_viewport3d_picking`,
  `test_handles3d`, `test_transform_handle_ui`, `test_2d_parity_group_b`, `test_phase_a`,
  `test_tileexec`, `test_bypass`, `test_knowledge`): **Ran 152 tests in 30.846 s, OK**.
- Full suite on the stacked tip `0ee9198` (`/tmp/nb-review/integ2-tip.log`, started 11:43 PM):
  **Ran 1390 tests in 917.242 s, OK (skipped=1), exit 0**, read at 10:30 AM on 2026-09-23 (the integrator wake that should have merged it at midnight died before it could).

**What landed.**

- **L1 step 2: 3D selection.** `nodebased/handles3d.py` (screen-to-ray as the inverse of
  `scene3d.project`, document-graph walk through `Scene3D` grouping, `Axis3D` parenting and
  `TransformGeo3D` baking to attribute each shape to its node, world-space bounds, nearest-hit ray test,
  bounds wireframe edges) and `Viewport3D`: a left click with under 3 px of motion picks the nearest
  object whose node carries a transform, selects it in the node graph (which opens its properties),
  and draws its world-space bounds box in an accent colour over the frame; a click on empty space
  clears; a drag orbits or pans and never picks; a selected node that disappears from the document
  clears the selection. Tests: `tests/test_handles3d.py` (11) and `tests/test_viewport3d_picking.py`
  (7, offscreen Qt).
  Limits: picking is always the CPU bounds test, whichever backend painted the frame (the wgpu
  viewport draws no object-id buffer); `ReadAlembic3D`, `ReadUSD3D`, `ReadGLTF3D`, `ReadSplat3D` and
  `Project3D` are walked over for parenting but are not pick targets; the outline is an axis-aligned
  bounds box, not a silhouette. Step 3 (gizmos) is next.
- **L2 step 2b: ten 2D nodes.** `Invert`, `Clamp`, `Multiply`, `Add`, `Gamma`, `Saturation` (single
  image, mask and mix, a `channels` selector on the first five, sharing Grade's and ColorCorrect's
  knob names and limits) and `Dissolve`, `Keymix`, `Copy`, `ChannelMerge` (two-input A/B nodes sharing
  Merge's mask, mix, bypass-passes-B and union-window convention via the new `MERGE_LIKE_KINDS`).
  Each has an evaluator kernel and a tile-path kernel asserted identical, region rules in `tiers.py`,
  knobs, theme colours, LIMITS/CHOICES, and a docs row flipped in `docs/PARITY_2D.md` (bundled copy
  byte-identical). Tests: `tests/test_2d_parity_group_b.py` (pixel assertions against hand-computed
  values, mask and mix, tile parity, bypass, spec coverage). Group (c) is next.

Limits: Linux only (RTX 3080 Ti); no Windows run; CI on the pushed commit not read; visual QA of 3D
selection and of the ten nodes on the real display still owed by Gonzo.

## Five lane steps merged in one stack (2026-09-22, 10:49 PM PDT)

`main` moved `572fd45` -> this docs commit, the one directly on top of `575bad4`: a fast-forward of the integration branch `integ-2026-09-22`, which
stacks, in this order, `openclaw/nb-viewer-handles` (2 commits), `openclaw/nb-2d-parity` (2),
`openclaw/nb-3d-astra-lane` (2), `openclaw/nb-particles` (2) and `openclaw/nb-fluids-spike` (1), followed
by this docs commit. All five lane branches sat on `572fd45`; the stack is their commits cherry-picked
onto each other. The only conflict was `TASKLOG.md` (the relight and particles lanes both added a section
at the top); both sections were kept.

**Evidence.** Full suite on each lane branch head, on `572fd45`, run by the integrator at 9:48 PM through
the three-slot wrapper (`/tmp/nb-review/<branch>.log`):

- `nb-viewer-handles` `597d856`: Ran 1315 tests in 903.463 s, OK (skipped=1), exit 0
- `nb-2d-parity` `09e3339`: Ran 1300 tests in 895.276 s, OK (skipped=1), exit 0
- `nb-3d-astra-lane` `2216479`: Ran 1314 tests in 890.211 s, OK (skipped=1), exit 0
- `nb-particles` `0e2dab6`: Ran 1306 tests in 843.811 s, OK (skipped=1), exit 0
- `nb-fluids-spike` `bcf6a0b`: Ran 1297 tests in 891.432 s, OK (skipped=1), exit 0

Full suite on the stacked tip `575bad4` (the code that merges; this docs commit changes only
`context/` and `TASKLOG.md`), `/tmp/nb-review/integ-tip.log`, started 10:29 PM: **Ran 1344 tests in 870.096 s, OK (skipped=1), exit 0**.

**What landed, per lane.**

- **Viewer handles (L1 step 1).** `nodebased/handles2d.py` (Qt-free geometry: forward map, pivot,
  box corners, edge midpoints, point-in-quad, pivot drag that preserves the rendered image, scale and
  rotate from a drag) and the `Viewer` integration in `app.py`: when a `Transform` is selected, the
  viewer draws its box, a dashed rotate ring, corner and edge scale handles and a round pivot marker;
  drag the box to translate, the ring to rotate, a corner or edge to scale uniformly, Ctrl-drag the
  pivot to move `center_x`/`center_y` with translation compensated so the image does not move; a
  numeric readout while dragging; one undo batch per drag; animated knobs are keyed; roto and tracker
  clicks keep priority; Escape cancels. Shortcut row added. Docs: "Transform viewer handle" section in
  `docs/ROTO_TRACKING.md` (bundled copy identical). Tests: `tests/test_handles2d.py` (10) and
  `tests/test_transform_handle_ui.py` (8, offscreen Qt: translate drag and undo, corner scales, ring
  rotates, Ctrl-drag pivot leaves the image unchanged, click does not edit, animated translate is
  keyed, tracker and roto priority). The lane's second commit is still titled "WIP ... pre-rebase
  checkpoint"; the content is the complete deliverable 1 of L1 and the title is left as history.
- **2D parity (L2 step 2a).** `Merge` gains `average`, `from` and `hypot`, completing Nuke's 19
  operations, on the evaluator and the tile path, with `docs/PARITY_2D.md` rows flipped.
- **Relight (L4 item 1, complete).** `Render3D` gains a `relight` output that carries a bundle of
  per-light diffuse and specular response layers (`Raster.layers`), CPU-only; a new 2D `Relight` node
  consumes it with ambient, diffuse, specular and mix knobs and up to eight `Light3D` inputs. Registered
  in `tiers.py` and excluded from the generic bypass sweep. Docs in `3D_FOUNDATION.md` and
  `3D_ROADMAP.md`. Written by GPT-6 Astra under a Sonnet supervisor.
- **Particles (L5 step 1).** `docs/SIMULATION.md` (time model, determinism rule, cache design, a
  `ParticleInstance` proposal for L3/L4, Nuke/Houdini knob mapping) and `nodebased/simcache.py`
  (memory plus disk LRU cache, `solve_to_frame`) with nine tests. No particle nodes yet.
- **Fluids (L6 step 1).** `docs/FLUIDS_SPIKE.md`: the sourced ecosystem read (OpenVDB, Mantaflow,
  PhiFlow, Taichi, Houdini Pyro, Nuke). Docs only; the solver spike is step 2.

**Held back.** `openclaw/nb-image-to-3d` (`164d243`, `6241b79`): both commits are self-titled WIP and the
lane worktree carries uncommitted edits to `docs/3D_FOUNDATION.md` (both copies) and an untracked
`docs/IMAGE_TO_3D.md`. Its suite result on the exact commit: Ran 1302 tests in 818.708 s, OK (skipped=1), exit 0 (`/tmp/nb-review/nb-image-to-3d.log`). It waits for the lane to
finish step 2 under the new cadence.

**Workflow change in this commit.** `context/lanes.md` standing rules now record DiMo's 2026-09-22
decisions: at most two lanes live, one bounded step per run, plain-English reports in #nodebased with
step N of M and a deliverable checklist, no timers and no automation, full suite once per merge
candidate, model routing (Luna/Terra for simple coding, Sol when needed, Astra only case by case, no GPT
models until Saturday 2026-09-26), and a progress board under evaluation. The first two lanes to resume
are L1 (viewer handles) and L2 (2D parity).

Limits: the lane branches were each tested alone on `572fd45` and the stack once at its tip; the
intermediate stacked commits were not individually re-tested. Linux only (RTX 3080 Ti); no Windows run,
CI on the pushed commit not yet read. Visual QA of the Transform handle on the real display is still
owed by Gonzo.

## Axis3D and TransformGeo3D merged (2026-09-22 12:52 PM PDT)

`main` moved `7e572e1` -> `fc9b644` (fast-forward of `openclaw/nb-3d-parity` after the lane rebased onto the current main tip; `origin/main` had not moved beyond `7e572e1`). Full suite at that exact commit in a clean temp worktree before the push: **Ran 1297 tests in 813.320 s, OK (skipped=1), exit 0** (`/tmp/nb-rv-l3/full.log`, with `flock /tmp/nb-gpu.lock`, `QT_QPA_PLATFORM=offscreen`, `PYTHONPATH=/tmp/nb-rv-l3`, shared `projects/nodebased/.venv`). The lane's own pre-commit log `/tmp/nb-l3/full3.log` reports the same count and verdict at 828.118 s; my independent run is in the same band.

`fc9b644` **3D node parity: Axis3D and TransformGeo3D (lane L3 step 1).** Nuke's `Axis` is a pure parenting transform; `TransformGeometry` bakes into vertices. Both land here as `Axis3D` and `TransformGeo3D`, sharing the existing `_XFORM` knob block (translate, rotate, rotate order, scale, uniform scale, pivot) that every geometry node and `Scene3D` already carry. `Axis3D` produces a scene; nothing wired produces an empty scene; the single `object` slot accepts geometry, light or scene (so it can parent any of them). `TransformGeo3D` requires a geometry on its `geo` slot; disabled passes the geometry through unbaked; the bake rewrites the geometry's own vertices and normals, with normals going through the inverse transpose and renormalised. Both ship with old-document upgrade (older files load and render unchanged because nothing in the old format referenced them).

Diff against `main`: `docs/3D_FOUNDATION.md +28, nodebased/core.py +23, nodebased/data/docs/3D_FOUNDATION.md +28, nodebased/imaging.py +19, nodebased/knobs.py +2, nodebased/scene3d.py +19, nodebased/theme.py +5-1, nodebased/tiers.py +2-1, tests/test_3d_axis_transform_geo.py +262` — all lane-owned per `context/lanes.md` (L3 owns the geometry primitives and hierarchy in `scene3d.py`, the 3D node entries in the shared registries, and `docs/3D_FOUNDATION.md`; `TASKLOG.md` was not touched here). Bundled doc copy is byte-identical (56,177 bytes; `tests.test_knowledge.test_bundled_docs_match_the_repository_docs` enforces it).

Targeted tests in the review worktree: `tests.test_3d_axis_transform_geo` — **18 OK in 0.081 s** (`Axis3DTests` 9 + `TransformGeo3DTests` 9).

My own repro at `/tmp/nb-rv-l3/myrepro.py`: SPECS, OUTPUT_TYPES, theme.py, tiers.py all carry `Axis3D` / `TransformGeo3D`; both SPECS entries share `_XFORM` (keys: `pivot_x`, `pivot_y`, `pivot_z`, `rot_order`, `rx`, `ry`, `rz`, `sx`, `sy`, `sz`, `tx`, `ty`, `tz`, `uscale`); `INPUT_TYPES["object"] == ("geometry", "light", "scene")` for Axis3D's slot and `INPUT_TYPES["geo"] == ("geometry",)` for TransformGeo3D's slot; `transform_geometry(card, translate=(3,0,0))` produces `mean_shift=(3.000,0.000,0.000)` (bakes translation into vertices); the matrix@pivot = pivot + tx property holds for the Nuke pivot convention (translate moves the object regardless of pivot; pivot is the rotation/scale centre — `matrix @ (2,0,0,1) = (7,0,0,1)` when tx=5); the identity transform leaves vertices byte-for-byte unchanged. **21 of 21 checks pass.**

Limits and honesty:
- The full suite at `fc9b644` runs on Linux (RTX 3080 Ti, wgpu-on-Vulkan, `rgba32float float32-blendable present`). No Windows run yet — the lane is still WIP for the rest of its step-2 work and the test file does not touch the runtime fixtures that the `e80d601`/`8d1681a` Windows fix covered.
- `Card3D.geometry_from_node` returns a `Geometry` whose `normals` attribute is `None` (card normals are face-derived at render time); the unit test for normal rotation under TransformGeo3D uses `Sphere3D` instead (`test_rotates_normals_through_the_inverse_transpose_and_keeps_them_unit_length`), and that test passes in the suite. My repro skips the normal-length check on cards and notes that the test suite covers it on spheres.
- The 1297 test count is the lane's claim; my independent run matches the count and the OK verdict.
- The matrix@pivot = pivot + tx property is asserted explicitly by the lane's `test_pivot_case_matches_the_hand_computed_matrix_and_moves_the_far_vertex` test (which passes in the suite) and reproduced by my repro.
- CI on `fc9b644` not yet read at merge time; that lands on the next watch.

L3 branch rebased onto the new main tip: `openclaw/nb-3d-parity` fast-forwarded to `fc9b644` (the lane was sitting at `1d17c09`, the rebase onto `7e572e1` produced `fc9b644`, and fast-forwarding the branch was a no-op). The lane's `STATUS.md` (last touched this morning, suite-green confirmation, "Standing down") stays as the lane-side handoff.

L4 / L5 / L6 / L7 / L1 / L2 status next watch: see `memory/2026-09-22.md` for the per-lane survey at this run's start; no other lane commit satisfied the merge gate in this run.

## Windows runtime-manager test fixture repair (2026-09-22)

Three Desktop conformance runs on `e80d601` / `8d1681a` failed only on Windows because two
runtime-manager tests created a Unix `venv/bin/python` fixture even though
`runtimes._python_path()` resolves `venv/Scripts/python.exe` on Windows. The runtime code was not
broken; the tests never reached their intended assertions. The fixture now asks `_python_path()`
for its executable location and places the fake command beside it. Local `tests.test_runtimes`
passes (11 tests). Windows CI is the platform proof still pending.

## Whole-package lane execution

DiMo expanded scope from only 2D-to-3D to viewer interaction, 2D and 3D nodes,
rendering/relighting, particles, fluids and image-to-3D. `context/lanes.md` is
the durable lane map. Suite concurrency is three shared runs; only GPU-exclusive
work uses the exclusive lock. Merge only exact-commit full-suite-green lane steps.
The 0.25.0 release is published; board state moved to whole-package parity.

## L5 step 1 committed on lane branch: simulation time model + disk cache (2026-09-22 8:26 AM PDT)

Lane branch `openclaw/nb-particles`, not yet merged to `main` (rebased onto `main` at
`572fd45` in this session; previously rested on `8f35b59`, the same base as the 2D
parity audit entry below). Commit `ac54038` **replaces** the temporary WIP squash
`823197e` noted in the 3:35 AM STATUS.md handoff (`git reset --soft 8f35b59` then one
real commit, per the lane's standing rule: no commit stands as "real" until the full
suite is green on that exact tree).

`ac54038` **L5 step 1: simulation time model, disk cache, tests.**
`docs/SIMULATION.md` (371 lines, byte-identical bundled copy under
`nodebased/data/docs/SIMULATION.md`): start frame / substeps-as-frame-unit-dt time
model, frame-by-frame forward solve contract (`state(f) = fold(step, substeps)` from the
last cached frame), determinism rule (`numpy.random.default_rng((seed, frame,
substep))`, no sequential generator state), disk-backed cache design (single global LRU
byte budget across memory + disk tiers, `run_key(upstream_digest, params)` identity,
corrupt-entry-is-a-miss, atomic temp+replace writes), a `ParticleInstance` proposal for
`scene3d.py` (explicitly flagged as an L3/L4 request, not shipped here), and a Nuke
ParticleSystem / Houdini POPs knob-vocabulary mapping table for milestone 2. `nodebased/
simcache.py` (267 lines): `SimCache` (memory+disk LRU, `run_key`, `State`, `get`/`put`/
`latest_at_or_before`, disk index rebuilt from `mtime`), `solve_to_frame` (forward-solve
loop, checkpoint every whole frame, cancel between substeps). `SimCache.shared()` wires
`NODEBASED_SIM_CACHE` / `NODEBASED_SIM_CACHE_MB` env overrides, mirroring `cachetier`.
`tests/test_simcache.py` (137 lines, 9 tests): determinism, restart (zero solver calls
on already-cached frames), invalidation, cancellation, budget eviction, before-start-
frame, run_key sensitivity, corrupt-entry-as-miss. `nodebased/knowledge.py` +
`tests/test_knowledge.py`: additive wiring for the new doc topic.

**Evidence:** full discovery on the original tree (`ac54038`, rebased onto `8f35b59`):
**Ran 1277 tests in 831.291 s, OK (skipped=1), exit 0** (`/tmp/nb-l5/full2.log`, under
`flock /tmp/nb-gpu.lock`, `QT_QPA_PLATFORM=offscreen`, shared `.venv`). Targeted run
earlier in the session (16/16, `test_simcache` + `test_knowledge`) plus a
`docs/SIMULATION.md` byte-identical diff check and a grep for home paths/emails (none
found) are recorded in `/tmp/nb-l5/STATUS.md`. Author's own evidence; no second
reviewer; CI not checked. A fresh full suite on the tree rebased onto `572fd45` is
queued this session (`/tmp/nb-l5/full4.log`); this note is being written ahead of that
run finishing and will be corrected below if the rebased-tree numbers differ.

**Limits:** the Houdini POPs column in the knob-mapping table is not independently
re-verified live (the fetch tool didn't return body text for the SideFX docs pages on
the day it was written); the Nuke column was read live from learn.foundry.com. The
`ParticleInstance` / `scene3d.py` wiring is a proposal only, not shipped — step 2 owns
building the smallest additive version itself if L3/L4 hasn't answered by the time it's
needed, per the lane's standing cross-lane rule.

**Next:** step 2 (`ParticleEmitter3D`, force nodes, `ParticleBounce3D`,
`ParticleCache3D`, CPU rendering through `Render3D`) starts once step 1 is rebased and
confirmed green on `main` at `572fd45` and merged. Do not start step 2 before that.

## On-demand uv runtime manager for SHARP merged (2026-09-22 5:52 AM PDT)

`main` moved `8f35b59` -> `e80d601` (fast-forward of `openclaw/nb-image-to-3d`; `origin/main` had not
moved). Full suite at that exact commit in a clean temp worktree before the push: **Ran 1279 tests
in 811.586 s, OK (skipped=1), exit 0** (`/tmp/nb-rv-l7/full.log`, with `flock /tmp/nb-gpu.lock`,
`QT_QPA_PLATFORM=offscreen`, `PYTHONPATH=/tmp/nb-rv-l7`, shared `projects/nodebased/.venv`). The
lane's own pre-commit log `/tmp/nb-l7/full2.log` reports 1279 OK skipped=1 in 829.192 s; my
independent run matches the same count and the OK verdict. CI on `e80d601` not read at merge time
— lands on the next watch.

`e80d601` **On-demand uv runtime manager for SHARP (L7 step 1).** New `nodebased/runtimes.py` (389
lines) and `tests/test_runtimes.py` (207 lines, 11 tests). `RECIPES['sharp']` pins the spike's
working recipe: Python 3.12, torch 2.14.0+cu130, gsplat 1.5.3 with `BUILD_NO_CUDA=1`, ml-sharp at
commit `aed6527`, the 2,809,738,232-byte checkpoint. Runtime lives under the user data directory
from `portable.py`, or an existing directory pointed at by `NODEBASED_RUNTIME_<NAME>` for dev /
tests. `status()` returns absent/installing/ready/broken with the reason; `install()` refuses up
front when `uv` is missing or the network is unreachable (a HEAD probe before any download),
reports progress through venv/packages/source/checkpoint stages via the existing progress-callback
pattern, honours cancellation by killing the subprocess and marking the runtime broken, verifies
the source archive and checkpoint against their pinned SHA-256 (with a path-traversal/symlink guard
on the tar extraction and a download size cap) before writing `ready.json`, and rejects `install()`
entirely when pointed at an external directory. `run(argv)` executes inside the runtime's venv as
a subprocess with a timeout, captured stdout/stderr/log file, and cancellation.

Diff against `main`: `TASKLOG.md +37, nodebased/runtimes.py +389, tests/test_runtimes.py +207` —
all lane-owned per `context/lanes.md` (L7 owns `nodebased/runtimes.py` and the `ImageToSplat` /
`ImageToMesh` / `WorldSculpt` node entries; `TASKLOG.md` is the shared per-change notes file). No
other lane's files touched.

My own repro at `/tmp/nb-rv-l7/repro.py`-style script verified: no `torch` side-effect on
`runtimes` import (a separate `dir(sys.modules)` check before and after); public surface
(`status`, `install`, `run`, `RECIPES`); the `sharp` recipe carries `python`, `packages`, `source`
(url + sha256), `checkpoint` (url + sha256 + size), and `name`; `install()` under a
`NODEBASED_RUNTIME_SHARP` override raises ValueError before any subprocess; `run('sharp', ...)` on
an absent runtime raises; `status('bogus')` raises KeyError. **0 fails across 9 checks.**

Targeted runtime tests in the review worktree (independent rerun):
`tests.test_runtimes.RuntimeTests` — **11 OK in 0.015 s**.

Limits and honesty:
- No live SHARP run was attempted here: `install()` would need network + `uv` + 2.8 GB of
  checkpoint bytes; `nodebased/runtimes.py` does not pull torch into the package (verified at
  import), but `install()` itself opens a connection. The 11 tests cover the install and run
  flows with fake recipes, subprocesses, and openers, and assert each state transition without
  any network call.
- `status()` under a `NODEBASED_RUNTIME_<NAME>` override that points at a non-existent directory
  returns `RuntimeStatus(state='absent')` rather than raising — this is the documented
  behaviour ("the runtime is absent from the user's perspective"), and the lane's
  `test_override_status_install_refusal_and_run` covers it. My initial repro assertion was too
  strict.
- L7's lane worktree (`/var/home/omid/.openclaw/worktrees/dddceeaf03a43b80/nb-image-to-3d`) has
  uncommitted step-2 WIP (modified `core.py`, `knobs.py`, `theme.py`, `tiers.py`, `imaging.py`,
  `docs/3D_FOUNDATION.md`, `nodebased/data/docs/3D_FOUNDATION.md`; new untracked
  `nodebased/imageto3d.py` and `tests/test_imageto3d.py`). The lane branch was NOT rebased onto
  the new main — the working tree is dirty and the lane owns its own rebase when step-2 commits.

Next on L7: `ImageToSplat` itself (step 2), running SHARP through this runtime manager,
caching by input digest + parameters, the real SHARP run under the GPU lock with timing/VRAM
in `docs/IMAGE_TO_3D.md`. Lane session `agent:main:dashboard:3f7e9c25-...` is alive; the next
watch will check for step-2 work and whether it is a merge candidate.

## 2D node parity audit merged (2026-09-22 2:25 AM PDT)

`main` moved `8b49dfa` -> `af84898` (fast-forward of `openclaw/nb-2d-parity`; `origin/main` had not moved).
Full suite at that exact commit before the push: **Ran 1268 tests in 811.228 s, OK (skipped=1), exit 0**
(`/tmp/nb-rv-l2/full.log`, with `flock /tmp/nb-gpu.lock`, `QT_QPA_PLATFORM=offscreen`,
`PYTHONPATH=/tmp/nb-rv-l2`). L1's `5130332` (handles2d geometry, 11:42 PM Sept 21) was older but
not merge-worthy: its commit message named targeted tests only (9 tests in `test_handles2d.py`),
and the L1 brief requires a green full suite on the exact tree before any merge.

`af84898` **2D node parity audit against Nuke 17 (`docs/PARITY_2D.md`).** 22,295-byte doc, 146 rows
across all 11 of Nuke's 2D toolbar groups (Image, Draw, Time, Channel, Color, Filter, Keyer,
Merge, Transform, Metadata, Other), ranked by judged daily use within each group, marked
supported / partial / missing against `main` at `8b49dfa`. Bundled copy in
`nodebased/data/docs/PARITY_2D.md` is byte-identical (verified by
`tests.test_knowledge.test_bundled_docs_match_the_repository_docs`). No code changed.

My own repro (`/tmp/nb-rv-l2/repro.py`-style script) verified the audit against current main:
the 14 supported rows all map to existing nodes in `core.SPECS`; the 8 partial rows have
real-but-degraded behaviour; the 103 missing rows correctly identify what NodeBased has
not yet shipped. **The Merge operations audit** is independently verifiable: the audit
claims plus/screen/multiply/difference/mask/stencil/under/atop/xor/max/min already exist,
and `MERGE_OPERATIONS` on main (`nodebased/core.py:218`) is exactly
`("over", "under", "plus", "minus", "multiply", "screen", "max", "min", "difference", "divide",
"mask", "stencil", "in", "out", "atop", "xor")` — all the audit's "already exists" claims
match, and `average` / `from` / `hypot` are correctly identified as missing (the lane's
step-2 build group).

Limits and honesty:
- Every claim in the doc was double-checked against the node set on `main` at `8b49dfa`, not
  against my memory of what shipped in what release. The "supported" rows trace to
  `core.SPECS` entries; "missing" rows name a Nuke reference doc page as the source for the
  audit's existence claim.
- The Merge operations audit is independently verifiable from `nodebased/core.py:218`. Other
  groups (Filter, Color, etc.) are based on the lane's reading of `core.SPECS` and the
  `nodebased/*.py` modules; they may have small omissions — a doc this dense will not catch
  every partial case, and reviewers should treat the missing column as a directional
  starting point rather than a completeness proof.
- The audit does not include every Nuke 2D node — it covers the 2D toolbar groups and
  explicitly excludes Nuke's 3D, Particles, Deep, and Views groups (those are L3, L5, L6).
  The numbering inside each group is a judgment of daily-use rank, not alphabetical.
- Author's own evidence; L2's brief says the audit ships first as the document the lane's
  later commits flip rows in. CI on `af84898` not read at merge time; that lands on the
  next watch if green.
- L2's lane session has uncommitted step-2 work (the average / from / hypot Merge
  operations, from `/tmp/nb-l2/STATUS.md`); the L2 branch was NOT rebased onto the new
  main because the working tree is dirty. The lane owns that rebase when its step-2
  commit is ready.

L2 session `agent:main:dashboard:90bbdfb8-...` is alive and working on step 2; no other
lane had a committed step ahead of main with a green full suite.

## All lanes open (2026-09-21, about 11:38 PM PDT)

DiMo's direction in #nodebased at 11:28 PM PDT widens the work from the 2D-to-3D pipe to the whole
package: the 2D and 3D node sets (parity with Nuke and parts of Houdini), the 2D and 3D viewers
(movable pivots, on-screen handles for moving objects in 2D and in X, Y and Z), particles and fluids,
and the 2D-to-3D pipe, all at once. This supersedes the 2026-09-20 4:46 PM scope directive.

The lane map is `context/lanes.md`: seven lanes (L1 viewer interaction, L2 2D node parity, L3 3D node
parity, L4 rendering and relighting on the Astra lane, L5 simulation foundation and particles, L6 fluids
solver spike, L7 2D-to-3D), the standing rules (green exact-commit full suite before any merge, Nuke
knob direction, file ownership per lane, the GPU lock `/tmp/nb-gpu.lock`), each lane's owned files and
first deliverables, and the 0.26.0 cadence. Lanes run as Sonnet 5 supervisors driving Codex workers in
their own worktrees; Gonzo integrates. The `nodebased-lanes-watch` automation reviews and merges
between Gonzo's turns.

Not verified at the time of writing: Desktop conformance on `0cd6a30` and `cf98f61` was still running.

## ReadGLTF3D merged (2026-09-21 10:59 PM PDT)

`main` moved `6fd7924` -> `0cd6a30` (fast-forward of `gonzo/gltf-reader`; `origin/main` had not moved). Full suite at
that exact commit before the push: Ran 1268 tests in 833.5 s, OK (skipped=1), exit 0 (`/tmp/nb-gltf-chain/full.log`).
Desktop conformance run 35692878369 on `0cd6a30` was in progress at 11:00 PM; its result is not recorded here yet.

`ReadGLTF3D` loads glTF 2.0 `.glb`/`.gltf` meshes (triangles, strips, fans; normals, UVs, baked node transforms,
quantized attributes, texture transforms) with an in-house NumPy reader in `nodebased/gltfio.py`, no extra package.
Base colour factors and base colour textures become the surface colour and texture, decoded sRGB into the working
space, with alpha honoured only for `BLEND`/`MASK`. Not loaded: points, lines, cameras, lights, skins, morph targets,
animations, vertex colours, and metallic/roughness/normal/occlusion/emissive maps; Draco, meshopt, KTX2 and sparse
accessors are refused by name.

Next is ImageToSplat (SHARP). The SHARP spike measured 13 s wall and 11.9 GB peak VRAM per image (1,179,648 splats),
and its output reads correctly through `splats.read_ply` with `orientation='colmap'`.

## v0.25.0 published and verified (2026-09-21 3:48 PM PDT)

<https://github.com/neodimo/NodeBased/releases/tag/v0.25.0>, annotated tag `v0.25.0` on `c307b8a`, published 3:47 PM
PDT; not a draft, not a prerelease. Full suite at `c307b8a` before any push: Ran 1241 tests in 812.9 s, OK (skipped=1),
exit 0. Tag runs green: Build release packages 35660691963, Desktop conformance 35660691978 (tag) and 35660690684 (main).

Assets, all uploaded with nonzero size: AppImage (138,906,104 bytes), Windows setup exe (68,619,953), portable zip
(99,511,725), SHA256SUMS (318, lists all three packages), and the 14 release-media files (`01-bypass.gif/.mp4`,
`02-splats-cpu-vs-gpu.gif/.mp4`, `03-gpu-raytrace-passes-sheet.png`, `04-catch-shadows.gif/.mp4/-timelapse.mp4`,
`05-progress-1080p.gif/.mp4/-timelapse.mp4`, `06-cast-shadows-pair.png`, `manifest-clips.json`,
`manifest-stills.json`). All 12 media URLs in the release body return 200 with the uploaded byte size.

Still true after 0.25.0: none of the GPU work has run on a real Windows GPU (CI is WARP/D3D12); the CPU-only paths listed
in the notes' known limits stay CPU-only. Post-release lane work waits for DiMo.

## Windows GPU parity gate cleared; splat shadows were dead on D3D12 (2026-09-21 5:02 AM PDT)

`main` moved `2a3c790` -> `e8938c3` (fast-forward of `gonzo/win-d3d12-splat-shadows`, cut from
`2a3c790`; `origin/main` had not moved). Full suite at that exact commit: **1241 OK, 1 skipped,
811.9 s, exit 0** (`/tmp/nb-half/full_d3d12.log`, 4:47 AM). This unblocks the 0.25.0 release gate
that the 10:40 PM packaging dry run failure created.

`279dce5` **GPU tests name their adapter.** `gpu3d.adapter_report()` prints device, adapter type,
backend, driver, wgpu version, the rgba target format and whether `float32-blendable` is present.
`tests/test_3d_gpu_adapter_report.py` puts it at the top of every GPU test log, so a parity number
can never again be read without knowing which adapter produced it. GitHub's Windows runner is
`Microsoft Basic Render Driver`, type CPU, **backend D3D12**, wgpu 0.32.0, target `rgba16float`.

`63c773d` **Splat shadows work on D3D12.** `gpurt_render`'s `splat_visibility` packs each caster BVH
node as three `vec4<f32>`: bounds in `xyz`, child link bitcast into `w`. A leaf's link is `-1`, whose
bits are a NaN. Indexing `low[a]`/`high[a]` dynamically let D3D12 carry that NaN into the
`d[a]==0.` axis comparison, so every leaf was rejected and **a light whose direction has an
exactly-zero component cast no splat shadow at all** — the plain unshadowed surface colour, out by up
to 0.48 against a 2e-3 tolerance. The loop now indexes `let lo=low.xyz; let hi=high.xyz;`.

The cause is identified rather than guessed: on the runner (diag run 35566349801), the baseline
shader gives 0.281615 error and shadow depth 0 for light `(1,0,3)` but 0.000200 for `(1,1e-3,3)`;
the `vec3`-copy shader gives 0.000200 for both; and the **untouched** shader with leaf links packed
as `0` instead of `-1` also gives 0.000200. Two independent NaN removals both fix it. A bitcast round
trip of `0xFFFFFFFF` through the storage buffer is clean, so the upload is not at fault.

This is a **D3D12 codegen problem, not a WARP problem**, so discrete Windows GPUs are in scope —
which is why the alternative of skipping the asserts on WARP was not taken. It never shipped:
`gpurt_render.py` does not exist in `v0.24.0`.

`e8938c3` **Splat parity tolerance follows the adapter.** The other 12 runner failures were half
precision, not a defect: without `float32-blendable`, `gpu3d` renders the rgba layers to
`rgba16float` while the parity tests compare against a float32 CPU reference. Reproduced here by
forcing `rgba16float` on an RTX 3080 Ti — the same tests fail at 8.5e-3 and 4.6e-3, including the
identical `test_baked_sizes_and_termination`. `tests/gpu_precision.py` widens the splat parity
tolerances **x20 only on a half-float target** and prints the reason once; float32 adapters keep
2e-3 / 2e-4 / 3e-3 unchanged. x20 comes from the runner's worst case, `test_stable_ties_and_termination`
at 2.71e-2 max and 1.67e-3 mean; x10 was tried first and failed at 0.0271 > 0.02.
**This widening is Gonzo's call, taken with DiMo told explicitly and able to veto it.**
`tests/test_3d_gpu_rt_splats.py` was left strict at 2e-3 — the ray tracer passes there unwidened.

Windows proof, temporary branch `gonzo/win-d3d12-verify` (deleted after): run 35595977177, the 17
failing tests plus the new ones **38 OK**, the six already-green GPU modules **77 OK**, no regression.

**Limits.** All evidence is Gonzo's; nobody reviewed the branch. No real Windows GPU has run this —
the measurements come from a D3D12 software adapter, and "discrete Windows GPUs were affected" is
inference from the backend, not a measurement. CI on `e8938c3` was not read before the merge. A
packaging dry run on `main` (35597198965) is the real release-gate proof and was still running.
`tests/test_3d_gpu_rt_splat_slab.py` guards the shader's shape rather than its behaviour, because no
adapter available here reproduces the miscompile.

**0.25.0 release notes must carry this**, alongside the bypass fix already noted: splat shadows on
Windows D3D12, and the fact that the parity gate now reports its adapter.

## GPU ray-traced milestones 3 and 4 merged (2026-09-20 9:09 PM PDT)

`main` moved `4f6c4ba` -> `749e092` (straight fast-forward of the lane's two `openclaw/nb-3d-astra-lane`
commits onto `4f6c4ba`; the lane was cut from `c4e7dca`, the only conflict was `TASKLOG.md`, and the
resolution kept the bypass-fix entry on top and added m4 above it after the m3 commit was applied).
Review worktree `/tmp/nb-rv-m3m4` (clean temp checkout off the rebased lane, shared
`projects/nodebased/.venv`, `PYTHONPATH=<worktree>`) was used.

`9473fb3` **GPU ray tracing milestone 3: every AOV output on the GPU tracer.** New
`gpurt_render.render(state, scene, camera, width, height, background, ambient, output, samples,
cancel)` for every output except `splats` (`render_beauty` kept): first-hit data passes `depth`,
`normals`, `position`, `uv`, `object_id` (nearest surface with alpha > 0, no AA, background
ignored), composited light-class passes `albedo`, `diffuse`, `specular`, `emission`. Data outputs
ignore `samples` and `background`; light outputs cap samples at 4 and the peel-shade-composite loop
takes the same path as `rgba`; the band dispatch now passes the output index in the params buffer.
`gpu3d.render(mode='raytrace')` accepts these for triangle-mesh scenes; `splats` in the scene,
`projection`, the `splats` output, and relit splats with a shadowed light still raise
`Unsupported` (`auto` -> CPU, `gpu` -> clear ValueError). Per-triangle `at[0, 3] = object_id` so the
index matches the geometry position, not the material offset (different mip counts made the old
material-offset scheme non-monotonic). Tests: `tests/test_3d_gpu_rt_aovs.py` (290 lines), plus a
routing test, a validation test, an unsupported-before-device-access test, and a one-line change to
`test_3d_gpu_rt_integration.py`'s unsupported-`depth` case (now asserts unsupported `splats`).

`749e092` **GPU ray tracing milestone 4: splat casters on the GPU tracer.** In `gpurt_render.py`:
casters built on the CPU as `scene3d._SplatCasters` builds them (read-only use; opacity *
opacity_scale * cast switch, 1/255 keep rule), uploaded once with their BVH (packing into the
existing eight storage bindings; capability bumped from 7 to 8 storage buffers per stage), GPU-side
ellipsoid transmittance multiplied into the mesh shadow rays (same formula, 3-sigma limit, 1e-3
cutoff, bias and light-distance rules, float32), the tracer's depth kept in the same pass, and the
visible splat layer composited with `gpusplat.render_layer` at the inner resolution.
`gpu3d.render(mode='raytrace')` routes rgba splat scenes with opaque meshes; `Unsupported`
unchanged for relit splats under a shadowed light, `shadow_catch` (evaluated first), transparent or
projected meshes with splats, non-rgba outputs with splats, the `splats` output. Tests:
`tests/test_3d_gpu_rt_splats.py` (150 lines): analytic single-splat transmittance, mesh-only,
splat-only, both, splat between two meshes, point + directional, distance limit, shadows-off spies,
cast-off == no casters, two instances, scales, a grazing thin-large splat, fallbacks, capability
and memory refusals, cancellation, band independence. The integration test changes the blanket
splat rejection into a data-pass rejection and counts eight bindings in the capability expectation.

Evidence at `749e092` (reviewer worktree, RTX-class adapter via wgpu's `max-storage-buffers-per-shader-stage: 8`):

- Targeted OK in 11.9 s: `tests.test_3d_gpu_rt_aovs`, `tests.test_3d_gpu_rt_splats`,
  `tests.test_3d_gpu_rt_integration`, `tests.test_3d_gpu_rt_render`, `tests.test_3d_gpu_rt`,
  `tests.test_3d_gpu_splats`, `tests.test_3d_gpu_tiles`, `tests.test_3d_gpu_shadows` =
  **93 OK**.
- Full discovery **1237 OK (1 skipped) in 812.6 s** (lane's claim on its own pre-bypass tree was
  1228 OK, 1 skipped, 802 s; the +9 is the bypass fix's `test_bypass.py`, expected).
  Log: `/tmp/nb-rv-m3m4-full.log`. `exit 0`.
- My repro `/tmp/nb-rv-m3m4-rpr/repro_m3m4.py` on the same adapter:

  **m3 AOVs on the documented set** (textured + lit + shadowed directional scene, 96x72,
  interior tolerance 2e-3, IoU >= .99): `rgba` 1.19e-7, `depth` 9.54e-7, `normals` 6.62e-24,
  `albedo` 1.79e-7, `diffuse` 5.96e-8, `specular` 3.86e-10, `emission` 3.73e-8, `position`
  2.38e-6, `uv` 4.17e-7, `object_id` 0 (exact). IoU 1.000 for every output.

  **m3 edge cases**: sphere (24 segs) + ground (`rgba` 4.99e-7, `depth` 3.34e-6, `normals`
  9.84e-7, `albedo` 1.79e-7, `object_id` 0 exact, all IoU 1.0); coincident stack of three
  cards at z=-.5, .3, 1 (`rgba`, `depth`, `normals`, `albedo` all 0 difference, IoU 1.0); a
  near-clipped textured 4x4 plane rotated 80 degrees (`rgba` 1.79e-7, `uv` 3.58e-7, `depth`
  1.43e-6, IoU 1.0); emissive-only scene at `emission` output (0 difference, IoU 1.0).

  **m4 splat casters** (interior tolerance 2e-3, IoU >= .99): 1,500 casters + visible splats
  over a floor 3.04e-6 (IoU 1.0); a single splat between two occluder cards 3.58e-7 (IoU 1.0);
  `cast_shadows=False` (the GPU path must drop the caster set) 1.25e-6 (IoU 1.0); cast-off on
  is byte-identical to cast-off off in the GPU path.

  **m4 timing rerun** (1920x1080, mesh floor + 200,000 casters + visible + shadowed
  directional light, this adapter): 3.15 s cold, 2.28 s warm, 2.28 s warm. Lane claimed
  2.8-3.1 s on the RTX 3080 Ti; mine is in the band.

  **Fallbacks** (must raise): transparent mesh + splats raises
  `transparent meshes mixed with splats are CPU-only`; relit splat + shadowed light raises
  `splat shadows are CPU-only`; splat data pass raises `splat data passes are CPU-only`;
  splats output raises `splats output is CPU-only`; shadow_catch on a relit splat raises
  `caught splat shadows are CPU-only`; bad output name raises `ValueError: Unknown 3D render
  output 'bogus'`. **0 fails across 27 checks.**

Honest limits at `749e092`:

- The capability check lists `max-storage-buffers-per-shader-stage >= 8`; an adapter with 7 or
  fewer storage buffers fails capability and routes to CPU. Milestone 2 already required 7; the
  bump to 8 is the caster pack. The memory refusal message names both memory numbers
  (caster buffer + device binding limit) as the lane intended.
- m3 covers `rgba` and the documented AOVs on triangle-mesh scenes with the GPU tracer; splats
  in the scene, projected geometry, and the `splats` output are still CPU-only.
- m4 covers rgba splat scenes with opaque meshes; relit splats with shadowed lights, splat
  shadows on relit splats, splat shadow catching, transparent or projected meshes mixed with
  splats, splat data passes, and the `splats` output are still CPU-only. The visible splat
  layer is composited at the inner (supersampled) resolution, then the box filter — same as the
  CPU composite path.
- Casters are built on the CPU exactly as `scene3d._SplatCasters` builds them; the lane's own
  packing test (`test_packing_matches_cpu_and_memory`) compares positions, opacity, rotation
  matrix, and inverse scale to the CPU records and asserts byte equality.
- Per-splat visibility (shadow catching on relit splats) stays on the CPU.
- Not yet tried on the real Nelson capture in the running app; the lane's llvmpipe + 3.4M-splat
  bench (44.7 s cold / 33.4 s warm, 310 MiB caster upload by scaling) is the closest analogue
  and is not a comparison against the CPU tracer.
- Not run on Windows.

0.25.0 queue from here (DiMo's 4:46 PM scope directive, unchanged): no more lane work for 0.25.0;
the Astra lane queue is complete. Release: tag, screenshots, video per Release policy. The
bypass-fix 0.24.0 bug and the Merge bypass behaviour change stay on the 0.25.0 notes list.

## Node bypass fixed and merged (2026-09-20 8:27 PM PDT)

`main` moved `c4e7dca` -> `45ff6fd` (straight fast-forward of `gonzo/bypass-fix`; `origin/main` had
not moved since the branch was cut, so no rebase was needed and the full-suite result below is for
the exact commit now on `main`). Bug report from DiMo at 8:02 PM against 0.24.0, with a screenshot.

**What was wrong** (reproduced headless before any change, `Checker -> Grade -> Merge.B`,
`Constant -> ColorCorrect -> Merge.A`):

- Bypassed Merge: the evaluator raised `KeyError: 'grade'` and the viewer printed the bare text
  `'grade'`; the tile path raised `IndexError`. `Evaluator.evaluate_raster` walked only the first
  input of a bypassed node, then built the kernel's input list from all declared slots and looked
  up values it had never computed.
- Bypassed Grade did nothing in the viewer: `TileExecutor._render_tile` gathered the passed-through
  input and then ran the node's kernel on it anyway (mean 0.78876 bypassed and enabled; the
  evaluator gave the correct passthrough 0.38469). The viewer uses the tile path.
- The rule "which input passes" was written six times (imaging x2, tileexec x4) and two copies
  disagreed (first slot vs first wired input). `_first_generator` ignored bypass.

**The fix:** one rule, `core.bypass_slot(node)`, used at every site in `imaging.py` and
`tileexec.py`. A bypassed node never runs its kernel on either path. `app.py`: internal
(non-ValueError/OSError) preview failures read "Internal error while evaluating (KeyError: ...)...
a NodeBased bug". `docs/AGENT_PROTOCOL.md` and the bundled copy updated.

**Behaviour change:** a bypassed Merge passes **B** (its background, as in Nuke), or A when B is
unwired. It used to try A. Project3D passes `geometry`; every other node its first declared input.

**Evidence at `45ff6fd`:**

- `tests/test_bypass.py`, 9 tests: the rule; every single-input filter bypassed equals its input
  bit-for-bit on the evaluator and, where tiled, on the tile path, and re-enabling restores it;
  bypassed Merge equals B on both paths even with A's branch broken; only-A-wired passes A; bypass
  mid-chain downstream of a Merge; mask ignored; toggling costs at most one cache miss; a real
  window toggling Merge and Grade off/on with no render error; the error wording. Three mutants
  killed (kernel still run on the tile path, Merge passing A, evaluator reading all slots).
- Full discovery **1211 OK (1 skipped) in 815.685 s**, finished 8:22 PM, exit 0 at `45ff6fd`.
  Log: `/tmp/nb-shc/full-bypass.log`. No existing test needed changing.
- Toggle latency in the window (960x540, offscreen Qt, this machine): first toggle 260-320 ms,
  later toggles 160-180 ms. The full-suite run printed merge off 271 ms / on 179 ms, grade off
  268 ms / on 179 ms.

**Limits and open item:** DiMo has not tried this in a build yet; the fix is on `main` only, no
release carries it. About 125 ms of each toggle is the CPU display conversion (`to_qimage`) that
runs on every viewer update, plus the 35 ms preview debounce; evaluation itself is a cache lookup.
That conversion cost is untouched. Proposed as the next item after 0.25.0's already-started work,
unless DiMo pulls it in sooner.

**0.25.0 release notes must list:** bypass fixed as a 0.24.0 bug, and the Merge bypass behaviour
change (passes B).


## GPU ray-traced milestone 2 merged (2026-09-20 7:10 PM PDT)

`main` moved `9aa4895` -> `cf2ce2a` (straight fast-forward of the lane's `openclaw/nb-3d-astra-lane`
commit onto `9aa4895`; lane rebased cleanly, no conflicts). The review worktree at `/tmp/nb-rv-m2`
(clean temp checkout off the rebased lane, shared `projects/nodebased/.venv`, `PYTHONPATH=<worktree>`)
was used.

`cf2ce2a` **GPU ray tracing milestone 2: `gpurt_render.py` shades and composites peeled hits on
the GPU** (textures/mips, lights, specular, emission, BVH shadows); `Render3D` `Mode` `raytrace`
runs on `Backend` `gpu`/`auto` for `rgba` triangle meshes; 70-130x faster than the CPU tracer at
1080p on RTX 3080 Ti. New module `nodebased/gpurt_render.py` (317 lines): host prep
(world triangles + CPU's BVH, per-triangle attributes, geometry table, all textures with their
CPU mip chains packed in one float32 buffer, lights table) and a compute kernel with the
peel-shade-composite loop per ray, hits kept on the GPU, shadow rays through the same BVH;
row bands of `GPU_RT_RAYS_PER_SUBMISSION` (1<<19) with cancellation between bands;
`gpu3d._state` requests up to 8 storage buffers per stage (was 4); `gpu3d.render(mode='raytrace')`
routes supported scenes (rgba, no splats, no projection, capability ok) to it, everything
else raises `Unsupported` (auto -> CPU, gpu -> clear ValueError); `imaging.py` no longer
forces raytrace mode onto the CPU. Tests: `tests/test_3d_gpu_rt_render.py` (11) and
`tests/test_3d_gpu_rt_integration.py` (8) added; existing test changed:
`test_3d_raytrace_render.py::test_graph_cache_upgrade_choices_and_backends` (asserted 'ray-traced
mode is CPU-only for now'; now checks that raytrace mode is routed through `gpu3d.render` for
auto and gpu). Found on real hardware (per TASKLOG): the shared-edge duplicate rule (CPU
thresholds 1e-10) did not fire under the RTX's float32 rounding, so a card's two triangles
both shaded at a diagonal pixel. Thresholds widened to 1e-5 with a comment.

Evidence at `cf2ce2a`:

- Targeted OK in 16.3 s: `tests.test_3d_gpu_rt_render` 11, `tests.test_3d_gpu_rt_integration`
  5, `tests.test_3d_raytrace_render` 12, `tests.test_3d_gpu_rt` 8, `tests.test_3d_gpu_tiles` 11,
  `tests.test_3d_gpu_shadows` 25 = **72 OK**.
- Full discovery **1202 OK (1 skipped) in 791.975 s** (lane claimed 1202 OK, 1 skipped, 794 s).
  Log: `/tmp/nb-rv-m2-full.log`.
- My repro `/tmp/m2_repro.py` on the RTX 3080 Ti Vulkan adapter:
  - random soup 2000 tris at 96x72: IoU 1.000, interior max 0 (unlit) / 1.192e-7 (lit + shadowed).
  - 16x16 shared-edge grid + diagonal sweep at 96x96: 9216 covered pixels both sides, interior
    max 0. No edge cracks.
  - 12 coincident alpha-0.5 cards at z=0 (samples=2): IoU 1.000, interior max 5.96e-8.
  - 5 alpha-0.5 cards stacked (samples=2): IoU 1.000, interior max 5.96e-8.
  - 70 alpha-1.0 stacked: opaque-terminates at first surface, raises ValueError when MAX_HITS_PER_RAY
    exceeded (no ValueError raised here; only 1 surface per ray, well under 64).
  - sphere (1026 tris) + ground + directional shadowed light at 96x72: IoU 1.000, interior max 2.98e-8.
  - near-clipped textured ground (UV scale 16, mip select across near plane): IoU 1.000,
    interior max 1.79e-7.
  - timing rerun at 1920x1080 RTX 3080 Ti (lit, specular, shadowed sphere + ground):
    1026 tris **0.344 s** (6.03M px/s), 10002 tris **0.390 s** (5.32M px/s), 40002 tris
    **1.507 s** (1.38M px/s). Lane claimed 0.35 / 0.56 / 1.91 s; mine is at or better.

Honest limits at `cf2ce2a`:

- `gpu3d.render(mode='raytrace')` on the GPU only covers `rgba`; every AOV still routes to CPU.
  Splats in the scene still raise `Unsupported`; the splat casters milestone (4) wires those.
  Camera-projected geometry still raises `Unsupported`.
- The shared-edge duplicate rule thresholds widened to 1e-5 (from CPU's 1e-10) for float32.
  On the RTX the rule fires correctly at shared edges; on lower-precision GPUs it may
  misfire more often. The test asserts interior error <= 2e-3 (silhouette IoU >= .99).
- Needs seven compute storage buffers per stage; the device request was raised from 4 to 8.
  Adapters reporting fewer than 7 fail capability and route to CPU.
- Capability / memory checks fail with clear `gpu3d.Unsupported(reason)` strings; `auto`
  falls back to CPU and `gpu` raises; both verified by integration tests.
- Not yet tried on the real Nelson capture in the running app (3.4M-splat scene has no
  triangle meshes, so this milestone does not affect it).
- Not run on Windows.

0.25.0 queue after this merge (DiMo's 4:46 PM scope directive, unchanged):

- Lane: GPU ray-traced milestone 3 (AOV outputs on the GPU tracer), then milestone 4 (splat
  cast-shadows). One commit per milestone after a green full suite.
- My loose ends: Gonzo lane for progress bar + ETA in the desktop app (already merged at
  `c53eac1`) and `gonzo/splat-cast-toggle` (merged at `398e809`); the next Gonzo lane item
  is `gonzo/splat-cast-toggle` rebased and merged (already done).
- Release: tag, screenshots, video per Release policy.

## GPU ray-traced milestone 1 + banded GPU submissions merged (2026-09-20 6:15 PM PDT)

`main` moved `c45c080` -> `03b3cf8` (straight fast-forward of the lane's two `openclaw/nb-3d-astra-lane` commits onto `c45c080`, no rebase needed). The review worktree at `/tmp/nb-rv-rt2` (clean temp checkout, shared `projects/nodebased/.venv`, `PYTHONPATH=<worktree>`) was used:

- `715c968` **Banded GPU submissions with mid-frame cancellation.** Per-adapter
  budgets bind one GPU submission; rows above the shadow budget are dispatched in
  bands of `GPU_MAX_BANDS = 64`, capped at the render height. HD shadowed renders above
  the previous ~19k-triangle refusal now run: RTX 3080 Ti at 1920x1080, 40k triangles 1.49 s,
  90k 3.53 s. `gpu3d._band_plan(state, work, path, height)` returns the contiguous
  `(start, stop)` row splits; cancelling between bands is safe. Pre/post cancel checks
  bound the actual work. The change is bit-identical to the single-submission path (the
  shader does not depend on band size; only the dispatch is banded). 11 new tests in
  `tests/test_3d_gpu_tiles.py`, including band-planning boundaries, force-bands, observed
  device ownership, mid-band cancel propagation, and an exact-raster parity run with
  bands (1-row, 20-row, single) and `GPU_FORCE_BANDS` overrides.

- `03b3cf8` **GPU ray tracing milestone 1: gpurt.py on wgpu.** New
  `nodebased/gpurt.py` (283 lines) computes BVH closest-hit and peeled nearest-K hits on
  the GPU; closest-hit agrees with the CPU `TriangleSet.nearest_hits` to relative depth
  2e-4, barycentrics 1e-3, inclusive 1e-6 barycentric edge band, |det| > 1e-10. Edges,
  inclusive boundary rays, and the `(t, primitive)` lexicographic peel cursor for
  `all_hits` are all preserved. Capability check requires 4 storage buffers per
  shader stage, 64 MiB minimums, compute workgroup 64, and a working compute pipeline
  on the live adapter; failure surfaces as `gpu3d.Unsupported(reason)` so `auto` falls
  back to CPU and `gpu` raises the same reason. Bounds/cursors/stack are f32; the
  result is exposed as f64. `primary_rays` is f32 and matches the renderer's actual
  inverse view basis. 19 new tests in `tests/test_3d_gpu_rt.py`; existing
  raytracing/CPU parity tests untouched.

Evidence at `03b3cf8`:

- Targeted: `tests.test_3d_gpu_rt` 9 OK in 2.5 s, `tests.test_3d_gpu_tiles` 11 OK
  in 1.7 s, `tests.test_3d_gpu_bvh` + `tests.test_3d_gpu_shadows` + `tests.test_3d_gpu_splats`
  all 30+ relevant tests OK. Plus the lane's "Shared" GPU tests under
  `tests.test_3d_gpu.GPUComparison` and `test_3d_gpu_splats.GPUSplats` all OK as part
  of the full run.
- Full discovery **1180 OK (1 skipped) in 790.792 s** (lane tip-2 commit message claimed
  1157 OK; the run on top of `c45c080` adds the new tests: `test_3d_gpu_rt.py` 9 +
  `test_3d_gpu_tiles.py` 11, plus a few extras ⇒ 1180 net). Log: `/tmp/nb-rv-rt2/full.log`.
- My repro `/tmp/nb-rt-gpurepro/repro.py` on the RTX 3080 Ti, 3080 Ti Vulkan
  adapter: random soups at 500 and 5000 triangles, k=1, 2048 rays: where both
  the CPU and GPU return a hit, the primitive matches in 100% of cases (518/518 and
  1228/1228), and `t` differs by a relative 1.91e-06 / 3.71e-06 (well inside the
  2e-4 tolerance). Edge cases (all in `/tmp/nb-rt-gpurepro/repro.py`):
  - 16x16 shared-edge grid with a diagonal sweep, 6450/6450 pixels, both sides
    report a hit on every pixel (no edge cracks).
  - 12 coincident alpha-1 cards at z=0: GPU `all_hits` returns primitives
    `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` on both rays, identical to CPU.
  - 70 cards at z=0..69 with alpha 0.5: GPU returns 20 hits per ray with
    primitive lists and `t` values identical to CPU (within rtol 2e-4).
- 3080 Ti timing rerun: 10k-triangle BVH at 1080p, 1M primary rays, k=1 single submit:
  **5.05M rays/s** = 198 ms; k=8 single submit: **1.17M rays/s** = 851 ms; full HD
  1920x1080 single submit (2,073,600 rays): **3.20M rays/s** = 648 ms. Falls inside the
  lane's claimed 2.7-5.4M rays/s range; consistent with the lane's `40k 1.49 s` and
  `90k 3.53 s` HD shadowed render claims, where the wall-clock is dominated by
  the shadow budget at the same BVH traversal cost.

Honest limits at `03b3cf8`:

- `gpurt.nearest_hits`/`all_hits` is the milestone-1 wire; the integration into
  `gpu3d.render` (primary rays + closest-hit shading per pixel) is the next slice.
- The capability check lists `max-storage-buffers-per-shader-stage` >= 4; an
  adapter with 4 or more storage buffers and a working compute pipeline is fine.
  The lane's own failing-storage-buffer test had been raised to 4 by an earlier
  commit (see context/state.md "GPU BVH traversal" note from 2026-09-19 12:25 PM);
  this commit does not relax that.
- Barycentric coordinates may belong to different bases when edge ties select
  different triangles — the test asserts primitives match on edge ties, not
  `(u, v)`. Matches CPU.
- Relit splats, splats shading meshes, transparent layered AOVs, every data pass
  with splats, the `splats` output, shadow rays that need bands across shadow
  budgets — all unchanged. The GPU ray tracer milestone 1 only covers closest-hit
  + peeled nearest-K; shading port and AOVs are milestones 2 and 3 in the Omid
  4:46 PM scope.
- `gpurt` capability / memory checks fail with clear `gpu3d.Unsupported(reason)`
  strings; `auto` falls back to CPU and `gpu` raises; both verified.

Queue, in the order DiMo approved at 4:46 PM: GPU ray-traced mode milestone 2
(shading port, hits kept on the GPU), milestone 3 (AOV outputs), milestone 4
(splat cast-shadows), one commit per milestone after a green full suite.

## ReadSplat3D Cast shadows on/off merged (2026-09-20 5:28 PM PDT)

`main` moved `6354c07` -> `398e809` (straight fast-forward of `gonzo/splat-cast-toggle`). The branch
held uncommitted work on the old first catcher commit `c3db60c`; it was committed as WIP, rebased onto
`6354c07` (the superseded `c3db60c` dropped with `rebase --skip`), finished with tests and docs, and
amended into one commit. `ReadSplat3D` has a `Cast shadows` choice (`splat_cast_shadows`, `on`/`off`,
default `on`, older documents upgrade to `on`; `SplatInstance.cast_shadows`). `off` is for environment
captures whose sky shell or walls otherwise block every light: the instance leaves the shadow casters
(caster opacity zeroed in `_SplatCasters`, so receiver slots and offsets stay valid), still renders,
still receives when relit and still catches mesh shadows. The flag is in the caster cache key; an empty
caster set short-circuits to full visibility; non-casters cost nothing in the splat shadow budget; when
nothing casts and nothing is relit no splat BVH is built.

Evidence at `398e809` (worktree `/tmp/nb-cast`, shared `.venv`): full discovery 1160 OK (1 skipped) in
809 s, log `/tmp/nb-shc/full-cast.log`, exit line `exit 0 398e809 05:25 PM`; `origin/main` had not moved
during the run. `tests/test_3d_splat_cast_toggle.py` has 7 tests: default equals the old constructor byte
for byte; off equals the no-splat render byte for byte in raster and raytrace while on darkens more than
100 pixels; off builds no caster set or BVH and passes a budget of 1, and a 50-splat shell beside a relit
splat only counts while it casts; a relit splat ignores a non-caster and a non-casting relit splat still
goes black under a card; with two casters only the one left on casts; flipping the switch on a warm
cache changes the result both ways; knob, validation and old-document upgrade. Three mutants fail the
tests (flag out of the cache key, casters ignoring the flag, budget counting non-casters). Author's own
evidence; no second reviewer; CI on `398e809` not checked in this run.

Limits: all or nothing per node; the GPU path still refuses splats + meshes + a shadowed light even when
nothing casts, so `auto` renders that on the CPU; not tried on the real capture yet.

0.25.0 started-work list after this merge: the progress UI and the cast toggle are done. Left: GPU ray
tracer milestones 2-4 (Astra lane) and, before them, the watch's merge of `1bc0478` + `b58db92`, which
were still outside `main` at 5:28 PM (the watch had a second review suite running in `/tmp/nb-rv-rt2`
from about 5:10 PM). Then the release-media pass and the release itself.

## Progress bar + time left in the desktop app merged; 0.25.0 scope narrowed to started work (2026-09-20 5:12 PM PDT)

`main` moved `dd37918` -> `c53eac1` (straight fast-forward of `gonzo/progress-ui`; `origin/main`
was still `dd37918` after a fetch at 5:09 PM, so no rebase and no targeted rerun). The desktop
app now passes a progress callback into slow CPU splat frames: new `nodebased/renderprogress.py`
(`ThreadProgress`, installed once as `Evaluator.progress`, routes events to the handler the
calling thread registered; `progress_text`, `format_duration`); `app.py` has a `QProgressBar`
in the status bar, a `progress` signal from the preview worker whose callback raises
`Cancelled` when the request was superseded (so a cancel lands between tiles), the same
generation guard as interim pictures, the bar hidden when the request ends, and the
single-image export reporting on the GUI thread. Because the router is always installed,
every render the window starts is interactive and is no longer refused by the splat budget;
the agent CLI and batch evaluators have no callback and still refuse.

Evidence at `c53eac1` (worktree `/tmp/nb-progress`, shared `.venv`): full discovery 1153 OK
(1 skipped) in 809 s, log `/tmp/nb-shc/full-progress.log`, exit line `exit 0 c53eac1 05:04 PM`.
`tests/test_render_progress.py` has 6 tests: wording and rounding; per-thread routing with a
second thread, nesting and an exception; a window renders a splat frame with
`SPLAT_WORK_BUDGET` patched to 1 (refused without a callback), sees prepare/splats/done,
shows the bar with "Rendering splats" while it runs and hides it after; stale and cancelled
requests cannot move the bar. One mutant (router not installed) fails the window test with a
refusal. This is the author's own evidence; nobody else has reviewed the branch, and CI on
`c53eac1` was not checked in this run.

Limits: no bar inside a frame for thumbnails or sequence writes; GPU renders report nothing;
the ETA is the lane's estimator, uncalibrated by scene type; not yet tried on the real
capture in the running app (that happens in the release-media pass).

Not in `main` at 5:12 PM: the lane's `1bc0478` (banded GPU submissions) and `b58db92` (GPU ray
tracing milestone 1). The status watch reviewed both and its review suite was green at
4:36 PM (1157 OK, 1 skipped, `/tmp/nb-rv-rt/full.log`), but the merge it was expected to make
around 4:55 PM has not happened. The watch owns that merge; it will now need a rebase onto
`c53eac1` (expected conflicts: `TASKLOG.md` and the two `3D_FOUNDATION.md` copies).

### Scope directive for 0.25.0 (from DiMo, 2026-09-20 4:46 PM PDT)

DiMo: wrap up started work before going full steam on 2D-to-3D. So 0.25.0 = finish started
work only, and nothing new is queued.

- In (started): GPU ray tracer milestones 2-4 (Astra lane, stacked on `b58db92`); this
  progress UI (done, `c53eac1`); `gonzo/splat-cast-toggle` (worktree `/tmp/nb-cast`,
  uncommitted work on the old `c3db60c`, to be rebased onto `main`, finished with tests, full
  suite, merged).
- Out unless DiMo says otherwise (not started): relight passes and a 2D Relight node, normals,
  shadow offset/blur, kept specular, `WriteSplat3D`, sphere rows/columns, matrix readout,
  multichannel EXR, particles, volumes, a glTF/GLB reader, and all 2D-to-3D work (SHARP image
  to splat, WorldSculpt/Pixal3D splat to meshes). 2D-to-3D stays parked until 0.25.0 ships.
- Estimate given to DiMo at 4:47 PM: Monday 2026-09-21 around noon; best case about 1 AM if
  the lane does not hit its Codex limit overnight. An estimate, with the media gate below
  still inside it.
- Cut line: if the GPU shading port balloons, ship with raytrace mode on the CPU and
  `gpurt.py` documented as groundwork.
- The release-media gate (stills and clips from the real app, see Release policy) still
  applies to 0.25.0.

## Shadow catcher at Relight 0 merged; shadow cache recorded (2026-09-20 3:45 PM PDT)

`main` moved `e90e47b` -> `5c82999` (straight fast-forward of `gonzo/shadow-catcher`, which
had been rebased onto `e90e47b` at 3:20 PM; the first version `c3db60c` was green but
conflicted with stage 2). `ReadSplat3D` has a `Catch shadows` slider (`splat_shadow_catch`,
0..1, default 0, older documents upgrade to 0; `SplatInstance.shadow_catch`). The captured
colour is multiplied by `1 - strength * (1 - lit/total)`, where lit/total comes from
ambient, light intensity x luminance and mesh transmittance, applied before the `Relight`
blend, so a capture keeps its look and still takes shadows from CG meshes. Only meshes
cast; the rays start at visible splat centres and are cached by cloud, node matrix,
meshes and light geometry. The splat caster BVH is now built lazily, so a catch-only
render never builds it. Catch rays have their own budget refusal. The GPU path refuses
the combination explicitly (`Unsupported('caught splat shadows are CPU-only')`, stated as
its own clause in `gpu3d.py` with a case in `tests/test_3d_gpu_splat_render.py`), so
`auto` falls back to the CPU and `gpu` errors.

Evidence at `5c82999` (worktree `/tmp/nb-catcher`, shared `.venv`): full discovery 1147 OK
(1 skipped) in 788 s, log `/tmp/nb-shc/full-catch2.log`, exit line carries `5c82999`.
`origin/main` had not moved during the run, so no second rebase and no rerun was needed.
`tests/test_3d_splat_shadow_catch.py` has 10 tests: off is byte-identical and traces
nothing; analytic factors for ambient, an unshadowed fill light, half strength, a
half-transparent card and a coloured sun; `Relight` .5 equals .5 caught + .5 relit and
`Relight` 1 ignores catching byte for byte; raster equals raytrace; alpha, data passes
and `shadows=False` untouched; cache invalidation per card, light and node, each variant
starting from a warm base cache. Two cache-key mutants (node matrix dropped, mesh dropped)
fail the tests. This is the author's own evidence; nobody else has reviewed the branch.

Limits: only meshes cast caught shadows; centre-sampled, so big soft splats blur the
shadow edge; CPU only; not shown in the viewport.

Owed from 12:12 PM, recorded here: `main` had moved `a4e3761` -> `8c0b797`
(`gonzo/splat-shadow-cache`). Splat shadows are cached between renders: a shared caster
BVH plus a per-splat visibility store keyed by casters (cloud identity, matrix, scale,
opacity), mesh occluders (blake2b of triangles + alpha + bias) and light geometry
(direction for directional, position for point). Colour, intensity, ambient, `Relight`
and the camera are outside the key, so changing them traces nothing. Caps: 2 caster
sets, 1 GiB casters (206 bytes per caster, 0.65 GiB for the 3.4M-splat capture), 256 MiB
visibility. 200k-splat shell at 640x360: 36.4 s cold, 5.3 s warm; a light move costs
34.8 s. The cold image equals the uncached build byte for byte. 9 new tests, 3 key
mutants killed, full suite 1105 OK (1 skipped) in 783 s (`/tmp/nb-shc/full.log`). Three
lane build-count tests now clear the cache first; no assertion was loosened.

Open: `gonzo/splat-cast-toggle` (worktree `/tmp/nb-cast`) holds uncommitted work stacked
on the old `c3db60c` and needs rebasing onto `5c82999`. Next for Gonzo's lane: wire
`Evaluator.progress` into the desktop UI (progress bar + ETA).

## Splat budget stage 2: progress callback + ETA for CPU splat frames (2026-09-20 2:23 PM PDT)

`main` moved `313b97f` -> `6d07fb1` (straight fast-forward of the lane's `6d07fb1`; the
lane was sitting on `313b97f` so there was no conflict): `scene3d.render(..., progress=cb)`
emits ordered events `("prepare", 0.0, {})`, `("splats", fraction, info)` with
`tile_work` + `estimate_seconds` on every update and `eta_seconds` after the first, then
`("done", 1.0, {...eta_seconds: 0.0})`; `accumulate_splats(..., progress=cb(done, total))`
in tile evaluations, at most ~200 per render, monotonic, last `(tile_work, tile_work)`,
band shares partition the full-frame `tile_work`; `prepare_splats(..., enforce_budget=True)`
(flag is `False` for the interactive path so both refusal guards are skipped but tile_work
is still measured); `estimate_seconds(tile_work)` and `estimate_eta_seconds(done, total,
elapsed)` (reference rate before 2% done, then linear extrapolation); `Evaluator.progress`
(default `None`, forwarded to CPU `Render3D` only, cache hits silent). A render with a
progress callback is interactive: the splat work budget does not refuse it; shadow and
mesh budgets are unchanged. `imaging.py` threads `self.progress` into the CPU render call
only; 7 new tests in `tests/test_3d_splat_progress.py`.

Review evidence at `6d07fb1` (clean temp worktree `/tmp/nb-rv-stage2`, shared
`projects/nodebased/.venv`, `PYTHONPATH=<worktree>` for standalone scripts):

- Targeted: 142 tests OK in 12.8 s, 1 skipped (the four relevant modules plus
  `test_3d_splat_progress`, `test_3d_splat_relight`, `test_3d_splat_shadows`,
  `test_3d_splat_shadow_cache`, `test_3d_splats`).
- Full discovery 1137 OK (1 skipped) in 762.6 s (lane's run 761 s).
- Reviewer repros (28 claims, `/tmp/nb-rv-stage2-repro/repro.py`): event shape and
  ordering correct (prepare first, done last at 1.0, fractions monotonic, splats events
  all carry `tile_work` + `estimate_seconds`, ≤200 splats events); rgba/splats/depth bit-
  identical with and without a progress callback; non-interactive refuses above the
  splat budget; interactive (with progress) renders successfully even with a 1-tile
  budget; pre-bin guard refuses huge bbox pairs when `enforce_budget=True` and is
  skipped when `False`; `Cancelled` from the callback aborts the render and a follow-up
  render still works; `estimate_eta_seconds` at 0 done uses the reference rate, at done
  == total returns 0, is strictly decreasing as done grows past 2%; `Evaluator.progress`
  forwards to `scene3d.render`, cache hits emit zero events; ETA at 10% and 50% is
  within the bounds the lane documented (pessimistic early, accurate late).

Honest limits at `6d07fb1`: the desktop app has not yet wired `Evaluator.progress` into
the UI — the callback is the API, the wiring is separate work; no ETA smoothing or
calibration by scene type; the callback is capped near 200 events per render, so very
fine-grained progress is not available; shadow and mesh budgets are unchanged and still
apply in the interactive path; `prepare_splats(..., enforce_budget=False)` is internal-
facing and not exposed in the public docs beyond the docstring.

Queue from here, in the order Omid approved at 11:50 AM: (3) GPU ray-traced mode,
tiled GPU submissions, GPU job cancellation. No particles or volumes yet.

## GPU splat rendering wired into Render3D (step 2, 2026-09-20 1:40 PM PDT)

`main` moved `ab931f7` -> `0ed4dad` (cherry-pick of the lane's `5156259`; the lane was on
`a4e3761` so a straight fast-forward was impossible and the conflict was only `TASKLOG.md`).
`gpu3d.render` now renders splats through `gpusplat.render_layer` for `rgba` in `raster` mode when
every mesh is opaque, for baked splats and for relit splats whose lights have no shadows: mesh
rgba, then mesh view depth from the depth pass at the inner (supersampled) resolution, then the
splat layer, then a composite identical to the CPU's, then the box filter. `auto` falls back to
the CPU and `gpu` raises a clear error for the data passes and the `splats` output, transparent
or projected meshes mixed with splats, a shadowed light with relit splats, baked splats that
cast shadows onto meshes, an adapter without vertex-stage storage buffers, and a render that
would exceed the GPU memory cap. `imaging.py` no longer forces splat scenes onto the CPU.
13 new tests in `tests/test_3d_gpu_splat_render.py`; three existing "splats are CPU-only"
assertions were updated to the new world (one uses a half-transparent card as the CPU-only case,
the others skip `rgba` and assert the remaining outputs still raise).

Review evidence at `0ed4dad` (clean temp worktree, shared `projects/nodebased/.venv`,
`PYTHONPATH=<worktree>` for standalone scripts):

- Targeted: 78 tests OK in 7.6 s (the four relevant modules: `test_3d_gpu_splat_render`,
  `test_3d_splat_render`, `test_3d_read_splat_node`, `test_3d_gpu_splats`, 1 skipped).
- Full discovery 1121 OK (1 skipped) in 760.2 s (lane's run 754 s).
- Reviewer repros (14 claims): CPU vs GPU parity within 8.3e-7 on baked-only splats, 8.0e-7 on
  splats + opaque mesh, 8.3e-7 on relit directional and relit point without shadows, 2.4e-7 at
  samples=2 with and without a 35-degree opaque card; mesh-depth parity 0 to 2.1e-4 against the
  CPU depth pass (tilted cards at 43 and 78 degrees sit at the documented `5e-4` limit because of
  float32 interpolation order on the RTX 3080 Ti; the llvmpipe run held 2e-5); `auto` falls back
  to the CPU byte-exactly when `gpu3d.available()` is False, when capability fails, and when the
  GPU memory cap refuses; `gpu` raises a clear ValueError with the right reason in all of those;
  data passes on splat scenes (`depth`, `normals`, `position`, `uv`, `object_id`, `diffuse`,
  `splats`) raise on `gpu` and fall back on `auto`; pre-set cancellation raises `Cancelled`
  before adapter acquisition; cancellation propagated through `gpusplat.render_layer` raises
  `Cancelled`; baked splats + a shadow light with no mesh render on the GPU (`last_shadow_path
  == 'brute'`, splat shadows skipped because there is no mesh); Evaluator end-to-end: gpu ~= cpu
  within 3e-3, auto ~= cpu within 3e-3 with actual float-blend noise max 3.6e-7. Repro scripts
  in `/tmp/nb-wiring-repro/repro.py`; review log in `/tmp/nb-rv-wiring-full.log`.

Honest limits at `0ed4dad`: the GPU subset is exactly what the lane claimed (rgba, raster,
opaque meshes, baked or unshadowed relit). Shadows on relit splats, splats casting shadows onto
meshes, transparent-mesh layering, every data/AOV pass with splats and the `splats` output are
still CPU-only. The wgpu renderer still sorts on the CPU each frame and needs vertex-stage
storage buffers; the real-capture timings (`0.3-0.4 s warm` on the 3.4M-splat Nelson capture)
are from the step 1 module and are still valid because the wired path calls the same
`gpusplat.render_layer`. The 2 GiB GPU memory cap is unchanged.

Queue from here, in the order Omid approved at 11:50 AM: (2) splat budget stage 2 (progress +
ETA inside a frame, then stop refusing large CPU renders), then (3) GPU ray-traced mode, tiled
GPU submissions, GPU job cancellation. No particles or volumes yet.

## GPU splat layer (step 1 of post-0.24.0 work, 2026-09-20 12:39 PM PDT)

`main` moved `8c0b797` -> `f31f589` (cherry-pick of the lane's `57afed9`; the lane was sitting on
`a4e3761` so a straight fast-forward was impossible and the conflict was only `TASKLOG.md`):
`nodebased/gpusplat.py` 381 lines, `tests/test_3d_gpu_splats.py` 240 lines (12 tests). The new
module is NOT yet wired into `Render3D` or `gpu3d.render`; that is the lane's next commit (gpu3d.py,
imaging.py and a GPU splat render test are uncommitted in the lane worktree).

The layer matches the CPU renderer within the documented tolerance (2e-3 max / 2e-4 mean). Review
in a clean temp worktree at `f31f589` (`/tmp/nb-rv-gpusplat`, shared `projects/nodebased/.venv`,
`PYTHONPATH=<worktree>` for standalone scripts):

- 12 GPU splat tests OK in 1.4 s; full discovery **1108 OK (1 skipped), 756.9 s** (lane's own
  run on its tree: 819 s).
- Reviewer repros all PASS: 1500-splat SH-deg-1 rgb max 2.15e-6 alpha max 3.7e-6; mixed in/out
  frustum centre pixel identical to a solo render; SH-deg-3 350 splats rgb max 2.76e-6 alpha max
  3.6e-6; mesh-in-front alpha zero/nonzero mask matches CPU; mesh-far-behind splat alpha identical
  to a solo render; Relight 0 vs CPU rgb max 2.35e-6; Relight 1 with one light rgb max 1.31e-6;
  Relight 1 with a per-splat shadow visibility map rgb max 6.26e-7; empty / zero-splat instances
  return zero arrays; memory cap refusal raises `ValueError("MiB")`; cancelled render raises
  `Cancelled`. Last_timings at 200k 1080p warm: depth_sort_ms 18.2, colour_ms 0.02, upload_ms
  10.6, gpu_ms 138.6.
- Honest timings on this machine, 3080 Ti Vulkan, rgba32float: 200k splats 1920x1080
  cold 529.8 ms / warm 174.3 ms (lane's machine 700 ms cold / 90 ms warm); CPU refuses 1920x1080
  with the default budget and benchmarks at 5.8 s for 200k splats at 640x360.

Known limits at `f31f589`: the module is not used by `Render3D` yet, so the existing CPU
rasterizer still draws splats; `gpu3d.render` raises `Unsupported("splats are not implemented by
the wgpu backend yet")` for any render that contains splats. The integration commit will plug
into the four fallback paths the lane already drafted (transparent meshes, data passes, the
`splats` output, GPU shadow visibility upload) and add the documented `auto` fallback for the
integrated behaviour. The queue from there is: budget stage 2 (progress + ETA in a frame, then
stop refusing large CPU renders), GPU ray-traced mode, tiled GPU submissions, GPU job
cancellation. No particles or volumes.

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

`main` moved `e4993a4` -> `5156d72` -> `0bdff59` (2026-09-20 12:55 AM, Gonzo on Fable 5.1). `5156d72` is docs only: no tracked
file names a home directory or an account address any more (`TASKLOG.md`, `docs/BENCHMARKS-v0.17-playback.md`); the address
remains in git history, and rewriting history is DiMo's call. `0bdff59` gives `Card3D`, `Cube3D`, `Sphere3D`, `ReadGeo3D` and
`Scene3D` the uniform scale, rotation order and pivot that only `ReadSplat3D` had, as one Nuke-ordered block
(`core._XFORM`, `knobs._XFORM_KNOBS`); `upgrade_document` fills identity values into older documents. No renderer change:
`scene3d._transform_from` already read those fields. Evidence: `tests/test_3d_transform_knobs.py` (7 tests, including a
stripped document that upgrades, validates and renders byte-identical), the real Sphere3D and Scene3D panels inspected as
pictures, full suite at `0bdff59` 1091 OK (1 skipped) in 758 s. Still missing from the 3D UX requirement: sphere rows and
columns, pole treatment, local/world matrix readout. Rows and columns replace the animatable `segments` parameter (schema
rename plus curve migration), held until after 0.24.0.

Packaging dry run 35497374275 was dispatched on `e4993a4` at 12:37 AM (publish off), the first frozen build containing the
GPU viewport. It predates `0bdff59` and shadows B2, so the release still needs a dry run at the final HEAD.
That dry run finished green on ubuntu-22.04 and windows-latest (publish skipped).

`main` moved `7feb462` -> `8c02b70` (2026-09-20 1:31 AM, watch on Fable 5.1): splat shadows B2, the lane's `4be7e75`
cherry-picked onto `7feb462` (a context-only commit had landed meanwhile, so it could not fast-forward; the trees differ only
by `context/splat-relighting-plan.md`). Splats now cast shadows onto mesh fragments through one per-render splat BVH shared
with relit-splat shadows; casting does not depend on `Relight`; casters below alpha 1/255 are omitted; shadows darker than
`SPLAT_SHADOW_CUTOFF` = 1e-3 count as fully dark; only view-visible splats get shadow rays; `traverse(tmin=)` skips BVH nodes
that end before a ray's start offset, which is exact because `SplatSet._factors` cuts every splat at 3 sigma. Lane numbers
(CPU, 200,000-splat shell, one shadowed directional light, relit): 640x360 78.4-86.5 s before, 36.1 s after (5.3 s without
shadows); 1920x1080 44.1 s (12.1 s without); mesh floor under the shell at 320x180 4.0 s -> 5.7 s. Still a batch/reference
feature, CPU only. Review evidence: lane suite on its exact tree 1096 OK (1 skipped) in 783 s; my clean-checkout suite at
`8c02b70` 1096 OK (1 skipped) in 800 s; my own repro (3000 rotated anisotropic splats, 700 rays, six tmin/tmax variants:
pruned and unpruned bit-identical, brute-force agreement to 1e-12; a 400-splat blob over a floor card darkens 86 of 4225
pixels, brightens none, raster and raytrace agree to 2.1e-6, `shadows=False` ignores splats, `relight` 0 and 1 give the same
diffuse AOV); 640x360 timing re-measured at 40.5 s while a suite shared the machine. Known wording slip carried into
`docs/3D_FOUNDATION.md`: it says the shell's splats "all face the camera"; the real reason every splat gets a ray is that the
whole shell is inside the view and culling is frustum plus alpha. The lane fixes that in its tightening commit. The earlier
version of this commit (`3b58964`/`a9acaf3`/`bb889b6`) claimed a suite that never ran on it; `4be7e75` says so in its message.
Remaining before 0.24.0: the lane's tightening commit (docs known-limits, roadmap status), CI on Linux and Windows at the
final HEAD, a packaging dry run at the final HEAD, the release notes.

`main` moved `8c02b70` -> `a1e446d` (state note) -> `1c5a89d` (2026-09-20 1:47 AM, watch on Opus 5): the documentation
tightening pass, and the last queued Astra-lane item for 0.24.0. The lane's `437c9b1` (docs only: `docs/3D_FOUNDATION.md`
"Known limits" rewritten against the code, `docs/3D_ROADMAP.md` status per deliverable, `TASKLOG.md`, both bundled copies
under `nodebased/data/docs/`) was cherry-picked onto `a1e446d` as `5231c1c`; `1c5a89d` amends it with three corrections
found in review:

- the shell's splats do not "all face the camera" — the whole shell is inside the frustum and culling is frustum plus alpha,
  which is why all 200,000 get a shadow ray;
- "Known limits" now names the real-capture refusal at 1920x1080 (2,339M tile evaluations, estimated 142 s) and says
  1280x720 is the largest standard size that renders;
- the platform line states what CI actually does: the full suite runs on Linux and Windows per commit, but neither the `gpu`
  nor the `usd` extra is installed, so 81 (Linux) / 82 (Windows) of 1079 tests skip (read from the CI log of run
  35497416740), and nothing GPU-related has ever run on Windows or macOS.

`1c5a89d` also drops a roadmap header that credited 0.24.0 with the USD and Alembic readers; both shipped in 0.23.0
(checked against `docs/RELEASE_NOTES.md` and the `v0.23.0` tree). Evidence: lane suite on its exact tree 1096 OK
(1 skipped) in 802 s; my clean-checkout suite at `1c5a89d` 1096 OK (1 skipped) in 756 s; `tests.test_knowledge` 7 OK.
No product code changed in either commit.

**The Astra lane's 0.24.0 queue is now empty and the feature freeze holds.** Release-gate state at 1:48 AM: lane items done,
`gonzo/3d-ux` merged, CI for `1c5a89d` queued (run 35500538889), packaging dry run at the final HEAD dispatched as run
35500544508 (publish off). Left before the tag: both of those green, then verify
`artifacts/astra-specs/release-0.24.0-notes-draft.md` claim by claim, bump `pyproject.toml` and `nodebased/__init__.py`,
prepend the notes, sync the bundled copy, full suite, commit, tag `v0.24.0`, push, and check the tag build's AppImage,
Windows setup exe, portable zip and SHA256SUMS before calling anything shipped.

## 0.24.0 released (2026-09-20)

v0.24.0 is published from `bca5d3b` (release published 3:42 AM PDT):
<https://github.com/neodimo/NodeBased/releases/tag/v0.24.0>. Verified on the release page: AppImage (138,840,568 bytes),
Windows setup exe (68,557,418), portable zip (99,444,897), SHA256SUMS (318); not a draft, not a prerelease. Desktop
conformance on `bca5d3b` (run 35504356557), the tag build (run 35504358897) and the packaging dry run on `3976fa1` (run 35502192355) are green on Linux and
Windows. The first dry run (35500544508) failed on Windows because the dense-mesh viewport frame-time test measured 72 ms
against a 50 ms limit on the runner's software adapter; `3976fa1` skips that check on software adapters and the docs say
what the release workflow really runs on Windows. Limits are listed in `docs/RELEASE_NOTES.md`.

Hard-requirement status after 0.24.0: splats delivered (CPU render only); splat relighting delivered in the final render
(lights, mutual mesh/splat shadows, both render modes) and only as a proxy in the viewport (opaque discs that follow
`Relight`, no blending, no shadows), so the viewport half of that requirement is still open; USD and Alembic as in 0.23.0.

**Post-release queue, not started. The Astra lane is idle (clean tree, branch level with `main`) and DiMo has not yet
authorized post-release lane work; he was told so in #nodebased at 4:10 AM.** Suggested order when he says go:

1. GPU splat path (wgpu), which also unlocks a real relit splat preview in the viewport (the open half of the relighting
   requirement) and ends the CPU-only timings in the known limits.
2. Splat budget stage 2: progress callback plus ETA inside a frame, after which the app stops refusing large renders
   (1920 x 1080 on the 3.4M-splat capture is refused today at 2,339M tile evaluations).
3. GPU ray-traced mode, tiled GPU submissions (lifts the roughly 19k-triangle HD shadow refusal), GPU job cancellation.
4. Particles, volumes.
5. Held UX items: Sphere3D rows and columns replacing `segments` (schema rename plus curve migration), pole treatment,
   local/world matrix readout; multichannel EXR / more than one AOV per node.
6. Small debts only if test-covered and bit-identical: USD stage opened twice, Alembic re-decode.

The 20-minute status watch (`nodebased-3d-status-watch`) was removed once the release was verified; whoever restarts the
lane should set up a new watch with a fresh payload, since the old one described the 0.24.0 gate.

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
then watch both tag workflows and confirm the published assets. Then, in the same turn as the
confirmation, post the release to #nodebased with the `message` tool: the release URL, a plain-English
list of what is in it, the builds, and what landed after the tag. A release is not reported until that
post exists (DiMo, 2026-09-23 10:51 PM: the v0.26.0 announcement never reached the channel; the
release turn finished without posting).

### Release media: slow burn, big features only (standing, from DiMo 2026-09-23 3:29 PM PDT)

Supersedes the 2026-09-20 rule that made screenshots and video a release gate. The release itself
comes first: when `main` is green and the work is merged, cut it without waiting for media. Media is
a slow burn afterwards, and only for the really big features (a headline 3D interaction, a new
renderer path, a new pipeline), never for every node or knob. When a still or clip is ready, append
it to the existing GitHub release (edit the release notes and attach the asset) and send DiMo one
updated release message saying what was added. The bundled `docs/RELEASE_NOTES.md` copy still has to
match `docs/` byte for byte, so a media addition after the tag is a small docs commit on `main`
plus the release edit, not a new tag.

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
