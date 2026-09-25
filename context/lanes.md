# NodeBased lanes

Opened 2026-09-21 at about 11:38 PM PDT on DiMo's direction in #nodebased (11:28 PM PDT):

> This is mostly regarding the feature for the 2D to 3D splat or geometry pipe. We also need to keep in
> mind the entire NodeBased software package. The 2D and 3D nodes, which are very very very much
> underbaked. The viewers for 2D and 3D that need more polish and have better user interface controls
> including pivot points (that can be changed), visuals for user interfaces of moving an object in 2D and
> 3D in X, Y, and Z if 3D. The nodes in NodeBased have a long way to go until we at the very least have
> parity with Nuke and parts of Houdini. We still have the particles and fluids simulations to get
> through. So think more than just the 2D to 3D for now. Let's make sure we are working all the lanes.

This supersedes the 2026-09-20 4:46 PM scope directive ("finish started work before going full steam").
0.25.0 shipped at 3:47 PM PDT on 2026-09-21. Every lane below is open at once. Each one delivers in
small merged steps; nothing waits for a lane to "finish" before it can ship in a release.

This file is the map. `state.md` keeps the dated evidence per merge, `TASKLOG.md` the per-change notes,
`docs/3D_ROADMAP.md` the 3D milestones and gates. When a lane's scope or owner changes, edit this file
in the same commit as the change that caused it.

## Standing rules for every lane

- **Branch and worktree.** One branch per lane, cut from `main`, in its own worktree. `main` is the
  integration branch. Nothing lands on `main` without (1) the full suite green on the exact commit that
  merges, (2) a review by the integrator (Gonzo in the #nodebased session; there is no automation),
  (3) a dated section at the top of `context/state.md` and a `TASKLOG.md` entry that state the
  evidence and the limits plainly. Several lane branches that share a base may be stacked into one
  integration branch and tested once at the tip; the tip is then the commit that merges.
- **Cadence (DiMo, 2026-09-22, 9:46 PM PDT).** At most two lanes are live at once. A lane run does one
  bounded step: one supervisor run, one worker run, the targeted tests for the lane's own files. It
  then posts a plain-English report in #nodebased (what changed, what the tests said, step N of M with
  the step's deliverable checklist, what is blocked and on what, what decision it needs) and stops.
  Continuous mode (DiMo, 2026-09-23, 2:39 PM PDT): when a lane's step has merged, its next step starts
  at once from a pre-written brief (`scratch/nb-lanes/auto/briefs/` in Gonzo's workspace), driven by a
  10-minute integrator check on Sonnet (`scratch/nb-lanes/auto/tick.py`) that merges only after its own
  targeted rerun and a green full suite, posts one plain line per merge, and stops a lane on any red or
  irregularity ("Gonzo needed"). Lanes never restart themselves. "Gonzo pause" halts everything. The
  full suite runs once per merge candidate, by the integrator, not per lane commit.
  Three lanes at once (DiMo, 2026-09-24, 6:08 PM PDT, "Go" on refilling the three empty slots): the cap
  is three live lanes. Each lane carries a plan backlog (`scratch/nb-lanes/auto/state.json`, briefs in
  `scratch/nb-lanes/auto/briefs/`); when a plan's last step merges the tick archives it into the lane's
  history and starts the next approved plan, so a slot only goes empty when no plan is written for it.
  Retired lanes (plan complete, no backlog): L1 (viewer handles, retired 2026-09-23 after step 4 of 4)
  and L3 (3D parity, retired 2026-09-24 after step 4 of 4). Live as of 6:13 PM PDT 2026-09-24: L2 (2D
  parity, third pass), L4 (rendering, reopened on Claude Sonnet 5 as the worker, no per-item gate) and
  L5 (particles, step 2). All three of those plans were complete by 11:25 AM PDT 2026-09-25 (L2 step 4c
  and L5 step 2c at 10:25 PM 9/24, L4 step D at 11:25 AM 9/25). Second round (DiMo, 2026-09-25 2:48 PM
  PDT, "Proceed with your lane suggestions"), live from 2:52 PM: L2 runs the **2D Viewer parity** plan
  (inputs, A/B and wipe; gain, gamma, zebra, display; ROI, proxy, format masks) and for it **owns the
  `Viewer` class in `app.py`** (handed over from retired L1), with a fourth node pass queued behind it;
  L4 runs rendering plan 2 (particles on the GPU and in the 3D viewport, splat normals pass, WriteSplat3D,
  multichannel EXR) and for it may edit `viewport3d.py`/`viewportgpu.py` only to carry particles and Spot
  lighting through (handed over from retired L1; gizmos and handles untouched); L6 (fluids) runs steps 2
  and 3 of its spike. L5 is retired (plan complete); GPU drawing of particles moved into L4 plan 2.
  L7 (2D-to-3D pipe) is ON HOLD per DiMo 2026-09-24 6:21 PM PDT: "important
  workflow, but not until we get everything else in a better place"; issue #7 and its card say so, and
  it enters no slot and no backlog until DiMo lifts the hold.
  **The board** is the GitHub project "NodeBased lanes" (<https://github.com/users/neodimo/projects/1>,
  public, chosen by DiMo on 2026-09-22 at 10:48 PM PDT). One repo issue per lane, label `lane` (#1 L1,
  #2 L2, #3 L3, #4 L4, #5 L5, #6 L6, #7 L7), carries the lane's step checklist; the project fields
  `Lane status` (Live, Paused, Blocked, Done), `Step`, `Blocked on`, `Branch` and `Last suite` carry the
  rest. A lane updates its issue checklist and fields through `gh issue edit` and `gh project item-edit`
  in the same turn as its #nodebased report, at the start of a step, at its end, and when it blocks. A
  new lane gets a new `lane` issue and a card before it starts.
- **Tests before claims.** Every behaviour a commit claims has a test, and pixel claims assert pixels.
  Full suite: `python -m unittest discover -s tests` (about 14 minutes; 1268 tests on 2026-09-21). Run it
  with the shared environment `projects/nodebased/.venv` and `PYTHONPATH=<worktree>` so the GPU and USD
  tests do not skip. Run the full suite through the lane suite wrapper `/tmp/nb-suite.sh <log> [worktree]`
  (source copy `scratch/nb-lanes/suite.sh` in Gonzo's workspace): up to three suites run at once, each
  holding a shared lock on `/tmp/nb-gpu.lock`. SHARP runs, benchmarks and anything that needs the whole
  RTX 3080 Ti keep the exclusive `flock /tmp/nb-gpu.lock <command>`, which waits for running suites and
  holds off new ones: the card has 12 GB and the SHARP runtime alone takes 11.9 GB of it. A lane run
  resumed by message is capped at 30 minutes, so write the lane's STATUS.md within the first five minutes
  and before any wait longer than 20 minutes, and end the turn rather than poll a queued suite.
- **Knobs follow Nuke** (state.md, "3D UX requirements", 2026-09-19): XYZ numeric fields for transforms,
  uniform scale, rotation and transform order, pivot, polygon counts as rows and columns, read-only
  local and world matrices; sliders only where a bounded scalar is natural. 3D nodes have rounded
  shapes; Scene3D, Light3D and Camera3D are circles. Interaction must feel snappy: no full graph
  re-evaluation on a mouse move.
- **File ownership** keeps lanes from colliding. Shared registries (`core.py` SPECS/LIMITS/CHOICES,
  `knobs.py`, `tiers.py`, `theme.py`, the properties labels in `app.py`, `docs/`) are touched by
  everyone in small additive edits; conflicts there are expected and are resolved on rebase. Anything
  else belongs to the lane named below, and another lane that needs a change there asks for it
  instead of making it.
- **Public repo.** No home paths, no email addresses, no captured assets in tracked files.
  `assets/splats/scene.ply` is read-only and never committed or derived from in the repo.
- **Words.** 12-hour clock everywhere. "Shipped" only after a tag and a published release. Separate
  what was measured from what is inferred.
- **Models (DiMo, 2026-09-22).** Lanes are a Claude Sonnet 5 supervisor in the worktree driving a worker
  through `codex exec`. Simple coding goes to the cheaper GPT-5.6 models (Luna, Terra); Sol when the
  task needs it; GPT-6 Astra only for big-picture work DiMo approves case by case, never as a routine
  worker. When the GPT models are unavailable, Sonnet 5 is the worker and Haiku handles mechanical
  steps; Fable and Opus are not spent on lane work. The supervisor owns scope, review, tests, commits
  and the report; the worker's report is never trusted without `git log` and a test log. On a usage
  limit, the supervisor records the retry time in its report and stops; nobody restarts it on a timer.

## Lanes

### L1. Viewer interaction: pivots and on-screen handles, 2D and 3D

**Status: complete and retired** (step 4 of 4, camera and light handles, merged `a691c2e` 2026-09-23;
gizmos in 0.26.0, handles in 0.27.0). Open request against its files: the editor viewport still lights
a Spot like a Directional light (L4 step A, 2026-09-24, request in `TASKLOG.md`).

**Why.** DiMo, 11:28 PM: pivot points you can move, and visible controls for moving an object in 2D and
in X, Y and Z in 3D. Today the 2D `Viewer` draws roto and tracker overlays only, and the 3D
`Viewport3D` orbits, pans and dollies with no selection and no handles.

**Owns.** `nodebased/viewport3d.py`, `nodebased/viewportgpu.py`, the `Viewer` class and the 3D viewport
wiring in `nodebased/app.py`, and new modules for the handles (for example `handles2d.py`,
`handles3d.py`). Visual QA on the real display and GPU is Gonzo's. **Retired 2026-09-24** (plan
complete). Hand-over from 2026-09-25 2:48 PM PDT: the `Viewer` class belongs to L2 for its Viewer parity
plan; L4 may touch `viewport3d.py`/`viewportgpu.py` only to carry particles and Spot lighting through.

**Deliverables, in order.**

1. **2D Transform handle.** When the properties panel shows a `Transform`, the Viewer draws its
   handle over the image: a pivot marker (drag sets `center_x`/`center_y` without moving the image, as
   Nuke's Ctrl-drag does), a translate drag on the box or centre, a rotate ring, and scale handles on
   the corners and edges, with a numeric readout while dragging. Each drag is one undo step and keys
   animated knobs the way roto point drags do. The handle never steals a roto or tracker click.
2. **3D selection.** Clicking in the 3D viewport picks the nearest object whose node carries a
   transform (object id from the GPU renderer where present, a ray test against bounds on the CPU
   path). The picked node's properties open; a selection outline or bounds box shows it.
3. **3D gizmos.** Translate (X, Y, Z arrows plus the three plane squares), rotate (three rings) and
   scale (three axis cubes plus uniform) on the selected object, driving `tx ty tz`, `rx ry rz`,
   `sx sy sz`/`uscale`. A pivot mode moves `pivot_x/y/z` without moving the object. Hotkeys for the
   modes are the lane's call, recorded in the in-app shortcuts table. Dragging updates only the
   object's transform in the viewport (no scene re-evaluation) so it feels immediate.
4. **Camera and light handles** (position and target for `Camera3D` and `Light3D`).

**Tests.** Offscreen Qt tests: a drag sets the expected parameters, undo restores them, the pivot drag
leaves the rendered image unchanged, screen-to-axis projection round-trips, mode hotkeys, and the
handles stay out of roto and tracker interaction.

### L2. 2D node parity with Nuke

**Why.** DiMo: the 2D nodes are underbaked and parity with Nuke is the bar. The 2D set today: Read,
Constant, Checker, Grade, ColorCorrect, Blur, Transform, Crop, Shuffle, ChannelShuffle, Roto, Tracker,
Merge, Premult, Unpremult, Dot, Switch, Viewer, Write.

**Owns.** New kernel modules for 2D nodes (for example `nodebased/ops2d_*.py`), their tile-path
implementations in `tileexec.py`, and `docs/PARITY_2D.md`.

**Deliverables, in order.**

1. **The audit, first commit.** `docs/PARITY_2D.md`: the node classes a Nuke 17.1 compositor uses
   daily, grouped as Nuke's menus group them (Image, Draw, Time, Channel, Color, Filter, Keyer, Merge,
   Transform, Metadata, Other), each marked supported, partial or missing against `main`, sourced
   from the Nuke reference guide and ranked by daily use. Merge's operation list is audited too.
2. **Nodes, by rank.** Expected early rows: Reformat (a real format model), Invert, Clamp, Saturation,
   Multiply/Add/Gamma as separate nodes, Merge operations beyond over (plus, screen, multiply,
   difference, mask, stencil, under, atop, xor, max, min, average), Dissolve, Keymix, Copy,
   ChannelMerge, Erode/Dilate, Median, Sharpen, Glow, Keyer (luminance and chroma), HueKeyer, Difference
   key, Ramp, Radial, Rectangle, Noise, Mirror, CornerPin, FrameHold, TimeOffset, Retime, Text.
   Each node ships with: kernel on the evaluator and the tile path, mask and mix where Nuke has them,
   bypass behaviour, LIMITS/CHOICES, knobs, a shortcut where Nuke has one, a docs row flipped from
   missing to supported, and tests with pixel assertions.

### L3. 3D node parity with Nuke and parts of Houdini

**Status: complete and retired** (step 4 of 4, MergeGeo3D, Normals3D, DisplaceGeo3D, merged `54597bc`
2026-09-24; all four steps ship in 0.27.0).

**Why.** DiMo: the 3D nodes are underbaked. Today: Card3D, Cube3D, Sphere3D, Scene3D, Camera3D,
Light3D, Project3D, Render3D, ReadGeo3D, ReadSplat3D, ReadAlembic3D, ReadAlembicCamera3D, ReadUSD3D,
ReadUSDCamera3D, ReadGLTF3D, WriteGeo3D.

**Owns.** Geometry primitives and hierarchy in `nodebased/scene3d.py` (not the renderer, materials,
shadows or splat code, which are L4's), the 3D node entries in the registries, `docs/3D_FOUNDATION.md`
node tables.

**Deliverables, in order.**

1. **Axis3D** (a transform that parents whatever is wired into it, chainable) and **TransformGeo3D**
   (applies a transform to input geometry, with a pivot).
2. **Rows and columns** on Card3D and Sphere3D, pole handling on the sphere, **Cylinder3D**, a
   subdivided plane; **read-only local and world matrices** in the properties panel of every
   transformable node (the UX requirement from 2026-09-19).
3. **Camera3D film back**: focal length, horizontal and vertical aperture, and the derived field of
   view, so a camera from Alembic, USD or glTF keeps its lens. **Light3D** spot cone and falloff.
4. **MergeGeo3D** (any number of inputs, replacing the eight fixed Scene3D slots over time),
   **Normals3D** (recompute or flip), **DisplaceGeo3D** (image-driven displacement).

Every node follows the Nuke knob direction and ships with tests that assert vertices, matrices and
rendered pixels.

### L4. Rendering and relighting

**Why.** The queue that the 0.25.0 scope cut, now open. Worktree `nb-3d-astra-lane`, branch
`openclaw/nb-3d-astra-lane`. Originally the Astra lane (GPT-6 Astra, one go-ahead per item); reopened
2026-09-24 6:13 PM PDT as a continuous lane on Claude Sonnet 5 with no per-item gate (DiMo "Go",
6:08 PM). Current plan (4 steps, briefs `L4-stepA..D`): spot cone and falloff in every renderer (step A
reported done 6:25 PM at `2e15004`), shadow offset and blur controls, kept specular, GPU shadows on
relit splats.

**Owns.** `raytrace.py`, `gpurt.py`, `gpurt_render.py`, `gpu3d.py`, `gpusplat.py`, `splatraster.py`,
`splatshade.py`, `splats.py`, the materials, shadows and splat code in `scene3d.py`, `renderprogress.py`.

**Queue, in order.** Relight passes out of Render3D and a 2D `Relight` node; a normals pass; shadow
offset and blur controls; kept specular; GPU shadows on relit splats and shadow catching (per-splat
visibility is still CPU); `WriteSplat3D`; multichannel EXR out of Render3D. Stays out of the viewport,
node graph and properties UI files (L1) and the geometry primitives (L3).

### L5. Simulation foundation and particles

**Status: live** since 2026-09-24 6:13 PM PDT on Claude Sonnet 5, worktree `nb-particles`, branch
`openclaw/nb-particles`, plan "Particles: emitter and cache, forces, bounce and rendering" (3 steps,
briefs `L5-step2a..c`; step 2a reported done 6:45 PM at `e880540`).

**Why.** DiMo: particles and fluids still have to be done. Roadmap milestone 5 sets the gate:
deterministic emitters, forces, collisions, instancing, disk-backed caches, scrubbing without
re-solving, reproducible seeds, timestep handling, restart, invalidation, cancellation, budgets.

**Owns.** New modules `nodebased/simcache.py` and `nodebased/particles.py`, the particle node entries,
`docs/SIMULATION.md`. L6 builds on the same cache and time model, so this lane writes them first and
keeps their interface small. **Retired 2026-09-25** (step 2 complete 9/24 10:25 PM PDT); the GPU and
viewport drawing of particles is L4 plan 2, step E.

**Deliverables, in order.**

1. **Design, first commit.** `docs/SIMULATION.md`: the simulation time model (start frame, substeps,
   seed), the disk cache under the user data directory keyed by the node's upstream digest, scrubbing
   through cached frames, what invalidates a cache, cancellation and memory budgets, and how a
   simulation becomes typed scene data that Render3D draws. Read Nuke's ParticleSystem and Houdini
   POPs for the knob vocabulary.
2. **Particles.** `ParticleEmitter3D` (from a point, or the points, surface or volume of input
   geometry; rate, life, initial velocity, size, colour, seed), forces as separate nodes
   (`ParticleGravity3D`, `ParticleDrag3D`, `ParticleTurbulence3D`, `ParticleWind3D`), `ParticleBounce3D`
   against scene geometry, `ParticleCache3D`, and rendering as points, spheres or camera-facing cards
   through Render3D.

### L6. Fluids solver spike

**Why.** DiMo: fluids. The roadmap says evaluate an established solver and volume ecosystem before
committing to a custom solver, and keep importing and rendering caches separate from solving.

**Owns.** `docs/FLUIDS_SPIKE.md`, `nodebased/fluid2d.py`, `tools/benchmark_fluid.py`, and later the
volume import and render modules. No nodes until the spike concludes.

**Deliverables, in order.**

1. **Ecosystem read, sourced.** OpenVDB and its Python bindings (wheel availability on Linux and
   Windows, cp312), Mantaflow, PhiFlow, Taichi, and what Houdini's Pyro and Nuke's lack of fluids imply
   for us. Which one we would import caches from (VDB first), and whether any is packageable inside
   the app the way `usd-core` is.
2. **A 2D smoke solver spike** in NumPy: MAC grid, semi-Lagrangian advection, buoyancy, vorticity
   confinement, pressure projection with a conjugate-gradient solve, deterministic, cancellable, with
   tests (mass conservation, divergence after projection, determinism) and measured frame times at
   256 by 256 and 512 by 512. A wgpu compute variant if the NumPy numbers show it is needed.
3. **The verdict** in `docs/FLUIDS_SPIKE.md`: custom solver, embedded library, or import-only, with
   the numbers that decide it.

### L7. 2D-to-3D pipe

**Status: ON HOLD** (DiMo, 2026-09-24, 6:21 PM PDT: "That is an important workflow, but not until we get
everything else in a better place"). Step 1 is mid-way on its branch as WIP. Issue #7 and the board card
say on hold; the lane gets no slot and no backlog until DiMo lifts it.

**Why.** The lane DiMo asked for first (10:21 PM PDT). `ReadGLTF3D` merged at 10:59 PM. The SHARP spike
(`workspace/scratch/sharp-spike/`) measured 13 s and 11.9 GB peak VRAM per image, output correct with
`orientation='colmap'`.

**Owns.** `nodebased/runtimes.py` (on-demand uv runtimes under the user data directory, with progress
and a plain refusal when offline), `nodebased/imageto3d.py`, the `ImageToSplat`, `ImageToMesh` and
`WorldSculpt` node entries, `docs/IMAGE_TO_3D.md`.

**Deliverables, in order.**

1. **ImageToSplat** running SHARP as a subprocess in an on-demand runtime, cached by input hash and
   parameters, orientation fixed to colmap, a splat scene out that Render3D and the viewport draw.
   Torch stays out of the package. Tests use a fake runner; the real run on the RTX 3080 Ti happens
   under the GPU lock and its timing goes in the docs.
2. **ImageToMesh** (Pixal3D) through the same runtime manager, GLB out through `ReadGLTF3D`'s reader.
3. **WorldSculpt** once its VRAM need is measured against the 12 GB card.

### Integration (Gonzo)

Review and merge lane commits, run the exact-commit full suite, write the state note, keep this map
current, do the visual QA on the real display, capture release media, cut releases per the standing
release policy. The `nodebased-lanes-watch` automation does the review-and-merge job between Gonzo's
turns and restarts lanes that stopped on a usage limit.

## Release cadence

0.26.0 shipped 7:50 PM PDT 2026-09-23 and 0.27.0 shipped 7:24 PM PDT 2026-09-24 (DiMo, 5:04 PM: "Cut
v0.27 when you believe it's ready today, i don't want to rush it"). The release now runs through the
same tick (`tick.py release <version> <notes> [summary]`): release commit in its own worktree, full
suite at that exact commit, annotated tag, tag workflows polled, one announcement in #nodebased, and
lane merges wait while a release is in flight.

0.26.0 when the first merged step of L1 (the 2D Transform handle or the 3D translate gizmo), at least
three L2 nodes, `Axis3D`/`TransformGeo3D`, and `ImageToSplat` are on `main` with media; earlier if a
user-facing fix warrants it. Simulation and fluids ship when their gates in the roadmap are met, not
before.
