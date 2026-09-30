# NodeBased 3D roadmap

Status: milestone 1 shipped in 0.22.0 (2026-09-18). 0.23.0 added the optional GPU backend, USD and
Alembic import, camera projection, hard shadows, materials and AOVs. 0.24.0 adds the ray-traced render mode,
the GPU BVH shadow path, Gaussian splats (reader, node, rendering, relighting, shadows) and the GPU
viewport, all with the limits listed in [3D_FOUNDATION.md](3D_FOUNDATION.md) ("Known limits"). This
document is a plan plus status notes, not a list of shipped capabilities; what exists is described in
3D_FOUNDATION.md.

## Product target

A compositing-first 3D/2.5D workspace with an interactive viewport and a camera
render that feeds the ordinary image graph and Write. User scope additionally
includes Gaussian splats, ray tracing, particles and fluid simulation. Full
Nuke 17.1 parity is a multi-milestone target requiring a feature-by-feature audit;
neither a mesh viewer nor empty node types satisfy it.

## Verified reference baseline

Foundry's 17.1 release notes document Gaussian-splat relighting and shadows,
animated splat import through GeoSequencer, splat export, Hydra 2.0 delegates,
USD layer composition and render outputs. GeoRender accepts scene and camera
inputs, supports delegate-specific settings/AOVs, and renders USD materials
(not Nuke materials). These are separate requirements, not synonyms for drawing
triangles. Legacy projection, particles, deep and geometry workflows still need
an inventory; fluid simulation is an explicit user request regardless of parity.

Primary references, checked 2026-09-18:
- https://learn.foundry.com/nuke/17.1v1/content/release_notes/nuke_17.1.html
- https://learn.foundry.com/nuke/17.1v1/content/comp_environment/usd-3d-comp/rendering-georender.html

## Architecture boundaries

- Keep UI/orchestration in Python; use compiled/GPU engines for expensive
  rendering and simulation. A reference rasterizer can establish correctness
  but must be labeled honestly and cannot substantiate performance claims.
- Distinguish typed image, geometry, scene and camera connections. Reject invalid
  connections without damaging documents. Maintain schema migrations, undo,
  animation and the agent protocol alongside UI changes.
- Separate editor navigation from authored render cameras. Orbiting a viewport
  must not silently edit a camera or invalidate the final image.
- Render scene-linear premultiplied float32 RGBA into the existing pipeline.
  Display transforms occur once, in the viewer. Image export retains graph
  precision; any reduced-precision preview/cache needs explicit provenance.
- Scene identity must include time, upstream geometry/material/image inputs,
  camera, lighting, settings and resolved asset identity. Navigation-only state
  must not invalidate image caches. Cancellation and memory budgets are required.
- Preserve an interchange boundary for USD and native render delegates rather
  than coupling all future features to a first-stage mesh implementation.
  USD/Hydra adoption requires packaging, license and Windows/Linux viability
  experiments before making it a mandatory dependency.

## Milestones and release gates

### 1. Usable foundation (shipped in 0.22.0)
Scene assembly, camera, transforms, basic card/cube geometry, navigable viewport
with grid/axes, and camera render into the 2D graph/Write. Textured cards if ready.
Gate: actual rendered pixel assertions, occlusion/camera/transform tests,
invalid connection rejection, old-document loading, undo/redo, live orbit/pan/
dolly/frame interaction, and an exported image matching the chosen render tree.
Record remaining limitations and backend in release notes.

### 2. Production 2.5D and interchange
Image cards, camera projection, camera/geometry animation, scene hierarchy,
geometry import/export and a tested USD subset. Define units, handedness, axis,
lens/filmback, image origin and alpha conventions before import adapters.
Gate: known camera/projection fixtures, animated round trips, missing assets,
cache invalidation and textured-card color/alpha correctness.

### 3. Lighting and rendering
Materials, lights, shadows, motion blur, depth of field and named AOVs; integrate
a viable compiled ray/path-tracing backend after an actual packaging spike.
Keep interactive preview and final-render quality explicitly distinguishable.
Gate: convergent reference scenes, cancellation, deterministic seeds where
supported, transparent/holdout correctness, memory limits and real GPU timings.
Deep output is a separate audited capability, never inferred from a depth AOV.

### 4. Gaussian splats
Import supported splat representations with covariance/opacity/color preserved,
camera-correct anisotropic rendering and mesh/splat depth interaction. Then
animated sequences, exports, relighting and shadow controls.
Gate: reference assets and images, transparent compositing, ordering artifacts,
animation identity, bounded memory and measured throughput. A point cloud is
not equivalent to a Gaussian-splat renderer.

### 5. Particles and volumes/fluids
Deterministic emitters, forces, collisions, instancing and disk-backed simulation
caches. Separate importing/rendering simulation caches from actually solving
fluids. Evaluate an established solver/volume ecosystem before committing to a
custom solver; support scrubbing through cached simulation without re-solving.
Gate: reproducible seeds, timestep handling, restart/checkpoint behavior,
collision fixtures, simulation invalidation, cancellation and resource budgets.

Gate status, particles (L5 step 2c, 2026-09-24; evidence in `docs/SIMULATION.md` and
`tests/test_particles_*.py`, `tests/test_simcache.py`):

- [x] Reproducible seeds (per-substep seeded streams; bit-identical across sessions and jumps).
- [x] Timestep handling (substeps, semi-implicit Euler, swept collisions independent of the timestep).
- [x] Restart/checkpoint behaviour (every frame checkpointed to memory and disk, a new session does not re-solve).
- [x] Collision fixtures (`ParticleBounce3D`: analytic rebound heights, Coulomb friction, kill, no tunnelling over 200 frames).
- [x] Simulation invalidation (any emitter, force, bounce or collider change starts a new run).
- [x] Cancellation and resource budgets (cancel stops a solve and keeps banked frames; memory, disk and particle caps).
- [x] Deterministic emitters and forces; rendering as points, spheres and cards on the CPU renderer.
- [x] Instancing (L4 step A, 2026-09-27): `Instance3D` copies up to eight variant meshes onto
  points from any particle stream or geometry (`scene3d.instances_from_node`,
  `tests/test_3d_instance.py`). Knobs: `scale`/`scale_random`, `orient` (none, velocity, point
  normal, random), `rotate_random`, `spin` per frame of age, `variant` (cycle, seeded random, or
  the point's particle id), `color_from_points`, `seed`. The CPU raster and ray-traced Render3D
  paths, shadows, WriteGeo3D and the OBJ/USD exporters all accept it (`scene3d.resolve_instances`
  expands an `InstanceSet` into ordinary geometries just before each of those reads
  `Scene.geometries`). The scene representation itself shares each source mesh's vertex and
  triangle arrays by reference across every instance: measured, 100k instances of a 1k-triangle
  mesh cost the ~19 KB of that one mesh, not 100M triangles' worth (`InstanceMemoryTests`).
  Decision measured against the alternative (a two-level BVH, one bottom tree per source mesh
  plus a top tree of instance transforms): `render()`'s existing ray-traced path already flattens
  the WHOLE scene to one `(triangle_count, 3, 11)` float32 world-space attribute array before
  shading, at `13.2 GB` for 100k instances of a 1k-triangle mesh (measured: `100_000 * 1000 * 3 *
  11 * 4` bytes) — regardless of instancing, that flatten is `render()`'s existing design for
  every geometry in a scene. Building the two-level BVH to also bound *that* array for extreme
  instance counts is real, sizeable, shared-renderer surgery (shading, shadows, reflections and
  liquids all read the flattened arrays together); it is out of this step's scope and deferred,
  named here so it is not silently assumed done. Moderate instance counts (what the CPU raster,
  ray trace and shadow tests above actually exercise) render correctly today.
- [ ] GPU renderer and 3D viewport do not draw particles or instances (requests written for L4
  step B and L1); Instance3D's viewport/GPU fallback is step B.
  - [x] GPU **ray tracer** (`nodebased.gpuinstance`, L4 step B part 1, 2026-09-27): traces
    `scene.instances` without flattening them, the alternative step A measured and deferred. A
    top-level tree (`raytrace.Bvh.build` over each instance's world bounds) points into one
    bottom-level tree per unique source mesh, built once in that mesh's own local space; a leaf
    transforms the ray into that instance's space (unnormalized, so the hit `t` stays valid in
    both spaces) before descending. Measured: 5,000 instances of a 72-triangle sphere upload that
    one sphere's ~1.7 KB of triangles once, not 5,000 times; only the ~160-byte-per-instance
    transform/tint table grows with instance count (`tests/test_3d_gpu_instance.py`
    `test_memory_stays_per_source_mesh_not_per_instance`, the 100k-instance case from step A's own
    memory test). Beauty, shadows (hard and soft, including instances shadowing each other) and
    the `depth`/`normals`/`albedo`/`diffuse`/`specular`/`emission`/`position`/`object_id` AOVs
    match the CPU ray-traced reference to within 2e-6 on the scenes tested, well inside the
    existing 2e-3 tolerance; per-instance tint (`color_from_points`) reaches the shader.
    Routed from `gpu3d.render(mode='raytrace')` only for a scene made entirely of instances (no
    ordinary geometry, splats, particles or volumes alongside them -- combining the two flattened
    and two-level structures in one shader is future work, named here rather than assumed done);
    a mixed scene, or any instance source with a texture, the liquid material or a projection, now
    raises `Unsupported` instead of the silent drop every GPU path gave instances before this step
    (`test_mixed_scene_is_unsupported_not_silently_dropped`). The `uv` AOV and multi-surface
    transparency across overlapping instances (order-independent peeling) are not implemented;
    single-surface alpha compositing is correct for the opaque or non-overlapping scenes
    instancing is used for today. The GPU **raster** renderer (`gpu3d.py`'s path for Render3D's own
    final output) still does not draw instances; the interactive 3D **viewport** now does (below).
  - [x] Interactive 3D **viewport** (`viewportgpu.py`, L4 step X2 of 2 part 3, 2026-09-30): draws
    every `scene.instances` copy with one hardware-instanced, indexed draw call per (InstanceSet,
    source variant) present, instead of the CPU flattening `expand_instances` the rest of the
    codebase uses. Per-instance model and normal matrices and straight-rgba tint
    (`InstanceSet.colors`) are uploaded once into a single vertex buffer per `InstanceSet`
    (`ViewportRenderer._build_instances`), grouped by source variant and cached on the identity of
    the set's own arrays exactly like every other upload here, so an unchanged scene (orbiting,
    nothing re-evaluated) costs nothing after the first frame. The source mesh itself is
    deduplicated into a real vertex/index buffer (`_instance_mesh`) rather than the triangle soup
    the one-draw-per-object mesh path uses, since an instanced draw pays that per-vertex cost once
    per copy: measured, 100,000 copies of a 990-triangle sphere render at ~40 fps (~25 ms/frame)
    indexed against ~18 fps (~55 ms/frame) with the soup, on an RTX 3080 Ti
    (`tests/test_3d_viewport_instances.py`
    `test_100k_copies_of_a_1000_triangle_mesh_stay_above_30fps`). Shading matches the mesh path's
    own `mesh_fragment` (headlight/unlit/lit/PBR, the dome, the key light's shadow map read but not
    cast into); picking an instance copy resolves to its Instance3D node
    (`handles3d.pick_instance_sets`, vectorized over every copy in a set so a 100,000-instance pick
    is one ray test, not a Python loop) via `InstanceSet.node_key`, set by `imaging.py` when
    `Instance3D` evaluates. The CPU fallback viewport (no adapter) now also shows instances, through
    `resolve_instances` like the rest of the codebase, so it has no GPU instancing to draw them
    with. Left out, both stated limits: instanced copies neither cast into the key light's shadow
    map nor carry a projector.
- [x] Moving colliders (L4 step A, 2026-09-27): `ParticleBounce3D.animated` tracks a collider's own
  rigid transform per frame and sweeps a substep's motion in that collider's own local frame, so a
  moving or rotating surface throws the particles it hits with its own velocity at the hit point
  (`tests/test_particles_animated.py`); off by default, so an existing document still solves
  bit-identically. The collider's own mesh is assumed rigid while it animates (only its transform
  moves, not its vertices).
- [x] Particle-to-particle collisions (L4 step B, 2026-09-27): `ParticleCollide3D` collides
  particles against each other as spheres of `collide_radius` or `radius_from_size`, with a uniform grid
  broad phase rebuilt every substep and `iterations` passes of position-based contact correction
  (`restitution`, Coulomb `friction`, `sleep_threshold` so a settled pile stops jittering); composes
  with a chained, moving or static `ParticleBounce3D` and with `ParticleCache3D` (`tests/test_particles_collide.py`).
  Deterministic (contacts sorted by particle id before any floating-point sum) and vectorised with
  NumPy; measured about 700 ms/frame (4 substeps) for 20,000 particles packed at a 40 percent volume
  fraction (`tools/benchmark_particle_collide.py`, docs/SIMULATION.md). No GPU path; one radius per
  particle, no particle-versus-splat or -volume collision, `iterations` is a fixed count rather than
  an early-exit solver.
- [x] Volumes, VDB import and a CPU 3D smoke and fire solver with its nodes exist (L6 steps A to C, `docs/FLUIDS_SPIKE.md`); the GPU-resident solver (multigrid pressure, GPU substep, sparse tiles, L6 step D) is built; FLIP liquids with a level-set mesh and a splash tag are built on the CPU (L6 step E; a GPU-resident FLIP and liquid refraction are not).

Gate status, fluids (L6 step 3, 2026-09-25; evidence and numbers in `docs/FLUIDS_SPIKE.md`, code in
`nodebased/fluid2d.py`, `tests/test_fluid2d.py`, `tools/benchmark_fluid.py`):

- [x] Solver-versus-library evaluation. Decided: no ecosystem library is embeddable on cp312 Linux and
  Windows (OpenVDB conda-only, Mantaflow no wheel, PhiFlow slow and torch-bound, Taichi a kernel
  language). Recommendation: a custom solver, with import-only as the fallback; awaits DiMo's choice.
- [x] Importing and rendering separated from solving (the volume member and renderer come first, shared by both routes).
- [x] Reproducible determinism, restart and checkpoint, cancellation and cost measurement, in 2D only
  (bit-identical runs, `simcache` checkpoints, mid-solve cancel, 256 and 512 squared frame times).
- [ ] Volume scene member, `Render3D` volume drawing and a VDB reader: not built. No pip route exists for VDB.
- [ ] 3D solver: extrapolated only (about 3 s per substep at 128 cubed in NumPy); not built or measured.
- [ ] Fluid collision fixtures and resource budgets for volume caches (a 256 cubed checkpoint is about 400 MB).

### 6. Parity audit and extensions
Maintain supported/partial/missing rows against verified Nuke 17.1 workflows,
with a runnable acceptance scene per claimed feature. Compare rendering quality
and throughput under equivalent settings/hardware; no blanket "better renderer"
claim without published measurements. Full parity remains unverified until the
audit closes, including applicable legacy workflows.

## Required deliverables (owner requirements, 2026-09-19)

Each item needs real tests and a runnable acceptance scene before it is called supported. None is
implemented yet unless a later section says so.

- **A. Gaussian splats.** 3DGS `.ply` import preserving covariance, opacity and SH; camera-correct
  anisotropic rendering; mesh/splat depth interaction. Gate: reference asset and image, ordering
  artifacts, transparent compositing, bounded memory, measured throughput.
- **B. Gaussian splat relighting**, in both the 3D viewport and the final render. Baked SH colour
  alone does not satisfy this. Splats must respond to scene lights, cast and receive shadows, and sit
  inside a ray-traced render with ordinary geometry (reference: V-Ray in Houdini). The design has to
  state its approximations: per-splat normal estimation (shortest covariance axis, oriented toward the
  views), the split of baked radiance into an albedo-like term and lighting, the light response on the
  GPU path, and shadows through the ray-traced path. The viewport is an interactive approximation and
  the render is ray-traced quality; each must be labelled as such. Gate: light-move and shadow
  fixtures, splat/mesh mutual shadowing, viewport-versus-render comparison.
- **C. Alembic (`.abc`)**: meshes, cameras, xforms, time-sampled animation. There is no PyAlembic wheel
  on PyPI and the PyPI package named `alembic` is the SQLAlchemy migration tool: never install or depend
  on it. A packaging spike (`docs/3D_ALEMBIC_SPIKE.md`) comes first: a pure-Python/NumPy Ogawa reader for
  the needed subset, a vendored compiled reader with Linux and Windows wheels, or another verified
  pip-installable route. No Alembic node until a route works on both platforms.
- **D. USD (`.usd/.usda/.usdc/.usdz`)** through the optional `usd-core` wheel (cp312 for Linux and
  Windows; license field `LicenseRef-TOST-1.0`): stage load, meshes with UVs and normals, cameras,
  xforms, time samples, layer composition as USD resolves it, and export after import. Optional extra
  like `gpu`, with clean degradation and a clear error state when missing. `usd-core` does not include
  the usdAbc plugin, so USD does not solve Alembic.

### Status of the required deliverables (as of the 0.24.0 branch)

- **A. Gaussian splats: implemented for the CPU reference.** `.ply` reader/writer (3DGS binary and ASCII,
  SH degrees 0-3), `ReadSplat3D`, EWA rendering with the reference Jacobian clamp, per-pixel depth
  ordering against opaque and transparent meshes in both render modes, splat data passes and a `splats`
  output, banded memory (150 MiB at 1080p), a tile-work budget (2 billion evaluations, refusal for now).
  Tested with generated fixtures, one third-party-generated file and one real 3.4-million-splat capture
  (read-only, not in the repository). Gaps: the GPU renderer covers only `rgba` with opaque meshes and unshadowed relighting (the viewport shows a
  proxy), CPU renders of real captures take tens of seconds to minutes, the shading AOVs ignore splats, `.splat`/compressed formats are
  not read.
- **B. Splat relighting: implemented for the CPU final render.** `Relight` mixes baked and re-lit colour from
  estimated normals and the SH DC term; meshes and splats shadow relit splats and splats shadow meshes
  through a splat BVH. Approximations are stated in 3D_FOUNDATION.md (baked lighting is not removed,
  normals are guesses, closest-approach shadows, residual self-shadowing about 3%). The viewport shows a
  relit layout proxy without shadows. Gaps: no GPU path, shadows are slow (tens of seconds for 200,000 splats),
  no viewport-versus-render comparison beyond the proxy's labelling. Update 2026-09-26 ("Splat relighting 2"):
  de-lighting (step B) and environment light with physically based shading, traced visibility and mesh
  reflections (step C) are CPU-reference features with the GPU drawing the same per-splat colours; see
  SPLAT_RELIGHTING.md and 3D_FOUNDATION.md "Physically based splat shading". The viewport still shows neither.
- **C. Alembic: partial.** In-house pure-Python Ogawa reader, `ReadAlembic3D`, `ReadAlembicCamera3D`.
  Blender-authored fixtures only; op-stack transforms verified with hand-built arrays; no curves, points,
  subdivision or materials; every visible mesh is decoded per evaluation; Windows unverified.
- **D. USD: partial.** `ReadUSD3D`, `ReadUSDCamera3D`, USD export from `WriteGeo3D` behind the optional
  `usd` extra; meshes, cameras, transforms, time samples, composition, up axis and units. No materials,
  lights, point instancers or camera export; the stage is opened twice per evaluation (fingerprint and
  load) and the cache key includes the frame even for static stages.
- **E. glTF 2.0: meshes and base colours.** `ReadGLTF3D` (in-house reader, no extra package): `.glb` and
  `.gltf`, hierarchy baked, base colour factor and texture, quantized and interleaved attributes. No
  cameras, animation, skins, morph targets, vertex colours, Draco or KTX2. Written for the image-to-3D
  tools (Pixal3D, WorldSculpt, SAM 3D), whose GLBs are the target; verified so far on generated files only.

## Design: relight passes and multichannel plumbing (written before code, 2026-09-21)

**Status (2026-09-22):** the `Raster.layers` field, `Render3D`'s `relight` bundle output, and the
`Relight` 2D node are implemented and tested (docs/3D_FOUNDATION.md "Relight passes"). Deliverable
L4.1 is done. Item 7 (multichannel EXR) is done too (2026-09-25, L4 plan 2 step H): `Render3D` has a
`multichannel` output whose `passes` knob (beauty, normals, depth, relight, and the volume_* layers) fills `Raster.layers` under
Nuke layer names; `Write` writes them as one EXR part and `Read` returns them (docs/3D_FOUNDATION.md
"Multichannel output"). It reuses this bundle for the per-light layers, so it keeps the bundle's limits.

It designs deliverable L4.1: a `Render3D` output that bundles several passes from one evaluation, and
a 2D `Relight` node that recombines them the way Nuke's `Relight` does. Deliverable L4.7 (multichannel
EXR) reuses the same bundle.

**Why one evaluation.** Getting albedo, normals, world position, and per-light diffuse/specular
today costs one full `Render3D` evaluation per pass: five or more re-rasterizations of the same
triangles for one frame. A 2D `Relight` node that recolours lights after the fact needs those passes
together, and should not force the mesh to re-render for every light or colour tweak.

**Carrier: `Raster.layers`.** `Raster` gains an optional fourth field, `layers: dict[str, Raster] |
None = None`, default `None`. Every existing `Raster` construction path (`Raster.of`, `with_pixels`,
`aligned`, `fit`) is untouched and keeps producing `layers=None`, so nothing downstream that already
holds a `Raster` changes behaviour. Only the one node that populates `layers` and the one node that
reads it need to know the field exists. This is deliberately not a new typed connection kind (no
"layers" value type parallel to image/scene/camera): the bundle still flows down the graph as an
ordinary `image` connection, so `Viewer`, `Write` and every 2D node keep working on `.pixels`
unmodified; only `Relight` (and later multichannel `Write`) look at `.layers`.

**`Render3D` output `relight`.** A new `render_output` choice, `"relight"`, added to
`scene3d.RENDER_OUTPUTS` but to neither `LIGHT_OUTPUTS` nor `DATA_OUTPUTS` (it is its own kind, not a
single-channel image). `Render3D`'s evaluated `Raster` for this output carries the ordinary beauty
image as `.pixels` (so `Viewer`/`Write` show something sane if wired directly) and these channels in
`.layers`, each an ordinary premultiplied `Raster`:

- `albedo`, `normals`, `position`: identical values to the existing single-purpose outputs of the
  same name (`normals`/`position` are first-hit, unantialiased, exactly like today's data outputs;
  `albedo` is the premultiplied flat/textured surface colour before lighting).
- `diffuse`, `specular`, `emission`: identical to today's single-purpose outputs (kept for
  convenience and as a parity check against the per-light channels below).
- `diffuse_L0`, `diffuse_L1`, ... and `specular_L0`, `specular_L1`, ..., one pair per enabled light
  (`light.intensity > 0`), in the same order as `scene.lights` (the order the light nodes were wired
  into `Scene3D`). These are **unitless response terms**, not multiplied by the light's colour or
  intensity: `diffuse_L{i}` is `max(dot(N, to_light), 0) * shadow_visibility` (the traced shadow term,
  when that light has `Shadows` on, is already folded in here — this is the expensive part, computed
  once); `specular_L{i}` is `geometry.specular * pow(max(dot(N, half), 0), shininess) * front *
  shadow_visibility`. A 2D node can recombine them with **new** light colours and intensities without
  re-tracing: `diffuse_total = albedo * (ambient + sum_i diffuse_L{i} * color_i * intensity_i)`,
  `specular_total = sum_i specular_L{i} * color_i * intensity_i`. Summed with the render's original
  ambient and lights, this reproduces `diffuse`/`specular` exactly (a tested identity).

**Known limits of this milestone, stated up front rather than discovered by a user:**
- `relight` output is **raster-mode only**; `Render3D` `Mode` `raytrace` raises a clear error for it
  (the GPU/raytrace ports are separate future work, matching how every other output gained raytrace
  and GPU support one milestone at a time).
- `relight` output does **not supersample**; it always renders at one sample regardless of the node's
  `Samples` knob, like the existing data outputs. Antialiasing the bundle is future work.
- `relight` output for this milestone does **not include splats**; a scene with splats raised a clear error
  rather than silently omitting them. Splat-only scenes are supported since Splat relighting 2 step B (see
  docs/3D_FOUNDATION.md, "Delight (intrinsic decomposition)"); a scene that mixes splats with geometry or
  particles still raises (`"the relight bundle output does not support scenes with splats that also hold
  geometry or particles yet"`).
- The per-light channels use the **render's own camera** for the eye/half-vector; wiring a different
  `Camera3D` into `Relight` does not re-project specular. `Relight`'s `Camera3D` input is accepted for
  forward compatibility (later milestones that need it) but is not read by this milestone's math; this
  is stated in the node's own docs so nobody is misled by an unused input.

**The `Relight` node.** A 2D node: one `image` input (must carry the `relight` bundle; a clear error
names the required `Render3D` output when it does not), one optional `camera` input (`Camera3D`,
unused for now, see above), and `light0`..`light7` optional inputs (`Light3D`, the same eight-slot
pattern as `Scene3D`'s `object0`..`object7`), paired by **index** with the `diffuse_Li`/`specular_Li`
channels — wire lights to `Relight` in the same order they were wired into the original `Scene3D`.
Knobs: `Ambient` (colour, replaces the render's own ambient for the relit term), `Diffuse` and
`Specular` (0..1 multipliers on the recombined terms, a bounded scalar so sliders are right), `Mix`
(0..1, blend between the original beauty and the relit result, `0` reproduces the input exactly).
Output: `mix * (albedo * (ambient + sum_i diffuse_Li * light_i.color * light_i.intensity * diffuse_knob)
+ sum_i specular_Li * light_i.color * light_i.intensity * specular_knob) + (1 - mix) * input.pixels`,
alpha unchanged from the bundle's coverage. A wired light beyond the channel count, or fewer wired
lights than channels, contributes/loses that light's term; document the pairing-by-index rule plainly
so a mismatch is a user error, not a silent one.

**Order of work (this deliverable only):** the `Raster.layers` field and the `relight` bundle output
on `Render3D` first, with golden-array tests proving every *existing* output is bit-identical to
before; then the `Relight` node.

Rendering status (milestone 3): shadows with per-light bias, blur and samples for soft shadows (L4 step B),
Blinn-Phong specular and emission, named AOVs (one per `Render3D`), a CPU BVH and CPU ray-traced mode, wgpu raster with shadows (brute-force or BVH per adapter type).
GPU shadows on relit splats and splat shadow catching are built (L4 step D). Not built: transparent-mesh layering with splats on the GPU, GPU splat AOVs,
reflections, global illumination, physically based materials, shadows
in the viewport, per-object shadow flags.

## Design: Gaussian splats and relighting (written before implementation)

Status: the data model, `.ply` reader/writer, SH evaluation, a CPU baked-colour renderer with mesh depth
interaction and the `ReadSplat3D` node exist (see 3D_FOUNDATION.md); GPU rendering, relighting, shadows and the viewport display
are not built. Splat relighting (per-splat Lambert from estimated normals and SH-DC albedo, `Relight` mix) with shadows from meshes
and from other splats onto relit splats now exists on the CPU; splats shadowing meshes, the GPU paths and the
viewport display are still design only.

**Data model.** A splat cloud is arrays of N Gaussians: position (3), scale (3, stored in log space in
3DGS files, linear in memory), rotation quaternion (4, w x y z, normalised), opacity (logit in files,
0..1 in memory) and SH coefficients (degree 0-3, DC plus rest, RGB). The 3D covariance is R S S^T R^T.
Clouds sit in the scene beside triangle geometry (`Scene.splats`) and inherit `Scene3D` transforms.
3DGS captures are usually COLMAP-oriented (+Y down, +Z forward), so the reader offers an orientation
choice rather than assuming Y-up; SH colour is treated as sRGB-encoded (that is how 3DGS trains) and is
converted to scene-linear unless told otherwise.

**Rendering (baked look).** Each splat is projected to a 2D Gaussian with the EWA/Jacobian approximation
(covariance through the perspective Jacobian plus the 3DGS 0.3 px low-pass dilation), coloured by
evaluating SH for the view direction, sorted by view depth and alpha-composited front to back with
alpha = opacity x exp(-1/2 d^T Sigma2d^-1 d), skipping alpha below 1/255. Meshes render first; a splat
fragment is hidden where the mesh is nearer at that pixel (splat centre depth versus mesh depth).
Approximations to state in the docs: a splat is depth-tested by its centre, so a mesh cutting through a
splat does not slice it; transparent meshes are not depth-sorted against splats; there is no
order-independent handling of overlapping splats at similar depths beyond the sort.

**Relighting design (approximation, not inverse rendering).**
- *Normals.* The shortest scale axis of a splat is its estimated normal (the corresponding column of R),
  flipped per shaded ray to face the viewer. This is reliable for flat, surface-like splats and unreliable
  for near-isotropic ones; a confidence 1 - s_min/s_mid downweights the directional term toward a
  neutral (view-facing) normal for blobs.
- *Albedo versus lighting.* 3DGS stores baked radiance, not albedo. The view-independent SH DC term is used
  as the albedo-like colour; higher-order SH is treated as the view-dependent/specular residual of the
  original capture. The capture's own lighting stays baked into that colour, so relighting is honest only
  as a controlled blend: `Relight` mixes between the baked look and the re-lit look
  (albedo x (ambient + sum of light contributions)), and the docs must say the baked lighting is not
  removed.
- *Shading.* Lambert (and optionally the existing Blinn-Phong term) per splat from the estimated normal
  and every `Light3D`, evaluated at the splat centre, then composited like the baked colour.
- *Shadows and mutual shadowing (final render).* The ray-traced path puts triangles and splats in one BVH
  (splat = ellipsoid bounds, primitive kind array). Shadow rays from a mesh fragment to a light accumulate
  transmittance through splats (each contributes 1 - opacity x exp(-1/2 d_min^2), d the Mahalanobis distance
  of the ray's closest approach) as well as triangles; shadow rays from a splat centre go through both. Splats
  therefore cast shadows on meshes, meshes on splats, and splats on splats.
- *Viewport.* An interactive approximation: baked or Lambert-relit splats without shadow rays (or with a
  cheap shadow map), clearly labelled as preview; the final render is the ray-traced result.
- *Order of work.* reader + data model + baked CPU renderer (with mesh depth), then the GPU splat path and
  node, then relighting without shadows, then splats in the BVH for shadows, then the GPU ray-traced path.

## Where things stand

Shipped in 0.22.0: typed scene graph, card/cube/sphere/OBJ geometry, textured cards, nested
scenes, directional and point lights, antialiasing, depth and normal passes, near clipping, and
the navigable viewport with camera look-through. All on the CPU reference rasterizer.

Since 0.22.0 (0.23.0 and the 0.24.0 branch): camera projection with optional occlusion, OBJ/USD export,
USD and Alembic import, an optional wgpu backend, hard shadows, materials, named AOVs, a BVH and a
ray-traced mode, Gaussian splats with relighting and shadows, and a GPU viewport. See the status notes
above for what is partial.

Next, after 0.24.0, in order (the GPU splat path for `rgba` and the splat progress callback have landed; the
desktop app still has to consume the callback): the GPU ray-traced mode is complete for its planned scope (mesh beauty and AOVs, splat casters onto meshes; banded GPU
submissions with cancel points have landed); particles; volumes and fluids.
