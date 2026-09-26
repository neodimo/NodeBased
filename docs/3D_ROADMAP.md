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
- [ ] Instancing (a mesh per particle) is not built.
- [ ] GPU renderer and 3D viewport do not draw particles (requests written for L4 and L1).
- [ ] Colliders are frozen at the emitter's start frame; no particle-to-particle collisions.
- [ ] Volumes and fluids (L6), and importing simulation caches, are not started as nodes.

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
  no viewport-versus-render comparison beyond the proxy's labelling.
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
`multichannel` output whose `passes` knob (beauty, normals, depth, relight) fills `Raster.layers` under
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
