# NodeBased 3D roadmap

Status: milestone 1 shipped in 0.22.0 (2026-09-18), with the textured-card and hierarchy parts
of milestone 2 and the Lambert-light part of milestone 3. This document is a plan, not a list
of shipped capabilities; what exists is described in [3D_FOUNDATION.md](3D_FOUNDATION.md).

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

Status of D (unreleased): `ReadUSD3D`, `ReadUSDCamera3D` and USD export from `WriteGeo3D` exist behind the
optional `usd` extra, with fixture tests (composition, instancing, time samples, cameras, round trip);
materials, lights, point instancers and camera export are not done, so D is partial. Known
performance debt: each evaluation opens the stage once to fingerprint its layers and again to load it,
and the cache key always includes the frame, so a static stage reloads every frame; caching the layer
list per root-file identity and dropping the frame for stages without time samples is open work.

Status of C (unreleased): the packaging spike chose an in-house pure-Python Ogawa reader. `ReadAlembic3D`
and `ReadAlembicCamera3D` read meshes, xforms, cameras and time samples with fixture tests against a
Blender-written archive. Partial: Windows unrun, only Blender-authored files tested, transform op stacks
verified only by hand-built arrays, no curves/points/subd/materials, and every evaluated mesh is decoded
into RAM (no lazy streaming), so C is not closed.

Status of milestone 3 (unreleased): hard shadows (`Light3D` `Shadows`) run on the CPU reference renderer and
on the wgpu backend, both brute-force ray tests with work budgets; the GPU path is tested against the CPU
one. Not done: shadows in the viewport, per-object shadow flags, soft shadows, an acceleration structure,
materials, named AOVs, and a ray/path-traced mode.

Revised order after the wgpu backend slice: (USD import and occlusion-aware projection done); the Alembic
spike and import; shadows, materials, named AOVs and a ray/path-traced mode on the GPU backend (B needs
it); Gaussian splats, then splat relighting; particles; volumes and fluids.

## Where things stand

Shipped in 0.22.0: typed scene graph, card/cube/sphere/OBJ geometry, textured cards, nested
scenes, directional and point lights, antialiasing, depth and normal passes, near clipping, and
the navigable viewport with camera look-through. All on the CPU reference rasterizer.

Milestone 2 progress (unreleased): camera projection with a fixture suite, animated camera/geometry
round trips and OBJ export are implemented and tested on the CPU reference renderer. Occlusion-aware
projection is in as an approximate depth-map option. Still open in milestone 2: Alembic, USD materials/lights,
missing-asset policy beyond a clear error.

Milestone 3 spike done (unreleased): [3D_BACKEND_SPIKE.md](3D_BACKEND_SPIKE.md) measured wgpu and
moderngl against the CPU reference and chose wgpu as an optional extra. A wgpu raster path
(rgba, depth, normals, lights, textures, transparency) sits behind `Render3D`'s `Backend` knob with the
CPU renderer as reference and fallback.

Next, in order: shadows, materials, named AOVs and a ray/path-traced mode on that backend (milestone
3); then Gaussian splat import and rendering (milestone 4). Splats, ray tracing,
particles and fluids are not implemented, and a CPU NumPy rasterizer is the wrong place to
start them.
