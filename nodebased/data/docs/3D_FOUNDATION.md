# 3D in NodeBased

NodeBased has a 3D scene graph that renders into the ordinary 2D comp. Geometry, lights and a
camera are assembled by `Scene3D`; `Render3D` turns that into the same float32, scene-linear,
premultiplied RGBA image every other node produces, so Grade, Merge, Roto and Write simply
follow it.

This page says what exists today and, just as plainly, what does not. The long-range plan —
USD, ray tracing, Gaussian splats, particles, fluids, Nuke parity — is in
[3D_ROADMAP.md](3D_ROADMAP.md) and none of it should be inferred from this page.

## Nodes

| Node | Output | What it does |
| --- | --- | --- |
| `Card3D` | geometry | A flat card. Connect `image` to texture it: this is the 2.5D workhorse. |
| `Cube3D` | geometry | A cube; each face carries the full texture. |
| `Sphere3D` | geometry | A smooth-shaded lat/long sphere with a spherical UV map. |
| `ReadGeo3D` | geometry | A Wavefront OBJ from disk: polygons (triangulated), UVs and normals. |
| `Project3D` | scene | Projects `image` through a `Camera3D` onto `geometry` (a geometry or a whole scene). See below. |
| `ReadUSD3D` | scene | A USD stage as a scene (optional `usd-core`). |
| `ReadUSDCamera3D` | camera | A USD camera (optional `usd-core`). |
| `ReadSplat3D` | scene | A 3D Gaussian splat cloud from a 3DGS `.ply` (baked-colour rendering on the CPU only). |
| `ReadAlembic3D` | scene | Polygon meshes from an Alembic (Ogawa) `.abc` as a scene. |
| `ReadAlembicCamera3D` | camera | A camera from an Alembic `.abc`. |
| `WriteGeo3D` | scene | Passes its scene through and exports it to Wavefront OBJ on request. |
| `Light3D` | light | Directional or point light aimed from its position at its target. |
| `Camera3D` | camera | Position, target, roll, vertical field of view, near and far planes. |
| `Scene3D` | scene | Up to eight geometry, light or scene inputs under one transform. |
| `Render3D` | image | Renders `scene` through `camera` at its own width and height. |

Connections are typed. An image cannot be wired where a scene is expected, and a rejected
connection leaves the document untouched. Viewing a geometry, light, camera or scene node
reports that it is not an image rather than failing obscurely; view the `Render3D`.

Every geometry node, `Scene3D` and `ReadSplat3D` carry the same transform block, in Nuke's
order: rotation order (`XYZ` by default; `XYZ` means Rx @ Ry @ Rz, so Z acts first), translate,
rotate (degrees), scale, uniform scale (multiplies all three scales) and pivot (the point that
rotation and scale hold still). The matrix is T(translate) @ T(pivot) @ R @ S @ T(-pivot).
Documents saved before uniform scale, rotation order and pivot existed load with the identity
values and render exactly as before. Geometry nodes also have an RGBA surface colour. With a texture connected the colour tints it, so leave it white for the plate as shot.
All numeric parameters animate and accept expressions like any other knob.

**Hierarchy** is nesting: wire a `Scene3D` into another `Scene3D` and everything inside
inherits the parent's transform — lights included.

## Camera projection

`Project3D` is the Project3D-style 2.5D workflow: wire an `image`, the projecting `camera` and the
`geometry` (or scene) that receives it. The texture is looked up per pixel from the world position
of the surface through the projection camera, so it stays stuck to the geometry when the render
camera moves, and it is perspective-correct on any surface, not just cards. The projection camera
may differ from the render camera and animates like any other camera.

- Filmback aspect is the image's aspect; `fov` is vertical, as for every camera.
- `Outside` decides what lies beyond the projection frustum (outside the image, nearer than the
  camera's near plane, past its far plane, or behind it): `transparent` draws nothing there and does
  not occlude, `clamp` smears the edge pixels (surfaces behind the projector are still dropped).
- `Backfaces` `skip` drops surfaces facing away from the projection camera, which stops a plate
  from bleeding through to the back of a card or cube.
- `Occlusion` `depth` stops the image landing on surfaces that the projection camera cannot see
  because other geometry in the scene is in the way. It renders a depth map of the whole scene from the
  projection camera (texture aspect, longest side capped at 512 pixels) and rejects fragments behind it.
  It is an approximation: the comparison uses the largest finite depth among the 3x3 neighbours plus a
  bias of max(0.2% of depth, 0.001) so flat cards, spheres and cubes never shadow themselves, which
  leaves a thin leak along occluder silhouettes and softens shadow edges at the map's resolution. Any
  occluder pixel with alpha above zero occludes, transparent ones included. It costs one extra CPU
  depth render per render call (only when some geometry asks for it) and is CPU-only: a projected
  scene with `Backend` `auto` renders on the CPU, and `gpu` reports it as unsupported. Default `off`.
- The geometry's own UVs and texture are ignored while projected; the colour still tints.

## Exporting geometry

`WriteGeo3D` exports its upstream scene to a Wavefront OBJ: **Export current frame**, or
**Export frame range** with a padded path such as `geo.%04d.obj`. Every transform (including
`Scene3D` hierarchy and keyframed animation) is baked into world-space positions, so an animated
scene becomes one OBJ per frame. Positions, UVs and per-vertex normals are written and read back
by `ReadGeo3D`; colours, textures, lights, cameras and projections are not exported. The file is
written atomically and the text is deterministic.

## USD import and export (optional)

Install the optional extra: `pip install nodebased[usd]` (`usd-core`, license field
`LicenseRef-TOST-1.0`). Without it the USD nodes report a clear error and nothing else changes.

- `ReadUSD3D` loads a `.usd/.usda/.usdc/.usdz` stage as a scene: meshes with UVs and normals, world
  transforms baked in, evaluated at the current frame (the frame number is used directly as the USD
  time code; there is no fps conversion). Composition (sublayers, references, variants, native
  instances) is resolved by USD. `Root prim` limits the load to a subtree. Invisible prims and
  non-`default`/`render` purposes are skipped.
- `ReadUSDCamera3D` loads a perspective camera (first one, or a chosen prim) into the ordinary camera
  type: position, orientation, roll, vertical FOV from focal length and vertical aperture, clipping.
  Horizontal aperture/film-back aspect, lens distortion, depth of field and shutter are ignored;
  non-uniform scale, shear and orthographic cameras are rejected.
- **Up axis and units.** A Z-up stage is rotated to Y-up on import (stage +Z becomes +Y), for meshes and
  cameras. `metersPerUnit` is applied when the stage authors it, so a centimetre stage comes in at
  metre scale; a stage that never authored it is left alone (USD's 0.01 fallback would shrink
  hand-written stages). Export always writes Y-up and 1 m, so Y-up/1 m files round-trip unchanged.
- Not loaded: materials (only constant `displayColor`/`displayOpacity`), subdivision (base cage is
  used), point instancers, curves, volumes, lights. Face holes are dropped.
- The render cache is keyed on the root file plus every file-backed layer the stage uses, so editing a
  sublayer re-renders.
- `WriteGeo3D` chooses the writer by extension. A `.usd/.usda/.usdc/.usdz` path writes meshes (points,
  normals, `st` UVs, display colour). **Export current frame** writes a static stage; **Export frame
  range** writes one file with time samples (topology must not change across frames); a padded
  pattern writes one file per frame instead. Import followed by export followed by import reproduces
  the geometry in tests.

## Alembic import

`ReadAlembic3D` and `ReadAlembicCamera3D` read Alembic `.abc` files with an in-house, read-only, pure
Python/NumPy Ogawa reader (`nodebased/alembicio.py`). It needs no extra package and never uses the
PyPI package called `alembic`, which is an unrelated database tool.

- **Meshes** (`AbcGeom_PolyMesh_v1`): positions, topology, indexed or face-varying UVs and normals. World
  transforms (including animated parents and the `inherits` flag) are baked in, points are interpolated
  linearly between the two stored samples around the requested time, faces are fan-triangulated, and the
  winding Alembic stores is reversed so front faces are counter-clockwise. Hidden objects, and objects under
  a hidden parent, are skipped. `Root object` limits the load to a subtree.
- **Cameras** (`AbcGeom_Camera_v1`): position, orientation and roll from the camera's transform chain,
  vertical FOV from focal length and vertical aperture, clip planes, focus distance as the look-at
  distance. Film offsets, lens squeeze, overscan, shutter and depth of field are ignored.
- **Time.** Node time in seconds is frame divided by the document fps, so frame 1 at 24 fps reads
  t = 1/24 s, which is how Blender writes its archives. There is no offset knob yet.
- **Units and axes** are used as authored: Alembic carries no unit metadata and exporters convert to
  Y-up.
- **Skipped:** curves, points, subdivision meshes, NURBS, face sets and materials are not loaded; the
  inspector says how many objects were skipped. HDF5-backed archives are rejected with a clear message.
- **Memory.** The file is memory-mapped, not read whole, and no file handle stays open after a read. Each
  evaluation still decodes every visible mesh into memory as NumPy arrays and bakes the whole scene, so a
  cache with huge meshes needs that much RAM per evaluated frame. There is no lazy or per-object streaming
  yet. Large production caches are unmeasured.
- **Verification.** Tested against one small archive written by Blender 5.3, with hand-derived constants.
  Transform op stacks are verified only with hand-built arrays; archives exported from Maya, Houdini or
  other tools have **not** been tested. Windows has **not** been run.

## Rendering

- **Unlit until lit.** A scene with no lights renders surfaces at their authored colour and
  texture, which is what projecting plates onto cards wants. Add a `Light3D` and surfaces become
  Lambert-shaded by the lights plus `Render3D`'s `ambient`. Surfaces are two-sided.
- **Materials.** Every geometry node has `Specular` (0..1), `Shininess` (exponent) and `Emission`.
  Specular is Blinn-Phong: for each light, `specular x light colour x intensity x max(N.H, 0)^shininess`,
  white (the light's colour, not tinted by the surface), zero where the surface faces away from the light,
  multiplied by shadow visibility and by the surface alpha (output stays premultiplied); ambient gives no
  specular. Emission adds the surface's own albedo (colour x texture, premultiplied) times `Emission`, lit
  or not, unshadowed. All default to 0, so old documents render as before. Same formulas on the CPU
  renderer and the wgpu backend, which are tested against each other. Not implemented: metalness, physically
  based (GGX) lobes, reflections, transmission, textured material maps, USD/Alembic materials.
- **Shadows** (CPU reference only). `Light3D` has a `Shadows` knob (off by default; old documents are
  unchanged). With it on, `Render3D` traces a ray from every shaded fragment to the light through all
  triangles in the scene, so every geometry casts and receives shadows; there are no per-object flags yet.
  A hit multiplies the light by (1 - the geometry's colour alpha), so an alpha 0.5 blocker halves it and
  alpha 0 casts nothing; texture alpha is not considered. Ambient is never shadowed. Shadows are hard
  (no soft or area lights). On the CPU the rays run through a bounding volume hierarchy
  (`nodebased/raytrace.py`, built once per render; scenes of 64 triangles or fewer use a plain loop), with
  results identical to the brute-force test. Measured for 960x540 rays (one per pixel) on this machine: about
  3.1 s for a 10,002-triangle scene and 4.0 s for 99,858 triangles (roughly 130,000-165,000 rays/s), plus a
  61-590 ms build; a 250,000-primitive build took about 1.5 s. A render whose estimated cost (rays x 16 x
  log2 triangles, plus the build) exceeds the built-in budget is refused with an error rather than hanging.
  The 3D viewport does not show shadows. The wgpu `Backend` implements the same shadow rules
  (same bias, alpha transmission and light handling) and is tested against the CPU reference on the same
  scenes (interior agreement and shadow edges within one pixel). Two GPU paths exist: a brute-force loop
  over all triangles per shadowed fragment, and BVH traversal in the fragment shader (used only when the
  adapter exposes enough storage buffers; otherwise the brute loop runs and results are identical).
  Which one runs depends on the adapter type and triangle count (discrete GPU above 20,000 triangles,
  integrated above 5,000, software above 500), because the measured winner differs: on the RTX 3080 Ti
  the brute loop was faster than the BVH up to 40,000 triangles (shadow cost 99 ms versus 286 ms at
  960x540, 40,002 triangles) while on the Radeon 8060S the BVH was about 4x faster at that size
  (283 ms versus 1,105 ms) and on llvmpipe faster from about 1,000 triangles. Measured brute throughput was
  roughly 40-300e9 pair tests/s (RTX 3080 Ti, noisy), 19-60e9 (Radeon 8060S) and 0.4-0.5e9 (llvmpipe).
  A first version of these numbers (15e9/s) was wrong: it divided whole-render time, which is dominated by
  host-side preparation (about 270 ms at 10,000 triangles and 1.1 s at 40,000), by the shadow work. The
  GPU refuses a render above a per-adapter-type work budget up front (brute: 4e10 discrete, 1e10
  integrated, 3e8 software, 2e9 unknown; BVH estimate units: 4e8, 4e8, 1e7, 2e8) meant to keep one
  submission near a second. The BVH estimate (16 x log2 triangles per ray) is an average: scenes with long thin overlapping
  triangles can cost far more per ray and are not guarded. The budgets refuse rather than split work: at
  1920x1080 with one shadowed light a discrete GPU refuses roughly 19,000 triangles or more on the brute
  path and the BVH path's 4e8 estimate is exceeded at similar sizes; tiled submissions (which would also
  give cancellation points) are planned, not built. A submitted GPU job cannot be cancelled, so the budget is the only safeguard;
  other adapters and Windows are unmeasured. GPU frame time for large meshes is currently limited by
  per-triangle host preparation, not by the shader. Projected geometry still renders on the CPU.
- **Mode.** `Render3D` has a `Mode` knob: `raster` (default, what every existing document uses) or
  `raytrace`, a CPU-only alternative that finds visibility by casting one ray per sub-sample through the
  bounding volume hierarchy instead of rasterizing triangles. It shares the shading code with the rasterizer
  and is tested for agreement with it on every output (beauty and all AOVs, interior pixels within 1e-4, at 1
  and 2 samples): same lights, shadows, specular, emission, textures, projections, transparency composited
  front to back, near/far clipping and antialiasing sample positions. It adds no new lighting features yet
  (no reflections, soft shadows or global illumination), and at present it is a foundation, not a quality
  upgrade. Per-pixel it sorts hits along the ray, so interpenetrating transparent surfaces composite correctly
  where the rasterizer's per-triangle sort can be wrong. Hits are found in bounded batches of the 8 nearest surfaces per ray (depth peeling), so
  surfaces hidden behind an opaque one cost almost nothing (70 stacked opaque cards render, as in the
  rasterizer). At most 64 surfaces may be composited along one ray, counting every shaded surface up to and
  including the first one that stops it (alpha 0.999 or more); a ray that would composite more, for example
  70 stacked half-transparent cards, raises an error. and a render over the CPU budget is refused. Measured once, 20,166 triangles at
  320x180, 1 sample, shadows on: rasterizer 4.57 s, ray-traced 1.01 s (CPU, this machine). It is
  CPU-only: `Backend` `auto` renders it on the CPU and `gpu` reports it as unsupported, and the viewport
  stays on the rasterizer.
- **Ties between coincident surfaces.** When surfaces of different colours sit at exactly the same
  distance along a ray, the ray-traced mode composites them in ascending primitive order front to back,
  while the rasterizer composites stable ties back to front, so their beauty results can differ for such
  exactly coincident geometry (transparency amounts still agree).
- **Gaussian splats (CPU reference renderer only).** `ReadSplat3D` loads a 3DGS `.ply` into the scene
  (`Scene.splats`, read by `nodebased/splats.py`) and `rgba` renders composite it:
  each Gaussian is projected with the EWA/perspective-Jacobian approximation plus the 3DGS 0.3 px
  low-pass, coloured from its spherical harmonics for the view direction (sRGB converted to scene-linear),
  sorted by depth and alpha-composited front to back. Splats and meshes are ordered per pixel: a splat fragment's depth is where the pixel ray meets the
  plane through the splat centre with its estimated normal (the shortest axis), falling back to the centre
  depth for grazing planes, when the plane depth strays more than 3 scale units, and for near-isotropic
  splats (smallest scale over the middle scale above 0.8), whose estimated normal is arbitrary; every mesh fragment
  (opaque or transparent) is merged with the splat fragments in exact depth order, so a splat can sit between
  two transparent cards or be partly hidden by an opaque mesh along the line where the surfaces cross.
  Each splat's screen-space covariance uses the 3DGS reference's clamped Jacobian (x/z and y/z limited to
  1.3 x tan(fov/2), the screen position stays unclamped), so large splats just outside the frame no longer
  smear across it. Measured on a real 3,409,742-splat outdoor capture (a CC BY 4.0 scan; the file is not
  part of the repository) at 640x360 on the CPU: about 51 s per frame (1,287 million
  tile evaluations, see the budget note below); the CPU reference is a correctness tool, and captures of that size
  need the GPU splat path (not built) for interactive use. Splats among themselves stay ordered by centre depth (as 3DGS does), so overlapping splats at nearly equal
  depth can still blend in the wrong order. Scenes with only opaque meshes use a fast path; when transparent
  meshes are present the mesh fragments come from primary rays even in `raster` mode (both modes then give
  identical results); at most 16 mesh surfaces may lie in front of a ray's terminating surface. The layered path works in
  horizontal bands sized so its mesh-layer buffers stay near 64 MiB whatever the resolution (measured: a
  1920x1080 frame with one transparent card and a small cloud peaked at 154 MiB, 272 MiB at 2x2 supersampling;
  the full-frame version needed about 0.9 GiB and 3 GiB), with identical results for any band size. Plain `raster`
  renders (beauty and `splats`) with only opaque meshes use the rasterizer's own depth buffer, not primary
  rays, and are not subject to the ray-traced budget (1920x1080, a 20,000-triangle opaque grid plus one
  splat: 2.0 s at 1 sample, 2.7 s at 2, the same as without the splat). In the data passes the mesh values
  come from the rasterizer exactly as without splats, and a splat replaces a pixel only where its first hit
  is nearer than the mesh's. In and differs slightly from a pure rasterizer render
  at silhouette edges (measured up to 3e-4 in a textured test scene). Splats also appear in the data passes and in a splat-only pass. In `depth`, `normals`,
  `position`, `uv` and `object_id` a pixel's first hit is either the first mesh fragment with alpha above zero
  or the depth where the splats' accumulated opacity first reaches 0.5, whichever is nearer along the ray;
  a splat hit reports its per-pixel plane depth, its estimated normal (flipped to face the viewer), the world
  position of that point, zero UVs, and object id = number of meshes + 1 + the splat instance's index.
  The `splats` output (`Render3D` `Output`) is the splats' premultiplied contribution to the beauty pass,
  attenuated or hidden by meshes in front of them, without the mesh colour; it is all zeros without splats.
  The shading passes (`albedo`, `diffuse`, `specular`, `emission`) still ignore splats, because splats have
  no shading model until `Relight` is used (below). The wgpu backend does not render splats (`auto` uses the CPU). The 3D viewport shows splats as a layout proxy only (see "The 3D viewport").
- **Splat relighting (CPU).** `ReadSplat3D` has a `Relight` slider (0 = the baked
  colours, exactly as before; 1 = re-lit). Per splat, at its centre, the colour becomes
  `albedo x (ambient + sum of Lambert x light colour x intensity)` where the albedo is the SH DC term
  (view-independent colour) and the normal is the splat's shortest axis flipped to face the camera; splats
  that are nearly round (smallest scale close to the middle scale) have no reliable normal and use the
  viewer-facing direction instead, blended by `1 - s_min/s_mid`. The render's ambient and every enabled
  `Light3D` (directional or point) apply; the result is mixed with the baked colour by `Relight`. What this
  is not: the capture's own lighting is baked into the DC colour and is not removed, so relighting an
  unevenly lit capture double-lights it; normals are guesses from splat shape; specular and higher-order SH
  are not re-lit; the shading passes (`albedo`,
  `diffuse`, `specular`, `emission`) still ignore splats. The shading is a reusable function
  (`nodebased/splatshade.py`, `shade_splats`) meant to be called by the viewport later.
  **Shadows on relit splats (CPU).** For every `Light3D` with `Shadows` on, each relit splat sends a ray from its
  centre to the light. Meshes shadow it through the same triangle BVH as mesh shadows (alpha transmission), and
  other splats shadow it through a splat BVH: each splat along the ray attenuates the light by
  `1 - min(0.99, opacity x exp(-d^2/2))`, with d the Mahalanobis distance of the ray's closest approach to that
  splat (splats beyond 3 sigma are ignored). This is a closest-approach approximation of a Gaussian's optical
  depth, not a volume integral. So that a surface made of overlapping splats does not shadow itself, the emitter
  itself is skipped and other splats overlapping its own thickness (closest approach earlier than 2.5 x its
  largest scale) only count beyond that distance; on a sphere of splats this leaves a residual self-shadowing
  of up to about 3.5% (mean 3.2%) relative to the unshadowed relit render. Splats do not yet shadow meshes.
  Cost is not proportional to what the shared budget estimates: one shadowed light on 200,000 splats at
  640x360 took 86.5 s against 5.3 s without shadows (about 2,500 rays per second on that dense shell; the
  estimate is per-ray average and ignores how many overlapping splats each ray crosses), so treat splat shadows as
  a slow reference until a density-aware budget and the GPU path exist.
  **Timing (CPU, 1920x1080, 200,000 splats, a sphere shell):** baked 12.3 s, relit without shadows 12.1 s (measured
  while another job was running on the machine, so treat as approximate); relighting itself costs almost nothing, shadows
  are the expensive part. Splats closer to the camera than 0.2 view units are culled even when the camera's near plane is
  smaller (the 3DGS reference does the same; large splats right in front of the eye otherwise smear the foreground).
  For the 3D viewport, `nodebased/splatshade.py` offers `instance_colors` (the exact per-splat linear colours the
  final render uses, relit or baked) and `instance_geometry` (world positions, rotations, sizes, opacity); the CPU
  renderer uses the same functions, so a GPU splat drawer can match it.
  This is the baked-colour look only when `Relight` is 0. `ReadSplat3D` knobs: file, orientation
  (`as_authored` or `colmap`, the +Y-down/+Z-forward frame of 3DGS/COLMAP captures), colour space
  (`srgb` default), SH degree clamp, opacity and footprint multipliers, and Nuke-style transform fields
  (translate, rotate, scale as XYZ numeric fields, uniform scale, rotation order, pivot). The decoded cloud
  is cached per file (path, size, modification time, orientation, colour space; LRU, 4 clouds / about 1 GiB),
  so re-evaluating does not re-read the file. Tested with generated fixtures (a chequer plane of flat
  splats plus meshes in front and behind), and read once from a third-party-generated 3DGS-layout file (an
  image-to-splat tool's output, 1,161 splats, SH degree 3); no real photogrammetry capture has been tried. Timings measured on the CPU (synthetic
  clouds, 320x180): 1k splats 47 ms, 20k 488 ms, 100k 2.4 s; a budget refuses renders that would exceed
  2 billion tile evaluations (`SPLAT_WORK_BUDGET`). The budget counts tile work: the number of (splat, pixel)
  tests the accumulation will actually do (splats binned to each 16x16 tile times that tile's pixels), which
  predicts time far better than bounding-box pixel counts did (five test scenes with 21-51 million bounding-box
  pixel pairs took 1.0 s to 49 s; tile work ran at 16-17 million evaluations per second on all translucent ones).
  2 billion is roughly 120 s at that rate on the development machine; it is a runaway guard, not a promise
  (early exit on opaque scenes makes real renders faster, and other machines differ). The refusal message
  gives the estimate. Measured on the real capture: 1,287 million evaluations at 640x360 (estimate 78 s, actual
  about 51 s) and 1,726 million at 1280x720 (estimate 105 s); neither is refused. A progress callback so an
  interactive app could show time left instead of refusing is planned, not built.
- **Textures** are perspective-correct, bilinear, and mip-mapped per triangle so distant cards
  do not shimmer. Texture alpha is respected and stays premultiplied.
- **Transparency** composites in depth order. Opaque surfaces use the z buffer; transparent
  ones test it without writing. Interpenetrating transparent surfaces can still sort wrongly.
- **Clipping.** Geometry crossing the near plane is clipped, not dropped, so ground planes can
  run under the camera.
- **Fill rule.** A pixel centre lying exactly on an edge belongs to exactly one triangle (top-left rule),
  so shared edges of transparent geometry are never blended twice and never leave gaps.
- **Antialiasing** is `samples`×`samples` supersampling (1–4).
- **Output / AOVs.** `Render3D`'s `Output` selects one named pass per node (several AOVs mean several
  `Render3D` nodes, each re-rendering; there is no multichannel file output yet):
  - Beauty and shading passes, antialiased and composited in depth order. `rgba` includes the
    background colour; the others never do:
    `rgba`, `albedo` (colour x texture, unlit), `diffuse` (albedo x ambient plus shadowed Lambert; equals
    albedo when the scene is unlit), `specular` and `emission`. Identity, tested on CPU and GPU:
    `diffuse + specular + emission` equals `rgba` rendered over a transparent background.
  - Data passes, first hit only, never antialiased, background ignored, alpha = coverage: `depth`
    (view-space distance), `normals` (world space), `position` (world xyz), `uv` (texture or projection
    UVs, zero without UVs) and `object_id` (1-based index of the geometry in the scene, in the red channel).
    They are image data, not deep data. Transparent geometry with alpha above zero counts as a hit.
  - Measured GPU-versus-CPU differences on an RTX 3080 Ti: position/uv up to about 3.5e-4 (float32
    interpolation order); object ids match exactly.
- The background defaults to transparent, ready to Merge over a plate.
- Renders are cached like any other node and honour proxy tiers and cancellation.

**Backend.** `Render3D` has a `Backend` knob. `cpu` (the default, and what every existing document
uses) is the reference rasterizer described below. `auto` uses an optional wgpu GPU rasterizer
(`nodebased/gpu3d.py`) when the `wgpu` package (`pip install nodebased[gpu]`) and an adapter exist,
and silently falls back to the CPU renderer otherwise or when the scene uses something the GPU path
lacks (currently camera-projected geometry). `gpu` requires the GPU path and reports an error instead
of falling back. The GPU path renders rgba (flat colour, textures, Lambert lights, sorted
transparency, supersampling), depth and normals. It is tested for agreement with the CPU renderer on
interior pixels, not bit-identity: edge coverage differs slightly, textures are sampled in half
precision, and colour is float32 only on adapters that can blend float32 targets (otherwise half
precision). Timings and the backend decision are in [3D_BACKEND_SPIKE.md](3D_BACKEND_SPIKE.md); the
3D viewport has its own interactive wgpu renderer (`nodebased/viewportgpu.py`): meshes are uploaded once and
cached, each object is one draw call against a depth buffer, and orbiting rewrites only the camera matrix.
Measured on an RTX 3080 Ti at 960x600, a 64-segment sphere paints in about 1 ms per frame (the CPU reference
takes about 400 ms) and a 65,536-triangle sphere in about 1.4 ms. The viewport sorts transparency per object,
shades at most 16 lights, shows no shadows, and lets camera projections show through occluders. Without a
usable adapter it falls back to the CPU reference renderer and says so in its header line. The viewport
renderer has not been run on Windows.

The CPU renderer is a deterministic NumPy **CPU reference rasterizer**. It is correct and tested
against analytic answers, and it is not fast: think cards, primitives and modest meshes. Scenes
over 250,000 triangles are refused with an error instead of hanging the session. Measured GPU-versus-CPU timings, with their caveats, are in the spike document; nothing else here is a performance claim.

## Conventions

Right-handed, +Y up; the camera looks down −Z. World units are unitless. `fov` is vertical.
UV (0,0) is the bottom-left of the texture. Near/far are distances along the view axis.

## The 3D viewport

Toolbar → **3D viewport** opens a dockable editor view (it is saved with the workspace).

- Orbit: left drag · Pan: middle drag · Dolly: wheel · **F** frames the scene · **C** looks
  through the authored camera.
- It shows what `Render3D` will see — the scene wired into the `Render3D` upstream of the viewed
  node, evaluated through the real graph, so textures, animation, expressions and disabled nodes
  all match. Before anything is wired it shows the loose geometry in the document.
- Textures are evaluated at quarter resolution, and the view renders at half size while
  dragging, to stay responsive on the CPU renderer.
- The ground grid, axes, camera frustum and light markers are editor chrome: depth-tested
  against the scene and never part of a render. A headlight shades unlit scenes for legibility;
  that too is viewport-only.
- Navigation is local. Orbiting never edits `Camera3D`, so inspecting a shot cannot change it.
- **Gaussian splats are a layout proxy, not the render.** On the GPU every splat is an opaque
  camera-facing disc in its base (SH degree 0) colour, sized from the splat's middle axis and kept
  between 1 and 2.5 pixels in radius, depth-tested against meshes and editor lines. There is no
  blending and no view-dependent colour, so it reads like a coloured point cloud: enough to place
  cameras, lights and geometry against a capture, never a preview of the final look. `Relight`,
  `Opacity` and `Scale` are followed (relighting uses the viewport's lights and ambient, without
  shadows; splats fainter than 0.05 after `Opacity` are hidden). At most 1,000,000 discs are drawn
  per cloud; larger clouds are strided evenly and the bottom-left note says so ("1 in 4 of
  3,409,742"). Moving the node re-uploads nothing. Measured on an RTX 3080 Ti at 1280x720, the
  3.4M-splat Nelson Ghost Town capture paints in about 4 ms per frame after a 0.3 s first upload.
  Without a GPU the fallback renders the meshes and marks up to 200,000 splat centres per cloud
  as depth-tested 2 x 2 points, ignoring `Relight`; it never runs the splat rasterizer, which
  takes seconds to minutes per frame and refuses large captures. **F** frames the 10th to 90th
  percentile box of a cloud widened by a quarter, because captures wrap their subject in a far
  shell of sky and haze splats that would otherwise push the view out.

## Not here yet

Geometry/camera import beyond OBJ (FBX; Alembic covers meshes and cameras only, no curves/points/subd/materials; USD covers mesh import/export and camera import only, with no USD materials or lights), materials, viewport shadows, soft shadows, GPU cancellation and per-triangle host-side preparation cost,
physically based specular, motion blur, depth of field, deep output, ray tracing, Gaussian splats, particles,
fluids, GPU support for projected geometry, ray tracing on the GPU, and in-viewport transform handles.
