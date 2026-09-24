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
| `Card3D` | geometry | A flat card. Connect `image` to texture it: this is the 2.5D workhorse. `rows`/`columns` (Nuke's own names) subdivide it into a grid of quads with correct UVs; a subdivided plane is just a `Card3D` with `rows`/`columns` above 1. |
| `Cube3D` | geometry | A cube; each face carries the full texture. |
| `Sphere3D` | geometry | A smooth-shaded lat/long sphere with a spherical UV map. `rows` (latitude bands) and `columns` (longitude segments) are Nuke's own names; the pole rings emit no degenerate or duplicated triangles. |
| `Cylinder3D` | geometry | A cylinder along +Y: `cyl_radius`, `cyl_height`, `rows` (height segments), `columns` (radial segments), and `cyl_caps` (`closed` or `open`) for flat end caps. |
| `ReadGeo3D` | geometry | A Wavefront OBJ from disk: polygons (triangulated), UVs and normals. |
| `Axis3D` | scene | A pure Nuke-style transform: parents whatever geometry, light or scene is wired into its one optional input. Chaining `Axis3D` nodes composes transforms in order through ordinary scene nesting. With nothing wired it is an empty scene. |
| `TransformGeo3D` | geometry | Bakes its transform into the input geometry's own vertices (and normals, through the inverse transpose), rather than adding a parent matrix. Unlike `Axis3D`, this acts before the geometry's own transform and any further parenting. |
| `Project3D` | scene | Projects `image` through a `Camera3D` onto `geometry` (a geometry or a whole scene). See below. |
| `ReadUSD3D` | scene | A USD stage as a scene (optional `usd-core`). |
| `ReadUSDCamera3D` | camera | A USD camera (optional `usd-core`). |
| `ReadSplat3D` | scene | A 3D Gaussian splat cloud from a 3DGS `.ply` (baked-colour rendering on the CPU only). |
| `ReadAlembic3D` | scene | Polygon meshes from an Alembic (Ogawa) `.abc` as a scene. |
| `ReadAlembicCamera3D` | camera | A camera from an Alembic `.abc`. |
| `ReadGLTF3D` | scene | Meshes from a glTF 2.0 `.glb` or `.gltf`, with base colours and textures. |
| `WriteGeo3D` | scene | Passes its scene through and exports it to Wavefront OBJ on request. |
| `Light3D` | light | Directional or point light aimed from its position at its target. |
| `Camera3D` | camera | Position, target, roll, film back (`focal`, `haperture`, `vaperture`), near and far planes; the field of view is derived. See below. |
| `Scene3D` | scene | Up to eight geometry, light or scene inputs under one transform. |
| `Render3D` | image | Renders `scene` through `camera` at its own width and height. |
| `Relight` | image | A 2D node: recombines `Render3D`'s `relight` bundle with new light colour/intensity, in comp. |

Connections are typed. An image cannot be wired where a scene is expected, and a rejected
connection leaves the document untouched. Viewing a geometry, light, camera or scene node
reports that it is not an image rather than failing obscurely; view the `Render3D`.

Every geometry node (including `Cylinder3D`), `Scene3D`, `Axis3D`, `TransformGeo3D` and
`ReadSplat3D` carry the same
transform block, in Nuke's order: rotation order (`XYZ` by default; `XYZ` means Rx @ Ry @ Rz, so Z
acts first), translate, rotate (degrees), scale, uniform scale (multiplies all three scales) and
pivot (the point that rotation and scale hold still). The matrix is
T(translate) @ T(pivot) @ R @ S @ T(-pivot). Documents saved before uniform scale, rotation order
and pivot existed load with the identity values and render exactly as before. Geometry nodes also
have an RGBA surface colour. With a texture connected the colour tints it, so leave it white for
the plate as shot. All numeric parameters animate and accept expressions like any other knob.

**Two ways to move geometry.** `Axis3D` only ever adds another parent matrix — like wiring
something into a `Scene3D` with one slot — so a chain of `Axis3D` nodes composes exactly as nested
`Scene3D`s do, and the object it carries (geometry, light or a whole scene) keeps its own identity.
`TransformGeo3D` instead bakes its transform straight into the incoming geometry's own vertices and
normals, so the result is new geometry at rest, with an identity transform of its own; anything
wired above it (another `Axis3D`, a `Scene3D`) still parents the *baked* geometry as usual. Disabling
either passes its input through: `Axis3D` at the identity (structure preserved, nothing moved),
`TransformGeo3D` unbaked (vertices untouched), the same convention a disabled 2D `Transform` follows.
**Read-only local/world matrix readouts** (the 2026-09-19 3D UX direction) are in the properties
panel of every node that carries a transform: `Card3D`, `Cube3D`, `Sphere3D`, `Cylinder3D`,
`Scene3D`, `Axis3D`, `TransformGeo3D`, `Camera3D` and `Light3D`. Local is the node's own transform
matrix (translation-only for `Camera3D`/`Light3D`, which aim through `target_x/y/z` rather than a
rotation knob); world multiplies in every `Scene3D`/`Axis3D` ancestor's own transform, nearest
first, the same accumulation `scene_from_node` does at render time. Both are four rows of four
numbers and update whenever a knob (its own or an ancestor's) changes. `Camera3D` is never
parented by a `Scene3D`/`Axis3D` (only geometry, lights and scenes can be), so its world matrix
always equals its local one. A node wired into more than one parent shows the first parent found
rather than every one — a structural preview, not a claim about a canonical single parent.

**Camera film back** (lane L3 step 3, Nuke's model). `Camera3D` no longer stores a field of view; it
stores the lens:

| Knob | Default | Meaning |
|---|---|---|
| `focal` | 22.5390978 mm | Focal length. The default is the value that makes the vertical field of view exactly 45 degrees, what `Camera3D` always rendered with. |
| `haperture` | 24.576 mm | Horizontal film-back aperture (Nuke's default). |
| `vaperture` | 18.672 mm | Vertical film-back aperture (Nuke's default). |

The vertical field of view is `2 * atan(vaperture / (2 * focal))`, the horizontal one the same with
`haperture`; both appear as read-only readouts under the film-back knobs in the properties panel.
`Camera.fov` remains the one stored lens value every renderer, the splat rasteriser and the handles
read, so nothing downstream changed; `Camera.focal` and `Camera.hfov` are derived from it, so the lens
can never disagree with itself. The field of view is rounded to nine decimals, which is what makes the
default film back give exactly 45.0. The render aspect still comes from `Render3D`'s width and height;
`haperture` only feeds the horizontal-FOV readout. **Old documents:** a saved `Camera3D` with only `fov`
loads with Nuke's apertures and the focal length solved from that `fov`, so it renders pixel-identically.
An animated `fov` curve becomes a `focal` curve keyed to the same angles (exact at every key and for
constant curves; a linear curve now interpolates the focal length between keys). An expression on `fov`
cannot be converted and is dropped on load; the stored field of view still applies. **Readers keep the
lens:** Alembic and USD cameras fill all three knobs from the file (Alembic apertures are centimetres in
the file, converted to millimetres); `gltfio.load_camera` turns a glTF camera's `yfov` and `aspectRatio`
into the equivalent film back (vertical aperture 18.672 mm, focal length from `yfov`, horizontal aperture
`18.672 * aspectRatio`, or 24.576 mm when the file gives no aspect). A glTF camera is read through that
function only; `ReadGLTF3D` still loads meshes alone and there is no glTF camera node yet.

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
  type: position, orientation, roll, film back (focal length and both apertures), clipping. Lens distortion, depth of field and shutter are ignored;
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

## glTF import

`ReadGLTF3D` reads glTF 2.0 files, binary `.glb` or JSON `.gltf` with external or data-URI buffers and
images, with an in-house pure Python/NumPy reader (`nodebased/gltfio.py`) and no extra package. This is
the format the image-to-3D tools write (Pixal3D, WorldSculpt, SAM 3D Objects all produce GLB), so it is
the door their results come through.

- **Meshes:** triangle, strip and fan primitives, indexed or not, with per-vertex normals and UVs. World
  transforms (TRS or matrix, through the node hierarchy) are baked in, a mirroring transform reverses the
  winding so front faces stay counter-clockwise, and zero-area triangles are dropped. glTF is right-handed,
  Y-up, in metres, the same as the 3D scene, so nothing is converted except V, which glTF measures from
  the top of the image. Interleaved buffers, quantized attributes (`KHR_mesh_quantization`) and
  `KHR_texture_transform` are handled. `Root node` limits the load to a named node's subtree.
- **Materials:** the base colour factor becomes the surface colour and the base colour texture becomes the
  surface texture (`KHR_materials_pbrSpecularGlossiness` diffuse is accepted too). Textures are decoded
  as sRGB into the working space; the material's alpha is used only when `alphaMode` is `BLEND` or `MASK`,
  otherwise the surface is opaque, as the specification says. Textures larger than 2048 pixels on their
  longest side are box-filtered down. Metallic, roughness, normal, occlusion and emissive maps have no home
  in the renderer and are ignored.
- **Not loaded:** points, lines, cameras, lights, skins, morph targets, animations and vertex colours.
  Draco and meshopt compression and KTX2 textures are refused by name when a file requires them; sparse
  accessors are refused.
- The render cache is keyed on the file plus every external buffer and image a `.gltf` points at, so
  editing either re-renders.

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
  GPU limits the work one submission may carry by a per-adapter-type budget (brute: 4e10 discrete, 1e10
  integrated, 3e8 software, 2e9 unknown; BVH estimate units: 4e8, 4e8, 1e7, 2e8) meant to keep one
  submission near a second. The BVH estimate (16 x log2 triangles per ray) is an average: scenes with long thin overlapping
  triangles can cost far more per ray and are not guarded. A render whose work exceeds one submission
  is split into horizontal bands, each with its own submission and read-back (scissored passes; the results are
  bit-identical to a single submission, tested), up to 64 bands (`GPU_MAX_BANDS`); beyond that it is refused with the
  same 'Shadow rays exceed the GPU budget' message. Before banding, 1920x1080 with one shadowed light was refused
  above roughly 19,000 triangles on a discrete GPU; now, measured on an RTX 3080 Ti, 40,002 triangles took 1.49 s and
  90,002 triangles 3.53 s (BVH path). Cancellation is honoured before each band's submission and after its read-back,
  so latency is bounded by one band; a band that is already submitted still runs to completion. Replaying the draw
  list per band multiplies host preparation, so scenes that fit one submission are never banded;
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
  320x180, 1 sample, shadows on: rasterizer 4.57 s, ray-traced 1.01 s (CPU, this machine). On the GPU
  (`Backend` `gpu` or `auto`) `raytrace` runs as a compute ray tracer for every output except `splats` on triangle-mesh scenes, and for `rgba`
  on scenes that also contain splats when every mesh is opaque (no projected geometry; see "GPU ray tracing" below); everything else
  renders on the CPU with `auto` and is reported as unsupported by `gpu`. The viewport stays on the
  rasterizer.
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
  need the GPU splat path (below) for interactive use. Splats among themselves stay ordered by centre depth (as 3DGS does), so overlapping splats at nearly equal
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
  of up to about 3.5% (mean 3.2%) relative to the unshadowed relit render. Splats also cast shadows onto meshes: a mesh fragment
  lit by a light with `Shadows` on multiplies its light by the same splat transmittance along its shadow ray
  (splats cast whether or not they are relit; splats whose opacity is below 1/255 are ignored). One splat BVH
  is built once per caster set (see below) and shared by both directions. Two cheap speed-ups: only splats that survive view culling
  get shadow rays, and BVH nodes lying entirely before a ray's start offset are skipped (exact, verified
  bit-identical); shadows darker than 0.1% (`SPLAT_SHADOW_CUTOFF`) count as fully dark.
  Measured on this machine (CPU, a 200,000-splat sphere shell, opacity 0.6, one directional shadowed light,
  relit): 640x360 took 78-86 s before the pruning and 36.1 s after (5.3 s without shadows); 1920x1080 took 44.1 s
  after (12.1 s without shadows); the whole shell is inside the view and culling is by frustum and alpha, never by facing, so all 200,000
  splats get a shadow ray at either size. A mesh floor receiving splat shadows costs little (320x180 with 200,000 splats: 4.0 s without shadows,
  5.7 s with). That is still slow for interactive use and the shared budget estimate does not see overlap
  density, so treat splat shadows as a batch/reference feature until the GPU path exists.
  **Shadows are kept between renders.** A splat's shadow depends on the casters, the mesh occluders and where the
  light is; it does not depend on the camera, the light's colour or intensity, ambient or `Relight`. The caster
  BVH and each splat's traced visibility are therefore cached across renders and only splats not yet seen are
  traced, so a camera move or a colour change over a lit capture pays for shadows once. Same 200,000-splat shell
  at 640x360: 36.4 s cold, then 5.3 s for the same frame, a moved camera, a recoloured light and `Relight` 0.5
  (5.3 s is the unshadowed cost); moving the light traces again (34.8 s, the BVH is reused). Warm renders are
  byte-identical to cold ones. Moving the light, the splat node, or any mesh occluder, or changing the node's
  `Opacity` or `Splat scale`, starts over. The cache holds about 206 bytes per caster (0.65 GiB for a 3.4M-splat
  capture), at most two caster sets and 1 GiB, and is per process, never written to disk.
  **Timing (CPU, 1920x1080, 200,000 splats, a sphere shell):** baked 12.3 s, relit without shadows 12.1 s (measured
  while another job was running on the machine, so treat as approximate); relighting itself costs almost nothing, shadows
  are the expensive part. Splats closer to the camera than 0.2 view units are culled even when the camera's near plane is
  smaller (the 3DGS reference does the same; large splats right in front of the eye otherwise smear the foreground).
  For the 3D viewport, `nodebased/splatshade.py` offers `instance_colors` (the exact per-splat linear colours the
  final render uses, relit or baked) and `instance_geometry` (world positions, rotations, sizes, opacity); the CPU
  renderer uses the same functions, so a GPU splat drawer can match it.
  **GPU splat rendering (`Backend` `gpu` or `auto`).** `nodebased/gpusplat.py` renders the
  same splat layer as the CPU renderer on the wgpu backend: CPU depth sort per frame (stable, back to front),
  a compute pass that projects each splat (clamped Jacobian, 0.3 px dilation, near cull at 0.2) and
  evaluates view-dependent SH degrees 1-3 on the GPU, then one instanced draw of alpha-blended quads with the
  CPU's per-pixel plane depth rule and an opaque-mesh depth texture. Baked colours of degree 0 are computed once
  and cached; relit colours are still computed on the CPU each call. Held against the CPU renderer: at most
  2e-3 (mean 2e-4) in the tests, 3.6e-3 worst case measured on a 200,000-splat shell with heavy overdraw
  (the CPU stops accumulating a pixel at 1e-4 transmittance, the GPU does not). Measured on an RTX 3080 Ti:
  200,000 splats at 1920x1080 in 0.09 s warm (CPU reference 12.4 s); the 3.4-million-splat capture in 0.33-0.42 s
  warm at 640x360, 1280x720 and 1920x1080 (CPU: about 51 s at 640x360, refused above), with an 11.6 s first call
  that builds and uploads the static buffers. It needs vertex-stage storage buffers (`check_capability` says
  when an adapter lacks them) and refuses renders whose buffers would exceed the adapter's limits or a 2 GiB
  cap. `Render3D` uses it for `rgba` output in `raster` mode when every mesh is opaque (no transparent or
  projected meshes), for baked splats and for relit splats whose lights have no shadows; the GPU mesh
  render supplies the opaque mesh depth, splats are composited over the mesh image before supersampling. Everything
  else stays on the CPU with no silent differences: `auto` falls back and `gpu` raises a clear error for the data
  passes and the `splats` output, transparent or projected meshes mixed with splats, a shadowed light with relit
  splats, baked splats that cast shadows onto meshes, an adapter without vertex-stage storage buffers, and a
  render that would exceed the GPU memory cap. Measured by two people on an RTX 3080 Ti: 200,000 splats at
  1920x1080 0.09-0.17 s warm.
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
  about 51 s) and 1,726 million at 1280x720 (estimate 105 s); neither is refused.
  **Progress and interactive renders.** `scene3d.render(..., progress=callback)` (and `Evaluator.progress`, which
  the desktop app can set) reports `("prepare", 0.0, {})`, then `("splats", fraction, info)` with `info` holding
  `tile_work`, `estimate_seconds` (reference rate) and, on updates, `eta_seconds`, and finally `("done", 1.0, ...)`.
  The fraction is over all tile work of the whole render (across the memory bands), at most about 200 updates,
  and the callback may raise `Cancelled` to abort. A render with a progress callback is treated as interactive: the
  splat work budget does not refuse it (the shadow and mesh budgets still apply). Renders without a callback
  (batch, agent, tests) are refused above the budget exactly as before. Measured on the real capture with
  progress: 1280x720 took 68 s (reference-rate estimate 105 s) and 1920x1080 took 95 s (estimate 142 s;
  refused before). The ETA is pessimistic early (at 10% it said 89 s and 120 s against 51 s and 74 s remaining,
  because the top of the frame is the heaviest) and within about 10 s from half way.
  **In the desktop app** the window installs one progress router on its evaluator (`renderprogress.py`), so
  every render it starts is interactive: a viewer frame or a single-image export over the budget shows a
  progress bar in the status bar with the percentage and the time left, and is not refused. Until 5% is done
  the text gives the reference-rate total and calls it rough; after that it gives the extrapolated time left.
  Cancelling (a newer edit, a scrub) is noticed between splat tiles. Thumbnails and sequence writes are not
  refused either but show no bar inside a frame. The agent CLI and batch renders have no callback and refuse
  as before. The GPU path reports no progress.
- **Catching shadows without relighting (CPU).** `ReadSplat3D` has a `Catch shadows` slider (0 = off, the
  default, byte-identical to before). With it up, meshes between a `Shadows`-on light and the capture darken the
  capture's OWN colours, so a CG object dropped into a scan grounds itself while the scan keeps its look; `Relight`
  can stay at 0. Each splat's captured colour is multiplied by `1 - Catch * (1 - L_blocked / L_open)`, where `L` is
  the scene's light arriving at the splat centre: the render's ambient plus every light's intensity times its
  luminance, shadowed lights weighted by the mesh transmittance toward them (a half-transparent card lets half
  through). Ambient and lights with `Shadows` off dilute a caught shadow exactly as they would on a mesh. There is
  no Lambert term, because the capture's own shading is already in its colours and its normals are guesses. **Only
  meshes cast caught shadows:** the capture already contains the shadows its own splats threw when it was
  photographed, and no splat BVH is built for this, so it is cheap (one mesh shadow ray per visible splat per
  shadowed light, cached between renders like relit shadows). With `Relight` between 0 and 1 the caught capture is
  what gets blended with the relit result; at `Relight` 1 catching changes nothing, since relit splats already
  carry traced shadows. Limits: a splat object placed in a splat environment does not cast a caught shadow onto
  it (use `Relight` for that); the shadow is evaluated at splat centres, so large soft splats blur its edge;
  the viewport does not show it; alpha and the data passes are untouched. It is CPU-only: with `Backend` `auto`
  a caught-shadow render falls back to the CPU, and `gpu` refuses it (`caught splat shadows are CPU-only`).
- **Cast shadows on/off per capture (CPU).** `ReadSplat3D` has a `Cast shadows` choice (`on` is the default
  and is byte-identical to before; older documents load with `on`). Set it to `off` for an environment capture:
  a scan's sky shell, ceiling or far walls otherwise sit between every light and the scene and black out the
  meshes and relit splats inside it. `off` only removes the instance from the shadow casters. It still renders,
  still receives splat and mesh shadows when `Relight` is up, and still catches mesh shadows with
  `Catch shadows`. Instances left `on` keep casting onto everything, including onto an `off` instance. A
  non-casting instance costs nothing in the splat shadow budget, and when no instance casts and nothing is
  relit no splat BVH is built, so a mesh render inside such a capture equals the same render without the
  capture's shadows byte for byte. The switch is part of the shadow cache key, so flipping it never reuses stale
  visibility. Limits: it is all or nothing per `ReadSplat3D` (no per-region or per-light control), and the GPU
  path still refuses any scene that mixes splats, meshes and a `Shadows`-on light, even when nothing casts, so
  `auto` renders it on the CPU.


- **GPU ray tracing (milestones 1 and 2: visibility and beauty shading for triangle meshes).** `nodebased/gpurt.py` answers the
  ray tracer's visibility queries on the wgpu backend with a compute shader: BVH traversal (the same flat BVH
  as the CPU, bounds rounded outward to float32), closest hit and the K nearest hits per ray (K up to 8) ordered
  by (distance, primitive) with a peeling cursor, peeled all-hit lists, and primary rays built with the CPU's
  exact maths. It works in float32 with an inclusive edge tolerance of 1e-6 (no cracks on shared edges,
  tested) and a 1e-10 determinant threshold. Against the CPU ray tracer on 20,000 random rays through a
  20,000-triangle soup: identical hit/miss decisions, identical primitives, largest distance difference
  7.7e-6 relative, barycentrics within 3e-5; the tests hold 2e-4 relative distance and 1e-3 barycentric
  (tie and shared-edge cases may pick a neighbouring triangle). Measured on an RTX 3080 Ti, 1920x1080 primary rays
  on a sphere-and-ground scene: closest hit 5.4 million rays/s at 1,000 triangles, 3.3 million at 10,000 and
  2.7 million at 100,000 (the CPU: about 92,000, 30,000 and 13,000 rays/s, measured on a subset); eight nearest
  hits 1.0-1.2 million rays/s; full peeled all-hit lists only 0.28-0.30 million rays/s, because each peel is a
  separate submission and read-back. Milestone 2 (`nodebased/gpurt_render.py`, used by `Render3D` `Mode` `raytrace` with `Backend`
  `gpu`/`auto`) keeps hits on the GPU: one compute thread per ray peels surfaces in front-to-back order (the same
  (distance, primitive) cursor and shared-edge duplicate rule as the CPU, with the duplicate thresholds widened to
  1e-5 for float32), shades them with the CPU's model (textures with the CPU's per-triangle mip level and manual
  bilinear sampling from a packed buffer, tint, ambient and Lambert, directional and point lights, Blinn-Phong
  specular, emission, two-sided normals) and traces shadow rays through the same BVH (alpha transmission, bias and
  light-distance rules as the CPU), compositing premultiplied 'over' until an opaque surface, at most 64 surfaces
  (same error text as the CPU). It renders in row bands of up to 524,288 rays per submission with cancellation between bands. Held against
  the CPU ray tracer (interior pixels): 2e-3 in the tests, maxima 1.4e-6 on llvmpipe and below 1e-4 on an
  RTX 3080 Ti in the scenes measured. Triangles crossing the near plane pick the second sub-triangle's mip level exactly as the CPU does
  (tested on a near-clipped textured ground plane). Measured
  on the RTX 3080 Ti at 1920x1080 with a lit, specular, shadowed sphere on a ground plane: 0.35 s at 1,026 triangles,
  0.56 s at 10,002 and 1.91 s at 40,002 (CPU ray tracer, extrapolated from 320x180: about 33 s, 71 s and 127 s).
  Needs seven compute storage buffers (the device now requests up to 8). Milestone 3 adds every output except `splats` (`gpurt_render.render(..., output=...)`), with the CPU's
  semantics: the data passes `depth`, `normals`, `position`, `uv` and `object_id` take the first surface whose
  alpha is above zero (no antialiasing, background ignored, alpha = coverage), and `albedo`, `diffuse`, `specular`
  and `emission` are composited like the beauty pass without the background, so `diffuse + specular + emission`
  equals `rgba` over a transparent background on the GPU too (within 1e-5, tested). Held against the CPU tracer on
  a textured, specular, shadowed test scene with an alpha-0.5 blocker (interior pixels, RTX 3080 Ti): largest
  differences 1.4e-5 for depth, 5.3e-6 for position and normals, 2.6e-6 for `rgba`, 1.3e-6 for `uv`, 0 for `object_id`
  (exact), coverage IoU 1.0 for every output; the tests hold 2e-3 for the composited passes, 2e-4 relative for depth,
  1e-3 for position, `uv` and normals, and exact object ids. At 1920x1080 with 10,000 triangles the beauty pass took 0.66 s and the
  AOVs 0.77-1.2 s.
  Milestone 4 adds splats to `rgba` on the GPU tracer with the same scope as the GPU raster splat path: every
  mesh opaque, splats baked or relit without a shadowed light acting on them. Splats cast shadows onto meshes
  through GPU-side ellipsoid transmittance: the casters are built on the CPU exactly as the CPU renderer builds them
  (opacity scaled by `opacity_scale`, zero for instances with `Cast shadows` off, casters fainter than 1/255
  dropped), uploaded once with their BVH, and each mesh shadow ray multiplies its mesh transmittance by the
  ellipsoid transmittance (same closest-approach formula, same 3-sigma limit, same 1e-3 cutoff, same bias and
  light-distance rules), all in float32. The visible splat layer is composited with the GPU splat renderer over the
  tracer's image using the tracer's depth. Held against the CPU tracer (interior pixels, RTX 3080 Ti): 1.0e-4 for a
  floor shadowed by 20,000 splats, 1.0e-4 for a splat between two meshes and 1.1e-6 for a grazing ray through a thin,
  large splat (the tests hold 2e-3). Measured at 1920x1080 on the RTX: a mesh floor lit through 200,000 splats
  (casters and visible) 2.8-3.1 s with an 18 MiB caster upload; the real 3.4-million-splat capture as casters and
  visible splats over a floor mesh 33-45 s (not compared with the CPU, which would take far longer). Still on the CPU:
  relit splats under a shadowed light, splats with `Shadow catch`, transparent or projected meshes mixed with
  splats, every non-`rgba` output of a splat scene and the `splats` output.
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
usable adapter it falls back to the CPU reference renderer and says so in its header line. On Windows the
viewport renderer has only run in the release workflow's tests, on a software adapter; nobody has run it on a
real Windows GPU.

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

## Relight passes (multichannel bundle)

`Render3D`'s `Output` has a `relight` choice: one CPU raster evaluation that returns the ordinary
beauty image plus, in `Raster.layers`, `albedo`, `normals`, `position`, `diffuse`, `specular`,
`emission`, and per enabled light `diffuse_L0`/`specular_L0`, `diffuse_L1`/`specular_L1`, ... in
`Scene3D`'s wiring order. The per-light channels are unitless response terms (Lambert/Blinn-Phong
times the traced shadow visibility for that light, when it has `Shadows` on) with no light colour,
intensity or ambient folded in, so a downstream node can recombine them with different light values
without re-rasterizing the mesh; summed with the render's own lights and ambient they reproduce the
`diffuse`/`specular` outputs exactly. `raster.Raster` gained an optional `layers` field for this;
every other Raster (including ones derived by `with_pixels`/`aligned`/`fit`) still has `layers=None`,
so nothing else in the pipeline is affected. `relight` is raster-mode only (`Mode` `raytrace` and
`Backend` `gpu` both raise a clear error), always renders at one sample regardless of `Samples`, and
does not support scenes with splats (a clear error names each limit). See "Design: relight passes and
multichannel plumbing" in docs/3D_ROADMAP.md for the full design.

**The `Relight` node** consumes the bundle: an `image` input (must carry `.layers`, or a clear error
names the required `Render3D` `Output`), an optional `camera` input (`Camera3D`, accepted but not yet
used by the shading math), and `light0`..`light7` optional inputs (`Light3D`, paired **by index** with
the bundle's `diffuse_Li`/`specular_Li` channels — wire lights to `Relight` in the same order they were
wired into the original `Scene3D`; a light wired past the bundle's channel count contributes nothing,
without an error). Knobs: `Ambient` (a colour, independent of the render's own ambient), `Diffuse` and
`Specular` (0..1, scale only the per-light response sums, not `Ambient`, so either at 0 turns off that
kind of light without killing the ambient fill), `Mix` (0..1, blend between the recombined result and
the input's own beauty pixels; `0` reproduces the input exactly, byte for byte). Output:
`mix * (albedo * (ambient + diffuse_knob * sum_i diffuse_Li * light_i.color * light_i.intensity)
+ specular_knob * sum_i specular_Li * light_i.color * light_i.intensity) + (1 - mix) * input.pixels`,
alpha unchanged. With the same lights, colours and intensities as the original render, `Diffuse` and
`Specular` at 1 and `Mix` at 1, this reproduces `diffuse + specular` from the bundle exactly (not
`emission`, which `Relight` does not model in this milestone). A disabled `Relight` node passes its
`image` input through unchanged.

## Known limits

What does not exist, and what exists with caveats. Each item is a fact about the code at this commit.

**Rendering**
- One AOV per `Render3D` node; several passes mean several nodes, each re-rendering. There is no multichannel
  (EXR) output.
- Equal-depth ties: where two surfaces of different colours sit at exactly the same distance along a ray, the
  ray-traced mode composites them in ascending primitive order front to back while the rasterizer composites
  stable ties back to front, so `raster` and `raytrace` beauty can differ for exactly coincident geometry.
  Transparency amounts still agree. (Reason: the two modes discover surfaces differently, by ray hits sorted by
  (t, primitive) versus a per-triangle depth sort, and neither has a defined answer for a true tie.)
- A shadow ray that lands exactly on the shared diagonal of a transparent card counts both triangles (alpha 0.5
  card gives visibility 0.25 instead of 0.5). Raster and ray-traced modes build shadow triangles in float32 and
  float64 respectively, so a shadow-edge sample can flip between modes on grid-aligned scenes (seen up to 0.158
  over 320 pixels).
- Shadows are hard: no soft or area lights, no per-object cast/receive flags, none in the viewport.
- No reflections, global illumination, path tracing, motion blur, depth of field, deep output or physically
  based materials (specular is Blinn-Phong).
- The rasterizer refuses scenes over 250,000 triangles; the CPU ray tracer and shadow paths have work budgets.

**GPU (optional `wgpu` extra)**
- `Backend` `auto` falls back to the CPU renderer for projected geometry, ray-traced renders of the `splats` output, splat scenes outside the supported subset (see
  "GPU ray tracing" and "GPU splat rendering" above), splats outside
  the supported GPU subset (see "GPU splat rendering" above), and when the adapter cannot render `rgba32float`
  or has too few storage buffers.
- A submitted GPU job cannot be interrupted, but heavy shadow renders are split into banded submissions (up to 64) so
  cancellation latency is one band and HD renders with a shadowed light and tens of thousands of triangles are no
  longer refused (40k triangles at 1080p: 1.5 s). Renders needing more than 64 bands of the per-adapter budget are
  still refused. Frame-to-band replay costs host time in proportion to the number of bands.
- Frame time for large meshes on the GPU is dominated by per-triangle host preparation, not the shader.
- The GPU ray tracer covers every output except `splats` for triangle meshes, and `rgba` for splat scenes with opaque
  meshes and no shadowed relit splats or shadow catching; no projected geometry. A caster upload larger than the
  adapter's memory cap or storage binding limit is refused (`auto` then renders on the CPU). The GPU splat renderer sorts on the
  CPU each frame, needs vertex-stage storage buffers, and its first call for a large cloud builds and uploads static
  buffers (11.6 s for the 3.4-million-splat capture, then 0.3-0.4 s per frame).

**Gaussian splats**
- Beauty rendering and relighting without shadows run on the GPU for the supported subset; shadows on
  relit splats, splats casting shadows, transparent meshes mixed with splats and every data/AOV pass with splats are
  CPU-only. The viewport draws a layout proxy, not the render.
- CPU time is large for real captures: a 3.4-million-splat capture took about 51 s at 640x360; renders whose
  tile work exceeds 2 billion evaluations (roughly 120 s) are refused when there is no progress callback. With one (the interactive path) nothing is refused: 1280x720
  took 68 s and 1920x1080 (2,339 million evaluations) 95 s on the real capture.
- Shadowed relighting costs tens of seconds for 200,000 splats (36 s at 640x360, 44 s at 1080p here).
- Relighting treats the SH DC term as albedo, so lighting baked into a capture stays; normals come from splat
  shape; splats are ordered by centre depth against each other; the shading AOVs ignore splats.
- Only 3DGS `.ply` files are read (no `.splat`, compressed or animated formats). Verified on generated fixtures,
  one third-party-generated file and one real capture.

**Interchange**
- No FBX. USD: no materials, lights, point instancers or camera export; the stage is opened twice per
  evaluation. Alembic: no curves, points, subdivision or materials; every visible mesh is decoded per
  evaluation; only Blender-written archives were tested.

**Platform**
- CI runs the whole test suite on Linux and Windows for every commit, but installs neither the `gpu` (wgpu) nor
  the `usd` extra, so the GPU and USD tests skip there (81 of 1079 on Linux, 82 on Windows at `5156d72`). The release workflow
  installs both extras before its own full run: its Windows runner has only a software adapter, so the GPU
  tests run there with their frame-time checks skipped, and its Linux runner has no adapter, so they skip.
  GPU timings, real-capture results and everything seen by eye come from one Linux machine with a discrete
  GPU; no real GPU has been used on Windows, and nothing has been run on macOS.
