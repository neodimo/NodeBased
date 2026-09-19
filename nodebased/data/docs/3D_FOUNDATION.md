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
| `WriteGeo3D` | scene | Passes its scene through and exports it to Wavefront OBJ on request. |
| `Light3D` | light | Directional or point light aimed from its position at its target. |
| `Camera3D` | camera | Position, target, roll, vertical field of view, near and far planes. |
| `Scene3D` | scene | Up to eight geometry, light or scene inputs under one transform. |
| `Render3D` | image | Renders `scene` through `camera` at its own width and height. |

Connections are typed. An image cannot be wired where a scene is expected, and a rejected
connection leaves the document untouched. Viewing a geometry, light, camera or scene node
reports that it is not an image rather than failing obscurely; view the `Render3D`.

Every geometry node has translate, rotate (degrees, XYZ order), scale, and an RGBA surface
colour. With a texture connected the colour tints it, so leave it white for the plate as shot.
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
- Not implemented: occlusion-aware projection (a surface hidden from the projector behind other
  geometry still receives the image), so stacked geometry needs `Backfaces` and manual layering.
- The geometry's own UVs and texture are ignored while projected; the colour still tints.

## Exporting geometry

`WriteGeo3D` exports its upstream scene to a Wavefront OBJ: **Export current frame**, or
**Export frame range** with a padded path such as `geo.%04d.obj`. Every transform (including
`Scene3D` hierarchy and keyframed animation) is baked into world-space positions, so an animated
scene becomes one OBJ per frame. Positions, UVs and per-vertex normals are written and read back
by `ReadGeo3D`; colours, textures, lights, cameras and projections are not exported. The file is
written atomically and the text is deterministic.

## Rendering

- **Unlit until lit.** A scene with no lights renders surfaces at their authored colour and
  texture, which is what projecting plates onto cards wants. Add a `Light3D` and surfaces become
  Lambert-shaded by the lights plus `Render3D`'s `ambient`. Surfaces are two-sided. There are no
  shadows and no specular yet.
- **Textures** are perspective-correct, bilinear, and mip-mapped per triangle so distant cards
  do not shimmer. Texture alpha is respected and stays premultiplied.
- **Transparency** composites in depth order. Opaque surfaces use the z buffer; transparent
  ones test it without writing. Interpenetrating transparent surfaces can still sort wrongly.
- **Clipping.** Geometry crossing the near plane is clipped, not dropped, so ground planes can
  run under the camera.
- **Antialiasing** is `samples`×`samples` supersampling (1–4).
- **Output** selects `rgba`, `depth` (view-space distance in RGB, coverage in alpha) or
  `normals` (world-space, coverage in alpha). Data passes are never antialiased and ignore the
  background colour, because averaged depths and normals are values that exist nowhere in the
  scene. These are image outputs, not deep data.
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
3D viewport still uses the CPU renderer. The GPU path has not been run on Windows or on CI hardware.

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

## Not here yet

Occlusion-aware projection, geometry/camera import beyond OBJ (and any export beyond OBJ) (USD, Alembic, FBX), materials, shadows,
specular, motion blur, depth of field, deep output, ray tracing, Gaussian splats, particles,
fluids, a GPU path for the viewport, GPU support for projected geometry, ray tracing on the GPU, and in-viewport transform handles.
