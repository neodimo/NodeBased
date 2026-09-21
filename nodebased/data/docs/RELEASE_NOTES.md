# NodeBased 0.25.0 — splats and ray tracing move to the GPU, and bypass works

## Fixed in this release

- **Bypassing (disabling) a node now works.** In 0.24.0, bypassing a `Merge` put the text `'grade'`
  in the viewer, and bypassing a `Grade` changed nothing on screen. The first was a crash whose whole
  message was the name of a node; the second was the viewer's tiled path running the node it was
  meant to skip. Both evaluation paths now share one rule, and a bypassed node never runs.
  **Behaviour change:** a bypassed `Merge` passes **B**, its background, as Nuke does (A when B is
  not connected). A saved script that contains a bypassed `Merge` will look different.

  ![Bypassing a Merge, then a Grade; the viewer follows each toggle](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/01-bypass.gif)
  [Full-quality clip (MP4, real time)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/01-bypass.mp4)
- **Splat shadows appear on Windows.** On a Direct3D 12 backend, a light aimed straight down an
  axis cast no splat shadow at all in the GPU ray tracer: the surface came back at its plain
  unshadowed colour. Caught by the release packaging gate before any tag, so no released build ever
  had it. GPU test logs now also name the adapter, its backend and whether the colour target is
  float32 or half float, so a parity number can be read with the hardware that produced it.
- **Internal errors say what they are.** If NodeBased itself fails while evaluating, the viewer reads
  "Internal error while evaluating (KeyError: 'grade'). This is a NodeBased bug", and no longer a bare
  word.

## What changed since 0.24.0

- **Gaussian splats render on the GPU.** With `Backend` on `auto` or `gpu`, `Render3D` draws splats
  with wgpu for the beauty image in `raster` mode when every mesh is opaque: baked colours, or relit
  splats whose lights cast no shadows. 200,000 splats at 1920 x 1080 take 530 ms cold and 174 ms warm
  on an RTX 3080 Ti (reviewer's re-measurement); 0.24.0 took about 12 s on the CPU. Output agrees with
  the CPU render within 3e-3 in the tests and within 1e-6 in the reviewer's own comparisons. Anything
  outside that scope renders on the CPU under `auto` and is refused with the reason under `gpu`.

  ![200,000 splats: one camera move on the CPU, then an orbit on the GPU](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/02-splats-cpu-vs-gpu.gif)
  [Full clip, real time (MP4, 75 s): 25 s per camera move on the CPU, 360 to 530 ms on the GPU at 1280 x 720](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/02-splats-cpu-vs-gpu.mp4)
- **A ray-traced mode on the GPU.** `Mode` `raytrace` now runs on the GPU for mesh scenes: textures,
  lights, specular, emission, shadows through transparent surfaces, up to 64 surfaces per ray, and
  every data pass (`depth`, `normals`, `position`, `uv`, `object_id`, `albedo`, `diffuse`, `specular`,
  `emission`). A lit, shadowed 40,002-triangle scene at 1920 x 1080 takes 1.5 s on an RTX 3080 Ti
  (reviewer) against roughly two minutes on the CPU tracer (extrapolated from 320 x 180, so the ratio
  is rough). Passes agree with the CPU tracer within 1.4e-5; object ids are exact.

  ![GPU ray-traced beauty with the albedo, diffuse, specular, normals and depth passes](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/03-gpu-raytrace-passes-sheet.png)
- **Splats cast shadows in the GPU tracer.** Splats shadow meshes on the GPU in `raytrace` mode, and
  the visible splats are composited over the traced image. 200,000 splats casting and visible over a
  floor at 1920 x 1080: about 3 s cold, 2.3 s warm (reviewer). The 3.4-million-splat capture renders
  this way in 33 to 45 s and is not refused (lane's figure; not compared against the CPU).
- **Shadows caught on a capture without relighting it.** `ReadSplat3D` has a `Catch shadows` slider.
  CG meshes between a shadowed light and the capture darken the capture's own colours, so an object
  dropped into a scan grounds itself while the scan keeps its look; `Relight` can stay at 0. At 0 the
  render is byte-identical to before and traces nothing.

  ![Catch shadows swept from 0 to 0.85: the cube grounds itself on the capture](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/04-catch-shadows.gif)
  [Time-lapse (MP4): render waits play at 40x and say so on screen; the pictures hold in real time](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/04-catch-shadows-timelapse.mp4)
  · [the raw take, real time (MP4, 13 min)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/04-catch-shadows.mp4).
  Each slider step re-renders the capture for about 131 s on the CPU at 960 x 540; switching the
  slider off and on again afterwards hits the shadow cache and takes about 125 ms.
  Capture: Nelson Ghost Town by Paolo Tosolini (superspl.at), CC BY 4.0.
- **Cast shadows on or off per capture.** An environment capture's sky shell or walls otherwise block
  every light in the scene. With `Cast shadows` off the capture still renders, still receives and
  catches shadows, costs nothing in the shadow budget and builds no splat BVH.

  ![Cast shadows on: the sky shell blocks the sun. Off: the scene is sun lit](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/06-cast-shadows-pair.png)
- **Splat shadows are cached.** Visibility per splat is kept between renders, keyed by the casters,
  the mesh occluders and the light's geometry. Changing colour, intensity, ambient, `Relight` or the
  camera traces nothing. A 200,000-splat shell at 640 x 360: 36.4 s the first time, 5.3 s after.
  Moving a light still costs a full trace.
- **Large splat frames show progress and are no longer refused in the app.** A slow CPU splat frame
  shows a progress bar in the status bar with a percentage and the time left, a newer edit cancels it
  between tiles, and single-image export gets the same bar. The 3.4M-splat capture at 1920 x 1080,
  refused in 0.24.0, renders in about 95 s on the CPU. The estimate is pessimistic before 5% and
  within about 10 s from half way. The agent CLI and batch renders still refuse over-budget frames.

  ![The progress bar and time left during a 1080p render of the capture, played at 10x](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/05-progress-1080p.gif)
  [Time-lapse at 10x (MP4)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/05-progress-1080p-timelapse.mp4) · [real time (MP4, 3.5 min)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/05-progress-1080p.mp4).
  The clip's scene is the capture plus a cube and a shadowed sun, and takes 208 s; the 95 s above is
  the capture alone. Capture: Nelson Ghost Town by Paolo Tosolini (superspl.at), CC BY 4.0.
- **HD frames with heavy shadows no longer refused on the GPU.** The GPU mesh renderer splits a frame
  into bands when one submission would be too much work, with cancellation between bands. 90,002
  triangles with a shadowed light at 1920 x 1080: 3.5 s. Output is bit-identical to a single
  submission. 0.24.0 refused such frames above about 19,000 triangles.

## Known limits

- **Still CPU-only for splats:** relit splats under a shadowed light, the shadow catcher, transparent
  or projected meshes mixed with splats, and every data pass of a scene that contains splats. `auto`
  falls back; `gpu` names the reason.
- **The GPU ray tracer needs 8 storage buffers per shader stage.** Adapters with fewer fall back to
  the CPU. It does not yet trace projected geometry.
- **The real capture as a shadow caster is heavy:** 33 to 45 s a frame on an RTX 3080 Ti. Turn
  `Cast shadows` off on environment captures.
- **Caught shadows** come from meshes only, are sampled at splat centres (large soft splats blur the
  edge), and do not show in the 3D viewport.
- **Progress** appears for viewer frames and single-image export. Thumbnails and sequence writes are
  no longer refused either, but show no bar inside a frame. GPU renders report no progress.
- **Bypass latency:** in the test window (960 x 540, offscreen) a toggle takes about 300 ms the
  first time and 170 ms after; evaluation is a cache lookup, and most of the rest is the CPU display
  conversion every viewer update pays. In the recorded clip above, on a real display with a small
  graph, each toggle reached a finished frame in 62 to 89 ms. The conversion cost is unchanged in
  this release.
- **Windows:** the GPU tests run in the release build on CI's software adapter (WARP on Direct3D 12,
  half-float colour) and pass there; none of the GPU work in this release has run on a real Windows
  GPU.
- **The 3D viewport still shows splats as opaque discs.** Relit, blended splats in the viewport are
  not in this release.

## Moved to 0.26

Relight passes and a 2D Relight node, better splat normals, shadow offset and blur controls, kept
specular, `WriteSplat3D`, sphere rows and columns, the matrix readout, multichannel EXR, particles,
volumes, a glTF reader, and the 2D-to-3D integration (SHARP image to splat, Pixal3D image to mesh,
WorldSculpt splat to meshes).

## How the media was made

Every still and clip above comes from the real app or a real render at commit `fa996cd` on an RTX
3080 Ti, made by `tools/release_media_stills.py`, `tools/release_media_clips.py` and
`tools/release_media_timelapse.py`. The step timings behind each clip are in
[manifest-clips.json](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/manifest-clips.json) and [manifest-stills.json](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/manifest-stills.json).
Time-lapses speed up render waits only and say so on screen.

# NodeBased 0.24.0 — ray-traced mode, Gaussian splats and a faster 3D viewport

## What changed since 0.23.0

- **A ray-traced render mode.** `Render3D` has a `Mode` knob: `raster` (the default, and what
  every existing document uses) or `raytrace`, a CPU-only alternative that finds visibility with
  one ray per sub-sample through a BVH and shares the rasterizer's shading, so both modes agree
  on every output. CPU shadows now use the same BVH. On the GPU backend, shadow rays traverse a
  BVH on adapters that support it.
- **Gaussian splats.** `ReadSplat3D` loads 3D Gaussian Splatting `.ply` captures (SH degrees 0 to
  3, `as_authored` or `colmap` orientation) with Nuke-style transform knobs, and `Render3D`
  renders them on the CPU with an EWA splat rasterizer. Splats and meshes are ordered per pixel,
  in both render modes, against opaque and transparent geometry.
- **Splats in the passes.** `depth`, `normals`, `position`, `uv` and `object_id` include splats
  (first hit where accumulated splat opacity reaches 0.5), and a new `splats` output gives the
  splats' premultiplied contribution on its own.
- **Splat relighting.** `ReadSplat3D` has a `Relight` slider: 0 keeps the captured colours
  exactly, 1 re-lights each splat from the scene's `Light3D` nodes and the render's ambient,
  using the splat's flattest axis as its normal. With `Shadows` on a light, meshes and other
  splats shadow relit splats, and splats cast shadows onto meshes.
- **Real captures render.** The reference 3DGS Jacobian clamp removes the full-frame veil that
  large off-screen splats caused on a 3.4-million-splat capture, splats closer than 0.2 view
  units are culled as the 3DGS reference does, and memory for mixed mesh and splat frames stays
  near 150 MiB at 1080p through banded buffers (it was about 0.9 GiB).
- **A splat work budget that matches the real cost.** Renders are refused up front by counted
  tile work (splats per 16 x 16 tile), with an estimate in seconds and remedies in the message,
  instead of by bounding-box pairs. The 3.4M-splat capture is accepted at 640 x 360 and at
  1280 x 720.
- **An interactive GPU viewport.** The 3D viewport has its own wgpu renderer:
  meshes upload once, each object is one draw call, and orbiting only rewrites a camera matrix
  (about 1 ms per frame for a 64-segment sphere at 960 x 600 on an RTX 3080 Ti, against about
  400 ms on the CPU reference). Without a usable adapter it falls back to the CPU renderer and
  says so.
- **Splats in the viewport, as a layout proxy.** Each splat is drawn as a small opaque disc in
  its base colour, up to 1,000,000 per cloud, following `Relight`, `Opacity` and `Scale`. The
  3.4M-splat capture orbits at about 3 to 4 ms per frame on an RTX 3080 Ti, and F frames the
  bulk of a cloud rather than its outliers.
- **3D panels follow Nuke.** XYZ rows with a keyframe diamond per axis, typed
  fields for sizes and open-ended magnitudes, sliders only for bounded values. 3D nodes have
  rounded ends in the graph; `Scene3D`, `Light3D` and cameras are circles.
- **One transform block everywhere.** `Card3D`, `Cube3D`,
  `Sphere3D`, `ReadGeo3D` and `Scene3D` gain uniform scale, rotation order and pivot, in the same
  order as `ReadSplat3D`. Older documents load unchanged and render the same.
- **Navigation.** Alt + left drag pans the node graph and the viewer, Alt + scroll zooms, and
  touchpad zoom is proportional.

See [3D in NodeBased](3D_FOUNDATION.md) for conventions and details.

## Known limits

- **Splats render on the CPU only.** The GPU backend does not render splats (`auto` uses the
  CPU). Expect seconds to minutes per frame: 200,000 splats at 1080p take about 12 s, the
  3.4M-splat capture at 640 x 360 about 50 s. Large renders are refused up front rather than
  left to run, and there is no progress bar inside a frame. With the default budget the
  3.4M-splat capture is refused at 1920 x 1080 (estimated 142 s); 1280 x 720 is the largest
  standard size it renders at (estimated 105 s).
- **Splat shadows are slow, CPU-only, and a batch/reference feature.** One shadowed light on a
  200,000-splat shell (relit) took 36.1 s at 640 x 360 (down from 78-86 s before an exact BVH
  speed-up; 5.3 s without shadows) and 44.1 s at 1920 x 1080 (12.1 s without shadows). A mesh
  floor receiving shadows from the same 200,000 splats at 320 x 180 took 5.7 s against 4.0 s without.
  Shadow cost depends on how many overlapping splats each ray crosses, which the up-front render
  budget does not see, so a dense capture can take far longer than these figures. Surfaces made of
  overlapping splats keep a residual self-shadowing of about 3%; shadows darker than 0.1% count as
  fully dark; each splat's shadow is a closest-approach approximation of a Gaussian's density, not a
  volume integral.
- **Relighting is an approximation.** The capture's own lighting stays baked into the colours,
  so an unevenly lit capture is double-lit; normals are guessed from splat shape; specular and
  view-dependent colour are not re-lit; `albedo`, `diffuse`, `specular` and `emission` ignore
  splats.
- **The viewport's splats are not a preview of the render.** Opaque
  discs, no blending, no view-dependent colour, no splat shadows; clouds above 1,000,000 splats
  are strided. Without a GPU the viewport marks splat centres as points.
- **The ray-traced mode is CPU only**, hard shadows only, and no reflections, refraction or
  global illumination. At most 16 mesh surfaces may lie in front of a ray's terminating surface
  when transparent meshes and splats mix.
- The viewport shows no shadows, sorts transparency per object and shades at most 16 lights.
- Not in this release: a GPU splat path, GPU ray tracing, particles, volumes. These are staged
  in [3D_ROADMAP.md](3D_ROADMAP.md).
- Windows behaviour is proven by CI only; nobody has driven this release by hand on Windows. The
  GPU viewport and GPU renderer have run there only in the release workflow's tests, on a software
  adapter, never on a real Windows GPU.

# NodeBased 0.23.0 — GPU rendering, USD, Alembic, shadows, materials and AOVs

## What changed since 0.22.0

- **An optional GPU backend.** `Render3D` has a `Backend` knob: `cpu` (the default, and what
  every existing document uses), `gpu` (wgpu) or `auto`. `auto` falls back to the CPU renderer
  when no capable adapter is present or a GPU render fails; `gpu` reports the reason instead.
  Adapters that cannot render to `rgba32float` (downlevel GL/GLES class) count as unavailable.
  The CPU rasterizer remains the reference and the GPU path is tested against it.
- **USD import and export.** `ReadUSD3D` loads `.usd/.usda/.usdc/.usdz` stages as scenes (meshes
  with UVs and normals, world transforms, up axis and `metersPerUnit` honoured), `ReadUSDCamera3D`
  loads a camera, and `WriteGeo3D` writes USD when the path has a USD extension. Release builds
  bundle `usd-core`; from source, install `nodebased[usd]`.
- **Alembic import.** `ReadAlembic3D` and `ReadAlembicCamera3D` read Ogawa `.abc` files with an
  in-house, read-only reader: polygon meshes, transforms and cameras, with time interpolation.
  No extra package is needed.
- **Camera projection and OBJ export.** `Project3D` projects an image through a camera onto any
  geometry, perspective-correct, with `Outside`, `Backfaces` and an approximate depth-map
  `Occlusion` option. `WriteGeo3D` exports OBJ.
- **Shadows.** `Light3D` has a `Shadows` switch: hard shadows on the CPU renderer and on the GPU
  backend, which is parity-tested against the CPU result.
- **Materials.** Blinn-Phong specular and emission on both backends.
- **AOVs.** `Render3D`'s `Output` adds `albedo`, `diffuse`, `specular`, `emission`, `position`,
  `uv` and `object_id` alongside `rgba`, `depth` and `normals`, on CPU and GPU.
- **Rasterizer fix.** A top-left fill rule covers shared triangle edges exactly once, and the
  GPU backend matches it.

See [3D in NodeBased](3D_FOUNDATION.md) for conventions and details.

## Known limits

- **Alembic is partial.** Verified against Blender-exported files only. Polygon meshes,
  transforms and cameras; no curves, points, subdivision surfaces or materials. Every mesh is
  decoded again on each evaluation, so heavy archives are slow.
- **USD is partial.** Meshes and cameras only: no USD materials, lights, point instancers,
  curves or volumes. The stage is opened twice per evaluation.
- **GPU shadows are brute force** (every shadow ray tests every triangle) under a per-adapter
  work budget; a render above the budget is refused up front. A submitted GPU job cannot be
  cancelled. The 3D viewport does not show shadows.
- **One AOV pass per `Render3D` node.** Several passes mean several nodes; there is no
  multichannel EXR output.
- Materials stop at Blinn-Phong: no metalness, GGX, reflections, transmission or texture maps.
- Not in this release: ray tracing, Gaussian splats, particles, fluids. These are staged in
  [3D_ROADMAP.md](3D_ROADMAP.md).
- Windows behaviour is proven by CI only; nobody has driven this release by hand on Windows.

# NodeBased 0.22.0 — a 3D scene graph and viewport

## What changed since 0.21.1

- **3D nodes that render into the comp.** `Card3D`, `Cube3D`, `Sphere3D`, `ReadGeo3D` (Wavefront
  OBJ), `Light3D`, `Camera3D`, `Scene3D` and `Render3D`. `Render3D` produces the same float32,
  scene-linear, premultiplied RGBA as every other node, so Grade, Merge and Write follow it.
  Ports are typed: a mismatched connection is refused and the document is left untouched.
- **Textured cards for 2.5D.** Connect any image to a geometry node's `image` input. Textures
  are perspective-correct, mip-mapped, respect alpha and stay premultiplied. Upstream 2D edits
  re-render the 3D scene.
- **Lights.** Scenes are unlit until they contain a `Light3D` (directional or point); then
  surfaces are Lambert-shaded with an `ambient` term on `Render3D`.
- **Scene hierarchy.** `Scene3D` has its own transform and accepts geometry, lights and other
  scenes; nested scenes inherit their parent's transform.
- **Render controls.** Supersampled antialiasing (1–4), transparent background by default, and
  `rgba`, `depth` or `normals` output. Near-plane clipping, depth-sorted transparency,
  cancellation, proxy tiers and result caching all apply. Every numeric 3D parameter animates
  and takes expressions.
- **A 3D viewport.** Toolbar → 3D viewport. Orbit, pan, dolly, **F** to frame the scene, **C**
  to look through the authored camera. It shows the scene your `Render3D` will see — textures,
  animation and disabled nodes included — with a depth-tested grid, axes, camera frustum and
  light markers. Navigation never edits the authored camera. The dock is saved with the workspace.
- **Type-aware node creation.** A new node wires itself to the selection only where the types
  fit: select a plate and create a `Card3D` to texture it; select a card and create a `Scene3D`
  to collect it.

See [3D in NodeBased](3D_FOUNDATION.md) for conventions and details.

## Known limits

- The renderer is a CPU reference rasterizer: correct, tested against analytic answers, and
  not fast. It refuses scenes above 250,000 triangles. There is no GPU scene backend yet.
- No shadows, specular, materials, motion blur, depth of field or deep output. `depth` and
  `normals` are image passes.
- Not in this release: camera projection, USD/Alembic/FBX, ray tracing, Gaussian splats,
  particles, fluids, and in-viewport transform handles. These are staged in
  [3D_ROADMAP.md](3D_ROADMAP.md); Nuke 17.1 parity is a long-range target, not a claim.
- Interpenetrating transparent surfaces can sort incorrectly.

# NodeBased 0.21.1 — a properties panel that fits, and a workspace that remembers

## What changed since 0.21.0

- **The properties panel reflows to its dock.** Panels no longer paint wider than the dock and
  clip their right edge. Form labels wrap above their field when a row runs out of room, notes
  word-wrap, numeric fields share the width that is left, and long button and check box labels
  shorten (full text in the tooltip). Stacked panels follow the dock as it is widened or
  narrowed, and sideways scrolling is gone.
- **The window reopens the way you left it.** Window size and placement, dock layout (including
  the Agent dock) and the viewer/node-graph divider are saved on close and restored on launch.
  An unusable saved layout falls back to the default instead of half-applying.
- **Workspace → Default workspace.** A new menu puts the window, panels and dividers back where a
  fresh install has them, in place.
- **A bolder app icon.** The icon now fills its whole tile, with no border, so it reads at
  taskbar and dock sizes.

## Known limits

- On Wayland, applications cannot set their own window position, so only the window size and
  layout are restored there; the compositor chooses where the window appears.

# NodeBased 0.21.0 — direct GPU viewport rendering and packaged GPU verification

## What changed since 0.20.0

- **Direct GPU viewport rendering.** Scene-linear float32 frames stay float32 through
  computation and are transformed directly into the active viewport framebuffer.
  Channel isolation, exposure, alpha, checkerboard compositing, OCIO views, and
  top-down image orientation are covered by offscreen-GL tests.
- **Packaged Linux GPU smoke coverage.** The release workflow launches the real AppImage
  under Xvfb/GLX and requires the packaged application to report the GPU display path.
- **Cache precision remains explicit.** Full-resolution scene-linear cache entries remain
  isolated from display output; half-float storage is only permitted where the cache
  correctness gates prove its range and round-trip behavior.

## Fixes

- Release GPU resources while their owning OpenGL context is current, avoiding deferred
  texture destruction after the context has gone away.

# NodeBased 0.20.0 — an agent in the app, Nuke knobs and a pixel readout

## What changed since 0.19.0

- **An Agent panel with a real terminal.** View → Agent (or the Agent menu) opens a docked
  terminal that starts Claude Code or Codex in the project directory. The agent talks to the
  running application, so it acts on the comp you are looking at rather than a description of
  it.
- **The application explains itself to the agent.** A knowledge module ships its own docs
  inside the package: architecture, the time model, playback, colour management, roto and
  tracking, the agent protocol and the release history. The agent reads topics on demand
  instead of guessing.
- **`nodebased-mcp`.** A stdio MCP server exposes the live document over the local bridge:
  describe, inspect, edit, undo, redo, view, errors, reference context, knowledge and issue
  filing. Save, load and render stay behind a switch that is off by default.
- **The agent can file GitHub issues.** Reports go through the `gh` CLI, fall back to the REST
  API with a token, and are recorded in a local log. The switch is in the Agent panel.
- **Nuke's knob types.** Parameters render as float sliders with a soft-range ruler,
  XY pairs, colour knobs with a swatch and picker, checkboxes, dropdowns and file fields,
  each with an animation button. Values remain scene-linear floats around 0–1, never 0–255.
- **Expressions on any knob.** Press `=` with the cursor in a knob, or right-click and choose
  Enter expression. Expression-driven knobs take their own colour, and Clear expression puts
  the plain value back.
- **Stacked properties panels.** Double-click pins a node's properties below the current one,
  Nuke-style, with a per-panel close and a clear-all. The stack holds five panels by default;
  Preferences and the dock's spin box change the cap.
- **A User tab on every node,** alongside the node's own tab and the Node tab from 0.19.0,
  with Revert and Close.
- **Pixel values under the pointer.** The lower right of the viewer reads out full-resolution
  coordinates with Nuke's bottom-left origin, raw float RGBA and a colour swatch. It follows
  the pointer while you draw roto shapes too.
- **Hotkeys match Nuke.** Node creation keys were checked against Nuke's defaults and
  reassigned, and select-all, copy, cut, paste and duplicate were added. The viewer gains
  Ctrl+= / Ctrl+- zoom, Ctrl+1 for 1:1, and J/K/L shuttle.
- **Help → Keyboard shortcuts** lists every binding from the single table the menus use, so
  the dialog cannot drift from the application.

## Fixes

- The pixel readout was dead in the running application even though its tests passed: a second
  `mouseMoveEvent` on the viewer shadowed the readout's handler. The handlers are merged and
  the tests now hover through the real event path.
- Three tests made platform assumptions that only failed on Windows: a stored boolean read as
  raw Ini text, a log file opened twice, and a `/tmp` path used as a directory.

## Known limits

- The 0.17.0 playback limits are unchanged: native 24 fps at 4K ACES 2.0 is still not reached,
  and performance evidence remains from one Linux/X11 machine.
- The Agent panel has been driven by hand on Linux/X11 only: Claude Code starts and reaches its
  trust prompt. Signing in and running a full agent session through the panel, on any platform,
  is still unproven.
- The MCP server has been exercised against a running application with the window hidden, not
  through a full agent session.

# NodeBased 0.19.0 — Node tab, postage stamps and a quieter viewer

## What changed since 0.18.0

- **Merge reads like Nuke's.** `B` is the top-centre trunk input, `A` joins from the left, and
  Merge gains an optional `mask` on the right. Where the mask is zero the result is `B`. Older
  comps open with the mask unwired and render identically.
- **New nodes land underneath the selection.** Adding a node while one is selected wires it in
  and places it directly below, stacking down the same column on repeated adds.
- **Scrubbing reuses what playback cached at full resolution.** The display cache is keyed by
  frame and region and is checked before tiles are composed, and a cached whole frame also
  serves a zoomed-in crop.
- **Postage stamps.** Read, Constant and Checker nodes show a thumbnail of their output by
  default; any other node can turn one on. Stamps render only while the viewer is idle and
  never delay a frame. Settings → Interface can switch them all off.
- **A Node tab in the properties panel.** Each node has a second tab with a label (drawn on
  the node in place of its type), an Enabled switch and the postage-stamp switch.
- **The viewer only re-renders when its picture can change.** Moving, renaming or labelling a
  node, or editing a branch the viewer is not showing, no longer triggers a render.
- **Dots win the click.** Clicking a Dot selects it even zoomed out, and Ctrl-clicking a Dot no
  longer starts a noodle insert underneath it.
- **Cleaner cards and an accent colour.** The thick coloured bar on the left of each node is
  gone; the family colour stays in the outline. Settings → Interface adds an accent colour
  (presets or custom) on top of any theme, stored per machine.
- **Tidier example comp.** The demo's `A` branch sits to the left of the Merge, so noodles no
  longer cross.

## Compatibility

- Projects are now document version 12. Older projects upgrade on open; 0.18.0 and earlier
  cannot open a project saved by 0.19.0.

## Known limits

- The 0.17.0 playback limits are unchanged: native 24 fps at 4K ACES 2.0 is still not reached,
  and performance evidence remains from one Linux/X11 machine.
- The Node tab does not yet offer Nuke's tile colour, font or hide-input controls.

# NodeBased 0.18.0 — Nuke-familiar graph, properties panel and export

## What changed since 0.17.0

- **Node titles and disabled state read at a glance.** Titles are centred and larger, and a
  disabled or bypassed node dims and carries a large X across it instead of announcing itself
  only in small subtitle text. A disabled Merge still shows its `B` input, so the wiring stays
  visible while the node is off.
- **Input ports sit where a compositor expects them.** `A` is on the left edge and `mask` on
  the right, with ordinary inputs still centred along the top. Dot control points draw above
  noodles so a Dot still looks like a dot where connections cross it.
- **Properties-panel context menus stay open.** The knob and curve menus were parented to the
  panel, and the rebuild triggered by opening one destroyed the menu mid-show — the menu
  appeared to flash and vanish. The window owns them now.
- **The viewer no longer resizes the layout.** Comp format and playback status text fed back
  into widget size hints, so a 4K format or a long status line could redistribute the splitter
  during playback. Panel sizes now change only when the artist drags them. The status label
  elides for painting and keeps the full string for tooltips and the agent bridge.
- **Viewer nodes behave like Nuke's.** Viewing a node rewires every Viewer to it as one undo
  step, with cycle and self-view guards, and the viewer connection draws faint, dashed and
  arrowless so it does not read as a real input of the node it inspects.
- **Write nodes do the real exporting.** A Write renders its own upstream tree at full
  resolution, validates its parameters against the chosen file type, refuses an unpadded path
  for a frame range rather than overwriting one file every frame, passes its input through
  unchanged, and can be bypassed like any other node.
- **Read opens a sequence-aware browser.** Numbered frames batch into a single padded entry,
  grouping is on by default and can be switched off to list every frame, padding width and
  extension separate two sequences in the same folder, a lone numbered file stays a still, and
  a hole in a range is reported rather than quietly hidden.
- **Chrome and theming.** Check for updates is right-justified at the trailing edge of the
  toolbar, and the settings dialog offers theme colours, which persist as a preference and are
  never written into the document. An unknown stored theme falls back instead of failing.

## Known limits

- The 0.17.0 playback limits are unchanged: native 24 fps at 4K ACES 2.0 is still not reached,
  and performance evidence remains from one Linux/X11 machine.
- Theme choices cover the application palette; per-node colour overrides are not exposed.
- The sequence browser reads directory listings rather than image headers, so it groups by file
  naming and does not validate that batched frames share a format.

# NodeBased 0.17.0 — faster 4K playback ingest and parallel decode-ahead

## What changed since 0.16.0

- **Proxy playback is substantially faster on the measured 4K sequence.** The standard
  auto-proxy ACES 2.0 path improved from 2.10 fps to 10.21–11.42 fps on the development
  machine, a measured **4.9–5.4x** gain. The reproducible harness, stage timings, and exact
  hardware scope are recorded in `docs/BENCHMARKS-v0.17-playback.md`.
- **Decode and ingest do less memory work.** ACEScg premultiply handling now broadcasts a
  one-channel alpha factor instead of constructing full RGBA-sized factor arrays; ordinary
  RGB channel layouts take a slice fast path; and proxy decimation uses strided accumulation
  rather than a slow multi-axis reshape reduction. The tier-2 and tier-4 decimation stages
  measured about 8x faster in isolation while retaining the existing pixel tolerances.
- **Bounded parallel decode-ahead.** During proxy playback, a dedicated worker pool warms
  decoded Read frames ahead of the single-owner preview evaluator. The cache is memory-bounded,
  suppresses duplicate requests, drops stale work after seeks or edits, and never runs for
  full-resolution tier-1 playback where measurements showed it was counterproductive.
- **Source replacement and shutdown are safe.** Decode-cache identity includes the resolved
  file path, size, and modification time, including the reference-frame identity used by
  `missing=black`, so replacing media at the same path cannot return stale pixels. Window close
  cancels queued decode work and remains prompt even if a native decode is already in flight.
- **Playback QA is repeatable.** `tools/playback_qa.py` exercises a real desktop window and
  reports throughput and ordering; its ordering check accepts the expected end-to-start loop
  while still rejecting an ordinary backward jump.

## Known limits

- Native 24 fps at 4K ACES 2.0 is not reached. The best measured result is 11.42 fps at the
  normal tier-2 auto-proxy resolution; full-resolution playback improved only modestly, from
  1.18 fps to 1.38 fps.
- Performance evidence is from one Linux/X11 machine and one 100-frame 4K EXR sequence.
  Windows is covered by the conformance suite, not by equivalent real-hardware playback
  measurements. Decode-worker tuning, display-cache hashing, Qt scene rebuilds, and QImage
  conversion remain candidates for the next performance pass.
- Negative-origin/overscan behavior through the pre-existing tiled proxy path remains outside
  this release's playback-performance scope.

# NodeBased 0.16.0 — fast display transform, agent reference loop, roto drawing and pixel tracking

## What changed since 0.15.0

- **The ACES 2.0 view transform is no longer the playback bottleneck.** OCIO's CPU processor
  releases the GIL, so the display transform now runs across a thread pool. The output is
  bit-identical to the single-threaded path. Measured medians: 4K ACES 2.0 dropped from
  2170 ms to 176 ms per frame, and 4K sRGB from 129 ms to 20 ms. When an OpenGL 4 context is
  available, ACES 2.0 runs as an OCIO-generated GLSL shader instead. That path measured 47 ms
  at 4K and 5 ms at HD on the development machine's integrated GPU, within 1 8-bit code value
  of the CPU reference. sRGB stays on the CPU path, because GPU upload and readback made it
  slower. `NODEBASED_DISPLAY_GPU=0` forces the CPU path, and the viewer status shows which
  backend is active. Method and full numbers: `docs/BENCHMARKS-v0.16-display.md`.
- **Reference-image agent loop.** Nodes can be tagged `Reference for agent`, and the GUI endpoint
  captures the viewed and tagged images (schema v10, `reference_context`). The new
  `nodebased-agent-loop` client sends those captures and your prompt to a vision model
  (Anthropic by default, key from `ANTHROPIC_API_KEY`). It validates the proposed edits against
  the node schema and applies each iteration as one revision-guarded batch, which is one undo
  step. `--dry-run` applies nothing, and by default every batch asks for confirmation.
  NodeBased itself still makes no model or network calls.
- **Roto shapes can be drawn and edited in the viewer.** Draw closed shapes and drag existing
  points directly on the image. Edits are validated, undoable, and preserve animated points.
- **Tracker analyses real pixels.** Pick a point, and the Tracker follows it forward with
  normalised cross-correlation and sub-pixel refinement. Weak matches are reported as possible
  occlusion instead of writing bad motion. Results commit as one undoable edit, and cancelling
  leaves the document unchanged.

## Known limits

- The GPU display path was benchmarked on one Linux machine only. CI runners have no GPU and
  exercise the CPU fallback. Other GPUs, drivers, and Windows GL contexts are unverified. A
  display-less eGPU used through PRIME offload measured slower than the integrated GPU there.
- The agent loop's live model call was not exercised in automated tests.
- Tracking is forward-only point tracking. Roto has no transform handles, track linking, or
  ROI-limited evaluation yet.

# NodeBased 0.15.0 — expressions, keyframes, timeline, and roto foundation

## What changed since 0.14.0

- **Expressions are available in numeric knobs.** Set, edit, and clear formulas directly
  from the inspector. Expressions support `frame` and references to other knobs, resolve
  deterministically through the renderer, reject dependency cycles with their full path,
  and participate in atomic undo/redo.
- **Animated controls are visible and usable.** Numeric knobs expose keyframe controls;
  formula-driven controls show their resolved value and reject conflicting base-value or
  keyframe edits.
- **Timeline readability and cache feedback.** Adaptive tick marks/frame labels, blue
  keyframe underlines, and orange cached-frame underlines make temporal state visible.
- **Roto, Tracker, and ChannelShuffle groundwork.** Schema v8 added deterministic CPU
  roto mattes and channel routing. Interactive shape drawing and image-based tracker
  solving remain future work.
- **Expression-error feedback is stable.** Rejected formula edits stay visible in the
  inspector even if an asynchronous preview status update occurs.

# NodeBased 0.14.0 — playback correctness fixes and live agent diagnostics

## What changed since 0.13.0

- **Fixed: the display cache never actually hit during playback.** Its key
  hashed the whole document, including the live playhead position.
  Read-ahead builds every prefetch request from one document snapshot taken
  while the playhead is still on the current frame, so a frame warmed
  several frames ahead got hashed with the *old* playhead value — a
  guaranteed miss once that frame actually became current. Every read-ahead
  result was being thrown away. Fixed by normalizing the playhead to the
  frame actually being evaluated before hashing.
- **Measured: the ACES 2.0 view transform costs roughly 10x what sRGB
  costs on identical pixels**, and that cost scales with resolution (HD
  ~550ms vs ~56ms; 4K ~2.2s vs ~170ms). This is inherent to OCIO's built-in
  ACES 2.0 implementation, not a caching bug, and is the real reason native
  4K/ACES playback is still not real-time — moving the transform off the
  CPU is real, scoped, unstarted future work.
- **Nuke-style sequential playback fallback.** When rendering can't sustain
  real time, the transport used to keep following the wall clock anyway,
  so it could show frames wildly out of order — including jumping
  backward — once a slow render finally finished. Playback now degrades to
  strictly in-order, slower-than-real-time frames instead, matching Nuke's
  own fallback behavior. Verified against a real 4K sequence: before, 9
  frames displayed in 20s in the order 62, 14, 64, 17, 68, 18, 69, 22, 72;
  after, 9 frames in 25s as 2, 3, 4, 5, 6, 7, 8, 9, 10.
- **Live render-error visibility for an attached agent.** A new `errors`
  operation on the local agent bridge surfaces every evaluation failure —
  including ones on frames that never reached the screen — instead of
  requiring a human to read the status bar. See `docs/AGENT_PROTOCOL.md`.
- **Fixed: GitHub Release pages showed the entire changelog, not just the
  version being published.** The release workflow now extracts only the
  current version's section from the release notes.

# NodeBased 0.13.0 — branch-insert node placement and viewer format guides

## What changed since 0.12.0

- **Adding a node with a selection wires it into that node's branch.**
  Tab-search or a node hotkey used to always drop the new node at the last
  click position, unconnected. If a node is selected and the new node type
  has an input, it now lands near the selection, connects from its output,
  and takes over any existing downstream connection the selected node had —
  the same splice used by the existing Ctrl-drag-a-noodle-midpoint gesture —
  so it's inserted inline in the branch rather than forking a dead end.
  Generators with no input (Read/Constant/Checker) are unaffected and keep
  click-position placement.
- **Viewer format guides, Nuke-style.** A dotted outline now traces the
  display window at any zoom level, and a resolution readout ("3840 x
  2160") sits just outside its bottom-right corner. Implemented as a
  `drawForeground` paint rather than scene items, specifically so it can
  never perturb `itemsBoundingRect()` — the measurement the 0.12.0
  stuck-corner regression test depends on staying exact.

# NodeBased 0.12.0 — playback caching and proxy-resolution playback

## What changed since 0.11.0

- **EXR writes default to 16-bit half at ZIPS (1-scanline) compression.**
  Applies to export and the agent's `render` op alike; both bit depth and
  compression remain explicitly overridable. Half/ZIPS is now verified to
  match the tile path exactly against the reference evaluator on real 4K
  source (max abs delta 0.0, zero fallbacks).
- **Display-ready frame cache.** Looping playback, scrubbing back onto a
  frame already shown, or returning a paused parameter to a value it held
  before is now a cache hit instead of paying the ACES 2.0 view transform
  again — measured 3.14s cold vs 0.08s on an identical repeat at 4K. Keyed
  on the whole document plus frame/tier/view/exposure/channel/background, so
  a graph edit is correctly a miss and undoing it is correctly a hit again.
- **Proxy-resolution playback.** Sources above HD now play back at an
  automatically chosen lower tier — the same technique every NLE and
  compositor uses for this — and restore full quality the instant playback
  stops. Never overrides a tier chosen manually. Cuts a cold 4K frame from
  ~3.1s to ~1.1s. Read-ahead now also warms the transformed display cache,
  not just the raw composite, ahead of the playhead.
- **Fixed: viewer stuck zoomed into a stale corner.** Reconnecting the
  viewer to a differently sized source used to clamp its first frame to
  whatever fraction of the old image's viewport happened to overlap the new
  canvas, so the image could get stuck zoomed into a small corner with no
  way to recenter it. The first frame of a resized source now always
  requests the complete canvas.
- **Fixed: a schema-upgrade step could silently skip its own successor.**
  One upgrade step stamped the generic `SCHEMA_VERSION` constant instead of
  the literal version it actually produced — harmless while that step
  happened to be the last one, but the next schema version added would have
  landed a document tagged current while missing its newest section. Fixed
  and guarded by a test that checks no upgrade step makes this mistake.

## Known scope

Native 4K playback at full (untier'd) quality is not real-time — the ACES
2.0 display transform's CPU cost is the remaining bottleneck for that
specific case. Proxy-resolution playback above closes most of the practical
gap; moving the transform itself off the CPU is unscoped follow-up work.

# NodeBased 0.11.0 — ACEScg color pipeline and project settings

- **Linear ACEScg float32 working space.** Read inputs convert from their tagged
  or selected source space into ACEScg before graph evaluation. Untagged EXRs
  fall back to Linear Rec.709 instead of being assumed to already match the
  working space. EXR output is tagged ACEScg; PNG output converts from ACEScg
  through OCIO rather than applying only an sRGB transfer curve.
- **Correct premultiplied display.** Viewer transforms now unpremultiply,
  transform straight color, then re-associate alpha. Semi-transparent pixels no
  longer disagree between the viewer and PNG output.
- **ACES 2.0 Rec.709 default view.** New projects use the ACES 2.0 SDR 100-nit
  Rec.709 view on the sRGB display.
- **Project settings, schema v7.** `Edit → Project settings…` (`S`) exposes the
  bundled OCIO config, ACEScg working space, display, saved default view, and
  black/checker viewer background. Settings edits are validated, atomic,
  undoable, saved in the project, and available through the Dispatcher protocol.
- **Deterministic slow-playback coverage.** The transport regression now delays
  the tile executor actually used by the viewer and renders a tiny graph, so
  hosted-runner raster speed cannot decide whether the test passes.

# NodeBased 0.10.0 — animation curves

## What changed since 0.9.1

- **Animation curves (document schema v6).** Any numeric node parameter can hold
  a curve of keys with `constant` or `linear` interpolation. Curves are an
  evaluation-time overlay: a node's stored parameters are never mutated, so a
  graph with no curves renders byte-identically to a v5 document.
- **Endpoint hold.** Before the first key and after the last, a curve holds that
  key's value, matching Nuke's extrapolation.
- **Atomic curve editing.** `set_key`, `delete_key` and `clear_curve` are
  Dispatcher operations with undo/redo, and deleting a node drops its curves in
  the same transaction, so a document can never reference a curve on a node that
  is gone. The agent CLI exposes the same operations.
- **Animation reaches the tile viewport.** Curves resolve at the tile executor's
  API boundary, so animated parameters render identically through the tile path
  and the reference evaluator, at every proxy tier.
- **Transparency displays over black.** The viewer composites transparent and
  partially transparent regions over pure black instead of a checkerboard. The
  checker tinted every pixel it showed through, so soft alpha edges displayed
  brighter than the graph produced them.

## Upgrade

Documents at schema v1 through v5 upgrade in place on open, chained through to
v6, gaining an empty animation section. No stored parameter changes, so an
upgraded comp renders exactly as it did before.

## Acceptance evidence

357 tests pass offscreen on the release commit. Animated parameters are pinned
against the reference evaluator at frames 1/5/10 across proxy tiers 1/2/4, and
the frames are asserted to differ, so a parameter frozen at its base value
cannot pass by matching an equally frozen reference.

## Known scope

Animated `Transform` and `Crop` render correctly but take the full-frame
fallback rather than the tile path; those kernels are not tile-supported yet.
Desktop verification is offscreen, so interactive playback feel on real
hardware is unverified.

# NodeBased 0.9.1 — Windows cache-root repair

## What changed since 0.9.0

- **Windows release-test repair.** Disk-cache initialization now survives
  sanitized Windows environments with no profile variables, using `TEMP` or
  the process directory as a safe fallback. A regression test covers it.

# NodeBased 0.9.0 — tile-native viewport and EXR data windows

## What changed since 0.8.0

- **Bounded tile evaluation.** Supported graphs evaluate in 256px tiles, with
  halo-aware Blur and explicit fallback telemetry for unsupported kernels.
- **EXR data windows survive.** Read/evaluation/tile requests preserve offsets
  and overscan beyond the display frame, including negative origins.
- **Bounded source Reads.** Full-resolution tile requests acquire the requested
  source region instead of decoding then slicing a full image.
- **Visible viewport scheduling.** The Viewer requests its visible data-window
  rectangle; panning requests newly exposed pixels. Export remains a complete
  full-resolution reference render.
- **Cache/runtime foundation.** Adaptive 8GiB-max RAM cache, persistent disk
  spill, full/half/quarter proxy tiers, and 24fps delivery controls.

## Acceptance evidence

At 4K on the recorded Linux CPU host, a centered 1920×1080 tile viewport on a
supported graph measured 312.232ms cold and 208.855ms grade-edit p50, versus
2404.175ms and 2052.728ms for the full reference frame. See
`docs/BENCHMARKS-v0.9-4k.md` for scope and machine details.

# NodeBased 0.8.0 — bounded playback and read-ahead

## What changed since 0.7.0

- **Forward playback.** Space or the transport button plays the inclusive comp
  range at document FPS and loops at the end.
- **Bounded read-ahead.** One display request is prioritized ahead of at most
  three future frames. Evaluation stays serial so one worker owns the existing
  LRU cache.
- **Deadline behavior.** The playhead derives from elapsed wall time. Slow
  frames are counted and skipped instead of building a latency spiral.
- **Cancellation and display safety.** Scrubs, edits, channel/view changes and
  stop cancel obsolete work. A result must match both the active generation and
  current frame before it may enter the Viewer; prefetch results only warm cache.
- **Undo-safe transport.** Playback updates the persisted playhead through the
  validated command boundary without consuming the 100-slot artist undo stack.
- **Explicit proxy boundary.** This release only schedules full-quality frames.
  Unimplemented proxy tiers are rejected rather than faking a proxy by scaling
  an already fully-evaluated Viewer image.

## Acceptance evidence

The playback contract in `docs/PLAYBACK.md` fixes a 16 ms enqueue budget, a
three-frame queue cap, cooperative cancellation, stale-frame exclusion, and
wall-clock deadline behavior. Automated coverage exercises the queue, real EXR
sequence cache warming, UI transport, undo preservation and wrong-frame rejection.

# NodeBased 0.7.0 — time foundation and image sequences

## What changed since 0.6.5

- **Time is now first-class.** Schema v5 stores a composition frame range,
  current frame, and frame rate. The evaluator receives an explicit timeline
  frame, leaving a clean boundary for future clips, retimes, tracks, nested
  timelines, and animated parameters.
- **Image-sequence Read.** Read accepts printf (`plate.%04d.exr`) and hash
  (`plate.####.exr`) patterns, a source-frame offset, and explicit missing-frame
  behavior: error, hold nearest, or transparent black at the sequence format.
- **Time-aware caching.** A sequence frame changes the Read fingerprint and its
  dependent branch. Static sources and unrelated branches retain their cached
  pixels while the playhead moves.
- **Minimal timeline strip.** First/current/last controls and scrubbing sit under
  the Viewer. Left/Right step; Home/End jump to the range boundaries.
- **Agent time control.** `describe` advertises time/sequence capabilities,
  `time` edits the range atomically, and headless `render` accepts a frame.
- **Windows installer identity.** The NSIS installer/uninstaller and Start Menu
  shortcut now explicitly use the NodeBased icon.

## Validation

Schema migration, sequence resolution, frame offsets, missing-frame policies,
static/temporal cache behavior, timeline UI/undo synchronization, and a real
frame-specific agent render are covered by the full automated suite.

# NodeBased 0.6.5 — the NodeBased icon

## What changed since 0.6.4

- **NodeBased has its app icon.** A folded green ribbon forming an `N`, on a
  dark rounded tile, with real transparent alpha outside the tile. Used by the
  desktop window, Windows executable/installer, and Linux AppImage.

# NodeBased 0.6.4 — visible reroute handles

## What changed since 0.6.3

- **Ctrl now visibly exposes every noodle's reroute handle.** Each connected
  noodle gets a high-contrast circular midpoint marker while Ctrl is held. The
  marker responds immediately to the global Ctrl key event, even when Qt focus
  is elsewhere in the workspace.
- **Dots behave as reroutes.** A Dot is a compact circular node with a visible
  selected outline. Its body is selectable and draggable after creation; its
  sockets remain at the top and bottom rather than covering the center.
- **App icon plumbing landed, placeholder graphic.** The desktop window,
  Windows executable/installer, and Linux AppImage now load an icon file from
  `assets/`, so a final mark drops in with no code change. The mark bundled in
  this release is an early draft that was rejected for being too detailed;
  design is still in progress and this asset will be replaced.

## Validation

141 automated tests cover Ctrl-handle state, Dot selection and movement,
transactional noodle insertion, source asset loading, and the existing graph,
compositor, media, agent, updater, and packaging behavior.

# NodeBased 0.6.3 — Dot crash repair and reliable graph sockets

## What changed since 0.6.2

- **Fixed the Dot/Switch graph crash.** Their missing theme colors caused a
  scene rebuild exception after creation, leaving a partially cleared graph and
  apparently vanished noodles. Both nodes now render safely.
- **Dots are compact reroute points.** Ctrl-dragging a noodle midpoint shows a
  live Dot preview that follows the cursor and only changes the graph on drop.
- **Larger socket hit targets.** Visible ports remain small, but their grab area
  is 26 px; the graph recognizes the visible socket child as its port target so
  active wires cannot be accidentally dropped on it.

## Validation

138 automated tests cover Dot creation without clearing graph edges, live Ctrl
preview placement, atomic insertion, symmetric wiring, and the full compositor
and packaging suite.

# NodeBased 0.6.2 — safe noodles and viewer controls

## What changed since 0.6.0

- **Safe input rewiring.** Picking up an existing input preserves its original
  connection until a valid output is dropped. Esc or a drop on empty canvas
  leaves the comp intact; a completed gesture replaces only that input.
- **Ctrl-drag a noodle midpoint to insert a Dot.** Holding Ctrl reveals a center
  handle on every connected noodle. Drag one to place a Dot that is atomically
  spliced into the connection, so it cannot leave a half-disconnected graph.
- **Viewer hotkeys:** `R`, `G`, `B`, and `A` solo their channel; pressing the
  currently soloed channel again returns to RGB. `F` fits the image; `H` is the
  matching home/frame alias for the single-image 2D viewer.

## Validation

137 automated tests cover direct/reverse wiring, transactional input rewire,
Ctrl midpoint Dot insertion, channel toggles, and framing keys, in addition to
the compositor, media, schema, agent, and updater suites.

# NodeBased 0.6.0 — masks, mix, graph routing and selection

## What changed since 0.5.1

- **Reusable mask + mix controls** now apply to Grade, ColorCorrect, Blur,
  Transform and Crop. Their optional `mask` input gates the processed result by
  mask alpha, and `mix` blends it back with the source. Existing projects retain
  the exact prior behavior: an absent mask and `mix: 1.0` are the defaults.
- **Dot** adds a zero-cost graph-routing passthrough. **Switch** selects between
  two image inputs with its `which` control; both are available to keyboard node
  search and to the agent interface.
- **Document schema v4** upgrades v3 projects by adding optional filter-mask
  slots and default mix values. v1 through v3 projects continue through the
  tested upgrade chain.
- Agents can discover Dot, Switch, filter mask ports and mix parameters through
  `describe`, and build/render schema-v4 graphs through the local JSON-lines
  interface.

## Validation

135 automated tests cover schema migration, premultiplied mask/mix behavior,
Dot/Switch evaluation, desktop graph behavior, and agent graph construction.
Native display validation of the new optional mask ports remains a human QA item.

# NodeBased 0.5.1 — graph interaction hotfix

## What changed since 0.5.0

- Restores normal **round-capped noodles** at a more readable 3.25 px stroke.
  Arrowheads are painted separately, preventing Qt from filling the curve into
  a ribbon.
- Inputs now wire in **either direction**: drag an output to an input, or drag
  an empty/connected top input to an output. A 24-screen-pixel magnetic target
  makes ports much less finicky at any graph zoom.
- **Tab is pointer-contextual.** When the mouse is over the node graph, Tab is
  captured before Qt focus traversal and opens node search even if a dock or
  control currently owns keyboard focus.

# NodeBased 0.5.0 — graph interaction repair

## What changed since 0.4.0

- **Direct noodle drag wiring.** Drag from any output port directly onto an
  input port to connect. The connection preview follows the pointer and the
  destination resolves on mouse release, so wires no longer depend on a fragile
  click sequence. Existing click-to-connect and click-a-connected-input-to-pick-
  up-and-rewire remain available.
- **Legible directional connections.** Saved and pending noodles now terminate
  in small arrowheads that point into their input. Ports stay above noodles in
  the drawing order, making them easy to grab.
- **Centered top inputs.** A single input is exactly at a node's top centre;
  multiple inputs fan symmetrically around that centre. Outputs stay centered
  at the bottom.
- **Nuke-style Tab search.** Tab opens a keyboard-first popup search at the
  last graph click. Enter creates the selected node there. The placement code
  finds the nearest vacant slot, so a new node never lands on top of an
  existing node. Single-key creation still works.

The interaction coverage now includes direct drag-to-connect, input-centering,
non-overlapping placement, and Tab-search filtering, alongside the existing
rewire/disconnect tests.

# NodeBased 0.4.0 — Full Merge operations, real Transform, Premult/Unpremult

Fourth desktop release. Where 0.3.0 added new nodes, this one makes the
compositing core itself honest: the operations an artist actually reaches for
now behave the way they do in Nuke.

## What changed since 0.3.0

- **All 16 Nuke-style Merge operations.** Merge gains an `operation` choice:
  over, under, plus, minus, multiply, screen, max, min, difference, divide,
  mask, stencil, in, out, atop, xor. Inputs are premultiplied, A is the
  foreground and B is the background, matching Nuke's wiring convention.
  `divide` guards against zero. The 0.3.0 `over`/mix behaviour is preserved
  byte-identical, so existing comps render exactly as before.
- **Transform is a real transform.** Float `translate_x`/`translate_y`,
  `rotate` in degrees, `scale`, and a `center`, with a filter choice of
  nearest, bilinear or cubic. It is inverse-mapped and sub-pixel filtered, and
  does not wrap at the edges. Defaults reduce to identity, so 0.3.0 projects
  render unchanged.
- **Premult and Unpremult** are new single-input nodes. Unpremult leaves RGB
  untouched wherever alpha is 0, so it never emits NaN or inf into the rest of
  the graph.
- **Document schema v3, with a tested upgrade path.** Opening a v2 project
  converts old integer Transform `x`/`y` into float `translate_x`/
  `translate_y`, defaults rotate/scale/center/filter to identity, and gives
  Merge `operation: "over"` where it is missing. The v1 → v2 → v3 chain is
  covered by tests. The .nbcomp compatibility promise holds: projects saved by
  0.1.0, 0.2.0 and 0.3.0 all still open.
- The new node types, parameters and choices reach `agent.describe` and the
  inspector automatically, from the same SPECS/LIMITS/CHOICES tables the
  desktop UI reads. No separate agent-protocol update was needed.

## What changed in 0.3.0

- **Wire pick-up/rewire.** Clicking an already-connected input port unhooks its
  wire into a pending connection from the same source, which can then be dropped
  on a different input, or on empty space to disconnect. Real unhook/rehook, not
  delete-and-recreate.
- **Four new Nuke-parity nodes**, with keyboard shortcuts and inspector controls:
  **ColorCorrect** (lift/gamma/gain/saturation, sign-safe under fractional gamma
  on negative HDR values), **Blur** (separable box blur, O(n) via a cumulative
  sum, edge padding avoids darkened borders), **Crop** (masks to a rectangle
  without resizing the canvas), and **Shuffle** (remaps output channels from any
  input channel or a constant 0/1).
- All new node types and the null-source disconnect/rewire semantics are exposed
  automatically to the agent protocol's `describe` operation, driven by the same
  SPECS/LIMITS/CHOICES tables the desktop UI reads.
- Projects saved by 0.2.0 load unchanged; the graph format did not change.

## What changed since 0.1.0 (also in 0.2.0)

- **EXR read and write** through OpenImageIO. Float RGBA in, zip-compressed float
  RGBA out. HDR and negative values survive a round trip exactly.
- **Read node color controls.** Input space (Auto, sRGB, Linear Rec.709, ACEScg,
  ACES2065-1, Raw), alpha interpretation (Auto, Straight, Premultiplied), EXR layer
  selection and EXR subimage/part selection. Auto reads EXR as linear/premultiplied
  and PNG/JPEG/TIFF as sRGB/straight.
- **OpenColorIO display views.** sRGB, ACES 2.0 SDR (Rec.709) and Linear, using the
  built-in `cg-config-v4.0.0_aces-v2.0_ocio-v2.5` ACES config. No external OCIO
  config file or `OCIO` environment variable is required, in packages or from source.
- **TIFF input** alongside EXR, PNG and JPEG.
- **Export image** replaces Export PNG and offers float EXR or 8-bit sRGB PNG. The
  headless agent `render` operation follows the same rule by file extension.
- **Darker theme.** Neutral charcoal surfaces with teal, amber, lavender and
  soft-blue node accents, focus rings on inputs and a distinct update button.
- Data windows are composited onto the display window, so cropped EXRs keep their
  position instead of shifting to the origin.
- Projects saved by 0.1.0 load unchanged; Read nodes gain the new controls at their
  Auto defaults.

## Downloads

- **Windows portable ZIP:** extract the entire `NodeBased` folder to a writable
  location and run `NodeBased.exe`. No installer or administrator access required.
  Keep `_internal` and `portable.marker` beside the executable. Update cache and
  rollback files stay under its `portable-data` folder. No registry installation.
- **Windows installer:** per-user installation; no elevation requested.
- **Linux AppImage:** mark executable and run. No installation or Python setup.
  Requires an x86_64 Linux desktop with glibc 2.35+ and the relevant graphics
  libraries. If FUSE is unavailable, run with `APPIMAGE_EXTRACT_AND_RUN=1`.

These early binaries are unsigned. Workplace application-control policies still
apply, including to portable programs.

## Update button

Same explicit flow as GameStore: **Check for updates → Download v… → progress →
Restart to update**. There are no automatic downloads or surprise restarts.
Unsaved work can be saved, discarded or used to cancel the restart. Downloads use
GitHub's published asset SHA-256 digests and are rechecked before installation.
Windows portable builds update in place using a separate helper; installer builds
run the per-user installer. AppImages replace their original file atomically.
A previous portable bundle/AppImage is kept for manual rollback. Keep the app in
a writable location for updates. Source checkouts use Git/pip instead.

## Scope and known limits

Still an M0-class prototype, not a production compositor. Color management runs on
the CPU per preview: on the development machine a 960 × 540 frame took ~58 ms
through the sRGB view and ~202 ms through ACES 2.0, so display transforms are not
yet interactive at high resolution and need a GPU/LUT path. Deep EXR is rejected
rather than flattened. Reads are full-frame with an 8192-per-axis cap. Embedded ICC
profiles are ignored; set Input space explicitly for non-sRGB LDR sources.
Sequences, timeline, 3D and AI generation remain roadmap work. macOS arm64 comes
later. Offscreen tests and packaged launch checks do not establish native GPU or
display performance, or workplace-policy acceptance.
