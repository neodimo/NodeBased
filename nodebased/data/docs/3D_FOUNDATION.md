# 3D foundation

The first 3D milestone is deliberately bounded. `Card3D`, `Cube3D`, `Camera3D`,
`Scene3D`, and `Render3D` provide typed scene assembly and a practical card/cube →
camera → image → Write workflow. `Render3D` emits the existing float32,
scene-linear, premultiplied RGBA raster, so ordinary 2D nodes can follow it.

The renderer is a deterministic NumPy CPU/reference rasterizer with a z buffer,
perspective projection, flat RGBA materials, transforms, and a transparent/solid
background option. It is suitable for correctness and small foundation scenes;
it makes no GPU throughput claim. The editor's 3D viewport uses this same reference
path. Its orbit, pan, dolly, and frame state is transient UI state and never edits
the authored `Camera3D` node or the camera used by `Render3D`.

Connections are typed: image, geometry, scene, and camera ports reject mismatches
atomically. Existing schema upgrades, undo/redo, animation and agent discovery remain
available; 3D parameter animation uses the existing numeric knob mechanism.

Out of scope here: textured cards, USD/Hydra, materials/lights/shadows, AOVs, deep
output, ray tracing, Gaussian splats, particles, fluids, GPU scene evaluation, and
full Nuke parity. Those remain roadmap milestones and must not be inferred from this
foundation.
