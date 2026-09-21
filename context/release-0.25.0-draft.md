# DRAFT: NodeBased 0.25.0 release notes (not released)

Drafted 2026-09-20 9:40 PM PDT against `main` `3aeb393`. This is a working draft, kept out of
`docs/RELEASE_NOTES.md` until the release commit so the bundled-copy test and the release workflow's
notes extraction are not disturbed. Every number below comes from `context/state.md` or `TASKLOG.md`
and names who measured it. `[MEDIA: ...]` marks a still or clip the release gate requires (Release
policy, "Release notes carry screenshots and video"); none has been captured yet.

Scope is DiMo's 4:46 PM directive of 2026-09-20: work that was already started, plus the bypass bug
he reported in shipped 0.24.0. Everything else planned for 0.25.0 moved to 0.26.

---

# NodeBased 0.25.0 — splats and ray tracing move to the GPU, and bypass works

## Fixed in this release

- **Bypassing (disabling) a node now works.** In 0.24.0, bypassing a `Merge` put the text `'grade'`
  in the viewer, and bypassing a `Grade` changed nothing on screen. The first was a crash whose whole
  message was the name of a node; the second was the viewer's tiled path running the node it was
  meant to skip. Both evaluation paths now share one rule, and a bypassed node never runs.
  **Behaviour change:** a bypassed `Merge` passes **B**, its background, as Nuke does (A when B is
  not connected). A saved script that contains a bypassed `Merge` will look different.
  [MEDIA: clip, the graph from the bug report: bypass Merge, bypass Grade, viewer follows each toggle]
- **Splat shadows appear on Windows.** On a Direct3D 12 backend, a light aimed straight down an
  axis cast no splat shadow at all in the GPU ray tracer: the surface came back at its plain
  unshadowed colour. Caught by the release packaging gate before any tag, so no released build ever
  had it. GPU test logs now also name the adapter, its backend and whether the colour target is
  float32 or half float, so a parity number can be read with the hardware that produced it.
  [MEDIA: still, the same splat-cast shadow on Windows and Linux side by side]
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
  [MEDIA: clip, same 200k-splat frame on CPU and GPU with the timer visible]
- **A ray-traced mode on the GPU.** `Mode` `raytrace` now runs on the GPU for mesh scenes: textures,
  lights, specular, emission, shadows through transparent surfaces, up to 64 surfaces per ray, and
  every data pass (`depth`, `normals`, `position`, `uv`, `object_id`, `albedo`, `diffuse`, `specular`,
  `emission`). A lit, shadowed 40,002-triangle scene at 1920 x 1080 takes 1.5 s on an RTX 3080 Ti
  (reviewer) against roughly two minutes on the CPU tracer (extrapolated from 320 x 180, so the ratio
  is rough). Passes agree with the CPU tracer within 1.4e-5; object ids are exact.
  [MEDIA: still, ray-traced beauty plus three passes side by side]
- **Splats cast shadows in the GPU tracer.** Splats shadow meshes on the GPU in `raytrace` mode, and
  the visible splats are composited over the traced image. 200,000 splats casting and visible over a
  floor at 1920 x 1080: about 3 s cold, 2.3 s warm (reviewer). The 3.4-million-splat capture renders
  this way in 33 to 45 s and is not refused (lane's figure; not compared against the CPU).
- **Shadows caught on a capture without relighting it.** `ReadSplat3D` has a `Catch shadows` slider.
  CG meshes between a shadowed light and the capture darken the capture's own colours, so an object
  dropped into a scan grounds itself while the scan keeps its look; `Relight` can stay at 0. At 0 the
  render is byte-identical to before and traces nothing.
  [MEDIA: clip, cube over the Nelson capture, Catch shadows 0 to 0.85]
- **Cast shadows on or off per capture.** An environment capture's sky shell or walls otherwise block
  every light in the scene. With `Cast shadows` off the capture still renders, still receives and
  catches shadows, costs nothing in the shadow budget and builds no splat BVH.
- **Splat shadows are cached.** Visibility per splat is kept between renders, keyed by the casters,
  the mesh occluders and the light's geometry. Changing colour, intensity, ambient, `Relight` or the
  camera traces nothing. A 200,000-splat shell at 640 x 360: 36.4 s the first time, 5.3 s after.
  Moving a light still costs a full trace.
- **Large splat frames show progress and are no longer refused in the app.** A slow CPU splat frame
  shows a progress bar in the status bar with a percentage and the time left, a newer edit cancels it
  between tiles, and single-image export gets the same bar. The 3.4M-splat capture at 1920 x 1080,
  refused in 0.24.0, renders in about 95 s on the CPU. The estimate is pessimistic before 5% and
  within about 10 s from half way. The agent CLI and batch renders still refuse over-budget frames.
  [MEDIA: clip, the 1080p Nelson frame in the app with the bar and ETA running]
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
- **Bypass latency:** a toggle takes about 300 ms the first time and 170 ms after. Evaluation is a
  cache lookup; about 125 ms is the CPU display conversion every viewer update pays. Unchanged in
  this release.
- **Windows:** the GPU tests skip on CI's software adapter, and none of the GPU work in this release
  has run on a real Windows GPU.
- **The 3D viewport still shows splats as opaque discs.** Relit, blended splats in the viewport are
  not in this release.

## Moved to 0.26

Relight passes and a 2D Relight node, better splat normals, shadow offset and blur controls, kept
specular, `WriteSplat3D`, sphere rows and columns, the matrix readout, multichannel EXR, particles,
volumes, a glTF reader, and the 2D-to-3D integration (SHARP image to splat, Pixal3D image to mesh,
WorldSculpt splat to meshes).

---

## Media checklist (release gate)

| # | What | Kind | Source |
| - | ---- | ---- | ------ |
| 1 | Bypass Merge and Grade, viewer follows | clip | the running app, the bug-report graph |
| 2 | 200k splats, CPU against GPU | clip | the running app, synthetic cloud |
| 3 | GPU ray-traced beauty + passes | still | Render3D outputs, mesh scene |
| 4 | Catch shadows on the capture | clip | the running app, Nelson capture (read-only) |
| 5 | Progress bar and ETA at 1080p | clip | the running app, Nelson capture (read-only) |
| 6 | Cast shadows off on an environment shell | still pair | Render3D outputs |

Record beside each file the commit it was captured at and the script or scene that made it. Nothing
derived from the capture goes into the repo; media goes on the release page, MP4s as release assets.
