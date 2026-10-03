# v0.34 — shadows on 100,000 instanced copies: GPU culling and detail levels

Plan "Rendering 6", step R3. The 3D viewport's shadow maps now take their `Instance3D` casters from a per-light list
that a compute pass builds on the GPU (`viewportgpu._CULL_SHADER`): each copy's bounding sphere is tested against the
light's six frustum planes, and the survivors are sorted into three detail levels by how many shadow-map texels they
span, then drawn by indirect calls. 0.33.0 held 30 fps only when most copies fell outside the lights' view; this page
measures the case it listed as a known limit, every copy inside four lights' views at once.

Measured on October 3, 2026 (PDT) on the culling's commit `af52846` (the tool's `--cull` option, which selects the 0.33.0
path for the "before" rows, was added after it and does not change the "after" path), Linux 7.2.7-ogc1.1.fc44.x86_64, Python
3.12.13, NumPy 2.5.3, wgpu-py 0.32.0, on this workstation with no test suite running. Script: `tools/benchmark_instances.py`.

## What was measured

- **Scene.** 100,000 copies of a 990-triangle sphere (radius 1) scattered uniformly in a box, four shadow-casting
  Directional lights at 90 degree steps, PBR floor, 1920 by 1080, the viewport's `ViewportRenderer.render` including the
  read-back. Wall time is the best of three batches of ten frames after a warm-up frame (one batch of two frames on
  llvmpipe, which takes seconds a frame).
- **`outside`.** The copies scatter over +-200 units while the lights frame an 8-unit floor, so almost every copy is outside
  every light's view: the case 0.33.0 already held at 30 fps (the lights keep 4, 4, 2 and 8 of the 100,000).
- **`in_view`.** A 440-unit floor, copies over +-150 units: every light frames the whole scatter, and the tool reads back that
  each of the four lights keeps all 100,000 copies. The camera is the small one `tests/test_3d_viewport_shadows.py` uses, so
  the camera pass draws the few copies near it and the row isolates the shadow cost. This is the budget case.
- **`unshadowed`.** The `in_view` scene with the lights' shadows off: what the same frame costs without the shadow passes.
- **`in_view wide` and `unshadowed wide`.** The same two scenes through a camera that frames the whole scatter, so the
  camera pass draws every copy at full detail. That pass alone is over the budget on the RTX 3080 Ti (39.4 ms, 25.4 fps
  without shadows); these rows are reported for context and are not the budget case.
- **Before** is `--cull cpu`: the 0.33.0 path, where NumPy tests every copy against each light's planes and draws the
  survivors at full detail. It stays in the code as the fallback for a device that cannot bind the lists.
- **Detail levels.** A copy whose bounding sphere spans 32 texels or more in the light's shadow map draws as the full mesh;
  8 to 32 texels as a vertex-clustered copy of the mesh on a 6-cell grid (252 of the sphere's 990 triangles); under 8
  texels as a 3-cell one (48 triangles). In `in_view` the spheres span about 3 texels, so all 400,000 shadow draws take the
  coarsest level. `tests/test_3d_viewport_shadow_cull.py` renders 1,500 such copies both ways: the picture differs by less
  than half a grey level on average.

## NVIDIA GeForce RTX 3080 Ti (discrete), 1920 by 1080

| case | before | after | 30 fps budget |
| --- | ---: | ---: | --- |
| outside | 31.0 ms (32.3 fps) | 27.2 ms (36.7 fps) | within |
| in_view | 146.9 ms (6.8 fps) | 26.3 ms (38.0 fps) | within |
| unshadowed (in_view scene) | 24.4 ms (41.0 fps) | 25.5 ms (39.2 fps) | within |
| in_view wide | 161.3 ms (6.2 fps) | 42.4 ms (23.6 fps) | over |
| unshadowed wide | 38.8 ms (25.8 fps) | 39.4 ms (25.4 fps) | over |

## AMD Radeon 8060S (integrated), 1920 by 1080

| case | before | after | 30 fps budget |
| --- | ---: | ---: | --- |
| outside | 34.5 ms (29.0 fps) | 30.8 ms (32.4 fps) | within |
| in_view | 140.2 ms (7.1 fps) | 33.9 ms (29.5 fps) | over by 0.6 ms |
| unshadowed (in_view scene) | 28.9 ms (34.6 fps) | 29.0 ms (34.5 fps) | within |
| in_view wide | 141.3 ms (7.1 fps) | 35.4 ms (28.3 fps) | over |
| unshadowed wide | 30.7 ms (32.6 fps) | 29.8 ms (33.5 fps) | within |

The integrated adapter's own camera pass takes 29 ms, so four shadowed lights at 100,000 copies land at 29.5 fps, just
under the line; the test that holds 30 fps is for the discrete card and skips here with a message that points to this page.

## llvmpipe (software), 1920 by 1080

| case | before | after |
| --- | ---: | ---: |
| outside | not measured | 3307 ms (0.3 fps) |
| in_view | 48,759 ms (0.02 fps) | 4636 ms (0.2 fps) |
| unshadowed (in_view scene) | not measured | 3649 ms (0.3 fps) |

Software rendering is not a real-time target; the rows show the culling and detail levels cut the in-view shadow cost
from 45 seconds to about one second a frame there too.

## Reading the numbers

- On the workstation card four shadow-casting lights over all 100,000 copies cost about 1 ms on top of the unshadowed frame
  (26.3 ms against 25.5 ms, inside run-to-run noise), against 122 ms before.
- The detail levels carry that result. Culling alone cannot help when every copy is in view; what the lights cannot resolve
  (a 1-unit sphere is about 3 texels in a 440-unit frustum) is no longer drawn as 990 triangles.
- Not measured here: the camera pass at 100,000 full-detail copies, which is over the budget on its own; Windows.
