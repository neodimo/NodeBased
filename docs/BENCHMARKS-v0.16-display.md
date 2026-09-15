# v0.16 display-transform benchmarks

Status: measured evidence for the viewer display-transform speedup. Numbers below are the
authority for any future claim about this path; do not restate a faster number without a
new measurement in this file.

## Machine

- CPU: 32 logical cores (`os.cpu_count()`), Linux `7.2.4-ogc3.1.fc44.x86_64`, x86_64.
- GPU: dual-GPU laptop/eGPU rig. Integrated **AMD Radeon 8060S (Strix Halo, radeonsi/Mesa
  26.2.2, OpenGL 4.6 core)** owns the display and is the GLX/EGL default vendor. A
  **discrete NVIDIA RTX 3080 Ti** is present as a USB4 eGPU (driver 615.71.09) but is not
  the default render target — see "AMD vs NVIDIA" below for why that turned out to matter.
- Qt: PySide6 6.11.2. OCIO: PyOpenColorIO 2.5.2, `ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5`.
- Display context: `QT_QPA_PLATFORM=xcb DISPLAY=:0` (real X server on Wayland via XWayland).
  `QT_QPA_PLATFORM=offscreen` was also tested (see "Offscreen" below): on this specific
  machine it *also* gets a real, driver-backed GL context, because the offscreen QPA
  platform can still reach EGL/DRM directly. That is a property of this machine, not a
  guarantee — the fallback path exists precisely because a cloud CI runner's `offscreen`
  platform typically cannot do this (no GPU, no DRM render node).

## How to reproduce

```
QT_QPA_PLATFORM=offscreen .venv/bin/python -m nodebased.bench_display --backend cpu
QT_QPA_PLATFORM=xcb DISPLAY=:0 .venv/bin/python -m nodebased.bench_display --backend all --iterations 30 --json
```

`nodebased/bench_display.py` times `nodebased.color.display_rgb`'s underlying primitives in
isolation, on deterministic synthetic float32 ACEScg-shaped data (negatives, >1 highlights,
mid-grey body — see `synthetic_frame`), independent of file I/O or the rest of the evaluator.
20-30 iterations after 1 warmup call; `min`/`median`/`max` reported in milliseconds.

## Baseline (git history before this work, `52bb039` lineage)

`display_rgb`'s ACES 2.0 branch built a brand-new `DisplayViewTransform` processor via
`config().getProcessor(transform).getDefaultCPUProcessor()` on every single call, then ran
one single-threaded `applyRGB`. `sRGB` already used the `lru_cache`d `processor()`.

| resolution | view     | median   |
| ---------- | -------- | -------- |
| HD (1920×1080) | sRGB     | 27.99 ms |
| HD             | ACES 2.0 | 537.45 ms |
| 4K (3840×2160) | sRGB     | 129.40 ms |
| 4K             | ACES 2.0 | 2169.50 ms |

This matches the task brief's stated baseline (HD ~550 ms / 4K ~2.2 s, ACES 2.0 ~10x sRGB).

## Step 1: cache the ACES 2.0 processor alone (measured separately, then discarded as insufficient)

Caching only the processor construction (`display_processor()` in `nodebased/color.py`)
was tested in isolation, single-threaded:

| resolution | view     | median (cached, single-thread) |
| ---------- | -------- | ------------------------------- |
| HD | ACES 2.0 | 538.15 ms |
| 4K | ACES 2.0 | 2171.83 ms |

**No measurable improvement.** Processor construction itself is cheap (~11 ms cold, ~0 ms
cached); `applyRGB` is what costs 530-2170 ms, so caching alone does not touch the real
cost. This is why "cheap win" here had to be about `applyRGB` itself, not about the object
that owns it.

## Step 2: threaded CPU `applyRGB` (the real CPU-side win)

`PyOpenColorIO`'s `CPUProcessor.applyRGB` releases the GIL, so chunking the image into
row bands and applying the *same* processor to each band across a `ThreadPoolExecutor`
(`nodebased.color.apply_threaded`, `_CPU_WORKERS = min(os.cpu_count(), 16)` in a persistent
pool) is real parallelism, not contended bytecode. This is exact, not approximate: each
chunk is a contiguous C-order slice of the same array, so the output is bit-identical to
one full-image `applyRGB` call (`tests/test_display_transform.py::ApplyThreadedTests`
asserts exact equality, including non-evenly-divisible odd sizes).

| resolution | view     | median (CPU, cached + 16-way threaded) | speedup vs. baseline |
| ---------- | -------- | --------------------------------------- | --------------------- |
| HD | sRGB     | 4.95 ms  | 5.7x |
| HD | ACES 2.0 | 45.94 ms | **11.7x** |
| 4K | sRGB     | 19.90 ms | 6.5x |
| 4K | ACES 2.0 | 176.30 ms | **12.3x** |

Optimization-flag sweep (`OPTIMIZATION_DEFAULT` / `LOSSLESS` / `VERY_GOOD` / `GOOD` /
`DRAFT` via `getOptimizedCPUProcessor`) made no measurable difference (within ~1.5%,
noise-level) — the ACES 2.0 RRT/ODT cost here is dominated by the per-pixel math itself
(exponentials, table lookups), not by op-graph structure a flag can simplify away.

## Step 3: GPU path

`nodebased/gpudisplay.py` builds each view's `GpuShaderDesc` (GLSL 4.0) once, uploads its
LUT textures (two 1D textures for ACES 2.0: `reach_m_table` R32F and `gamut_cusp_table`
RGB32F; zero for sRGB, which is pure matrix+gamma), and reuses a `QOpenGLShaderProgram` +
input texture + FBO across calls. Design notes, in order of how much they mattered:

1. **Input texture is RGB32F, not RGBA32F.** The first working version padded a per-pixel
   alpha=1.0 in NumPy before every upload; that padding *alone* cost ~32 ms at 4K — as much
   as the rest of the GPU path combined. The constant alpha is now supplied inside the
   fragment shader (`vec4(c, 1.0)`) instead.
2. **Readback is `GL_RGB`/`GL_FLOAT` into a persistent per-size buffer with
   `GL_PACK_ALIGNMENT=1`**, not `GL_RGBA` into a fresh `bytearray` per call. RGB float32
   rows are already 4-byte aligned (12 bytes/pixel), so this is a tight, correct readback
   that also skips the "drop the alpha channel" NumPy copy a `GL_RGBA` readback needed.
3. **No vertex buffer.** The full-screen triangle is generated in the vertex shader from
   `gl_VertexID` (the standard attributeless-triangle trick); only an empty VAO is bound,
   because core-profile draw calls require *a* bound VAO even with no attributes.
4. **Uniform binding uses raw `glUseProgram`/`glGetUniformLocation`/`glUniform1i`
   (`QOpenGLExtraFunctions`), not `QOpenGLShaderProgram.setUniformValue`.** The Qt wrapper
   method produced `GL_INVALID_OPERATION` and an all-black result for the ACES 2.0 shader
   specifically (multiple sampler uniforms); the raw calls work correctly. Shader
   compile/link still goes through `QOpenGLShaderProgram` for its `.log()` diagnostics.
5. `QOpenGLTexture` (not raw `glTexImage1D`/`glTexImage3D`) builds the LUT textures,
   because this PySide6 build's `QOpenGLExtraFunctions` does not expose `glTexImage1D` at
   all (`hasattr` is `False`); `QOpenGLTexture.Target1D`/`Target3D` resolve their own GL
   entry points independently and work.

| resolution | view     | median (GPU, AMD iGPU) | vs. threaded CPU | vs. baseline |
| ---------- | -------- | ------------------------ | ----------------- | ------------ |
| HD | sRGB     | 6.11 ms  | 1.2x **slower** | 4.6x |
| HD | ACES 2.0 | 5.37 ms  | **8.6x faster** | **100.1x** |
| 4K | sRGB     | 46.15 ms | 2.3x **slower** | 2.8x |
| 4K | ACES 2.0 | 46.73 ms | **3.8x faster** | **46.4x** |

The GPU path's cost is dominated by the fixed per-call texture upload + readback, not by
which transform it runs — HD ACES 2.0 and HD sRGB finish in essentially the same time on
GPU (5.4 vs. 6.1 ms), and the same is true at 4K (46.7 vs. 46.1 ms). That is a real,
measured regression for `sRGB` if the GPU path were used unconditionally, worst at 4K
(2.3x slower than just running threaded CPU).

### Why `display_rgb` only routes ACES 2.0 to the GPU

Given the above, `nodebased/color.py` routes only `view == 'ACES 2.0'` through the GPU path
(`_GPU_PREFERRED_VIEWS`); `sRGB` always uses the threaded CPU path. This is a deliberate
refinement of "auto-select GPU when available," not a literal always-prefer-GPU policy,
because the literal version is a measured regression on the cheap view. ACES 2.0 is also
the shipped default display view for new projects (`docs/COLOR_MANAGEMENT.md`), so this
routing targets the case real playback actually hits by default. A future proxy/tile
system that also wants GPU acceleration for `sRGB` frames smaller than roughly HD should
re-measure before flipping that switch, since the upload/readback overhead is largely
fixed cost and may cross over at smaller sizes.

### AMD vs. NVIDIA (why the default vendor was kept)

The RTX 3080 Ti is reachable via PRIME render offload (`__NV_PRIME_RENDER_OFFLOAD=1
__GLX_VENDOR_LIBRARY_NAME=nvidia`), confirmed working (`NVIDIA GeForce RTX 3080
Ti/PCIe/SSE2` reported by `glGetString`). It is measurably **slower** for this workload:

| resolution | view | GPU (AMD iGPU, default) | GPU (NVIDIA RTX 3080 Ti, PRIME offload) |
| --- | --- | --- | --- |
| HD | sRGB | 6.11 ms | 32.48 ms |
| HD | ACES 2.0 | 5.37 ms | 33.10 ms |
| 4K | sRGB | 46.15 ms | 147.29 ms |
| 4K | ACES 2.0 | 46.73 ms | 147.56 ms |

This machine's NVIDIA GPU is an eGPU that does not own the display (see
`projects/bazzite-gpu-session` notes: the AMD Strix Halo iGPU owns HDMI/output, the RTX
3080 Ti is attached over USB4). PRIME render offload adds a real cross-GPU synchronization
cost to every upload/readback round trip. `gpudisplay.py` does not force a vendor; it uses
whatever `QOpenGLContext` the platform hands back, which on this machine is correctly the
faster option. A different machine with a single discrete GPU should not see this gap.

## Correctness gate

`tests/test_display_transform.py::GpuDisplayTests.test_gpu_matches_cpu_within_one_8bit_code_value`
compares the GPU and CPU paths on a wide-gamut/HDR test image (`hdr_gamut_image`): random
ACEScg-range noise plus fixed rows of saturated primaries/secondaries (up to 16.0),
negative values, exact zero, and near-zero (`1e-7`). Measured on this machine:

| view | max abs float error | max 8-bit code-value error (after quantization) |
| --- | --- | --- |
| sRGB | 1.26e-05 | 1 |
| ACES 2.0 | 2.19e-05 | 1 |

Tolerance is `<= 1` code value after 8-bit quantization, matching the task brief; both
views are at that boundary, consistent with float32 rounding differences between the CPU
and GPU evaluations of the same OCIO-generated math, not a correctness gap.

## Fallback behavior

- `NODEBASED_DISPLAY_GPU=0` forces the CPU path unconditionally
  (`GpuDisplayDisabledTests::test_env_var_disables_gpu`).
- No `QApplication`/`QGuiApplication` instance (a bare script, an early-loaded test module):
  `GpuDisplay.__init__` raises before touching `QOffscreenSurface`, which otherwise
  segfaults rather than raising when no Qt application exists yet. Falls back to CPU.
- `QT_QPA_PLATFORM=offscreen` with no working GL (the CI case on GitHub-hosted Ubuntu/
  Windows runners): `QOpenGLContext.create()`/`makeCurrent()` fail, caught and converted to
  `GpuUnavailable`, falls back to CPU. Not directly reproducible on this machine (its
  `offscreen` platform does get real GL — see "Machine" above), but the exception handling
  is unconditional and structural, not conditioned on which platform is active.
- A GPU failure mid-session (simulated in
  `GpuDisplayTests::test_forced_gpu_failure_falls_back_to_correct_cpu_pixels` by making
  `render()` raise) falls back to CPU for that call and returns pixel-identical output to
  the CPU reference; it is a per-call fallback rather than a permanent one, so a transient
  fault does not give up GPU acceleration for the rest of the session.
- `nodebased.color.display_rgb` never lets a `gpudisplay.GpuUnavailable` escape; every path
  either returns GPU output or the threaded CPU path's output.

## Test counts

- Focused: `QT_QPA_PLATFORM=xcb DISPLAY=:0 .venv/bin/python -m unittest
  tests.test_display_transform` — **12 tests passed in 0.19s** (exercises the real GPU
  path). Same file under `QT_QPA_PLATFORM=offscreen` — **12 tests passed in 0.20s** (also
  exercises the real GPU path on this machine, per the "Machine" note above).
- Full: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests` —
  **514 tests passed in 127.9s** (502 pre-existing + 12 new).

## Real-app evidence

A real 4K EXR (`noise_test_4k.0001.exr`, ACEScg-tagged) was loaded into a live `Window`
under `QT_QPA_PLATFORM=xcb DISPLAY=:0` (no `--smoke-test`, a direct script drove the
window and read `viewer_info.text()` after the first frame rendered). Status text:

```
3840 × 2160  ·  993 ms  ·  tiles 0 hit/270 miss  ·  cache 0.0 / 8192 MiB  ·  display GPU
```

`993 ms` is the full cold tile-compose-plus-decode time for a first-time 4K frame (270
tile misses, EXR decode included), not the isolated display-transform cost measured above;
`display GPU` confirms the GPU backend is what actually served this real render, not just
the isolated benchmark.

## Unverified

- No GPU-less machine was available to reproduce the CI fallback path end-to-end (only the
  code path and exception handling were verified, plus the `NODEBASED_DISPLAY_GPU=0`
  forced-CPU path, which exercises the same `display_rgb` fallback branch).
- Windows GL context creation/fallback is unverified; only Linux/xcb and Linux/offscreen
  were tested on this machine.
- The GPU-vs-CPU crossover resolution below which GPU stops winning even for ACES 2.0 was
  not measured (only HD and 4K); a proxy tier well below HD may not benefit from GPU.
