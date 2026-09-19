# 3D backend spike: GPU scene rendering options

Measured 2026-09-19 with `tools/spike_3d_backends.py` on the development machine (Linux, Python 3.12,
32 logical CPUs; RTX 3080 Ti on a USB4 eGPU plus the Radeon 8060S iGPU). One machine, one run per row,
no Windows measurements. Every GPU timing includes reading the frame back to a NumPy array. Scenes render
960×540 with a fixed camera (z=5, 45° vertical FOV) and flat-shaded triangles: a 32-segment sphere
(1,024 triangles), and jittered grids of 20,000 and 100,000 triangles.

## Candidates

| Package | Wheel (cp312 / py3) | License | Notes |
| --- | --- | --- | --- |
| `wgpu` 0.32.0 (wgpu-py, wraps wgpu-native) | one `py3-none` wheel each for manylinux x86_64, `win_amd64`, macOS | BSD-2-Clause | Vulkan / DX12 / Metal behind one API. Explicit adapter choice. Raster and compute in one API. Pulls `cffi` and `rendercanvas` (BSD-2-Clause). The wgpu-native binary it bundles was not license-audited here. |
| `moderngl` 5.12.0 (+ `glcontext` 3.0.0) | cp312 wheels for Linux x86_64, `win_amd64`, macOS | MIT | OpenGL 3.3+. Needs a GL context: EGL/GLX on Linux, WGL on Windows. No compute path we would rely on, no adapter selection (it used the NVIDIA GL driver in every run). |
| CPU reference `scene3d.render` | n/a | n/a | NumPy. Correctness reference. |

## Results (RTX 3080 Ti, Vulkan; median ms per frame including readback)

| Scene | CPU reference | moderngl | wgpu raster | wgpu compute ray-trace (brute force) |
| --- | ---: | ---: | ---: | ---: |
| sphere, 1,024 tris | 165 | 5.7 | 5.7 | 7.0 |
| 20,000 tris | 2,957 | 27.2 | 6.2 | 49.0 |
| 100,000 tris | over the 60 s budget, skipped | 27.3 | 5.3 | 241 |

Radeon 8060S iGPU through wgpu/Vulkan: raster 2.6 / 2.6 / 3.4 ms, compute 1.9 / 27.6 / 135 ms
for the same three scenes. Context/device/pipeline creation ranged from about 15 ms to 550 ms for wgpu and
100 to 170 ms for moderngl on the first scene. Adapters enumerated by wgpu: RADV STRIX_HALO (Vulkan),
RTX 3080 Ti (Vulkan and OpenGL), llvmpipe (Vulkan, CPU).

Correctness against the CPU reference on the sphere and 20k scenes: coverage IoU 1.0 for every backend;
view-space depth differs by at most 4.0e-4 (raster) and 2.6e-5 (compute) with median error below 3e-6.
The 100k scene has no CPU baseline, so it is timing only.

## What the numbers do and do not say

- GPU raster is one to three orders of magnitude faster than the NumPy reference on this machine.
  That is a measurement on this machine, not a portable claim.
- moderngl's median on the larger scenes is dominated by something the script did not isolate (its
  minimum was 4.5 ms against a 27 ms median), so its numbers are less trustworthy than wgpu's.
- The compute ray tracer tests every triangle for every pixel. Its cost grows linearly with triangle
  count, so it is only a feasibility check of compute plus readback. A real ray/path tracer needs a BVH.
- CI has no GPU. llvmpipe timings, Windows timings and Windows adapter behaviour are not measured.
- The rgba16float colour target used in the raster path holds half-precision colour. A backend that
  hands pixels to the comp graph must state its precision, see the roadmap architecture rules.

## Decision

Adopt **wgpu** as the GPU scene backend, as an optional extra (`pip install nodebased[gpu]`), never a
hard requirement. Reasons: permissive license; one wheel per OS with no system packages; the same API
serves the raster path now and compute (ray tracing, splat sorting, particles) later; explicit adapter
selection matters on this machine's dual-GPU setup; the raster path was the fastest and steadiest. moderngl
is not adopted: it duplicates raster, lacks compute and adapter control, and its timings were noisy.
The CPU rasterizer stays as the reference, as the fallback when no adapter or no wgpu wheel exists, and
as the path the test suite uses on GPU-less runners.
