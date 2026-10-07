# Lane 4 Rendering 8 step T1 — light linking everywhere

Complete on `openclaw/nb-3d-astra-lane`: linked splat sets and instance sets render in the GPU raster and ray-traced modes with no
CPU fallback, and liquid glints, lit particle sets (whitewater included) and smoke honour their links in the CPU, GPU and
path-traced renderers and the viewport; an excluded light casts no shadow from any of them. Checked against the CPU reference on
the RTX 3080 Ti, the AMD Radeon 8060S and llvmpipe (the GPU path tracer's smoke cases on NVIDIA only). Not run: Windows, CI.
Integration is pending.

# Lane 6 N1 — liquids on the GPU end to end

# Lane 4 Rendering 7 step S3 — artist-facing mixed-scene render gate

Complete on `openclaw/nb-3d-astra-lane`: `examples/mixed_scene` renders through the application's Write node into one float EXR
(`R G B A depth.Z object_id.R position.X/Y/Z`), re-read and held to the CPU reference on three adapters; a real-display session
rendered and wrote it from the UI. Beauty runs on the GPU with a transparent pane; depth, position and object_id fall back to the CPU
there and run on the GPU with an opaque pane. Not run: Windows, CI. Integration is pending.

# Lane 6 N1 — measuring the warmed liquid step

Complete on `openclaw/nb-fluids-spike`: `FluidLiquidSolver3D` with `pressure = resident` keeps particles, the MAC grid, the
extrapolation and the surface field on the card between substeps. A warmed 128-cubed substep measures 31.0 ms on the RTX 3080 Ti
(the 100 ms bar and the 40 ms goal both met), 55.7 ms on the AMD Radeon 8060S and 429.4 ms on llvmpipe; the surface level set
takes 10.4 ms on the card against 6,857.3 ms on the CPU. Tests (CPU agreement, bit-identical restart, cancel inside a substep,
kernels, level set) passed on all three adapters. Opt-in; viscosity, surface tension, a narrow band and auto-resize fall back to
`gpu` with a stated reason. Not run: Windows, the full suite. Integration is pending.

# Lane 4 Rendering 7 step S2 — transparent meshes with splats on the GPU

Complete on `openclaw/nb-3d-astra-lane`: `rgba` of scenes with splats and transparent meshes renders on the GPU in `raster`
and `raytrace` mode and matches the CPU reference (three adapters; the real capture's few outlying pixels are float32 rounding
of depth order). Tests, benchmark tool and docs are committed. Not run: Windows, the full suite. Integration is pending.

# Lane 4 Rendering 7 step S1 — GPU data passes for splat scenes

Complete on `openclaw/nb-3d-astra-lane`: `depth`, `position` and `object_id` of scenes with splats and opaque meshes render
on the GPU in `raster` mode and match the CPU reference (three adapters; the real capture's differences are float32
rounding at the 0.5 opacity threshold and the cube's raster edge). Tests, benchmark tool and docs are committed.
Not run: Windows, the full suite. Integration is pending.

# Lane 8 E3 — queue chains and LAN workers

Complete on `openclaw/nb-2d-parity-b`: persistent chains, queue panel, TCP remote worker protocol,
artifact synchronization, documentation and CPU coverage. The targeted jobs, workers, artifacts,
documentation and desktop checks passed. Animal/Windows remains untested; integration is pending.

# Lane 6 H2 finish — branch handoff

All requested code, measurements, and three-adapter targeted checks are committed on `openclaw/nb-fluids-spike`. `docs/FLUIDS_SPIKE.md` records the sparse/dense 256³ and 512³ comparisons and the 256³ path-traced plume before/after timing and 64-sample noise check. The retained tiles are compact; existing `Volume` consumers still expand them for rendering. Awaiting Gonzo's review, integration suite, and merge. No Windows or real-display run by this lane.
