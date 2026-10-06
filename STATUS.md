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
