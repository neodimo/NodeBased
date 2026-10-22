# Lane 6 step B — PARTIAL (2026-10-22)

Optimized RTX 3080 Ti frame-60 960 × 540 fixed-32 render: 22.735 s, down from 200.38 s with 16 volume-shadow steps. Optimized 1920 × 1080 adaptive render: 83.511 s, mean 8.379 spp, cap 64, 99.405% pixels under threshold 0.01. NVIDIA CPU/GPU frame comparisons 1/60/120 passed; targeted adaptive/majorant modules passed on NVIDIA, forced integrated invocation, and llvmpipe (12 each). Radeon 8060S 64-cell preview: 134.31 warm FPS; 13.4 ms stop-to-full-detail. New stills remain visibly noisy and the stream reads weakly. A6000 and Windows unverified. Evidence/docs landed in `1b267f28`; exact next-owner handoff is in TASKLOG.md and context/state.md. Scratch profile was removed after durable evidence was saved.

# Lane 8 step A finish 1 — active (2026-10-20, 11:02 AM PDT)

Scope from the posted partial report: bring RTX drag-first-picture below 0.5 s, reach 24 fps cached tier-4 playback, and address the ranked full-resolution Roto / tier-4 Roto costs. Original brief and COMMON.md read. Branch is at integrated `7eeafdcf` (same as local `main` ref); worktree clean at start. Committed first fixes at 4919011a. Added a 128 MiB bounded resolved-matte cache, with shape edits and proxy tiers in its key; animated source frames reuse static shape coverage. Targeted Roto, playback and preview checks pass (108). RTX benchmark waits behind Lane 4 at the exclusive card lock; next remeasure all adapters.

# Lane 8 T2 finish 1 — PARTIAL (2026-10-20)

Real-display measurements on the merged 1920 × 1080 QA comp: RTX 3080 Ti drag first picture 1.142 s after the move (after release), settle 3.888 s; scrub first picture 0.595 s, with tier 8/4/1 at 0.595/0.978/4.914 s. Playback: 0.125 fps at tier 1, 2.624 fps at tier 4, 12.84 fps with automatic tier changes 1→2→4 against a 24 fps target. llvmpipe during Lane 4's integrated-adapter progression benchmark: drag first picture 1.414 s after release, settle 4.416 s; scrub first picture 0.808 s, settle 9.446 s; playback 0.125 fps at tier 1, 1.618 fps at tier 4, 0.125 fps automatic. Latency and rate gates remain unmet. Detailed logs are in `scratch/nb-lanes/run/qa-L8-T2-{llvmpipe-1020-r2,rtx3080ti-1020}.log`.


# Lane 8 T2 initial implementation handoff (PARTIAL, 2026-10-19)

Implementation and targeted checks are committed on `openclaw/nb-2d-parity-b`. 72 targeted tests passed. Progressive edit/scrub previews, adaptive playback tiering, and PlanarTracker worker progress/cancel are in place. Real-display latency and playback FPS on 12 GB and llvmpipe remain unverified because the integrator full suite holds the shared GPU lock. Gonzo owns follow-up using `/var/home/omid/.openclaw/workspace/scratch/nb-qa/1015-roi/scripts/s12_t2.py` after the lock clears.

# Lane 6 P4 finish 1 — stop-time cache reuse (active, 2026-10-12)

The preview budgets pass. Root cause is fixed in `11370fc`: whitewater now loads liquid frames from the upstream ParticleCache3D store, avoiding eviction by its smaller store. Six focused P4 modules pass (22 tests). A clean current-code 120-frame bake is at `scratch/nb-lanes/run/L6-p4-cache`; its 49.32 GB cache contains 120 whitewater states and 119 surface meshes. The Radeon stop benchmark is queued behind the active full-suite GPU lock; stop results at frames 1, 60 and 120 remain pending.

# Lane 8 NL7 — motion — active (2026-10-09, 3:18 PM PDT)

Shared animator, Reduce motion preference, family panel transition, eased Properties controls, graph node glides, and playback playhead are implemented. 76 targeted UI/motion tests and 7 documentation tests pass. Screenshot and completion notes are committed on this branch; integration and display QA remain with Gonzo.

# Lane 6 P2 — interactive cached-fluid preview — PARTIAL (2026-10-09)

Preview implementation, UI behavior, benchmark records and docs are committed. Active 64³/48³,
24-frame Hot pour preview measured 14.96 FPS on Radeon and 11.06 FPS on llvmpipe. The production
256³/192³ preset was not benchmarked, so its playback target remains unverified. Targeted checks
passed. Waiting for integration; Gonzo owns deciding any production-size follow-up.

# Lane 6 N3 finish 1 — active (2026-10-08, 12:57 AM PDT)

The full-size Radeon bake was regenerated (1,940.5 s) and committed with updated metrics in `benchmarks/hot_pour/radeon-full-bake.json` and `docs/SIMULATION.md`; cache is at `/tmp/hot-pour-integrated-full-1008`. Playback command is queued on `/tmp/nb-gpu.lock` behind a parallel `python -m unittest discover -s tests` process, active for >32 min; do not interrupt it. RTX stays pending card reset. Remaining: playback, stills, reduced Radeon/llvmpipe runs, targeted Hot pour tests, and final issue/board/report update.

The full-size Radeon bake was regenerated (1,940.5 s) and committed with updated metrics in `benchmarks/hot_pour/radeon-full-bake.json` and `docs/SIMULATION.md`; cache is at `/tmp/hot-pour-integrated-full-1008`. Playback command is queued on `/tmp/nb-gpu.lock` behind a parallel `python -m unittest discover -s tests` process, active for >32 min; do not interrupt it. RTX stays pending card reset. Remaining: playback, stills, reduced Radeon/llvmpipe runs, targeted Hot pour tests, and final issue/board/report update.

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

# Lane 6 S2 — hot-pour picture quality (PARTIAL, 2026-10-15)

The corrected 64-cell bake completed on the RTX 3080 Ti in 34.94 seconds. The plate now carries one glass and the transparent holdout aligns with it. Corrected 32-sample 960 × 540 frames 60, 90 and 120 took 201.06, 201.88 and 200.04 seconds; the frame-120 CPU holdout crop matched the plate exactly in all 32 interior pixels. `tests.test_hot_pour_shot` and `tests.test_knowledge` pass. The 8/32/128 sweep on frames 60 and 90 predates the plate cleanup; corrected full sweep remains incomplete. Progressive first-look/settle and Radeon preview/stop were not measured: the progressive command waited 30 minutes while the integrator full-suite runner held the shared GPU lock. The 2.2 GB generated cache was moved with `gio trash` to `/var/home/omid/.local/share/Trash/files/hot-pour-shot`; it remains recoverable and allocated, so permanent cleanup is handed off. The exclusive GPU-lock wait expired behind the integrator suite; no bypass was attempted. Needs Gonzo: rerun `run/progressive-L6-s2.py` and `tools/benchmark_hot_pour.py` at 64 cells after the lock clears.

## Lane 8 step A continuation — 10/22 2:14 AM PDT
Final-code RTX, uncontended llvmpipe, and integrated laptop-stand-in measurements are committed with the updated before/after comparison. RTX first picture is 0.474 s during drag; llvmpipe is 0.517 s during drag; integrated stand-in still produces no in-drag picture and takes 45.358 s to settle. Tier-4 playback is 19.363, 18.834, and 8.100 fps respectively, below the 24 fps target. The earlier contention-affected llvmpipe run was excluded. Targeted docs-copy and decode-ahead checks pass; the targeted preview/Roto/playback set also passes. Issue and board updated; Needs Gonzo: continue toward the remaining performance gates.

## Lane 8 step A continuation — 10/22 3:15 AM PDT
The worker profile identified repeated paint compositing and multichannel EXR reads among the frame costs. A solid-color paint-over fast path and shared beauty/layer EXR decode are committed; focused paint, media, bounding-box, cached-playback, and documentation-copy tests pass. Final-code adapter runs measure tier-4 playback at 22.089 fps RTX, 21.215 llvmpipe and 13.611 integrated. RTX drag first picture is 0.516 s during movement; llvmpipe is 0.497 s; the integrated stand-in produces no in-drag frame and takes 45.361 s to settle. The 24 fps target remains open on all adapters; the 0.5 s target passes only on llvmpipe. Issue and board will be paused with this partial result.
