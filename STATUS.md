# Lane 6 step B finish 2 — PARTIAL (2026-10-23, 2:34 AM PDT)

Production-size gates ran on RTX 3080 Ti at 256³ liquid / 192³ steam. Pour keyed at frame 60; profiled first edited scene evaluation took 346.208 s (before display render), dominated by collider voxelization (188.852 s). Timeline reached frame 70 at 16.487 s, missing the 12 FPS target. The 120-frame proxy total, full-detail stage split, cup/steam edits and integrated Radeon remain incomplete; A6000 is estimated only. Corrected harness sets the viewer output; prior blank-view results are superseded. Production scratch was moved to trash after capturing the committed JSON. Docs knowledge tests pass. Commit: ed77ee22.

# Lane 8 step B finish 6 — active (2026-10-22)

**RTX read-ahead A/B on current main, 1080p, 10 s uncached, pipeline enabled.** Decode-ahead on: first-pass 5.680, 5.382, 5.449 fps (mean 5.504); node evaluation 173.59, 177.40, 176.49 ms (mean 175.83); display conversion 80.13, 83.39, 83.66 ms (mean 82.39). Decode-ahead off: 4.626, 4.580, 4.557 fps (mean 4.588); node evaluation 209.60, 211.77, 220.08 ms (mean 213.82); display conversion 85.17, 85.55, 87.55 ms (mean 86.09). Each run has Lane 2 per-request stage marks in `scratch/nb-lanes/run/finish6-read-ahead-{on,off}-{1,2,3}.json`.

**Finding (measured):** disabling the pool costs about 38 ms in node evaluation and about 4 ms in display conversion, reducing the fresh rate around 20%. The latest 4.68 fps finish-5 result is not reproducible on current main with the pool active: the three on-runs average 5.50 fps, overlapping the prior 5.63 baseline within observed run spread. That older single run also recorded 18 pool hits but had no lane-stage timeline, so the cause of its low reading cannot be isolated. The evidence does not support read-ahead or threaded color ingest as the cause; there is no read-ahead code change to make from this A/B. Lane 2's timeline documents shared CPU/interpreter and memory-bandwidth cost from overlapping evaluation and display conversion (node evaluation 137 -> 150 ms; display conversion 40 -> 55 ms) while frame interval improves through overlap.

**Transform follow-up:** the largest fresh-frame stage is node evaluation. Its sampled frame-2 leaders were Transform 28 ms and Roto 26 ms. The horizontal fractional Transform made an unnecessary full-frame copy when its integer vertical rows needed no crop or padding; reusing the horizontal result cut the standalone 1080p median from 24.918 to 23.933 ms (3.95%), with exact repeated pixels. `tests.test_playback_profile_optimizations`, `tests.test_boundingbox` and `tests.test_tileexec` pass (76 tests). Evidence is in `scratch/nb-lanes/run/finish6-transform-{before,after}.log` and `tests-L8-finish6-transform.log`.

Part 1 A/B table is committed; part 2 queues the cold current-frame Reads before the first preview request without changing timer cadence. The playback-start and Read-ahead integration tests pass. No Viewer or Direct picture changes are included.

# Lane 8 step B finish 5 — PARTIAL (2026-10-22)

Read-ahead entries already carry working-space ACEScg pixels. The pool returned 18 prepared-frame hits in under 0.01 ms; two first-frame misses fell back to synchronous decoding and exceed the 5 ms per-Read target. A dedicated 1080p opaque-alpha conversion measurement improved from a 20.88 ms warmed median to 4.21 ms, with exact pixels. The owned EXR float32 buffer is reused and redundant alpha work is skipped. Raster.fit already returns the original pixel buffer for unchanged bounds; a regression now pins it.

Fresh uncached first-pass 1080p playback: RTX 4.68 fps, Radeon 5.585 fps, llvmpipe 5.643 fps, and four-core integrated laptop stand-in 4.226 fps. The 16/24 fps target and prior 4K 24 fps / one-second recovery gates remain open. JSON and logs are deliberate local scratch under `scratch/nb-lanes/run/`; no benchmark inputs were regenerated.

Pixel and targeted integration tests passed across the three implementation commits. Needs Gonzo: investigate first-frame misses with the playback scheduler owner and continue from `finish5-*.json`. Issue #8 and the lane board are Paused; project status is blocked pending follow-up.

# Lane 8 step B finish 4 — PARTIAL (2026-10-22)

On the 1080p RTX fresh-frame sample, frame 2 evaluation was 152.925 ms. Sampled nodes total 114 ms (Transform 30, Blur 24, Roto 22, Merge 18, Grade 16, ColorCorrect 4), leaving 38.925 ms outside those estimates. A warmed direct evaluator breakdown attributed 66.4 ms to two Read calls, including about 31 ms in input color conversion, plus 9.34 ms in ten Raster.fit calls. This separate direct run suggests Read/decode and format-fit work explain the missing evaluator time, but it is not the same app frame. Lane 2's trace commits were not on main at this finish.

Commit `8d0d5812` adds a pixel-exact cubic identity Transform fast path. On a 1920×1080 input the direct identity call fell from a 2.17 ms warmed median to about 0.002 ms by reusing the unchanged input. The animated playback comp is not identity-valued, so this does not improve its gate rate. The comp's Transform timing is consistent with earlier 21–22 ms measurements; the 30 ms stack estimate is too coarse to establish a regression. The attempted full-width sampling alternative was slower and discarded.

Latest comparable gates remain open: RTX fresh pass 5.627 fps; 16 fps intermediate and 24 fps target missed. The 4K run was 2.895 fps with 1,032 ms full-resolution recovery. Latest available stand-ins: llvmpipe fresh pass 5.215 fps; integrated Radeon 2.3 fps; laptop stand-in 3.422 fps. The 2D Blur, Transform, Roto, Merge and Grade kernels run on the CPU; existing GPU color conversion is display-side and outside those nodes.

Pixel regression and targeted node/core/tier/knob/properties checks passed. Benchmark and test logs remain local in workspace scratch; intended source change is committed on this lane. Needs Gonzo: continue from `L8-finish3-rtx-final-1022.json` after Lane 2's trace work is integrated and close the playback gates. Issue #8 and board are Paused.

# Lane 8 step B finish 3 — PARTIAL (2026-10-22)

Lane 2 owns scheduling, display conversion, handoff and frame overlap; this pass changed node kernels only. A fresh corrected uncached RTX 3080 Ti 1080p frame measured 5.627 fps first pass. Frame-2 sampled node costs: Transform 30 ms, Blur 24 ms, Roto 22 ms, Merge 18 ms, Grade 16 ms (2 ms/sample). Roto coverage storage changed float64 accumulators to float32; the short llvmpipe sample estimate moved 30→26 ms. Neutral Grade now returns an independent identity copy; its sample estimate moved 20→18 ms. The common unmasked Merge Over path reuses its output buffer; a five-run 1080p microbenchmark moved 10.671→8.020 ms. Each committed fix has a pixel regression. Required targeted tests passed: Roto/core/knobs/tiers/FluidPropertiesPanel 160; imaging/core/knobs/tiers/FluidPropertiesPanel 105 then 106. CPU profile and final RTX JSON/logs are local in `scratch/nb-lanes/run/`.

Blur remains about 24 ms in the fresh RTX frame. Float32 cumulative-sum and sliding-window alternatives were slower, so neither was kept. The 16 fps intermediate and 24 fps RTX 1080p gates, 4K 24 fps and sub-second still recovery remain open (last 4K result: 2.895 fps, 1,032 ms); A6000 remains unmeasured. Issue #8 and lane board are Paused. Needs Gonzo: continue node optimization from `L8-finish3-rtx-final-1022.json` and close the remaining playback gates.

# Lane 6 step A finish 2 — PARTIAL (2026-10-22, 6:33 PM PDT)

The bounded Radeon 64-cell cup-move re-bake is 28.34 s after the Hot pour steam cache was reduced to half-precision density+temperature; the paired full-channel measurement was 29.06 s. Three cancellation trials pass under 50 ms on both RTX 3080 Ti (21.443 ms worst) and Radeon 8060S (0.535 ms worst). Focused GPU, rigid, Hot pour, cache, surface and documentation checks pass. Real-display QA remains unverified in this headless session. The production-size bake is deferred to step B as directed; available space is 5.7 GB against an 8.50 GB historical cache. Evidence and handoff are in `TASKLOG.md` and `docs/FLUIDS_SPIKE.md`.

## Previous status — finish 1

Lane 6 step A of 2 was PARTIAL. Progressive timeline, stale state and cancellation signaling were committed; solver-cost reduction, Radeon cup-move gate, measured cancellation latency, display QA and production-size refresh remained open.

Complete-frame solver callbacks, Hot pour edit profiles, and artist-facing progressive timeline/stale state/cancellation wiring are committed. UI regressions plus CPU and llvmpipe simulation checks pass. First solved volume frames remain under 0.31 s on RTX and Radeon. The Radeon cup-move bake takes 32.43 s, 2.43 s over the 30 s target; no cost-reducing change is verified. A new production-size bake is deferred because 5.7 GB is free while the historical cache occupies about 8.5 GB. Real-display QA and measured <50 ms worker cancellation remain open; see `TASKLOG.md` and `docs/FLUIDS_SPIKE.md`.

# Lane 8 step A of 2 — in progress (2026-10-22)

Base is the newly merged live-picture plan. Building a generated animated 1080p/4K comp and a real-window benchmark with frame-stage profiling; initial GPU baseline is queued behind an existing RTX lock holder.

# Lane 8 step B finish 1 — DONE (2026-10-22, 1:30 PM PDT)

The three left-out drag gates are closed. Solid-over RotoPaint now limits raster/composite work to clipped stroke bounds and matches full-tile output byte-for-byte. The field driver moves the Properties slider horizontally on a visible control and reports each drag's picture count and maximum interval. On the RTX 3080 Ti, RotoPaint settles in 0.849 s; Properties produces 8 pictures and settles in 0.702 s. Maximum in-drag picture gaps are 0.182 s (Roto), 0.202 s (RotoPaint), 0.278 s (Tracker), 0.082 s (PlanarTracker), and 0.195 s (Properties). The scrub and playback thresholds documented below also pass. Focused CPU/UI/docs, playback-counter, and GPU lens-view tests pass. Raw field runs and test logs are retained under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`. No off-site hardware result is claimed. Needs Gonzo: review/integrate and perform real-display/Windows visual QA.

# Lane 6 finish 2 — PARTIAL (2026-10-22, 6:20 AM PDT)

Fresh frame 60/90/120 on the RTX 3080 Ti now show a continuous golden stream from spout to pool and a smoother denoised liquid surface. CPU/GPU comparisons pass. The 960 × 540 fixed-32 profile took 10.689 s. The fully converged 1920 × 1080 profile took 60.857 s (0.857 s over the strict limit). Steam remains grainy and the catcher shadow too crisp. The next candidate (smoke96, turbulence off, broader/fewer shadow samples) passed CPU tests but could not be rendered: the integration suite and Lane 2 GPU gates held the exclusive lock ahead of this work. Lane 6's queued render was canceled at the task time limit; no peer process was interrupted. Evidence in `benchmarks/hot_pour/`, fresh stills in `docs/images/hot_pour_shot/`, and detailed handoff in `TASKLOG.md`.

# Lane 6 finish 1 — local gates complete; external hardware QA unavailable (2026-10-22, 4:15 AM PDT)

RTX 3080 Ti: fixed 32-sample 960 × 540 frame 60 took 35.55 s. Adaptive 1920 × 1080 Render mode at threshold 0.01 converged every pixel in 45.46 s (adaptive cap 2,048; mean 11.71 samples/pixel). The showcase now disables steam cast shadows and uses four transmission bounces; 16–64 adaptive still renders took 14.95–15.68 s. CPU/GPU comparisons passed at frames 1/60/120 on NVIDIA and AMD Radeon 8060S; adaptive/majorant tests passed under NVIDIA and forced-integrated invocations. Visual review: golden stream reads as a pour; cup/pool still has speckle and the table shadow remains hard-edged. A6000 and Windows runs remain unverified. Evidence is in `benchmarks/hot_pour/hot-pour-render-profile-2026-10-22.json` and `benchmarks/hot_pour/hot-pour-shot-2026-10-22.json`; output logs in the lane run folder.

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
# Lane 6 step A finish 1 — active (2026-10-22, 5:42 PM PDT)

Scope is the prior PARTIAL's left-outs: artist-facing progressive timeline, stale status, edit cancellation, cost reduction and a fresh production-size bake. The `Window` now queues sequential non-display renders behind the visible edit preview; the timeline marks solved simulation frames separately, the old picture stays labeled stale until the current edit appears, and forward playback stops at the contiguous solved prefix. Queue/timeline/state tests pass. Fresh production-size cache needs ~8.5 GB based on the historical record, while the filesystem has 5.7 GB free; no such bake is running. Real-display QA and measured cancellation latency, cost reduction, and fresh production timing remain open. Worktree is `openclaw/nb-fluids-spike`.

# Lane 8 step B finish 2 — PARTIAL (2026-10-22)

The merged base is `39cc8a78`. Accepted-picture 1080p uncached first-pass results are 5.451 fps on the RTX 3080 Ti, 5.215 fps on llvmpipe and 3.422 fps on the four-CPU / 2×-scale laptop stand-in; the 16 fps and 12 fps gates remain open. The RTX's overall 7.3 fps includes cache repeats. The 4K RTX run presented tiers 1/2/4/8 at 2.895 fps and restored the full-resolution still in 1,032 ms; the 4K llvmpipe run reached 2.512 fps and restored in 1,417 ms. Both remain below the 24 fps / one-second gates. Fourteen proxy/tier tests, seven docs-copy tests and the benchmark self-check pass. The benchmark now requires Viewer generation acceptance before counting a display; the previous 7.277 fps first-pass estimate counted completed handoffs without that check. New 4K inputs occupy 175 MB under the lane run directory and will be removed after measurements.

## Lane 8 step B finish 6 — PARTIAL (2026-10-22)

RTX read-ahead A/B on the merged Lane 2 pipeline (three 10-second uncached 1080p runs each): on 5.680/5.382/5.449 fps (mean 5.504), off 4.626/4.580/4.557 (mean 4.588). The prior 4.68 fps finish-5 result is not reproduced with read-ahead on; it is close to the no-read-ahead mean. The available comparison does not identify the cause of that single-run discrepancy. With read-ahead on, RTX stage means were 175.83 ms node evaluation and 82.39 ms display conversion in the A/B set; the later final run measured 167.914 and 76.477 ms. The normal benchmark warms frame 1 before playback, so it does not establish end-to-end cold first-frame miss latency.

At play start, cold current-frame Reads are now submitted to the existing read-ahead path before the preview enters the playback queue; playback cadence and queue behavior are unchanged. A regression verifies call ordering. The final adapter 10-second uncached 1080p first-pass rates are RTX 5.701 fps, Radeon 5.680, llvmpipe 6.749, and four-core integrated stand-in 4.272. None reaches the 16 fps intermediate target. Node evaluation remains the largest Lane 2 stage (167.914 ms RTX final run); Transform was the largest sampled node, and its no-op vertical row fit now returns the original image. Its isolated fractional-horizontal microbenchmark improved 24.918→23.933 ms median (3.95%) with exact pixels. Focused tests: 59 prefetch/tile tests and 76 Transform/bounding-box/tile tests passed. RTX 4K and full-resolution stop-time gates were not rerun.

Evidence JSON/logs and microbenchmark logs are deliberate local scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/finish6-*`; they are not committed. Three implementation/measurement commits are local on `openclaw/nb-2d-parity-b`, not pushed. Step B remains partial; 1080p/4K targets and an end-to-end cold-start latency measurement remain open. Needs Gonzo: integrate/review commits `1b1df0bc`, `9bf0689b`, `8aac3a31`; prioritize remaining node-evaluation cost using the final timeline/profile JSON and coordinate the still-open playback gates. The one-off 4.68 fps comparison cannot distinguish run variance from a specific contention cause.
