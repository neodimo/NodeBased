# R6-3: bound GPU image memory with ROI, device tiles and safe storage reuse

**Proposed · rank 3 · 2026-10-23 · owner: assigned 2D lane · 5–8 additional engineer-days.**
Conditional on R6-1/R6-2 evidence showing memory pressure or substantial avoidable ROI work.
Read their reports, `docs/GPU_2D_PLAN.md`, `context/lanes.md`, current `context/state.md`
and dispatcher rules. Fast-forward the assigned clean worktree from origin/main. Work
directly without additional workers; no push or main edits. Coordinate viewer changes
through Gonzo. Do not start by rewriting the CPU tile scheduler.

## Goal

Make resident execution useful under the laptop's memory budget and on partial image
requests, with identical CPU pixels. Keep full-frame 1080p playback honest and unchanged.
This step does not introduce a new filtering model, approximate Blur or half precision.

## Deliver in order

1. Reuse `tiers.Region`, `TileRegion`, `resolve_halo`, `TileKey` semantics and existing
   CPU tile tests. Derive input requests for pointwise nodes, Merge and separable Blur;
   preserve data/display windows and negative origins. Blur halos use the exact rounded
   radius/edge behavior; tile boundaries must never become image-border clamp boundaries.
   Device cache keys include source digest, frame/tier, region/halo, format and device epoch.
   Keep unsupported graphs on the existing CPU full-frame fallback.
2. Add mapped-input ROI for validated Transform modes only, inverse-mapping output bounds
   and expanding by the actual reconstruction support. Test rotation, fractional and
   integer translation, scale/minification, center/pivot, masks, overscan and edge pixels.
   Admit Transform into device tiling only after full CPU/gpu tiled/untiled agreement.
   Unsupported filters/motion blur stay explicit fallbacks. Do not claim dispatch workgroup
   tiling alone bounds full-frame texture allocations.
3. Add a liveness-aware budget planner and completed-resource pool. Count every scratch,
   input/output, staging and readback resource; retire by last consumer and completion token.
   Enforce the plan before allocation. Use existing `gpumemory` instrumentation; identify
   driver/process memory separately. Prove budget-pressure behavior with forced low caps,
   branching graphs, large halos, overscan, A/B and repeated seeks. No allocation retry loop
   that blocks interaction; fall back once or return a precise error if CPU cannot render.
4. Batch compatible passes and reuse storage. Only fuse pointwise arithmetic where pixel
   ordering/masks are preserved. **Do not concatenate sequential transforms or collapse
   box filters**: those change the current CPU reference. A semantic optimization requires
   a separate authorized CPU change. Compare fused and unfused output and timings.
5. Commit correctness evidence and measurements under `benchmarks/2d-gpu/R6-3/`, and update
   PERFORMANCE in your lane's section with mirrored docs. Commit each tested part.

## Workload, quality and performance gates

Use R6-1's realistic fixture, plus 4K full-frame and 25%/10% requested ROI cases, awkward
non-tile-aligned regions, wide Blur and negative-origin EXRs. Reuse reproducible generators;
store large generated inputs only in dispatcher's disk scratch and clean them afterward.
Compare CPU full-frame, CPU tiled, GPU full-frame and GPU tiled **at the same requested
region and quality**. Require every finite pixel within `2e-5 + 2e-5*abs(cpu)` or stricter
existing tests, with no seams and correct metadata/layers/fallback behavior.

- No retained-memory growth across 100 seek/edit/play cycles after warmup. Device allocation
  peak stays under the configured cap including in-flight resources; report counts/bytes.
- At 25% ROI, require at least **2× reduction in measured GPU processing time** versus
  full-frame for the nominated pointwise-heavy case; record halo/transfer overhead. If
  the workload does not benefit, report that and do not globally enable tiling.
- Full-frame 1080p accepted-picture median/p95 must not regress more than **5%** versus
  R6-2 on paired runs; retain **24 fps / p95 ≤41.67 ms** wherever it already passed.
  ROI/proxy playback never counts as passing that full-frame target.
- Report first-result and full-resolution settle separately at 4K. No universal 4K 24 fps
  claim is required; provide measured bottleneck and working-set evidence.

Run on **RTX 3080 Ti 12 GB** (1 GiB base, at most 2 GiB after pressure proof),
**A6000 48 GB** (same cap for comparison, optional 4 GiB capacity row), and **actual Iris Xe
+ RTX 4070 laptop** (512 MiB initial, at most 1 GiB after proof). Include real internal-display
presentation and adapter/backend/power identity. Repeat pixel/memory tests on **Radeon 8060S**
(512 MiB), **llvmpipe and Windows software D3D12** (128 MiB small correctness cases, CPU default).
Unavailable tiers stay unverified/PARTIAL with field-kit command and Gonzo owner. No inference
of speed from card capacity or stand-ins. Protect optional device features and numeric packing.

Run new tests and affected `tests.test_tiles`, `tests.test_tileexec`, `tests.test_tile_layers`,
`tests.test_tile_cancel_is_a_cancel`, `tests.test_transform_resample_perf`,
`tests.test_gpumemory`, `tests.test_playback_pipeline` and `tests.test_knowledge` before docs
commits. Project venv, worktree PYTHONPATH, offscreen Qt for tests; acquire `/tmp/nb-gpu.lock`
for GPU work and use existing per-adapter runner. Keep test summaries small; no full suite.

Before reporting done, write your completion note to `TASKLOG.md` and `context/state.md`
in the assigned worktree, the lane-local equivalent of `projects/nodebased/TASKLOG.md`:
what was done, evidence versus inference, exact artifacts and committed/scratch status,
done/partial/blocked and unverified gates, next owner plus concrete command, failure mode
if relevant. Newest TASKLOG entry first. Update assigned issue/card and any tracked task
through the provided status script in the same closeout; never write main's notes.
Needs Gonzo: review whether measured ROI/memory gains justify enabling the path, integrate,
and route outstanding hardware checks. Leave general transform concatenation out.
