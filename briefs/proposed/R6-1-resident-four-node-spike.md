# R6-1: measure the realistic comp with Transform, Blur, Merge and Grade resident on the GPU

**Proposed · rank 1 · 2026-10-23 · owner: assigned 2D lane · estimate: 4–6 engineer-days.**
This is a bounded implementation-and-measurement spike, not approval for a graph rewrite.
Read `docs/GPU_2D_PLAN.md`, `docs/GPU_2D_COMPOSITING.md`, `context/lanes.md`, the top of
`context/state.md`, and the dispatcher's standing lane brief. Use the assigned lane's
worktree/branch only; fast-forward from origin/main first with no local changes. Work
directly without additional workers. Do not change main or push. Ask Gonzo to route any
viewer-owned wiring change; this spike can return one CPU Raster at its final boundary.

## Goal and fixed workload

Keep the four native operations resident on one device while matching CPU pixels, and
measure whether this meaningfully closes the **170 ms node-stage / 24 fps 1080p** gap.
Use `tests/fixtures/playback_comp.nbcomp` unchanged in topology and animated parameters:
Read → Grade → Transform → Blur → Merge → ColorCorrect, with separate Read background and
Roto mask inputs. CPU ColorCorrect and Roto remain in the end-to-end timing. No identity
replacement, lower-resolution frame, cached replay or missing mask counts as success.

## Deliver in order, commit each tested part

1. **Baseline/harness.** Read the current `tools/benchmark_2d_playback.py` and existing
   `benchmarks/2d-playback/pipeline-1022-*.timeline.json`. Generate the deterministic EXR
   sources with its `--generate-inputs` option in the dispatcher's large disk scratch
   (`NB_RUN`, not RAM-backed temporary storage). Record source-generation command, fixture
   hash, revision, environment/backend, hardware and driver. Measure baseline on this
   revision; retain the existing 48-frame comparison and add a 240-distinct-frame fixture
   range for the sustained gate. Extend only the range using the fixture's existing curve
   extrapolation; record that variant and its hash. Output/source caches must not turn a
   repeated loop into “fresh playback.” Decode-ahead may remain enabled and is timed.
2. **Opt-in island.** Add a bounded GPU implementation and explicit resident artifact in
   new 2D modules (suggested `nodebased/gpu2d.py`). Use float32 storage/arithmetic and the
   existing CPU formulas. Upload plate/background/Roto inputs, execute Grade, Transform,
   Blur and Merge on the same device with **zero intermediate host readbacks**, then
   materialize once for CPU ColorCorrect. Pipeline/resource creation is cached per device.
   Expose an opt-in backend to the benchmark; keep CPU as default during the spike. Preserve
   source digests, frame/tier, masks/mix, windows and cancellation. Unsupported modes use
   a reported CPU boundary; the nominated fixture must execute all four on GPU.
3. **Pixel and failure gates.** Add `tests/test_gpu2d.py` plus CPU-only capability/fallback
   tests. Use `_grade`, `_transform`/`_resample`, `_blur`, `_merge_gated` as truth. Blur is
   separable edge-padded box blur; its wide-radius CPU path accumulates in float64. Port
   validated radii first and explicitly fall back outside them. Cubic Transform is
   Catmull–Rom; hardware bilinear is not a replacement. Test negative/HDR values, alpha 0/1,
   masks and mix 0/partial/1, odd sizes, border impulses, fractional translations, rotation,
   nonuniform scale, pivot, negative data origins, all supported Merge modes and combined
   fixture frames. Preserve stricter existing tolerances; require every finite fixture
   channel `abs(gpu-cpu) <= 2e-5 + 2e-5*abs(cpu)`, report max/mean/p99 error. Nonfinite
   results must match CPU behavior or force explicit fallback. No tolerance widening to
   pass a bad kernel. Verify no-device, unsupported-feature/parameter, over-budget,
   cancellation, device-loss and stale-generation behavior without GPU hardware too.
4. **Measure and decide.** Add `tools/benchmark_gpu2d.py` (new, not an existing command)
   with CPU/per-node-roundtrip/resident modes, the same inputs and JSON output. Measure
   predecoded kernels separately from full graph and accepted display. Count texture/buffer
   allocations, live/peak bytes, upload/download bytes, submissions, CPU boundaries, shader
   compile/cold-start, CPU reference cost and GPU completion. Do not use enqueue time as
   compute time. Use GPU timestamps when supported, otherwise bounded completion timing
   with the method labelled. Integrate counters with `playback_trace`; keep its accepted
   generation/frame stamps and account for overlap. Commit compact measurements under
   `benchmarks/2d-gpu/R6-1/`, reproducible commands and results under your own lane's section
   in `docs/PERFORMANCE.md`, with bundled copy byte-identical.

## Acceptance gates, fixed before optimizing

Run three independent fresh passes per measured hardware tier after separate pipeline warmup
on sacrificial frames. Report cold shader setup separately. For the common 48-frame run,
report every accepted fresh frame; for the sustained run use all 240 unique frames and
finish the pass, not a timeout that selects only easy frames. Disable persistent/output
replay caches; static unchanged inputs and shaders may be reused. Verify displayed content
and generation, no drops/duplicates counted as progress, and no end-of-run drain counted
inside a timed window. Report total duration, mean interval, p50/p95/p99 and first-picture
latency. Use the same cache policy for CPU and GPU. Preserve raw per-frame stage records.

**Spike go/no-go (RTX 3080 Ti and actual RTX 4070 laptop separately):**

- Pixel gates pass, zero readbacks between the four kernels, allocations stay within cap.
- Four-node resident execution **p95 ≤20 ms**; uploads + resident execution + single
  terminal readback **p95 ≤30 ms**, including completion rather than dispatch only.
- Realistic full graph node-stage median improves **at least 2×** versus its same-machine
  CPU baseline. Full-resolution accepted fresh playback reaches **16 fps RTX 3080 Ti** and
  **12 fps laptop** (intermediate gates), with p95 intervals ≤62.5 / ≤83.33 ms respectively.
- Final product target remains **24 fps / 41.67 ms** on both, with p95 interval ≤41.67 ms.
  Record this separately as pass/fail; an intermediate spike pass is not “24 fps shipped.”

If pixels or transfers fail, stop expansion and deliver the measured reason. If the kernel
wins but the whole comp misses, identify the measured remaining CPU/transfer stage and mark
the unmet gate; do not remove ColorCorrect or Roto. A defensible negative result is useful.

## Hardware, memory and portability

- **RTX 3080 Ti 12 GB:** primary measured tier, 1 GiB initial total 2D device allocation cap.
- **A6000 48 GB:** repeat identical workload with the same 1 GiB cap; same kernel gates and
  16 fps intermediate / 24 fps final target. Larger VRAM is capacity, not a timing result.
- **Iris Xe + RTX 4070 Windows laptop:** priority field tier. Start 512 MiB cap, allow 1 GiB
  only after logging available adapter and actual VRAM. Record backend, render/present
  devices, internal/external display route, resolution/scaling and power state. Include
  actual accepted pictures on the screen. No Radeon/Xvfb stand-in can pass this row.
- **Radeon 8060S:** required pixel/failure tests and measured full comp, initial 512 MiB cap;
  separate integrated workstation and four-CPU/2×-scale stand-in results.
- **llvmpipe and Windows Microsoft Basic Render Driver/D3D12:** explicit kernel correctness
  mode with 128 MiB cap/small fixtures; ordinary CPU fallback must work. No speed gate.
  Software wgpu execution and the numpy CPU evaluator are separately identified.

Avoid optional float32 blending/filtering and fp16 requirements; use compute/manual sampling.
Do not pack integer metadata as float bit patterns. Query limits before requesting devices.
The v0.42.0 package failure and successful v0.42.1 follow-up are documented in the plan;
neither proves these new shaders portable. After WGSL/buffer edits, run new GPU tests on
NVIDIA, Radeon and llvmpipe via the dispatcher's adapter runner, and on Windows D3D12.
Unavailable A6000/laptop/Windows results remain unverified; report PARTIAL for missing
required hardware gates and name Gonzo as the owner of the field-kit command/artifacts.

## Commands, verification and closeout

Use the project venv as `PY`, `PYTHONPATH` set to this worktree, `QT_QPA_PLATFORM=offscreen`
for tests. `NB_RUN` is the dispatcher's explicit disk scratch path. Example existing
baseline commands (adapter selection and real display supplied by the dispatcher):

```sh
"$PY" tools/benchmark_2d_playback.py --generate-inputs "$NB_RUN/plates" --size 1080p --frames 48
"$PY" tools/benchmark_2d_playback.py --plate-pattern "$NB_RUN/plates/plate.%04d.exr" --background-pattern "$NB_RUN/plates/background.%04d.exr" --size 1080p --frames 48 --cache-mode uncached --seconds 10 --output "$NB_RUN/cpu.json" --timeline-output "$NB_RUN/cpu.timeline.json"
```

Check the script's current CLI first and capture actual commands, not imagined flags.
GPU runs acquire the shared `/tmp/nb-gpu.lock`; do CPU work while it is held, then follow
the dispatcher's bounded lock wait. Never kill another process or occupy the whole GPU.
Do not run the full suite. Run new tests plus affected references:
`tests.test_imaging`, `tests.test_tiles`, `tests.test_tileexec`,
`tests.test_playback_profile_optimizations`, `tests.test_transform_resample_perf`,
`tests.test_2d_parity_aa1_merge`, `tests.test_playback_pipeline`, and `tests.test_knowledge`
before docs commits. Do not rerun unrelated 3D/ML suites. Summarize test logs, preserve
compact evidence, and remove generated source/cache bytes after retaining their generator
and measurements. No generated EXR sequence belongs in git.

Before reporting done, write the completion note to `TASKLOG.md` and `context/state.md`
in the assigned worktree (the lane-local equivalent of `projects/nodebased/TASKLOG.md`):
what was done, evidence versus inference, exact artifacts and committed/scratch status,
done/partial/blocked and unverified gates, next owner plus concrete command, and failure
mode if correcting one. Newest TASKLOG entry goes first. Update assigned issue/card and
any tracked project task through the provided status script in the same closeout; never
edit main's notes. Needs Gonzo: review the measured go/no-go and route missing field tiers.
