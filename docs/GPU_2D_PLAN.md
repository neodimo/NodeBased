# GPU-resident 2D evaluation: NodeBased fit and decision gates

**2026-10-23 · Lane 3, R6 part 2.** Read-only code assessment at main snapshot
`2e0a2b9d`, fast-forwarded into `openclaw/nb-research` before work. No GPU work executed.
Production precedents and dated sources: [GPU compositing survey](GPU_2D_COMPOSITING.md).
All effort/gain ranges below are engineering estimates, not local results.

## Baseline: distinguish the brief from the committed measurements

The assignment rounds the remaining CPU nodes to **170 ms/frame**, with approximately
**5.7 fresh fps**. This is the decision baseline, not a new measurement. On inspected main,
[PERFORMANCE](PERFORMANCE.md), “Lane 8: realistic comp playback evaluation,” records
stack-sampled after-cost estimates at 1080p: Transform **67.7**, Grade **27.6**, Merge
**18.8**, Blur **18.8**, Roto **11.0 ms**. These are samples from an earlier run, not
independently timed kernels that can be added to a later trace without qualification.

The newer “Where a fresh frame's time goes” table records **137.0 → 150.1 ms** evaluation,
**5.32 → 6.50 fresh fps** on RTX 3080 Ti with pipelining; **4.18 fps** on the four-CPU,
2×-scale Radeon laptop stand-in; Direct picture on Xvfb **4.82 fps**. Display and node
work overlap, so summing all stage durations overstates the throughput interval. Conversely,
GPU kernel time alone understates time to an accepted displayed frame.

Evidence is already committed in `benchmarks/2d-playback/pipeline-1022-*.timeline.json`.
A6000 and the actual Windows Iris Xe + RTX 4070 laptop remain unmeasured. Rebaseline the
exact current fixture and revision in the spike; never present stand-in numbers as laptop
numbers. The target remains **24 fresh fps / 41.67 ms interval at 1080p**, with no reduced
resolution hidden in that claim. The existing intermediate gates are 16 fps RTX / 12 fps
laptop, and must be labelled intermediate.

## Read map: existing contracts and proposed attachment points

| Current code read | Existing contract | Proposed use; work not implemented here |
| --- | --- | --- |
| `nodebased/imaging.py`, `Evaluator.evaluate`, `evaluate_raster` | `evaluate` returns the display-window ndarray; `evaluate_raster` retains data/display windows, frame/tier, hashes and cache behavior. Single-owner retained evaluator. | Keep these public CPU return contracts. Add an opt-in resident result entry point or explicit internal island executor; materialize only at a CPU consumer or public ndarray boundary. |
| `nodebased/raster.py`, `Raster` | H×W×4 float32 pixels, `data`, `display`, named `layers`, `meta`; constructor calls `np.asarray`. | A separate `DeviceRaster` sibling/internal artifact, never a texture secretly passed into `Raster`. Carry the same windows/metadata, representation version and CPU materialization adapter. |
| `nodebased/tileexec.py`, `TileExecutor.supports_tiled`, compose/evaluate path; `tiles.py`, `tiers.py` | Tile artifacts and hashes contain tier, requested region and halo. Unsupported ancestors (including ordinary Transform) take full-frame evaluation. Named outputs have restrictions. | Phase 1 uses explicit full-frame islands and existing CPU tile fallback. Later device tiles reuse `Region`, halo rules and cache keys, with mapped input ROI for transforms. Do not mark Transform tiled merely because its kernel dispatch uses workgroups. |
| `nodebased/decodepool.py`, `DecodeAheadPool`; evaluator `decode_pool` | Bounded immutable decoded CPU Raster cache, epoch-discard on seek, one default worker; decode plus colour ingest before graph processing. | Keep OIIO/OCIO ingest unchanged. Upload a decoded frame once per device/content/tier key, bounded separately from RAM cache. Decode readiness is not GPU completion; do not call evaluator from decode threads. |
| `nodebased/directviewer.py`, `RetainedPresenter`, `FrameLedger`, `StagingRing`, `pick_display_adapter` | CPU picture/linear array upload; presenter-owned, surface-compatible device; newest generation and device epoch; three staging slots; bounded loss recovery. | A later viewer-owner change can accept a same-device texture lease. Different devices require an explicit, measured bridge or CPU presentation. A texture handle from another wgpu device is invalid, even on the same adapter. |
| `nodebased/carddisplay.py` | wgpu OCIO shader matches `to_qimage` ordering, premultiply handling and A/B comparison. Unsupported LUT forms fall back to CPU display bytes. | Reuse shader/pipeline construction for final resident colour conversion where compatible; preserve unsupported-view fallback and display tests. |
| `nodebased/gpudisplay.py` | Existing OpenGL display conversion/context lifecycle, separate from wgpu. | Do not confuse this with same-device wgpu presentation or assume OpenGL/wgpu resource sharing. |
| `nodebased/gpupicture.py`, `Recorder`, `register`, `take` | Played 3D frames get an 8-bit display pass in the render submission, then readback registered against the returned CPU float array. | Reuse the ordering/colour precedent, not the registry as a resident-frame API: it still reads back pictures and float data. |
| `nodebased/gpumemory.py`, `MemoryTracker`, `instrument` | App-owned allocation byte accounting; wgpu report does not expose total adapter memory in bytes. | Track 2D textures, buffers, staging and readback, distinguish app tally from process/driver peak; enforce caps before allocation. |
| `nodebased/playback_trace.py`, `tools/benchmark_2d_playback.py`, `tools/timeline_2d_playback.py` | Request/worker/conversion/accepted-picture stage marks and fresh-frame accounting. | Add upload/compute/download/CPU-boundary counters without removing existing timestamps or charging pipeline overlap twice. |

### Concrete island for the supplied realistic comp

`tests/fixtures/playback_comp.nbcomp` contains this topology (parameter animation remains
in the fixture):

```text
Read plate (CPU decode) -> Grade -> Transform -> Blur --+
                                                      Merge -> ColorCorrect -> viewer
Read background (CPU decode) -------------------------+
Roto (CPU matte, separate input branch) ---------------+ mask
```

Grade/Transform/Blur/Merge form the first resident island. Upload plate, background and
Roto matte; retain the intermediate images on the same device; read back the island's
output once for the CPU ColorCorrect. Roto is a side input, so it need not force a readback
between Blur and Merge. This distinction materially changes the expected transfer cost.

The first spike also measures a predecoded micro-pipeline through the same four kernels,
but a microbenchmark pass is not a realistic-comp pass. Keep Roto and ColorCorrect in the
end-to-end measurement. A later ColorCorrect implementation can remove the output CPU
boundary if its pixels pass; do not silently delete or neutralize it to hit the gate.

### Proposed resident image and lifecycle

A `DeviceRaster` holds a texture or storage buffer, device identity/epoch, completion token,
width/height/row layout, channel/format, data/display windows, colour/alpha conventions,
named-layer policy, source digest, frame/tier and generation. It owns a counted resource
lease; consumers borrow it. No implicit `__array__` readback. Explicit `to_cpu()` waits once,
records transfer bytes/time and produces the normal Raster. Initially reject unsupported
layers/metadata behavior from the island and evaluate them on CPU intact.

The evaluator continues to own graph traversal and semantic cache keys. Partition only
validated nodes/parameters into maximal connected islands. Capability checks include
format, device limits, window size, mask/mix, operation mode and supported filter. Side
branches upload only when content changes. A fork retains its image until the final
consumer and all submitted work finish. Cancelling a generation prevents presentation;
it does not permit freeing buffers still referenced by GPU commands. Device loss invalidates
all leases in that epoch, releases completed work safely, and retries the whole affected
frame on CPU from retained CPU inputs. Do not swallow unrelated graph errors as GPU fallback.

No unbounded future-frame GPU cache. At most two evaluating/submitted frames plus the
currently displayed lease initially; the exact budget may force fewer. Cache shader modules,
pipelines and immutable parameter layouts per device. Submission should encompass the
island rather than wait after every node. Reuse storage only after its last read and GPU
completion. A first implementation can keep separate passes; shader fusion is optional.

## Pixel truth: accelerate the existing operations

- **Grade:** `_grade` applies exposure and multiply to premultiplied RGB, then adds
  `alpha * offset`. Preserve alpha, mask/mix and operation ordering. Do not replace it
  with an unpremultiplied textbook grade or clamp HDR/negative values.
- **Transform:** `_transform`, `_resample`, `_resample_cubic_separable` use existing
  coordinate/filter/border behavior. Cubic is Catmull–Rom, not hardware bilinear. Preserve
  fixed/preserved bounds, center/pivot, nonuniform scaling, clamp, masks and motion-blur
  fallback. Integer-translation optimization must stay exactly aligned.
- **Blur:** `_blur` is separable **box blur**, not Gaussian. `_box_blur_axis` rounds radius,
  uses edge padding, float32 shifted accumulation for small radii and float64 cumulative
  sums for large radii. A WGSL float32 prefix sum can lose precision; initially port the
  tested small-radius fixture path, fall back explicitly outside the validated range,
  and separately test large radii. Do not claim broad Blur coverage from radius-two tests.
- **Merge:** match `_merge_gated`/`_merge_op`, actual A/B convention, premultiplied algebra,
  optional/missing inputs, mask, mix, alpha and window alignment. Float32 compute or shader
  arithmetic avoids requiring float32 fixed-function blending support.
- **ColorCorrect:** remains CPU for the first proposal. Later port `_color_correct`,
  especially tonal ranges and negative/HDR behavior, against its dedicated tests.

Keep `tests.test_imaging`, `tests.test_tiles`, `tests.test_tileexec`,
`tests.test_playback_profile_optimizations`, `tests.test_transform_resample_perf`,
`tests.test_colorcorrect_nuke`, `tests.test_2d_parity_group_3a_merge`,
`tests.test_2d_parity_aa1_merge` as semantic references as relevant to the actual changes.
The existing golden files include transform filters, Grade negative/HDR and Merge modes.
New GPU tests must compare to these CPU operations, plus combined fixture frames, not
only to another shader implementation. Test random and analytic images, transparent edges,
negative/HDR pixels, odd sizes, border impulses, negative data origins, masks and fractional
frames. CPU reference tests must also run when there is no graphics device.

Proposed initial float32 gate: every finite fixture pixel satisfies
`abs(gpu - cpu) <= 2e-5 + 2e-5 * abs(cpu)`; report maximum, mean and p99 error separately,
and preserve any stricter existing test. This is a proposed acceptance criterion, not an
observed error. Overflow/NaN behavior must agree or explicitly trigger CPU fallback; never
omit bad pixels from the score. A failing algorithm is fixed or left on CPU, not silently
granted a looser tolerance. Viewer display's existing ~1/255 agreement is a separate gate.
Do not import the 3D half-float splat tolerance multiplier into 2D tests.

Start storage and arithmetic at float32, using `textureLoad`/manual taps or float buffers.
Do not require float32 filtering, float32 blending or float16 arithmetic merely to run the
four kernels. Existing Direct viewer downconverts some large display uploads to half;
that display policy is not permission to halve compositing precision or change exports.

## ROI, transform concatenation and filter fusion: separate stages

For pointwise kernels, output ROI maps directly to input ROI. Merge intersects/aligns the
requested region with each input data window; Blur expands it by its exact kernel support.
Transform inverse-maps output bounds and expands by reconstruction support; rotation,
negative origins and minification need dedicated tests. The existing CPU tile path remains
the oracle. Boundaries between tiles must not become clamp edges of the original image.

Fusion's transform concatenation can improve quality and speed by resampling once, but
NodeBased's CPU reference currently evaluates nodes sequentially. Concatenating two cubic
transforms changes those pixels even when the matrices compose exactly. Do not introduce
that semantic change in a parity spike. Likewise two box filters are not one wider box.
Safe future work includes command batching, lifetime reuse and fusing pointwise expressions
with matching order/mask semantics; floating-point reassociation still needs testing.
General transform concatenation requires an independently approved CPU semantic change.

## Budget and adapter policy (proposed, to be measured)

| Tier | Initial 2D allocation cap, including transfers | Performance/compatibility stance |
| --- | --- | --- |
| RTX 3080 Ti, 12 GB | 1 GiB for spike; later at most 2 GiB after pressure tests | Main 1080p gate; USB4 transfer evidence in Direct viewer makes residency particularly valuable. |
| RTX A6000, 48 GB | Same 1 GiB spike for fair comparison; later up to 4 GiB opt-in cache | Same 24 fps goal. Capacity is larger; compute and transfer speeds are not inferred from VRAM. |
| RTX 4070 mobile + Iris Xe | Start 512 MiB, permit 1 GiB after actual adapter/VRAM check | Laptop is priority acceptance hardware. Measure render/present devices, backend, power mode and scanout route. CPU bridge may cost the gain. |
| Radeon 8060S integrated | 512 MiB initial cap and shared-RAM accounting | Required correctness and measured performance; shared bandwidth competes with decode/CPU nodes. Four-core stand-in remains explicitly a stand-in. |
| llvmpipe / Windows software D3D12 | CPU evaluator default; 128 MiB bounded explicit GPU-kernel test mode | Portability/correctness gates, no hardware-acceleration claim or 24 fps requirement. |

Caps are conservative engineering choices, not detected free VRAM. Overscan, layers, A/B
viewer, blur scratch, staging and retained CPU copies all count. Query actual per-resource
limits before allocation; OOM fallback must not depend on a full device reset. An oversized
frame can stay on CPU until device tiling is validated. Reserve headroom for 3D/ML/desktop;
never claim app-owned counts equal total adapter occupancy.

### Windows release evidence and concrete risk

The brief's v0.42.0 failure is verified by [package run 39937770296](https://github.com/neodimo/NodeBased/actions/runs/39937770296),
failed on 2026-10-23. Its Windows logs identify **Microsoft Basic Render Driver, CPU type,
D3D12, wgpu 0.32.0**. Direct-viewer minification errors enter `_start_device` at
`request_device_sync`; fluid-first-look also hits an explicitly NVIDIA-only path. These
are distinct failures; do not attribute them all to shader precision. The same run also
has an Ubuntu first-look assertion failure. This research does not fix any of them.

At sweep time [v0.42.1 package run 39961043927](https://github.com/neodimo/NodeBased/actions/runs/39961043927)
and its conformance run report success. Therefore the old package failure is portability
history, not a claim that the latest inspected package is still red. A green package does
not verify the proposed new GPU island or the actual hybrid laptop.

Risk controls: request only supported limits/features; test unavailable device, invalid
texture size, shader failure and loss recovery; separate CPU fallback from software-GPU
execution; validate D3D12 as well as Linux Vulkan. Do not skip all CPU correctness tests
because one GPU feature is missing. A software-adapter pass cannot substitute for laptop
presentation, and a 3080 Ti pass cannot substitute for AMD shader correctness.

## Expected gain and effort: a model with explicit assumptions

The sampled four operations total **132.9 ms**. Against the brief's 170 ms node envelope,
that leaves **37.1 ms** on CPU before transfers. If the resident four take **8–20 ms** and
boundary transfers/synchronization **3–12 ms**, the modeled node path is **48.1–69.1 ms**,
a **2.46–3.53×** node-stage speedup (about **14–21 fps** as a ceiling before other critical
work). These are hypotheses borrowed from parallelism/data movement reasoning, not vendor
or local timings. If 37.1 ms really remains, infinitely fast four-node kernels still leave
only 4.57 ms of a 24 fps budget for everything else. Thus a four-node win alone is unlikely
to finish the playback goal; ColorCorrect, read/decode readiness, Roto and final presentation
must be reprofiled. Do not mix this illustrative model with the newer 150.1 ms trace as if
they were one measurement.

| Ranked work | Engineer-days, including targeted tests | Expected benefit / stop condition |
| --- | --- | --- |
| 1. Measured resident four-node spike | **4–6 days** | Test the modeled 2.5–3.5× node gain. Stop expansion if pixels fail, transfers erase gain, or real fixture fails the intermediate frame gate. |
| 2. Integrate resident result, CPU islands and final viewer; add ColorCorrect | **6–10 additional days** | Remove avoidable full-frame readback/reupload. Hypothesis: total node path 15–35 ms on discrete hardware if most remaining CPU pixel work moves; 24 fps still requires display/decode budget. |
| 3. Budgeted device ROI/tiles and safe pass reuse | **5–8 additional days** | At 25% requested area, pointwise work may approach a fourfold reduction, limited by halos, dispatch and transfers. Full-frame 1080p may gain little; no 24 fps claim from cropped playback. |

Total staged estimate **15–24 engineer-days**, excluding waiting for hardware and integration
queue. Broad node parity, transform semantic redesign, arbitrary OpenFX hosting, all temporal
nodes and video GPU decode are outside this estimate. A failing spike can still deliver
useful evidence; it must report PARTIAL against unmet performance or hardware gates.

## Ranked recommendation (only these three)

1. [R6-1: measured four-node resident spike](../briefs/proposed/R6-1-resident-four-node-spike.md).
2. [R6-2: resident evaluator and viewer boundary](../briefs/proposed/R6-2-resident-evaluator-and-viewer.md), conditional on the spike.
3. [R6-3: budgeted ROI and device tiles](../briefs/proposed/R6-3-budgeted-roi-and-tiles.md), conditional on actual memory/ROI evidence.

Gonzo owns assignment and integration. A 2D lane owns the kernels/evaluator work; changes
to Direct presenter/app wiring require the viewer-owning lane's coordination. None of these
proposals is started by this research. A6000/laptop unavailable means unverified, never an
estimated pass. Preserve CPU behavior, then earn the laptop speed claim with accepted
fresh pictures on the real machine.
