# v0.35 portable final-render benchmark

Lane 4, step P1, October 7 to 8, 2026. Baseline for the harness: `6807d16`, after Rendering 8; harness commit `abcce2e`; measured on branch commit `7bae83e` (same renderer code). All three adapters ran the full matrix at 640 by 360: AMD Radeon 8060S (RADV, Mesa 26.2.2), llvmpipe, and the NVIDIA RTX 3080 Ti (after the host reboot, one bounded `Z1-fixed64` case first, then the other seven). No kernel watchdog error was seen in the case outputs; the kernel log itself was not readable from the worker, so a quiet kernel log is unverified. The raw per-case JSON is in `benchmarks/portable_render/`.

## Method and exact rerun

`tools/benchmark_portable_render.py` uses X1, Y1 and Z1 from `tools/benchmark_adaptive.py`: each at fixed 64 samples and adaptive threshold 0.003. The sparse smoke plume comes from `tools/benchmark_volume_majorant.py`, at fixed 32 samples with the two-level grid and the single-box control. All cases use a fixed seed and a CPU-path-tracer image reference with the same settings. Quality is PSNR and mean/maximum absolute error on clipped, sRGB-encoded RGB. One warm-up and three timed renders produce a median wall time. Per-case JSON includes the adapter and backend, settings, seed, resolution, renderer phases and sample counts; smoke includes collision counts. CPU references are prepared before taking the GPU lock. Each GPU case is its own bounded subprocess and lock turn, saving JSON immediately. Timeouts and unavailable adapters are explicit records.

From the lane worktree, with the project virtualenv active, after any other lane releases the GPU lock. First validate one RTX case in a short turn; only include it in the full matrix if that case succeeds without a new watchdog error:

```sh
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen python tools/benchmark_portable_render.py --matrix --adapters discrete --cases Z1-fixed64 --size 640x360 --out /tmp/nodebased-portable-p1 --timeout 120 --lock-wait 60
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen python tools/benchmark_portable_render.py --matrix --adapters integrated,cpu,discrete --size 640x360 --out /tmp/nodebased-portable-p1 --timeout 900 --lock-wait 60
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen python tools/benchmark_portable_render.py --report /tmp/nodebased-portable-p1
```

`--force` reruns already recorded cases, including unavailable or timed-out ones. Keep RTX to one bounded validation case first, and stop on a renewed watchdog error. The default matrix excludes RTX until it has passed that check.

## Measurements and next recommendation

The six X1, Y1 and Z1 case pairs were re-measured on October 8 in step Q1 (renderer commit `a6c04e2`) and the two X1 cases on October 9 in step R1 (renderer commit `6007642`); the smoke rows were re-measured in step Q2 (renderer commit `c94b4eb`; the P1 smoke times are in the Q2 section). Median of three timed renders after one warm-up. "Passes" is the number of render passes the adaptive loop or fixed loop ran; "mean spp" is the mean samples per pixel. PSNR and max error compare each adapter with the CPU path tracer at the same settings and seed (clipped sRGB RGB). Only the smoke grid cases (74 to 75 dB) and the RTX X1 adaptive row (90.1 dB, one pixel; see step R1) carry noticeable GPU-to-CPU differences; everything else is above 97 dB. Before step R1, X1 read 54 to 55 dB on all three adapters (max error 0.014 to 0.064) because the CPU path tracer mishandled the occlusion map, and the P1 and Q1 sections below quote those older X1 numbers as measured then.

| adapter | backend | case | status | median | passes | mean spp | PSNR vs CPU | max error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| llvmpipe | Vulkan | X1-adaptive0.003 | ok | 263 ms | 7 | 19.0 | 105.3 dB | 0.001 |
| llvmpipe | Vulkan | X1-fixed64 | ok | 619 ms | 64 | 64.0 | 109.4 dB | 0.000 |
| llvmpipe | Vulkan | Y1-adaptive0.003 | ok | 763 ms | 7 | 38.0 | 104.3 dB | 0.002 |
| llvmpipe | Vulkan | Y1-fixed64 | ok | 1251 ms | 64 | 64.0 | 104.6 dB | 0.002 |
| llvmpipe | Vulkan | Z1-adaptive0.003 | ok | 147 ms | 7 | 18.3 | 159.2 dB | 0.000 |
| llvmpipe | Vulkan | Z1-fixed64 | ok | 374 ms | 64 | 64.0 | 122.6 dB | 0.000 |
| llvmpipe | Vulkan | smoke-box | ok | 1025 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| llvmpipe | Vulkan | smoke-grid | ok | 367 ms | 32 | 32.0 | 75.2 dB | 0.053 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | X1-adaptive0.003 | ok | 43 ms | 7 | 19.0 | 90.1 dB | 0.022 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | X1-fixed64 | ok | 87 ms | 64 | 64.0 | 110.3 dB | 0.001 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Y1-adaptive0.003 | ok | 53 ms | 7 | 38.0 | 97.4 dB | 0.006 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Y1-fixed64 | ok | 90 ms | 64 | 64.0 | 97.6 dB | 0.006 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Z1-adaptive0.003 | ok | 20 ms | 7 | 18.3 | 156.2 dB | 0.000 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Z1-fixed64 | ok | 44 ms | 64 | 64.0 | 127.4 dB | 0.000 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | smoke-box | ok | 71 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | smoke-grid | ok | 59 ms | 32 | 32.0 | 73.6 dB | 0.053 |
| AMD Radeon 8060S | Vulkan | X1-adaptive0.003 | ok | 55 ms | 7 | 19.0 | 104.8 dB | 0.002 |
| AMD Radeon 8060S | Vulkan | X1-fixed64 | ok | 101 ms | 64 | 64.0 | 106.9 dB | 0.001 |
| AMD Radeon 8060S | Vulkan | Y1-adaptive0.003 | ok | 90 ms | 7 | 38.0 | 101.9 dB | 0.003 |
| AMD Radeon 8060S | Vulkan | Y1-fixed64 | ok | 115 ms | 64 | 64.0 | 102.2 dB | 0.003 |
| AMD Radeon 8060S | Vulkan | Z1-adaptive0.003 | ok | 17 ms | 7 | 18.3 | 151.5 dB | 0.000 |
| AMD Radeon 8060S | Vulkan | Z1-fixed64 | ok | 39 ms | 64 | 64.0 | 127.4 dB | 0.000 |
| AMD Radeon 8060S | Vulkan | smoke-box | ok | 102 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| AMD Radeon 8060S | Vulkan | smoke-grid | ok | 61 ms | 32 | 32.0 | 73.9 dB | 0.053 |

What the numbers show (measured):

- **Y1 adaptive is faster than Y1 fixed 64 on all three adapters after step Q1** (AMD 90 against 115 ms, llvmpipe 763 against 1251 ms, RTX 53 against 90 ms). In P1 it was slower on AMD (183 against 117 ms) and llvmpipe (1548 against 1187 ms) and level on RTX, because it averaged 72.7 samples per pixel; the Q1 section below has the cause and the change. Y1 adaptive now averages 38.0 samples per pixel.
- X1 and Z1 adaptive are faster than fixed on every adapter (mean 19.0 and 18.3 spp after Q1; 19.6 and 20.3 in P1): X1 55 against 101 ms on AMD after R1, Z1 20 against 44 ms on RTX.
- **Smoke majorant grid:** collisions fall from 112.6 tentative per path (single box) to 15.4 (grid), 7.3 times fewer, with real collisions unchanged at 0.19. After step Q2 the grid case is faster than the box case on every adapter: 1.7 times on AMD (102 to 61 ms), 2.8 times on llvmpipe (1025 to 367 ms) and 1.2 times on the RTX (71 to 59 ms). In P1 the RTX was level (144 against 143 ms); step Q2 found the cause in the dispatch size, not the collision loop.
- The RTX is 1.2 to 1.3 times faster than the Radeon on X1 and Y1 fixed, level on Z1 fixed (42 against 44 ms), slower on Z1 adaptive (24 against 20 ms) and level on the smoke grid after Q2 (59 against 61 ms).

Ranked recommendation for the next rendering brief (evidence above; no optimization started here):

1. **Y1 adaptive sampling.** Done in step Q1 (below): adaptive is faster than fixed on every adapter and scene. It was the only scene where adaptive lost on two adapters; it averaged more samples than fixed.
2. **Smoke on the RTX.** Done in step Q2 (below): the render no longer leaves the card idle (143 to 59 ms) and the grid is 17 percent faster than the single box. The collision loop is a small part of the remaining RTX time; what is left is per-path shader cost.
3. X1 quality: 54 to 55 dB on all adapters pointed at a scene or reference difference. Done in step R1 (below): the CPU reference was wrong, X1 now reads 105 to 110 dB on the Radeon and llvmpipe and 110 dB (fixed) on the RTX.

Known limits: one resolution (640 by 360), one seed per scene, three timed frames, so differences under about 10 percent are not established. Lock contention with other lanes was heavy; the timings were taken inside short exclusive lock turns, but a lane holding only the shared side of the lock cannot overlap them. RTX validation covers these eight cases on one boot only. No speed target is claimed.

## Step Q1: why Y1 adaptive loses to fixed 64

Lane 4, step Q1, October 8, 2026. This section was written from measurements taken before any renderer change (the commit that adds it changes no renderer file). Tool: `tools/benchmark_adaptive_overhead.py`, Y1 at 640 by 360, median of three renders after a warm-up, in a GPU lock turn, on the code of `515b6f8`.

**Stopping rule** (`pathtrace.pixel_noise`, the same estimate in the render shader, the sparse `reduce` shader and the CPU loop): every pixel takes 16 samples, then 8 more per pass, and stops at the first pass boundary where the variance of its mean luminance over (mean + 0.02) squared is under the threshold, or at `Max samples` (256). The decision is per pixel, never per tile. `Path samples` (the fixed 64 the benchmark compares with) is **ignored** by an adaptive render, so nothing ties its spend to the fixed render it replaces.

**Why the mean is 72.7 samples.** The sample counts of Y1 at threshold 0.003 (CPU path tracer; the GPU gives the same mean, 72.7):

- 52.3 percent of the pixels (the sky and the unlit floor) stop at the 16-sample minimum.
- 43.4 percent want more than 64 samples (the sphere and the floor under the hard sun; the bulk stops between 104 and 152), and 3.3 percent reach the 256 cap.
- Those 43.4 percent hold 86 percent of all the samples, and the part of their counts above 64 is 34.7 samples per pixel of the 72.7 mean. A fixed render gives every pixel 64.

So at 0.003 the rule asks for more than 64 samples on average because the threshold is tighter than fixed 64 delivers on the hard-sun pixels (fixed 64 reads 37.3 dB against a 1024-sample render; the uncapped adaptive render reads 39.9 dB). The rule is doing what it says, and the compaction is correct.

**Per-pass cost by phase** (adaptive at a threshold no pixel reaches, 16 minimum and 64 maximum samples, so every row runs the same shader work in 2 to 25 passes; sky pixels still stop at 16 because their noise reads exactly 0; slope of wall time against pass count):

| adapter | fixed 64 | cost per added pass | passes 2 to 25 span | what the phases say |
| --- | ---: | ---: | ---: | --- |
| AMD Radeon 8060S | 113.0 ms | -0.3 ms (noise) | 92.7 to 103.8 ms | `encode` 1.5 ms at 7 passes, 3.8 ms at 25; the waits (`dispatch`) are 2 to 8 |
| llvmpipe | 1232 ms | +0.2 ms | 837 to 868 ms | `encode` 649 ms at 7 passes and 653 ms at 25 |

The per-pass overhead is a small share: the full 31-pass adaptive render spends 5.3 ms of 177 ms in `encode` on the Radeon (3 percent). The llvmpipe `encode` phase that the P1 table reads as per-pass host cost is the shader work itself, because llvmpipe executes a submitted pass on the submitting thread; it does not grow with the number of passes (649 ms at 7 passes, 653 ms at 25). The compaction, the convergence test and the one-word waits are not what makes Y1 slow.

**What does make it slow.** The pixels left open are the expensive ones. Seven passes at 40.2 samples on average (sky stopped, the rest at 64) take 92.7 ms on the Radeon, 2.3 ms per mean sample, against 113 ms for fixed 64 (1.8 ms per sample): sky pixels are almost free to sample, the sphere and the sunlit floor are not. The uncapped adaptive render takes 72.7 samples on average and 177 ms. A trial that stopped the whole render once the average reached 64 (a global sample count, tried and dropped) still took 142.7 ms at a mean of 62.8 samples on the Radeon, because that average is spent on those pixels. A cap on every pixel cannot cost more than fixed 64 at any pixel, so the fix is that: `Path samples` becomes the most any pixel of an adaptive render may take.

Also seen while measuring: after the last open pixel has stopped, the host loop still encodes and submits every remaining pass of the 31 (each a compaction over the whole image plus empty dispatches). It waits on the card every fourth pass by reading 16 bytes of the accumulator, which says nothing about whether any pixel is left. The change reads the list's length (one word) at that same wait and ends the loop when it is zero.

### Step Q1: the change and the numbers after it

**The change.** `Path samples` is now the most any pixel of an adaptive render may take (`PathSettings.clamped` holds `Max samples`, and with it `Min samples`, to it; a cap below 2 is 2). The CPU loop and the GPU loop both read the clamped settings, so they stop the same pixels at the same pass. The GPU host loop also stops encoding once no pixel is open: the wait after every fourth pass reads the open-pixel count (one 4-byte word, instead of 16 accumulator bytes) and a zero ends the loop, so a render whose last open pixel has stopped no longer encodes the passes that remain. Nothing in the shaders changed. In the benchmark the adaptive case leaves `Path samples` at its default of 64, the fixed budget it is compared with.

**Rerun.** `tools/benchmark_portable_render.py --matrix` on the three adapters, the six X1, Y1 and Z1 cases (CPU references re-rendered with the same settings), each case in its own bounded lock turn; `tools/benchmark_adaptive_overhead.py` for the pass-cost and quality-against-1024-samples rows. Before the change (P1) against after, adaptive time (median) next to the fixed 64 time of the same rerun, then the share of fixed 64's time:

| adapter | scene | fixed 64 | adaptive before | adaptive after | after / fixed | mean spp before to after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AMD Radeon 8060S | X1 | 104 ms | 72 ms | 53 ms | 51% | 19.6 to 19.0 |
| AMD Radeon 8060S | Y1 | 115 ms | 183 ms | 90 ms | 79% | 72.7 to 38.0 |
| AMD Radeon 8060S | Z1 | 39 ms | 20 ms | 17 ms | 43% | 20.3 to 18.3 |
| llvmpipe | X1 | 623 ms | 300 ms | 271 ms | 44% | 19.6 to 19.0 |
| llvmpipe | Y1 | 1251 ms | 1548 ms | 763 ms | 61% | 72.7 to 38.0 |
| llvmpipe | Z1 | 374 ms | 167 ms | 147 ms | 39% | 20.3 to 18.3 |
| RTX 3080 Ti | X1 | 97 ms | 61 ms | 43 ms | 44% | 19.6 to 19.0 |
| RTX 3080 Ti | Y1 | 90 ms | 83 ms | 53 ms | 59% | 72.7 to 38.0 |
| RTX 3080 Ti | Z1 | 44 ms | 24 ms | 20 ms | 47% | 20.3 to 18.3 |

Adaptive is faster than fixed on all nine adapter and scene pairs. The narrowest margin is Y1 on the Radeon (90 against 115 ms, 79 percent of fixed 64); everywhere else adaptive takes 61 percent of fixed 64's time or less. The pass cost is no longer in play: the fit of wall time against pass count (`tools/benchmark_adaptive_overhead.py`, same shader work in 2 to 25 passes) gives between -0.2 and +0.3 ms per added pass on the Radeon and the RTX for X1, Y1 and Z1 (0.16 ms on llvmpipe, measured on Y1 only), against 40 to 120 ms for a whole render.

**Quality, two ways.** (1) Against the CPU path tracer at the same settings and seed, the metric of the table above. The adaptive rows move by at most 0.1 dB on X1 (54.5 to 54.6 dB on all three adapters); on Y1 llvmpipe rises from 99.0 to 104.3 dB and the Radeon from 99.7 to 101.9 dB, while the RTX falls from 98.9 to 97.4 dB, a drop of 1.5 dB, which is more than the 0.5 dB the step allowed. The RTX fixed 64 row reads 97.6 dB in both runs and the maximum error moves from 0.005 to 0.006 of full scale, so the RTX adaptive row now sits where its fixed row sits (inference: that is the level of the RTX's float rounding against the CPU, not a change in what the render shows). Z1 rises on all three adapters, from 102 to 116 dB up to 152 to 159 dB. (2) Against a 1024-sample fixed render of the same scene (another seed, the Radeon; the RTX agrees to 0.01 dB), which is the real image quality:

| scene | fixed 64 | adaptive 0.003 before (cap 256) | adaptive 0.003 after (cap 64) | mean spp before to after |
| --- | ---: | ---: | ---: | ---: |
| X1 | 46.68 dB | 44.99 dB | 44.74 dB | 19.6 to 19.0 |
| Y1 | 37.29 dB | 39.94 dB | 37.02 dB | 72.7 to 38.0 |
| Z1 | 48.59 dB | 46.04 dB | 45.43 dB | 20.3 to 18.3 |

So the cap costs Y1 2.9 dB against the old adaptive render (which bought that quality with 72.7 samples per pixel and 1.5 times fixed 64's time on the Radeon) and 0.27 dB against fixed 64, in 79 percent of fixed 64's time on the Radeon and 59 percent on the RTX. On X1 and Z1 the cap costs 0.25 and 0.6 dB against the old adaptive render; the capped render sits 1.9 and 3.2 dB below fixed 64 (the old one 1.7 and 2.6 dB). A threshold of 0.003 is a speed setting on those two scenes; a user who wants Y1's old 39.9 dB sets `Path samples` back to 256 and pays the old time.

Raw rows of the overhead tool: `benchmarks/adaptive_overhead/<code>-<adapter>-<scene>.json`, `base` on the P1 code (`515b6f8`) and `new` on `a6c04e2`. The Y1 llvmpipe row of `new` was taken while other lanes were using the CPU, so its wall times carry more noise than the rest.

Unverified or limited: one resolution, one seed, three timed renders, so margins under about 10 percent (none among the nine pairs) would not be established; llvmpipe timings share the CPU with the other lanes' work; the real-display and Windows checks are not part of this step; the quality-against-1024 rows come from the overhead tool, one scene run per adapter (Y1 on all three, X1 and Z1 on the Radeon and the RTX).

## Step Q2: smoke on the RTX

Lane 4, step Q2, October 8, 2026. Profile written from measurements taken before any renderer change. Tool: `tools/profile_smoke_phases.py` (one JSON line per variant in `benchmarks/smoke_phases/`, each run in a GPU lock turn on the RTX 3080 Ti, 640 by 360, fixed 32 samples, 4 bounces, median of three renders after a warm-up). The timestamp-query feature is not enabled on the device (it lists `float32-blendable` only), so phases come from the renderer's own laps plus dispatch-shape experiments.

**Phase profile of the render as shipped (P1 code).** Grid case, then the single-box case: scene build 0.3 and 0.3 ms, pack (host-side grid build) 0.8 and 0.2 ms, upload 2.4 ms, dispatch 106 and 125 ms, readback of the image 2.8 ms, post-processing on the host 4.7 ms; totals 130 and 137 ms. The renderer takes bands of about 65 000 paths (`GPU_PATHS_PER_SUBMISSION` divided by `SOFT_SLOWDOWN` 8), one sample per pass, so a 640 by 360 sample is 4 dispatches and the render is 128 dispatches, each followed by a wait on the card (0.83 ms per band in the grid case). There is no mesh in the scene, so there is no surface traversal phase; the grid is built on the host, not on the card.

**What the dispatch shape does** (median ms of the whole render, grid then box):

| dispatch shape | dispatches and waits | grid | box |
| --- | ---: | ---: | ---: |
| shipped: 4 bands per sample | 128 | 127 | 140 |
| 2 bands per sample | 64 | 85 | 94 |
| 1 band per sample (the whole image) | 32 | 57 | 70 |
| shipped bands, the waits batched (one per about 11 ms of work) | 7 | 108 | 118 |
| 4 samples per thread, shipped bands | 120 | 225 | 239 |
| 8 samples per thread, shipped bands | 120 | 378 | 378 |

Batching the waits barely helps (127 to 108 ms), so the round trips are not the cost. Bigger dispatches are: a band of 65 000 paths leaves most of the card idle (the 3080 Ti keeps about 120 000 threads resident), and the same work as one dispatch over the whole image takes 2.2 times less. Giving each thread several samples makes it worse, because the threads of a pass then run one sample after another.

**Where the dispatch time goes after the dispatch fix** (the same render with bands grown to the image, grid then box; one variable changed at a time): camera aimed away from the plume, every ray leaving the box: 12.7 ms of dispatch, the fixed cost of the shader per path. With the plume in view: 42 and 48 ms. Bounces 0, 1, 2, 4: 27, 35, 41, 45 ms in the grid case and 32, 40, 45, 48 in the box case. Shadow steps 1, 4, 16, 64: 47, 47, 39, 45 ms (grid), no measurable effect. Work-group edge 4, 8, 16, 32: 50, 41, 44, 72 ms (grid), 70, 55, 51, 86 (box), so 8 stays. Ambient light off: slower, not faster.

Reading the table: the collision loop is the difference between the box and the grid, 4 to 7 ms of a 40 to 48 ms dispatch, even though the grid does 7.6 times fewer tentative collisions. On this card the 112 collisions per path of the single box are cheap; the cost sits in the rest of the path (12.7 ms of fixed shader cost, about 14 ms for the camera flight into the plume and its first scatter vertex, 6 to 8 ms per further bounce). That part is the same in both cases, so it caps what the grid can win on the RTX. Alongside it the host-side phases (upload 2.4, readback 2.8, post-processing 4.7 ms) are about 12 ms that the grid cannot touch.

Measured above: dispatch shape, the floor, bounce, shadow-step and work-group effects. Inference: that the remaining plume cost is per-path shader work (register pressure and divergence of the smoke code) rather than memory traffic; no in-shader timer exists to separate those.

### Step Q2: the change and the numbers after it

**The change.** A smoke or splat render takes its rows in bands whose size follows the measured time of the last dispatch. It starts at the old size (65 000 paths, so a slow adapter begins as short as before), and after each wait the band grows to about 8 ms of work (`BAND_TARGET_SECONDS`), at most four times larger per step, never smaller than the start and never more than 4 million paths (`BAND_MAX_PATHS`). Surface-only renders and adaptive renders are unchanged. Nothing in the shaders changed; the sample sums do not depend on how the rows are cut, and `tests/test_gpu_band_growth.py` renders the same image with fixed and grown bands and requires them to be identical, with fewer than half the waits.

**Rerun** (`tools/benchmark_portable_render.py --matrix`, the two smoke cases on each adapter, own bounded lock turn each, CPU references from P1 reused; three timed renders in brackets):

| adapter | case | P1 | after Q2 | timed renders after | PSNR vs CPU, P1 to after |
| --- | --- | ---: | ---: | --- | --- |
| RTX 3080 Ti | smoke-box | 144 ms | 71 ms | 70, 78, 71 | 84.8 to 84.8 dB |
| RTX 3080 Ti | smoke-grid | 143 ms | 59 ms | 59, 60, 58 | 73.6 to 73.6 dB |
| AMD Radeon 8060S | smoke-box | 157 ms | 102 ms | 102, 99, 103 | 84.8 to 84.8 dB |
| AMD Radeon 8060S | smoke-grid | 113 ms | 61 ms | 62, 59, 61 | 73.9 to 73.9 dB |
| llvmpipe | smoke-box | 1031 ms | 1025 ms | 1025, 1031, 1005 | 84.8 to 84.8 dB |
| llvmpipe | smoke-grid | 515 ms | 367 ms | 367, 368, 366 | 75.2 to 75.2 dB |

On the RTX the grid case is 17 percent faster than the box case (59 against 71 ms, every grid run under every box run). The AMD gain is kept and grows (the grid case 1.9 times faster than P1, the box case 1.5 times); llvmpipe keeps its grid gain and the grid case is 1.4 times faster than P1. The llvmpipe box case did not change: its dispatches already take longer than 8 ms, so its bands stay at the start size. The images are the same to the digit of the table (PSNR against the CPU, mean and maximum error unchanged). The collision counters are unchanged (15.4 tentative per path with the grid, 112.6 with the box).

Same-session before and after, from `tools/profile_smoke_phases.py` (`fixed` is the old shape, `grow` the new one; median of five renders, two repeats; `benchmarks/smoke_phases/prof9.jsonl`): RTX grid 127 and 126 to 56 and 58 ms, RTX box 129 and 147 to 61 and 66 ms; AMD grid 112 and 115 to 57 and 61 ms, AMD box 152 and 163 to 102 and 102 ms; llvmpipe grid 496 and 498 to 361 and 359 ms.

**What bounds the RTX now** (measured in the profile above, with the inference named there): about 12 ms of every 59 ms render is host-side (upload, image readback and post-processing, none of which the grid changes), 12.7 ms of the dispatch is the shader's fixed cost per path, and the rest is the smoke vertices' own work. The collision loop is the 4 to 7 ms the box adds, which is why the grid wins 17 percent and not the 7 times its collision count suggests. A bigger win on the RTX would need a cheaper smoke vertex (light sampling and the phase-function continuation after each real collision) or a leaner fixed path; both are shader changes with the AMD work-group limits behind them (`WG_SIZE_SPLATS_AND_VOLUMES`), left for a later step.

Unverified or limited: one resolution (640 by 360), one scene and seed, three timed renders; the RTX gain was measured on one boot with no watchdog or device-lost error seen in the case outputs (the kernel log is not readable from the worker); llvmpipe timings share the CPU with other lanes; no Windows or real-display run; surface-only renders were not changed and not re-measured (their bands are the same as in P1).

## Step R1: why X1 reads 55 dB against the CPU path tracer

Lane 4, step R1, October 9, 2026. The finding below was written from measurements taken before any renderer change (the commit that first added this section changed no renderer file; base `e08f6b4`); the change and the rerun follow it.

**Where the images differ** (measured; X1 fixed 64 at 640 by 360, llvmpipe against the CPU path tracer, the same 55.2 dB and 0.014 maximum error as the P1 and Q1 tables). Every pixel off the sphere is identical (mean difference exactly 0). On the sphere the mean absolute difference is 0.0067 in linear light, and the difference has two shapes. The left half of the sphere (the left column of the 2 by 1 metallic-roughness map, metallic 0.9 and roughness 0.2) differs by noise: the GPU is 0.0049 brighter there on average, speckled pixel by pixel. The right half (roughness 0.8, metallic 0.1, mostly diffuse) differs smoothly across a disc, the GPU darker. The difference follows the sphere's material.

**Which feature** (measured; 256 by 144, 32 samples, llvmpipe against CPU, one scene feature removed at a time):

| X1 with | PSNR vs CPU |
| --- | ---: |
| everything (as benchmarked) | 55.2 dB |
| no occlusion texture | 106.9 dB |
| no environment light | 75.9 dB |
| no metallic-roughness map | 50.9 dB |
| no base-colour texture | 54.1 dB |
| no normal map, no emissive map, no rectangle light, no point light, no shadows (each alone) | 54.7, 53.8, 55.1, 54.7, 55.2 dB |

Removing the occlusion texture alone brings the two backends to 106.9 dB, the same as Y1 and Z1. Removing the environment light lifts the number to 75.9 dB, which still leaves a gap; no other single removal moves it by more than a few dB. X1 is the only benchmark scene with an occlusion map.

**Why** (inference from reading both renderers; step R1's fix tests it). Occlusion scales the diffuse colour of the hit, here by 0.58. The two backends apply that scale in different places:

- GPU (`make_lobe` in `nodebased/gpupathtrace.py`): the scale goes into the lobe's diffuse colour first, so the specular-or-diffuse choice probability (`p_spec`), the BSDF response and the sampling density (`pdf`) are all computed from the scaled colour. Light samples, MIS weights and BSDF samples agree with one another.
- CPU (`_surface_event` in `nodebased/pathtrace.py`): the scaled colour is used to choose between the specular and the diffuse lobe when it samples a bounce, but `bsdf_eval` rebuilds the lobe from the unscaled colour to get its `pdf`. The `pdf` therefore describes a different choice probability than the one the sampler used. That `pdf` divides the BSDF sample's weight and feeds the MIS weight of every light sample, so with an occlusion map the CPU estimate converges to the wrong image. Where the two probabilities differ most (the left half: a near-mirror lobe on a surface that also has a diffuse lobe) the CPU image is wrong in a speckled way; on the right half the bias is smooth.

So the CPU path tracer is the image that is wrong, and the GPU image is the one that is consistent. Fix in the next commit: the CPU lobe carries the occlusion the way the GPU lobe does.

### Step R1: the change and the numbers after it

**The change** (`6007642`, `nodebased/pathtrace.py`). `_lobes` and `bsdf_eval` take the hit's occlusion and scale the diffuse colour with it before the lobe choice probability, the responses and the `pdf` are derived, the way the GPU `make_lobe` does. `_surface_event` passes the occlusion in and no longer multiplies it into the diffuse response separately at the four places it did. Without an occlusion map (factor 1) nothing changes. The GPU renderer is untouched. This is a change to the CPU reference: every earlier CPU render of a scene with an occlusion map (the CPU path tracer is the reference of the benchmark and of the 1024-sample quality rows) was biased toward a different image than the GPU draws; scenes without an occlusion map render exactly as before.

**Rerun** (`tools/benchmark_portable_render.py --matrix`, the two X1 cases on each adapter, each in its own bounded exclusive GPU lock turn, the CPU references re-rendered with the fixed code; commit `6007642`):

| adapter | case | PSNR before to after | max error before to after | median before to after |
| --- | --- | ---: | ---: | ---: |
| AMD Radeon 8060S | X1 fixed 64 | 55.2 to 106.9 dB | 0.014 to 0.001 | 104 to 101 ms |
| AMD Radeon 8060S | X1 adaptive 0.003 | 54.6 to 104.8 dB | 0.064 to 0.002 | 53 to 55 ms |
| llvmpipe | X1 fixed 64 | 55.2 to 109.4 dB | 0.014 to 0.000 | 623 to 619 ms |
| llvmpipe | X1 adaptive 0.003 | 54.6 to 105.3 dB | 0.064 to 0.001 | 271 to 263 ms |
| RTX 3080 Ti | X1 fixed 64 | 55.2 to 110.3 dB | 0.014 to 0.001 | 97 to 87 ms |
| RTX 3080 Ti | X1 adaptive 0.003 | 54.6 to 90.1 dB | 0.064 to 0.022 | 43 to 43 ms |

Five of the six rows are above the 97 dB the other scenes read. The RTX adaptive row reads 90.1 dB, below that line, because of one pixel (measured: per-pixel sample counts of the CPU and the RTX renders of X1 adaptive 0.003, mean 19.0 on both; they differ at exactly one pixel of 230 400, x 249, y 179, which stops after 40 samples on the CPU and 48 on the RTX, one pass later). Everywhere else the two images differ by at most 0.009 in linear light. The stopping rule reads a pixel whose noise estimate sits on the threshold, and the RTX's float32 sums land on the other side of it (inference: the sample counts are the only difference, and the Radeon and llvmpipe rows read above 104 dB). That is the same kind of single-pixel decision the Q1 section saw on Y1, and it is not a rendering defect, so it is left as it is. Timings moved by at most the run-to-run spread of the earlier rows (no renderer change on the GPU side).

**Tests** (`tests/test_3d_occlusion_lobe.py`): two CPU tests that the occlusion scales the diffuse colour of the lobe (a dielectric hit with occlusion 0.5 evaluates exactly like the same hit with half the base colour: both responses and the `pdf`; and an occlusion of 1 leaves it unchanged), and a regression test that renders X1 at 160 by 90 and 32 samples on the CPU and the default GPU adapter and compares the sphere's crop: it measured 48.4 dB (largest error 0.0136) on llvmpipe before the change and 102.3 dB after, and the test asks for more than 85 dB and 0.004. Before the change all three tests fail; after it they pass. The test modules `test_3d_occlusion_lobe`, `test_3d_pbr_textures`, `test_3d_pathtrace`, `test_3d_pathtrace_gpu_soft` and `test_benchmark_portable_render` (148 tests) passed on the default adapter (the RTX), on the Radeon through `force-adapter.py integrated` and on llvmpipe through `force-adapter.py cpu`; the splat, volume, denoise, progressive, sparse and GPU ray-trace modules (126 tests) passed on the default adapter.

Unverified or limited: one resolution and seed per scene; the RTX adaptive row's one-pixel explanation is an inference from the sample counts; Windows and the real display were not checked; the 1024-sample quality rows of the Q1 section (X1 46.68, 44.99 and 44.74 dB) were measured against the old CPU reference and have not been re-measured.
