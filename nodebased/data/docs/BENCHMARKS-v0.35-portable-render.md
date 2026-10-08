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

The six X1, Y1 and Z1 case pairs were re-measured on October 8 in step Q1 (renderer commit `a6c04e2`); the smoke rows are the P1 measurements. Median of three timed renders after one warm-up. "Passes" is the number of render passes the adaptive loop or fixed loop ran; "mean spp" is the mean samples per pixel. PSNR and max error compare each adapter with the CPU path tracer at the same settings and seed (clipped sRGB RGB). Two of the 24 rows carry noticeable GPU-to-CPU differences: X1 (54 to 55 dB, max error 0.014 to 0.064, the same on all three adapters, so it comes from the scene and not one driver) and the smoke grid cases (74 to 75 dB). Everything else is above 97 dB.

| adapter | backend | case | status | median | passes | mean spp | PSNR vs CPU | max error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| llvmpipe | Vulkan | X1-adaptive0.003 | ok | 271 ms | 7 | 19.0 | 54.6 dB | 0.064 |
| llvmpipe | Vulkan | X1-fixed64 | ok | 623 ms | 64 | 64.0 | 55.2 dB | 0.014 |
| llvmpipe | Vulkan | Y1-adaptive0.003 | ok | 763 ms | 7 | 38.0 | 104.3 dB | 0.002 |
| llvmpipe | Vulkan | Y1-fixed64 | ok | 1251 ms | 64 | 64.0 | 104.6 dB | 0.002 |
| llvmpipe | Vulkan | Z1-adaptive0.003 | ok | 147 ms | 7 | 18.3 | 159.2 dB | 0.000 |
| llvmpipe | Vulkan | Z1-fixed64 | ok | 374 ms | 64 | 64.0 | 122.6 dB | 0.000 |
| llvmpipe | Vulkan | smoke-box | ok | 1031 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| llvmpipe | Vulkan | smoke-grid | ok | 515 ms | 32 | 32.0 | 75.2 dB | 0.053 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | X1-adaptive0.003 | ok | 43 ms | 7 | 19.0 | 54.6 dB | 0.064 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | X1-fixed64 | ok | 97 ms | 64 | 64.0 | 55.2 dB | 0.014 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Y1-adaptive0.003 | ok | 53 ms | 7 | 38.0 | 97.4 dB | 0.006 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Y1-fixed64 | ok | 90 ms | 64 | 64.0 | 97.6 dB | 0.006 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Z1-adaptive0.003 | ok | 20 ms | 7 | 18.3 | 156.2 dB | 0.000 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Z1-fixed64 | ok | 44 ms | 64 | 64.0 | 127.4 dB | 0.000 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | smoke-box | ok | 144 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | smoke-grid | ok | 143 ms | 32 | 32.0 | 73.6 dB | 0.053 |
| AMD Radeon 8060S | Vulkan | X1-adaptive0.003 | ok | 53 ms | 7 | 19.0 | 54.6 dB | 0.064 |
| AMD Radeon 8060S | Vulkan | X1-fixed64 | ok | 104 ms | 64 | 64.0 | 55.2 dB | 0.014 |
| AMD Radeon 8060S | Vulkan | Y1-adaptive0.003 | ok | 90 ms | 7 | 38.0 | 101.9 dB | 0.003 |
| AMD Radeon 8060S | Vulkan | Y1-fixed64 | ok | 115 ms | 64 | 64.0 | 102.2 dB | 0.003 |
| AMD Radeon 8060S | Vulkan | Z1-adaptive0.003 | ok | 17 ms | 7 | 18.3 | 151.5 dB | 0.000 |
| AMD Radeon 8060S | Vulkan | Z1-fixed64 | ok | 39 ms | 64 | 64.0 | 127.4 dB | 0.000 |
| AMD Radeon 8060S | Vulkan | smoke-box | ok | 157 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| AMD Radeon 8060S | Vulkan | smoke-grid | ok | 113 ms | 32 | 32.0 | 73.9 dB | 0.053 |

What the numbers show (measured):

- **Y1 adaptive is faster than Y1 fixed 64 on all three adapters after step Q1** (AMD 90 against 115 ms, llvmpipe 763 against 1251 ms, RTX 53 against 90 ms). In P1 it was slower on AMD (183 against 117 ms) and llvmpipe (1548 against 1187 ms) and level on RTX, because it averaged 72.7 samples per pixel; the Q1 section below has the cause and the change. Y1 adaptive now averages 38.0 samples per pixel.
- X1 and Z1 adaptive are faster than fixed on every adapter (mean 19.0 and 18.3 spp after Q1; 19.6 and 20.3 in P1): X1 53 against 104 ms on AMD, Z1 20 against 44 ms on RTX.
- **Smoke majorant grid:** collisions fall from 112.6 tentative per path (single box) to 14.9 (grid), 7.6 times fewer, with real collisions unchanged at 0.19. Wall time improves 1.39 times on AMD (157 to 113 ms), 2.0 times on llvmpipe (1031 to 515 ms) and not at all on RTX (144 against 143 ms), where about 125 ms of dispatch time remains in both. The RTX smoke cost is therefore not collision work.
- The RTX is 1.2 to 1.3 times faster than the Radeon on X1 and Y1 fixed, level on Z1 fixed (42 against 44 ms), slower on Z1 adaptive (24 against 20 ms) and slower on the smoke grid (143 against 113 ms).

Ranked recommendation for the next rendering brief (evidence above; no optimization started here):

1. **Y1 adaptive sampling.** Done in step Q1 (below): adaptive is faster than fixed on every adapter and scene. It was the only scene where adaptive lost on two adapters; it averaged more samples than fixed.
2. **Smoke on the RTX.** The grid removes 7.6 times the collision work but the RTX does not get faster, so something other than the collision loop (dispatch occupancy or memory access for the grid) bounds it there. Needs a phase-level profile on the RTX before any change.
3. X1 quality: 54 to 55 dB on all adapters points at a scene or reference difference worth one look before it is used for a visual claim.

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
