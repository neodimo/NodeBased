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

Median of three timed renders after one warm-up. "Passes" is the number of render passes the adaptive loop or fixed loop ran; "mean spp" is the mean samples per pixel. PSNR and max error compare each adapter with the CPU path tracer at the same settings and seed (clipped sRGB RGB). Two of the 24 rows carry noticeable GPU-to-CPU differences: X1 (54 to 55 dB, max error 0.014 to 0.064, the same on all three adapters, so it comes from the scene and not one driver) and the smoke grid cases (74 to 75 dB). Everything else is above 97 dB.

| adapter | backend | case | status | median | passes | mean spp | PSNR vs CPU | max error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| llvmpipe | Vulkan | X1-adaptive0.003 | ok | 300 ms | 31 | 19.6 | 54.5 dB | 0.064 |
| llvmpipe | Vulkan | X1-fixed64 | ok | 615 ms | 64 | 64.0 | 55.2 dB | 0.014 |
| llvmpipe | Vulkan | Y1-adaptive0.003 | ok | 1548 ms | 31 | 72.7 | 99.0 dB | 0.004 |
| llvmpipe | Vulkan | Y1-fixed64 | ok | 1187 ms | 64 | 64.0 | 104.6 dB | 0.002 |
| llvmpipe | Vulkan | Z1-adaptive0.003 | ok | 167 ms | 31 | 20.3 | 115.7 dB | 0.001 |
| llvmpipe | Vulkan | Z1-fixed64 | ok | 362 ms | 64 | 64.0 | 122.6 dB | 0.000 |
| llvmpipe | Vulkan | smoke-box | ok | 1031 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| llvmpipe | Vulkan | smoke-grid | ok | 515 ms | 32 | 32.0 | 75.2 dB | 0.053 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | X1-adaptive0.003 | ok | 61 ms | 31 | 19.6 | 54.5 dB | 0.064 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | X1-fixed64 | ok | 87 ms | 64 | 64.0 | 55.2 dB | 0.014 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Y1-adaptive0.003 | ok | 83 ms | 31 | 72.7 | 98.9 dB | 0.005 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Y1-fixed64 | ok | 88 ms | 64 | 64.0 | 97.6 dB | 0.006 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Z1-adaptive0.003 | ok | 24 ms | 31 | 20.3 | 109.0 dB | 0.002 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | Z1-fixed64 | ok | 42 ms | 64 | 64.0 | 127.4 dB | 0.000 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | smoke-box | ok | 144 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| NVIDIA GeForce RTX 3080 Ti | Vulkan | smoke-grid | ok | 143 ms | 32 | 32.0 | 73.6 dB | 0.053 |
| AMD Radeon 8060S | Vulkan | X1-adaptive0.003 | ok | 72 ms | 31 | 19.6 | 54.5 dB | 0.064 |
| AMD Radeon 8060S | Vulkan | X1-fixed64 | ok | 103 ms | 64 | 64.0 | 55.2 dB | 0.014 |
| AMD Radeon 8060S | Vulkan | Y1-adaptive0.003 | ok | 183 ms | 31 | 72.7 | 99.7 dB | 0.005 |
| AMD Radeon 8060S | Vulkan | Y1-fixed64 | ok | 117 ms | 64 | 64.0 | 102.2 dB | 0.003 |
| AMD Radeon 8060S | Vulkan | Z1-adaptive0.003 | ok | 20 ms | 31 | 20.3 | 102.4 dB | 0.004 |
| AMD Radeon 8060S | Vulkan | Z1-fixed64 | ok | 44 ms | 64 | 64.0 | 127.4 dB | 0.000 |
| AMD Radeon 8060S | Vulkan | smoke-box | ok | 157 ms | 32 | 32.0 | 84.8 dB | 0.028 |
| AMD Radeon 8060S | Vulkan | smoke-grid | ok | 113 ms | 32 | 32.0 | 73.9 dB | 0.053 |

What the numbers show (measured):

- **Y1 adaptive is slower than Y1 fixed 64 on AMD (183 against 117 ms) and llvmpipe (1548 against 1187 ms), and level on RTX (83 against 88 ms).** Adaptive spends a mean of 72.7 samples per pixel on Y1 (some pixels reach the 256 cap), more than the 64 fixed. On llvmpipe the adaptive run also spends 1248 ms in the `encode` phase against 300 ms of dispatch, so the CPU-side per-pass encoding of 31 passes dominates there; on AMD and RTX `encode` is 6 ms.
- X1 and Z1 adaptive are faster than fixed on every adapter (mean 19.6 and 20.3 spp): X1 72 against 103 ms on AMD, Z1 24 against 42 ms on RTX.
- **Smoke majorant grid:** collisions fall from 112.6 tentative per path (single box) to 14.9 (grid), 7.6 times fewer, with real collisions unchanged at 0.19. Wall time improves 1.39 times on AMD (157 to 113 ms), 2.0 times on llvmpipe (1031 to 515 ms) and not at all on RTX (144 against 143 ms), where about 125 ms of dispatch time remains in both. The RTX smoke cost is therefore not collision work.
- The RTX is 1.2 to 1.3 times faster than the Radeon on X1 and Y1 fixed, level on Z1 fixed (42 against 44 ms), slower on Z1 adaptive (24 against 20 ms) and slower on the smoke grid (143 against 113 ms).

Ranked recommendation for the next rendering brief (evidence above; no optimization started here):

1. **Y1 adaptive sampling.** It is the only scene where adaptive loses on two adapters, it averages more samples than fixed, and on the CPU-backed adapter its per-pass encode cost is the largest single phase. Investigate the stopping rule and the per-pass cost for scenes with a wide spread of per-pixel noise.
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
