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
