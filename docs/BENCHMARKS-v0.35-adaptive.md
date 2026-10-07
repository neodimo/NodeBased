# v0.35 — adaptive sampling in the path tracer, after the device-side mask

Plan "Rendering 8", step T2. The v0.34 measurements (`docs/BENCHMARKS-v0.34-adaptive.md`) showed adaptive sampling slower
than fixed 64 on both hardware GPUs at 1280 by 720 (X1 at threshold 0.01: 437 ms and 41.8 dB against fixed 64 at 311 ms and
46.8 dB). Since then the GPU path tracer keeps each pixel's running moments and done flag on the card, lists the pixels
still open with a compaction pass, dispatches only those through indirect dispatches, runs the last few noisy pixels sample
by sample in parallel, and reads nothing back before the finished image (docs/3D_FOUNDATION.md, "Adaptive sampling"). This page
repeats the measurement on the same three scenes and the same three adapters.

Measured October 7, 2026 between 12:16 AM and 12:25 AM PDT on commit `c51028c`, Linux 7.2.7-ogc1.1.fc44.x86_64, Python
3.12.13, NumPy 2.5.3, wgpu 0.32.0, on this workstation with the exclusive GPU lock held for every run. Script:
`tools/benchmark_adaptive.py` (`flock /tmp/nb-gpu.lock python tools/benchmark_adaptive.py --adapter default|integrated|cpu
--size 1280x720`). The RTX 3080 Ti was run a second time right after the first (the "repeat" column) to show how far one
number moves from run to run.

## Method

- **Scenes.** X1: a PBR sphere with a base-colour texture, a metallic-roughness map, a normal map, an occlusion map and an
  emissive map, lit by a Rect light, a Point light and a smooth environment. Y1: a PBR sphere on a floor under an
  environment with a hard sun. Z1: a cube casting a soft shadow from a Disc area light with 16 light samples. Default
  camera, 8 bounces, the path tracer's GPU backend.
- **Renders.** `fixed` at 16, 32, 64 and 128 samples (seed 1); `adaptive` at noise thresholds 0.05, 0.01, 0.006, 0.003,
  0.001 and 0.0003 with Render3D's other defaults (16 minimum samples, 256 maximum, passes of 8, seed 1); and the
  reference, `fixed` 1024 samples with seed 777.
- **Wall time** is the median of three renders after a warm-up render, from the call into the path tracer to the returned
  image. It includes building the scene, packing it and uploading it, which cost the same whatever the sampling; the fixed 16
  row is the nearest thing to that floor.
- **PSNR** is against the reference, over all pixels, on what the viewer shows (values clipped to 0 to 1, sRGB encoded, peak
  1). The reference carries noise of its own, so no row can reach infinity. The same seeds give the same PSNR on every
  adapter at the same size, which is why the two hardware GPUs share one PSNR column (llvmpipe renders a smaller image, so its
  PSNR differs a little).
- **Samples** is the per-pixel count the render took (`stats["samples"]`): the mean, and the largest count of any pixel
  (the smallest is 16 everywhere).
- **The bar.** On the RTX 3080 Ti some adaptive threshold reaches fixed 64's PSNR in less wall time than fixed 64 on at
  least two of the three scenes.

## NVIDIA GeForce RTX 3080 Ti, 1280 by 720

| scene | sampling | wall time | wall time (repeat) | PSNR | mean samples | max samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| X1 | fixed 16 | 103 ms | | 40.9 dB | 16.0 | 16 |
| X1 | fixed 32 | 145 ms | | 43.9 dB | 32.0 | 32 |
| X1 | fixed 64 | 326 ms | 230 ms | 46.8 dB | 64.0 | 64 |
| X1 | fixed 128 | 549 ms | | 49.5 dB | 128.0 | 128 |
| X1 | adaptive 0.05 | 78 ms | 77 ms | 40.9 dB | 16.0 | 184 |
| X1 | adaptive 0.01 | 85 ms | 86 ms | 41.8 dB | 16.4 | 256 |
| X1 | adaptive 0.006 | 91 ms | 90 ms | 43.0 dB | 17.2 | 256 |
| X1 | adaptive 0.003 | 103 ms | 101 ms | 45.0 dB | 19.4 | 256 |
| X1 | adaptive 0.001 | 141 ms | 139 ms | 47.8 dB | 28.1 | 256 |
| X1 | adaptive 0.0003 | 194 ms | 192 ms | 50.4 dB | 42.0 | 256 |
| Y1 | fixed 16 | 105 ms | | 31.7 dB | 16.0 | 16 |
| Y1 | fixed 32 | 158 ms | | 34.5 dB | 32.0 | 32 |
| Y1 | fixed 64 | 236 ms | 220 ms | 37.3 dB | 64.0 | 64 |
| Y1 | fixed 128 | 426 ms | | 40.0 dB | 128.0 | 128 |
| Y1 | adaptive 0.05 | 91 ms | 96 ms | 32.3 dB | 16.3 | 256 |
| Y1 | adaptive 0.01 | 135 ms | 137 ms | 35.3 dB | 28.8 | 256 |
| Y1 | adaptive 0.006 | 180 ms | 181 ms | 37.3 dB | 41.7 | 256 |
| Y1 | adaptive 0.003 | 297 ms | 293 ms | 40.0 dB | 72.5 | 256 |
| Y1 | adaptive 0.001 | 465 ms | 452 ms | 42.1 dB | 125.0 | 256 |
| Y1 | adaptive 0.0003 | 486 ms | 474 ms | 42.4 dB | 132.6 | 256 |
| Z1 | fixed 16 | 66 ms | | 42.8 dB | 16.0 | 16 |
| Z1 | fixed 32 | 77 ms | | 45.7 dB | 32.0 | 32 |
| Z1 | fixed 64 | 114 ms | 126 ms | 48.7 dB | 64.0 | 64 |
| Z1 | fixed 128 | 203 ms | | 51.4 dB | 128.0 | 128 |
| Z1 | adaptive 0.05 | 51 ms | 52 ms | 43.0 dB | 16.0 | 32 |
| Z1 | adaptive 0.01 | 53 ms | 54 ms | 44.4 dB | 16.8 | 120 |
| Z1 | adaptive 0.006 | 54 ms | 58 ms | 45.1 dB | 17.8 | 184 |
| Z1 | adaptive 0.003 | 57 ms | 56 ms | 46.1 dB | 20.2 | 256 |
| Z1 | adaptive 0.001 | 64 ms | 64 ms | 47.6 dB | 27.0 | 256 |
| Z1 | adaptive 0.0003 | 79 ms | 80 ms | 50.8 dB | 45.7 | 256 |

The repeat column shows that the adaptive rows repeat to within a few milliseconds while a fixed 64 render moves more: X1's
fixed 64 took 326 ms the first time and 230 ms the second (the same render, the same card), so the ratios below use the
faster, repeat time as well as the first.

**The bar, on this card: met on all three scenes.** The quickest adaptive render that reaches fixed 64's PSNR:

| scene | fixed 64 | quickest adaptive at or above that PSNR | share of fixed 64's time (first run / repeat) |
| --- | --- | --- | ---: |
| X1 | 46.8 dB in 326 ms (repeat 230 ms) | 0.001: 47.8 dB in 141 ms | 43% / 60% |
| Y1 | 37.3 dB in 236 ms (repeat 220 ms) | 0.006: 37.3 dB in 180 ms | 76% / 82% |
| Z1 | 48.7 dB in 114 ms (repeat 126 ms) | 0.0003: 50.8 dB in 79 ms | 70% / 64% |

Each scene needs its own threshold: Z1 needs 0.0003 to get past 48.7 dB (0.001 stops at 47.6 dB, 1.1 dB short, in 56% of
fixed 64's time), and Y1's hard sun and glossy highlight keep a large share of its pixels open to 128 samples or more at 0.001
(125 mean samples, 465 ms, twice fixed 64's time for 4.8 dB more than fixed 64). Against the v0.34 table at the same size the
same render at 0.01 is 85 ms on X1 (437 ms in v0.34), 135 ms on Y1 (437 ms) and 53 ms on Z1 (211 ms).

## AMD Radeon 8060S (integrated), 1280 by 720

PSNR and sample counts are the RTX table's. Fixed 64 is 278 ms on X1, 390 ms on Y1 and 109 ms on Z1.

| scene | sampling | wall time | mean samples |
| --- | --- | ---: | ---: |
| X1 | fixed 16 | 109 ms | 16.0 |
| X1 | fixed 32 | 165 ms | 32.0 |
| X1 | fixed 64 | 278 ms | 64.0 |
| X1 | fixed 128 | 493 ms | 128.0 |
| X1 | adaptive 0.05 | 91 ms | 16.0 |
| X1 | adaptive 0.01 | 99 ms | 16.4 |
| X1 | adaptive 0.006 | 112 ms | 17.2 |
| X1 | adaptive 0.003 | 143 ms | 19.4 |
| X1 | adaptive 0.001 | 257 ms | 28.1 |
| X1 | adaptive 0.0003 | 399 ms | 42.0 |
| Y1 | fixed 16 | 136 ms | 16.0 |
| Y1 | fixed 32 | 233 ms | 32.0 |
| Y1 | fixed 64 | 390 ms | 64.0 |
| Y1 | fixed 128 | 761 ms | 128.0 |
| Y1 | adaptive 0.05 | 138 ms | 16.3 |
| Y1 | adaptive 0.01 | 262 ms | 28.8 |
| Y1 | adaptive 0.006 | 385 ms | 41.7 |
| Y1 | adaptive 0.003 | 683 ms | 72.5 |
| Y1 | adaptive 0.001 | 1125 ms | 125.0 |
| Y1 | adaptive 0.0003 | 1181 ms | 132.6 |
| Z1 | fixed 16 | 60 ms | 16.0 |
| Z1 | fixed 32 | 71 ms | 32.0 |
| Z1 | fixed 64 | 109 ms | 64.0 |
| Z1 | fixed 128 | 185 ms | 128.0 |
| Z1 | adaptive 0.05 | 47 ms | 16.0 |
| Z1 | adaptive 0.01 | 44 ms | 16.8 |
| Z1 | adaptive 0.006 | 47 ms | 17.8 |
| Z1 | adaptive 0.003 | 52 ms | 20.2 |
| Z1 | adaptive 0.001 | 67 ms | 27.0 |
| Z1 | adaptive 0.0003 | 97 ms | 45.7 |

Quickest adaptive render at or above fixed 64's PSNR: X1 0.001 in 257 ms against 278 ms (93%), Y1 0.006 in 385 ms against
390 ms (99%), Z1 0.0003 in 97 ms against 109 ms (90%). On this card the win is small, a tenth of the time at best, and Y1
breaks even.

## llvmpipe (software Vulkan on the CPU), 160 by 90

The adapter where the cost of a sample is the cost of the pixels it covers. The image is smaller, so PSNR differs a
little from the tables above.

| scene | sampling | wall time | PSNR | mean samples | max samples |
| --- | --- | ---: | ---: | ---: | ---: |
| X1 | fixed 16 | 43 ms | 39.8 dB | 16.0 | 16 |
| X1 | fixed 32 | 71 ms | 42.9 dB | 32.0 | 32 |
| X1 | fixed 64 | 112 ms | 45.7 dB | 64.0 | 64 |
| X1 | fixed 128 | 210 ms | 48.3 dB | 128.0 | 128 |
| X1 | adaptive 0.05 | 38 ms | 40.0 dB | 16.1 | 128 |
| X1 | adaptive 0.01 | 46 ms | 41.2 dB | 17.4 | 256 |
| X1 | adaptive 0.006 | 52 ms | 42.3 dB | 18.6 | 256 |
| X1 | adaptive 0.003 | 58 ms | 43.9 dB | 21.3 | 256 |
| X1 | adaptive 0.001 | 75 ms | 45.7 dB | 30.2 | 256 |
| X1 | adaptive 0.0003 | 100 ms | 47.0 dB | 43.7 | 256 |
| Y1 | fixed 16 | 49 ms | 31.7 dB | 16.0 | 16 |
| Y1 | fixed 32 | 83 ms | 34.6 dB | 32.0 | 32 |
| Y1 | fixed 64 | 152 ms | 37.2 dB | 64.0 | 64 |
| Y1 | fixed 128 | 281 ms | 40.1 dB | 128.0 | 128 |
| Y1 | adaptive 0.05 | 42 ms | 32.2 dB | 16.3 | 112 |
| Y1 | adaptive 0.01 | 72 ms | 35.3 dB | 29.4 | 256 |
| Y1 | adaptive 0.006 | 90 ms | 37.5 dB | 42.8 | 256 |
| Y1 | adaptive 0.003 | 137 ms | 40.2 dB | 73.9 | 256 |
| Y1 | adaptive 0.001 | 197 ms | 42.1 dB | 126.3 | 256 |
| Y1 | adaptive 0.0003 | 204 ms | 42.4 dB | 133.8 | 256 |
| Z1 | fixed 16 | 12 ms | 42.2 dB | 16.0 | 16 |
| Z1 | fixed 32 | 21 ms | 45.4 dB | 32.0 | 32 |
| Z1 | fixed 64 | 41 ms | 48.5 dB | 64.0 | 64 |
| Z1 | fixed 128 | 80 ms | 51.2 dB | 128.0 | 128 |
| Z1 | adaptive 0.05 | 20 ms | 42.5 dB | 16.0 | 24 |
| Z1 | adaptive 0.01 | 22 ms | 43.9 dB | 16.9 | 112 |
| Z1 | adaptive 0.006 | 21 ms | 44.6 dB | 17.9 | 176 |
| Z1 | adaptive 0.003 | 25 ms | 46.0 dB | 20.6 | 256 |
| Z1 | adaptive 0.001 | 29 ms | 47.4 dB | 28.3 | 256 |
| Z1 | adaptive 0.0003 | 37 ms | 50.6 dB | 48.5 | 256 |

Quickest adaptive render at or above fixed 64's PSNR: X1 0.001 in 75 ms against 112 ms (67%), Y1 0.006 in 90 ms against
152 ms (59%), Z1 0.0003 in 37 ms against 41 ms (91%).

## Pass size

`Adaptive pass size` on the RTX 3080 Ti at threshold 0.001 (wall time, PSNR; one run each):

| pass size | X1 | Y1 | Z1 |
| ---: | --- | --- | --- |
| 4 | 173 ms, 47.7 dB | 481 ms, 42.1 dB | 69 ms, 47.4 dB |
| 8 | 142 ms, 47.8 dB | 470 ms, 42.1 dB | 64 ms, 47.6 dB |
| 16 | 176 ms, 47.9 dB | 461 ms, 42.2 dB | 65 ms, 47.7 dB |
| 32 | 195 ms, 48.2 dB | 477 ms, 42.2 dB | 69 ms, 48.0 dB |

Pass size 8 is the fastest on X1 and Z1 and within 3% of the fastest on Y1. Smaller passes cost more dispatches; larger ones
let a pixel run past its stopping point, which buys a little PSNR (X1 +0.4 dB at 32) for more time. The default stays 8.

## What the numbers say

1. **The bar is met on the RTX 3080 Ti, scene by scene.** X1 reaches 47.8 dB, above fixed 64's 46.8 dB, in 141 ms against 230
   to 326 ms; Y1 matches fixed 64's 37.3 dB in 180 ms against 220 to 236 ms; Z1 passes fixed 64's 48.7 dB at 50.8 dB in 79 ms
   against 114 to 126 ms. Adaptive 0.01 went from slower than fixed 64 (v0.34) to 26 to 37% of its time on X1, 46% on Z1
   and 57% on Y1.
2. **The saving per scene depends on how much of the frame is hard.** X1 and Z1 are mostly easy pixels with a small noisy
   region, so adaptive spends 16 to 46 samples on average. Y1's hard sun and glossy floor keep its mean at 125 samples at
   0.001, so it needs a looser threshold (0.006, 42 samples on average) to beat fixed 64's time at fixed 64's quality.
3. **The AMD card gains less** (a tenth of the time at best, Y1 breaks even) and the software adapter gains most on X1 and Y1.
   One threshold does not give fixed 64's quality at lower time on every scene and adapter; 0.006 does on Y1 and not on Z1
   (45.1 dB against 48.7 dB), 0.0003 does on Z1 and costs more than fixed 64 on Y1.
4. **Where the time goes now** (`--breakdown`, RTX 3080 Ti, 1280 by 720, median of three renders, milliseconds). Everything
   that sampling cannot shorten adds up to about 44 ms on X1 and Y1 (scene build 5, pack 4.5, upload 1, final readback 10,
   post-processing 23) and about 34 ms on Z1. The rest is the card's own dispatch time, and it tracks the samples taken: X1 at
   0.001 is 85 ms of dispatch for 28 mean samples, Y1 at 0.001 360 ms for 125, against 173 ms and 169 ms for fixed 64's 64
   samples, so a sample costs the card 2.6 to 2.7 ms in a fixed render and 2.9 to 3.0 ms in an adaptive one (X1 and Y1), about a tenth more. The host's
   per-pass work, which ate the saving in v0.34 (40 ms of mask update, 15 ms of uniform writes and 28 ms of flag readback on
   X1 at 0.01), is now 4.5 to 7.4 ms of recording passes and no readback. **The phase that holds Y1 back** is therefore the
   shader dispatch itself: fixed 64's quality there costs 42 samples a pixel on average at the best threshold (0.006), and the
   per-sample price is about that of a fixed render. The CPU denoise filter (8.3 to 8.9 seconds per frame in the part 1 breakdown) is outside these wall times.

## Recommended default threshold

Start at **0.001** for a render that should look like fixed 64 (a pixel stops when its standard error is under about 3% of
its brightness) and leave `Max samples` at 256. On the RTX at 1280 by 720 it scores 47.8 dB on X1 (fixed 64: 46.8 dB in 43 to
60% of the time), 42.1 dB on Y1 (fixed 64: 37.3 dB, at about twice fixed 64's time) and 47.6 dB on Z1 (fixed 64: 48.7 dB, in
56% of the time). Go to 0.003 for a faster, noisier render (X1 45.0 dB in 103 ms, Z1 46.1 dB in 57 ms, Y1 40.0 dB in 297
ms) and to 0.0003 for a cleaner one. For a hard-lit scene like Y1 where the time matters more than the last decibels, 0.006
matches fixed 64 in about 80% of its time.
