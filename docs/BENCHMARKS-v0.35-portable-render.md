# v0.35 portable final-render benchmark — measurement pending

Lane 4, step P1, October 7, 2026. Baseline for the harness: `6807d16`, after Rendering 8; harness commit `abcce2e`. The integration suite still holds the shared GPU lock, and Fluids has the first short turn when it releases. No GPU timing from this step is claimed yet. The RTX 3080 Ti responded after the host reboot, but its render path has not been validated since the watchdog lock. Its status is **candidate-available, deferred until a bounded locked turn**.

## Method and exact rerun

`tools/benchmark_portable_render.py` uses X1, Y1 and Z1 from `tools/benchmark_adaptive.py`: each at fixed 64 samples and adaptive threshold 0.003. The sparse smoke plume comes from `tools/benchmark_volume_majorant.py`, at fixed 32 samples with the two-level grid and the single-box control. All cases use a fixed seed and a CPU-path-tracer image reference with the same settings. Quality is PSNR and mean/maximum absolute error on clipped, sRGB-encoded RGB. One warm-up and three timed renders produce a median wall time. Per-case JSON includes the adapter and backend, settings, seed, resolution, renderer phases and sample counts; smoke includes collision counts. CPU references are prepared before taking the GPU lock. Each GPU case is its own bounded subprocess and lock turn, saving JSON immediately. Timeouts and unavailable adapters are explicit records.

From the lane worktree, with the project virtualenv active, after the suite and Fluids first turn. First validate one RTX case in a short turn; only include it in the full matrix if that case succeeds without a new watchdog error:

```sh
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen python tools/benchmark_portable_render.py --matrix --adapters discrete --cases Z1-fixed64 --size 640x360 --out /tmp/nodebased-portable-p1 --timeout 120 --lock-wait 60
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen python tools/benchmark_portable_render.py --matrix --adapters integrated,cpu,discrete --size 640x360 --out /tmp/nodebased-portable-p1 --timeout 900 --lock-wait 60
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen python tools/benchmark_portable_render.py --report /tmp/nodebased-portable-p1
```

`--force` reruns already recorded cases, including unavailable or timed-out ones. Keep RTX to one bounded validation case first, and stop on a renewed watchdog error. The default matrix excludes RTX until it has passed that check.

## Measurements and next recommendation

Pending the lock. The existing v0.35 adaptive report indicates scene-dependent adaptive sampling: on AMD at 1280 by 720, X1 adaptive 0.003 was faster than fixed 64, while Y1 adaptive 0.003 was slower. That evidence ranks **Y1 adaptive sampling and its per-pixel work** as the first bottleneck to investigate; smoke majorant cost is second pending this matrix's collision and phase counters. This is a next-brief recommendation, not an optimization begun here. No portable P1 speed or quality claim, nor RTX validation, is established by this page yet.
