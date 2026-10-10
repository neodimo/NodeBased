## 2026-10-10 — Lane 6 P2 finish: production preview benchmark (PARTIAL)

**What was done (evidence).** Ran the full Hot pour preset on the AMD Radeon 8060S: 256³ liquid, 192³ steam, 120 frames. Bake took 1,725.5 s; playback at tier 1/4 measured 0.30 median FPS (3,187.8 ms median fetch, 83.7 ms draw). Full detail measured 0.09 FPS. The production target of 12 FPS is missed. Focused tests passed: proxy volume bounds and integrated density, liquid silhouette bounds, stop restoring full tier/refinement, proxy control, and documentation bundle parity. The viewer's one-second test uses a synthetic frame; production-cache settle remains unverified.

**Artifacts.** Production measurements: `benchmarks/hot_pour/radeon-preview-production-2026-10-10.json` (committed with this note); Hot pour docs synchronized at `docs/FLUIDS_SPIKE.md` and `nodebased/data/docs/FLUIDS_SPIKE.md`. Raw run log and test log are at `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/L6-production-preview.log` and `tests-L6-finish-1010.log`. The 36.8 GB baked cache is deliberate scratch at `/tmp/hot-pour-preview-production-L6` pending cleanup.

**State.** PARTIAL. The production speed requirement is unmet; production-cache refinement within one second is unverified. Reduced 64³/48³ Radeon and llvmpipe rows remain in the P2 benchmark table.

**Next owner + concrete artifact.** Needs Gonzo: plan cache-resident proxy frames to eliminate repeated full-cache fetch/rebuild, then rerun the exact 256³/192³ workload with `tools/benchmark_hot_pour.py` and measure stop refinement against the production cache. Do not mark issue #6 P2 complete until both targets pass.

**Failure mode.** The previous partial handoff lacked the full production playback measurement. Proxying after evaluating each full-resolution cached frame does not make cache fetch/render fit the production target.

## 2026-10-10 — Lane 4 step R1 of 3: per-object render switches (Cast shadows, Receive shadows, Visible to camera)

- **What was done (evidence):** every mesh node (Card3D, Cube3D, Sphere3D, Cylinder3D, ReadGeo3D, ReadGLTF3D) has the three
  switches, all on; `Geometry` carries them and instance copies take their source's. Honoured by the CPU raster, CPU ray-traced,
  CPU path tracer, GPU raster, GPU ray-traced and GPU path tracer. `tests/test_3d_render_switches.py` (26 tests) passes on the RTX 3080 Ti,
  the Radeon 8060S (`force-adapter.py integrated`) and llvmpipe (`cpu`); the GPU path tracer matches the CPU one within the
  existing tolerance; 457 tests of the existing shadow, light-linking, instance, viewport, path-tracer and Cryptomatte modules pass
  (RTX). Old documents: golden pictures from the pre-change tree match (CPU bit for bit).
- **Artifacts:** committed on `openclaw/nb-3d-astra-lane`; docs/3D_FOUNDATION.md "Per-object render switches" (bundled copy identical),
  `tests/golden_render_switches.py` and `tests/data/golden/render_switches_*.npy`.
- **State:** DONE for all five parts. Unverified: Windows, real-display look of the knobs, CI. The test runs did not wait for
  `/tmp/nb-gpu.lock` (an exclusive waiter from another job had queued for 50 minutes, so a shared request would not return); they
  ran unlocked on the idle cards. Not honoured: GPU instanced ray tracer raises `Unsupported` (CPU fallback); viewport does not
  preview Receive shadows; Cryptomatte manifest still lists a hidden mesh's name.
- **Next owner:** Gonzo integrates; R2 (holdouts) and R3 (shadow catcher) read these switches.
- **Failure mode:** none open. A fair-queued exclusive `flock` blocks every later shared `flock`.

## 2026-10-10 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step R2 of 2: the 12 ms of host work around every GPU render (finish 1), Lane 6 (Fluids, GPT-6 Luna) step P2 of 2: interactive preview while scrubbing and playing, Lane 8 (2D parity B, GPT-6 Luna) step NL7 of 7: motion (integrator tick)

- **What was done:** `main` `9290977` -> `3bf3c00` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `3bf3c00`: Ran 4886 tests in 11963.644 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

