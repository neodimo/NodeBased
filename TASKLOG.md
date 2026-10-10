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

