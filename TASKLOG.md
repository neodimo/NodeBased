## 2026-10-10 — Lane 6 P2 production follow-up (PARTIAL)

- **What was done (evidence):** Added persistent tier-specific companion caches for reduced liquid-particle, whitewater-particle, and cached-volume frames; proxy markers keep `proxy_scene` from reducing an already reduced volume twice. Preview and full scene bounds/density tests, proxy controls, and stop-tier UI tests pass (91 focused tests); documentation parity passes (7 tests). The 120-frame Radeon production run completed: full detail 0.14 median FPS; cold 1/4 preview 0.49 median FPS (1,974.7 ms fetch + 74.5 ms draw). The 12 FPS target remains missed. Production preview-to-full-detail stop latency was not directly captured; the 7,093.4 ms full-detail fetch median does not establish the one-second target. The synthetic stop test still passes.
- **Artifacts:** Code/tests/docs and `benchmarks/hot_pour/radeon-preview-production-followup-2026-10-10.json` are committed locally on `openclaw/nb-fluids-spike`; not pushed. Detailed handoff: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L6.md`. Raw benchmark: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/L6-production-preview-v2.log`; final test logs: `tests-L6-focused-final.log`, `tests-L6-knowledge-final.log`. `/tmp/hot-pour-preview-production-L6-v2` (about 35 GB, deliberate scratch) is retained for next measurement; cleanup handoff after it is no longer needed: `gio trash /tmp/hot-pour-preview-production-L6-v2`. The exploratory warm-repeat probe was interrupted at frame 100 after 29.9 s; it produced no warm/stop result and is not cited as performance evidence.
- **State:** Partial. Cold production preview misses 12 FPS; production stop transition within one second remains unverified. Issue #6 stays unchecked.
- **Next owner + artifact:** Gonzo owns the next optimization/measurement; use `tools/benchmark_hot_pour.py --reuse-cache --playback --preview-tier 4` with the retained production cache, then instrument a real tier-4-to-tier-1 stop at a representative frame. Do not mark P2 done until both acceptance targets pass.

## 2026-10-10 — Lane 4 step R2 of 3: holdout objects that cut CG out where something in the filmed plate is in front of it

- **What was done (evidence):** a per-mesh **Holdout** switch (off by default) beside R1's switches on Card3D, Cube3D, Sphere3D,
  Cylinder3D, ReadGeo3D and ReadGLTF3D; `Geometry.holdout`, instance copies take their source's. A holdout hides what is behind it
  from the camera and is itself zero colour, zero alpha and no depth, position, normal, uv or id, in the CPU raster, CPU ray-traced
  mode (also with the thin lens), the CPU path tracer (also with a shutter), the GPU raster, the GPU ray-traced mode and the GPU
  path tracer, and in Cryptomatte (coverage of what it hides is reduced by its coverage). Splats, particles and smoke behind it are cut
  (CPU modes and both path tracers); in front of it they stay. It casts the shadow its Cast shadows switch gives it and is not met
  by reflection, refraction or bounce rays. `tests/test_3d_holdout.py` (46 tests): RTX 3080 Ti, Radeon 8060S
  (`force-adapter.py integrated`) and llvmpipe (`cpu`) all OK; with `tests.test_3d_render_switches` 72 tests OK on the RTX. The GPU
  raster and ray-traced modes match the CPU's modes (5e-4), the GPU path tracer the CPU one (1% of the mean); the 101 `test_3d_*`
  modules (1644 tests, RTX) and the knob, tier, core, bypass, catalog, knowledge, label and motion-blur modules (125 tests) pass. The
  existing GPU modules (260 tests) also pass on the Radeon and llvmpipe, except nine adapter-routing tests that mock `wgpu`
  (`tests.test_3d_gpu.NoGPURequired`, `tests.test_3d_gpu_rt_integration.RoutingTests`): they fail only under `force-adapter.py`,
  whose wrapper replaces `gpu3d._state`, and pass in the default run (inference: not run on the unchanged tree).
- **Artifacts (all committed on `openclaw/nb-3d-astra-lane`):** docs/3D_FOUNDATION.md "Holdouts" (bundled copy identical),
  `docs/images/holdout_recipe.png`, `examples/holdout_plate.png` (made by `tools/render_holdout_reference.py` from a fixed seed, no
  licence beyond the repository's; step R3 reuses it), `tests/test_3d_holdout.py`; code in scene3d.py, pathtrace.py, splatraster.py,
  motionblur.py, gpu3d.py, gpurt_render.py, gpupathtrace.py, gpuinstance.py, viewport3d.py, imaging.py, core.py, knobs.py.
- **State:** DONE, all four parts. No schema bump (`holdout` is read with a default of off; old documents render bit for bit, the R1
  golden pictures pass). Raises instead of drawing the holdout: the GPU raster and ray-traced modes and the GPU instanced tracer
  raise `gpu3d.Unsupported` for a holdout with splats, particles, smoke, a liquid, or an environment or area light the camera sees
  (`auto` falls back to the CPU); the CPU raster and ray-traced modes raise `ValueError` for a holdout with a liquid or with splat
  reflections or bounce light. Unverified: Windows, CI, the real-display look of the knob. Not done: the viewport draws a holdout as
  the stand-in it is (no preview of the cut); Cryptomatte's manifest still lists the holdout's name (its id never has coverage).
- **Next owner:** Gonzo integrates; R3 (shadow catcher) can read `Geometry.holdout` and reuse `examples/holdout_plate.png`.
- **Failure mode:** none open. A path tracer scene with a holdout samples every light by next-event estimation alone (like the other
  switches); a transparent pane of alpha .5 in the CPU path tracer reads alpha .25 over nothing (existing behaviour, not changed).

## 2026-10-10 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) Gonzo's GUI sprint: finish the new look against the mockup, Lane 4 (Rendering, Claude Sonnet 5.5) step R1 of 3: each object can be told not to cast shadows, not to take shadows, or not to be seen by the camera, Lane 6 (Fluids, GPT-6 Luna) step P2 of 2: interactive preview while scrubbing and playing (finish 1) (integrator tick)

- **What was done:** `main` `c020687` -> `6006290` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `6006290`: Ran 4919 tests in 11710.044 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
