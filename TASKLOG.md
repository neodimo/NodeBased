## 2026-10-10 — Lane 4 step R3 of 3: shadow catcher floor (DONE, one stated limit)

- **What was done (evidence):** the **Shadow catcher** switch on every mesh node (off by default; `Geometry.shadow_catcher`). A catcher renders black
  with alpha = 1 - shadowed/unshadowed light (ambient, environments and every light, measured on a white diffuse surface with the same samples
  for both, so an unshadowed point is exactly 0). Every renderer reaches it through `nodebased/shadowcatch.py`: the picture is drawn with each
  catcher as a holdout (step R2), the shadow layer is laid under it (coverage from the CPU raster's position/normal/id passes on the sample grid,
  shadow from the CPU raster's own shadow code for the raster and ray-traced modes and the path tracer's own light sampling for the path-traced
  modes). CPU raster, ray-traced, path tracer and GPU raster, ray-traced, path tracer all honour it. Checked against the "render the floor twice and
  divide" reference (shadowed floor with the object hidden from camera but casting, over the unshadowed floor): mean alpha difference 0.0000 in
  the raster and ray-traced modes, 0.0005 to 0.003 in the path tracer. Depth/normals/position/uv see the floor, the returned depth buffer too;
  the object-id pass and Cryptomatte (manifest included) do not. A nearer holdout cuts the layer; a translucent object lets the shadow show through.
  `tests/test_3d_shadow_catcher.py` (52 tests) OK on the RTX 3080 Ti, the Radeon 8060S (`force-adapter.py integrated`) and llvmpipe
  (`force-adapter.py cpu`) together with `tests.test_3d_splat_shadow_catch`; 334 related tests (knobs, tiers, core, bypass, catalog, knowledge,
  labels, panels, Cryptomatte, holdout, render switches, light linking, motion blur, GPU splat shadows, smoke shadow exchange, animation) and 387
  GPU, path tracer, ray-trace and viewport-mode tests OK on the RTX.
- **Artifacts (all committed on openclaw/nb-3d-astra-lane):** `nodebased/shadowcatch.py`; hooks in `scene3d.render`, `gpu3d.render`,
  `pathtrace.render`, `gpupathtrace.render`; `core.py`/`knobs.py`/`imaging.py` (switch, label, glTF); `cryptomatte3d.py`;
  `tests/test_3d_shadow_catcher.py`; `examples/shadow_catcher/shadow_catcher.nbcomp` (rebuilt by `tools/make_shadow_catcher_example.py`, uses
  `examples/holdout_plate.png`); `docs/images/shadow_catcher_recipe.png` (`tools/render_shadow_catcher_reference.py`); docs/3D_FOUNDATION.md
  sections "Shadow catcher floors" and "Putting CG onto a plate" (bundled copy identical). No schema bump.
- **State:** done, with one stated limit: the CG object's coloured light bounce onto the floor (colour bleed, reflection) is **not** caught as a
  tinted layer (the brief allowed leaving it out). Other limits: the shadow layer is measured on the CPU for every renderer (a very large
  scene's CPU shadow budget applies even with a GPU picture); the grid is capped at 12 million sub-pixels, so a frame near 4K is measured on a
  coarser grid; a catcher with a holdout-incompatible scene raises as a holdout does; no motion blur per-sub-frame shadow in the CPU path tracer
  (one layer at the middle of the shutter). Unverified: Windows, CI, the knob on a real display.
- **Next owner:** Gonzo integrates (no push from the lane). Nothing is queued after R3 in this plan.

## 2026-10-10 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step R2 of 3: holdout objects that cut CG out where something in the filmed plate is in front of it, Lane 6 (Fluids, GPT-6 Luna) step P2 of 2: interactive preview while scrubbing and playing (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step R5 fix: a click on a selected Roto node starts drawing again (integrator tick)

- **What was done:** `main` `baa86c1` -> `3a059dd` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `3a059dd`: Ran 4974 tests in 12909.718 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

