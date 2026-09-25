## 2026-09-24 — Lane 4 step A: Spot cone and falloff in every renderer

- **What was done:** CPU reference (`scene3d._shade_fragments` both loops, shadow rays, splat shadow
  rays), `gpu3d.py`, `gpurt_render.py` and `splatshade.py` apply `light_attenuation`. New tests in
  `tests/test_3d_spot_render.py`; targeted set of 373 tests OK. CPU Directional/Point renders hash-identical
  to the previous commit (checked with a stash A/B).
- **Committed vs scratch:** all committed on `openclaw/nb-3d-astra-lane`; nothing scratch.
- **Not done:** `viewportgpu.py` (lane 1) still lights a Spot like a Directional light. Request: add the
  cone terms (is spot, inner, outer, exponent) and falloff power to its 16-light table, treat Spot as positional
  (`place.w`), and multiply `radiance`/`specular` by the `attenuation()` function in `gpu3d.py`'s shader.
  `splatshade.shadow_catch` ignores cone and falloff (documented as a Known limit).
- **Next owner:** lane 1 for the viewport request; lane 4 continues at step B.

## 2026-09-24 — continuous mode merge: Lane 2 (2D parity) step 3b of 3: Defocus, DirBlur, DropShadow, Position, BlackOutside, AdjustBBox, Lane 3 (3D parity) step 4 of 4: MergeGeo3D, Normals3D, DisplaceGeo3D (integrator tick)

## 2026-09-24 — continuous mode merge: Lane 2 (2D parity) step 4c of 3: shared format registry, Grid, NoOp (integrator tick)

- **What was done:** `main` `a5a0c27` -> `777d3de` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `777d3de`: Ran 1910 tests in 1244.507 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

