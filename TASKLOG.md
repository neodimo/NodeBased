## 2026-09-25 — Lane 4 step C of 4: kept specular (partial)

- **What was done (tested):** `ReadSplat3D` `Keep specular` (`splat_specular`, 0..1, default 0, old documents load at 0). At k > 0 relit
  splats keep the capture's view-dependent SH residual as a highlight and the shadow catcher multiplies only the diffuse part
  (`splatshade.py`). Repro and pinned reference in `tests/test_3d_splat_specular.py` (peak luminance 3.36 direct, 0.0765 relit with it off);
  GPU relit splats match the CPU; mesh side: `Relight` `Specular` 0 removes and 1 restores the highlight of a shiny sphere.
  Docs: "Specular (splats)" in `docs/3D_FOUNDATION.md` (bundled copy identical). Targeted tests: 89 + 128 OK.
- **Not done:** per-light splat specular layers in the relight bundle (the bundle still rejects scenes with splats), a computed
  (light-driven) splat highlight, GPU shadow catching (step D). Committed on `openclaw/nb-3d-astra-lane`.
- **Next owner:** L4 step D; a bundle-with-splats design needs a splat material first.

## 2026-09-25 — continuous mode merge: Lane 4 (rendering) step B of 4: shadow offset and blur controls (integrator tick)

- **What was done:** `main` `947338c` -> `1883fff` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1883fff`: Ran 1937 tests in 1242.257 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

