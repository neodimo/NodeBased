## 2026-09-29 — Lane 6 H2 finish: sparse up-res and smoke skipping (branch handoff)

- **What was done (evidence):** CPU and GPU up-res reconstruct and advance only retained density/fuel tiles, with guided velocity, prior-frame transport, and seeded detail; stored tile fields used 13–17% of dense field bytes on the measured plumes. At 128³→512³, sparse GPU reconstruction took 0.653 s versus 1.862 s dense on the same input. A coarse-majorant GPU smoke path skips empty cells; a warmed 256³ plume render fell from 1.204 to 0.397 ms/sample. The 64-sample image difference was below baseline seed noise. Targeted tests passed on NVIDIA GeForce RTX 3080 Ti, AMD Radeon 8060S Graphics, and llvmpipe.
- **Artifacts:** `nodebased/fluid_upres.py`, `nodebased/fluid_upres_gpu.py`, `nodebased/gpupathtrace.py`, their targeted tests, reproducible scripts under `tools/`, and `docs/FLUIDS_SPIKE.md` with its byte-identical bundled copy. All intended artifacts are committed on `openclaw/nb-fluids-spike`; nothing was pushed by this lane.
- **State:** Done for H2's requested code, measurements, and adapter checks. Existing `Volume` consumers expand sparse tiles to dense arrays, so full playback memory stays higher than stored-frame memory. Full integration suite, Windows, and real-display visual QA are unverified here.
- **Next owner + concrete artifact:** Gonzo reviews and integrates branch `openclaw/nb-fluids-spike` using `docs/FLUIDS_SPIKE.md` H2 and the targeted tests. A later renderer/storage pass can remove dense `Volume` expansion.
- **Failure mode corrected:** The first H2 pass recorded dense baselines without building sparse compute or path-trace skipping. This finish pass implements both and preserves the original baseline table.

## 2026-09-29 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5) step G1: splats and smoke in the GPU path tracer give wrong pictures on AMD and on Windows' software driver, Lane 6 (Fluids, GPT-6 Luna) step H2 of 2: big grids: the sparse upres path, measured at production sizes (integrator tick)

## 2026-09-29 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5) step D1 of 2: close the partial rows: Roto and RotoPaint, DustBust speck detection, CurveTool, HSVTool, CrossTalk (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step F1 of 2: OFlow, VectorToMotion and the rest of the motion blur family (finish 1) (integrator tick)

- **What was done:** `main` `d31cdcd` -> `cbb1ecf` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `cbb1ecf`: Ran 3642 tests in 2379.074 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

