## 2026-10-11 — Lane 6 P3 of 3: lighter preview, fast by construction (PARTIAL)

- **What was done (evidence):** cached playback now reads a 48³ density/temperature artifact stored as 8-bit fields; liquid and whitewater companion states carry radii and cap at 20,000 points. Liquid surface extraction is bypassed while playing and full detail is requested on stop. Bounds and integrated density are restored on decode. Focused fluid, proxy, liquid surface, playback-control, and knowledge tests pass.
- **Artifacts:** commits `01a298a`, `215d022`, and `9ce5ef8`; synchronized `docs/FLUIDS_SPIKE.md` copies; production Radeon record `benchmarks/hot_pour/radeon-preview-P3-2026-10-11.json` (committed). Raw cache and adapter/test logs are temporary scratch under `scratch/nb-lanes/run/`, scheduled for cleanup before reporting.
- **State:** partial. Production Radeon cold/warm playback measured 4.07/4.32 FPS (245.7/231.5 ms load+draw). Tier-4 to tier-1 on frame 60 took 375.1 s (fetch/rebuild) plus 61 ms draw. The 80 ms and one-second goals fail. llvmpipe measurement is waiting behind an active full-suite lock and an earlier exclusive benchmark. P2 acceptance stays unchecked.
- **Next owner + concrete artifact:** Needs Gonzo: address production `FluidSurface3D` refinement and preview fetch cost; use the exact tier-4 to tier-1 path from `tools/benchmark_hot_pour.py` and finish the llvmpipe row.

## 2026-10-10 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step R2 of 3: holdout objects that cut CG out where something in the filmed plate is in front of it, Lane 6 (Fluids, GPT-6 Luna) step P2 of 2: interactive preview while scrubbing and playing (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step R5 fix: a click on a selected Roto node starts drawing again (integrator tick)

## 2026-10-11 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step R7 of 3: Roto feather grows outward from the edge, point by point, like Nuke (integrator tick)

- **What was done:** `main` `5d0d31b0` -> `115b85f2` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `115b85f2`: Ran 5181 tests in 15970.353 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
