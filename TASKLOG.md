## 2026-09-26 — continuous mode merge: Lane 2 (2D parity) step K3 of 3: Cryptomatte, Lane 4 (rendering) step A of 4: GPU volume rendering with real lighting, in Render3D and the viewport, Lane 6 (fluids) step D of 5: the GPU-resident solver (multigrid pressure, GPU advection, sparse tiles), Lane 8 (2D parity B, GPT-6 Luna) step F1a of 3: Matrix (3x3) and Laplacian (integrator tick)

- **What was done:** `main` `37ecb72` -> `18fede7` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `18fede7`: Ran 2568 tests in 1257.994 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

