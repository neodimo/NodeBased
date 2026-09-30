## 2026-09-30 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5) step I1 of 2: M2 gate: startup, time to first pixel and interaction latency, measured and gated, Lane 4 (Rendering, Claude Sonnet 5) step Y2 of 2: viewport shadows from every light, from splats and from instanced copies, Lane 8 (2D parity B, GPT-6 Luna) step J2 of 2: GridWarpTracker local motion and occlusion-aware point rejection (integrator tick)

- **What was done:** `main` `ef6bc94` -> `14c94ce` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `14c94ce`: Ran 3887 tests in 2470.564 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

