## 2026-09-26 — continuous mode merge: Lane 2 (2D parity) step S2a of 4: Backdrop and PostageStamp, Lane 4 (rendering) step C of 4: volume motion blur, shadows exchanged with the scene, and the control passes on the GPU, Lane 6 (fluids) step E of 5: FLIP liquids on the particle system, with surface extraction (integrator tick)

- **What was done:** `main` `7830399` -> `b15fab3` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b15fab3`: Ran 2654 tests in 1220.412 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

