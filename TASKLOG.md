## 2026-10-09 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step R2 of 2: the 12 ms of host work around every GPU render, Lane 6 (Fluids, GPT-6 Luna) step P1 of 2: cached playback that is not bound by reading the cache (finish 1) (integrator tick)

- **What was done:** `main` `05b2658` -> `8cbaeef` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8cbaeef`: Ran 4847 tests in 10457.366 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

