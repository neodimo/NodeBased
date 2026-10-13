## 2026-10-12 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) the release fix: the checks on main pass again on Linux and Windows (playback proxy wait, Windows dock, rail and knob-row metrics, the GPU lock import, the agent session's settings, UTF-8 source scan), Lane 3 (Research, GPT-6 Astra) step T1 of 3: the research radar, generative video and temporal models for the shot pipeline, Lane 4 (Rendering, Claude Sonnet 5.5) step L2 of 3: the camera solve recovers the lens, and survey points picked in the viewer fix the scale, Lane 6 (Fluids, GPT-6 Luna) step P4 of 4: Hot pour preview at 12 frames per second, and a stop that settles within a second, Lane 8 (2D parity B, GPT-6 Luna) step S2 of 3: planar tracking, a tracked plane that drives CornerPin and Roto (integrator tick)

- **What was done:** `main` `413023bf` -> `f4e96654` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `f4e96654`: Ran 5283 tests in 18885.651 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

