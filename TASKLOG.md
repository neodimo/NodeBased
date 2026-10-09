## 2026-10-09 — Lane 6 P1 cached playback finish (DONE)

- **What was done:** The queued reduced llvmpipe benchmark completed on the exclusive-lock path: 32/32 cells, 8 frames, 0.8 s bake, 42.78 FPS whole-scene and 46.83 FPS steam-only median. This run produced no particles or active steam tiles, so these throughput numbers describe empty-field playback. Full-size Radeon evidence remains 0.72 FPS steam-only / 0.09 FPS whole-scene; the frame-120 profile measured 1.653 s evaluation against 88.6 ms file read and 1.582 ms upload. The requested exact performance bound is documented; the 6 / 2 FPS targets remain unmet. The 65,536 MiB steam cache budget is committed from the prior pass.
- **Artifacts:** llvmpipe JSON `benchmarks/hot_pour/llvmpipe-reduced-2026-10-09.json`; synchronized `docs/SIMULATION.md` and `nodebased/data/docs/SIMULATION.md`; Radeon JSON remains `benchmarks/hot_pour/radeon-full-playback-2026-10-09.json`. All are committed in this worktree. Raw run evidence remains in the workspace scratch run directory.
- **Tests:** Hot pour, simcache, playback, and knowledge modules passed under the exclusive GPU lock.
- **State:** Done against the spec's measured-bounds alternative; RTX full-size row remains pending the card reset.
- **Next owner + concrete artifact:** Gonzo, for P2; use the original Lane 6 brief and `docs/SIMULATION.md` Hot pour section as the baseline.

## 2026-10-09 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step NL4 of 7: icons and the left column, Lane 6 (Fluids, GPT-6 Luna) step P1 of 2: cached playback that is not bound by reading the cache (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step NL6b of 7: Properties closer to the mockup (integrator tick)

- **What was done:** `main` `d1fffb9` -> `d4c5a89` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `d4c5a89`: Ran 4834 tests in 10344.487 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

