## 2026-10-07 — Lane 4 P1 portable rendering benchmark, PARTIAL (9:19 PM PDT)

Evidence: committed harness `abcce2e` persists one JSON per case and compares against a CPU path-tracer image. Follow-up fixes run exactly one warm-up, prepare CPU references outside the GPU lock, lock llvmpipe too, and record failed reference preparation. Targeted benchmark and knowledge tests passed (15 tests); the mirrored benchmark doc is byte-identical. CPU references for all eight cases at 64 by 36 exist only in local scratch, for harness preparation, and are not timing evidence. The integration suite still holds `/tmp/nb-gpu.lock`, so no P1 adapter benchmark has been run. The RTX responded after reboot but remains unvalidated by this step; no new watchdog probe was made.

Status: PARTIAL. Committed artifacts: benchmark tool, its tests, `docs/BENCHMARKS-v0.35-portable-render.md` and bundled copy. Local-only artifacts: `/tmp`-style scratch references and test logs, outside the repo. Needs Gonzo: after the suite releases the lock and Fluids gets its first short turn, run the exact bounded commands in the benchmark doc, review per-case JSON, fill measured tables and revise the bottleneck ranking. Failure mode: the original harness held the lock while preparing CPU references and warmed twice; both are fixed in this step. Full suite, AMD, llvmpipe and RTX P1 matrix remain unverified.

## 2026-10-07 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step Z2 of 2: shortcut-created nodes are wired to the selection like Nuke, Lane 8 (2D parity B, GPT-6 Luna) step R2 of 2: animating Roto shapes (integrator tick)

## 2026-10-07 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step AA1 of 2: S makes a Shuffle, Merge passes B through, numbers keep all their digits, Lane 8 (2D parity B, GPT-6 Luna) step R3 of 2: the Roto panel shows what is keyed and stays where you are (integrator tick)

- **What was done:** `main` `6807d16` -> `446ae02` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `446ae02`: Ran 4684 tests in 6322.780 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

