## 2026-10-08 — Lane 4 P1 finish 1: portable render benchmark measured, DONE (6:10 AM PDT)

- **What was done:** ran the full 8-case matrix on AMD Radeon 8060S, llvmpipe and (after one bounded Z1 case) the RTX 3080 Ti at 640x360; wrote measurements, ranked recommendation and limits into `docs/BENCHMARKS-v0.35-portable-render.md` (bundled copy identical); committed the 24 case JSON files in `benchmarks/portable_render/`. Evidence: the JSON files and tables. Inference: Y1 adaptive is the first bottleneck (more mean samples than fixed, loses on AMD and llvmpipe); RTX smoke is bound by something other than collisions.
- **State:** all committed on the lane branch; run scratch is under the run directory p1-amd, p1-cpu, p1-rtx (not tracked). Targeted tests: tests.test_knowledge and tests.test_benchmark_portable_render, 15 OK.
- **Unverified:** kernel log was unreadable from the worker, so absence of a new NVRM watchdog error rests on all eight RTX cases completing; one resolution, one seed, three timed frames.
- **Next owner:** Gonzo, write the next rendering brief from the ranked list in the doc. Failure mode: queued lock waits that outlive their timeout write "unavailable" records; delete them and rerun.

## 2026-10-08 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step P1 of 1: portable final-render benchmark on the working adapters, Lane 6 (Fluids, GPT-6 Luna) step N3 of 3: one production-sized scene that uses everything, and its numbers (finish 2) (integrator tick)

## 2026-10-08 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step AA2 of 2: new nodes land in the stream the way Nuke places them, Lane 6 (Fluids, GPT-6 Luna) step N3 of 3: one production-sized scene that uses everything, and its numbers (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step R4 of 2: Roto shows the plate before you draw, and the Tracker lists its points (integrator tick)

- **What was done:** `main` `84f1a8c` -> `7666387` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7666387`: Ran 4703 tests in 6509.807 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

