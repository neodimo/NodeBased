## 2026-09-30 — Lane 2 I2 review and desktop cache-budget follow-through (Gonzo)

- **What was done:** reviewed the isolated worker's I2 deliverable against `L2-m2b.md`. The worker's benchmark and gate exercised an opt-in combined cache budget, while the desktop app still used separate cache budgets. Wired the desktop preview's raster evaluator and tile cache to the shared 4 GiB default budget at `869a13e`; corrected the gate docs to distinguish cache bytes from process RSS. The worker's first two commits are `035b9ba` and `324bccd`.
- **Artifacts:** `869a13e` on `openclaw/nb-2d-parity` (committed locally, awaiting integrator merge); `docs/BENCHMARKS-v0.33-m2.md`, `docs/M1_GATE.md`, `nodebased/app.py`, `tests/test_desktop.py`. The worker's full report is deliberate scratch at `scratch/nb-lanes/run/report-L2.md`. The 8K benchmark's 14+ GB process-RSS peak is an observed limitation, not a 4 GiB result.
- **Evidence:** `tests.test_desktop` passed 153 tests on the app change (64.6 s); the newly added desktop budget-wiring check passed separately. Worker-reported 8K cache/recompute and 4K throughput tests passed; independent integrator suite for I2 has not run yet.
- **State:** I2 lane work committed, integration and independent full suite pending. The 4 GiB ceiling applies to the two accounted caches, not total process memory; whole-process 8K peak-RSS control remains open.
- **Next owner + concrete artifact:** integrator picks up `openclaw/nb-2d-parity` at `869a13e` for targeted and full suite. Gonzo tracks its merge and the separate process-RSS gap in `docs/M1_GATE.md`.
- **Failure mode:** a cache-bytes assertion was presented as a general memory ceiling while the production desktop constructor did not use that cache configuration. Check actual app wiring and distinguish process RSS from cache accounting before closing performance gates.

## 2026-09-30 — Lane 2 (2D parity, Claude Sonnet 5) step I2 of 2: M2 gate: 4K/8K memory ceilings and throughput (isolated worker)

- **What was done:** on `openclaw/nb-2d-parity` (worktree `nb-2d-parity`), starting from `75d27aa`
  (already at `main`'s tip `bf4d3ef` for that branch): `tools/benchmark_memory_throughput.py`
  (new) builds the ten-node graph the step's brief names (Read, Grade, Transform, Merge, Blur,
  Tracker, ColorCorrect with Roto feeding its `mask` input, Reformat, Write — Roto has no `image`
  input of its own) and plays a short real EXR sequence back through the full-frame evaluator and
  the tile-path executor at 4K and 8K, cold and warm on-disk cache; numbers recorded in
  `docs/BENCHMARKS-v0.33-m2.md`. `cachetier.SharedMemoryBudget` (new) ties the evaluator's raster
  cache and the tile executor's tile cache to one combined byte ceiling, opt-in via
  `TileExecutor(memory_budget=...)`, default 4096 MiB; each cache still evicts by its own least
  recent use, with the headroom either sees shrinking by whatever the other currently holds.
  `tests/test_memory_ceiling_gate.py` (new) drives four real 8K frames past a combined budget and
  asserts combined bytes stay within budget plus 10 percent and a post-eviction recompute is
  pixel-identical to the first render. `tests/test_m2_throughput_gate.py` (new) gates 4K tile-path
  fps at the measurement minus 30 percent headroom (0.29 fps measured, 0.20 fps budget), same
  skip-on-software-adapter convention as the I1 latency gate. One optimisation pass on the
  profile's dominant node (`Evaluator._resample`'s bilinear filter no longer reclips/revalidates
  each of the four corners independently), measured before/after on a standalone 4K array
  (~4-5 percent faster, checksum-identical output). `docs/M1_GATE.md` gets an M2 summary table.
  Commits `035b9ba` (code + tests), `324bccd` (docs) on `openclaw/nb-2d-parity`.
- **Not done:** the desktop app (`nodebased/app.py`) still uses two independent cache budgets, not
  the combined ceiling; only one of the ten graph nodes got an optimisation pass; no gated fps
  floor for 8K, only the measured table. Issue #2 checklist and board updated (Paused, awaiting
  the integrator); full report in `scratch/nb-lanes/run/report-L2.md`.
- **Evidence:** targeted tests green (`test_memory_ceiling_gate`, `test_m2_throughput_gate`,
  `test_golden_2d`/`test_2d_parity_v3_effects`/`test_transform_handle_ui` — 23, `test_imaging`/
  `test_2d_parity_group_3b`/`test_2d_parity_step_5c`/`test_phase_a`/`test_tiers`/`test_roto` — 224,
  `test_cachetier`/`test_cacheinspector`/`test_cache_correctness`/`test_tileexec` — 68). Full
  `tests/test_desktop.py`: 153 tests in 64.4 s, OK. Logs under `scratch/nb-lanes/run/tests-L2-*.log`.
## 2026-09-30 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step J2 of 2: GridWarpTracker local motion and occlusion-aware point rejection (finish 1) (integrator tick)

- **What was done:** `main` `75d27aa` -> `43ed16e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `43ed16e`: Ran 3890 tests in 2473.210 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
## 2026-09-30 — Lane 8 plan closed; Sonnet lane starts recovered (Gonzo)

- **What was done:** Lane 8 J2 finish was merged to `main` as `50708c0` at 1:55 PM, with 3,890 tests green at the stacked tip. Its plan is complete and paused pending DiMo's next direction. Lane 2 I2 and Lane 4 Y2 finish had both failed before their first reply at 12:16 PM because Sonnet hit its session limit. After reset, PID 843954 was only the Lane 2 dashboard follow-up, which made no edits; it was mistakenly identified as an independent worker. Gonzo then started one isolated Lane 2 Sonnet worker in its worktree (`agent:main:subagent:a70a3b2f-687b-4f4a-97c2-7469781eca98`) and recorded its real start. The Lane 4 dashboard follow-up did complete its Y2 finish, committing `d32deee`; its independent integrator suite is running.
- **Artifacts:** `scratch/nb-lanes/auto/state.json` has the lane notes; `scratch/nb-lanes/run/lane8-0930-1246.log` has the J2 finish report. This note and `context/state.md` are committed project records; no scratch artifact is claimed as committed.
- **State:** Lane 8 done for plan 6; display-level QA and Windows/CI verification remain unverified. Lane 2 I2 is running in the isolated worker, with output/tests pending. Lane 4 has a lane-branch commit and its independent suite/merge are pending.
- **Next owner + concrete artifact:** Gonzo checks the Lane 2/Lane 4 sessions and the next `python3 scratch/nb-lanes/auto/tick.py tick` result, then performs GridWarpTracker display QA. DiMo chooses any next Lane 8 plan.
- **Failure mode:** A failed Sonnet subagent was marked Live without doing work. Its own resumed Claude PID was mistaken for a separate lane worker, and that unverified interpretation reached the state note and channel. Inspect the session identity, worktree, report and commit—not just a PID or a `running` label—before declaring a lane recovered.
