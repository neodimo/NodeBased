## 2026-10-04 — Lane 8 E3 finish: shared artifact writes and stable ControlBundle exports

- **What was done:** Made content-addressed blob and metadata changes atomic and serialized per artifact across threads/processes; changed post-handshake worker failures to return a failure reply; fixed the ControlBundle EXR `DateTime` so repeat exports retain the same artifact ID.
- **Evidence:** The mixed thread/process artifact stress test and the remote malformed-job test pass. The remote shared-cache regression passed 20 consecutive runs with six busy Python loops. The required jobs/artifacts/workers and ControlBundle-related tests passed together (41 tests).
- **Artifacts:** Commits `108bc78` and `4856d0f` on `openclaw/nb-2d-parity-b`; not pushed. Test logs: `scratch/nb-lanes/run/tests-L8-E3-part1.log`, `scratch/nb-lanes/run/tests-L8-E3-part2.log`, `scratch/nb-lanes/run/tests-L8-E3-loaded-20.log`, and `scratch/nb-lanes/run/tests-L8-E3-final.log` (deliberate local scratch).
- **State:** Done on the lane branch. Full-suite integration, Windows execution, and Animal validation remain unverified.
- **Next owner + artifact:** Gonzo; integrate `openclaw/nb-2d-parity-b` at `5a7a8f8` and use the listed targeted logs. Validate the worker protocol on Animal/Windows when available.
- **Failure mode:** In-place blob/meta writes let concurrent workers observe truncated or interleaved metadata, and the server silently closed on JSON parse errors. Atomic replacement plus a per-artifact lock and explicit job failure replies address those paths.

## 2026-10-03 — Lane 8 E3: job chains and LAN workers

- **What was done:** Added a durable SQLite queue with dependency chains, retries, restart reuse, cancellation, priorities and artifact provenance. Added a queue panel and TCP workers using shared-secret authentication and content-addressed artifact synchronization. Documented the Animal worker command and its untested status.
- **Evidence:** `tests.test_jobs`, `tests.test_queuepanel`, `tests.test_workers`, `tests.test_artifacts`, `tests.test_knowledge` and `tests.test_desktop` passed together: 210 tests. The bundled generative guide is byte-identical.
- **Artifacts:** Commits `b06441b` and `86ccca9` on `openclaw/nb-2d-parity-b`; not pushed. Code: `nodebased/jobs.py`, `nodebased/queuepanel.py`, `nodebased/workers.py`, `nodebased/app.py`. Docs: `docs/GENERATIVE_CONDITIONING.md` and its bundled copy. Test log `scratch/nb-lanes/run/tests-L8-final.log` is deliberate local scratch.
- **State:** Done on the lane branch; integration and Animal/Windows validation remain unverified.
- **Next owner + artifact:** Gonzo; integrate `openclaw/nb-2d-parity-b` at `86ccca9`, use `scratch/nb-lanes/run/tests-L8-final.log`, then try `python -m nodebased.workers serve --bind <host:port>` on Animal.
- **Failure mode:** One earlier aggregate run reported a remote-fixture failure; the isolated remote test and final aggregate rerun passed. Cause was not reproduced.

## 2026-10-03 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step M3 of 3: several fluids in one scene, and playback that stays sparse end to end (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step E2 of 3: providers run in an isolated worker process, with progress, cancel and logs (integrator tick)

- **What was done:** `main` `f68e790` -> `f14170d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `f14170d`: Ran 4275 tests in 3294.422 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
