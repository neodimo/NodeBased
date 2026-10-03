## 2026-10-03 — Lane 6 M2 performance finish (partial)

- **What was done (evidence):** Replaced particle-coordinate `unique(axis=0)` sorting in sparse-tile discovery with direct occupancy marking and neighborhood dilation on the bounded tile lattice. GPU transfer parity remained green. Warmed RTX 3080 Ti dam-break measurements (two warm-up, two timed steps) are 136/486/1,562 ms per substep at 64³/96³/128³; pressure took 15/62/200 ms. The 128³ target remains under 40 ms.
- **Artifacts:** Commit `1f8ca37` in `openclaw/nb-fluids-spike`; docs and bundled copies updated. Benchmark/profile/test logs are deliberate scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/` (`benchmark-L6-final-1003.log`, `profile-L6-gpu128.log`, `adapter-L6-*.log`, `tests-L6-docs-1003.log`), local and uncommitted by design.
- **State:** Partial. Transfer tests passed on RTX 3080 Ti, AMD Radeon 8060S Graphics and llvmpipe; docs checks passed. The 128³ under-40-ms gate remains unverified and missed by measurement.
- **Next owner + concrete artifact:** Gonzo owns the remaining performance work; start from `nodebased/flip_gpu_transfer.py` and the benchmark `tools/benchmark_flip3d.py --sizes 128 --steps 2 --warmup 2 --gpu`.
- **Failure mode:** None; this records an unmet performance target, not a correctness regression.

## 2026-10-03 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step M2 of 3: liquids an artist can scrub: GPU particle transfers, surface tension and open walls (finish 1) (integrator tick)

## 2026-10-03 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step D3 of 3: the provider contract, and a stand-in provider that proves the chain end to end (finish 1) (integrator tick)

- **What was done:** `main` `6963b47` -> `2943310` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `2943310`: Ran 4259 tests in 3255.947 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

