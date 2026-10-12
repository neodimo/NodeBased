## 2026-10-12 — Lane 8 S2 of 3: PlanarTracker (PARTIAL)

- **What was done (evidence):** Added deterministic normalized-DLT homography fitting with fixed-seed RANSAC and synthetic tests for projective corner recovery, repeatability, and 25% outlier rejection.
- **Artifacts:** `nodebased/planar.py` and `tests/test_planar_tracking.py` are committed in `1dcb65b`; test log is `scratch/nb-lanes/run/tests-L8.log`.
- **State:** The fitting primitive passes its targeted tests. There is no PlanarTracker node or dense-flow feature matching integration yet; UI overlays, keyed corners, CornerPin export, Roto Planar follow, docs and full acceptance tests remain absent.
- **Next owner + concrete artifact:** Needs Gonzo to resume S2 in `openclaw/nb-2d-parity-b`, implementing feature correspondence from the existing flow, node registration/analysis, UI/export/follow, and docs.

## 2026-10-12 8:20 AM PDT — Lane 6 P4 of 4: hot-pour preview and stop (PARTIAL)

- **What was done:** Regenerated the 120-frame 256³ / 192³ Radeon production bake. A warmed frame-60 cProfile baseline took 397 ms, with fluid chain and geometry track work at the top. Implemented directional decoded-frame ring buffering, persistent volume texture updates, cached surface-mesh persistence and progressive tier-2 stop refinement. Production playback now measures 6.7 ms cold median / 7.8 ms p95 and 6.4 ms warm median / 8.3 ms p95 on Radeon; llvmpipe measures 23.0 / 31.3 ms cold and 22.1 / 29.6 ms warm. Targeted tests pass.
- **Artifacts:** source and tests committed in this worktree; profile/playback JSON and logs were temporary scratch under `scratch/nb-lanes/run/L6-p4-cache` (the 48.8 GB bake is removed before handoff); docs and bundled copy committed below. No public push.
- **State:** PARTIAL. Radeon playback meets the 80 ms budget. A production frame-1 stop entered whitewater reconstruction and did not reach tier 2 within two minutes; interrupted. Full stop timings for frames 1, 60 and 120 are unverified, as is the one-second target.
- **Next owner + concrete artifact:** Needs Gonzo: make the stopped preview reuse the production bake's liquid and whitewater artifacts, then run `tools/benchmark_hot_pour.py --reuse-cache --playback --stop-frames 1,60,120` against the production cache and document tier-2/full timings.
- **Failure mode:** playback benchmark's fresh stop evaluator fell through to simulation reconstruction for whitewater; timing it as surface refinement would misattribute solver work.

## 2026-10-12 — Lane 3 T1 of 3: generative-video research radar (DONE)

- **What was done — evidence:** read the stage 3/8 baseline and conditioning contract; searched the web and inspected primary model cards, repositories, licences and papers. Added a ranked, dated sweep covering the requested video families, small-card evidence, camera-control research and newer Wan/LTX/Hunyuan releases; separated published base-model numbers from unmeasured controlled workloads. Added mechanism comparisons, one bounded VACE brief and three non-scaling experiments. Four deliverable commits precede this final audit/handoff commit.
- **Artifacts:** `docs/RESEARCH_RADAR.md`, its byte-identical `nodebased/data/docs/RESEARCH_RADAR.md`, and `briefs/proposed/T1-vace-depth-reference.md`; committed locally on `openclaw/nb-research`, not pushed or integrated. Research source caches and per-part test logs were deliberate worktree scratch, discarded after verification; no weights or generated media exist from this task. Compact test evidence is retained here.
- **State:** documentation complete. `tests.test_knowledge` passed before each documentation commit (7 tests per run); final byte equality and local-link checks passed. No GPU work, model benchmark, visual review, Windows check, full-suite run or product implementation. Exact controlled VRAM/speed, byte determinism and verifier gains remain unverified. External searches were partly degraded; gated LTX-2.5 model-card text was unavailable, so its metadata/code evidence is explicitly scoped.
- **Next owner + concrete artifact:** Gonzo integrates this branch and decides whether to assign `briefs/proposed/T1-vace-depth-reference.md`. It specifies the fixed conditioning, immutable checkpoint, repeatability/quality gates, maximum active budget and unchanged-SDXL fallback. Update issue #3 and pause the lane board for integration; no further research step started.
- **Failure mode recorded:** the earlier stage 8 assessment reused base Wan T2V memory/timing as a VACE estimate and suggested an occlusion mask without establishing VACE edit-mask semantics. The radar/brief correct those assumptions. Related traps: download size is not VRAM; a small base benchmark does not prove a larger control adapter fits; SDXL AnimateDiff does not establish Union-ControlNet compatibility. Existing sprint history, parity table and roadmap are unchanged.

## 2026-10-12 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) Gonzo's end-to-end sprint, stage 8: temporal coherence, the generated shot stops flickering (flicker 15.4 to 2.6 display steps), Lane 6 (Fluids, GPT-6 Luna) step P4 of 4: Hot pour preview at 12 frames per second, and a stop that settles within a second, Lane 8 (2D parity B, GPT-6 Luna) step S1 of 3: open and stroked splines in Roto, like Nuke's open Bezier with a stroke width (integrator tick)

- **What was done:** `main` `3edf3774` -> `7330eb39` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7330eb39`: Ran 5238 tests in 18476.617 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

