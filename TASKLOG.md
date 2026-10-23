## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: an edit to the Hot pour shows its re-simulated first frames within a second (finish 2), Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 1) (integrator tick)

- **What was done:** `main` `b1aa277a` -> `1e750ce0` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1e750ce0`: Ran 5988 tests in 482.385 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step A finish 2 (PARTIAL)

- **What was done — evidence:** The 64-cell / 120-frame Radeon cup-move bake measured 29.06 s with float32/all steam-cache channels and 28.34 s with half-precision density+temperature, using the same warmed upstream setup and bounded per-node cache caps. The optimized measurement clears the 30 s gate; the 0.72 s difference is one paired measurement and may include timing variation. Cancellation after FLIP substep entry was measured in three trials on each adapter: RTX 3080 Ti median 0.554 ms / worst 21.443 ms; Radeon 8060S median 0.278 ms / worst 0.535 ms. New-edit first-frame solver time was at most 0.161 s RTX and 0.167 s Radeon. Real viewport presentation is not part of these timings.
- **Artifacts:** Cost harness `tools/benchmark_fluid_edits.py`; cancellation harness `tools/benchmark_fluid_edit_cancel.py`; final comparison JSONs `benchmarks/hot_pour/edit-cup-bounded-{baseline,optimized}-radeon-2026-10-22.json`, `edit-cancel-rtx-2026-10-22.json`, and `edit-cancel-radeon-2026-10-22.json`; updated Hot pour generator output and example graph; `docs/FLUIDS_SPIKE.md` and byte-identical bundled copy. These are committed in this worktree change. Targeted logs are in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/` and remain local scratch. All generated large cache directories from this pass were empty and removed; no production bake was started.
- **State:** PARTIAL. Cost and cancellation gates pass in the measured harnesses. Targeted GPU-liquid, rigid, preset, cache, surface and knowledge tests pass on RTX; resident solver tests pass on Radeon and llvmpipe. Real-display QA remains unverified because this session has no display session. Production-size refresh is deferred to step B by the brief; the historical cache is about 8.50 GB and free space was 5.7 GB.
- **Next owner + concrete artifact:** Gonzo owns real-display review of the timeline/stale indicator using the Lane 6 finish-1 UI and `docs/FLUIDS_SPIKE.md`. Step B owns a fresh production-size bake when scratch space permits.
- **Failure mode:** UI cancellation supplies `threading.Event`, while rigid pre-simulation expected `CancelToken.check()`; rigid stepping now accepts both interfaces. The first repeat benchmark reused a resident solver identity and missed instrumentation after trial one; each trial now receives a unique run identity. Generated preset/example files were regenerated after changing the preset builder.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: an edit to the Hot pour shows its re-simulated first frames within a second (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `f4aecaab` -> `68d134e1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `68d134e1`: Ran 5981 tests in 483.532 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step A finish 1 (PARTIAL)

- **What was done — evidence:** The timeline now shows the active simulation bake separately from ordinary display-cache marks. The previous picture stays visible with a stale indicator until the edited playhead frame replaces it. One background frame is queued at a time; forward playback stays within the contiguous solved range. A newer preview replaces queued work and signals active evaluation cancellation. The simulation identity is limited to the viewed solver dependency scope, so unrelated camera/light edits do not restart the bake.
- **Artifacts:** UI, queue, timeline, tests and both copies of `docs/FLUIDS_SPIKE.md` are committed in this worktree. No generated bake/cache artifact was created. Targeted UI log: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/tests-L6-finish-ui2.log`; docs log: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/tests-L6-finish-knowledge2.log`. Earlier CPU and llvmpipe logs are `tests-L6-finish-cpu.log` and `tests-L6-finish-llvmpipe.log` in the same run folder.
- **State:** PARTIAL. The UI/state tests, CPU and llvmpipe solver regressions, and docs check pass. The solver cost reduction has not been verified; Radeon cup movement remains 32.43 s versus a 30 s gate. A fresh production-size bake is deferred because available disk is 5.7 GB and the historical cache occupies about 8.5 GB. Real-display QA and measured worker-stop latency under 50 ms remain unverified.
- **Next owner + concrete artifact:** Gonzo owns real-display QA and the next measured optimization pass, using the warmed phase profiles and gate table in `docs/FLUIDS_SPIKE.md`. Regenerate the production-size baseline only when sufficient scratch disk is available.
- **Failure mode:** A broad “any edit to a visible fluid graph” invalidation would wastefully restart fluid baking for camera/light changes. Invalidation now checks the solver dependency scope; a regression test covers this distinction.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: an edit to the Hot pour shows its re-simulated first frames within a second, Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `6e5c5d12` -> `e4679e89` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e4679e89`: Ran 5974 tests in 488.078 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 2: the catcher shadow off the CPU), Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `09dd7655` -> `31880d7a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `31880d7a`: Ran 5971 tests in 504.316 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step C of 3: overlays and the pixel readout follow playback, and the Direct picture becomes a docked viewer DiMo can switch on, Lane 4 (Rendering, Claude Sonnet 5.5) step A of 2: find where 3D playback's time goes, commit the profile, and remove the biggest costs (integrator tick)

- **What was done:** `main` `0a388be9` -> `25f44534` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `25f44534`: Ran 5966 tests in 482.342 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `a5caf406` -> `4e23575d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4e23575d`: Ran 5864 tests in 475.996 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (integrator tick)

- **What was done:** `main` `907d56d7` -> `74523025` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `74523025`: Ran 5855 tests in 469.929 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step C of 3: Render mode settles in seconds: stop sampling where the picture is already clean, Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the drag picture within a third of a second, the scrub within a third, and playback at the set rate (finish 1) (integrator tick)

- **What was done:** `main` `e9534e70` -> `6eeba375` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `6eeba375`: Ran 5851 tests in 477.642 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step B finish 1 (DONE)

- **What was done — evidence:** Clipped solid-over paint work is byte-identical to full-tile rendering. A valid 1920 × 1800 virtual desktop exposed the Properties slider for an actual pointer drag. RTX 3080 Ti measurements pass all first-picture, maximum in-drag gap, full-tier settle, scrub, and playback gates. The completed table is in `docs/ROTO_TRACKING.md`; its bundled copy is byte-identical.
- **Artifacts:** Commits `9b150c7c`, `031f7e2b`, and `1c3f9bc5` are in this worktree. Test logs and field JSON are retained as local scratch in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/` (`L8-finish1-tests.log`, `L8-finish1-playback-tests.log`, `L8-finish1-lens-gpu.log`, and `live-picture-L8-finish1-rtx-tall.json`).
- **State:** DONE for the left-out finish scope. No off-site GPU/laptop run or real-display visual review is claimed.
- **Next owner + concrete artifact:** Gonzo reviews/integrates and performs real-display/Windows visual QA using `docs/ROTO_TRACKING.md` and the local RTX measurement JSON.
- **Failure mode:** A 1920 × 1080 virtual desktop placed the Properties control at y=1293, outside the screen, and produced no preview. Re-run with a virtual screen tall enough to contain the panel; the corrected 1920 × 1800 run measured the slider successfully.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (finish 1) (integrator tick)

- **What was done:** `main` `219263f7` -> `873cc57b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `873cc57b`: Ran 5819 tests in 483.785 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the drag picture within a third of a second, the scrub within a third, and playback at the set rate (integrator tick)

- **What was done:** `main` `031f9394` -> `b9fc0920` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b9fc0920`: Ran 5811 tests in 476.762 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `d40e3fb3` -> `4f678da1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4f678da1`: Ran 5806 tests in 496.606 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B finish 3 (PARTIAL)

- **What was done — evidence:** Fresh frame-60 and frame-90 stills now show a soft coherent steam plume, continuous liquid reaching the cup pool, a smooth pool, and a broad soft catcher shadow. RTX 3080 Ti and Radeon 8060S CPU/GPU comparisons passed on frames 1, 60 and 120. A fully converged 1920 × 1080 render at noise threshold 0.01 took 69.253 seconds; the best pass-size experiment took 68.235 seconds. Both miss the strict under-60-second target.
- **Artifacts:** Updated generator, graph, evidence runner, test, stills, profile JSON, and byte-identical docs copies in this worktree; large render cache is temporary scratch and was trashed before handoff.
- **State:** PARTIAL; the only remaining acceptance item is fully converged 1080p under 60 seconds.
- **Next owner + concrete artifact:** Gonzo coordinates a catcher/path-render cost reduction with L4 using `benchmarks/hot_pour/hot-pour-render-profile-2026-10-22.json`.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (finish 1) (integrator tick)

- **What was done:** `main` `66217360` -> `6434a596` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `6434a596`: Ran 5806 tests in 480.759 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `1e10ca44` -> `8d5fefe9` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8d5fefe9`: Ran 5805 tests in 490.616 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B finish 2: softer catcher, steam and strict 1080p gate still open (PARTIAL)

- **What was done — evidence:** Replaced the path-traced Point key with a 0.7 × 0.7 Rect key; fresh frames 60 and 90 show a visibly broader catcher shadow. Removed source noise and extra turbulence, then tuned volume density. The steam remains granular in both final stills. The full-cap 1080p render converged every pixel at noise 0.01 in 79.72 seconds. Lowering the nominal sample budget to 64 finished in 40.68 seconds but left 3.78% of pixels unconverged (max noise 0.52335). CPU/GPU comparisons passed at frames 1, 60 and 120 on the RTX 3080 Ti.
- **Artifacts:** `examples/hot_pour_shot/{hot_pour_shot.nbcomp,make_shot.py,render_evidence.py}`, `tests/test_hot_pour_shot.py`, both identical `docs/FLUIDS_SPIKE.md` copies, frames 60/90, and the benchmark JSONs are committed in this worktree. Generated caches under `workspace/scratch/nb-lanes/run/big-L6/finish5` through `finish10` were trashed after measurement. `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L6.md` is the handoff report.
- **State:** PARTIAL. Shadow softness passes visual review. Steam softness and settled 1080p under 60 seconds remain unverified/failed. A6000 and Windows are assigned to DiMo and are outside this pass.
- **Next owner + concrete artifact:** Gonzo to continue from `benchmarks/hot_pour/hot-pour-render-profile-2026-10-22.json`; reducing `pt_samples` caps adaptive convergence, so an under-minute fix needs the renderer-side catcher sampling cost reduced or another measured approach. L6 does not own `nodebased/shadowcatch.py`; coordinate that change with L4.
- **Failure mode:** `shadow_blur` on a Point light does not soften this path-traced catcher; the Rect light provides the actual penumbra.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `155e3974` -> `87791fd8` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `87791fd8`: Ran 5805 tests in 474.693 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step A: cold first-play profile and affinity-aware read-ahead (PARTIAL)

- **What was done — evidence:** Cold RTX profile found 22 serial Read decodes in 8 seconds (median 85.3 ms, p95 98.2 ms; 1.946 s total) and 22 RotoPaint passes (median 47.2 ms; 1.038 s total). The full-frame fallback did not consume decode-ahead, and optional `Roto.bg` was omitted from read traversal. Full Evaluator and tile paths now share immutable bounded Raster decodes, including named layers. Four-worker prefetch starved playback (15.564 / 15.571 / 7.617 fps); one worker restored discrete adapters near baseline. Disabling prefetch at four-or-fewer CPUs yielded 16.473 fps integrated with zero dropped frames. Final RTX / llvmpipe / integrated rates are 21.356 / 20.953 / 16.473 fps versus the 24 fps target.
- **Artifacts:** Commits `1711c3f6`, `0c67c6d7`, `16d4821c` on `openclaw/nb-2d-parity-b`, local and not pushed. Profile and final adapter JSONs live under `benchmarks/live-picture/`; final docs copies are byte-identical. Exact handoff: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L8.md`. The first integrated one-worker sample overlapped CPU tests and was trashed; its clean rerun is recorded separately.
- **Verification:** `tests.test_tileexec`, `tests.test_decodepool`, `tests.test_media`, `tests.test_boundingbox`, `tests.test_live_picture_bench`, `tests.test_roto_paint`: 114 passed. `py_compile`, `git diff --check`, docs-copy check passed.
- **State:** PARTIAL/BLOCKED: cold playback remains below 24 fps on every adapter.
- **Next owner + concrete artifact:** Gonzo continues using `benchmarks/live-picture/profile-1022-cold-rtx.json` and the final `after-*-cold-*.json` timelines; rerun all adapters under `/tmp/nb-gpu.lock`.
- **Failure mode:** Four simultaneous EXR decoders can starve preview/UI. Bound the pool using available CPU affinity. `NB_QA_PLATE` expects the sequence directory, not the filename pattern.

## 2026-10-22 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step B of 3: the mixed mesh and splat scene's first look under one second, and a sharper first look (finish 1) (integrator tick)

- **What was done:** `main` `9ce12b9a` -> `7f6acd4e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7f6acd4e`: Ran 5802 tests in 475.065 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `3bc0b4c3` -> `4b39823a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4b39823a`: Ran 5802 tests in 483.726 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (finish 1), Lane 4 (Rendering, Claude Sonnet 5.5) step B of 3: the mixed mesh and splat scene's first look under one second, and a sharper first look (integrator tick)

- **What was done:** `main` `5d612c6f` -> `644f3778` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `644f3778`: Ran 5802 tests in 472.720 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `e0963ac8` -> `89870cb0` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `89870cb0`: Ran 5766 tests in 543.776 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step A finish: corrected drag proof; playback cadence remains just under target (PARTIAL)

- **What was done — evidence:** The integrated stand-in's absent drag picture was a benchmark coordinate/window-size bug. Qt supplied logical pointer coordinates at 2× scaling, `xdotool` addressed physical X11 pixels, and the old 1920 × 1080 logical window exceeded the 1920 × 1080 Xvfb surface. The corrected harness scales pointer coordinates and sizes to the screen geometry. Fresh 8-second unprofiled runs delivered held-drag frames on RTX, llvmpipe and integrated stand-in; first picture from the first Qt-received move was 0.198 s, 0.193 s and 0.177 s. Cold tier-4 playback was 21.330, 20.979 and 15.463 fps. On separate warm-range timelines, playback was 23.842, 23.845 and 23.841 fps with respectively 192/192, 192/192 and 193/193 display-cache hits and zero graph evaluations. Thus the render graph is out of the playback interval, but the strict 24 fps target still misses every adapter.
- **Artifacts:** `tools/live_picture_bench.py`, `tools/_live_picture_driver.py`, `tests/test_live_picture_bench.py`, both byte-identical copies of `docs/ROTO_TRACKING.md`, and corrected cold/warm benchmark JSONs under `benchmarks/live-picture/`. All are local and included in this commit. The prior warm JSON's direct residency field was invalid because it queried `region=None`; playback-hit counters are recomputed from the event timeline and the invalid field is explicitly null. No push. Transient shell logs remain in `/tmp` and are not evidence artifacts.
- **Verification limit:** The corrected cache-residency verifier was not rerun end-to-end: other lanes held the exclusive GPU queue. The reported warm-cache hit counts are derived from the completed archived timelines (all playback lookups hit; zero graph evaluations), while the newly corrected direct residency field remains unverified live.
- **State:** PARTIAL. The 0.5-second first-move picture gate is met on all three adapters. The 24 fps playback gate remains open on all three. Focused tests, benchmark self-check, diff check and docs-copy check pass. No product rendering code changed in this handoff.
- **Next owner + concrete artifact:** Gonzo continues playback cadence investigation using `benchmarks/live-picture/after-*-warm-range.json` timelines and `tools/live_picture_bench.py --warm-playback-range`; preserve the strict measured rate and report if the 24 fps gate remains unattainable.
- **Failure mode:** A high-DPI benchmark that sends Qt logical coordinates directly to physical-pointer tooling can report a nonexistent adapter render failure. Keep the DPI conversion and physical-screen window geometry in the harness.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (integrator tick)

- **What was done:** `main` `ce3b26ae` -> `7dc30c79` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7dc30c79`: Ran 5762 tests in 481.719 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `386675a0` -> `7f93a326` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7f93a326`: Ran 5742 tests in 476.537 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 Step B finish 1: strict render gate and legible stills (PARTIAL for unavailable hardware validation)

- **What was done — measured:** RTX 3080 Ti fixed-32 frame 60 at 960 × 540: 35.55 s. Render-mode 1920 × 1080, adaptive threshold 0.01: 45.464 s, all pixels converged, max noise 0.009999989, mean 11.71 samples/pixel (max observed 1,560 under the 2,048 cap). New 16–64 adaptive showcase stills took 14.95/15.33/15.68 s. Visual review: the golden stream reads as a pour into the centered cup; steam is faint and the table shadow is still hard-edged. CPU/GPU comparisons at frames 1/60/120 passed on RTX and AMD Radeon 8060S; adaptive/majorant GPU modules passed on NVIDIA and forced-integrated invocations. Hot pour tests and knowledge tests pass. Preview evidence from Step B remains the earlier 64-cell Radeon proxy because preview code was unchanged.
- **Artifacts:** source commits `f8705d4c`, `81cb9b33`; docs, images, benchmark JSON and bundled docs commit `880218da`; all on `openclaw/nb-fluids-spike`, local and not pushed. Exact report: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L6.md`. Profiling and adapter logs are in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/profile-L6-finish1-settled2048.log` and `render-evidence-L6-finish1-64.log`, `render-integrated-compare-L6-finish1.log`, `tests-L6-finish1-{nvidia,integrated,shot,knowledge}.log`. Large task-owned cache moved to trash.
- **State:** local 12 GB render targets, frame comparisons, still refresh, and docs are complete. PARTIAL overall because the brief's left-out A6000 and Windows validation remain unverified. The 48 GB A6000 estimate is updated and explicitly unmeasured.
- **Next owner + concrete artifact:** Gonzo, when an A6000/Windows host is available, run `examples/hot_pour_shot/render_evidence.py` and compare its result with the dated profile and stills in `benchmarks/hot_pour/` and `docs/images/hot_pour_shot/`.
- **Failure mode:** one-step steam transmittance made the transparent liquid nearly black. Disabling steam cast shadows for this shot and allowing four transmission bounces restored the golden liquid while retaining the visible volume and geometry shadow.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `8662c7bb` -> `8f0b4c30` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8f0b4c30`: Ran 5742 tests in 483.002 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (integrator tick)

- **What was done:** `main` `d91eadd0` -> `9ffbd0b7` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `9ffbd0b7`: Ran 5740 tests in 478.929 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B of 2: the Hot pour settles in under a minute (PARTIAL)

- **What was done — evidence:** With one volume-shadow step, the RTX 3080 Ti frame-60 960 × 540 fixed-32 render took 22.735 s vs 200.38 s at 16 steps. The optimized 1080p adaptive run took 83.511 s, mean 8.379 spp, cap 64, 29 passes; 99.405% pixels passed noise threshold 0.01. Frame 60 NVIDIA CPU/GPU comparisons passed at frames 1/60/120. Integrated AMD Radeon 8060S 64-cell/120-frame preview: 29.7 s bake, 134.31 warm FPS, 13.4 ms stop-to-full-detail. Adaptive/volume targeted tests: 12/12 NVIDIA, 12/12 forced-integrated invocation, 12/12 llvmpipe; Hot pour tests 2/2; knowledge tests 7/7. The stills remain grainy and the pour stream reads weakly.
- **Artifacts:** Code commit `a71af0cd`; profile commit `e3d1ca0f`; docs, profile summary, preview JSON, comparison stills and baseline copies are committed as `1b267f28` on `openclaw/nb-fluids-spike` (local only, not pushed). Scratch logs under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`; large task cache removed before closure.
- **State:** Partial against the fully-settled 1080p and legible-pour goals. 0.595% of pixels remain above the threshold at the 64-sample cap; visual quality is not accepted as final. A6000 and Windows unverified.
- **Next owner + concrete artifact:** Gonzo to review the baseline/new frames in `docs/FLUIDS_SPIKE.md`, then improve the pour/shadow quality using `examples/hot_pour_shot/hot_pour_shot.nbcomp` and rerun `examples/hot_pour_shot/render_evidence.py`.
- **Failure mode:** the first scratch profile attempted to serialize NumPy image arrays directly; it was corrected to persist compact numeric summaries. The first optimized 1080p result was initially estimated from GPU subphase timing; final reporting uses the measured 83.511 s wall time.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `d3bb958d` -> `80fcbd48` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `80fcbd48`: Ran 5740 tests in 475.161 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: the Hot pour's first look in under a second (fix the red suite) (integrator tick)

- **What was done:** `main` `e10c77a6` -> `63be6ad5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `63be6ad5`: Ran 5739 tests in 495.779 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 3: edit on the Direct picture: clicks reach the same handles the Viewer would pick, Lane 4 (Rendering, Claude Sonnet 5.5) step A of 3: an edit in Render mode is picked up at once: passes stop midway instead of finishing first, Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `8e58480a` -> `3db8ecaf` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `3db8ecaf`: Ran 5739 tests in 472.117 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (integrator tick)

- **What was done:** `main` `5d3a10db` -> `a40e37c8` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a40e37c8`: Ran 5706 tests in 496.222 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: the Hot pour's first look in under a second (integrator tick)

- **What was done:** `main` `89c3b9e1` -> `05a538ec` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `05a538ec`: Ran 5704 tests in 507.047 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — Lane 6 step A finish pass: first look and refinement responsiveness (DONE on lane branch)

- **What was done — evidence:** the progressive worker now retains camera-independent liquid/geometry preparation across equivalent scene evaluations and uses shorter heavy-scene GPU dispatches. Warm acceleration build/packing fell from 144–146 ms to 1.5–2.9 ms. Camera-move cancellation measured 2.0 ms RTX / 1.8 ms Radeon. Frame-60 CPU/GPU mean absolute RGB error stayed 9.96e-5 on both adapters. Cold/warm press-to-picture: RTX 1.170 s / 0.214–0.249 s; Radeon 1.048 s / 0.225 s.
- **Artifacts:** `nodebased/gpupathtrace.py`, `tests/test_fluid_firstlook.py`, `benchmarks/hot_pour/firstlook-timeline-2026-10-20.json` and `benchmarks/hot_pour/firstlook-integrated-2026-10-20.json`, `docs/FLUIDS_SPIKE.md`, `docs/INTERACTIVE_VOLUMES_LIQUIDS.md` and byte-identical bundled copies are committed on `openclaw/nb-fluids-spike` (local, not pushed). Test logs are deliberate scratch: `tests-L6-finish-a.log`, `tests-L6-finish-a-integrated.log`, `tests-L6-finish-a-cpuadapter.log` and `tests-L6-finish-a-docs.log` in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`.
- **State:** done against the brief's timing, acceleration, cancellation and comparison gates. Windows and real-display review remain unverified.
- **Next owner + concrete artifact:** Gonzo; perform visual review on the real display, using the two first-look JSONs and `tools/benchmark_fluid_firstlook.py`.
- **Failure mode:** an identity-only scene key missed acceleration reuse because camera-move evaluation creates equivalent new scene objects; lowering the fixed-band target did not change adaptive dispatch sizes. Content comparison and adaptive chunk sizing address those misses.

## 2026-10-20 — continuous mode merge: Lane 3 (Research, GPT-6 Astra) step R5 of 1: a first look under a second for volumes and liquids: what production renderers do (integrator tick)

- **What was done:** `main` `403fc9ec` -> `392217d5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `392217d5`: Ran 5700 tests in 504.190 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — Lane 3 R5: interactive volumes/liquids research complete

- **What was done:** researched production preview/preparation/sampling strategies and recent papers; inspected main `15e28165` renderer paths; wrote a dated evidence register, implementation mapping with labelled effort estimates and three ranked proposed Rendering briefs using Fluids artifacts. Evidence: existing sparse majorants, scoped preparation reuse, device-buffer reuse and retracing progression are visible in code. Inference: persistent preparation, bounded previews and sample continuation are the most direct next experiments; their speed gains remain unmeasured.
- **Artifacts:** `docs/INTERACTIVE_VOLUMES_LIQUIDS.md`, dated R5 section and priority line in `docs/RESEARCH_RADAR.md`, byte-identical `nodebased/data/docs/` copies, and `briefs/proposed/R5-{1-persistent-render-preparation,2-budgeted-liquid-volume-preview,3-persistent-convergence-and-denoise}.md`. Committed on the research lane branch, not pushed; no app code, weights or GPU execution. Targeted knowledge-test logs are deliberate lane-run scratch, named `tests-L3-R5-part*.log`; integrator report is `report-L3.md` in the lane-run directory.
- **State:** research deliverables done; knowledge-document validation passed before each documentation commit. No implementation/hardware performance or image-quality gate claimed. The brief's new Hot pour timings are missing from inspected main, which still marks that sweep outstanding; source fetch limitations are recorded in the source register.
- **Next owner + concrete artifact:** Gonzo integrates and assigns the ranked `briefs/proposed/R5-*.md`; Rendering recovers the raw Hot pour timing record or records a fresh exact-scene baseline before comparative acceptance. A6000/laptop remain explicit unverified reporting tiers; shader/buffer work must include Radeon and llvmpipe correctness.
- **Failure mode guarded against:** proposing already implemented skipping/adaptation as absent, treating a vendor demo as target-card latency, conflating allocation reuse with sample accumulation, or calling a filtered/capped picture converged.

## 2026-10-20 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step S2 of 2: the Hot pour reads as a pour (finish 2: the auto-solver crash, then the card numbers) (finish 1) (integrator tick)

- **What was done:** `main` `15e28165` -> `91049701` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `91049701`: Ran 5700 tests in 486.583 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP2b of 2: overlays on the direct picture, the laptop check, and the default-viewer decision, Lane 4 (Rendering, Claude Sonnet 5.5) step H2 of 2: keep the image refining after the first look (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step T2 of 2: a live picture while the artist works: the drag refreshes within a third of a second and refines after, playback honours the proxy tier, analysis never freezes the window (finish 1) (integrator tick)

- **What was done:** `main` `5bea8768` -> `e9588d4b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e9588d4b`: Ran 5700 tests in 500.166 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

# Lane 8 T2 finish 1 — real-display measurements (PARTIAL, 2026-10-20)

- **What was done — evidence:** measured the merged viewer on the 1920 × 1080 QA comp. RTX 3080 Ti: drag first picture 1.142 s from the first move (after release), settle 3.888 s; uncached scrub showed tier 8 at 0.595 s, tier 4 at 0.978 s, tier 1 at 4.914 s. Playback at the 24 fps target: 0.125 fps (tier 1), 2.624 fps (tier 4), 12.84 fps (automatic), with automatic tier choices 1 → 2 → 4. llvmpipe, while Lane 4's integrated-adapter progression benchmark ran: drag first picture 1.414 s after release, settle 4.416 s; scrub first picture 0.808 s, settle 9.446 s; playback 0.125 fps (tier 1), 1.618 fps (tier 4), 0.125 fps (automatic, tiers 1 and 2). The merged docs test passed (7 tests) and bundled docs are byte-identical.
- **Inference:** both adapters miss the 0.3 s first-picture target; RTX automatic playback misses the 24 fps target. The llvmpipe values are a contention-affected sample. PlanarTracker's bad cube-face pick remains a selection error; the contained checker-floor failures indicate a separate robustness limitation.
- **Artifacts:** documentation and bundled copy committed on `openclaw/nb-2d-parity-b`; detailed real-display logs and test output are local scratch: `scratch/nb-lanes/run/qa-L8-T2-llvmpipe-1020-r2.log`, `qa-L8-T2-rtx3080ti-1020.log`, `tests-L8-T2finish1-knowledge.log`. No push. The first provisional playback log is excluded because its counter counted preview generations rather than changed frame numbers.
- **State:** partial; the implementation is useful, measurements are recorded, latency and set-rate gates remain unmet.
- **Next owner + concrete artifact:** Gonzo should scope any follow-on performance change using the two QA logs and this `docs/ROTO_TRACKING.md` entry, then repeat `scratch/nb-qa/1015-roi/scripts/s12_t2.py` on both adapters.
- **Failure mode:** do not use the first provisional FPS output; it counted preview generations as presented frames. The corrected counter counts a presented frame only when both the preview generation and current frame number change.

## 2026-10-20 — Lane 4 step H2 of 2 (keep the image refining), finish 1: DONE

- **What was done — evidence:** the earlier H2 pass landed the worker-thread Render mode (`nodebased/progressivejob.py`), denoised passes, preview/refining/settled labels, and the convergence tests against Render3D (mesh room, depth of field, lens, mixed mesh and splat; green on RTX 3080 Ti, Radeon 8060S, llvmpipe). This finish ran `tools/benchmark_progression.py` on a quiet card under the exclusive GPU lock at 1920 x 1080 on all three adapters, recorded the numbers in `docs/3D_FOUNDATION.md` (Lane 4 step notes, bundled copy byte-identical), and committed the result JSONs. Targeted rerun: tests.test_knowledge, test_3d_progressive_job, test_3d_progressive_convergence, test_3d_first_look: 64 tests OK.
- **Measured (furnished room / mixed mesh+splat):** RTX first image warm 0.52 s / 1.02 s, settled 8.2 s / 82.5 s, cancel 35 ms / 403 ms. Radeon 0.54 s / 0.83 s, settled 16.7 s / 34.9 s, cancel 22 ms / 406 ms. llvmpipe furnished only: 0.82 s warm, settled 97 s (correctness, no target claimed).
- **Inference / not measured:** A6000 and the hybrid Iris Xe + RTX 4070 laptop are unreachable here; Windows not run. The mixed scene misses the one-second target on the RTX (1.02 s warm).
- **Artifacts:** committed `benchmarks/progression/{rtx3080ti,radeon8060s,llvmpipe}-1020.json`; local only (not committed) time-stamped PNG frames under `scratch/nb-lanes/run/prog-L4/{rtx,amd,cpu}/`. Removed a stray empty tracked file `timeout`.
- **State:** done. Remaining slow: mixed-scene full-size passes and cancel latency (~0.4 s, a pass cannot be interrupted midway). Next owner: Gonzo; concrete artifact: a mid-pass cancel check in the path tracer (follow-up, not started).

## 2026-10-20 — Lane 2 VP2b: overlays and readout on the direct picture, laptop check kit, default-viewer memo (DONE)

- **What was done — evidence:** the Viewer's foreground painting is now one function (`Viewer.paint_overlays`); `Viewer.render_overlay` paints it for the direct view's transform into a premultiplied RGBA image that the card draws one texel per pixel over the picture (`RetainedPresenter.set_paint`, `PAINT_SHADER`). Drawn read-only: A/B wipe line, handles and B half, region-of-interest box, format masks, Roto, Tracker, Transform and other handles, corner labels. Pixel-compared against the Viewer's own widget at scale 1.0 and 1.75 (wipe, moved wipe, ROI, three mask modes, Roto, Transform) in `tests/test_directviewer_overlays.py`; a pixel agrees when a Viewer pixel within one step has the card's colour within 16/255. Pixel readout on hover reads the float frame (`Viewer.pixel_readout_info`, shared with the Viewer's readout) and is painted bottom left; a played frame is read with its own geometry (`Viewer.readout_geometry`). `tools/laptop_present_check.py` (field kit) with `--self-check` (ran here, no Qt or wgpu) and a `result.json` per run recording the adapter and backend used and the frame cadence; steps in `docs/FIELD_TEST.md`. Decision memo in `docs/DIRECT_VIEWER.md`: not yet the default.
- **Tests:** `tests.test_directviewer_overlays`, `test_directviewer`, `test_directviewer_playback`, `test_laptop_present_check`, `test_carddisplay`: Ran 93, OK on the RTX 3080 Ti, the Radeon (integrated) and llvmpipe (cpu), on a private Xvfb display and under the GPU lock (`scratch/nb-lanes/run/vp2b-{default,integrated,cpu}.log`). Targeted regression on the Viewer: compare, roto, transform handles, look, frame, mode, navigation, strip: 125 OK; desktop, punchlists, parity, lens view, artist tools, knowledge: 274 OK.
- **Inference / not measured:** no Windows number exists or is claimed. Clicks on the direct surface still only pan and zoom: no test shows a handle click lands on the Viewer's pick, so there is no routing setting. Overlays do not refresh per played frame; the real-display walk, A6000 and laptop runs are owed.
- **Failure mode found:** `StubPresenter` in the tests lacked the new paint members and made the widget fail over, which hid the dock in two display-only tests that skip without a display; stubs now carry `paint_size`, `paint_key`, `set_paint`.
- **State:** done; default stays off. **Next owner:** Gonzo schedules DiMo's laptop run and the real-display walk.

## 2026-10-20 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice, Lane 4 (Rendering, Claude Sonnet 5.5) step H2 of 2: keep the image refining after the first look (integrator tick)

- **What was done:** `main` `ae4d6c04` -> `5b4e2e25` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5b4e2e25`: Ran 5673 tests in 487.940 s, OK (skipped=16), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — Lane 2 VP1 respawn: already landed, re-verified on main (DONE, no new code)

- **What was done — evidence:** the tick spawned `L2-viewer-present-a.md` (VP1 of 2) at 5:02 AM although VP1 merged to main on 10/18 and 10/19 (`docs/DIRECT_VIEWER.md`, `nodebased/directviewer.py`, merges `bbb03482` and `2673f45b`) and VP2a followed. The lane state's merged list was empty because the plan name changed to the VP2 plan, which looks like the cause of the respawn. I fast-forwarded the lane branch to main (`ce83279b`, 0.41.0), changed no code, and reran the VP1 modules there: `tests.test_directviewer`, `tests.test_directviewer_playback`, `tests.test_carddisplay`: Ran 66, OK (skipped=4), on the offscreen platform (`scratch/nb-lanes/run/tests-L2-vp1-recheck.log`).
- **Inference / not re-measured:** the shared GPU lock was held by another process, so the three-adapter runs, the real-display walk and the cadence/CPU comparison were not repeated; the numbers in `docs/DIRECT_VIEWER.md` (10/18 and 10/19) stand as the record. Colour/alpha, A/B/ROI, Roto/Tracker hit order, Windows, native Wayland and device-loss gaps are listed in that file under "Known gaps (VP1)" and are unchanged.
- **Artifacts:** nothing new; branch `openclaw/nb-2d-parity` equals main at `ce83279b`, nothing uncommitted. Report: `scratch/nb-lanes/run/report-L2.md`.
- **State:** done (previously landed); the switch stays off by default.
- **Next owner + concrete artifact:** Gonzo should mark VP1 merged in the lane state (so the card stops offering it) and run the queued `briefs/L2-vp2b.md`; VP2 parity/Windows gate remains Gonzo's.
- **Failure mode:** a lane step whose plan name changes before it is recorded in `merged` is spawned again; check `git log --grep` on main for the step name before spawning.

## 2026-10-20 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice (finish 1), Lane 6 (Fluids, GPT-6 Luna) step S2 of 2: the Hot pour reads as a pour (finish 2: the auto-solver crash, then the card numbers) (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step T2 of 2: a live picture while the artist works: the drag refreshes within a third of a second and refines after, playback honours the proxy tier, analysis never freezes the window (integrator tick)

- **What was done:** `main` `9fe51455` -> `667e0e81` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite (parallel runner) at `aca39b06`, rebased onto 9fe51455 as `667e0e81` (adds only the roto timing-test headroom; tests.test_roto rerun OK): Ran 5641 tests in 642.700 s, OK (skipped=16), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

# Lane 8 T2 — live pictures, playback adaptation, background analysis (PARTIAL, 2026-10-19)

- **What was done — evidence:** committed progressive tier-8/tier-4/final refresh for drag snapshots, knob edits and uncached scrubs; adaptive playback tier reduction/restoration; PlanarTracker worker progress/cancel. Tracker analysis uses its existing worker path. The exact-HEAD targeted command passed 72 tests (`scratch/nb-lanes/run/tests-L8-T2-combined.log`, local scratch). Roto drag preview tests verify the real document and undo stack remain unchanged. The stored broad floor corners and a contained checker-floor region failed tracking at grids 8, 16 and 24; cube face is flat and featureless, a bad pick, while floor failures expose a robustness limitation. **Inference:** implementation addresses the interaction gaps but does not establish latency/rate gates.
- **Artifacts:** source/docs are committed on `openclaw/nb-2d-parity-b`; `docs/ROTO_TRACKING.md` and `nodebased/data/docs/ROTO_TRACKING.md` are byte-identical. Three implementation commits: `fae62337`, `3fbba399`, `3b0c3927`. Local QA driver and logs are under `/var/home/omid/.openclaw/workspace/scratch/nb-qa/1015-roi/scripts/` and `scratch/nb-lanes/run/`; no intended benchmark artifacts have been committed. No push.
- **State:** partial. First-picture-under-0.3-second, settled latency, 12 GB adapter and llvmpipe measurements, tier-1/tier-4/automatic playback FPS, and maintaining the artist-set rate are unverified. The integrator's full suite held `/tmp/nb-gpu.lock` through the brief's 30-minute wait; no lock bypass or process interruption.
- **Next owner + concrete artifact:** Gonzo should run `scratch/nb-qa/1015-roi/scripts/s12_t2.py` on display `:48`, with the RTX run under exclusive `/tmp/nb-gpu.lock`, then run the CPU-adapter variant after the GPU lock is released; append only observed values to both `docs/ROTO_TRACKING.md` copies. Recheck `scratch/nb-lanes/run/tests-L8-T2-combined.log` for the exact targeted results.
- **Failure mode:** do not call the feature gate done from code/tests alone; historically the broad floor track failed even after selecting the valid checker-floor region, so “bad plane pick” does not explain all of Finding 8.

## 2026-10-16 — continuous mode merge: Lane 3 (Research, GPT-6 Astra) step V1 of 1: how the fastest apps keep their interface at display rate, and what our Python + Qt + wgpu setup can reach, Lane 6 (Fluids, GPT-6 Luna) step S2 of 2: the Hot pour reads as a pour (finish 2: the auto-solver crash, then the card numbers), Lane 8 (2D parity B, GPT-6 Luna) step T1 of 2: the region of interest never breaks the picture (finish 1) (integrator tick)

## 2026-10-19 — Lane 2 VP2a (Opus sprint): the direct picture plays smoothly beside the Viewer and takes 4K pictures

- **What was done:** found what blocked playback at 4K: the Viewer's visible crop and the whole frame alternated, every
  size change made a new 133 MB float texture (draws up to 262 ms on the RTX 3080 Ti) followed by a 34 to 38 ms
  `write_texture` on the Qt thread. Uploads now go through a staging ring of three reused buffers filled on a filler
  thread; the draw records the copy only; one picture waits between offer and copy, newer offers replace it; textures
  grow and are reused, one per float format; 4K float frames go up as half floats (48 ms over the RTX's USB4 link as
  float32, 32 ms now). Pictures over the card's texture or buffer limit are refused by name and the Viewer shows them;
  the worker converts such frames itself. Closing the dock releases the device, buffers and the retained picture
  (newest only); the cache inspector shows its size.
- **Measured** (this workstation; private Xvfb display; the A6000 and the laptop not measured): graph press, drag and
  timeline scrub under 31 ms at the 95th percentile while the dock plays, 1080p and 4K, RTX and llvmpipe; a 4K picture
  costs the Qt thread about 1 ms on all three adapters (VP1: 11 to 25 ms); 241 of 241 frames reached the window in
  order at 1080p on the RTX and llvmpipe, 171 of 241 at 4K on the RTX (card side), 241 of 241 at 4K on llvmpipe.
- **Tested:** `tests.test_directviewer_playback` (new), `tests.test_directviewer`, `tests.test_carddisplay` on the
  RTX, the Radeon and llvmpipe and with wgpu hidden; `tests.test_artisttools`, `tests.test_artist_tools_wiring`,
  `tests.test_topbar`, `tests.test_e2e5_ui_punchlist`, `tests.test_knowledge`.
- **Not done:** real-display runs (workers may not use display :0); the Radeon cannot present to Xvfb; the hot pour
  comp was not played (a path-traced fluid shot that needs a bake); not pushed; no full suite. Report:
  `scratch/nb-lanes/run/report-L2-vp2a.md`.

## 2026-10-19 — Lane 2: sprint branches integrated (wire disconnect, Properties overhaul, viewer colour on the card)

- **What was done:** Lane 2's branch was reset to `main` (0.39.0; backup branch `backup/nb-2d-parity-pre-sprints-1019`) and
  the three finished 10/18 sprint branches were brought in by cherry-pick: `wire-disconnect-1018` (5 commits, clean),
  `props-overhaul-1018` (5 commits, clean; ColorCorrect migration is schema 24 -> 25, main was at 24), and the five
  commits of `astra-sprint-a-1018` that were not already on main (its first four are main's VP1 commits). Conflicts in
  `nodebased/directviewer.py` kept main's device-loss recovery, Wayland refusal and `needs_wgpu` skips and added the
  card display transform on top: recovery and close now also release the float texture, the retained frame for
  recovery is the whole ledger payload (QImage or linear), and close skips its idle wait on a device already lost.
  `docs/DIRECT_VIEWER.md` keeps main's known-gaps list with the sprint's colour and playback notes merged in.
- **CI fixes:** (1) props-overhaul's exit -11 in `tests.test_knob_sliders` was a stack overflow: with CI's DejaVu
  fonts PointsTo3D's x/y `WrappingRow` fit on one line only while the Properties scroll bar was hidden, so the row and
  the bar flipped each other through QScrollArea's resize handling (reproduced here with DejaVu fonts and PySide6
  6.12.0, 306,000 layout passes on one panel, gdb showed 10,000 nested resizes). `WrappingRow` now widens back only
  with `WIDEN_SLACK` (16px) to spare; regression test `WrappingRowHysteresisTests`. (2) The "cannot schedule new
  futures after shutdown" traces all came from `compact_display_cache` after the window closed; guarded. (3)
  `tests.test_carddisplay.CardDisplayAgreementTests` carries `needs_wgpu`. (4) `tests/test_tracker_completion.py`
  pins its old-document version to 15 instead of `SCHEMA_VERSION-1`.
- **Tested:** 31 targeted modules one interpreter each, with and without a wgpu-hiding shim, all OK; card tests also on
  the Radeon and llvmpipe through force-adapter.py; batch 20 of CI's run under DejaVu fonts with PySide6 6.12.0 offscreen
  and the slider/colour-picker modules on a private Xvfb display. Report: `scratch/nb-lanes/run/report-L2.md`.
- **Not done:** not pushed; no full suite; no Windows run; no real-display visual QA.

## 2026-10-19 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice (finish 1) (integrator tick)

## 2026-10-19 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step H1 of 2: a denoised 1080p first look within one second (integrator tick)

- **What was done:** `main` `e08cc94e` -> `0a89b108` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0a89b108`: Ran 5547 tests in 21350.370 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.


## 2026-10-19 — Lane 6 S2 finish attempt (PARTIAL)

- **What was done:** Queued the corrected still-sample matrix with the required exclusive GPU lock. The 30-minute wait expired while the integrator full-suite process retained a shared lock; no lock was bypassed. No new GPU timings were measured.
- **Artifacts:** Updated `docs/FLUIDS_SPIKE.md` and its byte-identical bundled copy. `tests.test_knowledge` passed. Lock-wait log: `scratch/nb-lanes/run/L6-matrix.log` (empty because the command never acquired the lock).
- **State:** PARTIAL. The matrix, progressive first-look/settle, and Radeon preview/stop remain unmeasured.
- **Next owner + concrete artifact:** Gonzo retries `scratch/nb-lanes/run/measure-L6-S2-matrix.py`, `scratch/nb-lanes/run/progressive-L6-s2.py`, and the 64-cell Radeon playback row from `tools/benchmark_hot_pour.py` after the shared GPU lock clears.
- **Failure mode:** An exclusive GPU benchmark can sit behind an integrator's shared suite lock; preserve the lock and report PARTIAL when the 30-minute wait expires.
## 2026-10-20 — Lane 6 step A cache-lifetime fix (DONE on lane branch)

- **What was done — evidence:** prepared scene entries now have explicit retention; ordinary first-look scopes clear on exit, volume previews retain equivalent acceleration across camera moves, and cancellation clears the entry. The retained-scene assertions now cover safe content-equivalent reuse and camera-independent preparation. Four requested test modules passed on the RTX 3080 Ti. A fresh Hot pour run measured 1.186 seconds cold and 0.374 seconds warm.
- **Artifacts:** `nodebased/gpupathtrace.py`, `nodebased/progressiverender.py`, `tests/test_3d_first_look.py` committed locally on `openclaw/nb-fluids-spike`; test log and timing JSON are deliberate scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`.
- **State:** done for this fix; full suite, AMD/llvmpipe reruns, Windows and real-display visual review remain unverified.
- **Next owner + concrete artifact:** Gonzo reviews the lane commit and merge-candidate results; use the test log and `tools/benchmark_fluid_firstlook.py` for follow-up.
- **Failure mode:** thread-local prepared-scene data previously survived a scope and could satisfy later first-look renders without preparing their scene; cancellation also left the entry behind.
## 2026-10-22 — Lane 6 B Hot pour finish 2 — PARTIAL

- **What was done — evidence:** Fresh RTX 3080 Ti frames 60/90/120 render in 41.90/40.87/41.63 s. They show a continuous golden stream from spout to pool, a smoother liquid surface after final denoising, a still-grainy plume and a still-crisp table-catcher shadow. CPU/GPU frame comparisons at 1/60/120 pass. Fixed-32 960 × 540 took 10.689 s. Adaptive 1920 × 1080 at noise threshold 0.01 converged every pixel (max noise 0.009999999) in 60.857 s, missing the strict under-60 target by 0.857 s.
- **Artifacts:** Source/generator, test, docs, evidence JSON and stills are committed locally as `1e09b529` on `openclaw/nb-fluids-spike`; not pushed. The latest unrendered candidate was reverted to the last measured settings. Render logs and cache are under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`; the finish5 queued render was canceled before starting after waiting behind another shared GPU job.
- **State:** PARTIAL. Continuity and pool smoothing improved; steam softness, shadow softness and full-HD-under-60 remain open.
- **Next owner + concrete artifact:** Gonzo; rerun `examples/hot_pour_shot/render_evidence.py --adapter default --workdir <cache-directory>` and `scratch/nb-lanes/run/profile-hot-pour-finish-final.py <worktree> <cache-directory> settled-1080p` after the shared GPU queue clears; tune steam density/noise and catcher-shadow softness while shaving at least 0.858 s.
- **Failure mode:** the 4096-sample adaptive cap left 0.014% of pixels above threshold; raising to 8192 fully converged but took 60.857 s. A temporary finish5 candidate could not be visually/performance validated because its queued render never acquired the GPU lock. An earlier bridge graph exceeded Scene3D's input slots; corrected by replacing the unused backdrop slot with the bridge at `object7`.
## 2026-10-22 — Lane 6 step A edit-driven Hot pour re-simulation — PARTIAL

- **What was done — evidence:** Added a complete-frame callback to the fluid cache bake path and a warmed edit benchmark covering pour velocity, cup movement, and steam vorticity. On the RTX 3080 Ti, each edited output chain completed in 30.15–50.19 s; first solved volume frame was 0.078–0.302 s. On the Radeon 8060S, edits completed in 23.68–32.43 s and first solved volume frame in 0.055–0.256 s. Radeon cup movement missed the strict <30 s full-chain gate by 2.43 s. CPU/llvmpipe and targeted regression suites passed. On-screen progressive presentation was not measured. The warm RTX diagnostic attributes 8.45–9.24 s to steam host work, 3.94–12.34 s to steam pressure, and 1.22–1.38 s to cache writes; cup movement also spends 14.29 s in whitewater output. Diagnostic timings are synchronized and for cost ranking.
- **Artifacts:** Changes, benchmark harness, warmed timing JSON, Radeon and RTX phase profiles, and mirrored docs are committed locally on `openclaw/nb-fluids-spike`; no push. Logs and temporary caches are deliberate scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`. Historical production-size Radeon bake evidence remains `benchmarks/hot_pour/radeon-full-bake.json`; no fresh production-size edit bake was run.
- **State:** PARTIAL. Cache bake callback and measurement harness are landed. Radeon cup-move time misses target. Artist-facing timeline/stale-result/edit-cancel wiring, cost reduction, on-screen first-volume proof, and a fresh production-size bake remain incomplete/unverified.
- **Next owner + concrete artifact:** Gonzo owns coordination with the app/UI lane to wire progressive frames and stale/cancel handling; use `tools/benchmark_fluid_edits.py` to optimize and remeasure the Radeon cup-move case, and the historical production artifact above as scale reference.
- **Failure mode:** The first harness timed edits from a cold cache and overstated recomputation. It was corrected to pre-bake the base graph, retain unchanged caches, and measure only the affected output chain; cold figures are marked superseded in the docs.
## 2026-10-22 — Lane 6 step A finish 1: progressive edit timeline — PARTIAL

- **What was done — evidence:** Fluid edits in the currently viewed chain now keep the prior picture visible with a `SIM STALE` label until the current edited frame replaces it. The timeline adds an amber simulation-frame band, and the window submits one background frame at a time through its existing single-owner preview queue. Playback is bounded to the contiguous solved prefix; stopping playback resumes the queue. Replacing the preview queue sets the active cancellation event. State, queue, timeline and view-dependency tests pass.
- **Artifacts:** UI wiring is in `nodebased/app.py`, queue support in `nodebased/playback.py`, ruler marks in `nodebased/timeline.py`; test in `tests/test_fluid_edit_timeline.py` and `tests/test_playback.py`. Mirrored docs are `docs/FLUIDS_SPIKE.md` and `nodebased/data/docs/FLUIDS_SPIKE.md`. Changes are committed locally on `openclaw/nb-fluids-spike`; no push. Production-size scratch was not created.
- **State:** PARTIAL. Real-display QA, measured <50 ms worker-stop latency, verified cost reduction and a fresh production-size bake remain outstanding. The historical production bake is ~8.50 GB and the workspace currently has 5.7 GB free, so a new run would not fit safely.
- **Next owner + concrete artifact:** Gonzo verifies the visible stale indicator and interactive playback on a real display, then owns the cost optimization using `benchmarks/hot_pour/edit-phase-profile-warm-rtx-2026-10-22.json` and `benchmarks/hot_pour/edit-phase-profile-integrated-2026-10-22.json`; rerun `tools/benchmark_fluid_edits.py` after each measured change. Retry production-scale timing only after scratch capacity exceeds the historical cache plus safe headroom.
- **Failure mode:** An initial state test caught the progress text staying at 1/N after a background frame completed; the per-frame label refresh is now covered by the ordered-enqueue test.
## 2026-10-22 — Lane 6 step B of 2 (PARTIAL)

- **What was done — evidence:** Fluid-node knobs now have look-only, keyed-simulation, or from-start classifications. The status row distinguishes look updates, a keyed edit and a from-start simulation edit. The keyed label reports the affected frame; the existing solver still starts the changed run from the beginning, so checkpoint reuse is not claimed.
- **Artifacts:** Commit `fd6dd08c` adds `nodebased/fluid_edits.py`, the UI status hook, classification tests, and a step note in `docs/FLUIDS_SPIKE.md` with its byte-identical bundled copy. Test logs are local scratch under the lane run folder; generated cache data was not created. The production-size cache was not baked because disk space is 5.7 GB versus an 8.5 GB historical cache.
- **State:** PARTIAL. Focused classification, timeline and registration tests passed on the default path, CPU and integrated-adapter harness; documentation consistency tests passed. Prefix checkpoint reuse, proxy-first/full refinement, old-cache stop behavior and the step 4 production gates remain unverified. Production-size gate is unmeasured for the disk reason above; A6000 estimate is explicitly labeled unmeasured in the docs.
- **Next owner + concrete artifact:** Needs Gonzo: implement prefix checkpoint reuse and proxy/full promotion, then add the pre-P4 cache stop test around `nodebased/fluid3d.py`/`nodebased/imaging.py`; re-run the step 4 measurements when scratch space allows.
