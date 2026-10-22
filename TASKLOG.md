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
