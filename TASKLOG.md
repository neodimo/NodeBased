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

