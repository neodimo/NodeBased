# Proposed worker step: survey-constrained lens refinement against stage 9

**Proposed 2026-10-12 · Owner: Gonzo to assign to the rendering/camera-solve lane.**
Research T2 does not execute this experiment. One bounded experiment; no new provider, weights, production UI, schema change, full suite or second-model fallback. Read `context/lanes.md` and the assigned lane's standing brief before starting. Work in that lane's own worktree and never edit main.

## Question and baseline

Can putting independently known survey landmarks inside the existing lens/pose fit remove the reported focal bias without worsening trajectory error or repeatability?

The [research radar](../../docs/RESEARCH_RADAR.md#tracking-and-reconstruction-swept-1012) records two evidence levels:

- The T2 assignment supplies stage-9 MASt3R-SLAM focal error **+12–20%**, **11 FPS**, deterministic repeats and survey reprojection **1.35 px mean**. Raw stage-9 evidence was absent from the checked sprint notes on 2026-10-12. These are reference values to verify, not this experiment's measurements.
- The committed stage-7 table (2026-10-10) reports DPV-SLAM at **960 × 540: 1.8 cm RMS**, **45 FPS tracking / 29 FPS including optimization**; DROID's short-shot deterministic result is separate. The L2 synthetic-track test's **<1% focal at 1920-wide** does not establish real plate performance.

Use the same shot and exact stage-9 provider configuration if artifacts are recoverable. Changing to DPVO is an explicitly labelled fallback comparison, never a claimed improvement over a missing MASt3R run. The current `examples/demo_shot/solve_camera.py` CLI exposes only `survey` and `dpvo`; do not invent a `--solver mast3r` option.

## Budget and preflight — 15 minutes

**Hard stop: 90 minutes active work.** Proposed settings and thresholds below are experiment design choices dated 2026-10-12, not measured claims.

1. Locate the original plate, ground-truth lens/film back, poses, distortion, survey observations, provider identity/revision and saved solver outputs. Record content digests and pixel conventions. Ground truth is for evaluation only. Survey fitting may use only the subset that an artist could measure independently; never derive survey world coordinates from estimated geometry or reveal the true focal as a prior.
2. Record the actual timeline count, crop/resize transform, camera-to-world convention and timing scope. Establish the mean and distribution underlying the quoted focal range. Recover stage-9 position error from its raw report; no value is supplied in this brief. If missing, leave that historical metric unavailable.
3. Check that survey points span depth and image radius, that enough are visible across separated frames, and that held-out observations exist. A scale-only distance or a nearly planar configuration cannot certify the lens. Use at least **six independently positioned, non-coplanar static landmarks** as a proposed minimum fixture, with **two landmarks entirely withheld** and at least **three separated frames** per landmark. If the existing shot cannot supply this, do not silently change the shot: report insufficient fixture and use the fallback below.
4. Reuse raw track/pose snapshots; `--save-solves` in `examples/demo_shot/solve_camera.py` already supports offline L2 replay for DPVO. CPU replay is preferred. If raw measurements are unavailable, a future assigned worker may make **two sequential runs of the already-installed provider** with fixed settings, under the normal GPU lock, **at most five minutes waiting for the lock**. No new weights or environments. Peak device usage must remain within available memory on the **12 GB** card; leave the desktop headroom. If a provider cannot export tracks/known survey observations, do not fake them from its predicted pointmap. Mark that comparison unavailable.

Persist a short worktree state note before any lock wait. Timing the replay is not timing the original solver.

## One implementation and comparison — 45 minutes

Make a small offline harness in deliberate lane scratch, leaving production behavior unchanged. Use the existing lens projection/refinement routines in `nodebased/lensrefine.py`, `nodebased/lensmodel.py` the survey parsing/coordinates in `nodebased/camera_solve.py`, and projection conventions in `nodebased/track3d.py` as references. Do not reimplement a provider or rewrite bundle adjustment.

Compare these three cases on exactly the same saved observations and masks:

- **A: saved provider solve**, unmodified.
- **B: merged L2 lens refinement + existing post-hoc survey alignment**, as the true current-code baseline. Reuse `estimate_lens`; confirm its required track schema before adapting a MASt3R snapshot.
- **C: survey-constrained refinement**, one shared log-focal and per-frame camera poses using the fixed surveyed landmarks, with the saved static tracks supplying additional constraints where compatible. Begin with radial coefficients fixed at B. Reuse existing Jacobian/projection machinery and a bounded LM solve; introduce the survey residual/Jacobian in the harness. Do not add unknown principal point, tangential terms or a zoom model in this step. A consistent surveyed world gauge replaces the arbitrary frozen first-camera gauge; verify the two are not simultaneously overconstraining the problem.

Weight survey residuals by declared click/measurement uncertainty; optimize against fitting landmarks only. Keep the same robust-loss schedule and deterministic observation order in B and C. One full-resolution survey-observation pass is allowed; do not regenerate every image or alter the scene. Stop if the small harness requires a production solver rewrite. Record residuals, accepted steps and any conditioning/rank failures.

## Evaluation — 20 minutes

Run B and C twice in fresh CPU processes on the same immutable inputs. Pin numerical library/thread settings. If two independently captured solver runs are available, repeat refinement on each and distinguish **fixed-input determinism** from **end-to-end determinism**. Do not round outputs before comparison.

Report a compact table with the following for A, B and C:

- **Lens:** `100 * abs(f_est_px - f_true_px) / f_true_px`, plus signed error; evaluation truth is never used in the optimizer. Proposed pass **≤1%** on every tested fixture/run, not just the average.
- **Position:** RMS and maximum camera-centre error in metres using only the survey-derived world placement. Also report the historical similarity-aligned metric separately if needed to reproduce stage 9. No new ground-truth Sim(3) alignment may hide a metric-scale failure. Proposed pass: no increase over B beyond **1 mm RMS**, and **≤2 cm RMS** as a separate target derived from the stage-7 reference. If B is already better, retain the stricter no-regression requirement. Rotation error is a regression diagnostic.
- **Pixel evidence:** mean, median and p95 held-out reprojection error at the same resolution; compare **1.35 px** only if the recovered stage-9 report uses the same observations/definition. Require improvement or no regression on the independently held-out set. Keep the original verifier untouched; do not score only the landmarks used for fitting.
- **Speed:** provider frontend time, provider final optimization, survey/refinement CPU time, and end-to-end `frames / total_wall_seconds`. The quoted **11 FPS** must retain its original definition. Report added latency even when frontend FPS is unchanged. Proposed bounded-refinement gate **≤30 s for the demo clip**; a lens win with slower end-to-end throughput is an offline-quality result, not a real-time win.
- **Resources and repeatability:** CPU peak RSS and available provider GPU peak with measurement method; exact camera/lens array identity and canonical content digests across fresh runs. Exclude volatile timing fields only. If raw runs differ, report it even when fixed-input CPU replay is identical.
- **Degeneracy check:** profile at least **five fixed focal values around C (±2%)**, reoptimizing poses at each; save the cost curve and repeat the fit with a different predetermined observation subset. A flat/unstable profile or divergent held-out performance fails the confidence gate. This local profile is a diagnostic, not a global confidence interval.

Accept the experiment only with observable constraints, ≤1% focal, held-out/position no-regression, bounded CPU cost and honest determinism reporting. If B already meets the target and C adds no held-out improvement, prefer B and record that no extra production feature is justified. Do not claim a filmed-plate result from the synthetic demo.

## Fallback, artifacts and handoff — 10 minutes

If source artifacts, geometry, runtime or numerical fitting fail the gates, retain the saved production solve and known/artist-supplied lens. A CPU synthetic-observation replay using `tests/test_camera_solve_lens.py` may diagnose the mathematics but **cannot complete the stage-9 comparison**. Do not download DroidCalib/VGGT/MegaSaM as a fallback within this step; they need separate briefs.

Commit the compact experiment table and method in `docs/experiments/T2-survey-constrained-lens.md`. Since bundled knowledge docs are flat, place a byte-identical copy at `nodebased/data/docs/T2-survey-constrained-lens.md` and explicitly compare bytes in addition to `tests.test_knowledge`. Record source versions, digests, fitting/held-out point IDs, unavailable metrics, exact deliberate-scratch artifact locations in the private report, and local-vs-committed status. Never put machine-home paths in tracked files. Remove regenerable heavy scratch before reporting; preserve only evidence with an explicit retention owner.

Run `tests.test_knowledge`; if production lens code is changed under a separately authorized scope, also run `tests.test_camera_solve_lens` and the directly affected tests. No full suite or claimed provider promotion. Gonzo owns reviewing the experiment record and assigning any product change.

**Mandatory final step:** before reporting done, write your completion note to the NodeBased project root `TASKLOG.md` and `context/state.md` **inside your worktree only** (this is the `projects/nodebased/TASKLOG.md` completion contract applied to an isolated checkout). Include what was done with evidence versus inference, exact artifact paths and local/committed status, done/partial/blocked and unverified items, next owner + concrete artifact, and any failure mode. Update the assigned lane issue/checklist and board in the same turn; mark a tracked task done only when the comparison is genuinely complete. No note means the task is not closed.
