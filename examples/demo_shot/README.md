# Demo shot: one plate, a solved camera, a generated backdrop, done by hand

The repeatable shot the end-to-end sprint is measured on (docs/END_TO_END_SPRINT.md). The sprint's
goal is a filmed plate; **the repository has no filmed footage, so the plate here is rendered by
NodeBased itself** (a small synthetic set through Render3D). Everything below is deterministic, runs on
the CPU and needs no model and no network. The generative step is the local `reproject` stand-in; the
real model comes in stage 3 and only swaps the provider.

## What is here

| File | What it is |
| --- | --- |
| `make_plate.py` | Renders the 24-frame plate to EXR and checks it against `plate.sha256.json`; `--k1` renders the lens plate instead (hashes in `plate_k1.sha256.json`); `--distorted` the plate of the lens round trip, 640 by 360 with k1 and k2 (hashes in `plate_distorted.sha256.json`). |
| `solve_camera.py` | Tracks the plate, solves the camera from it, compares with the rendering camera, writes `solve.json` and `demo_shot.nbcomp`; `--solver dpvo --lens estimate` solves with DPV-SLAM and recovers the lens (below); `--distorted` solves the distorted plate (the surveyed landmarks, or `--solver dpvo`). |
| `lens_benchmark.json` | The recorded focal length and k1 recovery runs (`solve_camera.py --solver dpvo --matrix`). |
| `diagnose_verdict.py` | Measures the verifier on the shot: each check's worst value, its threshold, and the same measurement for the plate against its own SceneState. |
| `demo_shot.nbcomp` | The shot graph. |
| `demo_shot_lens_dpvo.nbcomp`, `solve_lens_dpvo.json` | The same graph with the camera and lens of the DPV-SLAM solve on the card (`solve_camera.py --solver dpvo --distorted`), and the recovered lens and errors (not the default; see below). |
| `demo_shot_lens.nbcomp` | The shot graph of the distorted plate: plate, Undistort, the conditioning and the comp in the undistorted frame, Redistort, Write (below). |
| `solve_lens.json` | The landmark solve behind it: the lens recovered against the truth, the fit, the per-frame errors and the overscan. |
| `lens_check.py` | The lens round-trip error of a run, apart from the verifier's verdict. |
| `capture_lens_view.py` | Renders the viewer's undistorted view with its overscan into `docs/images/demo_shot/`. |
| `run_by_hand.py` | Headless run through the `nodebased.agent` dispatcher; prints artifact ids and the verdict. `--plate distorted` runs the lens shot. |
| `run_by_agent.py` | The same shot through MCP tools only: a scripted agent on an offscreen window (stage 4b). |
| `compare_runs.py` | Runs the hand run and the agent run with fresh stores and compares every id and the output pixels (`--plate distorted` for the lens shot). |
| `run_by_claude.py` | One real Claude Code session on the brief over MCP (needs the `claude` CLI and a login; network). |
| `BRIEF.md` | The one-sentence brief and the MCP tool sequence for the agent run (stage 4b). |
| `shot.py` | Shared scene, camera move and graph builder. |

## The plate

480 by 270, 24 frames, `plate.%04d.exr` (float, about 1.3 MB a frame) with beauty, `normals`, `depth`,
`motion` and `object_id`. The set: a checker-textured ground, a pillar and a crate as set pieces, the
mixed scene's cube and sphere (examples/mixed_scene), a backdrop card, a directional key light and an
environment light. The camera dollies in from z 8.0 to 6.6 while panning a little to the right.

The frames are not committed. They go to `$NODEBASED_CACHE/demo_shot/plate` (default
`~/.cache/nodebased/demo_shot/plate`). Regenerate and check them (about 24 s on 32 cores):

    python examples/demo_shot/make_plate.py            # --out DIR to choose the folder

The script compares each frame's pixel digest (the EXR header carries a write time, so file bytes are
not compared) with `plate.sha256.json` and exits 1 on a mismatch. `--update-hashes` rewrites that file.
The digests were recorded on this machine's CPU; another build of the renderer's dependencies may need
them re-recorded.

## The tracked camera

The camera is solved from the plate alone. Survey points of the set (the ground checker corners and the
top corners of the cube and crate, 49 in view) are placed in frame 1 the way an artist would click them
(the true projection rounded to a whole pixel), tracked through the plate with `nodebased.tracker`, and
the camera is fitted to the tracks on each frame by the verifier's own solver
(`conditioning_verify._fit_camera`), starting from a generic camera (default distance and lens) and
dropping tracks more than 1.5 px from the fit. Frame 1 starts from the generic guess; each later frame
starts from the previous solution. The graph's `camera` node holds the solved values as per-frame keys;
the rendering camera is used nowhere downstream.

Error against the camera that rendered the plate, 24 frames (`solve.json`, same measures as
`conditioning_verify`):

| | mean | worst |
| --- | --- | --- |
| position | 2.9 cm | 5.2 cm |
| rotation | 0.15 degrees | 0.32 degrees |
| field of view | 0.12 degrees | 0.37 degrees |
| reprojection | 0.47 px | 0.64 px |

The verifier's camera tolerance is 1 cm and 0.25 degrees, so the solved camera is close but outside
it on position. That does not enter the verdict: the verifier judges the generated pictures against
the exported camera (the solved one), not the solve against the rendering camera.

## The lens plate and the recovered lens

The plain plate has no distortion, and the solvers take the focal length they are given. Step L2 of the lens plan makes
`CameraSolve3D` recover both after the solve (docs/END_TO_END_SPRINT.md, "Stage 7, step L2"), and this folder is
its benchmark.

**The lens plate.** `make_plate.py --k1 -0.03 --size 960x540` renders the same camera move through a lens with
radial distortion `k1` (the Camera3D lens model, `nodebased/lensmodel.py`, units of half the frame height).
The set is rendered 1.35 times larger in each direction through the ideal camera and read through the lens model
into the plate's format (what Redistort does; `tests/test_camera_solve_lens.py` checks the two agree), so the
barrel-distorted corners show the set and not black. Beauty only: the data passes stay undistorted and are not
written. Any 16:9 size works; the benchmark uses 960 and 1920 wide.

**The benchmark.** Needs the NVIDIA card (it holds the GPU lock for the whole run, so it waits for a running suite):

    python examples/demo_shot/solve_camera.py --solver dpvo --lens estimate --distortion k1 --width 960 --plate-k1 -0.03
    python examples/demo_shot/solve_camera.py --solver dpvo --matrix        # writes lens_benchmark.json

The typed focal length is 1.2 times the truth in the estimate rows, the first-frame camera is the true one, and the solve is
compared with the rendering camera after a similarity fit (rotation from the camera orientations, scale and shift from
the positions: the solve is monocular).

<!-- lens-benchmark:start -->
Measured errors: not recorded yet, see docs/END_TO_END_SPRINT.md, "Stage 7, step L2" (the run waits for the GPU lock).
<!-- lens-benchmark:end -->

## The shot graph

`demo_shot.nbcomp`: `plate` (Read) and the solved `camera` (Camera3D); the layout (everything the plate
shows: ground, pillar, crate, backdrop, cube, sphere, key light, environment light) in a Scene3D, so the
exported SceneState describes the plate. The meshes are named (an object's id in the export hashes its
name: two unnamed cubes shared one id and their measurements collided);
`export` (WriteGeo3D, the SceneState and ControlBundle export) and `render` (Render3D, the passes the
bundle carries); `generate` (Generate, provider `reproject`, seed 7, the brief in its text control);
`grade` and `merge` (the result graded, then merged over the plate); `write` (Write, EXR). Two stages
of the shot are not graph nodes in this build and are run by the script: verification
(`verify`) and the generated read (a `ConditionedRead` node created at run time, because its path is
the generated sequence's artifact id). The Read path in the file points at `plate/`; the run script
sets it to wherever the plate was rendered.

The brief, "replace the backdrop with a rainy street, keep the cube and sphere, match the camera",
is in `generate`'s text control. `reproject` reports `text` as ignored (it also ignores normals, ids,
camera pose, reference frames and sampler settings; it uses depth, motion and seed). The run sends the
text anyway: one `generate` call does the shot, and the reply, the sequence's `generation.json` and its
provenance row record `text` as ignored. The stand-in carries each frame's layout render into the next
by its motion vectors; it does not paint a street.

## The distorted plate shot (step L3)

`python examples/demo_shot/run_by_hand.py --plate distorted` (and `run_by_agent.py`, `compare_runs.py` with the same
flag) runs the shot of a filmed, distorted plate: **plate, Undistort (with overscan), the layout, the conditioning
export and the generated picture in the undistorted frame, grade and merge, Redistort, Write**
(`demo_shot_lens.nbcomp`). The plate is 640 by 360 with `k1` -0.03 and `k2` 0.004 (`make_plate.py --distorted`,
hashes in `plate_distorted.sha256.json`; a plate in the cache that differs from them is rendered again). The camera and
the lens in the graph come from the landmark solve below (`solve_camera.py --distorted`); the graph also keeps the
`CameraSolve3D` node set up for a plate with no survey (`Focal length is` = estimate, `Distortion` = k1 k2 and a survey of
two picked points and their distance, which `shot.pick_survey` picks where the plate has texture on every frame), whose
result is the DPV-SLAM row of the table below. The overscan is the
margin Undistort grows the frame by (24 by 20 pixels here, the frame a multiple of 16 each way, which is what a
diffusion model wants), the export carries the lens of the overscan camera (the same physical lens on the larger
film back) and the resolution of the overscan frame, so a provider works undistorted and `Redistort` puts the
result back on the plate.

The report keeps two things apart. The **verdict** is the model's: lighting and objects are checked against the exported
SceneState, as in the plain shot. The **lens round trip** (`lens_check.py`, in the run's summary under
`lens_round_trip`) is the pipeline's: `plate_identity` is the plate sent through Undistort and Redistort against the
plate, and `result_vs_plate` is the written result against the plate; both are measured away from a 16 pixel edge
band, as a mean and a 99th percentile of the absolute difference, against limits in `lens_check.LIMITS`. With the
`reproject` stand-in both are judged. A real model restyles the picture, so for it only `plate_identity` is judged and
`result_vs_plate` is printed for reading.

Measured on this machine's CPU (24 frames, 640 by 360, no GPU used), with the camera and lens of the landmark solve the
hand run takes **325 s** (the plain shot's 155 s is at 480 by 270; the lens shot also renders the overscan frame and
resamples twice): verdict **PASS** (lighting pass, objects pass, the rest not conditioned as in the plain shot), plate
identity mean 0.0012 and p99 0.024, result against the plate mean 0.0026 and p99 0.059 (limits 0.003 / 0.06 and
0.01 / 0.12). The hand and the agent run give the same ids and the same output pixels (`compare_runs.py --plate
distorted`, 648 s for both):

| Artifact | Id |
| --- | --- |
| SceneState | `14a7f274019d8a8f43406ccfc141b1df58096b8edaf435bfe3ab56038fddab3d` |
| ControlBundle | `78a1f06a07031e526de0aba2a097d93d0a7fc371edea4cc80393e0a687313f72` |
| Generated sequence | `c68d664ea28725ab69086c946eac4e67fba5207ca1e9c06c83b0a4bbeea9d96a` |
| Verification report | `11be7d67a88603be6abc93412400a01f03eb705ca63586559e0a7ccc13a77578` |

**Why 640 by 360.** The verifier's object check allows 2 pixels. The `reproject` stand-in carries each layout render
along its motion vectors and its tracked error grows with the frame size: it passes at 640 and 704 wide and fails
at 768, 832 and 896 (two objects drift 2.2 to 2.4 pixels on a few frames). The shot is therefore 640 wide.

**The landmark solve (the default camera).** `python examples/demo_shot/solve_camera.py --distorted` (the processor, 8 s) clicks the
set's 50 visible surveyed landmarks on frame 1 (the true projection rounded to a whole pixel), follows them through the
distorted plate with the tracker and fits every frame's pose and one lens shared by all frames in a single adjustment
(`solve_lens_landmarks`: Levenberg-Marquardt with a Huber weight, outlier tracks dropped twice), starting from a focal
length typed 20 percent long and no distortion. Recovered against the truth: focal length 22.558 mm against 22.539
(+0.08%), `k1` -0.0314 against -0.03, `k2` 0.0042 against 0.004; camera position 2.2 cm worst over the 24 frames, rotation
0.19 degrees worst; 842 of 1200 tracked positions kept, 0.71 px RMS. The shot with that camera and lens gives the
**PASS** above. Starting from a focal length 15% short gives the same lens to within 1% (`tests/test_lens_shot_l3.py`).

**Why the solve with no survey cannot find `k1`.** The DPV-SLAM rows below come out with `k1` near 0 whatever the
settings, and the cause is the data. On the saved DPV-SLAM answer for this plate (768 points, 6144 tracked positions), a
bundle adjustment that holds the lens fixed gives the same reprojection error for a plate lens of zero, half the truth
and the truth: RMS 1.022, 1.023 and 1.030 px. Poses and points absorb the distortion on a dolly with this little
parallax. Letting only `k1 k2` move with the focal length known gives -0.008 and a formal error of 0.013, against the
true -0.03; letting the focal length move too gives a focal length 18.6% short. A wider plate does not help: at 960 by 540
the same estimate gives a focal length 34% long and `k1` -0.006. The surveyed landmarks break the tie, since their world
positions are known, so the lens is no longer traded against the structure.

**The DPV-SLAM solve on the card (10/13).** `python examples/demo_shot/solve_camera.py --solver dpvo --distorted` solved the
distorted plate on the RTX 3080 Ti and wrote `demo_shot_lens_dpvo.nbcomp` and `solve_lens_dpvo.json`. The lens estimate does
not hold at 640 by 360: the true focal length is 22.54 mm (typed 27.05 mm), the recovered one 24.80 mm (10.0% off); the
recovered `k1` is 0.0012 against the true -0.03 and `k2` 0.0001 against 0.004, so the lens is not found. The camera path is
good: fitted scale 0.995, position error 0.049 m RMS, rotation error 0.09 degrees mean and 0.18 maximum. A second solve in
`CardSolveTests` stayed inside its loose bounds (focal within 15%, `k1` within 0.03); the `k1` error of the run above, 0.031,
sits just outside, so the solver's result varies from run to run. The graph with that solve is not the committed default
because the shot fails with it:

| Run (640 by 360, 24 frames) | Camera in the graph | Verdict | Lens round trip | Time |
| --- | --- | --- | --- | --- |
| `reproject` stand-in, CPU | landmark solve (committed `demo_shot_lens.nbcomp`) | PASS | pass (result against plate mean 0.0026, p99 0.059) | 325 s |
| `reproject` stand-in, CPU | rendering camera | PASS | pass (mean 0.0022, p99 0.036) | 328 s |
| `reproject` stand-in, CPU | DPV-SLAM solve (`demo_shot_lens_dpvo.nbcomp`) | FAIL (objects) | plate identity pass, result against plate fails (mean 0.077, p99 0.90) | 287 s |
| `--provider sdxl_controlnet`, RTX 3080 Ti | DPV-SLAM solve | FAIL (lighting, objects) | plate identity pass (mean 0.0008, p99 0.014); result against plate printed, mean 0.27 | 739 s |
| `--provider sdxl_controlnet`, RTX 3080 Ti | rendering camera | FAIL (lighting, objects) | plate identity pass; result against plate printed, mean 0.28 | 910 s |

The model runs generate 24 frames at 1024 by 576 with CPU offload: 441 s (solved camera) and 579 s (rendering camera) for
the generation, peak 7.9 GiB of video memory. The model's lighting and objects checks fail with the rendering camera as
well, so the failure of the SDXL runs comes from the model's picture and not from the solve; the round-trip identity row
passes in every run. To run a graph other than the default, `run_by_hand.py --plate distorted --graph FILE`.
The committed default graph carries the landmark solve instead, and its stand-in row reads PASS (325 s).

## Running it

    python examples/demo_shot/run_by_hand.py [--plate plain|distorted|DIR] [--out DIR] [--cache DIR] [--solver survey|dpvo]

It starts `python -m nodebased.agent` on the graph and sends the same commands an attached agent sends
(`export_conditioning`, `generate`, `verify`, `edit`-style creates, `render`, `artifacts`), then prints
the artifact ids, the controls used and ignored, the per-check verdict and the provenance length. Output
frames go to `$NODEBASED_CACHE/demo_shot/out`, the artifact store to `.../demo_shot/store`.

`--solver dpvo` swaps the survey-point camera for one solved from the plate alone by a `CameraSolve3D` node (DPVO on
the NVIDIA card; set up with `projects/nb-providers/slam/setup.sh`, interpreter in `NODEBASED_SLAM_PYTHON`). The first-frame
camera is the node's anchor and two surveyed points fix the scale. Results and ids: `docs/END_TO_END_SPRINT.md`, stage 7.

Result on this machine's CPU (32 cores, no GPU used): **155 s** for the whole run at full size
(24 frames, 480 by 270), identical on a second run with a fresh store and output folder: same four
artifact ids and same output pixels (`tests/test_demo_shot.py` asserts it at 160 by 90, 4 frames).
The ids are listed in docs/END_TO_END_SPRINT.md under "Stage 4a". **The verdict is PASS**: lighting and
objects pass; camera, depth, motion and normals read "not conditioned" (`reproject` does not declare
those honoured; its camera is a resampled copy, measured up to 29 cm off, see below).

## Why the verdict was FAIL, and what changed (stage 4a fix)

The first hand run failed camera, lighting and objects. `diagnose_verdict.py` measured the plate itself
against its own SceneState and found the verifier and the export wrong, not only the stand-in:

| Finding | Measured | Fix |
| --- | --- | --- |
| The export rendered its references without the Render3D's ambient fill, and the verifier did the same | plate against the export: mean 0.05 per pixel, 32% of pixels differ | the SceneState records `ambient` per frame; export and verifier use it |
| The layout left out the pillar, crate and backdrop the plate shows | plate against its SceneState could not match | the layout holds everything the plate shows |
| Camera landmarks were tracked from the observed first frame: frame 1 scored zero by construction, later frames drifted with the dolly's change of scale, hidden vertices and straight-edge corners matched at random | plate against its own camera: 0.47 m worst | templates come from a render of the exported scene on the same frame; only visible vertices with corner structure count |
| Object ids hash the name; two unnamed cubes or cards shared an id | 138 px error on the backdrop | the demo's meshes are named |
| An object origin on a faintly shaded face matched 10 px away | crate, frames 9 and 15 of the plate | origins need corner structure like the camera landmarks |
| No shadow in a scene whose light has shadows off (the default) was counted as missing evidence | lighting failed on the plate | no shadow in the reference means no shadow direction is demanded |
| `reproject` 1.0 labelled frame N with frame N+1's picture (forward motion points at the next frame) | one dolly step, about 6 cm, in the camera solve | 1.1 carries frame N-1 into frame N; the first frame is copied |

After the fixes, with the camera that rendered the plate (the solved camera is up to 5 cm off it, so
against the solved graph the plate would fail the 1 cm check by construction), the plate against its own
SceneState passes camera, objects and lighting; thresholds are unchanged (1 cm, 0.25 degrees, 0.1 degrees,
2 px, 5 degrees, 0.05):

| check (worst over 24 frames) | threshold | plate | `reproject` 1.1 |
| --- | --- | --- | --- |
| camera position | 0.01 m | 0.0000 | 0.2933 |
| camera rotation | 0.25 degrees | 0.0369 | 0.4900 |
| camera field of view | 0.1 degrees | 0.0000 | 1.6490 |
| object origin | 2 px | 0.0000 | 0.7708 |
| light direction | 5 degrees | 1.5322 | 1.7491 |
| colour balance | 0.05 | 0.0000 | 0.0007 |

(shadow direction: the key light has shadows off, so none is demanded.) The generated column is a
resampled copy of the layout render, and pixel tracking at 480 by 270 cannot place a camera to 1 cm from
a copy that is off by a fraction of a pixel in places, so the camera check reads "not conditioned" for
`reproject`, whose description no longer declares camera pose honoured (it only decodes the motion
vectors). Reproduce the table: `python examples/demo_shot/diagnose_verdict.py` (about 4 minutes).

One `generate` call now does the shot: the text brief is recorded as ignored instead of refused, and
the reply carries the ConditionedRead node parameters (the hand run creates `generated` from them), the
same ones a one-attempt loop returns, with the same ids. The provider worker limits numpy's thread
pools itself and sizes its address-space cap from the job, so no caller sets thread variables: 24 frames
of 480 by 270 died on the 2 GiB cap before and run now.
