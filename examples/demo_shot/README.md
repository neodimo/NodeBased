# Demo shot: one plate, a solved camera, a generated backdrop, done by hand

The repeatable shot the end-to-end sprint is measured on (docs/END_TO_END_SPRINT.md). The sprint's
goal is a filmed plate; **the repository has no filmed footage, so the plate here is rendered by
NodeBased itself** (a small synthetic set through Render3D). Everything below is deterministic, runs on
the CPU and needs no model and no network. The generative step is the local `reproject` stand-in; the
real model comes in stage 3 and only swaps the provider.

## What is here

| File | What it is |
| --- | --- |
| `make_plate.py` | Renders the 24-frame plate to EXR and checks it against `plate.sha256.json`. |
| `solve_camera.py` | Tracks the plate, solves the camera from it, compares with the rendering camera, writes `solve.json` and `demo_shot.nbcomp`. |
| `diagnose_verdict.py` | Measures the verifier on the shot: each check's worst value, its threshold, and the same measurement for the plate against its own SceneState. |
| `demo_shot.nbcomp` | The shot graph. |
| `run_by_hand.py` | Headless run through the `nodebased.agent` dispatcher; prints artifact ids and the verdict. |
| `run_by_agent.py` | The same shot through MCP tools only: a scripted agent on an offscreen window (stage 4b). |
| `compare_runs.py` | Runs the hand run and the agent run with fresh stores and compares every id and the output pixels. |
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

## Running it

    python examples/demo_shot/run_by_hand.py [--plate DIR] [--out DIR] [--cache DIR]

It starts `python -m nodebased.agent` on the graph and sends the same commands an attached agent sends
(`export_conditioning`, `generate`, `verify`, `edit`-style creates, `render`, `artifacts`), then prints
the artifact ids, the controls used and ignored, the per-check verdict and the provenance length. Output
frames go to `$NODEBASED_CACHE/demo_shot/out`, the artifact store to `.../demo_shot/store`.

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
