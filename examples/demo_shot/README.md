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
| `demo_shot.nbcomp` | The shot graph. |
| `run_by_hand.py` | Headless run through the `nodebased.agent` dispatcher; prints artifact ids and the verdict. |
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
it on position; the score card reports that.

## The shot graph

`demo_shot.nbcomp`: `plate` (Read) and the solved `camera` (Camera3D); the layout (ground, cube, sphere,
key light, environment light; the backdrop is left out because it is what gets replaced) in a Scene3D;
`export` (WriteGeo3D, the SceneState and ControlBundle export) and `render` (Render3D, the passes the
bundle carries); `generate` (Generate, provider `reproject`, seed 7, the brief in its text control);
`grade` and `merge` (the result graded, then merged over the plate); `write` (Write, EXR). Two stages
of the shot are not graph nodes in this build and are run by the script: verification
(`verify`) and the generated read (a `ConditionedRead` node created at run time, because its path is
the generated sequence's artifact id). The Read path in the file points at `plate/`; the run script
sets it to wherever the plate was rendered.

The brief, "replace the backdrop with a rainy street, keep the cube and sphere, match the camera",
is in `generate`'s text control. `reproject` reports `text` as ignored (it also ignores normals, ids,
reference frames and sampler settings; it uses depth, motion, camera pose and seed). Sending the
text to the provider makes it refuse the call before any work ("cannot honour controls: text"), so
the run records that refusal and sends the call again without the text. The stand-in therefore
reprojects the layout's own render by its motion vectors; it does not paint a street.

## Running it

    python examples/demo_shot/run_by_hand.py [--plate DIR] [--out DIR] [--cache DIR]

It starts `python -m nodebased.agent` on the graph and sends the same commands an attached agent sends
(`export_conditioning`, `generate`, `verify`, `edit`-style creates, `render`, `artifacts`), then prints
the artifact ids, the controls used and ignored, the per-check verdict and the provenance length. Output
frames go to `$NODEBASED_CACHE/demo_shot/out`, the artifact store to `.../demo_shot/store`.

Result on this machine's CPU (32 cores, no GPU used): **107 s** for the whole run at full size
(24 frames, 480 by 270), identical on a second run with a fresh store and output folder: same four
artifact ids and same output pixels (`tests/test_demo_shot.py` asserts it at 160 by 90, 4 frames).
The ids are listed in docs/END_TO_END_SPRINT.md under "Stage 4a". The verdict is FAIL: the camera,
lighting and objects checks fail (the stand-in forward-warps the layout render by one frame of motion,
so objects and lighting do not match the next frame's scene, and the solved camera is outside the
verifier's tolerance), depth, motion and normals read "not conditioned".

The provider worker has a 2 GiB address-space cap. On a many-core machine the numpy thread pools spend
a third of it, so `run_by_hand.py` starts the agent with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` at 2; without that, generate fails above roughly 300 by 180 pixels with "failed to
map segment from shared object".
