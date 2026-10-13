# Shot workflow for an attached agent

The order of operations from a filmed plate to a graded, generated result, and the MCP tool that does
each step. Every tool call goes through the same bridge as a human edit, appears in the Agent panel's
activity log, and shares one undo stack. Read this with the `knowledge` topic `shot_workflow`.

## Order of operations

1. **Look first.** `knowledge` (`overview`, `shot_workflow`), `describe` for node and parameter
   names, `inspect` for the live document and its revision.
2. **Bring in the plate.** `edit` with a `Read` node (or `ReadBundle`); `view` it; `errors` shows
   render errors since a timestamp. Saving, loading and rendering files from `edit` needs the
   "Allow save / load / render from the agent" setting.
3. **Lay out the scene.** `edit` creates geometry nodes into a `Scene3D`, a `Camera3D` (animated
   to match the plate's tracked camera) and a `Render3D` fed by both. `workspace` with `3D` shows
   the viewport, `Composite` returns to the comp.
4. **Export the conditioning.** `export_conditioning` with the `Render3D` or `WriteGeo3D` node id and
   a frame range. Returns a SceneState artifact id and a ControlBundle artifact id (depth, normals,
   motion, object ids, camera pose, beauty). Files stay inside the artifact store.
5. **Generate.** Either one call, `generate` (scene_state, bundle, provider), which returns the
   generated-sequence artifact id, or a bounded `loop`: `start` with the attempt cap, spend cap,
   per-attempt estimate and its unit; poll `loop` `status` for the attempt list, verdicts and stop
   reason; `cancel` stops it. Providers today are the local stand-ins `null` (returns the beauty
   unchanged) and `reproject`, and the real model `sdxl_controlnet` (SDXL with ControlNet-Union on the NVIDIA
   card; pick it when the brief asks for a new picture, not a copy of the layout render). It honours depth,
   normals, motion, IDs, text, seed and `sampler_settings` (`steps`, `guidance`, `controlnet_conditioning_scale`,
   `control_mode`, `denoise_strength`) and ignores camera pose and reference frames. Each frame after the first
   starts from the one before, carried along the exported motion, so the backdrop and lighting hold between
   frames; the verification report's "Temporal" line measures what flicker is left. `denoise_strength` 1 makes
   every frame on its own. Text is a description of the picture, not an instruction. A 24-frame shot at 480 by 270 takes about six minutes
   to generate on the 3080 Ti (about nine for the whole demo run), and the same brief and seed give the same ids. Pass a `seed` (default 0) and, if the provider uses them,
   `sampler_settings`; the reply's `controls` map says which the provider used or ignored. A text brief goes in
   `options` `{"text": ...}`; a provider that does not use text still runs the shot and the reply's `ignored` list
   names it, so one call does the shot. The reply also carries `path`, `manifest`, `scene_state` and `node`: the
   ConditionedRead parameters, the same ones a loop's `use_result` returns. The seed and
   settings are part of the artifact id, so the same brief and seed give the same id. A loop reuses its
   seed on every attempt unless `vary_seed` is true (attempt n then uses seed + n - 1).
6. **Verify.** `verify` (scene_state, bundle, generated_sequence) returns the score card and a
   `determinism` section. The verifier judges beauty and every pass the sequence carries; a check with
   no pass behind it reads "not conditioned" and is left out of the verdict, and so does the camera check of a
   provider that does not declare camera pose honoured (`reproject` does not). The loop verifies each
   attempt itself.
7. **Bring it back into the comp.** `generate` (or a loop's `use_result`) returns the parameters for a
   `ConditionedRead` node whose `path` is the generated-sequence artifact id. Create the node with
   `edit`, then connect it, grade it and write it like any other input.
8. **Prove where it came from.** `artifacts` `provenance` with the generated-sequence id returns the
   chain of artifacts that produced it, including the SceneState and ControlBundle ids.

## Tools added for the pipeline

| Tool | Arguments | Returns |
| --- | --- | --- |
| `export_conditioning` | `id`, `first`, `last`, `camera` (only for a plain scene node) | `scene_state`, `control_bundle` artifact ids |
| `generate` | `scene_state`, `bundle`, `provider`, `options`, `seed`, `sampler_settings`, `output_pattern` | `generated_sequence` artifact id, `seed`, `sampler_settings`, `controls` |
| `verify` | `scene_state`, `bundle`, `generated_sequence` | `report` artifact id, `verdict`, `score_card`, `determinism` |
| `loop` | `action` = `start` (also `seed`, `sampler_settings`, `vary_seed`), `status`, `cancel`, `use_result` | loop state, attempts with their seeds, stop reason |
| `artifacts` | `action` = `list` (`kind`, `limit`) or `provenance` (`id`) | recent artifacts, or the provenance chain |
| `workspace` | `tab` = `Composite`, `3D`, `Simulate`, `Render` | the active tab |

`generate` with `output_pattern` writes frames outside the artifact store, so it needs the same
"Allow save / load / render from the agent" setting as `edit`'s disk commands. Without that
argument the stand-in providers only touch the artifact store.

A loop started from the GUI runs in the background and appears in the Conditioning Queue panel; poll
`status`. The headless dispatcher (`python -m nodebased.agent`) runs a loop to completion before it
replies. `workspace` needs the window and is refused headless.

## Worked example: the demo shot

`examples/demo_shot` (README.md there) holds a 24-frame plate rendered by NodeBased, a solved camera and
the shot graph `demo_shot.nbcomp`, which already has the layout, the export node `export` (WriteGeo3D),
`generate` (Generate, provider `reproject`, seed 7), `grade`, `merge` and `write`. The brief is one
sentence: "replace the backdrop with a rainy street, keep the cube and sphere, match the camera". The
scripted agent `run_by_agent.py` does the shot with exactly these calls, in this order, and
`compare_runs.py` checks that the hand run reaches the same ids.

1. `knowledge` `{"topic": "shot_workflow"}`, `describe`, `inspect`. Find the `Read`, `WriteGeo3D`,
   `Generate`, `Grade` and `Write` nodes in the document and read the provider and seed from the
   `Generate` node.
2. `edit` `[{"op": "set", "id": "plate", "param": "path", "value": "<plate folder>/plate.%04d.exr"}]`.
3. `export_conditioning` `{"id": "export", "first": 1, "last": 24}`.
4. `generate` `{"scene_state": ..., "bundle": ..., "provider": "reproject", "seed": 7, "options": {"text":
   "<the brief, character for character>"}}`. The text is part of the generated sequence's id. The reply
   lists `text` under `ignored`: `reproject` carries the layout render forward by its motion vectors and
   paints no street. Say so in your report; do not describe the result as a rainy street.
5. `verify` `{"scene_state": ..., "bundle": ..., "generated_sequence": ...}`.
6. `edit` `[{"op": "create", "id": "generated", "type": "ConditionedRead", "params": <the reply's node.params>},
   {"op": "connect", "id": "grade", "input": "image", "source": "generated"}, {"op": "set", "id": "write",
   "param": "path", "value": "<out folder>/shot.%04d.exr"}]`.
7. One `edit` per frame, `[{"op": "render", "id": "write", "path": "<out folder>/shot.0001.exr", "frame": 1,
   "bits": "half"}]`, frames 1 to 24 (needs the disk setting; a single call can hold several render ops,
   and `bits` defaults to half).
8. `artifacts` `{"action": "provenance", "id": <generated_sequence>}`: the chain holds the generated
   sequence, the provider description, the SceneState and the ControlBundle.

Expected from the demo plate (full size, 480 by 270, 24 frames; the first four ids are also what the
hand run gives):

| Result | Value |
| --- | --- |
| SceneState | `b52bc6b3536b15eca47d8c96c2d4d46c4d75532c516e5a09f31e3ea172cfdec7` |
| ControlBundle | `e734808faf740d55be8ab7429532227e08fe3810b9aacdacc25c6d2474a3c708` |
| Generated sequence | `47fdd43adee5f8bddc061dc9b9509802e8caadf1476d6e6c305ef1bc9440877d` |
| Verification report | `3f49adfa081dfcf1d5f6d3121cc4ae62319528f3bd06f7554c970a3993994db1` (verdict PASS) |
| Written frames, pixel digest | `ab52e1da767e59065ccfc024cd1e21397e3690fea87918a39475e8a5e260e766` |

The written frames are not an artifact (they live in the folder you chose and the EXR header holds a
write time), so they are compared by the SHA-256 of their pixels. Export from `render` or from `export`:
both give the same SceneState and ControlBundle. A different text, seed or provider gives a different
generated-sequence id.

## The distorted plate

A filmed plate is distorted by its lens, and the pictures the export makes (depth, normals, motion, ids, the layout
render) are those of an ideal camera. So a distorted plate is worked in the undistorted frame and bent back at the end:
plate, `Undistort`, solve, the layout and the conditioning in the undistorted frame, comp, `Redistort`, `Write`. The brief
for the agent is the same sentence plus "the plate is distorted". `examples/demo_shot/demo_shot_lens.nbcomp` is that shot
(`run_by_hand.py --plate distorted`, `run_by_agent.py --plate distorted`, `compare_runs.py --plate distorted`).

The lens steps, with the tool for each:

1. **Find the lens.** `inspect`: an `Undistort` node on the plate and a `Redistort` before the `Write`, both wired to the
   solved `Camera3D` (`camera`). Its `lens_k1`, `lens_k2` and `focal` are what the camera solve recovered
   (`CameraSolve3D` with `solve_focal` `estimate`, `solve_distortion` `k1 k2` and a Survey of two picks and their distance for
   the scale: the `solve` node of the graph, run once by `solve_camera.py --solver dpvo --distorted`, never at run time: a
   re-solve changes the camera by a millimetre, so the camera that feeds the export is the one kept in the document).
2. **Look at the straightened plate.** `view` the `Undistort` node. Its frame is larger than the format: the data window
   beyond the format is the overscan, the part of the undistorted picture that a Redistort will bend back into the corners.
   In the window, the viewer strip has "View undistorted" and "View plate" for a graph with an Undistort node.
3. **Export in the undistorted frame.** The export node `export` and the `render` node use `camera_os`, the overscan camera:
   the same pose and focal length on a film back taller by the overscan, so Render3D draws the whole undistorted frame at the
   pixel size of the plate, and the same physical lens restated in that larger format's units (the model works in units of
   half the format height: `k1` times the ratio squared, `k2` times its fourth power). `export_conditioning` therefore writes
   a SceneState whose camera has `lens` `distorted: true` and a resolution larger than the plate. A provider conditions on
   undistorted pictures; applying that lens to its result and cropping to the plate's format lands on the plate.
4. **Generate and verify as before.** `generate` reads the undistorted plate (`generate`'s image input is the `Undistort`
   node) and the export; the verifier measures the pictures against the export, all in the undistorted frame. The lens is not
   part of its checks.
5. **Bring it back.** The generated read goes into `fit`, a `Reformat` (to the plate's size, no resize, centred, keep the
   bounding box) that puts the overscan picture in the plate's format with its overscan intact, then `grade`, then `merge` over
   the undistorted plate, `Redistort`, `Write`. Connect `generated` to `fit`, not to `grade`: Merge needs matching formats.
6. **Check the round trip.** The run reports a `lens_round_trip` block apart from the verdict (examples/demo_shot/lens_check.py):
   the plate straightened and bent back against the plate (lens only, judged for every provider), and the written shot
   against the distorted plate (judged for `reproject`, which copies the layout; reported and not judged for a model that
   restyles the picture).

## Worked example: a smaller one, the smoke demo

The same repeatable demo as "Artist workflow and repeatable local demo" in
`docs/GENERATIVE_CONDITIONING.md`, driven through tools only. No paid provider, model download or
network request is involved, and the spend figures are estimates, not charges.

1. `edit`: create `Cube3D` `cube`, `Scene3D` `scene`, `Camera3D` `camera` (`tz` 7) and `Render3D`
   `render` (`width` 160, `height` 120), then connect `cube` to `scene` slot `object0`, `scene` and
   `camera` to `render`.
2. `export_conditioning` `{"id": "render", "first": 1, "last": 1}` returns the two artifact ids.
3. `loop` `{"action": "start", "scene_state": ..., "bundle": ..., "provider": "null",
   "max_attempts": 2, "spend_cap": 0.02, "estimate": 0.01, "unit": "credits (estimate)"}` returns a
   loop id. Poll `loop` `{"action": "status", "loop": ...}` until `finished` is true. The `null`
   sequence carries beauty only: camera, objects (by tracking) and lighting pass, depth, motion and
   normals read "not conditioned", so the first attempt passes and the loop stops as `pass` after one
   attempt (cumulative estimate 0.01, seed 0). The frame is 160 by 120 because the camera check needs
   enough pixels to track.
4. `loop` `{"action": "use_result", "loop": ...}` returns the `ConditionedRead` parameters; `edit`
   creates that node. Its `path` is the generated-sequence artifact id.
5. `artifacts` `{"action": "provenance", "id": <that path>}` lists the generated sequence, the
   provider description, and the SceneState and ControlBundle it was made from.
