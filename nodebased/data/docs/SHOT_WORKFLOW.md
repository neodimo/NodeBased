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
   unchanged) and `reproject`.
6. **Verify.** `verify` (scene_state, bundle, generated_sequence) returns the score card. The loop
   verifies each attempt itself.
7. **Bring it back into the comp.** `loop` `use_result` returns the parameters for a
   `ConditionedRead` node whose `path` is the generated-sequence artifact id. Create the node with
   `edit`, then connect it, grade it and write it like any other input.
8. **Prove where it came from.** `artifacts` `provenance` with the generated-sequence id returns the
   chain of artifacts that produced it, including the SceneState and ControlBundle ids.

## Tools added for the pipeline

| Tool | Arguments | Returns |
| --- | --- | --- |
| `export_conditioning` | `id`, `first`, `last`, `camera` (only for a plain scene node) | `scene_state`, `control_bundle` artifact ids |
| `generate` | `scene_state`, `bundle`, `provider`, `options`, `output_pattern` | `generated_sequence` artifact id |
| `verify` | `scene_state`, `bundle`, `generated_sequence` | `report` artifact id, `verdict`, `score_card` |
| `loop` | `action` = `start`, `status`, `cancel`, `use_result` | loop state, attempts, stop reason |
| `artifacts` | `action` = `list` (`kind`, `limit`) or `provenance` (`id`) | recent artifacts, or the provenance chain |
| `workspace` | `tab` = `Composite`, `3D`, `Simulate`, `Render` | the active tab |

`generate` with `output_pattern` writes frames outside the artifact store, so it needs the same
"Allow save / load / render from the agent" setting as `edit`'s disk commands. Without that
argument the stand-in providers only touch the artifact store.

A loop started from the GUI runs in the background and appears in the Conditioning Queue panel; poll
`status`. The headless dispatcher (`python -m nodebased.agent`) runs a loop to completion before it
replies. `workspace` needs the window and is refused headless.

## Worked example: the smoke demo

The same repeatable demo as "Artist workflow and repeatable local demo" in
`docs/GENERATIVE_CONDITIONING.md`, driven through tools only. No paid provider, model download or
network request is involved, and the spend figures are estimates, not charges.

1. `edit`: create `Cube3D` `cube`, `Scene3D` `scene`, `Camera3D` `camera` (`tz` 7) and `Render3D`
   `render` (`width` 16, `height` 12), then connect `cube` to `scene` slot `object0`, `scene` and
   `camera` to `render`.
2. `export_conditioning` `{"id": "render", "first": 1, "last": 1}` returns the two artifact ids.
3. `loop` `{"action": "start", "scene_state": ..., "bundle": ..., "provider": "null",
   "max_attempts": 2, "spend_cap": 0.02, "estimate": 0.01, "unit": "credits (estimate)"}` returns a
   loop id. Poll `loop` `{"action": "status", "loop": ...}` until `finished` is true. The loop's
   verifier is given the beauty frames only, so the score card is FAIL today and both attempts run;
   the attempt cap stops it as `attempts_exhausted`, inside the spend cap.
4. `loop` `{"action": "use_result", "loop": ...}` returns the `ConditionedRead` parameters; `edit`
   creates that node. Its `path` is the generated-sequence artifact id.
5. `artifacts` `{"action": "provenance", "id": <that path>}` lists the generated sequence, the
   provider description, and the SceneState and ControlBundle it was made from.
