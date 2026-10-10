# The agent run (stage 4b)

## The brief (the only thing the agent is told)

> replace the backdrop with a rainy street, keep the cube and sphere, match the camera

Setup the agent starts from: NodeBased running with `examples/demo_shot/demo_shot.nbcomp` open, the
plate rendered (`python examples/demo_shot/make_plate.py`), the "Allow save / load / render from the
agent" setting on, and the Read node's path pointing at the plate (`edit` `set`). The agent reads the
`shot_workflow` knowledge topic first.

## Expected tool sequence

Each step is an MCP tool call; ids come from the previous reply.

1. `knowledge` `{"topic": "shot_workflow"}`, then `inspect`: the graph already has the plate, the
   solved camera, the layout, the export node `export` and a `generate` node whose provider is
   `reproject`, seed 7, and whose text control holds the brief.
2. `export_conditioning` `{"id": "export", "first": 1, "last": 24}` returns `scene_state` and
   `control_bundle`.
3. `generate` `{"scene_state": ..., "bundle": ..., "provider": "reproject", "seed": 7}` with no text:
   the reply's `controls` says `text` is ignored, and sending it makes the provider refuse the call
   (stage 4a recorded that refusal), so the agent omits it and reports the brief as not honoured.
   The brief's text does not enter the artifact id when it is omitted, which is what makes the ids
   comparable with the hand run.
4. `verify` `{"scene_state": ..., "bundle": ..., "generated_sequence": ...}` returns the verdict and
   the score card (expected: FAIL on camera, lighting and objects; depth, motion and normals "not
   conditioned").
5. `loop` `{"action": "start", "scene_state": ..., "bundle": ..., "provider": "reproject",
   "max_attempts": 1, "spend_cap": 0.01, "estimate": 0.01, "unit": "credits (estimate)", "seed": 7}`,
   poll `loop` `status` until finished, then `loop` `{"action": "use_result", "loop": ...}`. This is
   the only tool that unpacks the ConditionedRead parameters (`path`, `manifest`, `scene_state`); the
   one-attempt loop regenerates with the same inputs and returns the same generated-sequence and
   report ids as steps 3 and 4 (checked in stage 4a).
6. `edit` creates the generated read and wires the rest: `create` `generated` as `ConditionedRead`
   with the `use_result` parameters; `connect` `grade` input `image` to `generated`; `set` `write`
   `path` to the output pattern.
7. `edit` `render` on node `write` for frames 1 to 24 (disk operations setting).
8. `artifacts` `{"action": "provenance", "id": <generated_sequence>}` for the chain.

## What to compare with the hand run

`run_by_hand.py` prints `scene_state`, `control_bundle`, `generated_sequence`, `report`, the verdict and
`output_pixel_digest`. The ids in docs/END_TO_END_SPRINT.md ("Stage 4a") are the hand run's. The agent
run must produce the same four ids and the same output digest (the EXR header holds a write time, so
compare pixels, not file bytes: `make_plate.frame_digest`).
