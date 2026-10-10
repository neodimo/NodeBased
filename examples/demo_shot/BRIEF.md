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
3. `generate` `{"scene_state": ..., "bundle": ..., "provider": "reproject", "seed": 7, "options":
   {"text": "<the brief>"}}`: one call. The provider does not use text, so the reply's `controls` says
   `text` is ignored and its `ignored` list names it; the record (generation manifest, provenance row)
   says the same. The reply also carries `path`, `manifest`, `scene_state` and `node`, the ConditionedRead
   parameters. The text is part of the artifact id, so the agent must send exactly the text in the
   `generate` node's text control (the brief's sentence as stored in the graph) for its ids to match the
   hand run's.
4. `verify` `{"scene_state": ..., "bundle": ..., "generated_sequence": ...}` returns the verdict and
   the score card (expected: PASS; lighting and objects pass, camera, depth, motion and normals "not
   conditioned").
5. Optional: a bounded `loop` `{"action": "start", ..., "max_attempts": 1, "seed": 7}` regenerates
   with the same inputs; its `use_result` returns the same ConditionedRead node as a `generate` call made without text and the
   same generated-sequence id (a loop's `text` selects text feedback, which a provider that ignores text
   stops on, so it carries none); `tests/test_agent.py` asserts it.
6. `edit` creates the generated read and wires the rest: `create` `generated` as `ConditionedRead`
   with the `node` parameters from step 3; `connect` `grade` input `image` to `generated`; `set` `write`
   `path` to the output pattern.
7. `edit` `render` on node `write` for frames 1 to 24 (disk operations setting).
8. `artifacts` `{"action": "provenance", "id": <generated_sequence>}` for the chain.

## What to compare with the hand run

`run_by_hand.py` prints `scene_state`, `control_bundle`, `generated_sequence`, `report`, the verdict and
`output_pixel_digest`. The ids in docs/END_TO_END_SPRINT.md ("Stage 4a") are the hand run's. The agent
run must produce the same four ids and the same output digest (the EXR header holds a write time, so
compare pixels, not file bytes: `make_plate.frame_digest`).
