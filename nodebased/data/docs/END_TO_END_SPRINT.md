# End-to-end sprint: one shot, by hand and by an agent

Owner: Gonzo. Asked for by DiMo, #nodebased, Sat 10/10 2:47 PM: "get us a good chunk closer to both our
UI goals and end agentic and deterministic generative workflows with our DCC; find a true end goal plan
and execution." Lanes keep running in parallel; this sprint runs on its own branch (`openclaw/nb-e2e`)
and merges through the same suite as everything else.

## The end goal (the thing we can show)

**One shot, twice.** A filmed plate goes into NodeBased, gets a tracked camera and a small 3D layout,
exports its conditioning (depth, normals, motion, IDs, camera pose, lighting), runs a real generative
model conditioned by that export, is verified against the scene, comes back into the comp as another
input, is graded and written out. Then the same shot is done a second time by an attached agent from
one sentence of brief, through the same commands, and produces the **same bytes** (pinned seed and
settings, full provenance chain). Both runs happen inside the new look.

That is the VISION.md thesis made concrete: generative and deterministic work in one graph, one undo
stack, one cache; the human and the agent use the same validated commands; non-determinism is
contained, never hidden.

## Where we stand (10/10)

- **UI:** stages 1 to 5 of the new look and the GUI sprint are on main. Properties (stage 6) and motion
  (stage 7) are finished on their branch and in the suite run that started at 1:58 PM. The four
  workspace tabs exist; Simulate and Render today only switch the viewer and open a dock.
- **Deterministic side:** camera tracking, 3D scene, Render3D passes, SceneState and ControlBundle
  export, CPU verification, content-addressed artifacts with provenance, queue chains, LAN workers,
  bounded loops: all built and tested.
- **Generative side:** the provider contract exists with two deterministic stand-ins (`null`,
  `reproject`). No real model has ever run through it. There is no seed field in the generate path.
- **Agentic side:** the Agent panel runs Claude Code or Codex against this session over MCP, but the
  MCP exposes only graph editing, viewing, errors, knowledge and issues. An attached agent cannot
  export conditioning, generate, verify, run a loop or look up provenance. That is the biggest gap
  between "agent edits a graph" and "agent delivers a shot".

## Stages

Each stage ends in a merge to main through the full suite and a short post in #nodebased. Each is sized
for one worker step (a few hours), reviewed and display-checked by Gonzo.

1. **The agent gets the whole pipeline.** New MCP tools: `export_conditioning`, `generate`, `verify`,
   `loop` (start, status, cancel), `artifacts` (list, provenance), `workspace` (switch tab), plus a
   `knowledge` topic "shot_workflow" that tells an agent the order of operations. Every tool goes
   through `Window.agent_command`, so it shows in the panel's activity log and respects the
   disk-ops setting. Test: an agent script builds the smoke demo from GENERATIVE_CONDITIONING.md
   through MCP alone. *Needs no decision; starts today.*
2. **Determinism is a contract.** `seed` and `sampler_settings` on Generate and in the provider
   description (honoured or not, like every other control); the artifact ID of a result covers them;
   a test runs the same brief twice and asserts byte-identical outputs with the stand-ins; a
   `determinism` line in the verification report. *Needs no decision.*
3. **The first real model.** Recommendation: a local depth-and-normals-conditioned image diffusion
   model that fits the RTX 3080 Ti's 12 GB, run per frame, no spend, no network. It exercises the
   full contract (depth, normals, IDs, camera as reprojection check) and is reproducible on this
   hardware with a pinned seed. A hosted video model with camera control comes next as a second
   provider; it needs an API key and money. *DiMo's call: local image model first, or wait for the
   video API.*
4. **Shot from a sentence.** The repeatable demo scene: a short plate with a tracked camera and the
   mixed scene laid in. An attached agent gets "replace the backdrop with a rainy street, keep the
   cube and sphere, match the camera" and delivers it through stage 1's tools; the human does the
   same by hand. Both results compared and archived with provenance.
5. **UI: finish and prove.** Display QA of the whole new look against the mockups on this machine's
   real display (the 9/30 method), punch list fixed; the Simulate and Render tabs get a real layout
   (Simulate: viewport + cache inspector + timeline; Render: queue + write nodes + output preview)
   or are removed; Agent panel and Conditioning Queue restyled to the new look since the demo lives
   in them.
6. **Release with the demo.** Tag, then a short capture of the two runs appended to the release.

## Order and cost

Stages 1 and 2 start now, one worker at a time (Claude week at 45%, resets Mon 10 PM; Codex 42%).
Stage 3 starts when DiMo picks the provider. Stage 5's QA runs alongside, by Gonzo on the real display.
Expected: stages 1, 2 and 5's QA this weekend; 3 and 4 next week; 6 when 4 holds up.

## Done means

An attached agent, given one sentence and this repository, produces the shot with the same artifact
IDs as the hand-made one, and the release notes link a capture of both. Anything short of that is
progress to report, not done.

## Stage 4a: the demo shot by hand (10/10)

Built before the real model exists, so stage 3 only swaps the provider. Files and how to run them:
`examples/demo_shot/README.md`; the agent's script: `examples/demo_shot/BRIEF.md`.

- **The plate** is rendered by NodeBased itself (the repository has no filmed footage): 24 frames,
  480 by 270, EXR with beauty, depth, normals, motion and object ids; a textured ground, two set
  pieces, the mixed scene's cube and sphere, a key light, an environment light, and a camera that
  dollies in with a small pan. Frames are not committed; `make_plate.py` regenerates them into the
  cache (about 24 s) and checks per-frame pixel digests against `plate.sha256.json`.
- **The solve** tracks 49 survey points with `nodebased.tracker` and fits the camera per frame from a
  generic starting camera. Against the camera that rendered the plate: position 2.9 cm mean and 5.2 cm
  worst, rotation 0.15 and 0.32 degrees, field of view 0.12 and 0.37 degrees, reprojection 0.64 px
  worst. The graph uses the solved camera. It converges; position is outside the verifier's 1 cm
  tolerance, which matters only when the plate is checked against it (the verdict judges the generated
  pictures against the exported, solved, camera).
- **The hand run** (`run_by_hand.py`, headless through `nodebased.agent`, provider `reproject` 1.1,
  seed 7, the brief in the Generate node's text control, sent in the one `generate` call and recorded
  as ignored): 155 s at full size on this machine's CPU, byte-identical on a second run with a fresh
  store. Artifact ids:
  - scene_state `b52bc6b3536b15eca47d8c96c2d4d46c4d75532c516e5a09f31e3ea172cfdec7`
  - control_bundle `e734808faf740d55be8ab7429532227e08fe3810b9aacdacc25c6d2474a3c708`
  - generated_sequence `47fdd43adee5f8bddc061dc9b9509802e8caadf1476d6e6c305ef1bc9440877d`
  - report `3f49adfa081dfcf1d5f6d3121cc4ae62319528f3bd06f7554c970a3993994db1`
  - verdict **PASS** (lighting and objects pass; camera, depth, motion, normals "not conditioned");
    output pixel digest `ab52e1da767e59065ccfc024cd1e21397e3690fea87918a39475e8a5e260e766`.
- **Stage 4a fix (10/10).** The first run's verdict was FAIL on camera, lighting and objects. The plate
  checked against its own SceneState failed too, so the cause was the verifier and the export as well as
  the stand-in: the export and verifier left out the Render3D's ambient (now recorded per frame in the
  SceneState), the layout left out set pieces the plate shows, camera landmarks were tracked from the
  observation's own first frame (zero error on frame 1 by construction, drift after) and are now matched
  against a render of the exported scene on the same frame (visible vertices with corner structure only),
  object ids collided for unnamed meshes of one type, a scene with no shadow-casting light was counted as
  missing shadow evidence, and `reproject` 1.0 put frame N+1's picture under frame N (1.1 carries frame
  N-1 forward). No threshold changed. Plate against its own SceneState: PASS on camera, objects and
  lighting (`diagnose_verdict.py`). `reproject` no longer declares camera pose honoured (a resampled copy
  measures up to 29 cm and 1.7 degrees off at this size), so its camera check reads "not conditioned".
  Also: one `generate` call does the shot (an unhonoured text brief is recorded as ignored in the reply,
  the manifest and the provenance row, not refused; colour space, frame count and size still refuse); the
  reply carries the ConditionedRead node parameters, identical to a loop's `use_result` and tested; the
  provider worker limits numpy's threads itself and sizes its cap from the job.
- **Findings for Gonzo.** (1) VerifyConditioning, SceneState and ControlBundle are not graph nodes;
  WriteGeo3D is the export node and verification is an op. (2) The headless `render` op writes RGBA only,
  not the layers. (3) A camera held to 1 cm cannot be verified from pixels at 480 by 270 for a provider
  whose output is not pixel-exact; stage 3's real model will need either a larger plate or a looser,
  stated camera tolerance for pixel-solved cameras, DiMo's call.

## Stage 4b: the same shot by an attached agent (10/10)

- **The scripted agent** (`examples/demo_shot/run_by_agent.py`, built on the reusable headless client
  `nodebased/mcpclient.py`): an offscreen NodeBased window with the `--agent` bridge on the shot graph, the
  stdio MCP server attached, and a fixed plan with no model in the loop. It gets the one-sentence brief and the
  plate folder and reaches the written frames through MCP tools alone (`knowledge`, `describe`, `inspect`,
  `edit`, `export_conditioning`, `generate`, `verify`, `edit` for the generated read, grade input and Write
  path, `edit` `render` per frame, `artifacts`). Its exact sequence is the worked example in
  docs/SHOT_WORKFLOW.md. The test is of the tools and the contract.
- **The comparison** (`compare_runs.py`, both runs with fresh stores, 309 s at full size): SceneState,
  ControlBundle, generated sequence, verification report, verdict, determinism section and the written
  frames' pixel digest all match, and equal the Stage 4a ids above. The graph needed no change: the hand run
  and the agent plan already build the same document. The written frames have no artifact id (they leave the
  store, and the EXR header holds a write time), so that row compares the SHA-256 of the pixels and says so.
- **Two bugs the scripted agent and the real one found.** (1) The desktop bridge had no `render` op: `edit`
  with a render command failed with "cannot run inside a group" (the headless dispatcher had it; both now
  share `agent.render_op`). (2) The MCP server dropped the connection after 30 s while the window was still
  exporting 24 frames; export, generate, verify, loop, artifacts and render replies now wait up to an hour.
- **A real agent, once** (`run_by_claude.py`: Claude Code 2.1, model claude-opus-5-5, `claude -p` with the
  nodebased MCP server registered the way the Agent panel does, a fresh store, given the brief, the plate
  folder and the output folder). 22 nodebased tool calls and 221 s, about 1.02 US dollars. Sequence:
  `knowledge` shot_workflow, `inspect`, `knowledge` protocol, `knowledge` limits, `edit` (plate and Write paths,
  view), `errors`, `export_conditioning` (on `render`, not `export`: same ids), `generate` (provider, seed and
  the graph's text verbatim), `verify`, `edit` (a ConditionedRead made from the reply, connected to `grade`),
  render of frame 1, one `edit` holding the other 23 renders, then `errors`, `artifacts` provenance and a few
  viewing calls. **Its ids matched the hand run's**: SceneState, ControlBundle, generated sequence and report,
  and the written frames' pixel digest, all equal to the Stage 4a values.
  - Where it went wrong: the live viewer reported "NameError: name 'self' is not defined" on the
    ConditionedRead chain, and the agent spent eight calls (`errors`, extra `view`, a temporary Read node,
    `reference_context`) deciding whether its frames were good. The cause was a real bug: the tile executor's
    canvas-size walk used `self` and a wrong variable inside a module function when a chain started at a
    ConditionedRead (fixed, with a test in tests/test_conditioned_read.py). It also said plainly that the
    rainy street was not generated, which is true of the `reproject` stand-in.
  - What SHOT_WORKFLOW.md now says so it would not wander: the worked example names the node ids of the demo
    graph and that `render` and `export` give the same ids, says to send the brief character for character,
    says to report `text` as ignored rather than describe a street, and lists the five expected ids to check
    against. Still open: the topic does not tell an agent how to confirm a result when the viewer shows an
    error (compare the written frames' pixels, not the viewer); worth one sentence once the viewer is clean
    on a full run.
