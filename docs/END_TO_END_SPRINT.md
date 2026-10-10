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
