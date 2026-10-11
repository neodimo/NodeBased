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
7. **Real-time camera tracking.** Today's solve needs survey points (`examples/demo_shot/solve_camera.py`
   fits the camera to 2D tracks of known points) and cannot work on a plate with no survey. The target
   is a learned monocular SLAM that returns per-frame poses, intrinsics and a sparse point cloud at or
   near frame rate on the RTX 3080 Ti, as a `CameraSolve3D` node backed by a GPU subprocess like the
   model provider. Candidates: DROID-SLAM (dense bundle adjustment, the accuracy reference, heavy),
   DPVO / DPV-SLAM (its successor from the same group, several times faster and a fraction of the
   memory, real-time on consumer cards, the first to run), MASt3R-SLAM (real-time dense SLAM on a 3D
   reconstruction prior, no fixed camera model, the one to watch). Benchmark on the demo plate, then on
   a filmed plate from DiMo. Measured: pose error against truth, frames per second, VRAM, and whether
   the result drives the conditioning export unchanged.

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

## Stage 3: the first real model, SDXL with ControlNet-Union (10/10)

DiMo chose a local model. `sdxl_controlnet` 1.0 is in (see `docs/GENERATIVE_CONDITIONING.md` section 8):
SDXL base, the fp16-fix VAE and ControlNet-Union (promax), run per frame on the RTX 3080 Ti through CUDA only,
in its own interpreter (`NODEBASED_SDXL_PYTHON`), so the repository environment still has no torch. It
honours depth, normals, text, seed and sampler settings; it ignores motion, IDs, camera pose and reference
frames. No temporal model: each frame is made from its own depth and normals with the same seed, so flicker
between frames is expected and the motion check reads "not conditioned".

**The demo shot through it** (`python examples/demo_shot/run_by_hand.py --provider sdxl_controlnet`): the
480 by 270 plate is resampled to the model's 1024 by 576 working size and back (the plate is not re-rendered).
Brief: "replace the backdrop with a rainy street, keep the cube and sphere, match the camera"; seed 7; defaults
30 steps, guidance 5.0, ControlNet scale 0.8, depth and normals together.

| | |
|---|---|
| Generation | about 340 s for 24 frames (about 14 s a frame), model load about 5 s; whole run about 540 s |
| Peak VRAM | 7.8 GiB allocated, 8.2 GiB reserved (model CPU offload on) |
| Verdict | FAIL: camera, depth, motion and normals "not conditioned" (the generated EXRs carry beauty only); lighting FAIL; objects FAIL |
| generated_sequence | `f06915babb08173436a06e03227a6a5be7a5694b4f5c74708501fb8c5cd45caf` |
| scene_state | `b52bc6b3536b15eca47d8c96c2d4d46c4d75532c516e5a09f31e3ea172cfdec7` |
| control_bundle | `e734808faf740d55be8ab7429532227e08fe3810b9aacdacc25c6d2474a3c708` |
| verification report | `f06177b1a6faf90bcfc75908456583a1f62f4d6f8352a84f8f9147bb90ff906e` |

The run was repeated once: the same four ids and the same output pixel digest. Stills (first, middle, last
frame of the graded output): `docs/images/demo_shot/sdxl_controlnet_frame_0001.png`, `_0012.png`, `_0024.png`.
What they show: the cubes, the sphere and the pillar stay where the exported depth and normals put them, the
lighting and the surfaces change from frame to frame (flicker), and the backdrop becomes a dark textured wall,
not a rainy street: SDXL reads the sentence as a description and the control images fix the layout. The verdict
is FAIL because the lighting and object checks measure the picture against the layout render and SDXL
restyles it; this is an honest result, not a tuning target for this step.

Things found on the way:

- The worker's address-space cap would kill CUDA, so a description may declare `local_gpu` and its job runs
  without the cap (tests: `tests/test_local_gpu.py`). Every other provider keeps the cap.
- A bug older than this stage: a `ConditionedRead` whose path is a generated-sequence artifact id rendered
  frame 1 for every frame (the cache key did not include the frame), so the demo's written shot (stand-in
  `reproject` too, checked on this branch's head) was one frame repeated. Fixed in the evaluator and the tile executor, with a regression test.
- Determinism on this card needed one rule: the memory strategy must not depend on free VRAM. A run that fell
  back to CPU offload after an out-of-memory error (another program was holding 1 GB of the card) gave other
  bytes than the same run without the fallback, so offload is now always on (`NODEBASED_SDXL_OFFLOAD=never`
  turns it off for a card to itself, about twice as fast, peak about 10.4 GiB).
- Nothing was downloaded for this step; every model came from the local Hugging Face cache.


## Stage 7: real-time camera tracking, the spike (10/10)

Question: can a learned monocular SLAM replace the survey-point solve, in real time on the RTX 3080 Ti, and
can its camera drive the conditioning export unchanged? Answer on the demo plate: yes for DPV-SLAM, with
conditions below. Not tried on a filmed plate: the demo plate is a clean synthetic render with exact ground
truth, so every number is a best case.

**Setup** (`projects/nb-providers/slam/setup.sh`, rerunnable; the sizes are in its header): no `nvcc` exists
here and the pip package `nvidia-cuda-nvcc-cu12` ships only `ptxas`, so the real nvcc 12.6.85 comes from NVIDIA's
redistributable archive (52 MB with cudart and cccl) and gcc 13 from conda-forge through micromamba (0.9 GB on
disk); the system gcc 16.2 breaks nvcc ("false_type is undefined"), and `-allow-unsupported-compiler` does not
help. Both repositories use torch's removed `Tensor.type()` in their dispatch macros (one `sed`). DPVO and
DROID-SLAM both install a `lietorch_backends`, so DROID-SLAM lives in its own folder. Downloads: DPVO 230 MB,
DROID-SLAM 169 MB, weights `dpvo.pth` 14 MB and `droid.pth` 16 MB, Eigen 3.6 MB. Torch 2.14 cu126 is shared
with the SDXL environment through a `.pth` file.

**Measured**, demo plate (24 frames, camera dolly 1.47 m), estimated camera centres aligned to truth with a
similarity transform (rotation from the camera orientations, because the path is nearly a straight line and
positions alone leave the roll about it free; scale and shift from positions). Frame rate is the per-frame
tracking after the 8-frame warm-up; peak VRAM is the solver process's own allocation.

| run | translation RMS / worst | rotation mean / worst | frames per second | peak VRAM |
|---|---|---|---|---|
| DPVO, 480 x 270 | 8.5 cm (5.8% of the path) / 16.8 cm | 0.15 / 0.28 deg | 251 (49 with the final optimisation) | 0.19 GB |
| DPVO, 960 x 540 | 2.3 cm (1.6%) / 5.9 cm | 0.09 / 0.15 deg | 45 (30) | 0.50 GB |
| DPV-SLAM (loop closure), 480 x 270 | 7.7 cm / 15.7 cm | 0.10 / 0.24 deg | 248 (48) | 0.47 GB |
| DPV-SLAM (loop closure), 960 x 540 | 1.8 cm (1.2%) / 5.3 cm | 0.05 / 0.10 deg | 45 (29) | 0.77 GB |
| DROID-SLAM, 960 x 540 (resized to 584 x 328) | 1.3 cm (0.9%) / 2.3 cm | 0.09 / 0.16 deg | 335 tracking only (30 with the final optimisation) | 3.2 GB |
| DROID-SLAM, 480 x 270 | 34.6 cm (24%) / 57 cm | 3.1 / 6.4 deg | same | 3.2 GB |

- **Resolution decides accuracy.** DROID-SLAM resizes both plates to the same 584 x 328, yet the 480 plate gives
  a 24% error and the 960 plate 0.9%: the 480 plate is too coarse for it. DPVO is the same story at a smaller
  gap. Use the plate at 960 wide or more.
- **The focal length is not recovered.** DPVO and DROID-SLAM both take the intrinsics they are given (square
  pixels, centre principal point). A focal length 15% too long or 13% too short moved DPVO's error from 2.0 cm to
  2.6 and 2.9 cm, so the lens does not need to be exact, but it has to come from the plate (the node's focal and
  vertical aperture). MASt3R-SLAM, which estimates the camera model, is the candidate that would fix this.
- **Loop closure** changes nothing measurable on a 24-frame dolly (there is no loop); it costs 0.27 GB.
- **DROID-SLAM keeps 3 keyframes** of the 24 (its motion filter); DPVO tracks every frame.
- **Determinism.** Two DROID-SLAM runs were bit-identical. DPVO is not: unseeded runs differ by up to 0.1 solve
  unit (random patch choice); seeded runs (`seed` on the node: torch, numpy, CUDA) differ by about 3e-4 solve
  units, because the bundle adjustment kernel adds floats with CUDA atomics. After placement that is about 1 mm
  of camera position, and two full hand runs of the shot with `--solver dpvo` gave different artifact ids
  (below). Rounding the keys to 1 mm makes equal keys likely (74% of the keys matched in a test, 96% at 1 cm),
  never certain. So the camera that feeds the export is the one solve kept in the document: the op runs on
  request, never when the graph evaluates, and a re-solve replaces the keys and the ids on purpose.

**The integration.** `nodebased/providers/camera_solve_dpvo.py` (numpy side plus the DPVO side, started as a
subprocess with `NODEBASED_SLAM_PYTHON`, `local_gpu` declared in `providers/solvers/camera_solve_dpvo.json`;
the description sits in its own folder so `generate` does not offer a solver as a conditioning provider), the
`CameraSolve3D` node (plate passes through, like `Generate`) and the agent op `solve_camera`
(`nodebased/camera_solve.py`; also served by the desktop bridge). The op reads the plate through the display
transform, runs the provider, places the solve in the world and writes into the document: a `Camera3D` with a
key on every frame (position, target and roll, rounded to 0.001), a `ReadSplat3D` holding the sparse cloud as
tiny coloured splats (the nearest existing scene type for points), and the id of the solve record (a new
artifact kind, `camera_solve`: provider and version, seed, scale and its source, every key) in the node's
`solve` parameter. The solve is up to scale and in its own frame, so two controls place it:

- **Anchor:** the camera on the first frame (position, target and roll on the node), the camera an artist sets
  by eye with the Camera3D handles; the solved motion is applied from there.
- **Scale:** the node's `unit_scale` (scene units per solve unit), or `survey` on the op: two points with their
  real distance and their pixel positions on two or more frames, triangulated with the solved cameras. Picking
  the survey pixels in the viewer is not built; the op takes them as numbers.

**The shot with that camera** (`run_by_hand.py --solver dpvo`, 480 x 270 committed plate, `reproject` 1.1, seed
7, the first-frame camera equal to the plate's true one and the two survey points clicked to a whole pixel):
verdict **PASS**, the same checks as the survey run (the camera check reads "not conditioned" for `reproject`,
so the verdict does not judge the camera; the ids show that the export did take it).

| | survey solve (stage 4a) | DPVO solve, first run |
|---|---|---|
| camera against the plate's: position mean / worst | 2.8 / 5.2 cm | 8.2 / 25.6 cm |
| rotation mean / worst | 0.15 / 0.32 deg | 0.17 / 0.41 deg |
| scene_state | `b52bc6b3536b15eca47d8c96c2d4d46c4d75532c516e5a09f31e3ea172cfdec7` | `93b0d86d3465ac158a6fc9bb840b9498b427a11cb16ae2255006b4592912a301` |
| control_bundle | `e734808faf740d55be8ab7429532227e08fe3810b9aacdacc25c6d2474a3c708` | `970a607815f782f9ea0501094ac75ee04cb7d5e28f848cbdbac0bd7b814436e5` |
| generated_sequence | `47fdd43adee5f8bddc061dc9b9509802e8caadf1476d6e6c305ef1bc9440877d` | `21a5ae37677942e8ef6b520d33fc4de830db984cc57e2cabc00ca1e401e45e0e` |
| report | `3f49adfa081dfcf1d5f6d3121cc4ae62319528f3bd06f7554c970a3993994db1` | `909872e0f516915919363aec8b63f04842f4557e3d4c3066e9520e640dcdeffb` |
| solve record | | `dc5ecaf2a007238dc3a60f5643454d5d7c02a6259d599172b824765c28c17fd9` |

A second full run gave scene_state `392bd71bba8425b454ebc2a469096e334ed4d71f56512a27300ac150bc3f546b` and a PASS
verdict: the camera keys moved by 1 mm, so every id after the solve differs (the survey run reproduces its ids
exactly). On the 960 x 540 re-render the same op, anchor and survey placement gives camera error 1.4 cm mean and
6.5 cm worst, 0.04 and 0.07 deg: better than the survey solve in rotation and mean position. The 480 run's 8 cm
is mostly the scale: two clicked points a pixel off at 480 wide put it 9% off; at 960 it is 1%. Run it on the
real card with `NODEBASED_SLAM_TEST=1 python -m unittest tests.test_camera_solve`.

**Recommendation.** DPV-SLAM first. It runs at 45 frames per second on a 960 x 540 plate in half a gigabyte, so
it is real time with room for the model provider next to it, and it is within a centimetre or two of DROID-SLAM
on this plate. DROID-SLAM is the more accurate and the only repeatable one, but uses six times the memory (3.2 GB) for the same
30 frames per second end to end on this short plate; keep it as the reference and a second provider. Neither is proven
beyond a clean synthetic render: the next step is a filmed plate from DiMo, and MASt3R-SLAM for lens recovery.
Not built: viewer picking of the survey points, a tile-path or queue job for the solve, and an automatic
ground-plane alignment (the anchor is typed by hand).
