# Proposed worker step: VACE depth + reference against the stage 8 shot

**Proposed 2026-10-12 by Lane 3. Not executed or scheduled.** One bounded experiment, no production provider, UI or model training. Owner after selection: Gonzo assigns a generative-pipeline worker. The research step authorizes neither downloads nor a GPU run; a separately assigned execution step must cover both. [Decision context](../../docs/RESEARCH_RADAR.md).

## Goal

Determine whether **Wan-AI/Wan2.1-VACE-1.3B-diffusers**, through `diffusers.WanVACEPipeline`, improves the existing shot while fitting the RTX 3080 Ti and preserving repeatability. Candidate checkpoint revision inspected in this sweep: `ec4d2cb062b548996b179d493fdd05340de702a1` (Hugging Face API, accessed 2026-10-12). Use this immutable revision unless a separately documented compatibility blocker requires a new proposal. Use only this model/configuration. Deliver a go/no-go record even when memory, runtime or image quality fails. No automatic adoption follows a pass.

## Inputs and setup — 15 minutes maximum

- Use the exact demo export and result identified in `benchmarks/demo_shot/sdxl-stage8-v1.2-canvas.json`; resolve its sequence/report artifact IDs and the matching SceneState/ControlBundle through the existing artifact store. Require all 24 frames and the original generated frame 1. Confirm the source digests against the stage 3/8 sprint records. If missing, report that prerequisite; rebuilding the baseline is a separate step.
- Preserve the original prompt, **“replace the backdrop with a rainy street, keep the cube and sphere, match the camera”**, and seed **7**. No prompt extension or network call in generation. Download/cache preparation, if separately authorized, must finish before the inference portion; report download time separately.
- Use an isolated provider environment. Record immutable checkpoint revision and file digests, interpreter, torch/diffusers/transformers/accelerate versions, CUDA/driver, sampler config and OCIO config digest. The baseline records diffusers 0.41.0; verify its VACE arguments against the installed signature rather than upgrading the existing SDXL environment.
- Fixed configuration: **30 steps, guidance 5.0, VACE conditioning scale 1.0, negative prompt empty**, checkpoint's shipped scheduler/config, bf16 transformer/text encoder and fp32 VAE, sequential CPU offload and VAE tiling. These are experiment settings, not a published optimum. Pin tiling settings. CPU generator seed 7; deterministic algorithms on, TF32 off, no xformers/TeaCache/compilation or adaptive memory fallback. If a required deterministic kernel is unavailable, record a failed repeatability prerequisite.

## Exact conditioning adapter — 10 minutes maximum

1. Read the synchronized `.controls/manifest.json` with `nodebased.control_bundle` and the corresponding SceneState. Assert dimensions, colour tags, frame ordering and digests. Do not estimate depth from beauty: exported `depth.Z` is camera-space metric depth.
2. Encode depth with the current provider's **sequence-wide inverse-depth range**: `255 * (1/d - 1/far) / (1/near - 1/far)`, clipped, near white, far/empty black; identical RGB channels. Determine valid near/far once from the whole shot, exclude empty samples using the export's validity conventions, and save these bounds. Resize depth with nearest-neighbour to avoid blending object boundaries. VACE's published annotator normalizes estimated depth per frame; this deliberate sequence-wide mapping is a distribution-mismatch risk to record, not a proved equivalent.
3. Preserve aspect: scale **480 × 270 → 832 × 468**, pad **6 rows top and bottom** to **832 × 480**. Depth padding is black. Convert the existing SDXL frame 1 from ACEScg to display sRGB through the baseline's fixed colour transform, scale with bicubic, and apply the same padding. Pass it as `reference_images=[reference]`; this is appearance guidance, not a guaranteed copied first frame.
4. Supply the 24 depth images as `video`, append one duplicate of depth frame 24, and request **25 frames** (Wan temporal size is `4k+1`). Use an **all-white mask for every depth frame**, following the depth-control path's default `mask=None`/ones behaviour in the cited pipeline. Do **not** substitute the temporal verifier's occlusion mask: a VACE edit mask means generate/retain, not valid/invalid correspondence. Save the actual tensors/images used.
5. Exported motion and bit-preserved IDs remain verifier inputs for warping/occlusion masks. Normals, camera matrices, lights and raw IDs are **not model inputs in this experiment**. Do not declare direct camera/lighting/normals control. The depth sequence implicitly reflects geometry/camera motion, which is a weaker statement.
6. Crop six padding rows off each generated frame, resize to **480 × 270**, use the current provider's inverse sRGB/Rec.709-to-ACEScg conversion and write beauty-only RGBA EXRs with alpha 1. Discard only generated frame 25. Map frames 1–24 to their original scene times: no interpolation, fps reinterpretation or resampling of time. The 16 fps training cadence is a distribution risk, not permission to change the shot.

## Execute and measure — 30 minutes maximum

Acquire the existing **exclusive** GPU lock before either run; do not overlap the integrator's suite. Waiting longer than five minutes ends this attempt as blocked, with no process left waiting. No weights, caches or generated media in a RAM-backed temporary directory.

Make **two fresh generations**, each capped at **15 minutes including model load**. Separate output stores; no reused output cache. Same conditioning bytes, reference, seed, settings and memory strategy. Log load/generation/decode/encode times, peak CUDA allocated/reserved memory, whole-device peak and host-RAM peak. Set an **11 GiB peak reserved/device-use target** to leave room on the 12 GB card; out-of-memory or timeout is a recorded no-go, not a trigger for changing precision or resolution. Ensure cleanup releases the lock and stops the model process.

## Decision and evidence — 15 minutes maximum

Use `nodebased.temporal` and the existing verification pipeline unchanged on the original 24-frame export. Source of exact baselines: `benchmarks/demo_shot/sdxl-stage8-v1.2-canvas.json` (2026-10-11), which records **2.644 mean, 7.6702 p95, 0.9895 compared fraction, 23 frame pairs, 584.15 s generation**. The sprint's displayed 2.6/7.7 and 584.2 are rounded.

**All promotion gates must pass:**

- Both runs produce byte-identical EXRs and matching content identities; report separately any volatile timing/report fields. Seed metadata is not a substitute for comparing files.
- Proposed meaningful gain: flicker **mean ≤2.0**, **p95 ≤7.6702**, all **23 pairs** measured, with unchanged masks/coverage (baseline **0.9895**, within rounding tolerance). These are experiment gates, not altered product thresholds. Current product acceptance remains mean ≤4 / p95 ≤12.
- First-run generation including the discarded extra frame **≤584.15 s**; memory within the target above. Include total runtime and host RAM even if this passes. No linear extrapolation from another GPU.
- Compare frames **1, 12 and 24** beside both layout and stage 8 output. Require a recognisable rainy-street backdrop, retained cube/sphere, no new severe edge ghosting or loss of detail. A frozen or blurred picture that wins flicker fails this gate. If no human/display review is possible, mark this gate unverified and do not promote.
- Report every lighting/object sub-check, raw observations and tracking failures against the stored stage 8 report. Require no new failure and no worsening of available numerical errors; identify any genuine improvement separately. Existing lighting/object FAILs may remain, so even a temporal win is **not a binding-compliant provider**. Camera remains undeclared; retain any diagnostic camera fit rather than claiming a pose lock. Do not copy input passes into output to manufacture verifier success.

**Fallback:** retain `sdxl_controlnet` 1.2 unchanged. On any no-go, document the limiting metric; do not download Fun or try another model within this step. A subsequent Fun V1.1 control/camera experiment is a separate decision.

## Deliverable and stop — 10 minutes maximum

Budget: **80 minutes active work**, plus at most five minutes lock wait; hard stop **90 minutes**. No production registration, UI changes, training or search over seeds. The proposed harness belongs to deliberate worker scratch; commit only the compact experiment record and reviewable evidence approved for the repository.

Write `docs/experiments/T1-vace-depth-reference.md` and its byte-identical bundled copy: resolved revisions, input/output digests, fixed settings, two-run timing/memory/flicker table, verifier deltas, visual-gate status and pass/no-go conclusion. Heavy images/logs stay in a deliberate disk-backed scratch directory; the report must name their exact location and cleanup/retention owner. Never put machine-home paths in tracked files. Run `tests.test_knowledge` with the shared environment, worktree `PYTHONPATH` and offscreen Qt before committing documentation.

Before reporting done, write your completion note to the assigned NodeBased worktree's `TASKLOG.md` and `context/state.md` (newest first): **what was done** with evidence vs inference; **artifacts** and local/committed/scratch status; **state** and unverified limits; **next owner + concrete artifact**; **failure mode** when applicable. Update the assigned lane issue/card; the step is experimental, so a documented no-go is still a completed investigation. Never edit the main checkout's notes or push the worker branch.

## Primary references (checked 2026-10-12)

- [VACE release/model table](https://github.com/ali-vilab/VACE), 2025-05-14; [input/mask guide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md), living page accessed above.
- [Depth annotator source](https://github.com/ali-vilab/VACE/blob/main/vace/annotators/depth.py), living source accessed above: per-frame normalization of predicted depth, replicated RGB.
- [diffusers VACE pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py), living source accessed above: `prepare_conditioning_inputs` defaults mask to ones, and inference enforces the VAE temporal-size rule. Pin the execution revision before adapting this recipe.
