# Proposed spike A: few-step SDXL with each step shown in the viewer

**Proposed 2026-10-15 · not executed, scheduled or adopted.** Gonzo assigns the provider and viewer owners together, or a worker explicitly authorized for both scopes. Research permission does not authorize weights, GPU execution or application edits. One bounded experiment: **SDXL-Lightning 4-step LoRA** on the existing SDXL + ControlNet-Union provider. No model sweep, training, production default change or new node. The proposed numerical gates below are requirements, not predicted results. [Evidence and ranking](../../docs/RESEARCH_RADAR.md#where-to-look-next).

## Question and reference measurements

Can a useful, depth-aligned preview of the requested appearance arrive within **1 second of a warm edit**, then refine to a valid final image without worsening temporal behavior or the verifier? All four denoising updates must be offered to the viewer, with superseded updates safely coalesced and their actual presentation times logged.

- On **2026-10-11**, RTX 3080 Ti stage 3 independent generation: **336.5 s / 24 frames = 14.0 s/frame** rounded; stage 8 canvas-guided: **584.15 s / 24 = 24.3 s/frame** rounded, working **1024 × 576**, delivery **480 × 270**. Generation excludes loading and is not event-to-display latency. Exact source: `benchmarks/demo_shot/sdxl-stage3-baseline.json`, `benchmarks/demo_shot/sdxl-stage8-v1.2-canvas.json`, `docs/END_TO_END_SPRINT.md` stage 8.
- Stage 8 uses **30 steps, guidance 5.0, control scale 0.8, strength 0.5, seed 7**, with CPU offload; peak reserved **8,346 MiB**. Flicker **2.644 mean / 7.6702 p95**, **23 pairs**, compared fraction **0.9895**; lighting/objects still fail. Record their actual baseline observations, never call temporal acceptance binding compliance.
- Flow reference is separately **1.34 s for both directions at 1080p**, four pyramid levels/eight iterations, RTX 3080 Ti, **2026-09-27**, `docs/PARITY_2D.md` Lane 2 V1. It motivates the subsequent NeuFlow v2 spike. **Do not add this number to SDXL generation or claim Spike A accelerates flow.** If reprojection/flow preparation lies on the measured preview path, instrument its actual cost. A separate Spike B must rerun this same pair contract before comparing NeuFlow.
- Reproduce the current provider on the same fixed inputs/configuration before comparing. Historical figures are context, not today's HEAD or A6000 measurements. Record revision and environment; never convert one card's timing into the other's.

## Setup and safe scope

Resolve the exact adapter revision and SHA-256 plus base/Union/VAE revisions and their licence text before any authorized download. Candidate: `ByteDance/SDXL-Lightning`, `sdxl_lightning_4step_lora.safetensors`; [card and schedule](https://huggingface.co/ByteDance/SDXL-Lightning), [paper](https://arxiv.org/abs/2402.13929), **2024-02**, checked **2026-10-15**. Use the card's Euler trailing schedule, correct prediction type and guidance **0**. This is the matched distilled method, not simply reducing the current sampler's step count.

Use the assigned worktree and isolated experiment environment; leave production provider 1.2 intact as the fallback. Pin CUDA/torch/diffusers, dtype, attention implementation and deterministic settings; no automatic precision, seed, memory or sampler fallback. No new attention kernel/compilation/TeaCache for this first test. Stop with a documented no-go if the installed Union pipeline cannot support the required callback and schedule safely.

The current provider creates T2I and img2img pipelines sharing components, and explicitly re-arms offload hooks at each switch to avoid double residency. Preserve that invariant. **Strength 0.5 truncates an img2img schedule**: do not assume a four-step distilled LoRA remains valid on an arbitrary truncated two-step trajectory. Inspect the exact scheduler calls and log timesteps for both paths. First demonstrate independent T2I with all four trained steps; for canvas-guided comparison require a justified compatible img2img schedule or report that leg as failed/unverified. Never quietly change the strength/conditioning contract just to get a fast result.

Recover stage-8's renderer conditions, prompt, input images, original frame times, masks and verifier artifacts; the committed JSON is a pointer, not all source pixels. If the inputs cannot be recovered, end blocked on the shot comparison; a synthetic callback test alone cannot pass this spike. Use existing authorized inputs only.

## The two hardware variants

These are separate runs, each capped at **90 minutes active work** including setup and reporting; no second card is assumed to exist or be reachable. If Gonzo assigns only one, complete it and explicitly leave the other untested. No silent substitution of Radeon, RTX 6000 Ada or RTX PRO 6000.

| Variant | Fixed residency policy | Proposed resource guardrail and comparison |
| --- | --- | --- |
| **12 GB RTX 3080 Ti** | Existing model CPU offload, re-armed on pipeline switches; batch one; shared components. Keep this policy for both baseline and candidate. | **≤11 GiB whole-device use**, report allocated/reserved/driver peaks and host RSS separately. Measure with the viewer present. If preview decoding causes OOM, record no-go; do not retry with a changed model/resolution. |
| **48 GB RTX A6000** | Baseline and candidate both fully GPU-resident; batch one first, no CPU/disk model offload. Use existing `NODEBASED_SDXL_OFFLOAD=never` route only after ownership review. | **≤44 GiB whole-device use**, with viewer. Verify no hidden offload and log resident components. Same working/delivery size, inputs and numerical mode. This isolates model-step benefit from residency benefit; compare resident vs offloaded totals only as a separately labelled comparison. |

Guardrails leave deliberate headroom and are experiment choices, not model requirements. Use the host's exclusive GPU-job coordination; do not overlap integration suites. Bound lock wait to **5 minutes**, then leave no waiter/process behind and report blocked. Large outputs go to deliberate disk-backed scratch, never RAM-backed temporary storage. Release model processes and GPU lock on every exit path.

## Preview channel and interaction semantics

1. Inspect `nodebased/providers/sdxl_controlnet.py`, the job/progress transport and viewer ownership before edits. Existing progress is emitted after a whole frame; no per-step presentation currently exists. Scope an experimental callback/transport and viewer path only if assigned; provider-only callbacks are insufficient for the user-visible gate.
2. For each denoising step, obtain a scheduler-correct predicted clean sample (or documented preview estimate); VAE-decode without consuming the generation RNG or changing final latents. Do not label a raw noisy latent as the clean-image estimate. Prefer the existing decoder first; a tiny decoder is a later separately licensed variant, not an unlogged optimization.
3. Send preview pixels with job/revision/frame/step identifiers and provisional status through a bounded newest-result channel. UI paint must remain on the UI thread. Record callback completion and actual display timestamps independently. A count of callbacks alone proves no visible preview.
4. Superseding an edit cancels/coalesces old work; no stale frame or stale final result may replace the new revision. Preserve the last valid display while preparing the next result, with an explicit stale/provisional label. Keep a single active sampling task and one latest pending request.
5. Preview decoding must not pollute offload-hook state, final caches, the unrefined guide canvas, verification or EXR byte identity. Repeated independent renders with previews enabled must produce the same final bytes as previews disabled for that candidate configuration.
6. A four-step result completes its schedule. An artist pause may trigger a separately identified quality render later; it must not append steps to a completed distilled trajectory or silently splice a different checkpoint into it.

## Run and score

Record cold model load/startup separately. After warm-up, use **10 scripted same-complexity edits**, each waited to completion for latency measurement, plus a separate **10-edit burst** to test cancellation/coalescing. These counts are proposed test design. Log event → queue → prepare/transfer → first useful presented preview → each presented step → final display and EXR completion. Include preview decode/transport/draw overhead, median, p95, maximum, stale-result count and peak queue depth. Do not count the cached old picture or a noise-only preview as first look.

On the fixed stage-8 sequence, run **two fresh generations**, same inputs/configuration/seed, separate output stores and no result-cache reuse. Compare final EXR bytes. With the same candidate settings, additionally compare representative frames **1/12/24** with preview callbacks disabled to catch RNG/offload side effects. Compare baseline and candidate at identical working/delivery sizes. No downscaled preview may masquerade as a full-resolution speed result; label its actual dimensions and score full-resolution final separately.

**All promotion gates:**

- Warm event-to-first-**useful**-display **p95 <1.0 s**, plus no action silently blocks longer than **60 s**; load/first-use latency must be reported separately. UI responsiveness, cancellation and revision safety pass. Final-generation timeout or missing final output is a failed gate even if previews appeared quickly.
- Warm final generation/frame **<14.0 s independent** and **<24.3 s canvas-guided** on the 3080 Ti, and faster than the rerun same-card baseline on either card. Report percentage changes only from matched local measurements. A6000 baseline must actually be run; those 3080 Ti numbers are reference thresholds, not A6000 baseline estimates.
- Final flicker **mean ≤2.644 / p95 ≤7.6702**, all **23 pairs**, compared fraction not reduced below **0.9895** except documented rounding. Product temporal thresholds remain mean ≤4 / p95 ≤12; stricter experiment nonregression is intentional. No unchanged-layout or blurred-output shortcut.
- Same candidate repeated runs give byte-identical final EXRs; preview-on/off has no final-byte effect. If upstream kernels cannot satisfy this, report failed determinism rather than weakening the gate.
- Side-by-side display review of early previews and final frames: recognizable requested rainy-street appearance, retained objects/depth alignment, no new severe ghosting. Show every intermediate stage for the representative frames and the real interaction recording. If a physical display/reviewer is unavailable, mark visual acceptance unverified and do not promote.
- Existing verifier stays unchanged. No new failed sub-check or worse available numerical result; retain the present lighting/object failures honestly. No provider binding-compliance claim until independently established.
- Fixed hardware-specific memory guardrail passes with the viewer active; all model/host/device costs and unsupported controls are disclosed.

**Fallback and decision:** if any gate fails, keep provider 1.2 unchanged. A generation speed win with a missed first-look gate is a partial result, not interactive acceptance. A valid early preview with failing final temporal/verification is also partial. Report the bottleneck: scheduling, Union conditioning, sampling, VAE, transfer, display, licence, input recovery or unsupported img2img schedule. Do not expand into Hyper-SD/LCM/video models in this step.

## Deliver, verify and stop

Land `docs/experiments/U1-few-step-sdxl-preview.md` with byte-identical bundled copy, source pins/licences, stage timings, both-card run/untested statuses, final-byte comparisons, unchanged-verifier results, screenshots/interaction evidence and pass/no-go rationale. Heavy artifacts remain deliberate scratch with retention/cleanup owner and a repo-safe pointer; no home paths in tracked documents. No default/provider promotion within this experiment.

Run `tests.test_knowledge` and targeted existing provider/temporal tests appropriate to the eventual changes (`tests.test_sdxl_controlnet`, `tests.test_temporal`, relevant generative/viewer tests); add meaningful preview-generation/cancel/cross-revision/final-byte tests. Do not run the full suite. A successful harness is not a display proof. Commit reviewable work, never push or touch the main checkout, then stop.

Before reporting done, write your completion note to the assigned NodeBased worktree's `TASKLOG.md` and `context/state.md`, newest first: **What was done** (evidence vs inference), **Artifacts** (exact paths and committed/local/scratch state), **State** (done/partial/blocked and unverified limits), **Next owner + concrete artifact**, and **Failure mode** when applicable. Update the assigned issue/card in the same completion step. Gonzo owns integration and any subsequent NeuFlow v2 assignment; no new worker starts automatically.
