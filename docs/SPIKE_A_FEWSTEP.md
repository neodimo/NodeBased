# Spike A: few-step SDXL preview (10/15)

Experiment from `briefs/proposed/U1-few-step-sdxl-preview.md`: **SDXL-Lightning 4-step LoRA** inside the existing SDXL + ControlNet-Union provider
(`nodebased/providers/sdxl_controlnet.py`, version 1.2, still the default), every denoising step offered to a viewer. Assigned to one worker with both
the provider and the viewer scope, **2026-10-15**. Only the **12 GB RTX 3080 Ti** variant ran; the **48 GB RTX A6000 variant is untested** and its run
recipe is at the end. Raw run summaries: `benchmarks/spike_a_fewstep/`. Pictures: `docs/images/spike_a/`.

## Verdict

**Partial: no promotion, provider 1.2 stays the default.** The generation speed gates pass; the first-look gate fails by a factor of four on the 12 GB
offloaded tier, and the verifier's light-direction number got worse.

| Gate (12 GB RTX 3080 Ti, offloaded) | Result | Verdict |
| --- | --- | --- |
| Warm event to first **useful** display, p95 under 1.0 s | 4.32 s (step 2 of 4, 10 edits); the first picture of any kind 4.06 s | **FAIL** |
| No action silently blocks over 60 s; UI responsiveness; cancellation; revision safety | longest wait 14.5 s (burst); viewer event loop max gap 8 ms (replay, see Limits); 10-edit burst: 1 cancelled, 8 coalesced, 1 completed, 0 stale pictures painted | pass |
| Warm final per frame under 14.0 s independent | 7.36 s (rerun same-card baseline 14.05 s, 47.6% less) | **pass** |
| Warm final per frame under 24.3 s canvas-guided | 14.28 s and 14.37 s (rerun baseline 24.31 s, about 41% less) | **pass** |
| Flicker mean at most 2.644, p95 at most 7.6702, 23 pairs, compared fraction 0.9895 | 2.195 / 6.460, 23 pairs, 0.9895 | **pass** |
| Two fresh runs byte-identical; previews on or off no byte change | pixel digests equal, 24 of 24 EXR files equal apart from the header write time; previews on equals off, 24 of 24 | **pass** |
| Side-by-side display review of previews and finals | my own review of the pictures only: layout, objects and depth hold from step 1, **no recognisable rainy street at any step**, finals paler and flatter than the baseline; no human display review | **unverified, not passed** |
| Verifier unchanged, no new failed sub-check, no worse available number | same sub-checks fail (lighting, objects); light-direction error **52.4 to 78.9 degrees (worse)**, colour balance 0.0714 to 0.0336 and object error 18.99 to 11.33 px (better) | **FAIL** |
| Memory guardrail at most 11 GiB whole device with the viewer | 10,738 MiB peak whole device (11,264 MiB limit) | pass (see Limits) |

**Bottleneck: weight transfer under model CPU offload, then the preview decode.** A warm single-frame edit spends about 3.7 s before the first step
finishes (text encoders, ControlNet and UNet copied to the card for every call) and 3 s after step 4 (the final VAE decode evicts the UNet again). The
steps themselves are fast: 0.14 s of sampling per step, and the preview decode (170 ms) is more than half of each 0.31 s step. The 48 GB variant, where
nothing is offloaded, is the test of whether a first look under one second is reachable at all.

## What was built (all behind opt-in settings)

- **Provider setting `fast_preview`** (`sampler_settings`, default off, not recorded when false, so every existing artifact id is unchanged). On: the
  Lightning 4-step LoRA is loaded before the offload hooks are armed, fused into the UNet and unloaded; the scheduler is Euler with `trailing` timestep
  spacing (epsilon prediction, the base scheduler's); `steps` is 4 and `guidance` 0 and nothing else is accepted for them; `denoise_strength` keeps its
  meaning (see Schedule). Provider description, version 1.2 and the offload invariant (hooks re-armed at every pipeline switch) are untouched.
- **Per-step previews**: `callback_on_step_end` on all three pipeline calls. The scheduler's clean-image estimate (`pred_original_sample`, captured by a
  wrapper on `scheduler.step` that changes nothing it computes) is decoded by a **separate resident copy of the VAE**, made before the offload hooks wrap
  the first one. Decoding through the pipeline's own VAE would evict the UNet at every step. No random numbers are drawn and the final latents are not touched.
  Enabled by `NODEBASED_SDXL_PREVIEW_DIR` (batch runs) or a `preview` callable on `run_frames`, or by the session below. Without either, the code path is
  the old one.
- **`nodebased/steppreview.py`** (Qt-free): `PreviewSession` keeps one model process warm (`sdxl_controlnet.py serve`) and speaks a line protocol: one
  active sampling task, one latest pending request (a newer edit stops the running one after its current step and replaces a waiting one, counted as
  coalesced), pictures tagged with edit id, revision, frame, stage and step, a `LatestSlot` one-place newest-wins channel, stale revisions dropped at
  arrival. All times are `time.monotonic()`, shared with the model process.
- **`nodebased/steppreviewview.py`**: `StepPreviewView`, the viewer widget. The reader thread only emits a queued signal; the QImage conversion and the
  paint happen on the UI thread. It never paints a picture older than the newest submitted revision, keeps the last valid picture up labelled
  "Superseded - waiting for the new preview" while the next one is prepared, labels every picture with step, size and "provisional" or "Final", and logs
  model-done, receive, drain, paint and flush times per picture. **It is not wired into the main window**: it is a self-contained widget plus bench.
- **Harnesses** (`tools/`): `spike_a_run.py` (one stage-8 style batch run with verification), `spike_a_interactive.py` (warm edits, burst, memory,
  UI lag, evidence frames, `--replay`), `spike_a_frames.py` (representative frames as bytes, previews off and on), `spike_a_phases.py` (per-call time
  inside one edit), `spike_a_sheets.py` (side-by-side sheets), `spike_a_replay_server.py`.
- **Field bundle**: `tools/make_field_bundle.py --everything` (and the new `--lightning`) copies the LoRA snapshot (three files: the LoRA, `LICENSE.md`,
  `README.md`) from the Hugging Face cache; `peft` joins the provider wheels. Tested with a fake cache (`tests.test_field_report`).
- **Tests**: `tests.test_sdxl_controlnet` (settings, previews through a stub model process, and a CUDA test that previews on or off give identical
  frames), `tests.test_steppreview` (channel, supersession and coalescing against a fake server, viewer revision safety), `tests.test_field_report`.

## Source pins and licences (checked 2026-10-15)

| Component | Repository, revision | Licence |
| --- | --- | --- |
| Adapter (new) | `ByteDance/SDXL-Lightning` @ `c9a24f48e1c025556787b0c58dd67a091ece2e44`, file `sdxl_lightning_4step_lora.safetensors`, 393,854,592 bytes, **sha256 `bf56cf2657efb15e465d81402ed481d1e11c4677e4bcce1bc11fe71ad8506b79`** (equals the Hub's LFS hash; recomputed locally after download) | CreativeML Open RAIL++-M (Bytedance 2024, Stability AI 2023), `LICENSE.md` sha256 `58a5efadd5103ca0906bfc86817ad2d02ebd0856a79e5034756cadd02fa97066`, 14,143 bytes |
| Base | `stabilityai/stable-diffusion-xl-base-1.0` @ `462165984030` (head unchanged) | OpenRAIL++ |
| VAE | `madebyollin/sdxl-vae-fp16-fix` @ `207b116dae70` | MIT |
| ControlNet | `xinsir/controlnet-union-sdxl-1.0` @ `801a4a3fa3d4` | Apache-2.0 |

Licence terms that matter here, from the adapter's `LICENSE.md` (full text is in the Hugging Face cache snapshot and on the stick bundle, which must carry
it): royalty-free copyright and patent grants for the Model and Derivatives; redistribution allowed provided the **use-based restrictions (Attachment A) are
passed on as an enforceable provision** and recipients get a copy of the licence and notice of changes; Attachment A forbids use in violation of law, to
exploit or harm minors, to generate verifiably false information to harm others, to generate personal identifiable information that can harm an individual,
to defame or harass, for fully automated decisions that affect legal rights, to discriminate on behaviour, personal characteristics or protected
categories, to exploit vulnerable groups, to give medical advice or interpret medical results, and for justice, law-enforcement, immigration or asylum
prediction. No change to a licence position of the three existing models. Environment: torch 2.14.1+cu126, diffusers 0.41.0, transformers 5.19.0,
**peft 0.21.2 installed into the provider environment for this spike** (the LoRA loader needs it; `projects/nb-providers/sdxl/setup.sh` and the stick's
provider wheels list it), driver 615.71.09, RTX 3080 Ti 12 GB. The card's own recommendation is the full UNet checkpoint for plain SDXL; the LoRA was used
because the brief names it (it also keeps the base weights shared with the existing pipelines).

## Schedule inspection (CPU, `diffusers` scheduler only)

| Path | Timesteps | Note |
| --- | --- | --- |
| Base scheduler | Euler, epsilon, `leading` spacing, scaled-linear betas | 30 steps start at 958 |
| Lightning, independent text-to-image | Euler `trailing`, 4 steps: **999, 749, 499, 249** (sigmas 14.615, 4.082, 1.613, 0.693, 0) | all four trained steps |
| Lightning, image-to-image at strength 0.5 | `int(4 * 0.5)` = 2 steps: **499, 249**, start sigma 1.613 | the suffix of the trained trajectory |
| Baseline 1.2 image-to-image at strength 0.5 | 15 of 30 steps, first timestep 499 | the same starting noise level |

The truncated Lightning schedule is a suffix of the trained one and begins at the baseline's own noise level (timestep 499), which is why strength 0.5 was
kept unchanged rather than re-derived. Strengths below 0.25 would run no step and are refused with `fast_preview`.

## Inputs recovered and method

Stage 8's inputs were recoverable from the repository: `examples/demo_shot/make_plate.py` regenerates the 24-frame plate (hashes match the committed ones),
`run_by_hand.py` exported the scene state (`665ebf9f…`) and control bundle (`ee3299fc…`) in 2.5 minutes on the CPU. The prompt, seed 7 and the
sampler settings are the committed ones. The **baseline was rerun at the same HEAD today**: its verification report id is `ca01c65e…`, the same id the
10/11 run recorded, with the same 583.5 s generation, peak memory and flicker, so the baseline reproduces to the digit. Each candidate run used its own
fresh copy of the artifact store and its own output folder. Card runs held `/tmp/nb-gpu.lock` for their whole length.

## Measurements, 12 GB RTX 3080 Ti, model CPU offload, batch one, 1024 x 576 working, 480 x 270 delivery

Batch runs (24 frames, seed 7):

| Run | Generation | Per frame | Load | Peak allocated / reserved | Flicker mean / p95 (23 pairs, 0.9895) | Lighting, objects |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline 1.2, canvas-guided, 30 steps | 583.5 s | **24.31 s** | 5.2 s | 7,899 / 8,346 MiB | 2.644 / 7.6702 | FAIL, FAIL |
| `fast_preview`, canvas-guided, run 1 | 342.6 s | **14.28 s** | 10.3 s | 7,763 / 8,072 MiB | 2.195 / 6.460 | FAIL, FAIL |
| `fast_preview`, canvas-guided, run 2 (fresh store) | 344.9 s | 14.37 s | 10.4 s | 7,763 / 8,072 MiB | 2.195 / 6.460 | FAIL, FAIL |
| `fast_preview`, canvas-guided, previews on | 369.1 s | 15.38 s | 10.5 s | 9,186 / 10,500 MiB | 2.195 / 6.460 | FAIL, FAIL |
| Baseline, independent (strength 1.0) | 337.2 s | **14.05 s** | 5.2 s | 7,760 / 8,228 MiB | 15.43 / 48.27 | FAIL, FAIL |
| `fast_preview`, independent | 176.6 s | **7.36 s** | 10.3 s | 7,626 / 7,948 MiB | 12.01 / 28.27 | FAIL, objects not conditioned |

Previews cost 7.2% of generation time in batch (142 pictures, 25 s) and 1.4 GiB of reserved memory. The independent runs' flicker is the model's
frame-to-frame variation (no canvas); it fails the product thresholds (mean 4, p95 12) in both the baseline and the candidate, and is not the
flicker gate's subject.

Verifier means over the frames (the verifier itself is unchanged), canvas-guided: light-direction error 52.43 to 78.92 degrees (**worse**), colour-balance
error 0.0714 to 0.0336, object error 18.99 to 11.33 px. Independent: light direction 79.11 to 79.84 degrees, colour balance 0.0557 to 0.0269; the
object check, FAIL at 14.73 px in the baseline, reads "not conditioned" for the candidate (no data; not investigated).

Determinism: runs 1 and 2 have the same pixel digest (`b019a5b5…`) and 24 of 24 EXR files that differ only in the header's capture date (4 bytes);
previews on gives that same digest. The CUDA test `test_fast_preview_previews_on_or_off_give_byte_identical_frames` pins the same property in the suite.

**Interaction, warm single-frame edit through `PreviewSession` and `StepPreviewView`** (same process stays loaded; 2 warm-up edits, then 10 scripted edits
on frames 1, 12 and 24 with ten different briefs; the model process was loaded cold in 11.9 s, reported apart):

| Event to display, seconds | Median | p95 | Max |
| --- | --- | --- | --- |
| First picture (step 1 of 4) | 4.00 | 4.06 | 4.10 |
| First useful picture (step 2 of 4) | 4.31 | 4.32 | 4.32 |
| Step 4 of 4 | 4.93 | 4.94 | 4.95 |
| Final delivered picture | 7.97 | 7.99 | 7.99 |
| EXR written (colour conversion included) | 7.98 | 8.02 | 8.06 |

"Useful" was fixed before the runs as step 2: its picture is 28 dB PSNR from step 4 on all three representative frames (step 1: 24.5 dB, luma
correlation 0.95 to 0.98 with step 4); counting step 1 does not change the verdict. Each step costs 0.31 s, of which the preview decode is 0.17 s;
model-done to receive is under 0.1 ms and receive to paint 2.2 to 3.7 ms per picture. Pictures are 1024 x 576, shown scaled; the delivered final is 480 x 270.
**Burst**: 10 edits 150 ms apart; the running edit was cancelled after its current step, 8 waiting edits were replaced before they ran, the newest
completed and its final was on screen 14.45 s after the first edit; no picture of an older revision was painted or arrived after a newer one; slot and
pending queue peak depth 1. **Memory with the viewer up**: whole device 10,738 MiB peak (11,264 MiB limit; 19 MiB idle before), the model process
10,714 MiB of it (CUDA allocated 9,050, reserved 10,376 MiB), model process host RSS peak 12.1 GiB, viewer process 1.0 GiB.

## Display review

Sheets `docs/images/spike_a/sheet_frame_01.png`, `_12.png`, `_24.png` hold, per representative frame: the committed stage 8 and stage 3 baseline
stills, the candidate's canvas-guided final, the four denoising steps of an independent edit and its delivered picture.

- Step 1 is grainy but already carries the layout: both boxes, the sphere and the floor plane sit at the depth-aligned positions; step 2 is smooth
  and clean; steps 3 and 4 add backdrop texture. Objects and depth stay aligned through all four steps and into the guided final; I saw no new severe ghosting
  in frames 1, 12 and 24 of the guided final.
- **The requested rainy street does not appear** at any step: the backdrop is a grey wall with abstract texture, the finals read paler, flatter and less
  contrasty than the 30-step baseline, whose backdrop is stripes or bricks and also not a rainy street. Guidance 0 gives the text brief little hold.
- `docs/images/spike_a/viewer_edit_1_grabs.png` is the viewer widget (offscreen grab) at each paint of one edit: step 1, 2, 3, 4 provisional, then Final. Real-screen evidence: see Real screen below. No person has reviewed these pictures; the gate stays unverified.

## Real screen

**Not run.** A pass of the same bench on the workstation's real display (Xwayland, screen captures of the window during an edit) was queued behind the
GPU lock and withdrawn when the lock stayed held by a 5-hour integration suite past the 30-minute bound (queued 10:10 PM, withdrawn 10:35 PM on
2026-10-15). All display evidence here is offscreen widget grabs; no physical display or human reviewer was involved, so display acceptance is unverified.
`tools/spike_a_interactive.py` runs on a real screen with `QT_QPA_PLATFORM=xcb DISPLAY=:0` (the window opens without taking focus).

## Limits and what was not done

- **A6000 variant untested.** No substitution of any other card.
- **Card parts left out because the GPU lock was held past the 30-minute bound** (other workers' suites from 9:00 PM on): the per-call phase probe
  (`tools/spike_a_phases.py`, offloaded and a labelled resident diagnostic), the real-screen pass, a repeat of the live interaction run with clean lag
  accounting, a comparison of the warm preview process's delivered frames against a cold batch process (`tools/spike_a_frames.py` with the whole-sequence
  depth normalisation), and the CUDA unit test run (`RealModelTests`). The first-look latency is therefore **not attributed call by call**: the 3.7 s before
  step 1 and the 3 s after step 4 are inferred from the 0.31 s step cadence, the batch per-frame times (two offloaded passes of about 6.5 s each per
  guided frame) and the weight sizes (text encoders 1.7 GB, ControlNet 2.5 GB, UNet 5.1 GB copied per call). `tools/spike_a_phases.py` and the revised `tools/spike_a_frames.py` were written and
  syntax-checked but **never executed**. A resident variant of the 12 GB card (UNet and
  ControlNet kept on the card) was never measured and is the cheapest next test of that inference.
- Previews on or off for the representative frames 1, 12 and 24 was checked with `spike_a_frames.py` as first written, which normalised the depth control
  images over those three frames only: all three delivered pictures are byte-identical with previews on and off (`frames-previews-on-off.json`). The
  full 24-frame canvas-guided runs with previews on and off are byte-identical as well (above).
- **UI responsiveness** was measured by replaying a real run's pictures and timing through the real `PreviewSession` and `StepPreviewView` (max event-loop gap
  8 ms over 21,579 ticks, no gap over 100 ms, 2.6 to 3.1 ms per picture), because the card was held by other workers' suites when the live repeat was due.
  The first live interaction run's lag figure counted the bench's own blocking calls and is not used.
- The viewer widget is **not wired into the main window** and was not measured over a GPU-backed viewport; the memory figure had a CPU-painted widget
  and no other process on the card.
- The canvas-guided path's four fill steps (the text-to-image pass that fills the warp's holes) also emit previews, labelled `fill`; a guided frame's two
  image-to-image steps are labelled `guided`. The interaction bench measured independent single-frame edits only; no interactive canvas-guided edit loop
  exists in the app, and none was built.
- Strength 0.5 on 4 steps was kept as is; no other strength, step count or adapter (2-step, 8-step, full UNet) was tried. No tiny decoder, no resolution
  change, no attention kernel change.
- Lighting and objects still fail, as before; the guided candidate's light direction is worse.
- Not run: the lens-distorted shot, `--solver` variants, the full test suite.

## Run recipe for the 48 GB RTX A6000 (Claude Code on the work machine, from the stick bundle v2)

Follow `docs/FIELD_CLAUDE.md`'s rules (measure and report, no commits, nothing leaves the machine but the zip DiMo moves; if the card is shared, stop and
say so). Everything below is PowerShell under `%USERPROFILE%\nb\nodebased` with the venv active; replace `E:\nb-bundle` with the stick. The bundle must have
been made with `--everything` (or `--lightning`) after this spike, so that it holds the LoRA snapshot and the `peft` wheel.

1. **Install and point the cache at the stick** as `BUNDLE-README.txt` says (step 4 sets `HF_HUB_CACHE` and `HF_HUB_OFFLINE=1`). Check the adapter:
   `Get-FileHash -Algorithm SHA256 (Get-ChildItem $env:HF_HUB_CACHE\models--ByteDance--SDXL-Lightning\snapshots\*\sdxl_lightning_4step_lora.safetensors)`
   must read `BF56CF2657EFB15E465D81402ED481D1E11C4677E4BCE1BC11FE71AD8506B79`. `pip list` must show `peft`.
2. **Inputs** (CPU, a few minutes): `$env:NODEBASED_CACHE="$env:USERPROFILE\nb\spike-a\cache"`; `python examples\demo_shot\make_plate.py`;
   `python examples\demo_shot\run_by_hand.py --out "$env:USERPROFILE\nb\spike-a\out-reproject" --cache "$env:USERPROFILE\nb\spike-a\store"`. Note the printed
   `scene_state` and `control_bundle` ids (the 3080 Ti run printed `665ebf9f…` and `ee3299fc…`; equal ids mean equal inputs). Make one copy of the `store` folder per
   run below (`store-a6000-base`, `-base-i`, `-c1`, `-c2`, `-cp`, `-ci`); a run must never see another run's results.
3. **Fully resident, no hidden offload** for every run below: `$env:NODEBASED_SDXL_OFFLOAD="never"`. Each run's summary must show `model_run.cpu_offload` false;
   if it shows true, stop: the comparison is void. Nothing else may use the card; check `nvidia-smi` first (record the memory already used).
4. **Baseline, provider 1.2 unchanged** (30 steps), canvas-guided then independent:
   `python tools\spike_a_run.py --store <store> --scene-state <id> --bundle <id> --out <runs>\a6000-base\out --label a6000-base`
   and the same with `--settings '{"denoise_strength": 1.0}'` and label `a6000-base-i`. On Windows there is no GPU lock file to take; the provider's own
   lock call is a no-op.
5. **Candidate**: `--settings '{"fast_preview": true}'`, labels `a6000-c1` and `a6000-c2` (two fresh stores, the byte-identity comparison), `a6000-cp` with
   `--preview-dir <runs>\a6000-cp\previews` (previews on; its `output_pixel_digest` must equal `a6000-c1`'s), and `a6000-ci` with
   `'{"fast_preview": true, "denoise_strength": 1.0}'` (independent).
6. **Representative frames as bytes**: `python tools\spike_a_frames.py --store <store> --scene-state <id> --bundle <id> --out <runs>\frames` prints `"identical": true`
   when previews on and off give the same bytes for frames 1, 12 and 24.
7. **Interaction**, viewer present: `python tools\spike_a_interactive.py --store <store> --scene-state <id> --bundle <id> --out <runs>\interactive --grab`
   (offscreen Qt by default; leave `QT_QPA_PLATFORM` unset to show the window on the screen, which then also needs a screen capture by hand for the
   recording). It needs `nvidia-smi` on the PATH for the memory samples.
8. **Gates for this card** (the brief's numbers, none of them predictions): first useful display p95 under 1.0 s; guided and independent final per frame faster
   than `a6000-base` and `a6000-base-i` measured today (the 14.0 and 24.3 s figures belong to the 3080 Ti); flicker mean at most 2.644 and p95 at most 7.6702
   over 23 pairs with compared fraction 0.9895; `a6000-c1` and `a6000-c2` same digest; whole-device memory peak at most 44 GiB (45,056 MiB) with the viewer up;
   verifier sub-checks no worse than `a6000-base` (compare light direction, colour balance, object error as in the table above). Report the offloaded
   3080 Ti numbers from this page only as a separately labelled comparison.
9. Copy `<runs>\*\*.summary.json`, `interactive\interactive.json`, `frames\frames.json`, the `evidence` and `grabs` folders into the field report folder.
   Never invent a number; a step that did not run is reported as not run.

## Files

`nodebased/providers/sdxl_controlnet.py`, `nodebased/steppreview.py`, `nodebased/steppreviewview.py`, `tools/spike_a_*.py`, `tools/make_field_bundle.py`,
`tests/test_steppreview.py`, `tests/test_sdxl_controlnet.py`, `tests/test_field_report.py`, `benchmarks/spike_a_fewstep/`, `docs/images/spike_a/`. Large
run folders (about 3 GB: stores, outputs, previews) stay in scratch and are owned by Gonzo for cleanup.
