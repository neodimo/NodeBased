## 2026-10-31 — continuous mode merge: Lane 6 (Fluids, Claude Sonnet 5.5) step B of 2: the flame has a hot core and licking tongues (integrator tick)

- **What was done:** `main` `a2db7871` -> `9583035e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `9583035e`: Ran 7589 tests in 540.903 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31, 1:20 AM PDT — Lane 6 fire shape, step B of 2: the flame has a hot core and tongues

- **What was done — evidence:** swept the existing solver controls on the 64-cell shot (21 rows, `benchmarks/fire_shot/fire-shape-sweep-2026-10-31.json`; cooling is the strongest lever, burn expansion and confinement add tongues, turbulence of 3 or more makes dust, confinement of 3 diverges); found the temperature term is already a blackbody and the flat part is the flame glow's single kelvin, so added the off-by-default Render3D `volume_flame_follow_temperature` on both raymarchers, both path tracers and the fire-light grid; kept the shape-turbulence noise draws (bake 40 s to 9 s, bit-identical); retuned the shot. Gate numbers on the RTX 3080 Ti: core in the lowest third 6 to 17 of 17 frames, tongues per slice 0.53 to 1.13, bake 8.4 s, frame 0.495 / 0.440 / 0.417 s. The pre-step fire document renders bit for bit the same. 348 targeted tests pass on the RTX 3080 Ti, AMD Radeon 8060S and llvmpipe.
- **Inference / not run:** the look in the stills is my reading of saved images, not the real display. No Windows run, no full suite, no CI read.
- **Artifacts (committed on `openclaw/nb-fluids-spike`, not pushed):** `nodebased/volumerender.py`, `gpuvolume.py`, `gpupathtrace.py`, `ptvolume.py`, `imaging.py`, `core.py`, `knobs.py`, `fluid3d.py`, `fluid_gpu_solver.py`, `examples/fire_shot/{make_shot,render_evidence,flame_metrics}.py` and `fire_shot.nbcomp`, `tests/test_flame_follow.py`, `tests/test_flame_metrics.py`, `tests/test_fire_shot.py`, `benchmarks/fire_shot/fire-shape-*.json`, `docs/images/fire_shot/fire-shape-*`, `docs/FLUIDS_SPIKE.md` ("Fluids 13 B shape"), `docs/3D_FOUNDATION.md` (bundled copies identical). Scratch only: bake caches under `scratch/nb-lanes/run/big-L6/` (deleted).
- **State:** DONE against the gate (profile and tongue counts move the right way, bake and frame time under the limits, old document unchanged, tests green on three adapters). Did not move the right way, so unfinished against the brief's prose: smoke eddies (vorticity 185 to 68), the image-row measures, the colour ratio by height; grain rose 0.0056 to 0.0141. Schema unchanged.
- **Next owner + concrete artifact:** Gonzo judges `docs/images/fire_shot/fire-shape-before-after-2026-10-31.png` on the real display, then picks the next step (smoke eddies without dust, a grain pass on the brighter flame, or a Windows run of the new glow).
- **Failure mode corrected:** the colour was flat because the glow ignored temperature, so cooling the gas could not change it; and a profile measure over the flame's own span cannot separate a short flame from a cooling one (the third-over-third ratio stays near 1 for a flame that is hot to its last cell), so the sweep reads core position and tongues, not that ratio.

## 2026-10-31 — continuous mode merge: Lane 8 (2D on the GPU, GPT-5.6 Terra) GPU playback follow-up: retain native-half EXR samples through decode (integrator tick)

- **What was done:** `main` `94d85e16` -> `cbe77070` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `cbe77070`: Ran 7559 tests in 528.355 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step N of N: a NoTimeBlur node, and the project settings counted as Nuke's Root (integrator tick)

- **What was done:** `main` `4926b8bf` -> `2a035b71` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `2a035b71`: Ran 7558 tests in 535.811 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31, 12:15 AM PDT — Lane 2 fidelity step N: NoTimeBlur, and the project settings counted as Nuke's Root

- **What was done — evidence:** `NoTimeBlur` added (`rounding` rint/floor/ceil/none, `single` on by default). The evaluator reads its input at the rounded frame; `single` rounds the output frame, found on a per-thread stack of running `evaluate_raster` requests (`imaging._REQUEST_FRAMES`, nearest whole-frame request). Tests in `tests/test_2d_parity_notimeblur.py` (26, all pass): a TimeBlur (shutter 1, 4 divisions) over a keyed Constant at frame 5 averages frames 5,5,5,5 (rint), 4,4,5,5 (floor), 5,5,6,6 (ceil) with `single` off and reads frame 5 alone with it on; without NoTimeBlur it blurs; `none` forwards fractional frames; tiled equals full frame for every mode; bypass passes through. Root: 7 of 43 knobs paired with document fields (`time.first/last/fps`, `settings.color.config`, `settings.color.view`, `settings.viewer.proxy`), each shown acting by a test; 36 listed unsupported with a reason. Audit tool gained one rule (`document_settings`, `document_default`), stated in the report's scoring section. Fidelity 90.4 -> 90.9 (NoTimeBlur 0.0 -> 100.0, Root 0.0 -> 28.0). Targeted runs: new module, test_knobs, test_tiers, test_core, FluidPropertiesPanelTests, PropertiesLabelTests, test_bypass, test_nodecatalog, test_tiles, test_cache_correctness, the time/e2 parity modules, test_knowledge, test_nuke_fidelity_audit, test_2d_parity_step_w3: all OK.
- **Inference / not run:** Nuke's exact semantics of `single` are not in the guide beyond "use single frame"; the anchor rule (the output frame the request belongs to) is my reading and has no Nuke render behind it. No full suite, no CI read, no Windows run, no real Nuke renders, so the pixel-truth component is still not in any number.
- **Artifacts (committed on `openclaw/nb-2d-parity`, not pushed):** `nodebased/imaging.py` (`no_time_blur_frame`, `_REQUEST_FRAMES`), `core.py`, `knobs.py`, `nodecatalog.py`, `theme.py`, `tiers.py`, `tiles.py`, `tileexec.py`, `rotoahead.py`, `tools/audit_nuke_fidelity.py`, `docs/parity/nodebased_to_nuke.json`, `docs/PARITY_2D.md`, `docs/PARITY_2D_FIDELITY.md`, `docs/TIME_MODEL.md` (bundled copies identical), `tests/test_2d_parity_notimeblur.py`, `tests/test_tiles.py`. No schema bump.
- **State:** DONE for both parts. Known limit: a whole-frame TimeOffset between a TimeBlur and a NoTimeBlur hides the output frame from `single` (documented in PARITY_2D row 9 and TIME_MODEL).
- **Next owner + concrete artifact:** Gonzo integrates the branch tip through `tick.py tick`, then decides whether the 2D fidelity aspect is closed from `docs/PARITY_2D_FIDELITY.md`: what stays below 100 is 12 deliberate omissions (2.3 points), secondary knobs and per-format tabs of Roto/Write/Read and others (5.4), on-purpose default differences (1.5), Root's unpaired knobs (0.2), plus pixel truth that needs real Nuke renders (`tools/make_nuke_reference_pack.py`).
- **Failure mode:** my first anchor was the plain output frame, which lost a TimeOffset placed between the blur and NoTimeBlur; the nearest-whole-request rule fixes offsets above the blur only, and that is what the docs now say.

## 2026-10-30 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step M of N: Write delivers TIFF, JPEG and DPX, and takes Nuke's EXR options, Lane 6 (Fluids, Claude Sonnet 5.5) step A of 2: the fire and its smoke stop sparkling (integrator tick)

- **What was done:** `main` `06ba1e51` -> `c819ca42` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `c819ca42`: Ran 7532 tests in 526.911 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-30, 8:40 PM PDT — Lane 6 fire grain, step A of 2: the fire and its smoke stop sparkling

- **What was done — evidence:** attribution first, on the shot at frame 16 (128 samples, two seeds): `emission` grain 0.0110 (about its own mean 0.0133), `diffuse` 0.0055, `diffuse_indirect` 0.0013, specular 0; everything 0.0142. The emission estimator summed the fire's glow at delta tracking's tentative collisions (one global majorant because the volume has a flame channel). Replaced, for a single volume, by `vol_emission` (WGSL, `gpupathtrace.py`) and `ptvolume.expected_emission` (CPU twin, same hashed numbers, wired through `free_flight(..., deep=)` and one call in `pathtrace.py`): stratified jittered samples every 4 voxels (16 from a smoke vertex) of `emission * exp(-tau)`, step limited to optical depth 0.5, stop at optical depth 6, attenuation to a sample from a 0.75 own / 0.25 previous extinction blend. Measured on the RTX 3080 Ti: frame-16 grain 0.0142 -> 0.0056; emission grain 0.0110 -> 0.0021; render time 1.10 / 1.18 / 1.22 x for frames 8 / 16 / 24 (limit 1.25 x); mean image at 2048 samples within 0.013% of the old renderer (flame core within 0.15%). 277 targeted tests (volume path tracing, sparse volumes, flame mesh, data passes, fire look, fire shot and the new `tests/test_fire_grain.py`) pass on the RTX 3080 Ti, the AMD Radeon 8060S and llvmpipe.
- **Inference / not run:** "smoother" in the stills comes from the saved images, not the real display. No Windows run, no full suite, no CI read. The 0.25 blend weight was set on this shot (checked on frames 8, 16, 24 and the existing tests), not derived. The thick test blob in `test_fire_grain` agrees with the old estimator to about 1%, within the test's 3%. Several volumes keep the old collision sum.
- **Artifacts (committed on `openclaw/nb-fluids-spike`, not pushed):** `nodebased/gpupathtrace.py`, `nodebased/ptvolume.py`, `nodebased/pathtrace.py`, `tests/test_fire_grain.py`, `examples/fire_shot/{render_evidence.py,README.md}` (`--grain` mode), `benchmarks/fire_shot/fire-grain-2026-10-30.json`, `docs/images/fire_shot/fire-grain-before-after-{,zoom-}2026-10-30.png`, `docs/FLUIDS_SPIKE.md` ("Fluids 13 B grain") with the identical bundled copy.
- **State:** DONE for step A of 2 against the brief's gate. The time margin at frame 24 is thin (1.22 x of 1.25 x) and was measured on a card another process may share.
- **Next owner + concrete artifact:** Gonzo integrates the branch tip through `tick.py tick`, then judges `docs/images/fire_shot/fire-grain-before-after-2026-10-30.png` on the real display. The dominant term left is the smoke's own scattering (`diffuse` 0.0045 of 0.0056): a next-event estimate of the fire lighting the smoke from `volumerender.fire_light_grid` and an expected-value single scatter along the flight, which would change the smoke's lighting estimator (a separate step).
- **Failure mode corrected:** the "glitter" was the camera-ray fire term, not the smoke or the floor: a flame a few voxels wide met each ray in zero or one random collisions. More samples only halved it (1/sqrt(N)); integrating along the flight removed most of it. Watch for: the card keeps a scene's packed form between renders, so the old-estimator switch lives in the per-render uniform, not in the pack; and an identical render request is answered from the image cache in a millisecond, so timings need a new seed each repeat.

## 2026-10-30, 8:00 PM PDT — Lane 2 fidelity step M: Write delivers TIFF, JPEG and DPX and takes Nuke's EXR options

- **What was done — evidence:** `file_type` gains `tiff`, `jpeg` and `dpx` (Auto reads `.tif/.tiff/.jpg/.jpeg/.dpx`); `nodebased/writeformats.py` writes them through OpenImageIO with Nuke's knobs (TIFF data type and compression, JPEG quality and sub-sampling, DPX data type, fill, big endian, transfer) after the existing output transform and premultiplied handling, clamping before quantising. `media.write_exr` takes an `options` dict that `writenode.exr_options` builds from `exr_autocrop`, `exr_metadata`, `exr_interleave`, `exr_first_part`, `exr_write_full_layer_names`, `exr_truncate_channel_names` and `exr_dw_compression_level` (`dwaa`/`dwab` join the compression choices). Properties lists only the active format's knobs (`writenode.hidden_params`). `tests/test_write_formats.py` (47 tests) writes real files and reads them back with OpenImageIO: size, channels, bit depth, pixels to one code value, TIFF compression and DPX endianness/packing/transfer in the header, JPEG sampling factors in the frame header, EXR data window, attributes, parts and DWA level. An old Write (params popped) loads with the defaults and writes the same EXR (pixels, channel names and attributes, DateTime aside) and the same PNG bytes. Targeted runs across Write, render, knobs, tiers, core, panels, sliders, audit, knowledge, animation and bypass (367 tests, then the two modules that failed on the first pass plus knowledge and panels, 140) pass after the fixes below.
- **Fidelity:** the Write row stays 67.2 and the overall score 90.4. The reference data (`tools/build_nuke_reference.py`) marks Nuke's per-format option tabs as not counted, so pairing the fifteen knobs that reach the file in `docs/parity/nodebased_to_nuke.json` changes no number; `docs/PARITY_2D_FIDELITY.md` regenerated with no diff. `exr_standard_layer_name_format` is stored and shown but changes no byte with one view, so it is `extra` and Nuke's knob is `unsupported`. MOV, MXF and Python-callback rows untouched.
- **Inference / not run:** DPX `fill` follows the guide's wording (on = packed), which reads opposite to the knob name; not checked against a real Nuke file. Float TIFF Auto writes the working space (ACEScg) and TIFF stores no colour-space tag, so Read's Auto reads it back as sRGB. No Windows run, no full suite, no real-display look at the new Properties rows beyond the offscreen panel tests.
- **Found on the way:** OpenImageIO's reader mis-reads a 10-bit RGBA DPX with filled packing at some widths (40, 64); the file is correct (decoded from its bytes in the test). `tests/test_knob_sliders.py` skips knobs a Write hides for its format; the JPEG and DWA sliders are asserted in `test_write_formats.py`.
- **Artifacts (committed on `openclaw/nb-2d-parity`, not pushed):** `nodebased/writeformats.py`, `nodebased/writenode.py`, `nodebased/media.py`, `nodebased/core.py`, `nodebased/knobs.py`, `nodebased/app.py`, `tests/test_write_formats.py`, `docs/PARITY_2D.md` (Write row and fidelity step M note) with its bundled copy, `docs/parity/nodebased_to_nuke.json`.
- **State:** DONE for step M. Saved-document schema unchanged (28): new params default in `SPECS`, and `upgrade_document` fills them for old documents.
- **Next owner + concrete artifact:** Gonzo integrates the step M commits through `tick.py tick` and checks the exact-tip suite log; a later step could add Read for DPX and `Cineon`/`Targa`/`SGI` writers.
- **Failure mode guarded:** an old Write silently changing its output. The EXR and PNG paths only leave the unchanged call when every new knob is at its default (`exr_options` returns None), and `test_write_formats.OldDocumentTests` compares the files.

## 2026-10-30 — continuous mode merge: Lane 6 (Fluids, Claude Sonnet 5.5) step B of 2: a smoother, better sampled fire shot with honest before-and-after evidence (integrator tick)

- **What was done:** `main` `af7376b1` -> `d519c0b3` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `d519c0b3`: Ran 7472 tests in 518.681 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-30, 9:10 PM PDT — Lane 6 Fluids 13 B look: a finer, less grainy fire shot

- **What was done — evidence:** `make_shot.py` now takes `division_size`, `pt_samples`, `curl` and `source_noise`; the shot defaults to 64 cells, 128 samples and a seeded sideways wobble (`curl_keys`, seed 1029, 0.10) on the fuel source's velocity. `render_evidence.py` gained `--compare` (32 and 64 cells at 32 and 128 samples, frames 8, 16, 24, fixed seeds, contact sheet, image differences, flame occupancy per cached frame, grain from two seeds), `--curl-sweep` and richer default output. Measured on the RTX 3080 Ti: grain at frame 16 is 0.034 at 32 samples and 0.017 at 128 samples for both grids; 32 versus 64 cells differs 2 to 5 times more in the image than 32 versus 128 samples; the 32-cell cache has four empty height slices at frame 8 and the 64-cell cache has none; the 64-cell bake takes 7.7 s (0.74 s at 32 cells); a 128-sample frame takes 0.28 s warm. Curl 0.10 raises flame sway from 0.3 to 0.5 cells up to 1.4 to 1.9 cells. Mesh light on the new shot still reads (mean delta 0.022). `tests/test_fire_shot.py` (10 tests) plus fire, flame-mesh, sparse-fire-light, hot-pour and knowledge tests pass (62 + 15).
- **Inference / not run:** the "less grainy, finer, leans" verdict comes from the saved stills, not the real display. No Windows run, no AMD or llvmpipe pass (no shader or buffer change), no full suite.
- **Artifacts (committed on `openclaw/nb-fluids-spike`, not pushed):** `examples/fire_shot/{make_shot.py,render_evidence.py,fire_shot.nbcomp,README.md}`, `tests/test_fire_shot.py`, `benchmarks/fire_shot/fire-{look,curl,look-shot}-2026-10-30.json`, `docs/images/fire_shot/fire-{compare,curl}-2026-10-30.png` and `fire-look-{008,016,024}.png` plus the mesh-light-off still, `docs/FLUIDS_SPIKE.md` section "Fluids 13 B look" with the bundled copy. The simulation caches under `scratch/nb-lanes/run/big-L6/` are deleted.
- **State:** DONE for step B of 2. Real-display verdict unverified.
- **Next owner + concrete artifact:** Gonzo integrates the branch tip through `tick.py tick` and judges `docs/images/fire_shot/fire-compare-2026-10-30.png` on the real display. Needs Gonzo: a wind or curl-noise force in the solver would be a separate step with its own measured proposal; the upper smoke is still speckled at 128 samples.
- **Failure mode corrected:** the 13 B shot's straight grainy column with bright gaps came from a 32-cell grid (empty height slices in the cache) and 32 samples, not from the renderer; neither the renderer's fire math nor any solver parameter was changed.

## 2026-10-30 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step L of N: TVIScale gets Nuke's edge-preserving two-times enlargement, Lane 6 (Fluids, Claude Sonnet 5.5) step A of 2: sparse fire caches light nearby objects without expanding the volume (integrator tick)

- **What was done:** `main` `ae50d9fe` -> `295b2285` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `295b2285`: Ran 7464 tests in 534.106 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-30, 7:50 PM PDT — Lane 6 Fluids 13 B finish: sparse fire caches light nearby objects

- **What was done — evidence:** `volumerender.fire_light_grid()` and `mesh_fire_lights()` now accept sparse volumes. A shared `_fire_emission` helper serves the dense and sparse paths (dense output unchanged); the sparse path builds the same bounded coarse grid from rest values plus stored-tile differences, with no dense array. `gpuvolume.fire_light_view` uses it for sparse smoke lighting with a key covering tiles, rest values, shape, voxel size, origin, transform and light settings. New `tests/test_sparse_fire_lights.py` (13 tests) plus `tests.test_volume_fire_look`, `test_volume_flame_mesh`, `test_sparse_gpu_volumes`, `test_sparse_gpu_pathtrace` (84 tests) pass on the RTX 3080 Ti, Radeon 8060S and llvmpipe; `tests.test_knowledge` passes. No shader or buffer layout changed.
- **Inference / not run:** no Windows run, no full suite, no real-display visual check of a sparse fire shot.
- **Artifacts (committed on `openclaw/nb-fluids-spike`, not pushed):** commit `b9cce977`; `nodebased/volumerender.py`, `nodebased/gpuvolume.py`, `tests/test_sparse_fire_lights.py`, `docs/FLUIDS_SPIKE.md` and the identical bundled copy.
- **State:** DONE for step A of 2; step B not started.
- **Next owner + concrete artifact:** Gonzo integrates `b9cce977` through `tick.py tick` and checks the exact-tip suite log.
- **Failure mode corrected:** a cached sparse fire glowed but its floor stayed dark, because the light grid and mesh lights skipped sparse volumes. Left: a sparse volume whose temperature or flame is not stored in its own tiles still gets no halo.

## 2026-10-30, 7:50 PM PDT — Lane 2 fidelity step L: TVIScale is Nuke's edge-preserving two-times enlargement

- **What was done — evidence:** A new `TVIScale` always enlarges by 2: a bilinear 2x target (edge pixels carried outward), then `Iterations` (5) Jacobi steps of total-variation smoothing with four-neighbour weights `1/sqrt(difference^2 + A^2)` and a data term weighted by `Lambda` (5); `A` (0.0001) is floored at 1e-6; zero iterations is the bilinear target (`nodebased/tviscale.py`). The channels selector, mask input, mask channel, invert, inject, fringe and mix apply on the enlarged picture; named layers are enlarged too. `tests/test_tviscale_fidelity.py` (29 tests) asserts the output window is exactly twice the input's, each of Iterations, Lambda, A, channels, mix, filter, mask and mask channel changes pixels, more iterations lower the noise of a flat area, a larger Lambda keeps more noise, a sharp edge keeps a steeper step than a 3x3 box blur, tiles of 7, 16 and 32 pixels equal the full frame (with and without a mask), bypass passes the unscaled input, and a schema-27 graph with `power` and `filter` renders the same pixels (checked bit for bit against the code before this change, with a mask and mix). The schema is bumped 27 -> 28: migration sets `legacy_scale` on every saved TVIScale, which runs the old signed `power`, `filter`, mask and mix code unchanged. Fidelity row 19 goes 30.0 -> 100.0; overall 90.3 -> 90.4.
- **Artifacts:** `nodebased/tviscale.py`; edits in `core.py` (SPECS, LIMITS, migration), `imaging.py`, `tileexec.py`, `ops2d_chrome.py`, `knobs.py`, `nodecatalog.py`; `tests/test_tviscale_fidelity.py`; `docs/parity/nodebased_to_nuke.json`; `docs/PARITY_2D.md` and `docs/PARITY_2D_FIDELITY.md` with identical bundled copies.
- **State / unverified:** The solver is our reading of the guide's description: there is no Nuke render, so pixel equality with Nuke is unmeasured. The per-neighbour weights smooth slowly (about 13 % lower noise at the default 5 iterations on a bilinear 2x of Gaussian noise, 20 % at 30): the default look is gentle. Each channel is solved on its own (no colour coupling). The GPU 2D path does not carry TVIScale.
- **Next owner + concrete artifact:** Gonzo integrates the lane commit through `python3 scratch/nb-lanes/auto/tick.py tick` and reads its exact-tip full-suite log. If DiMo supplies a Nuke TVIScale render, `tools/make_nuke_reference_pack.py` and `--reference-dir` measure the pixel column for row 19.
- **Failure mode (legacy TVIScale):** a saved TVIScale that predates schema 28 is marked `legacy_scale` = 1 and keeps its old resize (power, filter, mask checked against the source, mix against the unscaled picture, output black outside the mask corner). If the migration did not run, such a node would silently become a 2x smoothing node and a `power` of -1 or 0 would be ignored; any new test that builds a TVIScale for the old behaviour must pass `legacy_scale=1`.

## 2026-10-30 — continuous mode merge: Lane 8 (2D on the GPU, GPT-5.6 Terra) step F of 6: ColorCorrect runs on the card, and exact half-float plates go up at half size (finish 1) (integrator tick)

- **What was done:** `main` `fa849477` -> `59b156f6` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `59b156f6`: Ran 7422 tests in 521.241 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-30, 9:15 AM PDT — GPU ColorCorrect/half-upload finish: measured negative result

- **What was done — evidence:** Repaired the GPU ColorCorrect test setup that silently skipped its comparisons, added a seeded 200-case sweep and exact-half eligibility/chain coverage, fixed a CPU curve-endpoint precision error and an EXR layer-only read error. Radeon 8060S, RTX 3080 Ti and llvmpipe each passed 186 focused tests; another 54 media tests passed. Three fresh interleaved playback passes per device/phase measured Radeon 18.21–18.78 fps after GPU ColorCorrect, then 13.21–13.33 with half-float preparation enabled. Explicit opt-in `NODEBASED_GPU2D_HALF_UPLOAD=1` retains the exact half-width upload; default-off restored Radeon to 18.14–18.90 fps in three fresh passes. The 24 fps goal remains unmet.
- **Artifacts:** Lane branch `openclaw/nb-2d-parity-b` commits `02649caa` and `77d069ce` contain code, tests, identical performance docs and committed `benchmarks/2d-gpu/R6-1/playback-colorcorrect-halfupload-1027f.json`. Raw timelines and test logs are deliberate local scratch under `scratch/nb-lanes/run/bench-L8-finish/` and `scratch/nb-lanes/run/tests-L8-*.log`; they are not committed. The generated plates under `scratch/nb-lanes/run/big-L8/plates/` remain deliberate scratch for the next decode investigation.
- **State:** Step F is done as a defensible measured negative result, pending exact-tip integration suite. The Radeon 24 fps gate, RTX 4070/A6000/Windows measurements and 240-frame playback are unverified/open. The half path is experimental until native-half decode removes its conversion cost.
- **Next owner + concrete artifact:** Gonzo integrates `77d069ce` through `python3 scratch/nb-lanes/auto/tick.py tick`, checks its exact-tip full-suite log, then briefs a bounded decode-retention experiment from `docs/PERFORMANCE.md` and the committed playback JSON. The next GPT job is chosen from the live budget and measured gate closure.
- **Failure mode:** Importing `DeviceTestCase` did not run `tests.test_gpu2d`'s module setup, so prior ColorCorrect GPU tests appeared green while skipped. A smaller transfer can still slow playback when preparing it consumes decode time; measure the whole first lap before changing a default.

## 2026-10-30 — continuous mode merge: Lane 8 (2D on the GPU, GPT-5.6 Terra) step F of 6: ColorCorrect runs on the card, and exact half-float plates go up at half size (finish 1) (integrator tick)

- **What was done:** `main` `1a60460d` -> `bf2bf204` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `bf2bf204`: Ran 7415 tests in 526.857 s, OK (skipped=39), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-29 — continuous mode merge: Lane 8 (2D on the GPU, GPT-5.6 Terra) step F of 6: ColorCorrect runs on the card, and exact half-float plates go up at half size (integrator tick)

- **What was done:** `main` `d6c3fc17` -> `644173f2` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `644173f2`: Ran 7413 tests in 520.729 s, OK (skipped=39), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-28, 11:10 PM PDT — NodeBased 0.47.0 publication confirmed

- **What was done:** Verified stable GitHub release `v0.47.0` is public with four uploaded assets, checksum manifest matches the three package digests, exact commit's Desktop conformance and Linux/Windows package-and-publish workflows succeeded, and the tick cleared `s["release"]`. Audited commit `010abf53`: only two version-line bumps and identical 0.47.0 additions to both release-notes copies. Verified the tick's one announcement in Releases → #nodebased, message `1565243612621578324`.
- **Artifacts:** Published release <https://github.com/neodimo/NodeBased/releases/tag/v0.47.0>; package workflow <https://github.com/neodimo/NodeBased/actions/runs/40813342879>; exact tag `v0.47.0` on `010abf53`; downloaded manifest `/tmp/NodeBased-0.47.0-SHA256SUMS` is deliberate local verification scratch. This note and `context/state.md` are intended repository notes on main; `status/nodebased.json` is the local status-board source, updated through `scripts/project_status.py`.
- **State:** Release done/published. The separate local-build timer recorded the on-host 0.47.0 build at 11:10 PM PDT and updated the stable launcher; manual real-display QA and pixel comparison against Nuke remain unverified. GPT lanes stay parked pending DiMo's Thursday priority choice.
- **Next owner + concrete artifact:** Gonzo performs real-display QA using the now-updated stable `/home/omid/AppImages/NodeBased.AppImage`; the local build ledger is `/home/omid/.local/share/nodebased/local-builds.json`. DiMo chooses the next GPT lane from #nodebased's priority question.

## 2026-10-28, 7:15 AM PDT — 0.46.1 built and installed locally

- **What was done:** The user timer built `v0.46.1` from its exact tag on this Linux host at 2:06 AM. `packaging/build.py` completed its frozen-AppImage smoke and live HTTPS probe before the builder installed the artifact. The local build ledger has no pending published tags. The stable `NodeBased.AppImage` link resolves to the new version; the desktop launcher and KDE task-manager pin use that stable target. The timer is enabled and its service exited successfully. Post-release main Desktop conformance passed on Linux and Windows for both the ZMerge merge and the subsequent release-state note.
- **Artifacts:** Local build `/home/omid/AppImages/NodeBased-0.46.1-linux-x86_64.AppImage` (140,057,080 bytes), build log `/home/omid/.local/share/nodebased/local-builds/v0.46.1.log`, ledger `/home/omid/.local/share/nodebased/local-builds.json`, and stable launcher `/home/omid/AppImages/NodeBased.AppImage` are local installed state, not repository artifacts. This note is committed/pushed on main. The independently published CI AppImage is at <https://github.com/neodimo/NodeBased/releases/tag/v0.46.1>.
- **State:** Local install done; frozen-app smoke and network checks passed via the packaging script. A manual visible-GUI launch remains unverified. Post-release main CI is green on both platforms.
- **Next owner + concrete artifact:** Gonzo uses `/home/omid/AppImages/NodeBased.AppImage` for real-display QA; the timer will keep future published versions current. DiMo's priority choice for the next GPT lane is pending in #nodebased.
- **Failure mode:** A published CI AppImage is a different artifact from the on-host build. Confirm the local ledger, build log, smoke result and stable launcher before reporting the local install done.

## 2026-10-28, 2:20 AM PDT — 0.46.1 published after Windows package repair

- **What was done:** The exact `v0.46.1` commit `503c4782` passed its full local suite and both Linux and Windows tag conformance. Both package jobs and the publish job passed. GitHub published four release assets at 2:03 AM; the `SHA256SUMS` entries match the three GitHub asset SHA-256 digests, and the manifest's downloaded bytes match GitHub's digest. The one release announcement, with DiMo ping, landed in Releases → #nodebased at 2:09 AM (message `1564929052349833269`). ZMerge merged into main afterward and is absent from this tag.
- **Artifacts:** Published release: <https://github.com/neodimo/NodeBased/releases/tag/v0.46.1>. Package workflow: <https://github.com/neodimo/NodeBased/actions/runs/40651078246>; tag conformance: <https://github.com/neodimo/NodeBased/actions/runs/40651078264>. `scratch/nb-lanes/run/release-0461-check/SHA256SUMS` is deliberate local verification scratch. This note and the status-board update are committed/pushed on main; release assets are published on GitHub. The `v0.46.0` tag remains pushed and unpublished.
- **State:** 0.46.1 release done. Real-display visual QA and real Nuke render comparison remain unverified; neither is claimed by the published release. The newer ZMerge main commit needs its own platform CI before any next release.
- **Next owner + concrete artifact:** Gonzo follows main's post-release Desktop conformance at `9ed23998` and the local Linux release installation separately. Future release work starts from `v0.46.1` and this published asset set.
- **Failure mode:** The 0.46.0 Windows package test assumed a software adapter supported GPU fire path tracing. The 0.46.1 repair tests CPU fallback there while retaining GPU/CPU comparison on capable adapters; the failed 0.46.0 tag was never moved or published.

## 2026-10-28 — continuous mode merge: Lane 2 (2D parity, GPT-5.6 Terra) step K of N: ZMerge chooses depth order and blends by the selected alpha (integrator tick)

- **What was done:** `main` `503c4782` -> `bb8bd82d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `bb8bd82d`: Ran 7410 tests in 511.407 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-28, 1:55 AM PDT — Lane 2 ZMerge depth and alpha controls recovered

- **What was done:** Recovered the stopped Terra worker's nine-file worktree diff. ZMerge now chooses near depth in either direction, blends its chosen nearer input using RGBA or a named alpha channel, preserves the old hard-switch result when `alpha_channel=none`, and matches full-frame and tiled output. Existing documents migrate to the old hard-switch behavior; new nodes expose `backwards` and `alpha_channel`. Evidence: the requested focused/compatibility checks passed (37 tests), pre-commit node/panel checks passed (82 tests), the added B-nearer regression passed, and the fidelity audit is current. The audit score changed from 90.2 to 90.3; ZMerge's row changed from 55.6 to 100.0.
- **Artifacts:** `nodebased/core.py`, `nodebased/imaging.py`, `nodebased/tileexec.py`, `nodebased/knobs.py`, `tests/test_2d_parity_step_d1.py`, `tests/test_tile_layers.py`, `docs/parity/nodebased_to_nuke.json`, and both copies of `PARITY_2D_FIDELITY.md` are committed on `openclaw/nb-2d-parity` with this note. Test logs in `scratch/nb-lanes/run/tests-L2-zmerge-recovery.log` and `tests-L2-zmerge-precommit.log` are deliberate local scratch.
- **State:** Lane step done on its branch; integration suite, cross-platform CI, real Nuke pixel comparison, and artist display review remain unverified. The 0.46.1 release tag is separate and does not include this lane step.
- **Next owner + concrete artifact:** Gonzo/integrator picks up branch `openclaw/nb-2d-parity` and `scratch/nb-lanes/run/report-L2.md`, runs the exact-tip stacked suite, then merges only on green. TVIScale stays held for Thursday's budget slice.
- **Failure mode:** The original worker exited after writing nine uncommitted files; a silent lane is not empty. Its first focused run passed and a second log lacked a terminal result. Recovery preserved the diff, reran the required checks, and added the B-nearer assertion before commit.

## 2026-10-27, 10:55 PM PDT — 0.46.0 package gate failed; 0.46.1 repair staged

- **What was done:** The 0.46.0 tag package run failed on Windows only. Its software graphics adapter correctly rejects GPU path-traced smoke/fire; two new tests had assumed every GPU adapter supported that path. Main now contains the test-harness repair, which asserts CPU fallback on unsupported adapters and retains GPU/CPU comparisons on supported ones. Focused tests passed on the local NVIDIA, Radeon and software-renderer paths. Fresh Linux and Windows Desktop conformance is running on the repair commit.
- **Artifacts:** Fix committed and pushed on main as `a8eb7c01` (`tests/test_volume_flame_mesh.py`). Failed package run: <https://github.com/neodimo/NodeBased/actions/runs/40627566266>. Replacement conformance: <https://github.com/neodimo/NodeBased/actions/runs/40640089541>. Patch-release notes and summary: `scratch/nb-lanes/auto/notes-0.46.1.md` and `summary-0.46.1.md` (local scratch; to be consumed by the release tick). The `v0.46.0` tag stays pushed and unpublished.
- **State:** Partial; full replacement CI and package jobs remain unverified. Do not announce a stable release yet.
- **Next owner + concrete artifact:** Gonzo checks Desktop conformance run `40640089541` and the latest main commit. On green Linux and Windows, arm `python3 scratch/nb-lanes/auto/tick.py release-no-preview 0.46.1 scratch/nb-lanes/auto/notes-0.46.1.md scratch/nb-lanes/auto/summary-0.46.1.md`, run `tick.py tick`, and follow exact-tip suite and tag workflows to publication. If red, repair the exact failure first.
- **Failure mode:** GPU availability alone does not imply the adapter supports GPU path-traced volumes; tests must check `gpupathtrace.soft_supported` and verify automatic CPU fallback where it is false.

## 2026-10-27 — continuous mode merge: Lane 6 (Fluids, GPT-6 Sol) step B of 2: flames that glow and light the scene around them (integrator tick)

- **What was done:** `main` `fa330061` -> `ac08e954` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `ac08e954`: Ran 7405 tests in 523.799 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-27 — continuous mode merge: Lane 2 (2D parity, GPT-5.6 Terra) step J of N: the filter, colour and keyer rows: Inpaint, LightWrap, ColorLookup, IDistort, DirBlur, Noise, Cryptomatte and the rest (finish 1) (integrator tick)

- **What was done:** `main` `aa4b8ade` -> `de3d65db` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `de3d65db`: Ran 7396 tests in 515.934 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-27 — continuous mode merge: Lane 2 (2D parity, GPT-5.6 Terra) step J of N: the filter, colour and keyer rows: Inpaint, LightWrap, ColorLookup, IDistort, DirBlur, Noise, Cryptomatte and the rest (integrator tick)

- **What was done:** `main` `04d38175` -> `4c06e706` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4c06e706`: Ran 7393 tests in 520.674 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-27 — continuous mode merge: Lane 6 (Fluids, GPT-6 Sol) step A of 2: the Hot pour looks right: a soft steam column, a real pouring stream, liquid you can see in the glass, a soft shadow (integrator tick)

- **What was done:** `main` `faa06788` -> `a805bb34` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a805bb34`: Ran 7391 tests in 508.914 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 4 (Generative, GPT-6 Sol) step A of 2: a second local generative provider with different conditioning, benchmarked against SDXL ControlNet (integrator tick)

- **What was done:** `main` `276415cc` -> `7fd30009` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7fd30009`: Ran 7391 tests in 521.916 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, GPT-5.6 Terra) step I of N: Read, Roto, Write, Group and the Tracker take the rest of Nuke's knobs, and Clamp clamps to values (finish 1), Lane 8 (2D on the GPU, GPT-6 Sol) step E of 5: the Roto matte draws fast enough for the Radeon to reach 24 fps (integrator tick)

- **What was done:** `main` `218b6e27` -> `1993c0af` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1993c0af`: Ran 7382 tests in 534.144 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, GPT-5.6 Terra) step I of N: Read, Roto, Write, Group and the Tracker take the rest of Nuke's knobs, and Clamp clamps to values (integrator tick)

- **What was done:** `main` `7ace3b8b` -> `e35b5256` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e35b5256`: Ran 7339 tests in 524.356 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, GPT-5.6 Terra) step H of N: the short knob rows: keyers, RotoPaint, Precomp, the time and metadata nodes, Glow, DropShadow and AddMix (resume) (finish 1) (integrator tick)

- **What was done:** `main` `2ec026ce` -> `093ba61e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `093ba61e`: Ran 7337 tests in 548.511 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 8 (2D on the GPU, Claude Sonnet 5.5) step D of 4: "Automatic" picks the card that is actually faster, and the Read stops paying 11 ms in file checks (integrator tick)

- **What was done:** `main` `8ac90404` -> `32ba3515` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `32ba3515`: Ran 7335 tests in 518.852 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step H of N: the short knob rows: keyers, RotoPaint, Precomp, the time and metadata nodes, Glow, DropShadow and AddMix (resume) (integrator tick)

- **What was done:** `main` `f90b6d84` -> `ffa7d4a6` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `ffa7d4a6`: Ran 7275 tests in 518.870 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 8 (2D on the GPU, Claude Sonnet 5.5) step C of 3: where the last 50 ms of the Radeon frame go, and the two-worker and GPU-default switches ready to flip (integrator tick)

- **What was done:** `main` `560840a6` -> `c8ee9c86` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `c8ee9c86`: Ran 7090 tests in 519.914 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step G of N: the Viewer and the Tracker take Nuke's knobs, plus Profile and CornerPin (resume) (integrator tick)

- **What was done:** `main` `9b0e1102` -> `53933a67` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `53933a67`: Ran 7062 tests in 520.382 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 8 (2D on the GPU, Claude Sonnet 5.5) step B of 2: the Roto matte is ready before the frame needs it, and four-CPU machines get the source workers (integrator tick)

- **What was done:** `main` `abae80a5` -> `65306a50` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `65306a50`: Ran 6937 tests in 507.830 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step A of 1: the first full-size frame after a rest reuses the scene packed for the half-size frames (resume 2) (finish 1) (integrator tick)

- **What was done:** `main` `9a7a1fca` -> `f30486f2` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `f30486f2`: Ran 6908 tests in 515.276 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step F of N: Roto, Text, Read and Write take the rest of Nuke's knobs, and the time nodes get Nuke's path mask (integrator tick)

- **What was done:** `main` `db12f300` -> `3ff19e79` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `3ff19e79`: Ran 6908 tests in 516.304 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 8 (2D on the GPU, Claude Sonnet 5.5) step A of 1: the Read decodes and the Roto matte of a frame run at the same time (integrator tick)

- **What was done:** `main` `4379060e` -> `e7ce209b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e7ce209b`: Ran 6790 tests in 513.130 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step E of N: Grade, Merge, Blur, Sharpen and Soften take Nuke's knobs, and every masked node gets Nuke's mask fringe, Lane 4 (Rendering, Claude Sonnet 5.5) step A of 1: the first full-size frame after a rest reuses the scene packed for the half-size frames (resume 2) (integrator tick)

- **What was done:** `main` `3c368b33` -> `d703209b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `d703209b`: Ran 6767 tests in 508.042 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-26 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step D of N: the time nodes, the keyers and the small channel nodes take Nuke's knobs, Lane 8 (2D on the GPU, Claude Sonnet 5.5) step C of 3: measure the GPU island against the gates and give the go/no-go (integrator tick)

- **What was done:** `main` `8dcd9693` -> `df49d0a4` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `df49d0a4`: Ran 6636 tests in 504.283 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-25 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step C of N: Roto, Read, Write, Crop, Premult and Unpremult take Nuke's knobs, and "clip to" changes pixels (integrator tick)

- **What was done:** `main` `499a0d2a` -> `4e87354d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4e87354d`: Ran 6537 tests in 491.052 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-25 — continuous mode merge: Lane 8 (2D on the GPU, Claude Sonnet 5.5) step B of 3: the GPU island matches CPU pixels everywhere, and fails safely (integrator tick)

- **What was done:** `main` `6e6e65d6` -> `a85129fe` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a85129fe`: Ran 6417 tests in 501.693 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-25 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of N: the shared Nuke node controls on every 2D node, plus Nuke's two-input Shuffle, Lane 8 (2D on the GPU, Claude Sonnet 5.5) step A of 3: the four-node GPU island runs the realistic comp, with zero readbacks between its kernels (integrator tick)

- **What was done:** `main` `2a5fcaa6` -> `695b0796` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `695b0796`: Ran 6393 tests in 500.272 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-25 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of N: measure how closely the 2D nodes match real Nuke nodes, and publish the score (integrator tick)

- **What was done:** `main` `5eccc56c` -> `b84823f5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b84823f5`: Ran 6286 tests in 488.047 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-25 — Lane 2 (2D parity, Claude Sonnet 5.5) step A of 1: the fidelity ruler: how closely each 2D node matches the real Nuke node, scored from the Nuke 17.0 Reference Guide

- **Why:** DiMo, 10/25 1:28 PM: "I'm very sure the 2d node parity is not mimicking actual Nuke node functionality yet." PARITY_2D.md counts 123 of 138 rows supported, but that grade means "a node does the main thing and has tests", and the golden images in `tests/data/golden/` lock NodeBased's own output. The most-used nodes are the thinnest (Grade has `exposure`, `multiply`, `offset`, `mix`; the guide's has ten knobs, a mask channel and an unpremult-by). This step builds the ruler. It changes no node behaviour.
- **What landed (measured):** `docs/parity/nuke_nodes.json`: the guide's tables for the 190 Nuke nodes named by the 138 rows (2,991 knob entries, 2,376 of them scored; sockets, names, tabs, defaults, `null` plus a note where the guide states none; no guide prose). `docs/parity/nodebased_to_nuke.json`: 177 pairings, 921 entries (220 same name, 237 renamed with a reason, 80 multi-param components, 34 panel features, 350 NodeBased-only). `tools/audit_nuke_fidelity.py` (re-runnable; `--write`, `--check`, `--reference-dir`) scores each node from knob coverage (main-tab knobs weigh 2, the rest 1), defaults, sockets, shared node chrome (channels, mask channel/invert/inject/fringe, unpremult-by, mix, bbox) and 47 formula cases on `Evaluator._kernel` against a 16 x 12 HDR+alpha fixture (the 30 Merge operations from the guide's algorithms, KeyMix, Gamma, Invert, Multiply, Add, Clamp, Premult, Unpremult, ColorMatrix; Grade and Exposure marked *convention* because the page states no formula); pixel truth is a column that reads `unmeasured`. `tools/make_nuke_reference_pack.py` writes the fixtures, one `.nk` per case and `render_all.sh` (not run: no Nuke here) and holds the loader the scorer uses once renders exist (per channel max abs error and worst pixel). `docs/PARITY_2D_FIDELITY.md` (bundled copy in `nodebased/data/docs`): overall **54.5** (56.1 counting only nodes NodeBased has), groups from Image 25.4 to Metadata 87.1, the 30 largest gaps. Formula findings: Grade's `offset` is scaled by alpha, 8 of 29 scored Merge operations differ from the guide's text outside 0..1, five channel nodes default `channels` to `rgb` where the guide says `all`. PARITY_2D.md gained a note saying what "supported" does not mean.
- **Tests:** `tests/test_nuke_fidelity_audit.py` (27: committed report equals a fresh run and its bundled copy, mapping complete and every pairing real, scoring rules, fixtures, loader on perfect, perturbed and missing renders, pack contents). Full suite on this tree (`suite2.sh`): Ran 6286 tests in 476.948 s, OK (skipped=35).
- **Not done:** no comparison with a real Nuke render, so the pixel-truth component is not in any number; the `.nk` scripts have not been opened in Nuke; the mapping is one reviewer's pass over 177 pairs (the top-ranked nodes got the closest look); 25 guide pages in the groups have no row in PARITY_2D.md (listed in the report); PlanarTracker has no guide page and Stabilize/Backdrop/Assert/PostageStamp/Input/Output/CopyBBox/VectorToMotion/UDIM Import list no knobs, so they are not scored.
- **Who:** Claude Sonnet 5.5 (Lane 2).

## 2026-10-25 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 1: NodeBased starts extremely fast, measured from launch to a usable window (integrator tick)

- **What was done:** `main` `695b5b96` -> `8e547ad6` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8e547ad6`: Ran 6259 tests in 489.964 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.



## 2026-10-25 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 1: the Direct picture never crashes the app on a Linux machine whose only graphics path is OpenGL (integrator tick)

- **What was done:** `main` `f3d28485` -> `c349d81d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `c349d81d`: Ran 6259 tests in 500.067 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-25 — Lane 2 (2D parity, Claude Sonnet 5.5) step A of 1: the Direct picture never takes the application down on a Linux machine whose only graphics path is OpenGL

- **Why:** The 0.43.0 and 0.43.1 package builds aborted in the Direct viewer tests on GitHub's Linux runners (llvmpipe through EGL is the only adapter). `e58904b7` only skipped the tests; the product took the same path, and a Rust panic aborts the process, so no `try/except` around `RetainedPresenter` can save an artist's unsaved work.
- **What landed (measured):** Reproduced here first: `tools/repro_gl_egl_race.py enum` and `twodev` abort with exit 134 and the CI text (`wgpu-hal-29.0.3/src/gles/egl.rs:182:14 ... BadAccess`), `single` survives; with `VK_ICD_FILENAMES` pointed at nothing and Mesa's EGL forced, the machine lists exactly the runner's one adapter (llvmpipe, OpenGL) and `main` with only the skip removed aborts (exit 134) in the Direct viewer tests. Cause: wgpu-hal's GLES backend gives every adapter the same EGL context but a lock of its own, and `Instance::enumerate_adapters` and `create_surface` make that context current with no adapter lock; two threads reaching it together make `eglMakeCurrent` answer BadAccess, the `unwrap()` panics inside an `extern "C"` function (`wgpuDevicePoll` in the CI trace) and Rust aborts. Fix (b): `nodebased/glguard.py`; on Linux the display request asks Vulkan alone (`backend_order`), `RetainedPresenter` refuses an OpenGL adapter before any format, device or buffer exists, `_start` checks `check_display_possible` before the surface is made (also refuses while the app's own 3D device is on OpenGL), and adapter enumeration in `gpu3d` goes through one gate, once, with the adapters kept alive (`_raw_adapters`), so no code here enumerates or drops an adapter while another thread holds the context. A Vulkan-only request with a busy OpenGL device on another thread survived 30092 requests; an OpenGL/auto request aborted.
- **Tests:** `OpenGLGuardTests` (stand-in adapters, no card needed: refusal on Linux only, GL-only machine, Vulkan-only requests, `_start` makes no surface and reports "direct viewer could not start: ..."), `tests/test_glguard.py` (one enumeration, never two at once, no cached failure); the card tests skip with the product's own reason where the guard refuses. Full suite on this exact tree (`suite2.sh`): Ran 6259 tests in 490.334 s, OK (skipped=35). The same Direct viewer, overlay, follow, playback, minify, card-display, graphics-adapter and 3D GPU modules on the simulated runner (Mesa EGL forced, no Vulkan driver, llvmpipe OpenGL the only adapter): Ran 255 tests, OK (skipped=145), no abort. An earlier full run caught 9 failures in `test_3d_gpu` / `test_3d_gpu_rt_integration` (mock `wgpu` modules met a stale adapter cache); the cache is now per `wgpu` module. CI not read; no Windows run (the guard is Linux-only, `glguard.affected`).
- **Not done:** Windows OpenGL (WGL) is not covered or claimed; a GL-only machine keeps its 3D views on the card as before (one device is safe on OpenGL).
- **Who:** Claude Sonnet 5.5 (Lane 2).

## 2026-10-24 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step B of 2: the four 3D scenes re-measured on the integrated card and the laptop stand-in, and the biggest cost there removed (finish 1) (integrator tick)

- **What was done:** `main` `f172aa42` -> `b5a07e2f` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b5a07e2f`: Ran 6245 tests in 537.667 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-24 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step B of 2: the four 3D scenes re-measured on the integrated card and the laptop stand-in, and the biggest cost there removed (integrator tick)

- **What was done:** `main` `132df25e` -> `0badef1e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0badef1e`: Ran 6216 tests in 485.541 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 1: the full-detail Hot pour refines within memory on all three machines (finish 1) (integrator tick)

- **What was done:** `main` `8dc0a3b1` -> `5b810c54` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5b810c54`: Ran 6196 tests in 485.912 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

# Lane 6 step A finish 1 — full-detail refinement memory (PARTIAL, 2026-10-23)

- **What was done — evidence:** A fresh RTX 3080 Ti run completed edited liquid and surface, then failed at whitewater frame 91 (16,174.4 MiB peak RSS, 9,821 MiB card allocation, zero swap). Warm-cache retry completed liquid, surface, whitewater and steam through frame 120; neighbor evaluation was reduced from 696,541 to 131,072. Retry total was 585.44 s; peak RSS 9,659.9 MiB, card allocation 5,706 MiB, zero swap. The integrated Radeon 8060S laptop stand-in completed all four stages under taskset CPUs 0–3 in 1,167.77 s, with 13,990.7 MiB peak RSS and zero swap.
- **Artifacts:** RTX and integrated JSON evidence and synchronized docs are committed in this worktree. Production caches were deleted after measurement. No push.
- **State:** PARTIAL only for direct A6000 measurement; this host exposes the RTX 3080 Ti, and no other connected device or SSH host is configured. A labelled 0.8–1.3× warm-RTX estimate remains in the docs. CPU/llvmpipe streaming, fluid-cache and timeline checks pass; knowledge-copy validation passes.
- **Next owner + concrete artifact:** Gonzo: if the direct A6000 measurement is still required, run `tools/benchmark_fluid_edit_stages.py --adapter default --full-only --edit pour_rate --max-process-rss-mb 16384` on the A6000 host; save its JSON under `benchmarks/hot_pour/` and refresh both `FLUIDS_SPIKE.md` copies.
- **Failure mode:** The first cold RTX whitewater attempt can exhaust the card buffer despite staying below process RSS and card-capacity gates; warm rerun with the adaptive neighbor budget completed all stages.

## 2026-10-23 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step A of 2: 3D scenes hold full rate from the first second, and the first picture comes up fast (finish 1) (integrator tick)

- **What was done:** `main` `2d19239d` -> `a2387bf8` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a2387bf8`: Ran 6196 tests in 486.034 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step A of 2: 3D scenes hold full rate from the first second, and the first picture comes up fast (integrator tick)

- **What was done:** `main` `8c6e2d2d` -> `7b69adb1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7b69adb1`: Ran 6193 tests in 482.396 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step A of 1: smoke and see-through objects with splats stay on the card, and Render mode settles the mixed scene faster (resume 1), Lane 6 (Fluids, GPT-6 Luna) step A of 1: the full-detail Hot pour refines within memory on all three machines (integrator tick)

- **What was done:** `main` `eed3dca0` -> `58b1f6aa` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `58b1f6aa`: Ran 6176 tests in 529.017 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

# Lane 6 step A — full-detail refinement memory (PARTIAL, 2026-10-23)

- **What was done — evidence:** Per-stage RSS, card-memory and swap sampling is implemented; caches/checkpoints are bounded and streamed, and whitewater neighbor work adapts to available card memory with a status message and reduced-budget retry. The cold 256³/192³ RTX run completed edited liquid and surface stages, then whitewater failed during frame 90 at 12,628.7 MiB RSS / 9,821 MiB card allocation. Warm retry completed whitewater through frame 120; steam reached frame 70 before timeout. Swap was zero.
- **Artifacts:** Commits `6beaa99a`, `97324671`, `81d1a8fa`, `cf144ec3` on `openclaw/nb-fluids-spike`, not pushed. Benchmark JSON and mirrored docs are committed; raw caches/logs remain deliberate local scratch pending cleanup.
- **State:** PARTIAL. Focused memory and knowledge checks passed; cache/timeline tests passed on CPU/llvmpipe, GPU whitewater checks passed on RTX/Radeon/llvmpipe. Laptop RSS run, complete steam stage and measured A6000 run remain open.
- **Next owner + concrete artifact:** Gonzo: run the integrated stand-in with `taskset -c 0-3` and the exact `tools/benchmark_fluid_edit_stages.py --full-only` harness; then resume the RTX steam stage from the recorded warm cache if available. JSON: `benchmarks/hot_pour/full-refinement-memory-rtx-2026-10-23.json`.
- **Failure mode:** The cold whitewater run exhausted GPU allocation during frame 90 despite staying within RSS/swap bounds. Warm-cache retry proved the adaptive neighbor reduction completes whitewater, while the 600-second bound still leaves steam incomplete.

## 2026-10-23 — Lane 6 step A: full-detail Hot pour memory — PARTIAL

- **What was done — evidence:** Added per-stage process RSS, process swap and card-memory sampling to `tools/benchmark_fluid_edit_stages.py --full-only`. Added disk-streamed simulation checkpoints, a 128 MiB evaluator-result cache, whitewater neighbor-list budgeting from live free card memory, deterministic quality reduction with a status line, explicit GPU-buffer release and a half-budget retry after allocation pressure. The 12 GiB virtual-memory cap stopped cleanly; the 16 GiB RSS-capped RTX run completed edited liquid and surface through 120 frames, then whitewater reached frame 80 and failed at frame 90. Stage measurements are in the benchmark JSON and synchronized docs. The latest retry that halves the budget has not yet run at production size.
- **Tests:** Memory-budget and cache-spill regressions pass. Cache/timeline tests pass on default and llvmpipe; whitewater GPU tests pass on RTX, Radeon and llvmpipe. Knowledge-copy tests pass.
- **Artifacts:** Commits `6beaa99a`, `97324671` and the documentation/notes commit are local on `openclaw/nb-fluids-spike`, not pushed. `benchmarks/hot_pour/full-refinement-memory-rtx-2026-10-23.json` and both `FLUIDS_SPIKE.md` copies are committed. Benchmark scratch at `scratch/nb-lanes/run/big-L6/stepA-retest/cache` is deliberate, generated data and must be deleted before handoff.
- **State:** PARTIAL. Full whitewater completion, steam stage, Radeon laptop RSS gate and real A6000 measurement are unverified. A6000 appears only as a labelled planning estimate.
- **Next owner + artifact:** Gonzo reruns `tools/benchmark_fluid_edit_stages.py --adapter default --full-only --edit pour_rate --max-process-rss-mb 16384` from this worktree after the exclusive GPU lock clears, then runs the integrated adapter with `taskset -c 0-3`.
- **Failure mode:** Whitewater's neighbor-list and GPU-buffer allocation can still fail on RTX even after free-memory sampling; retry at half the point budget is implemented to degrade and continue.

## 2026-10-23 — continuous mode merge: Lane 3 (Research, GPT-6 Astra) step R6 of 1: 2D compositing on the graphics card: how Nuke, Fusion, Resolve and Flame keep a node graph on the GPU, and what fits NodeBased (integrator tick)

- **What was done:** `main` `2e0a2b9d` -> `70cfa61a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `70cfa61a`: Ran 6140 tests in 481.325 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — Lane 3 R6: GPU-resident 2D compositing decision package — DONE

- **What was done — evidence:** Read the assigned brief and standing rules; fast-forwarded this research worktree to main snapshot `2e0a2b9d`. Added a dated, source-backed survey of Nuke 16/17/Blink, Fusion/Resolve, Flame, Natron/OpenFX, Copernicus and TouchDesigner; mapped the actual 2D evaluator, tile executor, decode pool and display code; wrote exactly three ranked runnable implementation briefs; added the radar sweep and its top next-look priority. Four documentation parts are in separate lane commits. No application code or GPU execution.
- **Inference kept separate:** Suggested float32 resident islands, allocation caps, 4–6 / 6–10 / 5–8 engineer-day stages and modeled speedups are proposals. Existing profiles differ from the brief's rounded 170 ms/5.7 fps baseline; they are not merged into a fictitious measurement. CPU ColorCorrect and Roto remain explicit costs. No A6000 or actual laptop performance is claimed.
- **Artifacts:** `docs/GPU_2D_COMPOSITING.md`, `docs/GPU_2D_PLAN.md`, `docs/RESEARCH_RADAR.md`, their byte-identical `nodebased/data/docs/` copies, and `briefs/proposed/R6-1-resident-four-node-spike.md`, `R6-2-resident-evaluator-and-viewer.md`, `R6-3-budgeted-roi-and-tiles.md`. Committed locally on `openclaw/nb-research`; not pushed. Temporary downloaded source/PDF/CI extracts and test logs under worktree `.r6-sources/` were discarded after extracting citations and verification summaries; no generated assets retained. Required integrator report is workspace scratch `scratch/nb-lanes/run/report-L3.md` (local, not committed).
- **Verification:** `tests.test_knowledge` passed before each docs part commit (7 tests, latest 0.021 s); explicit byte comparisons pass for all three changed doc pairs; new survey/plan/radar-section local links resolve; new tracked documents contain no home paths; `git diff --check` passes. Only documentation checks ran, no full suite or adapter tests. Public CI read confirmed v0.42.0 package failure and v0.42.1 package success; precise run links and scope are in the plan.
- **State:** DONE for R6 research parts 1–4. Proposed implementation, visual QA, new pixel gates and all new hardware performance measurements remain unstarted/unverified. Missing vendor scheduler internals and fetch limitations are recorded rather than guessed. Issue #3 and lane card are updated in closeout; no next worker starts here.
- **Next owner + concrete artifact:** Needs Gonzo: review/integrate these lane commits, then assign `briefs/proposed/R6-1-resident-four-node-spike.md` to a 2D lane. Prioritize the actual Windows hybrid laptop field kit and coordinate viewer ownership before R6-2. Use `docs/GPU_2D_PLAN.md` for CPU truth, transfer accounting, budgets and portability gates.
- **Failure mode guarded against:** A GPU-enabled node/display is not evidence of an end-to-end resident graph. Shader timing alone excludes transfers and CPU tails; cache repeats, proxies and stand-ins cannot establish full-resolution fresh laptop playback. Sequential-transform concatenation would change today's CPU reference pixels and is excluded from the acceleration spike.

## 2026-10-23 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (finish 4: the steam asks the liquid at the proxy tier) (finish 1) (integrator tick)

- **What was done:** `main` `fb299488` -> `82b88cd0` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `82b88cd0`: Ran 6140 tests in 487.314 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — Lane 6 finish 1: production proxy playback and adapter follow-up — PARTIAL

- **What was done — evidence:** `tools/benchmark_fluid_edit_stages.py` now reports the 120-frame forward bake separately from replaying the baked sequence in the retained evaluator. Production RTX 3080 Ti (256³ liquid / 192³ steam; pour key at frame 60), before/after per-device whitewater pipeline caching: first proxy 0.492→0.434 s with 16 FLIP and 1 smoke step; forward bake 32.40→29.89 s; cached replay 46.73→126.49 FPS (p95 24.95→10.12 ms, max 26.37→12.27 ms). The frame-120 split moved from 14.24 ms evaluation / 0.01 ms proxy / 7.85 ms draw to 1.99 / 0.01 / 7.10 ms. Radeon production replay moved 142.03→154.37 FPS (p95 7.35→7.07 ms); its bake varied 20.93→31.05 s, so no bake improvement is claimed. llvmpipe production first proxy measured 1.079→1.086 s; its full 64³/120-frame stand-in record remains the Oct 22 benchmark. Full-detail production split is incomplete: RTX ran out of device memory during base whitewater frame 31; Radeon completed the unchanged base prefix (553.5 s) and reached edited liquid frame 80/120, then was interrupted at ~14 GB RSS with swap active and no useful progress. No production refinement totals are claimed.
- **Artifacts:** Harness, JSON, docs and bundled docs are committed in this worktree; docs copies are byte-identical. Production JSON: `benchmarks/hot_pour/production-edit-refresh-rtx-2026-10-23.json`. Raw measurements/cProfile/logs are deliberate scratch in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/big-L6/finish1/`; generated sim caches were moved to user Trash (recoverable and still occupying disk; permanent cleanup is not performed). Branch is local; not pushed. Report: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L6.md`.
- **Tests:** `test_gpu_pipelines_are_reused_for_the_same_device`, `test_proxy_steam_geometry_sampling_never_requests_tier_one_liquid` and `test_resident_solid_composition_reuses_an_identical_animated_pose` pass. `tests.test_knowledge` passes all 7 tests; combined log: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/tests-L6-finish1-final.log`. The CLI accepts the production llvmpipe stand-in run.
- **State:** PARTIAL. The baked 12 FPS scrub gate passes on RTX and Radeon; first proxy is below one second on those GPUs, while llvmpipe production first proxy is 1.086 s. The complete production full-detail stage split remains missing.
- **Next owner + concrete artifact:** Gonzo owns a resource-bounded full-detail refinement measurement after the whitewater allocation/memory issue is resolved. Start from `tools/benchmark_fluid_edit_stages.py --full-only` and `benchmarks/hot_pour/production-edit-refresh-rtx-2026-10-23.json`; avoid rerunning the full-size Radeon prefix without a memory cap.
- **Failure mode:** The prior `proxy_120_frames_s` folded forward simulation into the scrub gate. The harness now distinguishes bake and cached replay; profiling also found per-frame whitewater pipeline construction, now cached by device. Full-detail whitewater on RTX exceeded device allocation; Radeon full refinement accumulated enough state to drive swap and was stopped before stage completion.

## 2026-10-23 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step A of 1: the mixed mesh and splat scene plays at 24 fps at full size, Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (finish 4: the steam asks the liquid at the proxy tier) (integrator tick)

- **What was done:** `main` `24be048a` -> `18a53f59` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `18a53f59`: Ran 6139 tests in 487.685 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — Lane 6 finish 4: steam uses the liquid proxy tier — PARTIAL

- **Why:** The production profile showed the tier-4 steam solver sampling its animated liquid geometry at tier 1, replaying hundreds of full-detail FLIP steps before the preview appeared.
- **What landed — evidence:** Fluid geometry tracks now propagate the evaluator's requested tier into both provider evaluation and digest lookahead. Tiered liquid point clouds provide a bounded proxy surface for steam emission and collision. Same-pose animated collider masks now retain a stable composed-solid identity, allowing the resident FLIP upload path to reuse it. A regression counts surface evaluation tiers and fails if tier-4 steam requests tier 1; another verifies same-pose solid/mask/system identity.
- **Tests:** Imaging/fluid-node/surface-cache/CPU fluid modules pass; resident GPU solver module passes on the RTX 3080 Ti; knowledge-copy tests pass and bundled docs are byte-identical.
- **Production evidence:** RTX 3080 Ti, 256³ liquid / 192³ steam, pour-rate key at frame 60: fresh first displayed proxy 0.375 s including drawing, with 16 proxy FLIP steps and one proxy steam step. The 120-frame proxy pass completed in 30.64 s (3.92 FPS; p95 336.14 ms; maximum 1,083.9 ms), so the 12 FPS scrub gate misses. The full-resolution refinement split and integrated Radeon were not measured.
- **Artifacts:** Code, tests, production JSON, mirrored docs, and this note are in the lane worktree. Raw evidence logs are deliberate scratch under `scratch/nb-lanes/run/big-L6/f4-first/`; the >1 GB generated cache is deleted before handoff. Not pushed.
- **State:** PARTIAL. The <1 s first-proxy gate and 120-frame completion pass; the 12 FPS timeline scrub gate remains open.
- **Next owner + concrete artifact:** Needs Gonzo: optimize the sequential proxy timeline using `tools/benchmark_fluid_edit_stages.py` and the `finish4_steam_proxy_tier` record in `benchmarks/hot_pour/production-edit-refresh-rtx-2026-10-23.json`; then measure the full-resolution refinement split and integrated Radeon if still in scope.
- **Failure mode:** Geometry track providers defaulted to tier 1 even when their downstream steam solver was tier 4. Surface preview output is a point cloud, so the tiered track now converts its bounds to a small proxy surface while full-detail geometry continues through the original path.

## 2026-10-23 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (finish 3: the proxy starts from a checkpoint) (integrator tick)

- **What was done:** `main` `17358279` -> `98fbab28` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `98fbab28`: Ran 6123 tests in 486.754 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — Lane 6 finish 3: proxy checkpoint reuse (PARTIAL)

- **What was done — evidence:** Added a dedicated exact per-frame checkpoint store for tiered liquid proxy solves and routed the tiered whitewater solver and its nested liquid solve through proxy stores. The frame-60 regression confirms checkpoints 1–59, one post-edit solve step, and bit-identical particle positions/IDs versus a clean replay. A second focused test verifies tiered whitewater reads the proxy solver store. The full focused fluid-cache module and knowledge-copy tests pass.
- **Production evidence:** RTX 3080 Ti, 256³ liquid / 192³ steam, pour key at frame 60: first displayed proxy measured 118.782 s with 504 FLIP and 1 smoke step. cProfile on a distinct warmed variant measured 120.355 s with the same counts and showed nested liquid replay through the full-detail store. After the whitewater routing change, a distinct edit variant measured 103.014 s with 496 FLIP and 1 smoke step; this is not a controlled before/after. First-proxy <1 s still misses. 120-frame and 12 FPS gates were not run.
- **Artifacts:** Source/test/harness/docs/JSON are committed locally in this worktree (not pushed); docs copies are identical. Production logs and profile remain as deliberate local scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/big-L6/`; the 6.2 GB production cache was moved to trash before handoff. Benchmark metrics are committed in `benchmarks/hot_pour/production-edit-refresh-rtx-2026-10-23.json` after commit.
- **State:** PARTIAL. Exact proxy checkpoint reuse is implemented and focused regressions pass; production latency gate and complete timeline remain open.
- **Next owner + concrete artifact:** Needs Gonzo: investigate remaining 496 FLIP steps and the one smoke step using `proxy-checkpoint-profile.log`, then rerun `tools/benchmark_fluid_edit_stages.py` on the RTX with fresh cache; finish 120 frames/12 FPS and remaining adapter/edit matrix.
- **Failure mode:** ParticleCache’s proxy path used a dedicated store, but whitewater’s nested liquid solve still used the full-detail cache, allowing replay despite valid proxy checkpoints. Routing it to the proxy store reduced the measured count on a distinct variant but did not close the gate.

## 2026-10-23 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (finish 2: the production-size gates) (finish 1) (integrator tick)

- **What was done:** `main` `6ac33bb8` -> `c5b52424` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `c5b52424`: Ran 6121 tests in 486.913 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — Lane 6 step B finish 2 (PARTIAL follow-up)

- **What was done — evidence:** A bounded cross-edit collider-pose cache is already committed at `decbff08`. Follow-up added checkpoint propagation for downstream liquid `ParticleCache3D` and `FluidWhitewater3D` streams. Fresh RTX 3080 Ti production profile (256³ liquid / 192³ steam; frame-60 pour edit) fell from the original 346.208 s to 122.454 s; collider solid voxelization fell from 188.852 s to 24.474 s and surface voxelization from 182.967 s to 21.424 s. A fresh displayed proxy still took 113.494 s, missing the <1 s gate. The profile still records 496 FLIP GPU steps and substantial domain-resize work. Focused surface-cache, fluid node and fluid solver tests passed (112); knowledge tests passed (7); bundled `FLUIDS_SPIKE.md` is byte-identical.
- **Artifacts:** Code/test/harness/docs/JSON are in this worktree; production measurements are committed to `benchmarks/hot_pour/production-edit-refresh-rtx-2026-10-23.json`. Raw logs and generated cache are under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`; generated cache was moved to trash before handoff. No push.
- **State:** PARTIAL. Still missing a passing <1 s first proxy, complete 120-frame proxy timings and 12 FPS hold, full-detail per-stage timing, cup-move and steam-edit production runs, and integrated Radeon runs. A6000 remains an unmeasured estimate only.
- **Next owner + concrete artifact:** Needs Gonzo: continue from `tools/benchmark_fluid_edit_stages.py` and the updated production JSON/profile; isolate the remaining FLIP replay/domain-resize work, then run the listed RTX and integrated production gates.
- **Failure mode:** Having a downstream `FluidLiquidSolver3D` checkpoint did not mean the edited picture would reuse it: `ParticleCache3D` and whitewater have distinct run keys and could replay a large prefix. Reuse hints now reach both; remaining production replay still dominates, so the fix is not yet a gate pass.

## 2026-10-23 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (finish 2: the production-size gates) (integrator tick)

- **What was done:** `main` `8b40b385` -> `7c3fe7d6` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7c3fe7d6`: Ran 6118 tests in 490.576 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-23 — Lane 6 step B finish 2: production gates (PARTIAL)

- **What was done — evidence:** The production gate harness now selects the `scene` viewer output, so its edit scope/checkpoint logic follows the interactive app. RTX 3080 Ti at 256³ liquid / 192³ steam, pour keyed at frame 60: cProfile measured the first edited scene evaluation at 346.208 s before proxy conversion/display. It attributes 188.852 s to collider solid voxelization, including 182.967 s in surface voxelization. Proxy timeline reached frame 70; that frame took 16.487 s and misses 12 FPS. The 120-frame total, full-detail split, cup/steam edits, integrated Radeon, and A6000 are incomplete or unmeasured. Docs-copy check passed (7 knowledge tests).
- **Artifacts:** Committed benchmark JSON `benchmarks/hot_pour/production-edit-refresh-rtx-2026-10-23.json`; harness `tools/benchmark_fluid_edit_stages.py`; table and evidence in `docs/FLUIDS_SPIKE.md` and identical bundle. Raw profiles/logs and cache scratch were moved to trash after recording the measurements; the committed JSON preserves the relevant metrics.
- **State:** PARTIAL. Production first-frame and 12 FPS gates miss; other requested adapter/edit/stage gates remain open. No solver optimization landed; profiling isolates collider voxelization as the largest measured cost.
- **Next owner + concrete artifact:** Needs Gonzo: optimize/retest collider voxelization from `nodebased/fluid3d.py` with a targeted regression, then resume RTX/Radeon gates with `tools/benchmark_fluid_edit_stages.py`.
- **Failure mode:** The first harness left `document["view"]` unset; `_fluid_simulation_scope` therefore returned an empty scope and the pour edit restarted from frame 1. Selecting `scene` corrects the benchmark classification. Profiling overhead is explicitly labeled and was excluded from any displayed-first-frame claim.

## 2026-10-23 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step A of 1: 3D comps at full size: the furnished room at 24 fps with the proxy off, and no tier flapping on the cube (integrator tick)

- **What was done:** `main` `240452a2` -> `2dbc03af` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `2dbc03af`: Ran 6118 tests in 487.945 s, OK (skipped=35), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step B of 2: 3D comps play at the comp's rate: the cube and the furnished room at 24 fps, the mixed scene progressively (resume 1), Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 6) (integrator tick)

- **What was done:** `main` `97729398` -> `74e1cffd` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `74e1cffd`: Ran 6102 tests in 486.402 s, OK (skipped=34), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step B finish 6 — PARTIAL

- **What was done — evidence:** On the merged Lane 2 pipeline, three uncached RTX 1080p read-ahead-on runs measured 5.680/5.382/5.449 fps (mean 5.504); off measured 4.626/4.580/4.557 (mean 4.588). The former 4.68 fps one-off is not reproduced with read-ahead on and is close to the no-read-ahead mean; these data cannot identify why that single run differed. Three implementation/measurement commits are `1b1df0bc`, `9bf0689b`, `8aac3a31`. Cold current-frame Reads are queued before the playback preview; targeted prefetch/tile tests passed (59). The final sampled RTX timeline has node evaluation 167.914 ms and display conversion 76.477 ms; node evaluation remains largest. A no-op vertical-row fit in Transform now returns the source image, and the isolated fractional-horizontal microbenchmark improved 24.918→23.933 ms median (3.95%) with exact pixels. Transform/bounding-box/tile regressions passed (76).
- **Artifacts:** Source, regression tests and STATUS.md are committed locally on `openclaw/nb-2d-parity-b`, not pushed. Local scratch evidence: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/finish6-read-ahead-on-{1,2,3}.json`, `finish6-read-ahead-off-{1,2,3}.json`, `finish6-final-{rtx,radeon,llvmpipe,laptop}.json` and corresponding `.log` files, `finish6-transform-before.log`, `finish6-transform-after.log`, `tests-L8-finish6-prefetch.log`, `tests-L8-finish6-transform.log`. These files are deliberate local scratch, not committed. Final first-pass rates: RTX 5.701 fps, Radeon 5.680, llvmpipe 6.749, four-core integrated stand-in 4.272. The benchmark warms frame 1 before playback, so end-to-end cold first-frame miss latency is unverified. RTX 4K/full-resolution stop gates were not rerun.
- **State:** PARTIAL. The 16 fps intermediate and 24 fps/4K gates remain unmet; the prior 4.68 fps discrepancy is unexplained. Focused tests passed as above.
- **Next owner + concrete artifact:** Gonzo to review/integrate these three commits and continue optimizing node evaluation from `finish6-final-rtx.json`; rerun cold-start miss timing with a benchmark that does not warm frame 1, then resume the 1080p/4K gates.
- **Failure mode:** The standard benchmark pre-renders frame 1 before starting its playback window, concealing first-frame cold Reads. Do not treat its frame-rate result as proof that the play-start misses were decoded asynchronously.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 2: the Direct picture's last gaps on this workstation: picking points, and filtered minification (integrator tick)

- **What was done:** `main` `58d04045` -> `da227d67` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `da227d67`: Ran 6055 tests in 503.305 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 2: uncached 2D playback overlaps the next frame's work with showing this one (integrator tick)

- **What was done:** `main` `bdeab15c` -> `70126645` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `70126645`: Ran 6044 tests in 483.552 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 5) (integrator tick)

- **What was done:** `main` `856fd60d` -> `845df0a6` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `845df0a6`: Ran 6026 tests in 475.322 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 4) (integrator tick)

- **What was done:** `main` `e6c503e0` -> `a6b885c5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a6b885c5`: Ran 6024 tests in 475.958 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 3) (integrator tick)

- **What was done:** `main` `65b02101` -> `1bf22fad` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1bf22fad`: Ran 6023 tests in 481.994 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B finish 1 (PARTIAL)

- **What was done — evidence:** A key at frame 60 restarts the fluid simulation from checkpoint 58 and solves frames 59–60; the changed result matches a complete replay bit for bit. The bake runs the 64-cell-capped proxy stream first, then full detail with isolated caches and per-frame promotion. An existing full-detail cache now supplies its proxy and full-detail view without additional solver steps, and the preview migration note appears once.
- **Artifacts:** Code and regression test in commit `1bc56588`; step A checkpoint/proxy implementation in `23963a80`; updated `docs/FLUIDS_SPIKE.md` and byte-identical bundled copy. Adapter logs are local scratch in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`.
- **State:** PARTIAL. The focused suite passes on RTX 3080 Ti, CPU/llvmpipe, and integrated Radeon 8060S; the docs parity check passes. The old-cache regression asserts zero additional solve steps and one migration callback. Production first-frame, 120-frame, full-refinement, and playback performance gates remain unmeasured. Production baking remains prohibited with 5.4 GB free against an 8.5 GB historical cache.
- **Next owner + concrete artifact:** Gonzo owns the production-size performance gates when disk capacity allows. Use `tests-L6-final-rtx-1022.log` and `docs/FLUIDS_SPIKE.md`.
- **Failure mode:** Before the preview-ring cache existed, a full-detail cache could be reused without a compact preview companion. The stop path now builds that companion from the full output entry and exposes the one-time migration in render status.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 1: Roto drawing and the zoom keys on the Direct picture, and a steady picture count (resume 1), Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 2) (integrator tick)

- **What was done:** `main` `a91609e1` -> `647fd5c8` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `647fd5c8`: Ran 6013 tests in 481.704 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the production-size Hot pour re-simulates only what changed (integrator tick)

- **What was done:** `main` `39cc8a78` -> `a98d95a2` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a98d95a2`: Ran 5993 tests in 479.331 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: an edit to the Hot pour shows its re-simulated first frames within a second (finish 2), Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the realistic comp plays uncached at 24 fps at 1080p, and the 4K comp plays progressively (finish 1) (integrator tick)

- **What was done:** `main` `b1aa277a` -> `1e750ce0` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1e750ce0`: Ran 5988 tests in 482.385 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step A finish 2 (PARTIAL)

- **What was done — evidence:** The 64-cell / 120-frame Radeon cup-move bake measured 29.06 s with float32/all steam-cache channels and 28.34 s with half-precision density+temperature, using the same warmed upstream setup and bounded per-node cache caps. The optimized measurement clears the 30 s gate; the 0.72 s difference is one paired measurement and may include timing variation. Cancellation after FLIP substep entry was measured in three trials on each adapter: RTX 3080 Ti median 0.554 ms / worst 21.443 ms; Radeon 8060S median 0.278 ms / worst 0.535 ms. New-edit first-frame solver time was at most 0.161 s RTX and 0.167 s Radeon. Real viewport presentation is not part of these timings.
- **Artifacts:** Cost harness `tools/benchmark_fluid_edits.py`; cancellation harness `tools/benchmark_fluid_edit_cancel.py`; final comparison JSONs `benchmarks/hot_pour/edit-cup-bounded-{baseline,optimized}-radeon-2026-10-22.json`, `edit-cancel-rtx-2026-10-22.json`, and `edit-cancel-radeon-2026-10-22.json`; updated Hot pour generator output and example graph; `docs/FLUIDS_SPIKE.md` and byte-identical bundled copy. These are committed in this worktree change. Targeted logs are in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/` and remain local scratch. All generated large cache directories from this pass were empty and removed; no production bake was started.
- **State:** PARTIAL. Cost and cancellation gates pass in the measured harnesses. Targeted GPU-liquid, rigid, preset, cache, surface and knowledge tests pass on RTX; resident solver tests pass on Radeon and llvmpipe. Real-display QA remains unverified because this session has no display session. Production-size refresh is deferred to step B by the brief; the historical cache is about 8.50 GB and free space was 5.7 GB.
- **Next owner + concrete artifact:** Gonzo owns real-display review of the timeline/stale indicator using the Lane 6 finish-1 UI and `docs/FLUIDS_SPIKE.md`. Step B owns a fresh production-size bake when scratch space permits.
- **Failure mode:** UI cancellation supplies `threading.Event`, while rigid pre-simulation expected `CancelToken.check()`; rigid stepping now accepts both interfaces. The first repeat benchmark reused a resident solver identity and missed instrumentation after trial one; each trial now receives a unique run identity. Generated preset/example files were regenerated after changing the preset builder.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: an edit to the Hot pour shows its re-simulated first frames within a second (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `f4aecaab` -> `68d134e1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `68d134e1`: Ran 5981 tests in 483.532 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step A finish 1 (PARTIAL)

- **What was done — evidence:** The timeline now shows the active simulation bake separately from ordinary display-cache marks. The previous picture stays visible with a stale indicator until the edited playhead frame replaces it. One background frame is queued at a time; forward playback stays within the contiguous solved range. A newer preview replaces queued work and signals active evaluation cancellation. The simulation identity is limited to the viewed solver dependency scope, so unrelated camera/light edits do not restart the bake.
- **Artifacts:** UI, queue, timeline, tests and both copies of `docs/FLUIDS_SPIKE.md` are committed in this worktree. No generated bake/cache artifact was created. Targeted UI log: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/tests-L6-finish-ui2.log`; docs log: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/tests-L6-finish-knowledge2.log`. Earlier CPU and llvmpipe logs are `tests-L6-finish-cpu.log` and `tests-L6-finish-llvmpipe.log` in the same run folder.
- **State:** PARTIAL. The UI/state tests, CPU and llvmpipe solver regressions, and docs check pass. The solver cost reduction has not been verified; Radeon cup movement remains 32.43 s versus a 30 s gate. A fresh production-size bake is deferred because available disk is 5.7 GB and the historical cache occupies about 8.5 GB. Real-display QA and measured worker-stop latency under 50 ms remain unverified.
- **Next owner + concrete artifact:** Gonzo owns real-display QA and the next measured optimization pass, using the warmed phase profiles and gate table in `docs/FLUIDS_SPIKE.md`. Regenerate the production-size baseline only when sufficient scratch disk is available.
- **Failure mode:** A broad “any edit to a visible fluid graph” invalidation would wastefully restart fluid baking for camera/light changes. Invalidation now checks the solver dependency scope; a regression test covers this distinction.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: an edit to the Hot pour shows its re-simulated first frames within a second, Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `6e5c5d12` -> `e4679e89` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e4679e89`: Ran 5974 tests in 488.078 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 2: the catcher shadow off the CPU), Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `09dd7655` -> `31880d7a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `31880d7a`: Ran 5971 tests in 504.316 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step C of 3: overlays and the pixel readout follow playback, and the Direct picture becomes a docked viewer DiMo can switch on, Lane 4 (Rendering, Claude Sonnet 5.5) step A of 2: find where 3D playback's time goes, commit the profile, and remove the biggest costs (integrator tick)

- **What was done:** `main` `0a388be9` -> `25f44534` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `25f44534`: Ran 5966 tests in 482.342 s, OK (skipped=32), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `a5caf406` -> `4e23575d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4e23575d`: Ran 5864 tests in 475.996 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where 2D comp playback's time goes on a real comp, commit the profile, and remove the biggest costs (integrator tick)

- **What was done:** `main` `907d56d7` -> `74523025` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `74523025`: Ran 5855 tests in 469.929 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step C of 3: Render mode settles in seconds: stop sampling where the picture is already clean, Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the drag picture within a third of a second, the scrub within a third, and playback at the set rate (finish 1) (integrator tick)

- **What was done:** `main` `e9534e70` -> `6eeba375` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `6eeba375`: Ran 5851 tests in 477.642 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step B finish 1 (DONE)

- **What was done — evidence:** Clipped solid-over paint work is byte-identical to full-tile rendering. A valid 1920 × 1800 virtual desktop exposed the Properties slider for an actual pointer drag. RTX 3080 Ti measurements pass all first-picture, maximum in-drag gap, full-tier settle, scrub, and playback gates. The completed table is in `docs/ROTO_TRACKING.md`; its bundled copy is byte-identical.
- **Artifacts:** Commits `9b150c7c`, `031f7e2b`, and `1c3f9bc5` are in this worktree. Test logs and field JSON are retained as local scratch in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/` (`L8-finish1-tests.log`, `L8-finish1-playback-tests.log`, `L8-finish1-lens-gpu.log`, and `live-picture-L8-finish1-rtx-tall.json`).
- **State:** DONE for the left-out finish scope. No off-site GPU/laptop run or real-display visual review is claimed.
- **Next owner + concrete artifact:** Gonzo reviews/integrates and performs real-display/Windows visual QA using `docs/ROTO_TRACKING.md` and the local RTX measurement JSON.
- **Failure mode:** A 1920 × 1080 virtual desktop placed the Properties control at y=1293, outside the screen, and produced no preview. Re-run with a virtual screen tall enough to contain the panel; the corrected 1920 × 1800 run measured the slider successfully.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (finish 1) (integrator tick)

- **What was done:** `main` `219263f7` -> `873cc57b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `873cc57b`: Ran 5819 tests in 483.785 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step B of 2: the drag picture within a third of a second, the scrub within a third, and playback at the set rate (integrator tick)

- **What was done:** `main` `031f9394` -> `b9fc0920` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b9fc0920`: Ran 5811 tests in 476.762 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `d40e3fb3` -> `4f678da1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4f678da1`: Ran 5806 tests in 496.606 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B finish 3 (PARTIAL)

- **What was done — evidence:** Fresh frame-60 and frame-90 stills now show a soft coherent steam plume, continuous liquid reaching the cup pool, a smooth pool, and a broad soft catcher shadow. RTX 3080 Ti and Radeon 8060S CPU/GPU comparisons passed on frames 1, 60 and 120. A fully converged 1920 × 1080 render at noise threshold 0.01 took 69.253 seconds; the best pass-size experiment took 68.235 seconds. Both miss the strict under-60-second target.
- **Artifacts:** Updated generator, graph, evidence runner, test, stills, profile JSON, and byte-identical docs copies in this worktree; large render cache is temporary scratch and was trashed before handoff.
- **State:** PARTIAL; the only remaining acceptance item is fully converged 1080p under 60 seconds.
- **Next owner + concrete artifact:** Gonzo coordinates a catcher/path-render cost reduction with L4 using `benchmarks/hot_pour/hot-pour-render-profile-2026-10-22.json`.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (finish 1) (integrator tick)

- **What was done:** `main` `66217360` -> `6434a596` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `6434a596`: Ran 5806 tests in 480.759 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `1e10ca44` -> `8d5fefe9` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8d5fefe9`: Ran 5805 tests in 490.616 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B finish 2: softer catcher, steam and strict 1080p gate still open (PARTIAL)

- **What was done — evidence:** Replaced the path-traced Point key with a 0.7 × 0.7 Rect key; fresh frames 60 and 90 show a visibly broader catcher shadow. Removed source noise and extra turbulence, then tuned volume density. The steam remains granular in both final stills. The full-cap 1080p render converged every pixel at noise 0.01 in 79.72 seconds. Lowering the nominal sample budget to 64 finished in 40.68 seconds but left 3.78% of pixels unconverged (max noise 0.52335). CPU/GPU comparisons passed at frames 1, 60 and 120 on the RTX 3080 Ti.
- **Artifacts:** `examples/hot_pour_shot/{hot_pour_shot.nbcomp,make_shot.py,render_evidence.py}`, `tests/test_hot_pour_shot.py`, both identical `docs/FLUIDS_SPIKE.md` copies, frames 60/90, and the benchmark JSONs are committed in this worktree. Generated caches under `workspace/scratch/nb-lanes/run/big-L6/finish5` through `finish10` were trashed after measurement. `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L6.md` is the handoff report.
- **State:** PARTIAL. Shadow softness passes visual review. Steam softness and settled 1080p under 60 seconds remain unverified/failed. A6000 and Windows are assigned to DiMo and are outside this pass.
- **Next owner + concrete artifact:** Gonzo to continue from `benchmarks/hot_pour/hot-pour-render-profile-2026-10-22.json`; reducing `pt_samples` caps adaptive convergence, so an under-minute fix needs the renderer-side catcher sampling cost reduced or another measured approach. L6 does not own `nodebased/shadowcatch.py`; coordinate that change with L4.
- **Failure mode:** `shadow_blur` on a Point light does not soften this path-traced catcher; the Rect light provides the actual penumbra.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `155e3974` -> `87791fd8` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `87791fd8`: Ran 5805 tests in 474.693 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step A: cold first-play profile and affinity-aware read-ahead (PARTIAL)

- **What was done — evidence:** Cold RTX profile found 22 serial Read decodes in 8 seconds (median 85.3 ms, p95 98.2 ms; 1.946 s total) and 22 RotoPaint passes (median 47.2 ms; 1.038 s total). The full-frame fallback did not consume decode-ahead, and optional `Roto.bg` was omitted from read traversal. Full Evaluator and tile paths now share immutable bounded Raster decodes, including named layers. Four-worker prefetch starved playback (15.564 / 15.571 / 7.617 fps); one worker restored discrete adapters near baseline. Disabling prefetch at four-or-fewer CPUs yielded 16.473 fps integrated with zero dropped frames. Final RTX / llvmpipe / integrated rates are 21.356 / 20.953 / 16.473 fps versus the 24 fps target.
- **Artifacts:** Commits `1711c3f6`, `0c67c6d7`, `16d4821c` on `openclaw/nb-2d-parity-b`, local and not pushed. Profile and final adapter JSONs live under `benchmarks/live-picture/`; final docs copies are byte-identical. Exact handoff: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L8.md`. The first integrated one-worker sample overlapped CPU tests and was trashed; its clean rerun is recorded separately.
- **Verification:** `tests.test_tileexec`, `tests.test_decodepool`, `tests.test_media`, `tests.test_boundingbox`, `tests.test_live_picture_bench`, `tests.test_roto_paint`: 114 passed. `py_compile`, `git diff --check`, docs-copy check passed.
- **State:** PARTIAL/BLOCKED: cold playback remains below 24 fps on every adapter.
- **Next owner + concrete artifact:** Gonzo continues using `benchmarks/live-picture/profile-1022-cold-rtx.json` and the final `after-*-cold-*.json` timelines; rerun all adapters under `/tmp/nb-gpu.lock`.
- **Failure mode:** Four simultaneous EXR decoders can starve preview/UI. Bound the pool using available CPU affinity. `NB_QA_PLATE` expects the sequence directory, not the filename pattern.

## 2026-10-22 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step B of 3: the mixed mesh and splat scene's first look under one second, and a sharper first look (finish 1) (integrator tick)

- **What was done:** `main` `9ce12b9a` -> `7f6acd4e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7f6acd4e`: Ran 5802 tests in 475.065 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `3bc0b4c3` -> `4b39823a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `4b39823a`: Ran 5802 tests in 483.726 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (finish 1), Lane 4 (Rendering, Claude Sonnet 5.5) step B of 3: the mixed mesh and splat scene's first look under one second, and a sharper first look (integrator tick)

- **What was done:** `main` `5d612c6f` -> `644f3778` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `644f3778`: Ran 5802 tests in 472.720 s, OK (skipped=19), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `e0963ac8` -> `89870cb0` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `89870cb0`: Ran 5766 tests in 543.776 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 8 step A finish: corrected drag proof; playback cadence remains just under target (PARTIAL)

- **What was done — evidence:** The integrated stand-in's absent drag picture was a benchmark coordinate/window-size bug. Qt supplied logical pointer coordinates at 2× scaling, `xdotool` addressed physical X11 pixels, and the old 1920 × 1080 logical window exceeded the 1920 × 1080 Xvfb surface. The corrected harness scales pointer coordinates and sizes to the screen geometry. Fresh 8-second unprofiled runs delivered held-drag frames on RTX, llvmpipe and integrated stand-in; first picture from the first Qt-received move was 0.198 s, 0.193 s and 0.177 s. Cold tier-4 playback was 21.330, 20.979 and 15.463 fps. On separate warm-range timelines, playback was 23.842, 23.845 and 23.841 fps with respectively 192/192, 192/192 and 193/193 display-cache hits and zero graph evaluations. Thus the render graph is out of the playback interval, but the strict 24 fps target still misses every adapter.
- **Artifacts:** `tools/live_picture_bench.py`, `tools/_live_picture_driver.py`, `tests/test_live_picture_bench.py`, both byte-identical copies of `docs/ROTO_TRACKING.md`, and corrected cold/warm benchmark JSONs under `benchmarks/live-picture/`. All are local and included in this commit. The prior warm JSON's direct residency field was invalid because it queried `region=None`; playback-hit counters are recomputed from the event timeline and the invalid field is explicitly null. No push. Transient shell logs remain in `/tmp` and are not evidence artifacts.
- **Verification limit:** The corrected cache-residency verifier was not rerun end-to-end: other lanes held the exclusive GPU queue. The reported warm-cache hit counts are derived from the completed archived timelines (all playback lookups hit; zero graph evaluations), while the newly corrected direct residency field remains unverified live.
- **State:** PARTIAL. The 0.5-second first-move picture gate is met on all three adapters. The 24 fps playback gate remains open on all three. Focused tests, benchmark self-check, diff check and docs-copy check pass. No product rendering code changed in this handoff.
- **Next owner + concrete artifact:** Gonzo continues playback cadence investigation using `benchmarks/live-picture/after-*-warm-range.json` timelines and `tools/live_picture_bench.py --warm-playback-range`; preserve the strict measured rate and report if the 24 fps gate remains unattainable.
- **Failure mode:** A high-DPI benchmark that sends Qt logical coordinates directly to physical-pointer tooling can report a nonexistent adapter render failure. Keep the DPI conversion and physical-screen window geometry in the harness.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the Direct picture shows the true float picture at rest, and the A/B compare runs on the card (integrator tick)

- **What was done:** `main` `ce3b26ae` -> `7dc30c79` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7dc30c79`: Ran 5762 tests in 481.719 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (finish 1) (integrator tick)

- **What was done:** `main` `386675a0` -> `7f93a326` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7f93a326`: Ran 5742 tests in 476.537 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 Step B finish 1: strict render gate and legible stills (PARTIAL for unavailable hardware validation)

- **What was done — measured:** RTX 3080 Ti fixed-32 frame 60 at 960 × 540: 35.55 s. Render-mode 1920 × 1080, adaptive threshold 0.01: 45.464 s, all pixels converged, max noise 0.009999989, mean 11.71 samples/pixel (max observed 1,560 under the 2,048 cap). New 16–64 adaptive showcase stills took 14.95/15.33/15.68 s. Visual review: the golden stream reads as a pour into the centered cup; steam is faint and the table shadow is still hard-edged. CPU/GPU comparisons at frames 1/60/120 passed on RTX and AMD Radeon 8060S; adaptive/majorant GPU modules passed on NVIDIA and forced-integrated invocations. Hot pour tests and knowledge tests pass. Preview evidence from Step B remains the earlier 64-cell Radeon proxy because preview code was unchanged.
- **Artifacts:** source commits `f8705d4c`, `81cb9b33`; docs, images, benchmark JSON and bundled docs commit `880218da`; all on `openclaw/nb-fluids-spike`, local and not pushed. Exact report: `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/report-L6.md`. Profiling and adapter logs are in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/profile-L6-finish1-settled2048.log` and `render-evidence-L6-finish1-64.log`, `render-integrated-compare-L6-finish1.log`, `tests-L6-finish1-{nvidia,integrated,shot,knowledge}.log`. Large task-owned cache moved to trash.
- **State:** local 12 GB render targets, frame comparisons, still refresh, and docs are complete. PARTIAL overall because the brief's left-out A6000 and Windows validation remain unverified. The 48 GB A6000 estimate is updated and explicitly unmeasured.
- **Next owner + concrete artifact:** Gonzo, when an A6000/Windows host is available, run `examples/hot_pour_shot/render_evidence.py` and compare its result with the dated profile and stills in `benchmarks/hot_pour/` and `docs/images/hot_pour_shot/`.
- **Failure mode:** one-step steam transmittance made the transparent liquid nearly black. Disabling steam cast shadows for this shot and allowing four transmission bounces restored the golden liquid while retaining the visible volume and geometry shadow.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `8662c7bb` -> `8f0b4c30` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8f0b4c30`: Ran 5742 tests in 483.002 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step B of 2: the Hot pour settles in under a minute (integrator tick)

- **What was done:** `main` `d91eadd0` -> `9ffbd0b7` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `9ffbd0b7`: Ran 5740 tests in 478.929 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — Lane 6 step B of 2: the Hot pour settles in under a minute (PARTIAL)

- **What was done — evidence:** With one volume-shadow step, the RTX 3080 Ti frame-60 960 × 540 fixed-32 render took 22.735 s vs 200.38 s at 16 steps. The optimized 1080p adaptive run took 83.511 s, mean 8.379 spp, cap 64, 29 passes; 99.405% pixels passed noise threshold 0.01. Frame 60 NVIDIA CPU/GPU comparisons passed at frames 1/60/120. Integrated AMD Radeon 8060S 64-cell/120-frame preview: 29.7 s bake, 134.31 warm FPS, 13.4 ms stop-to-full-detail. Adaptive/volume targeted tests: 12/12 NVIDIA, 12/12 forced-integrated invocation, 12/12 llvmpipe; Hot pour tests 2/2; knowledge tests 7/7. The stills remain grainy and the pour stream reads weakly.
- **Artifacts:** Code commit `a71af0cd`; profile commit `e3d1ca0f`; docs, profile summary, preview JSON, comparison stills and baseline copies are committed as `1b267f28` on `openclaw/nb-fluids-spike` (local only, not pushed). Scratch logs under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`; large task cache removed before closure.
- **State:** Partial against the fully-settled 1080p and legible-pour goals. 0.595% of pixels remain above the threshold at the 64-sample cap; visual quality is not accepted as final. A6000 and Windows unverified.
- **Next owner + concrete artifact:** Gonzo to review the baseline/new frames in `docs/FLUIDS_SPIKE.md`, then improve the pour/shadow quality using `examples/hot_pour_shot/hot_pour_shot.nbcomp` and rerun `examples/hot_pour_shot/render_evidence.py`.
- **Failure mode:** the first scratch profile attempted to serialize NumPy image arrays directly; it was corrected to persist compact numeric summaries. The first optimized 1080p result was initially estimated from GPU subphase timing; final reporting uses the measured 83.511 s wall time.

## 2026-10-22 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `d3bb958d` -> `80fcbd48` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `80fcbd48`: Ran 5740 tests in 475.161 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: the Hot pour's first look in under a second (fix the red suite) (integrator tick)

- **What was done:** `main` `e10c77a6` -> `63be6ad5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `63be6ad5`: Ran 5739 tests in 495.779 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-22 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 3: edit on the Direct picture: clicks reach the same handles the Viewer would pick, Lane 4 (Rendering, Claude Sonnet 5.5) step A of 3: an edit in Render mode is picked up at once: passes stop midway instead of finishing first, Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (finish 1) (integrator tick)

- **What was done:** `main` `8e58480a` -> `3db8ecaf` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `3db8ecaf`: Ran 5739 tests in 472.117 s, OK (skipped=17), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step A of 2: find where the live picture's time goes, commit the profile, and remove the biggest costs (integrator tick)

- **What was done:** `main` `5d3a10db` -> `a40e37c8` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a40e37c8`: Ran 5706 tests in 496.222 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step A of 2: the Hot pour's first look in under a second (integrator tick)

- **What was done:** `main` `89c3b9e1` -> `05a538ec` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `05a538ec`: Ran 5704 tests in 507.047 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — Lane 6 step A finish pass: first look and refinement responsiveness (DONE on lane branch)

- **What was done — evidence:** the progressive worker now retains camera-independent liquid/geometry preparation across equivalent scene evaluations and uses shorter heavy-scene GPU dispatches. Warm acceleration build/packing fell from 144–146 ms to 1.5–2.9 ms. Camera-move cancellation measured 2.0 ms RTX / 1.8 ms Radeon. Frame-60 CPU/GPU mean absolute RGB error stayed 9.96e-5 on both adapters. Cold/warm press-to-picture: RTX 1.170 s / 0.214–0.249 s; Radeon 1.048 s / 0.225 s.
- **Artifacts:** `nodebased/gpupathtrace.py`, `tests/test_fluid_firstlook.py`, `benchmarks/hot_pour/firstlook-timeline-2026-10-20.json` and `benchmarks/hot_pour/firstlook-integrated-2026-10-20.json`, `docs/FLUIDS_SPIKE.md`, `docs/INTERACTIVE_VOLUMES_LIQUIDS.md` and byte-identical bundled copies are committed on `openclaw/nb-fluids-spike` (local, not pushed). Test logs are deliberate scratch: `tests-L6-finish-a.log`, `tests-L6-finish-a-integrated.log`, `tests-L6-finish-a-cpuadapter.log` and `tests-L6-finish-a-docs.log` in `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`.
- **State:** done against the brief's timing, acceleration, cancellation and comparison gates. Windows and real-display review remain unverified.
- **Next owner + concrete artifact:** Gonzo; perform visual review on the real display, using the two first-look JSONs and `tools/benchmark_fluid_firstlook.py`.
- **Failure mode:** an identity-only scene key missed acceleration reuse because camera-move evaluation creates equivalent new scene objects; lowering the fixed-band target did not change adaptive dispatch sizes. Content comparison and adaptive chunk sizing address those misses.

## 2026-10-20 — continuous mode merge: Lane 3 (Research, GPT-6 Astra) step R5 of 1: a first look under a second for volumes and liquids: what production renderers do (integrator tick)

- **What was done:** `main` `403fc9ec` -> `392217d5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `392217d5`: Ran 5700 tests in 504.190 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — Lane 3 R5: interactive volumes/liquids research complete

- **What was done:** researched production preview/preparation/sampling strategies and recent papers; inspected main `15e28165` renderer paths; wrote a dated evidence register, implementation mapping with labelled effort estimates and three ranked proposed Rendering briefs using Fluids artifacts. Evidence: existing sparse majorants, scoped preparation reuse, device-buffer reuse and retracing progression are visible in code. Inference: persistent preparation, bounded previews and sample continuation are the most direct next experiments; their speed gains remain unmeasured.
- **Artifacts:** `docs/INTERACTIVE_VOLUMES_LIQUIDS.md`, dated R5 section and priority line in `docs/RESEARCH_RADAR.md`, byte-identical `nodebased/data/docs/` copies, and `briefs/proposed/R5-{1-persistent-render-preparation,2-budgeted-liquid-volume-preview,3-persistent-convergence-and-denoise}.md`. Committed on the research lane branch, not pushed; no app code, weights or GPU execution. Targeted knowledge-test logs are deliberate lane-run scratch, named `tests-L3-R5-part*.log`; integrator report is `report-L3.md` in the lane-run directory.
- **State:** research deliverables done; knowledge-document validation passed before each documentation commit. No implementation/hardware performance or image-quality gate claimed. The brief's new Hot pour timings are missing from inspected main, which still marks that sweep outstanding; source fetch limitations are recorded in the source register.
- **Next owner + concrete artifact:** Gonzo integrates and assigns the ranked `briefs/proposed/R5-*.md`; Rendering recovers the raw Hot pour timing record or records a fresh exact-scene baseline before comparative acceptance. A6000/laptop remain explicit unverified reporting tiers; shader/buffer work must include Radeon and llvmpipe correctness.
- **Failure mode guarded against:** proposing already implemented skipping/adaptation as absent, treating a vendor demo as target-card latency, conflating allocation reuse with sample accumulation, or calling a filtered/capped picture converged.

## 2026-10-20 — continuous mode merge: Lane 6 (Fluids, GPT-6 Luna) step S2 of 2: the Hot pour reads as a pour (finish 2: the auto-solver crash, then the card numbers) (finish 1) (integrator tick)

- **What was done:** `main` `15e28165` -> `91049701` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `91049701`: Ran 5700 tests in 486.583 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP2b of 2: overlays on the direct picture, the laptop check, and the default-viewer decision, Lane 4 (Rendering, Claude Sonnet 5.5) step H2 of 2: keep the image refining after the first look (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step T2 of 2: a live picture while the artist works: the drag refreshes within a third of a second and refines after, playback honours the proxy tier, analysis never freezes the window (finish 1) (integrator tick)

- **What was done:** `main` `5bea8768` -> `e9588d4b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e9588d4b`: Ran 5700 tests in 500.166 s, OK (skipped=20), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

# Lane 8 T2 finish 1 — real-display measurements (PARTIAL, 2026-10-20)

- **What was done — evidence:** measured the merged viewer on the 1920 × 1080 QA comp. RTX 3080 Ti: drag first picture 1.142 s from the first move (after release), settle 3.888 s; uncached scrub showed tier 8 at 0.595 s, tier 4 at 0.978 s, tier 1 at 4.914 s. Playback at the 24 fps target: 0.125 fps (tier 1), 2.624 fps (tier 4), 12.84 fps (automatic), with automatic tier choices 1 → 2 → 4. llvmpipe, while Lane 4's integrated-adapter progression benchmark ran: drag first picture 1.414 s after release, settle 4.416 s; scrub first picture 0.808 s, settle 9.446 s; playback 0.125 fps (tier 1), 1.618 fps (tier 4), 0.125 fps (automatic, tiers 1 and 2). The merged docs test passed (7 tests) and bundled docs are byte-identical.
- **Inference:** both adapters miss the 0.3 s first-picture target; RTX automatic playback misses the 24 fps target. The llvmpipe values are a contention-affected sample. PlanarTracker's bad cube-face pick remains a selection error; the contained checker-floor failures indicate a separate robustness limitation.
- **Artifacts:** documentation and bundled copy committed on `openclaw/nb-2d-parity-b`; detailed real-display logs and test output are local scratch: `scratch/nb-lanes/run/qa-L8-T2-llvmpipe-1020-r2.log`, `qa-L8-T2-rtx3080ti-1020.log`, `tests-L8-T2finish1-knowledge.log`. No push. The first provisional playback log is excluded because its counter counted preview generations rather than changed frame numbers.
- **State:** partial; the implementation is useful, measurements are recorded, latency and set-rate gates remain unmet.
- **Next owner + concrete artifact:** Gonzo should scope any follow-on performance change using the two QA logs and this `docs/ROTO_TRACKING.md` entry, then repeat `scratch/nb-qa/1015-roi/scripts/s12_t2.py` on both adapters.
- **Failure mode:** do not use the first provisional FPS output; it counted preview generations as presented frames. The corrected counter counts a presented frame only when both the preview generation and current frame number change.

## 2026-10-20 — Lane 4 step H2 of 2 (keep the image refining), finish 1: DONE

- **What was done — evidence:** the earlier H2 pass landed the worker-thread Render mode (`nodebased/progressivejob.py`), denoised passes, preview/refining/settled labels, and the convergence tests against Render3D (mesh room, depth of field, lens, mixed mesh and splat; green on RTX 3080 Ti, Radeon 8060S, llvmpipe). This finish ran `tools/benchmark_progression.py` on a quiet card under the exclusive GPU lock at 1920 x 1080 on all three adapters, recorded the numbers in `docs/3D_FOUNDATION.md` (Lane 4 step notes, bundled copy byte-identical), and committed the result JSONs. Targeted rerun: tests.test_knowledge, test_3d_progressive_job, test_3d_progressive_convergence, test_3d_first_look: 64 tests OK.
- **Measured (furnished room / mixed mesh+splat):** RTX first image warm 0.52 s / 1.02 s, settled 8.2 s / 82.5 s, cancel 35 ms / 403 ms. Radeon 0.54 s / 0.83 s, settled 16.7 s / 34.9 s, cancel 22 ms / 406 ms. llvmpipe furnished only: 0.82 s warm, settled 97 s (correctness, no target claimed).
- **Inference / not measured:** A6000 and the hybrid Iris Xe + RTX 4070 laptop are unreachable here; Windows not run. The mixed scene misses the one-second target on the RTX (1.02 s warm).
- **Artifacts:** committed `benchmarks/progression/{rtx3080ti,radeon8060s,llvmpipe}-1020.json`; local only (not committed) time-stamped PNG frames under `scratch/nb-lanes/run/prog-L4/{rtx,amd,cpu}/`. Removed a stray empty tracked file `timeout`.
- **State:** done. Remaining slow: mixed-scene full-size passes and cancel latency (~0.4 s, a pass cannot be interrupted midway). Next owner: Gonzo; concrete artifact: a mid-pass cancel check in the path tracer (follow-up, not started).

## 2026-10-20 — Lane 2 VP2b: overlays and readout on the direct picture, laptop check kit, default-viewer memo (DONE)

- **What was done — evidence:** the Viewer's foreground painting is now one function (`Viewer.paint_overlays`); `Viewer.render_overlay` paints it for the direct view's transform into a premultiplied RGBA image that the card draws one texel per pixel over the picture (`RetainedPresenter.set_paint`, `PAINT_SHADER`). Drawn read-only: A/B wipe line, handles and B half, region-of-interest box, format masks, Roto, Tracker, Transform and other handles, corner labels. Pixel-compared against the Viewer's own widget at scale 1.0 and 1.75 (wipe, moved wipe, ROI, three mask modes, Roto, Transform) in `tests/test_directviewer_overlays.py`; a pixel agrees when a Viewer pixel within one step has the card's colour within 16/255. Pixel readout on hover reads the float frame (`Viewer.pixel_readout_info`, shared with the Viewer's readout) and is painted bottom left; a played frame is read with its own geometry (`Viewer.readout_geometry`). `tools/laptop_present_check.py` (field kit) with `--self-check` (ran here, no Qt or wgpu) and a `result.json` per run recording the adapter and backend used and the frame cadence; steps in `docs/FIELD_TEST.md`. Decision memo in `docs/DIRECT_VIEWER.md`: not yet the default.
- **Tests:** `tests.test_directviewer_overlays`, `test_directviewer`, `test_directviewer_playback`, `test_laptop_present_check`, `test_carddisplay`: Ran 93, OK on the RTX 3080 Ti, the Radeon (integrated) and llvmpipe (cpu), on a private Xvfb display and under the GPU lock (`scratch/nb-lanes/run/vp2b-{default,integrated,cpu}.log`). Targeted regression on the Viewer: compare, roto, transform handles, look, frame, mode, navigation, strip: 125 OK; desktop, punchlists, parity, lens view, artist tools, knowledge: 274 OK.
- **Inference / not measured:** no Windows number exists or is claimed. Clicks on the direct surface still only pan and zoom: no test shows a handle click lands on the Viewer's pick, so there is no routing setting. Overlays do not refresh per played frame; the real-display walk, A6000 and laptop runs are owed.
- **Failure mode found:** `StubPresenter` in the tests lacked the new paint members and made the widget fail over, which hid the dock in two display-only tests that skip without a display; stubs now carry `paint_size`, `paint_key`, `set_paint`.
- **State:** done; default stays off. **Next owner:** Gonzo schedules DiMo's laptop run and the real-display walk.

## 2026-10-20 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice, Lane 4 (Rendering, Claude Sonnet 5.5) step H2 of 2: keep the image refining after the first look (integrator tick)

- **What was done:** `main` `ae4d6c04` -> `5b4e2e25` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5b4e2e25`: Ran 5673 tests in 487.940 s, OK (skipped=16), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-20 — Lane 2 VP1 respawn: already landed, re-verified on main (DONE, no new code)

- **What was done — evidence:** the tick spawned `L2-viewer-present-a.md` (VP1 of 2) at 5:02 AM although VP1 merged to main on 10/18 and 10/19 (`docs/DIRECT_VIEWER.md`, `nodebased/directviewer.py`, merges `bbb03482` and `2673f45b`) and VP2a followed. The lane state's merged list was empty because the plan name changed to the VP2 plan, which looks like the cause of the respawn. I fast-forwarded the lane branch to main (`ce83279b`, 0.41.0), changed no code, and reran the VP1 modules there: `tests.test_directviewer`, `tests.test_directviewer_playback`, `tests.test_carddisplay`: Ran 66, OK (skipped=4), on the offscreen platform (`scratch/nb-lanes/run/tests-L2-vp1-recheck.log`).
- **Inference / not re-measured:** the shared GPU lock was held by another process, so the three-adapter runs, the real-display walk and the cadence/CPU comparison were not repeated; the numbers in `docs/DIRECT_VIEWER.md` (10/18 and 10/19) stand as the record. Colour/alpha, A/B/ROI, Roto/Tracker hit order, Windows, native Wayland and device-loss gaps are listed in that file under "Known gaps (VP1)" and are unchanged.
- **Artifacts:** nothing new; branch `openclaw/nb-2d-parity` equals main at `ce83279b`, nothing uncommitted. Report: `scratch/nb-lanes/run/report-L2.md`.
- **State:** done (previously landed); the switch stays off by default.
- **Next owner + concrete artifact:** Gonzo should mark VP1 merged in the lane state (so the card stops offering it) and run the queued `briefs/L2-vp2b.md`; VP2 parity/Windows gate remains Gonzo's.
- **Failure mode:** a lane step whose plan name changes before it is recorded in `merged` is spawned again; check `git log --grep` on main for the step name before spawning.

## 2026-10-20 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice (finish 1), Lane 6 (Fluids, GPT-6 Luna) step S2 of 2: the Hot pour reads as a pour (finish 2: the auto-solver crash, then the card numbers) (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step T2 of 2: a live picture while the artist works: the drag refreshes within a third of a second and refines after, playback honours the proxy tier, analysis never freezes the window (integrator tick)

- **What was done:** `main` `9fe51455` -> `667e0e81` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite (parallel runner) at `aca39b06`, rebased onto 9fe51455 as `667e0e81` (adds only the roto timing-test headroom; tests.test_roto rerun OK): Ran 5641 tests in 642.700 s, OK (skipped=16), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

# Lane 8 T2 — live pictures, playback adaptation, background analysis (PARTIAL, 2026-10-19)

- **What was done — evidence:** committed progressive tier-8/tier-4/final refresh for drag snapshots, knob edits and uncached scrubs; adaptive playback tier reduction/restoration; PlanarTracker worker progress/cancel. Tracker analysis uses its existing worker path. The exact-HEAD targeted command passed 72 tests (`scratch/nb-lanes/run/tests-L8-T2-combined.log`, local scratch). Roto drag preview tests verify the real document and undo stack remain unchanged. The stored broad floor corners and a contained checker-floor region failed tracking at grids 8, 16 and 24; cube face is flat and featureless, a bad pick, while floor failures expose a robustness limitation. **Inference:** implementation addresses the interaction gaps but does not establish latency/rate gates.
- **Artifacts:** source/docs are committed on `openclaw/nb-2d-parity-b`; `docs/ROTO_TRACKING.md` and `nodebased/data/docs/ROTO_TRACKING.md` are byte-identical. Three implementation commits: `fae62337`, `3fbba399`, `3b0c3927`. Local QA driver and logs are under `/var/home/omid/.openclaw/workspace/scratch/nb-qa/1015-roi/scripts/` and `scratch/nb-lanes/run/`; no intended benchmark artifacts have been committed. No push.
- **State:** partial. First-picture-under-0.3-second, settled latency, 12 GB adapter and llvmpipe measurements, tier-1/tier-4/automatic playback FPS, and maintaining the artist-set rate are unverified. The integrator's full suite held `/tmp/nb-gpu.lock` through the brief's 30-minute wait; no lock bypass or process interruption.
- **Next owner + concrete artifact:** Gonzo should run `scratch/nb-qa/1015-roi/scripts/s12_t2.py` on display `:48`, with the RTX run under exclusive `/tmp/nb-gpu.lock`, then run the CPU-adapter variant after the GPU lock is released; append only observed values to both `docs/ROTO_TRACKING.md` copies. Recheck `scratch/nb-lanes/run/tests-L8-T2-combined.log` for the exact targeted results.
- **Failure mode:** do not call the feature gate done from code/tests alone; historically the broad floor track failed even after selecting the valid checker-floor region, so “bad plane pick” does not explain all of Finding 8.

## 2026-10-16 — continuous mode merge: Lane 3 (Research, GPT-6 Astra) step V1 of 1: how the fastest apps keep their interface at display rate, and what our Python + Qt + wgpu setup can reach, Lane 6 (Fluids, GPT-6 Luna) step S2 of 2: the Hot pour reads as a pour (finish 2: the auto-solver crash, then the card numbers), Lane 8 (2D parity B, GPT-6 Luna) step T1 of 2: the region of interest never breaks the picture (finish 1) (integrator tick)

## 2026-10-19 — Lane 2 VP2a (Opus sprint): the direct picture plays smoothly beside the Viewer and takes 4K pictures

- **What was done:** found what blocked playback at 4K: the Viewer's visible crop and the whole frame alternated, every
  size change made a new 133 MB float texture (draws up to 262 ms on the RTX 3080 Ti) followed by a 34 to 38 ms
  `write_texture` on the Qt thread. Uploads now go through a staging ring of three reused buffers filled on a filler
  thread; the draw records the copy only; one picture waits between offer and copy, newer offers replace it; textures
  grow and are reused, one per float format; 4K float frames go up as half floats (48 ms over the RTX's USB4 link as
  float32, 32 ms now). Pictures over the card's texture or buffer limit are refused by name and the Viewer shows them;
  the worker converts such frames itself. Closing the dock releases the device, buffers and the retained picture
  (newest only); the cache inspector shows its size.
- **Measured** (this workstation; private Xvfb display; the A6000 and the laptop not measured): graph press, drag and
  timeline scrub under 31 ms at the 95th percentile while the dock plays, 1080p and 4K, RTX and llvmpipe; a 4K picture
  costs the Qt thread about 1 ms on all three adapters (VP1: 11 to 25 ms); 241 of 241 frames reached the window in
  order at 1080p on the RTX and llvmpipe, 171 of 241 at 4K on the RTX (card side), 241 of 241 at 4K on llvmpipe.
- **Tested:** `tests.test_directviewer_playback` (new), `tests.test_directviewer`, `tests.test_carddisplay` on the
  RTX, the Radeon and llvmpipe and with wgpu hidden; `tests.test_artisttools`, `tests.test_artist_tools_wiring`,
  `tests.test_topbar`, `tests.test_e2e5_ui_punchlist`, `tests.test_knowledge`.
- **Not done:** real-display runs (workers may not use display :0); the Radeon cannot present to Xvfb; the hot pour
  comp was not played (a path-traced fluid shot that needs a bake); not pushed; no full suite. Report:
  `scratch/nb-lanes/run/report-L2-vp2a.md`.

## 2026-10-19 — Lane 2: sprint branches integrated (wire disconnect, Properties overhaul, viewer colour on the card)

- **What was done:** Lane 2's branch was reset to `main` (0.39.0; backup branch `backup/nb-2d-parity-pre-sprints-1019`) and
  the three finished 10/18 sprint branches were brought in by cherry-pick: `wire-disconnect-1018` (5 commits, clean),
  `props-overhaul-1018` (5 commits, clean; ColorCorrect migration is schema 24 -> 25, main was at 24), and the five
  commits of `astra-sprint-a-1018` that were not already on main (its first four are main's VP1 commits). Conflicts in
  `nodebased/directviewer.py` kept main's device-loss recovery, Wayland refusal and `needs_wgpu` skips and added the
  card display transform on top: recovery and close now also release the float texture, the retained frame for
  recovery is the whole ledger payload (QImage or linear), and close skips its idle wait on a device already lost.
  `docs/DIRECT_VIEWER.md` keeps main's known-gaps list with the sprint's colour and playback notes merged in.
- **CI fixes:** (1) props-overhaul's exit -11 in `tests.test_knob_sliders` was a stack overflow: with CI's DejaVu
  fonts PointsTo3D's x/y `WrappingRow` fit on one line only while the Properties scroll bar was hidden, so the row and
  the bar flipped each other through QScrollArea's resize handling (reproduced here with DejaVu fonts and PySide6
  6.12.0, 306,000 layout passes on one panel, gdb showed 10,000 nested resizes). `WrappingRow` now widens back only
  with `WIDEN_SLACK` (16px) to spare; regression test `WrappingRowHysteresisTests`. (2) The "cannot schedule new
  futures after shutdown" traces all came from `compact_display_cache` after the window closed; guarded. (3)
  `tests.test_carddisplay.CardDisplayAgreementTests` carries `needs_wgpu`. (4) `tests/test_tracker_completion.py`
  pins its old-document version to 15 instead of `SCHEMA_VERSION-1`.
- **Tested:** 31 targeted modules one interpreter each, with and without a wgpu-hiding shim, all OK; card tests also on
  the Radeon and llvmpipe through force-adapter.py; batch 20 of CI's run under DejaVu fonts with PySide6 6.12.0 offscreen
  and the slider/colour-picker modules on a private Xvfb display. Report: `scratch/nb-lanes/run/report-L2.md`.
- **Not done:** not pushed; no full suite; no Windows run; no real-display visual QA.

## 2026-10-19 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice (finish 1) (integrator tick)

## 2026-10-19 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step H1 of 2: a denoised 1080p first look within one second (integrator tick)

- **What was done:** `main` `e08cc94e` -> `0a89b108` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0a89b108`: Ran 5547 tests in 21350.370 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.


## 2026-10-19 — Lane 6 S2 finish attempt (PARTIAL)

- **What was done:** Queued the corrected still-sample matrix with the required exclusive GPU lock. The 30-minute wait expired while the integrator full-suite process retained a shared lock; no lock was bypassed. No new GPU timings were measured.
- **Artifacts:** Updated `docs/FLUIDS_SPIKE.md` and its byte-identical bundled copy. `tests.test_knowledge` passed. Lock-wait log: `scratch/nb-lanes/run/L6-matrix.log` (empty because the command never acquired the lock).
- **State:** PARTIAL. The matrix, progressive first-look/settle, and Radeon preview/stop remain unmeasured.
- **Next owner + concrete artifact:** Gonzo retries `scratch/nb-lanes/run/measure-L6-S2-matrix.py`, `scratch/nb-lanes/run/progressive-L6-s2.py`, and the 64-cell Radeon playback row from `tools/benchmark_hot_pour.py` after the shared GPU lock clears.
- **Failure mode:** An exclusive GPU benchmark can sit behind an integrator's shared suite lock; preserve the lock and report PARTIAL when the 30-minute wait expires.
## 2026-10-20 — Lane 6 step A cache-lifetime fix (DONE on lane branch)

- **What was done — evidence:** prepared scene entries now have explicit retention; ordinary first-look scopes clear on exit, volume previews retain equivalent acceleration across camera moves, and cancellation clears the entry. The retained-scene assertions now cover safe content-equivalent reuse and camera-independent preparation. Four requested test modules passed on the RTX 3080 Ti. A fresh Hot pour run measured 1.186 seconds cold and 0.374 seconds warm.
- **Artifacts:** `nodebased/gpupathtrace.py`, `nodebased/progressiverender.py`, `tests/test_3d_first_look.py` committed locally on `openclaw/nb-fluids-spike`; test log and timing JSON are deliberate scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`.
- **State:** done for this fix; full suite, AMD/llvmpipe reruns, Windows and real-display visual review remain unverified.
- **Next owner + concrete artifact:** Gonzo reviews the lane commit and merge-candidate results; use the test log and `tools/benchmark_fluid_firstlook.py` for follow-up.
- **Failure mode:** thread-local prepared-scene data previously survived a scope and could satisfy later first-look renders without preparing their scene; cancellation also left the entry behind.
## 2026-10-22 — Lane 6 B Hot pour finish 2 — PARTIAL

- **What was done — evidence:** Fresh RTX 3080 Ti frames 60/90/120 render in 41.90/40.87/41.63 s. They show a continuous golden stream from spout to pool, a smoother liquid surface after final denoising, a still-grainy plume and a still-crisp table-catcher shadow. CPU/GPU frame comparisons at 1/60/120 pass. Fixed-32 960 × 540 took 10.689 s. Adaptive 1920 × 1080 at noise threshold 0.01 converged every pixel (max noise 0.009999999) in 60.857 s, missing the strict under-60 target by 0.857 s.
- **Artifacts:** Source/generator, test, docs, evidence JSON and stills are committed locally as `1e09b529` on `openclaw/nb-fluids-spike`; not pushed. The latest unrendered candidate was reverted to the last measured settings. Render logs and cache are under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`; the finish5 queued render was canceled before starting after waiting behind another shared GPU job.
- **State:** PARTIAL. Continuity and pool smoothing improved; steam softness, shadow softness and full-HD-under-60 remain open.
- **Next owner + concrete artifact:** Gonzo; rerun `examples/hot_pour_shot/render_evidence.py --adapter default --workdir <cache-directory>` and `scratch/nb-lanes/run/profile-hot-pour-finish-final.py <worktree> <cache-directory> settled-1080p` after the shared GPU queue clears; tune steam density/noise and catcher-shadow softness while shaving at least 0.858 s.
- **Failure mode:** the 4096-sample adaptive cap left 0.014% of pixels above threshold; raising to 8192 fully converged but took 60.857 s. A temporary finish5 candidate could not be visually/performance validated because its queued render never acquired the GPU lock. An earlier bridge graph exceeded Scene3D's input slots; corrected by replacing the unused backdrop slot with the bridge at `object7`.
## 2026-10-22 — Lane 6 step A edit-driven Hot pour re-simulation — PARTIAL

- **What was done — evidence:** Added a complete-frame callback to the fluid cache bake path and a warmed edit benchmark covering pour velocity, cup movement, and steam vorticity. On the RTX 3080 Ti, each edited output chain completed in 30.15–50.19 s; first solved volume frame was 0.078–0.302 s. On the Radeon 8060S, edits completed in 23.68–32.43 s and first solved volume frame in 0.055–0.256 s. Radeon cup movement missed the strict <30 s full-chain gate by 2.43 s. CPU/llvmpipe and targeted regression suites passed. On-screen progressive presentation was not measured. The warm RTX diagnostic attributes 8.45–9.24 s to steam host work, 3.94–12.34 s to steam pressure, and 1.22–1.38 s to cache writes; cup movement also spends 14.29 s in whitewater output. Diagnostic timings are synchronized and for cost ranking.
- **Artifacts:** Changes, benchmark harness, warmed timing JSON, Radeon and RTX phase profiles, and mirrored docs are committed locally on `openclaw/nb-fluids-spike`; no push. Logs and temporary caches are deliberate scratch under `/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/`. Historical production-size Radeon bake evidence remains `benchmarks/hot_pour/radeon-full-bake.json`; no fresh production-size edit bake was run.
- **State:** PARTIAL. Cache bake callback and measurement harness are landed. Radeon cup-move time misses target. Artist-facing timeline/stale-result/edit-cancel wiring, cost reduction, on-screen first-volume proof, and a fresh production-size bake remain incomplete/unverified.
- **Next owner + concrete artifact:** Gonzo owns coordination with the app/UI lane to wire progressive frames and stale/cancel handling; use `tools/benchmark_fluid_edits.py` to optimize and remeasure the Radeon cup-move case, and the historical production artifact above as scale reference.
- **Failure mode:** The first harness timed edits from a cold cache and overstated recomputation. It was corrected to pre-bake the base graph, retain unchanged caches, and measure only the affected output chain; cold figures are marked superseded in the docs.
## 2026-10-22 — Lane 6 step A finish 1: progressive edit timeline — PARTIAL

- **What was done — evidence:** Fluid edits in the currently viewed chain now keep the prior picture visible with a `SIM STALE` label until the current edited frame replaces it. The timeline adds an amber simulation-frame band, and the window submits one background frame at a time through its existing single-owner preview queue. Playback is bounded to the contiguous solved prefix; stopping playback resumes the queue. Replacing the preview queue sets the active cancellation event. State, queue, timeline and view-dependency tests pass.
- **Artifacts:** UI wiring is in `nodebased/app.py`, queue support in `nodebased/playback.py`, ruler marks in `nodebased/timeline.py`; test in `tests/test_fluid_edit_timeline.py` and `tests/test_playback.py`. Mirrored docs are `docs/FLUIDS_SPIKE.md` and `nodebased/data/docs/FLUIDS_SPIKE.md`. Changes are committed locally on `openclaw/nb-fluids-spike`; no push. Production-size scratch was not created.
- **State:** PARTIAL. Real-display QA, measured <50 ms worker-stop latency, verified cost reduction and a fresh production-size bake remain outstanding. The historical production bake is ~8.50 GB and the workspace currently has 5.7 GB free, so a new run would not fit safely.
- **Next owner + concrete artifact:** Gonzo verifies the visible stale indicator and interactive playback on a real display, then owns the cost optimization using `benchmarks/hot_pour/edit-phase-profile-warm-rtx-2026-10-22.json` and `benchmarks/hot_pour/edit-phase-profile-integrated-2026-10-22.json`; rerun `tools/benchmark_fluid_edits.py` after each measured change. Retry production-scale timing only after scratch capacity exceeds the historical cache plus safe headroom.
- **Failure mode:** An initial state test caught the progress text staying at 1/N after a background frame completed; the per-frame label refresh is now covered by the ordered-enqueue test.
## 2026-10-22 — Lane 6 step B of 2 (PARTIAL)

- **What was done — evidence:** Fluid-node knobs now have look-only, keyed-simulation, or from-start classifications. The status row distinguishes look updates, a keyed edit and a from-start simulation edit. The keyed label reports the affected frame; the existing solver still starts the changed run from the beginning, so checkpoint reuse is not claimed.
- **Artifacts:** Commit `fd6dd08c` adds `nodebased/fluid_edits.py`, the UI status hook, classification tests, and a step note in `docs/FLUIDS_SPIKE.md` with its byte-identical bundled copy. Test logs are local scratch under the lane run folder; generated cache data was not created. The production-size cache was not baked because disk space is 5.7 GB versus an 8.5 GB historical cache.
- **State:** PARTIAL. Focused classification, timeline and registration tests passed on the default path, CPU and integrated-adapter harness; documentation consistency tests passed. Prefix checkpoint reuse, proxy-first/full refinement, old-cache stop behavior and the step 4 production gates remain unverified. Production-size gate is unmeasured for the disk reason above; A6000 estimate is explicitly labeled unmeasured in the docs.
- **Next owner + concrete artifact:** Needs Gonzo: implement prefix checkpoint reuse and proxy/full promotion, then add the pre-P4 cache stop test around `nodebased/fluid3d.py`/`nodebased/imaging.py`; re-run the step 4 measurements when scratch space allows.
## 2026-10-27 — Fluids 14 queued on Terra (Gonzo)

- **What was done:** Fluids 13's fire-lighting change is merged. The next plan has two bounded steps: sparse cached fire lighting without expanding the volume, followed by a 64-cell/32-vs-128-sample fire-shot comparison. This is a plan; neither result is measured yet.
- **Artifacts:** `scratch/nb-lanes/auto/briefs/L6-sparse-fire-1029.md`, `L6-flame-look-1029.md`, and `auto/state.json` are deliberate local workspace artifacts. Issue #6 and its board card carry the queue. This note and `context/state.md` are committed/pushed on main.
- **State:** Plan done; implementation held for the single GPT slot until Thursday 9:00 PM at the earliest, subject to the daily credit budget. Existing fire-shot real-display QA remains unverified.
- **Next owner + concrete artifact:** Gonzo checks the GPT budget and any competing lane at the hold time, then lets the tick launch `L6-sparse-fire-1029.md` on Terra. If another lane needs the same daily slice, DiMo chooses priority.
- **Failure mode:** The completed Fluids plan left the lane stopped with an empty backlog, causing repeated stall alerts. The new briefs, hold, and un-stopped queued state remove that empty-plan condition.
## 2026-10-30, 2:02 AM PDT — GPU ColorCorrect integration recovered; finish worker running

- **What was done:** Reproduced the stacked timing-report failure on Lane 8 and confirmed the same test passed on main. ColorCorrect now runs inside the GPU island, so the old assertion incorrectly demanded a CPU ColorCorrect timing row. Updated the benchmark assertion and report wording on the lane; the focused benchmark module and full exact-code-tip suite passed. The repaired code was merged and pushed to main. After the five-hour model window reset, launched the existing finish brief and verified its worker process is live. These are observed results; 24 fps playback is not inferred from the green tests.
- **Artifacts:** Product and test repair `b20e7158` and `644173f2`, merge note `1cbc5f50`, all committed/pushed on main. Full-suite log `scratch/nb-lanes/run/integ-gpu2d-fixed-1029.log` is deliberate local scratch (7,413 tests, exit 0). The PARTIAL report is `scratch/nb-lanes/run/report-L8.posted-1029-230912.md`; corrected original brief `scratch/nb-lanes/auto/briefs/L8-cc-1027.md`, finish brief `scratch/nb-lanes/auto/briefs/finish-L8-1029-2339.md`, and live worker log `scratch/nb-lanes/run/lane8-1030-0139.log` are workspace-local handoff artifacts. Existing EXR inputs are under `scratch/nb-lanes/run/big-L8/plates/`; verify them before reuse.
- **State:** Integration recovery done; feature step partial/in progress. Exact half-float qualification/upload, three-adapter before/after playback measurements, Radeon 24 fps gate, Windows CI, and real-display QA remain unverified. Other GPT lanes are parked while Lane 8 works.
- **Next owner + concrete artifact:** Lane 8 finish worker follows `scratch/nb-lanes/auto/briefs/finish-L8-1029-2339.md` and writes its completion report; Gonzo/integrator checks that report, the lane commit and exact-tip suite before merging. The 10-minute tick watches the running worker.
- **Failure mode:** A benchmark that times CPU kernels must stop expecting a per-node CPU row when that node moves into the resident GPU island. The first worker also searched for generated EXR plates inside its Git worktree; they are in workspace scratch. The model-credit exhaustion left the launch wake unanswered overnight until the reset.
## 2026-10-31 — Lane 8 native-half EXR decode

- **What was done:** Eligible passthrough half EXRs now decode as native OIIO half samples, widen once for evaluation, and reuse the contiguous RGBA half buffer for GPU upload. Reordered channels/alpha fill build exact half RGBA; converted, float and named-layer paths keep float decode. The upload default is enabled because all three measured tiers improved.
- **Artifacts:** Commits `a414a7ba` and `a4e758ec`; committed evidence `benchmarks/2d-gpu/R6-1/playback-nativehalf-1031.json`; raw timelines and X-server logs are deliberate scratch at `scratch/nb-lanes/run/bench-L8-native-half/`.
- **State:** Partial. Radeon native half reached 19.176–20.563 fps with p95 56.544–64.884 ms, so the 24 fps / 41.67 ms gate remains unmet. RTX and four-core rows also improved; 240-frame and Windows behaviour remain unverified.
- **Next owner + concrete artifact:** Gonzo uses `benchmarks/2d-gpu/R6-1/playback-nativehalf-1031.json` and `docs/PERFORMANCE.md` to select the next playback bottleneck.
- **Failure mode:** The earlier half path added a float32-to-float16 full-frame copy during decode, erasing its transfer saving. Native OIIO half decode removes that copy.
