## 2026-11-04 — continuous mode merge: Lane 9 (2D behaviour B, GPT-6 Luna) step Q: Flare's and Sparkles' remaining controls (finish 1) (integrator tick)

- **What was done:** `main` `e3700d8b` -> `32e237d2` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `32e237d2`: Ran 9325 tests in 590.600 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step S4: Nuke's rows and knob types on the fourth eight most-used nodes (integrator tick)

- **What was done:** `main` `d36cd6d4` -> `872c89cd` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `872c89cd`: Ran 9315 tests in 591.458 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 9 (2D behaviour B, GPT-6 Luna) step Q: Flare's and Sparkles' remaining controls (integrator tick)

- **What was done:** `main` `890114cd` -> `408cdade` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `408cdade`: Ran 9304 tests in 607.204 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 - Lane 9, step Q (Flare and Sparkles remaining controls)

- **What was done:** recorded the guide's value-unspecified Flare preset buttons and the `all` action as NOT_PIXEL reasons in `tools/nuke_fidelity_requirements.py`; regenerated `docs/PARITY_2D_FIDELITY.md` and copied it byte-identically to `nodebased/data/docs/`.
- **Evidence:** audit is unchanged: overall 95.1, Flare 79.7 (14/50 paired knobs), Sparkles 87.0 (14/32). `tests.test_nuke_fidelity_audit`, `tests.test_2d_behaviour_extra_draw`, `tests.test_knowledge`, `tests.test_knobs`, `tests.test_core`, `tests.test_tiles`, `tests.test_bypass`, `tests.test_tiers`, `tests.test_desktop.PropertiesLabelTests`, and `tests.test_desktop.FluidPropertiesPanelTests` passed.
- **Artifacts:** source and report changes are in this worktree and will be committed; test/audit logs are deliberate scratch under `scratch/nb-lanes/run/`.
- **State:** partial; no Flare/Sparkles pixel controls or cases implemented.
- **Next owner + concrete artifact:** Gonzo should take the remaining controls in the step brief and extend `tests/test_2d_behaviour_extra_draw.py` plus `tools/nuke_fidelity_cases_filter.py`.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step S3: Nuke's rows and knob types on the third eight most-used nodes (finish 1), Lane 9 (2D behaviour B, GPT-6 Luna) step P of 2: GridWarp and SplineWarp's remaining knobs, Flare, Sparkles and VolumeRays' secondary knobs, and VectorDistort's blur (finish 1) (integrator tick)

- **What was done:** `main` `ac8e1c9f` -> `83b4135a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `83b4135a`: Ran 9304 tests in 585.037 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step S3: Nuke's rows and knob types on the third eight most-used nodes (integrator tick)

- **What was done:** `main` `41c504f8` -> `cef86472` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `cef86472`: Ran 9290 tests in 583.678 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step S2: Nuke's rows and knob types on the next eight most-used nodes (integrator tick)

- **What was done:** `main` `403df01b` -> `7ef94d46` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7ef94d46`: Ran 9281 tests in 589.220 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step P of 2: the OCIO nodes', HSVTool's and Vectorfield's missing knobs, and Nuke's defaults on the time and metadata nodes (integrator tick)

- **What was done:** `main` `57403f7a` -> `86a43d6f` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `86a43d6f`: Ran 9272 tests in 591.254 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 - Lane 2 (2D behaviour), step P: the OCIO nodes', HSVTool's and Vectorfield's missing knobs, and Nuke's defaults on four time and metadata nodes

- **What was done:** (1) OCIODisplay gained `gain`, `gamma`, `saturation`, `channel_selector` and context `key1`..`value4`; OCIOColorspace gained `swap` and the context slots; OCIOLookTransform gained `reload`, `ignore_errors` and the context slots; OCIOFileTransform gained `working_space` and its `file_interpolation` is now paired with Nuke's `interpolation`; Colorspace gained `swap`, the Rec.709 camera curve, Cineon and the CIE illuminants A, B, C, D50, D55, D75 and E (`nodebased/ocioknobs.py`, `nodebased/ocio_nodes.py`). (2) HSVTool gained `HMask`, `SMask`, `BMask` (a channel of the input; a wired mask socket takes priority), their three inverts and `invert_alpha_mask`; `Conversion` pairs with `output_alpha` (`nodebased/hsvtoolknobs.py`). Vectorfield gained `reload`, `vector_file_type`, `vector_output_bitdepth` and `vector_normalize_output` (`lutio.read_3dl` takes `output_depth` and `normalize`). (3) A new AddTimeCode starts at 01:00:00:00, TimeEcho at 1 frame / max, TimeBlur at shutter 0.5 / start, ContactSheet at 3072 x 2048, 3 rows, no gap; schema 32 (`nodebased/olderdefaults.py`) gives a saved node the older value for every param it does not store.
- **Evidence:** new modules `test_2d_behaviour_ocio_p`, `test_2d_behaviour_hsv_vf_p`, `test_2d_behaviour_defaults_p`; the older OCIO, HSV, Colorspace, split-knob and timecode tests were updated for the new knobs and defaults; the targeted modules the brief names plus every module that greps for these kinds pass. Tiles of 7, 16 and 32 equal the frame with and without a mask for the OCIO nodes and HSVTool with the new knobs moved. No GPU kernel is touched, so nothing here needs `wgpu` hidden.
- **Scores (report regenerated, bundled copy identical):** overall 94.5 -> 94.7; OCIODisplay 74.7 -> 95.0, OCIOColorspace 81.4 -> 100.0, OCIOLookTransform 83.4 -> 99.3, OCIOFileTransform 87.2 -> 97.6, HSVTool 87.6 -> 96.3, Vectorfield 52.6 -> 82.8, Colorspace 85.5 -> 94.9 (math 14/20 -> 21/21), AddTimeCode 94.4 -> 100.0, TimeEcho 90.8 -> 98.7, TimeBlur 91.2 -> 97.1, ContactSheet 85.6 -> 91.7. "Knobs that change no pixels": 0.
- **Honest limits:** Colorspace's HSV, YCbCr and CIE choices stay out: the guide names the formats and gives no formula, so their cases are `not scored`. Gamma is the picture to the power of 1 / gamma and Display's saturation uses Rec. 709 luma weights (the guide states neither; marked as convention in the knob help). OpenColorIO renders `nearest` as `linear`, so Nuke does the same; `tetrahedral` and `best` differ only on a 3D LUT with coupled corners. `swap` is a checkbox that swaps the choices in the kernel, where Nuke's is a button that rewrites them. ContactSheet `startframe`/`endframe` keep 1 (the guide lists -1), BurnIn's `fontfamily` and text slots stay as before.
- **Not done:** HSVTool `huereset`, `satreset`, `brtreset` and OCIODisplay `layer`, Vectorfield `gpuExtrapolate` and `format_3dl_id` stay unpaired with reasons in the mapping. Schema bumped 31 -> 32: renumber if the other lane lands a bump first.

## 2026-11-04 — continuous mode merge: Lane 9 (2D behaviour B, GPT-6 Luna) step P of 2: GridWarp and SplineWarp's remaining knobs, Flare, Sparkles and VolumeRays' secondary knobs, and VectorDistort's blur (finish 1) (integrator tick)

- **What was done:** `main` `efe62b07` -> `0249c679` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0249c679`: Ran 9225 tests in 591.435 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step O of 2: the project settings' format and proxy, and SmartVector's remaining knobs (integrator tick)

- **What was done:** `main` `2078c314` -> `a419f980` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a419f980`: Ran 9220 tests in 593.071 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 - Lane 2 (2D behaviour), step O: SmartVector's remaining knobs (Root's primary knobs were already on main)

- **What was done:** SmartVector gained `match_strength` (Strength), `inpainting`, `matte_dilation`, `flicker_compensation`, `weight_red`, `weight_green` and `weight_blue`, each paired in `docs/parity/nodebased_to_nuke.json` and shown acting by `tests/test_2d_behaviour_smartvector_o.py` (weights: motion in one channel only; flicker: a brightness step, error against the motion with the switch on and off; strength: a noisy plate, lower is smoother; inpainting and dilation: a matte of known size). Kernel code is `nodebased/smartvectorknobs.py`, wired in the SmartVector range build in `imaging.py`. Schema 30 to 31: saved SmartVectors keep Rec. 709 weights, so their vectors are bit for bit as before (tested). SmartVector fidelity 59.3 to 94.2, knob coverage 3/11 to 10/11 (`export_write` stays unsupported: a button). Knobs that change no pixels: 0. Overall 94.5, Time group 98.0 to 98.4. Root's `format`, `proxy_type`, `proxy_format`, `proxySetting` and `Script_directory` were already paired by step N finish (Root 50.7); nothing new for Root in this step.
- **Evidence:** the new module, `test_2d_behaviour_smartvector_l`, `test_smartvector`, `test_2d_behaviour_transform_n`, `test_2d_behaviour_filter_g`, `test_nuke_fidelity_audit`, `test_knowledge`, `test_tiles`, `test_bypass`, `test_knobs`, `test_tiers`, `test_core`, `test_animation`, `test_cache_correctness`, `test_flow2b_gridwarp_tracker`, `PropertiesLabelTests` and `FluidPropertiesPanelTests` all OK; the SmartVector modules and `transform_n` also OK with `wgpu` hidden.
- **Honest limits:** a Strength above 1.5 changes the vectors but the solver does not keep getting rougher (its smoothing floors out near a smoothness of one); the guide does not say which side of the matte Matte Dilation grows, so the absent side grows; the SmartVector flow solver is the CPU Lucas-Kanade, so Strength is carried through its smoothness.
- **Not done:** Root's per-bit-depth colour spaces (`int8LUT`, `int16LUT`, `logLUT`, `floatLUT`: a project default for Read's `Auto` colour space, in `imaging.read_image`), `views` (no multi-view), `is_live`, `render_mode`; all stay listed unsupported with reasons.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step S1: Nuke's rows and knob types on the most-used nodes, and the ruler scores them, Lane 9 (2D behaviour B, GPT-6 Luna) step R2: the Read node's per-format option tabs (EXR, MOV, PSD) as Nuke shows them (integrator tick)

- **What was done:** `main` `037d7e4d` -> `9a6bf201` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `9a6bf201`: Ran 9202 tests in 585.432 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 9 (2D behaviour B, GPT-6 Luna) step R1: the Read node's Properties panel laid out like Nuke's, with Nuke's knob types (finish 1) (integrator tick)

- **What was done:** `main` `12dd3eac` -> `759efc39` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `759efc39`: Ran 9186 tests in 584.159 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step NT: every node's Node tab carries Nuke's Node-tab knobs (finish 1) (integrator tick)

- **What was done:** `main` `ed537804` -> `dc63c653` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `dc63c653`: Ran 9184 tests in 595.502 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step NT: every node's Node tab carries Nuke's Node-tab knobs, Lane 9 (2D behaviour B, GPT-6 Luna) step R1: the Read node's Properties panel laid out like Nuke's, with Nuke's knob types (integrator tick)

- **What was done:** `main` `55d95bf0` -> `299bef0e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `299bef0e`: Ran 9175 tests in 596.330 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step N of 2: Kronos, OFlow and TimeWarp get Nuke's remaining retime knobs and inputs, and the project settings' proxy and format (finish 1) (integrator tick)

- **What was done:** `main` `7631887c` -> `62b034ac` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `62b034ac`: Ran 9148 tests in 586.037 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-04 - Lane 2 (2D parity), 2D behaviour step N finish: Root's project format, proxy type and format, read-proxy-files setting and script directory

- **What was done (evidence: tests and the audit, both run):** `settings.project` gained `script_directory`, `format`, `proxy_type`, `proxy_format`
  and `read_proxy_files` (`nodebased/projectsettings.py`; additive, stored only when off its default, so no schema bump and older documents load
  unchanged, `tests/test_2d_parity_root_n.py`). Script directory: a Read's relative path resolves against it when the project directory is empty.
  Format: a Constant, Checker, Draw generator or Reformat created without a size takes the project format (`core._apply_project_format`); a
  registry rename follows and a delete clears it. Proxy type `format`: the `viewer_proxy` operation stores the tier nearest the project format's
  width over the proxy format's width (HD_1080 is the full size when no project format is set); the window combo follows the stored tier.
  `read_proxy_files` (Nuke's `proxySetting`): never, if larger, if nearest, always, decided in `readproxy.active` on the evaluator and the tile
  path (tile result equals the evaluator for each setting). Kernel changed: none beyond that decision; the default `always` is the old behaviour.
  Fidelity: Root 40.0 -> 50.7 (17 of 43 knobs paired), Other group 81.0 -> 81.5, overall 95.8 (unchanged to one decimal). "Knobs that change
  no pixels": 0. Unpaired with a reason: `is_live` (no live groups), `render_mode` (one evaluator).
- **Not done:** the project settings have no dialog in the window; the fields are set with the `settings` operation. The proxy format's
  stored tier loses the proxy-on flag when the proxy format is not smaller than the project format (tier 1). Nuke's default for proxySetting is
  if larger; NodeBased's is always, to keep old documents rendering as before.
- **Tests run:** tests.test_2d_parity_root_n, test_2d_parity_notimeblur, test_2d_behaviour_root_l, test_2d_fidelity_f_readwrite, test_proxy,
  test_nuke_fidelity_audit, test_knowledge, test_tiles, test_bypass, test_knobs, test_tiers, test_core, test_viewer_frame,
  test_new_look_viewer_strip_nl5, test_desktop.PropertiesLabelTests and FluidPropertiesPanelTests: all OK.
- **Next owner:** the integrator, to rebuild once on the stacked tree.

## 2026-11-04 — continuous mode merge: Lane 9 (2D behaviour B, GPT-6 Luna) step P of 2: GridWarp and SplineWarp's remaining knobs, Flare, Sparkles and VolumeRays' secondary knobs, and VectorDistort's blur (integrator tick)

- **What was done:** `main` `de6037a0` -> `6aa0dfa1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `6aa0dfa1`: Ran 9120 tests in 596.504 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step N of 2: Kronos, OFlow and TimeWarp get Nuke's remaining retime knobs and inputs, and the project settings' proxy and format, Lane 9 (2D behaviour B, Claude Sonnet 5.5) step O of 2: BumpBoss, Emboss, the motion blurs, EdgeExtend and Defocus get Nuke's missing knobs and defaults (finish 1) (integrator tick)

- **What was done:** `main` `b0df0d96` -> `fb67e53d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `fb67e53d`: Ran 9119 tests in 582.833 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 2 (2D parity), 2D behaviour step N (partial): Kronos, OFlow and TimeWarp retime knobs and inputs

- **What was done (evidence: tests and the audit, both run):** TimeWarp's Input length (`lookup_length`, the box filter's window width) with four
  scored cases; Kronos's Shutter Time, Automatic Shutter Time, Output, Matte Channel, Flicker Compensation, Vector Detail and Smoothness (paired,
  tested on a moving plate and a stamped plate) and its Matte, MoSrc, FgVecs and BgVecs inputs (paired, tested); OFlow's Automatic Shutter Time and
  Flicker Compensation. Kernel changed: a Kronos saved before this step keeps the output-speed-wide shutter (`legacy_shutter`, set by the load
  migration); TimeWarp's box with length 1 is the old blend, bit for bit. Fidelity: TimeWarp 82.4 -> 96.1, Kronos 65.8 -> 83.4, OFlow 82.4 -> 85.5,
  Time group 98.3 -> 98.8, overall 95.7 -> 95.8 (previous formula 95.3 unchanged). "Knobs that change no pixels": none listed (0).
  Unpaired with a reason in the mapping: Kronos `resetInputRange`, `motionEstimation`, `strength`, `resampleType`, `legacymodeNuke9`, `weightRed`,
  `weightGreen`, `weightBlue`, `vectorSpacing`, `showVectors`; OFlow `resetInputRange`, `resampleType`, the weights, `vectorSpacing`, `showVectors`.
- **Not done:** part 2 (Root: `format`, `proxy_type`, `proxy_format`/`proxySetting`, `Script_directory`, and `is_live`/`render_mode` listed unsupported);
  no schema bump was needed so far (SCHEMA_VERSION stays 29). Kronos matte/vector inputs have unit tests but no audit cases (the case harness feeds one
  input); the shutter, vector detail and smoothness cases stay "not scored" (the guide gives no numbers).
- **Next owner:** Lane 2 finish step: Root fields, with `RootPairingTests` in tests/test_2d_parity_notimeblur.py as the model.
- **Failure mode:** a Kronos saved before this step with `shutter_samples` above 1 and the motion method averages a shutter as wide as its output speed; the migration sets `legacy_shutter` so that stays.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step M of 2: new Read, Write, BurnIn, IBKColor and Keyer nodes start with Nuke's defaults, and IBKColor gets its second input (finish 1) (integrator tick)

- **What was done:** `main` `a139936d` -> `5c9b9a24` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5c9b9a24`: Ran 9070 tests in 601.391 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step O of 2: BumpBoss, Emboss, the motion blurs, EdgeExtend and Defocus get Nuke's missing knobs and defaults (integrator tick)

- **What was done:** `main` `16feae1a` -> `b121495d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b121495d`: Ran 9064 tests in 580.346 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step O finish: the inputs and Emboss knobs step O left out

- **What was done (evidence):** one code commit; new module tests.test_2d_behaviour_finish_o (30 tests). Targeted gate green: that module, the step O modules, tests.test_nuke_fidelity_audit, test_knowledge, test_tiles, test_tiers, test_bypass, test_knobs, test_core, test_cache_correctness, test_flow2_f1_finish, test_2d_parity_e1/step_5a/time_e1/step_f1b, test_2d_behaviour_filter_a/extra_filter/draw/filter_layout, tests.test_desktop.PropertiesLabelTests and FluidPropertiesPanelTests.
- **BumpBoss input B (third slot, `b`):** A's `height_channel` slope is lit onto B; B's alpha is kept; the node's mix dissolves toward B; the output has A's window with B fitted to it. B unconnected is A lit by its own slope (the one-input node as before); a saved node (`legacy_bumpboss`) ignores B. Tiles equal the frame with and without a mask (`tiers._bump_boss_rule` reads B where A is read).
- **EdgeExtend Matte input (third slot, `matte`):** the four `Matte ...` choices read its luminance (Rec. 709) and alpha and their inversions; unconnected it reads as black (convention, the guide is silent). The Source choices ignore the input. Saved nodes unchanged. Tiles equal the frame (`tiers._edge_extend_rule` pads the matte like the image).
- **MotionBlur2D 2D transf input (third slot, `transf`):** with a Transform or CornerPin wired there the picture is read once at the frame and averaged along that node's motion over the shutter (`Evaluator._transf_matrix`, `_transf_blur`: the content at each shutter sample sits where the transform puts it then, `sample @ inverse(current)`); any other node is refused. The guide's page describes only the shutter, so this reading is NodeBased's own. Without the wire the node samples its own input over time as before. Tracker and Stabilize are not accepted (their motion comes from track data).
- **Emboss:** Edge Channels (`edge_channels`: the brightness the slope is read from is the mean of those channels; rgb as before, none gives no relief), Blur size and Blur quality (`blur_size`, `blur_quality`: the EdgeDetect pre-blur before the slope is read). Defaults reproduce a saved node bit for bit. `blurquality` is NOT_PIXEL on the fixture (acts once the blur passes it), moved at size 12 in the new module. Threshold, Output and Erode size stay unsupported (the page gives them no description).
- **Still unsupported, reasons recorded in docs/parity/nodebased_to_nuke.json:** MotionBlur2D and MotionBlur3D Output UV, MotionBlur3D Z and distance (the page describes only the shutter, so nothing can be tested against it); Emboss Threshold, Output, Erode size.
- **Audit:** overall 95.7 (the page rounds the same); Filter group 96.6 -> 96.7. EdgeExtend 95.7 -> 100.0 (inputs 50 -> 100), BumpBoss 96.3 -> 100.0 (inputs 67 -> 100), MotionBlur2D 79.2 -> 87.5 (inputs 50 -> 100), Emboss 78.8 -> 84.8 (knobs 4/13 -> 7/13). "Knobs that change no pixels": 0 (375 of 375).

## 2026-11-03 - Lane 9 (2D behaviour), step O: Emboss, BumpBoss, EdgeExtend and Defocus get Nuke's missing knobs; the motion blurs get Nuke's defaults

- **What was done (evidence):** two code commits; new modules tests.test_2d_behaviour_lighting_o (23 tests) and tests.test_2d_behaviour_edge_defocus_o (20 tests); older modules whose expectations named the old defaults updated (test_2d_behaviour_filter_a, _extra_filter, _draw, _filter_layout, test_2d_parity_group_3b, test_tiers). Targeted gate green: those plus tests.test_nuke_fidelity_audit, test_knowledge, test_tiles, test_tiers, test_bypass, test_knobs, test_core, test_cache_correctness, test_node_rail, test_flow2_f1_finish, test_2d_parity_e1/step_5a/time_e1, the whole of test_desktop (210 tests, PropertiesLabelTests and FluidPropertiesPanelTests included).
- **Emboss (`nodebased/lightingbehaviour.py`):** Emboss Type (`emboss_type`: traditional adds the relief to the picture, effect is the relief about mid grey) and Edge Detector (`edge_detector`: simple, sobel, prewitt) are paired; angle now defaults to 0. Saved nodes read `effect` and `simple` (bit for bit). Edge channels, Threshold, Output, Erode size, Blur size and Blur quality are listed unsupported with a reason (the page gives them no effect on the relief).
- **BumpBoss:** the six guide knobs are paired (`lightposition_x/y`, `bumpsize`, `light_height` (renamed from `height` so the proxy-tier table does not read it as a pixel size), `intensity`, `minshadow`, `center_x/y`). The light leans from center toward the light position (Nuke's y runs up); light height 0 gives no emboss, negative flips it; picture x (1 + intensity x relief), floored at minshadow. The guide has no formula, so this is NodeBased's reading (the existing bumpboss case stays unscored). Saved nodes keep the one-input grey emboss (`legacy_bumpboss` 1). Input B is not built (NodeBased lights the one picture).
- **EdgeExtend (`nodebased/edgedefocusknobs.py`):** Matte (None, Source Alpha, Source Inverted Alpha; the four choices that read the Matte input are not offered), Detail Amount (0 leaves the 3 x 3 smooth colour, 1 the colour as it was) and Edge Mask (the filled-in edge in alpha). Saved nodes keep their picture. The Matte input is not built.
- **Defocus:** Quality smooths the disc's edge (each edge pixel carries the share the disc covers, found on a round(quality / 5) wide sample grid; 1 to 5 is the hard disc), Method is paired and changes no pixels (NOT_PIXEL: the guide puts the difference between accelerated and fullprecision below 0.00001). Saved nodes keep the hard disc (`legacy_defocus_disc` 1). The five exact-disc cases now run at quality 1. No schema bump.
- **MotionBlur2D / MotionBlur3D:** new nodes start at shutter 0.5 and shutter offset start (the guide); saved nodes keep their values (tested). Output UV, Z input and distance are listed unsupported with reasons (the page describes only the shutter). The 2D transf input and the MotionBlur3D Z input are not built.
- **Audit:** overall 95.6 -> 95.7 (previous formula 95.1 -> 95.2); Filter group 95.8 -> 96.6. BumpBoss 44.4 -> 96.3 (knobs 0/6 -> 6/6), Emboss 71.1 -> 78.8 (2/13 -> 4/13), EdgeExtend 82.6 -> 95.7 (3/6 -> 6/6), Defocus 88.6 -> 99.2 (3/5 -> 5/5), MotionBlur2D 68.1 -> 79.2 (defaults 33 -> 100), MotionBlur3D 63.9 -> 75.0 (defaults 33 -> 100). "Knobs that change no pixels": 0 (373 of 373 probed knobs act).
- **Not done:** BumpBoss input B; EdgeExtend Matte input and its four Matte choices; MotionBlur2D 2D transf input; MotionBlur Output UV, Z and distance; Emboss edge channels, threshold, output, erode and blur knobs; MotionBlur inputs and the knobs that act across frames have no multi-frame test here.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step N of 2: VectorDistort, PointsTo3D and Reconcile3D get Nuke's inputs and main knobs (integrator tick)

- **What was done:** `main` `7121124f` -> `709ddee2` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `709ddee2`: Ran 9021 tests in 588.405 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step N: VectorDistort, PointsTo3D and Reconcile3D get Nuke's inputs and main knobs

- **What was done (evidence):** two code commits; new module tests.test_2d_behaviour_transform_n (34 tests: a Noise plate that moves three whole pixels with a real SmartVector on it; a Camera3D sliding along x with a known 3D point). Targeted gate: tests.test_track3d, test_nuke_fidelity_audit, test_knowledge, test_bypass, test_knobs, test_core, test_tiers, test_tiles, test_smartvector, test_knob_sliders, test_2d_behaviour_filter_k, test_desktop.PropertiesLabelTests and FluidPropertiesPanelTests.
- **VectorDistort (`nodebased/vectordistortknobs.py`, flag `legacy_vectordistort`, saved nodes 1):** Hold Frame (on warps the reference frame; off carries the current frame back to the reference with the vectors reversed), Output (warped src, st-map, st-map inverse; a map takes no mask, mix or fade), Filter (the Transform group's list, Cubic by default; saved nodes keep bilinear), Channels (the shared strip's `restrict`, applied in the node's own branch). Frame Distance is listed `unsupported`: Nuke's SmartVector keeps vector sets for 1 to 64 frame hops and NodeBased's keeps one accumulated set per frame.
- **PointsTo3D (`nodebased/track3dknobs.py`):** `img` input (its display window is the plate the points are read against), Camera type (free move, the old triangulation, or nodal: Point A one focal length in front of the camera, B and C ignored; the unit is our reading, millimetres as scene units), point3D and point2D listed `panel` (read from `Evaluator.track3d_diagnostics[key]`: `position`, `point2d`). No schema bump (a missing `camera_mov_type` loads as free move through `transformbehaviour.UPGRADE_EXTRA`).
- **Reconcile3D:** `img` input (output resolution), `track`, `offset`, `stabilize` (the 2D transform as a per-frame translation `(projected - track) + offset`, reversed with stabilize, in the value's `transform` and each sample's `translate_x`/`translate_y`; the projected XY output does not change), XY output and XYZ output listed `panel` (`output`, `outputxyz` on the value), calculate output live listed `panel` (always live). Seven picture-resampling knobs (filter, clamp, black outside, motionblur, shutter, shutter offset, custom offset) are listed `unsupported`: this node outputs a track, not the moved picture.
- **Audit:** overall 94.8 -> 95.0 (previous formula 94.3 -> 94.5); Transform group 95.1 -> 96.6; VectorDistort 57.1 -> 90.5, PointsTo3D 55.0 -> 87.5, Reconcile3D 41.3 -> 76.7. "Knobs that change no pixels" stays 0.
- **Not done:** VectorDistort Blur Size still blurs the warped picture (the guide blurs the internal STMap); Frame Distance; GPU path for non-bilinear filters (the CPU path runs them); the Reconcile3D moved picture.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step M of 2: new Read, Write, BurnIn, IBKColor and Keyer nodes start with Nuke's defaults, and IBKColor gets its second input (integrator tick)

- **What was done:** `main` `8e40acae` -> `f9fbed7b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `f9fbed7b`: Ran 8987 tests in 595.623 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step M of 2: ZDefocus gets Nuke's main tab: focal point, depth of field, layers and output modes (integrator tick)

- **What was done:** `main` `a0488e12` -> `ef146d03` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `ef146d03`: Ran 8972 tests in 606.182 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step M: ZDefocus gets Nuke's main tab

- **What was done (evidence):** one code commit plus one for the filter knobs; targeted gate green: tests.test_2d_behaviour_zdefocus_m (35, new), _draw, _filter_k, test_2d_parity_step_f2, test_tile_layers, test_tiers, test_nuke_fidelity_audit, test_knowledge, test_tiles, test_bypass, test_knobs, test_core, test_desktop.PropertiesLabelTests and FluidPropertiesPanelTests, test_nodecatalog, test_cache_correctness (test_directviewer_hit_order, 245 s, green on the first commit).
- **ZDefocus (`nodebased/zdefocusknobs.py`, flag `legacy_zdefocus_focus`, saved nodes 1):** output (result, focal plane setup, layer setup, filter shape setup), show image, fill foreground, focal point (`focal_x`, `focal_y`), blur inside, automatic layer spacing, depth layers, layer curve; and the secondary knobs shape, aspect, catadioptric and its size, bloom, threshold, gain, gamma correction. Blur follows size * |d - edge of the depth-of-field slice| / d, clipped at maximum; the layers are the blur sizes between 0 and maximum.
- **Kernels changed:** `Evaluator._zdefocus` (evaluator and tile path share it); a focal point or the filter shape view needs the whole frame (`zdefocusknobs.needs_full_frame`, checked in `TileExecutor.supports_tiled`). Graphs saved before keep their picture (tested).
- **Not built, listed in the report:** output red / green / blue (the guide does not say what they show), roundness, inner size, inner feather, inner brightness, legacy resize mode, filter channel, use input channels, filter bounds, image filter, clamp image filter, GPU options, `aspect#2` (one `aspect` serves the disc and the bladed filter).
- **Report:** ZDefocus 74.8 -> 91.2 (knobs 9/36 -> 25/36); Filter group 94.9 -> 95.8; overall 94.4 -> 94.4; Knobs that change no pixels 0 -> 0.
- **Next owner:** Gonzo: decide on the bladed filter's roundness / inner polygon (no shape in the guide) and the image-filter resize knobs.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step L of 2: Write's secondary knobs, the project settings' main knobs, and SmartVector's outputs (integrator tick)

- **What was done:** `main` `3334d42c` -> `dc48cb50` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `dc48cb50`: Ran 8937 tests in 578.163 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - lane 2 (2D behaviour), step L of 2: Write's last knobs, the project settings on Root, and SmartVector's matte

- **What was done (evidence):** two code commits plus a docs commit; targeted gate green (the new Root and SmartVector modules, audit, knowledge, tiles, bypass, knobs, tiers, core, the properties-panel tests, test_smartvector, the Write and Read modules, the sequence-range offer tests, cache correctness).
- **Write:** the 29 knobs the report called missing are all the Metadata Export tab (MOV codec colour tags and MXF container knobs). Write makes EXR, PNG, TIFF, JPEG and DPX, so `tools/build_nuke_reference.py` files the tab with the other per-format tabs and `docs/parity/nuke_nodes.json` is patched to match (counted false). Every other Write knob was already paired and tested (`tests/test_2d_fidelity_c_write.py`). Write 70.9 -> 90.3, knobs 27/56 -> 27/27.
- **Root:** `settings.project` (`nodebased/projectsettings.py`), optional like `formats`: `name`, `label` (window title), `project_directory` (resolves a Read's relative path on create and on set), `lock_connections` (a `connect` is refused), `lock_range` (`offer_sequence_range` leaves the range). Root 30.0 -> 40.0, knobs 7/43 -> 12/43. Unsupported with reasons: Script_directory, format, proxy_type, proxy_format, proxySetting, is_live, render_mode and the colour, views and font knobs listed before.
- **SmartVector:** Matte input, `matte_channel`, `matte_output` (`nodebased/smartvectormatte.py`, evaluator in `imaging.py`): flow is estimated on frames with the other side blacked out; the side not output has zero vectors and coverage 0. SmartVector 17.1 -> 62.1, knobs 1/11 -> 3/11, sockets 50 -> 100. Listed unsupported with a reason: strengthReg, inpainting, matteDilation, export_write, flickerCompensation, weightRed/Green/Blue.
- **Legacy and schema:** no schema bump (the project group is optional; the SmartVector params and `matte` slot are filled on load). A SmartVector saved before keeps Matte Channel None and the same vectors (test).
- **Report:** overall 93.8 -> 94.7; "Knobs that change no pixels" reads 0 of 361 probed knobs, as it did at the head of this branch before the step (the brief named 13 for main at 69aa0d36); under the previous formula 94.2.
- **Not done:** Project Settings dialog fields for the Root group (the `settings` op and the agent reach them); the matte estimate is a stated reading of the guide (hard cut at 0.5), no Nuke render compared; inpainting not built.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step L of 2: Denoise, MotionBlur and VectorBlur get Nuke's main knobs and inputs (finish 1) (integrator tick)

- **What was done:** `main` `3d05b2d2` -> `5474baa4` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5474baa4`: Ran 8922 tests in 611.245 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step L finish: the rest of Denoise, MotionBlur and VectorBlur

- **What was done (evidence):** one code commit (`21a6d677`); targeted gate green: tests.test_2d_behaviour_filter_k, _l (35), _m (32, new), _n (36, new), _a, _f, _g, _layout, extra_filter, extra_draw, image, retime_vectors, 2d_parity_notimeblur, split_knobs, 2d_parity_step_5c, step_f2, time_d2, flow2_f1, flow2_f1_finish, flow_node_accuracy, flow_nodes_v1, nuke_fidelity_audit, knowledge, tiles, knobs, tiers, core, nodecatalog, 2d_fidelity_e_filters, h_filters, 2d_parity_group_3a_filters, cache_correctness, bypass, desktop PropertiesLabelTests and FluidPropertiesPanelTests. Each module was run alone.
  - MotionBlur (`nodebased/motionblurknobs.py`, flag `legacy_motionblur`, saved nodes 1): Method (Local or Regularized), Vector Detail (a vector every round(1/detail) pixels), Resampling (Bilinear, Lanczos4, Lanczos6 through the vector blur), Matte Channel with the Matte input, and BgVecs (vector = matte x FgVecs + (1 - matte) x BgVecs, exact on constants). With estimated vectors the matte keeps foreground and background vectors from averaging across its edge (measured: the vector error beside a 32 px square moving 3 px falls from 0.97 to 0.42).
  - VectorBlur (`nodebased/vectorblurknobs.py`): UV offset, Blur UV (none, uniform 3 x 3, linear 9 x 9 tent), Blur type (uniform or gaussian) with Motion falloff, Expand blur bbox (the window grows, the ring takes the edge vectors), Soft lines (three pixels wide) and Blur inside (3 x 3 where nothing moves). The param is `vector_blur_type` because `blur_type` is DirBlur's shared choice list. New nodes read the guide's gaussian default; saved nodes get `uniform` and read as before. `tests/test_2d_parity_step_5c.py` now names `vector_blur_type="uniform"` where it asserts equal weights.
  - Denoise (`nodebased/denoiseknobs.py`, flag `legacy_denoise` from step L): Tune Channels (Luminance and Chrominance Gain scale the threshold, defaults 1 and 5), Tune Frequencies (four bands of the removed noise, each switch and gain; each band comes from a window average at its own gain; with all bands on at 1 the whole noise, exactly), the Profile tab (curve, Tune Profile and the three gains, Profile Frame feeding the Automatic profile), Noise Analysis (box x y r t counted from the bottom, 80 x 80 minimum, Analysis Frame, Lock, held level and profile, Analyze Noise, Export, Import: three buttons on the panel), Temporal Processing (Frames to Blend and Frame Blending; neighbours moved onto the frame along the Motion input or along vectors estimated between the frames) and the `Motion` input.
- **Kernels changed:** `Evaluator._vector_blur` (resample filter, line profile, soft lines, blur inside; uniform and bilinear are bit for bit what they were), the uv-filter branch (UV offset, Blur UV, grow), the MotionBlur FgVecs and estimation branches and `_motion_flow`, `Evaluator._denoise` (takes `refs`), the Denoise pre-evaluation of other frames (roles neighbour, analysis, profile) and its temporal branch. No schema bump (the new keys come from SPECS defaults and `filterbehaviour.UPGRADE_EXTRA`).
- **Unsupported, with the reason in `docs/parity/nodebased_to_nuke.json`:** VectorBlur `mv_presets` (the guide gives no scale and offset values for the renderer presets) and `normalize` (it compensates for tearing and collisions of a scatter; NodeBased gathers along each vector, so a real version needs a scatter kernel for VectorBlur).
- **Report:** Denoise 75.5 -> 99.6 (knobs 11/37 -> 37/37); MotionBlur 41.7 -> 93.3 (6/6); VectorBlur 70.9 -> 90.1 (13/15); Filter group 94.9 -> 95.6; overall 94.0 -> 94.4; Knobs that change no pixels 0.
- **Next owner:** Gonzo, to decide whether VectorBlur gets a scatter kernel (for `normalize`) and where the renderer preset values come from.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step K of 2: IBKGizmo, MatchGrade and HueCorrect get Nuke's missing main knobs, and Nuke's layout on the time and metadata panels (finish 1) (integrator tick)

- **What was done:** `main` `ee9fbbd1` -> `0a51e76b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0a51e76b`: Ran 8854 tests in 590.085 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step L of 2: Denoise, MotionBlur and VectorBlur get Nuke's main knobs and inputs (integrator tick)

- **What was done:** `main` `120e8d02` -> `7cb91003` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7cb91003`: Ran 8822 tests in 581.385 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step L of 2: Denoise, MotionBlur and VectorBlur get Nuke's main knobs and inputs

- **What was done (evidence):** two code commits; targeted gate green (tests.test_2d_behaviour_filter_l 35 tests, filter_k, 2d_parity_step_5c, flow_node_accuracy, flow_nodes_v1, flow2_f1, flow2_f1_finish, cache_correctness, nuke_fidelity_audit, knowledge, tiles, bypass, knobs, tiers, core, desktop PropertiesLabelTests and FluidPropertiesPanelTests).
  - Denoise (new `nodebased/denoiseknobs.py`, flag `legacy_denoise`, saved nodes 1): Source, Noise model, Profile, Output, Lift blacks, Roll off, Smoothness, Luminance blend, Preserve edges and the `Noise` input. Output Noise is the source minus the result exactly. A higher amount lowers the residual variance of a plate with known noise. The evaluator and the tile path agree at tiles of 7, 16 and 32, with and without a mask; a node that measures a noise level (Automatic profile, or Noise wired) runs on the full frame. Saved graphs keep the bilateral of before bit for bit.
  - VectorBlur: Invert UV and Output (result or uvs) added; Scale, Offset, Method and UV alpha already acted and now each have a constant-vector test (streak = vector x scale).
  - MotionBlur: the `fgvecs` input (FgVecs) streaks the plate over Shutter Time at Shutter Samples points, centred, starting or ending at the frame; tested on a ramp moving a whole number of pixels and on a lone bright pixel (five pixels at a fifth each).
- **Kernels changed:** `Evaluator._denoise` (new, `denoiseknobs.denoise`; the temporal path and the tile path call it); the VectorBlur branch of the uv filter; a MotionBlur branch for FgVecs. No schema bump.
- **Left out, one line each:** Denoise: Motion input and temporal blending knobs (frame_blending, frames), the Noise Analysis tab, Tune Frequencies, Tune Channels and the Profile tab (all need Foundry's trained model or a profile curve). MotionBlur: Method, Vector Detail, Resampling (Lanczos), Matte Channel, BgVecs and Matte inputs (they act on motion estimation from neighbouring frames); the shutter defaults stay 1.0 and 4 (Nuke 0.75 and 3). VectorBlur: mv presets, blur uv, uv offset, blur type, motion falloff, expand blur bbox, soft lines, blur inside, normalize.
- **Report:** Denoise 33.7 -> 75.5 (knobs 2/37 -> 11/37, math 100 on 3 scored cases); MotionBlur 37.5 -> 41.7 (sockets 25 -> 50); VectorBlur 65.0 -> 70.9 (knobs 4/15 -> 6/15); Filter group 94.4 -> 94.9; overall 94.0 -> 94.0 (previous formula 93.4 -> 93.5); Knobs that change no pixels 0 of 358 (13 was the count before step K).

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step K of 2: IBKGizmo, MatchGrade and HueCorrect get Nuke's missing main knobs, and Nuke's layout on the time and metadata panels (integrator tick)

- **What was done:** `main` `c198ea8a` -> `a6cf9dc7` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `a6cf9dc7`: Ran 8787 tests in 587.057 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 2 (2D behaviour), step K of 2: IBKGizmo, MatchGrade, HueCorrect and GenerateLUT get Nuke's missing main knobs; the time and metadata panels follow Nuke's order

- **What was done (evidence):** three code commits, targeted gate green (the new `tests/test_2d_behaviour_knobs_k.py`, 29 tests, plus every module that greps for the changed kinds, audit, knowledge, tiles, bypass, knobs, tiers, core, split knobs and the properties-panel label tests).
- **Knobs added:** IBKGizmo `color` (screen type pick), `screen_range` (level), `luminance_level_enable` (ll_enable), `yellow`, `cyan`, `magenta`, `use_bg_chroma` (ubcr); `autolevels` unpaired (whole-frame statistics a tile cannot see). HueCorrect `mix_luminance`. MatchGrade `match_task`, `match_analysis`, `grade_mask`, `apply_to_mask`, `match_output`; `alignTargetToSource` and `autoCrop` unpaired (they build nodes upstream). GenerateLUT `lut_file_type` (cube or 3dl, with a new .3dl writer), `prelut_size`, `prelut_style`, `lut_input_bit_depth`, `lut_output_bit_depth`, `lut_clamp_output`.
- **Guide status:** the guide names the IBKGizmo knobs and says nothing about what they compute, so they are NodeBased's reading of the names, each checked as a property of the pixels, and the cases stay not scored. MatchGrade's Mask, Apply Grade to Masked Region and Output are described in the guide and are implemented as it states.
- **Legacy and schema:** no bump. `behaviour2d`: IBKGizmo `luminance_level_enable` (older documents 1, new 0), GenerateLUT `lut_clamp_output` (older 0, new 1). MatchGrade, HueCorrect, the other IBKGizmo knobs and the layouts default to the earlier pixels.
- **Audit before and after:** IBKGizmo 76.4 -> 95.7, MatchGrade 44.5 -> 66.2, HueCorrect 82.4 -> 100.0, GenerateLUT 30.9 -> 82.0; AddTimeCode 91.1 -> 94.4, BurnIn 84.6 -> 88.3, CurveTool 98.1 -> 100, AppendClip 97.6 -> 100, TimeBlur 92.6 -> 94.1, ContactSheet 88.6 -> 90.9, Precomp 97.1 -> 100. Color group 95.6 -> 96.6, Keyer 91.6 -> 92.2, Time 97.7 -> 97.8, overall 93.8 -> 94.0 (previous formula 93.4 -> 93.6). "Knobs that change no pixels": 13 on main, 13 now (new inert knobs are listed under not-probed with reasons).
- **Not done:** GenerateLUT `format_3dl_id` and the Generate button; MatchGrade's Ref Frames, Transform and CDL knobs; IBKGizmo `autolevels`.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step K of 2: the thirteen knobs that change nothing, and Nuke's labels and order on the filter and transform panels (integrator tick)

- **What was done:** `main` `69aa0d36` -> `530df4a7` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `530df4a7`: Ran 8758 tests in 590.295 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step K of 2: the thirteen knobs that change nothing, and Nuke's labels and order

- **What was done (evidence):** three code commits plus a report commit; targeted gate green (tests.test_2d_behaviour_filter_k 17 tests, filter_f, draw, transform, lens_j, filter_layout, retime_vectors, group_3b, step_5a, tiers, audit, knowledge, tiles, bypass, knobs, core, the properties-label and fluid-panel desktop tests).
  - LightWrap (new flag `legacy_lightwrap_luma`, saved nodes 1): Disable luminance based wrap divides the background light by its luminance (one brightness whatever the background); Enable Glow drops the matte factor so the wrap extends outside the alpha; LumaTolerance fades background darker than it out of the wrap; midtone bias (`gray`) is the contrast pivot and `white` the gamma fixed point of the CCorrect tab (run in ColorCorrect's order). Guide-silent details are stated in the code comments.
  - DirBlur (existing `legacy_dirblur_knobs`): Quality high/normal/low scales the picture down by 1/2/4, blurs and scales back up (new nodes default to high, Nuke's 1); PixelAspect widens the pixel (linear direction, radial circle); Holdout on keeps pixels whose alpha is not black out of a zoom blur's average. Quality scaling and a non-square pixel read the whole frame like zoom and radial (not tiled).
  - Dither `static_seed`, Soften and Laplacian `quailty`, EdgeDetect `blurquality`, Laplacian `crop`, DirBlur `holdout`: listed in NOT_PIXEL with reasons and tests (static seed across frames; quality caps the size only past it; crop to format is the box).
  - Labels and order: ZDefocus, Denoise, DegrainSimple, MotionBlur, VectorBlur, ZSlice, CornerPin, STMap, IDistort, PointsTo3D, Text now read the guide's labels and order (saved param names unchanged; Text puts the Draw chrome first, per the guide table; PointsTo3D shows each point's ref time next to the point).
- **Report:** overall 93.8 -> 94.0; Knobs that change no pixels 13 -> 0; Draw 91.4 -> 92.1, Filter 93.5 -> 94.4, Transform 94.8 -> 95.1. No schema bump.
- **Not done:** CurveTool, AppendClip, TimeBlur, ContactSheet, ZMerge, AddTimeCode, BurnIn and Precomp still show layout differences (not in this step's list); Holdout channel is the alpha only (a choosable channel needs a per-node holdout choice list, since `holdout` is a shared on/off key).

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step J of 2: Kronos and OFlow retime with Nuke's timing knobs, Lane 9 (2D behaviour B, Claude Sonnet 5.5) step J of 2: LensDistortion's lens models, checked against a stated distortion (integrator tick)

- **What was done:** `main` `de38d3b9` -> `e054f5cd` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e054f5cd`: Ran 8741 tests in 578.171 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step J of 2: LensDistortion's lens models checked against a stated distortion

- **What was done (evidence):** one code commit plus a docs commit, targeted gate green (lens modules, audit, knowledge, tiles, bypass, knobs, tiers, core, split knobs, nodecatalog, the properties-panel tests). Undistort and Redistort read Nuke's Model Preset: NukeX Classic (guide: xu = xd / (1 + k0 rd^2 + k1 rd^4), k0 and k1 the two Denominator s) and Custom (adds Numerator s and the Direction; Backward is used directly, Forward is inverted), Lens Type (Anamorphic reads the squeeze, Spherical ignores it), Centre, Mode (Undistort, Redistort, STMap into `vector.forward` and `vector.backward`), Filter (default now cubic), BBox (Auto, Input, Manual), Output Format and Distortion Scaling. The math is `lensmodel.RationalLens`, which shares the squeeze, centre, window growth and sampling code of the Brown-Conrady lens. Units: r in half the longer side (the guide gives no factor); Centre in pixels from the format middle, y up (also a reading).
- **Cases:** LensDistortion 0 -> 8 (7 scored pass, 1 unscored: the numerator's form is read from the Order and Exponent text). `tests/test_2d_behaviour_lens_j.py` (32 tests) checks the equation on the node's own STMap output, so no inverse runs in the test: grid-point radius, a named point, redistort, round trip, centre fixed point, squeeze x only, backward, numerator, every paired knob acting, filters, bbox, output format, scaling, layout order.
- **Audit before and after:** LensDistortion 21.0 -> 77.6, knobs 1/58 -> 20/58, math n/a -> 100 (7/7), acts 100, layout 100. Transform group 93.9 -> 94.8 (rank-weighted), overall 93.5 -> 93.6; previous formula 93.2 -> 93.3. Behaviour checked 82 -> 83 nodes.
- **Legacy and schema:** no bump. `lensdistortion.upgrade_params` gives a saved Undistort or Redistort the preset `Brown-Conrady (NodeBased)` (the `lens_*` knobs and `lens_preset` as before) and every other new knob at the value that changes nothing; a parameter dict without the new knobs reads as Brown-Conrady, so bit-for-bit older pixels are tested. Four places in three existing lens test files that build a node with `lens_k1` now name that preset (test_lens_l1, test_gpulens, test_camera_solve_lens).
- **Kernel changed:** `Evaluator._lens_warp` (window and display from BBox, Output Format, Distortion Scaling; STMap layers); the GPU card kernel stays Brown-Conrady only (`lens_backend` gpu says so for the rational model; auto and cpu run on the CPU).
- **Left out, one line each:** Analysis tab (detect, numFeatures, patchSize, featureSeparation, thresholds, solveDistortion, reject, reset, errorThreshold, overlay) needs grid detection and a solver; CaraVR Radial Degree 3 and 3DEqualizer models (named, equations not stated); Twist and Scale (anamorphic); Fisheye projection, focal, sensor size, distort-in-fisheye-space and Use Fisheye; Beam Splitter; Normalisation (Maximum only), Order and Exponent beyond the default; the three Python callback knobs and `renderProgress`; the GPU knobs (`useGPUIfAvailable`, `gpuName`); the Advanced-tab Link copies are paired as panel entries.
- **Not done:** no real Nuke render; the Centre unit and the radius unit are readings of the guide; the Tcl strings in the cases' `nuke_knobs` are untested against Nuke.

## 2026-11-03 - lane 2 (2D behaviour), step J of 2: Kronos and OFlow retime with Nuke's timing knobs

- **What was done (evidence):** (1) New `nodebased/retimeknobs.py` holds the frame arithmetic both retimers share; `Evaluator` calls it for Kronos and OFlow. Kronos gains Input Range (`input_start`, `input_end`), Channels (`retimed_channels`), Timing (`timing`: speed, input_speed, frame) and Input Speed (`input_speed`); the Method knob (`interpolation`, `method` on OFlow) now means what the guide says: frame is the nearest original frame, the new `blend` is the mix of the two neighbours, motion is the vector interpolation. OFlow gains Channels, Input Speed and the Method knob. Both panels follow the guide's knob order and labels. (2) `tools/nuke_fidelity_cases_time.py`: 30 scored and 9 not scored cases per node (guide-quoted frame, blend, output and input speed, keyed Frame, input range, retimed channels, a Noise plate moving 4 pixels a frame whose half-way frame must show 2, tolerance 0.01). `Case` gained a `tolerance` field and `_run_stamped_document` gained `time`, `green_scale`, `plate`, `size` and per-step curves. (3) `tests/test_2d_behaviour_retime_vectors.py`, 28 tests. (4) Mapping, report (`docs/PARITY_2D_FIDELITY.md` and the bundled copy) and `docs/PARITY_2D.md` updated.
- **Kernels changed (legacy flag):** Kronos `legacy_retime` (set to 1 on load for every saved Kronos: the timeline frame times the speed counted from zero, `frame` of 0 or more wins, project range only, `interpolation` "frame" blended). A new Kronos counts output speed from `input_start` (`input_start + (f - input_start) * speed`), limits the frame to the input range and the project range, and `frame` shows the nearest original frame. OFlow's frame mapping did not change (no flag); its new knobs default to what it always did. `tests/test_flow_node_accuracy.py` asserted the old Kronos mapping (speed 0.5 at output frame 3) on a fresh node; it now asks output frame 2.
- **Audit before and after (fidelity, math cases passing/total):** Kronos 33.1 (-) -> 65.8 (30/30); OFlow 61.4 (-) -> 82.4 (30/30); SmartVector 17.1 -> 17.1 (untouched). Time group 97.1 -> 97.7 (Time rank-weighted), overall 93.6 -> 93.7, under the previous formula 93.3 -> 93.3. Nodes with behaviour checked 91 -> 93 (the report's count).
- **Not done / open in the Time group:** keyed Output Speed and Input Speed do not accumulate (the speed is read at the current frame); negative output speed has no anchor in the guide; frames outside the input range hold (our reading); vector detail, smoothness, strength, occlusions, resampling, shutter time and samples are not scored; Kronos's Matte, FgVecs, BgVecs and MoSrc inputs, `output`, `matteChannel`, `strength`, `motionEstimation`, `flickerCompensation`, weights and `showVectors` are not built; Reset buttons missing. SmartVector's missing primary knobs: `strengthReg` (matching strength of the estimate), `matteChannel` (where the foreground matte comes from), `output` (foreground or background vectors), `inpainting` (infill absent vectors), `matteDilation` (dilate the matte before infilling); `matteChannel` and `output` need the Matte socket and a masked flow estimate, so they are not a plain reuse and are not built. The Time group still has no pixel truth against real Nuke renders.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step I of 2: the retime and frame nodes pick the frames Nuke picks (integrator tick)

- **What was done:** `main` `9b8dfa7e` -> `47890484` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `47890484`: Ran 8681 tests in 692.901 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - lane 2 (2D behaviour), step I of 2: the retime and frame nodes pick the frames Nuke picks

- **What was done (evidence):** (1) `tools/nuke_fidelity_cases_time.py` holds 165 cases for TimeOffset, FrameHold, Retime, FrameRange, TimeClip, AppendClip, TimeEcho, TimeBlur and TimeWarp; 146 are scored and 19 not scored with the reason. They run through a new `_run_stamped_document` in `tools/nuke_fidelity_cases.py`: a Constant whose red channel is the frame number (a linear curve keyed on every frame from -90 to 100), so the red value of the picture is the source frame the node picked, and a fractional frame reads back as itself. The module is registered in the case hook. `source` is `guide` where a sentence or example of the node's page states the frame (the statement quotes it) and `convention` where only the form is long-established. (2) Three kernels disagreed with the guide and were fixed on the evaluator (TimeBlur and TimeEcho have no kernel of their own on the tile path: `_temporal_tile` asks the evaluator, so tiles of 7, 16 and 32 equal the full frame, tested): TimeBlur's shutter window, TimeEcho's missing fade, FrameRange cutting fractional frames. TimeWarp gained the `nearest` filter. (3) `tests/test_2d_behaviour_time.py` (24 tests): case statuses, the guide's worked examples, older documents bit for bit, every paired knob moving the frame picked, tiles, registration.
- **Kernels changed (legacy flag):** TimeBlur `legacy_shutter_window` (start was the frame less the shutter to the frame, end the frame to the frame plus the shutter, custom centred on the offset; the guide opens at the frame plus the offset: start frame to frame + shutter, end frame - shutter to frame, custom frame + offset onward; same helper as Transform). TimeEcho `frames_fade` (Nuke's `framesfade`; the oldest n frames fade k/n, the rest at full; older nodes load 1, which fades nothing). FrameRange `legacy_fractional_frames` (an older FrameRange cuts every frame to a whole one). All three set by `upgrade_nodes` through `timeknobs.OLD`; no schema bump (the migration uses setdefault, as lanes 9 and 2 did before). TimeWarp's `lookup_filter` choices are now none, blend, nearest. The TimeEcho panel follows Nuke's order and labels (TimeEcho Method, Frames to look at, Frames to fade out).
- **Audit before and after (per node: fidelity, math cases passing/total):** TimeOffset 100.0 (- ) -> 100.0 (14/14); FrameHold 100.0 -> 100.0 (8/8); Retime 98.3 -> 98.8 (47/47); FrameRange 100.0 -> 100.0 (3/3); TimeClip 100.0 -> 100.0 (33/33); AppendClip 96.7 -> 97.6 (12/12); TimeEcho 54.2 -> 92.2 (9/9); TimeBlur 89.6 -> 92.6 (8/8); TimeWarp 75.0 -> 82.4 (12/12). Time group 96.1 -> 97.1 rank-weighted (plain mean 78.9 -> 82.7). Nodes with behaviour checked 82 -> 91 of 167. Overall 93.5 -> 93.6 (previous formula 93.2 -> 93.3).
- **Open in the Time group:** the box filters' weights (Retime box and shutter, TimeWarp box) are not in the guide; the fade curves of TimeClip and AppendClip and the dissolve weights and overlap are not either; FrameHold between its listed frames, at frame 0, negative frames and before the first frame, and a negative increment; whether TimeOffset reverses before or after its offset; which way a half rounds (Retime nearest, TimeWarp nearest); TimeClip's bounce before the first frame (the guide's example is Retime's copied); a TimeEcho fade longer than the frames looked at. TimeWarp has no `length` knob; Retime's and TimeWarp's defaults (box) and TimeEcho's (mode max, 1 frame) are still ours; the layout of AppendClip and TimeBlur is as before. No Nuke render exists for any of it (the reference pack renders one frame).
- **Not done:** the Time group has no pixel truth; Kronos, OFlow, SmartVector and VectorToMotion are untouched.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step I of 2: Roto's per-shape knobs and Tracker's secondary knobs (integrator tick)

- **What was done:** `main` `772f2ca6` -> `ae01e17b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `ae01e17b`: Ran 8657 tests in 578.163 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step I of 2: Roto's per-shape controls and the Tracker's Transform tab

- **What was done (evidence):** two commits, targeted gate green (modules listed in the report). (1) Roto: the per-shape knobs the guide's Shape, Transform and Lifetime tabs state now have pixel tests on shapes built in the payload (colour, invert, visible, translate, rotate, scale, skew X and Y, skew order, centre, open-spline width and end types, lifetime in five types on ten frames). Two kernels disagreed with the guide and are fixed: a shape blended in `minus` took what is below minus the shape, the guide says A - B (the shape taken from what is below) and lists `from` (B - A), `difference`, `exclusion`, `max`, `min` and `screen`, now offered and drawn; and an open spline's edge was a hard step half a pixel past half the width, so an odd width covered one row too many (width 3 covered 4); it is a one-pixel ramp now. Both sit behind `legacy_roto_shapes` (new Roto 0, a saved Roto 1: minus becomes from, the hard edge stays), tested bit for bit. (2) Tracker: the Transform tab's `black outside` and `clamp` were read the older way (opaque black / an edge extension that never limited a halo) and the window of black outside off ignored the format; a new Tracker now reads them as the Transform node does (`legacy_transform_edges` 0, black outside on by default), a saved one keeps its pixels (`legacy_transform_edges` 1, black outside 0). The tile path takes a new Tracker with black outside on and no clamp (a plain zero fill) and sends black outside off, clamp and motion blur to the full-frame evaluator.
- **Cases:** Roto 18 -> 36 (blend minus/from/difference/exclusion/max/min/screen scored; multiply unscored with its reason; rotate quarter and half turn, scale, centre, skew X, skew order XY and YX, open-spline width 4 and 3, square ends, rounded ends). Tracker 3 -> 5 (black outside on is black, off holds the edge).
- **Audit before and after:** Roto 89.6 -> 89.6 (math was already 100 on 18 cases; now 36), Tracker 90.7 -> 91.1 (defaults 79 -> 83, black outside now agrees). Groups Draw 88.7, Transform 93.9, both unchanged to one decimal; overall 93.3 -> 93.3.
- **Schema:** no bump (new params get their legacy value from `drawbehaviour` / `trackerknobs` on load).
- **Left out, with reasons in the mapping:** Roto's Clone tab (`source_*`, `shape_source`), `view` and `stereo_offset`, Roto's Tracking tab (all `unsupported`, as in step E); `feather_type` smooth0 and smooth1 (the guide places the centre but gives no curve); blend modes color-burn, color-dodge, hard-light, overlay and soft-light (names without a formula) and `multiply` (the guide's AB leaves the matte as it is; NodeBased composites it); Roto motion blur beyond the existing samples. Tracker's 13 secondary knobs stay `unsupported` (matching iterations, zoom window, warp types, regrab and keyframe-strip settings: no counterpart in NodeBased's one-shot analysis); `smoothT/R/S` stay one `smoothing` count (the guide does not say whether N frames means N either side); the Tracker's `filter` default stays bilinear (Nuke's Cubic would widen the output window by a pixel and move a test outside this step).
- **Not done:** no real Nuke render; the y axis and rotation direction of a Roto shape transform keep NodeBased's raster rows (the tests use turns and shifts that do not depend on the direction); shape colours are tested, the other channels of the blend modes are not scored by the matte cases.

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step H of 2: ScannedGrain, Flare, Glint and Sparkles compute what Nuke computes (integrator tick)

- **What was done:** `main` `477d0dbb` -> `7d19d55b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `7d19d55b`: Ran 8612 tests in 580.983 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step H of 2: GridWarp and SplineWarp get Nuke's inputs, mask controls and a checked warp (integrator tick)

- **What was done:** `main` `ae42d3b7` -> `97ccfc91` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `97ccfc91`: Ran 8555 tests in 583.205 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - Lane 9 (2D behaviour), step H: GridWarp and SplineWarp

- **What was done (evidence):** `nodebased/warpknobs.py` holds the tables and helpers; kernel in `Evaluator._warp_node`. GridWarp gains the `dst` and `bg` inputs (after the mask, so a saved edge keeps its slot; GridWarpTracker's `vectors` slot is untouched), `grid_output` (source / source warped / destination / destination warped), `background` and `background_mix`, and the shared chrome (channels, mask channel, invert, inject, fringe). SplineWarp gains `premultiply`, `outputMask` (a `mask_splinwarp` layer, alpha 1 where the warp moved the picture), `crop_to_format`, `boundary_bbox`, `layer_warp` and `pair_warp`, and `channels`. `key` and `curves` are paired as `panel` entries: the payload's `pairs` list is the curves, the keys are the animation keys of each point's scalars.
- **Cases:** GridWarp 3 scored (interior key moves the pixel under it by the offset and the corners stay; mix 0 is the source; output source is the source) + 1 unscored (background over the source: the guide's page is a video transcript with no formula); SplineWarp 2 scored (a curve paired with one shifted 3 pixels right moves the pixels on it; mix 0 is the input). All pass at the audit's 1e-5 tolerance.
- **Legacy:** no flag needed for GridWarp (defaults reproduce the old picture: background mix 0). A saved SplineWarp gets `crop_to_format` 0 and `outputMask` none on load (`warpknobs.upgrade_params`); `boundary_bbox` 1 is the old behaviour. No schema bump.
- **Audit before and after:** GridWarp 26.3 -> 70.2, SplineWarp 35.2 -> 82.1. Groups: Transform 92.9 -> 93.9, Filter 93.5 -> 93.5. Overall 93.2 -> 93.3.
- **Open:** GridWarp: `warp` and the morph output (the guide page is a transcript; no formula), the Source/Destination grid tabs (keys, visible, locked, resize to image), the two grid transform tabs, `render_res` (submesh), `source_color`/`destination_color`; the filter list is NodeBased's two choices. SplineWarp: `output` (A, A warped, B, B warped, AB morph; NodeBased's is image/stmap), the `B` input, the Transform tab, `rbf`, `previewrez`, layer and pair lists. Neither warp is on the tile path (full frame, as before). The acts probe is skipped for both (payload-carried data; `NOT_PROBED`); `tests/test_2d_behaviour_warp_h.py` moves each new knob on a warp.
- **Not done:** no real Nuke render; the background composite and the output mask are NodeBased's reading.

## 2026-11-03 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step G of 2: Bilateral, Sharpen and VectorCornerPin get Nuke's missing knobs and inputs (integrator tick)

- **What was done:** `main` `0b3564ac` -> `44bab365` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `44bab365`: Ran 8536 tests in 591.445 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
## 2026-11-03 - lane 2 (2D behaviour), step H of 2: ScannedGrain, Flare, Glint and Sparkles compute what Nuke computes

- **What was done (evidence):** (1) New module `nodebased/drawfxbehaviour.py` holds Nuke-form kernels for the four nodes, each behind a `legacy_*` flag. Flare is the guide's donut (inner colour inside the inner radius, the ring colour at the middle radius, black from the outer radius, `size mult`, `brightness`, a chroma shift along an angle). Glint casts `no. of rays` rays 360 / n degrees apart (the first pointing up, positive rotation clockwise), each `steps` copies of the highlights (what is above `tolerance`) out to `length`, colour running from `from color` to `to color`, `max` or added, `odd ray length`, `effect only`. Sparkles casts `rays Nb` rays of `size` pixels and `thickness` from a position, directions drawn from RandomState(seed), the larger value kept where they overlap. ScannedGrain adds (plate - `offset`) times the red, green and blue `amount` with `saturation` and `low clip` (the grain is not subtracted below it), the plate translated by seed + frame; `fullGrain` is added as the path knob (NodeBased reads the grain from the plate input). (2) 44 cases in `tools/nuke_fidelity_cases_extra.py`: 34 scored (Flare 7, Glint 12, Sparkles 9, ScannedGrain 6) and 10 not scored with the reason (fade curves, anamorph, chroma shift numbers, Flare alpha, Glint odd-ray choice and highlight strength, Sparkles ray fade, modes and random knobs, ScannedGrain frame selection). (3) Glint gained the channels and mask strip (`FILTER_CHROME`). (4) `PROBE_BASE` in `tools/nuke_fidelity_requirements.py` starts the knob probe for the three effects inside the 16 x 12 fixture (their defaults put the effect off the picture). (5) Mapping JSON, knobs (Nuke's order and labels), LIMITS, tiers pixel units, `docs/PARITY_2D.md` rows and step note, `docs/PARITY_2D_FIDELITY.md` regenerated and copied byte-identical.
- **Kernels changed (legacy flag):** Flare `legacy_flare`, Glint `legacy_glint`, Sparkles `legacy_sparkles`, ScannedGrain `legacy_scannedgrain`; `upgrade_nodes` sets each to 1 on a saved node (and the older Glint rays 4, length 12, tolerance 1 and Sparkles seed 1 when a saved node lacks them), and a new node has it 0. No schema bump. New Glint defaults: rays 2, length 50, tolerance 0.5 (Nuke's); new Sparkles seed 23. `NODE_LIMITS` lets Glint and Sparkles take a tolerance from 0 to 1000 (the shared name was capped at 1).
- **Wrong before:** Flare was a gaussian glow with streaks and ghosts, not a donut with radii and ring and inner colours; Glint's rays were fixed to eight directions of a line convolution with a `falloff`, no steps, colours or `max`; Sparkles was a random cross on bright pixels, with no position, rays, thickness or rotation; ScannedGrain scaled the grain by the input's luminance to a power and a stock preset, where the guide subtracts the offset and multiplies by the amounts, with no `fullGrain`.
- **Tests:** `tests/test_2d_behaviour_extra_draw.py` (57: cases, a mutation check that the cases bite, kernel behaviours, every paired knob acts, saved documents bit for bit against `tests/legacy_behaviour_d.py`, whole-frame jobs, panel order); older tests in `test_2d_parity_v3_effects.py` and `test_flow2b_gridwarp_tracker.py` set the legacy flags.
- **Score:** ScannedGrain 15.4 -> 100.0, Flare 35.4 -> 77.2, Glint 45.6 -> 94.0, Sparkles 40.6 -> 80.5; Draw group 88.7 -> 91.4, Filter 92.3 (unchanged: these four are Draw); overall 93.0 -> 93.2, previous formula 92.6 -> 92.9; behaviour checked 74 -> 78 of 167 nodes, nodes with a kernel and no case 4 -> 0. Inert knobs: Flare 1 -> 0, Glint 0 -> 0, Sparkles 0 -> 0 (the probe now starts inside the fixture), ScannedGrain n/a (only `fullGrain`, a path).
- **Left (needs a source or a real Nuke render):** Flare offset, anamorph, chroma spread, visibility, shape (corners, flattening), asymmetry and multiple-flare tabs (40 knobs); Glint aspect ratio, gamma and source channel; Sparkles' modes, anamorphic, relative rotation, chroma, crawl, broken rays and sparks; ScannedGrain frame range, min width, resize, flip, curves, precomputed table and the (un)premult strip; the splits (ScannedGrain amount is a colour knob NodeBased holds as three scalars, not yet split).

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step G of 2: EdgeBlur, GodRays, VolumeRays and BumpBoss compute what Nuke computes (finish 1) (integrator tick)

- **What was done:** `main` `b7dcb7c5` -> `40183c86` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `40183c86`: Ran 8501 tests in 581.018 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
## 2026-11-03 - lane 9 (2D behaviour), step G of 2: Bilateral, Sharpen and VectorCornerPin get Nuke's missing knobs and inputs

- **What was done (evidence):** two commits, targeted gate green. (1) Bilateral (`nodebased/bilateralknobs.py`): the `guide` input (wired through the evaluator's mask/mix path and the tile executor, region rule reads the guide a window's reach past the tile), `guide_channels` (Guide Channels), `positional_sigma`, `bilateral_filter` (gaussian or median, a weighted median under the same weights), Size as the window (radius `size // 2`, positional falloff sigma `positionalSigma * size` pixels; the guide states neither, so the exact smoothed picture is an unscored case), colour sigma default 0.4. Graphs saved before keep `legacy_bilateral` 1 (radius `ceil(size)`, colours from the picture's RGB, colour sigma 0.1 when absent); tested bit for bit against the old kernel, and Denoise's call is unchanged. (2) Sharpen: `quailty` is implemented (the blur is made on a scaled-down picture once the size passes it); it acts only above the size, so it is listed in `NOT_PIXEL` with that state and `tests/test_2d_behaviour_filter_g.py` moves it at size 40. (3) VectorCornerPin (`nodebased/vectorcornerpinknobs.py`): `user1`-`user4`, `to1`-`to4`, the From quad (`to1#2`), `output_mode` (warped src, st-map, st-map inverse) and `use_baked`; with no SmartVector wired the pins stay put; points are in Nuke's pixel space (y up), the default 0 0 / 100 0 / 100 100 / 0 100 stands for the format's corners; legacy corner points kept behind `legacy_vectorcornerpin` 1 (SmartVector still required there). Case runner gained `_run_plate_document` (`_plate` in a case's params) in `tools/nuke_fidelity_cases.py`.
- **Cases:** Bilateral 2 scored (size 1 and a tiny colour sigma leave the picture unchanged) + 2 unscored with reasons; VectorCornerPin 4 scored (identity, pins moved 3 right and 2 up, pins swapped = mirror, From quad picks the source), all pass with error 0.
- **Audit before and after:** Bilateral 64.2 -> 100.0, Sharpen 95.1 -> 98.5, VectorCornerPin 25.0 -> 93.3. Groups: Filter 91.2 -> 92.4, Transform 92.2 -> 92.9. Overall 92.8 -> 93.0 (95.2 -> 95.4 counting only the nodes NodeBased has); previous formula 92.5 -> 92.7.
- **Open (VectorCornerPin):** `bakeCorners` (a render over the frame range that writes the To pins; `use_baked` stands for it), `imageFilter` (the warp is bilinear), `frameDistance` and `blurSize` (they steer the SmartVector solve and the internal STMap), the user-keyframe buttons; the vectors' motion on the pins is not checked by a pixel case (the fixture has no vectors with motion; tests/test_smartvector.py runs the zero-motion rig). Bilateral: the guide's format conversion (the output takes the guide's format; here the guide is fitted to the source's window). Sharpen: `defaults` 85 (not in this step).
- **Not done:** no real Nuke render; no schema bump (flags are params set by `upgrade_nodes`); VectorCornerPin stays off the tile path (whole-frame node, as before).

## 2026-11-03 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step G of 2: EdgeBlur, GodRays, VolumeRays and BumpBoss compute what Nuke computes (integrator tick)

- **What was done:** `main` `5ff5eb05` -> `8d2a2633` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8d2a2633`: Ran 8497 tests in 602.893 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-03 - lane 2 (2D behaviour), step G finish: VolumeRays style, pre-blur and radial falloff

- **What was done (evidence):** re-read the Nuke 17.0 VolumeRays and BumpBoss pages and searched for the VolumeRays gizmo and BumpBoss formulas; neither exists in any source found, so BumpBoss's lighting model stays unbuilt. VolumeRays gained the knobs whose behaviour the guide states plainly: Volumetric Creation (RGB Luminance, Alpha Solid), Pre-Ray Blur (the Blur node's Gaussian on the ray source), Enable Radial Falloff, Radial Size and Radial Falloff (rays fade with distance from the centre, faster when higher; the curve `exp(-falloff * distance / size)` is ours, the guide gives none). Evaluator only: VolumeRays has no tile path (precedent: it stays outside tiles). Mapping: 5 knobs now paired; Edge Size and Edge Blur stay unsupported (Alpha Edges and Shadow Making styles not built).
- **Kernel changed:** VolumeRays (legacy: a saved node gets `style` RGB Luminance, `pre_blur` 0, `chk_radial` 0 on load, so it renders as before; a new node has radial falloff on, as Nuke's default). No schema bump.
- **Tests:** `tests/test_2d_behaviour_extra_filter.py` (4 new VolumeRays tests, 2 updated); the brief's targeted gate (208 tests) OK after the report was regenerated.
- **Score:** VolumeRays 71.5 -> 80.5, BumpBoss unchanged at 44.4; Filter group 92.3 (unchanged: 79.4 -> 79.7 on the requirements column); overall 93.0, previous formula 92.6. Inert knobs on VolumeRays: 0 of 11 probed.
- **Left (needs a source or a real Nuke render):** VolumeRays ray length, quality, flicker (4), initial and end volume colour, mask options (3), Alpha Edges and Shadow Making styles with Edge Size and Edge Blur; BumpBoss's lighting model (lightposition, bumpsize, height, intensity, minshadow, center).

## 2026-11-02 - lane 2 (2D behaviour), step G of 2: EdgeBlur, GodRays, VolumeRays and BumpBoss

- **What was done (evidence):** (1) EdgeBlur: 9 scored cases and 2 not scored in `tools/nuke_fidelity_cases_extra.py` (pixels outside the edge band unchanged; inside it, the box blur of the same size on the interior; tint and brightness; size 0); the kernel now takes Nuke's `filter`, `quality`, `crop to format`, matte channel, `tint` and `brightness` and reuses Blur's kernels (`nodebased/blurknobs.py`). (2) GodRays: 3 scored cases (identity at the defaults; the streak of one white pixel at x 12, 10, 8 with max on; from and to colour at the two ends) and 5 not scored; new kernel `nodebased/raysbehaviour.py`. (3) VolumeRays: 5 plain knobs implemented on the existing kernel; 1 not scored case. (4) BumpBoss: 1 not scored case; the 6 guide knobs recorded unsupported.
- **Kernels changed:** EdgeBlur (legacy: `kernel_filter` gaussian, `quality` 1000, `edge_mult` 1, `edgeblur_size` 4 set on load); GodRays (`legacy_godrays` 1 on load keeps the additive smear; new node default steps 5); VolumeRays (`comp_me` 1 on load, off for a new node). No schema bump.
- **Wrong before:** EdgeBlur had no filter choice, tint or brightness and its defaults were not Nuke's; GodRays added streaks to the picture at its defaults and its `translate` was one zoom number where Nuke has a vector; VolumeRays always added the rays on top.
- **Tests:** `tests/test_2d_behaviour_extra_filter.py` (37); the targeted gate in the brief, plus test_2d_behaviour_filter_a, test_2d_fidelity_b_chrome, test_2d_fidelity_e_fringe, test_2d_parity_v3_effects and test_nodecatalog, all OK.
- **Score:** EdgeBlur 53.6 -> 97.1, GodRays 45.5 -> 98.9, VolumeRays 36.7 -> 71.5, BumpBoss 31.4 -> 44.4; Filter group 89.1 -> 90.2, Draw group unchanged at 87.6; overall 92.2 -> 92.3 (previous formula 92.5 -> 92.6). Behaviour checked 68 -> 70 nodes (EdgeBlur, GodRays). VolumeRays and BumpBoss have only not-scored cases, so their `math` shows n/a and part of their gain is the end of the "0 unchecked" penalty, not new pixel checks.
- **Left:** VolumeRays' style, alpha options, ray length, pre-blur, quality, flicker (4), colour ramp (2), radial falloff (3) and mask options (3); BumpBoss's whole lighting model (needs a source with the formula, or a real Nuke render); GodRays' from and to colour are one gain each (Nuke's are RGB colours, splits pending); `output` of EdgeBlur; the order in which GodRays' translate, rotate, scale and skew combine, and their signs (the guide gives none).

## 2026-11-02 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step F of 2: Tracker, DropShadow, DirBlur and LightWrap compute what Nuke computes (integrator tick)

- **What was done:** `main` `81cb348c` -> `c803c0db` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `c803c0db`: Ran 8460 tests in 587.602 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 9 (2D behaviour), step F of 2: Tracker, DropShadow, DirBlur and LightWrap compute what Nuke computes

- **What was done (evidence):** (1) Cases in `tools/nuke_fidelity_cases_filter.py`: Tracker 3 (match-move and stabilise by a sub-pixel (2.5, 1.5) track delta against a bilinear resample of a ramp plate, and the reference frame as identity; `nuke_fidelity_cases._run_tracker_document` builds the document from `_tracks`/`_frame`/`_bg`), DropShadow 10 (8 scored: angle 180 right, 0 left, distance, opacity, colour, inherit input colour, enable off; 2 not scored: which way 90 casts, the softness filter), DirBlur 5 (4 scored: linear length 3 and 5 along x, angle 90 down the columns, length 1 identity; 1 not scored: zoom and radial paths), LightWrap 3 (2 scored: intensity 0 adds nothing, wrap only with no wrap is black; 1 not scored: the edge amount). 17 scored, all pass; 4 not scored, each with its reason. (2) Tracker analysis: an analytic two-blob feature moves (0.4, -0.25) pixel per frame; `tracker.analyse` recovers every frame within 0.2 pixel (measured worst 0.13); match-move and stabilise at frame 4 equal the plate shifted by the tracked delta to 1e-5. (3) Kernels changed: DropShadow angle now reads the guide's "0/360 is left, 180 is right" (a new node: offset angle = 180 - angle; default 225); saved graphs get `legacy_dropshadow_angle` 1 and keep -45 style angles. LightWrap's CCorrect tab (saturation, contrast, gamma, gain, offset and the three ranges' gamma, gain, offset) now grades the wrap through ColorCorrect's kernel; saved graphs get `legacy_lightwrap_grade` 1 (the knobs did nothing before). DirBlur's UseTarget (direction from BlurCenter to Linear Target), Samples (taps of zoom and radial) and Pixel offset (zoom) act in new nodes; saved graphs get `legacy_dirblur_knobs` 1; a new node's target is "480,240" (Nuke's). Evaluator and tile path share each kernel; tiles of 7, 16 and 32 equal the whole frame with and without a mask (DropShadow, DirBlur linear with a target, LightWrap with the grade). (4) Main tabs in the guide's order and labels: Tracker (24 of 35 -> all paired knobs in order; sections Matching, Auto-tracking, Keyframes, Transform), DropShadow (5 of 9 -> 9; `softness` reads Softness, `operation` reads Input), DirBlur (9 of 12 -> 12; BlurType, BlurLength, BlurAngle), LightWrap (21 of 27 -> 27). Old DropShadow tests that wrote the old angle (0 = right) now write 180.
- **Audit before and after:** Tracker 61.1 -> 90.7, DropShadow 66.5 -> 96.8, DirBlur 60.3 -> 89.1, LightWrap 66.1 -> 95.4. Groups: Filter 89.1 -> 91.2, Transform 89.2 -> 92.2, Draw 87.6 -> 88.7. Overall 92.2 -> 92.8; previous formula 92.5 -> 92.5. Inert knobs: Tracker n/a (not probed: tracks need frames); DropShadow 1 -> 0 (`operation` acts with bg wired, listed in NOT_PIXEL); DirBlur 7 -> 3 (`holdout`, `PixelAspect`, `Quality`; `BlurCenter`, `Samples`, `pixeloffset` are zoom/radial knobs listed in NOT_PIXEL because the default type is linear); LightWrap 19 -> 5 (`disableluma`, `enableglow`, `LumaTolerance`, `white`, `gray`; `shadows_gain` needs a dark wrap, listed in NOT_PIXEL).
- **Tracker secondary knobs still missing (13):** `max_iter` (max iterations of the match), `epsilon` (match resolution), `zoom_window_behavior`, `zoom_window_size`, `zoom_magnification_size`, `zoom_window_filter_behaviour` (the viewer's zoom window), `warp` (warp type: translate, rotate, scale, affine), `auto_regrab_pattern` (regrab when tracking stops), `regrab_when_offset` (regrab when the tracker is moved), `retrack_on_move` (retrack when a keyframe moves), `keyframe_display`, `keyframe_size`, `max_number_of_keyframe_tracks_to_display` (on-screen key-track limit).
- **Open:** DirBlur `holdout` (the knob is the 3D on/off enum; Nuke's is a channel menu and only matters for zoom), `PixelAspect` (stretch of the blur direction; needs a non-square-pixel format), `Quality` (Nuke's is a number, a scale factor of at least 1; ours is low/normal/high), `Samples` and `pixeloffset` for the linear type, the zoom/radial length units (the guide says pixels for every type; ours are percent and degrees), a new node's BlurType default (Nuke zoom, ours linear). DropShadow: which way the angle turns through 90 (the guide gives only left and right; the reading 180 - angle makes 90 up and the old default look the same), softness filter, Nuke's defaults (distance 5, opacity 0.7, softness 0). LightWrap: `disableluma`, `enableglow`, `LumaTolerance`, `white`, `gray`; the node's `mix` is the dissolve between the wrap and the colour-corrected wrap in Nuke and the node mix here; Intensity default 0 in Nuke, 1 here. Tracker: the 13 knobs above; per-channel splits none listed for this node.
- **Not done:** no real Nuke render; no schema bump (flags are params, set by `upgrade_nodes`).

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step F of 2: Cryptomatte's missing knobs, the OCIO nodes checked against a stated config, and Write's main knobs, Lane 9 (2D behaviour B, Claude Sonnet 5.5) step E of 2: Roto and RotoPaint compute what Nuke computes, and their panels read like Nuke's (integrator tick)

- **What was done:** `main` `4c40f139` -> `08eb6601` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `08eb6601`: Ran 8419 tests in 578.733 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 9 (2D behaviour), step E of 2: Roto and RotoPaint compute what Nuke computes; their main tabs follow the guide

- **What was done (evidence):** (1) 21 Roto cases and 6 RotoPaint cases in `tools/nuke_fidelity_cases_filter.py`, each with its shapes or strokes written in the node's payload (`nuke_fidelity_cases._run_payload_document` builds the document: `_shapes`, `_items`, `_bg` and `_second`). Roto: square and ellipse fill (interior and exterior scored, rim not), linear outward feather and negative feather inward (0.75, 0.25 at 0.5 and 1.5 pixels for a feather of 2), smooth feather half way at half the feather, shape and node opacity, invert, over and plus blends, translate, visible, output alpha and rgb on a wired plate, replace, premultiply rgb and rgba. RotoPaint: a hard dab is a disc of the brush diameter, a larger brush a larger disc, stroke opacity, clone offset against the plate, reveal against the second image. Not scored, each with its reason: the half-covered edge pixel, smooth0 and smooth1 curves, the minus blend's direction, the soft-brush falloff curve. (2) One kernel disagreed: with replace off the guide keeps the incoming alpha ("the incoming image already has an alpha channel which you want to throw away" is what replace is for), and NodeBased wrote the matte alone. Roto now draws the matte over the plate's alpha on the evaluator and the tile path, and premultiply rgba squares the resulting alpha (a*a). Older documents get `legacy_roto_alpha` = 1 from `drawbehaviour.upgrade_params` (no schema bump: `upgrade_nodes` adds it) and render bit for bit as before. (3) RotoPaint's `replace` changed nothing; it now clears the image the strokes are drawn on (the plate still feeds clone and background-source strokes), on all three paths (kernel, windowed kernel, tile path), behind `legacy_paint_replace` = 1 for older documents. (4) Main tabs in the guide's order and labels for both nodes (`NUKE_LAYOUT_LANE9`), the node-only `invert` of Roto relabelled "Invert matte". (5) `key` is paired as a panel knob on both nodes (the Roto panel's keyed-here marker).
- **Tests:** `tests/test_2d_behaviour_roto.py` (23): every case passes or is not scored with a reason, the matte over alpha, replace, legacy pixels bit for bit for both flags, tiles of 7, 16 and 32 equal the evaluator for both nodes and every option, premultiply rgba squares the alpha, a harder brush never paints less, main-tab order and labels, output, premultiply, replace, clip to and opacity each change pixels on both nodes. `tests/test_roto_bg_plate.py` updated (the plate's alpha stays outside the shape, replace restores the old matte-only alpha).
- **Score:** Roto 60.9 -> 89.6, RotoPaint 66.8 -> 95.2, Draw group 77.2 -> 87.6, overall 89.7 -> 90.7 (91.9 -> 93.0 counting only the nodes NodeBased has); under the previous formula 92.0 -> 92.2.
- **Left (listed in the report with the reasons):** Roto's 26 and RotoPaint's 14 remaining secondary knobs: the Clone tab and `shape_source` on Roto (Roto shapes carry no source; RotoPaint strokes do), `view`, `stereo_offset` and `source_view` (no multi-view documents), the Tracking tab's settings, Export tab and corner pin points (the Tracker node holds tracking), `format` on RotoPaint (the plate sets it), smooth0 and smooth1 feather types (the guide gives no curve).

## 2026-11-02 - lane 2 (2D behaviour), step F of 2: Cryptomatte's missing knobs, the OCIO nodes checked against a stated config, Write's QuickTime and MXF knobs

- **What was done (evidence):** two commits, targeted gate green. (1) Cryptomatte gained `manifest_source` (metadata, sidecar), `sidecar_filepath` and `preview_enabled`; Encryptomatte gained `matte_name_type` (auto, manual), `merge_operation` (over, under, plus, replace) and the Cryptomatte Layer drop-down is `layer_name`. `tests/test_2d_behaviour_crypto.py` (25) writes a real two-ID EXR: the embedded manifest and the sidecar JSON disagree about the sphere's id, so each source picks a different matte; Preview on paints `cryptomatte.preview` into the colour channels and leaves the alpha matte bit for bit (off: the input colour); auto names a matte after its source node (the walk passes `_matte_names` on the evaluator and the tile path), manual uses `id0`..`id7`; over takes the old set's coverage down by the new matte (B * (1 - a)), under takes the new matte down by the old total, plus keeps both, replace writes the set afresh; tiles of 7 equal the whole frame for all four. `exportWrite` (a button that adds a Write node) is unpaired with a reason. No schema bump: older nodes get `preview_enabled` 0, `matte_name_type` manual and `merge_operation` replace from the load-time defaults in `core.py` (they render as before; the pre-existing tests that typed names now set manual). (2) OCIO: `tools/nuke_fidelity_ocio_fixture.py` writes a small config (a 2x gain matrix, an exponent of 2.2, a `double` look, two displays with views, a bent 3-point 1D LUT); 15 cases in `tools/nuke_fidelity_cases_color.py` compute the expected pixels from those statements (power-law cases compare only where the expected value is at most 0.9, because OpenColorIO's fast pow is off by 2e-5, e.g. 1.0000223 for 1.0 ** 2.2). `tests/test_2d_behaviour_ocio.py` (22) runs each node through a graph, tiles of 7, 16 and 32 with and without a mask, and every paired knob acting.
- **Kernels changed:** (a) the five OCIO kernels divided the colour by alpha and multiplied it back; Nuke transforms the stored colour (its own (un)premult by does the dividing), so a new node no longer divides and `divide_by_alpha` 1 (set by the load-time defaults on saved nodes) keeps the older pixels; (b) OCIODisplay started from a hard-coded "ACEScg", which does not exist in other configs: it now starts from its new `src` (Nuke's Input colorspace, default "linear"; a space of that name, else the config's scene_linear role, which is ACEScg in the built-in config), so older documents render the same.
- **Probe:** `tools/nuke_fidelity_requirements.py` probes the OCIO nodes against the fixture config (a pass-through request, each space, look, display, view, file and direction moved); `invert` on OCIOLookTransform and OCIOFileTransform left `NOT_PIXEL` because the probe now names a look and a file.
- **Write:** the 29 counted missing knobs are all QuickTime and MXF tabs; Write's file types (`core.WRITE_FILE_TYPES`) are exr, png, tiff, jpeg, dpx, so none can act. They were already `unsupported` in the mapping; the reason now names the real file types, and `tests/test_2d_behaviour_write.py` pins that. Write's pixels and files are unchanged.
- **Numbers (from main `4c40f139`):** Keyer 89.3 -> 91.6, Color 95.1 -> 95.6, Image 86.7 -> 86.7 (rank-weighted). Nodes with behaviour checked 61 -> 66 (OCIOColorspace, OCIOLogConvert, OCIOLookTransform, OCIODisplay, OCIOFileTransform); nodes whose kernel runs with no case 19 -> 14. Knob coverage: Cryptomatte 1/4 -> 4/4, Encryptomatte 1/4 -> 3/4, OCIODisplay 3/17 -> 4/17, the other four OCIO nodes unchanged, Write 27/56 unchanged. Knobs that act: 286 of 320 -> 297 of 331 (34 inert on the fixture, unchanged). Overall 90.9 -> 91.1 (93.2 -> 93.4 counting only the nodes NodeBased has); previous formula 92.1 -> 92.3.
- **Still open in lane 2's groups:** Cryptomatte and Encryptomatte have no fixture case (the audit's one-picture probe cannot hold EXR layers, so `acts` and `math` are `-`; the tests above are the evidence); `exportWrite`; the matte list's wildcard (`*`) and Matte Output / Unpremultiply / Remove Channels are not paired; Preview on by default paints the id colours into the colour channels of the final view (the guide's Viewer overlay), so a graph that wants the input colour downstream sets Preview off. OCIO: the Context tab (key1..4, value1..4), `swap` (a button), `ignore_errors`, `working_space`, OCIODisplay gain, gamma, saturation, channel view and layer have no param (the guide states no formula for the display adjustments); an interpolation choice on `.cube` files changes nothing in OpenColorIO 2.5.2 (already listed). Write: the 29 QuickTime and MXF knobs (no writer). Elsewhere in the groups: nodes with no node or source (UDIM Import and the 25 nodes the guide lists that no row covers), commercial algorithms (Keylight, IBK internals), and the `not scored` cases listed in each earlier step.
- **Next owner:** the integrator regenerates `docs/PARITY_2D_FIDELITY.md` once on the stacked tree; Gonzo calls the aspect closed or names one more step (a Cryptomatte fixture the probe can hold, or a MOV/MXF writer).

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step E of 2: the Merge and Image nodes compute what Nuke computes, and their panels read like Nuke's (integrator tick)

- **What was done:** `main` `c34bf1f0` -> `2a83922b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `2a83922b`: Ran 8351 tests in 587.370 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 2 (2D behaviour), step E of 2: the Merge and Image nodes compute what the guide says, and their panels read like Nuke's

- **What was done (evidence):** two commits, targeted gate green (336 tests). (1) Merge group: `tools/nuke_fidelity_cases_merge.py` (registered at the tuple in `tools/nuke_fidelity_cases.py`) holds
  27 cases for Switch (2 scored, 2 not scored), Dissolve (4), TimeDissolve (3), AddMix (3 scored, nonlinear not scored), Blend (3 scored, all-zero weights not scored), CopyRectangle (2 scored, softness
  not scored) and Constant (4 scored, colour with alpha below 1 not scored); every unscored one carries its reason. The kernels already agreed with the guide; one changed: Switch used to raise
  for `which` past either end and now shows the nearest input (`core.switch_index`, used by the kernel, the raster path, `tiers._switch_rule` and the tile executor's source walks; no legacy flag
  because the old behaviour was an error, so no document rendered differently; `tests/test_phase_b.py` pinned the error and now pins the clamp). The guide does not say what Nuke shows there, so the
  case is `not scored`. CopyRectangle's five "inert" knobs were never inert: Nuke's default box (512 389 1536 1167) lies off a 16 x 12 picture, so the probe saw nothing. `PROBE_BASE` in
  `tools/nuke_fidelity_requirements.py` starts the probe with a box inside the fixture; the defaults are unchanged. TimeDissolve's `in`, `out` and `lut` act across frames (a graph-walk turn of the frame into
  `which`): `NOT_PIXEL` entries, and `tests/test_2d_behaviour_merge.py::TimeDissolveTests` shows the mix at the start, middle and end, a moved range, the smooth curve and an animated curve.
  (2) Image group: Read, Write and Constant main tabs in the guide's order and wording (`nodebased/knobs.py`; Read "File", "Proxy", "Reload", "Missing frames", "Input transform"; Write File, Proxy,
  Reload; Constant colour before size; TimeDissolve's ease reads "Curves"); the format-specific Write knobs sit between File type and Create directories; NodeBased-only knobs follow Nuke's.
  `tests/test_2d_behaviour_image.py` asserts the label order of each.
- **Not changed:** Viewer: the mapping pairs none of its controls (viewer state), so there is no panel order to compare. No schema bump. Merge, Premult, Unpremult, KeyMix, ZMerge kernels untouched.
- **Numbers (from main `b98a80d9`):** Merge group 91.7 -> 99.4 rank-weighted, Image group 80.9 -> 86.7. Nodes with behaviour checked 54 -> 61 (Switch, Dissolve, TimeDissolve, AddMix, Blend,
  CopyRectangle, Constant); nodes whose kernel runs and have no case 26 -> 19. Inert knobs on the fixture 42 -> 34 (all eight were in this step: CopyRectangle 5, TimeDissolve 3). Layout difference lines
  43 -> 39 (Read, Write, Constant, TimeDissolve cleared). Overall 89.0 -> 90.2 (NodeBased's own nodes only 91.2 -> 92.4); previous formula 92.0 -> 92.1.
- **Still open, Merge:** Switch and Dissolve take two inputs (the guide: any number, 0 to 9999); Switch past the end and fractional `which`, Blend all-zero weights, AddMix `nonlinear` and CopyRectangle
  `softness` are `not scored` for want of a stated form; the guide names Blend's checkbox `weight1`, so `normalize` cannot be written as a Nuke knob in a case (checked in the test instead);
  CopyRectangle's `area` counts rows from the top like Crop, and whether Nuke's y runs up is not in the guide (lane 9's Crop decision); Blend defaults 83 and chrome 80; ContactSheet layout (8 of 11),
  defaults 45; ZMerge label "Depth layer" vs "Z channel".
- **Still open, Image:** Constant stores the colour times its alpha (not scored: the guide does not say how Nuke stores it), defaults 75; Read defaults 35; Write 29 knobs missing, all on the Metadata
  Export tab; the five to take first: the QuickTime `mov64` Color Primaries and Transfer Function (first pair; 11 more codec copies follow) and MXF Codec, Pattern (OP pattern), FPS; CurveTool layout (8 of 11);
  UDIM Import is missing.
- **Next owner:** the integrator rebuilds `docs/PARITY_2D_FIDELITY.md` once on the stacked tree (lane 9 may also regenerate it).

## 2026-11-02 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step D of 4: Nuke's layout and width/height splits for the step A filter nodes (integrator tick)

- **What was done:** `main` `b98a80d9` -> `1b98bf85` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1b98bf85`: Ran 8324 tests in 606.409 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 9 (2D behaviour), step D of 4: Blur, Sharpen, Soften, Glow and Erode (filter) split their size; the step A main tabs follow the guide

- **What was done (evidence):** Nuke's `size` on Blur, Sharpen, Soften, Glow and Erode (filter) is a WH knob, so `radius`, `sharpen_size`, `soften_size`, `glow_size` and `filter_size` are now in `core.SPLITTABLE` as `wh` and laid out as `split` knobs. One reader, `blurknobs.size_value`, goes through `core.split_values` and returns a number (a scalar, or two equal entries) or a (width, height) pair; `blurknobs.filtered` takes either, blurring each axis at its own size (one quality scaling factor for both, from the larger entry). Glow's width channel scales both entries. Erode (filter) with two different sizes uses an ellipse whose fractional parts blend the four surrounding integer ellipses. The tile halo (`blurknobs.halo`, the Erode (filter) rules in `tiers.py` and `tiles.py`) reads the larger entry; `tiers.scale_params` now divides each entry of a list. Blur's GPU mapping raises `Unsupported` for a list. A scalar size takes the old code path, so older documents render bit for bit as before (no schema bump; schema 29 already holds lists). Main tabs for Blur, Sharpen, Soften, Glow, EdgeDetect, Laplacian, Erode (filter) and EdgeExtend are in the guide's order with its labels (`NUKE_LAYOUT_LANE9`, `NUKE_LABELS` in `nodebased/knobs.py`); EdgeExtend's Channels row now leads because `NUKE_LAYOUT_LANE9` places it first like any other node's.
- **Tests:** `tests/test_2d_behaviour_filter_layout.py` (18): a split size blurs differently across and down per node, equal entries equal the unsplit node bit for bit, tiles of 7, 16 and 32 equal the whole frame with and without a mask (including `quality` scaling), tier scaling, GPU refusal, one label-order test per node. `tests/test_2d_behaviour_draw.py` EdgeExtend layout test updated to the new order. Targeted gate (filter A, layout, knobs, tiers, core, split knobs, bypass, tiles, audit, knowledge, nodecatalog, label and panel tests, the older filter and draw modules) green.
- **Score:** overall 89.0 -> 89.7 (91.2 -> 91.9 counting only the nodes NodeBased has); under the previous formula unchanged at 92.0.
- **Not done:** Laplacian `quailty` and `crop` and Sharpen/Soften `quailty` still read as inert in the audit's probe (they need `NOT_PIXEL` entries or a probe context; not part of this step). No visual check of the split slider in the running app.

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step D of 4: every colour, keyer and channel knob changes the picture, and their panels read like Nuke's, Lane 9 (2D behaviour B, Claude Sonnet 5.5) fix: two background-thread hangs that keep stopping the full test suite (integrator tick)

- **What was done:** `main` `1e219846` -> `ad5da33b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `ad5da33b`: Ran 8306 tests in 576.777 s, OK (skipped=38), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 9 (2D behaviour), hangfix-1102: two background-thread hangs and the CI-only CornerPin failure

- **What was done (evidence):** (1) `cornerpin_perspective` failed on both CI runners: output pixel (7, 1) of the skewed quad has its centre exactly on the quad's top edge (source row exactly 0), so any float rounding flipped it. Moved to-point 2 from (14, 2) to (14, 2.6); every pixel centre is now at least 0.006 px from a source pixel boundary; audit regenerated (report unchanged at 94.4). (2) Queue panel close hang (`tests/test_queuepanel.py`, 10/7, 10/22, 11/2): `LoopRun.cancel` can land between the attempt row and `Queue.add_chain`; `run_until_idle` then skipped the cancelled chain's queued links for ever (`continue`) while still counting them unfinished, so it polled without end and `closeEvent`'s `shutdown(wait=True)` waited on it. Fix: `run_until_idle` marks links of a cancelled chain cancelled; it takes `stop` and `stop_grace` (nothing new starts, running links get their cancel event, a link still running after the grace is abandoned as cancelled); `LoopRun.run(loop_id, stop)` ends the attempt and the loop `cancelled`; `QueuePanel.closeEvent` sets its stop event first. Tests: `test_closing_the_panel_ends_a_running_loop_whose_worker_never_returns`, `test_cancelling_a_loop_before_its_chain_is_queued_does_not_hang_the_close` (both hang or fail on the old code). (3) Render worker hang: `GpuDisplay(None)` created a `QOffscreenSurface` off the GUI thread, which waits for the GUI thread (blocked in `agentloop.request`). Fix: `GpuDisplay.__init__` raises `GpuNeedsGuiThread` when it must make its own surface off the GUI thread; `get_display` and `_get_convert_display` return None without memoizing `_failed_reason`; `configure_surface(..., convert_surface=)` and `app.py` make the conversion thread's surface on the GUI thread. Tests: `GpuDisplayWorkerSurfaceTests` (a stand-in surface that blocks off the GUI thread; the old code fails two of them).
- **Measured:** 3D playback cube, 960x540, Xvfb, `tools/benchmark_3d_playback.py`: card pictures 146 of 146 and display GPU before and after; stage median "convert and emit" 0.37 ms before, 0.39 after. A first version imported `QThread` at the top of `GpuDisplay.__init__` and the stage "GPU render and wait" went 4 ms -> 13 ms (reproducible, six runs); importing it only where no surface is brought gives 3.8 ms again (final: whole frame 10.8 ms, was 11.4).
- **Runs:** `tests.test_queuepanel` 10 runs in a row green; the display-thread tests 10 runs green; targeted gate (display transform, gpu2d, playback, viewport, topbar, agentloop, core) green.
- **Not done:** the hang on the real xcb platform is not reproduced here (Qt's offscreen platform does not wait); the tests model the wait. CI on the branch is the Windows check.

## 2026-11-02 - lane 2 (2D behaviour), step D of 4: the colour, keyer and channel knobs act, and their panels read like Nuke's

- **What was done (evidence):** three commits, targeted gate green. (1) The 30 lane-2 knobs the audit called inert were each moved in the state where they act and shown to change pixels
  (`tests/test_2d_behaviour_knobs_color.py`, 24 tests): 29 now carry a `NOT_PIXEL` reason in `tools/nuke_fidelity_requirements.py` naming that state (a mask, a replace mode, an inside matte, a
  second layer, a Set button, a look or file); none was a missing kernel. One is unpaired: OCIOFileTransform `interpolation` (OpenColorIO 2.5.2 returns the same pixels for nearest, linear,
  tetrahedral and best on a bent .cube, checked in the test), its param `file_interpolation` stays as an `extra`. (2) Main tabs in Nuke's order and labels: Clamp, Exposure, ColorLookup,
  Keyer (A, B, C, D as four knobs), Keylight, IBKGizmo (`ublu` reads "Use bkg luminnance", the guide's own spelling of its label), AddMix (A, B), ShuffleCopy (1 in, 2 in), MatchGrade,
  OCIOColorspace, OCIOLookTransform, OCIODisplay, OCIOFileTransform (Input, Output, Invert direction, File), Vectorfield (Vectorfield file); ColorLookup's Set RGB, Set RGBA and Set A are `panel`
  entries in the mapping (they are buttons, `colorlookupknobs.set_points`). The Copy table was already in order. (3) Clamp `minimum` and `maximum` are `rgba` split knobs read through
  `Evaluator._channel_value` (the tile path calls the same kernel); scalar documents are bit for bit as before, no schema bump.
- **Numbers:** overall 87.6 -> 89.0 (counting only NodeBased's nodes 89.8 -> 91.2), previous formula 92.0 -> 92.0 (measured from main `1e219846`). Inert knobs
  on the fixture 72 -> 42 (lane 2 had 30, now 0). Layout differences 57 -> 43 lines (lane 2 had 14 nodes, now 0). Split knobs: nodes splitting all 8 -> 9 of 14 (Clamp). Nodes with behaviour
  checked 54 -> 54: Posterize, HistEQ and MinColor already carry cases from steps A and B, all `not scored` with reasons; no citable form was found to add (Posterize's page reads as a bit depth
  while the node is used as a level count, HistEQ's page gives no binning or range, MinColor's delta comes from the Find Pixel Delta button).
- **Changed pinned tests:** `tests/test_split_knobs.py` (Keyer, IBKGizmo and ScreenKeyer label lists, the Keyer range row became four knobs A to D).
- **Still open (Color, Keyer, Channel):** `math` unchecked for OCIOColorspace, OCIODisplay, OCIOFileTransform, OCIOLogConvert and OCIOLookTransform (no fixture colour config; a case needs a stated transform);
  knob coverage low for MatchGrade (3 of 33), OCIODisplay (3 of 17), OCIOLookTransform (4 of 14), OCIOColorspace (2 of 11), GenerateLUT, Vectorfield, Sampler, Histogram, IBKGizmo (7 of 15), HSVTool (14 of 25);
  ColorTransfer, Primatte, Ultimatte not in NodeBased; Cryptomatte and Encryptomatte 1 of 4. **Not done:** visual QA, GPU islands untouched, CI not read.

## 2026-11-02 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step C of 3: the draw nodes, Defocus and EdgeExtend compute what Nuke computes (integrator tick)

- **What was done:** `main` `1045deae` -> `d66b7cca` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `d66b7cca`: Ran 8256 tests in 601.077 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 9 (2D behaviour), step C of 3: the draw nodes, Defocus and EdgeExtend

- **What was done (evidence):** ten nodes read against the Nuke 17.0 pages and their kernels; three commits (Rectangle, Radial, Ramp; Noise, Grid, Grain, Dither; Defocus, ZDefocus, EdgeExtend).
  `tools/nuke_fidelity_cases_filter.py` gains the draw and defocus cases; `tests/test_2d_behaviour_draw.py`: the cases, every paired knob, tab order, tiles, older documents bit for bit. Targeted gate green.
- **Defocus:** aspect ratio now widens the disc without changing its height (the older node narrowed the height); new `scale` (splits into width and height). **ZDefocus:** new `zdefocus_math`
  (direct, depth, far = 0 and negatives), `size`; direct blur is depth x size clamped to maximum. **EdgeExtend:** new `source_premultiplied`, `premultiply`, `erode`. Older nodes keep their pixels (`legacy_defocus`, `legacy_zdefocus`, `legacy_edge_extend`, set on load; no schema bump).
- **Overall fidelity** 87.4 -> 87.6 over this commit (previous formula 92.0). **Not done:** Defocus quality and method; ZDefocus blur curve, layers, filter shapes (the page gives no formula); EdgeExtend matte choices, detail amount, edge mask, and the Channels row still trails its main tab.

## 2026-11-02 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step B of 3: Transform, Reformat, CornerPin and Crop compute what Nuke computes (integrator tick)

- **What was done:** `main` `f8775c44` -> `81851da6` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `81851da6`: Ran 8150 tests in 575.565 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 - lane 9 (2D behaviour), step B of 3: Transform, Reformat, CornerPin, Crop, Mirror, Position, Tile, AdjustBBox, BlackOutside

- **What was done (evidence):** read the Nuke 17.0 pages and the kernels on both paths; `tools/nuke_fidelity_cases_filter.py` gains 61 cases (49 pass, 0 fail,
  12 not scored with a reason); `tests/test_2d_behaviour_transform.py` (105 tests): the cases, every paired knob, tab order, tiles, older documents bit for bit.
  Older tests that pinned the old readings now set the `legacy_*` flags (group 2c5, 3b, lut/tile d3, phase A, tracker, handle UI, gpu2d). Targeted gate green.
- **Nodes with behaviour checked:** nine of nine (before: 0 scored on Transform, Reformat, Crop, Mirror, Tile, AdjustBBox; Position, CornerPin, BlackOutside had none).
  Overall fidelity 85.0 -> 87.0 (previous formula 91.3 -> 91.6).
- **Kernels changed (flags in `nodebased/transformbehaviour.py`, set on older saved nodes, no schema bump):** Transform axes, edges (black outside / clamp), shutter window; Reformat layout, box shape,
  clamp and black outside; CornerPin edges and shutter; Mirror window; Position direction and fill; AdjustBBox fill; Tile filter. New: Tile's filter, Transform and Reformat split scale.
- **GPU:** the 2D device path pins the older edge readings, runs Nuke's black outside off on the device, and sends clamp and a split scale to the CPU (tests/test_gpu2d.py OK on the RTX 3080 Ti, the AMD integrated adapter and llvmpipe).
- **Knobs unpaired:** Transform scale_x, scale_y, scale_mode (older per-axis scale); Reformat pixel_aspect (changes no pixels).
- **Not done / Needs Gonzo:** CornerPin direction enum vs Nuke's invert checkbox; Crop intersect default; Notch filter;.

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step C of 3: the keyers and the channel nodes compute what Nuke computes (integrator tick)

- **What was done:** `main` `b6a8835d` -> `59b98b2a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `59b98b2a`: Ran 8044 tests in 587.888 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02, 2D behaviour step C of 3 (Lane 2, Claude Sonnet 5.5): Keyer, HueKeyer, ChromaKeyer, Difference, ScreenKeyer, IBKColor, IBKGizmo, Copy, ChannelMerge, Shuffle

- **What was done (evidence):** read the Nuke 17.0 pages of the ten nodes and the kernels on both paths; `tools/nuke_fidelity_cases_color.py` gains 88 cases
  (46 pass, 0 fail, 42 not scored with a stated reason: Keyer 11 of 17, HueKeyer 5 of 8, Difference 2 of 6, Copy 8 of 11, ChannelMerge 13 of 16, Shuffle 7 of 8 scored;
  ChromaKeyer, Keylight, IBKColor and IBKGizmo have only not-scored cases because no page gives a formula). `tests/test_2d_behaviour_keyer.py` (78 tests): the cases,
  every paired main-tab knob, tiles of 7, 16 and 32 with and without a mask, older documents bit for bit (HueKeyer, ChannelMerge), and the stated words of the
  unscored nodes as properties of the pixels. Targeted gate (the module, audit, knowledge, tiles, bypass, knobs, tiers, core, split knobs, panel and label tests, and every
  module that names these kinds): 540 tests OK.
- **Nodes with behaviour checked:** 6 of 10 (before: 0): Keyer, HueKeyer, Difference, Copy, ChannelMerge, Shuffle. Overall fidelity 91.2 -> 91.3; nodes with behaviour checked 30 -> 36 of 167.
- **Kernels changed (each behind a switch an older document gets on load by `ops2d_chrome.upgrade_nodes`; no schema bump):** HueKeyer keyed a hue window (centre, width,
  softness, saturation range); Nuke's keys from two curves over hue (amount, saturation threshold), alpha = 1 - amount while invert (on by default) is on: now the curves,
  the window behind `older_hue_window`. ChannelMerge used the Merge vocabulary (over by default) and its multiply/divide let two negatives turn positive; now the guide's
  fourteen operations in `channel_operation` (union by default), the old choice behind `older_channel_operations`. ChromaKeyer's `premultiply` default is on (the page says
  so; a saved graph keeps its stored value). Keyer, Difference, Copy, Shuffle, ScreenKeyer, IBKColor and IBKGizmo were right where the pages say anything.
- **Knobs unpaired:** HueKeyer's hue_center, hue_width, hue_softness, sat_min, sat_max (extra, read only with Older hue window); ChannelMerge's old `operation` (extra).
- **Rule 5:** layouts and splits were built in step S for these nodes (tests/test_split_knobs.py has one order test per node, updated for HueKeyer and ChannelMerge). The Color knobs
  of these nodes (ChromaKeyer screen colour, alpha bias, despill bias, replace colour; Keylight's; IBKColor darks and lights) are already one param per channel or, for darks and
  lights, one scalar; no AColor knob remains to split. No step A node was left over.
- **Not done / open (Needs Gonzo):** IBKGizmo has 7 of 15 knobs: color, screen range, luminance level enable, autolevels, yellow, cyan, magenta, use bkg chroma have no
  param and no page says what they do; its screen types are green and blue where Nuke has C-blue, C-green and pick. IBKColor's darks/lights are per-channel offset and
  multiplier colours in Nuke (scripting names off, mult; the tutorial sets darks g to -0.08 and lights g to 0.75), NodeBased clamps the screen channel. ChannelMerge in/out/stencil
  read the page's lower-case a and b as the channel value (unsettled). HueKeyer's `channels` limits the written channels where Nuke computes the hue from them.
- **Remaining Color, Keyer and Channel nodes with no behaviour checked:** see the report; ChromaKeyer, Keylight, IBKColor, IBKGizmo (no published formula), plus the nodes of
  steps A and B that are all not-scored (HistEQ, MinColor) and the Channel/Color nodes outside lane 2's three steps (Remove, ShuffleCopy, ColorMatrix, ColorCorrect siblings).
- **Next owner:** Gonzo.

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the log, colourspace and lookup nodes compute what Nuke computes (integrator tick)

- **What was done:** `main` `b32de38b` -> `cc2bf235` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `cc2bf235`: Ran 7961 tests in 578.852 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02, 2D behaviour step B of 3 (Lane 2, Claude Sonnet 5.5), finished by Gonzo: CrossTalk, Expression, HistEQ, MinColor

- **What happened:** the step B worker committed Log2Lin/PLogLin, Colorspace and ColorLookup, then crashed at 1:34 AM (its CLI output passed the 8 MiB limit) with CrossTalk, Expression, HistEQ and MinColor written but uncommitted. Its own targeted runs of those modules were green (`run/tests-L2b-ex4.log`, `tests-L2b-mc4.log`); it had just started a full run.
- **What Gonzo did:** read the uncommitted work against the brief (all eight nodes have cases in `tools/nuke_fidelity_cases_color.py`), ran the targeted set (behaviour A and B, cache correctness, fidelity audit and `--check`, knowledge, split knobs, the parity modules naming these kinds, knobs, expressions, tiers, chrome: 395 tests). One failure: MinColor's channels knob was labelled `Search`, which breaks step S's rule that a node whose Nuke table opens with channels shows `Channels` first; relabelled (help text keeps the meaning).
- **New here:** `nodebased/nukeexpr.py` (Nuke's C-like expression language, parsed to a tree and evaluated in numpy, never as Python source), `nodebased/mincolorknobs.py` (MinColor adds its stored delta). Older documents keep their pixels through `older_expression_syntax` and HistEQ's 256 entries (`nodebased/behaviour2d.py`).
- **Not verified by the worker's own report:** the worker never wrote its report, so what it left out of these four nodes is unstated beyond the cases marked not scored. The full suite runs at the next tick.
- **Next owner:** Gonzo; step C of 3 (the Channel nodes).

## 2026-11-02 — continuous mode merge: Lane 9 (2D behaviour B, Claude Sonnet 5.5) step A of 3: Blur, the erodes and the convolution filters compute what Nuke computes (integrator tick)

- **What was done:** `main` `51ffc5d1` -> `0f99ba86` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0f99ba86`: Ran 7912 tests in 578.790 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-01 - lane 9 (2D behaviour), step A of 3: Blur, Erode, ErodeFilter, Median, Sharpen, Soften, Glow, Convolve, Matrix, Laplacian, EdgeDetect, Emboss

- **What was done (evidence):** read the Nuke 17.0 pages of all twelve nodes and NodeBased's kernels on the evaluator and tile path; wrote
  `tools/nuke_fidelity_cases_filter.py` (34 cases: 20 pass, 14 not scored with a stated reason, 0 fail) and `tests/test_2d_behaviour_filter_a.py`
  (cases, every paired main-tab knob, tiles of 7, 16 and 32 with and without a mask, old documents bit for bit). Targeted gate (the module, the audit,
  knowledge, tiles, bypass, knobs, tiers, core, panel and label tests, and every module that names these kinds): 590 tests OK.
- **Nodes with behaviour checked:** scored cases on 6 of 12 (before: 0) (Blur, Erode (fast), Median, Glow, Matrix, Laplacian); the other six (Erode (filter),
  Sharpen, Soften, Convolve, EdgeDetect, Emboss) have only `not scored` cases because the guide states no formula, with their stated facts tested in the
  module. Overall fidelity 90.9 -> 91.0.
- **Kernels changed (each behind a `legacy_*` flag, `nodebased/filterbehaviour.py`; an older document is given the flag on load by `ops2d_chrome.upgrade_nodes`, no schema bump):**
  Erode (fast): positive size grew dark pixels, the guide says it grows light ones (default size 1 -> 0). Erode (filter): negative size did nothing, the guide
  says it grows the light areas. Median: size was a radius, the guide's size 3 is the 3 x 3 area (default 1 -> 3). Matrix: results were not clipped, the guide says
  0 to 1. Laplacian: a fixed 3 x 3 cross, original minus blur, no knobs; now the blur at `laplacian_size` (filter, quality, crop) minus the original. Convolve:
  read the top-left corner of the red channel at `kernel_size`; now the filter channel (alpha), `normalize`, `use_input_channels`, `filter_bounds`, centred on the
  filter's box at any size (FFT above 15 x 15), and a new Convolve is a whole-frame job (no tile path). EdgeDetect: gains `edge_blur_size`, `edge_erode_size`,
  `edge_blur_quality` (default 0, so old documents are unchanged).
- **Knobs unpaired:** none. Paired and now acting: Convolve 4, Laplacian 4, EdgeDetect 3.
- **Not scored (guide silent):** Gaussian sigma for a size, triangle and quadratic weights, the quality down-scale method, Sharpen/Soften's scale of amount and
  the minimum/maximum role, Glow's response curve, EdgeDetect operator scales and the missing `simple` detector, Emboss's formula and its `optype`, edge handling
  at the picture's border, Matrix weight orientation.
- **Open risk:** the Matrix clip and the Erode (fast) sign follow the guide's text; a real Nuke render could contradict either. Needs Gonzo: a Nuke reference
  pack render settles both, and the Sharpen, Soften, Emboss and EdgeDetect scales.
- **Not done:** steps B and C of the lane (Transform and Draw groups); default values that differ from Nuke's (Blur size 8 against 0, Sharpen amount 0.5 against 1,
  Soften size 4 against 1, Glow size 8 against 15) stay, so existing scripts and tests that rely on them keep working.
- **Next owner:** Gonzo for step B. Branch `openclaw/nb-2d-behaviour`.

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step B of 3: the log, colourspace and lookup nodes compute what Nuke computes (integrator tick)

- **What was done:** `main` `aabf15e3` -> `e7b07b5a` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e7b07b5a`: Ran 7823 tests in 582.990 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step S of 4: per-channel knob splitting and Nuke's knob layout for the colour nodes (integrator tick)

- **What was done:** `main` `aa6820f3` -> `1f79a8c1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `1f79a8c1`: Ran 7807 tests in 580.428 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-02, 2D behaviour step S (Lane 2, Claude Sonnet 5.5): per-channel (split) knobs and Nuke's layout for the colour nodes

- **What was done, evidence:** a knob can hold one value per channel, like Nuke's AColor (r g b a) and WH (w h) knobs. `core.SPLITTABLE` lists the splittable params, `core.split_values(params, name, n)` reads a number or a list as a float32 array, validation applies per entry and refuses a list of the wrong length when it is set, `KnobGroup("split", ..., channels="rgba"|"wh")` lays one out, and the Properties panel gets a split button (4 or 2) with r g b a fields on one wrapping row, one undo step per toggle. Applied to Grade (blackpoint, whitepoint, lift, gain, multiply, offset, gamma), ColorCorrect (saturation, contrast, gamma, gain, offset for master, shadows, midtones, highlights) and Multiply, Add, Gamma, on the evaluator and the tile path (both call the same kernels; tiles of 7, 16 and 32 equal the frame with and without a mask). Pixel tests: each channel equals the unsplit node holding that entry, equals the Grade formula and the fidelity cases' ColorCorrect reference per channel, equal entries equal the unsplit node. The main tab of 26 nodes follows Nuke's knob order (tests/test_split_knobs.py, one test per node). `tests/test_split_knobs.py` has 80 tests.
- **Schema:** bumped to 29 (main and lane 9's branch were both at 28). Only a document that holds a list needs it; documents with numbers load and render as before. Gonzo: renumber if another lane lands a 29 first.
- **Paths that take a list:** the dispatcher `set`/`create`, `describe` (a new `split` block), the agent loop's validator, save/load/undo, expressions reading a split knob (they see entry 0). No Nuke script import or export exists in the code base; the Nuke reference pack (`tools/make_nuke_reference_pack.py`) writes only the cases' numbers. Scalar-only: nothing else.
- **Left out / follow-ups:** animation (a split knob cannot be keyed and a keyed knob cannot be split; core refuses both, the panel disables the button with a tooltip) and expressions driving a split knob; a split Grade or ColorCorrect runs on the processor, not the GPU island (`Unsupported`). Alpha takes its entry only when the node's channels include alpha and a knob is split; an unsplit Grade/ColorCorrect still passes alpha through whatever `channels` says (a pre-existing difference from Nuke, kept so old documents render the same). ColorCorrect's alpha handling (every entry but saturation, weights from the colour's luminance) is NodeBased's reading: no source states it. Blur and Erode size (WH) are lane 9's to adopt: `run/split-knobs-howto.md`.
- **Layout:** the shared mask strip moved to Nuke's order and wording (Mask, Inject, Invert, Fringe, (Un)premult by, Invert) with no heading, `channels` first where Nuke puts it first; NodeBased-only knobs (Exposure, Offset follows alpha, Divide by alpha, the older-maths switches) sit after Nuke's knobs and before the strip; Keyer's A B C D and HSVTool's three ranges are one row each (new `captions` on an xy row); Grade lost its "Levels" heading; labels in Nuke's words for Grade (Blackpoint, Whitepoint), Log2Lin (Operation), PLogLin and ChannelMerge. HistEQ, MinColor and ScreenKeyer keep NodeBased's layout (Nuke's table has different controls or no entry). The Reference Guide's screenshots are not machine readable, so which knobs share a row is changed only where the knob table shows one Nuke knob NodeBased holds as several params. Colorspace keeps its distinct "Transfer/Whitepoint/Primaries In/Out" labels (Nuke shows three dropdowns all labelled "input"). tests/test_2d_parity_feedback_y3.py looked up Grade's first field as Exposure; it now finds it by name.
- **Next owner + concrete artifact:** Gonzo: merge, then lane 9 adopts the split for Blur/Erode size from `run/split-knobs-howto.md`; lane 2 step B.

## 2026-11-01 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) step A of 3: ColorCorrect, Saturation and the hue tools compute what Nuke computes (integrator tick)

- **What was done:** `main` `5015cef7` -> `e875bb16` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `e875bb16`: Ran 7727 tests in 572.739 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-01, 2D behaviour step A of 3 (Lane 2, Claude Sonnet 5.5): the colour grading nodes compute what Nuke computes

- **What was done — evidence:** seven nodes (ColorCorrect, Saturation, HueCorrect, HSVTool, Toe, SoftClip, Posterize) now have pixel cases for the behaviour Nuke has, not only knob pairings. 75 cases in `tools/nuke_fidelity_cases_color.py`: 66 pass, 0 fail, 9 not scored with the reason (a new fixture `C` in `tools/nuke_fidelity_cases.py` gives the HSV cases colour inside 0 to 1). Sources: the 17.0 guide where it states the behaviour (`guide`); where it states none, A. Geremia's measurements of ColorCorrect, Saturation and HueCorrect on Nuke 17, Foundry's "Correcting HSV" page and Natron's HSVTool page (`convention`, cited in each case). Nodes with behaviour checked: 12 -> 18 overall; of this step's seven, 0 -> 6 (Posterize's only case is not scored). Overall fidelity 90.851 -> 90.930 (Color group 92.849 -> 93.531); the cases are what moved, the knob counts did not.
- **Kernels changed, and what was wrong:** ColorCorrect divided by alpha before correcting (Nuke corrects the stored colour; `divide_by_alpha`) on the evaluator, tiles and the GPU island (WGSL reads one more table entry; `tests.test_gpu2d` and `tests.test_gpu2d_colorcorrect` on the RTX 3080 Ti, the AMD iGPU and llvmpipe). Saturation had one grey (0.2126...) and no luminance math: now Rec 709 (0.2125/0.7154/0.0721), Ccir 601, Average, Maximum (`luminance_math`, with an old-weights choice), and a negative saturation is allowed. HueCorrect: lum was scaled by saturation (now a plain gain), suppression curves limited to the largest of the other two with 0 meaning nothing (now `min(others) + sup * (c - min(others))`, 1 meaning nothing, default curves 1), old grey (`older_curve_maths`). HSVTool: saturation and brightness adjustments scaled by (1 + adjustment), now added (`adjust_by_scaling`). SoftClip: preserve hue and brightness used the old grey (`older_grey`). Toe: default lift 0.05 -> Nuke's 0 for a new node (saved nodes store their own value).
- **Older documents:** each change has a knob set by `ops2d_chrome.upgrade_nodes` (values in `nodebased/behaviour2d.py`) for a document that lacks it, so a saved graph renders bit for bit as before; `tests/legacy_behaviour_a.py` freezes the older kernels and `tests/test_2d_behaviour_color_a.py` compares against them, on the evaluator and through a graph. No schema bump (the additive upgrade path, like Grade's `offset_by_alpha`). Existing tests that asserted the older numbers were updated: `test_colorcorrect_nuke`, `test_2d_parity_group_4a`, `test_2d_parity_w2`, `test_2d_parity_t2`, `test_2d_parity_step_5b`.
- **Knobs:** none unpaired. Toe's mask-invert checkbox is now paired with the page's `invert`. Every paired knob of the seven nodes changes the pixels (`tests/test_2d_behaviour_color_a.py`); Saturation's and HueCorrect's two (un)premult knobs are shown to leave the pixels alone, as they must (both nodes are degree-one homogeneous in the colour).
- **Not done:** ColorCorrect's and HueCorrect's `mix luminance` and ColorCorrect's `test` overlay (Geremia describes the overlay's channels; not built); HueCorrect's saturation threshold fade (two published points do not fix it); the shape of HSVTool's rolloff, its default output matte (the 17.0 page reads as a default that writes the hue matte into alpha; NodeBased keeps the source alpha), colour replacement and the H/S/B masks; Toe's rolloff curve; SoftClip's logarithmic curve between its end points; Posterize's Colors (the page says bit depth, the node behaves as a level count). All reported as not scored in `docs/PARITY_2D_FIDELITY.md`. No comparison with a real Nuke render exists yet; the Nuke script text of the HueCorrect and HSVTool curve and range knobs is unverified.
- **Next owner + concrete artifact:** Gonzo, step B of 3 (the Keyer nodes) and step C (the Channel nodes) of this lane; the Nuke reference pack (`tools/make_nuke_reference_pack.py`) is the route to the pixel-truth column. Failure mode to watch: lane 9 may also take schema 29 or add a fixture to `make_fixtures`; this step did neither bump the schema nor touch anything but adding fixture `C`.

## 2026-11-01 — continuous mode merge: Lane 4 (Generative, Claude Sonnet 5.5) step A of 1: generated footage gets its own camera limit, and the M5 score card on real clips (integrator tick)

- **What was done:** `main` `a0c43634` -> `19b20f60` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `19b20f60`: Ran 7690 tests in 569.048 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-11-01 — continuous mode merge: Lane 4 (Generative, Claude Sonnet 5.5) step A of 1: the video model learns the scene's distance scale (integrator tick)

- **What was done:** `main` `c0f78612` -> `ddcd904e` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `ddcd904e`: Ran 7683 tests in 569.124 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31 — continuous mode merge: Lane 4 (Generative, Claude Sonnet 5.5) step B of 2: a local video model that follows the scene's camera path (integrator tick)

- **What was done:** `main` `db238339` -> `bec1489c` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `bec1489c`: Ran 7673 tests in 576.902 s, OK (skipped=37), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31 — continuous mode merge: Lane 6 (Fluids, Claude Sonnet 5.5) step A of 1: the smoke above the fire curls, through the existing up-res node (finish 1) (integrator tick)

- **What was done:** `main` `1ac2f1ad` -> `9602249b` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `9602249b`: Ran 7658 tests in 568.084 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31, 12:45 PM PDT — Lane 6 smoke up-res, finish 1: factor 4 rendered (frame-time gate far from met, default stays off)

- **What was done — evidence:** the left-out item that can run here, the factor-4 render, is in. Added `render_evidence.py --upres-render` (stills, grain, image structure, times; no 17-frame field pass: that pass at factor 4 ran 25 minutes without finishing, processor-bound, and was stopped by its own PID) and `--timing-from` (reads a saved `--upres-time on` output; the in-harness timing call of factor 4 failed once for a reason not found, the same call run alone succeeded). Measured on the RTX 3080 Ti, 960 x 540, 128 samples, x4 / turbulence 2 / swirl 8 / grain 2: bake 2.2 s coarse + 70.8 s up-res; frame time median of five **6.5 s (frame 8), 43.9 s (16), 79.7 s (24)** against 0.89 / 0.82 / 0.79 s without; frame-16 grain 0.0067 (gate 0.0071 met); image smoke structure 0.080 / 0.095 / 0.091 against 0.065 / 0.084 / 0.080 (x2: 0.069 / 0.085 / 0.081). Stills: finer, more broken smoke with short streaks, still no curls. Targeted tests (96: fire grain, fire shot, flame metrics, up-res, reinjection, sparse GPU volumes, fluid edits) pass on the RTX, the Radeon 8060S and llvmpipe; `tests.test_knowledge` passes.
- **Artifacts (committed):** `examples/fire_shot/render_evidence.py`, `benchmarks/fire_shot/fire-smoke-upres-render-x4-2026-10-31.json`, `docs/images/fire_shot/fire-smoke-upres-before-after-x4-2026-10-31.png` and `...-zoom-x4-2026-10-31.png`, `docs/FLUIDS_SPIKE.md` (paragraph "Factor 4, rendered") with the bundled copy. Large scratch under `scratch/nb-lanes/run/big-L6/x4` removed.
- **State:** PARTIAL. Still not done: **Windows** (no Windows machine in this lane; the CI build is the only route), **the real display** (stills read from saved images; no display is available to this worker), and **curls**: the node adds density detail only (vorticity stays 55 to 68), so the smoke is mottled, not curling. Factor 4 is far outside the gate (frame time 6.5 to 80 s against 0.75 s, bake 75 s against 30 s). The default stays off.
- **Next owner + concrete artifact:** Gonzo: judge `docs/images/fire_shot/fire-smoke-upres-before-after-x4-2026-10-31.png` on the real display, and decide between a cheaper render path for a finer sparse volume (frame time grows with the plume; processor-bound between uploads) or velocity-level swirl inside `FluidUpres3D` (the only route to real curls and to higher vorticity). Lane 6 owns both. Windows: read the CI run on the merge.
- **Failure mode:** a 17-frame metric pass written for the 2x volumes scales with the fine voxel count and stalls at factor 4; render-only evidence is the route for large factors. A subprocess called from a long harness hid its error text (captured output); run it alone to see the failure.

## 2026-10-31 — continuous mode merge: Lane 6 (Fluids, Claude Sonnet 5.5) step A of 1: the smoke above the fire curls, through the existing up-res node (integrator tick)

- **What was done:** `main` `a6e34ebc` -> `59f4eacf` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `59f4eacf`: Ran 7658 tests in 575.317 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31, 11:10 AM PDT — Lane 6 smoke up-res: the fire shot's smoke through FluidUpres3D (eddy contrast up, frame-time gate not met, default stays off)

- **What was done — evidence:** wired `FluidUpres3D` between the fire cache and the render of the shaped 64-cell shot behind `make_shot.build_document(smoke_upres=True|dict)` (off by default; the committed `fire_shot.nbcomp` is unchanged and a test pins it). The first trial showed the node unusable on a continuing fire: from frame 2 it only carried the first frame's fine state and rescaled it to the coarse mass, so the flame and temperature vanished (fine flame sum 0.004 against coarse 0.17 at frame 16; the render had no fire). **Fixed as a separate bug fix** (`fluid_upres.upres_sparse_grid(..., previous_source=, frame=)`, `cached_upres(..., previous_source=, reinject=)`, the evaluator's up-res loop): every frame is the coarse field sampled on the fine grid plus the carried detail (carried fine minus carried coarse), new smoke gets new curl-noise detail seeded once and carried, and the older carry-only path is what runs when `frame` is not given. A second small fix: the loop read coarse frames through a disabled cache and solved the coarse simulation twice (8.5 s of 28 s); it now uses the node's store. Swept factor 2 (and 4) against turbulence, swirl size and grain (`render_evidence.py --upres-sweep`, 11 rows) and compared the pick (x2, turbulence 2, swirl 8, grain 2) with the shot as it is (`--upres`): smoke density contrast 0.44 to 0.91 (x4: 0.96); grain at frame 16 0.0065 to 0.0061 (gate 0.0071 met); bake 8.5 s to 17.3 s (x4: 75 s; gate 30 s met for x2); flame-field core 17/17 and tongues 1.13 to 1.12 (met); **128-sample frame 0.80 to 0.90 s to 2.4 to 3.7 s (gate 0.75 s NOT met; the shot without up-res is already at 0.80 to 0.90 s)**. The stills show billowy, mottled smoke in place of a smooth haze (eddy-scale structure, not curls); the image-space structure measure rises only 1 to 8 %.
- **Artifacts (committed):** `nodebased/fluid_upres.py`, `nodebased/imaging.py`, `tests/test_fluid_upres_reinject.py` (7 tests), `examples/fire_shot/{make_shot,render_evidence,flame_metrics}.py`, `tests/test_fire_shot.py` (+3), `tests/test_flame_metrics.py` (+7), `benchmarks/fire_shot/fire-smoke-upres-2026-10-31.json` and `fire-smoke-upres-sweep-2026-10-31.json`, `docs/images/fire_shot/fire-smoke-upres-before-after-2026-10-31.png`, `...-zoom-2026-10-31.png` and the frame-16 pair, `docs/FLUIDS_SPIKE.md` section "Fluids 13 B smoke up-res" with the bundled copy.
- **State:** DONE as a measurement step; gate NOT met, up-res stays off by default. 209 targeted tests (fire, fire shot, flame metrics, up-res, reinjection, sparse volumes, fluid edits) pass on the RTX 3080 Ti, the Radeon 8060S and llvmpipe; `tests.test_knowledge` passes. **Unverified:** Windows; the real display (the stills were read from saved images); factor 4 rendering (only its bake time was measured); the cause of the extra render time from the noise (a raised density bound is the likely reason, not isolated); the emission-luminance core measure reads 16/17 frames with the up-res (frame 22, a noise-brightened hot-smoke cell), the flame-field and image measures are unchanged. Old documents with a `FluidUpres3D` node now reinject too; their cached up-res frames are rebuilt.
- **Next owner + concrete artifact:** Gonzo: judge `docs/images/fire_shot/fire-smoke-upres-before-after-2026-10-31.png` on the real display; if the look is worth it, the open item is the path tracer's cost on a finer sparse volume (1.6 to 2.0 s without turbulence, 3.3 to 4.8 s with it) or a lower-cost eddy route (velocity-level detail in the up-res node, which would also raise vorticity, not just density contrast). Lane 6 owns both.
- **Failure mode:** the node as shipped looked fine in its own unit tests (a closed, source-free blob) and failed on the first continuing fire; always check a new route on the fire shot's flame field before trusting the smoke numbers. The 17-frame emission-core measure is sensitive near its 1/3 threshold (the plain x2 resample alone reads 15/17); compare the flame field and the rendered rows with it.

## 2026-10-31 — continuous mode merge: Lane 8 (2D on the GPU, GPT-5.6 Terra) GPU playback: a second decode-ahead thread against the 24 fps gate (integrator tick)

- **What was done:** `main` `37cef67f` -> `090faa1d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `090faa1d`: Ran 7641 tests in 556.075 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31 — Lane 8 GPU playback: decode-ahead worker sweep

- **What was done — evidence:** Ran three fresh interleaved 48-frame 1080p passes for each Radeon native-half matrix row (one, two and three decode workers; one- and two-frame lead). One worker/two-frame lead won there at 21.806 fps median and 53.634 ms p95, against 20.380 fps and 59.682 ms for today. The same setting regressed RTX 3080 Ti (10.403 fps / 146.633 ms versus 10.770 / 134.487) and improved the four-core stand-in (11.885 / 103.329 versus 10.330 / 105.887). Automatic decode workers therefore start at four CPUs; the normal full-size lead remains one frame. `tests.test_playbackprefs` and `tests.test_knowledge` pass (39 tests).
- **Artifacts:** `benchmarks/2d-gpu/R6-1/playback-decode-workers-1031.json`, `docs/PLAYBACK.md` and bundled copy, `nodebased/playbackprefs.py`, `tests/test_playbackprefs.py`; raw traces are deliberate scratch at `scratch/nb-lanes/run/bench-L8-decode-workers-1031/`. All tracked artifacts are uncommitted at this note; raw traces are deliberate scratch.
- **State:** DONE. Radeon gate is unmet: 2.194 fps and 11.964 ms p95 remain. Decode-ahead stays the limiting producer (29.931 ms busy per accepted frame); the 192 MiB peak in-flight source-pool allocation remains inside its 819 MiB cap. Linux/48-frame evidence only; Windows and 240-frame behavior unverified.
- **Next owner + concrete artifact:** Gonzo integrates the branch tip and uses `benchmarks/2d-gpu/R6-1/playback-decode-workers-1031.json` for the next playback decision.
- **Failure mode:** A Radeon-only longer decode lead looked globally attractive. RTX measurements showed it worsens both fps and p95, so the lead default was retained.

## 2026-10-31 — continuous mode merge: Lane 6 (Fluids, Claude Sonnet 5.5) step A of 1: the smoke above the fire stops looking like sand (finish 1) (integrator tick)

- **What was done:** `main` `7e8a6983` -> `b7525282` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `b7525282`: Ran 7641 tests in 573.653 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31, 3:09 AM PDT — Lane 6 smoke grain, finish 1: the camera ray's first smoke collision is counted, not drawn (gate met)

- **What was done — evidence:** the left-out item of the smoke step (grain from the smoke's own scattering, 0.0084 floor) was built as a camera-flight estimator on both tracers. For a path's first flight
  (`turn` 0) with fire sampling on and one volume that glows, the first collision is the distance where the optical depth reaches `-log(1 - xi)` (`ptvolume.stratified_event`, midpoint sum, 2-voxel steps; the shader's
  `vol_first_event`), `xi` the base-2 radical inverse of the sample index rotated per pixel (`stratified_number`, `strat_number`; integers, same 24 bits on both), and the first vertex absorbs by weight (throughput x albedo) instead of by
  coin flip. Measured on the shot (`render_evidence.py --strat-grain`, RTX 3080 Ti, 128 samples, seeds 1 and 2): frame-16 grain 0.0141 (pre fire sampling) / 0.0130 (fire sampling) to **0.0065**; frames 8 / 24 0.0066 / 0.0066;
  `diffuse` 0.0096 to 0.0039; time 1.07 / 1.01 / 1.03 x the fire-sampling renderer; 2048-sample mean within 0.025 % of it. Per-ray collision distribution against delta tracking: CDF gap at most 0.2 % at fine steps,
  2 % at the box entry with 2-voxel steps (the half-voxel density ramp). 220 targeted tests (28 new, `tests/test_fire_smoke_strat.py`) pass on the RTX 3080 Ti, the Radeon 8060S and llvmpipe.
- **Gate:** grain at most 0.0071 **met** (0.0065); time at most 1.3 x met; mean met; adapters met.
- **Inference / not run:** the split "stratified collision 0.0130 to 0.0088, absorption by weight 0.0088 to 0.0065" is from an intermediate run that is not a saved benchmark. Smoke eddies are not visible (soft vertical streaks that the
  2048-sample reference shares); stills read from saved images, not the real display. No Windows run, no full suite, no CI read, no adaptive-sampling benchmark, two volumes and splat scenes keep delta tracking.
- **Artifacts (committed on `openclaw/nb-fluids-spike`, not pushed):** `nodebased/ptvolume.py`, `pathtrace.py`, `gpupathtrace.py` (volume header flag bit 16), `tests/test_fire_smoke_strat.py`,
  `examples/fire_shot/render_evidence.py` (`--strat-grain`, `--rerun`), `benchmarks/fire_shot/fire-smoke-strat-2026-10-31.json`, `docs/images/fire_shot/fire-smoke-strat-before-after-*-2026-10-31.png`,
  `docs/FLUIDS_SPIKE.md` "Fluids 13 B smoke finish" with the bundled copy. No schema bump. Large scratch under `scratch/nb-lanes/run/big-L6/strat` removed.
- **State:** DONE.
- **Next owner + concrete artifact:** Gonzo: judge `docs/images/fire_shot/fire-smoke-strat-before-after-2026-10-31.png` and the zoom on the real display; decide whether the later smoke vertices (second scatter on) are worth the same by-weight absorption.
- **Failure mode recorded:** a 128-sample image mean of the thin test volume swung 6 to 13 % between seeds (heavy-tailed knot fireflies) and looked like a bias; the 2048-sample means agree within 0.3 %, so judge means at many samples. The `volume_fire_light` 0 floor of the smoke step runs delta tracking (fire sampling is off) and is not comparable with the stratified path.

## 2026-10-31, 2:55 AM PDT — 0.48.0 published (Gonzo)

- **What was done:** confirmed v0.48.0 published on `4e91828f` (release `06ba1e51` + the Windows import fix `2dd58148`, retagged 12:40 AM after the first tag's Windows tests failed); 4 assets, SHA256SUMS matches GitHub's digests, release commit only bumps versions and adds identical notes, announcement posted by the tick, local build installed at 2:32 AM.
- **Not done / unverified:** real-display QA; the intermittent Windows DPX byte-order test failure on main (`a2db7871`) is not yet investigated.
- **Next owner + artifact:** Gonzo, `tests/test_write_formats.py` DpxTests on Windows before 0.49.0.

## 2026-10-31 — continuous mode merge: Lane 6 (Fluids, Claude Sonnet 5.5) step A of 1: the smoke above the fire stops looking like sand (integrator tick)

- **What was done:** `main` `5b0bcfc8` -> `8174b545` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `8174b545`: Ran 7613 tests in 552.081 s, OK (skipped=36), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

## 2026-10-31, 3:00 AM PDT — Lane 6 smoke grain, step A of 1: the fire is sampled as a light at smoke vertices (gate missed)

- **What was done — evidence:** split frame-16 grain by component on the shot as committed (RTX 3080 Ti, 128 samples): total 0.0141, smoke `diffuse` 0.0107, flame `emission` 0.0025;
  with `volume_fire_light` 0 the total is 0.0084 (so the fire's light on smoke is 65 % of the variance, and 0.0084 is the floor no fire estimator can go below). Built next-event
  estimation of the fire at smoke vertices on both tracers (`ptvolume.FireEmitter`, `fire_nee_sample`, `fire_nee_value`, `fire_transmittance`, `expected_emission`'s `mis`; the shader's
  `vol_fire_light` and friends): emitter = the coarse emission grid before its blur, balance-heuristic weights in volume measure against the phase-sampled continuation glow, a near-field
  fade (a vertex inside a dense knot was 3.6 x noisier without it), midpoint-rule shadow transmittance (ratio tracking was built first and ran 1.7 to 4.6 x noisier per fire sample; kept as the
  test reference `ratio_transmittance`). Per vertex the fire's light is 1.0 to 3.3 x quieter with the same mean. On the shot (`render_evidence.py --smoke-grain`, 128 samples): frame-16 grain 0.0141 to
  0.0130, time 1.01 to 1.04 x, 2048-sample mean within 0.045 % of the previous renderer. 280 targeted tests pass on the RTX 3080 Ti, the Radeon 8060S and llvmpipe (24 new in `tests/test_fire_smoke_nee.py`).
- **Gate:** grain half of 0.0141 **missed** (0.0130 against at most 0.0071); time 1.3 x met; mean met; adapters met.
- **Inference / not run:** that the remaining grain is mostly whether and where a camera ray scatters in the thin plume (collision presence, about `sqrt((1 - p) / (p N))` relative) is reasoned from
  the 0.0084 floor and delta tracking's structure, not separately measured. Stills read from saved images, not the real display. No Windows run, no full suite, no CI read. Smoke eddies are not visible in the close-up (the grain is still there).
- **Artifacts (committed on `openclaw/nb-fluids-spike`, not pushed):** `nodebased/ptvolume.py`, `pathtrace.py`, `gpupathtrace.py` (packed volume header 10 to 16 vec4), `volumerender.py` (`fire_coarse_emission`, `fire_light_grid` unchanged),
  `tests/test_fire_smoke_nee.py`, `examples/fire_shot/render_evidence.py` (`--smoke-grain`), `benchmarks/fire_shot/fire-smoke-grain-2026-10-31.json`, `docs/images/fire_shot/fire-smoke-grain-before-after-*-2026-10-31.png`,
  `docs/FLUIDS_SPIKE.md` "Fluids 13 B smoke" with the bundled copy. No schema bump.
- **State:** DONE as an estimator (unbiased, tested, no slower), BLOCKED on the gate. Not done: the camera-flight estimator below.
- **Next owner + concrete artifact:** Gonzo decides on one more Lane 6 step: an expected-value single scatter along the camera flight (stratified in-scattering samples weighted by transmittance, lighting evaluated at a
  few of them by the vertex-lighting code, the real collision's own lighting skipped) or a forced first collision with the camera ray split into scattered and passing branches; measure frame-16 grain against 0.0071 and
  render time against 1.3 x. Start from "What dominates, and why" in `docs/FLUIDS_SPIKE.md`.
- **Failure mode recorded:** fire sampling cannot reach the gate on its own; a draw from the whole fire is noisier than the continuation inside a dense emitter (fixed by the near-field fade); two volumes and splat scenes keep the previous estimator.

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
## 2026-11-04 — Lane 9, step P (partial): VectorDistort Blur Size smooths the map

**What was done — evidence.** VectorDistort's modern path now blurs its internal displacement/ST map before sampling the source once. A hard-edge plate with a smooth displacement field verifies that map blur leaves the warped edge sharp. The existing `legacy_vectordistort` migration preserves the old pixel-blur path for saved graphs. No knobs were added or unpaired. GridWarp/SplineWarp and Flare/Sparkles/VolumeRays remain open.

**Artifacts.** `nodebased/vectordistortknobs.py`, `nodebased/imaging.py`, and `tests/test_2d_behaviour_transform_n.py`; committed locally on `openclaw/nb-2d-behaviour`. The regenerated `docs/PARITY_2D_FIDELITY.md` and bundled copy are byte-identical and unchanged from tracked content.

**State.** Partial. The VectorDistort behavior module, neighboring warp/draw/filter tests, audit, tiles, bypass, knobs, tiers, core, knowledge, and required properties-panel checks passed. `tests.legacy_behaviour_d` contains no unittest tests. The audit remains 95.8 overall (95.4 under the previous formula); Draw 92.1, Filter 96.7, Transform 96.6. The six target nodes' fidelity/knob coverage is unchanged; the audit's inert-knob section remains empty (0). The new VectorDistort pixel assertion covers the guide's map-blur direction; no audit case was added because its fixture cannot express a measured temporal SmartVector.

**Next owner + concrete artifact.** Needs Gonzo: continue groups 1 and 2 from `scratch/nb-lanes/auto/briefs/L9-behaviour-P-1103.md`, then rerun the audit and targeted gates.
