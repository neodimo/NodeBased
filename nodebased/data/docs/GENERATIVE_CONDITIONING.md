# Generative conditioning: the scene is the brief

## 1. The rule

A generative node in NodeBased is conditioned on the scene itself: cameras, objects, simulations, lights,
environments, time and space. Rendered passes (depth, normals, motion, IDs) travel with it as supporting evidence;
they never replace the scene data.

Everything the artist supplies is binding by default. A supplied camera is followed exactly; supplied object
motion is followed exactly; supplied lighting is followed. The generated result departs from any of them only
through an explicit control on the node or in the graph. Nothing is loosened because a model preferred otherwise.

What the artist does not supply is free, and the output says which parts were free.

## 2. What the conditioning export contains

One export per shot, frame-accurate over the shot's range, in the scene's own conventions: Y-up, one scene unit is
one metre (USD and Alembic imports are converted to this on load, docs/3D_FOUNDATION.md), frames at the project fps.
The scene is right-handed. Transforms use column vectors and the matrix is `T(position) @ T(pivot) @ R(order) @ S @ T(-pivot)`; `R(XYZ) = Rx @ Ry @ Rz`, so the rightmost rotation acts first (`nodebased/scene3d.py`, `Transform3D.matrix()`).

**Time.** First and last frame, fps, and for each frame its number and time in seconds. Every item below is sampled
per frame; a static item says so once.

**Camera** (Camera3D, or one imported from USD, Alembic or glTF), per frame:
- world transform (position, rotation with its order, or the 4x4 matrix) and roll;
- lens: vertical field of view, focal length in millimetres, horizontal and vertical aperture in millimetres
  (Nuke's film-back model, `nodebased/filmback.py`), near and far planes;
- depth of field: f-stop, focus distance, aperture blade count;
- lens distortion (`camera_data.lens`): `model` `brown_conrady`, `units` `film_back_half_height` (1.0 is half the
  format's height, origin at the format's centre, x right and y up), radial `k1` `k2` `k3`, tangential `p1` `p2`,
  `principal_point` `[cx, cy]`, `anamorphic_squeeze` (the model runs on the squeezed image) and `distorted`,
  false when every coefficient is zero. The pixels of the passes are those of the ideal camera, so a provider
  conditions on undistorted images and redistorts what it returns with this lens to land on the plate. The
  equations are in docs/3D_FOUNDATION.md "Lens distortion";
- output resolution and pixel aspect.

**Objects**, per frame, for every geometry and every instance of an InstanceSet:
- stable ID (the same value the Cryptomatte and ID passes carry), node name and type;
- world transform and world-space bounding box;
- for deforming meshes, the vertex positions or a reference to the cached geometry per frame.

**Simulations**, per frame: particle positions, velocities and IDs; fluid surface meshes or volume references;
rigid-body transforms and velocities. Large data is referenced by cache path, not inlined.

**Lighting**, per frame:
- every Light3D with its type (Directional, Point, Spot, Rect, Disc, Sphere), world position and aim, colour,
  intensity, cone and penumbra angle, falloff, area size, and whether it casts shadows;
- every Environment light: the equirectangular (lat-long) image itself at full scene-linear range, plus its
  rotation and blur. The model receives the image, not a summary of it.
- the colour space of all colour values (the project's working space, ACEScg by default).

**Passes** (supporting evidence), per frame: beauty, depth, normals, motion vectors, object IDs, and the camera
used to render them. These let a model check its own spatial reading against the scene.

## 3. Binding and the controls that loosen it

Each binding has one control on the generative node, and each defaults to fully locked:

| Binding | Default | Loosened by |
| --- | --- | --- |
| Camera motion | locked: 100 % of the supplied camera | `camera_freedom` 0 to 1, or keyed |
| Object motion | locked per object | `object_freedom` per object ID |
| Lighting | locked when any light or environment is supplied | `lighting_freedom` 0 to 1 |
| Layout (what is where) | locked | `layout_freedom` 0 to 1 |

A model that cannot honour a lock at the requested strength says so before it runs. It does not run and quietly
ignore the lock (VISION.md M5: "make unsupported controls visible rather than silently ignoring them").

## 4. Verification: how we know it held

Every generated shot is checked against the scene that conditioned it, and the result is stored with the shot.

**Camera.** Track the generated frames with NodeBased's own camera solve and compare against the supplied camera
per frame: position error in metres, rotation error in degrees, field-of-view error. A shot outside tolerance is
flagged or rejected; tolerances are set per project, defaulting tight.

**Objects.** Re-project each object's supplied position into the generated frames and compare with where the
object appears (via the generated ID or segmentation pass where the model provides one, else by tracking).
Error in pixels per object per frame.

**Lighting.** Harder, and this doc says where it cannot tell. Render the known geometry under the supplied lights
in NodeBased; compare dominant light direction, shadow direction and overall colour balance against the generated
frame. Stated plainly: this checks direction and colour, not exact intensity, and it needs visible surfaces and
shadows to work.

The report travels with the output: per binding, locked or loosened, the measured error, pass or fail.

## 5. Where NodeBased is today

- The scene graph already carries all of section 2 internally: Camera3D and imported cameras with film back and
  lens, per-frame transforms, InstanceSets, particles, fluids, rigid bodies, Light3D in seven types, and the
  Environment light with its lat-long image, rotation and blur.
- Render3D writes beauty and render passes, including motion vectors and Cryptomatte IDs, to multichannel EXR.
- SceneState export and CPU verification are implemented; the generative node with its lock controls remains future work.

## 6. Order of work (proposal)

1. This document approved; VISION.md M4 and M5 amended to match.
2. The scene-state export as its own node and file format, with round-trip tests (export, reload, compare).
3. Verification tools on their own, tested against NodeBased renders where the answer is known exactly.
4. Only then the first generative node, conditioned through that export and checked by that verification.

## 7. ControlBundle: synchronized image controls

The `.scene.json` mode of WriteGeo3D writes a `.controls/manifest.json` sidecar as part of the same export.
`nodebased.control_bundle.write_control_bundle(scene_state_path, output_dir, samples)` is also available for
already-rendered pass arrays. The exporter packages the CPU-rendered image controls for each sampled frame beside
the versioned SceneState contract. Each frame is a 32-bit float,
multichannel EXR; `manifest.json` is the authoritative schema and records the SceneState schema version, exact
frame range, pixel size/aspect, near/far clipping distances, every EXR channel, units, coordinate convention and
OCIO color-space role. The bundle contains:

| Layer | Units and convention |
| --- | --- |
| `R/G/B/A` | Beauty, scene-linear, tagged with the config's `scene_linear` role |
| `depth.Z` | Camera-space +Z distance in metres, clipped at the frame's near/far planes |
| `normals.X/Y/Z` | Unit normals; camera or world space is declared, with right-handed Y-up coordinates |
| `motion_forward.X/Y`, `motion_backward.X/Y` | Pixels per frame; +X right, +Y down, vectors point to next/previous frame; sampled at pixel centres |
| `object_id.id` | Cryptomatte uint32 ID bits preserved losslessly in float32; manifest maps hex IDs to SceneState object names |

The typed reader returns NumPy arrays wrapped with units, coordinate-space and color-space labels. Pass the
matching SceneState file to the reader: it checks the bundle version, schema version and SceneState file digest,
then checks manifest pixel dimensions before exposing data. Schema mismatch raises `SceneStateVersionMismatch`;
a different same-schema shot raises `SceneStateMismatch`. The EXR layer layout and JSON manifest are versioned
together so a consumer can reject a contract it does not understand.

Worked example for the lower-level API (the six arrays are the frame's beauty, depth, normals, forward/backward
motion and Cryptomatte object-ID pass; the renderer supplies them without color conversion):

```python
from nodebased.control_bundle import read_control_bundle, write_control_bundle

manifest = write_control_bundle(
    "shot.scene.json", "shot.controls",
    {1001: {"beauty": beauty, "depth": depth, "normals": normals,
            "motion_forward": forward, "motion_backward": backward,
            "object_ids": cryptomatte_id_float_bits}},
    first_frame=1001, last_frame=1001, normals_space="camera")
frame = read_control_bundle(manifest, "shot.scene.json")[0]
assert frame.depth.units == "metres"
assert frame.object_names  # hex Cryptomatte ID -> SceneState name
```

Samples must cover a contiguous range present in SceneState and exactly match each camera's resolution. The
motion fields use the convention already defined by `nodebased.motionblur.motion_vectors`: a per-frame screen
displacement, with the forward field pointing to the next frame and backward to the previous one. Object IDs are
the existing Cryptomatte name hashes, bit-preserved through EXR; background is zero. Data layers are Raw, and
beauty is the project's scene-linear OCIO role.

## 8. Provider contract and Generate

Provider descriptions are versioned JSON records under `nodebased/providers/`. Each declares its name and
version; acceptance for depth, normals, motion, IDs, camera pose, text, reference frames, `seed` and
`sampler_settings` (a description written before the last two existed reads as ignoring them); maximum width,
height and frame count; accepted colour spaces; local/remote execution; and output frame and honoured-control
claims. `nodebased.generative.ProviderDescription.load()` loads a description. The single provider entry point
is `nodebased.generative.generate(manifest, scene_state, output_pattern, provider, ...)`; it checks frame and
image limits and rejects reference inputs, a non-default seed or sampler settings that the selected provider cannot
honour. A text brief the provider does not honour does not stop the call: the shot runs and the text is recorded as
ignored (`ignored` in the reply, `ignored_controls` in the sequence's `generation.json` and in its provenance row),
so one call does the shot and the record says the text was not used. Colour space, frame count and size stay hard
refusals, since the provider cannot take those inputs at all. The
Generate node's panel names every control as used or ignored before invocation, and the MCP `generate`
and `loop` start replies carry the same `controls` map. Generated EXR sequences can be loaded by
`ConditionedRead`.

**Seed contract.** Generate has `seed` (a non-negative integer, default 0) and `sampler_settings` (a small
JSON object of at most 4096 bytes, default empty; the node stores it as JSON text). Both reach the provider
entry point (`generate(..., seed=0, sampler_settings=None)`). A provider whose description declares seed or
sampler settings ignored refuses a non-default value before work, like reference frames (text is the one advisory
control: recorded as ignored, not refused). The
sequence's artifact is a ZIP of the frames plus `generation.json`, which holds the provider name and
version, the provider-description ID, the seed, the sampler settings, whether each was honoured, the
SceneState and ControlBundle IDs and every option (the text prompt and the hashes of any reference
frames). Because the artifact ID is the hash of that archive, any change to one of those inputs gives a new
ID, even when a provider that ignores the change writes identical frames. `python -m nodebased.artifacts
provenance <id>` and the MCP `artifacts provenance` tool show provider, version, seed and settings on the
generated-sequence row. The stand-ins declare `seed` honoured (their output is bit-identical for every
seed, so a fixed seed reproduces the same bytes and the same ID) and `sampler_settings` ignored.

**What the verifier is given.** The verifier receives beauty plus every pass the generated EXR carries,
named like the bundle layers (`depth.Z`, `normals.X/Y/Z`, `motion_forward.X/Y`, `object_id.id`), each
compared with the exported pass. A check whose pass the sequence lacks (depth, motion, normals, or object
IDs when no tracked point answers) is labelled "not conditioned" and left out of the verdict, as the
intentionally unconditioned motion check always was. So is the camera check of a provider whose description does not
declare `camera_pose` honoured (its measurement stays in the report): the camera is solved from the picture's
pixels, which a resampled copy of the layout render cannot make exact (see `reproject` below).

**How the verifier measures the camera and objects.** Landmarks are the scene's own vertices that the camera sees
and that have corner structure (a straight edge or a faintly shaded face matches anywhere along it). The template
for each is cut from a render of the exported scene through the exported camera on that frame, with the exported
`ambient` (a Render3D setting that the SceneState records per frame), and matched in the observed beauty; the camera
is fitted to the matches and compared with the exported one. A picture that follows the SceneState matches at the
projected positions, so the plate rendered by the same scene scores zero error (the demo's check:
`examples/demo_shot/diagnose_verdict.py`, plate against its own SceneState passes camera, objects and lighting).
Templates taken from the observation's own first frame scored that frame zero by construction and drifted with the
dolly's change of scale (about 0.5 m worst on the demo plate). A light whose shadows are off casts nothing to
compare, so the shadow direction is not demanded when the reference render shows no shadow. A SceneState with no light has no key direction to
compare, so lighting then rests on colour balance alone. The unconditioned `null` baseline therefore gets
PASS on the checks beauty alone can answer (objects by tracking, lighting) and "not conditioned" on the rest,
the camera included: it is measured and reported (exactly, for a copy of the beauty) but `null` does not declare
camera pose honoured. The report also has a `determinism` section (provider, version, seed, sampler settings and
whether the provider declared seed honoured); it never changes the verdict. Generate takes the source plate on its required `image`
input and passes it through unchanged during ordinary evaluation; the explicit panel action writes the
provider result. Its provider description and bundle/control parameters are part of evaluator and tile cache
keys, so changing a setting invalidates the tap's cached result.

Two deterministic CPU stand-ins establish the contract without a model or network:

- **reproject 1.0** is local, accepts ACEScg sequences up to 8192 by 8192 and 10,000 frames. It forward-splats
  beauty using exported pixel motion and metric depth as a nearest-surface z-buffer; the validated SceneState
  camera pose supplies the camera context encoded by those vectors. Version 1.1: frame N is frame N-1's beauty carried by
  frame N-1's forward motion (forward motion points at the next frame, so version 1.0 produced frame N+1's picture
  under frame N's label, which the verifier rightly measured as a camera one frame off); the first frame is copied.
  It honours depth, motion and `seed`; it ignores normals, IDs, text, reference frames, sampler settings and camera
  pose, which only decodes the motion vectors. A resampled copy of the layout render measures up to 29 cm and
  1.7 degrees of field of view away from the exported camera on the 480 by 270 demo plate, far outside the verifier's
  1 cm, 0.25 degrees and 0.1 degrees, because pixel tracking at that size cannot place a camera that finely; the
  camera check therefore reads "not conditioned" for it. It returns RGBA OpenEXR frames.
- **null 1.0** has the same local limits and colour space, copies beauty unchanged, and honours only `seed`
  (recorded, no effect on the pixels). It is the deliberately unconditioned baseline.

The first real model is **sdxl_controlnet 1.2** (`nodebased/providers/sdxl_controlnet.py` and `.json`):
Stable Diffusion XL base with the fp16-fix VAE and ControlNet-Union (promax), local, on the NVIDIA card through
CUDA only (the RTX 3080 Ti; the AMD adapter and llvmpipe are not used). It honours depth, normals, motion, IDs,
text, `seed` and `sampler_settings` (`steps` 1 to 100 default 30, `guidance` 0 to 20 default 5.0,
`controlnet_conditioning_scale` 0 to 2 default 0.8, `control_mode` `depth_normals`, `depth` or `normals`,
`denoise_strength` 0.05 to 1 default 0.5; any other key is refused, and the artifact records the settings with
their defaults filled in). It ignores camera pose and reference frames. Limits are 1024 by 1024 and 240 frames; input and output colour space
is ACEScg.

- **Reprojection between frames (1.2).** Frames are made in order with the same seed (the loop's `vary_seed`
  changes the seed per attempt, never per frame). The first is text-to-image from its own depth and normals,
  and its picture starts a guide canvas. For each later frame the canvas is warped into it with the exported
  backward motion (`nodebased/temporal.py`, the same warp and mask the verifier's temporal line uses); pixels
  the warp cannot supply (disocclusions, an object ID that changed, the picture's edges, by the exported depth
  and IDs) are filled from that frame's own text-to-image picture, that becomes the canvas, and the canvas goes
  through an image-to-image pass at `denoise_strength`, with the same depth and normals control. The
  image-to-image results never go back into the canvas: in 1.1 each frame started from the previous generated
  picture, and the model's contrast and texture gain compounded over the passes until the picture broke up
  (flicker mean 19.0 against 15.4 without guidance). `denoise_strength` 1 turns the guidance off and
  makes every frame on its own, as 1.0 did. The verifier's motion, depth, normals and camera checks still read
  "not conditioned" (the generated EXRs carry beauty only); the temporal line measures the flicker.
- **Colour.** The model works in display sRGB. Depth and normals become 8-bit control images with their own
  rules: depth is inverse depth normalised with the nearest and farthest valid depth of the whole sequence
  (`255 * (1/d - 1/far) / (1/near - 1/far)`, near white, far and empty pixels black, so a surface keeps its grey
  from frame to frame); normals are camera-space unit vectors as `(n * 0.5 + 0.5) * 255`, with pixels that have
  no normal set to the flat colour (128, 128, 255). The longer side is resampled to 1024 for the model (both
  sides on the latent grid of 8) and the picture is resampled back to the plate size. The model's 8-bit sRGB
  picture is converted to scene-linear ACEScg (inverse sRGB encoding, then the Rec.709 to ACEScg matrix, through
  the fixed OCIO config) with alpha 1.
- **Process.** The model runs in its own interpreter, `NODEBASED_SDXL_PYTHON` (default: the provider
  environment kept outside the repository); the repository environment has no torch and never imports it. The
  entry point speaks the usual `generate` protocol and starts the model process with the job on disk. Weights
  come from the local Hugging Face cache, offline. The description declares `local_gpu`, so the worker does not
  put the address-space cap on the job (CUDA reserves terabytes of address space at start-up).
- **Determinism.** Fixed seed on a CUDA generator, deterministic algorithms on, TF32 off, no xformers. The same
  manifest twice gives byte-identical EXR frames on the RTX 3080 Ti (a test proves it and is skipped with a
  reason where there is no CUDA device). The fp16 pipeline runs with model CPU offload by default: the whole pipeline peaks near 10.4 GiB at the
  1024 by 576 working size, which fits the 12 GB card only when no other program holds it, and a fallback after
  an out-of-memory error changed the pixels, so the memory strategy never depends on what else runs
  (`NODEBASED_SDXL_OFFLOAD=never` keeps everything on the card, about twice as fast, for a card to itself).
  Reproducibility holds for the same card, driver and library versions; another GPU generation may differ in
  the last bits.

A real provider implements this Python entry point and supplies a description that accurately reports its
capabilities and honoured controls. These stand-ins establish plumbing and reporting only, not model quality;
`sdxl_controlnet` is the first model, and a hosted video model with camera control is the next, separately
approved step.

**Second local model: svd_image_video 1.0.** SVD-XT generates up to 25 frames from one RGB reference image. Generate uses an explicitly supplied reference image, or the first exported beauty frame when none is supplied. It accepts a seed and four sampler settings (`steps`, `fps`, `motion_bucket_id`, `noise_aug_strength`). Its capability card marks depth, normals, exported motion, IDs, camera pose and text ignored. `motion_bucket_id` sets overall motion amount; it is not the exported motion field or a camera trajectory. The generated EXRs carry beauty only, so the verifier measures appearance and camera from pixels but labels unavailable pass comparisons “not conditioned.” The model works at 1024 × 576 and its frames are resized to the requested plate; the output is scene-linear ACEScg. `NODEBASED_SVD_MODEL_DIR` must point at a local SVD-XT Diffusers snapshot containing the fp16 variants; `NODEBASED_SVD_PYTHON` selects a Python with torch, diffusers and Pillow (default: the existing local provider environment). The repository interpreter imports none of those model packages. Weights are read offline, the model runs in an isolated process, and model CPU offload plus UNet forward chunking and two-frame VAE decode keep its peak below the RTX 3080 Ti's 12 GB. `NODEBASED_SVD_RUN_INFO` optionally receives timing and CUDA memory statistics. This provider does not lock a camera, object or layout and is unsuitable for M5's strict-binding gate.

**Model choice.** [SVD-XT's model card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) says that it conditions on an input image and generates 25 frames. Its [Community License](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE.md) expressly permits local research and non-commercial use (commercial terms are conditional). The card recommends memory reductions; the [Diffusers SVD guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd) quantifies model offload, forward chunking and two-frame decode together at under 8 GB VRAM. This is a new conditioning input versus SDXL ControlNet, whose capability card rejects reference frames. It does not meet the more desirable trajectory or combined depth-plus-reference case.

**Runner-up: I2VGen-XL.** [Its card](https://huggingface.co/ali-vilab/i2vgen-xl) describes image and text conditioning and carries `license: mit`, but its final disclaimer says research/non-commercial only. The card offers no numeric VRAM requirement, so 12 GB suitability cannot be verified from it. We did not adopt a model with that licence contradiction and unverified memory fit.

**Runner-up: Wan2.1 I2V-14B 480P.** [Its card](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) declares Apache-2.0 and image-to-video conditioning. Its resource table is an embedded picture rather than a machine-readable number; the card only quantifies 8.19 GB for the *different* T2V-1.3B model, and describes 14B with offload on one GPU. A 12 GB fit for this I2V model is unverified, so it remains a candidate for a separately measured larger-card or quantified-offload experiment.

### Workers

Generate jobs run in a separate Python interpreter using a local socket with four-byte, network-order
length-prefixed JSON messages. A `Job` carries the provider description ID, ControlBundle artifact ID, options
and priority. The worker announces `submit`, streams `progress` (fraction and short text) and `log` lines,
accepts `cancel`, then returns `result` with an artifact ID or `failed` with a named error and log tail. The
UI remains responsive and keeps the previous successful artifact until the replacement completes.

The default worker limits are 2 GiB virtual address space (sized up from the frame size) and 30 minutes wall time per job; a provider whose description declares `local_gpu` runs without the address-space cap, since CUDA needs it. Cancellation is
cooperative; a process that ignores it is killed after a 1-second grace period. Worker stdout and stderr are
captured as a `worker_log` artifact and linked from the generated sequence provenance. Output frames are staged
and promoted only after success, so failed work does not replace the previous output.

A provider package is importable in the worker interpreter and includes its versioned JSON description plus
its Python entry point and dependencies. The deterministic built-ins use `nodebased.generative.generate`; an
installed provider entry point can be selected as `package.module:function` and receives options, a progress
callback and a cancellation event. Keep model/runtime dependencies in the worker environment; communicate only
serializable job values and artifact IDs across the process boundary. The bundled `reproject` and `null`
providers remain CPU-only and make no network calls.

## Queue and remote workers

`nodebased.jobs.Queue` stores chains and links in SQLite. A link names an installed `module:function`
operation, JSON options, input artifact IDs, an optional provider, priority, dependencies, and a retry cap.
Dependencies default to the preceding link in the chain, so an export → generate → verify chain starts each
link only after its predecessor has stored an artifact. A failed link is retried up to its cap; its final error
marks the chain failed and leaves dependent links unrun. Cancelling a chain cancels pending links and signals its
running worker. Pause stops new dispatch while active workers finish. Reordering changes chain priority.

Queue state and successful artifact IDs persist across app restarts. A completed link is reused when its
operation, options and resolved input artifact IDs have the same signature; changing an upstream artifact makes
the dependent link eligible to run again. Progress, current worker, elapsed time and output artifact ID appear in
the **Conditioning Queue** panel. Import a JSON chain using this shape:

```json
{
  "name": "shot 12",
  "priority": 4,
  "links": [
    {"name": "Export", "operation": "studio.shot:export", "options": {"shot": "12"}},
    {"name": "Generate", "operation": "studio.shot:generate", "provider": "reproject",
     "options": {"provider": "reproject"}},
    {"name": "Verify", "operation": "studio.shot:verify", "retry_cap": 1}
  ]
}
```

Each operation receives `(options, input_artifact_ids, progress, cancelled)` and returns bytes, JSON data, or
an existing artifact ID. Set `artifact_kind` in options when returning bytes/JSON; supported kinds include
`scene_state`, `control_bundle`, `generated_sequence` and `verification_report`. Both machines need the
operation's Python package. The provider description's `locality`
selects local versus remote dispatch when `provider` is supplied; the scheduler assigns remote work to the
currently least-loaded configured worker. `NODEBASED_REMOTE_WORKERS` is a JSON array of `{name, locality,
host, port}` records, and `NODEBASED_WORKER_SECRET` is the shared handshake secret. The TCP protocol verifies
artifact digests, transfers only the referenced artifacts needed by a task, and returns the produced artifact
and its provenance. Use a trusted network and keep the secret private.

Start a worker on a LAN machine with the same NodeBased package and operation code:

```powershell
$env:NODEBASED_WORKER_SECRET = "the shared secret"
py -m nodebased.workers serve --bind 0.0.0.0:8765
```

Configure the desktop scheduler with that machine's reachable LAN address and port. Animal (the Windows
machine) is the intended first LAN target; this workflow has not yet been tried on Animal or tested on Windows.
Automated coverage uses two local worker processes on distinct TCP ports and makes no model or external network
calls.

## Origin

DiMo, 2026-09-30 10:39 AM: "Regarding the intrinsic data. I don't only want the 2d render data feeding the models,
I want the literal camera, object, time and space information of these objects, sims, and environments to inform
the direction the generative content needs to be derived from. If I give it a camera, and say it needs to be 100%
movement of the camera I provided, that should be how it works by default unless told otherwise through methods of
the DCC."

DiMo, 10:54 AM: "It would also get lighting information provided. Either literal light nodes, or HDRI spheres with
their latlong inputs to drive the lighting in the image if it is supplied."

## Lane 8 step notes

**2026-10-03, step C2 (complete).** `python -m nodebased.conditioning_verify scene.scene.json observations.json report` writes one JSON report and readable `.txt` summary for the shot; the observation JSON is keyed by frame, with beauty and optional ID images in its adjacent NPZ. NodeBased's Tracker follows scene-space landmarks through the plate, then a CPU reprojection solve measures camera position, rotation and field-of-view errors. Object origins are reprojected and checked against the supplied ID pass, ID masks or Tracker points. The verifier freshly renders the exported scene on the CPU, estimates dominant light direction from its albedo/normals guides and the plate, measures shadow displacement and colour balance, and reports each binding's lock state and pass/fail. Lighting intensity is explicitly unchecked. Acceptance tests cover a known render, a 2-degree camera nudge on one frame, a 3-pixel object displacement and a rendered 20-degree key-light rotation. Camera solving requires six trackable, non-coplanar scene landmarks; direction and shadow checks need visible shaded surfaces and shadows. See `tests/test_conditioning_verify.py`.

**2026-10-03, step D1 (complete).** WriteGeo3D's SceneState export now writes a synchronized `.controls/manifest.json` sidecar and 32-bit multichannel EXRs for beauty, clipped metric depth, declared-space normals, forward/backward motion, and bit-preserved Cryptomatte IDs mapped to SceneState names. The manifest binds the bundle to the exact SceneState file and schema, camera dimensions, frame range, near/far, units, coordinates and the config's `scene_linear` role. Its typed reader refuses mismatched SceneState files, schema versions, pixel sizes, formats, channels or color tags. CPU round trips cover the animated plan-7 scene (three meshes, an instance set, particles, two lights and an environment), a reprojected surface depth, plane normal, motion displacement, object identity, and the OCIO tag. See `nodebased/control_bundle.py` and `tests/test_scene_state.py`.

**2026-10-03, step D2 (complete).** `ConditionedRead` uses the bundle manifest's shot frame range and
an explicit frame offset to align generated sequence frames. It centre-crops to fill the shot aspect
ratio, bilinearly resizes to the shot pixel dimensions, converts the tagged beauty through OCIO into
the project's ACEScg working space, and exposes beauty, depth, normals, forward/backward motion and
object IDs as named layers. A mismatched or missing colour-space tag is rejected. `VerifyConditioning`
adds CPU optical-flow residual mean and 95th-percentile errors in pixels per frame plus Spearman rank
correlation for tracked-landmark depth ordering. Its JSON now includes one per-shot `score_card` with
camera, objects, lighting, motion and depth lock state, error and pass/fail, and a headline PASS/FAIL;
the text report starts with the same verdict. Motion observations carry the next generated frame and
the matching exported `motion_forward` layer. Depth observations provide expected bundle depth and
generated apparent depth for each tracked landmark. Camera-solving requirements and unchecked light
intensity remain as described above. CPU tests cover a known passing shot, shifted imagery, inverted
depth order, layer exposure, resize and an incorrect colour tag.

**2026-10-03, step D3 (complete).** The Generate provider contract, capability panel and single Python
provider interface are in place. The local `reproject` stand-in z-buffers a deterministic forward motion
reprojection; `null` copies the beauty plate unchanged. Both declare limits, colour space, locality, outputs and
honoured controls. Tests exercise SceneState/ControlBundle export, both providers, ConditionedRead reloading the
output, motion capability visibility and early rejection of unsupported text. The verification score card
marks unsupported motion as "not conditioned" and leaves it out of the verdict. A real model remains a
separately approved step for DiMo to choose. Generate is a required-plate Write-like tap whose provider and
bundle settings invalidate its evaluator and tile cache entries; the panel action writes the sequence.

**2026-10-06, step W1 (Windows artifact locking; CI confirmation pending).** On Windows, artifact locks seek
to byte zero and lock it directly; they never read or write the locked byte because Windows enforces byte-range
locks on those operations. Nonblocking lock attempts retry for up to 30 seconds before naming the artifact in
a timeout. Linux continues to use `flock`. Queue shutdown closes its SQLite connections and joins worker
threads before temporary database files are removed. Windows CI confirmation is still required.

**2026-10-03, step E1 (complete).** Conditioning exports now receive content-addressed IDs with producer, version, source document, frame range, time and input links. SceneState and ControlBundle artifacts package their companion files; Generate stores a portable sequence archive and returns its ID; ConditionedRead can consume that ID; VerifyConditioning reports store their scene, bundle and sequence links. `python -m nodebased.artifacts provenance <id>` prints the full chain oldest first. The 2 GiB cache supports LRU collection while preserving registered references; the open graph refreshes its live artifact references automatically. Original export files remain available. CPU tests cover same-content deduplication, LRU order, GC pinning, missing IDs and the complete plan-7 scene chain. See `nodebased/artifacts.py` and `tests/test_artifacts.py`.

**2026-10-03, step E2 (complete).** Generate now runs in an isolated, memory- and time-limited process over a
length-prefixed local JSON socket. Progress and log lines stream to the UI, cancel is available in the node
panel, and sequence output is staged until success. Worker logs are stored as provenance-linked artifacts.
Targeted CPU tests cover successful IDs, provider failures, memory exhaustion, cooperative and forced cancel.
The offscreen Generate panel reports progress and exposes cancellation. See `nodebased/workers.py`.

**2026-10-03, step E3 (complete).** The persistent SQLite queue runs dependency-ordered chains, retries failed
links to a cap, preserves matching completed links over restarts, and records chain provenance on outputs. The
Conditioning Queue panel imports chains and controls run, pause, cancel and priority. LAN workers extend the
length-prefixed worker messages with a shared-secret handshake and content-verified artifact transfer. CPU tests
cover export → generate → verify ordering, failure and restart behaviour, cancellation, and two separate worker
processes. Animal/Windows has not been tried. See `nodebased/jobs.py` and `nodebased/queuepanel.py`.

**2026-10-04, step L1 of 2 (bounded generative loops with saved candidates and hard limits).** `nodebased/loops.py`
adds a persistent bounded Generate → VerifyConditioning orchestrator over the existing job queue. Attempt and
estimated-spend caps are enforced before dispatch; provider version/options, input/output IDs, score verdict and
terminal state are recorded. Requested reference feedback requires declared provider capability, and generated
artifacts remain referenced across later failures. CPU tests cover pass, feedback, limits, cancellation, provenance,
and reopen/history behavior. The included `null` provider and CPU score-card verifier are deterministic local
operations; no paid or real model is invoked. Other providers still require their own installed adapter.

**2026-10-06, step W1 of 1 (Windows artifact and queue shutdown).** Artifact blobs are immutable once
published: a staged file is linked into its content-addressed name only if that name is still absent, so
writers never replace a blob a reader may have open. Per-artifact in-process locks serialize Windows
threads alongside the existing cross-process lock. Atomic metadata replacement and temporary-file cleanup
retry brief Windows sharing violations. Queue and loop database helpers now commit or roll back and close
each SQLite connection at the end of its operation; closing the queue waits for worker threads, and closing
the panel waits for its runner. Linux targeted tests and loaded artifact stress passed. Windows CI remains
the platform proof for these file-sharing rules.

## 9. Artifacts and provenance

Conditioning outputs are kept in a content-addressed artifact store. SceneState exports, ControlBundle manifests, provider descriptions, generated sequences, and verification reports receive SHA-256 IDs; identical bytes share one stored object. The cache uses the user's cache directory by default; `NODEBASED_CACHE` changes the cache root. Its default budget is 2 GiB. `ArtifactStore.gc(budget)` removes least-recently-used unreferenced objects until the requested budget is met. The open graph automatically registers every artifact ID present in its node parameters; saved documents can also register references with `ArtifactStore.reference(document_path, ids)`. Existing export files remain in their original locations for compatibility. The export functions retain their path return by default; pass `return_artifact_id=True` to receive `(path, artifact_id)`.

Every artifact records its producer and version, input artifact IDs, source document, frame range and creation time. Generate returns an artifact ID for its zipped sequence; ConditionedRead can use that ID instead of a sequence path. VerifyConditioning records the scene and supplied bundle/sequence IDs. The node panels show generated and consumed IDs. To inspect the chain in dependency order, run `python -m nodebased.artifacts provenance <id>`; the oldest inputs appear first. A missing or evicted ID is reported by name.

## 10. Bounded generative loops

`nodebased.loops.LoopRun` persists an explicit finite sequence of `Generate` then
`VerifyConditioning` queue chains. A loop is not a render-graph edge. Its caller supplies SceneState,
ControlBundle and provider IDs/options, a positive finite attempt cap, a positive finite spend cap,
and a declared per-attempt estimate with units. Missing estimates are rejected; with an active cap,
the next attempt is refused before dispatch when its estimate would exceed the remaining budget.
These values are estimates only; NodeBased makes no claim about provider billing.

Each attempt records its seed. By default every attempt reuses the loop's `seed` (default 0), so a rerun
reproduces the first; with `vary_seed` the seed of attempt *n* is `seed + n - 1`. The loop's `seed`,
`sampler_settings` and `vary_seed` are part of its identity, and status output shows the seed per attempt.

Loop states are `queued`/`running`, then one of `pass`, `provider_failure`, `verification_failed`,
`attempts_exhausted`, `budget_exhausted`, `feedback_unsupported` or `cancelled`. Each attempt
persists its exact input/output IDs, provider name/version/options, score-card verdict, terminal state,
and spend estimate/unit. A failing score card can schedule another bounded attempt. Its generated
sequence is included as an input only when requested feedback is declared by the provider (for
example `reference_frames`); unsupported feedback stops before dispatch. Successful sequence IDs
remain referenced if a later attempt fails. Reopening a loop reuses a completed queue chain with
matching artifact inputs and preserves the attempt history.

Deterministic CPU example (using already-exported artifact IDs in an application):

```python
from nodebased.artifacts import ArtifactStore
from nodebased.jobs import Queue
from nodebased.loops import LoopRun

store = ArtifactStore()
queue = Queue("loop-jobs.sqlite", store)
loop = LoopRun("loops.sqlite", queue)
run_id = loop.create(scene_state_id=scene_id, control_bundle_id=controls_id,
    provider_id="null", provider_options={}, max_attempts=2,
    max_estimated_spend=0.02, estimated_spend_per_attempt=0.01,
    spend_unit="credits")
result = loop.run(run_id)
print(result["state"], result["cumulative"], result["attempts"])
queue.close()
```

The bundled `null` provider generates the unchanged ControlBundle beauty frames, and the loop's
verification operation runs the CPU `verify_conditioning` score-card against those generated frames
and the exact SceneState. This example invokes no paid or real model and makes no network request.
Only the installed `null` and `reproject` provider implementations run locally; other providers need
an explicitly installed operation adapter. Declared reference-frame feedback is passed as an artifact
input to that adapter, and unsupported feedback fails before dispatch.

### Artist workflow and repeatable local demo

Open **Conditioning Queue** from the workspace docks. Enter the exported SceneState and ControlBundle
artifact IDs, choose `null` or `reproject`, then set the attempt maximum and estimated-spend cap. The
estimate per attempt and its unit are explicit; the panel adds the estimate before dispatch and stops
when the next attempt would exceed the cap. Provider controls show supported and unsupported labels
before launch. Selecting text or previous-result feedback asks the chosen provider to honour it; an
unsupported selection is recorded as a stop reason before that next attempt starts.

The compact attempt list keeps each attempt's state, exact control/artifact IDs, score verdict,
cumulative estimate, queue chain and stop reason. Double-click an attempt for its queue links,
provenance, failure text and worker log when available. Cancel selected loop requests cancellation of
its active worker and prevents further attempts. Select a completed attempt and choose **Use selected
result in ConditionedRead** to add a ConditionedRead node whose path is that attempt's generated-sequence
artifact ID. This action does not call Generate; earlier artifacts remain independently inspectable.
Loop and queue history are SQLite-backed and restore when the application is reopened.

Repeatable smoke demo, using only existing local stand-ins:

1. Export SceneState and ControlBundle from a small local scene with one frame (a cube at 160 by 120; the
   camera check needs enough pixels to track, so a 16 by 12 frame cannot answer it), then copy their artifact
   IDs from the export result/provenance view into the queue panel.
2. Choose `null`, set maximum attempts to `2`, spend cap to `0.02`, estimate to `0.01` with unit
   `credits (estimate)`, and start. The unchanged beauty sequence is generated locally and checked by
   the CPU score-card verifier; camera, depth, motion and normals are "not conditioned", the other checks pass,
   and the passing first score stops the loop after one attempt and one 0.01 estimate.
3. Reopen the project and select the loop chain to inspect the restored attempt. Choose that attempt's
   result for ConditionedRead; its path should be the displayed generated-sequence ID.

No paid provider, model download, external network request, model-quality claim or actual-billing claim
is involved in this demo. The estimate is user/provider configuration, not a measured charge.
