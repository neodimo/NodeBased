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
version; acceptance for depth, normals, motion, IDs, camera pose, text and reference frames; maximum width,
height and frame count; accepted colour spaces; local/remote execution; and output frame and honoured-control
claims. `nodebased.generative.ProviderDescription.load()` loads a description. The single provider entry point
is `nodebased.generative.generate(manifest, scene_state, output_pattern, provider, ...)`; it checks frame and
image limits and rejects supplied text or reference inputs that the selected provider cannot honour. The
Generate node's panel names every control as used or ignored before invocation. Generated EXR sequences can
be loaded by `ConditionedRead`; the verifier labels an intentionally unconditioned motion check
"not conditioned" and excludes it from the verdict.

Two deterministic CPU stand-ins establish the contract without a model or network:

- **reproject 1.0** is local, accepts ACEScg sequences up to 8192 by 8192 and 10,000 frames. It forward-splats
  beauty using exported pixel motion and metric depth as a nearest-surface z-buffer; the validated SceneState
  camera pose supplies the camera context encoded by those vectors. It honours depth, motion and camera pose,
  ignores normals, IDs, text and reference frames, and returns RGBA OpenEXR frames.
- **null 1.0** has the same local limits and colour space, copies beauty unchanged, and honours no controls.
  It is the deliberately unconditioned baseline.

A real provider implements this Python entry point and supplies a description that accurately reports its
capabilities and honoured controls. A real model is a later, separately approved step; DiMo decides which
provider comes first. These stand-ins establish plumbing and reporting only, not model quality.

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
separately approved step for DiMo to choose.
