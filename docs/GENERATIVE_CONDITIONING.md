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
- Render3D already writes the passes, including motion vectors and Cryptomatte IDs, to multichannel EXR.
- Missing: a defined export of the scene state for a model (the schema above, versioned), the generative node with
  its lock controls, and the verification step.

## 6. Order of work (proposal)

1. This document approved; VISION.md M4 and M5 amended to match.
2. The scene-state export as its own node and file format, with round-trip tests (export, reload, compare).
3. Verification tools on their own, tested against NodeBased renders where the answer is known exactly.
4. Only then the first generative node, conditioned through that export and checked by that verification.

## Origin

DiMo, 2026-09-30 10:39 AM: "Regarding the intrinsic data. I don't only want the 2d render data feeding the models,
I want the literal camera, object, time and space information of these objects, sims, and environments to inform
the direction the generative content needs to be derived from. If I give it a camera, and say it needs to be 100%
movement of the camera I provided, that should be how it works by default unless told otherwise through methods of
the DCC."

DiMo, 10:54 AM: "It would also get lighting information provided. Either literal light nodes, or HDRI spheres with
their latlong inputs to drive the lighting in the image if it is supplied."

## Lane 8 step notes

**2026-10-03, step C2 (partial).** `python -m nodebased.conditioning_verify scene.scene.json observations.json report` writes a JSON report and readable `.txt` summary. The observation JSON is keyed by frame number; image arrays can use `{"array":"name"}` references into the adjacent NPZ. It measures camera transform/FOV differences, compares observed object-ID mask centroids with the exported ID pass, and compares supplied light/shadow direction measurements and frame colour balance. Reports carry per-binding lock state and pass/fail, and say explicitly that light intensity is not checked. CPU tests cover exact matches, a 2-degree camera rotation and a 20-degree key-light direction change. The camera solution and light/shadow direction measurements must currently be supplied by the caller; deriving these from a plate with NodeBased's Tracker and analyzing shadows remain open. See `tests/test_conditioning_verify.py`.
