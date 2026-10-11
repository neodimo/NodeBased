"""Camera solve from the plate alone: the `solve_camera` op behind the CameraSolve3D node.

A learned monocular SLAM (DPVO / DPV-SLAM, see providers/camera_solve_dpvo.py) returns one camera pose per
frame and a sparse point cloud, up to scale and in its own frame. This module turns that into the document's
own terms:

* The solve is placed in the world by an anchor: the camera of the first solved frame (the node's anchor
  position, target and roll, the camera an artist sets by eye on frame 1). Frames after it follow the solve.
* The scale comes from the node's `unit_scale` (scene units per solve unit), or from the node's Survey: points the
  artist picked in the viewer, each tracked through the plate and triangulated with the solved cameras, and a
  distance between two of them or a position for each of three or more (similarity alignment). With fewer picks
  the solve stays up to scale and the record says so.
* The lens is the typed focal length, or, with `solve_focal` estimate and/or `solve_distortion` k1 / k1 k2, the one
  `lensrefine` recovers from the solver's own tracked points (a bundle adjustment after the SLAM solve). The
  recovered focal and radial terms go to the Camera3D and, with their residuals, into the solve record.
* The result is a Camera3D with a key on every solved frame (rounded to 0.001) and a ReadSplat3D that carries
  the sparse cloud as tiny coloured splats (the nearest existing scene type for points).

Two solves of one plate are not bit-identical (the solver's bundle adjustment adds floats with CUDA atomics; a
seeded run differs by about 3e-4 solve units, about a millimetre of camera position after placement), and
rounding the keys makes equal ids likely, never certain. The camera that feeds the conditioning export is
therefore the one solve kept in the document: the op runs on request, never when the graph is evaluated, and
re-running it replaces the keys (and so the export ids) on purpose.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import filmback, lensmodel, lensrefine, scene3d, track3d
from .providers import camera_solve_dpvo as dpvo

KEY_DECIMALS = 3
CAMERA_PARAMS = ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll")
TARGET_DISTANCE = 1.0          # the look-at target sits this far (scene units) along the optical axis
SPLAT_RADIUS = 0.02            # scene units, the size of one cloud point
SH_DC = 0.28209479177387814    # degree-0 spherical harmonic basis: colour = 0.5 + SH_DC * f_dc
FOCAL_MODES = ("known", "estimate")
PICK_PATTERN_RADIUS, PICK_SEARCH_RADIUS = 8, 16      # tracker window for a surveyed pick
PICK_MIN_CONFIDENCE = 0.6                            # normalised correlation below which a tracked position is dropped
PICK_MAX_VIEWS = 12                                  # frames a pick is triangulated from (evenly spaced over the confident ones)


def pose_from_keys(tx, ty, tz, target_x, target_y, target_z, roll=0.0):
    """Camera-to-world 4x4 in OpenCV axes (x right, y down, z forward) of a Camera3D's position, target and roll."""
    camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(tx, ty, tz)), scene3d.Vec3(target_x, target_y, target_z),
                            45.0, roll=float(roll))
    eye, view = scene3d._view_basis(camera)
    right, up, back = (row.astype(np.float64) for row in view)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, -up, -back, eye
    return pose


def keys_from_pose(c2w):
    """Camera3D position, target and roll of a camera-to-world pose in OpenCV axes (inverse of pose_from_keys)."""
    eye, forward, right = c2w[:3, 3], c2w[:3, 2], c2w[:3, 0]
    flat = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(*map(float, eye))),
                          scene3d.Vec3(*map(float, eye + forward * TARGET_DISTANCE)), 45.0)
    _eye, view = scene3d._view_basis(flat)
    right0, up0 = view[0].astype(np.float64), view[1].astype(np.float64)
    roll = math.degrees(math.atan2(float(right @ up0), float(right @ right0)))
    target = eye + forward * TARGET_DISTANCE
    return {"tx": float(eye[0]), "ty": float(eye[1]), "tz": float(eye[2]), "target_x": float(target[0]),
            "target_y": float(target[1]), "target_z": float(target[2]), "roll": roll}


def place_in_world(c2w, points, anchor, scale):
    """Poses and points of the solve in the world: frame 0 lands on `anchor` (a 4x4 pose), lengths times `scale`."""
    c2w = np.asarray(c2w, np.float64)
    first_inverse = np.linalg.inv(c2w[0])
    placed = np.einsum("ij,njk->nik", anchor, _scale_translation(first_inverse @ c2w, scale))
    local = (np.asarray(points, np.float64) - c2w[0, :3, 3]) @ c2w[0, :3, :3]       # R1^T (p - t1)
    return placed, (local * scale) @ anchor[:3, :3].T + anchor[:3, 3]


def _scale_translation(poses, scale):
    poses = poses.copy()
    poses[:, :3, 3] *= scale
    return poses


def fov_from_intrinsics(height, fy):
    return math.degrees(2 * math.atan(height / (2.0 * fy)))


def solve_camera_object(c2w, fov, roll_keys=None, lens=(0.0, 0.0)):
    """A scene3d.Camera at the pose `c2w` with the given vertical field of view and radial lens (k1, k2)."""
    keys = keys_from_pose(np.asarray(c2w, np.float64))
    return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(keys["tx"], keys["ty"], keys["tz"])),
                          scene3d.Vec3(keys["target_x"], keys["target_y"], keys["target_z"]), fov, roll=keys["roll"],
                          lens_k1=float(lens[0]), lens_k2=float(lens[1]))


def triangulate_points(c2w, intrinsics, width, height, points, lens=(0.0, 0.0)):
    """Each point's position in the solve (solve units) from its pixel observations on the solved frames.

    `points` is a list of lists of observations `{"frame": index into the solve, "x": px, "y": py}` (top-left
    origin, on the plate: the lens is undone first), at least two frames each."""
    fov = fov_from_intrinsics(height, intrinsics[1])
    found = []
    for observations in points:
        if len(observations) < 2:
            raise ValueError("each survey point needs observations on at least two frames")
        cameras = [solve_camera_object(c2w[int(row["frame"])], fov, lens=lens) for row in observations]
        pixels = [(float(row["x"]), float(row["y"])) for row in observations]
        try:
            xyz, _residuals = track3d.triangulate(cameras, pixels, width, height)
        except ValueError as error:
            raise ValueError(f"survey point could not be triangulated: {error}") from error
        found.append(xyz)
    return found


def survey_scale(c2w, intrinsics, width, height, distance, points, lens=(0.0, 0.0)):
    """Scene units per solve unit from two surveyed points.

    `points` is two lists of observations `{"frame": index into the solve, "x": px, "y": py}` (top-left origin,
    at least two frames each); `distance` is the points' real distance. Each point is triangulated from the
    solved cameras (solve units); the scale is the real distance over the triangulated one."""
    if not distance or distance <= 0:
        raise ValueError("survey distance must be a positive number")
    if len(points) != 2:
        raise ValueError("survey needs exactly two points")
    found = triangulate_points(c2w, intrinsics, width, height, points, lens)
    apart = float(np.linalg.norm(found[0] - found[1]))
    if apart < 1e-9:
        raise ValueError("survey points triangulate to the same place")
    return float(distance) / apart


# ----------------------------------------------------------------------------- the node's Survey

NEEDED_FOR_SCALE = "a distance between two picks"
NEEDED_FOR_SIMILARITY = "positions for three picks"


def parse_survey(value):
    """The node's `survey` knob (JSON text, a dict or nothing) as {"picks": [...], "distances": [...]}.

    A pick is {"name", "frame", "x", "y", "position": [X, Y, Z] or None}: a 2D point the artist picked in the
    viewer on one frame, and optionally where it is in the scene. A distance is {"a", "b", "distance"} between
    two picks by name. Raises ValueError with a sentence for anything malformed."""
    if value in (None, ""):
        return {"picks": [], "distances": []}
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError(f"survey is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("survey must be an object with picks and distances")
    picks, names = [], set()
    for row in data.get("picks", []):
        name = str(row.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"survey pick names must be unique and not empty (got {name!r})")
        names.add(name)
        position = row.get("position")
        if position is not None:
            position = [float(v) for v in position]
            if len(position) != 3 or not all(math.isfinite(v) for v in position):
                raise ValueError(f"survey pick {name!r}: a position is three numbers")
        pick = {"name": name, "frame": int(row["frame"]), "x": float(row["x"]), "y": float(row["y"]), "position": position}
        if not (math.isfinite(pick["x"]) and math.isfinite(pick["y"])):
            raise ValueError(f"survey pick {name!r}: the picked pixel is not a number")
        picks.append(pick)
    distances = []
    for row in data.get("distances", []):
        a, b, distance = str(row["a"]), str(row["b"]), float(row["distance"])
        if a not in names or b not in names or a == b:
            raise ValueError(f"survey distance {a!r} - {b!r}: name two different picks of the list")
        if not distance > 0 or not math.isfinite(distance):
            raise ValueError(f"survey distance {a!r} - {b!r}: the distance must be a positive number")
        distances.append({"a": a, "b": b, "distance": distance})
    return {"picks": picks, "distances": distances}


def survey_plan(survey):
    """What a survey can fix: {"mode": "similarity" | "scale" | "up_to_scale", "pairs": [(a, b, real distance)],
    "positioned": [names], "picks": count, "message": a sentence for the panel}.

    Three or more picks with positions (not on one line) fix scale, orientation and position (a similarity). A
    distance between two picks, or positions for two, fix the scale alone and leave the placement to the first-frame
    anchor. Anything less leaves the solve up to scale, and the message says what would fix it."""
    survey = parse_survey(survey) if not isinstance(survey, dict) or "picks" not in survey else survey
    picks = survey["picks"]
    positioned = [p for p in picks if p["position"]]
    plan = {"picks": len(picks), "positioned": [p["name"] for p in positioned], "pairs": [],
            "positions": {p["name"]: p["position"] for p in positioned}}
    if len(positioned) >= 3:
        spread = np.array([p["position"] for p in positioned], np.float64)
        if np.linalg.matrix_rank(spread - spread.mean(axis=0), tol=1e-9 * max(1.0, np.abs(spread).max())) >= 2:
            plan.update(mode="similarity", message=f"Aligned to {len(positioned)} surveyed points: scale, orientation and "
                        "position come from the survey, not from the first-frame camera.")
            return plan
    pairs = {tuple(sorted((d["a"], d["b"]))): d["distance"] for d in survey["distances"]}
    for i, a in enumerate(positioned):
        for b in positioned[i + 1:]:
            pairs.setdefault(tuple(sorted((a["name"], b["name"]))),
                             float(np.linalg.norm(np.array(a["position"]) - np.array(b["position"]))))
    plan["pairs"] = [(a, b, d) for (a, b), d in sorted(pairs.items())]
    if plan["pairs"]:
        collinear = " The positioned picks lie on one line, so only their distances are used." if len(positioned) >= 3 else ""
        plan.update(mode="scale", message=f"Scale from {len(plan['pairs'])} surveyed "
                    f"distance{'s' if len(plan['pairs']) != 1 else ''}; the first-frame camera still places the solve.{collinear}")
        return plan
    count = len(picks)
    have = "no picks yet" if not count else f"{count} pick{'s' if count != 1 else ''}"
    plan.update(mode="up_to_scale", message=f"Up to scale ({have}): the survey needs {NEEDED_FOR_SCALE}, or "
                f"{NEEDED_FOR_SIMILARITY}. The Scene units per solve unit knob sets the scale until then.")
    return plan


def similarity_transform(source, target):
    """Umeyama: the scale s, rotation R and translation t with target ~ s R source + t (N >= 3 points not on a line)."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("a similarity needs at least three point pairs")
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    covariance = (target - mu_t).T @ (source - mu_s) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        fix[2, 2] = -1.0
    rotation = u @ fix @ vt
    variance = float(np.mean(np.sum((source - mu_s) ** 2, axis=1)))
    if variance < 1e-18:
        raise ValueError("the surveyed points triangulate to the same place")
    scale = float(np.sum(singular * np.diag(fix)) / variance)
    return scale, rotation, mu_t - scale * rotation @ mu_s


def survey_alignment(plan, found):
    """Resolve a plan against the triangulated picks (`found`: name -> position in solve units).

    Returns {"mode", "scale", "rotation", "translation", "residual", ...}: for "scale" the least-squares scale
    over the distances (rotation and translation None); for "similarity" the Umeyama transform; `residual` is the
    root mean square distance between the aligned picks and their surveyed positions (a distance's relative
    error times the distance for "scale")."""
    if plan["mode"] == "similarity":
        names = plan["positioned"]
        survey_positions = plan["positions"]
        scale, rotation, translation = similarity_transform([found[n] for n in names], [survey_positions[n] for n in names])
        aligned = scale * (np.array([found[n] for n in names]) @ rotation.T) + translation
        residual = float(np.sqrt(np.mean(np.sum((aligned - np.array([survey_positions[n] for n in names])) ** 2, axis=1))))
        return {"mode": "similarity", "scale": scale, "rotation": rotation, "translation": translation, "residual": residual}
    if plan["mode"] == "scale":
        real = np.array([d for _a, _b, d in plan["pairs"]], np.float64)
        apart = np.array([np.linalg.norm(found[a] - found[b]) for a, b, _d in plan["pairs"]], np.float64)
        if (apart < 1e-9).any():
            raise ValueError("survey points triangulate to the same place")
        scale = float(np.sum(real * apart) / np.sum(apart * apart))
        return {"mode": "scale", "scale": scale, "rotation": None, "translation": None,
                "residual": float(np.sqrt(np.mean((scale * apart - real) ** 2)))}
    raise ValueError("the survey leaves the solve up to scale")


def place_with_similarity(c2w, points, scale, rotation, translation):
    """Poses and points of the solve in the world under world = scale * rotation * solve + translation."""
    c2w, points = np.asarray(c2w, np.float64), np.asarray(points, np.float64)
    placed = c2w.copy()
    placed[:, :3, :3] = rotation @ c2w[:, :3, :3]
    placed[:, :3, 3] = scale * (c2w[:, :3, 3] @ rotation.T) + translation
    return placed, scale * (points @ rotation.T) + translation


def track_pick(frames, pick, first, last):
    """Where a picked pixel goes on the solved frames: observations `{"frame": index into the solve, "x", "y"}`
    from the pick's own frame outward with the tracker (normalised cross-correlation), keeping the confident ones,
    at most PICK_MAX_VIEWS evenly spaced. `frames` are the plate's frames `first`..`last` as arrays."""
    from . import tracker
    if not first <= pick["frame"] <= last:
        raise ValueError(f"survey pick {pick['name']!r} is on frame {pick['frame']}, outside the solve ({first} to {last})")
    gray = {first + i: np.asarray(f, np.float32).mean(axis=2) / 255.0 for i, f in enumerate(frames)}
    rows = {pick["frame"]: (pick["x"], pick["y"], 1.0)}
    for direction in ("forward", "backward"):
        confidence = {}
        track = tracker.analyse(gray, pick["frame"], (pick["x"], pick["y"]), pattern_radius=PICK_PATTERN_RADIUS,
                                search_radius=PICK_SEARCH_RADIUS, first_frame=first, last_frame=last, direction=direction,
                                progress=lambda frame, done, total, score, c=confidence: c.__setitem__(frame, float(score)))
        for frame, (x, y) in track.items():
            if frame != pick["frame"] and confidence.get(frame, 0.0) >= PICK_MIN_CONFIDENCE:
                rows[frame] = (float(x), float(y), confidence[frame])
    ordered = sorted(rows)
    if len(ordered) > PICK_MAX_VIEWS:
        ordered = [ordered[i] for i in np.unique(np.round(np.linspace(0, len(ordered) - 1, PICK_MAX_VIEWS)).astype(int))]
    if len(ordered) < 2:
        raise ValueError(f"survey pick {pick['name']!r} could not be followed to a second frame; pick a point with more texture")
    return [{"frame": frame - first, "x": rows[frame][0], "y": rows[frame][1]} for frame in ordered]


def cloud_splats(points, colors, radius=SPLAT_RADIUS):
    """The sparse cloud as a degree-0 SplatCloud of small, opaque, coloured round splats."""
    from .splats import SplatCloud
    points = np.asarray(points, np.float32)
    count = len(points)
    rgb = np.asarray(colors, np.float32) / 255.0   # DPVO already stores RGB
    sh = ((rgb - 0.5) / SH_DC).astype(np.float32)[:, None, :]
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (count, 1))
    return SplatCloud(points, np.full((count, 3), radius, np.float32), rotations, np.full(count, 0.99, np.float32), sh, 0)


def write_cloud(points, colors, path):
    from .splats import write_ply
    write_ply(cloud_splats(points, colors), path)
    return str(path)


def _plate_frames(evaluator, document, source, first, last):
    """The plate as display-sRGB uint8 frames, through the same display transform the viewer uses."""
    from .color import display_rgb
    frames = []
    for frame in range(first, last + 1):
        pixels = evaluator.evaluate(document, source, frame=frame)
        shown = np.clip(display_rgb(np.array(pixels[..., :3], np.float32), "sRGB"), 0.0, 1.0)
        frames.append(np.rint(shown * 255.0).astype(np.uint8))
    return frames


def _anchor_pose(params):
    return pose_from_keys(*(float(params[name]) for name in
                            ("anchor_tx", "anchor_ty", "anchor_tz", "anchor_target_x", "anchor_target_y",
                             "anchor_target_z", "anchor_roll")))


def lens_request(params):
    """(estimate the focal length, radial terms to estimate) from the node's lens knobs."""
    mode, distortion = str(params.get("solve_focal", "known")), str(params.get("solve_distortion", "none"))
    if mode not in FOCAL_MODES:
        raise ValueError(f"solve_camera: solve_focal must be one of {', '.join(FOCAL_MODES)} (got {mode!r})")
    lensrefine.distortion_terms(distortion)
    return mode == "estimate", distortion


def lens_warnings(lens, typed_focal_px, width):
    """Sentences for the panel when the recovered lens should not be trusted blindly."""
    notes = []
    if lens["rms_px_after"] > 2.0 * width / 960.0:
        notes.append(f"The tracked points fit the lens only to {lens['rms_px_after']:.2f} px"
                     + ("; try estimating k1." if lens["distortion"] == "none" else "."))
    if not 0.25 < lens["focal_px"] / typed_focal_px < 4.0:
        notes.append("The recovered focal length is far from the typed one; the move may not carry enough parallax to tell.")
    sigma = lens.get("focal_px_sigma")
    if sigma and sigma > 0.05 * lens["focal_px"]:
        notes.append("The focal length is poorly determined by this move (formal error above 5 percent).")
    return notes


def lens_record(result, typed_focal_mm, params, height):
    """The lens block of the solve record: what was asked, what was recovered, how well it fits."""
    vaperture = float(params["vaperture"])
    mm = lambda px: px * vaperture / height
    block = {"focal_mode": str(params.get("solve_focal", "known")), "distortion": str(params.get("solve_distortion", "none")),
             "focal_typed_mm": round(typed_focal_mm, 4), "focal_mm": round(mm(result["focal_px"]), 4),
             "focal_px": round(result["focal_px"], 3), "k1": round(result["k1"], 6), "k2": round(result["k2"], 6),
             "rms_px_before": round(result["rms_px_before"], 4), "rms_px_after": round(result["rms_px_after"], 4),
             "median_px_before": round(result["median_px_before"], 4), "median_px_after": round(result["median_px_after"], 4),
             "inliers_before": round(result["inliers_before"], 4), "inliers_after": round(result["inliers_after"], 4),
             "observations": result["observations"],
             "points": result["points_used"], "frames": result["frames_used"], "iterations": result["iterations"]}
    if result.get("focal_px_sigma") is not None:
        block["focal_mm_sigma"] = round(mm(result["focal_px_sigma"]), 4)
    for name in ("k1_sigma", "k2_sigma"):
        if result.get(name) is not None:
            block[name] = round(result[name], 6)
    return block


def solve_camera_op(evaluator, dispatcher, cmd, store=None, solver=None, workdir=None):
    """The `solve_camera` op: solve node `id`'s plate, write the Camera3D `camera` and a cloud node into the document.

    Optional: `first`, `last` (default: the document's range), `camera` (node id to write; default `<id>_camera`),
    `survey` ({"distance": d, "points": [[{"frame", "x", "y"}, ...], [...]]}, frame as a document frame number;
    replaces the node's own Survey). `solver` replaces the provider call (tests). Returns the ids, the scale, the
    recovered lens and how the solve ran."""
    import tempfile
    from .artifacts import ArtifactStore
    document = dispatcher.document
    key = cmd.get("id")
    node = document["nodes"].get(key)
    if node is None or node["type"] != "CameraSolve3D":
        raise ValueError(f"solve_camera: {key!r} is not a CameraSolve3D node")
    source = node["inputs"].get("image")
    if not source:
        raise ValueError("solve_camera: connect the plate to the CameraSolve3D image input")
    params = node["params"]
    if params["provider"] != dpvo.PROVIDER_NAME:
        raise ValueError(f"solve_camera: unknown camera solve provider {params['provider']!r}")
    estimate_focal, distortion = lens_request(params)
    survey_data = parse_survey(params.get("survey", ""))
    plan = survey_plan(survey_data)
    time_range = document["time"]
    first, last = int(cmd.get("first", time_range["first"])), int(cmd.get("last", time_range["last"]))
    if last - first + 1 < dpvo.MIN_FRAMES:
        raise ValueError(f"solve_camera needs at least {dpvo.MIN_FRAMES} frames (got {last - first + 1})")
    frames = _plate_frames(evaluator, document, source, first, last)
    height, width = frames[0].shape[:2]
    intrinsics = dpvo.intrinsics_for(width, height, params["focal"], params["vaperture"])
    run = solver or dpvo.run_solve
    with tempfile.TemporaryDirectory(prefix="camera-solve-", dir=workdir) as work:
        solved = run(frames, intrinsics, loop_closure=bool(params["loop_closure"]), seed=int(params["seed"]),
                     workdir=work)
    c2w, points, colors = solved["c2w"], solved["points"], solved["colors"]
    lens, lens_block, focal_px, radial = None, None, intrinsics[1], (0.0, 0.0)
    if estimate_focal or distortion != "none":
        if not solved.get("tracks"):
            raise ValueError("solve_camera: the solver returned no tracked points to estimate the lens from")
        lens = lensrefine.estimate_lens(c2w, points, solved["tracks"], width, height, intrinsics[1],
                                        estimate_focal=estimate_focal, distortion=distortion)
        c2w, points, colors = lens["poses"], lens["points"], colors[lens["point_index"]]
        focal_px, radial = float(lens["focal_px"]), (float(lens["k1"]), float(lens["k2"]))
        intrinsics = (focal_px, focal_px, intrinsics[2], intrinsics[3])
        lens_block = lens_record(lens, float(params["focal"]), params, height)
        lens_block["warnings"] = lens_warnings({**lens, "distortion": distortion}, dpvo.focal_pixels(
            params["focal"], params["vaperture"], height), width)
    survey, alignment = cmd.get("survey"), {"mode": plan["mode"], "message": plan["message"], "picks": plan["picks"]}
    similarity = None
    if survey:
        observations = [[{**row, "frame": int(row["frame"]) - first} for row in point] for point in survey["points"]]
        scale = survey_scale(c2w, intrinsics, width, height, survey["distance"], observations, radial)
        scale_source = "survey"
        alignment = {"mode": "scale", "message": "Scale from the surveyed distance of the command.", "picks": 2}
    elif plan["mode"] != "up_to_scale":
        needed = sorted({n for a, b, _d in plan["pairs"] for n in (a, b)}) if plan["mode"] == "scale" else plan["positioned"]
        picks = {p["name"]: p for p in survey_data["picks"]}
        found = dict(zip(needed, triangulate_points(
            c2w, intrinsics, width, height,
            [track_pick(frames, picks[n], first, last) for n in needed], radial)))
        aligned = survey_alignment(plan, found)
        scale, scale_source = aligned["scale"], "survey"
        alignment["residual"] = round(aligned["residual"], 6)
        if aligned["mode"] == "similarity":
            similarity = aligned
    else:
        scale, scale_source = float(params["unit_scale"]), "unit_scale"
    if similarity:
        placed, cloud = place_with_similarity(c2w, points, similarity["scale"], similarity["rotation"], similarity["translation"])
    else:
        placed, cloud = place_in_world(c2w, points, _anchor_pose(params), scale)
    camera_id = cmd.get("camera") or f"{key}_camera"
    cloud_id = f"{key}_points"
    keys = [keys_from_pose(pose) for pose in placed]
    focal = round(filmback.focal_from_fov(fov_from_intrinsics(height, focal_px), float(params["vaperture"])), 4) \
        if estimate_focal else float(params["focal"])
    lens_params = {"lens_k1": round(radial[0], 6), "lens_k2": round(radial[1], 6)}
    commands = []
    if camera_id not in document["nodes"]:
        commands.append({"op": "create", "id": camera_id, "type": "Camera3D", "name": "Solved camera",
                         "params": {"focal": focal, "vaperture": float(params["vaperture"]), **lens_params, **keys[0]}})
    for name in CAMERA_PARAMS:
        commands.append({"op": "set_curve", "id": camera_id, "param": name, "curve": {
            "interpolation": "linear", "keys": [{"frame": first + i, "value": round(row[name], KEY_DECIMALS)}
                                                for i, row in enumerate(keys)]}})
    # A focal curve from an earlier solve (the survey solve keys the lens per frame) would override the solved lens.
    animated = document.get("animation", {}).get("curves", {}).get(camera_id, {})
    for name in ("focal", *lens_params):
        if animated.get(name):
            commands.append({"op": "clear_curve", "id": camera_id, "param": name})
    commands.append({"op": "set", "id": camera_id, "param": "focal", "value": focal})
    commands.append({"op": "set", "id": camera_id, "param": "vaperture", "value": float(params["vaperture"])})
    for name, value in lens_params.items():
        commands.append({"op": "set", "id": camera_id, "param": name, "value": value})
    store = store or ArtifactStore()
    cloud_file = store.root / "camera-solve" / f"{hashlib.sha256(np.ascontiguousarray(cloud, np.float32).tobytes()).hexdigest()[:16]}.ply"
    cloud_file.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(np.round(cloud, KEY_DECIMALS), colors, cloud_file)
    if cloud_id not in document["nodes"]:
        commands.append({"op": "create", "id": cloud_id, "type": "ReadSplat3D", "name": "Solve point cloud",
                         "params": {"splat_path": str(cloud_file)}})
    else:
        commands.append({"op": "set", "id": cloud_id, "param": "splat_path", "value": str(cloud_file)})
    record = {"schema": 2, "provider": dpvo.PROVIDER_NAME, "version": dpvo.description()["version"],
              "seed": int(params["seed"]), "loop_closure": bool(params["loop_closure"]), "scale": round(scale, 6),
              "scale_source": scale_source, "alignment": alignment, "frames": [first, last], "camera": camera_id,
              "intrinsics_px": [round(v, 4) for v in intrinsics],
              "camera_keys": [{k: round(v, KEY_DECIMALS) for k, v in row.items()} for row in keys]}
    if lens_block:
        record["lens"] = lens_block
    solve_id = store.put(json.dumps(record, sort_keys=True, indent=1).encode(), "camera_solve", {
        "producer": "CameraSolve3D", "version": record["version"], "inputs": [], "document": "", "frame_range": [first, last],
        "time": time.time()})
    commands.append({"op": "set", "id": key, "param": "solve", "value": solve_id})
    dispatcher.execute({"op": "batch", "commands": commands})
    info = dict(solved.get("info", {}))
    return {"camera": camera_id, "points": cloud_id, "solve": solve_id, "scale": record["scale"],
            "scale_source": scale_source, "alignment": alignment, "lens": lens_block, "frames": [first, last],
            "cloud_points": int(len(points)), "run": info}


def panel_lines(params, record=None):
    """The sentences the CameraSolve3D panel shows: the survey's state before a solve, the last solve's lens and scale after."""
    try:
        plan = survey_plan(parse_survey(params.get("survey", "")))
        lines = [plan["message"]]
    except ValueError as error:
        lines = [f"Survey unreadable: {error}"]
    if record:
        lens = record.get("lens")
        if lens:
            line = f"Lens: focal {lens['focal_mm']:.3f} mm"
            if lens["focal_mode"] == "estimate":
                line += f" estimated (typed {lens['focal_typed_mm']:.3f} mm"
                if "focal_mm_sigma" in lens:
                    line += f", plus or minus {lens['focal_mm_sigma']:.3f}"
                line += ")"
            else:
                line += " as typed"
            if lens["distortion"] != "none":
                line += f", k1 {lens['k1']:.5f}" + (f" (plus or minus {lens['k1_sigma']:.5f})" if "k1_sigma" in lens else "")
                if lens["distortion"] == "k1 k2":
                    line += f", k2 {lens['k2']:.5f}"
            lines.append(line + ".")
            lines.append(f"Reprojection error, median {lens['median_px_before']:.2f} px before and {lens['median_px_after']:.2f} px "
                         f"after (rms {lens['rms_px_before']:.2f} and {lens['rms_px_after']:.2f}); "
                         f"{lens['observations']} tracked positions of {lens['points']} points on {lens['frames']} frames.")
            lines.extend(lens.get("warnings", []))
        alignment = record.get("alignment")
        if alignment:
            lines.append("Last solve: " + alignment["message"])
    return lines
