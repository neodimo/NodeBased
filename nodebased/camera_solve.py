"""Camera solve from the plate alone: the `solve_camera` op behind the CameraSolve3D node.

A learned monocular SLAM (DPVO / DPV-SLAM, see providers/camera_solve_dpvo.py) returns one camera pose per
frame and a sparse point cloud, up to scale and in its own frame. This module turns that into the document's
own terms:

* The solve is placed in the world by an anchor: the camera of the first solved frame (the node's anchor
  position, target and roll, the camera an artist sets by eye on frame 1). Frames after it follow the solve.
* The scale comes from the node's `unit_scale` (scene units per solve unit), or from two surveyed points: their
  world distance and their pixel positions on two or more frames (triangulated with the solved cameras).
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

from . import filmback, scene3d, track3d
from .providers import camera_solve_dpvo as dpvo

KEY_DECIMALS = 3
CAMERA_PARAMS = ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll")
TARGET_DISTANCE = 1.0          # the look-at target sits this far (scene units) along the optical axis
SPLAT_RADIUS = 0.02            # scene units, the size of one cloud point
SH_DC = 0.28209479177387814    # degree-0 spherical harmonic basis: colour = 0.5 + SH_DC * f_dc


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


def survey_scale(c2w, intrinsics, width, height, distance, points):
    """Scene units per solve unit from two surveyed points.

    `points` is two lists of observations `{"frame": index into the solve, "x": px, "y": py}` (top-left origin,
    at least two frames each); `distance` is the points' real distance. Each point is triangulated from the
    solved cameras (solve units); the scale is the real distance over the triangulated one."""
    if not distance or distance <= 0:
        raise ValueError("survey distance must be a positive number")
    if len(points) != 2:
        raise ValueError("survey needs exactly two points")
    fov = fov_from_intrinsics(height, intrinsics[1])
    found = []
    for observations in points:
        if len(observations) < 2:
            raise ValueError("each survey point needs observations on at least two frames")
        cameras, pixels = [], []
        for row in observations:
            keys = keys_from_pose(np.asarray(c2w[int(row["frame"])], np.float64))
            cameras.append(scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(keys["tx"], keys["ty"], keys["tz"])),
                                          scene3d.Vec3(keys["target_x"], keys["target_y"], keys["target_z"]),
                                          fov, roll=keys["roll"]))
            pixels.append((float(row["x"]), float(row["y"])))
        xyz, _residuals = track3d.triangulate(cameras, pixels, width, height)
        found.append(xyz)
    apart = float(np.linalg.norm(found[0] - found[1]))
    if apart < 1e-9:
        raise ValueError("survey points triangulate to the same place")
    return float(distance) / apart


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


def solve_camera_op(evaluator, dispatcher, cmd, store=None, solver=None, workdir=None):
    """The `solve_camera` op: solve node `id`'s plate, write the Camera3D `camera` and a cloud node into the document.

    Optional: `first`, `last` (default: the document's range), `camera` (node id to write; default `<id>_camera`),
    `survey` ({"distance": d, "points": [[{"frame", "x", "y"}, ...], [...]]}, frame as a document frame number).
    `solver` replaces the provider call (tests). Returns the ids, the scale and how the solve ran."""
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
    survey = cmd.get("survey")
    if survey:
        observations = [[{**row, "frame": int(row["frame"]) - first} for row in point] for point in survey["points"]]
        scale, scale_source = survey_scale(c2w, intrinsics, width, height, survey["distance"], observations), "survey"
    else:
        scale, scale_source = float(params["unit_scale"]), "unit_scale"
    placed, cloud = place_in_world(c2w, points, _anchor_pose(params), scale)
    camera_id = cmd.get("camera") or f"{key}_camera"
    cloud_id = f"{key}_points"
    keys = [keys_from_pose(pose) for pose in placed]
    focal = float(params["focal"])
    commands = []
    if camera_id not in document["nodes"]:
        commands.append({"op": "create", "id": camera_id, "type": "Camera3D", "name": "Solved camera",
                         "params": {"focal": focal, "vaperture": float(params["vaperture"]), **keys[0]}})
    for name in CAMERA_PARAMS:
        commands.append({"op": "set_curve", "id": camera_id, "param": name, "curve": {
            "interpolation": "linear", "keys": [{"frame": first + i, "value": round(row[name], KEY_DECIMALS)}
                                                for i, row in enumerate(keys)]}})
    # A focal curve from an earlier solve (the survey solve keys the lens per frame) would override the solved lens.
    if document.get("animation", {}).get("curves", {}).get(camera_id, {}).get("focal"):
        commands.append({"op": "clear_curve", "id": camera_id, "param": "focal"})
    commands.append({"op": "set", "id": camera_id, "param": "focal", "value": focal})
    commands.append({"op": "set", "id": camera_id, "param": "vaperture", "value": float(params["vaperture"])})
    store = store or ArtifactStore()
    cloud_file = store.root / "camera-solve" / f"{hashlib.sha256(np.ascontiguousarray(cloud, np.float32).tobytes()).hexdigest()[:16]}.ply"
    cloud_file.parent.mkdir(parents=True, exist_ok=True)
    write_cloud(np.round(cloud, KEY_DECIMALS), colors, cloud_file)
    if cloud_id not in document["nodes"]:
        commands.append({"op": "create", "id": cloud_id, "type": "ReadSplat3D", "name": "Solve point cloud",
                         "params": {"splat_path": str(cloud_file)}})
    else:
        commands.append({"op": "set", "id": cloud_id, "param": "splat_path", "value": str(cloud_file)})
    record = {"schema": 1, "provider": dpvo.PROVIDER_NAME, "version": dpvo.description()["version"],
              "seed": int(params["seed"]), "loop_closure": bool(params["loop_closure"]), "scale": round(scale, 6),
              "scale_source": scale_source, "frames": [first, last], "camera": camera_id,
              "intrinsics_px": [round(v, 4) for v in intrinsics],
              "camera_keys": [{k: round(v, KEY_DECIMALS) for k, v in row.items()} for row in keys]}
    solve_id = store.put(json.dumps(record, sort_keys=True, indent=1).encode(), "camera_solve", {
        "producer": "CameraSolve3D", "version": record["version"], "inputs": [], "document": "", "frame_range": [first, last],
        "time": time.time()})
    commands.append({"op": "set", "id": key, "param": "solve", "value": solve_id})
    dispatcher.execute({"op": "batch", "commands": commands})
    info = dict(solved.get("info", {}))
    return {"camera": camera_id, "points": cloud_id, "solve": solve_id, "scale": record["scale"],
            "scale_source": scale_source, "frames": [first, last], "cloud_points": int(len(points)), "run": info}
