"""CPU checks for a frame sequence against a versioned SceneState export.

Scene vertices seed NodeBased Tracker landmarks. A CPU reprojection solve estimates
the camera; exported object positions are compared to ID masks or tracked points;
the scene is rerendered with its exported lights for image-derived light checks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import json
from pathlib import Path

import numpy as np

from .scene_state import read_scene_state
from . import scene3d, tracker


@dataclass(frozen=True)
class Tolerances:
    camera_position_m: float = 0.01
    camera_rotation_degrees: float = 0.25
    camera_fov_degrees: float = 0.1
    object_pixels: float = 2.0
    light_direction_degrees: float = 5.0
    shadow_direction_degrees: float = 5.0
    colour_balance: float = 0.05
    motion_mean_pixels: float = 1.0
    motion_p95_pixels: float = 3.0
    depth_rank_correlation: float = 0.8


def _rotation_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _camera_axes(camera):
    _eye, view = scene3d._view_basis(camera)
    return view.T.astype(np.float64)


def _camera_measurement(reference_camera, measured_camera):
    a = _camera_axes(reference_camera)
    b = _camera_axes(measured_camera)
    pa = np.asarray(reference_camera.transform.position.array(), dtype=np.float64)
    pb = np.asarray(measured_camera.transform.position.array(), dtype=np.float64)
    return {
        "position_error_m": float(np.linalg.norm(pa - pb)),
        "rotation_error_degrees": _rotation_error(a, b),
        "fov_error_degrees": abs(float(reference_camera.fov - measured_camera.fov)),
    }


def _camera_parameters(camera):
    eye = np.asarray(camera.transform.position.array(), dtype=np.float64)
    direction = np.asarray(camera.target.array(), dtype=np.float64) - eye
    distance = float(np.linalg.norm(direction))
    direction /= max(distance, 1e-12)
    yaw = np.arctan2(direction[0], -direction[2])
    pitch = np.arcsin(np.clip(direction[1], -1, 1))
    return np.array((*eye, yaw, pitch, np.radians(camera.roll), np.radians(camera.fov))), max(distance, 1e-3)


def _camera_from_parameters(base, params, target_distance):
    px, py, pz, yaw, pitch, roll, fov = params
    direction = np.array((np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                          -np.cos(yaw) * np.cos(pitch)), dtype=np.float64)
    eye = np.array((px, py, pz), dtype=np.float64)
    transform = replace(base.transform, position=scene3d.Vec3(*eye.tolist()))
    target = eye + direction * target_distance
    return replace(base, transform=transform, target=scene3d.Vec3(*target.tolist()),
                   roll=float(np.degrees(roll)), fov=float(np.degrees(fov)))


def _camera_from_observation(base, value):
    if isinstance(value, scene3d.Camera):
        return value
    if not isinstance(value, dict) or "world_transform" not in value:
        return None
    matrix = np.asarray(value["world_transform"], dtype=np.float64)
    if matrix.shape != (4, 4):
        return None
    axes = matrix[:3, :3]
    eye = matrix[:3, 3]
    forward = -axes[:, 2]
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    distance = max(float(np.linalg.norm(base.target.array() - base.transform.position.array())), 1.0)
    right = np.cross(forward, np.array((0., 1., 0.)))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array((0., 0., 1.)))
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    roll = np.arctan2(np.dot(axes[:, 0], up), np.dot(axes[:, 0], right))
    transform = replace(base.transform, position=scene3d.Vec3(*eye.tolist()))
    target = eye + forward * distance
    return replace(base, transform=transform, target=scene3d.Vec3(*target.tolist()),
                   roll=float(np.degrees(roll)),
                   fov=float(value.get("vertical_fov_degrees", base.fov)))


def _fit_camera(base, world_points, image_points, width, height):
    """Damped finite-difference reprojection solve from tracked 3-D/2-D point pairs."""
    world = np.asarray(world_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or image.shape != (len(world), 2):
        raise ValueError("camera solve needs matching Nx3 world and Nx2 tracked points")
    if len(world) < 6 or np.linalg.matrix_rank(world - world.mean(axis=0)) < 3:
        raise ValueError("camera solve needs at least six non-coplanar tracked scene points")
    params, distance = _camera_parameters(base)

    def residual(values):
        candidate = _camera_from_parameters(base, values, distance)
        pixels, depth = scene3d.project(candidate, width, height, world)
        out = (pixels.astype(np.float64) - image).reshape(-1)
        out[~np.repeat(depth > candidate.near, 2)] = 1e4
        return out

    damping = 1e-3
    current = residual(params)
    for _ in range(50):
        jacobian = np.empty((len(current), len(params)), dtype=np.float64)
        steps = np.array((1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5))
        for column, step in enumerate(steps):
            shifted = params.copy(); shifted[column] += step
            jacobian[:, column] = (residual(shifted) - current) / step
        # Huber weighting keeps a single bad tracker match from rotating the whole solve.
        pair_errors = np.linalg.norm(current.reshape(-1, 2), axis=1)
        weights = np.repeat(np.minimum(1.0, 3.0 / np.maximum(pair_errors, 1e-12)), 2)
        weighted_j = jacobian * weights[:, None]
        weighted_r = current * weights
        normal = weighted_j.T @ weighted_j + damping * np.eye(len(params))
        delta = np.linalg.lstsq(normal, -(weighted_j.T @ weighted_r), rcond=None)[0]
        candidate = params + delta
        if not (np.radians(1.0) < candidate[6] < np.radians(179.0)):
            damping *= 10
            continue
        trial = residual(candidate)
        if np.dot(trial, trial) < np.dot(current, current):
            params, current = candidate, trial
            damping = max(damping * 0.3, 1e-9)
            if np.linalg.norm(delta) < 1e-7:
                break
        else:
            damping = min(damping * 10, 1e8)
    rms = float(np.sqrt(np.mean(current.reshape(-1, 2) ** 2)))
    return _camera_from_parameters(base, params, distance), rms


def _scene_anchor_points(scene):
    """Deterministic scene-space points usable as camera-solve tracker landmarks."""
    points = []
    for geometry in scene3d.resolve_instances(scene).geometries:
        if not len(geometry.vertices):
            continue
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        if len(vertices) > 48:
            indices = np.linspace(0, len(vertices) - 1, 48, dtype=int)
            vertices = vertices[indices]
        matrix = np.asarray(geometry.world_matrix(), dtype=np.float64)
        world = (matrix @ np.column_stack((vertices, np.ones(len(vertices)))).T).T[:, :3]
        points.extend(world)
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    # Remove coincident landmarks from seams and instanced source geometry.
    points = np.unique(np.round(np.asarray(points), decimals=7), axis=0)
    if len(points) > 64:
        points = points[np.linspace(0, len(points) - 1, 64, dtype=int)]
    return points


def _tracker_search_radius(width, height):
    return min(32, max(12, int(max(width, height) * .0125)))


def _angle_degrees(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))))


def _main_direction(scene):
    lights = [light for light in scene.lights if light.intensity > 0]
    if not lights:
        return None
    key = max(lights, key=lambda light: float(light.intensity))
    position, travel = key.world()
    if key.kind == "Directional":
        return -np.asarray(travel, dtype=np.float64)
    points = [np.asarray(g.world_matrix()[:3, 3], dtype=np.float64)
              for g in scene3d.resolve_instances(scene).geometries]
    center = np.mean(points, axis=0) if points else np.zeros(3)
    direction = np.asarray(position, dtype=np.float64) - center
    return direction / max(float(np.linalg.norm(direction)), 1e-12)


def _scene_with_key_direction(scene, to_light):
    lights = list(scene.lights)
    candidates = [i for i, light in enumerate(lights) if light.intensity > 0]
    if not candidates:
        return scene
    index = max(candidates, key=lambda i: float(lights[i].intensity))
    light = lights[index]
    parent = np.asarray(light.parent, dtype=np.float64)
    parent_inverse = np.linalg.inv(parent)
    desired = np.asarray(to_light, dtype=np.float64)
    desired /= max(float(np.linalg.norm(desired)), 1e-12)
    if light.kind == "Directional":
        travel = -desired
        local_travel = np.linalg.inv(parent[:3, :3]) @ travel
        local_travel /= max(float(np.linalg.norm(local_travel)), 1e-12)
        distance = float(np.linalg.norm(light.target.array() - light.position.array()))
        target = light.position.array() + local_travel * max(distance, 1.0)
        lights[index] = replace(light, target=scene3d.Vec3(*target.tolist()))
    else:
        geometry_positions = [np.asarray(g.world_matrix()[:3, 3], dtype=np.float64)
                              for g in scene3d.resolve_instances(scene).geometries]
        center = np.mean(geometry_positions, axis=0) if geometry_positions else np.zeros(3)
        world_position, _ = light.world()
        radius = max(float(np.linalg.norm(np.asarray(world_position) - center)), 1.0)
        new_world_position = center + desired * radius
        old_world_target = (parent @ np.append(light.target.array(), 1.0))[:3]
        old_world_position = np.asarray(world_position, dtype=np.float64)
        new_world_target = old_world_target + (new_world_position - old_world_position)
        local_position = (parent_inverse @ np.append(new_world_position, 1.0))[:3]
        local_target = (parent_inverse @ np.append(new_world_target, 1.0))[:3]
        lights[index] = replace(light, position=scene3d.Vec3(*local_position.tolist()),
                                target=scene3d.Vec3(*local_target.tolist()))
    return replace(scene, lights=tuple(lights))


def _estimate_light_direction(albedo, normals, observed):
    """Fit a robust Lambert direction from the CPU albedo and world-normal passes."""
    base = np.asarray(albedo, dtype=np.float64)
    normal = np.asarray(normals, dtype=np.float64)
    image = np.asarray(observed, dtype=np.float64)
    if base.shape != image.shape or normal.shape[:2] != image.shape[:2]:
        return None
    valid = ((base[..., 3] > .5) & (normal[..., 3] > .5)
             & (np.linalg.norm(normal[..., :3], axis=-1) > .5)
             & (base[..., :3].mean(axis=-1) > .015))
    if valid.sum() < 12:
        return None
    n = normal[..., :3][valid]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    if np.linalg.matrix_rank(n - n.mean(axis=0)) < 3:
        return None
    albedo_luma = base[..., :3][valid] @ np.array((.2126, .7152, .0722))
    observed_luma = image[..., :3][valid] @ np.array((.2126, .7152, .0722))
    y = observed_luma / np.maximum(albedo_luma, .02)
    usable = np.isfinite(y) & (y > 0) & (y < 4.0)
    n, y = n[usable], y[usable]
    if len(y) < 12:
        return None
    # A directional Lambertian model plus ambient term. Trim bright specular and
    # occlusion outliers using iteratively reweighted least squares.
    design = np.column_stack((n, np.ones(len(n))))
    weights = np.ones(len(n), dtype=np.float64)
    beta = np.zeros(4, dtype=np.float64)
    for _ in range(8):
        beta = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
        error = y - design @ beta
        scale = max(float(np.median(np.abs(error))) * 1.4826, 1e-4)
        weights = np.minimum(1.0, (1.5 * scale) / np.maximum(np.abs(error), 1e-12))
    vector = beta[:3]
    magnitude = float(np.linalg.norm(vector))
    return None if magnitude < 1e-5 else vector / magnitude


def _shadow_vector(unshadowed, shadowed, albedo):
    """Estimate 2-D shadow displacement from dark residuals on rendered surfaces."""
    clear = np.asarray(unshadowed, dtype=np.float64)
    shaded = np.asarray(shadowed, dtype=np.float64)
    base = np.asarray(albedo, dtype=np.float64)
    support = (base[..., 3] > .5) & (base[..., :3].mean(axis=-1) > .01)
    ratio = (shaded[..., :3].mean(axis=-1) + 1e-5) / (clear[..., :3].mean(axis=-1) + 1e-5)
    support &= np.isfinite(ratio)
    if support.sum() < 12:
        return None
    cut = float(np.quantile(ratio[support], .2))
    shadow = support & (ratio <= min(cut, .75))
    if shadow.sum() < 4:
        return None
    y, x = np.nonzero(shadow)
    sy, sx = float(y.mean()), float(x.mean())
    fy, fx = np.nonzero(support)
    vector = np.array((sx - float(fx.mean()), sy - float(fy.mean())))
    length = float(np.linalg.norm(vector))
    return None if length < 1.0 else vector / length


def _render_frame(scene, camera, width, height):
    """Render conditioning references on the CPU, in the same scene/camera space."""
    beauty = scene3d.render(scene, camera, width, height, output="rgba", mode="raster")
    albedo = scene3d.render(scene, camera, width, height, output="albedo", mode="raster")
    normals = scene3d.render(scene, camera, width, height, output="normals", mode="raster")
    diffuse = scene3d.render(scene, camera, width, height, output="diffuse", mode="raster", shadows=True)
    clear = scene3d.render(scene, camera, width, height, output="diffuse", mode="raster", shadows=False)
    return {"beauty": beauty, "albedo": albedo, "normals": normals,
            "diffuse": diffuse, "unshadowed": clear}


def _lighting_resolution(width, height, maximum=512):
    scale = min(1.0, maximum / max(int(width), int(height), 1))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _resize_for_lighting(image, width, height):
    image = np.asarray(image)
    if image.shape[:2] == (height, width):
        return image
    ys = np.linspace(0, image.shape[0] - 1, height).round().astype(int)
    xs = np.linspace(0, image.shape[1] - 1, width).round().astype(int)
    return image[ys[:, None], xs[None, :]]


def _track_camera_sequence(scene_state, observations, width, height):
    rows = scene_state["frames"]
    cameras = {int(row["frame"]): row["camera"] for row in rows
               if isinstance(row.get("camera"), scene3d.Camera)}
    if not cameras:
        return {}, {}
    reference_frame = min(cameras)
    image_frames = {int(frame): np.asarray(row["beauty"], dtype=np.float32)
                    for frame, row in observations.items() if row.get("beauty") is not None}
    if reference_frame not in image_frames:
        return {}, {frame: "reference image missing" for frame in cameras}
    anchors = _scene_anchor_points(rows[0]["scene"])
    reference_camera = cameras[reference_frame]
    seed_pixels, depths = scene3d.project(reference_camera, width, height, anchors)
    valid = ((depths > reference_camera.near) & (seed_pixels[:, 0] >= 12)
             & (seed_pixels[:, 1] >= 12) & (seed_pixels[:, 0] < width - 12)
             & (seed_pixels[:, 1] < height - 12))
    anchors, seed_pixels = anchors[valid], seed_pixels[valid]
    tracked = []
    for world, seed in zip(anchors, seed_pixels):
        try:
            result = tracker.analyse(image_frames, reference_frame, seed,
                                     first_frame=min(image_frames), last_frame=max(image_frames),
                                     pattern_radius=4, search_radius=_tracker_search_radius(width, height))
            tracked.append((world, result))
        except (tracker.AnalysisError, KeyError):
            continue
    solved, failures = {}, {}
    for frame_number, camera in cameras.items():
        pairs = [(world, points[frame_number]) for world, points in tracked if frame_number in points]
        if len(pairs) < 6:
            failures[frame_number] = f"only {len(pairs)} usable Tracker landmarks; need six non-coplanar points"
            continue
        try:
            fitted, rms = _fit_camera(camera, [p[0] for p in pairs], [p[1] for p in pairs], width, height)
            solved[frame_number] = (fitted, rms, len(pairs))
        except ValueError as error:
            failures[frame_number] = str(error)
    return solved, failures


def _track_object_positions(scene_state, observations, width, height):
    rows = scene_state["frames"]
    camera_frames = {int(row["frame"]): row["camera"] for row in rows
                     if isinstance(row.get("camera"), scene3d.Camera)}
    if not camera_frames:
        return {}
    reference_frame = min(camera_frames)
    images = {int(frame): np.asarray(row["beauty"], dtype=np.float32)
              for frame, row in observations.items() if row.get("beauty") is not None}
    if reference_frame not in images:
        return {}
    reference_camera = camera_frames[reference_frame]
    manifest = next((row["objects"] for row in rows if int(row["frame"]) == reference_frame), [])
    result = {frame: {} for frame in camera_frames}
    required = set()
    for frame, observation in observations.items():
        if observation.get("id_pass") is not None:
            continue
        explicit = set(observation.get("object_ids", {})) | set(observation.get("object_tracks", {}))
        required.update(row["id"] for row in manifest if row["id"] not in explicit)
    for obj in manifest:
        if obj["id"] not in required:
            continue
        world = np.asarray(obj["world_transform"], dtype=np.float64)[:3, 3]
        pixel, depth = scene3d.project(reference_camera, width, height, world[None, :])
        seed = pixel[0]
        if depth[0] <= reference_camera.near or not (12 <= seed[0] < width - 12 and 12 <= seed[1] < height - 12):
            continue
        try:
            points = tracker.analyse(images, reference_frame, seed,
                                     first_frame=min(images), last_frame=max(images),
                                     pattern_radius=4, search_radius=_tracker_search_radius(width, height))
        except (tracker.AnalysisError, KeyError):
            continue
        for frame, point in points.items():
            if frame in result:
                result[frame][obj["id"]] = [float(point[0]), float(point[1])]
    return result


def _centroid(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        return None
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [float(x.mean() + 0.5), float(y.mean() + 0.5)]


def _object_measurements(frame, observed_ids, tracked_positions, camera, width, height):
    """Project each exported object origin and compare it with its ID mask or tracked point."""
    results = []
    for row in frame["objects"]:
        observed = observed_ids.get(row["id"])
        observed_array = None if observed is None else np.asarray(observed, dtype=bool)
        if observed_array is not None and observed_array.shape[:2] != (height, width):
            observed_array = None
        actual = None if observed_array is None else _centroid(observed_array)
        source = "id_pass" if actual is not None else None
        if actual is None and row["id"] in tracked_positions:
            actual = [float(v) for v in tracked_positions[row["id"]]]
            source = "tracker"
        matrix = np.asarray(row["world_transform"], dtype=np.float64)
        projected, depth = scene3d.project(camera, width, height, matrix[:3, 3][None, :])
        reference = projected[0].astype(float).tolist() if depth[0] > camera.near else None
        error = (None if reference is None or actual is None else
                 float(np.linalg.norm(np.asarray(reference) - np.asarray(actual))))
        results.append({"object_id": row["id"], "error_pixels": error,
                        "reference_centroid": reference, "observed_centroid": actual,
                        "measurement_source": source})
    return results


def _colour_balance(reference, observed):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] < 3:
        return None
    ma = np.mean(a[..., :3], axis=(0, 1)); mb = np.mean(b[..., :3], axis=(0, 1))
    ma /= max(float(ma.sum()), 1e-12); mb /= max(float(mb.sum()), 1e-12)
    return float(np.linalg.norm(ma - mb))


def _lock_state(value):
    return str(value) if value in ("locked", "loosened", "free") else "locked"


def _rank_correlation(expected, observed):
    """Spearman rank correlation without a scipy dependency."""
    a, b = np.asarray(expected, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 2:
        return None
    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _motion_measurement(observation, tolerances):
    """Compare CPU optical flow of generated adjacent frames with bundle vectors."""
    expected = observation.get("motion_reference")
    first, second = observation.get("beauty"), observation.get("next_beauty")
    if expected is None or first is None or second is None:
        return {"mean_error_pixels": None, "p95_error_pixels": None, "pass": False,
                "reason": "generated frame pair or exported motion vectors missing"}
    from .opticalflow import flow_pair
    measured, _backward, _occlusion = flow_pair(first, second, backend="cpu")
    expected = np.asarray(expected, dtype=np.float32)
    if expected.shape != measured.shape:
        return {"mean_error_pixels": None, "p95_error_pixels": None, "pass": False,
                "reason": "generated flow and exported motion dimensions differ"}
    error = np.linalg.norm(measured - expected, axis=-1)
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return {"mean_error_pixels": None, "p95_error_pixels": None, "pass": False,
                "reason": "motion error has no finite pixels"}
    mean, p95 = float(np.mean(finite)), float(np.percentile(finite, 95))
    return {"mean_error_pixels": mean, "p95_error_pixels": p95,
            "pass": mean <= tolerances.motion_mean_pixels and p95 <= tolerances.motion_p95_pixels}


def _depth_measurement(observation, tolerances):
    landmarks = observation.get("depth_landmarks") or {}
    expected, observed = [], []
    control_depth = observation.get("control_depth")
    if control_depth is not None:
        control_depth = np.asarray(control_depth, dtype=np.float32)
        if control_depth.ndim == 3:
            control_depth = control_depth[..., 0]
    for item in landmarks.values():
        if not isinstance(item, dict) or "observed" not in item:
            continue
        value = item.get("expected")
        if value is None and control_depth is not None:
            pixel = item.get("pixel", item.get("xy"))
            if pixel is not None:
                x, y = map(int, pixel)
                if 0 <= y < control_depth.shape[0] and 0 <= x < control_depth.shape[1]:
                    value = control_depth[y, x]
        if value is not None and np.isfinite(value) and np.isfinite(item["observed"]):
            expected.append(float(value))
            observed.append(float(item["observed"]))
    correlation = _rank_correlation(expected, observed)
    return {"rank_correlation": correlation,
            "landmarks": len(expected),
            "pass": correlation is not None and correlation >= tolerances.depth_rank_correlation,
            **({} if correlation is not None else {"reason": "tracked landmark depth observations missing"})}


def verify_conditioning(scene_state_path, observations, output_path, *, tolerances=None):
    """Write ``.json`` and ``.txt`` shot reports.

    `observations` is keyed by integer frame. A row contains `beauty` (HxWxRGBA float
    frame) and may contain `id_pass` (integer render IDs), `object_ids` (masks keyed by
    exported ID), lock-state fields, or explicit solved-camera/direction measurements.
    Missing camera and object measurements are derived with NodeBased Tracker when the
    scene has enough visible features. Lighting estimation uses a fresh CPU scene render.
    """
    tolerances = tolerances or Tolerances()
    state = read_scene_state(scene_state_path)
    report = {"schema_version": 1, "scene_state": Path(scene_state_path).name,
              "intensity_checked": False, "frames": []}
    solved_cameras, solve_failures = _track_camera_sequence(
        state, observations, *state["frames"][0].get("resolution", (1920, 1080)))
    tracked_objects = _track_object_positions(
        state, observations, *state["frames"][0].get("resolution", (1920, 1080)))
    for frame in state["frames"]:
        number = int(frame["frame"]); observation = observations.get(number, {})
        reference_camera = frame.get("camera")
        solved = solved_cameras.get(number)
        measured_camera = (solved[0] if solved is not None else
                           _camera_from_observation(reference_camera, observation.get("camera")))
        camera = None
        if isinstance(reference_camera, scene3d.Camera) and isinstance(measured_camera, scene3d.Camera):
            camera = _camera_measurement(reference_camera, measured_camera)
            if solved is not None:
                camera["tracker_landmarks"] = solved[2]
                camera["tracker_reprojection_rms_pixels"] = solved[1]
            camera["pass"] = (camera["position_error_m"] <= tolerances.camera_position_m
                              and camera["rotation_error_degrees"] <= tolerances.camera_rotation_degrees
                              and camera["fov_error_degrees"] <= tolerances.camera_fov_degrees)
        elif isinstance(reference_camera, scene3d.Camera):
            camera = {"position_error_m": None, "rotation_error_degrees": None,
                      "fov_error_degrees": None, "pass": False,
                      "reason": solve_failures.get(number, "camera solve missing")}
        active_camera = measured_camera if isinstance(measured_camera, scene3d.Camera) else reference_camera
        width, height = frame.get("resolution", (1920, 1080))
        observed_ids = dict(observation.get("object_ids", {}))
        if observation.get("id_pass") is not None:
            id_image = np.asarray(observation["id_pass"])
            if id_image.ndim == 3:
                id_image = id_image[..., 0]
            for index, object_row in enumerate(frame["objects"], 1):
                observed_ids.setdefault(object_row["id"], id_image == index)
        objects = (_object_measurements(frame, observed_ids,
                                        {**tracked_objects.get(number, {}),
                                         **observation.get("object_tracks", {})}, active_camera,
                                        int(width), int(height))
                   if isinstance(active_camera, scene3d.Camera) else [])
        for item in objects:
            item["pass"] = item["error_pixels"] is not None and item["error_pixels"] <= tolerances.object_pixels
        lighting = {"intensity_checked": False}
        if (isinstance(active_camera, scene3d.Camera) and observation.get("beauty") is not None
                and isinstance(frame.get("scene"), scene3d.Scene)):
            scene = frame["scene"]
            light_width, light_height = _lighting_resolution(int(width), int(height))
            reference = _render_frame(scene, active_camera, light_width, light_height)
            observed = _resize_for_lighting(observation["beauty"], light_width, light_height).astype(np.float32)
            expected_direction = _main_direction(scene)
            estimated_direction = _estimate_light_direction(reference["albedo"], reference["normals"], observed)
            if estimated_direction is not None and expected_direction is not None:
                lighting["dominant_light_direction"] = estimated_direction.tolist()
                lighting["light_direction_error_degrees"] = _angle_degrees(expected_direction, estimated_direction)
            direction_scene = (_scene_with_key_direction(scene, estimated_direction)
                               if estimated_direction is not None else scene)
            clear = scene3d.render(direction_scene, active_camera, light_width, light_height, output="diffuse",
                                   mode="raster", shadows=False)
            altered_shadow = _shadow_vector(clear, observed, reference["albedo"])
            reference_shadow = _shadow_vector(reference["unshadowed"], reference["diffuse"], reference["albedo"])
            if altered_shadow is not None and reference_shadow is not None:
                lighting["shadow_direction_error_degrees"] = _angle_degrees(reference_shadow, altered_shadow)
                lighting["shadow_direction"] = altered_shadow.tolist()
            lighting["colour_balance_error"] = _colour_balance(reference["beauty"], observed)
        # Explicit measurements remain useful for callers whose generated sequence
        # already carries trusted direction annotations.
        if ("light_direction_error_degrees" not in lighting
                and observation.get("dominant_light_direction") is not None
                and observation.get("reference_light_direction") is not None):
            angle = _angle_degrees(observation["reference_light_direction"],
                                   observation["dominant_light_direction"])
            if angle is not None:
                lighting["light_direction_error_degrees"] = angle
        if ("shadow_direction_error_degrees" not in lighting
                and observation.get("shadow_direction") is not None
                and observation.get("reference_shadow_direction") is not None):
            angle = _angle_degrees(observation["reference_shadow_direction"],
                                   observation["shadow_direction"])
            if angle is not None:
                lighting["shadow_direction_error_degrees"] = angle
        if "colour_balance_error" not in lighting and observation.get("beauty") is not None:
            reference_beauty = frame.get("passes", {}).get("beauty")
            if reference_beauty is not None:
                lighting["colour_balance_error"] = _colour_balance(reference_beauty, observation["beauty"])
        lighting_metrics = ("light_direction_error_degrees", "shadow_direction_error_degrees",
                            "colour_balance_error")
        complete_metrics = all(key in lighting and lighting[key] is not None
                               and np.isfinite(lighting[key]) for key in lighting_metrics)
        lighting["pass"] = complete_metrics and all((
            lighting["light_direction_error_degrees"] <= tolerances.light_direction_degrees,
            lighting["shadow_direction_error_degrees"] <= tolerances.shadow_direction_degrees,
            lighting["colour_balance_error"] <= tolerances.colour_balance))
        if not lighting["pass"] and not complete_metrics:
            lighting["reason"] = "light direction, shadow direction, or colour evidence missing"
        for key in lighting_metrics:
            lighting.setdefault(key, None)
        motion = _motion_measurement(observation, tolerances)
        depth = _depth_measurement(observation, tolerances)
        bindings = {
            "camera": {"lock_state": _lock_state(observation.get("camera_lock_state")), **camera}
                      if camera else {"lock_state": _lock_state(observation.get("camera_lock_state")),
                                     "pass": False, "reason": "camera solve missing"},
            "objects": [{"lock_state": _lock_state((observation.get("object_lock_states") or {}).get(
                item["object_id"])), **item} for item in objects],
            "lighting": {"lock_state": _lock_state(observation.get("lighting_lock_state")), **lighting},
            "motion": {"lock_state": _lock_state(observation.get("motion_lock_state")), **motion},
            "depth": {"lock_state": _lock_state(observation.get("depth_lock_state")), **depth},
        }
        report["frames"].append({"frame": number, "bindings": bindings})
    # One per-shot score card, aggregating each requirement across the frame range.
    categories = ("camera", "objects", "lighting", "motion", "depth")
    score = {}
    for category in categories:
        values = []
        for row in report["frames"]:
            binding = row["bindings"][category]
            values.extend(binding if category == "objects" else [binding])
        passed = bool(values) and all(value.get("pass", False) for value in values)
        errors = []
        for value in values:
            errors.append({key: item for key, item in value.items()
                           if "error" in key or key == "rank_correlation"})
        locks = sorted({value.get("lock_state", "locked") for value in values})
        score[category] = {"lock_state": locks[0] if len(locks) == 1 else locks,
                           "error": errors, "pass": passed}
    report["score_card"] = {"bindings": score,
                            "verdict": "PASS" if all(item["pass"] for item in score.values()) else "FAIL"}
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def metric(value, unit):
        return "unavailable" if value is None else f"{float(value):.3f} {unit}"
    lines = [f"Conditioning verification: {json_path.stem}",
             f"Shot verdict: {report['score_card']['verdict']}",
             "Lighting intensity is not checked."]
    for item in report["frames"]:
        bindings = item["bindings"]
        lines.append(f"Frame {item['frame']}: camera {bindings['camera']['lock_state']}="
                     f"{'PASS' if bindings['camera']['pass'] else 'FAIL'}, "
                     f"objects={'PASS' if all(x['pass'] for x in bindings['objects']) else 'FAIL'}, "
                     f"lighting {bindings['lighting']['lock_state']}="
                     f"{'PASS' if bindings['lighting']['pass'] else 'FAIL'}")
        c = bindings["camera"]
        if c.get("position_error_m") is not None:
            lines.append(f"  Camera errors: {c['position_error_m']:.4f} m, "
                         f"{c['rotation_error_degrees']:.3f} degrees, FOV {c['fov_error_degrees']:.3f} degrees")
        elif c.get("reason"):
            lines.append(f"  Camera measurement unavailable: {c['reason']}")
        for obj in bindings["objects"]:
            error = metric(obj["error_pixels"], "px")
            lines.append(f"  Object {obj['object_id']} {obj['lock_state']}: {error} "
                         f"({obj.get('measurement_source') or 'no observation'}), "
                         f"{'PASS' if obj['pass'] else 'FAIL'}")
        light = bindings["lighting"]
        lines.append("  Lighting errors: "
                     f"direction {metric(light['light_direction_error_degrees'], 'degrees')}, "
                     f"shadow {metric(light['shadow_direction_error_degrees'], 'degrees')}, "
                     f"colour balance {metric(light['colour_balance_error'], 'normalised RGB')}; "
                     "intensity not checked")
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write CPU conditioning-verification reports.")
    parser.add_argument("scene_state", help="SceneState JSON export")
    parser.add_argument("observations", help="JSON frame observations with an adjacent NPZ for image arrays")
    parser.add_argument("output", help="Output report basename")
    args = parser.parse_args(argv)
    observation_path = Path(args.observations)
    raw = json.loads(observation_path.read_text(encoding="utf-8"))
    with np.load(observation_path.with_suffix(".npz"), allow_pickle=False) as arrays:
        def resolve(value):
            if isinstance(value, dict) and set(value) == {"array"}:
                return arrays[value["array"]].copy()
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value
        observations = {int(frame): resolve(row) for frame, row in raw.items()}
    verify_conditioning(args.scene_state, observations, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
