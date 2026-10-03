"""Versioned, lossless scene-state snapshots for generative conditioning.

The JSON carries the scene contract and typed records; numeric arrays are stored beside it in an
NPZ file so large geometry and simulation data do not inflate the structured document.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path

import numpy as np

from . import cryptomatte
from . import envlight, scene3d
from .splats import SplatCloud

SCHEMA_VERSION = 1
COORDINATES = {
    "handedness": "right-handed",
    "up_axis": "Y",
    "units": "metres",
    "matrix": "column vectors; T(position) @ T(pivot) @ R(order) @ S @ T(-pivot); R(XYZ) = Rx @ Ry @ Rz (rightmost acts first)",
}
_CLASSES = {c.__name__: c for c in (
    scene3d.Vec3, scene3d.Transform3D, scene3d.Geometry, scene3d.Light, scene3d.Camera,
    scene3d.ParticleInstance, scene3d.InstanceSet, scene3d.Volume, scene3d.Scene,
    scene3d.SplatInstance, scene3d.Projection, envlight.Environment, SplatCloud,
)}


def _encode(value, arrays, array_keys=None):
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):06d}"
        arrays[key] = value
        if array_keys is not None:
            array_keys[id(value)] = key
        return {"$array": key}
    if is_dataclass(value) and not isinstance(value, type):
        # Solver/cache handles are runtime services, not shot data. Their numerical state is
        # already present in the frame arrays; serializing a Python object graph would be brittle.
        omitted = {"stream", "sparse"} if isinstance(value, (scene3d.ParticleInstance, scene3d.Volume)) else set()
        return {"$type": type(value).__name__, "fields": {
            f.name: _encode(getattr(value, f.name), arrays, array_keys)
            for f in fields(value) if f.name not in omitted
        }}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(v, arrays, array_keys) for v in value]}
    if isinstance(value, list):
        return [_encode(v, arrays, array_keys) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode(v, arrays, array_keys) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"SceneState cannot encode {type(value).__name__}")


def _decode(value, arrays):
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    if not isinstance(value, dict):
        return value
    if "$array" in value:
        return arrays[value["$array"]].copy()
    if "$tuple" in value:
        return tuple(_decode(v, arrays) for v in value["$tuple"])
    if "$type" in value:
        cls = _CLASSES.get(value["$type"])
        if cls is None:
            raise ValueError(f"Unknown SceneState type {value['$type']!r}")
        return cls(**{k: _decode(v, arrays) for k, v in value["fields"].items()})
    return {k: _decode(v, arrays) for k, v in value.items()}


def _matrix(obj):
    if isinstance(obj, scene3d.Geometry):
        return obj.world_matrix()
    if isinstance(obj, scene3d.InstanceSet):
        return obj.parent
    if isinstance(obj, scene3d.ParticleInstance):
        return obj.matrix
    if isinstance(obj, scene3d.Volume):
        return obj.matrix
    if isinstance(obj, scene3d.Light):
        return obj.parent
    return getattr(obj, "matrix", getattr(obj, "parent", np.eye(4, dtype=np.float32)))


def _bounds(geometry):
    if not len(geometry.vertices):
        return [[0., 0., 0.], [0., 0., 0.]]
    v = np.asarray(geometry.vertices, dtype=np.float64)
    world = (np.asarray(geometry.world_matrix(), dtype=np.float64) @
             np.column_stack((v, np.ones(len(v)))).T).T[:, :3]
    return [world.min(axis=0).tolist(), world.max(axis=0).tolist()]


def _object_manifest(scene):
    """Object ids are uint32 Cryptomatte bits, exactly as emitted in the pass."""
    resolved = scene3d.resolve_instances(scene)
    instance_names = set()
    for instance_set in scene.instances:
        for i in range(len(instance_set)):
            instance_id = int(instance_set.ids[i]) if instance_set.ids is not None else i
            variant = int(instance_set.variant[i])
            source = instance_set.sources[variant]
            instance_names.add(f"{source.name or 'instance'}_{instance_id}")
    out = []
    for i, geom in enumerate(resolved.geometries, 1):
        name = geom.name or f"object{i}"
        out.append({"id": f"{cryptomatte.name_to_bits(name):08x}", "name": name,
                    "type": "instance" if name in instance_names else "geometry",
                    "world_transform": np.asarray(geom.world_matrix()).tolist(),
                    "world_bounding_box": _bounds(geom)})
    for i, splat in enumerate(resolved.splats, 1):
        name = splat.name or f"splat{i}"
        out.append({"id": f"{cryptomatte.name_to_bits(name):08x}", "name": name,
                    "type": "splat", "world_transform": np.asarray(_matrix(splat)).tolist(),
                    "world_bounding_box": None})
    particle_index = 0
    for particle in scene.particles:
        matrix = np.asarray(particle.matrix, dtype=np.float64)
        positions = np.asarray(particle.positions, dtype=np.float64)
        world = (matrix @ np.column_stack((positions, np.ones(len(positions)))).T).T[:, :3]
        for position in world:
            name = f"particle{particle_index}"
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = position
            out.append({"id": f"{cryptomatte.name_to_bits(name):08x}", "name": name,
                        "type": "particle", "world_transform": transform.tolist(),
                        "world_bounding_box": [position.tolist(), position.tolist()]})
            particle_index += 1
    return out


def write_scene_state(path, samples, *, first_frame=None, last_frame=None, fps=24.0,
                      resolution=(1920, 1080), pixel_aspect=1.0, working_color_space="ACEScg",
                      simulations=None):
    """Write one shot export. `samples` maps frame number to {scene, camera[, resolution]}.

    Optional simulation payloads (particles, fluid caches, rigid bodies) can be supplied per frame
    under `simulations`; arrays are automatically moved into the adjacent NPZ.
    """
    path = Path(path)
    frames = sorted(int(f) for f in samples)
    if not frames:
        raise ValueError("SceneState needs at least one sampled frame")
    first = frames[0] if first_frame is None else int(first_frame)
    last = frames[-1] if last_frame is None else int(last_frame)
    if first > last or any(f < first or f > last for f in frames):
        raise ValueError("SceneState sample frames must fall inside the declared range")
    arrays = {}
    array_keys = {}
    records = []
    for frame in frames:
        sample = samples[frame]
        scene = sample["scene"]
        if not isinstance(scene, scene3d.Scene):
            raise TypeError(f"frame {frame} scene must be a Scene")
        camera = sample.get("camera")
        r = sample.get("resolution", resolution)
        encoded_camera = _encode(camera, arrays, array_keys)
        encoded_scene = _encode(scene, arrays, array_keys)
        camera_data = None
        if isinstance(camera, scene3d.Camera):
            camera_data = {
                "world_transform": np.asarray(camera.transform.matrix()).tolist(),
                "target": [camera.target.x, camera.target.y, camera.target.z],
                "rotation_order": camera.transform.order, "roll_degrees": camera.roll,
                "vertical_fov_degrees": camera.fov, "focal_length_mm": camera.focal,
                "horizontal_aperture_mm": camera.haperture,
                "vertical_aperture_mm": camera.vaperture,
                "near": camera.near, "far": camera.far,
                "depth_of_field": {"f_stop": camera.fstop,
                                   "focus_distance_metres": camera.focus_distance,
                                   "aperture_blade_count": camera.aperture_blades},
                "resolution": list(r), "pixel_aspect": float(pixel_aspect),
            }
        records.append({
            "frame": frame, "time_seconds": (frame - first) / float(fps),
            "camera": encoded_camera, "camera_data": camera_data, "resolution": list(r),
            "pixel_aspect": float(pixel_aspect), "objects": _object_manifest(scene),
            "scene": encoded_scene,
            "lighting": _encode(scene.lights, arrays, array_keys),
            "environments": [{
                "image": f"{path.with_suffix('.npz').name}#{array_keys[id(e.rgb)]}",
                "rotation_degrees": e.rotation, "blur": e.blur,
                "intensity": e.intensity, "tint": list(e.tint),
            } for e in scene.environments if isinstance(e, envlight.Environment)],
            "environment_images": [None if not isinstance(e, envlight.Environment) else
                                    f"{path.with_suffix('.npz').name}#{array_keys[id(e.rgb)]}"
                                    for e in scene.environments],
            "simulations": _encode({**sample.get("simulations", {}),
                                    **(simulations or {}).get(frame, {})}, arrays, array_keys),
            "passes": _encode(sample.get("passes", {}), arrays, array_keys),
        })
    document = {"schema_version": SCHEMA_VERSION,
                "coordinates": {**COORDINATES, "fps": float(fps)},
                "frame_range": {"first": first, "last": last},
                "working_color_space": str(working_color_space), "frames": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    array_path = path.with_suffix(".npz")
    np.savez_compressed(array_path, **arrays)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_scene_state(path):
    """Read a SceneState JSON+NPZ pair and reconstruct typed per-frame values."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported SceneState schema version {doc.get('schema_version')!r}")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        frames = []
        for record in doc["frames"]:
            item = dict(record)
            item["camera"] = _decode(record["camera"], arrays)
            item["scene"] = _decode(record["scene"], arrays)
            item["lighting"] = _decode(record["lighting"], arrays)
            item["simulations"] = _decode(record["simulations"], arrays)
            item["passes"] = _decode(record.get("passes", {}), arrays)
            frames.append(item)
    return {**doc, "frames": frames}
