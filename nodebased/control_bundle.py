"""Versioned multichannel EXR control bundles for SceneState exports.

The caller supplies the CPU-rendered per-frame controls.  This module owns the stable file
contract, EXR layer names and typed reader so downstream conditioning never has to guess.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cryptomatte
from .color import config as ocio_config
from .scene_state import SCHEMA_VERSION, read_scene_state

BUNDLE_VERSION = 1
CONTROL_BUNDLE_DATETIME = "1980:01:01 00:00:00"
SHUTTER_CONVENTION = "sample at the pixel centre; vectors point from this frame to the adjacent frame"


class SceneStateVersionMismatch(ValueError):
    """The bundle and its referenced SceneState use different schemas."""


@dataclass(frozen=True)
class ControlPlane:
    values: np.ndarray
    units: str
    coordinate_space: str
    color_space: str


@dataclass(frozen=True)
class ControlFrame:
    frame: int
    beauty: ControlPlane
    depth: ControlPlane
    normals: ControlPlane
    motion_forward: ControlPlane
    motion_backward: ControlPlane
    object_ids: ControlPlane
    object_names: dict[str, str]


def _rgba(array, name, shape):
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 2:
        value = value[..., None]
    if value.shape[:2] != shape or value.ndim != 3 or value.shape[2] not in (1, 2, 3, 4):
        raise ValueError(f"{name} must be an HxW array matching the SceneState camera")
    out = np.zeros((*shape, 4), dtype=np.float32)
    out[..., :value.shape[2]] = value
    if value.shape[2] < 4:
        out[..., 3] = 1.0
    return out


def write_control_bundle(scene_state_path, output_dir, samples, *, first_frame=None,
                         last_frame=None, normals_space="camera", return_artifact_id=False):
    """Write `frame.####.exr` and a versioned manifest. Samples contain six named arrays."""
    state = read_scene_state(scene_state_path)
    if state["schema_version"] != SCHEMA_VERSION:
        raise SceneStateVersionMismatch(
            f"SceneStateVersionMismatch: expected {SCHEMA_VERSION}, got {state['schema_version']}")
    if normals_space not in ("camera", "world"):
        raise ValueError("normals_space must be 'camera' or 'world'")
    records = {int(row["frame"]): row for row in state["frames"]}
    frames = sorted(map(int, samples))
    if not frames:
        raise ValueError("ControlBundle requires at least one frame")
    first = frames[0] if first_frame is None else int(first_frame)
    last = frames[-1] if last_frame is None else int(last_frame)
    if first > last or frames != list(range(first, last + 1)):
        raise ValueError("ControlBundle samples must cover the declared contiguous frame range")
    if any(frame not in records for frame in frames):
        raise ValueError("ControlBundle frame range must be sampled in SceneState")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    import PyOpenColorIO as ocio
    scene_linear_space = ocio_config().getRoleColorSpace(ocio.ROLE_SCENE_LINEAR)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "scene_state_schema_version": SCHEMA_VERSION,
        "scene_state": str(Path(scene_state_path).name),
        "scene_state_sha256": hashlib.sha256(Path(scene_state_path).read_bytes()).hexdigest(),
        "frame_range": {"first": first, "last": last},
        "format": "OpenEXR 32-bit float, multichannel, scene-linear beauty",
        "frames": [],
    }
    layer_specs = {
        "beauty": ("beauty", ["R", "G", "B", "A"], "scene-linear", "scene coordinates; premultiplied RGBA", f"OCIO role scene_linear ({scene_linear_space})"),
        "depth": ("depth.Z", ["Z"], "metres", "camera-space +Z distance; positive forward", "Raw"),
        "normals": ("normals.X/Y/Z", ["X", "Y", "Z"], "unitless", f"{normals_space}-space; right-handed; +Y up", "Raw"),
        "motion_forward": ("motion_forward.X/Y", ["X", "Y"], "pixels per frame", "image coordinates; +X right, +Y down; next frame", "Raw"),
        "motion_backward": ("motion_backward.X/Y", ["X", "Y"], "pixels per frame", "image coordinates; +X right, +Y down; previous frame", "Raw"),
        "object_ids": ("object_id.id", ["id"], "Cryptomatte uint32 ID bits", "Cryptomatte name_to_bits; bit-preserved float32; zero is background", "Raw"),
    }
    for frame in frames:
        sample, camera = samples[frame], records[frame].get("camera_data")
        if camera is None:
            raise ValueError(f"SceneState frame {frame} has no camera")
        width, height = map(int, records[frame]["resolution"])
        if tuple(camera["resolution"]) != (width, height):
            raise ValueError(f"SceneState camera resolution mismatch at frame {frame}")
        shape = (height, width)
        arrays = {key: _rgba(sample[key], key, shape) for key in layer_specs}
        for key in ("beauty", "depth", "normals", "motion_forward", "motion_backward"):
            if not np.all(np.isfinite(arrays[key])):
                raise ValueError(f"{key} contains non-finite values")
        depth_values = arrays["depth"][..., 0]
        clipped = (depth_values > 0) & ((depth_values < camera["near"]) | (depth_values > camera["far"]))
        if np.any(clipped):
            raise ValueError(f"depth pass is outside SceneState near/far clipping at frame {frame}")
        ids = {str(obj["id"]): obj["name"] for obj in records[frame]["objects"]}
        ids_float = arrays["object_ids"][..., 0]
        known = {int(value, 16) for value in ids}
        observed = set(np.unique(ids_float.view(np.uint32)).tolist()) - {0}
        if not observed.issubset(known):
            raise ValueError(f"object ID pass has IDs absent from SceneState at frame {frame}")
        import OpenImageIO as oiio
        path = root / f"frame.{frame:04d}.exr"
        names = ["R", "G", "B", "A"]
        buffers = [arrays["beauty"]]
        for key in ("depth", "normals", "motion_forward", "motion_backward", "object_ids"):
            value = arrays[key]
            channels = {"depth": ("Z",), "normals": ("X", "Y", "Z"),
                        "motion_forward": ("X", "Y"), "motion_backward": ("X", "Y"),
                        "object_ids": ("id",)}[key]
            prefix = {"motion_forward": "motion_forward", "motion_backward": "motion_backward",
                      "object_ids": "object_id"}.get(key, key)
            names.extend(f"{prefix}.{channel}" for channel in channels)
            buffers.append(value[..., :len(channels)])
        pixels = np.ascontiguousarray(np.concatenate(buffers, axis=2), dtype=np.float32)
        spec = oiio.ImageSpec(width, height, pixels.shape[2], oiio.TypeDesc(oiio.FLOAT))
        spec.channelnames = names
        spec.attribute("oiio:ColorSpace", scene_linear_space)
        spec.attribute("compression", "zips")
        spec.attribute("DateTime", CONTROL_BUNDLE_DATETIME)
        out = oiio.ImageOutput.create(str(path))
        if out is None or not out.open(str(path), spec):
            raise ValueError(f"ControlBundle cannot write {path}")
        try:
            if not out.write_image(pixels):
                raise ValueError(f"ControlBundle EXR write failed: {out.geterror()}")
        finally:
            out.close()
        manifest["frames"].append({
            "frame": frame, "file": path.name, "pixel_size": [width, height],
            "pixel_aspect": records[frame]["pixel_aspect"],
            "near_metres": camera["near"], "far_metres": camera["far"],
            "shutter": SHUTTER_CONVENTION,
            "layers": {key: {"name": val[0], "channels": val[1], "units": val[2],
                              "coordinate_convention": val[3], "color_space": val[4]}
                       for key, val in layer_specs.items()},
            "object_ids": ids,
        })
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    from .artifacts import ArtifactStore
    store = ArtifactStore()
    scene_path = Path(scene_state_path)
    scene_sidecar = scene_path.with_suffix(scene_path.suffix + ".artifact.json")
    if scene_sidecar.exists():
        scene_id = json.loads(scene_sidecar.read_text(encoding="utf-8"))["artifact_id"]
    else:
        scene_id = store.put_files({scene_path.name: scene_path,
            scene_path.with_suffix(".npz").name: scene_path.with_suffix(".npz")}, "scene_state", {
            "producer": "SceneState", "version": SCHEMA_VERSION, "inputs": [],
            "document": str(scene_path), "frame_range": [first, last], "time": __import__("time").time()})
    bundle_files = {manifest_path.name: manifest_path}
    bundle_files.update({p.name: p for p in root.glob("frame.*.exr")})
    bundle_id = store.put_files(bundle_files, "control_bundle", {"producer": "ControlBundle", "version": BUNDLE_VERSION,
        "inputs": [{"id": scene_id}], "document": str(scene_path), "frame_range": [first, last], "time": __import__("time").time()})
    manifest_path.with_suffix(".artifact.json").write_text(
        json.dumps({"artifact_id": bundle_id}, indent=2) + "\n", encoding="utf-8")
    return (manifest_path, bundle_id) if return_artifact_id else manifest_path


def read_control_bundle(manifest_path, scene_state_path):
    """Read typed planes after proving the paired SceneState schema and file identity."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Unsupported ControlBundle version {manifest.get('bundle_version')!r}")
    version = manifest.get("scene_state_schema_version")
    if version != SCHEMA_VERSION:
        raise SceneStateVersionMismatch(
            f"SceneStateVersionMismatch: bundle expects {version}, reader supports {SCHEMA_VERSION}")
    actual_state = Path(scene_state_path)
    state = read_scene_state(actual_state)
    if state["schema_version"] != version:
        raise SceneStateVersionMismatch(
            f"SceneStateVersionMismatch: bundle expects {version}, file has {state['schema_version']}")
    digest = hashlib.sha256(actual_state.read_bytes()).hexdigest()
    if digest != manifest.get("scene_state_sha256"):
        raise ValueError("SceneStateMismatch: this ControlBundle was written for a different SceneState file")
    import OpenImageIO as oiio
    import PyOpenColorIO as ocio
    scene_linear_space = ocio_config().getRoleColorSpace(ocio.ROLE_SCENE_LINEAR)
    result = []
    for record in manifest["frames"]:
        path = manifest_path.parent / record["file"]
        image = oiio.ImageInput.open(str(path))
        if image is None:
            raise ValueError(f"Cannot read ControlBundle EXR {path}")
        try:
            spec = image.spec()
            if [spec.width, spec.height] != record["pixel_size"]:
                raise ValueError("ControlBundle pixel size does not match its manifest")
            if str(spec.format) != "float" or manifest.get("format") != (
                    "OpenEXR 32-bit float, multichannel, scene-linear beauty"):
                raise ValueError("ControlBundle format does not match its manifest contract")
            expected_channels = ["R", "G", "B", "A", "depth.Z", "normals.X", "normals.Y",
                                 "normals.Z", "motion_forward.X", "motion_forward.Y",
                                 "motion_backward.X", "motion_backward.Y", "object_id.id"]
            if set(spec.channelnames) != set(expected_channels) or len(spec.channelnames) != len(expected_channels):
                raise ValueError("ControlBundle EXR channels do not match its manifest contract")
            from .media import auto_space
            if auto_space(".exr", spec) != scene_linear_space:
                raise ValueError("ControlBundle EXR color-space tag does not match the scene_linear OCIO role")
            raw = image.read_image(oiio.FLOAT)
            data = np.asarray(raw, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
            channels = {name: i for i, name in enumerate(spec.channelnames)}
            def take(names):
                return data[..., [channels[name] for name in names]].copy()
            def plane(values, key):
                meta = record["layers"][key]
                return ControlPlane(values, meta["units"], meta["coordinate_convention"], meta["color_space"])
            ids = take(("object_id.id",))[..., 0]
            ids_u32 = ids.view(np.uint32)
            result.append(ControlFrame(
                record["frame"], plane(take(("R", "G", "B", "A")), "beauty"),
                plane(take(("depth.Z",)), "depth"),
                plane(take(("normals.X", "normals.Y", "normals.Z")), "normals"),
                plane(take(("motion_forward.X", "motion_forward.Y")), "motion_forward"),
                plane(take(("motion_backward.X", "motion_backward.Y")), "motion_backward"),
                plane(ids_u32, "object_ids"), record["object_ids"]))
        finally:
            image.close()
    return result
