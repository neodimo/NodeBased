"""CPU checks for a shot against a versioned SceneState export.

The verifier consumes measured camera solutions and object observations alongside
the rendered frame sequence. The scene-state exporter supplies the reference render
and object-ID pass; this module computes errors and writes a portable JSON report.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np

from .scene_state import read_scene_state


@dataclass(frozen=True)
class Tolerances:
    camera_position_m: float = 0.01
    camera_rotation_degrees: float = 0.25
    camera_fov_degrees: float = 0.1
    object_pixels: float = 2.0
    light_direction_degrees: float = 5.0
    shadow_direction_degrees: float = 5.0
    colour_balance: float = 0.05


def _rotation_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _camera_measurement(reference, measured):
    a = np.asarray(reference["world_transform"], dtype=np.float64)
    b = np.asarray(measured["world_transform"], dtype=np.float64)
    return {
        "position_error_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_error_degrees": _rotation_error(a, b),
        "fov_error_degrees": abs(float(reference["vertical_fov_degrees"])
                                  - float(measured["vertical_fov_degrees"])),
    }


def _centroid(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [float(x.mean() + 0.5), float(y.mean() + 0.5)]


def _object_measurements(frame, observed_ids, width, height):
    """Measure projected object-centre errors against a rendered object-ID pass.

    `observed_ids` maps the exported hexadecimal ID to an HxW segmentation mask.
    The reference pass uses the raster renderer's 1-based geometry IDs; its object
    order follows the manifest emitted by SceneState.
    """
    reference_ids = frame["passes"].get("object_id")
    if reference_ids is None:
        return [{"object_id": row["id"], "error_pixels": None,
                 "pass": False, "reason": "export has no object-ID pass"}
                for row in frame["objects"]]
    ids = np.asarray(reference_ids)
    if ids.ndim == 3:
        ids = ids[..., 0]
    results = []
    for index, row in enumerate(frame["objects"], 1):
        observed = observed_ids.get(row["id"])
        reference = _centroid(ids == index)
        actual = None if observed is None else _centroid(np.asarray(observed, dtype=bool))
        error = (None if reference is None or actual is None else
                 float(np.linalg.norm(np.asarray(reference) - np.asarray(actual))))
        results.append({"object_id": row["id"], "error_pixels": error,
                        "reference_centroid": reference, "observed_centroid": actual})
    return results


def _colour_balance(reference, observed):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] < 3:
        return None
    ma = np.mean(a[..., :3], axis=(0, 1)); mb = np.mean(b[..., :3], axis=(0, 1))
    ma /= max(float(ma.sum()), 1e-12); mb /= max(float(mb.sum()), 1e-12)
    return float(np.linalg.norm(ma - mb))


def verify_conditioning(scene_state_path, observations, output_path, *, tolerances=None):
    """Write ``.json`` and ``.txt`` shot reports.

    `observations` is keyed by integer frame. A row may contain `camera` (the camera
    estimate produced upstream), `object_ids` (ID masks keyed by exported ID), `beauty`,
    `dominant_light_direction`, and `shadow_direction`. Light and shadow directions
    are 3-vectors in world space; colour is measured from the supplied beauty image.
    """
    tolerances = tolerances or Tolerances()
    state = read_scene_state(scene_state_path)
    report = {"schema_version": 1, "scene_state": Path(scene_state_path).name,
              "intensity_checked": False, "frames": []}
    for frame in state["frames"]:
        number = int(frame["frame"]); observation = observations.get(number, {})
        camera = None
        if frame.get("camera_data") is not None and observation.get("camera") is not None:
            camera = _camera_measurement(frame["camera_data"], observation["camera"])
            camera["pass"] = (camera["position_error_m"] <= tolerances.camera_position_m
                              and camera["rotation_error_degrees"] <= tolerances.camera_rotation_degrees
                              and camera["fov_error_degrees"] <= tolerances.camera_fov_degrees)
        objects = _object_measurements(frame, observation.get("object_ids", {}),
                                       *frame.get("resolution", (0, 0)))
        for item in objects:
            item["pass"] = item["error_pixels"] is not None and item["error_pixels"] <= tolerances.object_pixels
        lighting = {"intensity_checked": False}
        supplied_lights = observation.get("dominant_light_direction")
        expected_lights = observation.get("reference_light_direction")
        if supplied_lights is not None and expected_lights is not None:
            a, b = np.asarray(supplied_lights, dtype=float), np.asarray(expected_lights, dtype=float)
            cosine = np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1)
            lighting["light_direction_error_degrees"] = float(np.degrees(np.arccos(cosine)))
        if observation.get("shadow_direction") is not None and observation.get("reference_shadow_direction") is not None:
            a, b = np.asarray(observation["shadow_direction"], dtype=float), np.asarray(observation["reference_shadow_direction"], dtype=float)
            cosine = np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1)
            lighting["shadow_direction_error_degrees"] = float(np.degrees(np.arccos(cosine)))
        if observation.get("beauty") is not None and frame.get("passes", {}).get("beauty") is not None:
            lighting["colour_balance_error"] = _colour_balance(frame["passes"]["beauty"], observation["beauty"])
        lighting_metrics = ("light_direction_error_degrees", "shadow_direction_error_degrees",
                            "colour_balance_error")
        lighting["pass"] = all(key in lighting for key in lighting_metrics) and all((
            lighting["light_direction_error_degrees"] <= tolerances.light_direction_degrees,
            lighting["shadow_direction_error_degrees"] <= tolerances.shadow_direction_degrees,
            lighting["colour_balance_error"] <= tolerances.colour_balance))
        if not lighting["pass"] and not all(key in lighting for key in lighting_metrics):
            lighting["reason"] = "light direction, shadow direction, or colour evidence missing"
        bindings = {
            "camera": {"lock_state": "locked", **camera} if camera else {"lock_state": "locked", "pass": False, "reason": "camera solve missing"},
            "objects": [{"lock_state": "locked", **item} for item in objects],
            "lighting": {"lock_state": "locked", **lighting},
        }
        report["frames"].append({"frame": number, "bindings": bindings})
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [f"Conditioning verification: {json_path.stem}",
             "Lighting intensity is not checked."]
    for item in report["frames"]:
        bindings = item["bindings"]
        lines.append(f"Frame {item['frame']}: camera={'PASS' if bindings['camera']['pass'] else 'FAIL'}, "
                     f"objects={'PASS' if all(x['pass'] for x in bindings['objects']) else 'FAIL'}, "
                     f"lighting={'PASS' if bindings['lighting']['pass'] else 'FAIL'}")
        if bindings["camera"].get("position_error_m") is not None:
            c = bindings["camera"]
            lines.append(f"  Camera errors: {c['position_error_m']:.4f} m, "
                         f"{c['rotation_error_degrees']:.3f} degrees, FOV {c['fov_error_degrees']:.3f} degrees")
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
