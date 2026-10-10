"""Solve the camera from the plate alone and compare it with the camera that rendered it.

Survey points of the set (known world coordinates) are clicked in the first frame (here: the true
projection rounded to a whole pixel, the precision of a click), tracked through the plate with
nodebased.tracker, and the camera is fitted to the tracks on every frame starting from a generic
guess that is not the true camera. The same fit and the same error measures are the ones the
conditioning verifier uses (nodebased.conditioning_verify).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

from nodebased import scene3d, tracker  # noqa: E402
from nodebased import conditioning_verify as cv  # noqa: E402

PATTERN_RADIUS = 7
SEARCH_RADIUS = 8
OUTLIER_PX = 1.5      # a track further than this from the fitted camera is dropped and the fit repeated


def truth_camera(frame):
    p = shot.camera_at(frame)
    return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(p["tx"], p["ty"], p["tz"])),
                          scene3d.Vec3(p["target_x"], p["target_y"], p["target_z"]),
                          fov_from_focal(shot.FOCAL))


def fov_from_focal(focal):
    from nodebased import filmback
    return filmback.fov_from_aperture(focal, filmback.DEFAULT_VAPERTURE)


def generic_guess():
    """The camera an artist would start from: the default distance and lens, looking at the origin."""
    return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0.0, 1.0, 7.0)), scene3d.Vec3(0.0, 0.0, 0.0),
                          fov_from_focal(30.0))


def _robust_fit(base, world, image, width, height):
    """Fit, drop tracks that disagree with the fit by more than OUTLIER_PX, fit again."""
    keep = np.ones(len(world), bool)
    for _ in range(4):
        camera, rms = cv._fit_camera(base, world[keep], image[keep], width, height)
        pixels, _depth = scene3d.project(camera, width, height, world)
        error = np.linalg.norm(pixels - image, axis=1)
        now = error < max(OUTLIER_PX, 3 * float(np.median(error[keep])))
        if now.sum() < 10 or (now == keep).all():
            break
        keep = now
    return camera, rms


def solve(frames, width, height):
    """`frames` maps frame number to an HxW luminance-capable array. Returns the report dict."""
    numbers = sorted(frames)
    first, last = numbers[0], numbers[-1]
    points = np.array(shot.landmarks(), dtype=np.float64)
    reference = truth_camera(first)
    clicks, depth = scene3d.project(reference, width, height, points)
    margin = 14 + SEARCH_RADIUS
    keep = ((depth > 0.5) & (clicks[:, 0] > margin) & (clicks[:, 1] > margin)
            & (clicks[:, 0] < width - margin) & (clicks[:, 1] < height - margin))
    points, clicks = points[keep], np.round(clicks[keep])
    tracks, used = [], []
    for world, click in zip(points, clicks):
        try:
            tracks.append(tracker.analyse(frames, first, click, pattern_radius=PATTERN_RADIUS,
                                          search_radius=SEARCH_RADIUS, first_frame=first, last_frame=last))
            used.append(world)
        except tracker.AnalysisError:
            continue
    used = np.array(used)
    base, solved, rows = generic_guess(), {}, []
    for frame in numbers:
        image = np.array([track[frame] for track in tracks], dtype=np.float64)
        camera, rms = _robust_fit(base, used, image, width, height)
        base = camera                      # the next frame starts from this one
        measured = cv._camera_measurement(truth_camera(frame), camera)
        solved[frame] = camera
        rows.append({"frame": frame, "rms_px": round(rms, 4), **{k: round(v, 5) for k, v in measured.items()}})
    return {"landmarks": len(used), "frames": rows}, solved


def camera_keys(solved):
    """Per-frame Camera3D parameter values of the solved cameras."""
    from nodebased import filmback
    out = {}
    for frame, cam in solved.items():
        eye, target = cam.transform.position.array(), cam.target.array()
        out[frame] = dict(tx=float(eye[0]), ty=float(eye[1]), tz=float(eye[2]), target_x=float(target[0]),
                          target_y=float(target[1]), target_z=float(target[2]), roll=float(cam.roll),
                          focal=float(filmback.focal_from_fov(cam.fov, filmback.DEFAULT_VAPERTURE)))
    return out


def summarise(report):
    rows = report["frames"]
    return {"landmarks": report["landmarks"],
            "max_position_error": max(r["position_error_m"] for r in rows),
            "max_rotation_error_degrees": max(r["rotation_error_degrees"] for r in rows),
            "max_fov_error_degrees": max(r["fov_error_degrees"] for r in rows),
            "max_reprojection_rms_px": max(r["rms_px"] for r in rows)}


def read_plate(folder, frames=shot.FRAMES):
    from nodebased.media import read_media_raster
    out = {}
    for frame in range(1, frames + 1):
        raster = read_media_raster(shot.plate_pattern(folder) % frame)
        out[frame] = np.asarray(raster.to_display(), dtype=np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", help="folder holding plate.%%04d.exr (default: the NodeBased cache)")
    args = parser.parse_args()
    frames = read_plate(Path(args.plate) if args.plate else shot.plate_dir())
    height, width = frames[1].shape[:2]
    report, solved = solve(frames, width, height)
    keys = camera_keys(solved)
    summary = summarise(report)
    (shot.HERE / "solve.json").write_text(json.dumps({"summary": summary, "frames": report["frames"],
                                                       "camera_keys": {str(f): k for f, k in keys.items()}},
                                                      indent=1) + "\n")
    shot.save_graph(shot.HERE / "demo_shot.nbcomp", width, height, keys)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
