"""Solve the camera from the plate alone and compare it with the camera that rendered it.

`--solver survey` (the default): survey points of the set (known world coordinates) are clicked in the first
frame (here: the true projection rounded to a whole pixel, the precision of a click), tracked through the plate
with nodebased.tracker, and the camera is fitted to the tracks on every frame starting from a generic guess that
is not the true camera. The same fit and the same error measures are the ones the conditioning verifier uses
(nodebased.conditioning_verify).

`--solver dpvo` solves the plate with no survey at all, through the CameraSolve3D node (the GPU; DPV-SLAM), and
`--lens estimate` lets the node recover the focal length (and `--distortion k1` the radial lens) from the
solver's own tracked points. The plate is rendered by make_plate.py at `--width` (a plain one, or the lens plate
with `--plate-k1`), the typed focal length is `--typed-focal-scale` times the truth, and the recovered lens and
camera are compared with the camera that rendered it. `--matrix` runs every combination and writes
lens_benchmark.json:

    python examples/demo_shot/solve_camera.py --solver dpvo --lens estimate --distortion k1 --width 960 --plate-k1 -0.03
    python examples/demo_shot/solve_camera.py --solver dpvo --matrix

`--distorted` is the solve of the lens round trip: it renders the distorted plate (make_plate.py --distorted), solves
the camera with the lens recovered (focal length, k1 and k2) and writes `demo_shot_lens.nbcomp` (the shot graph with the
solved camera, its overscan camera and the Undistort / Redistort round trip) and `solve_lens.json` (the recovered lens and
the errors against the truth). With the default `--solver survey` the camera comes from the set's surveyed landmarks
(clicked on frame 1, tracked through the plate, one adjustment over every frame's pose and the one shared lens); it runs
on the processor in seconds. With `--solver dpvo` the CameraSolve3D node solves the plate with no survey but two picks for
the scale (the GPU, under the GPU lock) and the result goes to `demo_shot_lens_dpvo.nbcomp` and `solve_lens_dpvo.json`,
which are not the shot's default (the lens does not come out of that solve, see the README). Later runs of the shot do
not solve again: the camera that feeds the export is the one kept in the graph.

    python examples/demo_shot/solve_camera.py --distorted
    python examples/demo_shot/solve_camera.py --solver dpvo --distorted
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
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



# ----------------------------------------------------------------------------- the lens from surveyed points

def solve_lens_landmarks(frames, width, height, plate_lens, typed_focal_mm, log=print):
    """Solve the camera of a distorted plate from the surveyed landmarks, and recover the lens with it.

    The landmarks are clicked on the first frame (the true projection through `plate_lens`, rounded to a whole pixel, the
    precision of a click) and followed through the plate with the tracker. A first camera per frame comes from the plain
    fit of `solve` (it takes the picture for an ideal one). Then one adjustment fits every frame's pose and the one
    lens shared by all of them (focal length, k1, k2), starting from `typed_focal_mm` and no distortion: the landmarks'
    world positions are known, so the lens is not entangled with the structure as in a solve with no survey (whose
    reprojection cost is flat across k1 on this move, see the README). Returns (cameras, lens block, rows)."""
    from nodebased import camera_solve as cs, filmback, lensrefine as lr
    numbers = sorted(frames)
    first, last = numbers[0], numbers[-1]
    points = np.array(shot.landmarks(), dtype=np.float64)
    reference = replace(truth_camera(first), **plate_lens)
    clicks, depth = scene3d.project_lens(reference, width, height, points)
    margin = 14 + SEARCH_RADIUS
    keep = ((depth > 0.5) & (clicks[:, 0] > margin) & (clicks[:, 1] > margin)
            & (clicks[:, 0] < width - margin) & (clicks[:, 1] < height - margin))
    points, clicks = points[keep], np.round(clicks[keep])
    tracks, used = [], []
    for world, click in zip(points, clicks):
        try:
            tracks.append(tracker.analyse(frames, first, click, pattern_radius=PATTERN_RADIUS, search_radius=SEARCH_RADIUS,
                                          first_frame=first, last_frame=last))
            used.append(world)
        except tracker.AnalysisError:
            continue
    world = np.array(used)
    image = np.array([[track[f] for track in tracks] for f in numbers], dtype=np.float64)       # (frames, points, 2)
    base, poses = generic_guess(), []
    for index, frame in enumerate(numbers):
        camera, _rms = _robust_fit(base, world, image[index], width, height)
        base = camera
        eye, target = camera.transform.position.array(), camera.target.array()
        poses.append(cs.pose_from_keys(*map(float, eye), *map(float, target), float(camera.roll)))
    rotation0, translation0 = lr.world_to_camera(np.array(poses))
    count, frame_count = len(world), len(numbers)
    start = np.array([math.log(typed_focal_mm * height / filmback.DEFAULT_VAPERTURE), 0.0, 0.0])
    mask = np.ones((frame_count, count), bool)

    def unpack(x):
        rotation = lr._exp_rotation(x[3:].reshape(-1, 6)[:, :3]) @ rotation0
        return rotation, x[3:].reshape(-1, 6)[:, 3:] + translation0, math.exp(x[0]), x[1], x[2]

    def residual(x):
        rotation, translation, focal, k1, k2 = unpack(x)
        camera = np.einsum("fij,pj->fpi", rotation, world) + translation[:, None, :]
        pixels = lr.camera_pixels(camera.reshape(-1, 3), focal, k1, k2, width, height).reshape(frame_count, count, 2)
        out = (pixels - image)
        out[camera[..., 2] <= 1e-6] = 1e3
        return out

    x = np.concatenate((start, np.zeros(frame_count * 6)))
    damping = 1e-3
    for stage in range(3):
        current = residual(x)[mask].reshape(-1)
        for _ in range(60):
            jacobian = np.empty((len(current), len(x)))
            for column in range(len(x)):
                step = 1e-6 if column >= 3 else 1e-5
                shifted = x.copy(); shifted[column] += step
                jacobian[:, column] = (residual(shifted)[mask].reshape(-1) - current) / step
            weights = np.repeat(np.minimum(1.0, 3.0 / np.maximum(np.linalg.norm(current.reshape(-1, 2), axis=1), 1e-12)), 2)
            normal = (jacobian * weights[:, None]).T @ (jacobian * weights[:, None]) + damping * np.eye(len(x))
            delta = np.linalg.solve(normal, -(jacobian * weights[:, None]).T @ (current * weights))
            trial = residual(x + delta)[mask].reshape(-1)
            if trial @ trial < current @ current:
                x, current = x + delta, trial
                damping = max(damping * 0.3, 1e-10)
                if np.linalg.norm(delta) < 1e-9:
                    break
            else:
                damping = min(damping * 10, 1e8)
        error = np.linalg.norm(residual(x), axis=2)
        mask = error < max(OUTLIER_PX, 3 * float(np.median(error[mask])))
    rotation, translation, focal_px, k1, k2 = unpack(x)
    error = np.linalg.norm(residual(x), axis=2)
    rms = float(np.sqrt(np.mean(error[mask] ** 2)))
    focal_mm = round(focal_px * filmback.DEFAULT_VAPERTURE / height, 4)
    fov = cs.fov_from_intrinsics(height, focal_px)
    placed = lr.camera_to_world(rotation, translation)
    cameras = {frame: cs.solve_camera_object(placed[i], fov, (k1, k2)) for i, frame in enumerate(numbers)}
    rows = [{"frame": frame, "rms_px": round(float(np.sqrt(np.mean(error[i][mask[i]] ** 2))), 4),
             **{k: round(v, 5) for k, v in cv._camera_measurement(truth_camera(frame), cameras[frame]).items()}}
            for i, frame in enumerate(numbers)]
    block = {"landmarks": count, "observations": int(mask.sum()), "of": int(mask.size), "focal_mm": focal_mm,
             "k1": round(float(k1), 5), "k2": round(float(k2), 5), "rms_px": round(rms, 4)}
    log(f"  landmark solve: {count} landmarks, {int(mask.sum())} of {mask.size} tracked positions used, "
        f"rms {rms:.3f} px, focal {focal_mm} mm, k1 {k1:.4f}, k2 {k2:.4f}")
    return cameras, block, rows


# ----------------------------------------------------------------------------- the DPVO lens benchmark

BENCHMARK = shot.HERE / "lens_benchmark.json"
WIDTHS = (960, 1920)
# (typed focal as a multiple of the truth, the node's focal mode, its distortion): what the artist could ask for
SETTINGS = ((1.0, "known", "none"), (1.0, "known", "k1"), (1.0, "estimate", "none"), (1.0, "estimate", "k1"),
            (1.2, "estimate", "none"), (1.2, "estimate", "k1"), (1.2, "estimate", "k1 k2"))


def lens_plate_folder(cache, width, k1):
    """The plate at `width` (16:9), rendered once into `cache` by make_plate.py; with `k1` the lens plate."""
    import make_plate
    folder = Path(cache) / "lens_plates" / f"w{width}_k1_{k1:g}"
    if not (folder / (shot.plate_pattern(".").split("/")[-1] % shot.FRAMES)).exists():
        make_plate.render_plate(folder, (width, width * 9 // 16), shot.FRAMES, k1=k1)
    return folder


def similarity_errors(document, camera_id, first, last):
    """How far the solved camera is from the rendering camera once the solve is fitted to it by a similarity.

    The rotation of the fit comes from the camera orientations (the path is nearly a straight line, so positions
    alone leave the roll about it free), the scale and shift from the positions. Returns the fitted scale, the root
    mean square position error (scene units: metres) and the mean and maximum rotation error in degrees. The solve
    is monocular, so scale and placement are not the thing under test."""
    from nodebased import camera_solve as cs
    curves = document["animation"]["curves"][camera_id]
    frames = range(first, last + 1)
    solved = [cs.pose_from_keys(*(np.interp(f, [k["frame"] for k in curves[name]["keys"]], [k["value"] for k in curves[name]["keys"]])
                                 for name in cs.CAMERA_PARAMS)) for f in frames]
    truth = [cs.pose_from_keys(*(shot.camera_at(f)[n] for n in ("tx", "ty", "tz", "target_x", "target_y", "target_z")), 0.0)
             for f in frames]
    u, _singular, vt = np.linalg.svd(sum(t[:3, :3] @ s[:3, :3].T for s, t in zip(solved, truth)))
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    spun = np.array([rotation @ s[:3, 3] for s in solved])
    want = np.array([t[:3, 3] for t in truth])
    spun_c, want_c = spun - spun.mean(axis=0), want - want.mean(axis=0)
    scale = float(np.sum(spun_c * want_c) / np.sum(spun_c * spun_c))
    position = scale * spun_c - want_c
    angles = [math.degrees(math.acos(float(np.clip((np.trace((rotation @ s[:3, :3]).T @ t[:3, :3]) - 1) / 2, -1, 1))))
              for s, t in zip(solved, truth)]
    return {"fitted_scale": scale, "position_rms_m": float(np.sqrt(np.mean(np.sum(position ** 2, axis=1)))),
            "rotation_mean_deg": float(np.mean(angles)), "rotation_max_deg": float(np.max(angles))}


def run_dpvo(width, plate_k1, settings, cache, folder=None, log=print, save_solves=None):
    """Solve the demo plate at `width` with each of `settings`; returns one result row per setting.

    The plate comes from `folder` or is rendered into `cache`. DPV-SLAM runs once per typed focal length (the
    refinement is a separate, deterministic step on what it returns), under the GPU lock."""
    from nodebased import camera_solve as cs, filmback
    from nodebased.artifacts import ArtifactStore
    from nodebased.core import Dispatcher
    from nodebased.gpulock import gpu_job
    from nodebased.imaging import Evaluator
    from nodebased.providers import camera_solve_dpvo as dpvo
    reason = dpvo.available()
    if reason:
        raise SystemExit(f"cannot run the camera solve: {reason}")
    folder = Path(folder) if folder else lens_plate_folder(cache, width, plate_k1)
    height = width * 9 // 16
    dispatcher = Dispatcher()
    start = shot.CAMERA_START
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Read", "params": {"path": shot.plate_pattern(folder)}},
        {"op": "create", "id": "solve", "type": "CameraSolve3D", "params": {
            "seed": 0, "anchor_tx": start["tx"], "anchor_ty": start["ty"], "anchor_tz": start["tz"],
            "anchor_target_x": start["target_x"], "anchor_target_y": start["target_y"], "anchor_target_z": start["target_z"]}},
        {"op": "connect", "id": "solve", "input": "image", "source": "plate"},
        {"op": "time", "first": 1, "last": shot.FRAMES, "current": 1}]})
    evaluator, store = Evaluator(), ArtifactStore(Path(cache) / "store")
    solves, rows = {}, []

    def solver(frames, intrinsics, **kwargs):
        key = round(intrinsics[1], 3)
        if key not in solves:
            started = time.perf_counter()
            solves[key] = dpvo.run_solve(frames, intrinsics, **kwargs)
            solves[key]["info"]["wall_s"] = round(time.perf_counter() - started, 1)
            if save_solves:      # the solver's raw answer, so the refinement can be rerun and studied without the card
                np.savez(Path(save_solves) / f"solve_w{width}_k1_{plate_k1:g}_f{key}.npz", c2w=solves[key]["c2w"],
                         points=solves[key]["points"], colors=solves[key]["colors"],
                         **{f"track_{n}": v for n, v in solves[key]["tracks"].items()})
            log(f"  DPVO at focal {intrinsics[1]:.1f} px: {solves[key]['info']['wall_s']} s, "
                f"{len(solves[key]['points'])} points, {len(solves[key]['tracks']['frame'])} tracked positions")
        return solves[key]

    with gpu_job(log=log):
        for scale, mode, distortion in settings:
            typed = shot.FOCAL * scale
            for name, value in (("focal", typed), ("solve_focal", mode), ("solve_distortion", distortion)):
                dispatcher.execute({"op": "set", "id": "solve", "param": name, "value": value})
            reply = cs.solve_camera_op(evaluator, dispatcher, {"op": "solve_camera", "id": "solve", "camera": "camera"},
                                       store=store, solver=solver)
            lens = reply["lens"]
            camera = dispatcher.document["nodes"]["camera"]["params"]
            row = {"width": width, "plate_k1": plate_k1, "typed_focal_scale": scale, "focal_mode": mode,
                   "distortion": distortion, "true_focal_mm": round(shot.FOCAL, 4),
                   "recovered_focal_mm": round(camera["focal"], 4), "recovered_k1": camera["lens_k1"],
                   "recovered_k2": camera["lens_k2"], "dpvo_wall_s": solves[round(dpvo.focal_pixels(typed, filmback.DEFAULT_VAPERTURE, height), 3)]["info"]["wall_s"],
                   "focal_error_percent": round(100.0 * (camera["focal"] / shot.FOCAL - 1.0), 3),
                   "k1_error": None if not plate_k1 and distortion == "none" else round(camera["lens_k1"] - plate_k1, 5),
                   "lens": lens, **{k: round(v, 5) for k, v in similarity_errors(
                       dispatcher.document, "camera", 1, shot.FRAMES).items()}}
            rows.append(row)
            log(f"  typed {scale:.2f}x {mode}/{distortion}: focal {row['recovered_focal_mm']} mm "
                f"({row['focal_error_percent']:+.2f}%), k1 {row['recovered_k1']:.4f} (plate {plate_k1:g}), "
                f"position {row['position_rms_m'] * 100:.1f} cm, rotation {row['rotation_mean_deg']:.3f} deg")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", help="folder holding plate.%%04d.exr (default: the NodeBased cache)")
    parser.add_argument("--solver", choices=("survey", "dpvo"), default="survey")
    parser.add_argument("--lens", choices=("known", "estimate"), default="known", help="dpvo: type the focal length or recover it")
    parser.add_argument("--distortion", choices=("none", "k1", "k1 k2"), default=None,
                        help="dpvo: radial terms to recover (default: k1 with --lens estimate, otherwise none)")
    parser.add_argument("--width", type=int, default=960, help="dpvo: plate width (16:9)")
    parser.add_argument("--plate-k1", type=float, default=0.0, help="dpvo: render the lens plate with this radial distortion")
    parser.add_argument("--typed-focal-scale", type=float, default=1.2, help="dpvo: the typed focal length over the truth")
    parser.add_argument("--distorted", action="store_true",
                        help="solve the distorted plate with the lens recovered; writes demo_shot_lens.nbcomp (--solver dpvo: demo_shot_lens_dpvo.nbcomp)")
    parser.add_argument("--matrix", action="store_true", help="dpvo: every plate, width and setting; writes lens_benchmark.json")
    parser.add_argument("--cache", default=str(shot.default_cache()), help="dpvo: where plates and the artifact store go")
    parser.add_argument("--save-solves", help="dpvo: also keep each raw solver answer here as an .npz (to rerun the refinement offline)")
    parser.add_argument("--out", help="dpvo: write the result rows here (default with --matrix: lens_benchmark.json)")
    args = parser.parse_args()
    if args.distorted:
        return main_distorted(args) if args.solver == "dpvo" else main_distorted_landmarks(args)
    if args.solver == "dpvo":
        return main_dpvo(args)
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


def solved_keys(document, camera_id, frames=shot.FRAMES):
    """Per-frame Camera3D parameter values of a solved camera node (curves where it has them, the node's value otherwise)."""
    node, curves = document["nodes"][camera_id], document["animation"]["curves"].get(camera_id, {})
    out = {}
    for frame in range(1, frames + 1):
        row = {}
        for name in ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll", "focal"):
            curve = curves.get(name)
            row[name] = float(np.interp(frame, [k["frame"] for k in curve["keys"]], [k["value"] for k in curve["keys"]])
                              if curve else node["params"][name])
        out[frame] = row
    return out


def solve_distorted(plate_folder, cache, log=print):
    """Solve the distorted plate with the CameraSolve3D node: typed focal length 1.2 times the truth, `estimate` for
    the focal length, k1 and k2, and a survey of two picks and their distance. Returns (document, reply, picks)."""
    from nodebased import camera_solve as cs
    from nodebased.artifacts import ArtifactStore
    from nodebased.core import Dispatcher
    from nodebased.gpulock import gpu_job
    from nodebased.imaging import Evaluator
    from nodebased.providers import camera_solve_dpvo as dpvo
    reason = dpvo.available()
    if reason:
        raise SystemExit(f"cannot run the camera solve: {reason}")
    size = shot.DISTORTED_SIZE
    picks, distance = shot.pick_survey(size, shot.lens_of(shot.DISTORTED_K1, shot.DISTORTED_K2), plate_folder=plate_folder)
    start = shot.CAMERA_START
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Read", "params": {"path": shot.plate_pattern(plate_folder)}},
        {"op": "create", "id": "solve", "type": "CameraSolve3D", "params": {
            "seed": 0, "focal": shot.FOCAL * shot.SOLVE_TYPED_FOCAL_SCALE, "solve_focal": "estimate", "solve_distortion": "k1 k2",
            "survey": shot.survey_json(picks, distance), "anchor_tx": start["tx"], "anchor_ty": start["ty"],
            "anchor_tz": start["tz"], "anchor_target_x": start["target_x"], "anchor_target_y": start["target_y"],
            "anchor_target_z": start["target_z"]}},
        {"op": "connect", "id": "solve", "input": "image", "source": "plate"},
        {"op": "time", "first": 1, "last": shot.FRAMES, "current": 1}]})
    store = ArtifactStore(Path(cache) / "store-solve")
    started = time.perf_counter()
    with gpu_job(log=log):
        reply = cs.solve_camera_op(Evaluator(), dispatcher, {"op": "solve_camera", "id": "solve", "camera": "camera"}, store=store)
    log(f"solved in {time.perf_counter() - started:.1f} s")
    return dispatcher.document, reply, (picks, distance)


def main_distorted_landmarks(args):
    """The default solve of the distorted plate: the surveyed landmarks and one lens shared by every frame."""
    import make_plate
    folder = Path(args.plate) if args.plate else shot.distorted_plate_dir()
    if not (folder / (shot.plate_pattern(".").split("/")[-1] % shot.FRAMES)).exists():
        make_plate.render_plate(folder, shot.DISTORTED_SIZE, shot.FRAMES, k1=shot.DISTORTED_K1, k2=shot.DISTORTED_K2)
    width, height = shot.DISTORTED_SIZE
    typed = shot.FOCAL * shot.SOLVE_TYPED_FOCAL_SCALE
    started = time.perf_counter()
    cameras, block, rows = solve_lens_landmarks(read_plate(folder), width, height,
                                                shot.lens_of(shot.DISTORTED_K1, shot.DISTORTED_K2), typed)
    seconds = time.perf_counter() - started
    lens = shot.lens_of(block["k1"], block["k2"])
    keys = camera_keys(cameras)
    margins = shot.overscan_margins(lens, shot.DISTORTED_SIZE)
    picks, distance = shot.pick_survey(shot.DISTORTED_SIZE, shot.lens_of(shot.DISTORTED_K1, shot.DISTORTED_K2), plate_folder=folder)
    record = {"schema": 2, "solver": "landmarks", "plate": {"size": list(shot.DISTORTED_SIZE), "k1": shot.DISTORTED_K1, "k2": shot.DISTORTED_K2},
              "typed_focal_mm": round(typed, 4), "true_focal_mm": round(shot.FOCAL, 4),
              "recovered": {"focal_mm": block["focal_mm"], "k1": block["k1"], "k2": block["k2"]},
              "errors": {"focal_percent": round(100.0 * (block["focal_mm"] / shot.FOCAL - 1.0), 3),
                         "k1": round(block["k1"] - shot.DISTORTED_K1, 5), "k2": round(block["k2"] - shot.DISTORTED_K2, 5),
                         "position_max_m": max(r["position_error_m"] for r in rows),
                         "rotation_max_deg": max(r["rotation_error_degrees"] for r in rows)},
              "fit": {k: block[k] for k in ("landmarks", "observations", "of", "rms_px")}, "seconds": round(seconds, 1),
              "survey": {"picks": picks, "distance": round(distance, 4), "note": "the CameraSolve3D node of the graph (a plate with no survey)"},
              "overscan_margins": list(margins), "frames": rows}
    shot.LENS_SOLVE.write_text(json.dumps(record, indent=1) + "\n")
    shot.save_lens_graph(shot.LENS_GRAPH, shot.DISTORTED_SIZE, keys, lens, margins, shot.survey_json(picks, distance),
                         plate_folder=shot.HERE / "plate_distorted")
    print(json.dumps({k: record[k] for k in ("recovered", "errors", "fit", "seconds", "overscan_margins")}, indent=1))
    print(f"wrote {shot.LENS_GRAPH.name} and {shot.LENS_SOLVE.name}")
    return 0


def main_distorted(args):
    import make_plate
    folder = Path(args.plate) if args.plate else shot.distorted_plate_dir()
    if not (folder / (shot.plate_pattern(".").split("/")[-1] % shot.FRAMES)).exists():
        make_plate.render_plate(folder, shot.DISTORTED_SIZE, shot.FRAMES, k1=shot.DISTORTED_K1, k2=shot.DISTORTED_K2)
    document, reply, (picks, distance) = solve_distorted(folder, args.cache)
    camera = document["nodes"]["camera"]["params"]
    lens = shot.lens_of(camera["lens_k1"], camera["lens_k2"])
    keys = solved_keys(document, "camera")
    margins = shot.overscan_margins(lens, shot.DISTORTED_SIZE)
    errors = similarity_errors(document, "camera", 1, shot.FRAMES)
    record = {"schema": 1, "solver": "dpvo", "plate": {"size": list(shot.DISTORTED_SIZE), "k1": shot.DISTORTED_K1, "k2": shot.DISTORTED_K2},
              "typed_focal_mm": round(shot.FOCAL * shot.SOLVE_TYPED_FOCAL_SCALE, 4), "true_focal_mm": round(shot.FOCAL, 4),
              "recovered": {"focal_mm": round(camera["focal"], 4), "k1": round(lens["lens_k1"], 5), "k2": round(lens["lens_k2"], 5)},
              "errors": {"focal_percent": round(100.0 * (camera["focal"] / shot.FOCAL - 1.0), 3),
                         "k1": round(lens["lens_k1"] - shot.DISTORTED_K1, 5), "k2": round(lens["lens_k2"] - shot.DISTORTED_K2, 5),
                         **{k: round(v, 5) for k, v in errors.items()}},
              "survey": {"picks": picks, "distance": round(distance, 4), "scale": reply["scale"], "scale_source": reply["scale_source"]},
              "overscan_margins": list(margins), "lens": reply["lens"], "run": reply["run"]}
    shot.LENS_SOLVE_DPVO.write_text(json.dumps(record, indent=1) + "\n")
    shot.save_lens_graph(shot.LENS_GRAPH_DPVO, shot.DISTORTED_SIZE, keys, lens, margins, shot.survey_json(picks, distance),
                         plate_folder=shot.HERE / "plate_distorted")
    print(json.dumps({k: record[k] for k in ("recovered", "errors", "overscan_margins")}, indent=1))
    print(f"wrote {shot.LENS_GRAPH_DPVO.name} and {shot.LENS_SOLVE_DPVO.name}")
    return 0


def main_dpvo(args):
    if args.matrix:
        runs = [(width, k1, SETTINGS) for k1 in (0.0, shot.LENS_K1) for width in WIDTHS]
    else:
        distortion = args.distortion or ("k1" if args.lens == "estimate" else "none")
        runs = [(args.width, args.plate_k1, ((args.typed_focal_scale if args.lens == "estimate" else 1.0, args.lens, distortion),))]
    rows = []
    from nodebased.gpulock import gpu_job
    if not args.plate:            # the plates are rendered on the processor first, so the card is only held while it is used
        for width, k1, _settings in runs:
            lens_plate_folder(args.cache, width, k1)
    with gpu_job(log=print):      # one wait for the card for the whole matrix; each solve inside sees the lock held
        for width, k1, settings in runs:
            print(f"plate {width} x {width * 9 // 16}, k1 {k1:g}")
            rows += run_dpvo(width, k1, settings, args.cache, folder=args.plate, save_solves=args.save_solves)
    out = Path(args.out) if args.out else (BENCHMARK if args.matrix else None)
    if out:
        out.write_text(json.dumps({"schema": 1, "shot": "demo_shot", "frames": shot.FRAMES, "true_focal_mm": shot.FOCAL,
                                   "lens_plate_k1": shot.LENS_K1, "rows": rows}, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
