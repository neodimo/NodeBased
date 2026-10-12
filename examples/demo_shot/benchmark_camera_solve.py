"""Benchmark the camera solvers behind CameraSolve3D on the demo plate or the stand-in plate (stage 9).

Renders the plate (or reuses it from --work), runs the solver twice with seed 0 through the provider's own
subprocess (the same call the `solve_camera` op makes, holding the GPU lock), and measures against the known
camera: pose error after a similarity alignment (rotation from the camera orientations, scale and shift from the
positions, as stage 7 did), the focal length the solver used or recovered against the true one, frames per
second after the warm-up, the solver process's peak card memory, and how far the two runs differ.

    python examples/demo_shot/benchmark_camera_solve.py --solver mast3r|dpvo [--plate demo|standin] [--size 960x540]
        [--focal-scale 1.0] [--runs 2] [--work DIR] [--json FILE]

DPV-SLAM is handed the true focal length times --focal-scale (1.15 shows what a wrong lens does); MASt3R-SLAM is
handed nothing. Plates over 1 GB never go in /tmp: pass --work on the big scratch disk.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

sys.path.insert(0, str(shot.HERE.parents[1]))


def align_and_measure(est, gt):
    """Similarity-align estimated camera-to-world poses to the truth; translation and rotation errors."""
    n = min(len(est), len(gt))
    est, gt = np.asarray(est[:n], np.float64), np.asarray(gt[:n], np.float64)
    # A nearly straight path leaves the roll about it free: rotation from the orientations (chordal mean).
    M = sum(g[:3, :3] @ e[:3, :3].T for e, g in zip(est, gt))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    p, q = est[:, :3, 3] @ R.T, gt[:, :3, 3]
    pc, qc = p - p.mean(0), q - q.mean(0)
    s = float((pc * qc).sum() / (pc ** 2).sum())
    aligned = s * p + (q.mean(0) - s * p.mean(0))
    terr = np.linalg.norm(aligned - q, axis=1)
    rerr = [float(np.degrees(np.arccos(np.clip((np.trace((R @ e[:3, :3]).T @ g[:3, :3]) - 1) / 2, -1, 1))))
            for e, g in zip(est, gt)]
    path = float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))
    return {"frames": n, "translation_rms_m": float(np.sqrt((terr ** 2).mean())), "translation_max_m": float(terr.max()),
            "translation_rms_pct_of_path": float(100 * np.sqrt((terr ** 2).mean()) / max(path, 1e-9)),
            "rotation_mean_deg": float(np.mean(rerr)), "rotation_max_deg": float(np.max(rerr)), "path_m": path}


def load_frames(folder, frames):
    """The plate as the op hands it to a solver: display sRGB, 8 bits."""
    from nodebased.color import display_rgb
    from nodebased.media import read_media_raster
    out = []
    for i in range(1, frames + 1):
        linear = np.array(read_media_raster(str(Path(folder) / f"plate.{i:04d}.exr")).to_display()[..., :3], np.float32)
        out.append(np.rint(np.clip(display_rgb(linear, "sRGB"), 0, 1) * 255).astype(np.uint8))
    return out


def plate(kind, size, work, frames=shot.FRAMES):
    """(folder, camera keys per frame, true focal in pixels, survey or None): rendered once into `work`."""
    folder = Path(work) / f"plate-{kind}-{size[0]}x{size[1]}"
    if kind == "demo":
        if not (folder / f"plate.{frames:04d}.exr").exists():
            import make_plate
            make_plate.render_plate(folder, size=size, frames=frames)
        keys = {f: dict(shot.camera_at(f), roll=0.0) for f in range(1, frames + 1)}
        import make_standin_plate as standin
        return folder, keys, standin.focal_pixels(size[1]), None
    import make_standin_plate as standin
    if not (folder / "truth.json").exists():
        standin.render(folder, size=size, frames=frames)
    truth = json.loads((folder / "truth.json").read_text())
    return folder, {int(f): row for f, row in truth["camera_keys"].items()}, truth["focal_px"], truth["survey"]


def run(solver, kind="demo", size=(960, 540), focal_scale=1.0, runs=2, work=None, seed=0, loop_closure=False, log=print,
        config=None):
    from nodebased import camera_solve as cs
    provider = cs.SOLVERS[f"camera_solve_{solver}"]
    work = Path(work or tempfile.mkdtemp(prefix="camera-solve-bench-"))
    folder, keys, true_focal, _survey = plate(kind, size, work)
    frames = load_frames(folder, len(keys))
    truth = np.stack([cs.pose_from_keys(**keys[f]) for f in sorted(keys)])
    height, width = frames[0].shape[:2]
    given = true_focal * focal_scale
    intrinsics = (given, given, width / 2.0, height / 2.0)
    results = []
    for index in range(runs):
        started = time.perf_counter()
        extra = {"config": config} if config else {}
        solved = provider.run_solve(frames, intrinsics, loop_closure=loop_closure, seed=seed,
                                    workdir=work / f"run-{solver}-{kind}-{width}-{index}", **extra)
        wall = time.perf_counter() - started
        focal = float(solved["intrinsics"][1]) if "intrinsics" in solved else given
        row = {**align_and_measure(solved["c2w"], truth), "focal_px": focal,
               "focal_error_pct": 100.0 * (focal / true_focal - 1.0), "wall_s": wall,
               "cloud_points": int(len(solved["points"])), "info": solved["info"], "_c2w": solved["c2w"]}
        if "c2w_slam" in solved:   # MASt3R-SLAM's own poses, before the focal refinement
            row["before_refinement"] = align_and_measure(solved["c2w_slam"], truth)
        log(f"{solver} {kind} {width}x{height} run {index + 1}: translation RMS {100 * row['translation_rms_m']:.2f} cm, "
            f"rotation {row['rotation_mean_deg']:.3f} deg, focal {focal:.1f} px ({row['focal_error_pct']:+.2f}%), "
            f"{solved['info'].get('fps_after_warmup', 0):.1f} fps, {solved['info'].get('peak_vram_mb', 0):.0f} MB")
        results.append(row)
    repeat = None
    if len(results) > 1:
        a, b = results[0]["_c2w"], results[1]["_c2w"]
        scale = max(np.ptp(a[:, :3, 3], axis=0).max(), 1e-9)
        repeat = {"identical": bool(np.array_equal(a, b)), "pose_max_abs_diff": float(np.abs(a - b).max()),
                  "position_max_diff_pct_of_extent": float(100 * np.abs(a[:, :3, 3] - b[:, :3, 3]).max() / scale),
                  "focal_diff_px": abs(results[0]["focal_px"] - results[1]["focal_px"])}
    for row in results:
        row.pop("_c2w")
    return {"solver": provider.PROVIDER_NAME, "version": provider.description()["version"], "plate": kind,
            "size": [width, height], "true_focal_px": true_focal, "given_focal_px": None if provider is cs.mast3r else given,
            "seed": seed, "loop_closure": loop_closure, "runs": results, "repeat": repeat}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--solver", choices=("mast3r", "dpvo"), required=True)
    parser.add_argument("--plate", choices=("demo", "standin"), default="demo")
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--loop-closure", action="store_true")
    parser.add_argument("--config", help="MASt3R-SLAM config overrides as JSON, e.g. '{\"tracking\": {\"match_frac_thresh\": 0.5}}'")
    parser.add_argument("--work")
    parser.add_argument("--json")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    result = run(args.solver, args.plate, size, args.focal_scale, args.runs, args.work, loop_closure=args.loop_closure,
                 config=json.loads(args.config) if args.config else None)
    result["config"] = json.loads(args.config) if args.config else {}
    text = json.dumps(result, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
