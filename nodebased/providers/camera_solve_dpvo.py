"""DPVO / DPV-SLAM camera solve, a local provider that runs on the NVIDIA card (stage 7 of the end-to-end sprint).

Two halves in one file, like the SDXL provider:

* The NodeBased side (imported by `nodebased.camera_solve`): numpy only. It writes the plate's frames as .npy
  files, starts the solver as a subprocess with the provider environment's interpreter (`NODEBASED_SLAM_PYTHON`),
  reads `progress i n` lines, and brings the poses and the sparse cloud back. Importing it needs no torch.
* The solver side (`python camera_solve_dpvo.py run JOB.json`, run by the provider environment's interpreter):
  loads DPVO from `NODEBASED_DPVO_DIR`, seeds it, feeds the frames one by one and writes `result.npz` and
  `info.json`. It imports nothing from NodeBased.

Monocular: the poses are up to scale and in the solver's own frame. `nodebased.camera_solve` places them.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROVIDER_NAME = "camera_solve_dpvo"
SOLVER_DIR = Path(__file__).with_name("solvers")
SLAM_HOME = Path.home() / ".openclaw/workspace/projects/nb-providers/slam"
DEFAULT_PYTHON = str(SLAM_HOME / ".venv/bin/python")
DEFAULT_DPVO_DIR = str(SLAM_HOME / "DPVO")
PYTHON_ENV = "NODEBASED_SLAM_PYTHON"
DPVO_DIR_ENV = "NODEBASED_DPVO_DIR"
RUN_INFO_ENV = "NODEBASED_SLAM_RUN_INFO"
MIN_FRAMES = 12            # DPVO initialises on the first 8 frames and needs a few more to say anything
RESULT_FILE, INFO_FILE = "result.npz", "info.json"


# ----------------------------------------------------------------------------- NodeBased side

def description():
    """The provider description (`providers/solvers/camera_solve_dpvo.json`) as a dict."""
    data = json.loads((SOLVER_DIR / f"{PROVIDER_NAME}.json").read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("kind") != "camera_solve":
        raise ValueError("Unsupported camera solve description")
    return data


def provider_python():
    return os.environ.get(PYTHON_ENV) or DEFAULT_PYTHON


def available():
    """Why the solver cannot run here (an empty string if it can): no interpreter, no CUDA, no DPVO."""
    python = provider_python()
    if not Path(python).exists():
        return f"the camera solve interpreter {python} does not exist (set {PYTHON_ENV}; projects/nb-providers/slam/setup.sh builds it)"
    probe = subprocess.run([python, "-I", "-c",
                            "import sys, torch; sys.path.insert(0, sys.argv[1]); import cuda_corr, dpvo.dpvo; "
                            "sys.exit(0 if torch.cuda.is_available() else 3)", dpvo_dir()],
                           capture_output=True, text=True)
    if probe.returncode == 3:
        return "torch in the camera solve environment reports no CUDA device"
    if probe.returncode:
        return f"the camera solve environment cannot import DPVO: {probe.stderr.strip()[-300:]}"
    return ""


def dpvo_dir():
    return os.environ.get(DPVO_DIR_ENV) or DEFAULT_DPVO_DIR


def focal_pixels(focal_mm, vaperture_mm, height):
    """Vertical focal length in pixels of a film back `vaperture_mm` tall (square pixels), the model Camera3D uses."""
    return float(height) * float(focal_mm) / float(vaperture_mm)


def intrinsics_for(width, height, focal_mm, vaperture_mm):
    """(fx, fy, cx, cy) in pixels for a pinhole with the principal point at the image centre."""
    f = focal_pixels(focal_mm, vaperture_mm, height)
    return (f, f, width / 2.0, height / 2.0)


def check_frames(name, frames, limits, min_frames):
    """Refuse what a solver cannot take before any process starts; returns (height, width)."""
    if len(frames) < min_frames:
        raise ValueError(f"{name} needs at least {min_frames} frames (got {len(frames)})")
    shapes = {np.asarray(f).shape for f in frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3 or next(iter(shapes))[2] != 3:
        raise ValueError(f"{name} needs same-size RGB frames")
    height, width = next(iter(shapes))[:2]
    if width > limits["max_resolution"][0] or height > limits["max_resolution"][1] or len(frames) > limits["max_frames"]:
        raise ValueError(f"{name} limits are {limits['max_resolution']} and {limits['max_frames']} frames")
    return height, width


def run_job(name, script, python, frames, job, workdir, progress=None, cancelled=None, env=None):
    """The subprocess contract shared by the camera solvers: write the frames as .npy files and `job.json`, run
    `python -I script run job.json` under the exclusive GPU lock, relay `progress i n` lines, and read
    `result.npz` (every array) and `info.json` back. A failure raises with the process's last lines."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, frame in enumerate(frames):
        path = workdir / f"frame_{index:05d}.npy"
        np.save(path, np.ascontiguousarray(frame, dtype=np.uint8))
        rows.append(str(path))
    job = {"frames": rows, **job, "result": str(workdir / RESULT_FILE), "info": str(workdir / INFO_FILE)}
    job_path = workdir / "job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", **(env or {}))
    # No address-space cap: CUDA reserves terabytes of virtual address space (workers.py, local_gpu).
    from ..gpulock import gpu_job  # the solver side of this file runs in another interpreter, so import here
    with gpu_job(cancelled):
        process = subprocess.Popen([python, "-I", str(Path(script).resolve()), "run", str(job_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        tail = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-40:]
                if line.startswith("progress "):
                    _, done, total = line.split()
                    if progress:
                        progress((int(done) / int(total), f"Frame {int(done)}: tracked"))
                if cancelled is not None and cancelled.is_set():
                    process.kill()
                    raise RuntimeError("Job cancelled")
            code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if code:
            raise RuntimeError(f"{name} solver process failed (exit {code}): " + " | ".join(tail[-8:])[-1200:])
    with np.load(job["result"]) as result:
        solved = {key: np.array(result[key]) for key in result.files}
    solved["c2w"] = solved["c2w"].astype(np.float64)
    solved["points"] = solved["points"].astype(np.float64)
    solved["colors"] = solved["colors"].astype(np.uint8)
    if "track_frame" in solved:   # where the network found each point on the other frames
        solved["tracks"] = {"frame": solved.pop("track_frame").astype(np.int64), "point": solved.pop("track_point").astype(np.int64),
                            "xy": solved.pop("track_xy").astype(np.float64), "weight": solved.pop("track_weight").astype(np.float64)}
    solved["info"] = json.loads(Path(job["info"]).read_text(encoding="utf-8"))
    if os.environ.get(RUN_INFO_ENV):   # timings and peak VRAM for a caller that wants them (never in the artifact)
        Path(os.environ[RUN_INFO_ENV]).write_text(json.dumps(solved["info"], indent=1), encoding="utf-8")
    return solved


def run_solve(frames, intrinsics, *, loop_closure=False, seed=0, workdir, progress=None, cancelled=None):
    """Solve the camera of a plate; returns {"c2w": (N,4,4), "points": (M,3), "colors": (M,3) uint8, "tracks": dict, "info": dict}.

    `frames` are N display-sRGB uint8 (H, W, 3) arrays, `intrinsics` is (fx, fy, cx, cy) in pixels. The poses
    are camera-to-world in OpenCV axes (x right, y down, z forward), up to scale, one per frame in order.
    `tracks` holds the solver's own measurements, for `nodebased.lensrefine`: `frame` (index into the poses),
    `point` (index into the points), `xy` (pixel position on the plate, top-left origin, at the plate's size) and
    `weight` (0 to 1). Each point has the position it was born at (weight 1) and one row per frame the network
    followed it to.
    """
    check_frames(PROVIDER_NAME, frames, description()["limits"], MIN_FRAMES)
    job = {"intrinsics": [float(v) for v in intrinsics], "loop_closure": bool(loop_closure), "seed": int(seed),
           "dpvo_dir": dpvo_dir()}
    return run_job(PROVIDER_NAME, __file__, provider_python(), frames, job, workdir, progress, cancelled)


# ----------------------------------------------------------------------------- solver side

def _tracks(slam, torch):
    """What the network measured: every patch where it was born, and where the update operator put it on each
    frame it was followed to (active and retired edges both), in full-resolution pixels (continuous coordinates, a
    pixel's centre at half an integer, as `lensmodel` has them). A patch is a point
    of `points_`, so `point` indexes the returned points; a keyframe's time stamp is the frame number."""
    pg, res, count = slam.pg, float(slam.RES), slam.m
    centre = 0.5     # feature j sits on plate pixel 4j (two stride-2 convolutions), whose centre is at 4j + 0.5
    stamps = np.asarray(pg.tstamps_)
    ii, jj, kk = (torch.cat((a, b)).cpu().numpy() for a, b in ((pg.ii_inac, pg.ii), (pg.jj_inac, pg.jj), (pg.kk_inac, pg.kk)))
    target = torch.cat((pg.target_inac.float(), pg.target.float()), dim=1)[0].cpu().numpy()
    weight = torch.cat((pg.weight_inac.float(), pg.weight.float()), dim=1)[0].cpu().numpy().mean(axis=1)
    keep = (kk < count) & (stamps[ii] != stamps[jj]) & np.isfinite(target).all(axis=1)
    born = pg.patches_[:, :, :2, 1, 1].cpu().numpy()                      # (keyframe, patch, xy) in feature-map pixels
    patch = np.arange(count)
    born_xy = born[patch // pg.M, patch % pg.M] * res + centre
    return {"frame": np.concatenate((stamps[patch // pg.M], stamps[jj[keep]])).astype(np.int64),
            "point": np.concatenate((patch, kk[keep])).astype(np.int64),
            "xy": np.concatenate((born_xy, target[keep] * res + centre)).astype(np.float64),
            "weight": np.concatenate((np.ones(count), weight[keep])).astype(np.float64)}


def _solve(job):
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, job["dpvo_dir"])
    import random
    import torch
    from scipy.spatial.transform import Rotation
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg.merge_from_file(f"{job['dpvo_dir']}/config/default.yaml")
    if job["loop_closure"]:
        cfg.merge_from_list(["LOOP_CLOSURE", "True"])
    intrinsics = torch.tensor(job["intrinsics"], dtype=torch.float64).cuda()
    torch.cuda.reset_peak_memory_stats()
    slam, times, total = None, [], len(job["frames"])
    with torch.no_grad():
        for t, path in enumerate(job["frames"]):
            rgb = np.load(path)
            h, w = rgb.shape[:2]
            image = torch.from_numpy(np.ascontiguousarray(rgb[:h - h % 16, :w - w % 16, ::-1])).permute(2, 0, 1).cuda()  # BGR like cv2
            if slam is None:
                slam = DPVO(cfg, f"{job['dpvo_dir']}/dpvo.pth", ht=image.shape[1], wd=image.shape[2])
            torch.cuda.synchronize()
            started = time.perf_counter()
            slam(t, image, intrinsics)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            print(f"progress {t + 1} {total}", flush=True)
        started = time.perf_counter()
        poses, _stamps = slam.terminate()
        terminate = time.perf_counter() - started
        points = slam.pg.points_.cpu().numpy()[:slam.m]
        colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    tracks = _tracks(slam, torch)
    c2w = np.tile(np.eye(4), (len(poses), 1, 1))
    c2w[:, :3, :3] = Rotation.from_quat(poses[:, 3:7]).as_matrix()   # x y z qx qy qz qw, camera to world
    c2w[:, :3, 3] = poses[:, :3]
    warm = np.array(times[8:]) if len(times) > 12 else np.array(times)
    info = {"frames": total, "fps_after_warmup": float(1 / warm.mean()),
            "fps_total": float(total / (sum(times) + terminate)), "terminate_s": terminate,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "loop_closure": job["loop_closure"],
            "seed": seed, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    np.savez(job["result"], c2w=c2w, points=points, colors=np.clip(colors, 0, 255).astype(np.uint8),
             **{f"track_{name}": value for name, value in tracks.items()})
    Path(job["info"]).write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _solve(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    else:
        raise SystemExit("usage: camera_solve_dpvo.py run JOB.json")
