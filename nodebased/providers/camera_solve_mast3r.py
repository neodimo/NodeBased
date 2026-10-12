"""MASt3R-SLAM camera solve, the second local camera solver on the NVIDIA card (stage 9 of the end-to-end sprint).

DPV-SLAM (camera_solve_dpvo.py) takes the focal length it is handed. MASt3R-SLAM needs no camera model: it tracks
on MASt3R's per-pixel 3D point maps and rays, so this provider also returns the focal length, read off the
keyframes' point maps. Same two halves and the same subprocess contract as DPVO (`run_job` there):

* The NodeBased side: numpy only. `run_solve` writes the frames and `job.json`, runs the solver side with the
  MASt3R environment's interpreter (`NODEBASED_MAST3R_PYTHON`) under the exclusive GPU lock and brings back
  the poses, the cloud and `intrinsics` (fx, fy, cx, cy in plate pixels).
* The solver side (`python camera_solve_mast3r.py run JOB.json`): MASt3R-SLAM from `NODEBASED_MAST3R_DIR`, in
  one process (the upstream script runs its back end in a second process; inline it is repeatable), frames
  resized to 512 wide like upstream. Every frame is tracked against the latest keyframe; at the end each
  frame's pose is its keyframe's optimised pose times the relative pose found when it was tracked.

The focal length: each keyframe's canonical point map gives, for every confident pixel, the ray x/z, y/z that
pixel sees; the focal is the robust (Weiszfeld, L1) fit of `pixel - centre = f * ray` with the principal point
at the image centre and square pixels, and the solve's focal is the median over keyframes, scaled back from
the 512-wide frame to the plate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROVIDER_NAME = "camera_solve_mast3r"
SOLVER_DIR = Path(__file__).with_name("solvers")
SLAM_HOME = Path.home() / ".openclaw/workspace/projects/nb-providers/slam"
DEFAULT_PYTHON = str(SLAM_HOME / ".venv-mast3r/bin/python")
DEFAULT_DIR = str(SLAM_HOME / "MASt3R-SLAM")
PYTHON_ENV = "NODEBASED_MAST3R_PYTHON"
DIR_ENV = "NODEBASED_MAST3R_DIR"
MIN_FRAMES = 8             # MASt3R-SLAM starts on one frame; eight give the focal fit a second keyframe or two
INPUT_SIZE = 512           # MASt3R's long side; upstream resizes every frame to it and crops to a multiple of 16
RECOVERS_FOCAL = True
CONFIDENCE = 1.5           # MASt3R confidence (1 = none) a point must pass for the focal fit and the cloud
CLOUD_STRIDE = 4           # one cloud point per 4 x 4 pixels of a keyframe
MAX_CLOUD_POINTS = 60000


def description():
    """The provider description (`providers/solvers/camera_solve_mast3r.json`) as a dict."""
    data = json.loads((SOLVER_DIR / f"{PROVIDER_NAME}.json").read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("kind") != "camera_solve":
        raise ValueError("Unsupported camera solve description")
    return data


def provider_python():
    return os.environ.get(PYTHON_ENV) or DEFAULT_PYTHON


def mast3r_dir():
    return os.environ.get(DIR_ENV) or DEFAULT_DIR


def available():
    """Why the solver cannot run here (an empty string if it can): no interpreter, no CUDA, no MASt3R-SLAM."""
    python = provider_python()
    if not Path(python).exists():
        return f"the MASt3R-SLAM interpreter {python} does not exist (set {PYTHON_ENV}; projects/nb-providers/slam/setup_mast3r.sh builds it)"
    weights = Path(mast3r_dir()) / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    if not weights.is_file():
        return f"the MASt3R weights are missing ({weights.name})"
    import subprocess
    probe = subprocess.run([python, "-I", "-c",
                            "import sys, torch; sys.path.insert(0, sys.argv[1]); import lietorch, mast3r_slam_backends, "
                            "mast3r_slam.tracker; sys.exit(0 if torch.cuda.is_available() else 3)", mast3r_dir()],
                           capture_output=True, text=True, cwd=mast3r_dir())
    if probe.returncode == 3:
        return "torch in the MASt3R-SLAM environment reports no CUDA device"
    if probe.returncode:
        return f"the MASt3R-SLAM environment cannot import its solver: {probe.stderr.strip()[-300:]}"
    return ""


def working_size(width, height):
    """(width, height, scale) of the frame MASt3R sees: long side resized to 512 (PIL, rounded), then centre-cropped
    to a multiple of 16; `scale` is plate pixels per working pixel (the crop keeps the centre and the focal)."""
    long_side = max(width, height)
    resized = (round(width * INPUT_SIZE / long_side), round(height * INPUT_SIZE / long_side))
    cx, cy = resized[0] // 2, resized[1] // 2
    cropped = ((2 * cx) // 16 * 16, (2 * cy) // 16 * 16)
    if cropped[0] == cropped[1] and width == height:
        cropped = (cropped[0], int(3 * cropped[0] / 4))
    return cropped[0], cropped[1], long_side / INPUT_SIZE


def focal_from_pointmap(points, confidence, width, height, threshold=CONFIDENCE, iterations=20):
    """Focal length in pixels of the pinhole that best explains a camera-frame point map (H*W, 3) on an image
    `width` by `height`: the L1 (Weiszfeld) fit of `pixel - centre = f * (x/z, y/z)` over confident pixels in
    front of the camera, principal point at the image centre, square pixels. Returns nan with too few pixels."""
    points = np.asarray(points, np.float64).reshape(height, width, 3)
    confidence = np.asarray(confidence, np.float64).reshape(height, width)
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    offset = np.stack([u - (width - 1) / 2.0, v - (height - 1) / 2.0], -1)
    z = points[..., 2]
    valid = (z > 1e-6) & (confidence > threshold) & np.isfinite(points).all(-1)
    if valid.sum() < 64:
        return float("nan")
    ray = points[..., :2][valid] / z[valid][:, None]
    offset = offset[valid]
    focal = float(np.median(np.linalg.norm(offset, axis=1) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)))
    for _ in range(iterations):
        weight = 1.0 / np.maximum(np.linalg.norm(focal * ray - offset, axis=1), 1e-6)
        focal = float((weight * (ray * offset).sum(1)).sum() / (weight * (ray * ray).sum(1)).sum())
    return focal


def run_solve(frames, intrinsics=None, *, loop_closure=False, seed=0, workdir, progress=None, cancelled=None):
    """Solve the camera and the lens of a plate; returns {"c2w", "points", "colors", "intrinsics", "info"}.

    `frames` are N display-sRGB uint8 (H, W, 3) arrays. `intrinsics` is ignored (MASt3R-SLAM is uncalibrated);
    the returned `intrinsics` is (fx, fy, cx, cy) in plate pixels, the focal as recovered, the principal point
    at the plate's centre. `loop_closure` turns the retrieval database (loop edges between keyframes) on.
    """
    from . import camera_solve_dpvo as dpvo   # the solver side of this file runs in another interpreter
    height, width = dpvo.check_frames(PROVIDER_NAME, frames, description()["limits"], MIN_FRAMES)
    job = {"loop_closure": bool(loop_closure), "seed": int(seed), "mast3r_dir": mast3r_dir()}
    solved = dpvo.run_job(PROVIDER_NAME, __file__, provider_python(), frames, job, workdir, progress, cancelled,
                          env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    focal = float(solved["info"]["focal_px"])
    if not np.isfinite(focal) or focal <= 0:
        raise RuntimeError(f"{PROVIDER_NAME} recovered no focal length (too few confident keyframe pixels)")
    solved["intrinsics"] = (focal, focal, width / 2.0, height / 2.0)
    return solved


# ----------------------------------------------------------------------------- solver side

class _LocalManager:
    """Stands in for multiprocessing.Manager: one process, so plain locks, values and lists do."""

    def RLock(self):
        import threading
        return threading.RLock()

    def Value(self, _typecode, value):
        import types
        return types.SimpleNamespace(value=value)

    def list(self):
        return []


def _solve(job):
    import warnings
    warnings.filterwarnings("ignore")
    os.chdir(job["mast3r_dir"])            # upstream loads config/ and checkpoints/ relative to its folder
    sys.path.insert(0, job["mast3r_dir"])
    import random
    import torch
    import lietorch
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    from mast3r_slam.config import config, load_config
    load_config("config/base.yaml")
    config["single_thread"], config["use_calib"] = True, False
    from mast3r_slam.frame import SharedKeyframes, create_frame
    from mast3r_slam.global_opt import FactorGraph
    from mast3r_slam.lietorch_utils import as_SE3
    from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono, resize_img
    from mast3r_slam.tracker import FrameTracker

    device = "cuda:0"
    total = len(job["frames"])
    first = np.load(job["frames"][0])
    plate_h, plate_w = first.shape[:2]
    h, w = (int(v) for v in resize_img(first.astype(np.float32) / 255.0, INPUT_SIZE)["true_shape"][0])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_mast3r(device=device)
    keyframes = SharedKeyframes(_LocalManager(), h, w, buffer=total + 2)
    tracker = FrameTracker(model, keyframes, device)
    graph = FactorGraph(model, keyframes, None, device)
    retrieval = load_retriever(model) if job["loop_closure"] else None
    load_s = time.perf_counter() - started

    def optimise(idx):
        """The upstream back end for one new keyframe, inline: edges to the previous keyframe (and retrieved ones)."""
        edges = [idx - 1] if idx > 0 else []
        if retrieval is not None:
            edges += retrieval.update(keyframes[idx], add_after_query=True, k=config["retrieval"]["k"],
                                      min_thresh=config["retrieval"]["min_thresh"])
        edges = sorted(set(edges) - {idx})
        if edges:
            graph.add_factors(edges, [idx] * len(edges), config["local_opt"]["min_match_frac"])
        graph.solve_GN_rays()

    anchors, times, lost, previous = [], [], 0, None   # anchors: (keyframe index, relative Sim3 data) per frame
    for t, path in enumerate(job["frames"]):
        rgb = np.load(path).astype(np.float32) / 255.0
        torch.cuda.synchronize()
        tick = time.perf_counter()
        pose = lietorch.Sim3.Identity(1, device=device) if previous is None else previous.T_WC
        frame = create_frame(t, rgb, pose, img_size=INPUT_SIZE, device=device)
        identity = lietorch.Sim3.Identity(1, device=device).data.cpu()
        if previous is None:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            keyframes.append(frame)
            optimise(len(keyframes) - 1)
            anchors.append((len(keyframes) - 1, identity))
        else:
            new_keyframe, _info, lost_track = tracker.track(frame)
            if lost_track:
                # Tracking failed against the last keyframe: hold the previous frame's pose (no relocalisation
                # without the retrieval database), and say so in the run info.
                lost += 1
                anchors.append(anchors[-1])
                frame.T_WC = previous.T_WC
            elif new_keyframe:
                keyframes.append(frame)
                optimise(len(keyframes) - 1)
                anchors.append((len(keyframes) - 1, identity))
            else:
                k = len(keyframes) - 1
                relative = lietorch.Sim3(keyframes.T_WC[k]).inv() * frame.T_WC
                anchors.append((k, relative.data.cpu()))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        previous = frame
        print(f"progress {t + 1} {total}", flush=True)

    n_kf = len(keyframes)
    c2w = np.tile(np.eye(4), (total, 1, 1))
    for t, (k, relative) in enumerate(anchors):
        pose = lietorch.Sim3(keyframes.T_WC[k]) * lietorch.Sim3(relative.to(device))
        x, y, z, qx, qy, qz, qw = as_SE3(pose).data.numpy().reshape(-1)
        qx, qy, qz, qw = (float(v) for v in (qx, qy, qz, qw))
        c2w[t, :3, :3] = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        c2w[t, :3, 3] = (x, y, z)

    focals, points, colors = [], [], []
    for k in range(n_kf):
        X = keyframes.X[k].double().cpu().numpy()
        C = (keyframes.C[k] / keyframes.N[k].clamp(min=1)).double().cpu().numpy().reshape(-1)
        focals.append(focal_from_pointmap(X, C, w, h))
        world = lietorch.Sim3(keyframes.T_WC[k]).act(keyframes.X[k]).double().cpu().numpy().reshape(h, w, 3)
        rgb = (keyframes.uimg[k].double().cpu().numpy() * 255.0).reshape(h, w, 3)
        keep = (C.reshape(h, w) > CONFIDENCE)[::CLOUD_STRIDE, ::CLOUD_STRIDE]
        points.append(world[::CLOUD_STRIDE, ::CLOUD_STRIDE][keep])
        colors.append(rgb[::CLOUD_STRIDE, ::CLOUD_STRIDE][keep])
    points, colors = np.concatenate(points), np.concatenate(colors)
    if len(points) > MAX_CLOUD_POINTS:
        step = int(np.ceil(len(points) / MAX_CLOUD_POINTS))
        points, colors = points[::step], colors[::step]
    focal_work = float(np.nanmedian(focals))
    scale = max(plate_w, plate_h) / INPUT_SIZE
    warm = np.array(times[4:]) if total > 8 else np.array(times)
    info = {"frames": total, "keyframes": n_kf, "lost_frames": lost, "working_size": [w, h],
            "focal_px": focal_work * scale, "focal_px_working": focal_work,
            "focal_px_per_keyframe": [f * scale for f in focals],
            "fps_after_warmup": float(1 / warm.mean()), "fps_total": float(total / sum(times)), "load_s": load_s,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "loop_closure": job["loop_closure"],
            "seed": seed, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    np.savez(job["result"], c2w=c2w, points=points, colors=np.clip(np.rint(colors), 0, 255).astype(np.uint8))
    Path(job["info"]).write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _solve(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    else:
        raise SystemExit("usage: camera_solve_mast3r.py run JOB.json")
