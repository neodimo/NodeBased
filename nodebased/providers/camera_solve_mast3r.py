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
  frame's pose is its keyframe's optimised pose times its pose relative to that keyframe.

The focal length: each keyframe's canonical point map gives, for every confident pixel, the ray x/z, y/z that
pixel sees; the focal is the robust (Weiszfeld, L1) fit of `pixel - centre = f * ray` with the principal point
at the image centre and square pixels, and the solve's focal is the median over keyframes, scaled back from
the 512-wide frame to the plate.

The poses: MASt3R-SLAM tracks a frame by aligning point maps (rays and distances), which leaves the camera off
by several centimetres on the demo plate. With the focal above held, every tracked frame's pose is refined by a
bundle adjustment on the reprojection of the tracker's own matches (`bundle_adjust`, numpy). Solving the focal
in that bundle too (self-calibration) is recorded as a diagnostic only: on a short dolly it swings by tens of
percent with whole-pixel match noise (the sprint doc, stage 9).
"""
from __future__ import annotations

import json
import math
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
MIN_REFINE_OBSERVATIONS = 200
MATCH_STRIDE = 8           # the pose refinement uses each keyframe's matches on an 8-pixel grid


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


def _sim3(data):
    """(translation, rotation matrix, scale) of lietorch Sim3 data [tx ty tz qx qy qz qw s]."""
    t, (qx, qy, qz, qw), scale = np.asarray(data[:3], np.float64), (float(v) for v in data[3:7]), float(data[7])
    rotation = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                         [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                         [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    return t, rotation, scale


def _rotation(rotvec):
    """Rotation matrices (N, 3, 3) of rotation vectors (N, 3) (Rodrigues)."""
    rotvec = np.asarray(rotvec, np.float64).reshape(-1, 3)
    angle = np.linalg.norm(rotvec, axis=1)[:, None, None]
    k = rotvec / np.maximum(angle[:, :, 0], 1e-12)
    K = np.zeros((len(k), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _project(segments, focal, width, height, log_rho, frame_params):
    """Residuals (n, 2) of every match and, per row, its segment's point index and its frame number."""
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    out, points, frames, row_frame = [], [], 0, []
    for seg, rho in zip(segments, log_rho):
        rays = np.column_stack([(seg["pixels"] - centre) / focal, np.ones(len(rho))])
        xyz = rays / np.exp(rho)[:, None]
        for frame in seg["frames"]:
            params = frame_params[frames]
            R = frame["R"] @ _rotation(params[:3])[0]
            moved = xyz[frame["index"]] @ R.T + frame["t"] + params[3:]
            z = np.maximum(moved[:, 2:3], 1e-6)
            out.append(focal * moved[:, :2] / z + centre - frame["observed"])
            row_frame.append(np.full(len(frame["index"]), frames))
            frames += 1
    return np.concatenate(out), np.concatenate(row_frame)


def bundle_adjust(segments, focal, width, height, free_focal=False, passes=2, outlier_pixels=3.0, iterations=40,
                  huber=1.0):
    """Bundle adjustment of the tracked frames on MASt3R's pixel matches, the focal held or (`free_focal`) solved.

    Each segment is one keyframe with the frames tracked against it: `pixels` (P, 2) keyframe pixel positions,
    `inv_depth` (P,) their inverse depths in the keyframe camera (from its point map), and `frames`, each with
    `R`, `t` (keyframe camera to frame camera, X_f = R X_k + t, the solve's pose), `index` (M,) the keyframe
    pixels it matched and `observed` (M, 2) where. Unknowns: every inverse depth, every frame's pose and, with
    `free_focal`, the one focal length (square pixels, principal point at the centre, pixel centres at integers
    like `focal_from_pointmap`). Levenberg-Marquardt on the reprojection error of every match (Huber, `huber` pixels), the
    depths eliminated with the Schur complement, derivatives by finite differences (each frame touches only its
    rows and each depth only its point's). Matches off by more than `outlier_pixels` after a pass are dropped for
    the next. Each segment keeps its scale (mean log inverse depth; the translations follow). Numpy only.
    Returns {"focal", "frames": [[(R, t), ...] per segment], "rms_pixels", "observations"}.
    """
    segments = [dict(seg, frames=[dict(f) for f in seg["frames"]]) for seg in segments]
    n_frames = sum(len(seg["frames"]) for seg in segments)
    log_rho = [np.log(np.asarray(seg["inv_depth"], np.float64)) for seg in segments]
    gauge = [rho.mean() for rho in log_rho]
    params = np.zeros((n_frames, 6))
    log_f = math.log(focal)
    offsets = np.cumsum([0] + [len(rho) for rho in log_rho])
    n_points, n_cam = offsets[-1], 1 + 6 * n_frames

    def point_rows():
        return np.concatenate([offsets[s] + frame["index"] for s, seg in enumerate(segments) for frame in seg["frames"]])

    def evaluate(log_f, log_rho, params):
        r, frame_of_row = _project(segments, math.exp(log_f), width, height, log_rho, params)
        return r, frame_of_row

    def cost(r):
        norm = np.linalg.norm(r, axis=1)
        return float(np.where(norm <= huber, 0.5 * norm ** 2, huber * (norm - 0.5 * huber)).sum())

    for _pass in range(passes):
        if n_frames == 0 or sum(len(f["index"]) for seg in segments for f in seg["frames"]) < 8:
            break
        lam = 1e-3
        r, frame_of_row = evaluate(log_f, log_rho, params)
        current = cost(r)
        pid = point_rows()
        for _ in range(iterations):
            eps = 1e-6
            # Jacobian columns: focal (all rows), each frame's 6 (its rows), each depth (its rows)
            jf = ((evaluate(log_f + eps, log_rho, params)[0] - r) / eps if free_focal else np.zeros_like(r))
            jc = np.zeros((len(r), 6, 2))
            for k in range(6):
                bumped = params.copy()
                bumped[:, k] += eps
                jc[:, k] = (evaluate(log_f, log_rho, bumped)[0] - r) / eps
            jp = (evaluate(log_f, [rho + eps for rho in log_rho], params)[0] - r) / eps     # (n, 2)
            norm = np.linalg.norm(r, axis=1)
            w = np.where(norm <= huber, 1.0, huber / np.maximum(norm, 1e-12))[:, None]       # IRLS weights
            # camera block: index 0 = focal, 1 + 6 j .. = frame j
            J = np.concatenate([jf[:, None, :], jc], axis=1)                              # (n, 7, 2)
            Hcc = np.zeros((n_cam, n_cam))
            gc = np.zeros(n_cam)
            Hcp = np.zeros((n_cam, n_points))
            Hpp = np.bincount(pid, (w * jp * jp).sum(1), n_points) + 1e-12
            gp = np.bincount(pid, (w * jp * r).sum(1), n_points)
            for j in range(n_frames):
                rows = frame_of_row == j
                cols = np.r_[0, 1 + 6 * j:7 + 6 * j]
                Jj, wj, rj = J[rows], w[rows], r[rows]
                Hcc[np.ix_(cols, cols)] += np.einsum("nac,nbc->ab", Jj * wj[:, None], Jj)
                gc[cols] += np.einsum("nac,nc->a", Jj * wj[:, None], rj)
                block = np.einsum("nac,nc->na", Jj * wj[:, None], jp[rows])                 # (n_j, 7)
                for a, col in enumerate(cols):
                    Hcp[col] += np.bincount(pid[rows], block[:, a], n_points)
            improved = False
            for _try in range(8):
                damped_pp = Hpp * (1 + lam)
                S = Hcc + lam * np.diag(np.diag(Hcc)) - (Hcp / damped_pp) @ Hcp.T
                b = -gc + (Hcp / damped_pp) @ gp
                try:
                    dc = np.linalg.solve(S + 1e-9 * np.eye(n_cam), b)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                dp = (-gp - Hcp.T @ dc) / damped_pp
                new_f = log_f + dc[0]
                new_params = params + dc[1:].reshape(-1, 6)
                new_rho = [rho + dp[offsets[s]:offsets[s + 1]] for s, rho in enumerate(log_rho)]
                # hold each segment's scale: shift its log inverse depths back and scale its translations
                start = 0
                for s, seg in enumerate(segments):
                    shift = new_rho[s].mean() - gauge[s]
                    new_rho[s] = new_rho[s] - shift
                    for k, frame in enumerate(seg["frames"]):
                        t = (frame["t"] + new_params[start + k, 3:]) * math.exp(shift)
                        new_params[start + k, 3:] = t - frame["t"]
                    start += len(seg["frames"])
                trial, _ = evaluate(new_f, new_rho, new_params)
                if cost(trial) < current:
                    log_f, log_rho, params, r, current = new_f, new_rho, new_params, trial, cost(trial)
                    lam = max(lam / 3, 1e-7)
                    improved = True
                    break
                lam *= 10
            if not improved or np.abs(dc).max() < 1e-7:
                break
        # drop the matches that do not fit, then solve again from here
        bad = np.linalg.norm(r, axis=1) > outlier_pixels
        start = 0
        for seg in segments:
            for frame in seg["frames"]:
                n = len(frame["index"])
                keep = ~bad[start:start + n]
                frame["index"], frame["observed"] = frame["index"][keep], frame["observed"][keep]
                start += n
    r, _ = evaluate(log_f, log_rho, params) if n_frames else (np.zeros((0, 2)), None)
    poses, start = [], 0
    for seg in segments:
        poses.append([(frame["R"] @ _rotation(params[start + k, :3])[0], frame["t"] + params[start + k, 3:])
                      for k, frame in enumerate(seg["frames"])])
        start += len(seg["frames"])
    norms = np.linalg.norm(r, axis=1)
    return {"focal": math.exp(log_f) if free_focal else focal, "frames": poses, "observations": int(len(norms)),
            "rms_pixels": float(np.sqrt(np.mean(norms ** 2))) if len(norms) else float("nan")}


def run_solve(frames, intrinsics=None, *, loop_closure=False, seed=0, workdir, progress=None, cancelled=None,
              config=None):
    """Solve the camera and the lens of a plate; returns {"c2w", "points", "colors", "intrinsics", "info"}.

    `frames` are N display-sRGB uint8 (H, W, 3) arrays. `intrinsics` is ignored (MASt3R-SLAM is uncalibrated);
    the returned `intrinsics` is (fx, fy, cx, cy) in plate pixels, the focal as recovered, the principal point
    at the plate's centre. `loop_closure` turns the retrieval database (loop edges between keyframes) on.
    `config` overrides upstream's config/base.yaml ({"tracking": {"match_frac_thresh": 0.5}}; benchmarks only).
    """
    from . import camera_solve_dpvo as dpvo   # the solver side of this file runs in another interpreter
    height, width = dpvo.check_frames(PROVIDER_NAME, frames, description()["limits"], MIN_FRAMES)
    job = {"loop_closure": bool(loop_closure), "seed": int(seed), "mast3r_dir": mast3r_dir(), "config": config or {},
           "debug": bool(os.environ.get("NODEBASED_MAST3R_DEBUG"))}
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
    for section, values in job.get("config", {}).items():
        config[section].update(values)
    from mast3r_slam.frame import SharedKeyframes, create_frame
    from mast3r_slam.global_opt import FactorGraph
    from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono, resize_img
    from mast3r_slam.tracker import FrameTracker

    import mast3r_slam.tracker as tracker_module
    device = "cuda:0"
    total = len(job["frames"])
    first = np.load(job["frames"][0])
    plate_h, plate_w = first.shape[:2]
    h, w = (int(v) for v in resize_img(first.astype(np.float32) / 255.0, INPUT_SIZE)["true_shape"][0])
    grid = (np.arange(MATCH_STRIDE // 2, h, MATCH_STRIDE)[:, None] * w
            + np.arange(MATCH_STRIDE // 2, w, MATCH_STRIDE)[None, :]).ravel()
    grid_t = torch.from_numpy(grid).to(device)
    captured = {}
    upstream_match = tracker_module.mast3r_match_asymmetric

    def capturing_match(model, frame_i, frame_j, idx_i2j_init=None):
        """The tracker's matcher, also keeping where each keyframe grid pixel was found in the frame."""
        out = upstream_match(model, frame_i, frame_j, idx_i2j_init=idx_i2j_init)
        idx, valid, _Xii, _Cii, Qii, _Xji, _Cji, Qji = out
        quality = torch.sqrt(Qii[idx[0]] * Qji).reshape(-1)       # as the tracker weighs a match
        captured["match"] = (idx[0].reshape(-1)[grid_t].cpu().numpy(), valid[0].reshape(-1)[grid_t].cpu().numpy(),
                             quality[grid_t].cpu().numpy())
        return out
    tracker_module.mast3r_match_asymmetric = capturing_match

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

    identity = lietorch.Sim3.Identity(1, device=device).data.cpu().numpy().reshape(-1)
    # anchors: per frame, (keyframe index, Sim3 data of the frame relative to it); matches: per tracked frame
    anchors, matches, times, lost, previous, frame_focals = [], [], [], 0, None, []
    for t, path in enumerate(job["frames"]):
        rgb = np.load(path).astype(np.float32) / 255.0
        torch.cuda.synchronize()
        tick = time.perf_counter()
        pose = lietorch.Sim3.Identity(1, device=device) if previous is None else previous.T_WC
        frame = create_frame(t, rgb, pose, img_size=INPUT_SIZE, device=device)
        if previous is None:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            keyframes.append(frame)
            optimise(len(keyframes) - 1)
            anchors.append((len(keyframes) - 1, identity))
        else:
            new_keyframe, _info, lost_track = tracker.track(frame)
            k = len(keyframes) - 1
            if lost_track:
                # Tracking failed against the last keyframe: hold the previous frame's pose (no relocalisation
                # without the retrieval database), and say so in the run info.
                lost += 1
                anchors.append(anchors[-1])
                frame.T_WC = previous.T_WC
            else:
                relative = (lietorch.Sim3(keyframes.T_WC[k]).inv() * frame.T_WC).data.cpu().numpy().reshape(-1)
                matches.append({"frame": t, "keyframe": k, "relative": relative, "match": captured["match"]})
                if new_keyframe:
                    keyframes.append(frame)
                    optimise(len(keyframes) - 1)
                    anchors.append((len(keyframes) - 1, identity))
                else:
                    anchors.append((k, relative))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
        if frame.X_canon is not None:
            frame_focals.append(focal_from_pointmap(frame.X_canon.double().cpu().numpy(),
                                                    frame.C.double().cpu().numpy().reshape(-1), w, h))
        previous = frame
        print(f"progress {t + 1} {total}", flush=True)
    tracker_module.mast3r_match_asymmetric = upstream_match

    n_kf = len(keyframes)
    kf_sim3 = [keyframes.T_WC[k].cpu().numpy().reshape(-1).astype(np.float64) for k in range(n_kf)]

    def c2w_of(k, rotation, centre):
        """Camera-to-world of a frame whose pose in keyframe k's camera is (rotation, centre)."""
        t_k, R_k, s_k = _sim3(kf_sim3[k])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R_k @ rotation, s_k * R_k @ centre + t_k
        return pose

    c2w_slam = np.stack([c2w_of(k, _sim3(rel)[1], _sim3(rel)[0]) for k, rel in anchors])

    focals, points, colors, segments, segment_kf = [], [], [], [], []
    for k in range(n_kf):
        X = keyframes.X[k].double().cpu().numpy()
        C = (keyframes.C[k] / keyframes.N[k].clamp(min=1)).double().cpu().numpy().reshape(-1)
        focals.append(focal_from_pointmap(X, C, w, h))
        world = lietorch.Sim3(keyframes.T_WC[k]).act(keyframes.X[k]).double().cpu().numpy().reshape(h, w, 3)
        rgb = (keyframes.uimg[k].double().cpu().numpy() * 255.0).reshape(h, w, 3)
        keep = (C.reshape(h, w) > CONFIDENCE)[::CLOUD_STRIDE, ::CLOUD_STRIDE]
        points.append(world[::CLOUD_STRIDE, ::CLOUD_STRIDE][keep])
        colors.append(rgb[::CLOUD_STRIDE, ::CLOUD_STRIDE][keep])
        # the keyframe's matches for the pose refinement: confident grid pixels in front of the camera
        usable = (C[grid] > CONFIDENCE) & (X[grid, 2] > 1e-6)
        position = np.full(len(grid), -1)
        position[usable] = np.arange(usable.sum())
        frames = []
        for row in (m for m in matches if m["keyframe"] == k):
            idx, valid, quality = row["match"]
            ok = valid.reshape(-1).astype(bool) & (quality > config["tracking"]["Q_conf"]) & usable
            if ok.sum() < 8:
                continue
            centre, rotation, _scale = _sim3(row["relative"])     # frame in keyframe coordinates
            frames.append({"frame": row["frame"], "R": rotation.T, "t": -rotation.T @ centre, "index": position[ok],
                           "observed": np.column_stack([idx[ok] % w, idx[ok] // w]).astype(np.float64)})
        if frames:
            segments.append({"pixels": np.column_stack([grid % w, grid // w])[usable].astype(np.float64),
                             "inv_depth": 1.0 / X[grid, 2][usable], "frames": frames})
            segment_kf.append(k)
    points, colors = np.concatenate(points), np.concatenate(colors)
    if len(points) > MAX_CLOUD_POINTS:
        step = int(np.ceil(len(points) / MAX_CLOUD_POINTS))
        points, colors = points[::step], colors[::step]
    focal_pointmap = float(np.nanmedian(focals))
    scale = max(plate_w, plate_h) / INPUT_SIZE

    # The lens is the point maps' focal. With it held, every tracked frame's pose (and the keyframe depths) is
    # refined on the reprojection of MASt3R's matches; the bundle with the focal free is kept as a diagnostic
    # (on a short dolly it is far less certain than the point maps, see the sprint doc, stage 9).
    if job.get("debug"):        # the refinement's input, for a benchmark that checks the matches against truth
        import pickle
        Path(job["result"]).with_name("segments.pkl").write_bytes(pickle.dumps(
            {"segments": segments, "segment_keyframes": segment_kf,
             "keyframe_frames": [int(keyframes.dataset_idx[k]) for k in range(n_kf)],
             "focal_pointmap": focal_pointmap, "working_size": [w, h]}))
    tick = time.perf_counter()
    refined = bundle_adjust(segments, focal_pointmap, w, h) if segments else None
    refine_s = time.perf_counter() - tick
    free = bundle_adjust(segments, focal_pointmap, w, h, free_focal=True) if segments else None
    c2w = c2w_slam.copy()
    if refined is not None and refined["observations"] >= MIN_REFINE_OBSERVATIONS:
        for seg, k, poses in zip(segments, segment_kf, refined["frames"]):
            for frame, (R, t) in zip(seg["frames"], poses):
                if anchors[frame["frame"]][0] == k:       # a frame that became a keyframe keeps the keyframe pose
                    c2w[frame["frame"]] = c2w_of(k, R.T, -R.T @ t)
    warm = np.array(times[4:]) if total > 8 else np.array(times)
    info = {"frames": total, "keyframes": n_kf, "lost_frames": lost, "working_size": [w, h],
            "focal_px": focal_pointmap * scale, "focal_px_working": focal_pointmap,
            "focal_px_per_keyframe": [f * scale for f in focals],
            "focal_px_per_frame": [f * scale for f in frame_focals],
            "focal_px_bundle_free": None if free is None else free["focal"] * scale,
            "refine": None if refined is None else {"observations": refined["observations"],
                                                    "rms_pixels_working": refined["rms_pixels"], "seconds": refine_s},
            "fps_after_warmup": float(1 / warm.mean()), "fps_total": float(total / (sum(times) + refine_s)),
            "load_s": load_s, "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "loop_closure": job["loop_closure"], "seed": seed, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)}
    np.savez(job["result"], c2w=c2w, c2w_slam=c2w_slam, points=points,
             colors=np.clip(np.rint(colors), 0, 255).astype(np.uint8))
    Path(job["info"]).write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _solve(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    else:
        raise SystemExit("usage: camera_solve_mast3r.py run JOB.json")
