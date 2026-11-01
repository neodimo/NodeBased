"""Local CameraCtrl-on-SVD-XT provider: an image-to-video model that follows the exported camera.

Torch, diffusers and the CameraCtrl code live only in the model interpreter. This module's repository side
turns the exported per-frame cameras into RealEstate10K pose rows and drives the isolated worker process;
`_model_main` (the worker side) runs inside the CameraCtrl virtual environment.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import threading

import numpy as np

PROVIDER_NAME = "cameractrl_svd"
MODEL_ENV = "NODEBASED_CAMERACTRL_MODEL_DIR"
PYTHON_ENV = "NODEBASED_CAMERACTRL_PYTHON"
SOURCE_ENV = "NODEBASED_CAMERACTRL_SRC"
VARIANT_ENV = "NODEBASED_CAMERACTRL_VARIANT"
RUN_INFO_ENV = "NODEBASED_CAMERACTRL_RUN_INFO"
GENERATION_SIZE = (576, 320)
# model directory layout: <dir>/<snapshot>/ is a Diffusers snapshot with the fp16 variants, <dir>/cameractrl/<ckpt> the adapter
VARIANTS = {"xt": {"snapshot": "svd-xt", "ckpt": "CameraCtrl_svdxt.ckpt", "frames": 25,
                   "config": "configs/train_cameractrl/svdxt_320_576_cameractrl.yaml"},
            "svd": {"snapshot": "svd", "ckpt": "CameraCtrl_svd.ckpt", "frames": 14,
                    "config": "configs/train_cameractrl/svd_320_576_cameractrl.yaml"}}
DEFAULT_PYTHON = str(Path.home() / ".openclaw/workspace/scratch/nb-lanes/run/big-L4/venv-cameractrl/bin/python")
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def resolve_settings(settings):
    """The same four sampler settings and bounds as svd_image_video."""
    from .svd_image_video import resolve_settings as resolve
    return resolve(settings)


def pose_rows(cameras, width, height):
    """RealEstate10K rows `[t, fx, fy, cx, cy, 0, 0, w2c (3 x 4, row-major)]`, one per camera.

    `width` and `height` are the size the video is generated at, which fixes the intrinsics' aspect.

    Conventions. NodeBased is right-handed, +Y up, and the camera looks down -Z (scene3d.py header).
    CameraCtrl and RealEstate10K use OpenCV: +X right, +Y down, the camera looks down +Z. The camera's
    world matrix is built from `scene3d._view_basis` (rows right, up, backward) and the eye position,
    `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`, and the row holds `w2c_cv = inverse(c2w_cv)`.
    The poses are absolute: CameraCtrl makes them relative to the first frame itself
    (`get_relative_pose(..., zero_first_frame_scale=True)`). Translation is in scene units; the model was
    trained on RealEstate10K, whose scale is arbitrary (structure from motion), so a unit of distance in
    a NodeBased scene is not a distance the model has learned.

    Intrinsics are normalised by image size. `fov` is the vertical field of view, so
    `fy = 0.5 / tan(radians(fov) / 2)`, `fx = fy * height / width` (square pixels) and `cx = cy = 0.5`.
    Pass the generation size as both the original pose size and the sample size to CameraCtrl: its aspect
    rescale is then a no-op.
    """
    from .. import scene3d
    rows = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, scene3d.Camera):
            raise ValueError("camera_pose needs a Camera on every frame of the SceneState")
        eye, view = scene3d._view_basis(camera)
        c2w_gl = np.eye(4)
        c2w_gl[:3, :3] = np.asarray(view, np.float64).T
        c2w_gl[:3, 3] = np.asarray(eye, np.float64)
        w2c = np.linalg.inv(c2w_gl @ GL_TO_CV)
        fy = 0.5 / math.tan(math.radians(camera.fov) / 2)
        fx = fy * height / width
        rows.append([float(index), fx, fy, 0.5, 0.5, 0.0, 0.0] + [float(v) for v in w2c[:3, :].reshape(-1)])
    return rows


def cameras_for(controls, scene_state_path):
    """The exported camera of each ControlBundle frame, in bundle order."""
    from ..scene_state import read_scene_state
    by_frame = {int(row["frame"]): row.get("camera") for row in read_scene_state(scene_state_path)["frames"]}
    missing = [c.frame for c in controls if c.frame not in by_frame]
    if missing:
        raise ValueError(f"the SceneState has no camera for frames {missing}")
    return [by_frame[c.frame] for c in controls]


def variant():
    name = os.environ.get(VARIANT_ENV, "xt")
    if name not in VARIANTS:
        raise ValueError(f"{VARIANT_ENV} must be one of {sorted(VARIANTS)}")
    return name


def run_frames(controls, *, cameras, reference_frames, seed, sampler_settings, workdir, progress=None, cancelled=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    name = variant()
    if len(controls) > VARIANTS[name]["frames"]:
        raise ValueError(f"the {name} CameraCtrl checkpoint generates at most {VARIANTS[name]['frames']} frames")
    model_dir = os.environ.get(MODEL_ENV)
    if not model_dir or not Path(model_dir).is_dir():
        raise RuntimeError(f"set {MODEL_ENV} to the directory holding the SVD snapshot and cameractrl/ checkpoint")
    source = os.environ.get(SOURCE_ENV)
    if not source or not Path(source, "cameractrl").is_dir():
        raise RuntimeError(f"set {SOURCE_ENV} to the CameraCtrl checkout (svd branch)")
    python = os.environ.get(PYTHON_ENV, DEFAULT_PYTHON)
    if not Path(python).is_file():
        raise RuntimeError(f"set {PYTHON_ENV} to the CameraCtrl virtual environment's Python")
    height, width = controls[0].beauty.values.shape[:2]
    from .svd_image_video import _reference_image
    np.save(workdir / "reference.npy", _reference_image(controls, reference_frames))
    gen_width, gen_height = GENERATION_SIZE
    spec = VARIANTS[name]
    job = {"model_dir": str((Path(model_dir) / spec["snapshot"]).resolve()),
           "checkpoint": str((Path(model_dir) / "cameractrl" / spec["ckpt"]).resolve()),
           "source": str(Path(source).resolve()), "config": spec["config"],
           "reference": str(workdir / "reference.npy"), "result_dir": str(workdir),
           "poses": pose_rows(cameras, gen_width, gen_height), "frames": len(controls),
           "width": width, "height": height, "generation_size": [gen_width, gen_height],
           "seed": seed, "settings": resolve_settings(sampler_settings), "decode_chunk_size": 2,
           "info": str(workdir / "info.json")}
    job_path = workdir / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    from ..gpulock import gpu_job
    with gpu_job(cancelled):
        process = subprocess.Popen([python, "-I", str(Path(__file__).resolve()), "run", str(job_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        if cancelled is not None:
            def stop_on_cancel():
                cancelled.wait()
                if process.poll() is None:
                    process.kill()
            threading.Thread(target=stop_on_cancel, daemon=True).start()
        tail = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-40:]
                if line.startswith("step "):
                    _, done, total = line.split()
                    if progress:
                        progress((0.9 * int(done) / int(total), f"Video denoising step {done} of {total}"))
                elif line.startswith("progress "):
                    _, done, total = line.split()
                    if progress:
                        progress((int(done) / int(total), f"Video frame {done}: generated"))
                else:
                    print(line, flush=True)
                if cancelled is not None and cancelled.is_set():
                    process.kill()
                    raise RuntimeError("Job cancelled")
            code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("Job cancelled")
        if code:
            raise RuntimeError(f"CameraCtrl model process failed (exit {code}): " + " | ".join(tail[-8:])[-1200:])
    info = json.loads(Path(job["info"]).read_text(encoding="utf-8"))
    info["checkpoint"] = spec["ckpt"]
    if os.environ.get(RUN_INFO_ENV):
        Path(os.environ[RUN_INFO_ENV]).write_text(json.dumps(info, indent=2), encoding="utf-8")
    from .sdxl_controlnet import srgb_to_acescg
    pictures = [srgb_to_acescg(np.load(workdir / f"result_{i:04d}.npy")) for i in range(len(controls))]
    return pictures, info


def _model_main(job_path):
    """Worker side, run by the CameraCtrl environment's Python: follows CameraCtrl's inference.py."""
    import sys
    import time
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sys.path.insert(0, job["source"])  # `-I` leaves the checkout off the path
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from cameractrl.models.pose_adaptor import CameraPoseEncoder
    from cameractrl.models.unet import UNetSpatioTemporalConditionModelPoseCond
    from cameractrl.pipelines.pipeline_animation import StableVideoDiffusionPipelinePoseCond
    from inference import Camera, get_relative_pose, ray_condition
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CameraCtrl requires CUDA")
    start = time.perf_counter()
    config = OmegaConf.load(str(Path(job["source"]) / job["config"]))
    root, half = job["model_dir"], torch.float16
    scheduler = EulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    extractor = CLIPImageProcessor.from_pretrained(root, subfolder="feature_extractor")
    encoder = CLIPVisionModelWithProjection.from_pretrained(root, subfolder="image_encoder", torch_dtype=half, variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(root, subfolder="vae", torch_dtype=half, variant="fp16")
    unet = UNetSpatioTemporalConditionModelPoseCond.from_pretrained(
        root, subfolder=config["unet_subfolder"], down_block_types=config["down_block_types"],
        up_block_types=config["up_block_types"], torch_dtype=half, variant="fp16")
    pose_encoder = CameraPoseEncoder(**config["pose_encoder_kwargs"])
    unet.set_pose_cond_attn_processor(enable_xformers=False, **config["attention_processor_kwargs"])
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
    missing, unexpected = pose_encoder.load_state_dict(checkpoint["pose_encoder_state_dict"])
    assert not missing and not unexpected
    _, unexpected = unet.load_state_dict(checkpoint["attention_processor_state_dict"], strict=False)
    assert not unexpected
    unet.to(half)
    pose_encoder.to("cuda", half)
    pipe = StableVideoDiffusionPipelinePoseCond(vae=vae, image_encoder=encoder, unet=unet, scheduler=scheduler,
                                                feature_extractor=extractor, pose_encoder=pose_encoder)
    pipe.enable_model_cpu_offload()  # the pose encoder is outside the offload sequence and stays on the card
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    width, height = job["generation_size"]
    cameras = [Camera(row) for row in job["poses"]]
    intrinsic = np.asarray([[c.fx * width, c.fy * height, c.cx * width, c.cy * height] for c in cameras], np.float32)
    c2ws = torch.as_tensor(get_relative_pose(cameras, zero_first_frame_scale=True))[None]
    plucker = ray_condition(torch.as_tensor(intrinsic)[None], c2ws, height, width, device="cpu")
    plucker = plucker.permute(0, 1, 4, 2, 3).contiguous().to("cuda", half)
    settings = job["settings"]
    steps = settings["steps"]
    image = Image.fromarray(np.load(job["reference"])).resize((width, height), Image.Resampling.LANCZOS)

    def on_step(_pipe, index, _timestep, kwargs):
        print(f"step {index + 1} {steps}", flush=True)
        return kwargs

    generator = torch.Generator(device="cuda").manual_seed(job["seed"])
    with torch.no_grad():
        frames = pipe(image, pose_embedding=plucker, height=height, width=width, num_frames=job["frames"],
                      num_inference_steps=steps, fps=settings["fps"], motion_bucket_id=settings["motion_bucket_id"],
                      noise_aug_strength=settings["noise_aug_strength"], min_guidance_scale=1.0, max_guidance_scale=3.0,
                      do_image_process=True, decode_chunk_size=job["decode_chunk_size"], generator=generator,
                      callback_on_step_end=on_step, output_type="pil").frames[0]
    for index, frame in enumerate(frames):
        resized = frame.resize((job["width"], job["height"]), Image.Resampling.LANCZOS)
        np.save(Path(job["result_dir"]) / f"result_{index:04d}.npy", np.asarray(resized.convert("RGB"), np.uint8))
        print(f"progress {index + 1} {len(frames)}", flush=True)
    info = {"load_seconds": round(loaded - start, 2), "generate_seconds": round(time.perf_counter() - loaded, 2),
            "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 1),
            "frames": len(frames), "generation_size": [width, height]}
    Path(job["info"]).write_text(json.dumps(info), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        raise SystemExit("usage: cameractrl_svd.py run JOB.json")
    _model_main(sys.argv[2])
