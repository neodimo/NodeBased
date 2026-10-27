"""Local SVD-XT image-to-video provider; torch lives only in the model interpreter."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading

import numpy as np

PROVIDER_NAME = "svd_image_video"
MODEL_ENV = "NODEBASED_SVD_MODEL_DIR"
PYTHON_ENV = "NODEBASED_SVD_PYTHON"
RUN_INFO_ENV = "NODEBASED_SVD_RUN_INFO"
DEFAULT_PYTHON = str(Path.home() / ".openclaw/workspace/projects/nb-providers/sdxl/.venv/bin/python")
DEFAULT_SETTINGS = {"steps": 25, "fps": 7, "motion_bucket_id": 127, "noise_aug_strength": 0.02}


def resolve_settings(settings):
    result = dict(DEFAULT_SETTINGS)
    settings = settings or {}
    unknown = set(settings) - set(result)
    if unknown:
        raise ValueError(f"unknown keys in SVD sampler settings: {sorted(unknown)}")
    for key, value in settings.items():
        if key in ("steps", "fps", "motion_bucket_id"):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            low, high = {"steps": (1, 100), "fps": (1, 30), "motion_bucket_id": (0, 255)}[key]
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            low, high = 0.0, 1.0
            value = float(value)
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        result[key] = value
    return result


def _reference_image(controls, references):
    if references:
        if len(references) != 1:
            raise ValueError("SVD accepts exactly one reference image")
        path = Path(references[0])
        if not path.is_file():
            raise ValueError(f"reference image does not exist: {path}")
        import OpenImageIO as oiio
        source = oiio.ImageInput.open(str(path))
        if source is None:
            raise ValueError(f"cannot read reference image: {path}")
        try:
            spec = source.spec()
            if path.suffix.lower() == ".exr":
                pixels = np.asarray(source.read_image(format=oiio.FLOAT)).reshape(spec.height, spec.width, spec.nchannels)
                if pixels.shape[2] >= 3:
                    from .. import color
                    rgba = np.ones((spec.height, spec.width, 4), np.float32)
                    rgba[..., :3] = pixels[..., :3]
                    color.processor(color.WORKING, color.INPUT_SPACES["sRGB"]).applyRGBA(rgba)
                    pixels = np.rint(np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
            else:
                pixels = np.asarray(source.read_image(format=oiio.UINT8)).reshape(spec.height, spec.width, spec.nchannels)
        finally:
            source.close()
        if pixels.shape[2] < 3:
            raise ValueError("reference image needs RGB channels")
        return pixels[..., :3].copy()
    from .. import color
    rgba = np.asarray(controls[0].beauty.values, np.float32).copy()
    rgba[..., 3] = 1
    color.processor(color.WORKING, color.INPUT_SPACES["sRGB"]).applyRGBA(rgba)
    return np.rint(np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def run_frames(controls, *, reference_frames, seed, sampler_settings, workdir, progress=None, cancelled=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model_dir = os.environ.get(MODEL_ENV)
    if not model_dir or not Path(model_dir).is_dir():
        raise RuntimeError(f"set {MODEL_ENV} to the downloaded SVD-XT model directory")
    python = os.environ.get(PYTHON_ENV, DEFAULT_PYTHON)
    if not Path(python).is_file():
        raise RuntimeError(f"set {PYTHON_ENV} to a Python with torch and diffusers")
    height, width = controls[0].beauty.values.shape[:2]
    np.save(workdir / "reference.npy", _reference_image(controls, reference_frames))
    job = {"model_dir": str(Path(model_dir).resolve()), "reference": str(workdir / "reference.npy"),
           "result_dir": str(workdir), "frames": len(controls), "width": width, "height": height,
           "seed": seed, "settings": resolve_settings(sampler_settings), "info": str(workdir / "info.json")}
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
                if line.startswith("progress "):
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
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("Job cancelled")
        if code:
            raise RuntimeError(f"SVD model process failed (exit {code}): " + " | ".join(tail[-8:])[-1200:])
    info = json.loads(Path(job["info"]).read_text(encoding="utf-8"))
    if os.environ.get(RUN_INFO_ENV):
        Path(os.environ[RUN_INFO_ENV]).write_text(json.dumps(info, indent=2), encoding="utf-8")
    from .sdxl_controlnet import srgb_to_acescg
    pictures = [srgb_to_acescg(np.load(workdir / f"result_{i:04d}.npy")) for i in range(len(controls))]
    return pictures, info


def _model_main(job_path):
    import time
    import torch
    from PIL import Image
    from diffusers import StableVideoDiffusionPipeline
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("SVD requires CUDA")
    start = time.perf_counter()
    pipe = StableVideoDiffusionPipeline.from_pretrained(job["model_dir"], torch_dtype=torch.float16,
                                                         variant="fp16", local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    settings = job["settings"]
    generator = torch.Generator(device="cuda").manual_seed(job["seed"])
    image = Image.fromarray(np.load(job["reference"])).resize((1024, 576), Image.Resampling.LANCZOS)
    frames = pipe(image, generator=generator,
                  num_frames=job["frames"], num_inference_steps=settings["steps"],
                  fps=settings["fps"], motion_bucket_id=settings["motion_bucket_id"],
                  noise_aug_strength=settings["noise_aug_strength"], decode_chunk_size=2).frames[0]
    for index, frame in enumerate(frames):
        resized = frame.resize((job["width"], job["height"]), Image.Resampling.LANCZOS)
        np.save(Path(job["result_dir"]) / f"result_{index:04d}.npy", np.asarray(resized.convert("RGB"), np.uint8))
        print(f"progress {index + 1} {len(frames)}", flush=True)
    info = {"load_seconds": round(loaded - start, 2), "generate_seconds": round(time.perf_counter() - loaded, 2),
            "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 1)}
    Path(job["info"]).write_text(json.dumps(info), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        raise SystemExit("usage: svd_image_video.py run JOB.json")
    _model_main(sys.argv[2])
