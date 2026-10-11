"""SDXL with ControlNet-Union (depth and normals), a local provider that runs on the NVIDIA card.

Two halves in one file, so the provider description, the conversions and the model call stay together:

* The NodeBased side (imported by `nodebased.generative`): numpy only. It turns each frame's exported depth
  and normals into the 8-bit control images the adapter expects, starts the model as a subprocess with the
  provider environment's interpreter (`NODEBASED_SDXL_PYTHON`) and brings the pictures back as ACEScg.
  Importing this module needs neither torch nor diffusers.
* The model side (`python sdxl_controlnet.py run JOB.json`, run by the provider environment's interpreter):
  loads SDXL base, the fp16-fix VAE and ControlNet-Union (promax) from the local Hugging Face cache and
  never reaches the network. It imports nothing from NodeBased.

Frames are generated one by one, each from that frame's own depth and normals, with the same seed. There is
no temporal model: frame-to-frame flicker is expected and motion is not conditioned.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROVIDER_NAME = "sdxl_controlnet"
DEFAULT_PYTHON = "/var/home/omid/.openclaw/workspace/projects/nb-providers/sdxl/.venv/bin/python"
PYTHON_ENV = "NODEBASED_SDXL_PYTHON"
RUN_INFO_ENV = "NODEBASED_SDXL_RUN_INFO"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_MODEL = "xinsir/controlnet-union-sdxl-1.0"
WORKING_LONG_SIDE = 1024
# ControlNet-Union promax control types: 0 openpose, 1 depth, 2 soft edge/scribble, 3 canny/lineart,
# 4 normal, 5 segment, 6 tile, 7 repaint.
CONTROL_TYPE_DEPTH, CONTROL_TYPE_NORMAL = 1, 4
CONTROL_MODES = {"depth_normals": (CONTROL_TYPE_DEPTH, CONTROL_TYPE_NORMAL),
                 "depth": (CONTROL_TYPE_DEPTH,), "normals": (CONTROL_TYPE_NORMAL,)}
DEFAULT_SETTINGS = {"steps": 30, "guidance": 5.0, "controlnet_conditioning_scale": 0.8,
                    "control_mode": "depth_normals"}
SETTING_RANGES = {"steps": (1, 100), "guidance": (0.0, 20.0), "controlnet_conditioning_scale": (0.0, 2.0)}
DEFAULT_PROMPT = "a photograph"
NEGATIVE_PROMPT = "blurry, low quality, watermark, text, deformed"
MIN_VALID_NORMAL_LENGTH = 0.5


# ----------------------------------------------------------------------------- NodeBased side

def resolve_settings(sampler_settings):
    """`sampler_settings` with defaults filled in; an unknown key or an out-of-range value is an error."""
    settings = dict(DEFAULT_SETTINGS)
    unknown = sorted(set(sampler_settings or {}) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"sdxl_controlnet sampler_settings has unknown keys: {', '.join(unknown)} "
                         f"(allowed: {', '.join(DEFAULT_SETTINGS)})")
    settings.update(sampler_settings or {})
    for key, (low, high) in SETTING_RANGES.items():
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"sdxl_controlnet {key} must be a number from {low} to {high}")
    if not isinstance(settings["steps"], int):
        raise ValueError("sdxl_controlnet steps must be a whole number")
    settings["guidance"] = float(settings["guidance"])
    settings["controlnet_conditioning_scale"] = float(settings["controlnet_conditioning_scale"])
    if settings["control_mode"] not in CONTROL_MODES:
        raise ValueError(f"sdxl_controlnet control_mode must be one of {', '.join(CONTROL_MODES)}")
    return settings


def working_size(width, height):
    """The model's working size: the longer side is 1024, both sides a multiple of 8 (SDXL's latent grid)."""
    scale = WORKING_LONG_SIDE / max(width, height)
    return (max(8, int(round(width * scale / 8)) * 8), max(8, int(round(height * scale / 8)) * 8))


def depth_range(depth_planes):
    """Nearest and farthest valid depth over the whole sequence (finite and above zero), or (0, 1) if none."""
    near, far = np.inf, 0.0
    for plane in depth_planes:
        values = np.asarray(plane, dtype=np.float32)
        valid = values[np.isfinite(values) & (values > 0)]
        if valid.size:
            near, far = min(near, float(valid.min())), max(far, float(valid.max()))
    return (0.0, 1.0) if not np.isfinite(near) else (near, far)


def depth_control_image(depth, near, far):
    """Metric depth to the 8-bit three-channel image the depth adapter expects.

    Rule: inverse depth, normalised with the nearest and farthest depth of the whole sequence (so a surface
    keeps its grey from frame to frame), near is white and far is black; pixels with no depth (background,
    infinity, zero) are black. 255 * (1/d - 1/far) / (1/near - 1/far).
    """
    values = np.asarray(depth, dtype=np.float32).reshape(depth.shape[:2])
    valid = np.isfinite(values) & (values > 0)
    out = np.zeros(values.shape, np.float32)
    if far > near:
        inverse = 1.0 / np.where(valid, values, 1.0)
        out = np.where(valid, (inverse - 1.0 / far) / (1.0 / near - 1.0 / far), 0.0)
    byte = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(byte[..., None], 3, axis=2)


def normals_control_image(normals):
    """Unit normals (camera space, +X right, +Y up, +Z toward the viewer) to the usual 8-bit normal map.

    rgb = (n * 0.5 + 0.5) * 255. Pixels with no normal (zero or non-finite) get the flat map colour
    (0.5, 0.5, 1.0), the colour of a surface facing the camera.
    """
    values = np.asarray(normals, dtype=np.float32)[..., :3]
    finite = np.all(np.isfinite(values), axis=2)
    values = np.where(finite[..., None], values, 0.0)
    length = np.sqrt((values * values).sum(axis=2))
    valid = finite & (length >= MIN_VALID_NORMAL_LENGTH)
    unit = values / np.maximum(length, 1e-6)[..., None]
    mapped = np.where(valid[..., None], unit * 0.5 + 0.5, np.array([0.5, 0.5, 1.0], np.float32))
    return np.rint(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)


def srgb_to_acescg(srgb8):
    """8-bit display sRGB pictures to scene-linear ACEScg RGBA (alpha 1) through the working OCIO config."""
    from .. import color
    rgba = np.ones((*srgb8.shape[:2], 4), np.float32)
    rgba[..., :3] = srgb8.astype(np.float32) / 255.0
    color.processor(color.INPUT_SPACES["sRGB"], color.WORKING).applyRGBA(rgba)
    rgba[..., 3] = 1.0
    return rgba


def provider_python():
    return os.environ.get(PYTHON_ENV) or DEFAULT_PYTHON


def available():
    """Why the model cannot run here (an empty string if it can): no interpreter, no CUDA."""
    python = provider_python()
    if not Path(python).exists():
        return f"the provider interpreter {python} does not exist (set {PYTHON_ENV})"
    probe = subprocess.run([python, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
                           capture_output=True, text=True)
    if probe.returncode == 3:
        return "torch in the provider environment reports no CUDA device"
    if probe.returncode:
        return f"the provider interpreter cannot import torch: {probe.stderr.strip()[-300:]}"
    return ""


def run_frames(controls, *, text, seed, sampler_settings, workdir, progress=None, cancelled=None):
    """Generate one picture per frame; returns (list of ACEScg RGBA arrays, run info).

    `controls` are ControlFrames (depth and normals planes). Control images and results go through `workdir`
    as .npy files; the model process reads a small job file and prints `progress i n` lines.
    """
    settings = resolve_settings(sampler_settings)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    near, far = depth_range([frame.depth.values for frame in controls])
    height, width = controls[0].beauty.values.shape[:2]
    rows = []
    for frame in controls:
        base = workdir / f"frame_{frame.frame:04d}"
        np.save(f"{base}.depth.npy", depth_control_image(frame.depth.values, near, far))
        np.save(f"{base}.normals.npy", normals_control_image(frame.normals.values))
        rows.append({"frame": frame.frame, "depth": f"{base}.depth.npy", "normals": f"{base}.normals.npy",
                     "result": f"{base}.result.npy"})
    job = {"frames": rows, "width": width, "height": height, "prompt": text or DEFAULT_PROMPT,
           "seed": int(seed), "settings": settings, "info": str(workdir / "info.json")}
    job_path = workdir / "job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    # No address-space cap here: CUDA reserves terabytes of virtual address space (see workers.py, local_gpu).
    process = subprocess.Popen([provider_python(), "-I", str(Path(__file__).resolve()), "run", str(job_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    tail = []
    try:
        for line in process.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            if line.startswith("progress "):
                _, done, total = line.split()
                if progress:
                    progress((int(done) / int(total), f"Frame {rows[min(int(done), len(rows)) - 1]['frame']}: generated"))
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
    if code:
        raise RuntimeError(f"sdxl_controlnet model process failed (exit {code}): " + " | ".join(tail[-8:])[-1200:])
    info = json.loads(Path(job["info"]).read_text(encoding="utf-8"))
    if os.environ.get(RUN_INFO_ENV):  # the run's timings and peak VRAM, for a caller that wants them (never in the artifact)
        Path(os.environ[RUN_INFO_ENV]).write_text(json.dumps(info, indent=1), encoding="utf-8")
    pictures = [srgb_to_acescg(np.load(row["result"])) for row in rows]
    return pictures, info


# ----------------------------------------------------------------------------- model side

def _cached_snapshot(repo):
    """The folder of a model's cached snapshot (Hugging Face cache layout); never touches the network."""
    from huggingface_hub import constants
    folder = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    revision = (folder / "refs" / "main").read_text().strip()
    return folder / "snapshots" / revision


def _load_pipeline(torch):
    from diffusers import AutoencoderKL, ControlNetUnionModel, StableDiffusionXLControlNetUnionPipeline
    from safetensors.torch import load_file
    snapshot = _cached_snapshot(CONTROLNET_MODEL)
    config = json.loads((snapshot / "config_promax.json").read_text())
    config.pop("_class_name", None)
    controlnet = ControlNetUnionModel.from_config(config)
    controlnet.load_state_dict(load_file(str(snapshot / "diffusion_pytorch_model_promax.safetensors")))
    controlnet = controlnet.to(torch.float16)
    vae = AutoencoderKL.from_pretrained(str(_cached_snapshot(VAE_MODEL)), torch_dtype=torch.float16)
    # A local folder skips the Hub's completeness check, which lists files of the repository that the fp16
    # snapshot deliberately does not hold (the fp32 weights, the alternative vae folders).
    base = _cached_snapshot(BASE_MODEL)
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        base, controlnet=controlnet, vae=vae, torch_dtype=torch.float16, variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _model_main(job_path):
    # Deterministic algorithms need this before cuBLAS starts (the parent sets it too).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from PIL import Image
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import diffusers.utils.logging as diffusers_logging
    diffusers_logging.set_verbosity_error()  # moving an fp16 pipeline off the card warns once per component
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: sdxl_controlnet runs on the NVIDIA card only")
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    settings = job["settings"]
    modes = list(CONTROL_MODES[settings["control_mode"]])
    width, height = job["width"], job["height"]
    work_w, work_h = working_size(width, height)
    started = time.perf_counter()
    pipe = _load_pipeline(torch)
    offloaded = False
    try:
        pipe.to("cuda")
    except torch.cuda.OutOfMemoryError:
        pipe.to("cpu")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        offloaded = True
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    names = {"depth": "depth", "normals": "normals"}
    total = len(job["frames"])
    for index, row in enumerate(job["frames"]):
        images = []
        for mode in modes:
            kind = "depth" if mode == CONTROL_TYPE_DEPTH else "normals"
            array = np.load(row[names[kind]])
            images.append(Image.fromarray(array).resize((work_w, work_h), Image.BICUBIC))
        arguments = dict(
            prompt=job["prompt"], negative_prompt=NEGATIVE_PROMPT, control_image=images, control_mode=modes,
            controlnet_conditioning_scale=[settings["controlnet_conditioning_scale"]] * len(modes),
            num_inference_steps=settings["steps"], guidance_scale=settings["guidance"],
            width=work_w, height=work_h, output_type="pil")
        generator = torch.Generator("cuda").manual_seed(job["seed"])
        try:
            picture = pipe(generator=generator, **arguments).images[0]
        except torch.cuda.OutOfMemoryError:
            if offloaded:
                raise
            pipe.to("cpu")
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
            offloaded = True
            generator = torch.Generator("cuda").manual_seed(job["seed"])
            picture = pipe(generator=generator, **arguments).images[0]
        np.save(row["result"], np.asarray(picture.resize((width, height), Image.LANCZOS), dtype=np.uint8))
        print(f"progress {index + 1} {total}", flush=True)
    import diffusers
    info = {"load_seconds": round(loaded - started, 2), "generate_seconds": round(time.perf_counter() - loaded, 2),
            "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "cpu_offload": offloaded, "working_size": [work_w, work_h], "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "diffusers": diffusers.__version__, "frames": total}
    Path(job["info"]).write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _model_main(sys.argv[2])
    else:
        raise SystemExit("usage: sdxl_controlnet.py run JOB.json (started by NodeBased, not by hand)")
