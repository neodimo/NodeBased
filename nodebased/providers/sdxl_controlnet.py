"""SDXL with ControlNet-Union (depth and normals), a local provider that runs on the NVIDIA card.

Two halves in one file, so the provider description, the conversions and the model call stay together:

* The NodeBased side (imported by `nodebased.generative`): numpy only. It turns each frame's exported depth
  and normals into the 8-bit control images the adapter expects, starts the model as a subprocess with the
  provider environment's interpreter (`NODEBASED_SDXL_PYTHON`) and brings the pictures back as ACEScg.
  Importing this module needs neither torch nor diffusers.
* The model side (`python sdxl_controlnet.py run JOB.json`, run by the provider environment's interpreter):
  loads SDXL base, the fp16-fix VAE and ControlNet-Union (promax) from the local Hugging Face cache and
  never reaches the network. It imports nothing from NodeBased.

Frames are generated in order with the same seed. The first is text-to-image from its own depth and normals.
Every later one is guided by a canvas carried from the one before (reprojection, version 1.2): the canvas (frame 1's
picture at first) is warped into this frame with the exported backward motion (`nodebased.temporal`, loaded by path
on the model side), the pixels the warp cannot supply (disocclusions, objects that appear, the edges, by the exported
depth and IDs) are filled from this frame's plain text-to-image picture, and the result is the next canvas and the
starting image of an image-to-image pass at the pinned `denoise_strength`. Image-to-image results never go back into
the canvas (1.1 chained them, and the picture broke up within a few frames). A strength of 1 turns the guidance off: every frame is made on its own, as in 1.0.
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
DEFAULT_PYTHON = str(Path.home() / ".openclaw/workspace/projects/nb-providers/sdxl/.venv/bin/python")
PYTHON_ENV = "NODEBASED_SDXL_PYTHON"
RUN_INFO_ENV = "NODEBASED_SDXL_RUN_INFO"
OFFLOAD_ENV = "NODEBASED_SDXL_OFFLOAD"  # "never" keeps the whole pipeline on the card (faster, needs the card to itself)
PREVIEW_DIR_ENV = "NODEBASED_SDXL_PREVIEW_DIR"  # set: every denoising step is also decoded and written here (spike A, see steppreview.py)
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"  # `fast_preview` (spike A): the 4-step distilled LoRA, fused into the base UNet
LIGHTNING_LORA = "sdxl_lightning_4step_lora.safetensors"
LIGHTNING_STEPS = 4
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_MODEL = "xinsir/controlnet-union-sdxl-1.0"
WORKING_LONG_SIDE = 1024
# ControlNet-Union promax control types: 0 openpose, 1 depth, 2 soft edge/scribble, 3 canny/lineart,
# 4 normal, 5 segment, 6 tile, 7 repaint.
CONTROL_TYPE_DEPTH, CONTROL_TYPE_NORMAL = 1, 4
CONTROL_MODES = {"depth_normals": (CONTROL_TYPE_DEPTH, CONTROL_TYPE_NORMAL),
                 "depth": (CONTROL_TYPE_DEPTH,), "normals": (CONTROL_TYPE_NORMAL,)}
DEFAULT_SETTINGS = {"steps": 30, "guidance": 5.0, "controlnet_conditioning_scale": 0.8,
                    "control_mode": "depth_normals", "denoise_strength": 0.5}
# `fast_preview` is opt-in: a setting set to False is not recorded, so a run without it keeps its artifact id.
OPTIONAL_SETTINGS = ("fast_preview",)
SETTING_RANGES = {"steps": (1, 100), "guidance": (0.0, 20.0), "controlnet_conditioning_scale": (0.0, 2.0),
                  "denoise_strength": (0.05, 1.0)}
DEFAULT_PROMPT = "a photograph"
NEGATIVE_PROMPT = "blurry, low quality, watermark, text, deformed"
MIN_VALID_NORMAL_LENGTH = 0.5


# ----------------------------------------------------------------------------- NodeBased side

def resolve_settings(sampler_settings):
    """`sampler_settings` with defaults filled in; an unknown key or an out-of-range value is an error.

    `fast_preview: true` selects the SDXL-Lightning 4-step LoRA (spike A): the sampler is the card's Euler "trailing"
    schedule at exactly 4 steps with guidance 0, so `steps` and `guidance` may be left out or given those values and
    nothing else. `denoise_strength` still applies: at 0.5 a guided frame takes the last 2 of the 4 trained steps.
    """
    given = dict(sampler_settings or {})
    settings = dict(DEFAULT_SETTINGS)
    unknown = sorted(set(given) - set(DEFAULT_SETTINGS) - set(OPTIONAL_SETTINGS))
    if unknown:
        raise ValueError(f"sdxl_controlnet sampler_settings has unknown keys: {', '.join(unknown)} "
                         f"(allowed: {', '.join(list(DEFAULT_SETTINGS) + list(OPTIONAL_SETTINGS))})")
    fast = given.pop("fast_preview", False)
    if not isinstance(fast, bool):
        raise ValueError("sdxl_controlnet fast_preview must be true or false")
    if fast:
        if given.get("steps", LIGHTNING_STEPS) != LIGHTNING_STEPS or given.get("guidance", 0) != 0:
            raise ValueError(f"sdxl_controlnet fast_preview runs the {LIGHTNING_STEPS}-step Lightning schedule at guidance 0: "
                             "leave steps and guidance out or give exactly those")
        settings.update(steps=LIGHTNING_STEPS, guidance=0.0)
    settings.update(given)
    for key, (low, high) in SETTING_RANGES.items():
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"sdxl_controlnet {key} must be a number from {low} to {high}")
    if not isinstance(settings["steps"], int):
        raise ValueError("sdxl_controlnet steps must be a whole number")
    settings["guidance"] = float(settings["guidance"])
    settings["controlnet_conditioning_scale"] = float(settings["controlnet_conditioning_scale"])
    settings["denoise_strength"] = float(settings["denoise_strength"])
    if fast:
        settings["fast_preview"] = True
        if settings["denoise_strength"] < 1.0 and int(LIGHTNING_STEPS * settings["denoise_strength"]) < 1:
            raise ValueError("sdxl_controlnet fast_preview needs denoise_strength of 0.25 or more (at least one of the 4 steps)")
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


def _single_plane(values):
    values = np.asarray(values, np.float32)
    return values[..., 0] if values.ndim == 3 else values


def write_control_files(controls, workdir):
    """The control images and warp inputs of every frame as .npy files under `workdir`: (job rows, width, height)."""
    workdir = Path(workdir)
    near, far = depth_range([frame.depth.values for frame in controls])
    height, width = controls[0].beauty.values.shape[:2]
    rows = []
    for frame in controls:
        base = workdir / f"frame_{frame.frame:04d}"
        np.save(f"{base}.depth.npy", depth_control_image(frame.depth.values, near, far))
        np.save(f"{base}.normals.npy", normals_control_image(frame.normals.values))
        # What the reprojection needs, at the plate size: backward motion, metric depth and object IDs.
        np.save(f"{base}.motion.npy", np.asarray(frame.motion_backward.values, np.float32)[..., :2])
        np.save(f"{base}.z.npy", _single_plane(frame.depth.values))
        np.save(f"{base}.ids.npy", _single_plane(frame.object_ids.values))
        rows.append({"frame": frame.frame, "depth": f"{base}.depth.npy", "normals": f"{base}.normals.npy",
                     "motion_backward": f"{base}.motion.npy", "z": f"{base}.z.npy", "ids": f"{base}.ids.npy",
                     "result": f"{base}.result.npy"})
    return rows, width, height


def run_frames(controls, *, text, seed, sampler_settings, workdir, progress=None, cancelled=None, preview=None):
    """Generate one picture per frame; returns (list of ACEScg RGBA arrays, run info).

    `controls` are ControlFrames (depth and normals planes). Control images and results go through `workdir`
    as .npy files; the model process reads a small job file and prints `progress i n` lines.

    Previews (spike A): with `preview` (a callable taking one event dict) or the environment variable
    `NODEBASED_SDXL_PREVIEW_DIR`, the model process also decodes the clean-image estimate after every denoising step
    and prints a `preview {json}` line per step (see `steppreview.py`); the pictures go under that folder (or `workdir`).
    Previews never change the generated pictures: the decode uses its own copy of the VAE and no random numbers.
    """
    settings = resolve_settings(sampler_settings)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows, width, height = write_control_files(controls, workdir)
    job = {"frames": rows, "width": width, "height": height, "prompt": text or DEFAULT_PROMPT,
           "seed": int(seed), "settings": settings, "info": str(workdir / "info.json")}
    preview_dir = os.environ.get(PREVIEW_DIR_ENV) or (str(workdir / "previews") if preview else None)
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        job["preview_dir"] = preview_dir
    job_path = workdir / "job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    # No address-space cap here: CUDA reserves terabytes of virtual address space (see workers.py, local_gpu).
    from ..gpulock import gpu_job  # the model side of this file runs in another interpreter, so import here
    with gpu_job(cancelled):
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
                elif line.startswith("preview "):
                    event = json.loads(line[len("preview "):])
                    event["received"] = time.monotonic()
                    if preview:
                        preview(event)
                    if os.environ.get(PREVIEW_DIR_ENV):
                        with open(Path(preview_dir) / "previews.jsonl", "a", encoding="utf-8") as log:
                            log.write(json.dumps(event) + "\n")
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


def _load_pipeline(torch, fast_preview=False):
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
    if fast_preview:
        # SDXL-Lightning 4-step LoRA (spike A), fused into the UNet, with the card's schedule: Euler, "trailing"
        # timestep spacing, epsilon prediction (the base scheduler's), guidance 0. The adapter is loaded before the
        # offload hooks are armed and unloaded after the fuse, so nothing of it stays resident but the fused weights.
        from diffusers import EulerDiscreteScheduler
        pipe.load_lora_weights(str(_cached_snapshot(LIGHTNING_REPO) / LIGHTNING_LORA), adapter_name="lightning")
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


def _load_temporal():
    """`nodebased/temporal.py` by path: the model side imports nothing from the NodeBased package."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "temporal.py"
    spec = importlib.util.spec_from_file_location("nodebased_temporal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_working(plane, work_w, work_h, motion=False):
    """A plate-size plane at the working size by nearest sampling (IDs and depth must not blend across edges);
    motion vectors are also rescaled to working-size pixels."""
    h, w = plane.shape[:2]
    rows = np.minimum(((np.arange(work_h) + 0.5) * h / work_h).astype(np.int64), h - 1)
    cols = np.minimum(((np.arange(work_w) + 0.5) * w / work_w).astype(np.int64), w - 1)
    out = np.asarray(plane, np.float32)[rows][:, cols]
    if motion:
        out = out * np.array([work_w / w, work_h / h], np.float32)
    return out


def guide_image(temporal, previous_picture, row, previous_row, work_w, work_h):
    """The previous working-size picture warped into this frame, and the mask of the pixels it can supply."""
    motion = _to_working(np.load(row["motion_backward"]), work_w, work_h, motion=True)
    depth, ids = (_to_working(np.load(row[key]), work_w, work_h) for key in ("z", "ids"))
    before_depth, before_ids = (_to_working(np.load(previous_row[key]), work_w, work_h) for key in ("z", "ids"))
    warped, _ = temporal.warp(np.asarray(previous_picture, np.float32), motion)
    mask = temporal.valid_mask(motion, depth, ids, before_depth, before_ids)
    return warped, mask


class _Cancelled(Exception):
    """Raised from the step callback when the edit being sampled was superseded (serve mode)."""


class _StepTap:
    """Remembers the scheduler's clean-image estimate (`pred_original_sample`) of the last step.

    The pipelines call `scheduler.step(..., return_dict=False)`, which for Euler returns (previous sample, predicted
    clean sample); the pipeline keeps the first. Wrapping the method changes nothing it computes or draws.
    """

    def __init__(self, scheduler):
        self.latest = None
        self._step = scheduler.step
        scheduler.step = self

    def __call__(self, *args, **kwargs):
        result = self._step(*args, **kwargs)
        self.latest = result[1] if isinstance(result, tuple) else result.pred_original_sample
        return result


class _Engine:
    """The loaded pipelines and the frame loop, shared by `run` (one job, then exit) and `serve` (many edits)."""

    def __init__(self, torch, settings, preview_dir=None):
        self.torch = torch
        self.settings = settings
        self.fast = bool(settings.get("fast_preview"))
        self.strength = settings.get("denoise_strength", 1.0)
        self.guided = self.strength < 1.0
        self.modes = list(CONTROL_MODES[settings["control_mode"]])
        started = time.perf_counter()
        self.pipe = _load_pipeline(torch, self.fast)
        self.preview_vae = None
        if preview_dir is not None:
            # A second, resident copy of the VAE for the previews, made before the offload hooks wrap the first: decoding
            # through the pipeline's own VAE would evict the UNet from the card at every step.
            import copy
            self.preview_vae = copy.deepcopy(self.pipe.vae).to("cuda")
        self.tap = _StepTap(self.pipe.scheduler) if self.preview_vae is not None else None
        self.refine = None
        if self.guided:
            from diffusers import StableDiffusionXLControlNetUnionImg2ImgPipeline
            import inspect
            # The same model objects, through the constructor: `from_pipe` left the second pipeline's models on the
            # card after an offloaded call (the first frame ran out of memory on the 12 GB card).
            keys = set(inspect.signature(StableDiffusionXLControlNetUnionImg2ImgPipeline.__init__).parameters) - {"self"}
            self.refine = StableDiffusionXLControlNetUnionImg2ImgPipeline(
                **{key: value for key, value in self.pipe.components.items() if key in keys})
            self.refine.set_progress_bar_config(disable=True)
            self.temporal = _load_temporal()
        # Model CPU offload always, unless NODEBASED_SDXL_OFFLOAD=never. The whole pipeline fits the 12 GB card only
        # when nothing else uses it, and a fallback after an out-of-memory error leaves the pipeline in a different
        # state (the same seed then gave different pixels), so the memory strategy never depends on what else runs.
        self.offloaded = os.environ.get(OFFLOAD_ENV) != "never"
        self.active = [None]
        if self.offloaded:
            self.ready(self.pipe)
        else:
            self.pipe.to("cuda")
        self.loaded_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        self.timesteps = {}  # pass name -> the timesteps the scheduler ran, for the log

    def ready(self, pipeline):
        """The pipeline about to run, with the offload hooks armed for it. The two pipelines share their models,
        and one pipeline's hooks never release the models after the other runs (the card ran out of memory), so
        the hooks are re-armed at every switch; the order of switches is fixed, so the pixels are too."""
        if self.offloaded and self.active[0] is not pipeline:
            pipeline.enable_model_cpu_offload()
            self.active[0] = pipeline
        return pipeline

    def _sampling(self, pipeline, stage, frame, sink, cancelled):
        """Keyword arguments that make `pipeline(...)` report every step to `sink` (empty without a sink)."""
        if sink is None and cancelled is None:
            return {}
        torch = self.torch
        vae = self.preview_vae

        def on_step(_pipeline, index, timestep, kwargs):
            if cancelled is not None and cancelled():
                raise _Cancelled()
            if sink is not None and self.tap is not None and self.tap.latest is not None:
                started = time.monotonic()
                with torch.no_grad():
                    clean = (self.tap.latest / vae.config.scaling_factor).to(torch.float16)
                    picture = vae.decode(clean, return_dict=False)[0]
                    picture = ((picture / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 2, 0)
                    array = picture.cpu().numpy()
                sink(frame=frame, stage=stage, step=index + 1, of=self._steps[0], array=array,
                     decode_seconds=time.monotonic() - started, timestep=float(timestep))
            if cancelled is not None and cancelled():
                raise _Cancelled()
            return kwargs
        return {"callback_on_step_end": on_step, "callback_on_step_end_tensor_inputs": ["latents"]}

    def run_job(self, job, *, sink=None, cancelled=None, on_frame=None):
        """Generate the job's frames in order; returns the guide canvas bookkeeping. `sink` gets every step's
        preview, `cancelled()` ends the job early with `_Cancelled`, `on_frame(index, total, row, picture)` follows
        each finished frame."""
        torch = self.torch
        from PIL import Image
        settings = job["settings"]
        width, height = job["width"], job["height"]
        work_w, work_h = working_size(width, height)
        names = {"depth": "depth", "normals": "normals"}
        total = len(job["frames"])
        # The guide canvas: frame 1's picture carried forward by the warp, its holes filled from each frame's own
        # text-to-image picture. Image-to-image results never enter it (chaining them compounded the model's contrast
        # and texture gain until the picture broke up by frame 6, version 1.1).
        previous_row, canvas, filled_fractions = None, None, []
        self._steps = [settings["steps"]]
        for index, row in enumerate(job["frames"]):
            if cancelled is not None and cancelled():
                raise _Cancelled()
            images = []
            for mode in self.modes:
                kind = "depth" if mode == CONTROL_TYPE_DEPTH else "normals"
                array = np.load(row[names[kind]])
                images.append(Image.fromarray(array).resize((work_w, work_h), Image.BICUBIC))
            arguments = dict(
                prompt=job["prompt"], negative_prompt=NEGATIVE_PROMPT, control_image=images, control_mode=self.modes,
                controlnet_conditioning_scale=[settings["controlnet_conditioning_scale"]] * len(self.modes),
                num_inference_steps=settings["steps"], guidance_scale=settings["guidance"],
                width=work_w, height=work_h, output_type="pil")
            follows = previous_row is not None and row["frame"] == previous_row["frame"] + 1
            try:
                if self.guided and follows:
                    warped, mask = guide_image(self.temporal, canvas, row, previous_row, work_w, work_h)
                    canvas = warped
                    if not mask.all():  # the holes come from this frame's own text-to-image picture
                        generator = torch.Generator("cuda").manual_seed(job["seed"])
                        self._steps = [settings["steps"]]
                        extra = self._sampling(self.pipe, "fill", row["frame"], sink, cancelled)
                        plain = np.asarray(self.ready(self.pipe)(generator=generator, **extra, **arguments).images[0], np.float32)
                        canvas = np.where(mask[..., None], warped, plain)
                    filled_fractions.append(round(1.0 - float(mask.mean()), 4))
                    start = Image.fromarray(np.rint(np.clip(canvas, 0, 255)).astype(np.uint8))
                    generator = torch.Generator("cuda").manual_seed(job["seed"])
                    # Image-to-image keeps the last int(steps * strength) steps of the schedule.
                    self._steps = [min(int(settings["steps"] * self.strength), settings["steps"])]
                    extra = self._sampling(self.refine, "guided", row["frame"], sink, cancelled)
                    picture = self.ready(self.refine)(generator=generator, image=start, strength=self.strength,
                                                      **extra, **arguments).images[0]
                else:
                    generator = torch.Generator("cuda").manual_seed(job["seed"])
                    self._steps = [settings["steps"]]
                    extra = self._sampling(self.pipe, "text2img", row["frame"], sink, cancelled)
                    picture = self.ready(self.pipe)(generator=generator, **extra, **arguments).images[0]
                    canvas = np.asarray(picture, np.float32)
            except _Cancelled:
                # Hooks that a pipeline call would have released on its way out.
                (self.active[0] or self.pipe).maybe_free_model_hooks()
                raise
            previous_row = row
            result = np.asarray(picture.resize((width, height), Image.LANCZOS), dtype=np.uint8)
            np.save(row["result"], result)
            if on_frame:
                on_frame(index, total, row, picture, result)
            else:
                print(f"progress {index + 1} {total}", flush=True)
        return filled_fractions

    def info(self, job, generate_seconds, filled_fractions):
        import diffusers
        import resource
        torch = self.torch
        width, height = job["width"], job["height"]
        work_w, work_h = working_size(width, height)
        return {"load_seconds": round(self.loaded_seconds, 2), "generate_seconds": round(generate_seconds, 2),
                "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                "host_rss_peak_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                "cpu_offload": self.offloaded, "working_size": [work_w, work_h],
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "diffusers": diffusers.__version__, "frames": len(job["frames"]),
                "denoise_strength": self.strength, "filled_fraction_per_frame": filled_fractions,
                **({"fast_preview": True, "sampler": "EulerDiscrete trailing, guidance 0",
                    "lora": f"{LIGHTNING_REPO}/{LIGHTNING_LORA}",
                    "lora_revision": _cached_snapshot(LIGHTNING_REPO).name} if self.fast else {}),
                **({"previews": True} if self.preview_vae is not None else {})}


def _setup_torch():
    # Deterministic algorithms need this before cuBLAS starts (the parent sets it too).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import diffusers.utils.logging as diffusers_logging
    diffusers_logging.set_verbosity_error()  # moving an fp16 pipeline off the card warns once per component
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: sdxl_controlnet runs on the NVIDIA card only")
    return torch


def _preview_sink(preview_dir, **tags):
    """The model side's preview writer: the picture as a .npy under `preview_dir`, then one `preview {json}` line."""
    folder = Path(preview_dir)

    def sink(*, frame, stage, step, of, array, decode_seconds, timestep):
        name = f"r{tags.get('revision', 0):04d}_f{frame:04d}_{stage}_{step}of{of}.npy"
        written = time.monotonic()
        np.save(folder / (name + ".tmp.npy"), array)
        os.replace(folder / (name + ".tmp.npy"), folder / name)
        print("preview " + json.dumps({**tags, "frame": frame, "stage": stage, "step": step, "of": of,
                                       "path": str(folder / name), "width": int(array.shape[1]),
                                       "height": int(array.shape[0]), "decode_seconds": round(decode_seconds, 4),
                                       "write_seconds": round(time.monotonic() - written, 4),
                                       "timestep": timestep, "done": time.monotonic()}), flush=True)
    return sink


def _model_main(job_path):
    torch = _setup_torch()
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    engine = _Engine(torch, job["settings"], job.get("preview_dir"))
    sink = _preview_sink(job["preview_dir"]) if job.get("preview_dir") else None
    began = time.perf_counter()
    filled = engine.run_job(job, sink=sink)
    Path(job["info"]).write_text(json.dumps(engine.info(job, time.perf_counter() - began, filled), indent=1), encoding="utf-8")


def _serve_main(settings_json):
    """Interactive edits for the viewer (spike A): load once, then read one JSON request per line on stdin.

    {"op": "edit", "id": N, "revision": R, "job": "JOB.json"} samples that job (one or more frames) and prints
    `preview {json}` per step, `final {json}` per frame and `done {json}` or `cancelled {json}` at the end.
    A request that arrives while another samples supersedes it: the running edit stops after its current step and only the
    newest waiting request is kept (older waiting ones are counted as coalesced). {"op": "quit"} ends the process.
    """
    import threading
    torch = _setup_torch()
    settings = resolve_settings(json.loads(settings_json))
    engine = _Engine(torch, settings, preview_dir="serve")
    print("ready " + json.dumps(engine.info({"frames": [], "width": 1024, "height": 576}, 0.0, []) | {"mono": time.monotonic()}), flush=True)
    lock, wake = threading.Lock(), threading.Event()
    state = {"pending": None, "coalesced": 0, "active_revision": None, "quit": False}

    def reader():
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            with lock:
                if request["op"] == "quit":
                    state["quit"] = True
                elif request["op"] == "edit":
                    request["arrived"] = time.monotonic()
                    if state["pending"] is not None:
                        state["coalesced"] += 1
                        print("coalesced " + json.dumps({"id": state["pending"]["id"], "revision": state["pending"]["revision"]}), flush=True)
                    state["pending"] = request
            wake.set()
        with lock:
            state["quit"] = True
        wake.set()
    threading.Thread(target=reader, daemon=True).start()

    def superseded():
        with lock:
            return state["quit"] or (state["pending"] is not None
                                     and state["pending"]["revision"] > state["active_revision"])
    while True:
        wake.wait()
        with lock:
            request, state["pending"] = state["pending"], None
            quitting = state["quit"]
            if request is None:
                wake.clear()
            else:
                state["active_revision"] = request["revision"]
        if request is None:
            if quitting:
                return
            continue
        job = json.loads(Path(request["job"]).read_text(encoding="utf-8"))
        tags = {"id": request["id"], "revision": request["revision"]}
        sink = _preview_sink(job["preview_dir"], **tags)

        def on_frame(index, total, row, picture, result):
            print("final " + json.dumps({**tags, "frame": row["frame"], "path": row["result"], "done": time.monotonic()}), flush=True)
        began = time.monotonic()
        print("started " + json.dumps({**tags, "mono": began, "queued_seconds": round(began - request["arrived"], 4)}), flush=True)
        try:
            engine.run_job(job, sink=sink, cancelled=superseded, on_frame=on_frame)
            print("done " + json.dumps({**tags, "seconds": round(time.monotonic() - began, 3), "mono": time.monotonic(),
                                    "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                                    "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}), flush=True)
        except _Cancelled:
            print("cancelled " + json.dumps({**tags, "seconds": round(time.monotonic() - began, 3), "mono": time.monotonic()}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _model_main(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "serve":
        _serve_main(sys.argv[2])
    else:
        raise SystemExit("usage: sdxl_controlnet.py run JOB.json (started by NodeBased, not by hand)")
