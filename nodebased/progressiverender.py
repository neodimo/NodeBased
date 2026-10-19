"""The 3D viewport's progressive "Render" mode (docs/SPLAT_RELIGHTING.md, "Production look" step R6
"next", closed): the R3/R4 path tracer (`pathtrace`) run in small, doubling sample steps so the
viewport stays interactive while the image converges toward what `Render3D` would produce, instead
of one long blocking trace.

Each step re-traces the whole image from a fresh seed at a higher sample count than the last,
rather than resuming a running accumulation across resolution changes -- simpler and safer, at
roughly twice the total sample cost of the final image (1+2+4+...+64 is about 2x 64). The very
first step after a reset (a new `key`: the camera moved, or the scene changed) renders at
``FIRST_PASS_SCALE`` under size, cheap enough to hold interactive frame rates while the camera is
still moving (measured on an RTX 3080 Ti through the ``auto`` backend: a two-mesh scene, quarter
size, 1 sample, 2-4 ms once the path tracer's pipelines are warm). Every later step is full size,
doubling the sample count up to ``SAMPLE_CAP``, where the state is ``converged``.

Denoising is not wired in here. ``pathtrace.render(..., output="denoise")`` was measured at about
1.1 seconds a call in this environment, independent of the sample count (`guide_aovs`'s normals and
depth passes fall outside the fast GPU beauty path and fall back to the slow CPU reference) -- a
poor fit for a per-step call in a progressive loop. The raw (noisy) accumulation is what this module
shows; wiring the denoiser in as an occasional final-settle pass, off the paint thread, is later work.

Interactive first look (lane 4, plan "Progress", step H1): `step_interactive` replaces the noisy quarter-size reset step with a
*denoised* preview of the full output size inside about a second. Stage one traces `PREVIEW_STAGES[0]` pixels (480 x 270 for a 1080p
view) at 4 samples, its guides (albedo, normals, depth) from the same prepared scene (`gpupathtrace.reusing_scene`: one scene build
and pack for all four GPU renders), filters it with `ptdenoise.denoise_strips` in float32 and enlarges it bilinearly to the output size;
stage two does the same at 960 x 540 and 16 samples. Both states say `preview=True` (an enlarged internal pass, not a render at the
output size) and `denoised=True`; the doubling steps then continue at full size exactly as `step` always did (32 then 64 samples), raw.
The one-second target is a measured one: `tools/benchmark_first_look.py` reports the time to the first displayed image.

Depth of field comes along for free: the viewed camera's own ``fstop``/``focus_distance``/lens knobs
(when it is a ``Camera3D`` looked through, not the orbit camera, which has none) are read by
``pathtrace.camera_rays`` on every sampled ray the same way ``Render3D`` reads them, so a step with
more than one sample already shows the thin-lens blur, sharper as the sample count climbs.

Motion blur (step X2) is the caller's job: pass ``moments``, a list of ``(scene, camera)`` across the
shutter (`viewport3d.py`'s ``_motion_moments``, built the same way `imaging.py`'s ``_motion_inputs``
builds Render3D's own), and each step traces `pathtrace.render_motion` over them instead of one
`pathtrace.render` call, at the same doubling sample count and low-res reset schedule as ever.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import gpupathtrace, pathtrace, ptdenoise, scene3d

FIRST_PASS_SCALE = 4     # the reset step renders at 1/this size on each side, for interactive framing
FIRST_PASS_SAMPLES = 1
SAMPLE_CAP = 64           # doubling stops here; the state is `converged` once it is reached at full size
MAX_BOUNCES = 4           # a lighter bounce budget than Render3D's own default, for interactivity
# Adaptive sampling (plan "Rendering 6", step R1): from `ADAPTIVE_FROM` samples a step stops each pixel on the path tracer's
# own noise estimate (`pathtrace.pixel_noise`) instead of giving every pixel every sample, and the view finishes early when
# every pixel has. The first steps (1, 2, 4 samples) are too few to estimate noise from, so they stay fixed.
ADAPTIVE_FROM = 8
NOISE_THRESHOLD = 0.01


@dataclass(frozen=True)
class ProgressiveState:
    key: object = None
    samples: int = 0                          # 0: nothing traced yet for this key
    width: int = 0
    height: int = 0
    image: np.ndarray | None = None            # premultiplied float32 (height, width, 4), or None
    seconds: float = 0.0                       # the last step's own render time
    low_res: bool = False                      # True while `image` is still the undersized reset step
    passes: int = 0                            # steps taken for this key (the "pass" the view shows)
    converged_fraction: float | None = None    # share of pixels under the noise threshold; None: not measured yet
    finished: bool = False                     # every pixel converged at full size: no further step is needed
    # Interactive first look (`step_interactive`). `image` is then enlarged to the output size, so `width`/`height` are the
    # output's and `traced_width`/`traced_height` are what was actually traced.
    preview: bool = False                      # True: `image` is an enlarged internal pass, not a render at the output size
    denoised: bool = False                     # True: `image` went through `ptdenoise` before it was shown
    stage: int = 0                             # preview stages done for this key (0 before the first look)
    traced_width: int = 0
    traced_height: int = 0
    phases: tuple = ()                         # ((name, seconds), ...) of the last first-look stage: render, guides, denoise, enlarge


def _drop_particles(scene):
    return replace(scene, particles=()) if getattr(scene, "particles", ()) else scene


def step(state, scene, camera, width, height, background, ambient, key, backend="auto",
        max_bounces=MAX_BOUNCES, moments=None):
    """One progressive step: `state` (or a fresh one when `key` differs from its own) advanced by a
    single, higher-sample-count re-trace. Particles are dropped first (the path tracer does not draw
    them, `pathtrace.check_scene`); everything else the viewport shows -- meshes, splats, volumes,
    lights and environments -- goes through unchanged.

    `moments`, when given, is `[(scene, camera)]` across the shutter (see the module docstring): the
    step traces `pathtrace.render_motion` over them instead of the single `scene`/`camera`, at the
    same sample-doubling schedule.
    """
    width, height = max(1, int(width)), max(1, int(height))
    if state is None or state.key != key:
        state = ProgressiveState(key=key)
    reset = state.samples == 0
    target = FIRST_PASS_SAMPLES if reset else min(state.samples * 2, SAMPLE_CAP)
    render_width = max(1, width // FIRST_PASS_SCALE) if reset else width
    render_height = max(1, height // FIRST_PASS_SCALE) if reset else height
    measured = target >= ADAPTIVE_FROM and not reset
    settings = pathtrace.PathSettings(
        samples=target, max_bounces=max_bounces,
        **(dict(sampling="adaptive", noise_threshold=NOISE_THRESHOLD, min_samples=ADAPTIVE_FROM, max_samples=target,
                adaptive_pass_size=ADAPTIVE_FROM) if measured else {}))
    stats = {"skip_noise_maps": True}      # only `converged` is read below: the GPU path tracer builds no variance or noise maps
    started = time.perf_counter()
    if moments:
        moments = [(_drop_particles(scene_at), camera_at) for scene_at, camera_at in moments]
        image = pathtrace.render_motion(moments, render_width, render_height, background, ambient=ambient,
                                        output="rgba", settings=settings, backend=backend, stats=stats)
    else:
        scene = _drop_particles(scene)
        image = pathtrace.render(scene, camera, render_width, render_height, background, ambient=ambient,
                                 output="rgba", settings=settings, backend=backend, stats=stats)
    elapsed = time.perf_counter() - started
    # `stats["converged"]` is there when one render measured it (the CPU reference and the GPU path tracer; a GPU
    # motion-blur render is several renders averaged and measures nothing)
    fraction = float(stats["converged"].mean()) if measured and "converged" in stats else None
    return ProgressiveState(key=key, samples=target, width=render_width, height=render_height,
                            image=image, seconds=elapsed, low_res=reset, passes=state.passes + 1,
                            converged_fraction=fraction, finished=fraction is not None and fraction >= 1.0)


def converged(state, width, height):
    """True once `state` has reached `SAMPLE_CAP` at the full `width`/`height` (not the low-res
    reset step), or every pixel is under the noise threshold there (`finished`, step R1): the terminal
    state, where further `step` calls for the same key are wasted work."""
    return (state is not None and (state.samples >= SAMPLE_CAP or state.finished)
           and state.width == max(1, int(width)) and state.height == max(1, int(height)))


# --- the interactive first look -------------------------------------------------------------------------------------------

# (pixels traced, samples) per preview stage. Measured on the RTX 3080 Ti with the furnished benchmark room
# (tools/benchmark_first_look.py): stage one warm in well under a second at 1080p, stage two about twice that.
PREVIEW_STAGES = ((480 * 270, 4), (960 * 540, 16))
PREVIEW_BOUNCES = 3
PREVIEW_WORKERS = 8
FIRST_LOOK_SECONDS = 1.0         # the target `first_look_seconds` is compared with; a goal, reported, never assumed


def internal_size(width, height, pixels):
    """The (width, height) of about `pixels` pixels with the aspect ratio of `width` by `height`, never above it."""
    scale = min(1.0, (pixels / float(width * height)) ** 0.5)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def enlarge(image, width, height):
    """`image` (h, w, c) enlarged to `width` by `height` with bilinear weights (pixel centres aligned), float32."""
    h, w = image.shape[:2]
    if (h, w) == (height, width):
        return np.asarray(image, np.float32)
    def axis(count, target):
        position = np.clip((np.arange(target) + 0.5) * count / target - 0.5, 0, count - 1)
        low = np.minimum(position.astype(np.int64), count - 1)
        high = np.minimum(low + 1, count - 1)
        return low, high, (position - low).astype(np.float32)
    y0, y1, wy = axis(h, height)
    x0, x1, wx = axis(w, width)
    image = np.asarray(image, np.float32)
    rows = image[y0] * (1 - wy)[:, None, None] + image[y1] * wy[:, None, None]
    return rows[:, x0] * (1 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]


_SRGB_LUT_SIZE = 8192
_SRGB_LUT = None


def display_rgba8(image):
    """The viewport's 8-bit sRGB RGBA for a premultiplied linear `image` (h, w, 4): the colour un-premultiplied, clipped to
    [0, 1] and passed through a table (within one level of the exact curve), alpha opaque. A float32 table lookup instead of
    a float64 power on every channel: about a tenth of the time at 1080p."""
    global _SRGB_LUT
    if _SRGB_LUT is None:
        x = np.linspace(0.0, 1.0, _SRGB_LUT_SIZE + 1)
        y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
        _SRGB_LUT = np.round(y * 255.0).astype(np.uint8)
    rgb = np.clip(image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0.0, 1.0)
    out = np.empty(image.shape[:2] + (4,), np.uint8)
    out[..., :3] = _SRGB_LUT[(rgb * _SRGB_LUT_SIZE + 0.5).astype(np.int32)]
    out[..., 3] = 255
    return out


def _preview(scene, camera, width, height, background, ambient, pixels, samples, backend, max_bounces, moments, cancel):
    """One denoised preview stage: trace `pixels` pixels at `samples` samples with guides from one prepared scene, filter,
    enlarge to `width` by `height`, composite over `background`. Returns (image float32, (traced_w, traced_h), phases)."""
    tw, th = internal_size(width, height, pixels)
    settings = pathtrace.PathSettings(samples=samples, max_bounces=max_bounces)
    phases, mark = [], time.perf_counter()

    def lap(name):
        nonlocal mark
        now = time.perf_counter()
        phases.append((name, now - mark))
        mark = now
    clear = (0.0, 0.0, 0.0, 0.0)
    stats = {"skip_noise_maps": True}
    with gpupathtrace.reusing_scene():
        if moments:
            moments = [(_drop_particles(scene_at), camera_at) for scene_at, camera_at in moments]
            beauty = pathtrace.render_motion(moments, tw, th, clear, ambient=ambient, output="rgba", settings=settings,
                                             backend=backend, stats=stats, cancel=cancel)
        else:
            beauty = pathtrace.render(scene, camera, tw, th, clear, ambient=ambient, output="rgba", settings=settings,
                                      backend=backend, stats=stats, cancel=cancel)
        lap("render")
        guides = pathtrace.guide_aovs(scene, camera, tw, th, settings, cancel=cancel, backend=backend)
        lap("guides")
    filtered = ptdenoise.denoise_strips(beauty, guides["albedo"], guides["normals"][..., :3], guides["depth"][..., 0],
                                        None, workers=PREVIEW_WORKERS, dtype=np.float32)
    lap("denoise")
    big = enlarge(filtered, width, height)
    image = pathtrace.over_background(big, background).astype(np.float32)
    lap("enlarge")
    return image, (tw, th), tuple(phases)


def step_interactive(state, scene, camera, width, height, background, ambient, key, backend="auto",
                     max_bounces=MAX_BOUNCES, moments=None, cancel=None, stages=None):
    """`step` with a denoised first look. For a fresh `key` the first call returns stage one of `PREVIEW_STAGES`, the second
    stage two (a stage whose pixel count is not below the output's is skipped: a small view just starts the doubling), and
    every later call is `step` unchanged. The preview states carry the output size with `preview=True`, `denoised=True` and
    what was traced in `traced_width`/`traced_height`; `cancel` (an event) stops a stage between passes with
    `cancellation.Cancelled`, leaving the caller's previous state as it was (nothing is stored here)."""
    width, height = max(1, int(width)), max(1, int(height))
    stages = PREVIEW_STAGES if stages is None else stages
    if state is None or state.key != key:
        state = ProgressiveState(key=key)
    if state.stage >= len(stages) or stages[state.stage][0] >= width * height or state.samples >= SAMPLE_CAP:
        return step(state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments)
    pixels, samples = stages[state.stage]
    started = time.perf_counter()
    image, traced, phases = _preview(_drop_particles(scene), camera, width, height, background, ambient, pixels, samples,
                                     backend, min(max_bounces, PREVIEW_BOUNCES), moments, cancel)
    return ProgressiveState(key=key, samples=samples, width=width, height=height, image=image,
                            seconds=time.perf_counter() - started, low_res=True, passes=state.passes + 1,
                            preview=True, denoised=True, stage=state.stage + 1,
                            traced_width=traced[0], traced_height=traced[1], phases=phases)
