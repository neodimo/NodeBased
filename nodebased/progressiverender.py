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

import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import gpupathtrace, pathtrace, ptdenoise, scenecache, scene3d
from .cancellation import Cancelled

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
    guides: object = field(default=None, compare=False, repr=False)   # (albedo, normals, depth) at the output size, kept for the next pass


def _drop_particles(scene):
    return replace(scene, particles=()) if getattr(scene, "particles", ()) else scene


drop_particles = _drop_particles      # the job drops them once per view, so every pass shares one prepared scene


def step(state, scene, camera, width, height, background, ambient, key, backend="auto",
        max_bounces=MAX_BOUNCES, moments=None, cancel=None):
    """One progressive step: `state` (or a fresh one when `key` differs from its own) advanced by a
    single, higher-sample-count re-trace. Particles are dropped first (the path tracer does not draw
    them, `pathtrace.check_scene`); everything else the viewport shows -- meshes, splats, volumes,
    lights and environments -- goes through unchanged.

    `moments`, when given, is `[(scene, camera)]` across the shutter (see the module docstring): the
    step traces `pathtrace.render_motion` over them instead of the single `scene`/`camera`, at the
    same sample-doubling schedule. `cancel` (an event) stops the trace between GPU passes with
    `cancellation.Cancelled`; nothing is stored here, so the caller's previous state stays as it was.
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
                                        output="rgba", settings=settings, backend=backend, stats=stats, cancel=cancel)
    else:
        scene = _drop_particles(scene)
        image = pathtrace.render(scene, camera, render_width, render_height, background, ambient=ambient,
                                 output="rgba", settings=settings, backend=backend, stats=stats, cancel=cancel)
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
FIRST_LOOK_SECONDS = 1.0         # the target the benchmark compares with; a goal, reported, never assumed
# The first stage sizes itself to the target from what the last one cost on this adapter (`_COST`): seconds per traced pixel,
# for surface scenes and for scenes with splats or smoke apart (those take a slower GPU variant, 10 to 20 times the cost per
# pixel). A scene never seen on this adapter starts at the stage's own pixel count, or at `HEAVY_FIRST_PIXELS` when it has
# splats or smoke; the next first look is sized to fit `FIRST_LOOK_SECONDS * FIT_FRACTION` less the enlargement's allowance.
FIT_FRACTION = 0.85
CATCHER_FIT_FRACTION = 0.3  # leave room for fixed scene/guide overhead on Hot pour-style catcher previews
ENLARGE_ALLOWANCE = 0.06
HEAVY_FIRST_PIXELS = 240 * 135
MIN_FIRST_PIXELS = 120 * 68
CATCHER_MIN_FIRST_PIXELS = 64 * 36  # keep a useful thumbnail while leaving room for fixed setup cost
STAGE_TWO_MAX_FIRST_SECONDS = 0.75      # a first stage slower than this (a heavy scene) goes straight to the doubling steps
_COST = {}
_BUILD = {}       # the same keys: seconds the last scene preparation took (the build and the pack), when one was made


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


def _thumbnail_guides(image):
    """Spatial-only guides for the one initial catcher thumbnail; later passes use traced guides."""
    h, w = image.shape[:2]
    albedo = np.zeros((h, w, 3), np.float32)
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = 1.0
    return {"albedo": albedo, "normals": normals, "depth": np.zeros((h, w, 1), np.float32)}


_SRGB_LUT_SIZE = 8192
_SRGB_LUT = None


DISPLAY_BAND_ROWS = 64


def display_rgba8(image, cancel=None):
    """The viewport's 8-bit sRGB RGBA for a premultiplied linear `image` (h, w, 4): the colour un-premultiplied, clipped to
    [0, 1] and passed through a table (within one level of the exact curve), alpha opaque. A float32 table lookup instead of
    a float64 power on every channel: about a tenth of the time at 1080p. With `cancel` (an event) the conversion runs in
    bands of `DISPLAY_BAND_ROWS` rows and raises `Cancelled` between them; the pixels are the same."""
    global _SRGB_LUT
    if _SRGB_LUT is None:
        x = np.linspace(0.0, 1.0, _SRGB_LUT_SIZE + 1)
        y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
        _SRGB_LUT = np.round(y * 255.0).astype(np.uint8)
    out = np.empty(image.shape[:2] + (4,), np.uint8)
    out[..., 3] = 255
    rows = image.shape[0] if cancel is None else DISPLAY_BAND_ROWS
    for y in range(0, image.shape[0], rows):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        band = image[y:y + rows]
        rgb = np.clip(band[..., :3] / np.maximum(band[..., 3:4], 1e-6), 0.0, 1.0)
        out[y:y + rows, :, :3] = _SRGB_LUT[(rgb * _SRGB_LUT_SIZE + 0.5).astype(np.int32)]
    return out


def _cost_key(scene):
    from . import gpu3d
    heavy = bool(getattr(scene, "splats", ()) or getattr(scene, "volumes", ()))
    try:
        adapter = gpu3d.adapter_report().splitlines()[0]
    except Exception:       # no adapter: the CPU reference takes the render, and costs its own way
        adapter = "cpu"
    return adapter, heavy, bool(scene3d.has_shadow_catchers(scene))


def first_stage_pixels(scene, stage_pixels, prepared=True):
    """How many pixels stage one traces for `scene`: `stage_pixels`, or fewer when the last first look of this kind of scene on this
    adapter cost too much per pixel to finish inside `FIRST_LOOK_SECONDS`; more than `stage_pixels` never. `prepared` False says the
    scene's preparation (`scenecache`) will be built again, and the last such build's time is taken off the budget first: a camera
    move or a material edit reuses the preparation and traces more pixels, a geometry or splat edit pays for the build."""
    key = _cost_key(scene)
    cost = _COST.get(key)
    if cost is None:
        return min(stage_pixels, HEAVY_FIRST_PIXELS) if key[1] else stage_pixels
    fraction = CATCHER_FIT_FRACTION if key[2] else FIT_FRACTION
    budget = FIRST_LOOK_SECONDS * fraction - ENLARGE_ALLOWANCE - (0.0 if prepared else _BUILD.get(key, 0.0))
    fit = budget / cost
    floor = CATCHER_MIN_FIRST_PIXELS if key[2] else MIN_FIRST_PIXELS
    return int(min(stage_pixels, max(floor, fit)))


def _prepared(scene, camera, ambient, backend):
    """True when the path tracer's preparation of `scene` for `camera` is cached (or there is nothing to prepare on the GPU)."""
    try:
        return scenecache.is_prepared(_drop_particles(scene), ambient, scene3d._view_basis(camera)[0])
    except Exception:
        return True


def _record_cost(scene, seconds, pixels, built=0.0):
    """Remember what a stage-one trace of `pixels` cost per pixel for this kind of scene; `built` is the part of `seconds` that
    was spent preparing the scene, which is kept apart (`_BUILD`) so a trace that reused the preparation is not charged for it."""
    key = _cost_key(scene)
    cost = max(seconds - built, 0.0) / max(pixels, 1)
    _COST[key] = cost if key not in _COST else 0.5 * _COST[key] + 0.5 * cost
    if built > 0.0:
        _BUILD[key] = built if key not in _BUILD else 0.5 * _BUILD[key] + 0.5 * built


def _preview(scene, camera, width, height, background, ambient, pixels, samples, backend, max_bounces, moments, cancel,
             *, trace_scene=None, retain_preparation=True):
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
    trace_scene = scene if trace_scene is None else trace_scene
    with gpupathtrace.reusing_scene(retain=retain_preparation):
        if moments:
            moments = [(_drop_particles(scene_at), camera_at) for scene_at, camera_at in moments]
            trace_moments = [(trace_scene if scene_at is scene else scene_at, camera_at) for scene_at, camera_at in moments]
            beauty = pathtrace.render_motion(trace_moments, tw, th, clear, ambient=ambient, output="rgba", settings=settings,
                                             backend=backend, stats=stats, cancel=cancel)
        else:
            beauty = pathtrace.render(trace_scene, camera, tw, th, clear, ambient=ambient, output="rgba", settings=settings,
                                      backend=backend, stats=stats, cancel=cancel)
        lap("render")
        guides = (_thumbnail_guides(beauty) if trace_scene is not scene else
                  pathtrace.guide_aovs(scene, camera, tw, th, settings, cancel=cancel, backend=backend, ambient=ambient))
        lap("guides")
    filtered = ptdenoise.denoise_strips(beauty, guides["albedo"], guides["normals"][..., :3], guides["depth"][..., 0],
                                        None, workers=PREVIEW_WORKERS, dtype=np.float32, cancel=cancel)
    lap("denoise")
    big = enlarge(filtered, width, height)
    image = pathtrace.over_background(big, background).astype(np.float32)
    lap("enlarge")
    return image, (tw, th), tuple(phases)


DENOISE_WORKERS = max(2, min(12, os.cpu_count() or 2))
GUIDE_SAMPLES = 4        # the guide passes need no more than the preview's samples (normals and depth are un-jittered)


def _denoised_step(state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments, cancel):
    """`step` with its picture filtered: the same render (adaptive sampling, `converged_fraction`, `finished`) traced over a clear
    background, filtered with the albedo, normals and depth guides, then composited over `background`. The guides are traced once
    for the view (they do not depend on the sample count) and carried in the returned state for the next pass."""
    scene = _drop_particles(scene)
    clear = (0.0, 0.0, 0.0, 0.0)
    with gpupathtrace.reusing_scene():
        raw = step(state, scene, camera, width, height, clear, ambient, key, backend, max_bounces, moments, cancel)
        started = time.perf_counter()
        guides = state.guides
        if guides is None or guides[0].shape[:2] != (raw.height, raw.width):
            settings = pathtrace.PathSettings(samples=GUIDE_SAMPLES, max_bounces=max_bounces)
            traced = pathtrace.guide_aovs(scene, camera, raw.width, raw.height, settings, cancel=cancel, backend=backend,
                                          ambient=ambient)
            guides = (traced["albedo"], traced["normals"][..., :3], traced["depth"][..., 0])
    guided = time.perf_counter()
    filtered = ptdenoise.denoise_strips(raw.image, guides[0], guides[1], guides[2], None, workers=DENOISE_WORKERS,
                                        dtype=np.float32, cancel=cancel)
    image = pathtrace.over_background(filtered, background).astype(np.float32)
    done = time.perf_counter()
    return replace(raw, image=image, denoised=True, guides=guides, seconds=raw.seconds + done - started,
                   phases=(("render", raw.seconds), ("guides", guided - started), ("denoise", done - guided)))


def _ladder_over(state, width, height, stages):
    """True when the preview stages are done for `state` (all taken, the next no smaller than the output, the cap reached, or a
    first stage too slow to follow with a dearer one): the doubling steps take over."""
    return (state.stage >= len(stages) or stages[state.stage][0] >= width * height or state.samples >= SAMPLE_CAP
            or (state.stage == 1 and state.seconds > STAGE_TWO_MAX_FIRST_SECONDS))


def next_samples(state, width, height, stages=None):
    """The sample count the next `step_interactive` call will reach for `state` (what a viewport can say about the pass in
    flight); the cap once `state` has it."""
    stages = PREVIEW_STAGES if stages is None else stages
    if state is None:
        return stages[0][1] if stages and stages[0][0] < width * height else FIRST_PASS_SAMPLES
    if not _ladder_over(state, width, height, stages):
        return stages[state.stage][1]
    return min(max(state.samples, 1) * 2, SAMPLE_CAP)


def step_interactive(state, scene, camera, width, height, background, ambient, key, backend="auto",
                     max_bounces=MAX_BOUNCES, moments=None, cancel=None, stages=None, denoise_passes=False):
    """`step` with a denoised first look. For a fresh `key` the first call returns stage one of `PREVIEW_STAGES`, the second
    stage two (a stage whose pixel count is not below the output's is skipped: a small view just starts the doubling), and
    every later call is `step` unchanged. The preview states carry the output size with `preview=True`, `denoised=True` and
    what was traced in `traced_width`/`traced_height`; `cancel` (an event) stops a stage between passes with
    `cancellation.Cancelled`, leaving the caller's previous state as it was (nothing is stored here). With `denoise_passes` the
    full-size doubling steps are filtered too (`_denoised_step`), so the picture never gets noisier than the preview it follows;
    without it they are the raw `step` as before."""
    width, height = max(1, int(width)), max(1, int(height))
    stages = PREVIEW_STAGES if stages is None else stages
    if state is None or state.key != key:
        state = ProgressiveState(key=key)
    if _ladder_over(state, width, height, stages):
        # the preview ladder is over for this key (`stage` says so, or the next `step_interactive` would start it again)
        if denoise_passes:
            return replace(_denoised_step(state, scene, camera, width, height, background, ambient, key, backend, max_bounces,
                                          moments, cancel), stage=len(stages))
        return replace(step(state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments, cancel),
                       stage=len(stages))
    pixels, samples = stages[state.stage]
    trace_scene = None
    if state.stage == 0 and not moments and scene3d.has_shadow_catchers(scene):
        # The initial picture is a framing aid; shadowcatch.render otherwise launches extra full
        # path-traced passes to measure a catcher shadow before the artist can see anything. Keep
        # the catcher as a holdout so the foreground/plate composition is intact, then let later
        # refinement passes render the real shadow layer.
        from . import shadowcatch
        trace_scene = shadowcatch.hold_scene(_drop_particles(scene))
    if state.stage == 0:
        pixels = first_stage_pixels(scene, pixels, _prepared(scene if trace_scene is None else trace_scene, camera, ambient, backend))
    started, built_before = time.perf_counter(), scenecache.BUILD_SECONDS
    image, traced, phases = _preview(_drop_particles(scene), camera, width, height, background, ambient, pixels, samples,
                                     backend, min(max_bounces, PREVIEW_BOUNCES), moments, cancel,
                                     trace_scene=trace_scene,
                                     retain_preparation=bool(getattr(scene, "volumes", ())))
    seconds = time.perf_counter() - started
    if state.stage == 0:
        _record_cost(scene, seconds - dict(phases)["enlarge"], traced[0] * traced[1], scenecache.BUILD_SECONDS - built_before)
    return ProgressiveState(key=key, samples=samples, width=width, height=height, image=image,
                            seconds=seconds, low_res=True, passes=state.passes + 1,
                            preview=True, denoised=True, stage=state.stage + 1,
                            traced_width=traced[0], traced_height=traced[1], phases=phases)
