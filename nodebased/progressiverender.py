"""The 3D viewport's progressive "Render" mode (docs/SPLAT_RELIGHTING.md, "Production look" step R6
"next", closed): the R3/R4 path tracer (`pathtrace`) run in small, doubling sample steps so the
viewport stays interactive while the image converges toward what `Render3D` would produce, instead
of one long blocking trace.

Each step re-traces the whole image at a higher sample count than the last (1+2+4+...+64 is about
2x 64 samples in all), except the full-size adaptive steps, which continue one accumulation (see
"Settling" below) and so cost no more than the final image. The very
first step after a reset (a new `key`: the camera moved, or the scene changed) renders at
``FIRST_PASS_SCALE`` under size, cheap enough to hold interactive frame rates while the camera is
still moving (measured on an RTX 3080 Ti through the ``auto`` backend: a two-mesh scene, quarter
size, 1 sample, 2-4 ms once the path tracer's pipelines are warm). Every later step is full size,
doubling the sample count up to ``SAMPLE_CAP``, where the state is ``converged``.

The picture is denoised in the interactive steps below (`step_interactive`); `step` alone shows the raw accumulation.

Interactive first look (lane 4, plan "Progress", step H1): `step_interactive` replaces the noisy quarter-size reset step with a
*denoised* preview of the full output size inside about a second. Stage one traces `PREVIEW_STAGES[0]` pixels (480 x 270 for a 1080p
view) at 4 samples, its guides (albedo, normals, depth) from the same prepared scene (`gpupathtrace.reusing_scene`: one scene build
and pack for all four GPU renders), filters it with `ptdenoise.denoise_strips` in float32 and enlarges it bilinearly to the output size;
stage two does the same at 960 x 540 and 16 samples. Both states say `preview=True` (an enlarged internal pass, not a render at the
output size) and `denoised=True`; the doubling steps then continue at full size exactly as `step` always did (32 then 64 samples), raw.
The one-second target is a measured one: `tools/benchmark_first_look.py` reports the time to the first displayed image.

Settling (lane 4, plan "Render mode responsiveness", step C): the full-size steps no longer re-trace the picture at twice the samples.
They continue one accumulation (`step(accumulate=True)`): the path tracer keeps its running sums, moments, counts and done flags
(`stats["keep_accumulation"]`, taken back through `stats["resume"]`), so each step gives the pixels still above the noise threshold their
next samples and leaves the cleared ones as they were; the result is the image a single render at that sample cap gives. The picture is
settled when every pixel is under `NOISE_THRESHOLD` or `SAMPLE_CAP` is reached, and `converged_fraction` and `converged_mask` say how much of
it is. The filter runs on the graphics card (`gpudenoise`, `denoise_picture`), so a 1080p picture is filtered in tens of milliseconds
instead of 1.4 s on the processor. A second preview stage that stage one's cost per pixel says would be slow is skipped
(`STAGE_TWO_MAX_SECONDS`). Motion blur and shadow catchers keep the earlier re-tracing steps.

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

import math
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import gpu3d, gpudenoise, gpupathtrace, pathtrace, ptdenoise, scenecache, scene3d
from .cancellation import Cancelled

FIRST_PASS_SCALE = 4     # the reset step renders at 1/this size on each side, for interactive framing
FIRST_PASS_SAMPLES = 1
SAMPLE_CAP = 64           # doubling stops here; the state is `converged` once it is reached at full size
MAX_BOUNCES = 4           # a lighter bounce budget than Render3D's own default, for interactivity
# Adaptive sampling (plan "Rendering 6", step R1): from `ADAPTIVE_FROM` samples a step stops each pixel on the path tracer's
# own noise estimate (`pathtrace.pixel_noise`) instead of giving every pixel every sample, and the view finishes early when
# every pixel has. The first steps (1, 2, 4 samples) are too few to estimate noise from, so they stay fixed.
ADAPTIVE_FROM = 8
NOISE_THRESHOLD = 0.0125   # relative variance of a pixel's mean luminance (a standard error of about 11 percent) under which it stops


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
    cost_per_pixel: float = 0.0                # seconds per traced pixel of stage one, the scene's preparation taken off (0: not measured)
    guides: object = field(default=None, compare=False, repr=False)   # (albedo, normals, depth) at the output size, kept for the next pass
    accumulation: object = field(default=None, compare=False, repr=False)   # the path tracer's running sums, for `step(accumulate=True)` to continue
    converged_mask: object = field(default=None, compare=False, repr=False)  # (height, width) bool: the pixels under the noise threshold; None: not measured


def _drop_particles(scene):
    return replace(scene, particles=()) if getattr(scene, "particles", ()) else scene


drop_particles = _drop_particles      # the job drops them once per view, so every pass shares one prepared scene


def step(state, scene, camera, width, height, background, ambient, key, backend="auto",
        max_bounces=MAX_BOUNCES, moments=None, cancel=None, accumulate=False):
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
    accumulating = accumulate and not moments and measured and not scene3d.has_shadow_catchers(scene)
    if accumulating:
        # Continue the render the last pass left (`stats["resume"]`): the pixels still above the noise threshold take their next
        # samples and the ones under it are left as they were, instead of tracing the whole picture again at twice the samples. The
        # result is the one a single render at this cap gives (the passes are the same ones, in the same order).
        stats["keep_accumulation"] = True
        stats["resume"] = state.accumulation
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
                            converged_fraction=fraction, finished=fraction is not None and fraction >= 1.0,
                            accumulation=stats.get("accumulation") if accumulating else None,
                            converged_mask=stats["converged"] if measured and "converged" in stats else None)


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
# Stage one may trace more than its 480 x 270 when the output is large and the adapter fast (step B of the Progress plan): up to this
# many pixels, never above a quarter of the output's (an enlargement of at least 2 x on each side), and only as far as the cost
# model below says the look still fits its target. A view smaller than that keeps the stage's own size, so the preview ladder is
# skipped exactly where it always was.
FIRST_STAGE_MAX_PIXELS = 640 * 360
PREVIEW_BOUNCES = 3
PREVIEW_WORKERS = 8
FIRST_LOOK_SECONDS = 1.0         # the target the benchmark compares with; a goal, reported, never assumed
WARM_FIRST_LOOK_SECONDS = 0.65   # the first stage's own target when the scene's preparation is cached (a camera move, a material edit):
                                 # the picture then reaches the screen under 0.8 s once the debounce and the display conversion are added
# The first stage sizes itself to the target from what the last one cost on this adapter (`_COST`): seconds per traced pixel,
# for surface scenes and for scenes with splats or smoke apart (those take a slower GPU variant, 10 to 20 times the cost per
# pixel). A scene never seen on this adapter starts at the stage's own pixel count, or at `HEAVY_FIRST_PIXELS` when it has
# splats or smoke; the next first look is sized to fit `FIRST_LOOK_SECONDS * FIT_FRACTION` less the enlargement's allowance.
FIT_FRACTION = 0.85
CATCHER_FIT_FRACTION = 0.3  # leave room for fixed scene/guide overhead on Hot pour-style catcher previews
ENLARGE_ALLOWANCE = 0.03
HEAVY_FIRST_PIXELS = 240 * 135
MIN_FIRST_PIXELS = 120 * 68
CATCHER_MIN_FIRST_PIXELS = 64 * 36  # keep a useful thumbnail while leaving room for fixed setup cost
STAGE_TWO_MAX_FIRST_SECONDS = 0.75      # a first stage slower than this (a heavy scene) goes straight to the doubling steps
STAGE_TWO_MAX_SECONDS = 1.2             # so does one whose second stage the first stage's cost per pixel says would take longer than this
_COST = {}
_BUILD = {}       # the same keys: seconds the last scene preparation took (the build and the pack), when one was made


def internal_size(width, height, pixels):
    """The (width, height) of about `pixels` pixels with the aspect ratio of `width` by `height`, never above it."""
    scale = min(1.0, (pixels / float(width * height)) ** 0.5)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


_BAND_POOL = None
BAND_WORKERS = max(2, min(8, os.cpu_count() or 2))
ENLARGE_MIN_ROWS = 96          # output rows per band at the least; a smaller picture is done in one go


def _band_pool():
    global _BAND_POOL
    if _BAND_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _BAND_POOL = ThreadPoolExecutor(BAND_WORKERS, thread_name_prefix="nodebased-bands")
    return _BAND_POOL


def _in_bands(rows, minimum, work):
    """`work(y0, y1)` over `rows` split into bands on a small thread pool (the array arithmetic inside releases the GIL); the
    bands are disjoint, so the result is the one a single call over all the rows gives."""
    count = max(1, min(BAND_WORKERS, rows // max(minimum, 1)))
    if count < 2:
        return work(0, rows)
    edges = np.linspace(0, rows, count + 1).astype(int)
    futures = [_band_pool().submit(work, int(edges[i]), int(edges[i + 1])) for i in range(count)]
    return [f.result() for f in futures]


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
    out = np.empty((height, width, image.shape[2]), np.float32)

    def band(lo, hi):
        rows = image[y0[lo:hi]] * (1 - wy[lo:hi])[:, None, None] + image[y1[lo:hi]] * wy[lo:hi][:, None, None]
        out[lo:hi] = rows[:, x0] * (1 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]
    _in_bands(height, ENLARGE_MIN_ROWS, band)
    return out


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
    a float64 power on every channel: about a tenth of the time at 1080p. The rows are converted in bands on a small thread pool;
    with `cancel` (an event) each band looks at it first and raises `Cancelled`, the pixels are the same."""
    global _SRGB_LUT
    if _SRGB_LUT is None:
        x = np.linspace(0.0, 1.0, _SRGB_LUT_SIZE + 1)
        y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
        _SRGB_LUT = np.round(y * 255.0).astype(np.uint8)
    out = np.empty(image.shape[:2] + (4,), np.uint8)
    out[..., 3] = 255

    def band(y0, y1):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        part = image[y0:y1]
        rgb = np.clip(part[..., :3] / np.maximum(part[..., 3:4], 1e-6), 0.0, 1.0)
        out[y0:y1, :, :3] = _SRGB_LUT[(rgb * _SRGB_LUT_SIZE + 0.5).astype(np.int32)]
    _in_bands(image.shape[0], DISPLAY_BAND_ROWS, band)
    return out


def tint_refining(rgba, mask, amount=0.45, colour=(255, 120, 40)):
    """A copy of the 8-bit picture `rgba` with the pixels the noise estimate has not yet cleared (`mask` False, as in
    `ProgressiveState.converged_mask`) blended `amount` of the way toward `colour`: the viewport's debug view of where sampling goes on."""
    out = np.array(rgba, copy=True)
    if mask is None or mask.shape != out.shape[:2]:
        return out
    open_pixels = ~np.asarray(mask, bool)
    tint = np.asarray(colour, np.float32)
    out[open_pixels, :3] = (out[open_pixels, :3] * (1.0 - amount) + tint * amount + 0.5).astype(np.uint8)
    return out


def _cost_key(scene):
    from . import gpu3d
    heavy = bool(getattr(scene, "splats", ()) or getattr(scene, "volumes", ()))
    try:
        adapter = gpu3d.adapter_report().splitlines()[0]
    except Exception:       # no adapter: the CPU reference takes the render, and costs its own way
        adapter = "cpu"
    return adapter, heavy, bool(scene3d.has_shadow_catchers(scene))


FIRST_STAGE_STEP = 1.06       # the first stage's size moves in steps of this factor (see `_snap`)


def _snap(pixels):
    """`pixels` rounded down onto a ladder of sizes `FIRST_STAGE_STEP` apart, so that a small change in the measured cost does not
    change the first look's size from one orbit to the next (it would otherwise creep up and down with the card's clock)."""
    if pixels <= MIN_FIRST_PIXELS:
        return int(pixels)
    rung = math.floor(math.log(pixels / MIN_FIRST_PIXELS) / math.log(FIRST_STAGE_STEP) + 1e-9)
    return int(MIN_FIRST_PIXELS * FIRST_STAGE_STEP ** rung)


def first_stage_pixels(scene, stage_pixels, prepared=None):
    """How many pixels stage one traces for `scene`: `stage_pixels`, or fewer when the last first look of this kind of scene on this
    adapter cost too much per pixel to finish inside its target; more than `stage_pixels` never. `prepared` says what the scene's
    preparation (`scenecache`) will cost: True, nothing (a camera move); False, what the last build took (a material edit repacks, a
    geometry or splat edit rebuilds), taken off the target; both aim for `WARM_FIRST_LOOK_SECONDS`. None (no information) aims for
    `FIRST_LOOK_SECONDS` with nothing taken off."""
    key = _cost_key(scene)
    cost = _COST.get(key)
    if cost is None:
        return min(stage_pixels, HEAVY_FIRST_PIXELS) if key[1] else stage_pixels
    fraction = CATCHER_FIT_FRACTION if key[2] else FIT_FRACTION
    target = FIRST_LOOK_SECONDS if prepared is None else WARM_FIRST_LOOK_SECONDS
    budget = target * fraction - ENLARGE_ALLOWANCE - (_BUILD.get(key, 0.0) if prepared is False else 0.0)
    fit = budget / cost
    floor = CATCHER_MIN_FIRST_PIXELS if key[2] else MIN_FIRST_PIXELS
    chosen = max(floor, fit)
    return int(stage_pixels) if chosen >= stage_pixels else _snap(chosen)


def _prepared(scene, camera, ambient, backend):
    """True when the path tracer's preparation of `scene` for `camera` is cached, False when the first look will build it, None when
    that cannot be told."""
    try:
        return scenecache.is_prepared(_drop_particles(scene), ambient, scene3d._view_basis(camera)[0])
    except Exception:
        return None


def _record_cost(scene, seconds, pixels, built=0.0):
    """Remember what a stage-one trace of `pixels` cost per pixel for this kind of scene; `built` is the part of `seconds` that
    was spent preparing the scene, which is kept apart (`_BUILD`) so a trace that reused the preparation is not charged for it."""
    key = _cost_key(scene)
    cost = max(seconds - built, 0.0) / max(pixels, 1)
    _COST[key] = cost if key not in _COST else 0.5 * _COST[key] + 0.5 * cost
    if built > 0.0:
        _BUILD[key] = built          # the last build is the best guess for the next one (a repack after a cold build is much cheaper)


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
    filtered = denoise_picture(beauty, guides["albedo"], guides["normals"][..., :3], guides["depth"][..., 0], backend, cancel,
                               PREVIEW_WORKERS)
    lap("denoise")
    # the bilinear weights sum to one, so the background goes under the small picture and the enlargement carries it along
    image = enlarge(pathtrace.over_background(filtered, background).astype(np.float32), width, height)
    lap("enlarge")
    return image, (tw, th), tuple(phases)


DENOISE_WORKERS = max(2, min(12, os.cpu_count() or 2))
_CARD_FILTER = {"broken": False}


def denoise_picture(beauty, albedo, normals, depth, backend="auto", cancel=None, workers=None):
    """The viewport's edge-aware filter (`ptdenoise`) over `beauty` with its guides, float32: on the graphics card when the path
    tracer runs there (`gpudenoise`, the same filter in about a fiftieth of the time at 1080p), else on the processor in strips. A card
    that refuses the picture (too large for one storage binding) or fails leaves this one on the processor; a failure other than a
    refusal turns the card filter off for the session."""
    if backend in ("auto", "gpu") and not _CARD_FILTER["broken"] and gpudenoise.available():
        try:
            return gpudenoise.denoise(beauty, albedo, normals, depth, cancel=cancel)
        except Cancelled:
            raise
        except gpu3d.Unsupported:
            pass
        except Exception:
            _CARD_FILTER["broken"] = True
    return ptdenoise.denoise_strips(beauty, albedo, normals, depth, None, workers=workers or DENOISE_WORKERS, dtype=np.float32,
                                    cancel=cancel)
GUIDE_SAMPLES = 1        # the guide passes need far less than the preview's samples: normals and depth are un-jittered, and one albedo sample moved the error on the silhouettes by 1 percent


_GUIDE_POOL = None


def _guide_pool():
    global _GUIDE_POOL
    if _GUIDE_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _GUIDE_POOL = ThreadPoolExecutor(1, thread_name_prefix="nodebased-guides")
    return _GUIDE_POOL


def _denoised_step(state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments, cancel):
    """`step` with its picture filtered: the same render (adaptive sampling, `converged_fraction`, `finished`) traced over a clear
    background, filtered with the albedo, normals and depth guides, then composited over `background`. The guides are traced once
    for the view (they do not depend on the sample count) and carried in the returned state for the next pass. On a scene with smoke the
    card's part of the guides comes first and the processor's part (the smoke's raymarch) runs on a second thread while the card
    traces the pass, so the card is not left idle for it."""
    scene = _drop_particles(scene)
    clear = (0.0, 0.0, 0.0, 0.0)
    if state is None or state.key != key:
        state = ProgressiveState(key=key)
    reset = state.samples == 0
    shape = (max(1, height // FIRST_PASS_SCALE), max(1, width // FIRST_PASS_SCALE)) if reset else (height, width)
    with gpupathtrace.reusing_scene():
        guides, traced, finish, future, card_part = state.guides, None, None, None, 0.0
        if guides is None or guides[0].shape[:2] != shape:
            guides = None
            settings = pathtrace.PathSettings(samples=GUIDE_SAMPLES, max_bounces=max_bounces)
            asked = time.perf_counter()
            traced, finish = pathtrace.guide_aovs_deferred(scene, camera, shape[1], shape[0], settings, cancel=cancel,
                                                           backend=backend, ambient=ambient)
            card_part = time.perf_counter() - asked
            if finish is not None:
                future = _guide_pool().submit(finish)
        try:
            raw = step(state, scene, camera, width, height, clear, ambient, key, backend, max_bounces, moments, cancel, accumulate=True)
        except BaseException:
            if future is not None:
                from concurrent.futures import wait
                wait([future])          # the pass stopped (a cancel set the event the march looks at): let the march end before returning
            raise
        started = time.perf_counter()
        if future is not None:
            traced = future.result()
        if guides is None:
            guides = (traced["albedo"], traced["normals"][..., :3], traced["depth"][..., 0])
    guided = time.perf_counter()
    filtered = denoise_picture(raw.image, guides[0], guides[1], guides[2], backend, cancel)
    image = pathtrace.over_background(filtered, background).astype(np.float32)
    done = time.perf_counter()
    return replace(raw, image=image, denoised=True, guides=guides, seconds=raw.seconds + card_part + done - started,
                   phases=(("render", raw.seconds), ("guides", card_part + guided - started), ("denoise", done - guided)))


def _ladder_over(state, width, height, stages):
    """True when the preview stages are done for `state` (all taken, the next no smaller than the output, the cap reached, or a
    first stage too slow to follow with a dearer one): the doubling steps take over."""
    return (state.stage >= len(stages) or stages[state.stage][0] >= width * height or state.samples >= SAMPLE_CAP
            or (state.stage == 1 and (state.seconds > STAGE_TWO_MAX_FIRST_SECONDS
                                      or state.cost_per_pixel * stages[1][0] * stages[1][1] / max(stages[0][1], 1) > STAGE_TWO_MAX_SECONDS)))


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
        return replace(step(state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments, cancel,
                            accumulate=True), stage=len(stages))
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
        pixels = max(pixels, min(FIRST_STAGE_MAX_PIXELS, width * height // 4))
        pixels = first_stage_pixels(scene, pixels, _prepared(scene if trace_scene is None else trace_scene, camera, ambient, backend))
    started, built_before = time.perf_counter(), scenecache.BUILD_SECONDS
    image, traced, phases = _preview(_drop_particles(scene), camera, width, height, background, ambient, pixels, samples,
                                     backend, min(max_bounces, PREVIEW_BOUNCES), moments, cancel,
                                     trace_scene=trace_scene,
                                     retain_preparation=bool(getattr(scene, "volumes", ())))
    seconds = time.perf_counter() - started
    cost = state.cost_per_pixel
    if state.stage == 0:
        built = scenecache.BUILD_SECONDS - built_before
        _record_cost(scene, seconds - dict(phases)["enlarge"], traced[0] * traced[1], built)
        cost = max(seconds - dict(phases)["enlarge"] - built, 0.0) / max(traced[0] * traced[1], 1)
    return ProgressiveState(key=key, samples=samples, width=width, height=height, image=image,
                            seconds=seconds, low_res=True, passes=state.passes + 1,
                            preview=True, denoised=True, stage=state.stage + 1,
                            traced_width=traced[0], traced_height=traced[1], phases=phases, cost_per_pixel=cost)
