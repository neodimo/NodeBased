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

from . import pathtrace, scene3d

FIRST_PASS_SCALE = 4     # the reset step renders at 1/this size on each side, for interactive framing
FIRST_PASS_SAMPLES = 1
SAMPLE_CAP = 64           # doubling stops here; the state is `converged` once it is reached at full size
MAX_BOUNCES = 4           # a lighter bounce budget than Render3D's own default, for interactivity


@dataclass(frozen=True)
class ProgressiveState:
    key: object = None
    samples: int = 0                          # 0: nothing traced yet for this key
    width: int = 0
    height: int = 0
    image: np.ndarray | None = None            # premultiplied float32 (height, width, 4), or None
    seconds: float = 0.0                       # the last step's own render time
    low_res: bool = False                      # True while `image` is still the undersized reset step


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
    settings = pathtrace.PathSettings(samples=target, max_bounces=max_bounces)
    started = time.perf_counter()
    if moments:
        moments = [(_drop_particles(scene_at), camera_at) for scene_at, camera_at in moments]
        image = pathtrace.render_motion(moments, render_width, render_height, background, ambient=ambient,
                                        output="rgba", settings=settings, backend=backend)
    else:
        scene = _drop_particles(scene)
        image = pathtrace.render(scene, camera, render_width, render_height, background, ambient=ambient,
                                 output="rgba", settings=settings, backend=backend)
    elapsed = time.perf_counter() - started
    return ProgressiveState(key=key, samples=target, width=render_width, height=render_height,
                            image=image, seconds=elapsed, low_res=reset)


def converged(state, width, height):
    """True once `state` has reached `SAMPLE_CAP` at the full `width`/`height` (not the low-res
    reset step): the terminal state, where further `step` calls for the same key are wasted work."""
    return (state is not None and state.samples >= SAMPLE_CAP
           and state.width == max(1, int(width)) and state.height == max(1, int(height)))
