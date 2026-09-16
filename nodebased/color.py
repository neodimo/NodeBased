"""Explicit fixed-config OCIO color pipeline. Working gamut is linear ACEScg.

Every buffer between a Read and the display is scene-linear float32 ACEScg with
premultiplied alpha. ACEScg is wider than Rec.709, so a value that would have
clipped at the edge of the old working gamut now survives the graph and only
meets a gamut boundary at the view transform.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np

# OCIO's CPU processor releases the GIL inside applyRGB, so row-chunking across a thread
# pool is a real parallel speedup rather than contended Python bytecode: measured ~10-12x
# at both HD and 4K for the ACES 2.0 view (docs/BENCHMARKS-v0.16-display.md). This is the
# CPU-side fallback used whenever no GPU display context is available; it changes nothing
# about the math, only how many rows are handed to `applyRGB` per call.
_CPU_WORKERS = min(os.cpu_count() or 1, 16)
_cpu_pool = ThreadPoolExecutor(max_workers=_CPU_WORKERS, thread_name_prefix='display-cpu')

CONFIG = 'ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5'
WORKING = 'ACEScg'
INPUT_SPACES = {'sRGB': 'sRGB Encoded Rec.709 (sRGB)', 'Linear Rec.709': 'Linear Rec.709 (sRGB)',
                'ACEScg': 'ACEScg', 'ACES2065-1': 'ACES2065-1', 'Raw': 'Raw'}
VIEWS = ('sRGB', 'ACES 2.0', 'Linear')
# Spaces that are already the working space, so ingest is a no-op. `Raw` is the explicit
# opt-out: it means "do not interpret these numbers", e.g. a normal or depth pass.
PASSTHROUGH = tuple(name for name, space in INPUT_SPACES.items() if space == WORKING) + ('Raw',)


@lru_cache(maxsize=1)
def config():
    import PyOpenColorIO as ocio
    return ocio.Config.CreateFromBuiltinConfig(CONFIG)


@lru_cache(maxsize=16)
def processor(source, target):
    return config().getProcessor(source, target).getDefaultCPUProcessor()


@lru_cache(maxsize=8)
def display_processor(view):
    """CPU processor for a display `view`, built once and reused.

    The ACES 2.0 view previously built a brand-new `DisplayViewTransform` processor on
    every call, which dwarfed every other cost in `display_rgb` (see
    docs/BENCHMARKS-v0.16-display.md). `sRGB` already went through the cached `processor()`
    below; this gives ACES 2.0 the same treatment via the same cache discipline.
    """
    if view == 'sRGB':
        return processor(WORKING, INPUT_SPACES['sRGB'])
    if view == 'ACES 2.0':
        import PyOpenColorIO as ocio
        transform = ocio.DisplayViewTransform(src=WORKING, display='sRGB - Display',
                                             view='ACES 2.0 - SDR 100 nits (Rec.709)')
        return config().getProcessor(transform).getDefaultCPUProcessor()
    raise ValueError(f'Unknown display view: {view}')


@lru_cache(maxsize=8)
def display_gpu_processor(view):
    """GPU processor for a display `view`, mirroring `display_processor` for `gpudisplay`."""
    if view == 'sRGB':
        return config().getProcessor(WORKING, INPUT_SPACES['sRGB']).getDefaultGPUProcessor()
    if view == 'ACES 2.0':
        import PyOpenColorIO as ocio
        transform = ocio.DisplayViewTransform(src=WORKING, display='sRGB - Display',
                                             view='ACES 2.0 - SDR 100 nits (Rec.709)')
        return config().getProcessor(transform).getDefaultGPUProcessor()
    raise ValueError(f'Unknown display view: {view}')


def to_working(rgba, space, associated=False):
    """Convert a decoded frame into the premultiplied ACEScg working representation.

    The unpremult/premult round trip below operates on the *whole* contiguous ``result``
    array rather than the ``result[..., :3]`` view: that view skips one float per pixel (the
    alpha channel), which is not C-contiguous, and NumPy's elementwise loop over it measured
    ~4x slower than the same divide/multiply on a full contiguous array of the same size
    (~144ms vs ~60ms for one 4K frame -- docs/BENCHMARKS-v0.17-playback.md). The alpha column
    of the scale factor is fixed at 1.0, so dividing/multiplying by it is a bit-exact no-op for
    every finite alpha value, and the real alpha is restored from the saved copy immediately
    after the color processor runs regardless.
    """
    result = np.array(rgba, dtype=np.float32, copy=True, order='C')
    alpha = result[..., 3:4].copy()
    if space not in INPUT_SPACES:
        raise ValueError(f'Unknown input color space: {space}')
    if associated:
        scale = np.where(np.abs(alpha) > 1e-8, alpha, 1).astype(np.float32)
        ones = np.ones_like(scale)
        divisor = np.concatenate([scale, scale, scale, ones], axis=2)
        result /= divisor
    if space not in PASSTHROUGH:
        processor(INPUT_SPACES[space], WORKING).applyRGBA(result)
    result[..., 3:4] = alpha
    channel_scale = scale if associated else alpha
    multiplier = np.concatenate([channel_scale, channel_scale, channel_scale,
                                 np.ones_like(channel_scale)], axis=2)
    result *= multiplier
    return result


# Measured on the readback-design GPU path (docs/BENCHMARKS-v0.16-display.md): its cost is
# dominated by the fixed per-call texture upload/readback, not by the transform it runs, so
# it wins big on the expensive ACES 2.0 view (~10x at HD, ~4x at 4K over the threaded CPU
# path) but is a net loss for the already-cheap sRGB view, worst at 4K (~2x slower than
# threaded CPU). Routing per view avoids "auto-select GPU" turning into a regression on the
# common proxy/preview path; ACES 2.0 is also the shipped default display view
# (docs/COLOR_MANAGEMENT.md), so this is the case that actually needs the GPU.
_GPU_PREFERRED_VIEWS = frozenset({'ACES 2.0'})


def display_rgb(rgb, view):
    if view == 'Linear':
        return rgb
    if view not in ('sRGB', 'ACES 2.0'):
        raise ValueError(f'Unknown display view: {view}')
    if view in _GPU_PREFERRED_VIEWS:
        from . import gpudisplay
        gpu = gpudisplay.get_display()
        if gpu is not None:
            try:
                return gpu.render(rgb, view)
            except gpudisplay.GpuUnavailable:
                pass  # Falls through to the CPU path below; the failure is already recorded.
    image = np.array(rgb, dtype=np.float32, copy=True, order='C')
    apply_threaded(display_processor(view), image)
    return image


def apply_threaded(cpu_processor, image):
    """Apply an OCIO CPU processor in-place, chunked across `_cpu_pool`.

    A single row-major `applyRGB` call is the reference behaviour this must match
    exactly: it is not an approximation, just the same processor invoked on row bands
    that happen to be contiguous C-order slices, so the pixel math is identical.
    """
    height = image.shape[0]
    workers = min(_CPU_WORKERS, height) or 1
    if workers <= 1:
        cpu_processor.applyRGB(image)
        return image
    chunks = np.array_split(image, workers, axis=0)
    list(_cpu_pool.map(cpu_processor.applyRGB, chunks))
    return image
