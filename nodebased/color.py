"""Explicit fixed-config OCIO color pipeline. Working gamut is linear ACEScg.

Every buffer between a Read and the display is scene-linear float32 ACEScg with
premultiplied alpha. ACEScg is wider than Rec.709, so a value that would have
clipped at the edge of the old working gamut now survives the graph and only
meets a gamut boundary at the view transform.
"""
from functools import lru_cache
import numpy as np

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


def to_working(rgba, space, associated=False):
    result = np.array(rgba, dtype=np.float32, copy=True, order='C')
    alpha = result[..., 3:4].copy()
    if space not in INPUT_SPACES:
        raise ValueError(f'Unknown input color space: {space}')
    if associated:
        scale = np.where(np.abs(alpha) > 1e-8, alpha, 1).astype(np.float32)
        result[..., :3] /= scale
    if space not in PASSTHROUGH:
        processor(INPUT_SPACES[space], WORKING).applyRGBA(result)
    result[..., 3:4] = alpha
    result[..., :3] *= scale if associated else alpha
    return result


def display_rgb(rgb, view):
    if view == 'Linear':
        return rgb
    image = np.array(rgb, dtype=np.float32, copy=True, order='C')
    if view == 'sRGB':
        processor(WORKING, INPUT_SPACES['sRGB']).applyRGB(image)
    elif view == 'ACES 2.0':
        import PyOpenColorIO as ocio
        transform = ocio.DisplayViewTransform(src=WORKING, display='sRGB - Display',
                                             view='ACES 2.0 - SDR 100 nits (Rec.709)')
        config().getProcessor(transform).getDefaultCPUProcessor().applyRGB(image)
    else:
        raise ValueError(f'Unknown display view: {view}')
    return image
