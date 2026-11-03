"""Behaviour of EdgeBlur, GodRays and BumpBoss as Nuke 17.0 describes them (lane 2, 2D behaviour step G).

EdgeBlur: "Blurs detected edges within the matte specified in the matte channel." The band is the matte's own morphological gradient
(box dilate minus box erode over `edge_mult * size` pixels), so a lower `edge_mult` gives a narrower, sharper band. Inside the band the
picture is the blur of `size` with the node's `filter` and `quality` (the Blur node's kernels, nodebased/blurknobs.py); `tint` multiplies
the red, green and blue of that blur and `brightness` all four channels ("the multiplier for the rgb channels / for all channels in the
edge blur"); outside the band the pixels are the input. `edge_matte` names the channel the edges are detected on (Nuke's control channel).
An older document's node keeps its Gaussian with no quality limit (blurknobs.OLD) and tint and brightness 1, which is what it did.

An older document's node of these kinds gets the flags and knob values of `upgrade_params` on load, so it renders exactly as before.
"""
from __future__ import annotations

import numpy as np

CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2, "alpha": 3}

NEW = {
    "EdgeBlur": {"edge_matte": "rgba.alpha", "edgeblur_tint": 1.0, "edgeblur_brightness": 1.0},
}
# What an older document's node of the kind gets on load: the knobs at the value that leaves it as it was, and where Nuke's default
# differs from the one NodeBased had, the old default (a saved node keeps its own size and edge multiplier).
UPGRADE = {
    "EdgeBlur": {"edge_matte": "rgba.alpha", "edgeblur_tint": 1.0, "edgeblur_brightness": 1.0, "edgeblur_size": 4.0, "edge_mult": 1.0},
}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(UPGRADE.get(kind, {}))


def channel_index(token, default=3):
    """The 0..3 channel a `rgba.red`-style token names; anything else (a layer NodeBased has no pixels for) reads `default`."""
    name = str(token or "").split(".")[-1].strip().lower()
    return CHANNEL_INDEX.get(name, default)


# ------------------------------------------------------------------------------------------------------------ EdgeBlur
def edge_halo(p):
    """The pixels each way a tile must hold for EdgeBlur: the blur's reach (with the quality scaling) or the band, whichever is wider."""
    import math
    from . import blurknobs
    size = abs(float(p.get("edgeblur_size", 0.0)))
    band = size * max(0.0, float(p.get("edge_mult", 1.0)))
    return max(blurknobs.halo("EdgeBlur", p), 0 if band < 0.5 else int(math.ceil(band)))


def edge_blur(image, p, origin=(0, 0)):
    """Nuke's EdgeBlur on `image`; `origin` is the array's top-left pixel on the canvas (the quality scaling's blocks sit on that grid)."""
    from . import blurknobs
    from .imaging import Evaluator
    size = abs(float(p.get("edgeblur_size", 4.0)))
    out = image.copy()
    band_width = size * max(0.0, float(p.get("edge_mult", 1.0)))
    if size < 0.5 or band_width < 0.5:
        return out
    matte = image[..., channel_index(p.get("edge_matte", "rgba.alpha")):][..., :1]
    band = np.clip(Evaluator._box_extreme(matte, band_width, use_max=True)
                   - Evaluator._box_extreme(matte, band_width, use_max=False), 0.0, 1.0)
    blurred = blurknobs.filtered(image, size, blurknobs._get("EdgeBlur", p, "kernel_filter"), blurknobs._get("EdgeBlur", p, "quality"), origin)
    tint, brightness = float(p.get("edgeblur_tint", 1.0)), float(p.get("edgeblur_brightness", 1.0))
    if tint != 1.0 or brightness != 1.0:
        blurred = blurred.copy()
        blurred[..., :3] *= tint
        blurred *= brightness
    for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
        out[..., c] = image[..., c] + band[..., 0] * (blurred[..., c] - image[..., c])
    return out.astype(np.float32)
