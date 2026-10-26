"""Nuke's Glow knobs of fidelity step H: `saturation`, `width_channel` (Nuke's `W`), `nonlinear`, `effect_only`, and the blur's filter shape
(`kernel_filter`, Nuke's `filter`), `quality` and `crop` (crop to format), which are Blur's own (nodebased/blurknobs.py, whose
shapes, quality scaling and window growth Glow reuses).

* `saturation` (default 1): the glow's colour is moved toward or away from its own luminance (`0.2126 r + 0.7152 g + 0.0722 b`):
  0 is a grey glow, 1 the glow as it was, above 1 a more colourful one.
* `nonlinear` (off): the part above the tolerance is encoded to the sRGB video curve (odd about zero, so negative and high values keep
  their order) before it is blurred and decoded after it. The guide gives no formula; this reading is NodeBased's.
* `effect_only` (off): the selected colour channels carry the glow alone instead of the glow added to the input; the alpha is the
  input's. The glow can then be added to other footage.
* `width_channel` (default none): a channel (`rgba.alpha`, `rgba.red`, `rgba.luminance`, or a layer channel such as `mask.a`) whose value, held
  to 0 .. 1, scales `glow_size` pixel by pixel: 0 gives the highlight with no spread, 1 the full size. It is drawn from nine blurs at 0,
  1/8 ... 1 of the size, mixed linearly, so a tile and the whole picture agree. The guide gives no more than the knob's name; that
  reading is NodeBased's.
* Older documents keep a box blur with no quality limit, as Glow always had; a new node starts with Nuke's gaussian at quality 15.
"""
from __future__ import annotations

import numpy as np

LEVELS = 8


def saturate(rgb, amount):
    """`rgb` (h, w, 3) moved away from its luminance by `amount` (1 leaves it as it is)."""
    amount = float(amount)
    if amount == 1.0:
        return rgb
    luma = (rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152 + rgb[..., 2:3] * 0.0722).astype(np.float32)
    return (luma + (rgb - luma) * np.float32(amount)).astype(np.float32)


def blurred(image, size, shape, quality, origin, width=None):
    """`image` blurred by `size`; with a `width` channel the size varies per pixel between nine fixed levels."""
    from . import blurknobs
    if width is None:
        return blurknobs.filtered(image, size, shape, quality, origin)
    weight = np.clip(np.asarray(width, np.float32), 0.0, 1.0) * LEVELS
    low = np.minimum(np.floor(weight).astype(int), LEVELS - 1)
    fraction = (weight - low).astype(np.float32)
    levels = [blurknobs.filtered(image, size * k / LEVELS, shape, quality, origin) for k in range(LEVELS + 1)]
    stack = np.stack(levels, axis=0)
    index = low[..., 0]
    rows, cols = np.indices(index.shape)
    return (stack[index, rows, cols] * (1.0 - fraction) + stack[index + 1, rows, cols] * fraction).astype(np.float32)
