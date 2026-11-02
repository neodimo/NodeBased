"""Behaviour fixes to the Transform group (lane 9, 2D behaviour step B): Transform, Reformat, CornerPin, Crop, Mirror, Position, Tile,
AdjustBBox and BlackOutside.

Each fix keeps the older reading behind a `legacy_*` parameter. A parameter dict that predates the knob reads as an older document's
node (a missing flag is the legacy value), and `upgrade_params` gives a saved node the explicit legacy value on load, so an older
document renders bit for bit as it did. A node created now gets the new reading from its SPECS defaults.

* Transform and CornerPin, the window of black outside off (`legacy_transform_edges`, CornerPin's only edge flag): the outside is filled with
  the outermost pixels, which only shows if the output window reaches the format, so it now does. Older nodes kept the moved
  picture's own window, and the viewer showed nothing outside it.
* Transform, black outside / clamp (`legacy_transform_edges`): the guide says "black outside" renders black pixels outside the image
  and, unchecked, fills the outside with the outermost pixels; "clamp" corrects the haloing of sharpening filters. Older nodes
  read black outside as "paint the outside opaque black" (and left it transparent black when off), and clamp as "clamp the
  sample position to the picture" (an edge extension), which never limited a halo.
* Transform, axis directions (`legacy_transform_axes`): Nuke counts y upwards, so a positive translate y moves the picture up the screen and
  a positive rotate turns it counter-clockwise; skew follows the same axis. NodeBased draws row 0 at the top, so the matrix is the
  same one conjugated by the vertical flip. Older nodes moved the picture down and turned it clockwise. `center` y keeps counting
  rows from the top, as Crop's box does (docs/PARITY_2D.md), because the matrix is built without the format's height.
* Reformat (`legacy_reformat_layout`): with center off "the lower left corners line up" (the guide), and turn "rotates the image 90 degrees
  counter-clockwise". NodeBased's rows run downwards, so the lower left is the last row; older nodes lined the top left corners up
  and their turn was the transpose (a reflection), not a rotation. A Reformat to a box with force this shape off makes the output
  "approximately the same shape as the original" (`reformat_box_shape`); older nodes always made it exactly the box (`box_fixed` 1
  on load). Black outside and clamp are the Transform ones, on the same filters.
* Mirror (`legacy_mirror_window`): "flips the input image around the center of the format area". A picture whose data window is not the
  whole format is reflected to the other side of the format's centre, with its window; older nodes flipped it inside its own window.
* Position (`legacy_position`): "Positive values move the image up or to the right" and it "fills the area between the original image
  and the moved image by repeating the edge pixels", so the window is the union of the two and the picture is carried to it. Older
  nodes moved a positive Y down and left the area between empty.
* AdjustBBox (`legacy_adjustbbox`): the area a growing box adds is filled with the outermost pixels (the BlackOutside page is the
  guide's evidence: it "removes stripes or other anomalies caused by the edge pixels of the bounding box being replicated ...
  after adjusting the bounding box with an AdjustBBox"); older nodes left it empty. `numpixels` splits into a width and a height.
* Tile (no flag, the filter): each tile is the picture scaled to the tile's size with the node's `filter`, as the guide's list has it
  (cubic by default); an older Tile loads with `bilinear`, the sampling it always used.
* Transform and CornerPin, shutter offset (`legacy_shutter_window`): the guide opens the shutter at the current frame for `start`
  and closes it there for `end`. Older nodes had the two the other way round.
* The filter list: Impulse (the existing `nearest`), Cubic, Keys, Simon, Rifman, Mitchell, Parzen, Lanczos4, Lanczos6 and Sinc4.
  Notch is named by the guide but no published kernel for it is at hand, so it is not offered.

Pure NumPy, so the evaluator, the tile executor and the tests read one arithmetic.
"""
from __future__ import annotations

import numpy as np

LEGACY = {
    "Transform": {"legacy_transform_edges": 1, "legacy_shutter_window": 1, "legacy_transform_axes": 1},
    "CornerPin": {"legacy_shutter_window": 1, "legacy_transform_edges": 1},
    "Reformat": {"legacy_reformat_layout": 1},
    "Mirror": {"legacy_mirror_window": 1},
    "Position": {"legacy_position": 1},
    "AdjustBBox": {"legacy_adjustbbox": 1},
}
NEW = {
    "Transform": {"legacy_transform_edges": 0, "legacy_shutter_window": 0, "legacy_transform_axes": 0},
    "CornerPin": {"legacy_shutter_window": 0, "legacy_transform_edges": 0},
    "Reformat": {"legacy_reformat_layout": 0, "box_fixed": 0, "clamp": 0, "black_outside": 1},
    "Mirror": {"legacy_mirror_window": 0},
    "Position": {"legacy_position": 0},
    "AdjustBBox": {"legacy_adjustbbox": 0},
}
# What an older document's node gets on load besides the legacy flags: the new knobs, at the value that leaves it as it was.
# A Transform saved before black outside had a stored value read it as off (the default then was 0).
UPGRADE_EXTRA = {"Transform": {"black_outside": 0},
                 # A Reformat saved before the step made its to-box output exactly the box (force this shape), took nothing from black
                 # outside or clamp, and had no value clamp.
                 "Reformat": {"box_fixed": 1, "clamp": 0, "black_outside": 1},
                 # A Tile saved before it had a filter scaled each tile with bilinear sampling.
                 "Tile": {"filter": "bilinear"}}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What an older document's node of `kind` gets on load."""
    return {**LEGACY.get(kind, {}), **UPGRADE_EXTRA.get(kind, {})}


def screen_matrix_terms(p):
    """(rotate degrees, skew x, skew y, translate y) as the matrix wants them: Nuke's, with the vertical axis turned over for NodeBased's
    top-down rows unless the node reads the older way. A dict that does not carry the flag (Tracker and Stabilize share the matrix) is
    an older reading."""
    rotate, kx, ky, ty = (float(p.get("rotate", 0.0)), float(p.get("skew_x", 0.0)), float(p.get("skew_y", 0.0)),
                          float(p.get("translate_y", 0.0)))
    if int(p.get("legacy_transform_axes", 1)):
        return rotate, kx, ky, ty
    return -rotate, -kx, -ky, -ty


def reformat_box_shape(src_w, src_h, box_w, box_h, resize_type, turn):
    """The output size of a Reformat to a box with force this shape off: the box in the direction the resize type names, the other
    direction in the source's proportion, rounded to whole pixels. `none` and `distort` have nothing to keep in proportion: the box.
    The box is the output's, so a turned Reformat solves it against its working format (the box with width and height exchanged,
    as `Evaluator._reformat_place` does) and swaps the answer back."""
    sw, sh = src_w, src_h
    bw, bh = (box_h, box_w) if turn else (box_w, box_h)
    if resize_type in ("none", "distort") or not sw or not sh:
        ow, oh = bw, bh
    elif resize_type == "width":
        ow, oh = bw, sh * bw / sw
    elif resize_type == "height":
        ow, oh = sw * bh / sh, bh
    else:
        scale = min(bw / sw, bh / sh) if resize_type == "fit" else max(bw / sw, bh / sh)
        ow, oh = sw * scale, sh * scale
    ow, oh = max(1, int(round(ow))), max(1, int(round(oh)))
    return (oh, ow) if turn else (ow, oh)


def is_legacy(kind, p, name):
    """True when `p` reads as an older document's node for the flag `name` (a missing flag is an older node)."""
    return bool(int(p.get(name, LEGACY[kind][name])))


# --------------------------------------------------------------------------------------------------------------------- filters
# Published reconstruction kernels. `a` is Keys' parameter (cubic convolution): Keys 1981 uses -0.5, Rifman 1973 uses -1; Simon
# sits between them (the guide orders the three by how much they sharpen). Mitchell is the Mitchell-Netravali B = C = 1/3 filter and
# Parzen the cubic B-spline (B = 1, C = 0), the smoothest of the cubic family. Lanczos and sinc are the windowed and the truncated
# sinc. The guide shows the curves as pictures and states no coefficients, so the exact weights are not scored; the facts it does
# state (smoothing against sharpening, impulse carries the original values) are tested.
KEYS_A = {"keys": -0.5, "simon": -0.75, "rifman": -1.0}


def _keys(a):
    def weight(x):
        x = np.abs(x)
        x2, x3 = x * x, x * x * x
        return np.where(x <= 1.0, (a + 2) * x3 - (a + 3) * x2 + 1,
                        np.where(x < 2.0, a * x3 - 5 * a * x2 + 8 * a * x - 4 * a, 0.0))
    return weight


def _mitchell(b, c):
    def weight(x):
        x = np.abs(x)
        x2, x3 = x * x, x * x * x
        inner = ((12 - 9 * b - 6 * c) * x3 + (-18 + 12 * b + 6 * c) * x2 + (6 - 2 * b)) / 6.0
        outer = ((-b - 6 * c) * x3 + (6 * b + 30 * c) * x2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6.0
        return np.where(x < 1.0, inner, np.where(x < 2.0, outer, 0.0))
    return weight


def _lanczos(radius):
    def weight(x):
        return np.where(np.abs(x) < radius, np.sinc(x) * np.sinc(x / radius), 0.0)
    return weight


def _sinc(radius):
    def weight(x):
        return np.where(np.abs(x) < radius, np.sinc(x), 0.0)
    return weight


KERNELS = {
    "keys": (_keys(KEYS_A["keys"]), 2), "simon": (_keys(KEYS_A["simon"]), 2), "rifman": (_keys(KEYS_A["rifman"]), 2),
    "mitchell": (_mitchell(1 / 3.0, 1 / 3.0), 2), "parzen": (_mitchell(1.0, 0.0), 2),
    "lanczos4": (_lanczos(2), 2), "lanczos6": (_lanczos(3), 3), "sinc4": (_sinc(2), 2),
}
FILTERS = tuple(KERNELS)
# The choice list of the `filter` knob: the three older names first (their kernels are untouched), then the guide's others.
CHOICES = ("nearest", "bilinear", "cubic", "impulse", *FILTERS)
# Pixels each way a filter reaches (the data window grows by this much).
SUPPORT = {"nearest": 1, "impulse": 1, "bilinear": 1, "cubic": 2, **{name: radius for name, (_fn, radius) in KERNELS.items()}}


def support(name):
    return SUPPORT.get(name, 2)


def resample(src, sx, sy, name, edge=False):
    """`src` sampled at the fractional pixel positions (`sx`, `sy`) (pixel i is at i) with the separable kernel `name`.

    The weights of the taps are divided by their sum, so a constant picture stays that constant at every position. Taps outside
    the picture read zero (black outside), or, with `edge`, the nearest edge pixel (the outermost pixels carried outwards)."""
    weight, radius = KERNELS[name]
    h, w = src.shape[:2]
    sx = np.asarray(sx, np.float64)
    sy = np.asarray(sy, np.float64)
    x0, y0 = np.floor(sx), np.floor(sy)
    fx, fy = sx - x0, sy - y0
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    offsets = range(1 - radius, radius + 1)
    wx = np.stack([weight(fx - o) for o in offsets], axis=-1)
    wy = np.stack([weight(fy - o) for o in offsets], axis=-1)
    wx = wx / wx.sum(axis=-1, keepdims=True)
    wy = wy / wy.sum(axis=-1, keepdims=True)
    source = src.astype(np.float64)
    total = np.zeros(sx.shape + (src.shape[2],), np.float64)
    zero = np.float64(0.0)
    for j, oy in enumerate(offsets):
        yi = y0 + oy
        for i, ox in enumerate(offsets):
            xi = x0 + ox
            if edge:
                tap = source[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
            else:
                inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                tap = np.where(inside[..., None], source[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], zero)
            total += (wy[..., j] * wx[..., i])[..., None] * tap
    return total.astype(np.float32)


def value_clamp(result, src):
    """Nuke's `clamp`: a filter with negative lobes cannot overshoot the range of the input, channel by channel."""
    if not src.size:
        return result
    return np.clip(result, src.min(axis=(0, 1)), src.max(axis=(0, 1))).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------ shutter
def shutter_window(frame, shutter, offset, custom, legacy):
    """The frames the shutter is open between. The guide: `start` opens it at the current frame, `end` closes it there, `custom`
    opens it at the current frame plus the offset.

    Older nodes (and the other nodes that share this branch) had `start` ending at the frame, `end` starting there and `custom`
    centred on the offset."""
    shutter = float(shutter)
    if offset == "start":
        return (frame - shutter, frame) if legacy else (frame, frame + shutter)
    if offset == "end":
        return (frame, frame + shutter) if legacy else (frame - shutter, frame)
    if offset == "custom":
        if legacy:
            return frame + float(custom) - shutter / 2, frame + float(custom) + shutter / 2
        return frame + float(custom), frame + float(custom) + shutter
    return frame - shutter / 2, frame + shutter / 2
