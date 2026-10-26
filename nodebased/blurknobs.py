"""Nuke's knobs on Blur, Sharpen and Soften (fidelity step E): the filter shape, the quality limit, crop to format, and on the
latter two the value range and Soften's amount.

The three nodes share one blur. `size` (Blur's `radius`, Sharpen's `sharpen_size`, Soften's `soften_size`) is the filter's reach
in pixels, and `filter` picks the kernel over that reach:

* box: every pixel within `round(size)` weighs the same (the blur NodeBased has always had).
* triangle: a tent, weight `1 - |k| / size`.
* quadratic: a quadratic B-spline spread over `size` (weight `0.75 - u^2` for `|u| <= 0.5`, `0.5 (1.5 - |u|)^2` up to 1.5, with
  `u = 1.5 k / size`).
* gaussian: sigma = `size / 3`, cut off at `ceil(size)` pixels (three sigma), which is how Soften has always been built.

Every kernel is separable, normalised to sum to 1, and reads edge-extended pixels, like the box blur. `quality` is the Nuke
guide's limit on the filter's size: when the reach exceeds it the picture is scaled down by a whole number of pixels so the
filter is no larger than `quality`, filtered, and scaled up again with linear interpolation. The blocks of that scaling sit on the
absolute pixel grid (a block is `factor` pixels wide from pixel 0 of the canvas), so a tile that holds `halo(...)` pixels around the
ones it wants computes exactly what the whole picture does. `crop` ("crop to format") on keeps the picture's data window as it comes;
off lets the window grow by the filter's reach, the new edge being transparent black.

Sharpen is the unsharp mask, `image + amount * (image - blur)`; Soften is `image + amount * (blur - image)`. For both,
`minimum` and `maximum` confine the effect to pixel values inside that range (a channel value outside it passes unchanged); the
guide gives the two knobs no description, so that reading is ours.

An older document's node keeps what it did: Blur and Sharpen a box blur, no quality limit, no value range. Pure NumPy, so the
evaluator, the tile executor and the tests read one arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

KINDS = ("Blur", "Sharpen", "Soften")
FILTERS = ("box", "triangle", "quadratic", "gaussian")
SIZE_PARAM = {"Blur": "radius", "Sharpen": "sharpen_size", "Soften": "soften_size"}
WIDE = 1000000.0
NO_QUALITY_LIMIT = 1000.0       # larger than any size the nodes accept (500), so an older node never scales the picture

NEW = {
    "Blur": {"kernel_filter": "gaussian", "quality": 15.0, "crop": 1},
    "Sharpen": {"kernel_filter": "gaussian", "quality": 15.0, "crop": 1, "minimum": 0.0, "maximum": 1.0},
    "Soften": {"kernel_filter": "gaussian", "quality": 15.0, "crop": 1, "minimum": 0.0, "maximum": 1.0, "amount": 1.0},
}
OLD = {
    "Blur": {"kernel_filter": "box", "quality": NO_QUALITY_LIMIT, "crop": 1},
    "Sharpen": {"kernel_filter": "box", "quality": NO_QUALITY_LIMIT, "crop": 1, "minimum": -WIDE, "maximum": WIDE},
    "Soften": {"kernel_filter": "gaussian", "quality": NO_QUALITY_LIMIT, "crop": 1, "minimum": -WIDE, "maximum": WIDE, "amount": 1.0},
}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(OLD.get(kind, {}))


def _get(kind, p, name):
    """A knob of `p`; a parameter dict that predates the knob reads as an older document's node."""
    return p.get(name, OLD[kind][name])


def reach(kind, p):
    """How many pixels each way the filter reads: 0 below half a pixel (the filter is the identity there), else `ceil(size)`."""
    size = abs(float(p.get(SIZE_PARAM[kind], 0.0)))
    return 0 if size < 0.5 else int(math.ceil(size))


def scale_factor(kind, p):
    """The whole-number factor `quality` scales the picture down by, or 1 when the filter already fits within it."""
    r = reach(kind, p)
    quality = float(_get(kind, p, "quality"))
    return 1 if r <= quality else int(math.ceil(r / quality))


def halo(kind, p):
    """The pixels each way a tile must hold around the ones it computes: the reach, or when `quality` scales the picture down,
    the scaled filter's reach plus two blocks (one for the partly filled block at the tile's edge, one for the interpolation)."""
    r = reach(kind, p)
    f = scale_factor(kind, p)
    if r == 0 or f == 1:
        return r
    return f * (int(math.ceil(abs(float(p.get(SIZE_PARAM[kind], 0.0))) / f)) + 2)


def needs_full_frame(kind, p):
    """True for what tiles cannot do: a data window that grows (crop off)."""
    return reach(kind, p) > 0 and not _get(kind, p, "crop")


def grown_window(kind, p, data):
    """The data window of the result with `crop` off: the input's window plus the filter's reach on every side."""
    r = reach(kind, p)
    return data.expand(r, r) if r else data


# ------------------------------------------------------------------------------------------------------------ the kernels
def weights(shape, size):
    """The 1-D kernel (float64, sums to 1) over integer offsets `-reach .. reach` for `shape` triangle, quadratic or gaussian."""
    size = abs(float(size))
    r = int(math.ceil(size))
    k = np.arange(-r, r + 1, dtype=np.float64)
    if shape == "triangle":
        w = np.maximum(0.0, 1.0 - np.abs(k) / size)
    elif shape == "quadratic":
        u = np.abs(1.5 * k / size)
        w = np.where(u <= 0.5, 0.75 - u * u, np.where(u < 1.5, 0.5 * (1.5 - u) ** 2, 0.0))
    elif shape == "gaussian":
        w = np.exp(-0.5 * (k / (size / 3.0)) ** 2)
    else:
        raise ValueError(f"Unknown blur filter {shape!r}; choose one of {', '.join(FILTERS)}")
    return w / w.sum()


def _weighted_axis(frame, w, axis):
    """A separable pass with the kernel `w` and edge padding, accumulated in float64."""
    radius = (len(w) - 1) // 2
    n = frame.shape[axis]
    pad = [(0, 0)] * frame.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(frame, pad, mode="edge").astype(np.float64)
    out = np.zeros(frame.shape, dtype=np.float64)
    for i, weight in enumerate(w):
        index = [slice(None)] * frame.ndim
        index[axis] = slice(i, i + n)
        out += weight * padded[tuple(index)]
    return out.astype(np.float32)


def _filtered_direct(image, size, shape):
    """The blur at `size` with no quality scaling."""
    from .imaging import Evaluator
    if float(size) < 0.5 and shape != "box":
        return image.copy()
    if shape == "box":
        return Evaluator._box_blur_axis(Evaluator._box_blur_axis(image, size, axis=1), size, axis=0) if size >= 0.5 else image.copy()
    r = int(math.ceil(abs(float(size))))
    if shape == "gaussian":
        sigma = abs(float(size)) / 3.0
        return Evaluator._gaussian_axis(Evaluator._gaussian_axis(image, r, sigma, axis=1), r, sigma, axis=0)
    w = weights(shape, size)
    return _weighted_axis(_weighted_axis(image, w, axis=1), w, axis=0)


def _upsample(small, factor, height, width, lead_y=0, lead_x=0):
    """`small` scaled up by `factor` with linear interpolation between pixel centres, cropped to (height, width) after the
    `lead_y` / `lead_x` pixels the first block starts before the picture."""
    def axis_terms(n_out, n_in, lead):
        src = np.clip((np.arange(n_out, dtype=np.float64) + lead + 0.5) / factor - 0.5, 0.0, n_in - 1.0)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, n_in - 1)
        return lo, hi, (src - lo).astype(np.float32)
    y0, y1, fy = axis_terms(height, small.shape[0], lead_y)
    x0, x1, fx = axis_terms(width, small.shape[1], lead_x)
    rows = small[y0] * (1 - fy)[:, None, None] + small[y1] * fy[:, None, None]
    return (rows[:, x0] * (1 - fx)[None, :, None] + rows[:, x1] * fx[None, :, None]).astype(np.float32)


def filtered(image, size, shape="box", quality=NO_QUALITY_LIMIT, origin=(0, 0)):
    """`image` blurred by `size` with filter `shape`. Below half a pixel it is the identity; when the filter would be wider than
    `quality` the picture is averaged down by a whole factor, filtered at the smaller size and scaled back up with linear
    interpolation (the guide's description of `quality`). `origin` is the picture's top-left pixel on the canvas: the blocks of
    the scaling start on multiples of the factor from canvas pixel 0, so a tile and the whole picture agree."""
    size = abs(float(size))
    if size < 0.5:
        return image.copy()
    r = int(math.ceil(size))
    quality = float(quality)
    if r > quality:
        factor = int(math.ceil(r / quality))
        height, width = image.shape[:2]
        lead_y, lead_x = int(origin[1]) % factor, int(origin[0]) % factor
        pad_y, pad_x = (-(lead_y + height)) % factor, (-(lead_x + width)) % factor
        padded = np.pad(image, ((lead_y, pad_y), (lead_x, pad_x), (0, 0)), mode="edge") if (lead_y or lead_x or pad_y or pad_x) else image
        h2, w2 = padded.shape[0] // factor, padded.shape[1] // factor
        small = padded.reshape(h2, factor, w2, factor, image.shape[2]).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
        blurred = _filtered_direct(small, size / factor, shape)
        return _upsample(blurred, factor, height, width, lead_y, lead_x)
    return _filtered_direct(image, size, shape)


# ---------------------------------------------------------------------------------------------------------- the nodes
def blur(image, p, origin=(0, 0)):
    """Blur's kernel; `origin` is the picture's top-left canvas pixel."""
    return filtered(image, p["radius"], _get("Blur", p, "kernel_filter"), _get("Blur", p, "quality"), origin)


def _confined(image, result, channel, lo, hi):
    """`result` for the channel values of `image` inside [lo, hi], `image` elsewhere."""
    if lo <= -WIDE and hi >= WIDE:
        return result
    value = image[..., channel]
    return np.where((value >= lo) & (value <= hi), result, value).astype(np.float32)


def sharpen(image, p, channel_sets, origin=(0, 0)):
    """Sharpen's kernel (the unsharp mask): `channel_sets` maps the channels knob to the channel indices it names."""
    amount = p.get("sharpen_amount", 0.5)
    blurred = filtered(image, p.get("sharpen_size", 1.0), _get("Sharpen", p, "kernel_filter"), _get("Sharpen", p, "quality"), origin)
    detail = image - blurred
    lo, hi = float(_get("Sharpen", p, "minimum")), float(_get("Sharpen", p, "maximum"))
    out = image.copy()
    for c in channel_sets[p.get("channels", "rgb")]:
        out[..., c] = _confined(image, image[..., c] + detail[..., c] * amount, c, lo, hi)
    return out


def soften(image, p, channel_sets, origin=(0, 0)):
    """Soften's kernel: the blur, `amount` of the way from the picture to it."""
    size = abs(float(p.get("soften_size", 4.0)))
    if size < 0.5:
        return image.copy()
    soft = filtered(image, size, _get("Soften", p, "kernel_filter"), _get("Soften", p, "quality"), origin)
    amount = float(_get("Soften", p, "amount"))
    lo, hi = float(_get("Soften", p, "minimum")), float(_get("Soften", p, "maximum"))
    out = image.copy()
    for c in channel_sets[p.get("channels", "rgba")]:
        mixed = soft[..., c] if amount == 1.0 else image[..., c] + amount * (soft[..., c] - image[..., c])
        out[..., c] = _confined(image, mixed, c, lo, hi)
    return out
