"""VectorBlur's preprocessing and blur knobs (lane 9, 2D behaviour step L finish), as the Nuke 17.0 Reference Guide states them.

* uv offset (`uv_offset`): "a constant value to add to the uv channels values" (so a vector file read raw takes -0.5). Added to u and v
  before anything else.
* blur uv (`blur_uv`): "applies the motion blur filter to the motion vectors themselves, before using them to blur the image ...
  smooth out gradients in the motion vectors". none leaves them; uniform is "a small uniform blur" (here a 3 x 3 average); linear is "a
  linearly-weighted blur ... a stronger effect than the uniform option" (here a tent over 9 x 9 pixels). The sizes are ours.
* blur type (`vector_blur_type`) and motion falloff (`motion_falloff`): "the profile to use along the direction of blur: uniform - a box-like
  profile with uniform weighting along the blur vector; gaussian - a gaussian profile with a softer fall-off towards the ends". The
  falloff "controls the steepness of the gaussian fall-off ... 0 gives a very sharp fall-off, while 1 gives a shallow fall-off": the
  weight at a distance x along the line (the line is one unit long, x in -0.5 .. 0.5 from its middle) is exp(-x^2 / (2 s^2)) with
  s = max(0.001, falloff / 2), so a falloff of 1 leaves the ends at 61% of the middle. The mapping from falloff to s is ours.
* expand blur bbox (`grow_bbox`): "the number of pixels added to the outside edge of image ... for cropped images with non-zero uv
  channels". The picture's window grows by that many pixels on each side; the vectors of the new ring are the nearest edge vectors.
* soft lines (`soft_lines`) and blur inside (`blur_inside`): "increase the softness of the motion blur by drawing wider lines"; here each
  sample is also taken one pixel to either side of the line (weights 1/4, 1/2, 1/4). "Apply a small blur inside regions with zero
  motion": with soft lines on, a pixel whose vector is zero takes the 3 x 3 average of its neighbours.

* normalize (`vector_normalize`, step X6): "normalize the results to compensate for tearing (holes appearing behind moving objects) and
  collisions (objects moving on top of one another) in the rendered output ... it tends to give a softer blur than not normalizing". The
  node blurs "each pixel into a straight line", which is a scatter. On (the guide's default) is the blur NodeBased always did, which divides
  by the weight taken at each pixel. Off draws each source pixel's line with its energy shared along it (the weights of one line sum to 1)
  and does not divide by the weight that arrives: a hole behind a moving object stays dark and transparent, and where lines pile up the
  picture is brighter (`scatter`). The guide gives no formula, so the off result is one reading of those two sentences, not a measured match;
  it ignores `vector_alpha` weighting and the vector location setting, which are gather refinements.

Not built, and listed in PARITY_2D_FIDELITY.md: `mv_presets` (the guide names the presets' purpose, "a scale and offset specific to the
renderer", and gives no values).

A saved node reads the older blur (`vector_blur_type` uniform, no preprocessing, no soft lines), so it renders as it did.
"""
from __future__ import annotations

import numpy as np

BLUR_UV = ("none", "uniform", "linear")
BLUR_TYPES = ("uniform", "gaussian")
NEW = {"blur_uv": "none", "uv_offset": 0.0, "vector_blur_type": "gaussian", "motion_falloff": 0.33, "grow_bbox": 0, "soft_lines": 0,
       "blur_inside": 1, "vector_normalize": 1}
UPGRADE = {**NEW, "vector_blur_type": "uniform"}
LIMITS = {"uv_offset": (-1000.0, 1000.0), "motion_falloff": (0.0, 1.0), "grow_bbox": (0, 4096), "soft_lines": (0, 1), "blur_inside": (0, 1), "vector_normalize": (0, 1)}
CHOICES = {"blur_uv": list(BLUR_UV), "vector_blur_type": list(BLUR_TYPES)}
TENT_RADIUS = 4


def _separable(plane, weights):
    radius = len(weights) // 2
    padded = np.pad(plane.astype(np.float64), radius, mode="edge")
    h, w = plane.shape
    rows = sum(wt * padded[:, i:i + w] for i, wt in enumerate(weights))
    return sum(wt * rows[i:i + h] for i, wt in enumerate(weights))


def blur_uvs(u, v, mode):
    """`blur uv`: the vectors blurred before they blur the picture."""
    mode = str(mode)
    if mode == "uniform":
        weights = np.full(3, 1.0 / 3.0)
    elif mode == "linear":
        ramp = np.array([TENT_RADIUS + 1 - abs(i) for i in range(-TENT_RADIUS, TENT_RADIUS + 1)], np.float64)
        weights = ramp / ramp.sum()
    else:
        return u, v
    return _separable(u, weights).astype(u.dtype), _separable(v, weights).astype(v.dtype)


def preprocess(u, v, p):
    """uv offset, then blur uv (invert uv and the scale follow, in the node)."""
    offset = float(p.get("uv_offset", 0.0))
    if offset:
        u, v = u + offset, v + offset
    return blur_uvs(u, v, p.get("blur_uv", "none"))


def line_weight(t, offset, p):
    """The weight of a sample at position `t` along the line that runs from `offset` to `offset + 1`: 1 for uniform, a gaussian
    centred on the middle of the line for gaussian. `t` may be an array."""
    if str(p.get("vector_blur_type", "uniform")) != "gaussian":
        return None
    sigma = max(0.001, float(p.get("motion_falloff", 0.33)) / 2.0)
    x = t - (offset + 0.5)
    return np.exp(-(x * x) / (2.0 * sigma * sigma))


def box3(image):
    padded = np.pad(image.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = image.shape[:2]
    return (sum(padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9.0).astype(np.float32)


def scatter(src, u, v, p):
    """VectorBlur with normalize off: each source pixel is drawn along its own vector instead of each pixel reading along its vector.

    The vector is (u, v) * vector_scale capped at max_length, as in the gather; the line runs from vector_offset to vector_offset + 1 of the
    vector, forward (the streak follows the motion) or backward, one sample about every pixel (or `samples` of them). The weights of one
    line (uniform, or the gaussian of blur type and motion falloff; soft lines adds a quarter of the weight a pixel to either side) sum to 1,
    and each sample is drawn on its four neighbouring pixels by bilinear weights. Nothing divides by what arrives, so a hole stays dark and
    transparent and a pile-up is brighter.
    """
    vx = u.astype(np.float64) * float(p["vector_scale"])
    vy = v.astype(np.float64) * float(p["vector_scale"])
    length = np.hypot(vx, vy)
    cap = float(p.get("max_length", 0.0))
    if cap > 0:
        shrink = np.where(length > cap, cap / np.maximum(length, 1e-12), 1.0)
        vx, vy, length = vx * shrink, vy * shrink, np.minimum(length, cap)
    steps = np.maximum(1, np.ceil(length - 1e-9)).astype(np.int32)
    sign = 1.0 if p["vector_method"] == "forward" else -1.0
    height, width = src.shape[:2]
    ix = np.arange(width, dtype=np.float64)[None, :] * np.ones((height, 1))
    iy = np.arange(height, dtype=np.float64)[:, None] * np.ones((1, width))
    offset = float(p["vector_offset"])
    fixed = int(p["samples"]) if p.get("samples") else 0
    count = fixed if fixed else int(steps.max()) + 1
    soft = bool(int(p.get("soft_lines", 0)))
    across_x, across_y = -vy / np.maximum(length, 1e-12), vx / np.maximum(length, 1e-12)
    pixels = src.astype(np.float64)

    def sample_at(k):
        if fixed:
            live = np.ones((height, width), bool)
            t = np.full((height, width), offset + (0.5 if fixed == 1 else k / (fixed - 1)))
        else:
            live = k <= steps
            t = offset + k / steps
        profile = line_weight(t, offset, p)
        share = live.astype(np.float64) * (1.0 if profile is None else np.asarray(profile, np.float64))
        return t, share

    line_total = np.zeros((height, width))
    for k in range(count):
        line_total += sample_at(k)[1]
    line_total = np.maximum(line_total, 1e-12)
    flat = np.zeros((4, height * width))
    taps = ((0.0, 1.0),) if not soft else ((0.0, 0.5), (-1.0, 0.25), (1.0, 0.25))

    def draw(px, py, weight):
        x0, y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        fx, fy = px - x0, py - y0
        for dx, dy, bilinear in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            cx, cy = x0 + dx, y0 + dy
            inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height) & (weight > 0)
            if not inside.any():
                continue
            index = (cy * width + cx)[inside]
            w = (weight * bilinear)[inside]
            for c in range(4):
                flat[c] += np.bincount(index, weights=w * pixels[..., c][inside], minlength=height * width)

    for k in range(count):
        t, share = sample_at(k)
        px, py = ix + sign * t * vx, iy + sign * t * vy
        for side, tap in taps:
            draw(px + side * across_x, py + side * across_y, share / line_total * tap)
    out = flat.reshape(4, height, width).transpose(1, 2, 0)
    if soft and int(p.get("blur_inside", 1)):
        out = np.where((length <= 1e-9)[..., None], box3(src), out)
    return out.astype(np.float32)
