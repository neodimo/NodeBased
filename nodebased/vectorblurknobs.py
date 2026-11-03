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

Not built, and listed in PARITY_2D_FIDELITY.md: `mv_presets` (the guide names the presets' purpose, "a scale and offset specific to the
renderer", and gives no values) and `normalize` (it compensates for tearing and collisions of a scatter, "blurring each pixel into a
straight line"; NodeBased's blur gathers along each pixel's vector, which divides by the samples taken and so has neither).

A saved node reads the older blur (`vector_blur_type` uniform, no preprocessing, no soft lines), so it renders as it did.
"""
from __future__ import annotations

import numpy as np

BLUR_UV = ("none", "uniform", "linear")
BLUR_TYPES = ("uniform", "gaussian")
NEW = {"blur_uv": "none", "uv_offset": 0.0, "vector_blur_type": "gaussian", "motion_falloff": 0.33, "grow_bbox": 0, "soft_lines": 0,
       "blur_inside": 1}
UPGRADE = {**NEW, "vector_blur_type": "uniform"}
LIMITS = {"uv_offset": (-1000.0, 1000.0), "motion_falloff": (0.0, 1.0), "grow_bbox": (0, 4096), "soft_lines": (0, 1), "blur_inside": (0, 1)}
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
