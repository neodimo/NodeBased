"""MotionBlur's estimation knobs and its BgVecs and Matte inputs (lane 9, 2D behaviour step L finish).

The Nuke 17.0 Reference Guide gives MotionBlur these controls besides Shutter Samples and Shutter Time:

* Method (`motionEstimation`): "Local - uses local block matching to estimate motion vectors. This method is faster to process, but
  can lead to artifacts in the output. Regularized - uses semi-global motion estimation to produce more consistent vectors between
  regions." Here Local is the window match alone (`opticalflow.flow_pair` with smoothness 0) and Regularized adds the window match's
  usual smoothing and then an edge-aware pass that averages each vector with its neighbours of similar colour, so vectors agree within
  a region and are free to differ across its edges. "Scripts loaded from previous versions of Nuke default to Local ... a new
  MotionBlur node defaults to Regularized": a saved node here reads as `legacy_motionblur` (the older estimate, which this module's
  Local and Regularized do not reproduce) and a new one as Regularized.
* Vector Detail (`vectorDetail`, 0.2): "Varies the density of the vector field. ... A value of 1 generates a vector at each pixel,
  whereas a value of 0.5 generates a vector at every other pixel." The two frames are averaged down by round(1 / detail), the vectors
  are found at that size and scaled and interpolated back up to every pixel.
* Resampling (`resampleType`): Bilinear, Lanczos4 or Lanczos6 sample the picture along each vector (the kernels of
  `transformbehaviour`; the Lanczos filters "provide some image sharpening").
* Matte Channel (`matteChannel`) and the `Matte` input: "an optional matte of the foreground, which may improve the motion estimation by
  reducing the dragging of pixels that can occur between foreground and background objects." With the vectors estimated here, the matte
  keeps the foreground's vectors from being averaged with the background's across its edge (normalised averages inside the matte and
  outside it, joined by the matte). The guide gives no formula; this is ours.
* `BgVecs`: "If you have separate vectors for the background and foreground, you should connect them to the appropriate inputs and
  supply the matte that was used to generate them to the Matte input." Each pixel's vector is then matte * FgVecs + (1 - matte) * BgVecs
  (without a matte, or with only one set, that set is used).

The matte is the current frame's: the guide's `Source Alpha` and `Source Inverted Alpha` read the Source, `Matte Luminance`, `Matte Alpha`
and their inverses read the Matte input.
"""
from __future__ import annotations

import numpy as np

METHODS = ("Local", "Regularized")
RESAMPLING = ("Bilinear", "Lanczos4", "Lanczos6")
MATTE_CHANNELS = ("None", "Source Alpha", "Source Inverted Alpha", "Matte Luminance", "Matte Inverted Luminance", "Matte Alpha",
                  "Matte Inverted Alpha")
NEW = {"legacy_motionblur": 0, "motion_method": "Regularized", "motion_vector_detail": 0.2, "motion_resampling": "Bilinear",
       "motion_matte_channel": "None"}
# A saved node keeps the older estimate (a four-level window match at smoothness 1, vectors at every pixel).
UPGRADE = {"legacy_motionblur": 1, "motion_method": "Local", "motion_vector_detail": 1.0, "motion_resampling": "Bilinear",
           "motion_matte_channel": "None"}
LIMITS = {"motion_vector_detail": (0.02, 1.0), "legacy_motionblur": (0, 1)}
CHOICES = {"motion_method": list(METHODS), "motion_resampling": list(RESAMPLING), "motion_matte_channel": list(MATTE_CHANNELS)}


def is_legacy(p):
    return bool(int(p.get("legacy_motionblur", 1)))


def resample_name(p):
    return str(p.get("motion_resampling", "Bilinear")).lower()


def matte_from_channel(token, source, matte):
    """The (h, w) matte the Matte Channel names, or None when it is None or names an input that is not wired."""
    token = str(token)
    if token == "None":
        return None
    if token.startswith("Source"):
        plane = np.asarray(source, np.float32)[..., 3]
    else:
        if matte is None:
            return None
        pixels = np.asarray(matte, np.float32)
        plane = pixels[..., 3] if token.endswith("Alpha") else (
            pixels[..., 0] * 0.2126 + pixels[..., 1] * 0.7152 + pixels[..., 2] * 0.0722)
    plane = np.clip(plane, 0.0, 1.0)
    return 1.0 - plane if "Inverted" in token else plane


def combine_vectors(fg, bg, matte):
    """FgVecs where the matte is 1, BgVecs where it is 0, mixed between (the guide's separate-vectors case)."""
    if bg is None or matte is None:
        return fg
    m = np.asarray(matte, np.float32)[..., None]
    return (fg * m + bg * (1.0 - m)).astype(np.float32)


def _down(a, step):
    h, w = a.shape[:2]
    ph, pw = (-h) % step, (-w) % step
    padded = np.pad(a, ((0, ph), (0, pw)) + (((0, 0),) if a.ndim == 3 else ()), mode="edge")
    shape = (padded.shape[0] // step, step, padded.shape[1] // step, step) + (() if a.ndim == 2 else (a.shape[2],))
    return padded.reshape(shape).mean(axis=(1, 3)).astype(np.float32)


def _up(field, step, height, width):
    from .opticalflow import _sample
    if step == 1:
        return field
    xs = np.clip((np.arange(width, dtype=np.float32) + 0.5) / step - 0.5, 0, field.shape[1] - 1)
    ys = np.clip((np.arange(height, dtype=np.float32) + 0.5) / step - 0.5, 0, field.shape[0] - 1)
    return (_sample(field, np.broadcast_to(xs, (height, width)), np.broadcast_to(ys[:, None], (height, width))) * step).astype(np.float32)


def _regularize(flow, guide, iterations=3, radius=2, sigma=0.08):
    """Average each vector with the neighbours whose guide value is close to its own (an edge-aware, semi-global smoothing)."""
    h, w = guide.shape
    gp = np.pad(guide, radius, mode="edge")
    for _ in range(iterations):
        fp = np.pad(flow, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        num = np.zeros_like(flow, dtype=np.float64)
        den = np.zeros((h, w), np.float64)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                g = gp[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
                weight = np.exp(-((g - guide) ** 2) / (2.0 * sigma * sigma))
                num += fp[radius + dy:radius + dy + h, radius + dx:radius + dx + w] * weight[..., None]
                den += weight
        flow = (num / np.maximum(den, 1e-12)[..., None]).astype(np.float32)
    return flow


def _box_mean(plane, radius):
    from .opticalflow import _box_sum
    return _box_sum(plane.astype(np.float64), radius) / float((2 * radius + 1) ** 2)


def split_by_matte(flow, matte, radius=3, iterations=2):
    """Average the vectors inside the matte with each other and the vectors outside it with each other, so neither drags the other."""
    m = np.clip(matte.astype(np.float64), 0.0, 1.0)
    for _ in range(iterations):
        inside, outside = _box_mean(m, radius), _box_mean(1.0 - m, radius)
        fg = np.stack([_box_mean(flow[..., c] * m, radius) for c in range(2)], axis=-1) / np.maximum(inside, 1e-6)[..., None]
        bg = np.stack([_box_mean(flow[..., c] * (1.0 - m), radius) for c in range(2)], axis=-1) / np.maximum(outside, 1e-6)[..., None]
        fg = np.where((inside > 1e-3)[..., None], fg, flow)
        bg = np.where((outside > 1e-3)[..., None], bg, flow)
        flow = (fg * m[..., None] + bg * (1.0 - m)[..., None]).astype(np.float32)
    return flow


def density_step(p, height, width):
    detail = min(1.0, max(0.02, float(p.get("motion_vector_detail", 1.0))))
    step = max(1, int(round(1.0 / detail)))
    while step > 1 and min(height, width) // step < 12:
        step -= 1
    return step


def pyramid_levels(step):
    """Pyramid levels for a field found at 1 / step size: the full-size estimate's four, less the octaves the averaging down already
    removed (ceil(log2 step)). Motion on the smaller pair is step times smaller, so the extra coarse levels only add tiny levels where
    the 15-pixel match windows see mostly edge padding; on the CPU estimator they made a 4-pixel pan read as 60 to 70 pixels
    (Desktop conformance on 8ab38cf, tests.test_flow_node_accuracy)."""
    return max(1, 4 - int(np.ceil(np.log2(step)))) if step > 1 else 4


def estimate_flow(first, second, p, matte=None):
    """The vectors (pixels per frame, height x width x 2, float32) from `first` to `second` that Method, Vector Detail and the matte give."""
    from .opticalflow import _luma, flow_pair
    first, second = np.asarray(first, np.float32), np.asarray(second, np.float32)
    height, width = first.shape[:2]
    step = density_step(p, height, width)
    low_a, low_b = (_down(first, step), _down(second, step)) if step > 1 else (first, second)
    regularized = str(p.get("motion_method", "Regularized")) == "Regularized"
    flow, _, _ = flow_pair(low_a, low_b, vector_detail=pyramid_levels(step), smoothness=1.0 if regularized else 0.0,
                           backend=p.get("flow_backend", "auto"))
    flow = np.asarray(flow, np.float32)
    if regularized:
        flow = _regularize(flow, _luma(low_a))
    if matte is not None:
        flow = split_by_matte(flow, _down(np.asarray(matte, np.float32), step) if step > 1 else np.asarray(matte, np.float32))
    return _up(flow, step, height, width)
