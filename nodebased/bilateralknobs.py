"""Bilateral as the Nuke 17.0 Reference Guide describes it (lane 9, 2D behaviour step G): the `guide` input, Guide Channels, Size as the
window, Positional Sigma and the Filter choice.

The guide's text: Size "sets the height and width of the area within which pixels are compared"; Positional Sigma "controls the amount that
nearby pixels affect the overall blur"; Colour Sigma "controls the amount of blur that occurs over edges, lower values preserve edges more
reliably"; the guide image "computes colour similarity while filtering" (a cross bilateral filter when it has the source's format), over the
Guide Channels; Filter is gaussian ("weighted to differences in the surrounding pixel colour and how close those pixels are") or median
("weighted according to the middle value pixel").

What the guide does not state, and this module therefore fixes by the classic form (Tomasi and Manduchi, ICCV 1998):

* the window is `size` pixels wide and high, centred on the pixel (radius `size // 2`);
* the positional weight is a Gaussian of the pixel distance with sigma `positionalSigma * size` pixels, the colour weight a Gaussian of the
  distance between the guide's channels at the two pixels with sigma `colourSigma`;
* median is the weighted median of each channel under those same weights.

An older document's node keeps its own reading (`legacy_bilateral` 1): the window radius `ceil(size)`, the positional falloff
`exp(-d^2 / 2 radius^2)` and the colour distance over the picture's RGB; it has no guide. A call that carries no flag reads as the older node,
which is what Denoise's call to the same filter relies on.
"""
from __future__ import annotations

import math

import numpy as np

FILTERS = ("gaussian", "median")
GUIDE_CHANNELS = ("all", "rgb", "alpha", "red", "green", "blue")
_INDICES = {"all": (0, 1, 2, 3), "rgb": (0, 1, 2), "alpha": (3,), "red": (0,), "green": (1,), "blue": (2,)}
LEGACY = {"legacy_bilateral": 1}
NEW = {"legacy_bilateral": 0, "positional_sigma": 0.4, "bilateral_filter": "gaussian", "guide_channels": "all"}
# What an older document's node gets on load: the flag, and each new knob at the value that leaves it as it was (colour sigma stays 0.1, Nuke's is 0.4).
UPGRADE = {"legacy_bilateral": 1, "positional_sigma": 0.4, "bilateral_filter": "gaussian", "guide_channels": "rgb", "colour_sigma": 0.1}


def is_legacy(p):
    return bool(int(p.get("legacy_bilateral", 1)))


def radius(p):
    """The window's reach in pixels each way: how far a tile must look past its edge."""
    size = max(0.0, float(p.get("spatial_size", 3.0)))
    return int(math.ceil(size)) if is_legacy(p) else int(size) // 2


def _window(image, guide, p, rad):
    h, w = image.shape[:2]
    pad = lambda a: np.pad(a, ((rad, rad), (rad, rad), (0, 0)), mode="edge")
    return pad(image.astype(np.float64)), pad(guide.astype(np.float64))


def bilateral(image, p, guide=None):
    """Bilateral on an (h, w, 4) picture. `guide` (same shape) holds the colours the weights are taken from; None means the picture itself."""
    rad = radius(p)
    if rad == 0:
        return image.copy()
    sigma_c = max(1e-5, float(p.get("colour_sigma", 0.1)))
    h, w = image.shape[:2]
    if is_legacy(p):
        sigma_s2 = max(1.0, 2.0 * rad * rad)
        guide_idx = (0, 1, 2)
    else:
        size = max(1.0, float(p.get("spatial_size", 3.0)))
        sigma_s = max(1e-5, float(p.get("positional_sigma", 0.4)) * size)
        sigma_s2 = 2.0 * sigma_s * sigma_s
        guide_idx = _INDICES.get(str(p.get("guide_channels", "all")), _INDICES["all"])
    reference = image if guide is None else guide
    padded, padded_ref = _window(image, reference, p, rad)
    centre = reference.astype(np.float64)[..., list(guide_idx)]
    samples, weights = [], []
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            spatial = math.exp(-(dx * dx + dy * dy) / sigma_s2)
            ref = padded_ref[rad + dy:rad + dy + h, rad + dx:rad + dx + w][..., list(guide_idx)]
            diff = ref - centre
            weight = spatial * np.exp(-np.sum(diff * diff, axis=2, keepdims=True) / (2 * sigma_c * sigma_c))
            samples.append(padded[rad + dy:rad + dy + h, rad + dx:rad + dx + w])
            weights.append(weight)
    if not is_legacy(p) and str(p.get("bilateral_filter", "gaussian")) == "median":
        return _weighted_median(np.stack(samples, axis=0), np.stack(weights, axis=0)[..., 0]).astype(np.float32)
    out = np.zeros((h, w, image.shape[2]), np.float64)
    total = np.zeros((h, w, 1), np.float64)
    for sample, weight in zip(samples, weights):
        out += sample * weight
        total += weight
    return (out / np.maximum(total, 1e-20)).astype(np.float32)


def _weighted_median(samples, weights):
    """The value at which half the total weight lies below, per channel. samples (k, h, w, c); weights (k, h, w)."""
    order = np.argsort(samples, axis=0, kind="stable")
    sorted_samples = np.take_along_axis(samples, order, axis=0)
    sorted_weights = np.take_along_axis(np.broadcast_to(weights[..., None], samples.shape), order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    half = cumulative[-1:] * 0.5
    index = np.argmax(cumulative >= half, axis=0)[None]
    return np.take_along_axis(sorted_samples, index, axis=0)[0]
