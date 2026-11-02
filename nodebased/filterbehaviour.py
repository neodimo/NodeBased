"""Behaviour fixes to the Filter nodes (lane 9, 2D behaviour step A): Erode (fast), Erode (filter), Median, Matrix, Laplacian and
Convolve.

Each fix keeps the older reading behind a `legacy_*` parameter. A parameter dict that predates the knob reads as an older
document's node (the `_get` pattern of blurknobs.py), and `upgrade_params` gives a saved node the explicit legacy value on load, so
an older document renders bit for bit as it did. A node created now gets the new reading from its SPECS defaults.

* Erode (fast): the guide says positive size increases the light pixels (it grows a matte) and negative size grows the dark ones.
  Older documents had it the other way round (`legacy_erode_sign` 1).
* Erode (filter): the guide says negative size lets brighter areas expand into darker ones; a negative size used to do nothing.
* Median: size is the width of the filter area in pixels (3 is the 3 x 3 area, the guide's default). Older documents had it a
  radius, so size 1 was the 3 x 3 area (`legacy_median_radius` 1).
* Matrix: "results that fall outside the 0-1 range are clipped" (the guide's note). Older documents were not clipped
  (`legacy_matrix_unclipped` 1).
* Laplacian: "applies a blur, and then subtracts the original from the blurred input", with Nuke's size, filter, quality and crop
  to format. Older documents had a fixed 3 x 3 cross, original minus blur, and no knobs (`legacy_laplacian` 1).
* Convolve: the filter image's own channel (the alpha, by default), normalize as a switch, the filter centred on the middle of its
  bounding box at any size. Older documents read the red channel of the top-left corner of the filter image at `kernel_size`,
  always normalised (`legacy_convolve` 1).

Pure NumPy, so the evaluator, the tile executor and the tests read one arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

LEGACY = {
    "Erode": {"legacy_erode_sign": 1},
    "Median": {"legacy_median_radius": 1},
    "Matrix": {"legacy_matrix_unclipped": 1},
    "Laplacian": {"legacy_laplacian": 1},
    "Convolve": {"legacy_convolve": 1},
}
NEW = {
    "EdgeDetect": {"edge_blur_size": 0.0, "edge_erode_size": 0.0, "edge_blur_quality": 15.0},
    "Erode": {"legacy_erode_sign": 0},
    "Median": {"legacy_median_radius": 0},
    "Matrix": {"legacy_matrix_unclipped": 0},
    "Laplacian": {"legacy_laplacian": 0},
    "Convolve": {"legacy_convolve": 0, "normalize": 1, "filter_channel": "rgba.alpha", "use_input_channels": 0, "filter_bounds": "shape"},
}
# What an older document's node of the kind gets on load besides the legacy flags: the new knobs, at the value that leaves it as it was.
UPGRADE_EXTRA = {"EdgeDetect": {"edge_blur_size": 0.0, "edge_erode_size": 0.0, "edge_blur_quality": 15.0},
                 "Laplacian": {"laplacian_size": 3.0},
                 "Convolve": {"normalize": 1, "filter_channel": "rgba.alpha", "use_input_channels": 0, "filter_bounds": "shape"}}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What an older document's node of `kind` gets on load."""
    return {**LEGACY.get(kind, {}), **UPGRADE_EXTRA.get(kind, {})}


def is_legacy(kind, p, name):
    """True when `p` reads as an older document's node for the flag `name` (a missing flag is an older node)."""
    return bool(int(p.get(name, LEGACY[kind][name])))


# ------------------------------------------------------------------------------------------------------------- Erode
def erode_size(p):
    """The signed size the box min/max filter is run at: negative takes the maximum, positive the minimum (shrinks light areas).

    `Evaluator._morph` erodes for a positive size; the guide's positive size grows light pixels, so a node of the new reading
    hands it the negated size."""
    size = float(p.get("erode_size", 1.0))
    return size if is_legacy("Erode", p, "legacy_erode_sign") else -size


# ------------------------------------------------------------------------------------------------------------ Median
def median_support(p):
    """Pixels each way the median window reaches: 0 for an identity, else half the odd width the size names.

    Size is the width of the filter area, so 3 is the 3 x 3 area. An even or fractional size is not covered by the guide; it
    rounds up to the next odd width (4 is the 5 x 5 area). Older nodes read size as a radius: ceil(size)."""
    size = float(p.get("median_size", 1.0))
    if is_legacy("Median", p, "legacy_median_radius"):
        return 0 if size < 0.5 else int(math.ceil(size))
    return 0 if size < 1.5 else int(math.ceil((size - 1.0) / 2.0))


# ------------------------------------------------------------------------------------------------------------ Matrix
def matrix_clip(image, p):
    """`image` clipped to 0 .. 1 for the Matrix of the new reading (the guide's note); unchanged for an older node."""
    if is_legacy("Matrix", p, "legacy_matrix_unclipped"):
        return image
    return np.clip(image, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------------------------- Laplacian
def laplacian_is_legacy(p):
    return is_legacy("Laplacian", p, "legacy_laplacian")


def laplacian(image, p, channel_sets, origin=(0, 0)):
    """Nuke's Laplacian: the blur of `laplacian_size` with the node's filter and quality, minus the original, on the channels named."""
    from . import blurknobs
    blurred = blurknobs.filtered(image, p.get("laplacian_size", 3.0), blurknobs._get("Laplacian", p, "kernel_filter"),
                                 blurknobs._get("Laplacian", p, "quality"), origin)
    out = image.copy()
    for c in channel_sets[p.get("channels", "rgba")]:
        out[..., c] = blurred[..., c] - image[..., c]
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------------------ Convolve
def convolve_is_legacy(p):
    return is_legacy("Convolve", p, "legacy_convolve")


def filter_planes(kernel, p):
    """The filter image as 2-D float64 planes, one per image channel (RGBA), each divided by its sum when `normalize` is on.

    `kernel` is the filter input's Raster. Its data window is the filter's bounding box (`filter_bounds` shape: the filter's centre
    is the centre of that box); `format` lays the box in the filter input's display window, so the filter's centre is the format's
    centre and the image can be offset inside it. The plane is the channel named by `filter_channel` (the alpha, by default), or
    with `use_input_channels` on each image channel's own."""
    from . import ops2d_chrome as chrome
    pixels = np.asarray(kernel.pixels, np.float64)
    if str(p.get("filter_bounds", "shape")) == "format" and kernel.display != kernel.data:
        shown = np.zeros((kernel.display.height, kernel.display.width, 4), np.float64)
        oy, ox = kernel.data.y - kernel.display.y, kernel.data.x - kernel.display.x
        top, left = max(oy, 0), max(ox, 0)
        bottom, right = min(oy + pixels.shape[0], shown.shape[0]), min(ox + pixels.shape[1], shown.shape[1])
        if bottom > top and right > left:
            shown[top:bottom, left:right] = pixels[top - oy:bottom - oy, left - ox:right - ox]
        pixels = shown
    if int(p.get("use_input_channels", 0)):
        planes = [pixels[..., c] for c in range(4)]
    else:
        plane = chrome.resolve_channel(p.get("filter_channel", "rgba.alpha"), pixels.astype(np.float32))[..., 0].astype(np.float64)
        planes = [plane] * 4
    out = []
    for plane in planes:
        total = float(plane.sum())
        out.append(plane / total if int(p.get("normalize", 1)) and abs(total) > 1e-12 else plane)
    return out


def _correlate_plane(channel, weights):
    """Each output pixel is the sum of the weights times the (edge-extended) pixels centred on it; the weights' centre is the pixel
    at (rows // 2, columns // 2). Direct for a small filter, by FFT for a large one."""
    kh, kw = weights.shape
    h, w = channel.shape
    top, left = kh // 2, kw // 2
    padded = np.pad(channel.astype(np.float64), ((top, kh - 1 - top), (left, kw - 1 - left)), mode="edge")
    if kh * kw <= 225:
        out = np.zeros((h, w), np.float64)
        for y in range(kh):
            for x in range(kw):
                if weights[y, x] != 0.0:
                    out += weights[y, x] * padded[y:y + h, x:x + w]
        return out
    shape = padded.shape
    spectrum = np.fft.rfft2(padded, s=shape) * np.conj(np.fft.rfft2(weights, s=shape))
    return np.fft.irfft2(spectrum, s=shape)[:h, :w]


def convolve(image, kernel, p):
    """Nuke's Convolve: each pixel becomes the sum of its neighbours times the filter image's pixels on them."""
    planes = filter_planes(kernel, p)
    out = np.empty(image.shape, np.float64)
    for c in range(image.shape[2]):
        out[..., c] = _correlate_plane(image[..., c], planes[c])
    return out.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------- EdgeDetect
def edge_blur_reach(p):
    """The pixels each way the pre-blur reads, with the picture scaled down to `edge_blur_quality` when the blur is wider."""
    from . import blurknobs
    return blurknobs.halo("Blur", {"radius": float(p.get("edge_blur_size", 0.0)), "quality": float(p.get("edge_blur_quality", 15.0))})


def edge_erode_reach(p):
    size = abs(float(p.get("edge_erode_size", 0.0)))
    return 0 if size < 0.5 else int(math.ceil(size))


def edge_halo(p):
    """The pixels each way a tile must hold around the ones it computes: the pre-blur, the one-pixel operator, the post erode."""
    return edge_blur_reach(p) + 1 + edge_erode_reach(p)


def edge_pre_blur(image, p, origin=(0, 0)):
    """The picture blurred by `edge_blur_size` (a gaussian, Blur's default filter) before the edges are found."""
    size = float(p.get("edge_blur_size", 0.0))
    if abs(size) < 0.5:
        return image
    from . import blurknobs
    return blurknobs.filtered(image, size, "gaussian", float(p.get("edge_blur_quality", 15.0)), origin)
