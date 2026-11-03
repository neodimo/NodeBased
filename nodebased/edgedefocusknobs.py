"""EdgeExtend's matte, detail amount and edge mask, and Defocus's quality and method (lane 9, 2D behaviour step O).

EdgeExtend (the Reference Guide's page):
* `matte`: "where the foreground matte is taken from to limit the effect": None, Source Alpha (the default) or Source Inverted Alpha.
  The four choices that read the Matte input (luminance, alpha, and their inversions) need that input, which NodeBased's node does not
  have; they are not offered.
* `detail_amount`: "how much of the original high frequency detail and noise is recovered after eroding or dilating the matte
  region". The colour the extension produces is smoothed by a 3 x 3 average, and `detail_amount` of the difference between the colour and
  that average is put back: 1 (the default) keeps the colour as it was, 0 leaves the smooth colour.
* `edge_mask`: "output an edge mask": on, the alpha channel carries 1 on the pixels the extension filled in from their neighbours and
  0 elsewhere (the pixels whose own colour was not trusted), so the edge can be worked on apart from the picture.
The guide gives no formula for either; both are NodeBased's reading and are unscored. A saved node has them at 1 and off, which leaves
it as it was.

Defocus: `quality` is "the interpolation used to smooth out the discs. Higher values produce better results, but take longer": the
disc edge pixels carry the share of the pixel the disc covers, found on a grid of `round(quality / 5)` samples across each pixel (1 to
8), so a low quality is the hard-edged disc NodeBased always drew and the default 20 is a smooth edge. A saved node keeps the hard
disc (`legacy_defocus_disc`). `method` (accelerated or fullprecision) is the guide's choice between two ways to the same picture; the
guide says the accelerated differences are below 0.00001, so NodeBased has one algorithm and the knob does not change pixels.
"""
from __future__ import annotations

import math

import numpy as np

NEW = {
    "EdgeExtend": {"matte": "Source Alpha", "detail_amount": 1.0, "edge_mask": 0},
    "Defocus": {"quality": 20.0, "defocus_method": "accelerated", "legacy_defocus_disc": 0},
}
UPGRADE = {
    "EdgeExtend": {"matte": "Source Alpha", "detail_amount": 1.0, "edge_mask": 0},
    "Defocus": {"quality": 20.0, "defocus_method": "accelerated", "legacy_defocus_disc": 1},
}
MATTES = ("Source Alpha", "Source Inverted Alpha", "None")


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(UPGRADE.get(kind, {}))


def matte_plane(image, p):
    alpha = image[..., 3]
    choice = str(p.get("matte", "Source Alpha"))
    if choice == "Source Inverted Alpha":
        return 1.0 - alpha
    if choice == "None":
        return np.ones_like(alpha)
    return alpha


def add_detail(rgb, p):
    """`rgb` is the colour the extension produced; put back `detail_amount` of what a 3 x 3 average takes out of it."""
    amount = float(p.get("detail_amount", 1.0))
    if amount == 1.0:
        return rgb
    height, width = rgb.shape[:2]
    pad = np.pad(rgb.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    smooth = sum(pad[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)) / 9.0
    return (smooth + amount * (rgb - smooth)).astype(np.float32)


def disc_weights(rx, ry, quality):
    """The weights of Defocus's disc of semi-axes (rx, ry): each pixel's share covered by the ellipse, found on a `round(quality / 5)` wide
    grid of samples (1 to 8); the weights sum to 1."""
    samples = max(1, min(8, int(round(float(quality) / 5.0))))
    nx, ny = int(math.ceil(rx + 0.5)), int(math.ceil(ry + 0.5))
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    ys = (np.arange(-ny, ny + 1)[:, None] + offsets[None, :]).reshape(-1)
    xs = (np.arange(-nx, nx + 1)[:, None] + offsets[None, :]).reshape(-1)
    inside = (xs[None, :] / max(rx, 1e-6)) ** 2 + (ys[:, None] / max(ry, 1e-6)) ** 2 <= 1.0 + 1e-9
    weights = inside.reshape(2 * ny + 1, samples, 2 * nx + 1, samples).mean(axis=(1, 3))
    return weights / weights.sum()


def soft_reach(rx, ry):
    """Pixels the soft disc reads each way: the half pixel the edge samples add to the semi-axis, rounded up."""
    return int(math.ceil(max(rx, ry) + 0.5))


def convolve(image, weights):
    """`image` averaged with the (point-symmetric) `weights`, edge pixels repeated, by FFT."""
    kh, kw = weights.shape
    ny, nx = kh // 2, kw // 2
    padded = np.pad(image, ((ny, ny), (nx, nx), (0, 0)), mode="edge").astype(np.float64)
    ph, pw = padded.shape[:2]
    kernel = np.zeros((ph, pw))
    kernel[:kh, :kw] = weights
    spectrum = np.fft.rfft2(kernel)
    out = np.fft.irfft2(np.fft.rfft2(padded, axes=(0, 1)) * spectrum[..., None], s=(ph, pw), axes=(0, 1))
    return out[kh - 1:, kw - 1:][:image.shape[0], :image.shape[1]]
