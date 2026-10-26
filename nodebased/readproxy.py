"""Read's proxy (fidelity step F): a smaller version of the clip that stands in for it while the proxy tier is on.

When the viewer renders at a proxy tier (the document's tier is 2, 4 or 8) and the Read names a `proxy` file, the Read reads that file
instead of `path`: the main file is not opened at all, so a proxy plays on a machine that does not have the full-size media.
The proxy picture is fitted to the size the full-size Read would have had at that tier, so everything downstream sees the same
canvas whichever file is behind it. That size comes from the main file's display window; `proxy_format` replaces it with a
named format (`file`, the default, keeps the file's own). When the main file cannot be found the proxy keeps its own size.
At tier 1 the proxy is ignored.

The fit is a bilinear sample at pixel centres, which is an exact 2 x 2 average for a half-size proxy.
"""
from __future__ import annotations

import numpy as np

from .tiers import Region


def active(params, tier):
    return int(tier) != 1 and bool(str(params.get("proxy", "") or "").strip())


def proxy_params(params):
    """The Read's parameters with the proxy file as its path."""
    return {**params, "path": str(params["proxy"]).strip()}


def target_size(display, params, tier):
    """(width, height) the fitted proxy takes at `tier`: the full-size display (or the named proxy format) divided by the tier,
    rounded up as the decimation of a full-size Read is."""
    from .core import REFORMAT_FORMATS
    from .readknobs import parse_format

    size = parse_format(str(params.get("proxy_format", "file")), REFORMAT_FORMATS) or (display.width, display.height)
    tier = int(tier)
    return (size[0] + tier - 1) // tier, (size[1] + tier - 1) // tier


def fit_pixels(pixels, width, height):
    """`pixels` (h, w, c) resampled to height x width with bilinear weights at pixel centres."""
    h, w = pixels.shape[:2]
    if (h, w) == (height, width):
        return pixels
    def axis(count, target):
        position = (np.arange(target, dtype=np.float64) + 0.5) * count / target - 0.5
        low = np.clip(np.floor(position).astype(int), 0, count - 1)
        high = np.clip(low + 1, 0, count - 1)
        weight = np.clip(position - low, 0.0, 1.0)
        return low, high, weight.astype(np.float32)
    y0, y1, wy = axis(h, height)
    x0, x1, wx = axis(w, width)
    rows = pixels[y0] * (1.0 - wy)[:, None, None] + pixels[y1] * wy[:, None, None]
    out = rows[:, x0] * (1.0 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]
    return out.astype(np.float32)


def fit_raster(raster, width, height):
    """The proxy Raster fitted to width x height. Its whole data window is taken as the picture, so the data and display windows of
    the result are both the fitted canvas; a proxy with overscan keeps none."""
    from .raster import Raster

    window = Region(0, 0, width, height)
    return Raster(fit_pixels(raster.pixels, width, height), window, window, None, raster.meta)


def main_display(params, frame):
    """The display window of the full-size file `params["path"]` would give at `frame`, or None when it cannot be found (the
    proxy then keeps its own size)."""
    from . import imaging, readknobs

    try:
        knobs = dict(params)
        path = knobs.pop("path")
        extra = {key: knobs[key] for key in readknobs.KEYS if key in knobs and key not in ("proxy", "proxy_format")}
        extra.update({key: knobs[key] for key in ("colorspace", "alpha_mode", "layer") if key in knobs})
        _data, display = imaging.read_image_bounds(path, subimage=knobs.get("subimage", 0),
                                                   frame_offset=knobs.get("frame_offset", 0),
                                                   missing=knobs.get("missing", "error"), frame=frame, **extra)
        return display
    except (OSError, ValueError, KeyError):
        return None


def fit_to_tier(raster, original, frame, tier):
    """The proxy `raster` fitted to the canvas the full-size Read `original` would have at `tier`."""
    display = main_display(original, frame)
    named = str(original.get("proxy_format", "file")) != "file"
    if display is None and not named:
        return raster
    from .tiers import Region
    width, height = target_size(display or Region(0, 0, raster.display.width, raster.display.height), original, tier)
    return fit_raster(raster, width, height)
