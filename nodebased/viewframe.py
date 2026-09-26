"""Viewer region of interest and format masks: the geometry, free of Qt so tests can assert it.

Both are display-side choices. The region of interest narrows what is evaluated (the request to
the tile executor is clipped to it); a format mask only paints over the picture."""
from __future__ import annotations

import math

# How dark the outside of a mask is: "half" lets the picture show through, "full" is opaque black.
MASK_DARKEN = {"half": 0.5, "full": 1.0}


def roi_pixels(rect, x, y, width, height):
    """The ROI rectangle `rect` (fractions [x0, y0, x1, y1] of the canvas) as integer pixels of the
    canvas whose top-left is (x, y) and size (width, height). Rounds outward so the clip never
    loses part of what the artist drew, and is at least one pixel each way."""
    x0 = x + min(width - 1, max(0, math.floor(rect[0] * width + 1e-9)))
    y0 = y + min(height - 1, max(0, math.floor(rect[1] * height + 1e-9)))
    x1 = x + max(x0 - x + 1, min(width, math.ceil(rect[2] * width - 1e-9)))
    y1 = y + max(y0 - y + 1, min(height, math.ceil(rect[3] * height - 1e-9)))
    return x0, y0, x1, y1


def mask_aspect(mask, width, height, pixel_aspect=1.0):
    """The aspect ratio a mask name stands for; "format" is the frame's own."""
    if mask == "format":
        return float(width) * pixel_aspect / float(height)
    return float(mask)


def mask_rect(mask, width, height, pixel_aspect=1.0):
    """The part of a `width` x `height` frame the mask keeps, as (x0, y0, x1, y1) in pixels: the
    largest centred rectangle of that aspect. Pillarboxes a frame narrower than the mask's aspect
    the other way round, as Nuke's masks do."""
    aspect = mask_aspect(mask, width, height, pixel_aspect)
    frame_aspect = float(width) * pixel_aspect / float(height)
    if aspect >= frame_aspect:
        kept_h = width * pixel_aspect / aspect
        top = (height - kept_h) / 2.0
        return 0.0, top, float(width), top + kept_h
    kept_w = height * aspect / pixel_aspect
    left = (width - kept_w) / 2.0
    return left, 0.0, left + kept_w, float(height)


def mask_bars(mask, width, height, pixel_aspect=1.0):
    """The outside of the mask as whole-pixel rectangles (x0, y0, x1, y1): the rows above and below,
    or the columns left and right. Edges round to the nearest pixel boundary."""
    x0, y0, x1, y1 = mask_rect(mask, width, height, pixel_aspect)
    top, bottom = round(y0), round(y1)
    left, right = round(x0), round(x1)
    bars = []
    if top > 0:
        bars.append((0, 0, width, top))
    if bottom < height:
        bars.append((0, bottom, width, height))
    if left > 0:
        bars.append((0, 0, left, height))
    if right < width:
        bars.append((right, 0, width, height))
    return bars
