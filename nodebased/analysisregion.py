"""Qt-free geometry and measurement for the viewer-drawn analysis regions.

MinColor and CurveTool analyse a box, Sampler a line. The viewer draws each one over the image,
and a drag edits the same knobs the properties panel does; on release the node's result knobs are
re-measured so the number on screen matches the box that was just drawn.
"""
from __future__ import annotations

import numpy as np

BOX_KINDS = ("MinColor", "CurveTool")
LINE_KINDS = ("Sampler",)
BOX_PARAMS = ("box_x", "box_y", "box_width", "box_height")
LINE_PARAMS = ("sample_x0", "sample_y0", "sample_x1", "sample_y1")
DEFAULT_SIZE = 64.0


def region_params(kind):
    return LINE_PARAMS if kind in LINE_KINDS else BOX_PARAMS


def shown_box(values):
    """The box as drawn: an empty (0 x 0) box means the whole image and is drawn as a default
    square that a drag then turns into an explicit box."""
    w = values["box_width"] if values["box_width"] > 0 else DEFAULT_SIZE
    h = values["box_height"] if values["box_height"] > 0 else DEFAULT_SIZE
    return values["box_x"], values["box_y"], w, h


def box_after_drag(start, part, dx, dy):
    """Corner/edge/body drag of a box given its knobs at the start of the drag.

    `part` is "body" or the compass name of a grip ("nw", "e", ...); like a Crop box, the opposite
    side stays put and the size never drops below one pixel.
    """
    x, y, w, h = shown_box(start)
    if part == "body":
        x += dx; y += dy
    else:
        if "w" in part: x += dx; w -= dx
        if "e" in part: w += dx
        if "n" in part: y += dy; h -= dy
        if "s" in part: h += dy
    return {"box_x": x, "box_y": y, "box_width": max(1.0, w), "box_height": max(1.0, h)}


def min_color_result(pixels, data_origin, box, mode="minimum"):
    """RGBA of the darkest (or brightest) pixel inside the box (0-size box: the whole image),
    or None when the box holds no pixels. `data_origin` is the raster's top-left data position."""
    from .imaging import Evaluator
    crop = pixels
    x, y, w, h = box
    if w > 0 and h > 0:
        px, py = max(0, int(x - data_origin[0])), max(0, int(y - data_origin[1]))
        crop = pixels[py:py + int(h), px:px + int(w)]
    if crop.size == 0:
        return None
    rgba, _ = Evaluator._min_color(crop, mode)
    return np.asarray(rgba, dtype=np.float64)
