"""Sampler's Normalize X (lane 2, 2D behaviour step R).

Nuke 17.0's Sampler turns the scanline between its two points into lookup curves, and `normalize_x` ("normalize x") scales the curves' x values to
0 to 1 across the line; off, x is the distance along the line. NodeBased's Sampler passes its image through unchanged and shows the line as a plot in
its panel, so the knob acts on that readout: the plot's x axis runs 0 to 1 (on, Nuke's default and how the plot always ran) or in pixels along the
line (off), and is labelled so. `lut`, the curve output Nuke feeds to a ColorLookup, has no counterpart: the readout is not a curve knob.
"""
from __future__ import annotations

import math

NEW = {"normalize_x": 1}


def new_params(kind):
    return dict(NEW) if kind == "Sampler" else {}


def upgrade_params(kind):
    """A saved Sampler plotted its samples across the line's width, which is what Normalize X on shows."""
    return dict(NEW) if kind == "Sampler" else {}


def line_length(start, end):
    """The length of the sampled line in pixels."""
    return math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def x_values(count, start, end, normalize):
    """The x value of each of `count` samples along the line from `start` to `end`: 0 to 1 when `normalize`, else pixels from the start."""
    count = max(int(count), 1)
    scale = 1.0 if normalize else line_length(start, end)
    return [scale * (i / (count - 1) if count > 1 else 0.0) for i in range(count)]


def axis_labels(start, end, normalize):
    """The captions under the plot's left and right ends."""
    return ("0", "1") if normalize else ("0", f"{line_length(start, end):.0f} px")


def readout(pixels, params, origin=(0, 0)):
    """What the panel plots for one frame: the x value and the RGBA of each sample along the line, with the channels `sample_channels` draws. The image
    itself is not changed by the Sampler."""
    import numpy as np
    from .imaging import Evaluator
    start, end = (params["sample_x0"], params["sample_y0"]), (params["sample_x1"], params["sample_y1"])
    values = Evaluator._sample_line(np.asarray(pixels, np.float32), start, end, origin)
    channels = {"alpha": (3,), "rgba": (0, 1, 2, 3)}.get(str(params.get("sample_channels", "rgb")), (0, 1, 2))
    return {"x": x_values(len(values), start, end, bool(int(params.get("normalize_x", 1)))), "values": values[:, channels], "channels": channels,
            "labels": axis_labels(start, end, bool(int(params.get("normalize_x", 1))))}
