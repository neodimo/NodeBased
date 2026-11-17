"""The Histogram node's plot (Nuke's `hist` knob, lane 2 step W2).

The guide: "The horizontal axis represents brightness (the numerical value of a pixel), with the shadows on the left, midtones in the middle, and
highlights on the right. The vertical axis represents pixel count (the percentage of pixels with that value)", and "the histogram updates as you
adjust these controls". So the plot has a fixed 0 to 1 horizontal axis (values outside it fall into the first and last column, the way the node
clamps superblacks and superwhites), one curve per colour channel, and each column is a percentage of the picture's pixels. The plot shows the
picture as it leaves the node (the input range, midtone and output range applied), beside the picture as it came in.
"""
from __future__ import annotations

import numpy as np

BINS = 64


def percentages(pixels, bins=BINS):
    """Per colour channel, the percentage of the picture's pixels in each of `bins` equal columns over brightness 0 to 1: a (3, bins) array whose rows
    each add up to 100. A value below 0 or above 1 is counted in the first or last column."""
    rgb = np.asarray(pixels, dtype=np.float64)[..., :3]
    count = max(1, rgb.shape[0] * rgb.shape[1])
    out = np.zeros((3, int(bins)), np.float64)
    for channel in range(3):
        values = np.clip(rgb[..., channel].ravel(), 0.0, 1.0)
        out[channel] = np.histogram(values, bins=int(bins), range=(0.0, 1.0))[0] * 100.0 / count
    return out


def plot_data(pixels, params, bins=BINS):
    """(input percentages, output percentages) for a Histogram node's `params`: the picture as it came in and as the node leaves it."""
    from .imaging import Evaluator
    levels = {name: params.get(name, default) for name, default in (("black", 0.0), ("white", 1.0), ("gamma", 1.0), ("black_out", 0.0), ("white_out", 1.0))}
    try:
        result = Evaluator._histogram_levels(np.asarray(pixels, dtype=np.float32), levels)
    except ValueError:
        result = np.asarray(pixels, dtype=np.float32)
    return percentages(pixels, bins), percentages(result, bins)
