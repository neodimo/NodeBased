"""Reusable, serialisable curves used by colour nodes and their editor.

The wire format is deliberately small JSON so curve edits remain ordinary knob edits (and
therefore one document undo step apiece). Points are [x, y] pairs; x values are strictly
increasing. Smooth interpolation uses the same smoothstep as the original HueCorrect model.
"""
from __future__ import annotations

import json
import math

import numpy as np


def encode(points, interpolation="linear"):
    return json.dumps({"interpolation": interpolation,
                       "points": [[float(x), float(y)] for x, y in points]},
                      separators=(",", ":"))


def decode(value):
    try:
        curve = json.loads(value)
        if (not isinstance(curve, dict) or set(curve) != {"interpolation", "points"}
                or curve["interpolation"] not in ("linear", "smooth")
                or not isinstance(curve["points"], list) or len(curve["points"]) < 2):
            raise ValueError
        points = curve["points"]
        previous = -math.inf
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)
                    or point[0] <= previous):
                raise ValueError
            previous = point[0]
        return curve
    except (TypeError, json.JSONDecodeError, ValueError):
        raise ValueError("Colour curve must contain ordered finite [x, y] points") from None


def evaluate(curve, x):
    points = curve["points"]
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[:2]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    if x >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2:]
        return y1 + (x - x1) * (y1 - y0) / (x1 - x0)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            if curve["interpolation"] == "smooth":
                t = t * t * (3.0 - 2.0 * t)
            return y0 * (1.0 - t) + y1 * t
    return points[-1][1]


def wrapped_hue(curve, hue):
    """Evaluate periodic hue curves whose endpoints meet at 0 and 360."""
    return evaluate(curve, float(hue) % 360.0)


def evaluate_array(curve, values):
    points = np.asarray(curve["points"], dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    values = np.asarray(values, dtype=np.float64)
    result = np.interp(values, x, y)
    result = np.where(values < x[0], y[0] + (values - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), result)
    result = np.where(values > x[-1], y[-1] + (values - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), result)
    if curve["interpolation"] == "smooth":
        for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            mask = (values >= x0) & (values <= x1)
            t = (values[mask] - x0) / (x1 - x0)
            t = t * t * (3.0 - 2.0 * t)
            result[mask] = y0 * (1.0 - t) + y1 * t
    return result.astype(np.float32)


def default_curve(x0=0.0, x1=1.0, y=1.0):
    return encode(((x0, y), (x1, y)))


def anchor_curve(values):
    """Convert six historic 60-degree anchors to the exact old smoothstep curve."""
    return encode(tuple((i * 60.0, float(v)) for i, v in enumerate(values)) +
                  ((360.0, float(values[0])),), "smooth")
