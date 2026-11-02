"""ColorLookup's Set RGB, Set RGBA and Set A buttons (2D behaviour lane 2, step B).

Nuke's `source` and `target` are colour knobs (one value per channel, r g b a): "Set RGB" adds a point to the red, green and blue curves that
maps the source value of that channel to its target value, "Set RGBA" adds the same on the alpha curve too, and "Set A" only on alpha. The
point is added to the curve as it stands (a key already at the source value takes the target value), so a curve built up from several
source and target pairs keeps its earlier points.
"""
from __future__ import annotations

import json

from . import colorcurves
from .core import split_values

CHANNELS = ("red", "green", "blue", "alpha")


def _with_point(curve, x, y):
    points = curve["points"]
    near = [i for i, point in enumerate(points) if abs(point[0] - x) < 1e-9]
    if near:
        points[near[0]][1] = y
        return curve
    keyed = "slopes" in curve
    if keyed and points[0][0] < x < points[-1][0]:
        index = colorcurves.insert_key(curve, x)
        points[index][1] = y
        return curve
    index = sum(1 for point in points if point[0] < x)
    points.insert(index, [x, y])
    if keyed:
        before, after = points[index - 1] if index else None, points[index + 1] if index + 1 < len(points) else None
        neighbour = before or after
        chord = (y - neighbour[1]) / (x - neighbour[0]) if neighbour[0] != x else 0.0
        curve["slopes"].insert(index, [chord, chord])
        curve["modes"].insert(index, "linear")
        curve["broken"].insert(index, False)
    return curve


def set_points(params, channels):
    """{curve param: new curve text} for the buttons' channels (`red`, `green`, `blue`, `alpha`), from the node's source and target."""
    source = split_values(params, "source", 4, default=0.0)
    target = split_values(params, "target", 4, default=0.0)
    out = {}
    for name in channels:
        index = CHANNELS.index(name)
        curve = _with_point(colorcurves.decode(params[f"curve_{name}"]), float(source[index]), float(target[index]))
        out[f"curve_{name}"] = json.dumps(curve, separators=(",", ":"))
    return out
