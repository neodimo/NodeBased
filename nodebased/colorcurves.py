"""Reusable, serialisable curves used by colour nodes and their editor.

The wire format is deliberately small JSON so curve edits remain ordinary knob edits (and
therefore one document undo step apiece). Points are [x, y] pairs; x values are strictly
increasing. Smooth interpolation uses the same smoothstep as the original HueCorrect model.
"""
from __future__ import annotations

import json
import math

import numpy as np


def encode(points, interpolation="linear", slopes=None, modes=None, broken=None):
    """Encode legacy curves unchanged, or keyed Hermite curves when slopes are supplied."""
    data = {"interpolation": interpolation,
            "points": [[float(x), float(y)] for x, y in points]}
    if slopes is not None:
        data["slopes"] = [[float(pair[0]), float(pair[1])] for pair in slopes]
        data["modes"] = list(modes or ["broken"] * len(points))
        data["broken"] = list(broken or [True] * len(points))
    return json.dumps(data, separators=(",", ":"))


def decode(value):
    try:
        curve = json.loads(value)
        if (not isinstance(curve, dict) or set(curve) not in ({"interpolation", "points"},
                    {"interpolation", "points", "slopes", "modes", "broken"})
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
        if "slopes" in curve:
            n = len(points)
            if (len(curve["slopes"]) != n or len(curve["modes"]) != n or len(curve["broken"]) != n
                    or any(len(s) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in s)
                           for s in curve["slopes"])
                    or any(m not in ("smooth", "linear", "constant", "broken") for m in curve["modes"])
                    or any(not isinstance(b, bool) for b in curve["broken"])):
                raise ValueError
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
            if "slopes" in curve:
                mode = curve["modes"][points.index([x0, y0])]
                if mode == "constant":
                    return y0 if x < x1 else y1
                if mode == "linear":
                    return y0 + (y1 - y0) * t
                slopes = curve["slopes"]
                if mode == "smooth":
                    left = (points[min(points.index([x0, y0]) + 1, len(points)-1)][1] -
                            points[max(points.index([x0, y0]) - 1, 0)][1]) / (
                            points[min(points.index([x0, y0]) + 1, len(points)-1)][0] -
                            points[max(points.index([x0, y0]) - 1, 0)][0])
                    right_i = points.index([x1, y1])
                    right = (points[min(right_i + 1, len(points)-1)][1] -
                             points[max(right_i - 1, 0)][1]) / (
                             points[min(right_i + 1, len(points)-1)][0] -
                             points[max(right_i - 1, 0)][0])
                else:
                    left, right = slopes[points.index([x0, y0])][1], slopes[points.index([x1, y1])][0]
                h = x1 - x0
                return ((2*t**3 - 3*t**2 + 1)*y0 + (t**3 - 2*t**2 + t)*h*left +
                        (-2*t**3 + 3*t**2)*y1 + (t**3 - t**2)*h*right)
            if curve["interpolation"] == "smooth":
                t = t * t * (3.0 - 2.0 * t)
            return y0 * (1.0 - t) + y1 * t
    return points[-1][1]


def wrapped_hue(curve, hue):
    """Evaluate periodic hue curves whose endpoints meet at 0 and 360."""
    return evaluate(curve, float(hue) % 360.0)


def evaluate_array(curve, values):
    """Vectorised curve evaluation, kept exactly consistent with the scalar `evaluate` above
    (including its keyed Hermite tangent handles) so a full-image kernel like CrossTalk's, and a
    single point sample, always agree.
    """
    points = np.asarray(curve["points"], dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    values = np.asarray(values, dtype=np.float64)
    if "slopes" not in curve:
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
    n = len(points)
    modes, slopes = curve["modes"], np.asarray(curve["slopes"], dtype=np.float64)
    # The "smooth" auto-tangent is the central difference through neighbouring keys, matching the
    # scalar branch exactly (an end key uses its one neighbour).
    auto = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        auto[i] = (y[hi] - y[lo]) / (x[hi] - x[lo])
    result = np.empty_like(values, dtype=np.float64)
    below, above = values <= x[0], values >= x[-1]
    result[below] = y[0] + (values[below] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    result[above] = y[-1] + (values[above] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(n - 1):
        x0, x1, y0, y1, mode = x[i], x[i + 1], y[i], y[i + 1], modes[i]
        mask = (values >= x0) & (values <= x1)
        if not mask.any():
            continue
        seg = values[mask]
        if mode == "constant":
            result[mask] = np.where(seg < x1, y0, y1)
            continue
        t = (seg - x0) / (x1 - x0)
        if mode == "linear":
            result[mask] = y0 + (y1 - y0) * t
            continue
        left, right = (auto[i], auto[i + 1]) if mode == "smooth" else (slopes[i][1], slopes[i + 1][0])
        h = x1 - x0
        result[mask] = ((2*t**3 - 3*t**2 + 1)*y0 + (t**3 - 2*t**2 + t)*h*left +
                        (-2*t**3 + 3*t**2)*y1 + (t**3 - t**2)*h*right)
    return result.astype(np.float32)


def _auto_slope(points, i):
    """Central-difference tangent through a key's neighbours (an end key uses its one neighbour)."""
    lo, hi = max(i - 1, 0), min(i + 1, len(points) - 1)
    return (points[hi][1] - points[lo][1]) / (points[hi][0] - points[lo][0])


def ensure_keyed(curve):
    """Give a legacy curve explicit tangent data without changing its shape.

    A linear curve becomes all-linear keys; the legacy smoothstep segment is exactly the cubic
    Hermite segment with zero end tangents, so it becomes flat-tangent "broken" keys.
    """
    if "slopes" in curve:
        return curve
    n = len(curve["points"])
    smooth = curve["interpolation"] == "smooth"
    curve["slopes"] = [[0.0, 0.0] for _ in range(n)] if smooth else [
        [_auto_slope(curve["points"], i)] * 2 for i in range(n)]
    curve["modes"] = ["broken" if smooth else "linear"] * n
    curve["broken"] = [False] * n
    return curve


def segment_slopes(curve, j):
    """The (start, end) slopes the segment from key j to key j+1 is drawn with."""
    points, mode = curve["points"], curve["modes"][j]
    if mode == "smooth":
        return _auto_slope(points, j), _auto_slope(points, j + 1)
    if mode == "broken":
        return curve["slopes"][j][1], curve["slopes"][j + 1][0]
    chord = (points[j + 1][1] - points[j][1]) / (points[j + 1][0] - points[j][0])
    return chord, chord


def bake_segment(curve, j):
    """Store segment j's current slopes explicitly and make it a "broken" segment, so a
    neighbouring edit cannot move it through an automatic tangent."""
    ensure_keyed(curve)
    if curve["modes"][j] != "broken":
        start, end = segment_slopes(curve, j)
        curve["slopes"][j][1], curve["slopes"][j + 1][0] = start, end
        curve["modes"][j] = "broken"


def set_tangent(curve, index, side, slope, break_tangent=False):
    """Set key `index`'s tangent: side 0 is the incoming (left) handle, 1 the outgoing (right).

    A smooth key keeps both handles on one slope; with `break_tangent` (Nuke's Ctrl-drag) the key
    becomes broken and only this side moves. A broken key stays broken until its mode is reset.
    """
    ensure_keyed(curve)
    n = len(curve["points"])
    if break_tangent:
        curve["broken"][index] = True
    sides = (side,) if curve["broken"][index] else (0, 1)
    for s in sides:
        j = index - 1 if s == 0 else index
        if 0 <= j < n - 1:
            bake_segment(curve, j)
        curve["slopes"][index][s] = float(slope)


def insert_key(curve, x):
    """Add a key on the curve at `x` without moving the curve.

    The new key sits at the curve's own value with the curve's own derivative, so the two
    resulting cubic segments reproduce the original one exactly; the neighbouring segments keep
    their slopes (any automatic tangent that the new key would have shifted is stored first).
    Returns the new key's index.
    """
    ensure_keyed(curve)
    points = curve["points"]
    if not points[0][0] < x < points[-1][0]:
        raise ValueError("A key can only be added inside the curve's range")
    j = max(i for i in range(len(points) - 1) if points[i][0] < x)
    # Inserting shifts the automatic tangent of both ends of segment j, which three segments use.
    for k in range(max(j - 1, 0), min(j + 2, len(points) - 1)):
        if curve["modes"][k] == "smooth":
            bake_segment(curve, k)
    y = evaluate(curve, x)
    mode = curve["modes"][j]
    x0, x1 = points[j][0], points[j + 1][0]
    slope = 0.0
    if mode == "broken":
        t, h = (x - x0) / (x1 - x0), x1 - x0
        m0, m1 = curve["slopes"][j][1], curve["slopes"][j + 1][0]
        y0, y1 = points[j][1], points[j + 1][1]
        # Derivative of the cubic Hermite segment at t.
        slope = ((6*t*t - 6*t) * y0 + (3*t*t - 4*t + 1) * h * m0 +
                 (-6*t*t + 6*t) * y1 + (3*t*t - 2*t) * h * m1) / h
    elif mode == "linear":
        slope = (points[j + 1][1] - points[j][1]) / (x1 - x0)
    points.insert(j + 1, [float(x), float(y)])
    curve["slopes"].insert(j + 1, [slope, slope])
    curve["modes"].insert(j + 1, mode)
    curve["broken"].insert(j + 1, False)
    return j + 1


def move_point(curve, index, x, y):
    """Move a key, keeping the x order strict; returns the (x, y) actually stored."""
    points = curve["points"]
    lo = points[index - 1][0] + 1e-5 if index > 0 else -math.inf
    hi = points[index + 1][0] - 1e-5 if index + 1 < len(points) else math.inf
    points[index] = [min(max(float(x), lo), hi), float(y)]
    return tuple(points[index])


def delete_key(curve, index):
    if len(curve["points"]) <= 2:
        return
    ensure_keyed(curve)
    for name in ("points", "slopes", "modes", "broken"):
        curve[name].pop(index)


def default_curve(x0=0.0, x1=1.0, y=1.0):
    return encode(((x0, y), (x1, y)))


def anchor_curve(values):
    """Convert six historic 60-degree anchors to the exact old smoothstep curve."""
    return encode(tuple((i * 60.0, float(v)) for i, v in enumerate(values)) +
                  ((360.0, float(values[0])),), "smooth")
