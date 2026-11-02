"""Behaviour fixes to the Draw nodes (lane 9, 2D behaviour step C): Rectangle, Radial, Ramp, Noise and Grid, plus the shared ramp.

Each fix keeps the older reading behind a `legacy_*` parameter. A parameter dict that predates the knob reads as an older document's
node (a missing flag is the legacy value), and `upgrade_params` gives a saved node the explicit legacy value on load, so an older
document renders bit for bit as it did. A node created now gets the new reading from its SPECS defaults.

* Axes (`legacy_draw_axes`, Rectangle, Radial, Ramp, Noise, Grid): Nuke counts y upwards. The Draw pages say the ramp's point 0 end is
  "by default, the bottom end" and its point 1 end "the top end" (point 0 100, 100 and point 1 100, 400). NodeBased draws row 0 at
  the top, so the vertical coordinate of every position knob (the area's y, the ramp's points, the grid's offset, the noise's translate
  and centre) now counts rows from the bottom of the format. Older nodes counted them from the top.
* Ramp sampling (the same flag): a colour ramp is sampled at the pixel's centre, a half pixel in from its index, which is where Nuke
  measures its coordinates (the Expression node's x and y are pixel centres, as 2D behaviour lane 2 found). Older nodes sampled at the
  index.
* Ramp type (`ramp_type`, Ramp): the guide's list is linear, plinear, smooth0, smooth1 and smooth; the older Ramp was only linear.
* Radial (`plinear`, `ramp`, `p0`, `p1`, `color0`): "perceptually linear" and the colour tab the Radial page lists next to Rectangle's.
  Older Radial nodes get the ramp off and plinear off, so they draw as before.

Pure NumPy, so the evaluator, the tile executor and the tests read one arithmetic.
"""
from __future__ import annotations

import numpy as np

FLAG = "legacy_draw_axes"
KINDS = ("Rectangle", "Radial", "Ramp", "Noise", "Grid")
RAMP_TYPES = ("linear", "plinear", "smooth0", "smooth1", "smooth")
# The exponent a "perceptually linear" ramp uses: the value a display gamma of about 2.2 shows as steps that look even.
PLINEAR_GAMMA = 2.2
RAMP_POINTS = {"p0_x": 100.0, "p0_y": 100.0, "p1_x": 100.0, "p1_y": 400.0,
               "color0_red": 0.0, "color0_green": 0.0, "color0_blue": 0.0, "color0_alpha": 1.0}

NEW = {kind: {FLAG: 0} for kind in KINDS}
NEW["Ramp"] = {FLAG: 0, "ramp_type": "linear"}
NEW["Radial"] = {FLAG: 0, "plinear": 0, "ramp": "none", **RAMP_POINTS}
NEW["Grid"] = {FLAG: 0, "ramp": "none", **RAMP_POINTS}
LEGACY = {kind: {FLAG: 1} for kind in KINDS}
LEGACY["Ramp"] = {FLAG: 1, "ramp_type": "linear"}
LEGACY["Radial"] = {FLAG: 1, "plinear": 0, "ramp": "none", **RAMP_POINTS}
LEGACY["Grid"] = {FLAG: 1, "ramp": "none", **RAMP_POINTS}
LIMITS = {FLAG: (0, 1), "plinear": (0, 1)}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What an older document's node of `kind` gets on load."""
    return dict(LEGACY.get(kind, {}))


def is_legacy(p):
    """True when `p` reads as an older document's draw node (a missing flag is an older node)."""
    return bool(int(p.get(FLAG, 1)))


COVERAGE_KINDS = ("Rectangle", "Radial", "Ramp", "Grid")


def inverts_coverage(kind, p):
    """True when the node's `invert` is applied to the shape's coverage, before the colour and its alpha: the guide's "inverts the new
    rectangle and non-rectangle areas". An older node inverted the finished alpha instead (a colour with alpha 0.8 then gave 0.2 inside
    and 1 outside)."""
    return kind in COVERAGE_KINDS and not is_legacy(p)


def cover(p, coverage):
    """`coverage` (0..1) inverted when the node's invert is on and follows Nuke."""
    if not is_legacy(p) and int(p.get("invert", 0)):
        return 1.0 - coverage
    return coverage


def rows(p, y0, h):
    """The vertical coordinate of each of `h` rows starting at absolute row `y0`: the row index in an older node, the index counted
    from the bottom of the format (`height - 1 - row`) in a node that follows Nuke."""
    index = np.arange(y0, y0 + h, dtype=np.float64)
    return index if is_legacy(p) else float(p["height"]) - 1.0 - index


def ease(t, kind):
    """The ramp's amount for a position `t` along it (0 at point 0, 1 at point 1, already clamped): the guide's five ramp types.

    `smooth0` eases into point 0 (the colours there are spread wider), `smooth1` into point 1 and `smooth` into both; the guide
    gives the words and not the curves, so these are the usual quadratic and smoothstep forms. `plinear` is the linear ramp seen
    through a display gamma."""
    if kind == "smooth0":
        return t * t
    if kind == "smooth1":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if kind == "plinear":
        return t ** PLINEAR_GAMMA
    return t


def ramp_position(p, x0, y0, w, h):
    """Where along the ramp from point 0 to point 1 each pixel sits (unclamped, shape (h, w)). A node that follows Nuke measures
    from the pixel's centre and from the bottom of the format; an older one from the pixel's index and the top."""
    legacy = is_legacy(p)
    xs = np.arange(x0, x0 + w, dtype=np.float64)
    ys = rows(p, y0, h)
    if not legacy:
        xs, ys = xs + 0.5, ys + 0.5
    xx, yy = np.meshgrid(xs, ys)
    p0x, p0y = float(p.get("p0_x", 0.0)), float(p.get("p0_y", 0.0))
    dx, dy = float(p.get("p1_x", 1.0)) - p0x, float(p.get("p1_y", 0.0)) - p0y
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return np.zeros_like(xx)
    return ((xx - p0x) * dx + (yy - p0y) * dy) / denom


def ramp_weight(p, x0, y0, w, h, kind):
    """The 0..1 weight of the high colour at each pixel for the ramp `kind` of the node's p0 and p1."""
    t = np.clip(ramp_position(p, x0, y0, w, h), 0.0, 1.0)
    return ease(t, kind)


def ramp_colour(p, x0, y0, w, h, kind, high):
    """Straight RGBA across the area: `color0_*` at point 0 to `high` (r, g, b, a) at point 1."""
    weight = ramp_weight(p, x0, y0, w, h, kind)[..., None]
    low = np.array([p.get("color0_red", 0.0), p.get("color0_green", 0.0), p.get("color0_blue", 0.0), p.get("color0_alpha", 1.0)],
                   dtype=np.float64)
    return low[None, None, :] * (1.0 - weight) + np.asarray(high, dtype=np.float64)[None, None, :] * weight
