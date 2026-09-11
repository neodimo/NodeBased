"""Deterministic CPU rasterisation of resolved roto shapes into a premultiplied matte.

Consumes the *resolved* shapes `nodebased.shapes.resolve_shapes` produces — plain numbers, no
animatable-scalar objects — so this module never has to know what a curve is. That separation is
why the cache can digest the resolved payload and get a static shape's entry to survive a scrub.

Output convention (docs/ROTO_TRACKING.md): float32, scene-linear, premultiplied, `rgb == a ==
coverage`. A premultiplied white matte composites correctly through every existing kernel with no
special case and reads as sensible greyscale in the viewer.

Two limitations are deliberate and recorded rather than hidden:

* Vertical antialiasing is quantised to `1 / SUBSAMPLES` while horizontal coverage is exact and
  continuous. The scanline fill takes `SUBSAMPLES` sub-scanlines per pixel row and computes exact
  fractional overlap along each of them.
* Feather is a separable box blur of the shape's coverage, which softens symmetrically about the
  edge instead of growing the shape outward the way a true offset-curve feather does. A feather
  wider than a thin shape's waist will thin that shape. Nuke does not behave this way.
"""
from __future__ import annotations

import math

import numpy as np

# Sub-scanlines per pixel row. Vertical coverage quantises to 1/SUBSAMPLES; horizontal is exact.
SUBSAMPLES = 8

# Cubic spans flatten to between MIN and MAX segments, chosen from the control polygon's length so
# that identical geometry always flattens identically and the cache digest stays meaningful.
MIN_SEGMENTS = 4
MAX_SEGMENTS = 64
SEGMENT_PIXELS = 3.0


def flatten_points(points):
    """Flatten a closed list of resolved bezier points to a polyline of (x, y) vertices.

    Tangent handles are stored relative to their point (the Nuke/After Effects convention), so a
    span whose outgoing and incoming handles are both zero is a straight line and emits exactly one
    segment. That is what lets a polygon be a bezier with no handles rather than a second shape
    type — no `is_polygon` flag, no divergent code path.
    """
    count = len(points)
    vertices = []
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % count]
        x0, y0 = start["x"], start["y"]
        x3, y3 = end["x"], end["y"]
        x1, y1 = x0 + start["out_x"], y0 + start["out_y"]
        x2, y2 = x3 + end["in_x"], y3 + end["in_y"]
        vertices.append((x0, y0))
        if start["out_x"] == 0.0 and start["out_y"] == 0.0 and end["in_x"] == 0.0 and end["in_y"] == 0.0:
            # Straight span: the closing polyline edge from this vertex to the next is the segment.
            continue
        control = (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x1, y2 - y1)
                   + math.hypot(x3 - x2, y3 - y2))
        segments = min(max(int(math.ceil(control / SEGMENT_PIXELS)), MIN_SEGMENTS), MAX_SEGMENTS)
        for step in range(1, segments):
            t = step / segments
            u = 1.0 - t
            vertices.append((u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * x3,
                             u * u * u * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y3))
    return np.asarray(vertices, dtype=np.float64)


def _add_span(row, x0, x1):
    """Accumulate exact horizontal coverage of the interval [x0, x1) into one pixel row."""
    width = row.shape[0]
    x0 = min(max(x0, 0.0), float(width))
    x1 = min(max(x1, 0.0), float(width))
    if x1 <= x0:
        return
    first = int(math.floor(x0))
    last = int(math.floor(x1))
    if first >= width:
        return
    if first == last:
        row[first] += x1 - x0
        return
    row[first] += (first + 1) - x0
    if last > first + 1:
        row[first + 1:last] += 1.0
    if last < width:
        row[last] += x1 - last


def coverage(points, width, height):
    """Scanline-fill one shape's resolved points into an HxW coverage array in [0, 1].

    Non-zero winding, so a self-intersecting shape fills the way every other package fills it and
    a hole is drawn by winding a sub-loop the other way.
    """
    matte = np.zeros((height, width), dtype=np.float64)
    polyline = flatten_points(points)
    if polyline.shape[0] < 3:
        return matte.astype(np.float32)
    ax, ay = polyline[:, 0], polyline[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    # A horizontal edge never crosses a sub-scanline; dropping it also keeps the slope finite.
    rising = ay != by
    ax, ay, bx, by = ax[rising], ay[rising], bx[rising], by[rising]
    if ax.size == 0:
        return matte.astype(np.float32)
    direction = np.where(by > ay, 1, -1)
    top, bottom = np.minimum(ay, by), np.maximum(ay, by)
    slope = (bx - ax) / (by - ay)
    # Only rows the shape actually touches are walked; an off-screen shape costs nothing.
    first_row = max(0, int(math.floor(top.min())))
    last_row = min(height, int(math.ceil(bottom.max())) + 1)
    for row in range(first_row, last_row):
        accumulator = matte[row]
        for sub in range(SUBSAMPLES):
            scanline = row + (sub + 0.5) / SUBSAMPLES
            hit = (top <= scanline) & (scanline < bottom)
            if not hit.any():
                continue
            crossings = ax[hit] + (scanline - ay[hit]) * slope[hit]
            winding = direction[hit]
            order = np.argsort(crossings, kind="stable")
            crossings, winding = crossings[order], winding[order]
            running = np.cumsum(winding)
            for index in np.nonzero(running[:-1] != 0)[0]:
                _add_span(accumulator, crossings[index], crossings[index + 1])
    return (matte / SUBSAMPLES).astype(np.float32)


def feather(cover, radius):
    """Soften coverage by a separable box blur. See the module docstring for what this is not."""
    if radius < 0.5:
        return cover
    # Reuse the Blur node's own filter rather than growing a second box-blur implementation that
    # can drift from it. Imported here because imaging imports this module for the Roto kernel.
    from .imaging import Evaluator
    return Evaluator._box_blur_axis(Evaluator._box_blur_axis(cover, radius, axis=1), radius, axis=0)


def rasterise(shapes, width, height, invert=False):
    """Rasterise resolved shapes into a premultiplied HxWx4 float32 matte."""
    matte = np.zeros((int(height), int(width)), dtype=np.float32)
    for shape in shapes:
        cover = feather(coverage(shape["points"], int(width), int(height)), shape["feather"])
        # Clip before combining: box-blur round-off can leave a coverage a hair outside [0, 1], and
        # both combine operators only stay in range for inputs that are in range.
        cover = np.clip(cover, 0.0, 1.0) * np.float32(shape["opacity"])
        if shape["mode"] == "union":
            matte = matte + cover - matte * cover
        else:
            matte = matte * (1.0 - cover)
    if invert:
        matte = 1.0 - matte
    # rgb == a == coverage: a premultiplied white matte.
    return np.repeat(matte[..., None], 4, axis=2).astype(np.float32)
