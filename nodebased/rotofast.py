"""The faster Roto and RotoPaint matte drawer.

`nodebased.roto` keeps the shape model, the cache contract and the old drawer; this module holds the kernels
that replace the old drawer's heavy loops, with the same arithmetic and a much smaller bill:

* **Closed shapes** (`coverage`). The old drawer sorted every edge against every sub-scanline of a band, a
  dense ``(sub-scanlines x edges)`` problem repeated over the whole height of the shape. Here each edge
  produces only the crossings it really has (an edge spans a few sub-scanlines, not all of them), the
  crossings are sorted once, the non-zero winding is one running sum, and the spans of all sub-scanlines are
  folded into their pixel row as they are accumulated. The sample positions, the winding rule and the
  horizontal clip are the old drawer's, so the coverage agrees with it to float rounding (about 1e-7); the
  dense work left is one running sum over the shape's own rows.
* **Open strokes** (`stroke_coverage`). Every segment was measured against every pixel of the patch. A
  segment now touches only its own bounding box widened by the stroke's reach, taken once for the whole
  stroke so the nearest segment of every pixel that can be lit is still the one that decides it.
* **Signed-distance feather** (`edge_feather`). The per-segment normal and offset set-up is one array
  operation per span instead of a Python loop of small ``numpy`` calls, and each segment's distance field is
  evaluated on its bounding box from two one-dimensional axes instead of two full ``mgrid`` planes.

Everything heavy is a ``numpy`` call over a large array, which releases the interpreter lock, so a worker
drawing a matte does not hold the thread that evaluates the frame while it works. The old drawer stays in
`nodebased.roto` and is chosen with ``NODEBASED_ROTO_DRAWER=legacy`` (or `roto.set_drawer("legacy")`) for one
release.
"""
from __future__ import annotations

import math

import numpy as np

SUBSAMPLES = 8


def _first_at_or_above(value, origin, lower):
    """For each edge bound: the first sub-scanline index ``s`` whose centre ``origin + (s + 0.5) / 8`` is not
    below ``value`` (``lower`` is the smallest allowed index), by the same comparison the old drawer made."""
    s = np.ceil((value - origin) * SUBSAMPLES - 0.5)
    s = np.clip(s, -1e9, 1e9).astype(np.int64)
    centre = lambda index: origin + (index + 0.5) / SUBSAMPLES        # noqa: E731 - exact binary fractions
    s = np.where(centre(s - 1) >= value, s - 1, s)
    s = np.where(centre(s) < value, s + 1, s)
    return np.maximum(s, lower)


def coverage(polyline, width, height, origin=(0, 0)):
    """Non-zero-winding coverage of a closed polyline in an ``height x width`` float32 array in [0, 1].

    Horizontal coverage is exact, vertical coverage is sampled on ``SUBSAMPLES`` sub-scanlines per row (the
    old drawer's rule, kept so the two agree)."""
    width, height = int(width), int(height)
    matte = np.zeros((height, width), dtype=np.float32)
    if polyline.shape[0] < 3 or width <= 0 or height <= 0:
        return matte
    ax, ay = polyline[:, 0], polyline[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    rising = ay != by
    ax, ay, bx, by = ax[rising], ay[rising], bx[rising], by[rising]
    if ax.size == 0:
        return matte
    ox, oy = map(int, origin)
    direction = np.where(by > ay, 1, -1).astype(np.int8)
    top, bottom = np.minimum(ay, by), np.maximum(ay, by)
    slope = (bx - ax) / (by - ay)
    limit = height * SUBSAMPLES
    first = _first_at_or_above(top, float(oy), 0)
    last = np.minimum(_first_at_or_above(bottom, float(oy), 0), limit)       # exclusive
    count = np.maximum(last - first, 0)
    total = int(count.sum())
    if total == 0:
        return matte
    # One crossing per (edge, sub-scanline) pair the edge covers.
    edge = np.repeat(np.arange(ax.size), count)
    starts = np.cumsum(count) - count
    scan = np.arange(total, dtype=np.int64) - np.repeat(starts, count) + np.repeat(first, count)
    y = oy + (scan + 0.5) / SUBSAMPLES
    x = ax[edge] + (y - ay[edge]) * slope[edge]
    np.subtract(x, ox, out=x)
    np.clip(x, 0.0, float(width), out=x)
    # Sorting by (sub-scanline, x) in one key: x is inside [0, width], so the key never mixes two lines.
    key = scan * float(width + 2) + x
    order = np.argsort(key)
    x = x[order]
    scan = scan[order]
    winding = np.cumsum(direction[edge][order], dtype=np.int32)
    inside = winding[:-1] != 0
    left, right = x[:-1], x[1:]
    valid = inside & (right > left) & (left < width)
    if not valid.any():
        return matte
    line = scan[:-1][valid] // SUBSAMPLES                       # the pixel row each span belongs to (ascending)
    x0, x1 = left[valid], right[valid]
    f0, f1 = np.floor(x0).astype(np.intp), np.floor(x1).astype(np.intp)
    # Bands of rows keep the float64 accumulators small however tall the shape is.
    band = max(8, BAND_CELLS // max(width, 1))
    row = int(line[0])
    end_row = int(line[-1]) + 1
    while row < end_row:
        stop = min(end_row, row + band)
        lo, hi = np.searchsorted(line, (row, stop))
        if hi > lo:
            _accumulate(matte, row, stop, line[lo:hi] - row, x0[lo:hi], x1[lo:hi], f0[lo:hi], f1[lo:hi], width)
        row = stop
    return matte


# float64 cells one accumulation band holds (16 MB each for the three arrays at this size).
BAND_CELLS = 1 << 21


def _accumulate(matte, row, stop, line, x0, x1, f0, f1, width):
    """Exact horizontal coverage of the spans ``[x0, x1)`` of one band of pixel rows, written into ``matte``."""
    rows = stop - row
    same = f0 == f1
    # Fractional end pixels (partial) and a difference array for the whole pixels between them (run).
    tail = ~same & (f1 < width)
    partial = np.bincount(
        np.concatenate((line * width + f0, line[tail] * width + f1[tail])),
        weights=np.concatenate((np.where(same, x1 - x0, f0 + 1 - x0), x1[tail] - f1[tail])),
        minlength=rows * width)
    full = f0 + 1 < f1
    stride = width + 1
    count = int(full.sum())
    if count:
        run = np.bincount(
            np.concatenate((line[full] * stride + f0[full] + 1, line[full] * stride + f1[full])),
            weights=np.concatenate((np.ones(count), -np.ones(count))), minlength=rows * stride)
        area = np.cumsum(run.reshape(rows, stride)[:, :width], axis=1)
        area += partial.reshape(rows, width)
    else:
        area = partial.reshape(rows, width)
    area *= 1.0 / SUBSAMPLES
    matte[row:stop] = area


def _spans(points, closed=False):
    """Per-span arrays of a point list: for each cubic span its end points, handle points, feather vectors and
    the number of segments it flattens to (the same rule as `roto.flatten_points`)."""
    from .roto import MAX_SEGMENTS, MIN_SEGMENTS, SEGMENT_PIXELS
    count = len(points)
    spans = count if closed else count - 1
    start = [points[i] for i in range(spans)]
    end = [points[(i + 1) % count] for i in range(spans)]
    get = lambda items, key: np.fromiter((item.get(key, 0.0) for item in items), np.float64, len(items))   # noqa: E731
    sx, sy, ex, ey = get(start, "x"), get(start, "y"), get(end, "x"), get(end, "y")
    p1x, p1y = sx + get(start, "out_x"), sy + get(start, "out_y")
    p2x, p2y = ex + get(end, "in_x"), ey + get(end, "in_y")
    straight = ((get(start, "out_x") == 0) & (get(start, "out_y") == 0)
                & (get(end, "in_x") == 0) & (get(end, "in_y") == 0))
    control = np.hypot(p1x - sx, p1y - sy) + np.hypot(p2x - p1x, p2y - p1y) + np.hypot(ex - p2x, ey - p2y)
    segments = np.where(straight, 1, np.clip(np.ceil(control / SEGMENT_PIXELS), MIN_SEGMENTS, MAX_SEGMENTS)).astype(np.int64)
    feathers = (get(start, "feather_x"), get(start, "feather_y"), get(end, "feather_x"), get(end, "feather_y"))
    return (sx, sy, p1x, p1y, p2x, p2y, ex, ey), segments, feathers


def _bezier(curves, t):
    """Points of cubic spans ``curves`` (eight arrays) at parameters ``t`` (one per row), the old formula."""
    x0, y0, x1, y1, x2, y2, x3, y3 = curves
    u = 1.0 - t
    return (u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
            u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3)


def stroke_polyline(points):
    """Flattened open stroke: vertices (n x 2) and the per-vertex feather vector (n x 2), as the old builder made them."""
    if len(points) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    curves, segments, (fa, fb, fc, fd) = _spans(points)
    span = np.repeat(np.arange(len(segments)), segments)
    first = np.cumsum(segments) - segments
    step = np.arange(span.size) - np.repeat(first, segments)
    t = step / segments[span]
    x, y = _bezier(tuple(c[span] for c in curves), t)
    # A one-segment span is the straight line between its end points: exactly the formula at t = 0.
    vertices = np.column_stack((x, y))
    offsets = np.column_stack((fa[span] * (1 - t) + fc[span] * t, fb[span] * (1 - t) + fd[span] * t))
    last = points[-1]
    vertices = np.vstack((vertices, [[last["x"], last["y"]]]))
    offsets = np.vstack((offsets, [[last.get("feather_x", 0.0), last.get("feather_y", 0.0)]]))
    return vertices, offsets


def stroke_coverage(vertices, offsets, width, height, stroke_width, feather, falloff, origin, caps, first_segment_hint=None):
    """Alpha of an open polyline stroke: ``vertices`` (n x 2) and per-vertex feather ``offsets`` (n x 2).

    The old drawer measured every segment against the whole patch; here a segment is measured on its own
    bounding box widened by ``reach``, the farthest any pixel can be lit from the stroke (half the width plus
    the largest feather plus the half pixel), the same for every segment. A pixel nearer than ``reach`` to
    any segment lies in that segment's box, and the nearest segment is the one that wins, exactly as before;
    a pixel farther than ``reach`` from all of them is dark in the old drawer too."""
    h, w = int(height), int(width)
    ox, oy = map(int, origin)
    half = max(0.0, float(stroke_width)) * .5
    capped = tuple(caps) != ("rounded", "rounded")
    seg_start, seg_end = vertices[:-1], vertices[1:]
    vec = seg_end - seg_start
    denom_all = vec[:, 0] ** 2 + vec[:, 1] ** 2
    live = np.flatnonzero(denom_all > 1e-12)
    best = np.full((h, w), np.inf, dtype=np.float32)
    local_feather = np.zeros((h, w), dtype=np.float32)
    if live.size == 0:
        return _stroke_alpha(best, local_feather, half, feather, falloff)
    # Largest extent any pixel can take: the stroke feather plus the feather projected on a normal.
    norm_off = np.hypot(offsets[:, 0], offsets[:, 1])
    reach = half + abs(float(feather)) + float(norm_off.max()) + 1.0
    if capped:
        nearest = np.full((h, w), -1, dtype=np.int32)
        reach_map = np.zeros((h, w), dtype=np.float32)
        across = np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32) + np.float32(ox + .5)
    ys = np.arange(h, dtype=np.float32) + np.float32(oy + .5)
    for index in live:
        p0 = seg_start[index]
        p1 = seg_end[index]
        x_lo = max(0, int(math.floor(min(p0[0], p1[0]) - reach - ox - 1)))
        x_hi = min(w, int(math.ceil(max(p0[0], p1[0]) + reach - ox + 1)))
        y_lo = max(0, int(math.floor(min(p0[1], p1[1]) - reach - oy - 1)))
        y_hi = min(h, int(math.ceil(max(p0[1], p1[1]) + reach - oy + 1)))
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        vx, vy = vec[index]
        denom = denom_all[index]
        px = xs[None, x_lo:x_hi]
        py = ys[y_lo:y_hi, None]
        t_raw = ((px - p0[0]) * vx + (py - p0[1]) * vy) / denom
        t = np.clip(t_raw, 0.0, 1.0)
        # The distance stays float64 and is compared with the float32 running minimum, as the old drawer did: at a
        # vertex shared by two segments the tie is settled by that rounding, and the feather offset follows the winner.
        dist = np.hypot(px - (p0[0] + t * vx), py - (p0[1] + t * vy))
        view = best[y_lo:y_hi, x_lo:x_hi]
        take = dist < view
        length = math.sqrt(denom)
        normal = (vy / length, -vx / length)
        a, b = offsets[index], offsets[index + 1]
        offset = np.abs((a[0] * (1 - t) + b[0] * t) * normal[0] + (a[1] * (1 - t) + b[1] * t) * normal[1])
        np.copyto(view, dist, where=take)
        np.copyto(local_feather[y_lo:y_hi, x_lo:x_hi], offset.astype(np.float32), where=take)
        if capped:
            np.copyto(nearest[y_lo:y_hi, x_lo:x_hi], np.int32(index), where=take)
            overshoot = np.where(t_raw < 0.0, t_raw, np.where(t_raw > 1.0, t_raw - 1.0, 0.0)) * length
            np.copyto(reach_map[y_lo:y_hi, x_lo:x_hi], overshoot.astype(np.float32), where=take)
            sideways = np.abs((px - p0[0]) * vy - (py - p0[1]) * vx).astype(np.float32) / np.float32(length)
            np.copyto(across[y_lo:y_hi, x_lo:x_hi], sideways, where=take)
    if capped:
        first_segment = int(live[0])
        last_segment = int(live[-1])
        for cap, segment, outside in ((caps[0], first_segment, reach_map < 0), (caps[1], last_segment, reach_map > 0)):
            if cap == "rounded":
                continue
            beyond = (nearest == segment) & outside
            if cap == "butt":
                best[beyond] = np.inf
            else:
                best[beyond] = np.where(np.abs(reach_map[beyond]) <= half, across[beyond], np.inf)
    return _stroke_alpha(best, local_feather, half, feather, falloff)


def _stroke_alpha(best, local_feather, half, feather, falloff):
    extent = float(feather) + local_feather
    with np.errstate(invalid="ignore"):
        alpha = np.where(extent >= 0,
                         np.where(best <= half, 1.0,
                                  np.clip((half + extent + .5 - best) / np.maximum(extent, 1e-6), 0.0, 1.0)),
                         np.clip(half + extent + .5 - best, 0.0, 1.0)).astype(np.float32)
    if falloff == "smooth":
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    elif falloff == "gaussian":
        alpha = np.exp(-4.0 * np.square(1.0 - alpha)).astype(np.float32)
    return alpha.astype(np.float32)


def edge_feather(points, vertices, cover, feather, falloff, origin):
    """Signed-distance feather of ``cover`` for a closed shape (see `roto.edge_feather` for the model).

    The per-span set-up (segment end points, normals and offsets) is array arithmetic over all segments at
    once. The nearest-segment search keeps the old rule, the first strictly nearest segment in outline
    order, but only records which segment won; the feather radius is then read once for every pixel from
    its winner instead of being written on every update."""
    height, width = cover.shape
    ox, oy = map(int, origin)
    if len(vertices) < 3:
        return cover
    curves, segments, (fa, fb, fc, fd) = _spans(points, closed=True)
    area = float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1]))
    winding = 1.0 if area >= 0 else -1.0
    span = np.repeat(np.arange(len(segments)), segments)
    first = np.cumsum(segments) - segments
    step = np.arange(span.size) - np.repeat(first, segments)
    n = segments[span]
    t0, t1 = step / n, (step + 1) / n
    span_curves = tuple(c[span] for c in curves)
    qx, qy = _bezier(span_curves, t0)
    q2x, q2y = _bezier(span_curves, t1)
    tx, ty = q2x - qx, q2y - qy
    length = np.hypot(tx, ty)
    keep = length >= 1e-10
    if not keep.any():
        return cover
    nx, ny = winding * ty / np.where(keep, length, 1.0), -winding * tx / np.where(keep, length, 1.0)

    def delta(s):
        return (fa[span] * (1 - s) + fc[span] * s) * nx + (fb[span] * (1 - s) + fd[span] * s) * ny
    d0, d1 = delta(t0), delta(t1)
    qx, qy, q2x, q2y, d0, d1 = (a[keep] for a in (qx, qy, q2x, q2y, d0, d1))
    radius = abs(float(feather)) + float(max(np.abs(d0).max(), np.abs(d1).max()))
    if radius <= 0:
        return cover
    vx, vy = q2x - qx, q2y - qy
    denom = vx * vx + vy * vy
    left = np.maximum(ox, np.floor(np.minimum(qx, q2x) - radius - 1)).astype(np.int64) - ox
    right = np.minimum(ox + width, np.ceil(np.maximum(qx, q2x) + radius + 1)).astype(np.int64) - ox
    top = np.maximum(oy, np.floor(np.minimum(qy, q2y) - radius - 1)).astype(np.int64) - oy
    bottom = np.minimum(oy + height, np.ceil(np.maximum(qy, q2y) + radius + 1)).astype(np.int64) - oy
    best = np.full((height, width), np.inf, dtype=np.float32)
    winner = np.full((height, width), -1, dtype=np.int32)
    xs = np.arange(width, dtype=np.float64) + (ox + .5)
    ys = (np.arange(height, dtype=np.float64) + (oy + .5))[:, None]
    for i in np.flatnonzero((left < right) & (top < bottom)):
        l, r, tp, b = left[i], right[i], top[i], bottom[i]
        px = xs[l:r]
        py = ys[tp:b]
        scale = 1.0 / denom[i]
        t = np.clip((px - qx[i]) * (vx[i] * scale) + (py - qy[i]) * (vy[i] * scale), 0.0, 1.0)
        dx = px - (qx[i] + t * vx[i])
        dy = py - (qy[i] + t * vy[i])
        dist2 = dx * dx + dy * dy
        view = best[tp:b, l:r]
        take = dist2 < view
        np.copyto(view, dist2, where=take)
        np.copyto(winner[tp:b, l:r], np.int32(i), where=take)
    lit = winner >= 0
    local_radius = np.full((height, width), float(feather), dtype=np.float32)
    if lit.any():
        rows, cols = np.nonzero(lit)
        w = winner[rows, cols]
        px, py = xs[cols], ys[rows, 0]
        t = np.clip(((px - qx[w]) * vx[w] + (py - qy[w]) * vy[w]) / denom[w], 0.0, 1.0)
        local_radius[rows, cols] = float(feather) + d0[w] + (d1[w] - d0[w]) * t
    distance = np.sqrt(best, dtype=np.float32)
    signed = np.where(cover > 0.0, -distance, distance)
    positive = local_radius >= 0
    alpha = np.zeros_like(cover, dtype=np.float32)
    ramp = np.zeros_like(cover, dtype=np.float32)
    np.divide(signed, local_radius, out=ramp, where=positive & (local_radius > 0))
    alpha[positive] = np.clip(1.0 - ramp[positive], 0.0, 1.0)
    negative = ~positive
    np.divide(signed, np.abs(local_radius), out=ramp, where=negative & (local_radius < 0))
    alpha[negative] = np.clip(-ramp[negative], 0.0, 1.0)
    if falloff == "smooth":
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    elif falloff == "gaussian":
        alpha = np.exp(-4.0 * np.square(1.0 - alpha)).astype(np.float32)
    alpha[positive & (signed >= np.abs(local_radius))] = 0.0
    alpha[negative & (signed >= 0.0)] = 0.0
    alpha = np.where(positive & (distance < 0.75) & (cover > 0), np.maximum(alpha, cover), alpha)
    return alpha.astype(np.float32)
