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


def flatten_points(points, closed=True):
    """Flatten resolved Bezier points to a polyline of (x, y) vertices.

    Tangent handles are stored relative to their point (the Nuke/After Effects convention), so a
    span whose outgoing and incoming handles are both zero is a straight line and emits exactly one
    segment. That is what lets a polygon be a bezier with no handles rather than a second shape
    type — no `is_polygon` flag, no divergent code path.
    """
    count = len(points)
    vertices = []
    for index in range(count if closed else count - 1):
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
    if not closed and count:
        vertices.append((points[-1]["x"], points[-1]["y"]))
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


def _flatten_cached(points, cache):
    if cache is None:
        return flatten_points(points)
    key = id(points)
    vertices = cache.get(key)
    if vertices is None:
        vertices = flatten_points(points)
        cache[key] = vertices
    return vertices


def coverage(points, width, height, origin=(0, 0), *, flattened_cache=None):
    """Scanline-fill one shape's resolved points into an HxW coverage array in [0, 1].

    Non-zero winding, so a self-intersecting shape fills the way every other package fills it and
    a hole is drawn by winding a sub-loop the other way.
    """
    matte = np.zeros((height, width), dtype=np.float64)
    polyline = _flatten_cached(points, flattened_cache)
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
    ox, oy = map(int, origin)
    first_row = max(0, int(math.floor(top.min() - oy)))
    last_row = min(height, int(math.ceil(bottom.max() - oy)) + 1)
    for row in range(first_row, last_row):
        accumulator = matte[row]
        for sub in range(SUBSAMPLES):
            scanline = oy + row + (sub + 0.5) / SUBSAMPLES
            hit = (top <= scanline) & (scanline < bottom)
            if not hit.any():
                continue
            crossings = ax[hit] + (scanline - ay[hit]) * slope[hit]
            winding = direction[hit]
            order = np.argsort(crossings, kind="stable")
            crossings, winding = crossings[order], winding[order]
            running = np.cumsum(winding)
            for index in np.nonzero(running[:-1] != 0)[0]:
                _add_span(accumulator, crossings[index] - ox, crossings[index + 1] - ox)
    return (matte / SUBSAMPLES).astype(np.float32)


def stroke_coverage(points, width, height, stroke_width, feather=0.0, falloff="linear", origin=(0, 0)):
    """Rasterise an open cubic as round-capped, round-joined stroke coverage."""
    h, w = int(height), int(width)
    vertices = []
    offsets = []
    for index in range(len(points) - 1):
        start, end = points[index], points[index + 1]
        x0, y0 = start["x"], start["y"]
        x1, y1 = x0 + start["out_x"], y0 + start["out_y"]
        x2, y2 = end["x"] + end["in_x"], end["y"] + end["in_y"]
        x3, y3 = end["x"], end["y"]
        control = (math.hypot(x1-x0, y1-y0) + math.hypot(x2-x1, y2-y1) + math.hypot(x3-x2, y3-y2))
        n = 1 if start["out_x"] == start["out_y"] == end["in_x"] == end["in_y"] == 0 else min(max(int(math.ceil(control / SEGMENT_PIXELS)), MIN_SEGMENTS), MAX_SEGMENTS)
        a = np.asarray((start.get("feather_x", 0.0), start.get("feather_y", 0.0)), dtype=np.float64)
        b = np.asarray((end.get("feather_x", 0.0), end.get("feather_y", 0.0)), dtype=np.float64)
        for step in range(n):
            t = step / n; u = 1.0 - t
            vertices.append((u**3*x0+3*u*u*t*x1+3*u*t*t*x2+t**3*x3,
                             u**3*y0+3*u*u*t*y1+3*u*t*t*y2+t**3*y3))
            offsets.append(a*(1-t)+b*t)
    vertices.append((points[-1]["x"], points[-1]["y"]))
    offsets.append(np.asarray((points[-1].get("feather_x", 0.0), points[-1].get("feather_y", 0.0))))
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 2:
        return np.zeros((h, w), dtype=np.float32)
    ox, oy = map(int, origin)
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx.astype(np.float32) + ox + .5, yy.astype(np.float32) + oy + .5
    best = np.full((h, w), np.inf, dtype=np.float32)
    local_feather = np.zeros((h, w), dtype=np.float32)
    for index, (p0, p1) in enumerate(zip(vertices[:-1], vertices[1:])):
        vx, vy = p1 - p0
        denom = vx*vx + vy*vy
        if denom <= 1e-12:
            continue
        t = np.clip(((px-p0[0])*vx + (py-p0[1])*vy) / denom, 0.0, 1.0)
        dist = np.hypot(px-(p0[0]+t*vx), py-(p0[1]+t*vy))
        take = dist < best
        normal = np.asarray((vy, -vx), dtype=np.float64) / math.sqrt(denom)
        offset = np.abs((offsets[index][0] * (1-t) + offsets[index+1][0] * t) * normal[0] +
                        (offsets[index][1] * (1-t) + offsets[index+1][1] * t) * normal[1])
        np.copyto(best, dist, where=take)
        np.copyto(local_feather, offset, where=take)
    half = max(0.0, float(stroke_width)) * .5
    extent = float(feather) + local_feather
    alpha = np.where(extent >= 0,
                     np.where(best <= half, 1.0,
                              np.clip((half + extent + .5 - best) / np.maximum(extent, 1e-6), 0.0, 1.0)),
                     np.clip(half + extent + .5 - best, 0.0, 1.0)).astype(np.float32)
    if falloff == "smooth":
        alpha = alpha*alpha*(3.0-2.0*alpha)
    elif falloff == "gaussian":
        alpha = np.exp(-4.0*np.square(1.0-alpha)).astype(np.float32)
    return alpha.astype(np.float32)


def feather(cover, radius):
    """Soften coverage by a separable box blur. See the module docstring for what this is not."""
    if radius < 0.5:
        return cover
    # Reuse the Blur node's own filter rather than growing a second box-blur implementation that
    # can drift from it. Imported here because imaging imports this module for the Roto kernel.
    from .imaging import Evaluator
    return Evaluator._box_blur_axis(Evaluator._box_blur_axis(cover, radius, axis=1), radius, axis=0)


def edge_feather(points, cover, feather, falloff="linear", origin=(0, 0), *, flattened_cache=None):
    """Rasterise a signed-distance feather with linearly interpolated vertex offsets.

    Work is restricted to each segment's feather-expanded bounds. This keeps the common small-shape
    case proportional to outline length rather than image area and requires only NumPy.
    """
    height, width = cover.shape
    ox, oy = map(int, origin)
    vertices = _flatten_cached(points, flattened_cache)
    if len(vertices) < 3 or feather == 0 and not any(
            p.get("feather_x", 0) or p.get("feather_y", 0) for p in points):
        return cover
    # Match the adaptive Bezier flattening above; offsets follow the corresponding outline spans.
    offsets = []
    area = float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) -
                        np.roll(vertices[:, 0], -1) * vertices[:, 1]))
    winding = 1.0 if area >= 0 else -1.0
    for i, point in enumerate(points):
        end = points[(i + 1) % len(points)]
        x0, y0 = point["x"], point["y"]
        x1, y1 = x0 + point["out_x"], y0 + point["out_y"]
        x2, y2 = end["x"] + end["in_x"], end["y"] + end["in_y"]
        x3, y3 = end["x"], end["y"]
        control = (math.hypot(x1-x0, y1-y0) + math.hypot(x2-x1, y2-y1) + math.hypot(x3-x2, y3-y2))
        n = 1 if point["out_x"] == point["out_y"] == end["in_x"] == end["in_y"] == 0 else min(max(int(math.ceil(control / SEGMENT_PIXELS)), MIN_SEGMENTS), MAX_SEGMENTS)
        a = float(point.get("feather_x", 0.0)); b = float(point.get("feather_y", 0.0))
        c = float(end.get("feather_x", 0.0)); d = float(end.get("feather_y", 0.0))
        # The pair is an artist-space vector; its projection onto the local normal changes the
        # signed offset. Segment normals are computed below so x/y handles remain intuitive.
        for step in range(n):
            t = step / n
            u = 1.0 - t
            if n == 1:
                q = np.asarray((x0, y0), dtype=np.float64)
                q2 = np.asarray((x3, y3), dtype=np.float64)
            else:
                q = np.asarray((u**3*x0+3*u*u*t*x1+3*u*t*t*x2+t**3*x3,
                                u**3*y0+3*u*u*t*y1+3*u*t*t*y2+t**3*y3))
                v = (step+1)/n; z = 1-v
                q2 = np.asarray((z**3*x0+3*z*z*v*x1+3*z*v*v*x2+v**3*x3,
                                 z**3*y0+3*z*z*v*y1+3*z*v*v*y2+v**3*y3))
            tangent = q2-q
            length = float(np.linalg.norm(tangent))
            if length < 1e-10:
                continue
            normal = winding * np.asarray((tangent[1], -tangent[0])) / length
            start_t, end_t = step / n, (step + 1) / n
            start_delta = float(np.dot(np.asarray((a*(1-start_t)+c*start_t,
                                                    b*(1-start_t)+d*start_t)), normal))
            end_delta = float(np.dot(np.asarray((a*(1-end_t)+c*end_t,
                                                  b*(1-end_t)+d*end_t)), normal))
            offsets.append((q, q2, start_delta, end_delta))
    if not offsets:
        return cover
    radius = abs(float(feather)) + max(max(abs(item[2]), abs(item[3])) for item in offsets)
    if radius <= 0:
        return cover
    best = np.full((height, width), np.inf, dtype=np.float32)
    local_radius = np.full((height, width), float(feather), dtype=np.float32)
    for p0, p1, delta0, delta1 in offsets:
        left = max(ox, int(math.floor(min(p0[0], p1[0])-radius-1)))
        right = min(ox+width, int(math.ceil(max(p0[0], p1[0])+radius+1)))
        top = max(oy, int(math.floor(min(p0[1], p1[1])-radius-1)))
        bottom = min(oy+height, int(math.ceil(max(p0[1], p1[1])+radius+1)))
        if left >= right or top >= bottom:
            continue
        yy, xx = np.mgrid[top:bottom, left:right]
        vx, vy = p1-p0
        denom = vx*vx+vy*vy
        t = np.clip(((xx+.5-p0[0])*vx+(yy+.5-p0[1])*vy)/denom, 0.0, 1.0)
        dx = xx+.5-(p0[0]+t*vx); dy = yy+.5-(p0[1]+t*vy)
        dist2 = dx*dx+dy*dy
        view = best[top-oy:bottom-oy,left-ox:right-ox]
        take = dist2 < view
        np.copyto(view, dist2, where=take)
        rv = local_radius[top-oy:bottom-oy,left-ox:right-ox]
        np.copyto(rv, float(feather)+delta0+(delta1-delta0)*t, where=take)
    distance = np.sqrt(best, dtype=np.float32)
    signed = np.where(cover > 0.0, -distance, distance)
    positive = local_radius >= 0
    alpha = np.zeros_like(cover, dtype=np.float32)
    ramp = np.zeros_like(cover, dtype=np.float32)
    np.divide(signed, local_radius, out=ramp, where=positive & (local_radius > 0))
    alpha[positive] = np.clip(1.0-ramp[positive], 0.0, 1.0)
    negative = ~positive
    np.divide(signed, np.abs(local_radius), out=ramp, where=negative & (local_radius < 0))
    alpha[negative] = np.clip(-ramp[negative], 0.0, 1.0)
    if falloff == "smooth":
        alpha = alpha*alpha*(3.0-2.0*alpha)
    elif falloff == "gaussian":
        alpha = np.exp(-4.0*np.square(1.0-alpha)).astype(np.float32)
    alpha[positive & (signed >= np.abs(local_radius))] = 0.0
    alpha[negative & (signed >= 0.0)] = 0.0
    # Preserve subpixel coverage on the original contour; the feather interior is fully solid.
    alpha = np.where(positive & (distance < 0.75) & (cover > 0), np.maximum(alpha, cover), alpha)
    return alpha.astype(np.float32)


def _shape_bounds(shape):
    """Conservative pixel-space bounds, including Bezier control hull and edge expansion."""
    cached = shape.get("_raster_bounds")
    if cached is not None:
        return cached
    points = [point for layer in [shape.get("points", []), *shape.get("_motion_samples", ())]
              for point in layer]
    if not points:
        shape["_raster_bounds"] = (0.0, 0.0, 0.0, 0.0)
        return shape["_raster_bounds"]
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    xs += [float(p["x"]) + float(p.get(k, 0.0)) for p in points for k in ("in_x", "out_x")]
    ys += [float(p["y"]) + float(p.get(k, 0.0)) for p in points for k in ("in_y", "out_y")]
    pad = abs(float(shape.get("feather", 0.0)))
    pad += max((math.hypot(float(p.get("feather_x", 0.0)),
                          float(p.get("feather_y", 0.0))) for p in points), default=0.0)
    if not shape.get("closed", True):
        pad += max(0.0, float(shape.get("stroke_width", 1.0))) * .5 + 1.0
    else:
        pad += 1.0
    shape["_raster_bounds"] = min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad
    return shape["_raster_bounds"]


def rasterise(shapes, width, height, invert=False, *, region=None, stats=None, flattened_cache=None,
              shape_cache=None):
    """Rasterise resolved shapes into a premultiplied RGBA float32 image."""
    h, w = int(height), int(width)
    if region is None:
        rx, ry, rw, rh = 0, 0, w, h
    else:
        rx, ry, rw, rh = map(int, region)
        if rw < 0 or rh < 0 or rx < 0 or ry < 0 or rx + rw > w or ry + rh > h:
            raise ValueError("Roto region must lie inside the image")
    if flattened_cache is None:
        flattened_cache = {}
    # The legacy feather is a pair of separable box filters. Evaluate its complete support
    # around the request so cropped and full-frame pixels stay identical at tile edges.
    box_support = max((int(round(max(0.0, float(s.get("feather", 0.0)))))
                       for s in shapes if s.get("closed", True)
                       and s.get("feather_model", "box") == "box"), default=0)
    pad = box_support
    x0, y0 = max(0, rx - pad), max(0, ry - pad)
    x1, y1 = min(w, rx + rw + pad), min(h, ry + rh + pad)
    tw, th = x1 - x0, y1 - y0
    matte = np.zeros((th, tw), dtype=np.float32)
    rgb = np.zeros((th, tw, 3), dtype=np.float32)
    roi = (x0, y0, tw, th)
    if region is None:
        x0, y0, tw, th = 0, 0, w, h
        roi = (x0, y0, tw, th)
        matte = np.zeros((th, tw), dtype=np.float32)
        rgb = np.zeros((th, tw, 3), dtype=np.float32)
    rx0, ry0, rx1, ry1 = roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]
    for shape in shapes:
        if not shape.get("visible", True):
            continue
        # A full-frame request used to allocate, fill and feather a full-frame coverage plane
        # once per shape. The shape's conservative bounds already include its feather/stroke
        # expansion, so rasterise just that patch and composite it into the working canvas.
        # Inverted shapes are the exception: their coverage outside the outline is non-zero and
        # therefore affects every pixel.
        inverted = bool(shape.get("invert", False))
        if region is not None and not inverted:
            bx0, by0, bx1, by1 = _shape_bounds(shape)
            if bx1 <= rx0 or bx0 >= rx1 or by1 <= ry0 or by0 >= ry1:
                if stats is not None:
                    stats["skipped_shapes"] = stats.get("skipped_shapes", 0) + 1
                continue
        if inverted:
            sx0, sy0, sx1, sy1 = x0, y0, x1, y1
        else:
            bx0, by0, bx1, by1 = _shape_bounds(shape)
            sx0, sy0 = max(x0, int(math.floor(bx0))), max(y0, int(math.floor(by0)))
            sx1, sy1 = min(x1, int(math.ceil(bx1))), min(y1, int(math.ceil(by1)))
            if sx1 <= sx0 or sy1 <= sy0:
                if stats is not None:
                    stats["skipped_shapes"] = stats.get("skipped_shapes", 0) + 1
                continue
        sw, sh = sx1 - sx0, sy1 - sy0
        if stats is not None:
            stats["rasterised_shapes"] = stats.get("rasterised_shapes", 0) + 1
        samples = shape.get("_motion_samples", ())
        falloff = shape.get("feather_falloff", "linear")

        def render_cover():
            if not shape.get("closed", True):
                layers = [shape["points"], *samples]
                covers = [stroke_coverage(points, sw, sh, shape.get("stroke_width", 1.0),
                                          shape["feather"], falloff, origin=(sx0, sy0))
                          for points in layers]
                result = sum(covers) / len(covers)
            elif shape.get("feather_model", "box") == "edge":
                layers = [shape["points"], *samples]
                covers = [edge_feather(points, coverage(points, sw, sh, origin=(sx0, sy0),
                                       flattened_cache=flattened_cache), shape["feather"], falloff,
                                       origin=(sx0, sy0), flattened_cache=flattened_cache)
                          for points in layers]
                result = sum(covers) / len(covers)
            else:
                result = coverage(shape["points"], sw, sh, origin=(sx0, sy0),
                                  flattened_cache=flattened_cache)
                if samples:
                    result = (result + sum(coverage(points, sw, sh, origin=(sx0, sy0),
                                                    flattened_cache=flattened_cache)
                                           for points in samples)) / (len(samples) + 1)
                result = np.clip(feather(result, shape["feather"]), 0.0, 1.0)
                if falloff == "smooth":
                    result = result * result * (3.0 - 2.0 * result)
                elif falloff == "gaussian":
                    result = np.exp(-4.0 * np.square(1.0 - result)).astype(np.float32)
            return 1.0 - result if inverted else result

        cover = (render_cover() if shape_cache is None else
                 shape_cache(shape, (sx0, sy0, sw, sh), (w, h), render_cover))
        color = np.asarray(shape.get("color", [1, 1, 1, 1]), dtype=np.float32)
        alpha = cover * np.float32(shape["opacity"]) * color[3]
        source = color[:3] * alpha[..., None]
        py0, py1 = sy0 - y0, sy1 - y0
        px0, px1 = sx0 - x0, sx1 - x0
        matte_patch = matte[py0:py1, px0:px1]
        rgb_patch = rgb[py0:py1, px0:px1]
        # The legacy shape operation remains the coverage combine; the Nuke blend knob controls
        # colour/alpha compositing independently and defaults to over for old documents.
        if shape["mode"] == "subtract":
            matte_patch *= 1.0 - alpha
            rgb_patch *= 1.0 - alpha[..., None]
            continue
        blend = shape.get("blend_mode", "over")
        if blend == "plus":
            rgb_patch[:] = np.clip(rgb_patch + source, 0.0, 1.0)
            matte_patch[:] = np.clip(matte_patch + alpha, 0.0, 1.0)
        elif blend == "minus":
            rgb_patch[:] = np.clip(rgb_patch - source, 0.0, 1.0)
            matte_patch[:] = np.clip(matte_patch - alpha, 0.0, 1.0)
        elif blend == "multiply":
            rgb_patch *= 1.0 - alpha[..., None] + color[:3] * alpha[..., None]
            matte_patch[:] = alpha + matte_patch * (1.0 - alpha)
        else:
            rgb_patch[:] = source + rgb_patch * (1.0 - alpha[..., None])
            matte_patch[:] = alpha + matte_patch * (1.0 - alpha)
    if invert:
        matte = 1.0 - matte
    pixels = np.concatenate((rgb, matte[..., None]), axis=2).astype(np.float32)
    return np.ascontiguousarray(pixels[ry-y0:ry-y0+rh, rx-x0:rx-x0+rw])
