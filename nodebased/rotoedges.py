"""Image edges for Roto: snap a point to the nearest strong edge, fit a shape to the edges around it.

Nuke's Roto has neither. The artist's cost is placing points on the edge they meant, so this module
reads the image the Roto node sees at its input and offers two operations on it:

* ``snap(point, radius)`` moves a point to the strongest edge within ``radius`` pixels.
* ``fit(shape, band, iterations)`` refines every point of a closed or open spline to the edges
  within ``band`` pixels (an active contour: each point moves along its normal toward the strongest
  edge, a smoothness term keeps spacing and curvature close to the original).

Edge strength is the gradient magnitude of the **maximum over the RGB channels** of the image.
Rec. 709 luminance can cancel an edge between a saturated colour and a grey of the same luma (a
red subject on a green-grey wall has almost no luma edge); the maximum over RGB keeps nearly every
visible boundary, and for a grey plate it equals the luminance. The image is blurred with a
Gaussian (sigma 1 px by default) and differentiated with the Scharr kernel, normalised so that a
unit step gives a gradient of 1 per pixel.

Only a region of interest is ever computed: a window around the cursor for ``snap`` and a band
around the shape for ``fit``, aligned to a coarse grid so that nearby gestures hit the cache. The
cache key is (node, frame, input digest, region); an edited input changes the digest and misses.

Coordinates are Roto data coordinates: pixel (i, j) covers x in [j, j + 1) and y in [i, i + 1), so
its centre is (j + 0.5, i + 0.5). Pure numpy.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

SIGMA = 1.0
ALIGN = 32                 # regions are rounded out to this grid so neighbouring gestures share a field
ABSOLUTE_FLOOR = 0.02      # a gradient below this (image units per pixel) is never an edge
RELATIVE_THRESHOLD = 0.2   # ... nor one below this fraction of the region's gradient range
POINT_FIELDS = ("x", "y", "in_x", "in_y", "out_x", "out_y")


# -- gradient --------------------------------------------------------------------------------

def edge_source(image):
    """The scalar image edges are measured on: the maximum over RGB (see the module docstring)."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array
    channels = array[..., :3] if array.shape[2] >= 3 else array[..., :1]
    return channels.max(axis=2)


def _gaussian_kernel(sigma):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum(), radius


def _convolve_axis(array, kernel, axis):
    radius = len(kernel) // 2
    padded = np.pad(array, [(radius, radius) if a == axis else (0, 0) for a in range(2)], mode="edge")
    out = np.zeros_like(array)
    length = array.shape[axis]
    for offset, weight in enumerate(kernel):
        window = [slice(None), slice(None)]
        window[axis] = slice(offset, offset + length)
        out += weight * padded[tuple(window)]
    return out


def gradient(scalar, sigma=SIGMA):
    """Smoothed gradient (gx, gy) of a 2-D array: Gaussian, then Scharr; a unit step gives 1.0."""
    scalar = np.asarray(scalar, dtype=np.float32)
    kernel, _ = _gaussian_kernel(sigma)
    smooth = _convolve_axis(_convolve_axis(scalar, kernel, 0), kernel, 1)
    padded = np.pad(smooth, 1, mode="edge")
    # central difference across one axis, [3, 10, 3] / 16 smoothing across the other
    dx = (padded[:, 2:] - padded[:, :-2]) * 0.5
    gx = (3.0 * dx[:-2] + 10.0 * dx[1:-1] + 3.0 * dx[2:]) / 16.0
    dy = (padded[2:, :] - padded[:-2, :]) * 0.5
    gy = (3.0 * dy[:, :-2] + 10.0 * dy[:, 1:-1] + 3.0 * dy[:, 2:]) / 16.0
    return gx, gy


class EdgeField:
    """Gradient magnitude and components over one pixel box ``(x0, y0, x1, y1)`` of an image."""

    def __init__(self, gx, gy, box, sigma=SIGMA):
        self.gx = np.ascontiguousarray(gx, dtype=np.float32)
        self.gy = np.ascontiguousarray(gy, dtype=np.float32)
        self.mag = np.hypot(self.gx, self.gy).astype(np.float32)
        self.box = tuple(int(v) for v in box)
        self.sigma = sigma
        self._corner = None

    # -- geometry helpers ------------------------------------------------------------------
    @property
    def shape(self):
        return self.mag.shape

    def contains(self, x, y, margin=1.0):
        x0, y0, x1, y1 = self.box
        return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin

    def _sample(self, array, x, y):
        """Bilinear sample of `array` at data coordinates (pixel centres at +0.5); edge-clamped."""
        x0, y0, _, _ = self.box
        fx = np.asarray(x, dtype=np.float64) - x0 - 0.5
        fy = np.asarray(y, dtype=np.float64) - y0 - 0.5
        h, w = array.shape
        fx = np.clip(fx, 0.0, w - 1.0)
        fy = np.clip(fy, 0.0, h - 1.0)
        ix = np.minimum(fx.astype(np.int64), max(w - 2, 0))
        iy = np.minimum(fy.astype(np.int64), max(h - 2, 0))
        tx, ty = fx - ix, fy - iy
        ix1 = np.minimum(ix + 1, w - 1)
        iy1 = np.minimum(iy + 1, h - 1)
        top = array[iy, ix] * (1 - tx) + array[iy, ix1] * tx
        bottom = array[iy1, ix] * (1 - tx) + array[iy1, ix1] * tx
        return top * (1 - ty) + bottom * ty

    def magnitude_at(self, x, y):
        return self._sample(self.mag, x, y)

    def threshold(self):
        """The weakest gradient that counts as an edge in this region."""
        span = float(self.mag.max() - self.mag.min()) if self.mag.size else 0.0
        return max(ABSOLUTE_FLOOR, RELATIVE_THRESHOLD * span)

    # -- snap ------------------------------------------------------------------------------
    def snap(self, point, radius):
        """The strongest edge within `radius` px of `point` (sub-pixel), or `point` unchanged."""
        px, py = float(point[0]), float(point[1])
        radius = float(radius)
        x0, y0, x1, y1 = self.box
        h, w = self.mag.shape
        lo_x = max(0, int(math.floor(px - radius - x0 - 0.5)))
        hi_x = min(w, int(math.ceil(px + radius - x0 - 0.5)) + 1)
        lo_y = max(0, int(math.floor(py - radius - y0 - 0.5)))
        hi_y = min(h, int(math.ceil(py + radius - y0 - 0.5)) + 1)
        if lo_x >= hi_x or lo_y >= hi_y:
            return (px, py)
        mag = self.mag[lo_y:hi_y, lo_x:hi_x]
        gx = self.gx[lo_y:hi_y, lo_x:hi_x]
        gy = self.gy[lo_y:hi_y, lo_x:hi_x]
        xs = (np.arange(lo_x, hi_x) + x0 + 0.5)[None, :]
        ys = (np.arange(lo_y, hi_y) + y0 + 0.5)[:, None]
        distance = np.hypot(xs - px, ys - py)
        # A pixel is an edge location when it is a local maximum of the gradient along the gradient direction.
        norm = np.maximum(mag, 1e-9)
        dx, dy = gx / norm, gy / norm
        ahead = self._sample(self.mag, xs + dx, ys + dy)
        behind = self._sample(self.mag, xs - dx, ys - dy)
        peak = (mag >= ahead) & (mag >= behind)
        score = mag * (1.0 - 0.5 * distance / max(radius, 1e-6))
        score = np.where(peak & (distance <= radius) & (mag >= self.threshold()), score, -1.0)
        index = int(np.argmax(score))
        if score.flat[index] <= 0:
            return (px, py)
        iy, ix = divmod(index, score.shape[1])
        cx, cy = float(xs[0, ix]), float(ys[iy, 0])
        ux, uy = float(dx[iy, ix]), float(dy[iy, ix])
        m0 = float(mag[iy, ix])
        mp = float(self._sample(self.mag, cx + ux, cy + uy))
        mm = float(self._sample(self.mag, cx - ux, cy - uy))
        denom = mm - 2.0 * m0 + mp
        t = 0.0 if abs(denom) < 1e-12 else 0.5 * (mm - mp) / denom
        t = max(-1.0, min(1.0, t))
        return (cx + t * ux, cy + t * uy)

    # -- corners ---------------------------------------------------------------------------
    def corner_response(self):
        """Smallest eigenvalue of the smoothed structure tensor (large where two edges meet)."""
        if self._corner is None:
            kernel, _ = _gaussian_kernel(1.5)
            jxx = _convolve_axis(_convolve_axis(self.gx * self.gx, kernel, 0), kernel, 1)
            jyy = _convolve_axis(_convolve_axis(self.gy * self.gy, kernel, 0), kernel, 1)
            jxy = _convolve_axis(_convolve_axis(self.gx * self.gy, kernel, 0), kernel, 1)
            trace = jxx + jyy
            root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
            self._corner = np.maximum(0.5 * (trace - root), 0.0).astype(np.float32)
        return self._corner

    def snap_corner(self, point, radius):
        """The strongest corner within `radius` px of `point`, or None."""
        response = self.corner_response()
        if response.size == 0 or float(response.max()) <= 0.0:
            return None
        px, py = float(point[0]), float(point[1])
        x0, y0, _, _ = self.box
        h, w = response.shape
        lo_x = max(0, int(math.floor(px - radius - x0 - 0.5)))
        hi_x = min(w, int(math.ceil(px + radius - x0 - 0.5)) + 1)
        lo_y = max(0, int(math.floor(py - radius - y0 - 0.5)))
        hi_y = min(h, int(math.ceil(py + radius - y0 - 0.5)) + 1)
        if lo_x >= hi_x or lo_y >= hi_y:
            return None
        window = response[lo_y:hi_y, lo_x:hi_x]
        xs = (np.arange(lo_x, hi_x) + x0 + 0.5)[None, :]
        ys = (np.arange(lo_y, hi_y) + y0 + 0.5)[:, None]
        distance = np.hypot(xs - px, ys - py)
        floor = max(ABSOLUTE_FLOOR ** 2, 0.2 * float(response.max()) * 0.5)
        score = window * (1.0 - 0.5 * distance / max(radius, 1e-6))
        score = np.where((distance <= radius) & (window >= floor), score, -1.0)
        index = int(np.argmax(score))
        if score.flat[index] <= 0:
            return None
        iy, ix = divmod(index, score.shape[1])
        cx, cy = float(xs[0, ix]), float(ys[iy, 0])
        out = [cx, cy]
        for axis in (0, 1):
            step = [0.0, 0.0]; step[axis] = 1.0
            a = float(self._sample(response, cx - step[0], cy - step[1]))
            b = float(window[iy, ix])
            c = float(self._sample(response, cx + step[0], cy + step[1]))
            denom = a - 2.0 * b + c
            t = 0.0 if abs(denom) < 1e-12 else max(-1.0, min(1.0, 0.5 * (a - c) / denom))
            out[axis] += t
        return self._refine_corner((out[0], out[1]))

    def _refine_corner(self, guess, window=5.0):
        """Least-squares intersection of the edge lines through the pixels around `guess`.

        The structure-tensor peak sits a little inside a blurred corner; solving
        sum(g g^T) q = sum(g g^T p) over the neighbourhood (every pixel's gradient is normal to the
        edge through it) puts the corner where the edges meet.
        """
        x0, y0, _, _ = self.box
        h, w = self.mag.shape
        gx0 = int(max(0, math.floor(guess[0] - window - x0 - 0.5)))
        gx1 = int(min(w, math.ceil(guess[0] + window - x0 - 0.5) + 1))
        gy0 = int(max(0, math.floor(guess[1] - window - y0 - 0.5)))
        gy1 = int(min(h, math.ceil(guess[1] + window - y0 - 0.5) + 1))
        if gx0 >= gx1 or gy0 >= gy1:
            return guess
        gx = self.gx[gy0:gy1, gx0:gx1].astype(np.float64)
        gy = self.gy[gy0:gy1, gx0:gx1].astype(np.float64)
        xs, ys = np.meshgrid(np.arange(gx0, gx1) + x0 + 0.5, np.arange(gy0, gy1) + y0 + 0.5)
        keep = (np.hypot(gx, gy) >= self.threshold()) & (np.hypot(xs - guess[0], ys - guess[1]) <= window)
        if int(keep.sum()) < 4:
            return guess
        gx, gy, xs, ys = gx[keep], gy[keep], xs[keep], ys[keep]
        a = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
        if abs(np.linalg.det(a)) < 1e-9 * max(float(np.trace(a)) ** 2, 1e-12):
            return guess
        b = np.array([np.sum(gx * gx * xs + gx * gy * ys), np.sum(gx * gy * xs + gy * gy * ys)])
        q = np.linalg.solve(a, b)
        if math.hypot(q[0] - guess[0], q[1] - guess[1]) > window:
            return guess
        return (float(q[0]), float(q[1]))

    # -- fit -------------------------------------------------------------------------------
    def fit(self, shape, band, iterations=12):
        """Refine a shape's points to the image edges within `band` px; returns new point dicts.

        `shape` is a dict with `points` (resolved scalars: x, y, in_x, in_y, out_x, out_y, tangents
        relative to their point) and `closed`. Points with other fields keep those untouched,
        which keeps feather offsets relative to the moved point. The input is not modified.
        """
        return fit_points(self, shape["points"], bool(shape.get("closed", True)), band, iterations,
                          curve_type=shape.get("curve_type", "bezier"))


def _normals(positions, closed):
    count = len(positions)
    tangents = np.zeros_like(positions)
    if count < 2:
        return tangents, tangents
    if closed:
        tangents = np.roll(positions, -1, axis=0) - np.roll(positions, 1, axis=0)
    else:
        tangents[1:-1] = positions[2:] - positions[:-2]
        tangents[0] = positions[1] - positions[0]
        tangents[-1] = positions[-1] - positions[-2]
    length = np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-9)
    tangents = tangents / length[:, None]
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return tangents, normals


def _is_corner(points, positions, closed):
    """Corner points are the ones the artist cusped: both handles negligible and the path turns."""
    count = len(points)
    flags = np.zeros(count, dtype=bool)
    for i in range(count):
        if not closed and i in (0, count - 1):
            continue
        a, b, c = positions[i - 1], positions[i], positions[(i + 1) % count]
        chord = max(min(np.hypot(*(b - a)), np.hypot(*(c - b))), 1e-9)
        handle = max(math.hypot(points[i].get("in_x", 0.0), points[i].get("in_y", 0.0)),
                     math.hypot(points[i].get("out_x", 0.0), points[i].get("out_y", 0.0)))
        if handle > 0.15 * chord:
            continue
        v1, v2 = b - a, c - b
        cosine = float(np.dot(v1, v2) / max(np.hypot(*v1) * np.hypot(*v2), 1e-9))
        flags[i] = cosine < math.cos(math.radians(30.0))
    return flags


def _normal_target(field, position, normal, band, threshold):
    """Offset along `normal` (within +-band) of the strongest edge there, or None."""
    steps = np.arange(-band, band + 0.001, 0.25)
    xs = position[0] + steps * normal[0]
    ys = position[1] + steps * normal[1]
    along = np.abs(field._sample(field.gx, xs, ys) * normal[0] + field._sample(field.gy, xs, ys) * normal[1])
    if len(along) < 3:
        return None
    peak = np.zeros(len(along), dtype=bool)
    peak[1:-1] = (along[1:-1] >= along[:-2]) & (along[1:-1] >= along[2:])
    peak[0] = along[0] > along[1]
    peak[-1] = along[-1] > along[-2]
    score = along * (1.0 - 0.5 * np.abs(steps) / max(band, 1e-6))
    score = np.where(peak & (along >= threshold), score, -1.0)
    index = int(np.argmax(score))
    if score[index] <= 0:
        return None
    t = float(steps[index])
    if 0 < index < len(along) - 1:
        a, b, c = along[index - 1], along[index], along[index + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            t += 0.25 * max(-1.0, min(1.0, 0.5 * (a - c) / denom))
    return t


def _smooth_displacement(target, weight, closed, strength):
    """Solve  min sum w_i (d_i - t_i)^2 + strength * sum (d[i-1] - 2 d[i] + d[i+1])^2."""
    count = len(target)
    if count < 3:
        return np.where(weight > 0, target, 0.0)
    second = np.zeros((count, count))
    rows = range(count) if closed else range(1, count - 1)
    for i in rows:
        row = np.zeros(count)
        row[(i - 1) % count] += 1.0; row[i] -= 2.0; row[(i + 1) % count] += 1.0
        second += np.outer(row, row)
    system = np.diag(weight) + strength * second + 1e-6 * np.eye(count)
    return np.linalg.solve(system, weight * target)


def fit_points(field, points, closed, band, iterations=12, curve_type="bezier", smoothness=0.15):
    """The active contour behind `EdgeField.fit`; see there."""
    count = len(points)
    if count < 2 or band <= 0:
        return [dict(p) for p in points]
    original = np.array([[float(p["x"]), float(p["y"])] for p in points])
    positions = original.copy()
    tangent_pairs = [(float(p.get("in_x", 0.0)), float(p.get("in_y", 0.0)),
                      float(p.get("out_x", 0.0)), float(p.get("out_y", 0.0))) for p in points]
    corners = _is_corner(points, original, closed)
    threshold = field.threshold()
    neighbour_ratio = None
    if count >= 3:
        prev_d = np.hypot(*(original - np.roll(original, 1, axis=0)).T)
        next_d = np.hypot(*(np.roll(original, -1, axis=0) - original).T)
        neighbour_ratio = prev_d / np.maximum(prev_d + next_d, 1e-9)
    for _ in range(max(1, int(iterations))):
        tangents, normals = _normals(positions, closed)
        target = np.zeros(count)
        weight = np.zeros(count)
        extra = np.zeros((count, 2))
        for i in range(count):
            if corners[i]:
                hit = field.snap_corner(positions[i], band)
                if hit is not None:
                    move = np.array(hit) - positions[i]
                    target[i] = float(np.dot(move, normals[i]))
                    extra[i] = float(np.dot(move, tangents[i])) * tangents[i]
                    weight[i] = 1.0
                continue
            offset = _normal_target(field, positions[i], normals[i], band, threshold)
            if offset is not None:
                target[i], weight[i] = offset, 1.0
        displacement = _smooth_displacement(target, weight, closed, smoothness)
        moved = positions + displacement[:, None] * normals + extra
        if neighbour_ratio is not None:
            # keep the original spacing: slide each smooth point along its tangent toward where it
            # sat between its neighbours before the fit
            before = np.roll(moved, 1, axis=0)
            after = np.roll(moved, -1, axis=0)
            ideal = before + neighbour_ratio[:, None] * (after - before)
            slide = np.einsum("ij,ij->i", ideal - moved, tangents)[:, None] * tangents
            slide[corners] = 0.0
            if not closed:
                slide[0] = slide[-1] = 0.0
            moved = moved + 0.5 * slide
        step = float(np.max(np.hypot(*(moved - positions).T)))
        positions = moved
        if step < 0.02:
            break
    # tangents follow the new neighbours: turn and scale each by how its chord changed
    old_tangent, _ = _normals(original, closed)
    new_tangent, _ = _normals(positions, closed)
    out = []
    for i, p in enumerate(points):
        item = dict(p)
        item["x"], item["y"] = float(positions[i, 0]), float(positions[i, 1])
        if not corners[i]:
            if closed:
                old_chord = np.hypot(*(original[(i + 1) % count] - original[i - 1]))
                new_chord = np.hypot(*(positions[(i + 1) % count] - positions[i - 1]))
            else:
                lo, hi = max(i - 1, 0), min(i + 1, count - 1)
                old_chord = np.hypot(*(original[hi] - original[lo]))
                new_chord = np.hypot(*(positions[hi] - positions[lo]))
            scale = float(new_chord / old_chord) if old_chord > 1e-9 else 1.0
            angle = math.atan2(new_tangent[i, 1], new_tangent[i, 0]) - math.atan2(old_tangent[i, 1], old_tangent[i, 0])
            cs, sn = math.cos(angle), math.sin(angle)
            ix, iy, ox, oy = tangent_pairs[i]
            item["in_x"], item["in_y"] = scale * (cs * ix - sn * iy), scale * (sn * ix + cs * iy)
            item["out_x"], item["out_y"] = scale * (cs * ox - sn * oy), scale * (sn * ox + cs * oy)
        out.append(item)
    return out


# -- regions and cache -----------------------------------------------------------------------

def aligned_box(x0, y0, x1, y1, width, height, align=ALIGN):
    """Round a box out to the grid and clamp it to a `width` x `height` image."""
    ax0 = max(0, int(math.floor(x0 / align)) * align)
    ay0 = max(0, int(math.floor(y0 / align)) * align)
    ax1 = min(int(width), int(math.ceil(x1 / align)) * align)
    ay1 = min(int(height), int(math.ceil(y1 / align)) * align)
    return (ax0, ay0, max(ax1, ax0), max(ay1, ay0))


def field_from_image(image, box, sigma=SIGMA):
    """Compute the field over `box` of `image` (HxW or HxWxC); the margin makes border pixels exact."""
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    margin = int(math.ceil(3.0 * sigma)) + 1
    mx0, my0 = max(0, x0 - margin), max(0, y0 - margin)
    mx1, my1 = min(width, x1 + margin), min(height, y1 + margin)
    scalar = edge_source(image[my0:my1, mx0:mx1])
    gx, gy = gradient(scalar, sigma)
    return EdgeField(gx[y0 - my0:y1 - my0, x0 - mx0:x1 - mx0],
                     gy[y0 - my0:y1 - my0, x0 - mx0:x1 - mx0], box, sigma)


class EdgeCache:
    """A small least-recently-used cache of fields keyed (node, frame, input digest, region)."""

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.items = OrderedDict()
        self.hits = 0
        self.misses = 0

    def field(self, node, frame, digest, box, image_provider, sigma=SIGMA):
        key = (node, frame, digest, tuple(box), sigma)
        found = self.items.get(key)
        if found is not None:
            self.items.move_to_end(key)
            self.hits += 1
            return found
        self.misses += 1
        field = field_from_image(image_provider(), box, sigma)
        self.items[key] = field
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return field

    def clear(self):
        self.items.clear()
        self.hits = self.misses = 0


CACHE = EdgeCache()


def snap(field, point, radius):
    """`EdgeField.snap` as a function."""
    return field.snap(point, radius)


def fit(field, shape, band, iterations=12):
    """`EdgeField.fit` as a function."""
    return field.fit(shape, band, iterations)
