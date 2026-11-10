"""Roto outline from one click: find the subject under a point and return a closed Bezier outline.

Nuke's Roto has no equivalent: the artist clicks every point of the first outline by hand. Here a
click inside the subject (and, for a harder subject, a rough box around it) is enough.

Method
------
1. **Segment** the subject with a continuous min-cut, the convex relaxation of GrabCut's graph cut.
   A graph cut needs a max-flow solver, which numpy does not have; the relaxation
   ``min_u  sum g |grad u| + sum f u,  0 <= u <= 1`` is solved with the Chambolle-Pock primal-dual
   iteration, which is a handful of whole-array operations per step and needs nothing else.
   * ``f`` is the data term: the log-likelihood ratio of each pixel's colour under a foreground
     and a background colour mixture (a few full-covariance Gaussians fitted by EM). The
     foreground model starts from a small disc around the click; the background model from the
     frame border (or from everything outside the box, when one is given).
   * ``g`` is the edge weight: small across strong colour edges, so the cut prefers to follow them.
   * As in GrabCut the mixtures are refitted from the result and the cut is solved again.
   * The work is done on a copy of the image reduced to at most ``WORK_SIZE`` pixels on the long
     side. Without a box a second pass runs on the subject's bounding box at a finer scale.
2. **Contour**: keep the region connected to the click, fill its holes and trace its 0.5 level with
   marching squares (sub-pixel), in image coordinates.
3. **Edge refinement**: the contour is resampled evenly and refined to the full-resolution edges
   with the active contour of ``rotoedges`` (Lane 9, step I1).
4. **Bezier fit**: corners are found from the turning angle; Schneider's least-squares cubic
   fitting runs between them. Its error tolerance grows until there are at most ``MAX_POINTS``
   points, so straight runs get few points and curves more; spans that are straight get zero
   tangents; a result with fewer than ``MIN_POINTS`` points is subdivided to reach it.
5. A last short fit to edges pass moves the Bezier points onto the edges.

Pure numpy; no new dependency. Coordinates are Roto data coordinates (pixel ``(i, j)`` covers
``x`` in ``[j, j + 1)``, ``y`` in ``[i, i + 1)``). Holes in the subject are filled: the outline is
the silhouette.
"""
from __future__ import annotations

import math

import numpy as np

from . import rotoedges

WORK_SIZE = 256            # long side of the image the segmentation runs on
MIN_POINTS = 8
FINAL_FIT = True           # the closing fit to edges pass (switchable for measurement)
SECOND_PASS = True
MAX_POINTS = 40
SEED_RADIUS = 0.02         # of the long side of the working image (at least 2 px)
BORDER = 0.025             # width of the background sample strip, of the long side
ROUNDS = (160, 90, 60)     # primal-dual iterations for each GrabCut round
DATA_WEIGHT = 0.12         # lambda: data term against the boundary length term
DATA_CLIP = 6.0            # a pixel's evidence is limited to this many nats
MIN_HOLE_FRACTION = 0.01   # a hole must cover at least this share of the subject's area ...
MIN_HOLE_CELLS = 16        # ... and at least this many cells of the working image (about 4 x 4 px at full size or less)
COMPONENTS = 3             # colour mixture size
REGULARISE = 5e-4          # added to the colour covariances (sigma about 0.022)


class OutlineError(Exception):
    """The click did not lead to a subject (nothing found, or the region is the whole frame)."""


# -- reduction ---------------------------------------------------------------------------------

def _rgb(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.clip(array[..., :3], 0.0, 4.0)


def _resample(array, box, size):
    """Area-average the pixel box ``(x0, y0, x1, y1)`` of `array` down to ``size`` = (height, width)."""
    x0, y0, x1, y1 = box
    crop = array[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    out_h, out_w = size
    fy, fx = max(1, height // out_h), max(1, width // out_w)
    if fy > 1 or fx > 1:
        h2, w2 = height - height % fy, width - width % fx
        crop = crop[:h2, :w2].reshape(h2 // fy, fy, w2 // fx, fx, -1).mean(axis=(1, 3))
        height, width = crop.shape[:2]
    if (height, width) == (out_h, out_w):
        return crop.astype(np.float32)
    ys = np.clip((np.arange(out_h) + 0.5) * height / out_h - 0.5, 0, height - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * width / out_w - 0.5, 0, width - 1)
    iy, ix = np.minimum(ys.astype(int), height - 2 if height > 1 else 0), np.minimum(xs.astype(int), width - 2 if width > 1 else 0)
    ty, tx = (ys - iy)[:, None, None], (xs - ix)[None, :, None]
    iy1, ix1 = np.minimum(iy + 1, height - 1), np.minimum(ix + 1, width - 1)
    top = crop[iy][:, ix] * (1 - tx) + crop[iy][:, ix1] * tx
    bottom = crop[iy1][:, ix] * (1 - tx) + crop[iy1][:, ix1] * tx
    return (top * (1 - ty) + bottom * ty).astype(np.float32)


# -- colour models -----------------------------------------------------------------------------

class _Mixture:
    """A few full-covariance Gaussians fitted by EM, started from a split along the main axis."""

    def __init__(self, samples, components=COMPONENTS, iterations=6):
        samples = np.asarray(samples, dtype=np.float64)
        count = len(samples)
        k = max(1, min(components, count // 12))
        mean = samples.mean(axis=0)
        if k > 1:
            centred = samples - mean
            axis = np.linalg.svd(centred[:: max(1, count // 4000)], full_matrices=False)[2][0]
            order = np.argsort(centred @ axis)
            labels = np.empty(count, dtype=int)
            labels[order] = np.minimum(np.arange(count) * k // count, k - 1)
        else:
            labels = np.zeros(count, dtype=int)
        self.k = k
        resp = np.zeros((count, k))
        resp[np.arange(count), labels] = 1.0
        for _ in range(iterations):
            self._m_step(samples, resp)
            resp = self._responsibilities(samples)
        self._m_step(samples, resp)

    def _m_step(self, samples, resp):
        weight = resp.sum(axis=0) + 1e-9
        self.weights = weight / weight.sum()
        self.means = (resp.T @ samples) / weight[:, None]
        self.precision, self.logdet = [], []
        for j in range(self.k):
            d = samples - self.means[j]
            cov = (d * resp[:, j:j + 1]).T @ d / weight[j] + REGULARISE * np.eye(3)
            self.precision.append(np.linalg.inv(cov))
            self.logdet.append(float(np.linalg.slogdet(cov)[1]))

    def _components(self, pixels):
        out = np.empty((len(pixels), self.k))
        for j in range(self.k):
            d = pixels - self.means[j]
            maha = np.einsum("ni,ij,nj->n", d, self.precision[j], d)
            out[:, j] = math.log(self.weights[j] + 1e-12) - 0.5 * maha - 0.5 * self.logdet[j]
        return out

    def _responsibilities(self, pixels):
        comp = self._components(pixels)
        comp -= comp.max(axis=1, keepdims=True)
        resp = np.exp(comp)
        return resp / resp.sum(axis=1, keepdims=True)

    def log_likelihood(self, pixels):
        comp = self._components(np.asarray(pixels, dtype=np.float64))
        top = comp.max(axis=1)
        return top + np.log(np.exp(comp - top[:, None]).sum(axis=1))


# -- the cut -----------------------------------------------------------------------------------

def _edge_weight(rgb):
    """g: near 1 in flat colour, small across strong colour edges."""
    gx = np.zeros(rgb.shape[:2], np.float32)
    gy = np.zeros(rgb.shape[:2], np.float32)
    sx = np.zeros(rgb.shape[:2], np.float32)
    for c in range(3):
        cx, cy = rotoedges.gradient(rgb[..., c], 0.8)
        sx += cx * cx + cy * cy
    magnitude = np.sqrt(sx)
    scale = max(0.03, 0.5 * float(np.percentile(magnitude, 98)))
    return np.maximum(np.exp(-(magnitude / scale) ** 2), 0.04).astype(np.float32)


def _primal_dual(f, g, fixed_fg, fixed_bg, u, iterations):
    """min sum g |grad u| + sum f u over 0 <= u <= 1 (Chambolle-Pock; step sizes satisfy tau sigma 8 < 1)."""
    tau = sigma = 0.35
    u = u.astype(np.float32).copy()
    ubar = u.copy()
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    for _ in range(iterations):
        px[:, :-1] += sigma * (ubar[:, 1:] - ubar[:, :-1])
        py[:-1, :] += sigma * (ubar[1:, :] - ubar[:-1, :])
        norm = np.maximum(1.0, np.hypot(px, py) / g)
        px /= norm
        py /= norm
        div = px.copy()
        div[:, 1:] -= px[:, :-1]
        div += py
        div[1:, :] -= py[:-1, :]
        new = np.clip(u + tau * (div - f), 0.0, 1.0)
        new[fixed_fg] = 1.0
        new[fixed_bg] = 0.0
        ubar = 2.0 * new - u
        u = new
    return u


def _grow(seed, allowed, limit=4000):
    """The part of `allowed` connected (4-way) to `seed`."""
    reached = seed & allowed
    for _ in range(limit):
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= allowed
        if np.array_equal(grown, reached):
            break
        reached = grown
    return reached


def _body_and_silhouette(mask, seed_xy):
    """(body, silhouette): the component of `mask` holding the seed, and that component with its holes filled."""
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    h, w = mask.shape
    seed = np.zeros_like(mask)
    sx, sy = min(max(sx, 0), w - 1), min(max(sy, 0), h - 1)
    if not mask[sy, sx]:
        near = np.argwhere(mask)
        if not len(near):
            return mask & False, mask & False
        d = np.hypot(near[:, 0] - sy, near[:, 1] - sx)
        sy, sx = near[int(np.argmin(d))]
    seed[sy, sx] = True
    body = _grow(seed, mask)
    outside = np.zeros_like(mask)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside = _grow(outside & ~body, ~body)
    return body, ~outside


def _silhouette(mask, seed_xy):
    """The component of `mask` holding the seed, with its holes filled."""
    return _body_and_silhouette(mask, seed_xy)[1]


def _components(mask):
    """The 4-connected components of a boolean mask, as a list of boolean masks (largest first)."""
    remaining = {(int(i), int(j)) for i, j in np.argwhere(mask)}
    found = []
    while remaining:
        start = remaining.pop()
        stack, cells = [start], [start]
        while stack:
            i, j = stack.pop()
            for neighbour in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    stack.append(neighbour)
                    cells.append(neighbour)
        found.append(cells)
    found.sort(key=len, reverse=True)
    out = []
    for cells in found:
        part = np.zeros_like(mask)
        index = np.array(cells)
        part[index[:, 0], index[:, 1]] = True
        out.append(part)
    return out


def find_holes(body, silhouette, min_fraction=MIN_HOLE_FRACTION, min_cells=MIN_HOLE_CELLS):
    """The interior holes of a subject: components of `silhouette & ~body` covering at least `min_fraction` of the
    silhouette's area and `min_cells` cells (smaller ones are specks, not holes)."""
    area = float(silhouette.sum())
    threshold = max(float(min_cells), min_fraction * area)
    return [part for part in _components(silhouette & ~body) if part.sum() >= threshold]


def _segment(rgb, seed, box, init=None, rounds=ROUNDS, want_body=False):
    """The subject mask of the working image `rgb` (HxWx3), as a float field u in [0, 1].

    `seed` and `box` are (x, y) and (x0, y0, x1, y1) in working pixels; `init` is an earlier mask to start from.
    """
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    radius = max(2.0, SEED_RADIUS * long_side)
    fixed_fg = np.hypot(xx + 0.5 - seed[0], yy + 0.5 - seed[1]) <= radius
    border = max(2, int(round(BORDER * long_side)))
    fixed_bg = np.zeros((h, w), bool)
    fixed_bg[:border] = fixed_bg[-border:] = True
    fixed_bg[:, :border] = fixed_bg[:, -border:] = True
    sample_bg = fixed_bg.copy()
    if box is not None:
        bx0, by0, bx1, by1 = box
        outside = (xx + 0.5 < bx0) | (xx + 0.5 > bx1) | (yy + 0.5 < by0) | (yy + 0.5 > by1)
        fixed_bg |= outside
        sample_bg = fixed_bg
        fixed_bg = fixed_bg & ~fixed_fg
    if init is not None:
        sure = np.zeros_like(init)
        eroded = init.copy()
        eroded[1:] &= init[:-1]; eroded[:-1] &= init[1:]; eroded[:, 1:] &= init[:, :-1]; eroded[:, :-1] &= init[:, 1:]
        fixed_fg = fixed_fg | sure | eroded
    pixels = rgb.reshape(-1, 3)
    g = _edge_weight(rgb)
    fg_samples = fixed_fg
    u = None
    for iterations in rounds:
        if int(fg_samples.sum()) < 4 or int(sample_bg.sum()) < 4:
            raise OutlineError("not enough pixels to model the subject and the background")
        fg_model = _Mixture(rgb[fg_samples])
        bg_model = _Mixture(rgb[sample_bg])
        evidence = (-fg_model.log_likelihood(pixels) + bg_model.log_likelihood(pixels)).reshape(h, w)
        f = (DATA_WEIGHT * np.clip(evidence, -DATA_CLIP, DATA_CLIP)).astype(np.float32)
        start = (f < 0).astype(np.float32) if u is None else u
        u = _primal_dual(f, g, fixed_fg, fixed_bg, start, iterations)
        body, mask = _body_and_silhouette(u >= 0.5, seed)
        if not mask.any():
            raise OutlineError("no subject found at that point")
        # the next round learns from the settled interior and the settled exterior (a hole is not interior when holes are kept)
        learn = body if want_body else mask
        inner = learn.copy()
        inner[1:] &= learn[:-1]; inner[:-1] &= learn[1:]; inner[:, 1:] &= learn[:, :-1]; inner[:, :-1] &= learn[:, 1:]
        fg_samples = inner | fixed_fg
        far = ~_dilate(mask, 2)
        sample_bg = far | sample_bg if box is None else (far & ~outside) | sample_bg
    return (mask, body) if want_body else mask


def _dilate(mask, steps=1):
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:] |= out[:-1]; grown[:-1] |= out[1:]; grown[:, 1:] |= out[:, :-1]; grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


# -- contour -----------------------------------------------------------------------------------

def trace_contour(mask):
    """The longest closed 0.5 level of `mask` (a boolean array) as an (n, 2) array of (x, y), in the mask's
    pixel centre coordinates plus 0.5 (so data coordinates for a mask whose pixel (i, j) covers [j, j+1))."""
    field = np.pad(mask.astype(np.float32), 1)
    kernel, _ = rotoedges._gaussian_kernel(0.8)
    field = rotoedges._convolve_axis(rotoedges._convolve_axis(field, kernel, 0), kernel, 1)
    inside = field >= 0.5
    h, w = field.shape
    tl, tr = inside[:-1, :-1], inside[:-1, 1:]
    bl, br = inside[1:, :-1], inside[1:, 1:]
    code = tl.astype(int) * 8 + tr * 4 + br * 2 + bl
    cells = np.argwhere((code != 0) & (code != 15))

    def crossing(kind, i, j):
        if kind == 0:   # horizontal edge from (i, j) to (i, j+1)
            a, b = field[i, j], field[i, j + 1]
            t = (0.5 - a) / (b - a)
            return (j + t, float(i))
        a, b = field[i, j], field[i + 1, j]
        t = (0.5 - a) / (b - a)
        return (float(j), i + t)

    links = {}
    for i, j in cells:
        i, j = int(i), int(j)
        edges = []
        if inside[i, j] != inside[i, j + 1]: edges.append((0, i, j))          # top
        if inside[i, j + 1] != inside[i + 1, j + 1]: edges.append((1, i, j + 1))  # right
        if inside[i + 1, j] != inside[i + 1, j + 1]: edges.append((0, i + 1, j))  # bottom
        if inside[i, j] != inside[i + 1, j]: edges.append((1, i, j))          # left
        if len(edges) == 4:
            centre = field[i:i + 2, j:j + 2].mean() >= 0.5
            top, right, bottom, left = edges
            pairs = [(top, left), (bottom, right)] if centre == bool(inside[i, j]) else [(top, right), (bottom, left)]
        else:
            pairs = [(edges[0], edges[1])]
        for a, b in pairs:
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in links:
        if start in seen:
            continue
        loop, previous, current = [], None, start
        while current not in seen:
            seen.add(current)
            loop.append(current)
            options = [n for n in links[current] if n != previous]
            nxt = options[0] if options else None
            if nxt is None or nxt in seen and nxt != start:
                break
            previous, current = current, nxt
        loops.append(loop)
    if not loops:
        raise OutlineError("no boundary found")
    best = max(loops, key=len)
    points = np.array([crossing(*edge) for edge in best], dtype=np.float64)
    # pad of 1 and pixel centres at +0.5: data x = index - 1 + 0.5
    return points - 0.5


def _resample_polygon(points, spacing, maximum=400):
    """Evenly spaced points along a closed polygon."""
    closed = np.vstack([points, points[:1]])
    seg = np.hypot(*(closed[1:] - closed[:-1]).T)
    total = float(seg.sum())
    count = int(min(max(8, round(total / max(spacing, 1e-6))), maximum))
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, count, endpoint=False)
    xs = np.interp(targets, cumulative, closed[:, 0])
    ys = np.interp(targets, cumulative, closed[:, 1])
    return np.stack([xs, ys], axis=1), total


# -- Bezier fitting (Schneider, Graphics Gems) ---------------------------------------------------

def _bernstein(u):
    v = 1.0 - u
    return v ** 3, 3 * u * v * v, 3 * u * u * v, u ** 3


def _chord_parameters(points):
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*(points[1:] - points[:-1]).T))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(points))


def _generate(points, u, t1, t2):
    p0, p3 = points[0], points[-1]
    b0, b1, b2, b3 = _bernstein(u)
    a1, a2 = t1[None, :] * b1[:, None], t2[None, :] * b2[:, None]
    c00, c01, c11 = (a1 * a1).sum(), (a1 * a2).sum(), (a2 * a2).sum()
    tmp = points - (p0[None, :] * (b0 + b1)[:, None] + p3[None, :] * (b2 + b3)[:, None])
    x0, x1 = (a1 * tmp).sum(), (a2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = float(np.hypot(*(p3 - p0)))
    alpha1 = alpha2 = seg / 3.0
    if abs(det) > 1e-12:
        a_1, a_2 = (x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det
        if a_1 > 1e-6 * seg and a_2 > 1e-6 * seg:
            alpha1, alpha2 = a_1, a_2
    return np.array([p0, p0 + t1 * alpha1, p3 + t2 * alpha2, p3])


def _evaluate(bez, u):
    b0, b1, b2, b3 = _bernstein(u)
    return b0[:, None] * bez[0] + b1[:, None] * bez[1] + b2[:, None] * bez[2] + b3[:, None] * bez[3]


def _max_error(points, bez, u):
    d = np.einsum("ij,ij->i", *(2 * [points - _evaluate(bez, u)]))
    index = int(np.argmax(d))
    return float(d[index]), index


def _reparameterise(points, u, bez):
    v = 1.0 - u
    first = 3 * (bez[1] - bez[0]); second = 3 * (bez[2] - bez[1]); third = 3 * (bez[3] - bez[2])
    d1 = (v * v)[:, None] * first + (2 * v * u)[:, None] * second + (u * u)[:, None] * third
    d2 = v[:, None] * 2 * (second - first) + u[:, None] * 2 * (third - second)
    diff = _evaluate(bez, u) - points
    numerator = (diff * d1).sum(axis=1)
    denominator = (d1 * d1).sum(axis=1) + (diff * d2).sum(axis=1)
    safe = np.abs(denominator) > 1e-12
    return np.clip(np.where(safe, u - numerator / np.where(safe, denominator, 1.0), u), 0.0, 1.0)


def _fit_span(points, t1, t2, tol):
    """Cubic Beziers (a list of 4x2 arrays) through `points`, with unit end tangents t1 (forward) and t2 (backward)."""
    count = len(points)
    if count == 2:
        d = float(np.hypot(*(points[1] - points[0]))) / 3.0
        return [np.array([points[0], points[0] + t1 * d, points[1] + t2 * d, points[1]])]
    u = _chord_parameters(points)
    bez = _generate(points, u, t1, t2)
    error, split = _max_error(points, bez, u)
    if error < tol * tol:
        return [bez]
    if error < 16 * tol * tol:
        for _ in range(6):
            u = _reparameterise(points, u, bez)
            bez = _generate(points, u, t1, t2)
            error, split = _max_error(points, bez, u)
            if error < tol * tol:
                return [bez]
    split = min(max(split, 1), count - 2)
    centre = points[split - 1] - points[split + 1]
    length = float(np.hypot(*centre))
    centre = centre / length if length > 1e-12 else t2
    return _fit_span(points[:split + 1], t1, centre, tol) + _fit_span(points[split:], -centre, t2, tol)


def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def find_corners(points, perimeter, angle=50.0):
    """Indices of points where a closed, evenly spaced polygon turns sharply (local maxima of the turning angle)."""
    count = len(points)
    spacing = perimeter / count
    window = max(2, int(round(min(max(0.02 * perimeter, 3.0), 12.0) / max(spacing, 1e-6))))
    if count < 4 * window:
        return []
    before = points - np.roll(points, window, axis=0)
    after = np.roll(points, -window, axis=0) - points
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = (before * after).sum(axis=1)
    turn = np.degrees(np.abs(np.arctan2(cross, dot)))
    corners = []
    for i in range(count):
        if turn[i] < angle:
            continue
        neighbourhood = turn[[(i + k) % count for k in range(-window, window + 1)]]
        if turn[i] >= neighbourhood.max() and (i not in corners):
            corners.append(i)
    # merge corners closer than a window
    merged = []
    for i in corners:
        if merged and (i - merged[-1]) % count <= window:
            continue
        merged.append(i)
    return merged


def fit_bezier(points, perimeter, tolerance, corners):
    """Bezier points {x, y, in_x, in_y, out_x, out_y} for a closed, evenly spaced polygon."""
    count = len(points)
    breaks = sorted(set(corners))
    if len(breaks) < 2:
        breaks = sorted(set(breaks + [0 if not breaks else (breaks[0] + count // 2) % count]
                            + ([count // 2] if not breaks else [])))
        if len(breaks) < 2:
            breaks = [0, count // 2]
    k = 3
    spans = []
    for a, b in zip(breaks, breaks[1:] + [breaks[0] + count]):
        idx = [(a + n) % count for n in range(b - a + 1)]
        chain = points[idx]
        if len(chain) < 3:
            chain = np.array([chain[0], (chain[0] + chain[-1]) / 2, chain[-1]])
        def tangent(index, sign, chain=chain):
            if index in corners:
                one = chain[min(k, len(chain) - 1)] - chain[0] if sign > 0 else chain[max(len(chain) - 1 - k, 0)] - chain[-1]
                return _unit(one)
            centre = points[(index + k) % count] - points[(index - k) % count]
            return _unit(centre) if sign > 0 else _unit(-centre)
        t1, t2 = tangent(a % count, +1), tangent(b % count, -1)
        spans.append(_fit_span(chain, t1, t2, tolerance))
    beziers = [b for span in spans for b in span]
    return _to_points(beziers)


def _to_points(beziers):
    count = len(beziers)
    out = []
    for i, bez in enumerate(beziers):
        previous = beziers[i - 1]
        out.append({"x": float(bez[0, 0]), "y": float(bez[0, 1]),
                    "in_x": float(previous[2, 0] - bez[0, 0]), "in_y": float(previous[2, 1] - bez[0, 1]),
                    "out_x": float(bez[1, 0] - bez[0, 0]), "out_y": float(bez[1, 1] - bez[0, 1])})
    return out


def _straighten(points, tolerance=0.3):
    """Zero the handles of every span whose handles lie on the chord (a straight run)."""
    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        chord = np.array([b["x"] - a["x"], b["y"] - a["y"]])
        length = float(np.hypot(*chord))
        if length < 1e-9:
            continue
        normal = np.array([-chord[1], chord[0]]) / length
        h1 = np.array([a["out_x"], a["out_y"]]); h2 = np.array([b["in_x"], b["in_y"]])
        if abs(float(h1 @ normal)) < tolerance and abs(float(h2 @ normal)) < tolerance:
            a["out_x"] = a["out_y"] = 0.0
            b["in_x"] = b["in_y"] = 0.0
    return points


def _subdivide(points, minimum):
    """Split the longest spans at t = 0.5 until there are `minimum` points."""
    points = [dict(p) for p in points]
    while len(points) < minimum:
        lengths = [math.hypot(points[(i + 1) % len(points)]["x"] - p["x"], points[(i + 1) % len(points)]["y"] - p["y"])
                   for i, p in enumerate(points)]
        i = int(np.argmax(lengths))
        a, b = points[i], points[(i + 1) % len(points)]
        p0 = np.array([a["x"], a["y"]]); p1 = p0 + [a["out_x"], a["out_y"]]
        p3 = np.array([b["x"], b["y"]]); p2 = p3 + [b["in_x"], b["in_y"]]
        p01, p12, p23 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
        p012, p123 = (p01 + p12) / 2, (p12 + p23) / 2
        mid = (p012 + p123) / 2
        a["out_x"], a["out_y"] = (p01 - p0).tolist()
        b["in_x"], b["in_y"] = (p23 - p3).tolist()
        points.insert(i + 1, {"x": float(mid[0]), "y": float(mid[1]),
                              "in_x": float(p012[0] - mid[0]), "in_y": float(p012[1] - mid[1]),
                              "out_x": float(p123[0] - mid[0]), "out_y": float(p123[1] - mid[1])})
    return points


# -- the whole thing ---------------------------------------------------------------------------

def _work_region(width, height, box):
    if box is None:
        return (0, 0, width, height)
    x0, y0, x1, y1 = box
    pad = 0.1 * max(x1 - x0, y1 - y0) + 4
    return (max(0, int(math.floor(x0 - pad))), max(0, int(math.floor(y0 - pad))),
            min(width, int(math.ceil(x1 + pad))), min(height, int(math.ceil(y1 + pad))))


def _mask_in_region(rgb, region, seed, box, init_box=None, init_mask=None, want_body=False):
    """Segment the subject inside the pixel `region` of `rgb`; returns (mask, scale, work_box) with the mask
    on a grid of `scale` full-resolution pixels per cell whose origin is the region's corner."""
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    scale = max(1.0, max(width, height) / WORK_SIZE)
    size = (max(8, int(round(height / scale))), max(8, int(round(width / scale))))
    small = _resample(rgb, region, size)
    sx, sy = size[1] / width, size[0] / height
    work_seed = ((seed[0] - x0) * sx, (seed[1] - y0) * sy)
    work_box = None if box is None else ((box[0] - x0) * sx, (box[1] - y0) * sy, (box[2] - x0) * sx, (box[3] - y0) * sy)
    init = None
    if init_mask is not None:
        init = init_mask(size, region)
    result = _segment(small, work_seed, work_box, init=init, rounds=ROUNDS if init is None else ROUNDS[1:], want_body=want_body)
    if want_body:
        return result[0], result[1], (1.0 / sx, 1.0 / sy), region
    return result, (1.0 / sx, 1.0 / sy), region


def _find_subject(image, seed, box, holes):
    """Segment the subject under `seed`. Returns (rgb, mask, body, x0, y0, sx, sy, width, height): the silhouette `mask`
    (holes filled) and, when `holes` is set, the `body` that keeps them (otherwise `body` is None), on a grid whose cell
    (i, j) covers [x0 + j * sx, x0 + (j + 1) * sx) by the same in y, in full-resolution pixels."""
    array = np.asarray(image)
    height, width = array.shape[:2]
    if not (0 <= seed[0] < width and 0 <= seed[1] < height):
        raise OutlineError("the click is outside the picture")
    rgb = _rgb(array)
    region = _work_region(width, height, box)
    if holes:
        mask, body, (sx, sy), region = _mask_in_region(rgb, region, seed, box, want_body=True)
    else:
        (mask, (sx, sy), region), body = _mask_in_region(rgb, region, seed, box), None
    x0, y0 = region[0], region[1]
    # a second, finer pass on the subject itself when the first was coarse
    if SECOND_PASS and max(sx, sy) > 1.6:
        ys, xs = np.nonzero(mask)
        pad = 0.12 * max(xs.max() - xs.min(), ys.max() - ys.min()) + 3
        fine = (max(0, int(math.floor(x0 + (xs.min() - pad) * sx))), max(0, int(math.floor(y0 + (ys.min() - pad) * sy))),
                min(width, int(math.ceil(x0 + (xs.max() + 1 + pad) * sx))), min(height, int(math.ceil(y0 + (ys.max() + 1 + pad) * sy))))
        coarse, coarse_region, coarse_scale = (body if holes else mask), region, (sx, sy)

        def init_mask(size, fine_region, coarse=coarse, coarse_region=coarse_region, coarse_scale=coarse_scale):
            fy = (np.arange(size[0]) + 0.5) * (fine_region[3] - fine_region[1]) / size[0] + fine_region[1]
            fx = (np.arange(size[1]) + 0.5) * (fine_region[2] - fine_region[0]) / size[1] + fine_region[0]
            iy = np.clip(((fy - coarse_region[1]) / coarse_scale[1]).astype(int), 0, coarse.shape[0] - 1)
            ix = np.clip(((fx - coarse_region[0]) / coarse_scale[0]).astype(int), 0, coarse.shape[1] - 1)
            return coarse[iy][:, ix]
        box_fine = None if box is None else box
        if holes:
            mask, body, (sx, sy), region = _mask_in_region(rgb, fine, seed, box_fine, init_mask=init_mask, want_body=True)
        else:
            mask, (sx, sy), region = _mask_in_region(rgb, fine, seed, box_fine, init_mask=init_mask)
        x0, y0 = region[0], region[1]
    area = float(mask.sum())
    if area < 12:
        raise OutlineError("the subject is too small")
    if area > 0.9 * mask.size:
        raise OutlineError("the region found fills the picture; draw a box around the subject")
    return rgb, mask, body, x0, y0, sx, sy, width, height


def _bezier_outline(rgb, mask, x0, y0, sx, sy, width, height, min_points, max_points):
    """The closed Bezier outline of the boolean `mask` (grid as in `_find_subject`)."""
    contour = trace_contour(mask)
    # the mask cell (i, j) covers [j, j + 1) working units, `sx`, `sy` full-resolution pixels per unit
    polygon = np.stack([x0 + contour[:, 0] * sx, y0 + contour[:, 1] * sy], axis=1)
    spacing = max(2.0, 0.9 * max(sx, sy))
    dense, perimeter = _resample_polygon(polygon, spacing)
    # refine the dense polygon to the full-resolution edges
    margin = int(math.ceil(3 * max(sx, sy))) + 8
    bx0, by0 = dense.min(axis=0) - margin
    bx1, by1 = dense.max(axis=0) + margin
    box_px = rotoedges.aligned_box(bx0, by0, bx1, by1, width, height)
    field = rotoedges.field_from_image(rgb, box_px)
    band = max(2.0, 1.4 * max(sx, sy) + 1.0)
    pts = [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0} for x, y in dense]
    refined = rotoedges.fit_points(field, pts, True, band, iterations=10)
    refined_xy = np.array([[p["x"], p["y"]] for p in refined])
    refined_xy, perimeter = _resample_polygon(refined_xy, spacing)
    corners = find_corners(refined_xy, perimeter)
    tolerance = max(0.7, 0.0008 * perimeter)
    result = None
    for _ in range(12):
        result = _straighten(fit_bezier(refined_xy, perimeter, tolerance, corners))
        if len(result) <= max_points:
            break
        tolerance *= 1.4
    result = _subdivide(result, min_points)
    # snap the corner points to the true corners, then a short fit to edges for the rest
    if not FINAL_FIT:
        return result
    return _guarded_fit(field, result)


def outline_from_click(image, seed, box=None, min_points=MIN_POINTS, max_points=MAX_POINTS):
    """A closed Bezier outline of the subject under `seed` = (x, y) in the `image` (HxWx3/4 float).

    `box` = (x0, y0, x1, y1) optionally confines the subject. Returns a list of point dicts
    ``x, y, in_x, in_y, out_x, out_y`` (tangents relative to their point), or raises `OutlineError`.
    Holes in the subject are filled: this is the silhouette (see `outline_with_holes`).
    """
    rgb, mask, _, x0, y0, sx, sy, width, height = _find_subject(image, seed, box, holes=False)
    return _bezier_outline(rgb, mask, x0, y0, sx, sy, width, height, min_points, max_points)


def outline_with_holes(image, seed, box=None, min_points=MIN_POINTS, max_points=MAX_POINTS,
                       min_hole_fraction=MIN_HOLE_FRACTION, min_hole_cells=MIN_HOLE_CELLS):
    """The outline of the subject under `seed` and one closed outline per interior hole: ``(outer, [hole, ...])``.

    `outer` is what `outline_from_click` returns (the silhouette). Each hole is a region inside the silhouette that the
    segmentation left out of the subject, as long as it covers at least `min_hole_fraction` of the silhouette's area and
    `min_hole_cells` cells of the working image; smaller ones are specks and stay part of the subject. Holes are returned
    biggest first. Rendered with the holes as shapes that subtract, the matte is the subject with its holes cut out.
    """
    rgb, mask, body, x0, y0, sx, sy, width, height = _find_subject(image, seed, box, holes=True)
    outer = _bezier_outline(rgb, mask, x0, y0, sx, sy, width, height, min_points, max_points)
    holes = []
    for part in find_holes(body, mask, min_hole_fraction, min_hole_cells):
        try:
            holes.append(_bezier_outline(rgb, part, x0, y0, sx, sy, width, height, min_points, max_points))
        except Exception:
            continue    # a hole whose contour cannot be traced is left as part of the subject
    return outer, holes


def _edge_score(field, points):
    """Mean gradient magnitude along the flattened outline (what the fit is meant to raise)."""
    from . import roto
    outline = np.array(roto.flatten_points(points, closed=True))
    return float(field.magnitude_at(outline[:, 0], outline[:, 1]).mean())


def _guarded_fit(field, points, band=1.5):
    """The closing pass of ``rotoedges`` fit to edges, kept only when it raises the edge response along the
    outline and moves no point further than `band`. The Bezier already follows the refined contour, so the
    pass can only polish; on corners it can round them off, and then the result is not used."""
    candidate = rotoedges.fit_points(field, points, True, band, iterations=6)
    moved = max(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for a, b in zip(points, candidate))
    if moved <= band + 0.25 and _edge_score(field, candidate) > _edge_score(field, points):
        return candidate
    return points


def bspline_controls(points, samples_per_span=2):
    """Control points of a closed uniform cubic B-spline that passes through the outline.

    The B-Spline tool stores control points, not points on the curve, so the Bezier outline is
    sampled (each point and the middle of each span) and the controls ``Q`` solved from
    ``P_i = (Q_(i-1) + 4 Q_i + Q_(i+1)) / 6``. A B-spline has no corners: sharp tips round off.
    """
    count = len(points)
    samples = []
    for i, a in enumerate(points):
        b = points[(i + 1) % count]
        p0 = np.array([a["x"], a["y"]]); p1 = p0 + [a["out_x"], a["out_y"]]
        p3 = np.array([b["x"], b["y"]]); p2 = p3 + [b["in_x"], b["in_y"]]
        for k in range(samples_per_span):
            t = k / samples_per_span
            samples.append(_evaluate(np.array([p0, p1, p2, p3]), np.array([t]))[0])
    samples = np.array(samples)
    n = len(samples)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, (i - 1) % n] += 1.0
        matrix[i, i] += 4.0
        matrix[i, (i + 1) % n] += 1.0
    controls = np.linalg.solve(matrix / 6.0, samples)
    return [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0} for x, y in controls]
