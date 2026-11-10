"""Roto point tracking: follow a deforming shape by tracking each control point (lane 9, step I2).

"Track shape" planar-tracks the shape's bounding quad, so the outline follows a rigid or perspective move but not a
subject that bends or stretches. This module tracks every control point with its own small patch and regularises the
result with the planar track, so a shape follows a deforming subject and stays coherent.

Pure numpy, no document and no Qt: `track_points` takes a frame source and the shape's points at the reference frame and
returns, for each tracked frame, the new points (anchor, tangents and feather offsets), a confidence per point and the
points whose confidence fell below the threshold.

Per frame, per point:

1. The search is centred on the planar track's position for the point (the plane the shape sat on at the reference frame,
   moved by the planar homography of this frame) plus the offset the point's own match had from the plane when it last
   matched; the search window is `search_radius` pixels around that.
2. A normalised cross-correlation of a `(2 * patch_radius + 1)` pixel patch (grabbed again from the frame whenever its match
   falls under `REGRAB`, so a slow deformation does not defeat it and a good match does not accumulate error) finds the best integer position, and a few Lucas-Kanade steps on the zero-mean patch
   refine it to a fraction of a pixel. The confidence is the correlation at the refined position.
3. The point's position is the planar prediction pulled toward its own match by the confidence: nothing below `threshold`
   (the point falls back to the planar prediction and is flagged), all of it above `threshold + BLEND_RANGE`.
4. Each confident point's offset from the plane is then pulled toward the confidence-weighted average of its neighbours'
   offsets along the shape by `smoothing`, so a noisy match does not tear the outline. Flagged points are left on the
   planar prediction and take no part in their neighbours' averages.
5. Tangent and feather offsets turn and scale with the planar map at the point.

Coordinates are pixels with pixel centres at +0.5, the convention the Tracker and the planar tracker use.
"""
from __future__ import annotations

import math
from concurrent.futures import CancelledError

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PATCH_RADIUS = 7
SEARCH_RADIUS = 12
THRESHOLD = 0.7          # a match under this correlation flags the point and leaves it on the planar prediction
BLEND_RANGE = 0.1        # the own match takes over fully this far above the threshold
REGRAB = 0.92            # the patch is grabbed again from the frame only when its match falls under this
SMOOTHING = 0.25         # how far a confident point is pulled toward its neighbours' offsets from the plane
LK_ITERATIONS = 12
RIDGE = 0.1              # how hard the stretch and shear terms are held toward zero (relative to the patch's gradient energy)
LK_STEP_LIMIT = 1.0      # one refinement step never moves more than this
LK_TOTAL_LIMIT = 1.5     # the refinement never ends more than this from the integer peak
OFFSET_FIELDS = (("in_x", "in_y"), ("out_x", "out_y"), ("feather_x", "feather_y"))
POINT_FIELDS = ("x", "y", "in_x", "in_y", "out_x", "out_y", "feather_x", "feather_y")


class PointTrackError(ValueError):
    """The shape cannot be tracked (too few points, bad range)."""


# -- the patch matcher ---------------------------------------------------------------------------------------------------
def _bilinear(image, xs, ys):
    """Catmull-Rom (bicubic) samples at index coordinates, clamped at the image edge.

    Cubic rather than bilinear: bilinear blurs a half-pixel sample more than an integer one, and a template and a candidate
    blurred differently pull the match half a pixel's worth of bias apart.
    """
    height, width = image.shape
    xs = np.clip(xs, 0.0, width - 1.000001)
    ys = np.clip(ys, 0.0, height - 1.000001)
    x0 = np.floor(xs).astype(np.intp)
    y0 = np.floor(ys).astype(np.intp)
    fx, fy = xs - x0, ys - y0

    def weights(f):
        f2, f3 = f * f, f * f * f
        return (-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1.0, -1.5 * f3 + 2.0 * f2 + 0.5 * f, 0.5 * f3 - 0.5 * f2)
    wx, wy = weights(fx), weights(fy)
    result = 0.0
    for j in range(4):
        rows = np.clip(y0 + j - 1, 0, height - 1)
        line = 0.0
        for i in range(4):
            line = line + image[rows, np.clip(x0 + i - 1, 0, width - 1)] * wx[i]
        result = result + line * wy[j]
    return result


def _offsets(radius):
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    return np.meshgrid(axis, axis)


def sample_patch(image, x, y, radius):
    """The `(2r+1)^2` patch of `image` centred at index coordinates (x, y), bilinear."""
    u, v = _offsets(radius)
    return _bilinear(image, x + u, y + v)


def _inside(image, x, y, radius):
    height, width = image.shape
    return radius + 1 <= x <= width - 2 - radius and radius + 1 <= y <= height - 2 - radius


def _refine(image, template, x, y, radius):
    """Lucas-Kanade steps on the zero-mean patch; returns the refined index coordinates of the patch centre.

    The patch is allowed to stretch and shear a little (a six-parameter affine warp, the four linear terms held back by a
    small ridge), so a point on a subject that is deforming is not pulled toward the side that deforms least.
    """
    t0 = template - template.mean()
    t_std = float(t0.std())
    if t_std < 1e-9:
        return x, y, -1.0
    start = (x, y)
    u, v = _offsets(radius)
    warp = np.zeros(6)                    # dx, dy, then the 2x2 linear part's departure from identity: a, b, c, d
    ridge = np.diag([0.0, 0.0, RIDGE, RIDGE, RIDGE, RIDGE])
    for _ in range(LK_ITERATIONS):
        wu = u + warp[2] * u + warp[3] * v
        wv = v + warp[4] * u + warp[5] * v
        px, py = x + warp[0] + wu, y + warp[1] + wv
        if not _inside(image, x + warp[0], y + warp[1], radius):
            break
        w = _bilinear(image, px, py)
        gx = _bilinear(image, px + 0.5, py) - _bilinear(image, px - 0.5, py)
        gy = _bilinear(image, px, py + 0.5) - _bilinear(image, px, py - 0.5)
        w0 = w - w.mean()
        w_std = float(w0.std())
        if w_std < 1e-9:
            break
        scale = t_std / w_std
        columns = np.stack([(gx).ravel(), (gy).ravel(), (gx * u).ravel(), (gx * v).ravel(),
                            (gy * u).ravel(), (gy * v).ravel()], axis=1)
        columns = scale * (columns - columns.mean(axis=0))
        error = (t0 - scale * w0).ravel()
        normal = columns.T @ columns
        normal = normal + ridge * max(float(np.trace(normal)) / 6.0, 1e-9)
        try:
            delta = np.linalg.solve(normal, columns.T @ error)
        except np.linalg.LinAlgError:
            break
        step = math.hypot(delta[0], delta[1])
        if step > LK_STEP_LIMIT:
            delta[:2] *= LK_STEP_LIMIT / step
        delta[2:] = np.clip(delta[2:], -0.05, 0.05)
        warp += delta
        warp[2:] = np.clip(warp[2:], -0.5, 0.5)
        if step < 0.003 and float(np.abs(delta[2:]).max()) < 1e-3:
            break
    if math.hypot(warp[0], warp[1]) > LK_TOTAL_LIMIT:
        warp[:] = 0.0
    x, y = start[0] + float(warp[0]), start[1] + float(warp[1])
    if not _inside(image, x, y, radius):
        return start[0], start[1], _correlation(template, sample_patch(image, start[0], start[1], radius))
    wu = u + warp[2] * u + warp[3] * v
    wv = v + warp[4] * u + warp[5] * v
    return x, y, _correlation(template, _bilinear(image, x + wu, y + wv))


def _correlation(template, patch):
    a = template - template.mean()
    b = patch - patch.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denominator <= 1e-12:
        return -1.0
    return float((a * b).sum()) / denominator


def match_patch(image, template, centre_x, centre_y, radius=PATCH_RADIUS, search=SEARCH_RADIUS):
    """Find `template` in `image` around index coordinates (centre_x, centre_y).

    Returns `(x, y, correlation)` in index coordinates, or None when the template is flat or no whole patch fits in the
    search window. The best integer position of the zero-mean normalised cross-correlation is refined to a fraction of a
    pixel.
    """
    height, width = image.shape
    t0 = template - template.mean()
    t_norm = math.sqrt(float((t0 * t0).sum()))
    if t_norm < 1e-9:
        return None
    cx, cy = int(round(centre_x)), int(round(centre_y))
    x_lo, x_hi = max(cx - search, radius + 1), min(cx + search, width - 2 - radius)
    y_lo, y_hi = max(cy - search, radius + 1), min(cy + search, height - 2 - radius)
    if x_hi < x_lo or y_hi < y_lo:
        return None
    region = image[y_lo - radius:y_hi + radius + 1, x_lo - radius:x_hi + radius + 1]
    windows = sliding_window_view(region, (2 * radius + 1, 2 * radius + 1))
    count = float((2 * radius + 1) ** 2)
    mean = windows.mean(axis=(2, 3))
    variance = (windows * windows).mean(axis=(2, 3)) - mean * mean
    spread = np.sqrt(np.maximum(variance, 0.0) * count)
    numerator = np.einsum("yxij,ij->yx", windows, t0)
    score = np.where(spread > 1e-9, numerator / np.maximum(spread * t_norm, 1e-30), -1.0)
    peak = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[peak] <= -1.0 + 1e-9:
        return None
    starts = [(float(x_lo + peak[1]), float(y_lo + peak[0]))]
    if abs(centre_x - starts[0][0]) > 0.5 or abs(centre_y - starts[0][1]) > 0.5:
        if _inside(image, centre_x, centre_y, radius):      # the predicted position is the other place a good match may sit
            starts.append((float(centre_x), float(centre_y)))
    best = None
    for px, py in starts:
        rx, ry, corr = _refine(image, template, px, py, radius)
        if best is None or corr > best[2] + 1e-9:
            best = (rx, ry, corr)
    return best


# -- the planar prediction -----------------------------------------------------------------------------------------------
def bounding_quad(points):
    xs, ys = [float(p["x"]) for p in points], [float(p["y"]) for p in points]
    return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]


def planar_matrices(get_frame, points, reference_frame, frame_numbers, *, samples_per_axis=16, vector_detail=4,
                    smoothness=1.0, threshold=2.0, cancel=None, progress=None):
    """Planar-track the shape's bounding quad; returns `{frame: 3x3 matrix}` mapping the reference frame to each frame.

    `get_frame(frame)` returns a 2-D float image; the planar tracker needs a contiguous frame range, so every frame between
    the first and last of `frame_numbers` and the reference is read.
    """
    from . import planar
    reference_frame = int(reference_frame)
    numbers = sorted({reference_frame, *map(int, frame_numbers)})
    frames = {}
    for frame in range(numbers[0], numbers[-1] + 1):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        gray = np.asarray(get_frame(frame), dtype=np.float32)
        frames[frame] = np.repeat(gray[..., None], 3, axis=2)
    results = planar.track_region(frames, np.asarray(bounding_quad(points), dtype=np.float64), reference_frame,
                                  samples_per_axis=samples_per_axis, vector_detail=vector_detail,
                                  smoothness=smoothness, backend="cpu", threshold=threshold, cancel=cancel,
                                  progress=progress)
    return {frame: np.asarray(result["matrix"], dtype=np.float64) for frame, result in results.items()}


def _project(matrix, xy):
    from .planar import project_points
    return project_points(matrix, xy)


def _jacobian(matrix, xy):
    """The 2x2 linear part of the homography at `xy` (a central finite difference over one pixel)."""
    base = np.asarray(xy, dtype=np.float64)
    probes = _project(matrix, np.array([base + (1, 0), base - (1, 0), base + (0, 1), base - (0, 1)]))
    return np.column_stack(((probes[0] - probes[1]) / 2.0, (probes[2] - probes[3]) / 2.0))


# -- smoothing along the shape -------------------------------------------------------------------------------------------
def smooth_offsets(offsets, weights, closed, strength):
    """Pull each point's offset toward the weight-averaged offsets of its neighbours along the shape.

    Points with zero weight are returned unchanged and count for nothing in the averages.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    count = len(offsets)
    result = offsets.copy()
    if count < 3 or strength <= 0.0:
        return result
    for i in range(count):
        if weights[i] <= 0.0:
            continue
        neighbours = [(i - 1) % count, (i + 1) % count] if closed else [j for j in (i - 1, i + 1) if 0 <= j < count]
        total = sum(weights[j] for j in neighbours)
        if total <= 0.0:
            continue
        average = sum(weights[j] * offsets[j] for j in neighbours) / total
        result[i] = (1.0 - strength) * offsets[i] + strength * average
    return result


# -- the tracker ---------------------------------------------------------------------------------------------------------
def frame_order(reference_frame, first_frame, last_frame, direction="forward"):
    """The frames after the reference to track, nearest first, for `direction` forward, backward or both (two passes)."""
    reference_frame, first_frame, last_frame = int(reference_frame), int(first_frame), int(last_frame)
    if first_frame > last_frame:
        raise PointTrackError("the first frame must not come after the last frame")
    forward = list(range(reference_frame + 1, last_frame + 1))
    backward = list(range(reference_frame - 1, first_frame - 1, -1))
    if direction == "forward":
        return [forward]
    if direction == "backward":
        return [backward]
    if direction == "both":
        return [forward, backward]
    raise PointTrackError("direction must be forward, backward or both")


def track_points(get_frame, points, reference_frame, frames, *, planar=None, closed=True, patch_radius=PATCH_RADIUS,
                 search_radius=SEARCH_RADIUS, threshold=THRESHOLD, smoothing=SMOOTHING, cancel=None, progress=None):
    """Track `points` from `reference_frame` through `frames` (an ordered list, nearest the reference first).

    `get_frame(frame)` returns a 2-D float image whose pixel (column, row) centre is at (column + 0.5, row + 0.5), the same
    coordinates as `points`, a list of dicts holding `x`, `y` and any of `in_x`, `in_y`, `out_x`, `out_y`, `feather_x`,
    `feather_y` (offsets from the anchor) at the reference frame. `planar` maps frame to the 3x3 matrix taking the reference
    frame's plane to that frame (`planar_matrices`); frames it does not list keep the last matrix, and None means the plane
    does not move.

    Returns `{frame: {"points": [...], "confidence": [...], "flagged": [...]}}` for every frame tracked, in order.
    """
    if len(points) < 2:
        raise PointTrackError("point tracking needs at least two points")
    reference_frame = int(reference_frame)
    frames = [int(f) for f in frames]
    count = len(points)
    reference = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64)
    offsets = [{field: float(p.get(field, 0.0)) for pair in OFFSET_FIELDS for field in pair} for p in points]
    reference_image = np.asarray(get_frame(reference_frame), dtype=np.float64)
    templates = [sample_patch(reference_image, reference[i, 0] - 0.5, reference[i, 1] - 0.5, patch_radius)
                 for i in range(count)]
    residual = np.zeros((count, 2), dtype=np.float64)     # each point's last own match minus the plane's position
    matrix = np.eye(3)
    output = {}
    for step, frame in enumerate(frames, start=1):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        if planar is not None and frame in planar:
            matrix = np.asarray(planar[frame], dtype=np.float64)
        image = np.asarray(get_frame(frame), dtype=np.float64)
        plane = _project(matrix, reference)                        # positions on the plane, pixel-centre convention
        own = np.zeros((count, 2))
        confidence = np.zeros(count)
        for i in range(count):
            centre = plane[i] - 0.5 + residual[i]
            match = match_patch(image, templates[i], centre[0], centre[1], patch_radius, search_radius)
            if match is None:
                own[i], confidence[i] = plane[i], 0.0
                continue
            own[i] = (match[0] + 0.5, match[1] + 0.5)
            confidence[i] = max(0.0, min(1.0, match[2]))
            if confidence[i] >= threshold:
                if confidence[i] < REGRAB:      # a patch that still matches well is kept, so its small errors do not add up
                    templates[i] = sample_patch(image, match[0], match[1], patch_radius)
                residual[i] = own[i] - plane[i]
        weight = np.clip((confidence - threshold) / BLEND_RANGE, 0.0, 1.0)
        blended = weight[:, None] * (own - plane)
        blended = smooth_offsets(blended, weight, closed, smoothing)
        position = plane + blended
        flagged = confidence < threshold
        position[flagged] = plane[flagged]
        new_points = []
        for i in range(count):
            jacobian = _jacobian(matrix, reference[i])
            point = {"x": float(position[i, 0]), "y": float(position[i, 1])}
            for fx, fy in OFFSET_FIELDS:
                moved = jacobian @ np.array([offsets[i][fx], offsets[i][fy]])
                point[fx], point[fy] = float(moved[0]), float(moved[1])
            new_points.append(point)
        output[frame] = {"points": new_points, "confidence": [float(c) for c in confidence],
                         "flagged": [bool(f) for f in flagged]}
        if progress is not None:
            progress(frame, step, len(frames))
    return output


# -- keys ----------------------------------------------------------------------------------------------------------------
def key_scalar(value, updates, reference_frame, auto_key=True):
    """Write `{frame: number}` into an animatable scalar as keys, following the Roto autokey rules.

    An already keyed scalar gets (or has changed) a key at each frame. A static scalar becomes keyed only with Autokey on,
    and then also keeps its current number as a key at the reference frame so the reference frame does not move; with Autokey
    off a static scalar is returned unchanged.
    """
    if not updates:
        return value
    if isinstance(value, dict) and value.get("curve"):
        result = {**value, "curve": {**value["curve"], "keys": [dict(k) for k in value["curve"]["keys"]]}}
        keys = result["curve"]["keys"]
    else:
        if not auto_key:
            return value
        base = float(value["value"] if isinstance(value, dict) else value)
        result = {"value": base, "curve": {"interpolation": "smooth", "keys": []}}
        keys = result["curve"]["keys"]
        updates = {int(reference_frame): base, **updates}
    index = {int(k["frame"]): k for k in keys}
    for frame, number in updates.items():
        if int(frame) in index:
            index[int(frame)]["value"] = float(number)
        else:
            keys.append({"frame": int(frame), "value": float(number)})
    keys.sort(key=lambda k: k["frame"])
    return result


def is_animated(shape):
    return any(isinstance(value, dict) and value.get("curve") for p in shape.get("points", []) for value in p.values())
