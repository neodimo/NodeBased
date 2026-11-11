"""Strokes that stick to the plate: carry a paint stroke with the picture (lane 9, step I7, past Nuke).

In Nuke a RotoPaint stroke painted on one frame stays where it was painted; to make it follow a moving plate the artist links
it to a tracker or builds a SmartVector and VectorDistort. Here a stroke gets a "Stick to plate" command that works out how the
picture under it moves and keys the stroke's points to follow.

Pure numpy, no document and no Qt. `stick_stroke` takes a frame source and the stroke's points at the painted frame and returns,
for each frame, the new points, the local scale and rotation of the motion (the brush size and the clone offset carry with
those), the confidence, and the frames and points that fell back to the planar motion of the stroke's region.

How it works, per frame after the painted one:

1. Dense optical flow (`opticalflow.flow_pair`) from the previous frame to this one is composed into a dense displacement field
   from the painted frame: `D(x) = D(x) + flow(x + D(x))`. Chaining is accurate for small steps but errors add up.
2. Every `CHECK_EVERY` frames the current frame is warped back by `D` and a second, direct flow against the painted frame (a
   small residual, so it is solved well) is added into `D`, so drift does not build up.
3. The stroke's points move by `D` sampled around each point (a small neighbourhood average). The local scale and rotation of
   `D` over the stroke give the brush size and clone offset factors.
4. The picture under the stroke, warped back by `D`, is compared with the painted frame (normalised cross-correlation at anchor
   points along the stroke). Where it no longer matches (the plate is covered, or the flow was wrong) the stroke falls back to
   the planar motion of its region (`planar.track_region`), which the occluding object does not outvote, and the frame is
   reported. `D` is then reset to the planar motion so the chain recovers when the plate shows again.

Coordinates are pixels with pixel centres at +0.5, the convention the Tracker, the planar tracker and Track points use.
"""
from __future__ import annotations

import math
from concurrent.futures import CancelledError

import numpy as np

from . import opticalflow

CHECK_EVERY = 3          # a direct check against the painted frame this often (and on the first frame after a reset)
NEIGHBOURHOOD = 3        # radius in pixels of the average of the displacement around each point
ANCHORS = 48             # the appearance check uses at most this many points along the stroke
PATCH_RADIUS = 6         # pixel radius of the patch compared with the painted frame
MATCH_THRESHOLD = 0.6    # a patch whose correlation with the painted frame falls under this is low confidence
FALLBACK_FRACTION = 0.25  # more than this share of low anchors puts the whole stroke on the planar motion
FLOW_DETAIL = 4
FLOW_SMOOTHNESS = 1.0
MAX_PLANAR_PADDING = 96.0
PLANE_SAMPLES = 12       # the region's plane is fitted through a grid of this many samples per side
PLANE_MATCH = 0.8        # a sample takes part in the plane when it still matches the painted frame this well
PLANE_MIN_SAMPLES = 12   # fewer matching samples than this and the planar tracker answers instead


class StickError(ValueError):
    pass


def frame_order(reference_frame, first_frame, last_frame, direction="forward"):
    """The frames after the painted one, nearest first, for `direction` forward, backward or both (two passes)."""
    from . import rotopointtrack
    try:
        return rotopointtrack.frame_order(reference_frame, first_frame, last_frame, direction)
    except rotopointtrack.PointTrackError as error:
        raise StickError(str(error)) from None


def _sample_vec(field, xs, ys):
    return opticalflow._sample(field, np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32))


def _anchor_indices(points, count):
    """Indices of at most `count` points spread evenly along the stroke's length."""
    n = len(points)
    if n <= count:
        return np.arange(n)
    steps = np.hypot(*np.diff(points, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(steps)))
    if along[-1] <= 0:
        return np.linspace(0, n - 1, count).round().astype(int)
    targets = np.linspace(0.0, along[-1], count)
    return np.unique(np.clip(np.searchsorted(along, targets), 0, n - 1))


def _patch_offsets(radius):
    side = np.arange(-radius, radius + 1, dtype=np.float32)
    gx, gy = np.meshgrid(side, side)
    return gx.ravel(), gy.ravel()


def _ncc(reference, current, xs, ys, radius):
    """Zero-mean correlation of `radius` patches of two images at the same index positions; flat patches score 0."""
    ox, oy = _patch_offsets(radius)
    px = (np.asarray(xs, np.float32)[:, None] + ox[None, :])
    py = (np.asarray(ys, np.float32)[:, None] + oy[None, :])
    a = opticalflow._sample(reference, px, py)
    b = opticalflow._sample(current, px, py)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denominator > 1e-9, (a * b).sum(axis=1) / denominator, 0.0)
    return score


def _average_displacement(field, points, radius):
    """The field averaged over a (2 radius + 1) square around each point (points in pixel-centre coordinates)."""
    side = np.arange(-radius, radius + 1, dtype=np.float32)
    gx, gy = np.meshgrid(side, side)
    px = (points[:, 0:1] - 0.5 + gx.ravel()[None, :]).astype(np.float32)
    py = (points[:, 1:2] - 0.5 + gy.ravel()[None, :]).astype(np.float32)
    return opticalflow._sample(field, px, py).mean(axis=1)


def local_motion(field, points, spread):
    """Mean 2x2 map of the displacement field over `points`: (scale, rotation in radians) of the motion around the stroke."""
    return local_motion_shear(field, points, spread)[:2]


def local_motion_shear(field, points, spread):
    """(scale, rotation in radians, shear) of the motion around the stroke.

    The mean 2x2 map J of the picture under the stroke is split as ``J = s R(theta) K``: `s` the scale (square root of the
    determinant), `R` the turn (as before, from the antisymmetric part) and ``K`` what is left, whose off-diagonal part is the
    shear: ``K = [[1, k], [k, 1]]`` to first order, `k` returned. A picture that only turns and scales has shear 0."""
    h, w = field.shape[:2]
    s = float(max(2.0, spread))
    anchors = points[_anchor_indices(points, ANCHORS)]
    bx, by = anchors[:, 0] - 0.5, anchors[:, 1] - 0.5
    right = _sample_vec(field, np.clip(bx + s, 0, w - 1), by)
    left = _sample_vec(field, np.clip(bx - s, 0, w - 1), by)
    down = _sample_vec(field, bx, np.clip(by + s, 0, h - 1))
    up = _sample_vec(field, bx, np.clip(by - s, 0, h - 1))
    run_x = np.clip(bx + s, 0, w - 1) - np.clip(bx - s, 0, w - 1)
    run_y = np.clip(by + s, 0, h - 1) - np.clip(by - s, 0, h - 1)
    good = (run_x > 0) & (run_y > 0)
    if not good.any():
        return 1.0, 0.0, 0.0
    dxx = ((right - left)[good] / run_x[good, None])
    dyy = ((down - up)[good] / run_y[good, None])
    jacobian = np.eye(2) + np.array([[dxx[:, 0].mean(), dyy[:, 0].mean()], [dxx[:, 1].mean(), dyy[:, 1].mean()]])
    determinant = float(np.linalg.det(jacobian))
    scale = math.sqrt(determinant) if determinant > 1e-6 else 1.0
    rotation = math.atan2(jacobian[1, 0] - jacobian[0, 1], jacobian[0, 0] + jacobian[1, 1])
    scale = min(max(scale, 0.25), 4.0)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    residual = np.array([[cosine, sine], [-sine, cosine]]) @ jacobian / scale        # (s R)^-1 J
    shear = float(np.clip(0.5 * (residual[0, 1] + residual[1, 0]), -0.9, 0.9))
    return float(scale), float(rotation), shear


def padded_quad(points, brush_size, bounds):
    """The stroke's bounding quad grown by the brush and some surrounding picture, kept inside `bounds` (width, height)."""
    xs, ys = points[:, 0], points[:, 1]
    pad = min(MAX_PLANAR_PADDING, max(32.0, float(brush_size) * 1.5))
    width, height = bounds
    left, right = max(0.0, xs.min() - pad), min(float(width), xs.max() + pad)
    top, bottom = max(0.0, ys.min() - pad), min(float(height), ys.max() + pad)
    if right - left < 8:
        left, right = max(0.0, left - 4), min(float(width), right + 4)
    if bottom - top < 8:
        top, bottom = max(0.0, top - 4), min(float(height), bottom + 4)
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _quad_grid(quad, side):
    """A `side` x `side` grid of sample points inside the quad (pixel-centre coordinates)."""
    left, top = min(q[0] for q in quad), min(q[1] for q in quad)
    right, bottom = max(q[0] for q in quad), max(q[1] for q in quad)
    us = np.linspace(0.06, 0.94, side)
    return np.array([(left + (right - left) * a, top + (bottom - top) * b) for b in us for a in us], dtype=np.float64)


def _planar_field(matrix, shape):
    """The displacement field of a homography over an image of `shape` (index coordinates)."""
    from .planar import project_points
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    grid = np.stack((xx.ravel() + 0.5, yy.ravel() + 0.5), axis=1)
    moved = project_points(matrix, grid)
    return (moved - grid).reshape(h, w, 2).astype(np.float32)


def stick_stroke(get_frame, points, reference_frame, frames, *, brush_size=25.0, direction_hint=None,
                 backend="cpu", check_every=CHECK_EVERY, match_threshold=MATCH_THRESHOLD, cancel=None, progress=None):
    """Move a stroke's `points` with the picture through `frames` (an ordered list, nearest the painted frame first).

    `get_frame(frame)` returns a 2-D float image whose pixel (column, row) centre is at (column + 0.5, row + 0.5), the same
    coordinates as `points`, a list of dicts holding `x` and `y` (and anything else, which is ignored) at `reference_frame`.

    Returns `{frame: {"points": [(x, y), ...], "scale": s, "rotation": r, "confidence": c, "fallback": bool,
    "low": [point indices on the planar motion]}}` for every frame tracked, in order. `scale` is the local scale of the motion
    around the stroke (1 where the picture does not stretch) and `rotation` its turn in radians.
    """
    if not points:
        raise StickError("a stroke with no points cannot stick to the plate")
    reference_frame = int(reference_frame)
    frames = [int(f) for f in frames]
    reference_points = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64)
    count = len(reference_points)
    reference_image = np.asarray(get_frame(reference_frame), dtype=np.float32)
    if reference_image.ndim != 2:
        raise StickError("the picture under the stroke must be a single channel")
    height, width = reference_image.shape
    if not (np.isfinite(reference_points).all()):
        raise StickError("the stroke has points that are not finite numbers")
    anchors = _anchor_indices(reference_points, ANCHORS)
    anchor_points = reference_points[anchors]
    spread = max(4.0, float(brush_size) * 0.5)
    field = np.zeros((height, width, 2), np.float32)         # painted-frame pixel -> displacement in the current frame
    previous = reference_image
    planar_cache = {}
    output = {}
    last_checked = None

    quad = padded_quad(reference_points, brush_size, (width, height))
    grid = _quad_grid(quad, PLANE_SAMPLES)
    last_plane = [None]

    def region_plane(warped):
        """The plane the stroke's region sits on: a robust homography through the samples that still match the painted frame."""
        from . import planar
        good = _ncc(reference_image, warped, grid[:, 0] - 0.5, grid[:, 1] - 0.5, 5) >= PLANE_MATCH
        if int(good.sum()) < PLANE_MIN_SAMPLES:
            return None
        moved = grid[good] + _sample_vec(field, grid[good, 0] - 0.5, grid[good, 1] - 0.5)
        try:
            matrix, inliers, _rms = planar.fit_homography_ransac(grid[good], moved, threshold=1.0)
        except Exception:
            return None
        if int(inliers.sum()) < PLANE_MIN_SAMPLES or not np.isfinite(matrix).all():
            return None
        return matrix

    def tracked_plane(frame):
        """The planar tracker's answer for the region, used when too little of it still matches to fit a plane directly."""
        if not planar_cache:
            from . import planar
            numbers = sorted({reference_frame, *frames})
            images = {}
            for number in range(numbers[0], numbers[-1] + 1):
                if cancel is not None and cancel.is_set():
                    raise CancelledError()
                gray = np.asarray(get_frame(number), dtype=np.float32)
                images[number] = np.repeat(gray[..., None], 3, axis=2)
            try:
                results = planar.track_region(images, np.asarray(quad, dtype=np.float64), reference_frame,
                                              samples_per_axis=12, vector_detail=FLOW_DETAIL, smoothness=FLOW_SMOOTHNESS,
                                              backend="cpu", threshold=2.0, cancel=cancel)
                planar_cache.update({f: np.asarray(r["matrix"], dtype=np.float64) for f, r in results.items()})
            except planar.PlanarTrackingError:
                planar_cache["unavailable"] = True
        return None if planar_cache.get("unavailable") else planar_cache.get(frame)

    def planar_for(frame, warped):
        matrix = region_plane(warped)
        if matrix is None:
            matrix = tracked_plane(frame)
        if matrix is None:
            matrix = last_plane[0]
        if matrix is not None:
            last_plane[0] = matrix
        return matrix


    for step, frame in enumerate(frames, start=1):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        image = np.asarray(get_frame(frame), dtype=np.float32)
        forward, _backward, _occluded = opticalflow.flow_pair(
            previous, image, vector_detail=FLOW_DETAIL, smoothness=FLOW_SMOOTHNESS, flow_on="luminance", backend=backend)
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        step_flow = opticalflow._sample(forward, xx + field[..., 0], yy + field[..., 1])
        field = (field + step_flow).astype(np.float32)
        # The picture at this frame, warped back to the painted frame's pixels.
        warped = opticalflow._sample(image, xx + field[..., 0], yy + field[..., 1])
        if last_checked is None or (step % max(1, int(check_every)) == 0):
            residual, _bw, _oc = opticalflow.flow_pair(reference_image, warped, vector_detail=FLOW_DETAIL,
                                                       smoothness=FLOW_SMOOTHNESS, flow_on="luminance", backend=backend)
            nudged = opticalflow._sample(field, xx + residual[..., 0], yy + residual[..., 1]) + residual
            field = nudged.astype(np.float32)
            warped = opticalflow._sample(image, xx + field[..., 0], yy + field[..., 1])
            last_checked = step
        score = _ncc(reference_image, warped, anchor_points[:, 0] - 0.5, anchor_points[:, 1] - 0.5, PATCH_RADIUS)
        low_anchor = score < match_threshold
        moved = reference_points + _average_displacement(field, reference_points, NEIGHBOURHOOD)
        scale, rotation, shear = local_motion_shear(field, reference_points, spread)
        low = np.zeros(count, dtype=bool)
        fallback = False
        if low_anchor.any():
            matrix = planar_for(frame, warped)
            if matrix is not None:
                from .planar import project_points
                plane = project_points(matrix, reference_points)
                if float(low_anchor.mean()) > FALLBACK_FRACTION:
                    fallback = True
                    low[:] = True
                    moved = plane
                    plane_field = _planar_field(matrix, (height, width))
                    field = plane_field
                    scale, rotation, shear = local_motion_shear(plane_field, reference_points, spread)
                else:
                    nearest = np.abs(np.arange(count)[:, None] - anchors[None, :]).argmin(axis=1)
                    low = low_anchor[nearest]
                    moved = np.where(low[:, None], plane, moved)
            else:
                low[:] = low_anchor[np.abs(np.arange(count)[:, None] - anchors[None, :]).argmin(axis=1)]
        output[frame] = {"points": [(float(x), float(y)) for x, y in moved], "scale": float(scale),
                         "rotation": float(rotation), "shear": float(shear), "confidence": float(np.clip(score.mean(), 0.0, 1.0)),
                         "fallback": bool(fallback), "low": [int(i) for i in np.flatnonzero(low)]}
        previous = image
        if progress is not None:
            progress(frame, step, len(frames))
    return output


def key_stroke(item, results, reference_frame):
    """A copy of stroke `item` whose points, brush size and clone offset follow `results` of `stick_stroke`, as keys.

    The points' x and y become animatable scalars keyed at every tracked frame and at the painted frame (where the stroke stays
    exactly as it was painted). `stick` holds the scale and rotation of the motion as scalars keyed the same way, 1 and 0 at the
    painted frame; the painter multiplies the brush size by the scale and turns and scales the clone offset by them.
    """
    from . import rotopointtrack as point_track
    out = dict(item)
    reference_frame = int(reference_frame)
    points = []
    for index, point in enumerate(item["points"]):
        moved = dict(point)
        for axis, field in enumerate(("x", "y")):
            series = {frame: data["points"][index][axis] for frame, data in results.items()}
            moved[field] = point_track.key_scalar(point[field], series, reference_frame, auto_key=True)
        points.append(moved)
    out["points"] = points
    stick = dict(item.get("stick") or {})
    fields = [("scale", 1.0), ("rotation", 0.0)]
    if any(abs(data.get("shear", 0.0)) > 1e-6 for data in results.values()) or "shear" in stick:
        fields.append(("shear", 0.0))                  # keyed only when the picture shears, so an unsheared stroke is saved as before
    for field, neutral in fields:
        series = {frame: data.get(field, neutral) for frame, data in results.items()}
        stick[field] = point_track.key_scalar(stick.get(field, neutral), series, reference_frame, auto_key=True)
    out["stick"] = stick
    return out


def fallback_frames(results):
    """Frames where the whole stroke was on the planar motion, and the frames where only some points were, in order."""
    whole = sorted(f for f, data in results.items() if data["fallback"])
    partial = sorted(f for f, data in results.items() if data["low"] and not data["fallback"])
    return whole, partial


def describe_fallback(name, results):
    """A status-bar phrase naming the frames a stroke fell back to the planar motion, or an empty string."""
    whole, partial = fallback_frames(results)
    parts = []
    for label, frames in (("planar motion", whole), ("planar motion for some points", partial)):
        if frames:
            runs, start, last = [], frames[0], frames[0]
            for frame in frames[1:]:
                if frame != last + 1:
                    runs.append((start, last)); start = frame
                last = frame
            runs.append((start, last))
            parts.append(f"{label} on frames " + ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in runs))
    return f"{name}: " + "; ".join(parts) if parts else ""


# -- a stroke with a transform of its own -----------------------------------------------------------------------------------
# A stroke's points are keyed in its own space and Roto then draws them through the stroke's transform (centre, translate,
# rotate, scale, skew, and the 4 x 4 matrix). The picture is followed where the stroke is drawn, so every frame's tracked points
# are taken back through the inverse of that frame's transform before they are keyed, as Track points does for shapes.

def _transform_parts(item, frame):
    """(A, translate, E, offset, stored_centre, brush_factor) of the stroke's transform at `frame`: the linear part A (skew, then
    scale, then rotate), the translation, the matrix `extra_matrix` as (2 x 2 linear, offset), the centre where one is stored
    (None where the default, the middle of the points' bounding box, applies) and the factor the brush size scales by."""
    from . import shapes as shape_model
    stored = item.get("transform") or {}
    t = {name: shape_model.resolve_scalar(stored.get(name, default), int(frame), name)
         for name, default in shape_model.TRANSFORM_DEFAULTS.items()}
    t["skew_order"] = stored.get("skew_order", "XY")
    angle = math.radians(t["rotate"])
    rotate = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    skew = np.array(shape_model._skew_matrix(t)).reshape(2, 2) if (t["skew_x"] or t["skew_y"]) else np.eye(2)
    linear = rotate @ np.diag([t["scale_x"], t["scale_y"]]) @ skew
    centre = [float(t["center_x"]) if "center_x" in stored else None, float(t["center_y"]) if "center_y" in stored else None]
    matrix = item.get("extra_matrix")
    if matrix is not None:
        m = [float(v) for v in matrix]
        extra, offset = np.array([[m[0], m[1]], [m[4], m[5]]]), np.array([m[3], m[7]])
    else:
        extra, offset = np.eye(2), np.zeros(2)
    area = abs(float(np.linalg.det(linear))) * abs(float(np.linalg.det(extra)))
    return linear, np.array([t["translate_x"], t["translate_y"]]), extra, offset, centre, math.sqrt(area)


def stroke_brush_factor(item, frame):
    """How the stroke's own transform scales its brush size at `frame` (the square root of the transform's area; 1 for none)."""
    if "transform" not in item and "extra_matrix" not in item:
        return 1.0
    return _transform_parts(item, frame)[5]


def unresolve_stroke_points(item, rendered, frame):
    """The stroke's own-space points that Roto draws at `rendered` (a list of (x, y)) through the transform at `frame`.

    Inverts, in the order Roto applies them: the 4 x 4 matrix, then the translation, then skew, scale and rotation about the
    centre. A centre the stroke does not store is the middle of its own-space points' bounding box, which depends on the answer;
    that closes exactly, because the bounding box of the points moves with the centre as ``c - A^-1 c`` and the answer is
    ``c = A * middle(A^-1 (R - t))``. Returns an (n, 2) array. A stroke with no transform returns `rendered` unchanged."""
    rendered = np.asarray(rendered, dtype=np.float64).reshape(-1, 2)
    if "transform" not in item and "extra_matrix" not in item:
        return rendered
    linear, translate, extra, offset, stored, _ = _transform_parts(item, frame)
    try:
        inverse = np.linalg.inv(linear)
        inverse_extra = np.linalg.inv(extra)
    except np.linalg.LinAlgError:
        return rendered
    drawn = (rendered - offset) @ inverse_extra.T                    # undo the matrix, applied last
    base = (drawn - translate) @ inverse.T                           # A^-1 (R - t)
    low, high = base.min(axis=0), base.max(axis=0)
    middle = (low + high) / 2.0
    centre = np.array([0.0, 0.0])
    known = [c is not None for c in stored]
    if not any(known):
        centre = linear @ middle
    elif all(known):
        centre = np.array(stored)
    else:
        axis = 0 if not known[0] else 1                              # the axis that is not stored
        other = 1 - axis
        centre[other] = stored[other]
        if abs(inverse[axis, axis]) > 1e-12:
            centre[axis] = (middle[axis] - inverse[axis, other] * centre[other]) / inverse[axis, axis]
        else:
            centre[axis] = middle[axis]
    return base + centre - centre @ inverse.T


def localise_results(item, results, reference_frame):
    """Take `stick_stroke` results measured on a stroke drawn through its own transform back into its own space.

    Each frame's points go through the inverse of that frame's transform, and the scale (which multiplies the brush size)
    is divided by what the transform itself does to the brush on that frame, so the drawn brush follows the picture's scale
    from its size on the painted frame. Strokes with no transform come back untouched."""
    if "transform" not in item and "extra_matrix" not in item:
        return results
    base = stroke_brush_factor(item, reference_frame)
    out = {}
    for frame, data in results.items():
        local = unresolve_stroke_points(item, data["points"], frame)
        factor = stroke_brush_factor(item, frame)
        out[frame] = dict(data, points=[(float(x), float(y)) for x, y in local],
                          scale=float(data["scale"] * (base / factor if factor > 1e-9 else 1.0)))
    return out
