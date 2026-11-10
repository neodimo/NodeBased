"""Ripple editing for Roto and RotoPaint shapes.

Nuke's "Deleting or Rippling Keyframes" page describes it: with the ripple edit button on, a change made to a point at
one frame is applied as the same relative adjustment to that point across all frames or a chosen range of frames, so a
non-animated correction can be made to a shape that is being animated.

`ripple_shapes` takes the shapes as they were and the same shapes after an ordinary viewer edit, and returns the edited
shapes with every changed point scalar rippled: each key of that scalar inside the range moves by the same offset the
edit gave it at the current frame, and keys outside the range stay put. The caller commits the result in the one command
that the ordinary edit used, so the whole ripple is one undo step.
"""
import copy
import math

from . import shapes as shape_model

MODES = ("all", "from start", "to end", "range")
DEFAULT_MODE = "all"
EPSILON = 1e-9


def frame_window(mode, current, first, last, start=None, end=None):
    """The (low, high) frames a ripple touches, both included, for one of Nuke's four range choices."""
    first, last, current = int(first), int(last), int(current)
    if mode == "from start":
        return first, current
    if mode == "to end":
        return current, last
    if mode == "range":
        a = first if start is None else int(round(float(start)))
        b = last if end is None else int(round(float(end)))
        return (a, b) if a <= b else (b, a)
    return first, last


def settings_window(shared, document_time):
    """The ripple window for the stored global settings, or None when ripple edit is off."""
    if not bool(shared.get("ripple_edit", False)):
        return None
    mode = shared.get("ripple_mode", DEFAULT_MODE)
    if mode not in MODES:
        mode = DEFAULT_MODE
    return frame_window(mode, document_time["current"], document_time["first"], document_time["last"],
                        shared.get("ripple_from"), shared.get("ripple_to"))


def _bounds(name):
    return shape_model.SHAPE_LIMITS.get(name, (-math.inf, math.inf))


def _has_curve(value):
    return isinstance(value, dict) and bool(value.get("curve"))


def ripple_scalar(old, new, frame, low, high, name=None):
    """`new` is `old` after an edit at `frame`; return the scalar with that edit rippled over frames `low`..`high`."""
    frame = int(frame)
    before = shape_model.resolve_scalar(old, frame)
    after = shape_model.resolve_scalar(new, frame)
    delta = after - before
    if abs(delta) < EPSILON:
        return new
    if not _has_curve(old):
        # Nothing is animated: the same offset at every frame is simply the new static value.
        return (copy.deepcopy(old) | {"value": after}) if isinstance(old, dict) else float(after)
    lo, hi = _bounds(name)
    result = copy.deepcopy(old)
    keys = result["curve"]["keys"]
    for key in keys:
        if low <= key["frame"] <= high:
            key["value"] = min(max(float(key["value"]) + delta, lo), hi)
    if low <= frame <= high:
        # Make the edited frame land exactly on the number the artist set, adding a key only when the shifted curve misses it.
        shifted = shape_model.resolve_scalar(result, frame)
        if abs(shifted - after) > 1e-6:
            index = {int(k["frame"]): k for k in keys}
            if frame in index:
                index[frame]["value"] = float(after)
            else:
                keys.append({"frame": frame, "value": float(after)})
                keys.sort(key=lambda k: k["frame"])
    return result


POINT_SCALARS = (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS, *shape_model.POINT_TENSION_FIELDS)


def ripple_shapes(old_shapes, new_shapes, frame, low, high):
    """Ripple every changed point scalar of `new_shapes` against `old_shapes`; shapes that gained or lost points are left as edited."""
    result = copy.deepcopy(new_shapes)
    for shape, before in zip(result, old_shapes):
        points, old_points = shape.get("points", []), before.get("points", [])
        if len(points) != len(old_points):
            continue
        for point, old_point in zip(points, old_points):
            for field in POINT_SCALARS:
                if field in point and field in old_point and point[field] != old_point[field]:
                    point[field] = ripple_scalar(old_point[field], point[field], frame, low, high, field)
    return result
