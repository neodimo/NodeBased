"""Image-aware in-betweens for Roto shapes (past Nuke).

Between two keyed frames a Roto shape is interpolated blindly, so the artist redraws every
in-between that the interpolation misses. ``fill_inbetweens`` gives them a draft instead: for each
frame strictly between two keys it takes today's interpolated shape, refines it to the image edges
of *that frame's* input with ``rotoedges.fit``, and proposes a key only on the points that the fit
moved by more than a threshold. Points that did not move keep the interpolation, and a frame whose
points all stayed put gets no key at all, so a shape the interpolation already fits gains nothing.

The module is pure (numpy and the existing curve helpers). It reads shapes through two callbacks so
the app, the tests and any batch tool supply them their own way:

* ``resolve(frame)`` returns the shape as the renderer sees it at ``frame``:
  ``{"points": [{x, y, in_x, in_y, out_x, out_y, ...}], "closed": bool, "curve_type": str}``.
* ``field_for(frame, box)`` returns a ``rotoedges.EdgeField`` over the data-space ``box``
  ``(x0, y0, x1, y1)`` of the input image at ``frame``, or ``None`` when there is no image.

The proposed keys are in the coordinates ``resolve`` returned. ``write_keys`` writes them into a
stored shape (``local`` maps a frame's resolved point into the shape's local space, the inverse of
the shape and tracker transforms) and keeps every other value of the shape's curves.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from . import animation, rotoedges

DEFAULT_BAND = 8.0          # px each point may travel toward an edge
DEFAULT_THRESHOLD = 0.75    # px a point must move before it gets a key
DEFAULT_STEP = 1            # fill every Nth frame between the keys
ITERATIONS = 12
NEW_CURVE_INTERPOLATION = "smooth"       # what Autokey gives a point coordinate that had no curve
FIELDS = rotoedges.POINT_FIELDS


@dataclass
class Result:
    """What a fill proposes for one shape.

    ``keys``      frame -> {point index -> the fitted point (all of its fields)}; only moved points.
    ``movement``  frame -> mean distance in px the fit moved the points of that frame (every
                  frame examined is listed, keyed or not).
    ``examined``  the frames the fit ran on.
    """
    keys: dict = field(default_factory=dict)
    movement: dict = field(default_factory=dict)
    examined: list = field(default_factory=list)
    skipped: list = field(default_factory=list)       # frames left alone: no image, or already keyed

    @property
    def frames_keyed(self):
        return len(self.keys)

    @property
    def key_count(self):
        """How many point keys are proposed (a key moves one point at one frame)."""
        return sum(len(points) for points in self.keys.values())


def frames_between(start, end, step=DEFAULT_STEP):
    """The frames strictly between two keys, every `step`-th from the first key's neighbour."""
    step = max(1, int(step))
    return list(range(int(start) + step, int(end), step))


def _outline_box(shape, margin):
    points = shape["points"]
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    try:
        from . import roto
        outline = roto.flatten_points(points, closed=shape.get("closed", True),
                                      curve_type=shape.get("curve_type", "bezier"))
        xs += [x for x, _ in outline]
        ys += [y for _, y in outline]
    except Exception:
        pass
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def fit_frame(shape, field_for, frame, band=DEFAULT_BAND, iterations=ITERATIONS):
    """The shape fitted to `frame`'s image, or None when that frame has no image."""
    if len(shape["points"]) < 2:
        return None
    field_ = field_for(frame, _outline_box(shape, band + 6.0))
    if field_ is None:
        return None
    return field_.fit({"points": shape["points"], "closed": shape.get("closed", True),
                       "curve_type": shape.get("curve_type", "bezier")}, band, iterations)


def fill_inbetweens(resolve, field_for, start, end, band=DEFAULT_BAND, threshold=DEFAULT_THRESHOLD,
                    step=DEFAULT_STEP, iterations=ITERATIONS, skip_frames=()):
    """Propose keys for the frames strictly between `start` and `end` (see the module docstring).

    `skip_frames` are frames that already hold a key of the artist's; they are never overwritten.
    """
    result = Result()
    skip = {int(f) for f in skip_frames}
    for frame in frames_between(start, end, step):
        if frame in skip:
            result.skipped.append(frame)
            continue
        base = resolve(frame)
        fitted = fit_frame(base, field_for, frame, band, iterations)
        if fitted is None:
            result.skipped.append(frame)
            continue
        result.examined.append(frame)
        moves = [math.hypot(float(new["x"]) - float(old["x"]), float(new["y"]) - float(old["y"]))
                 for old, new in zip(base["points"], fitted)]
        result.movement[frame] = sum(moves) / len(moves)
        moved = {i: fitted[i] for i, distance in enumerate(moves) if distance > threshold}
        if moved:
            result.keys[frame] = moved
    return result


# -- writing keys into a stored shape --------------------------------------------------------------

def key_frames(shape):
    """Every frame at which any coordinate of the stored shape has a key."""
    frames = set()
    for point in shape.get("points", []):
        for name in FIELDS:
            value = point.get(name)
            if isinstance(value, dict) and value.get("curve"):
                frames.update(int(k["frame"]) for k in value["curve"]["keys"])
    return sorted(frames)


def surrounding_keys(shape, frame):
    """(start, end): the keyed frames around `frame`, or None when it has no key on both sides.

    On a keyed frame the pair runs from that key to the next one.
    """
    frames = key_frames(shape)
    before = [f for f in frames if f <= frame]
    after = [f for f in frames if f > frame]
    if not before or not after:
        return None
    return before[-1], after[0]


def _with_key(value, frame, replacement, start, end):
    """A stored point coordinate with a key added at `frame`; a static one is first pinned at both ends."""
    if isinstance(value, dict) and value.get("curve"):
        updated = copy.deepcopy(value)
        updated["curve"] = animation.merge_key(updated["curve"], frame, replacement)
        updated["value"] = float(replacement)
        return updated
    base = float(value["value"]) if isinstance(value, dict) else float(value)
    curve = {"interpolation": NEW_CURVE_INTERPOLATION,
             "keys": [{"frame": int(start), "value": base}, {"frame": int(end), "value": base}]}
    curve = animation.merge_key(curve, frame, replacement)
    return {"value": float(replacement), "curve": curve}


def write_keys(shape, keys, start, end, local=None):
    """Write `Result.keys` into the stored `shape` (in place); returns the number of point keys.

    `local(frame, point)` converts a resolved point of that frame into the shape's local space
    (identity when omitted). Every field of a moved point is keyed; other frames keep their values.
    """
    count = 0
    for frame, moved in sorted(keys.items()):
        for index, point in moved.items():
            if index >= len(shape["points"]):
                continue
            new = local(frame, point) if local else point
            target = shape["points"][index]
            for name in FIELDS:
                target[name] = _with_key(target.get(name, 0.0), frame, float(new[name]), start, end)
            count += 1
    return count
