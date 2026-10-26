"""Nuke's RotoPaint settings of fidelity step H that live on each shape and stroke of the node's payload, and on the node itself.

A RotoPaint payload is an ordered list of items, a `shape` (a Roto spline drawn as a white matte over the picture) or a `stroke`
(a brush path). Step H lets an item carry the settings Nuke's tabs show for the selected curve. Each is optional and an item
without it draws as it always did.

Both kinds:

* `locked`: the item cannot be edited in the viewer; the pixels do not change.
* `source`: what the item paints. `color` (the default) paints the stroke's colour, or white for a shape; `foreground` paints the
  `input2` image where the item's matte is; `background` paints the node's plate (the picture before any item drew), which is how
  a stroke removes earlier paint.
* `transform` (translate, rotate, scale, skew, centre, skew order) and `extra_matrix`: the Transform tab. The item's points and a
  stroke's brush size move with it, the same arithmetic as Roto's shapes (`shapes.apply_shape_transform`).
* `motion_blur`, `motion_blur_samples`, `motion_blur_shutter`, `motion_blur_offset_type`, `motion_blur_offset`: the Motion Blur
  tab, and the node's `shape_motionblur_on` and `global_motionblur*` knobs. An item blurs by drawing at several times across the
  shutter and averaging; an item with no animation is sharp.

Shapes also take every Roto shape setting (`feather_on`, `falloff_power`, `feather_falloff`, `invert`, `cap_start`, `cap_end`,
`closed`, `stroke_width`, `feather_model`, `lifetime`).

Strokes:

* `effect`: the Stroke tab's effect parameter, 0 (the default) meaning the brush's own settings. Blur and sharpen: the filter
  radius in pixels (otherwise a sixth of the brush size); smear: how far the colour is dragged; dodge and burn: the strength.
* `dynamic_hardness`: pressure scales the brush's hardness along the stroke.
* `buildup`: overlapping dabs add up, `1 - (1 - a)(1 - a_dab)`, instead of the strongest dab alone setting the opacity.
* `write_on`: `{"start": s, "end": e}`, fractions of the stroke's length; only that part of the path is drawn.
* `source_transform` (rotate, scale, skew X and Y, skew order, centre), `source_round`, `source_filter`, `source_black_outside`:
  the Clone tab. A clone samples its source rotated, scaled and skewed about the centre (before the source offset translates it);
  `source_round` rounds the offset to whole pixels; `source_black_outside` makes the pixels the source does not cover black and
  transparent instead of repeating its edge; `source_filter` is the resampling filter.
"""
from __future__ import annotations

import math

import numpy as np

from . import rotoknobs, shapes

SOURCES = ("color", "foreground", "background")
CLONE_FILTERS = ("impulse", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch")
# Item settings both kinds take.
COMMON_FIELDS = ("locked", "source", "transform", "extra_matrix", "motion_blur", *shapes.MOTION_BLUR_FIELDS)
SHAPE_ITEM_FIELDS = (*COMMON_FIELDS, "invert", "feather_falloff", "feather_model", "closed", "stroke_width", "feather_on",
                     "falloff_power", "lifetime", "cap_start", "cap_end")
STROKE_ITEM_FIELDS = (*COMMON_FIELDS, "effect", "dynamic_hardness", "buildup", "write_on", "source_transform", "source_round",
                      "source_filter", "source_black_outside")
SOURCE_TRANSFORM_DEFAULTS = {"rotate": 0.0, "scale_x": 1.0, "scale_y": 1.0, "skew_x": 0.0, "skew_y": 0.0,
                             "center_x": 0.0, "center_y": 0.0}

# The node's own settings: the output mask layer and the global motion blur are Roto's (nodebased/rotoknobs.py); the premultiply
# knob is the same `premultiply_mode` Roto has.
PARAMS = {"premultiply_mode": "none", **rotoknobs.PARAMS}
KINDS = ("RotoPaint",)


def new_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def premultiply(pixels, params):
    """The node's `premultiply`: rgb (and with rgba the alpha too) multiplied by the alpha of the result."""
    mode = params.get("premultiply_mode", "none")
    if mode not in ("rgb", "rgba"):
        return pixels
    out = pixels.copy()
    alpha = pixels[..., 3:4]
    out[..., :3] = pixels[..., :3] * alpha
    if mode == "rgba":
        out[..., 3:4] = alpha * alpha
    return out


def _number(value):
    return type(value) in (float, int) and not isinstance(value, bool) and math.isfinite(value)


# ------------------------------------------------------------------------------------------------------------ validation
def validate_common(item, where):
    """Check the settings both kinds take, raising ValueError with `where` on the first fault."""
    if "locked" in item and type(item["locked"]) is not bool:
        raise ValueError(f"{where}.locked must be a boolean")
    if item.get("source", "color") not in SOURCES:
        raise ValueError(f"{where}.source must be one of {list(SOURCES)}")
    probe = {"name": "x", "mode": "union", "opacity": 1.0, "feather": 0.0,
             "points": [{field: 0.0 for field in shapes.POINT_FIELDS}] * 3}
    for field in ("transform", "extra_matrix", "motion_blur", *shapes.MOTION_BLUR_FIELDS):
        if field in item:
            probe[field] = item[field]
    shapes.validate_shape(probe, where)


def validate_stroke_extras(item, where):
    validate_common(item, where)
    if "effect" in item and not (_number(item["effect"]) and 0.0 <= item["effect"] <= 4096.0):
        raise ValueError(f"{where}.effect must be a number between 0 and 4096")
    for field in ("dynamic_hardness", "buildup", "source_round", "source_black_outside"):
        if field in item and type(item[field]) is not bool:
            raise ValueError(f"{where}.{field} must be a boolean")
    if "write_on" in item:
        write_on = item["write_on"]
        if (not isinstance(write_on, dict) or set(write_on) != {"start", "end"}
                or not all(_number(write_on[k]) and 0.0 <= write_on[k] <= 1.0 for k in write_on)):
            raise ValueError(f"{where}.write_on defines start and end between 0 and 1")
    if item.get("source_filter", "cubic") not in CLONE_FILTERS:
        raise ValueError(f"{where}.source_filter must be one of {list(CLONE_FILTERS)}")
    transform = item.get("source_transform", {})
    if not isinstance(transform, dict) or set(transform) - set(SOURCE_TRANSFORM_DEFAULTS) - {"skew_order"}:
        raise ValueError(f"{where}.source_transform contains unsupported fields")
    if transform.get("skew_order", "XY") not in shapes.SKEW_ORDERS:
        raise ValueError(f"{where}.source_transform.skew_order must be one of {list(shapes.SKEW_ORDERS)}")
    for field, value in transform.items():
        if field != "skew_order" and not _number(value):
            raise ValueError(f"{where}.source_transform.{field} must be a number")


def validate_shape_extras(item, where):
    """The shape item's extra settings: those of Roto's shape and the common ones, checked as a Roto shape is."""
    validate_common(item, where)
    probe = {k: item[k] for k in ("name", "mode", "opacity", "feather", "points")}
    for field in SHAPE_ITEM_FIELDS:
        if field in item and field not in ("locked", "source"):
            probe[field] = item[field]
    shapes.validate_shape(probe, where)


# --------------------------------------------------------------------------------------------------- resolving
def shape_input(item):
    """The Roto shape a shape item is resolved as: its five required fields and every setting it carries."""
    shape = {k: item[k] for k in ("name", "mode", "opacity", "feather", "points")}
    for field in SHAPE_ITEM_FIELDS:
        if field in item and field not in ("locked", "source"):
            shape[field] = item[field]
    return shape


def needs_resolving(item):
    return any(field in item for field in ("transform", "extra_matrix", "motion_blur", *shapes.MOTION_BLUR_FIELDS))


def resolve_stroke(item, frame, params):
    """The stroke at `frame` with its transform applied and, when motion blur applies, one set of points per shutter sample.

    A stroke with none of that comes back untouched. The brush size follows the transform's scale (the square root of its area)."""
    fake = {"name": "stroke", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": False,
            "stroke_width": float(item["brush"]["size"]),
            "points": [{**{field: 0.0 for field in shapes.POINT_FIELDS}, "x": float(p["x"]), "y": float(p["y"])}
                       for p in item["points"]]}
    for field in ("transform", "extra_matrix", "motion_blur", *shapes.MOTION_BLUR_FIELDS):
        if field in item:
            fake[field] = item[field]
    if not any(field in item for field in ("transform", "extra_matrix")) and shapes.motion_blur_plan(fake, params) is None:
        return item
    resolved = shapes._resolve_shape(fake, frame, params, blur=True)

    def points(raw):
        return [dict(old, x=new["x"], y=new["y"]) for old, new in zip(item["points"], raw)]
    out = dict(item)
    out["points"] = points(resolved["points"])
    scale = resolved["stroke_width"] / max(float(item["brush"]["size"]), 1e-9)
    if scale != 1.0:
        out["brush"] = {**item["brush"], "size": max(0.1, float(item["brush"]["size"]) * scale)}
    samples = resolved.get("_motion_samples")
    if samples and any(sample != resolved["points"] for sample in samples):
        out["_motion_samples"] = [points(raw) for raw in samples]       # a stroke that does not move is sharp
    return out


def shape_picture(item, plate, reveal, base):
    """The picture a shape paints (rgba, the size of `base`): the second image for `foreground`, the plate for `background`, or
    None for the default `color`, where the shape paints white. With nothing wired to the second input foreground paints black."""
    source = item.get("source", "color")
    if source == "color":
        return None
    picture = reveal if source == "foreground" else plate
    if picture is None or np.asarray(picture).shape[:2] != base.shape[:2]:
        return np.zeros_like(base)
    return np.asarray(picture, np.float32)


def write_on_points(points, start, end):
    """The part of a stroke's path between the fractions `start` and `end` of its length, the cut ends interpolated (position and
    pressure). One point stands for a path of no length: it is drawn when the range covers any of it."""
    if len(points) < 2:
        return points if end > start else []
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
    total = lengths[-1]
    if total <= 0.0:
        return points if end > start else []
    lo, hi = start * total, end * total
    if hi <= lo:
        return []

    def at(distance):
        for index in range(1, len(points)):
            if distance <= lengths[index] or index == len(points) - 1:
                span = lengths[index] - lengths[index - 1]
                t = 0.0 if span <= 0.0 else (distance - lengths[index - 1]) / span
                a, b = points[index - 1], points[index]
                return {**a, "x": a["x"] + (b["x"] - a["x"]) * t, "y": a["y"] + (b["y"] - a["y"]) * t,
                        "pressure": a["pressure"] + (b["pressure"] - a["pressure"]) * t}
        return dict(points[-1])
    middle = [dict(p) for p, length in zip(points, lengths) if lo < length < hi]
    return [at(lo), *middle, at(hi)]


# -------------------------------------------------------------------------------------------------------- clone
def clone_needs_whole_source(stroke):
    """True when a clone's source is rotated, scaled or skewed, so a tile may read it from anywhere in the source."""
    transform = stroke.get("source_transform") or {}
    moved = any(transform.get(k, d) != d for k, d in SOURCE_TRANSFORM_DEFAULTS.items() if k not in ("center_x", "center_y"))
    return moved


def clone_coordinates(stroke, x, y):
    """Where in the source each destination pixel (`x`, `y`, image coordinates of the pixel centres) reads: the inverse of
    skew, then scale, then rotate about the centre, then the source offset (rounded to whole pixels with `source_round`)."""
    dx, dy = stroke["source_offset"]
    if stroke.get("source_round"):
        dx, dy = round(dx), round(dy)
    t = {**SOURCE_TRANSFORM_DEFAULTS, **(stroke.get("source_transform") or {})}
    cx, cy = t["center_x"], t["center_y"]
    u, v = x - dx - cx, y - dy - cy
    angle = math.radians(t["rotate"])
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * u + s * v, -s * u + c * v                      # undo the rotation
    u, v = u / (t["scale_x"] or 1e-9), v / (t["scale_y"] or 1e-9)
    if t["skew_x"] or t["skew_y"]:
        a, b, cc, d = shapes._skew_matrix({"skew_x": t["skew_x"], "skew_y": t["skew_y"], "skew_order": t.get("skew_order", "XY")})
        det = a * d - b * cc
        if abs(det) > 1e-12:
            u, v = (d * u - b * v) / det, (-cc * u + a * v) / det
    return u + cx, v + cy


def items_need_full_frame(items):
    """True when a clone stroke resamples its source through a transform or a filter wider than a pixel: the source window a
    tile would fetch is then not the one the stroke reads, so the whole picture is drawn at once."""
    for item in items:
        if item.get("kind") == "stroke" and item.get("tool") == "clone":
            if clone_needs_whole_source(item) or item.get("source_filter", "impulse") != "impulse":
                return True
    return False


def effective_effect(stroke, fallback):
    """The stroke's effect parameter, or `fallback` while it is 0."""
    effect = float(stroke.get("effect", 0.0))
    return effect if effect > 0.0 else fallback


# ------------------------------------------------------------------------------------------------ clone resampling
# The clone's resampling kernels (the guide names the eight filters and gives no formula, so these are our readings): impulse is
# the nearest pixel; cubic and keys are the Keys cubic convolution with a = -0.5 (Catmull-Rom, one function under both names);
# simon and rifman are the same family with a = -0.75 and -1.0, each a little sharper; mitchell is the Mitchell-Netravali filter
# (B = C = 1/3); parzen is the cubic B-spline (soft); notch is the two-pixel tent (the softest, bilinear).
_KEYS_A = {"cubic": -0.5, "keys": -0.5, "simon": -0.75, "rifman": -1.0}


def _keys(x, a):
    x = np.abs(x)
    near = (a + 2.0) * x ** 3 - (a + 3.0) * x ** 2 + 1.0
    far = a * x ** 3 - 5.0 * a * x ** 2 + 8.0 * a * x - 4.0 * a
    return np.where(x <= 1.0, near, np.where(x < 2.0, far, 0.0))


def _mitchell(x, b, c):
    x = np.abs(x)
    near = ((12 - 9 * b - 6 * c) * x ** 3 + (-18 + 12 * b + 6 * c) * x ** 2 + (6 - 2 * b)) / 6.0
    far = ((-b - 6 * c) * x ** 3 + (6 * b + 30 * c) * x ** 2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6.0
    return np.where(x < 1.0, near, np.where(x < 2.0, far, 0.0))


def kernel_weight(name, x):
    """The weight of a source pixel `x` pixels from the sample point under the filter `name`."""
    if name in _KEYS_A:
        return _keys(x, _KEYS_A[name])
    if name == "mitchell":
        return _mitchell(x, 1.0 / 3.0, 1.0 / 3.0)
    if name == "parzen":
        return _mitchell(x, 1.0, 0.0)
    if name == "notch":
        return np.clip(1.0 - np.abs(x), 0.0, 1.0)
    raise ValueError(f"unknown clone filter {name!r}")


def sample_source(source, xf, yf, filter_name, origin=(0, 0), bounds=None, black_outside=False):
    """Read `source` at the fractional image coordinates (`xf`, `yf`) under a clone filter.

    `source` holds the pixels from image position `origin`; `bounds` is the whole source's rectangle (x, y, width, height) when
    `source` is only a window of it. A coordinate beyond the source repeats its edge pixel, or reads transparent black with
    `black_outside`."""
    height, width = source.shape[:2]
    if bounds is None:
        bx, by, bw, bh = 0, 0, width, height
        ox, oy = 0, 0
    else:
        bx, by, bw, bh = bounds
        ox, oy = int(origin[0]), int(origin[1])

    def pixel(ax, ay):
        inside = (ax >= bx) & (ax < bx + bw) & (ay >= by) & (ay < by + bh)
        cx = np.clip(np.clip(ax, bx, bx + bw - 1) - ox, 0, width - 1)
        cy = np.clip(np.clip(ay, by, by + bh - 1) - oy, 0, height - 1)
        value = source[cy, cx]
        return value * inside[..., None] if black_outside else value

    if filter_name == "impulse":
        return pixel(np.rint(xf).astype(int), np.rint(yf).astype(int))
    x0 = np.floor(xf).astype(int)
    y0 = np.floor(yf).astype(int)
    fx, fy = (xf - x0).astype(np.float32), (yf - y0).astype(np.float32)
    taps = (-1, 0, 1, 2)
    wx = [kernel_weight(filter_name, fx - o).astype(np.float32) for o in taps]
    wy = [kernel_weight(filter_name, fy - o).astype(np.float32) for o in taps]
    total = sum(wx)
    out = np.zeros(xf.shape + (source.shape[2],), np.float32)
    for oy_, weight_y in zip(taps, wy):
        row = np.zeros_like(out)
        for ox_, weight_x in zip(taps, wx):
            row += pixel(x0 + ox_, y0 + oy_) * (weight_x / total)[..., None]
        out += row * (weight_y / sum(wy))[..., None]
    return out
