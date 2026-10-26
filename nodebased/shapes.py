"""Structured per-node document payloads: roto shapes, tracker tracks and paint layers.

Deliberately free of NumPy and Qt, for the same reason `tiers.py` is: this is the *data model*
for `document["node_data"]`, and `core.validate` has to be able to reject a malformed shape
without importing a rasteriser.

The one rule that shapes everything here: a time-varying number inside a shape or a track is not
a new animation system. It is the schema v6 curve envelope from `docs/ANIMATION.md`, resolving
with the same semantics, wrapped next to the static value it falls back to:

    12.5                                              # static
    {"value": 12.5, "curve": {"interpolation": "linear", "keys": [...]}}

`validate_curve` and `evaluate_curve` are `nodebased.animation`'s, imported rather than
reimplemented. The spike this module grew from carried its own copy because v6 was unmerged at
the time; deleting that copy also removes one behavioural difference worth stating. The spike
resolved a frame outside the key range to the *base* value; v6 resolves it by **endpoint hold**
-- the first or last key's value -- matching Nuke. Delegating adopts endpoint hold, so a shape
point and a node parameter now extrapolate identically. The base value is what applies when a
scalar carries no curve at all.
"""
from __future__ import annotations

import math

from .animation import (CURVE_INTERPOLATIONS, FRAME_LIMITS, CurveError, evaluate_curve,
                        validate_curve as _validate_animation_curve)

SHAPE_MODES = ("union", "subtract")
MINIMUM_SHAPE_POINTS = 3
MAXIMUM_SHAPE_POINTS = 2000
MAXIMUM_SHAPES = 200
MAXIMUM_TRACKS = 200

# Bounds for the scalars inside a payload. These are separate from `core.LIMITS` on purpose:
# `core.LIMITS` is keyed by *node parameter* name and is part of the agent-visible parameter
# surface, while these belong to the payload schema.
SHAPE_LIMITS = {
    "x": (-65536.0, 65536.0), "y": (-65536.0, 65536.0),
    "in_x": (-65536.0, 65536.0), "in_y": (-65536.0, 65536.0),
    "out_x": (-65536.0, 65536.0), "out_y": (-65536.0, 65536.0),
    "feather_x": (-500.0, 500.0), "feather_y": (-500.0, 500.0),
    "opacity": (0.0, 1.0), "feather": (-500.0, 500.0), "enabled": (0.0, 1.0),
    "stroke_width": (0.0, 4096.0),
    "error": (0.0, 1.0),
    "translate_x": (-65536.0, 65536.0), "translate_y": (-65536.0, 65536.0),
    "rotate": (-360000.0, 360000.0), "scale_x": (-1000.0, 1000.0),
    "scale_y": (-1000.0, 1000.0), "center_x": (-65536.0, 65536.0),
    "center_y": (-65536.0, 65536.0),
    "skew_x": (-89.0, 89.0), "skew_y": (-89.0, 89.0),
}
TRANSFORM_DEFAULTS = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                     "scale_x": 1.0, "scale_y": 1.0, "center_x": 0.0, "center_y": 0.0,
                     "skew_x": 0.0, "skew_y": 0.0}
SKEW_ORDERS = ("XY", "YX")
# Roto's per-shape and global motion blur (Nuke's Motion Blur tab), lifetime and open-spline end caps.
OFFSET_TYPES = ("centred", "start", "end", "custom")
LIFETIME_TYPES = ("all", "single", "start_to", "to_end", "range")
END_CAPS = ("rounded", "square", "butt")
MOTION_BLUR_FIELDS = ("motion_blur_samples", "motion_blur_shutter", "motion_blur_offset_type", "motion_blur_offset")
MOTION_BLUR_DEFAULTS = {"samples": 1, "shutter": 0.5, "offset_type": "start", "offset": 0.0}
POINT_FIELDS = ("x", "y", "in_x", "in_y", "out_x", "out_y")
FEATHER_POINT_FIELDS = ("feather_x", "feather_y")

# Payload shape per node type. A node type absent from this table may not appear in `node_data`.
NODE_DATA_SCHEMA = {"Roto": "shapes", "Tracker": "tracks", "Stabilize": "tracks", "PlanarTracker": "tracks", "RotoPaint": "items",
                    "SplineWarp": "pairs", "GridWarp": "grid"}

# Pixel-unit scalars inside a payload — these must be scaled by a proxy tier change alongside
# `tiers.PIXEL_UNIT_PARAMS`, or a shape keys a different part of the frame at tier 2.
# See docs/EVALUATION_TIERS.md clause C3.
PIXEL_UNIT_SCALARS = frozenset({"x", "y", "in_x", "in_y", "out_x", "out_y", "feather",
                                "feather_x", "feather_y", "stroke_width",
                                "translate_x", "translate_y", "center_x", "center_y"})


def _is_number(value):
    return type(value) in (float, int) and not isinstance(value, bool) and math.isfinite(value)


def validate_curve(curve, where):
    """Apply `nodebased.animation`'s curve rules, re-raising with this payload's location.

    One engine, one set of rules: a curve on a shape point is the same object as a curve on a
    node parameter, so a key list that a knob would reject must not survive inside a roto shape.
    Only the message gains `where`, because "curve must define exactly ..." on its own gives an
    artist no way to find which point of which shape is wrong.
    """
    try:
        _validate_animation_curve(curve)
    except CurveError as error:
        raise ValueError(f"{where}: {error}") from None


def validate_scalar(value, name, where):
    """An animatable scalar: a plain number, or {'value': number, 'curve': curve | null}."""
    lo, hi = SHAPE_LIMITS[name]
    if _is_number(value):
        base = float(value)
    elif isinstance(value, dict):
        if set(value) not in ({"value"}, {"value", "curve"}):
            raise ValueError(f"{where}.{name}: an animatable scalar defines 'value' and optionally 'curve'")
        if not _is_number(value["value"]):
            raise ValueError(f"{where}.{name}: 'value' must be a finite number")
        base = float(value["value"])
        if value.get("curve") is not None:
            validate_curve(value["curve"], f"{where}.{name}")
    else:
        raise ValueError(f"{where}.{name} must be a number or an animatable scalar object")
    if not lo <= base <= hi:
        raise ValueError(f"{where}.{name} must be between {lo} and {hi}")


def resolve_scalar(value, frame, name=None):
    """Resolve an animatable scalar at `frame`, clamped to its declared bounds.

    A present curve always wins, at every frame, including outside its key range — that is v6's
    rule and `evaluate_curve` is where it lives. The base value applies only when there is no
    curve. Clamping mirrors v6's "resolved value clamps to LIMITS", against `SHAPE_LIMITS` here
    because these scalars are not node parameters.
    """
    if _is_number(value):
        resolved = float(value)
    else:
        curve = value.get("curve")
        at = int(frame) if float(frame).is_integer() else float(frame)      # a motion-blur sample may fall between frames
        resolved = float(evaluate_curve(curve, at) if curve else value["value"])
    if name is not None:
        lo, hi = SHAPE_LIMITS[name]
        resolved = min(max(resolved, lo), hi)
    return resolved


def validate_shape(shape, where):
    required = {"name", "mode", "opacity", "feather", "points"}
    optional = {"visible", "locked", "invert", "feather_falloff", "feather_model", "color", "blend_mode", "motion_blur", "track_link", "transform", "track_follow", "closed", "stroke_width",
                "feather_on", "falloff_power", "lifetime", "extra_matrix", "cap_start", "cap_end", *MOTION_BLUR_FIELDS}
    if not isinstance(shape, dict) or not required.issubset(shape) or set(shape) - required - optional:
        raise ValueError(f"{where}: a shape defines exactly the required fields plus supported optional settings")
    if not isinstance(shape["name"], str) or not 1 <= len(shape["name"]) <= 128:
        raise ValueError(f"{where}.name must contain 1-128 characters")
    if shape["mode"] not in SHAPE_MODES:
        raise ValueError(f"{where}.mode must be one of {list(SHAPE_MODES)}")
    if "closed" in shape and type(shape["closed"]) is not bool:
        raise ValueError(f"{where}.closed must be a boolean")
    validate_scalar(shape.get("stroke_width", 1.0), "stroke_width", where)
    validate_scalar(shape["opacity"], "opacity", where)
    validate_scalar(shape["feather"], "feather", where)
    if shape.get("feather_model", "box") not in ("box", "edge"):
        raise ValueError(f"{where}.feather_model is unsupported")
    if any(k in shape and type(shape[k]) is not bool for k in ("visible", "locked", "invert", "motion_blur")):
        raise ValueError(f"{where}: visibility, lock, invert and motion blur flags must be booleans")
    if "feather_on" in shape and type(shape["feather_on"]) is not bool:
        raise ValueError(f"{where}.feather_on must be a boolean")
    if "falloff_power" in shape and not (_is_number(shape["falloff_power"]) and 0.0 <= shape["falloff_power"] <= 16.0):
        raise ValueError(f"{where}.falloff_power must be a number between 0 and 16")
    for field in ("cap_start", "cap_end"):
        if shape.get(field, "rounded") not in END_CAPS:
            raise ValueError(f"{where}.{field} must be one of {list(END_CAPS)}")
    if "motion_blur_samples" in shape and not (type(shape["motion_blur_samples"]) is int and 1 <= shape["motion_blur_samples"] <= 64):
        raise ValueError(f"{where}.motion_blur_samples must be a whole number between 1 and 64")
    if "motion_blur_shutter" in shape and not (_is_number(shape["motion_blur_shutter"]) and 0.0 <= shape["motion_blur_shutter"] <= 8.0):
        raise ValueError(f"{where}.motion_blur_shutter must be a number between 0 and 8")
    if shape.get("motion_blur_offset_type", "start") not in OFFSET_TYPES:
        raise ValueError(f"{where}.motion_blur_offset_type must be one of {list(OFFSET_TYPES)}")
    if "motion_blur_offset" in shape and not (_is_number(shape["motion_blur_offset"]) and -8.0 <= shape["motion_blur_offset"] <= 8.0):
        raise ValueError(f"{where}.motion_blur_offset must be a number between -8 and 8")
    life = shape.get("lifetime")
    if life is not None:
        if (not isinstance(life, dict) or life.get("type") not in LIFETIME_TYPES
                or set(life) - {"type", "start", "end"}
                or any(k in life and type(life[k]) is not int for k in ("start", "end"))):
            raise ValueError(f"{where}.lifetime defines a type ({', '.join(LIFETIME_TYPES)}) and whole-number start and end frames")
        if life["type"] == "range" and life.get("end", 0) < life.get("start", 0):
            raise ValueError(f"{where}.lifetime range ends before it starts")
    matrix = shape.get("extra_matrix")
    if matrix is not None and (not isinstance(matrix, list) or len(matrix) != 16 or not all(_is_number(v) for v in matrix)
                               or any(abs(v) > 65536 for v in matrix)):
        raise ValueError(f"{where}.extra_matrix must be 16 finite numbers (a 4 x 4 matrix, row by row)")
    if shape.get("feather_falloff", "linear") not in ("linear", "smooth", "gaussian"):
        raise ValueError(f"{where}.feather_falloff is unsupported")
    if shape.get("blend_mode", "over") not in ("over", "plus", "minus", "multiply"):
        raise ValueError(f"{where}.blend_mode is unsupported")
    link = shape.get("track_link")
    if link is not None and (not isinstance(link, dict) or set(link) != {"tracker_id", "track_name"}
                             or not isinstance(link["tracker_id"], str) or not link["tracker_id"]
                             or not isinstance(link["track_name"], str) or not link["track_name"]):
        raise ValueError(f"{where}.track_link must identify a Tracker and track name")
    if shape.get("track_follow", "translate") not in ("translate", "full", "planar"):
        raise ValueError(f"{where}.track_follow must be translate or full")
    transform = shape.get("transform", {})
    if not isinstance(transform, dict) or set(transform) - set(TRANSFORM_DEFAULTS) - {"skew_order"}:
        raise ValueError(f"{where}.transform contains unsupported fields")
    if transform.get("skew_order", "XY") not in SKEW_ORDERS:
        raise ValueError(f"{where}.transform.skew_order must be one of {list(SKEW_ORDERS)}")
    for field in TRANSFORM_DEFAULTS:
        validate_scalar(transform.get(field, TRANSFORM_DEFAULTS[field]), field, f"{where}.transform")
    color = shape.get("color", [1.0, 1.0, 1.0, 1.0])
    if not isinstance(color, list) or len(color) != 4:
        raise ValueError(f"{where}.color must be RGBA values between 0 and 1")
    for channel, component in enumerate(color):
        validate_scalar(component, "opacity", f"{where}.color[{channel}]")
        base = component if _is_number(component) else component["value"]
        if not 0 <= base <= 1:
            raise ValueError(f"{where}.color[{channel}] must be between 0 and 1")
    points = shape["points"]
    minimum = MINIMUM_SHAPE_POINTS if shape.get("closed", True) else 2
    if not isinstance(points, list) or not minimum <= len(points) <= MAXIMUM_SHAPE_POINTS:
        raise ValueError(f"{where}.points must hold between {minimum} and {MAXIMUM_SHAPE_POINTS} points")
    for index, point in enumerate(points):
        if (not isinstance(point, dict) or not set(POINT_FIELDS).issubset(point)
                or set(point) - set(POINT_FIELDS) - set(FEATHER_POINT_FIELDS)):
            raise ValueError(f"{where}.points[{index}] defines {list(POINT_FIELDS)} and optional feather offsets")
        for field in (*POINT_FIELDS, *FEATHER_POINT_FIELDS):
            if field not in point:
                continue
            validate_scalar(point[field], field, f"{where}.points[{index}]")


# A track's own T, R and S switches and its pattern and search areas (fidelity step G, nodebased/trackerknobs.py); absent means the node's.
TRACK_OPTIONS = ("use_t", "use_r", "use_s", "pattern_radius", "search_radius")


def validate_track(track, where):
    if (not isinstance(track, dict) or not {"name", "enabled", "x", "y"}.issubset(track)
            or set(track) - {"name", "enabled", "x", "y", "error", *TRACK_OPTIONS}):
        raise ValueError(f"{where}: a track defines exactly name, enabled, x and y, with optional error, "
                         f"{', '.join(TRACK_OPTIONS)}")
    for field in ("use_t", "use_r", "use_s"):
        if field in track and track[field] not in (0, 1):
            raise ValueError(f"{where}.{field} must be 0 or 1")
    for field in ("pattern_radius", "search_radius"):
        if field in track and (type(track[field]) is not int or not 1 <= track[field] <= 256):
            raise ValueError(f"{where}.{field} must be a whole number from 1 to 256")
    if not isinstance(track["name"], str) or not 1 <= len(track["name"]) <= 128:
        raise ValueError(f"{where}.name must contain 1-128 characters")
    for field in ("enabled", "x", "y"):
        validate_scalar(track[field], field, where)
    if "error" in track:
        validate_scalar(track["error"], "error", where)


def validate_payload(kind, payload, where):
    """Validate one node's `node_data` entry against the payload schema for its node type."""
    slot = NODE_DATA_SCHEMA.get(kind)
    if slot is None:
        raise ValueError(f"{where}: {kind} nodes do not carry node_data")
    if kind == "SplineWarp":
        if not isinstance(payload, dict) or set(payload) != {"pairs"} or not isinstance(payload["pairs"], list):
            raise ValueError(f"{where}: SplineWarp payload defines a pairs list")
        if len(payload["pairs"]) > MAXIMUM_SHAPES:
            raise ValueError(f"{where}.pairs exceeds {MAXIMUM_SHAPES}")
        for i, pair in enumerate(payload["pairs"]):
            at = f"{where}.pairs[{i}]"
            if not isinstance(pair, dict) or set(pair) != {"name", "source", "destination"} or not isinstance(pair["name"], str):
                raise ValueError(f"{at} defines name, source and destination")
            source, destination = pair["source"], pair["destination"]
            if not isinstance(source, list) or not isinstance(destination, list) or not 2 <= len(source) <= MAXIMUM_SHAPE_POINTS or len(source) != len(destination):
                raise ValueError(f"{at} curves must have matching 2-{MAXIMUM_SHAPE_POINTS} points")
            for side, points in (("source", source), ("destination", destination)):
                for j, point in enumerate(points):
                    p_at = f"{at}.{side}[{j}]"
                    if not isinstance(point, dict) or set(point) != set(POINT_FIELDS):
                        raise ValueError(f"{p_at} defines exactly {list(POINT_FIELDS)}")
                    for field in POINT_FIELDS:
                        validate_scalar(point[field], field, p_at)
        return
    if kind == "GridWarp":
        if not isinstance(payload, dict) or set(payload) != {"source", "destination"}:
            raise ValueError(f"{where}: GridWarp payload defines source and destination grids")
        source, destination = payload["source"], payload["destination"]
        if not isinstance(source, list) or not isinstance(destination, list) or not 2 <= len(source) <= 15 or len(source) != len(destination):
            raise ValueError(f"{where}: grids need matching 2-15 rows")
        columns = len(source[0]) if source and isinstance(source[0], list) else 0
        if not 2 <= columns <= 15:
            raise ValueError(f"{where}: grids need 2-15 columns")
        for side, grid in (("source", source), ("destination", destination)):
            for i, row in enumerate(grid):
                if not isinstance(row, list) or len(row) != columns:
                    raise ValueError(f"{where}.{side}[{i}] must have {columns} columns")
                for j, point in enumerate(row):
                    p_at = f"{where}.{side}[{i}][{j}]"
                    if not isinstance(point, dict) or set(point) not in ({"x", "y"}, set(POINT_FIELDS)):
                        raise ValueError(f"{p_at} defines x and y, with optional Bezier tangents")
                    for field in (POINT_FIELDS if set(point) == set(POINT_FIELDS) else ("x", "y")):
                        validate_scalar(point[field], field, p_at)
        if len(destination[0]) != columns:
            raise ValueError(f"{where}: source and destination grid dimensions differ")
        return
    if not isinstance(payload, dict) or set(payload) != {slot}:
        raise ValueError(f"{where}: a {kind} payload defines exactly {slot!r}")
    items = payload[slot]
    if slot == "items":
        if not isinstance(items, list) or len(items) > MAXIMUM_SHAPES + 10000:
            raise ValueError(f"{where}.items must be a list of at most 10200 entries")
        for index, item in enumerate(items):
            at = f"{where}.items[{index}]"
            if not isinstance(item, dict) or item.get("kind") not in ("shape", "stroke"):
                raise ValueError(f"{at}: expected a shape or stroke item")
            if item["kind"] == "shape":
                required_shape = {"kind", "name", "mode", "opacity", "feather", "points", "blend", "visible"}
                if set(item) != required_shape:
                    raise ValueError(f"{at}: malformed layered shape")
                shape = {k: item[k] for k in ("name", "mode", "opacity", "feather", "points")}
                validate_shape(shape, at)
                if item["blend"] not in ("over", "add", "multiply", "screen") or type(item["visible"]) is not bool:
                    raise ValueError(f"{at}: invalid shape layer settings")
            else:
                required = {"kind", "name", "points", "brush", "tool", "lifetime", "color", "source_offset", "source_frame", "opacity", "blend", "visible", "follow_track", "patch_blend"}
                if set(item) != required or not isinstance(item["name"], str) or not 1 <= len(item["name"]) <= 128:
                    raise ValueError(f"{at}: malformed paint stroke")
                if item["tool"] not in ("paint", "eraser", "clone", "reveal", "blur", "sharpen", "smear", "dodge", "burn"):
                    raise ValueError(f"{at}.tool is unsupported")
                if not _is_number(item["patch_blend"]) or not 0 <= item["patch_blend"] <= 1:
                    raise ValueError(f"{at}.patch_blend must be between 0 and 1")
                if (item["blend"] not in ("over", "add", "multiply", "screen")
                        or type(item["visible"]) is not bool or not _is_number(item["opacity"])
                        or not 0 <= item["opacity"] <= 1):
                    raise ValueError(f"{at}: invalid layer settings")
                if not isinstance(item["points"], list) or not item["points"]:
                    raise ValueError(f"{at}.points must contain at least one point")
                for p in item["points"]:
                    if not isinstance(p, dict) or set(p) != {"x", "y", "pressure"} or any(not _is_number(p[k]) for k in p):
                        raise ValueError(f"{at}.points must contain finite x, y and pressure")
                    if any(abs(p[k]) > 65536 for k in ("x", "y")):
                        raise ValueError(f"{at}.points coordinates must be between -65536 and 65536")
                    if not 0 <= p["pressure"] <= 1: raise ValueError(f"{at}.pressure must be between 0 and 1")
                brush = item["brush"]
                if not isinstance(brush, dict) or set(brush) != {"size", "hardness", "opacity", "spacing", "strength"}:
                    raise ValueError(f"{at}.brush must define size, hardness, opacity, spacing and strength")
                if (any(not _is_number(brush[k]) for k in brush)
                        or not (0.1 <= brush["size"] <= 4096 and 0 <= brush["hardness"] <= 1
                                and 0 <= brush["opacity"] <= 1 and 0.01 <= brush["spacing"] <= 4
                                and 0 <= brush["strength"] <= 1)):
                    raise ValueError(f"{at}.brush values are out of range")
                life = item["lifetime"]
                if not isinstance(life, dict) or life.get("mode") not in ("single", "range", "all", "from_current"):
                    raise ValueError(f"{at}.lifetime is invalid")
                expected_life = ({"mode", "first", "last"} if life["mode"] == "range" else
                                 {"mode", "first"} if life["mode"] in ("single", "from_current") else {"mode"})
                if set(life) != expected_life:
                    raise ValueError(f"{at}.lifetime has unexpected frame fields")
                if life["mode"] in ("single", "range", "from_current") and type(life.get("first")) is not int:
                    raise ValueError(f"{at}.lifetime.first must be an integer frame")
                if life["mode"] == "range" and (type(life.get("last")) is not int or life["last"] < life["first"]):
                    raise ValueError(f"{at}.lifetime range is invalid")
                if not isinstance(item["color"], list) or len(item["color"]) != 4 or any(not _is_number(x) for x in item["color"]):
                    raise ValueError(f"{at}.color must be RGBA")
                if not isinstance(item["source_offset"], list) or len(item["source_offset"]) != 2 or any(not _is_number(x) for x in item["source_offset"]):
                    raise ValueError(f"{at}.source_offset must be XY")
                if item["source_frame"] != "relative" and type(item["source_frame"]) is not int:
                    raise ValueError(f"{at}.source_frame must be a frame or 'relative'")
                follow = item["follow_track"]
                if follow is not None and (not isinstance(follow, dict) or set(follow) != {"node_id", "track_index"}
                                           or not isinstance(follow["node_id"], str)
                                           or type(follow["track_index"]) is not int or follow["track_index"] < 0):
                    raise ValueError(f"{at}.follow_track must be null or identify a Tracker and track index")
        return
    limit = MAXIMUM_SHAPES if slot == "shapes" else MAXIMUM_TRACKS
    if not isinstance(items, list) or len(items) > limit:
        raise ValueError(f"{where}.{slot} must be a list of at most {limit} entries")
    validator = validate_shape if slot == "shapes" else validate_track
    for index, item in enumerate(items):
        validator(item, f"{where}.{slot}[{index}]")


def validate_node_data(node_data, nodes):
    """Validate `document["node_data"]` against the document's nodes.

    Orphan entries are an error rather than being dropped: silently discarding an artist's roto
    because a node id went missing is the kind of data loss that is only noticed at review.
    """
    if not isinstance(node_data, dict):
        raise ValueError("node_data must be an object keyed by node id")
    for key, payload in node_data.items():
        if key not in nodes:
            raise ValueError(f"node_data[{key!r}] does not name a node in this document")
        validate_payload(nodes[key]["type"], payload, f"node_data[{key!r}]")
        if nodes[key]["type"] == "RotoPaint":
            for index, item in enumerate(payload["items"]):
                follow = item.get("follow_track") if item.get("kind") == "stroke" else None
                if follow is not None:
                    tracker_id = follow["node_id"]
                    if tracker_id not in nodes or nodes[tracker_id]["type"] != "Tracker":
                        raise ValueError(f"node_data[{key!r}].items[{index}]: follow_track must name a Tracker")
                    tracks = node_data.get(tracker_id, {}).get("tracks", [])
                    if follow["track_index"] >= len(tracks):
                        raise ValueError(f"node_data[{key!r}].items[{index}]: track index is out of range")
        if nodes[key]["type"] == "Roto":
            for index, shape in enumerate(payload["shapes"]):
                link = shape.get("track_link")
                if link is None: continue
                tracker_id = link["tracker_id"]
                if tracker_id not in nodes or nodes[tracker_id]["type"] not in ("Tracker", "Stabilize", "PlanarTracker"):
                    raise ValueError(f"node_data[{key!r}].shapes[{index}]: track_link must name a Tracker")
                tracks = node_data.get(tracker_id, {}).get("tracks", [])
                if not any(track["name"] == link["track_name"] for track in tracks):
                    raise ValueError(f"node_data[{key!r}].shapes[{index}]: track name is missing")
        if nodes[key]["type"] == "GridWarp":
            rows, cols = nodes[key]["params"].get("rows", 5), nodes[key]["params"].get("columns", 5)
            if len(payload["source"]) != rows or len(payload["source"][0]) != cols:
                raise ValueError(f"node_data[{key!r}]: grid dimensions must match rows and columns knobs")


def payload_slot(kind):
    """The `node_data` key a node type owns, or None if it carries no payload."""
    return NODE_DATA_SCHEMA.get(kind)


def empty_payload(kind):
    slot = NODE_DATA_SCHEMA.get(kind)
    if kind == "SplineWarp":
        return {"pairs": []}
    if kind == "GridWarp":
        return None
    return {slot: []} if slot else None


def motion_blur_plan(shape, params=None):
    """The motion-blur settings that apply to one shape, or None when it is sharp.

    Nuke's Motion Blur tab has per-shape settings and global ones. The node's `shape_motionblur_on` (default on) lets the
    shapes' own settings count; `global_motionblur_on` applies the global samples, shutter and offset to every shape that
    has no motion blur of its own. A shape saved before these settings existed (only `motion_blur: true`) keeps its
    original three-sample blur across the neighbouring frames, which is the plan `{"legacy": True}`."""
    params = params or {}
    shapes_on = bool(params.get("shape_motionblur_on", 1))
    own = bool(shape.get("motion_blur", False)) and shapes_on
    if own:
        if not any(key in shape for key in MOTION_BLUR_FIELDS):
            return {"legacy": True}
        return {"samples": int(shape.get("motion_blur_samples", MOTION_BLUR_DEFAULTS["samples"])),
                "shutter": float(shape.get("motion_blur_shutter", MOTION_BLUR_DEFAULTS["shutter"])),
                "offset_type": shape.get("motion_blur_offset_type", MOTION_BLUR_DEFAULTS["offset_type"]),
                "offset": float(shape.get("motion_blur_offset", MOTION_BLUR_DEFAULTS["offset"]))}
    if params.get("global_motionblur_on", 0) and not shape.get("motion_blur", False):
        return {"samples": int(params.get("global_motionblur", 1)),
                "shutter": float(params.get("global_motionblur_shutter", 0.5)),
                "offset_type": str(params.get("global_motionblur_shutter_offset_type", "start")),
                "offset": float(params.get("global_motionblur_shutter_offset", 0.0))}
    return None


def motion_blur_times(plan, frame):
    """The frames a shutter plan samples: `samples + 1` evenly spaced times across the open shutter, ends included."""
    shutter = max(0.0, plan["shutter"])
    kind = plan["offset_type"]
    if kind == "centred":
        start = frame - shutter / 2.0
    elif kind == "end":
        start = frame - shutter
    elif kind == "custom":
        start = frame + plan["offset"]
    else:
        start = float(frame)
    n = max(1, int(plan["samples"]))
    return [start + shutter * i / n for i in range(n + 1)]


def in_lifetime(life, frame):
    """True when a shape with this lifetime exists at `frame` (Nuke's lifetime type, from and to)."""
    if not life or life.get("type", "all") == "all":
        return True
    start, end = life.get("start", 1), life.get("end", 1)
    kind = life["type"]
    if kind == "single":
        return int(frame) == start
    if kind == "start_to":
        return int(frame) <= end
    if kind == "to_end":
        return int(frame) >= start
    return start <= int(frame) <= end


def resolve_shapes(payload, frame, params=None):
    """Resolve every animatable scalar in a Roto payload at `frame`.

    Returned shapes are plain numbers throughout, which is what both the rasteriser and the cache
    digest consume — the digest hashes the *resolved* payload so a static shape keeps its cache
    entry across a scrub while an animated one re-keys. See docs/ANIMATION.md.

    `params` is the Roto node's own parameters; only its global motion-blur settings are read from it.
    """
    return [_resolve_shape(shape, frame, params, blur=True) for shape in (payload or {}).get("shapes", [])]


def _resolve_shape(shape, frame, params, blur):
    def points_at(at_frame):
        return [{**{field: resolve_scalar(point[field], at_frame, field) for field in POINT_FIELDS},
                 **{field: resolve_scalar(point.get(field, 0.0), at_frame, field)
                    for field in FEATHER_POINT_FIELDS}}
                for point in shape["points"]]
    resolved_points = points_at(frame)
    center_x = (min(p["x"] for p in resolved_points) + max(p["x"] for p in resolved_points)) / 2.0
    center_y = (min(p["y"] for p in resolved_points) + max(p["y"] for p in resolved_points)) / 2.0
    resolved = {
        "name": shape["name"],
        "mode": shape["mode"],
        "opacity": resolve_scalar(shape["opacity"], frame, "opacity"),
        "feather": resolve_scalar(shape["feather"], frame, "feather"),
        "closed": shape.get("closed", True),
        "stroke_width": resolve_scalar(shape.get("stroke_width", 1.0), frame, "stroke_width"),
        "feather_model": shape.get("feather_model", "box"),
        "points": resolved_points,
        "transform": {field: resolve_scalar(shape.get("transform", {}).get(field,
                          center_x if field == "center_x" else center_y if field == "center_y" else default), frame, field)
                      for field, default in TRANSFORM_DEFAULTS.items()},
        "track_follow": shape.get("track_follow", "translate"),
    }
    if shape.get("transform", {}).get("skew_order", "XY") != "XY":
        resolved["transform"]["skew_order"] = shape["transform"]["skew_order"]
    for field, default in (("visible", True), ("locked", False), ("invert", False),
                           ("feather_falloff", "linear"), ("color", [1, 1, 1, 1]),
                           ("blend_mode", "over"), ("motion_blur", False)):
        resolved[field] = copy_value = shape.get(field, default)
        if field == "color":
            resolved[field] = [resolve_scalar(channel, frame, "opacity") for channel in copy_value]
    # Settings added with the Nuke parity pass: each appears in the resolved shape only when it differs from the default,
    # so the resolved payload of an older document (and its cache digest) is unchanged.
    if shape.get("feather_on", True) is False:
        resolved["feather"] = 0.0
        for point in resolved_points:
            point["feather_x"] = point["feather_y"] = 0.0
    if shape.get("falloff_power", 1.0) != 1.0:
        resolved["falloff_power"] = float(shape["falloff_power"])
    for field in ("cap_start", "cap_end"):
        if shape.get(field, "rounded") != "rounded":
            resolved[field] = shape[field]
    if shape.get("extra_matrix") is not None:
        resolved["extra_matrix"] = [float(v) for v in shape["extra_matrix"]]
    if not in_lifetime(shape.get("lifetime"), frame):
        resolved["visible"] = False
    plan = motion_blur_plan(shape, params) if blur else None
    if plan is not None and plan.get("legacy"):
        # A compact one-frame shutter centered on the current frame. Static shapes naturally
        # collapse to one sample; keyed shapes acquire temporal coverage in the same raster path.
        resolved["motion_blur"] = True
        resolved["_motion_samples"] = [points_at(int(frame) - 1), points_at(int(frame) + 1)]
    elif plan is not None and plan["shutter"] > 0:
        # Each sample is the whole shape (animated points and transform) at its own time, already transformed.
        sets = []
        for at in motion_blur_times(plan, frame):
            sets.append(_resolve_shape(shape, at, params, blur=False)["points"])      # transformed by the call itself
        resolved["motion_blur"] = True
        resolved["_motion_samples"] = sets
        resolved["_motion_only"] = True
        resolved["_motion_final"] = True
    apply_shape_transform(resolved)
    return resolved


def _skew_matrix(t):
    """The 2 x 2 skew of a shape transform as (a, b, c, d): skew X then Y for order XY, Y then X for order YX (degrees)."""
    kx = (1.0, math.tan(math.radians(t.get("skew_x", 0.0))), 0.0, 1.0)
    ky = (1.0, 0.0, math.tan(math.radians(t.get("skew_y", 0.0))), 1.0)
    first, second = (kx, ky) if t.get("skew_order", "XY") == "XY" else (ky, kx)
    return (second[0]*first[0] + second[1]*first[2], second[0]*first[1] + second[1]*first[3],
            second[2]*first[0] + second[3]*first[2], second[2]*first[1] + second[3]*first[3])


def apply_shape_transform(shape):
    """Apply the resolved Nuke-style shape transform to points and tangent vectors in place.

    About the centre: skew, then scale, then rotate, then translate. A shape's `extra_matrix` (Nuke's 4 x 4 `transform_matrix`
    knob, row by row) is applied last, to image coordinates."""
    t = shape.get("transform", TRANSFORM_DEFAULTS)
    cx, cy = t["center_x"], t["center_y"]
    sx, sy = t["scale_x"], t["scale_y"]
    angle = math.radians(t["rotate"])
    c, s = math.cos(angle), math.sin(angle)
    skewed = bool(t.get("skew_x", 0.0) or t.get("skew_y", 0.0))
    k = _skew_matrix(t) if skewed else None
    matrix = shape.get("extra_matrix")
    area = abs(sx * sy) * (abs(k[0]*k[3] - k[1]*k[2]) if skewed else 1.0)
    if matrix is not None:
        area *= abs(matrix[0]*matrix[5] - matrix[1]*matrix[4])
    if shape.get("feather_model") == "edge":
        shape["feather"] *= math.sqrt(area)
    shape["stroke_width"] *= math.sqrt(area)
    def vector(x, y):
        if skewed:
            x, y = k[0]*x + k[1]*y, k[2]*x + k[3]*y
        x, y = x * sx, y * sy
        return c * x - s * y, s * x + c * y
    point_sets = [shape["points"]]
    if shape.get("feather_model") == "edge" and not shape.get("_motion_final"):
        point_sets.extend(shape.get("_motion_samples", []))
    for point_set in point_sets:
        for point in point_set:
            x, y = vector(point["x"] - cx, point["y"] - cy)
            point["x"] = cx + x + t["translate_x"]
            point["y"] = cy + y + t["translate_y"]
            for xfield, yfield in (("in_x", "in_y"), ("out_x", "out_y"), ("feather_x", "feather_y")):
                point[xfield], point[yfield] = vector(point.get(xfield, 0.0), point.get(yfield, 0.0))
            if matrix is not None:
                point["x"], point["y"] = (matrix[0]*point["x"] + matrix[1]*point["y"] + matrix[3],
                                          matrix[4]*point["x"] + matrix[5]*point["y"] + matrix[7])
                for xfield, yfield in (("in_x", "in_y"), ("out_x", "out_y"), ("feather_x", "feather_y")):
                    vx, vy = point.get(xfield, 0.0), point.get(yfield, 0.0)
                    point[xfield], point[yfield] = matrix[0]*vx + matrix[1]*vy, matrix[4]*vx + matrix[5]*vy


def feather_handle_points(points, feather):
    """Return viewer handle centres: outline normal offset plus each point's X/Y delta."""
    coords = [(float(p["x"]), float(p["y"])) for p in points]
    area = sum(coords[i][0]*coords[(i+1)%len(coords)][1] -
               coords[(i+1)%len(coords)][0]*coords[i][1] for i in range(len(coords)))
    winding = 1.0 if area >= 0 else -1.0
    result = []
    for i, point in enumerate(points):
        prev = coords[(i-1) % len(coords)]; nxt = coords[(i+1) % len(coords)]
        tx, ty = nxt[0]-prev[0], nxt[1]-prev[1]
        length = math.hypot(tx, ty) or 1.0
        nx, ny = winding*ty/length, -winding*tx/length
        result.append({"x": coords[i][0]+nx*feather+float(point.get("feather_x", 0.0)),
                       "y": coords[i][1]+ny*feather+float(point.get("feather_y", 0.0))})
    return result


def transform_point_group(points, *, center, scale_x=1.0, scale_y=1.0,
                          rotate=0.0, translate_x=0.0, translate_y=0.0):
    """Return transformed copies of selected points, including their tangent vectors."""
    angle = math.radians(float(rotate))
    c, s = math.cos(angle), math.sin(angle)
    cx, cy = map(float, center)
    result = []
    for original in points:
        point = dict(original)
        x, y = (float(point["x"]) - cx) * scale_x, (float(point["y"]) - cy) * scale_y
        point["x"] = cx + c*x - s*y + translate_x
        point["y"] = cy + s*x + c*y + translate_y
        for xf, yf in (("in_x", "in_y"), ("out_x", "out_y")):
            x, y = float(point[xf]) * scale_x, float(point[yf]) * scale_y
            point[xf], point[yf] = c*x - s*y, s*x + c*y
        x, y = float(point.get("feather_x", 0.0))*scale_x, float(point.get("feather_y", 0.0))*scale_y
        point["feather_x"], point["feather_y"] = c*x-s*y, s*x+c*y
        result.append(point)
    return result


def unresolve_roto_point(point, shape, document, frame, tier=1):
    """Invert the shape and Tracker transforms so a viewer edit stays in local point space."""
    result = dict(point)
    t = resolve_shapes({"shapes": [shape]}, frame)[0]["transform"]
    link = shape.get("track_link")
    nodes = document.get("nodes", {})
    track_node = nodes.get(link["tracker_id"]) if link else None
    if track_node is not None:
        data = document.get("node_data", {}).get(link["tracker_id"], {})
        reference = int(track_node["params"].get("reference_frame", 1))
        if shape.get("track_follow", "translate") == "planar" and track_node["type"] == "PlanarTracker":
            from . import planar as planar_model
            ref_corners = planar_model.resolve_corners(data, reference)
            current_corners = planar_model.resolve_corners(data, frame)
            inverse = planar_model.invert_homography(
                planar_model.homography_from_quads(ref_corners, current_corners))
            anchor = [[result["x"]*tier, result["y"]*tier]]
            mapped_anchor = planar_model.project_points(inverse, anchor)[0] / tier
            for xf, yf in (("in_x", "in_y"), ("out_x", "out_y")):
                endpoint = [[result["x"]*tier + result.get(xf, 0.0)*tier,
                             result["y"]*tier + result.get(yf, 0.0)*tier]]
                mapped_endpoint = planar_model.project_points(inverse, endpoint)[0] / tier
                result[xf], result[yf] = float(mapped_endpoint[0]-mapped_anchor[0]), float(mapped_endpoint[1]-mapped_anchor[1])
            result["x"], result["y"] = map(float, mapped_anchor)
        elif shape.get("track_follow", "translate") == "full":
            from .tracker import solve
            solved = solve(data, frame, track_node["params"])
            angle = math.radians(solved["rotate"]); c, s = math.cos(angle), math.sin(angle)
            scale = solved["scale"]
            cx, cy = t["center_x"], t["center_y"]
            for xf, yf in (("x", "y"), ("in_x", "in_y"), ("out_x", "out_y")):
                if xf == "x":
                    x, y = result["x"] - cx - solved["translate_x"]/tier, result["y"] - cy - solved["translate_y"]/tier
                else:
                    x, y = result[xf], result[yf]
                if abs(scale) > 1e-12:
                    result[xf] = (c*x + s*y) / scale
                    result[yf] = (-s*x + c*y) / scale
            result["x"] += cx; result["y"] += cy
        else:
            track = next((item for item in data.get("tracks", [])
                          if item["name"] == link["track_name"]), None)
            if track is not None:
                dx = (resolve_scalar(track["x"], frame, "x") - resolve_scalar(track["x"], reference, "x")) / tier
                dy = (resolve_scalar(track["y"], frame, "y") - resolve_scalar(track["y"], reference, "y")) / tier
                result["x"] -= dx; result["y"] -= dy
    angle = math.radians(t["rotate"]); c, s = math.cos(angle), math.sin(angle)
    sx, sy = t["scale_x"], t["scale_y"]
    cx, cy = t["center_x"], t["center_y"]
    tx, ty = t["translate_x"], t["translate_y"]
    x, y = result["x"] - cx - tx, result["y"] - cy - ty
    if abs(sx) > 1e-12 and abs(sy) > 1e-12:
        result["x"] = (c*x + s*y) / sx + cx
        result["y"] = (-s*x + c*y) / sy + cy
        for xf, yf in (("in_x", "in_y"), ("out_x", "out_y")):
            x, y = result[xf], result[yf]
            result[xf] = (c*x + s*y) / sx
            result[yf] = (-s*x + c*y) / sy
    return result


def unresolve_feather_vector(vector, shape, document, frame):
    """Invert the shape and full-Tracker linear transforms for a feather handle vector."""
    x, y = map(float, vector)
    link = shape.get("track_link")
    if link and shape.get("track_follow", "translate") == "planar":
        tracker_node = document.get("nodes", {}).get(link["tracker_id"])
        if tracker_node is not None and tracker_node["type"] == "PlanarTracker":
            from . import planar as planar_model
            data = document.get("node_data", {}).get(link["tracker_id"], {})
            ref = int(tracker_node["params"].get("reference_frame", 1))
            forward = planar_model.homography_from_quads(
                planar_model.resolve_corners(data, ref), planar_model.resolve_corners(data, frame))
            inverse = planar_model.invert_homography(forward)
            cx, cy = shape["transform"].get("center_x", 0.0), shape["transform"].get("center_y", 0.0)
            mapped = planar_model.project_points(inverse, [[cx, cy], [cx+x, cy+y]])
            x, y = map(float, mapped[1]-mapped[0])
    elif link and shape.get("track_follow", "translate") == "full":
        tracker_node = document.get("nodes", {}).get(link["tracker_id"])
        if tracker_node is not None:
            from .tracker import solve
            tracked = solve(document.get("node_data", {}).get(link["tracker_id"], {}), frame,
                            tracker_node["params"])
            scale = float(tracked["scale"])
            if abs(scale) > 1e-12:
                angle = math.radians(tracked["rotate"]); c, s = math.cos(angle), math.sin(angle)
                x, y = (c*x+s*y)/scale, (-s*x+c*y)/scale
    t = resolve_shapes({"shapes": [shape]}, frame)[0]["transform"]
    angle = math.radians(t["rotate"]); c, s = math.cos(angle), math.sin(angle)
    if abs(t["scale_x"]) > 1e-12 and abs(t["scale_y"]) > 1e-12:
        x, y = (c*x+s*y)/t["scale_x"], (-s*x+c*y)/t["scale_y"]
    return x, y


def resolve_roto_shapes(payload, frame, document=None, tier=1, params=None):
    """Resolve transforms and tracker links for both the evaluator and viewer overlay. `params` are the Roto node's own
    (its global motion blur)."""
    resolved = resolve_shapes(payload, frame, params)
    if document is None:
        return resolved
    from . import tracker as tracker_model
    raw_shapes = (payload or {}).get("shapes", [])
    nodes = document.get("nodes", {})
    for index, shape in enumerate(resolved):
        link = raw_shapes[index].get("track_link") if index < len(raw_shapes) else None
        tracker_node = nodes.get(link["tracker_id"]) if link else None
        if tracker_node is None:
            continue
        tracker_payload = document.get("node_data", {}).get(link["tracker_id"], {})
        reference = int(tracker_node["params"].get("reference_frame", 1))
        if shape.get("track_follow", "translate") == "planar" and tracker_node["type"] == "PlanarTracker":
            from . import planar as planar_model
            try:
                reference_corners = planar_model.resolve_corners(tracker_payload, reference)
                current_corners = planar_model.resolve_corners(tracker_payload, frame)
                matrix = planar_model.homography_from_quads(reference_corners, current_corners)
                planar_model.apply_homography_to_shape(shape, matrix, tier=tier)
            except ValueError:
                continue
        elif shape.get("track_follow", "translate") == "full":
            solved = tracker_model.solve(tracker_payload, frame, tracker_node["params"])
            # Solve maps the plate about its own centroid; transfer that similarity about the
            # Roto shape's centre, then apply the track translation.
            sx = sy = solved["scale"]
            angle = math.radians(solved["rotate"])
            c, s = math.cos(angle), math.sin(angle)
            cx, cy = shape["transform"]["center_x"], shape["transform"]["center_y"]
            dx, dy = solved["translate_x"] / tier, solved["translate_y"] / tier
            if shape.get("feather_model") == "edge":
                shape["feather"] *= abs(sx)
            point_sets = [shape["points"]]
            if shape.get("feather_model") == "edge" or shape.get("_motion_final"):
                point_sets.extend(shape.get("_motion_samples", []))
            for point_set in point_sets:
                for point in point_set:
                    x, y = (point["x"] - cx) * sx, (point["y"] - cy) * sy
                    point["x"], point["y"] = cx + c*x - s*y + dx, cy + s*x + c*y + dy
                    for xf, yf in (("in_x", "in_y"), ("out_x", "out_y"), ("feather_x", "feather_y")):
                        point[xf], point[yf] = (point.get(xf, 0.0)*sx*c-point.get(yf, 0.0)*sy*s,
                                                 point.get(xf, 0.0)*sx*s+point.get(yf, 0.0)*sy*c)
        else:
            track = next((item for item in tracker_payload.get("tracks", [])
                          if item["name"] == link["track_name"]), None)
            if track is None:
                continue
            dx = (resolve_scalar(track["x"], frame, "x") - resolve_scalar(track["x"], reference, "x")) / tier
            dy = (resolve_scalar(track["y"], frame, "y") - resolve_scalar(track["y"], reference, "y")) / tier
            for point_set in ([shape["points"], *shape.get("_motion_samples", [])] if shape.get("_motion_final")
                              else [shape["points"]]):
                for point in point_set:
                    point["x"] += dx
                    point["y"] += dy
    return resolved


def split_cubic_point(start, end, t):
    """Return De Casteljau's exact subdivision point and replacement adjacent handles."""
    t = min(1.0, max(0.0, float(t)))
    p0 = (float(start["x"]), float(start["y"]))
    p1 = (p0[0] + float(start["out_x"]), p0[1] + float(start["out_y"]))
    p3 = (float(end["x"]), float(end["y"]))
    p2 = (p3[0] + float(end["in_x"]), p3[1] + float(end["in_y"]))
    lerp = lambda a, b: (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)
    a, b, c = lerp(p0, p1), lerp(p1, p2), lerp(p2, p3)
    d, e = lerp(a, b), lerp(b, c)
    f = lerp(d, e)
    new_start, new_end = dict(start), dict(end)
    new_start["out_x"], new_start["out_y"] = a[0]-p0[0], a[1]-p0[1]
    new_end["in_x"], new_end["in_y"] = c[0]-p3[0], c[1]-p3[1]
    point = {"x": f[0], "y": f[1], "in_x": d[0]-f[0], "in_y": d[1]-f[1],
             "out_x": e[0]-f[0], "out_y": e[1]-f[1]}
    return new_start, point, new_end


def resolve_tracks(payload, frame):
    """Resolve a Tracker payload at `frame`. `enabled` resolves then thresholds at >= 0.5."""
    tracks = []
    for track in (payload or {}).get("tracks", []):
        tracks.append({
            "name": track["name"],
            "enabled": resolve_scalar(track["enabled"], frame, "enabled") >= 0.5,
            "x": resolve_scalar(track["x"], frame, "x"),
            "y": resolve_scalar(track["y"], frame, "y"),
            "error": resolve_scalar(track.get("error", 0.0), frame, "error"),
        })
    return tracks


def resolve_warp_data(kind, payload, frame):
    """Resolve animatable curve/grid point coordinates at a timeline frame."""
    if kind == "SplineWarp":
        def curve(points):
            return [[resolve_scalar(p[field], frame, field) for field in POINT_FIELDS] for p in points]
        return [{"name": pair["name"], "source": curve(pair["source"]),
                 "destination": curve(pair["destination"])}
                for pair in (payload or {}).get("pairs", [])]
    if kind == "GridWarp":
        return {side: [[[
                    resolve_scalar(p.get(axis, 0.0), frame, axis) for axis in POINT_FIELDS]
                    for p in row] for row in (payload or {}).get(side, [])]
                for side in ("source", "destination")}
    return None


def resolve_paint_items(payload, frame):
    """Return frame-resolved ordered RotoPaint items. Legacy Roto payloads are untouched."""
    items = []
    for item in (payload or {}).get("items", []):
        resolved = dict(item)
        if item["kind"] == "shape":
            resolved.update(resolve_shapes({"shapes": [{k: item[k] for k in
                              ("name", "mode", "opacity", "feather", "points")}]}, frame)[0])
        items.append(resolved)
    return items
