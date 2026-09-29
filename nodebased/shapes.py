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
    "opacity": (0.0, 1.0), "feather": (0.0, 500.0), "enabled": (0.0, 1.0),
    "error": (0.0, 1.0),
}
POINT_FIELDS = ("x", "y", "in_x", "in_y", "out_x", "out_y")

# Payload shape per node type. A node type absent from this table may not appear in `node_data`.
NODE_DATA_SCHEMA = {"Roto": "shapes", "Tracker": "tracks", "Stabilize": "tracks", "RotoPaint": "items",
                    "SplineWarp": "pairs", "GridWarp": "grid"}

# Pixel-unit scalars inside a payload — these must be scaled by a proxy tier change alongside
# `tiers.PIXEL_UNIT_PARAMS`, or a shape keys a different part of the frame at tier 2.
# See docs/EVALUATION_TIERS.md clause C3.
PIXEL_UNIT_SCALARS = frozenset({"x", "y", "in_x", "in_y", "out_x", "out_y", "feather"})


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
        resolved = float(evaluate_curve(curve, int(frame)) if curve else value["value"])
    if name is not None:
        lo, hi = SHAPE_LIMITS[name]
        resolved = min(max(resolved, lo), hi)
    return resolved


def validate_shape(shape, where):
    if not isinstance(shape, dict) or set(shape) != {"name", "mode", "opacity", "feather", "points"}:
        raise ValueError(f"{where}: a shape defines exactly name, mode, opacity, feather and points")
    if not isinstance(shape["name"], str) or not 1 <= len(shape["name"]) <= 128:
        raise ValueError(f"{where}.name must contain 1-128 characters")
    if shape["mode"] not in SHAPE_MODES:
        raise ValueError(f"{where}.mode must be one of {list(SHAPE_MODES)}")
    validate_scalar(shape["opacity"], "opacity", where)
    validate_scalar(shape["feather"], "feather", where)
    points = shape["points"]
    if not isinstance(points, list) or not MINIMUM_SHAPE_POINTS <= len(points) <= MAXIMUM_SHAPE_POINTS:
        raise ValueError(f"{where}.points must hold between {MINIMUM_SHAPE_POINTS} and "
                         f"{MAXIMUM_SHAPE_POINTS} points; open splines are not representable")
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != set(POINT_FIELDS):
            raise ValueError(f"{where}.points[{index}] defines exactly {list(POINT_FIELDS)}")
        for field in POINT_FIELDS:
            validate_scalar(point[field], field, f"{where}.points[{index}]")


def validate_track(track, where):
    if (not isinstance(track, dict) or not {"name", "enabled", "x", "y"}.issubset(track)
            or set(track) - {"name", "enabled", "x", "y", "error"}):
        raise ValueError(f"{where}: a track defines exactly name, enabled, x and y, with optional error")
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
                expected = {"kind", "name", "mode", "opacity", "feather", "points", "blend", "visible"}
                if set(item) != expected:
                    raise ValueError(f"{at}: malformed layered shape")
                shape = {k: item[k] for k in ("name", "mode", "opacity", "feather", "points")}
                validate_shape(shape, at)
                if item["blend"] not in ("over", "add", "multiply", "screen") or type(item["visible"]) is not bool:
                    raise ValueError(f"{at}: invalid shape layer settings")
            else:
                required = {"kind", "name", "points", "brush", "tool", "lifetime", "color", "source_offset", "source_frame", "opacity", "blend", "visible", "follow_track"}
                if set(item) != required or not isinstance(item["name"], str) or not 1 <= len(item["name"]) <= 128:
                    raise ValueError(f"{at}: malformed paint stroke")
                if item["tool"] not in ("paint", "eraser", "clone", "reveal", "blur", "sharpen", "smear", "dodge", "burn"):
                    raise ValueError(f"{at}.tool is unsupported")
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


def resolve_shapes(payload, frame):
    """Resolve every animatable scalar in a Roto payload at `frame`.

    Returned shapes are plain numbers throughout, which is what both the rasteriser and the cache
    digest consume — the digest hashes the *resolved* payload so a static shape keeps its cache
    entry across a scrub while an animated one re-keys. See docs/ANIMATION.md.
    """
    shapes = []
    for shape in (payload or {}).get("shapes", []):
        shapes.append({
            "name": shape["name"],
            "mode": shape["mode"],
            "opacity": resolve_scalar(shape["opacity"], frame, "opacity"),
            "feather": resolve_scalar(shape["feather"], frame, "feather"),
            "points": [{field: resolve_scalar(point[field], frame, field) for field in POINT_FIELDS}
                       for point in shape["points"]],
        })
    return shapes


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
