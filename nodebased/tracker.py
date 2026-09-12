"""Solve a similarity transform from tracker tracks.

A Tracker is a Transform whose numbers are solved from `document["node_data"]` instead of typed
into the inspector, so everything here produces the six fields `nodebased.imaging` already knows
how to apply and `nodebased.tiers._transform_rule` already knows how to invert. Nothing in this
module rasterises; it turns two frames' worth of track positions into geometry.

The solve is the closed-form least-squares 2D similarity (Umeyama without reflection): the single
rotation, uniform scale and translation that best carries the reference-frame track positions onto
this frame's. Least squares rather than "use the first two tracks" because a real track set is
noisy and an artist expects adding a fourth good track to improve the result, not to be ignored.

What is deliberately absent, and named rather than hidden:

* No perspective, no shear, no per-track weighting. Four-corner-pin is a different node.
* No analysis. Track positions are authored through the `set_tracks` op; nothing in this build
  looks at pixels to produce them. `docs/ROTO_TRACKING.md` records that gap.
* Tracks are matched **by index**, not by name, because a track list is an ordered payload and
  index is what `set_tracks` preserves. A track disabled at either frame is dropped from the
  solve at that frame rather than contributing a stale position.
"""
from __future__ import annotations

import math

from . import shapes

# The fields a solve produces, in the vocabulary `Transform` already speaks. `tiers` imports this
# name so the region rule and the kernel cannot disagree about what a solved transform contains.
SOLVED_FIELDS = ("translate_x", "translate_y", "rotate", "scale", "center_x", "center_y")

IDENTITY = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0,
            "center_x": 0.0, "center_y": 0.0}

# Below this, the reference points are effectively one point and no rotation or scale is
# recoverable from them — the fit would amplify float noise into a visible spin.
MINIMUM_SPREAD = 1e-9


def _usable(payload, frame):
    """Resolved (x, y) per track index at `frame`, or None where the track is disabled there."""
    resolved = shapes.resolve_tracks(payload, frame)
    return [(track["x"], track["y"]) if track["enabled"] else None for track in resolved]


def _pairs(payload, reference_frame, frame):
    reference = _usable(payload, reference_frame)
    current = _usable(payload, frame)
    return [(r, c) for r, c in zip(reference, current) if r is not None and c is not None]


def _similarity(pairs):
    """Least-squares rotation, uniform scale and the two centroids for matched point pairs."""
    count = len(pairs)
    rx = sum(r[0] for r, _ in pairs) / count
    ry = sum(r[1] for r, _ in pairs) / count
    cx = sum(c[0] for _, c in pairs) / count
    cy = sum(c[1] for _, c in pairs) / count
    dot = cross = spread = 0.0
    for (r, c) in pairs:
        ux, uy = r[0] - rx, r[1] - ry
        vx, vy = c[0] - cx, c[1] - cy
        dot += ux * vx + uy * vy
        cross += ux * vy - uy * vx
        spread += ux * ux + uy * uy
    if spread <= MINIMUM_SPREAD:
        # Every reference point sits on top of every other: translation is the whole answer.
        return 0.0, 1.0, (rx, ry), (cx, cy)
    rotate = math.degrees(math.atan2(cross, dot))
    scale = math.hypot(dot, cross) / spread
    return rotate, scale, (rx, ry), (cx, cy)


def solve(payload, frame, params):
    """Solve one Tracker node's transform at `frame`.

    Returns a dict over `SOLVED_FIELDS`. A Tracker with no payload, no enabled track present at
    both frames, or every component switched off resolves to identity — never to "leave the
    region rule undefined", because an unsolved data-dependent node is exactly the silent
    full-frame fall back clause C2 forbids.
    """
    reference_frame = int(params.get("reference_frame", 1))
    pairs = _pairs(payload, reference_frame, int(frame))
    if not pairs:
        return dict(IDENTITY)
    if len(pairs) == 1:
        # One track carries no orientation and no size, so asking for rotation or scale from it
        # would be inventing them. Translation is all it actually measures.
        (r, c), = pairs
        rotate, scale, origin, moved = 0.0, 1.0, r, c
    else:
        rotate, scale, origin, moved = _similarity(pairs)

    if not int(params.get("apply_rotate", 1)):
        rotate = 0.0
    if not int(params.get("apply_scale", 1)):
        scale = 1.0
    translate = (moved[0] - origin[0], moved[1] - origin[1])
    if not int(params.get("apply_translate", 1)):
        translate = (0.0, 0.0)

    if params.get("mode", "match_move") == "match_move":
        return {"translate_x": translate[0], "translate_y": translate[1],
                "rotate": rotate, "scale": scale,
                "center_x": origin[0], "center_y": origin[1]}

    # Stabilise is the inverse of the *gated* match-move transform, so switching a component off
    # removes it from the correction rather than leaving an uninverted remainder behind. Inverting
    # `dst = moved + s*R(t)*(src - origin)` about `moved` gives scale 1/s, rotation -t, and a
    # translation back to the reference centroid.
    inverse_scale = 1.0 / scale if abs(scale) > MINIMUM_SPREAD else 1.0
    return {"translate_x": -translate[0], "translate_y": -translate[1],
            "rotate": -rotate, "scale": inverse_scale,
            "center_x": moved[0], "center_y": moved[1]}


def solve_from_document(document, key, frame):
    """Convenience wrapper: solve the Tracker node `key` in `document` at `frame`."""
    node = document["nodes"][key]
    payload = document.get("node_data", {}).get(key)
    return solve(payload, frame, node["params"])
