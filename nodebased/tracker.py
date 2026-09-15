"""Solve a similarity transform from tracker tracks.

A Tracker is a Transform whose numbers are solved from `document["node_data"]` instead of typed
into the inspector, so everything here produces the six fields `nodebased.imaging` already knows
how to apply and `nodebased.tiers._transform_rule` already knows how to invert. Nothing in this
module rasterises; it turns two frames' worth of track positions into geometry.

The solve is the closed-form least-squares 2D similarity (Umeyama without reflection): the single
rotation, uniform scale and translation that best carries the reference-frame track positions onto
this frame's. Least squares rather than "use the first two tracks" because a real track set is
noisy and an artist expects adding a fourth good track to improve the result, not to be ignored.

Current boundaries, named rather than hidden:

* No perspective, no shear, no per-track weighting. Four-corner-pin is a different node.
* Pixel analysis is a bounded forward point tracker. It does not provide planar tracking,
  perspective, per-track weighting, or automatic occlusion recovery.
* Tracks are matched **by index**, not by name, because a track list is an ordered payload and
  index is what `set_tracks` preserves. A track disabled at either frame is dropped from the
  solve at that frame rather than contributing a stale position.
"""
from __future__ import annotations

import math
from concurrent.futures import CancelledError

import numpy as np

from . import shapes

# The fields a solve produces, in the vocabulary `Transform` already speaks. `tiers` imports this
# name so the region rule and the kernel cannot disagree about what a solved transform contains.
SOLVED_FIELDS = ("translate_x", "translate_y", "rotate", "scale", "center_x", "center_y")

IDENTITY = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0,
            "center_x": 0.0, "center_y": 0.0}

# Below this, the reference points are effectively one point and no rotation or scale is
# recoverable from them — the fit would amplify float noise into a visible spin.
MINIMUM_SPREAD = 1e-9
MINIMUM_MATCH_SCORE = 0.5


class AnalysisError(ValueError):
    """Actionable failure from pixel analysis; callers can show this directly to an artist."""


def _analysis_pixels(image):
    pixels = getattr(image, "pixels", image)
    array = np.asarray(pixels, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        gray = (array[..., 0] * np.float32(0.2126) + array[..., 1] * np.float32(0.7152)
                + array[..., 2] * np.float32(0.0722))
    elif array.ndim == 2:
        gray = array
    else:
        raise AnalysisError("pixel analysis requires an HxW or HxWxRGBA float image")
    window = getattr(image, "data", None)
    origin = (int(getattr(window, "x", 0)), int(getattr(window, "y", 0)))
    if gray.size == 0 or not np.isfinite(gray).all():
        raise AnalysisError("pixel analysis requires finite, non-empty image pixels")
    return np.ascontiguousarray(gray, dtype=np.float64), origin


def _radii(pattern_radius, search_radius):
    for name, value in (("pattern_radius", pattern_radius), ("search_radius", search_radius)):
        if type(value) is not int or value < 1:
            raise AnalysisError(f"{name} must be an integer >= 1")
    if search_radius < pattern_radius:
        raise AnalysisError("search_radius must be >= pattern_radius")
    return pattern_radius, search_radius


def _window(image, center_x, center_y, radius, origin, label):
    height, width = image.shape
    ix = int(round(center_x - origin[0] - 0.5))
    iy = int(round(center_y - origin[1] - 0.5))
    left, top = ix - radius, iy - radius
    right, bottom = ix + radius + 1, iy + radius + 1
    if left < 0 or top < 0 or right > width or bottom > height:
        raise AnalysisError(f"{label} window is out of bounds at ({center_x:g}, {center_y:g}); "
                            f"need {radius}px margin inside data window origin {origin}")
    return image[top:bottom, left:right], ix, iy


def _ncc(pattern, candidate):
    a = pattern - pattern.mean(dtype=np.float64)
    b = candidate - candidate.mean(dtype=np.float64)
    denominator = math.sqrt(float(np.dot(a.ravel(), a.ravel()) * np.dot(b.ravel(), b.ravel())))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(a.ravel(), b.ravel()) / denominator)


def _parabola(left, center, right):
    denominator = left - 2.0 * center + right
    if not math.isfinite(denominator) or abs(denominator) <= 1e-15:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (left - right) / denominator))


def match_pattern(reference, image, point, pattern_radius=8, search_radius=16, search_point=None):
    """Deterministic zero-mean NCC with integer peak and parabolic refinement."""
    pattern_radius, search_radius = _radii(pattern_radius, search_radius)
    ref, ref_origin = _analysis_pixels(reference)
    current, origin = _analysis_pixels(image)
    pattern, _, _ = _window(ref, float(point[0]), float(point[1]), pattern_radius,
                            ref_origin, "Pattern")
    if float(pattern.std(dtype=np.float64)) <= 1e-12:
        raise AnalysisError("pattern window has insufficient texture (zero variance)")
    search_point = point if search_point is None else search_point
    seed_x = int(round(float(search_point[0]) - origin[0] - 0.5))
    seed_y = int(round(float(search_point[1]) - origin[1] - 0.5))
    scores = {}
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            x, y = seed_x + dx, seed_y + dy
            if x - pattern_radius < 0 or y - pattern_radius < 0 or \
                    x + pattern_radius + 1 > current.shape[1] or y + pattern_radius + 1 > current.shape[0]:
                continue
            score = _ncc(pattern, current[y-pattern_radius:y+pattern_radius+1,
                                         x-pattern_radius:x+pattern_radius+1])
            if math.isfinite(score):
                scores[(x, y)] = score
    if not scores:
        raise AnalysisError("search window is out of bounds; no complete candidate windows remain")
    peak = max(scores, key=lambda key: (scores[key], -key[1], -key[0]))
    px, py = peak
    if scores[peak] < MINIMUM_MATCH_SCORE:
        raise AnalysisError(f"no reliable match (best NCC score {scores[peak]:.3f} < "
                            f"{MINIMUM_MATCH_SCORE:.3f}); the point may be occluded")
    dx = _parabola(scores.get((px - 1, py), scores[(px, py)]), scores[(px, py)],
                   scores.get((px + 1, py), scores[(px, py)]))
    dy = _parabola(scores.get((px, py - 1), scores[(px, py)]), scores[(px, py)],
                   scores.get((px, py + 1), scores[(px, py)]))
    return (float(origin[0] + px + 0.5 + dx), float(origin[1] + py + 0.5 + dy), scores[peak])


def analyse(frames, reference_frame, point, pattern_radius=8, search_radius=16,
            first_frame=None, last_frame=None, cancel=None, progress=None):
    """Ordered forward analysis; frames is a mapping or callable and no document is mutated."""
    _radii(pattern_radius, search_radius)
    get_frame = frames if callable(frames) else lambda frame: frames[frame]
    reference_frame = int(reference_frame)
    first = int(reference_frame if first_frame is None else first_frame)
    last = int(last_frame if last_frame is None else last_frame)
    if first != reference_frame:
        raise AnalysisError("forward analysis must start at the reference frame")
    if last < reference_frame:
        raise AnalysisError("analysis range must include the reference frame")
    result = {first: (float(point[0]), float(point[1]))}
    previous = result[first]
    total = last - first + 1
    reference = get_frame(first)
    for offset, frame in enumerate(range(first + 1, last + 1), start=1):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        match = match_pattern(reference, get_frame(frame), point, pattern_radius, search_radius,
                              search_point=previous)
        previous = match[:2]
        result[frame] = previous
        if progress is not None:
            progress(frame, offset, total, match[2])
    return result


analyze = analyse
match = match_pattern


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
