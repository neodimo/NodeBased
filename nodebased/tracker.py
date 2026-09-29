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


def _analysis_pixels(image, channels="luminance"):
    pixels = getattr(image, "pixels", image)
    array = np.asarray(pixels, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        if channels in (None, "luminance", "luma"):
            gray = (array[..., 0] * np.float32(0.2126) + array[..., 1] * np.float32(0.7152)
                    + array[..., 2] * np.float32(0.0722))
        else:
            indices = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
            requested = (("red", "green", "blue") if channels == "rgb" else
                         (channels,) if isinstance(channels, str) else tuple(channels))
            try:
                selected = [array[..., indices[name.lower()]] for name in requested]
            except (KeyError, IndexError, AttributeError) as error:
                raise AnalysisError("channels must be luminance or red, green, blue, alpha") from error
            if not selected:
                raise AnalysisError("at least one tracking channel is required")
            gray = np.mean(np.stack(selected, axis=0), axis=0)
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


def _subpixel_peak(pattern, image, x, y, radius):
    """Fit the NCC peak after cubic sampling a compact subpixel neighbourhood."""
    offsets = np.linspace(-0.5, 0.5, 41)
    step = float(offsets[1] - offsets[0])
    scores = np.full((len(offsets), len(offsets)), -np.inf, dtype=np.float64)
    height, width = image.shape
    base = np.arange(-radius, radius + 1, dtype=np.float64)
    for iy, dy in enumerate(offsets):
        ys = y + base + dy
        y0 = np.floor(ys).astype(int)
        if y0[0] < 1 or y0[-1] + 2 >= height:
            continue
        fy = ys - y0
        wy = np.stack((-0.5*fy + fy**2 - 0.5*fy**3,
                       1 - 2.5*fy**2 + 1.5*fy**3,
                       0.5*fy + 2*fy**2 - 1.5*fy**3,
                       -0.5*fy**2 + 0.5*fy**3), axis=1)
        for ix, dx in enumerate(offsets):
            xs = x + base + dx
            x0 = np.floor(xs).astype(int)
            if x0[0] < 1 or x0[-1] + 2 >= width:
                continue
            fx = xs - x0
            wx = np.stack((-0.5*fx + fx**2 - 0.5*fx**3,
                           1 - 2.5*fx**2 + 1.5*fx**3,
                           0.5*fx + 2*fx**2 - 1.5*fx**3,
                           -0.5*fx**2 + 0.5*fx**3), axis=1)
            candidate = np.zeros((len(base), len(base)), dtype=np.float64)
            for j in range(4):
                for i in range(4):
                    candidate += (image[np.ix_(y0+j-1, x0+i-1)]
                                  * wy[:, j, None] * wx[None, :, i])
            scores[iy, ix] = _ncc(pattern, candidate)
    iy, ix = np.unravel_index(int(np.argmax(scores)), scores.shape)
    dx = offsets[ix]; dy = offsets[iy]
    if 0 < ix < len(offsets)-1:
        dx += step * _parabola(scores[iy, ix-1], scores[iy, ix], scores[iy, ix+1])
    if 0 < iy < len(offsets)-1:
        dy += step * _parabola(scores[iy-1, ix], scores[iy, ix], scores[iy+1, ix])
    return dx, dy


def _phase_refine(pattern, candidate, radius):
    """Estimate the fractional translation from low-frequency Fourier phase slope."""
    size = pattern.shape[0]
    window = np.outer(np.hanning(size), np.hanning(size))
    a = np.fft.fft2((pattern-pattern.mean())*window)
    b = np.fft.fft2((candidate-candidate.mean())*window)
    ky, kx = np.meshgrid(np.fft.fftfreq(size)*size, np.fft.fftfreq(size)*size, indexing="ij")
    limit = max(2, min(6, radius//2))
    selected = ((np.abs(kx) <= limit) & (np.abs(ky) <= limit)
                & ((kx != 0) | (ky != 0)))
    weight = np.abs(a*np.conj(b))[selected]
    if not np.any(weight > 1e-12):
        return None
    design = (-2*np.pi/size)*np.stack((kx[selected], ky[selected]), axis=1)
    phase = np.angle(b*np.conj(a))[selected]
    root_weight = np.sqrt(weight)
    solution, _, rank, _ = np.linalg.lstsq(design*root_weight[:, None], phase*root_weight, rcond=None)
    if rank < 2 or not np.isfinite(solution).all():
        return None
    return (max(-0.5, min(0.5, float(solution[0]))),
            max(-0.5, min(0.5, float(solution[1]))))


def match_pattern(reference, image, point, pattern_radius=8, search_radius=16, search_point=None,
                  channels="luminance"):
    """Deterministic zero-mean NCC with integer peak and parabolic refinement."""
    pattern_radius, search_radius = _radii(pattern_radius, search_radius)
    ref, ref_origin = _analysis_pixels(reference, channels)
    current, origin = _analysis_pixels(image, channels)
    pattern, pattern_x, pattern_y = _window(ref, float(point[0]), float(point[1]), pattern_radius,
                                            ref_origin, "Pattern")
    point_offset_x = float(point[0]) - (ref_origin[0] + pattern_x + 0.5)
    point_offset_y = float(point[1]) - (ref_origin[1] + pattern_y + 0.5)
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
    candidate = current[py-pattern_radius:py+pattern_radius+1,
                        px-pattern_radius:px+pattern_radius+1]
    refined = _phase_refine(pattern, candidate, pattern_radius)
    dx, dy = refined if refined is not None else _subpixel_peak(pattern, current, px, py, pattern_radius)
    return (float(origin[0] + px + 0.5 + dx + point_offset_x),
            float(origin[1] + py + 0.5 + dy + point_offset_y), scores[peak])


def analyse(frames, reference_frame, point, pattern_radius=8, search_radius=16,
            first_frame=None, last_frame=None, cancel=None, progress=None, direction="forward",
            channels="luminance", adaptive_update=False):
    """Track from the reference toward one timeline direction without mutating a document.

    Backward analysis uses ``first_frame`` as its inclusive endpoint; forward analysis uses
    ``last_frame``. Progress receives (frame, completed, total, confidence).
    """
    _radii(pattern_radius, search_radius)
    get_frame = frames if callable(frames) else lambda frame: frames[frame]
    reference_frame = int(reference_frame)
    first = int(reference_frame if first_frame is None else first_frame)
    last = int(last_frame if last_frame is None else last_frame)
    if direction not in ("forward", "backward"):
        raise AnalysisError("direction must be 'forward' or 'backward'")
    if first > reference_frame or last < reference_frame:
        raise AnalysisError("analysis range must include the reference frame")
    frame_numbers = (range(reference_frame + 1, last + 1) if direction == "forward"
                     else range(reference_frame - 1, first - 1, -1))
    result = {reference_frame: (float(point[0]), float(point[1]))}
    previous = result[reference_frame]
    total = len(frame_numbers)
    reference = get_frame(reference_frame)
    pattern_point = tuple(point)
    for offset, frame in enumerate(frame_numbers, start=1):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        current_image = get_frame(frame)
        match = match_pattern(reference, current_image, pattern_point, pattern_radius, search_radius,
                              search_point=previous, channels=channels)
        previous = match[:2]
        result[frame] = previous
        if progress is not None:
            progress(frame, offset, total, match[2])
        if adaptive_update:
            reference, pattern_point = current_image, previous
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


def solve(payload, frame, params, force_stabilise=False):
    """Solve one Tracker node's transform at `frame`.

    Returns a dict over `SOLVED_FIELDS`. A Tracker with no payload, no enabled track present at
    both frames, or every component switched off resolves to identity — never to "leave the
    region rule undefined", because an unsolved data-dependent node is exactly the silent
    full-frame fall back clause C2 forbids.
    """
    reference_frame = int(params.get("reference_frame", 1))
    smoothing = max(0, int(params.get("smoothing", 0)))
    if smoothing:
        def smoothed_positions(center):
            samples = [shapes.resolve_tracks(payload, sample_frame)
                       for sample_frame in range(center - smoothing, center + smoothing + 1)]
            tracks = []
            for index in range(len((payload or {}).get("tracks", []))):
                usable = [sample[index] for sample in samples if sample[index]["enabled"]]
                tracks.append(None if not usable else
                              (sum(item["x"] for item in usable) / len(usable),
                               sum(item["y"] for item in usable) / len(usable)))
            return tracks
        reference, current = smoothed_positions(reference_frame), smoothed_positions(int(frame))
        pairs = [(r, c) for r, c in zip(reference, current) if r is not None and c is not None]
    else:
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

    if not force_stabilise and params.get("mode", "match_move") == "match_move":
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
