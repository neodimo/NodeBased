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

from . import shapes, trackerknobs

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


def _analysis_pixels(image, channels="luminance", prefilter="none", clamp=False):
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
    gray = np.ascontiguousarray(gray, dtype=np.float64)
    if clamp:
        gray = np.clip(gray, 0.0, 1.0)
    return trackerknobs.prefiltered(gray, prefilter), origin


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


def _bilinear(image, xs, ys):
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    if x0.min() < 0 or y0.min() < 0 or x0.max() + 1 >= image.shape[1] or y0.max() + 1 >= image.shape[0]:
        raise AnalysisError("warped pattern extends outside the image")
    fx, fy = xs - x0, ys - y0
    a, b = image[y0, x0], image[y0, x0 + 1]
    c, d = image[y0 + 1, x0], image[y0 + 1, x0 + 1]
    return a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + d*fx*fy


def _fit_pattern_warp(pattern, image, cx, cy, radius, warp, max_iter, epsilon):
    """Refine a translated match with a least-squares local similarity/affine fit."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    target = (pattern - pattern.mean()) / max(float(pattern.std()), 1e-12)
    # x/y, angle radians, log scale, affine shear x/y
    p = np.array([cx, cy, 0., 0., 0., 0.])
    enabled = {"Translate/Rotate": (0, 1, 2), "Translate/Scale": (0, 1, 3),
               "Translate/Rotate/Scale": (0, 1, 2, 3), "Affine": (0, 1, 2, 3, 4, 5)}.get(warp)
    if enabled is None:
        raise AnalysisError(f"unknown Tracker warp type: {warp}")
    def sample(q):
        angle, scale = q[2], math.exp(q[3])
        ca, sa = math.cos(angle)*scale, math.sin(angle)*scale
        xs = q[0] + ca*xx - sa*yy + q[4]*yy
        ys = q[1] + sa*xx + ca*yy + q[5]*xx
        v = _bilinear(image, xs, ys)
        return (v-v.mean()) / max(float(v.std()), 1e-12)
    for _ in range(max(1, int(max_iter))):
        residual = (sample(p)-target).ravel()
        cols = []
        for i in enabled:
            h = 0.05 if i < 2 else 1e-3
            q = p.copy(); q[i] += h
            cols.append(((sample(q)-target).ravel()-residual)/h)
        try:
            delta, *_ = np.linalg.lstsq(np.column_stack(cols), -residual, rcond=None)
        except np.linalg.LinAlgError:
            break
        delta = np.clip(delta, [-1.0 if i < 2 else -0.05 for i in enabled],
                        [1.0 if i < 2 else 0.05 for i in enabled])
        old_error = float(np.dot(residual, residual))
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125):
            trial = p.copy()
            for i, step in zip(enabled, delta): trial[i] += float(step) * factor
            try:
                error = float(np.sum((sample(trial)-target).ravel() ** 2))
            except AnalysisError:
                continue
            if error <= old_error:
                p, accepted = trial, True
                break
        if not accepted: break
        if float(np.linalg.norm(delta)) <= max(1e-6, float(epsilon)): break
    return float(p[0]), float(p[1]), math.degrees(float(p[2])), math.exp(float(p[3])), _ncc(target, sample(p))


def _ssd_score(pattern, candidate):
    """One minus the squared difference over the pattern's own energy: 1 for an exact copy, falling as the levels differ."""
    energy = float(np.sum((pattern - pattern.mean(dtype=np.float64)) ** 2))
    if energy <= 1e-15:
        return float("nan")
    return 1.0 - float(np.sum((pattern - candidate) ** 2)) / energy


def match_pattern(reference, image, point, pattern_radius=8, search_radius=16, search_point=None,
                  channels="luminance", metric="ncc", prefilter="none", clamp=False, min_score=MINIMUM_MATCH_SCORE,
                  warp="Translate", max_iter=100, epsilon=0.01, return_warp=False):
    """Deterministic zero-mean NCC with integer peak and parabolic refinement.

    `metric` "ssd" compares the levels themselves instead of their pattern, so a change of exposure or gain moves the best match
    (Nuke's `adjust for luminance changes` off); `prefilter` and `clamp` prepare both images first (nodebased/trackerknobs.py);
    a best score under `min_score` is an error."""
    pattern_radius, search_radius = _radii(pattern_radius, search_radius)
    ref, ref_origin = _analysis_pixels(reference, channels, prefilter, clamp)
    current, origin = _analysis_pixels(image, channels, prefilter, clamp)
    compare = _ncc if metric == "ncc" else _ssd_score
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
            score = compare(pattern, current[y-pattern_radius:y+pattern_radius+1,
                                            x-pattern_radius:x+pattern_radius+1])
            if math.isfinite(score):
                scores[(x, y)] = score
    if not scores:
        raise AnalysisError("search window is out of bounds; no complete candidate windows remain")
    peak = max(scores, key=lambda key: (scores[key], -key[1], -key[0]))
    px, py = peak
    if scores[peak] < min_score:
        raise AnalysisError(f"no reliable match (best {'NCC' if metric == 'ncc' else 'SSD'} score {scores[peak]:.3f} < "
                            f"{min_score:.3f}); the point may be occluded")
    candidate = current[py-pattern_radius:py+pattern_radius+1,
                        px-pattern_radius:px+pattern_radius+1]
    refined = _phase_refine(pattern, candidate, pattern_radius)
    dx, dy = refined if refined is not None else _subpixel_peak(pattern, current, px, py, pattern_radius)
    mx, my, angle, scale, score = float(px + dx), float(py + dy), 0.0, 1.0, scores[peak]
    if warp != "Translate":
        mx, my, angle, scale, score = _fit_pattern_warp(pattern, current, mx, my, pattern_radius,
                                                        warp, max_iter, epsilon)
    result = (float(origin[0] + mx + 0.5 + point_offset_x),
              float(origin[1] + my + 0.5 + point_offset_y), score)
    return (*result, angle, scale) if return_warp else result


def analyse(frames, reference_frame, point, pattern_radius=8, search_radius=16,
            first_frame=None, last_frame=None, cancel=None, progress=None, direction="forward",
            channels="luminance", adaptive_update=False, roi_frame=None,
            metric="ncc", prefilter="none", clamp=False, predict=False, grab=None, max_error=None, info=None,
            warp="Translate", max_iter=100, epsilon=0.01, pattern_point=None, regrab_when_offset=True):
    """Track from the reference toward one timeline direction without mutating a document.

    Backward analysis uses ``first_frame`` as its inclusive endpoint; forward analysis uses
    ``last_frame``. Progress receives (frame, completed, total, confidence).

    The Tracker's Nuke knobs (nodebased/trackerknobs.py): `metric`, `prefilter` and `clamp` prepare the matching;
    `predict` centres each search on where the motion so far says the point will be; `grab` (a `trackerknobs.Grab`) says
    when the pattern is grabbed again from the frame just tracked, and `adaptive_update` is the older spelling of "every
    frame"; `max_error` stops the analysis, keeping the frames tracked so far, at the first frame whose error (one minus the
    score) is above it, and says where in `info` (`stopped_at`, `error`).
    """
    _radii(pattern_radius, search_radius)
    get_frame = frames if callable(frames) else lambda frame: frames[frame]
    reference_frame = int(reference_frame)
    first = int(reference_frame if first_frame is None else first_frame)
    last = int(reference_frame if last_frame is None else last_frame)
    if direction not in ("forward", "backward"):
        raise AnalysisError("direction must be 'forward' or 'backward'")
    if first > reference_frame or last < reference_frame:
        raise AnalysisError("analysis range must include the reference frame")
    frame_numbers = (range(reference_frame + 1, last + 1) if direction == "forward"
                     else range(reference_frame - 1, first - 1, -1))
    if grab is None and adaptive_update:
        grab = trackerknobs.Grab("every frame")
    result = {reference_frame: (float(point[0]), float(point[1]))}
    previous = result[reference_frame]
    velocity = (0.0, 0.0)
    total = len(frame_numbers)
    # A manually offset track can start at a new coordinate while retaining the old
    # reference template when regrab-when-offset is disabled.
    pattern_point = tuple(point if regrab_when_offset or pattern_point is None else pattern_point)
    reference = (roi_frame(reference_frame, pattern_point, pattern_radius + 2)
                 if roi_frame is not None else get_frame(reference_frame))
    template_warp = [0.0, 1.0]
    for offset, frame in enumerate(frame_numbers, start=1):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        centre = (previous[0] + velocity[0], previous[1] + velocity[1]) if predict else previous
        margin = search_radius + pattern_radius + 2 + (int(math.ceil(max(abs(velocity[0]), abs(velocity[1])))) if predict else 0)
        current_image = (roi_frame(frame, centre, margin) if roi_frame is not None else get_frame(frame))
        match = match_pattern(reference, current_image, pattern_point, pattern_radius, search_radius,
                              search_point=centre, channels=channels, metric=metric, prefilter=prefilter, clamp=clamp,
                              min_score=MINIMUM_MATCH_SCORE if max_error is None else -math.inf,
                              warp=warp, max_iter=max_iter, epsilon=epsilon, return_warp=warp != "Translate")
        error = max(0.0, 1.0 - match[2])
        if max_error is not None and error > max_error:
            if info is not None:
                info.update(stopped_at=frame, error=error)
            break
        velocity = (match[0] - previous[0], match[1] - previous[1])
        previous = match[:2]
        result[frame] = previous
        if warp != "Translate":
            warp_total = [template_warp[0] + match[3], template_warp[1] * match[4]]
            if info is not None:
                info.setdefault("warps", {})[frame] = {"rotate": warp_total[0], "scale": warp_total[1]}
        if progress is not None:
            progress(frame, offset, total, match[2])
        if grab is not None and grab.due(offset, error):
            reference, pattern_point = current_image, previous
            if warp != "Translate":
                template_warp = warp_total
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


def _entries(payload, source_frame, frame, smoothing=0):
    """[(source point, destination point, (T, R, S))] for every track usable at both frames.

    A track is usable at a frame where it is enabled. With `smoothing` each point is the mean over the frames either side."""
    tracks = (payload or {}).get("tracks", [])
    flags = [trackerknobs.track_flags(track) for track in tracks]
    if smoothing:
        def smoothed_positions(center):
            samples = [shapes.resolve_tracks(payload, sample_frame)
                       for sample_frame in range(center - smoothing, center + smoothing + 1)]
            positions = []
            for index in range(len(tracks)):
                usable = [sample[index] for sample in samples if sample[index]["enabled"]]
                positions.append(None if not usable else
                                 (sum(item["x"] for item in usable) / len(usable),
                                  sum(item["y"] for item in usable) / len(usable)))
            return positions
        reference, current = smoothed_positions(int(source_frame)), smoothed_positions(int(math.floor(frame)))
    else:
        reference, current = _usable(payload, source_frame), _usable(payload, frame)
    return [(r, c, f) for r, c, f in zip(reference, current, flags) if r is not None and c is not None]


def _jitter_entries(payload, frame, period, removing):
    """Each track's position at `frame` against its mean over `period` frames about it: raw to mean when `removing` the jitter
    (the picture follows the mean path), mean to raw when adding it."""
    tracks = (payload or {}).get("tracks", [])
    flags = [trackerknobs.track_flags(track) for track in tracks]
    half = max(1, int(period) // 2)
    samples = [shapes.resolve_tracks(payload, frame + offset) for offset in range(-half, half + 1)]
    raw = shapes.resolve_tracks(payload, frame)
    entries = []
    for index, track in enumerate(raw):
        usable = [sample[index] for sample in samples if sample[index]["enabled"]]
        if not track["enabled"] or not usable:
            continue
        mean = (sum(item["x"] for item in usable) / len(usable), sum(item["y"] for item in usable) / len(usable))
        point = (track["x"], track["y"])
        entries.append((point, mean, flags[index]) if removing else (mean, point, flags[index]))
    return entries


def _fit(entries):
    """(rotate, scale, origin, moved) for the entries: translation from the tracks with T, rotation from those with R, scale from S."""
    if len(entries) == 1:
        # One track carries no orientation and no size, so asking for rotation or scale from it
        # would be inventing them. Translation is all it actually measures.
        (r, c, _), = entries
        return 0.0, 1.0, r, c
    if all(all(flags) for _, _, flags in entries):
        return _similarity([(r, c) for r, c, _ in entries])
    translating = [(r, c) for r, c, flags in entries if flags[0]]
    rotating = [(r, c) for r, c, flags in entries if flags[1]]
    scaling = [(r, c) for r, c, flags in entries if flags[2]]
    rotate = _similarity(rotating)[0] if len(rotating) >= 2 else 0.0
    scale = _similarity(scaling)[1] if len(scaling) >= 2 else 1.0
    base = translating or [(r, c) for r, c, _ in entries]
    origin = (sum(r[0] for r, _ in base) / len(base), sum(r[1] for r, _ in base) / len(base))
    moved = (sum(c[0] for _, c in base) / len(base), sum(c[1] for _, c in base) / len(base)) if translating else origin
    return rotate, scale, origin, moved


def own_transform_key(params):
    """The node's own transform knobs as a string. A solved transform replaces them while the live link is on, so they would
    leave a cached picture's key alone; carried in the effective parameters, an edit to one still changes the key (and
    recomputes a picture it did not change), which is what the cache rule asks of every parameter."""
    return ",".join(f"{name}={params.get(name, default)}" for name, default in IDENTITY.items())


def manual_transform(params):
    """The node's own transform knobs, as the six solved fields: what a Tracker applies when its live link is off or its mode is none."""
    return {name: float(params.get(name, default)) for name, default in IDENTITY.items()}


def solve(payload, frame, params, force_stabilise=False):
    """Solve one Tracker node's transform at `frame`.

    Returns a dict over `SOLVED_FIELDS`. A Tracker with no payload, no enabled track present at
    both frames, or every component switched off resolves to identity — never to "leave the
    region rule undefined", because an unsolved data-dependent node is exactly the silent
    full-frame fall back clause C2 forbids.

    The mode is the node's `mode` (Nuke's `transform`); with it at `none`, or the live link off, the answer is the node's own
    transform knobs instead (a Stabilize is always solved).
    """
    mode = "stabilise" if force_stabilise else str(params.get("mode", "match_move"))
    if not force_stabilise and (mode == "none" or not int(params.get("livelink_transform", 1))):
        return manual_transform(params)
    one_point = mode.endswith("_1pt")
    mode = mode[:-4] if one_point else mode
    if mode in ("remove_jitter", "add_jitter"):
        entries = _jitter_entries(payload, frame, params.get("jitter_period", 10), mode == "remove_jitter")
    else:
        reference_frame = int(params.get("reference_frame", 1))
        entries = _entries(payload, reference_frame, frame, max(0, int(params.get("smoothing", 0))))
    if one_point:
        entries = entries[:1]
    if not entries:
        return dict(IDENTITY)
    rotate, scale, origin, moved = _fit(entries)
    # A single point has no positional baseline for R/S. The Auto-Tracking warp fit records
    # those components on that track, so its per-frame orientation and size can drive output.
    if payload and len(entries) == 1:
        reference_tracks = _usable(payload, int(params.get("reference_frame", 1)))
        current_tracks = _usable(payload, frame)
        for index, track in enumerate(payload.get("tracks", [])):
            if reference_tracks[index] is None or current_tracks[index] is None:
                continue
            if "warp_rotation" in track or "warp_scale" in track:
                if "warp_rotation" in track and trackerknobs.track_flags(track)[1]:
                    rotate = (shapes.resolve_scalar(track["warp_rotation"], frame, "warp_rotation")
                              - shapes.resolve_scalar(track["warp_rotation"], int(params.get("reference_frame", 1)), "warp_rotation"))
                if "warp_scale" in track and trackerknobs.track_flags(track)[2]:
                    ref_scale = shapes.resolve_scalar(track["warp_scale"], int(params.get("reference_frame", 1)), "warp_scale")
                    now_scale = shapes.resolve_scalar(track["warp_scale"], frame, "warp_scale")
                    if ref_scale > MINIMUM_SPREAD:
                        scale = now_scale / ref_scale
                break

    if not int(params.get("apply_rotate", 1)):
        rotate = 0.0
    if not int(params.get("apply_scale", 1)):
        scale = 1.0
    translate = (moved[0] - origin[0], moved[1] - origin[1])
    if not int(params.get("apply_translate", 1)):
        translate = (0.0, 0.0)

    if mode != "stabilise":
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
