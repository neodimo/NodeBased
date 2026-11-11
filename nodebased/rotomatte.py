"""Shapes from a matte over a frame range (past Nuke).

``rotooutline.matte_outlines`` turns one matte picture into editable Bezier shapes. This module does it for a run of
frames: the matte is traced on every Nth frame (the key frames), each region is matched to the one it came from on the
previous key by overlap, and its outline is resampled from the matched start point so that point k of a shape sits in
the same place on the shape at every key. The result is ordinary animated Roto shapes: ``points`` carry one linear curve
per coordinate, a ``lifetime`` range says where the region exists, and the frames between keys are interpolated and then
filled with the existing in-betweens fit (``rotoinbetween``), which refines each frame to that frame's own matte edge.

Regions that split, merge, appear or vanish start or end a shape. The gap in which that happens is traced frame by frame so
the lifetime changes on the right frame; every such event is returned by name.

Pure numpy and the Roto shape model; no GUI. ``matte_for(frame)`` supplies a float H x W matte for a frame.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import rotoedges, rotoinbetween, rotooutline, shapes as shape_model

DEFAULT_STEP = 4                 # key every Nth frame
MATCH_MIN_IOU = 0.1              # overlap, as a share of the union, that makes two outlines the same region
CORNER_WINDOW = 0.08             # a corner is moved to a corner this close (share of the perimeter) on the next key
FILL_BAND = 8.0                  # px an in-between point may move toward the matte edge
FILL_THRESHOLD = 0.75            # px a point must move before the fill keys it (the fit has a half-pixel grid tie)
DEFAULT_TOLERANCE = 1.0          # px an in-between frame's outline may sit from the matte's edge before the frame is keyed from the matte
MAX_ROUNDS = 12                  # key-and-refit passes; every pass keys the worst frame of each gap that is over tolerance
CACHE_BYTES = 256 * 1024 * 1024  # mattes held while measuring
NAME_PREFIX = "matte"


@dataclass
class Track:
    """One shape through time: its outline on every frame it was keyed."""
    name: str
    mode: str
    feathers: list = field(default_factory=list)
    keys: dict = field(default_factory=dict)         # frame -> list of point dicts
    polygons: dict = field(default_factory=dict)     # frame -> dense polygon
    fractions: list = field(default_factory=list)    # where each point sits along the dense outline, 0 to 1
    corners: list = field(default_factory=list)      # which points are corners
    errors: dict = field(default_factory=dict)       # frame -> worst distance of the fitted curve from the traced outline

    @property
    def first(self):
        return min(self.keys)

    @property
    def last(self):
        return max(self.keys)

    @property
    def alive_at(self):
        return self.last


@dataclass
class Event:
    frame: int
    kind: str          # appeared, vanished, split, merged
    shape: str
    other: str = ""    # the shape it split from or merged into

    def __str__(self):
        if self.kind == "split":
            return f"{self.shape} split off {self.other} at frame {self.frame}"
        if self.kind == "merged":
            return f"{self.shape} merged into {self.other} at frame {self.frame}"
        return f"{self.shape} {self.kind} at frame {self.frame}"


@dataclass
class Sequence:
    """The traced shapes, as `Track`s, and what happened to them."""
    tracks: list
    events: list
    key_frames: list
    start: int
    end: int
    added_keys: list = field(default_factory=list)   # frames keyed from the matte because the in-betweens were over tolerance

    def summary(self):
        names = ", ".join(str(e) for e in self.events)
        return names


# -- one outline against the next ---------------------------------------------------------------

def _bbox(polygon):
    return polygon.min(axis=0), polygon.max(axis=0)


def _overlap(a, b):
    """(intersection, union) in pixels of two closed polygons."""
    (ax0, ay0), (ax1, ay1) = _bbox(a)
    (bx0, by0), (bx1, by1) = _bbox(b)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return 0, 1
    x0, y0 = int(math.floor(min(ax0, bx0))), int(math.floor(min(ay0, by0)))
    x1, y1 = int(math.ceil(max(ax1, bx1))) + 1, int(math.ceil(max(ay1, by1))) + 1
    ma = rotooutline.polygon_mask(a, x0, y0, x1 - x0, y1 - y0)
    mb = rotooutline.polygon_mask(b, x0, y0, x1 - x0, y1 - y0)
    return int((ma & mb).sum()), max(int((ma | mb).sum()), 1)


def match(previous, current):
    """Match outlines by overlap. `previous` and `current` are lists of dense polygons; returns
    ``(pairs, overlaps)``: pairs are ``(previous index, current index)`` that are each other's best overlap (and at least
    `MATCH_MIN_IOU` of the union), overlaps maps every overlapping pair to its share of the union."""
    overlaps = {}
    for i, a in enumerate(previous):
        for j, b in enumerate(current):
            inside, union = _overlap(a, b)
            if inside:
                overlaps[(i, j)] = inside / union
    pairs = []
    for (i, j), share in sorted(overlaps.items(), key=lambda item: -item[1]):
        if share < MATCH_MIN_IOU:
            continue
        if any(i == p or j == q for p, q in pairs):
            continue
        best_i = max((s for (a, b), s in overlaps.items() if a == i), default=0)
        best_j = max((s for (a, b), s in overlaps.items() if b == j), default=0)
        if share >= best_i - 1e-12 and share >= best_j - 1e-12:
            pairs.append((i, j))
    return pairs, overlaps


def _arc_fractions(points, dense):
    """Where each fitted point lies along the dense outline (0 to 1): the nearest dense vertex of each."""
    xy = np.array([[p["x"], p["y"]] for p in points])
    near = np.argmin(((xy[:, None, :] - dense[None, :, :]) ** 2).sum(-1), axis=1)
    return (near / len(dense)).tolist()


def _corner_flags(points, dense):
    perimeter = float(np.hypot(*(np.roll(dense, -1, axis=0) - dense).T).sum())
    corners = rotooutline.find_corners(dense, perimeter)
    count = len(dense)
    near = np.array([round(f * count) for f in _arc_fractions(points, dense)])
    return [any(min((n - c) % count, (c - n) % count) <= 2 for c in corners) for n in near]


def _at(dense, position):
    """The point at fractional index `position` of the closed dense polygon."""
    count = len(dense)
    position = position % count
    i = int(math.floor(position))
    t = position - i
    return dense[i % count] * (1 - t) + dense[(i + 1) % count] * t


def correspond(track, dense, fractions=None):
    """New points on `dense` for `track`'s points: the same number, in the same order, starting from the matched start
    point. Returns ``(points, fractions, worst fit error)``.

    The start is the position on the new outline where the previous points (moved by the shift of the outlines' centres) fit
    best. Corners are moved to a nearby corner of the new outline, and the points between follow proportionally.
    """
    previous = track.keys[track.last]
    count = len(dense)
    prev_xy = np.array([[p["x"], p["y"]] for p in previous])
    prev_dense = track.polygons[track.last]
    shift = dense.mean(axis=0) - prev_dense.mean(axis=0)
    phi = np.array(track.fractions, dtype=np.float64)
    phi = (phi - phi[0]) % 1.0
    offsets = phi * count
    # the start whose points land closest to where the previous points went
    starts = np.arange(count)
    index = (starts[:, None] + offsets[None, :]) % count
    lower = np.floor(index).astype(int)
    t = index - lower
    landed = dense[lower % count] * (1 - t)[..., None] + dense[(lower + 1) % count] * t[..., None]
    wanted = prev_xy[None, :, :] + shift
    cost = ((landed - wanted) ** 2).sum(axis=(1, 2))
    start = int(np.argmin(cost))
    # corners: snap to a corner of the new outline when one is close
    perimeter = float(np.hypot(*(np.roll(dense, -1, axis=0) - dense).T).sum())
    new_corners = rotooutline.find_corners(dense, perimeter)
    limit = CORNER_WINDOW * count
    taken = set()

    def nearest(here):
        best, best_distance = None, limit
        for c in new_corners:
            distance = min((c - here) % count, (here - c) % count)
            if distance <= best_distance and c not in taken:
                best, best_distance = c, distance
        return best

    if track.corners and track.corners[0]:
        found = nearest(start)
        if found is not None:
            start = found
            taken.add(found)
    anchors = {0: 0.0}
    for i, flag in enumerate(track.corners):
        if flag and i:
            found = nearest((start + offsets[i]) % count)
            if found is not None:
                taken.add(found)
                anchors[i] = float((found - start) % count)
    # piecewise-linear map of the previous offsets onto the new ones between the anchors (kept in order)
    ordered = sorted((i, a) for i, a in anchors.items())
    targets = offsets.copy()
    sources = [(offsets[i], a) for i, a in ordered] + [(float(count), float(count))]
    keep_order = all(sources[k][1] < sources[k + 1][1] for k in range(len(sources) - 1))
    if keep_order and len(ordered) > 1:
        src = np.array([s for s, _ in sources]); dst = np.array([d for _, d in sources])
        targets = np.interp(offsets, src, dst)
    positions = (start + targets) % count
    xy = np.array([_at(dense, p) for p in positions])
    points, worst = _fit_fixed(xy, positions, dense, track.corners)
    return points, ((positions / count)).tolist(), worst


def _fit_fixed(xy, positions, dense, corners):
    """Bezier points through `xy` (which lie at fractional indices `positions` of the closed `dense` polygon) with handles
    fitted to the polygon between them. Returns ``(points, worst distance of the curve from the polygon)``."""
    count, n = len(dense), len(xy)
    k = 3
    spans, worst = [], 0.0
    tangents_out, tangents_in = [], []
    for i in range(n):
        around = _at(dense, positions[i] + k) - _at(dense, positions[i] - k)
        central = rotooutline._unit(around)
        if corners[i]:
            tangents_out.append(rotooutline._unit(_at(dense, positions[i] + k) - xy[i]))
            tangents_in.append(rotooutline._unit(_at(dense, positions[i] - k) - xy[i]))
        else:
            tangents_out.append(central)
            tangents_in.append(-central)
    beziers = []
    for i in range(n):
        j = (i + 1) % n
        a, b = positions[i], positions[j]
        if b <= a:
            b += count
        inner = [m % count for m in range(int(math.floor(a)) + 1, int(math.ceil(b)))]
        chain = np.vstack([xy[i][None, :], dense[inner] if inner else np.zeros((0, 2)), xy[j][None, :]])
        if len(chain) == 2:
            chain = np.array([chain[0], (chain[0] + chain[1]) / 2, chain[1]])
        u = rotooutline._chord_parameters(chain)
        bez = rotooutline._generate(chain, u, tangents_out[i], tangents_in[j])
        for _ in range(4):
            u = rotooutline._reparameterise(chain, u, bez)
            bez = rotooutline._generate(chain, u, tangents_out[i], tangents_in[j])
        error, _ = rotooutline._max_error(chain, bez, u)
        worst = max(worst, math.sqrt(max(error, 0.0)))
        beziers.append(bez)
    points = rotooutline._straighten(rotooutline._to_points(beziers))
    return points, worst


# -- the sequence ----------------------------------------------------------------------------

def key_frames_for(start, end, step=DEFAULT_STEP, extra=()):
    """Every `step`-th frame from `start`, and `end` itself, plus any `extra` frames inside the range."""
    step = max(1, int(step))
    frames = list(range(int(start), int(end) + 1, step))
    if frames[-1] != int(end):
        frames.append(int(end))
    if extra:
        frames = sorted(set(frames) | {int(f) for f in extra if int(start) <= int(f) <= int(end)})
    return frames


def _trace_frame(matte_for, frame, threshold, min_area, min_points, max_points):
    try:
        return rotooutline.matte_outlines(matte_for(frame), threshold, min_area, min_points, max_points)
    except rotooutline.OutlineError:
        return []


def trace_sequence(matte_for, start, end, step=DEFAULT_STEP, threshold=rotooutline.MATTE_LEVEL,
                   min_area=rotooutline.MATTE_MIN_AREA, min_points=rotooutline.MIN_POINTS,
                   max_points=rotooutline.MAX_POINTS, taken_names=(), extra_keys=(), cache=None):
    """Trace the matte on every `step`-th frame of `start` to `end` and follow each region (see the module docstring).

    `extra_keys` are frames traced as well (keyed from the matte between the regular keys); `cache` is a dict that keeps
    each traced frame so that tracing again with more keys does not trace a frame twice.

    Raises `OutlineError` when the first frame has no region."""
    start, end = int(start), int(end)
    if end < start:
        raise ValueError("the range ends before it starts")
    keys = key_frames_for(start, end, step, extra_keys)
    cache = {} if cache is None else cache

    def _traced(matte_for, frame, threshold, min_area, min_points, max_points):
        if frame not in cache:
            cache[frame] = _trace_frame(matte_for, frame, threshold, min_area, min_points, max_points)
        return cache[frame]

    names = set(taken_names)
    tracks, alive, events = [], [], []

    def new_name():
        index = 1
        while f"{NAME_PREFIX}{index}" in names:
            index += 1
        names.add(f"{NAME_PREFIX}{index}")
        return f"{NAME_PREFIX}{index}"

    def begin(shape, frame, event=None):
        track = Track(new_name(), shape.mode)
        track.keys[frame], track.polygons[frame] = shape.points, shape.polygon
        track.feathers.append(shape.feather)
        track.fractions = _arc_fractions(shape.points, shape.polygon)
        track.corners = _corner_flags(shape.points, shape.polygon)
        tracks.append(track)
        alive.append(track)
        return track

    first = _traced(matte_for, start, threshold, min_area, min_points, max_points)
    if not first:
        raise rotooutline.OutlineError("the matte has no region above the threshold and minimum area on the first frame")
    for shape in first:
        begin(shape, start)

    def advance(frame_a, frame_b, shapes_b):
        """Match the tracks alive at frame_a to `shapes_b`; returns the events (without frames' refinement) and applies the keys."""
        prev_tracks = list(alive)
        pairs, overlaps = match([t.polygons[t.last] for t in prev_tracks], [s.polygon for s in shapes_b])
        pairs = [(i, j) for i, j in pairs if prev_tracks[i].mode == shapes_b[j].mode]
        matched_prev = {i for i, _ in pairs}
        matched_cur = {j for _, j in pairs}
        found = []
        for i, j in pairs:
            track, shape = prev_tracks[i], shapes_b[j]
            points, fractions, worst = correspond(track, shape.polygon)
            track.keys[frame_b], track.polygons[frame_b] = points, shape.polygon
            track.fractions, track.errors[frame_b] = fractions, worst
            track.feathers.append(shape.feather)
        for i, track in enumerate(prev_tracks):
            if i in matched_prev:
                continue
            into = [j for (a, j), s in overlaps.items() if a == i and j in matched_cur]
            other = ""
            if into:
                j = max(into, key=lambda j: overlaps[(i, j)])
                other = next(prev_tracks[p].name for p, q in pairs if q == j)
            found.append(("merged" if other else "vanished", track, other))
            alive.remove(track)
        for j, shape in enumerate(shapes_b):
            if j in matched_cur:
                continue
            origin = [i for (i, b), s in overlaps.items() if b == j]
            other = prev_tracks[max(origin, key=lambda i: overlaps[(i, j)])].name if origin else ""
            track = begin(shape, frame_b)
            found.append(("split" if other else "appeared", track, other))
        return found

    previous = start
    for frame in keys[1:]:
        shapes = _traced(matte_for, frame, threshold, min_area, min_points, max_points)
        snapshot = [(t, dict(t.keys), dict(t.polygons), dict(t.errors), list(t.feathers), list(t.fractions)) for t in alive]
        before = len(tracks)
        found = advance(previous, frame, shapes)
        if found and frame - previous > 1:
            # something changed inside this gap: undo and walk it frame by frame so the event lands on its real frame
            for track in tracks[before:]:
                names.discard(track.name)
            del tracks[before:]
            alive[:] = [item[0] for item in snapshot]
            for track, keys_, polygons, errors, feathers, fractions in snapshot:
                track.keys, track.polygons, track.errors, track.feathers, track.fractions = keys_, polygons, errors, feathers, fractions
            step_prev = previous
            for inner in range(previous + 1, frame + 1):
                inner_shapes = shapes if inner == frame else _traced(matte_for, inner, threshold, min_area, min_points, max_points)
                for kind, track, other in advance(step_prev, inner, inner_shapes):
                    events.append(Event(inner, kind, track.name, other))
                step_prev = inner
        else:
            for kind, track, other in found:
                events.append(Event(frame, kind, track.name, other))
        previous = frame
    frames = sorted({f for t in tracks for f in t.keys})
    return Sequence(tracks, events, frames, start, end)


# -- stored shapes -----------------------------------------------------------------------------------

POINT_FIELDS = shape_model.POINT_FIELDS


def _scalar(values):
    """A stored coordinate: a plain number when it never changes, else a linear curve through its keys."""
    frames = sorted(values)
    first = values[frames[0]]
    if all(abs(values[f] - first) < 1e-9 for f in frames):
        return float(first)
    return {"value": float(first), "curve": {"interpolation": "linear",
                                              "keys": [{"frame": int(f), "value": float(values[f])} for f in frames]}}


def stored_shape(track, template=None):
    """The Roto shape dict for a track (animated points, range lifetime, the measured feather)."""
    count = len(track.keys[track.first])
    points = []
    for i in range(count):
        points.append({name: _scalar({frame: key[i][name] for frame, key in track.keys.items()}) for name in POINT_FIELDS})
    feather = float(np.median(track.feathers)) if track.feathers else 0.0
    shape = {"name": track.name, "mode": track.mode, "opacity": 1.0, "feather": feather,
             "feather_model": "box" if feather > 0 else "edge", "closed": True, "stroke_width": 1.0,
             "points": points, "color": [1.0, 1.0, 1.0, 1.0], "blend_mode": "over", "source": "color",
             "feather_on": True, "falloff_power": 1.0, "feather_falloff": "linear",
             "lifetime": {"type": "range", "start": int(track.first), "end": int(track.last)}}
    if template:
        shape.update({k: copy.deepcopy(v) for k, v in template.items() if k not in ("name", "mode", "points", "lifetime", "feather", "feather_model", "closed")})
    return shape


def stored_shapes(sequence, template=None):
    """Every track as a Roto shape, outers before the holes that cut them (as `matte_outlines` orders them)."""
    order = sorted(sequence.tracks, key=lambda t: (t.mode == "subtract", sequence.tracks.index(t)))
    return [stored_shape(track, template) for track in order]


SCORE_BLUR = 3.0       # px; the matte is softened this much so that the score grows with the distance from the edge


def softened(image, sigma=SCORE_BLUR):
    """The matte blurred, for `level_error`: a hard matte alone says only inside or outside, a softened one says how far."""
    kernel, _ = rotoedges._gaussian_kernel(sigma)
    return rotoedges._convolve_axis(rotoedges._convolve_axis(image.astype(np.float32), kernel, 0), kernel, 1)


def level_error(image, points, closed=True, level=rotooutline.MATTE_LEVEL):
    """How far an outline sits from the matte's `level` contour: the root mean square of (softened matte - level) sampled
    along the flattened outline (`image` comes from `softened`). 0 is an outline lying exactly on the level."""
    from . import roto
    outline = np.array(roto.flatten_points(points, closed=closed))
    height, width = image.shape[:2]
    fx = np.clip(outline[:, 0] - 0.5, 0.0, width - 1.001)
    fy = np.clip(outline[:, 1] - 0.5, 0.0, height - 1.001)
    ix, iy = fx.astype(int), fy.astype(int)
    tx, ty = fx - ix, fy - iy
    top = image[iy, ix] * (1 - tx) + image[iy, ix + 1] * tx
    bottom = image[iy + 1, ix] * (1 - tx) + image[iy + 1, ix + 1] * tx
    sampled = top * (1 - ty) + bottom * ty
    return float(np.sqrt(np.mean((sampled - level) ** 2)))


def fill_between_keys(shapes, matte_for, band=FILL_BAND, threshold=FILL_THRESHOLD, step=1, level=rotooutline.MATTE_LEVEL):
    """The existing in-betweens fit on every frame between a shape's keys, against that frame's matte (edited in place).

    The fit refines the interpolated shape to the matte's edge of the frame (``rotoinbetween.fill_inbetweens``). A frame's
    keys are kept only when they bring the outline closer to the matte's `level` contour than the interpolation was
    (`level_error`), so the fill can only improve a frame. Returns the number of point keys added."""
    added = 0
    cache = {}

    def matte(frame):
        if frame not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[frame] = np.asarray(matte_for(frame), dtype=np.float32)
        return cache[frame]

    soft_cache = {}

    def soft(frame):
        if frame not in soft_cache:
            if len(soft_cache) > 8:
                soft_cache.clear()
            soft_cache[frame] = softened(matte(frame))
        return soft_cache[frame]

    def field_for(frame, box):
        image = matte(frame)
        height, width = image.shape[:2]
        region = rotoedges.aligned_box(*box, width, height)
        if region[2] - region[0] < 3 or region[3] - region[1] < 3:
            return None
        return rotoedges.field_from_image(image[..., None].repeat(3, axis=2), region)

    for index, stored in enumerate(shapes):
        frames = rotoinbetween.key_frames(stored)
        for a, b in zip(frames, frames[1:]):
            if b - a < 2:
                continue

            def resolve(frame, index=index):
                return shape_model.resolve_shapes({"shapes": shapes}, frame)[index]

            result = rotoinbetween.fill_inbetweens(resolve, field_for, a, b, band=band, threshold=threshold,
                                                   step=step, skip_frames=frames)
            for frame in list(result.keys):
                base = resolve(frame)["points"]
                moved = [dict(p) for p in base]
                for point_index, fitted in result.keys[frame].items():
                    moved[point_index] = fitted
                if level_error(soft(frame), moved, level=level) >= 0.98 * level_error(soft(frame), base, level=level):
                    del result.keys[frame]
            added += rotoinbetween.write_keys(stored, result.keys, a, b)
    return added


# -- adaptive keys ---------------------------------------------------------------------------------------

def _big_enough(inside, start, minimum, seen):
    """Whether the region of `inside` holding pixel `start` has at least `minimum` pixels (flood fill that stops there)."""
    from collections import deque
    height, width = inside.shape
    area, pending, own = 0, deque([start]), {start}
    while pending:
        y, x = pending.popleft()
        area += 1
        if area >= minimum:
            seen.update(own)
            return True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and inside[ny, nx] and (ny, nx) not in own:
                own.add((ny, nx))
                pending.append((ny, nx))
    return False


def _window_sum(image, radius, axis):
    """Sum of `image` over the 2 * radius + 1 pixels centred on each pixel along `axis` (zeros beyond the picture)."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius + 1, radius)
    total = np.cumsum(np.pad(image.astype(np.float64), pad), axis=axis)
    length = image.shape[axis]
    upper = np.take(total, range(2 * radius + 1, 2 * radius + 1 + length), axis=axis)
    lower = np.take(total, range(0, length), axis=axis)
    return upper - lower


def outline_error(drawn, matte, level=rotooutline.MATTE_LEVEL, reach=4.0, min_area=1):
    """How far the drawn shapes' edge sits from the matte's edge, in pixels, to a fraction of a pixel.

    `drawn` is the alpha the shapes draw and `matte` the matte, both float H x W. Where they disagree the difference in
    coverage, summed across the edge along the rows (h) and along the columns (v), is how far the edge has moved in x and in y;
    the distance perpendicular to the edge is h * v / sqrt(h^2 + v^2). The result is the worst such distance, capped
    by `reach`. Mattes other than at the 50 % level are compared as the hard region above `level`. Matte regions under
    `min_area` pixels are not traced and are left out."""
    drawn = np.asarray(drawn, dtype=np.float64)
    inside = matte >= level
    flat = (matte if abs(level - 0.5) < 1e-6 else inside).astype(np.float64)
    difference = drawn - flat
    ys, xs = np.nonzero((np.abs(difference) > 0.02) | ((drawn >= 0.5) != inside))
    if not len(ys):
        return 0.0
    if min_area > 1:
        keep, seen, small = [], set(), set()
        for y, x in zip(ys.tolist(), xs.tolist()):
            if not inside[y, x] or (y, x) in seen:
                keep.append(True)
            elif (y, x) in small:
                keep.append(False)
            else:
                big = _big_enough(inside, (y, x), min_area, seen)
                keep.append(big)
                if not big:
                    small.add((y, x))
        mask = np.array(keep, dtype=bool)
        ys, xs = ys[mask], xs[mask]
        if not len(ys):
            return 0.0
    radius = int(math.ceil(reach))
    height, width = inside.shape
    y0, y1 = max(int(ys.min()) - radius, 0), min(int(ys.max()) + radius + 1, height)
    x0, x1 = max(int(xs.min()) - radius, 0), min(int(xs.max()) + radius + 1, width)
    crop = difference[y0:y1, x0:x1]
    h = np.abs(_window_sum(crop, radius, 1))[ys - y0, xs - x0]
    v = np.abs(_window_sum(crop, radius, 0))[ys - y0, xs - x0]
    root = np.sqrt(h * h + v * v)
    perpendicular = np.divide(h * v, root, out=np.zeros_like(root), where=root > 1e-9)
    return float(min(perpendicular.max(), reach))


class _Mattes:
    """Mattes of the frames asked for, kept (oldest dropped first) within a memory budget."""

    def __init__(self, matte_for, budget=CACHE_BYTES):
        self.matte_for, self.budget, self.held = matte_for, budget, {}

    def __call__(self, frame):
        frame = int(frame)
        if frame not in self.held:
            image = np.asarray(self.matte_for(frame), dtype=np.float32)
            while self.held and sum(v.nbytes for v in self.held.values()) + image.nbytes > self.budget:
                del self.held[next(iter(self.held))]
            self.held[frame] = image
        return self.held[frame]


def frame_error(shapes, matte, frame, level=rotooutline.MATTE_LEVEL, min_area=1, reach=4.0):
    """`outline_error` of the stored `shapes` at `frame` against `matte` (a float H x W picture)."""
    from . import roto
    height, width = matte.shape[:2]
    drawn = roto.rasterise(shape_model.resolve_shapes({"shapes": shapes}, frame), width, height)[..., 3]
    return outline_error(drawn, matte, level, reach, min_area)


def _frames_to_key(keyed, errors, tolerance):
    """One frame to key in each gap between neighbouring keyed frames that holds a frame over `tolerance`: of the frames
    over it, the one nearest the middle of the gap, so that every key halves the gap it lands in (a path that curves
    leaves an error that falls with the square of the gap)."""
    chosen = []
    for a, b in zip(keyed, keyed[1:]):
        over = [f for f in range(a + 1, b) if errors.get(f, 0.0) > tolerance]
        if over:
            middle = (a + b) / 2.0
            chosen.append(min(over, key=lambda f: (abs(f - middle), f)))
    return chosen


def shapes_from_matte(matte_for, start, end, step=DEFAULT_STEP, threshold=rotooutline.MATTE_LEVEL,
                      min_area=rotooutline.MATTE_MIN_AREA, fill=True, taken_names=(), template=None,
                      min_points=rotooutline.MIN_POINTS, max_points=rotooutline.MAX_POINTS, tolerance=DEFAULT_TOLERANCE):
    """The whole job: ``(shapes, sequence)``. `shapes` are stored Roto shapes keyed every `step`-th frame of `start` to
    `end` and filled in between (unless `fill` is off); `sequence.events` names the regions that split, merged, appeared
    or vanished.

    With `fill` on and a `tolerance` above 0, every in-between frame is then measured against its own matte
    (`frame_error`). A frame whose outline sits farther than `tolerance` px from the matte's edge is keyed from the matte,
    with the same point correspondence, and the frames around it are fitted again; this repeats until every frame is within
    tolerance or keyed. `sequence.added_keys` lists the frames keyed that way."""
    cache = {}
    sequence = trace_sequence(matte_for, start, end, step, threshold, min_area, min_points, max_points, taken_names, cache=cache)
    shapes = stored_shapes(sequence, template)
    if not (fill and len(sequence.key_frames) > 1):
        return shapes, sequence
    mattes = _Mattes(matte_for)
    fill_between_keys(shapes, mattes, level=threshold)
    if not tolerance or tolerance <= 0:
        return shapes, sequence
    reach = max(float(tolerance), 1.0) + 3.0
    extra = set()
    for _ in range(MAX_ROUNDS):
        keyed = list(sequence.key_frames)
        errors = {f: frame_error(shapes, mattes(f), f, threshold, min_area, reach)
                  for f in range(int(start), int(end) + 1) if f not in set(keyed)}
        worst = _frames_to_key(keyed, errors, tolerance)
        if not worst:
            break
        extra.update(worst)
        sequence = trace_sequence(matte_for, start, end, step, threshold, min_area, min_points, max_points, taken_names,
                                  extra_keys=extra, cache=cache)
        shapes = stored_shapes(sequence, template)
        fill_between_keys(shapes, mattes, level=threshold)
    regular = set(key_frames_for(start, end, step))
    sequence.added_keys = sorted(f for f in sequence.key_frames if f not in regular and f in extra)
    return shapes, sequence
