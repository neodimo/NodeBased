"""Reduce keys for Roto shapes and RotoPaint strokes (past Nuke).

Track points, Fill in-betweens and Stick to plate key every frame. That is accurate and heavy to edit: moving one point
means fixing every key. ``reduce_item`` removes keys while every frame stays within a tolerance (default 0.5 px).

What is measured. For each frame between the first and last key the module takes the outline the shape had before, and the
outline it would have with the reduced keys, and compares them point for point:

* the outline of a shape is sampled along its curves (eight samples per span, Bezier or B-spline, tangents and tension
  included), and each feather handle (point plus feather offset), after the shape's own transform;
* the outline of a stroke is its points, plus the effect of Stick to plate's scale and turn on the brush radius and on the
  clone offset, in px.

The interpolation used between keys is the one Roto uses (``smooth``, ``linear`` or ``constant`` curves, endpoint hold), so
the error is what the artist would see, not a difference of raw control values.

What is removed. Keys go at the same frames for all the points of an item, so a shape stays editable as a whole; a frame is
kept when any curve of the item needs it. The first and last key always stay, and so do keys outside the frame range when
one is given. The search adds back the key at the frame with the worst error until every frame is inside the tolerance, so
corners in the motion (a stop, a direction change) stay and straight or eased stretches between them lose their keys.
With ``interpolation="auto"`` the keys of a shape whose curves are smooth or linear are tried both ways (as stored, and
linear) and the way that needs fewer keys is written; ``"keep"`` leaves each curve's interpolation as it was.
"""
from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import shapes as shape_model

DEFAULT_TOLERANCE = 0.5          # px
SAMPLES_PER_SPAN = 8
MAX_FRAMES = 20001
SHAPE_FIELDS = (*shape_model.POINT_FIELDS, *shape_model.FEATHER_POINT_FIELDS, *shape_model.POINT_TENSION_FIELDS)
INTERPOLATIONS = ("auto", "keep")


class ReduceError(ValueError):
    """Raised for an item that cannot be reduced (too many frames, bad tolerance)."""


@dataclass
class Reduction:
    """What reducing one item did. ``before`` and ``after`` count keyed frames of the item."""
    item: dict
    before: int = 0
    after: int = 0
    max_error: float = 0.0
    kept: list = field(default_factory=list)
    interpolation: str = ""          # "" when untouched, else the interpolation written ("as stored" keeps each curve's own)
    changed: bool = False
    within_tolerance: bool = True


# -- sampling the stored scalars ------------------------------------------------------------------------------------------
def _is_stroke(item):
    return item.get("kind") == "stroke" or "brush" in item


def _curve(value):
    return value.get("curve") if isinstance(value, dict) else None


def _interp(key_frames, key_values, frames, mode):
    """`animation.evaluate_curve` over many frames and many curves at once: key_values is (curves, keys)."""
    kf = np.asarray(key_frames, dtype=np.float64)
    kv = np.atleast_2d(np.asarray(key_values, dtype=np.float64))
    frames = np.asarray(frames, dtype=np.float64)
    if len(kf) == 1:
        return np.repeat(kv[:, :1], len(frames), axis=1)
    idx = np.clip(np.searchsorted(kf, frames, side="right") - 1, 0, len(kf) - 2)
    f0, f1 = kf[idx], kf[idx + 1]
    t = np.clip((frames - f0) / (f1 - f0), 0.0, 1.0)
    if mode == "constant":
        t = np.where(frames >= f1, 1.0, 0.0)
    elif mode == "smooth":
        t = t * t * (3.0 - 2.0 * t)
    return kv[:, idx] * (1.0 - t) + kv[:, idx + 1] * t


def _series(value, frames, name):
    """The resolved scalar at every frame (as `shapes.resolve_scalar` gives it, clamped to its bounds)."""
    lo, hi = shape_model.SHAPE_LIMITS[name]
    curve = _curve(value)
    if curve:
        keys = curve["keys"]
        out = _interp([k["frame"] for k in keys], [[k["value"] for k in keys]], frames, curve["interpolation"])[0]
    else:
        base = float(value["value"]) if isinstance(value, dict) else float(value)
        out = np.full(len(frames), base)
    return np.clip(out, lo, hi)


@dataclass
class _Channel:
    path: tuple              # ("point", index, field) or ("stick", name)
    curve: dict | None
    series: np.ndarray       # the resolved scalar on every frame
    mode: str = ""           # the curve's own interpolation


def _channels(item, frames):
    """Every scalar of the item that the outline depends on, sampled on `frames`."""
    channels = []
    names = ("x", "y") if _is_stroke(item) else SHAPE_FIELDS
    for index, point in enumerate(item["points"]):
        for name in names:
            if name not in point:
                continue
            value = point[name]
            curve = _curve(value)
            channels.append(_Channel(("point", index, name), curve, _series(value, frames, name),
                                     curve["interpolation"] if curve else ""))
    if _is_stroke(item):
        for name, neutral in (("scale", 1.0), ("rotation", 0.0), ("shear", 0.0)):
            stick = item.get("stick") or {}
            if name in stick:
                value = stick[name]
                curve = _curve(value)
                limit = {"rotation": "warp_rotation", "shear": "stick_shear"}.get(name, "warp_scale")
                channels.append(_Channel(("stick", name, ""), curve, _series(value, frames, limit),
                                         curve["interpolation"] if curve else ""))
    return channels


# -- the outline the renderer draws --------------------------------------------------------------------------------------
def _bezier_samples(f, closed):
    x, y = f["x"], f["y"]
    count = x.shape[0]
    spans = count if closed else count - 1
    if spans < 1:
        return np.stack([x.T, y.T], axis=-1)
    nxt = np.roll(np.arange(count), -1)
    a = np.arange(spans)
    b = nxt[a]
    t = (np.arange(SAMPLES_PER_SPAN) / SAMPLES_PER_SPAN)[:, None, None]       # (T, 1, 1)
    out = []
    for axis, o_name, i_name in ((x, "out_x", "in_x"), (y, "out_y", "in_y")):
        p0, p3 = axis[a], axis[b]
        p1 = p0 + f[o_name][a]
        p2 = p3 + f[i_name][b]
        u = 1.0 - t
        out.append(u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3)      # (T, spans, frames)
    samples = np.stack(out, axis=-1)                                           # (T, spans, frames, 2)
    samples = samples.transpose(2, 1, 0, 3).reshape(x.shape[1], spans * SAMPLES_PER_SPAN, 2)
    if not closed:
        samples = np.concatenate([samples, np.stack([x[-1], y[-1]], axis=-1)[:, None, :]], axis=1)
    return samples


def _bspline_samples(f, closed):
    x, y = f["x"].copy(), f["y"].copy()
    count = x.shape[0]
    tension = f.get("tension")
    if tension is not None and np.any(tension):
        idx = np.arange(count)
        before = np.roll(idx, 1) if closed else np.maximum(idx - 1, 0)
        after = np.roll(idx, -1) if closed else np.minimum(idx + 1, count - 1)
        if not closed:
            before[0], after[-1] = 0, count - 1
        k = 0.5 * tension
        x = x + k * (x - (x[before] + x[after]) * 0.5)
        y = y + k * (y - (y[before] + y[after]) * 0.5)
    if closed:
        spans = count
        order = [(np.arange(count) + s) % count for s in range(4)]
    else:
        padded = np.concatenate([[0], np.arange(count), [count - 1]])
        spans = max(0, len(padded) - 3)
        order = [padded[s:s + spans] for s in range(4)]
    if spans < 1:
        return np.stack([x.T, y.T], axis=-1)
    t = (np.arange(SAMPLES_PER_SPAN) / SAMPLES_PER_SPAN)[:, None, None]
    u = 1.0 - t
    weights = (u ** 3 / 6.0, (3 * t ** 3 - 6 * t ** 2 + 4) / 6.0, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6.0, t ** 3 / 6.0)
    out = []
    for axis in (x, y):
        out.append(sum(w * axis[o][None] for w, o in zip(weights, order)))      # (T, spans, frames)
    samples = np.stack(out, axis=-1).transpose(2, 1, 0, 3).reshape(x.shape[1], spans * SAMPLES_PER_SPAN, 2)
    return samples


def _affines(item, frames, series):
    """Per-frame (linear 2x2, offset) of the item's own transform, or None when it has none."""
    if "transform" not in item and "extra_matrix" not in item:
        return None
    px = series[("x")]
    py = series[("y")]
    cx_all = (px.min(axis=0) + px.max(axis=0)) / 2.0
    cy_all = (py.min(axis=0) + py.max(axis=0)) / 2.0
    stored = item.get("transform", {})
    mats = np.zeros((len(frames), 2, 2))
    offs = np.zeros((len(frames), 2))
    for n, frame in enumerate(frames):
        t = {name: shape_model.resolve_scalar(stored.get(name, float(cx_all[n]) if name == "center_x" else float(cy_all[n]) if name == "center_y" else default),
                                              int(frame), name)
             for name, default in shape_model.TRANSFORM_DEFAULTS.items()}
        if stored.get("skew_order", "XY") != "XY":
            t["skew_order"] = stored["skew_order"]
        probe = {"transform": t, "points": [{"x": px_, "y": py_, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0,
                                             "feather_x": 0.0, "feather_y": 0.0} for px_, py_ in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))],
                 "stroke_width": 1.0, "feather": 0.0}
        if item.get("extra_matrix") is not None:
            probe["extra_matrix"] = [float(v) for v in item["extra_matrix"]]
        shape_model.apply_shape_transform(probe)
        o, a, b = probe["points"]
        offs[n] = (o["x"], o["y"])
        mats[n] = ((a["x"] - o["x"], b["x"] - o["x"]), (a["y"] - o["y"], b["y"] - o["y"]))
    return mats, offs


def _outline(item, fields, extras, frames, affine):
    """Positions that stand for the item's picture on every frame: shape (frames, samples, 2)."""
    if _is_stroke(item):
        points = np.stack([fields["x"].T, fields["y"].T], axis=-1)                  # (frames, P, 2)
        if affine is not None:
            mats, offs = affine
            points = np.einsum("fij,fpj->fpi", mats, points) + offs[:, None, :]
        scale = extras.get("scale", np.ones(len(frames)))
        rotation = extras.get("rotation", np.zeros(len(frames)))
        shear = extras.get("shear", np.zeros(len(frames)))
        radius = 0.5 * float(item["brush"]["size"])
        dx0, dy0 = (float(v) for v in (item.get("source_offset") or (0.0, 0.0)))
        dx, dy = dx0 + shear * dy0, dy0 + shear * dx0
        cosine, sine = np.cos(rotation) * scale, np.sin(rotation) * scale
        clone = np.stack([cosine * dx - sine * dy, sine * dx + cosine * dy], axis=-1)[:, None, :]
        brush = np.stack([radius * scale, np.zeros_like(scale)], axis=-1)[:, None, :]
        return np.concatenate([points, clone, brush], axis=1)
    closed = bool(item.get("closed", True))
    sampler = _bspline_samples if item.get("curve_type", "bezier") == "bspline" else _bezier_samples
    samples = sampler(fields, closed)
    handles = np.stack([(fields["x"] + fields["feather_x"]).T, (fields["y"] + fields["feather_y"]).T], axis=-1)
    both = np.concatenate([samples, handles], axis=1)
    if affine is not None:
        mats, offs = affine
        both = np.einsum("fij,fpj->fpi", mats, both) + offs[:, None, :]
    return both


# -- the reduction ---------------------------------------------------------------------------------------------------------
def key_frames_of(item):
    """Every frame at which any point coordinate (or a stroke's stuck scale or turn) has a key."""
    frames = set()
    scalars = [value for point in item.get("points", []) for value in point.values()]
    scalars += list((item.get("stick") or {}).values()) if _is_stroke(item) else []
    for value in scalars:
        curve = _curve(value)
        if curve:
            frames.update(int(k["frame"]) for k in curve["keys"])
    return sorted(frames)


def reduce_item(item, tolerance=DEFAULT_TOLERANCE, first=None, last=None, interpolation="auto"):
    """Reduce the keys of one shape or stroke; returns a `Reduction` (the item itself is never modified).

    `first` and `last` limit which keys may be removed (keys outside stay); the error is checked on every frame between the
    item's first and last key.
    """
    tolerance = float(tolerance)
    if not tolerance > 0.0 or not math.isfinite(tolerance):
        raise ReduceError("The tolerance must be more than 0 px")
    if interpolation not in INTERPOLATIONS:
        raise ReduceError(f"interpolation must be one of {list(INTERPOLATIONS)}")
    keyed = key_frames_of(item)
    if len(keyed) <= 2:
        return Reduction(copy.deepcopy(item), len(keyed), len(keyed), 0.0, list(keyed))
    if keyed[-1] - keyed[0] + 1 > MAX_FRAMES:
        raise ReduceError(f"Reduce keys is limited to {MAX_FRAMES:,} frames")
    frames = np.arange(keyed[0], keyed[-1] + 1)
    channels = _channels(item, frames)
    lo = keyed[0] if first is None else int(first)
    hi = keyed[-1] if last is None else int(last)
    if lo > hi:
        lo, hi = hi, lo
    fixed = sorted({keyed[0], keyed[-1], *[k for k in keyed if k < lo or k > hi]})
    fixed_set = set(fixed)
    candidates = [k for k in keyed if k not in fixed_set]

    point_count = len(item["points"])
    names = ("x", "y") if _is_stroke(item) else SHAPE_FIELDS
    base_fields = {name: np.zeros((point_count, len(frames))) for name in names}
    extras = {}
    animated = []
    for channel in channels:
        if channel.path[0] == "point":
            base_fields[channel.path[2]][channel.path[1]] = channel.series
        else:
            extras[channel.path[1]] = channel.series
        if channel.curve:
            animated.append(channel)
    if not animated:
        return Reduction(copy.deepcopy(item), len(keyed), len(keyed), 0.0, list(keyed))
    affine = _affines(item, frames, {"x": base_fields["x"], "y": base_fields["y"]}) if ("transform" in item or "extra_matrix" in item) else None
    original = _outline(item, base_fields, extras, frames, affine)

    def solve(force_linear):
        modes = ["linear" if force_linear and c.mode != "constant" else c.mode for c in animated]
        kept = list(fixed)
        pool = list(candidates)
        errors = np.zeros(len(frames))
        stuck = np.zeros(len(frames), dtype=bool)

        def rebuild(i0, i1):
            """Error on frames[i0:i1 + 1] with the current keys."""
            sub = frames[i0:i1 + 1]
            fields = {name: array[:, i0:i1 + 1].copy() for name, array in base_fields.items()}
            sub_extras = {name: array[i0:i1 + 1].copy() for name, array in extras.items()}
            kf = np.asarray(kept, dtype=np.float64)
            kidx = kf.astype(int) - int(keyed[0])
            for mode in set(modes):
                group = [c for c, m in zip(animated, modes) if m == mode]
                values = np.stack([c.series[kidx] for c in group])
                result = _interp(kf, values, sub, mode)
                for c, row in zip(group, result):
                    if c.path[0] == "point":
                        fields[c.path[2]][c.path[1]] = row
                    else:
                        sub_extras[c.path[1]] = row
            sub_affine = None if affine is None else (affine[0][i0:i1 + 1], affine[1][i0:i1 + 1])
            candidate = _outline(item, fields, sub_extras, sub, sub_affine)
            return np.sqrt(((candidate - original[i0:i1 + 1]) ** 2).sum(axis=-1)).max(axis=1)

        errors[:] = rebuild(0, len(frames) - 1)
        while True:
            live = np.where(stuck, -1.0, errors)
            worst = int(np.argmax(live))
            if live[worst] <= tolerance:
                break
            frame = int(frames[worst])
            at = bisect.bisect_left(kept, frame)
            a, b = kept[at - 1], kept[at]
            position = bisect.bisect_left(pool, frame)
            options = [pool[p] for p in (position - 1, position) if 0 <= p < len(pool) and a < pool[p] < b]
            if not options:
                stuck[a - int(keyed[0]):b - int(keyed[0]) + 1] = True
                continue
            choice = min(options, key=lambda c: abs(c - frame))
            pool.remove(choice)
            bisect.insort(kept, choice)
            at = bisect.bisect_left(kept, choice)
            i0, i1 = kept[at - 1] - int(keyed[0]), kept[at + 1] - int(keyed[0])
            errors[i0:i1 + 1] = rebuild(i0, i1)
        return kept, float(errors.max()), modes

    attempts = [solve(False)]
    if interpolation == "auto" and any(c.mode == "smooth" for c in animated) and not any(c.mode == "constant" for c in animated):
        attempts.append(solve(True))
    kept, max_error, modes = min(enumerate(attempts), key=lambda e: (e[1][1] > tolerance, len(e[1][0]), e[0]))[1]
    if len(kept) >= len(keyed) and modes == [c.mode for c in animated]:
        return Reduction(copy.deepcopy(item), len(keyed), len(keyed), 0.0, list(keyed))
    result = copy.deepcopy(item)
    kidx = np.asarray(kept, dtype=int) - int(keyed[0])
    mode_of = {c.path: m for c, m in zip(animated, modes)}
    for channel in animated:
        keys = [{"frame": int(f), "value": float(channel.series[i])} for f, i in zip(kept, kidx)]
        curve = {"interpolation": mode_of[channel.path], "keys": keys}
        if channel.path[0] == "point":
            _, index, name = channel.path
            target = result["points"][index]
            target[name] = {**target[name], "curve": curve}
        else:
            result["stick"][channel.path[1]] = {**result["stick"][channel.path[1]], "curve": curve}
    written = "as stored" if modes == [c.mode for c in animated] else "linear"
    return Reduction(result, len(keyed), len(kept), max_error, list(kept), written, True, max_error <= tolerance + 1e-9)


def reduce_items(items, indices, tolerance=DEFAULT_TOLERANCE, first=None, last=None, interpolation="auto"):
    """Reduce the shapes and strokes at `indices` of a payload list; returns (new list, summary dict).

    The summary has `reduced` (items changed), `examined` (items with keys to look at), `before`, `after` (keyed frames summed
    over the examined items), `max_error` and `outside_tolerance` (names of items that could not reach the tolerance, which
    happens only when a key outside the frame range holds the error).
    """
    out = copy.deepcopy(items)
    summary = {"reduced": 0, "examined": 0, "before": 0, "after": 0, "max_error": 0.0, "outside_tolerance": [],
               "shapes": 0, "strokes": 0}
    for index in indices:
        item = items[index]
        if not item.get("points") or item.get("kind") not in (None, "shape", "stroke"):
            continue
        result = reduce_item(item, tolerance, first, last, interpolation)
        if not result.before:
            continue
        summary["examined"] += 1
        summary["strokes" if _is_stroke(item) else "shapes"] += 1
        summary["before"] += result.before
        summary["after"] += result.after
        summary["max_error"] = max(summary["max_error"], result.max_error)
        if not result.within_tolerance:
            summary["outside_tolerance"].append(item.get("name", f"item {index + 1}"))
        if result.changed or result.after < result.before:
            out[index] = result.item
            summary["reduced"] += 1
    return out, summary


def describe(summary, tolerance=DEFAULT_TOLERANCE):
    """The status-bar sentence for a `reduce_items` summary."""
    if not summary["examined"]:
        return "Reduce keys: the selection has no animated shapes or strokes"
    parts = [f"{n} {word}{'s' if n != 1 else ''}" for n, word in ((summary["shapes"], "shape"), (summary["strokes"], "stroke")) if n]
    text = (f"Reduce keys: kept {summary['after']} of {summary['before']} keys on {' and '.join(parts)}"
            f" (every frame within {tolerance:g} px; largest difference {summary['max_error']:.2f} px)")
    if summary["outside_tolerance"]:
        text += " · outside the tolerance: " + ", ".join(summary["outside_tolerance"])
    return text
