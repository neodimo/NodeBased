"""Nuke's knobs on the four Time-menu nodes that retime a clip: TimeOffset, FrameHold, Retime and TimeClip.

Each node turns the timeline frame into the frame its input is evaluated at. `plan(...)` does that and answers with a
`TimePlan`: the input frames to evaluate with their weights, whether the frame is black, and a gain for TimeClip's fades.
Pure Python (no NumPy), so the evaluator and the tests share it.

Frame arithmetic, in Nuke's order:

* TimeOffset: `frame - time_offset` (the legacy `reverse` knob flips the sign of the offset, as it always did); then
  `reverse_input` flips the result about the input clip's range (`first + last - t`), so a reversed clip keeps the span an
  offset clip would have; then `rounding_mode`: "Whole frames" rounds down, "Fractional time" keeps the exact time.
* FrameHold: `first_frame` while `increment` is 0; else `first_frame + k * increment`, `k = floor((frame - first) / increment)`
  and 0 before the first frame. Rounded by `rounding_mode` like TimeOffset.
* Retime: output frame `f` maps to input `x = input_first + (f - output_first) * speed` (`input_last - ...` when `reverse`).
  A range end or start that is not locked is the input clip's own; with both output ends locked the speed is
  `(input_last - input_first) / (output_last - output_first)`. `before` / `after` (hold, loop, bounce, black, or continue,
  which keeps the line going) say what the input frame becomes outside the input range. `filter`: none passes the
  fractional frame on, nearest rounds it (ties to the even frame, as Retime always did), box averages the input frames under
  a box of width `|speed| * shutter` centred on `x`, each weighted by how much of its unit cell the box covers.
* TimeClip: the `frame` knob (`frame_mode`: expression, start at, offset; the same as Read's), then the legacy
  `time_offset`, then the clip range `[first, last]` cut to the original range `[origfirst, origlast]`, with `before` /
  `after`; `reverse` plays the range backwards; `fade_in` and `fade_out` dim the clip's first and last frames by
  `(k + 1) / (fade + 1)` at the k-th frame from that end (this reading of the fade is ours: the guide gives no curve).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

KINDS = ("TimeOffset", "FrameHold", "Retime", "TimeClip")
ROUNDING_MODES = ("Whole frames", "Fractional time")
RETIME_RANGE = ("continue", "loop", "bounce", "hold", "black")
RETIME_FILTERS = ("none", "nearest", "box")
MAX_BOX_FRAMES = 64

# TimeOffset and FrameHold's path mask (fidelity step F, nodebased/pathmask.py): off, with no patterns, injecting nothing.
PATH_MASK = {"enable_path_mask": 0, "mask_patterns": "", "inject_mask": 0}

# What a new node gets (Nuke's defaults where the guide states them) and what an older document's node gets, which is
# what that node did before the knob existed.
NEW = {
    "TimeOffset": {"rounding_mode": "Whole frames", "reverse_input": 0, **PATH_MASK},
    "FrameHold": {"rounding_mode": "Whole frames", **PATH_MASK},
    "Retime": {"input_range_start_lock": 0, "input_range_end_lock": 0, "output_range_start_lock": 0,
               "output_range_end_lock": 0, "reverse": 0, "retime_before": "hold", "retime_after": "hold",
               "retime_filter": "box", "shutter": 1.0},
    "TimeClip": {"fade_in": 5.0, "fade_out": 5.0, "frame_mode": "expression", "frame_expr": "", "origfirst": 0,
                 "origlast": 0, "reverse": 0},
}
OLD = {
    "TimeOffset": {"rounding_mode": "Whole frames", "reverse_input": 0, **PATH_MASK},
    "FrameHold": {"rounding_mode": "Whole frames", **PATH_MASK},
    "Retime": {"input_range_start_lock": 1, "input_range_end_lock": 0, "output_range_start_lock": 1,
               "output_range_end_lock": 0, "reverse": 0, "retime_before": "continue", "retime_after": "continue",
               "retime_filter": "nearest", "shutter": 1.0},
    "TimeClip": {"fade_in": 0.0, "fade_out": 0.0, "frame_mode": "expression", "frame_expr": "", "origfirst": 0,
                 "origlast": 0, "reverse": 0},
}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(OLD.get(kind, {}))


@dataclass
class TimePlan:
    """The input frames to evaluate (frame, weight), the frame to size a black result from, and a gain on the result."""
    samples: list = field(default_factory=list)
    size_frame: float = 0
    gain: float = 1.0

    @property
    def black(self):
        return not self.samples


def _tidy(value):
    """An int for a whole number, else a float: a whole frame stays cacheable and compares equal to the int."""
    value = float(value)
    return int(round(value)) if abs(value - round(value)) < 1e-9 else value


def _floor(value):
    return int(math.floor(float(value) + 1e-9))


def _rounded(value, mode):
    return _tidy(value) if str(mode) == "Fractional time" else _floor(value)


def _apply_policy(frame, lo, hi, policy):
    """`frame`, which lies outside [lo, hi], under `policy`: (frame, black). `continue` keeps it."""
    nearest = lo if frame < lo else hi
    if policy == "continue":
        return frame, False
    if policy == "black":
        return nearest, True
    if policy == "loop":
        count = hi - lo + 1
        return lo + math.fmod(math.fmod(frame - lo, count) + count, count), False
    if policy == "bounce":
        if hi == lo:
            return lo, False
        period = 2.0 * (hi - lo)
        r = math.fmod(math.fmod(frame - lo, period) + period, period)
        return lo + (r if r <= hi - lo else period - r), False
    return nearest, False   # hold


def _range_frame(frame, lo, hi, before, after):
    """Map `frame` into [lo, hi] under the before/after policy: (frame, black, outside)."""
    if lo > hi:
        lo, hi = hi, lo
    if lo <= frame <= hi:
        return frame, False, False
    mapped, black = _apply_policy(frame, lo, hi, before if frame < lo else after)
    return mapped, black, True


def _expression(text, frame):
    from .expressions import evaluate
    text = str(text).strip()
    if not text:
        return float(frame)

    def lookup(reference):
        raise ValueError("TimeClip: the frame expression can use `frame` and arithmetic, not other knobs")
    return evaluate(text, frame, lookup)


def _one(frame, size_frame=None, gain=1.0):
    return TimePlan([(frame, 1.0)], frame if size_frame is None else size_frame, gain)


def time_offset_plan(params, frame, clip):
    offset = float(params.get("time_offset", 0))
    t = frame + offset if params.get("reverse") else frame - offset
    if params.get("reverse_input"):
        lo, hi = clip
        t = lo + hi - t
    return _one(_rounded(t, params.get("rounding_mode", "Whole frames")))


def frame_hold_plan(params, frame, clip):
    first = float(params.get("first_frame", 1))
    increment = float(params.get("increment", 0))
    if increment <= 0 or frame <= first:
        t = first
    else:
        t = first + math.floor((frame - first) / increment + 1e-9) * increment
    return _one(_rounded(t, params.get("rounding_mode", "Whole frames")))


def retime_numbers(params, clip):
    """(input_first, input_last, output_first, output_last, speed) with the locks applied."""
    in_lo, in_hi = clip
    in_first = float(params["input_range_start"]) if params.get("input_range_start_lock", 1) else float(in_lo)
    in_last = float(params["input_range_end"]) if params.get("input_range_end_lock", 0) else float(in_hi)
    length = in_last - in_first
    speed = float(params.get("speed", 1.0))
    out_first_locked = bool(params.get("output_range_start_lock", 1))
    out_last_locked = bool(params.get("output_range_end_lock", 0))
    if out_first_locked and out_last_locked:
        out_first, out_last = float(params["output_range_start"]), float(params["output_range_end"])
        if out_last != out_first:
            speed = length / (out_last - out_first)
    else:
        span = length / speed if speed else 0.0
        if out_last_locked:
            out_last = float(params["output_range_end"])
            out_first = out_last - span
        else:
            out_first = float(params["output_range_start"]) if out_first_locked else in_first
            out_last = out_first + span
    return in_first, in_last, out_first, out_last, speed


def retime_plan(params, frame, clip):
    in_first, in_last, out_first, _out_last, speed = retime_numbers(params, clip)
    reverse = bool(params.get("reverse"))
    lo, hi = min(in_first, in_last), max(in_first, in_last)
    # The policy follows the output side (frames before output.first, frames after it), whichever way the input runs.
    policy = params.get("retime_before", "hold") if frame < out_first else params.get("retime_after", "hold")

    def settle(x):
        if lo <= x <= hi:
            return x, False
        return _apply_policy(x, lo, hi, policy)

    step = (frame - out_first) * speed
    x = in_last - step if reverse else in_first + step
    mode = str(params.get("retime_filter", "box"))
    width = min(abs(speed) * float(params.get("shutter", 1.0)), MAX_BOX_FRAMES)
    if mode == "box" and width > 0:
        low, high = x - width / 2.0, x + width / 2.0
        weights = {}
        for cell in range(int(math.floor(low + 0.5 - 1e-9)), int(math.floor(high - 0.5 + 1e-9)) + 2):
            overlap = min(high, cell + 0.5) - max(low, cell - 0.5)
            if overlap > 1e-9:
                mapped, black = settle(float(cell))
                if not black:
                    key = int(round(mapped))
                    weights[key] = weights.get(key, 0.0) + overlap / width
        nearest, _ = settle(x)
        return TimePlan([(k, round(w, 9)) for k, w in sorted(weights.items())], int(round(nearest)))
    mapped, black = settle(x)
    if mode == "none":
        return TimePlan([] if black else [(_tidy(mapped), 1.0)], _tidy(mapped))
    whole = int(round(mapped))
    return TimePlan([] if black else [(whole, 1.0)], whole)


def clip_bounds(params, clip=None):
    """(first, last) of a TimeClip after the original range cuts it, or None when it plays every frame."""
    if params.get("frame_range_type", "custom") == "all":
        return None
    first, last = int(params.get("first", 1)), int(params.get("last", 100))
    lo, hi = min(first, last), max(first, last)
    orig_first, orig_last = int(params.get("origfirst", 0)), int(params.get("origlast", 0))
    if not (orig_first == 0 and orig_last == 0):
        olo, ohi = min(orig_first, orig_last), max(orig_first, orig_last)
        lo, hi = max(lo, olo), min(hi, ohi)
        if lo > hi:
            lo = hi = min(max(first, olo), ohi)
    return lo, hi


def time_clip_plan(params, frame, clip):
    mode = str(params.get("frame_mode", "expression"))
    text = params.get("frame_expr", "")
    bounds = clip_bounds(params)
    if mode == "start at":
        start = _expression(text, frame) if str(text).strip() else float(frame)
        origin = bounds[0] if bounds is not None else float(clip[0])
        value = frame - start + origin
    elif mode == "offset":
        value = frame + (_expression(text, frame) if str(text).strip() else 0.0)
    else:
        value = _expression(text, frame)
    t = _floor(value + 0.5) - int(params.get("time_offset", 0))
    if bounds is None:
        return _one(t, gain=1.0)
    lo, hi = bounds
    mapped, black, outside = _range_frame(t, lo, hi, params.get("before", "hold"), params.get("after", "hold"))
    mapped = int(mapped)
    gain = 1.0
    if not outside:
        fade_in, fade_out = float(params.get("fade_in", 0)), float(params.get("fade_out", 0))
        position, count = t - lo, hi - lo + 1
        if fade_in > 0 and position < fade_in:
            gain *= (position + 1) / (fade_in + 1)
        if fade_out > 0 and count - position <= fade_out:
            gain *= (count - position) / (fade_out + 1)
    if params.get("reverse"):
        mapped = lo + hi - mapped
    if black:
        return TimePlan([], mapped, gain)
    return TimePlan([(mapped, 1.0)], mapped, gain)


def plan(kind, params, frame, clip):
    """The `TimePlan` for one of KINDS at timeline `frame`. `clip` is the input clip's (first, last), the timeline's when
    nothing upstream states a range."""
    if kind == "TimeOffset":
        return time_offset_plan(params, frame, clip)
    if kind == "FrameHold":
        return frame_hold_plan(params, frame, clip)
    if kind == "Retime":
        return retime_plan(params, frame, clip)
    if kind == "TimeClip":
        return time_clip_plan(params, frame, clip)
    raise ValueError(f"{kind} is not a retiming kind")
