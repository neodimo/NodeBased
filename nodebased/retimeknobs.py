"""Kronos and OFlow's timing knobs as Nuke 17.0 states them (lane 2, 2D behaviour step J).

Both nodes turn the timeline frame into the source frame to show, then into a picture by the Method knob. The frame comes from `Timing`
(`timing2`):

* `speed` (Output Speed, `timingOutputSpeed`): "double speed halves the duration of the clip and half speed doubles the duration", so output
  frame `f` shows source frame `input_start + (f - output_start) * speed`. The guide does not say which output frame shows the first input
  frame; the reading used is that the output starts where the input range starts (OFlow's NodeBased-only `output_start` moves it, Kronos
  has none). Negative speeds are allowed, and run the clip backwards.
* `input_speed` (Input Speed, `timingInputSpeed`): the same frame for a constant speed, but "do not allow you to use negative speed values",
  so a negative value is read as 0 (the clip holds on its first frame).
* `frame` (`timingFrame2`): "the source frame at the current frame in the timeline", the knob's own value at that frame (a curve sets it).

The picked frame is limited to the input range `[input_start, input_end]` ("the first / last frame of input to use"), which in NodeBased is
also limited to the project's frame range. What is shown outside it (hold) is our reading: the guide does not say.

`interpolation` (`method` on OFlow): "Frame - the nearest original frame is displayed" (a tie, exactly half way, goes to the even frame as
Retime's nearest always has: the guide does not break ties), "Blend - a mix between two frames is used for the in-between frame" (the two
neighbouring frames, weighted by how close each is: a cross-dissolve), "Motion - vector interpolation".

`retimed_channels` (`retimedChannels`): "the channels affected by the retime". The others are the source's own at the current frame.

Legacy (`legacy_retime`): Kronos used to take `frame` as the source frame when it was 0 or more, else the timeline frame times `speed`
from zero (not from the input range), limited to the project's range only; its `interpolation` "frame" meant a blend, and it had no input
range, input speed or channels. A node saved before this step keeps all of that. OFlow's frame mapping was already the one above; its
new knobs default to what it always did (motion, every channel).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

KINDS = ("Kronos", "OFlow")
TIMINGS = ("speed", "input_speed", "frame")
METHODS = ("frame", "blend", "motion")
FLAG = "legacy_retime"

NEW = {
    "Kronos": {"input_start": 1.0, "input_end": 100.0, "input_speed": 1.0, "retimed_channels": "all", FLAG: 0, "timing": "speed"},
    "OFlow": {"input_speed": 1.0, "retimed_channels": "all", "interpolation": "motion"},
}
# What a node saved before this step gets: Kronos's old rule, flagged. A missing flag reads as legacy.
OLD = {
    "Kronos": {"input_start": 1.0, "input_end": 100.0, "input_speed": 1.0, "retimed_channels": "all", FLAG: 1, "timing": "speed", "frame": -1.0},
    "OFlow": {"input_speed": 1.0, "retimed_channels": "all", "interpolation": "motion"},
}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(OLD.get(kind, {}))


def is_legacy(kind, params):
    """True for a Kronos saved before this step (a missing flag reads as legacy); OFlow's frame mapping never changed."""
    return kind == "Kronos" and bool(int(params.get(FLAG, 1)))


@dataclass
class Retime:
    target: float          # the source frame to show, inside [lo, hi]
    lo: int
    hi: int
    method: str            # "frame", "blend" or "motion"


def nearest_frame(value):
    """The whole frame nearest `value`, a tie going to the even frame."""
    return float(round(value))


def plan(kind, params, frame, first, last):
    """The source frame, the usable frame range and the method for output frame `frame`, with the project's frames `first` to `last`."""
    frame = float(frame)
    if is_legacy(kind, params):
        target = float(params["frame"]) if float(params["frame"]) >= 0 else frame * float(params["speed"])
        lo, hi = int(first), int(last)
        method = params["interpolation"] if params["interpolation"] in ("frame", "motion") else "motion"
        return Retime(min(float(hi), max(float(lo), target)), lo, hi, "blend" if method == "frame" else method)
    start = float(params["input_start"])
    timing = params.get("timing", "speed")
    if timing == "frame":
        target = float(params["frame"])
    else:
        speed = float(params["input_speed"]) if timing == "input_speed" else float(params["speed"])
        if timing == "input_speed":
            speed = max(0.0, speed)
        origin = float(params["output_start"]) if "output_start" in params else start
        target = start + (frame - origin) * speed
    lo = max(int(math.floor(start)), int(first))
    hi = min(int(math.ceil(float(params["input_end"]))), int(last))
    if hi < lo:
        raise ValueError(f"{kind}: input_end must be at or after input_start")
    method = params.get("interpolation", "motion")
    target = min(float(hi), max(float(lo), target))
    if method == "frame":
        target = min(float(hi), max(float(lo), nearest_frame(target)))
    return Retime(target, lo, hi, method if method in METHODS else "motion")


def blend(a, b, alpha):
    """The cross-dissolve of two neighbouring frames: `alpha` of the way from `a` to `b`."""
    return a * (1.0 - alpha) + b * alpha


def keep_unretimed(pixels, source, channels):
    """`pixels` (the retimed picture) with the channels `retimed_channels` leaves out replaced by `source`'s (the input at the current frame).
    Both are (h, w, 4) arrays on the same window."""
    from .ops2d_chrome import parse_channels
    chosen = parse_channels(channels).rgba
    if all(chosen):
        return pixels
    out = source.copy()
    for index, take in enumerate(chosen):
        if take:
            out[..., index] = pixels[..., index]
    return out
