"""Nuke's Read-node knobs that change what is read: the time mapping (`frame_mode` / `frame_expr`, `first` / `last` with
`before` / `after`, `origfirst` / `origlast`), `read_format` (Nuke's `format`), `raw`, `premultiplied` and `auto_alpha`.

Every default reproduces what Read did before the knob existed, so `active(params)` is False for an old document and the
partial-region reads and cached decodes it relied on are untouched. A Read with any of these changed is decoded whole
and the region cut from it, the same precedent as the layer-bearing EXR.

The time mapping, in Nuke's order:

    timeline frame -> the `frame` knob (expression: the value of the expression; start at N: the clip's first frame shows at N;
                      offset: the frame plus N) -> the legacy `frame_offset` -> the clip range [first, last] with `before` /
                      `after` (hold, loop, bounce, black) -> the original range [origfirst, origlast], where a frame outside
                      it is missing and `missing` (Nuke's on_error) decides.

Pure Python, no NumPy: `tileexec` fingerprints and `core.validate` call it.
"""
from __future__ import annotations

import math
import re

from . import pathcontext

DEFAULTS = {"read_format": "file", "first": 0, "last": 0, "before": "hold", "after": "hold", "frame_mode": "expression",
            "frame_expr": "", "origfirst": 0, "origlast": 0, "raw": 0, "premultiplied": 0, "auto_alpha": 1,
            # Fidelity step F: localization (nodebased/localize.py), the reload counter, the proxy (nodebased/readproxy.py) and
            # the context variables (nodebased/pathcontext.py).
            "localizationPolicy": "from auto-localize path", "reload": 0, "proxy": "", "proxy_format": "file", **pathcontext.PARAMS}
KEYS = tuple(DEFAULTS)
RANGE_BEHAVIOURS = ("hold", "loop", "bounce", "black")
FRAME_MODES = ("expression", "start at", "offset")
_SIZE = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$")


def extras(params):
    """The Nuke Read knobs present in `params`, with defaults for any that are missing."""
    return {key: params.get(key, DEFAULTS[key]) for key in KEYS}


def active(params):
    """True when any of these knobs differs from the value that leaves Read as it was."""
    from . import localize
    return any(params.get(key, default) != default for key, default in DEFAULTS.items()) or localize.applies(params)


def time_active(params):
    return any(params.get(key, DEFAULTS[key]) != DEFAULTS[key]
               for key in ("first", "last", "frame_mode", "frame_expr", "origfirst", "origlast"))


def _range(first, last):
    first, last = int(first), int(last)
    return None if (first == 0 and last == 0) else (min(first, last), max(first, last))


def _evaluate_frame(text, frame):
    from .expressions import evaluate
    text = str(text).strip()
    if not text:
        return float(frame)

    def lookup(reference):
        raise ValueError("Read: the frame expression can use `frame` and arithmetic, not other knobs")
    return evaluate(text, frame, lookup)


def mapped_frame(params, frame, sequence_first=None):
    """(source frame, black). `black` is True when `before` / `after` is black and the frame falls outside the clip range.

    `sequence_first` is the first frame on disk, which `start at` needs when `first` is not set."""
    mode = str(params.get("frame_mode", "expression"))
    text = params.get("frame_expr", "")
    if mode == "expression":
        value = _evaluate_frame(text, frame)
    elif mode == "start at":
        start = _evaluate_frame(text, frame) if str(text).strip() else float(frame)
        clip = _range(params.get("first", 0), params.get("last", 0))
        origin = clip[0] if clip is not None else (sequence_first if sequence_first is not None else 0)
        value = frame - start + origin
    elif mode == "offset":
        value = frame + (_evaluate_frame(text, frame) if str(text).strip() else 0.0)
    else:
        raise ValueError(f"Read: unknown frame mode {mode!r}")
    source = int(math.floor(value + int(params.get("frame_offset", 0)) + 0.5))
    clip = _range(params.get("first", 0), params.get("last", 0))
    if clip is None:
        return source, False
    lo, hi = clip
    count = hi - lo + 1
    if lo <= source <= hi:
        return source, False
    behaviour = str(params.get("before" if source < lo else "after", "hold"))
    if behaviour == "hold":
        return (lo if source < lo else hi), False
    if behaviour == "black":
        return source, True
    if behaviour == "loop":
        return lo + (source - lo) % count, False
    if behaviour == "bounce":
        if count == 1:
            return lo, False
        period = 2 * (count - 1)
        index = (source - lo) % period
        return lo + (index if index < count else period - index), False
    raise ValueError(f"Read: unknown range behaviour {behaviour!r}")


def outside_original(params, source):
    """True when `source` lies outside the original range: the file is then treated as missing."""
    original = _range(params.get("origfirst", 0), params.get("origlast", 0))
    return original is not None and not (original[0] <= source <= original[1])


def clamp_to_original(params, source):
    original = _range(params.get("origfirst", 0), params.get("origlast", 0))
    return source if original is None else min(max(source, original[0]), original[1])


def resolve_policy(params, frame, sequence_first=None):
    """(source frame, missing policy, forced) for the file lookup.

    `forced` is True when the frame is black or a checkerboard whether or not a file exists for it: `before` / `after` black
    outside the clip range, or a frame outside the original range under the black / checkerboard `missing` policies. A frame
    outside the original range under `hold` is clamped into it; under `error` it raises."""
    source, black = mapped_frame(params, frame, sequence_first)
    policy = str(params.get("missing", "error"))
    if black:
        return source, "black", True
    if outside_original(params, source):
        if policy == "hold":
            return clamp_to_original(params, source), "hold", False
        if policy in ("black", "checkerboard"):
            return source, policy, True
        raise ValueError(f"Read: frame {source} is outside the original range "
                         f"{int(params.get('origfirst', 0))}-{int(params.get('origlast', 0))}")
    return source, policy, False


def parse_format(text, presets):
    """(width, height) for a Read `format`: 'file' is None, a preset name, or 'WIDTHxHEIGHT'."""
    text = str(text).strip()
    if text in ("", "file"):
        return None
    if text in presets:
        entry = presets[text]
        width, height = (entry[0], entry[1]) if isinstance(entry, (tuple, list)) else (entry["width"], entry["height"])
        return int(width), int(height)
    match = _SIZE.match(text)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Read: format {text!r} is neither 'file', a format name nor WIDTHxHEIGHT")
