"""Animation curves for time-varying numeric node parameters.

Independently designed from Qt and from the renderer: a curve is just a list of keys plus an
interpolation choice. The evaluator asks ``resolve_params`` for the per-frame resolved params
of a node, and that is the only contract this module exposes to the rest of the engine. The
document's stored ``node["params"]`` is never mutated by an animation edit.

Scope of this MVP:
    * Numeric parameters only (float or int defaults; bool excluded).
    * Two interpolations: ``constant`` (step-and-hold, value held until the next key) and
      ``linear`` (lerp between the bracketing keys).
    * Out-of-range frames fall back to the node's stored parameter value. The curve is a
      *patch* over a static base, not a replacement. This keeps "no animation" the
      cheap default and lets the existing graph render byte-identically when no curves exist.
    * Int parameters round their resolved value at evaluation time so a key with float=1.6
      evaluates to 2 for an int param. Range is clamped to LIMITS defensively.

What this module deliberately does NOT include (and why):
    * Bezier tangents, expressions, clips/tracks, playback, dopesheet — each is a follow-on
      that fits the same ``(node_id, param) -> curve`` slot. ``docs/ANIMATION.md`` documents
      how those will plug in without breaking this layer.
"""
from __future__ import annotations

import bisect
import math


# Stable, ordered list. Documented to agents in `describe` and asserted by validate().
CURVE_INTERPOLATIONS = ("constant", "linear")
DEFAULT_INTERPOLATION = "linear"

# Frame number is an integer; the range is wide enough to admit pre-rolls and tests, not so
# wide it lets a typo render everything frozen at +1e9.
FRAME_LIMITS = (-1_000_000, 1_000_000)


class CurveError(ValueError):
    """Raised for any structural problem with a curve dict, a key, or a curve edit."""


def validate_curve(curve):
    """Validate a single curve dict in isolation. Raises CurveError on any rule violation.

    The shape is exactly:
        {"interpolation": "constant" | "linear",
         "keys": [{"frame": int, "value": number}, ...]}
    Keys must be strictly increasing in frame and contain no duplicates. Values must be finite.
    Frames must fall inside ``FRAME_LIMITS``. The list must be non-empty (an "empty curve" is
    expressed by simply omitting it from the document).
    """
    if not isinstance(curve, dict):
        raise CurveError("curve must be an object")
    if set(curve) != {"interpolation", "keys"}:
        raise CurveError("curve must define exactly 'interpolation' and 'keys'")
    interp = curve["interpolation"]
    if interp not in CURVE_INTERPOLATIONS:
        raise CurveError(f"interpolation must be one of {list(CURVE_INTERPOLATIONS)}")
    keys = curve["keys"]
    if not isinstance(keys, list) or not keys:
        raise CurveError("keys must be a non-empty list")
    prev_frame = None
    for i, k in enumerate(keys):
        if not isinstance(k, dict) or set(k) != {"frame", "value"}:
            raise CurveError(f"key[{i}] must define exactly 'frame' and 'value'")
        f = k["frame"]
        v = k["value"]
        if type(f) is not int or isinstance(f, bool):
            raise CurveError(f"key[{i}].frame must be an integer (got {type(f).__name__})")
        lo, hi = FRAME_LIMITS
        if f < lo or f > hi:
            raise CurveError(f"key[{i}].frame={f} out of frame range [{lo}, {hi}]")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise CurveError(f"key[{i}].value must be a number (got {type(v).__name__})")
        if not math.isfinite(v):
            raise CurveError(f"key[{i}].value must be a finite number")
        if prev_frame is not None and f <= prev_frame:
            raise CurveError(f"key frames must be strictly increasing (got {prev_frame} then {f})")
        prev_frame = f


def evaluate_curve(curve, frame):
    """Resolve the curve at ``frame``. Returns the numeric value, or ``None`` to mean
    "the curve does not define this frame — caller should use the base parameter".

    Semantics:
        * frame < first frame                          -> None (before curve, base applies)
        * frame > last frame                           -> None (after curve, base applies)
        * frame == key.frame                           -> that key's value
        * constant interpolation, between keys         -> the previous key's value (step-and-hold)
        * linear interpolation, between keys           -> linear interpolation
    """
    frames = [k["frame"] for k in curve["keys"]]
    if frame < frames[0] or frame > frames[-1]:
        return None
    values = [k["value"] for k in curve["keys"]]
    idx = bisect.bisect_right(frames, frame) - 1
    if curve["interpolation"] == "constant":
        return values[idx]
    if curve["interpolation"] == "linear":
        if idx < 0 or frames[idx] == frame:
            return values[idx]
        k0f, k1f = frames[idx], frames[idx + 1]
        t = (frame - k0f) / (k1f - k0f)
        return values[idx] * (1.0 - t) + values[idx + 1] * t
    raise CurveError(f"Unknown interpolation: {curve['interpolation']}")


def coerce_value_for_param(value, param_spec_default, limits):
    """Match ``value`` to the parameter's expected numeric type. Int params round; both clamp.

    This is the only place the int/float distinction crosses module boundaries. ``limits`` is
    ``(lo, hi)`` from the project's LIMITS dict; passing ``None`` skips the clamp. The return
    type matches the spec default's numeric type (so callers can drop it into a params dict
    without surprising the validator).
    """
    if param_spec_default is None:
        return value
    if isinstance(param_spec_default, int) and not isinstance(param_spec_default, bool):
        value = int(round(float(value)))
    else:
        value = float(value)
    if limits is not None:
        lo, hi = limits
        if value < lo:
            value = lo
        elif value > hi:
            value = hi
    return value


def resolve_params(node, animation_section, frame, spec_params, limits):
    """Compute the per-frame resolved params dict for ``node``.

    Inputs:
        * ``node``                   -- the graph node dict (read-only).
        * ``animation_section``      -- either ``None`` or ``doc["animation"]["curves"][node_id]``;
                                        a dict mapping ``param_name -> curve`` for this node.
        * ``frame``                  -- int timeline frame to resolve at.
        * ``spec_params``            -- ``SPECS[kind]["params"]`` for type information.
        * ``limits``                 -- the project's LIMITS dict for clamping.

    Returns a shallow copy of ``node["params"]`` with per-param overrides applied where a
    curve defines the frame. Never mutates the input.
    """
    resolved = dict(node["params"])
    if not animation_section:
        return resolved
    for param, curve in animation_section.items():
        if param not in node["params"]:
            continue
        spec_default = spec_params.get(param)
        # Only numeric spec defaults are eligible for animation (bool is intentionally not).
        if not isinstance(spec_default, (int, float)) or isinstance(spec_default, bool):
            continue
        value = evaluate_curve(curve, int(frame))
        if value is None:
            continue
        resolved[param] = coerce_value_for_param(value, spec_default, limits.get(param))
    return resolved


def merge_key(curve, frame, value):
    """Return a new curve dict with the key at ``frame`` set/replaced and the key list kept
    strictly sorted. Does not mutate the input. Validation is the caller's responsibility."""
    new_keys = [dict(k) for k in curve["keys"] if k["frame"] != frame]
    new_keys.append({"frame": int(frame), "value": float(value)})
    new_keys.sort(key=lambda k: k["frame"])
    return {"interpolation": curve["interpolation"], "keys": new_keys}


def drop_key(curve, frame):
    """Return a new curve dict without the key at ``frame``. Returns ``None`` if the resulting
    key list would be empty (so the caller can delete the curve entry from the document).
    Raises CurveError if no key exists at ``frame``."""
    new_keys = [dict(k) for k in curve["keys"] if k["frame"] != frame]
    if len(new_keys) == len(curve["keys"]):
        raise CurveError(f"No key at frame {frame}")
    if not new_keys:
        return None
    return {"interpolation": curve["interpolation"], "keys": new_keys}
