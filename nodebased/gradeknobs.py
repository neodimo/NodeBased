"""Nuke's knobs on Grade (fidelity step E): the levels, gamma, reverse and the two clamps.

Nuke's Grade is a linear ramp followed by a gamma and the clamps. With `A` the slope and `B` the shift:

    A   = multiply * (white - black) / (whitepoint - blackpoint)
    B   = add + black - A * blackpoint
    out = A * in + B
    out = out ** (1 / gamma)            where out > 0
    out = max(out, 0)                   when black clamp is on
    out = min(out, 1)                   when white clamp is on

`reverse` runs it backwards: the opposite gamma (`out ** gamma` where out > 0), then the opposite ramp (`(out - B) / A`), then
the clamps. The Nuke guide states the knobs but no formula; this is Nuke's long-standing form (marked *convention* in
docs/PARITY_2D_FIDELITY.md).

NodeBased's Grade has always had `exposure`, `multiply` and `offset`. They keep their meaning: `exposure` is a stop count, so it
multiplies the slope by `2 ** exposure`; `multiply` is Nuke's multiply; `offset` is Nuke's `add`, except that NodeBased's
offset has always been scaled by the pixel's alpha (`offset_by_alpha`, on in an older document and off in a new node, where
the offset is Nuke's flat one). Nuke's `black` and `white` are the knobs its panel labels lift and gain.

Pure NumPy, so the evaluator, the tile executor and the GPU island's mapping read the same numbers.
"""
from __future__ import annotations

import numpy as np

# What a new node gets (Nuke's defaults) and what an older document's node gets, which is what that node did before the knob
# existed. Where they differ the old value is a NodeBased-only choice: no clamp, and an offset that follows alpha.
NEW = {"blackpoint": 0.0, "whitepoint": 1.0, "black": 0.0, "white": 1.0, "gamma": 1.0, "reverse": 0, "black_clamp": 1,
       "white_clamp": 0, "offset_by_alpha": 0}
OLD = dict(NEW, black_clamp=0, offset_by_alpha=1)
KINDS = ("Grade",)
SPAN_FLOOR = 1e-12            # a whitepoint equal to the blackpoint would divide by zero: the span is held at this size


def new_params(kind):
    return dict(NEW) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(OLD) if kind in KINDS else {}


def _get(p, name):
    """A knob of `p`; a parameter dict that predates the knob reads as an older document's node (the OLD value)."""
    return p.get(name, OLD.get(name, 0.0))


def is_legacy(p):
    """True when every step-E knob is at the value that leaves the node's old arithmetic untouched (the old path then runs
    byte for byte)."""
    return (float(_get(p, "blackpoint")) == 0.0 and float(_get(p, "whitepoint")) == 1.0
            and float(_get(p, "black")) == 0.0 and float(_get(p, "white")) == 1.0
            and float(_get(p, "gamma")) == 1.0 and not _get(p, "reverse")
            and not _get(p, "black_clamp") and not _get(p, "white_clamp")
            and (bool(_get(p, "offset_by_alpha")) or float(p.get("offset", 0.0)) == 0.0))


def terms(p):
    """`(slope, shift)` of the linear ramp as plain floats: `shift` is the part that does not depend on alpha. The slope carries
    `exposure` and `multiply`; `offset` joins the shift unless it follows alpha (`offset_by_alpha`)."""
    span = float(_get(p, "whitepoint")) - float(_get(p, "blackpoint"))
    if abs(span) < SPAN_FLOOR:
        span = SPAN_FLOOR if span >= 0 else -SPAN_FLOOR
    slope = (2.0 ** float(p.get("exposure", 0.0))) * float(p.get("multiply", 1.0)) * (
        float(_get(p, "white")) - float(_get(p, "black"))) / span
    flat_offset = 0.0 if _get(p, "offset_by_alpha") else float(p.get("offset", 0.0))
    shift = flat_offset + float(_get(p, "black")) - slope * float(_get(p, "blackpoint"))
    return slope, shift


def alpha_offset(p):
    """The offset that is scaled by alpha (0 for a node whose offset is Nuke's flat one)."""
    return float(p.get("offset", 0.0)) if _get(p, "offset_by_alpha") else 0.0


def grade_rgb(rgb, alpha, p):
    """Grade applied to an (h, w, 3) float array, `alpha` its (h, w, 1) alpha. Returns float32."""
    slope, shift = terms(p)
    follow = alpha_offset(p)
    gamma = float(_get(p, "gamma"))
    black_clamp, white_clamp = bool(_get(p, "black_clamp")), bool(_get(p, "white_clamp"))
    shifted = np.float32(shift)
    if _get(p, "reverse"):
        # Undo the gamma first, then the ramp: x = ((y ** gamma) - B) / A.
        y = rgb.astype(np.float32)
        if gamma != 1.0:
            y = np.where(y > 0, np.power(np.maximum(y, 0), np.float32(gamma)), y).astype(np.float32)
        if follow != 0.0:
            y = y - alpha * np.float32(follow)
        if abs(slope) < SPAN_FLOOR:
            out = y                        # a flat ramp cannot be undone: the colour stays as it is
        else:
            out = ((y - shifted) / np.float32(slope)).astype(np.float32)
    else:
        out = rgb * np.float32(slope) + shifted
        if follow != 0.0:
            out = out + alpha * np.float32(follow)
        if gamma != 1.0:
            out = np.where(out > 0, np.power(np.maximum(out, 0), np.float32(1.0 / gamma)), out)
    if black_clamp:
        out = np.maximum(out, np.float32(0.0))
    if white_clamp:
        out = np.minimum(out, np.float32(1.0))
    return out.astype(np.float32)


def grade(image, p):
    """Grade on an (h, w, 4) float array: colour graded, alpha kept."""
    out = np.empty_like(image, dtype=np.float32)
    out[..., :3] = grade_rgb(image[..., :3], image[..., 3:4], p)
    out[..., 3] = image[..., 3]
    return out
