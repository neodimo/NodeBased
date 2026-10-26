"""Nuke's AddMix knobs of fidelity step H: the channels that take part (`Achannels`, `Bchannels`, `merge_output`; Nuke's `A`, `B`,
`output`), `premultiplied`, `nonlinear` and the two lookup curves (`lut_a`, `lut_b`; Nuke's `lut`).

Nuke: the alpha of input A indexes two lookup curves; the first multiplies input A, the second input B, and the results are added.
With the curves at their defaults, A's curve rising from 0 to 1 and B's falling from 1 to 0, that is a premultiplied A over B, which
is what NodeBased's AddMix always drew. In NodeBased's terms, per pixel with `i` the lookup index:

    out colour = A colour * lut_a(i) + B colour * lut_b(i)
    out alpha  = lut_a(i)             + B alpha  * lut_b(i)

(A's alpha counts as a colour of 1, so the default gives `a + b (1 - a)`.)

* `premultiplied`: A's colour is divided by A's alpha (zero where it is zero) before the curve multiplies it.
* `nonlinear`: the index is A's alpha remapped from the range zero to infinity into zero to one, `ln(1 + a) / (1 + ln(1 + a))`. The
  guide gives no formula; this logarithmic remap is NodeBased's reading.
* `lut_a`, `lut_b`: a curve as `x:y` points separated by spaces, commas or semicolons, joined by straight lines and held flat beyond
  the first and last. The defaults are `0:0 1:1` and `0:1 1:0`. A curve with fewer than two points or a point that is not a number
  is refused with a message naming the knob.
* `Achannels`, `Bchannels`, `merge_output`: which of R, G, B and A are read from each input (an unselected channel counts as zero)
  and which are written (the others are B's). A layer name is refused: only the colour channels and alpha take part.

The mask and mix gate the result against B as they always did. With every knob at its default the old arithmetic runs unchanged.
"""
from __future__ import annotations

import re

import numpy as np

PARAMS = {"Achannels": "rgba", "Bchannels": "rgba", "merge_output": "rgba", "premultiplied": 0, "nonlinear": 0,
          "lut_a": "0:0 1:1", "lut_b": "0:1 1:0"}
KINDS = ("AddMix",)


def new_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def is_default(p):
    """True when every knob is at its default, so the original premultiplied-over arithmetic applies."""
    return all(_norm(p.get(name, default)) == _norm(default) for name, default in PARAMS.items())


def _norm(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_curve(text, name):
    """The points of a curve as sorted (x, y) arrays."""
    points = []
    for token in re.split(r"[\s,;]+", str(text).strip()):
        if not token:
            continue
        match = re.fullmatch(r"([-+0-9.eE]+):([-+0-9.eE]+)", token)
        try:
            if not match:
                raise ValueError
            points.append((float(match.group(1)), float(match.group(2))))
        except ValueError:
            raise ValueError(f"AddMix: {name} holds x:y points, not {token!r}") from None
    if len(points) < 2:
        raise ValueError(f"AddMix: {name} needs at least two x:y points")
    points.sort()
    xs = np.array([x for x, _ in points], np.float32)
    ys = np.array([y for _, y in points], np.float32)
    if np.any(np.diff(xs) <= 0):
        raise ValueError(f"AddMix: the points of {name} need different x values")
    return xs, ys


def flags(text, name):
    """R, G, B, A flags of a channel set; a layer name is refused."""
    from .ops2d_chrome import parse_channels
    chosen = parse_channels(text)
    if chosen.layers:
        raise ValueError(f"AddMix: {name} takes R, G, B and A, not the layer {chosen.layers[0]!r}")
    return np.array(chosen.rgba, np.float32)


def lookup_index(alpha, nonlinear):
    if not nonlinear:
        return alpha
    log = np.log1p(np.maximum(alpha, 0.0))
    return (log / (1.0 + log)).astype(np.float32)


def combine(p, a, b):
    """The merged picture of A and B (both (h, w, 4)) before the mask and mix."""
    read_a, read_b = flags(p.get("Achannels", "rgba"), "Achannels"), flags(p.get("Bchannels", "rgba"), "Bchannels")
    write = flags(p.get("merge_output", "rgba"), "merge_output").astype(bool)
    a, b = a * read_a, b * read_b
    xs_a, ys_a = parse_curve(p.get("lut_a", PARAMS["lut_a"]), "lut_a")
    xs_b, ys_b = parse_curve(p.get("lut_b", PARAMS["lut_b"]), "lut_b")
    alpha = a[..., 3:4]
    index = lookup_index(alpha, bool(int(p.get("nonlinear", 0))))
    weight_a = np.interp(index, xs_a, ys_a).astype(np.float32)
    weight_b = np.interp(index, xs_b, ys_b).astype(np.float32)
    colour = a[..., :3]
    if int(p.get("premultiplied", 0)):
        colour = np.where(alpha != 0.0, colour / np.where(alpha != 0.0, alpha, 1.0), 0.0).astype(np.float32)
    merged = np.empty_like(b)
    merged[..., :3] = colour * weight_a + b[..., :3] * weight_b
    merged[..., 3:4] = weight_a + b[..., 3:4] * weight_b
    out = b.copy()
    out[..., write] = merged[..., write]
    return out.astype(np.float32)


def gate(b, merged, mask, mix):
    """Dissolve between B and the merged picture by `mix` and the mask's alpha."""
    if mask is not None and mask.shape[:2] != b.shape[:2]:
        raise ValueError("Merge mask must match the merged format")
    weight = (np.float32(mix) if mask is None else mask[..., 3:4] if mix == 1.0 else np.float32(mix) * mask[..., 3:4])
    return (merged * weight + b * (1.0 - weight)).astype(np.float32)
