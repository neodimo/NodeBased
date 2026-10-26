"""Nuke's knobs on Merge (fidelity step E): the video colorspace, alpha masking, the output bounding box, the channels that
take part, and metadata and range from.

Nuke's Merge reads the channels named by `Achannels` and `Bchannels` (a channel left out counts as zero), combines them
with `operation`, and writes the result to the channels named by `output`; the other channels are B's, untouched. `also_merge`
adds layers that are merged the same way. Around that:

* `sRGB` (Nuke's "video colorspace"): the colour channels go to the video colorspace before the operation and come back to
  linear after it. NodeBased uses the sRGB curve, mirrored about zero so negative and HDR values keep their sign and order.
* `screen_alpha` ("alpha masking"): where one image has no alpha the other is passed through unchanged, and the output alpha is
  `a + b - a*b`. It changes nothing on the operations that already combine alpha that way (over, under, atop, in, out, mask,
  stencil, xor, matte, conjoint-over, disjoint-over, copy), which is the guide's "disabled where it makes no difference".
* `merge_bbox` (Nuke's `bbox`, "set bbox to"): the output's data window is the union of the two inputs', their intersection,
  A's or B's. Data outside it is cut off.
* `metainput` and `rangeinput`: which input's metadata and frame range the node passes on (B, A or, for All, both), as on Copy.

The mask and mix gate the whole result against B, as they always did. Every default draws what Merge did before the knob existed.
Pure NumPy and the shared channel parser, so the evaluator, the tile executor and the tests read one arithmetic.
"""
from __future__ import annotations

import numpy as np

KINDS = ("Merge",)
BBOX_CHOICES = ("union", "intersection", "A", "B")
# Operations whose alpha already combines as a + b - a*b or is the operation's own business: `screen_alpha` leaves them alone.
SCREEN_ALPHA_SKIPS = frozenset({"over", "under", "atop", "in", "out", "mask", "stencil", "xor", "matte", "conjoint-over",
                                "disjoint-over", "copy"})
# What a new node gets and what an older document's node gets (they are the same: every one of these leaves the old Merge as it was).
NEW = {"sRGB": 0, "screen_alpha": 0, "merge_bbox": "union", "metainput": "B", "rangeinput": "B", "Achannels": "rgba",
       "Bchannels": "rgba", "merge_output": "rgba", "also_merge": "none"}
OLD = dict(NEW)
ARRAY_KNOBS = ("sRGB", "screen_alpha", "Achannels", "Bchannels", "merge_output")


def new_params(kind):
    return dict(NEW) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(OLD) if kind in KINDS else {}


# ----------------------------------------------------------------------------------------------------- the channel sets
def _set(text):
    from .ops2d_chrome import parse_channels
    return parse_channels(text)


def channel_flags(text):
    """The R, G, B, A flags a `Achannels` / `Bchannels` / `output` value turns on (a layer name turns none of them on)."""
    return _set(text).rgba


def layer_names(text):
    return list(_set(text).layers)


def plain_channels(p):
    """True when Achannels, Bchannels and output are plain rgba, so the node needs no layer to do its job."""
    return all(str(p.get(name, "rgba")).strip().lower() in ("rgba", "all") for name in ("Achannels", "Bchannels", "merge_output"))


def array_default(p):
    """True when no array-level knob is changed: the old arithmetic then runs byte for byte."""
    return (not p.get("sRGB", 0) and not p.get("screen_alpha", 0) and plain_channels(p))


def uses_layers(p):
    """True when a channel set or `also_merge` names a layer, which only the whole-image path can read."""
    if any(layer_names(p.get(name, "rgba")) for name in ("Achannels", "Bchannels", "merge_output")):
        return True
    return str(p.get("also_merge", "none")).strip().lower() not in ("none", "")


def needs_full_frame(p):
    """True for what tiles cannot do: a bounding box other than the union, and layers."""
    return str(p.get("merge_bbox", "union")) != "union" or uses_layers(p)


def bbox_window(p, a_data, b_data):
    """The data window of the output for `merge_bbox`."""
    choice = str(p.get("merge_bbox", "union"))
    if choice == "intersection":
        return a_data.intersect(b_data)
    if choice == "A":
        return a_data
    if choice == "B":
        return b_data
    return a_data.union(b_data)


# ------------------------------------------------------------------------------------------------------------ sRGB curve
def to_srgb(x):
    """Linear to the sRGB video curve, odd about zero."""
    x = np.asarray(x, np.float32)
    mag = np.abs(x)
    encoded = np.where(mag <= 0.0031308, mag * 12.92, 1.055 * np.power(np.maximum(mag, 1e-30), 1.0 / 2.4) - 0.055)
    return (np.sign(x) * encoded).astype(np.float32)


def from_srgb(x):
    """The sRGB video curve back to linear, odd about zero."""
    x = np.asarray(x, np.float32)
    mag = np.abs(x)
    decoded = np.where(mag <= 0.04045, mag / 12.92, np.power((mag + 0.055) / 1.055, 2.4))
    return (np.sign(x) * decoded).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------ the merge
def _keep(pixels, flags):
    """`pixels` with the channels not in `flags` set to zero."""
    if all(flags):
        return pixels
    return (pixels * np.array(flags, np.float32)).astype(np.float32)


def merge_full(p, a, b, merge_op):
    """The ungated result of the operation on the channels that take part, before the output channels pick it up. `merge_op(op, a, b)`
    is the evaluator's operation."""
    op = str(p.get("operation", "over"))
    a_in = _keep(a, channel_flags(p.get("Achannels", "rgba")))
    b_in = _keep(b, channel_flags(p.get("Bchannels", "rgba")))
    srgb = bool(p.get("sRGB", 0))
    if srgb:
        a_in, b_in = a_in.copy(), b_in.copy()
        a_in[..., :3], b_in[..., :3] = to_srgb(a_in[..., :3]), to_srgb(b_in[..., :3])
    full = merge_op(op, a_in, b_in)
    if p.get("screen_alpha", 0) and op not in SCREEN_ALPHA_SKIPS:
        aa, ba = a_in[..., 3:4], b_in[..., 3:4]
        merged = np.where(ba == 0, a_in, np.where(aa == 0, b_in, full))
        merged[..., 3:4] = aa + ba - aa * ba
        full = merged.astype(np.float32)
    if srgb:
        full = full.copy()
        full[..., :3] = from_srgb(full[..., :3])
    return full


def merge_arrays(p, a, b, mask, merge_op):
    """Merge A onto B with the step-E knobs (`a`, `b`, `mask` at one output region). The mask and mix gate the result against B."""
    mix = float(p.get("mix", 1.0))
    full = merge_full(p, a, b, merge_op)
    if mask is None and mix == 1.0:
        gated = full
    else:
        gate = np.float32(mix) if mask is None else (mask[..., 3:4] if mix == 1.0 else np.float32(mix) * mask[..., 3:4])
        gated = full * gate + b * (1 - gate)
    flags = channel_flags(p.get("merge_output", "rgba"))
    if all(flags):
        return gated.astype(np.float32)
    out = b.copy()
    out[..., list(flags)] = gated[..., list(flags)]
    return out.astype(np.float32)


# ----------------------------------------------------------------------------------------------------- layers (whole image)
def _layer_or_main(raster, tokens, out_box):
    """The (h, w, 4) pixels a channel set reads from `raster`: its main picture, or the first named layer."""
    names = [name for name in tokens if name]
    if not names:
        return raster.fit(out_box)
    layers = raster.layers or {}
    name = names[0]
    if name not in layers:
        raise ValueError(f"Merge: channels name the layer {name!r}, but the input has no such layer"
                         + (f" (it has {', '.join(sorted(layers))})" if layers else " (it carries no named layers)"))
    return layers[name].fit(out_box)


def merge_raster(p, a, b, mask, out_box, merge_op, display):
    """The whole-image Merge for a node whose channel sets or `also_merge` name layers: returns (pixels, layers). `a`, `b` are Rasters,
    `mask` a Raster or None. The main picture is B's unless `output` names a layer, which then holds the merge instead."""
    from .raster import Raster

    mix = float(p.get("mix", 1.0))
    a_set, b_set, o_set = (_set(p.get(name, "rgba")) for name in ("Achannels", "Bchannels", "merge_output"))
    a_pixels = _layer_or_main(a, a_set.layers, out_box)
    b_pixels = _layer_or_main(b, b_set.layers, out_box)
    mask_pixels = None if mask is None else mask.fit(out_box)
    named = {**p, "Achannels": "rgba" if a_set.layers else p.get("Achannels", "rgba"),
             "Bchannels": "rgba" if b_set.layers else p.get("Bchannels", "rgba"), "merge_output": "rgba"}
    merged = merge_arrays(named, a_pixels, b_pixels, mask_pixels, merge_op)
    base = b.fit(out_box)
    layers = dict(b.layers or {})
    pixels = base
    if o_set.layers:
        for name in o_set.layers:
            layers[name] = Raster(merged, out_box, display)
        flags = o_set.rgba
        if any(flags):
            pixels = base.copy()
            pixels[..., list(flags)] = merged[..., list(flags)]
    else:
        flags = o_set.rgba
        pixels = merged if all(flags) else _write_flags(base, merged, flags)
    extra = str(p.get("also_merge", "none")).strip()
    if extra.lower() not in ("none", ""):
        names = sorted(set((a.layers or {}) ) | set(b.layers or {})) if extra.lower() == "all" else _set(extra).layers
        for name in names:
            la = a.layers.get(name) if a.layers else None
            lb = b.layers.get(name) if b.layers else None
            if la is None and lb is None:
                raise ValueError(f"Merge: also merge names the layer {name!r}, but neither input has it")
            zero = np.zeros((out_box.height, out_box.width, 4), np.float32)
            pa = zero if la is None else la.fit(out_box)
            pb = zero if lb is None else lb.fit(out_box)
            sub = {**p, "Achannels": "rgba", "Bchannels": "rgba", "merge_output": "rgba"}
            layers[name] = Raster(merge_arrays(sub, pa, pb, mask_pixels, merge_op), out_box, display)
    return pixels.astype(np.float32), (layers or None)


def _write_flags(base, merged, flags):
    out = base.copy()
    out[..., list(flags)] = merged[..., list(flags)]
    return out
