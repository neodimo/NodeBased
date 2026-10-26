"""Nuke's knobs on the small image and channel nodes of fidelity step D: Constant, Invert, Copy, Remove and CopyMetaData,
and the Keyer, ChromaKeyer and Difference mattes.

Pure Python helpers (NumPy only where a kernel needs it) that the evaluator, the tile executor and the tests share, so a
knob means the same on both paths.
"""
from __future__ import annotations

import re

import numpy as np

# What a new node gets (Nuke's defaults where the guide states them) and what an older document's node gets, which is what
# that node did before the knob existed. Where they differ the old value is a NodeBased-only choice.
NEW = {
    "Constant": {"first": 1, "last": 1, "channels": "rgba"},
    "Invert": {"clamp": 0},
    "Copy": {"channels": "none", "metainput": "B", "rangeinput": "B"},
    "Remove": {"channels2": "none", "channels3": "none", "channels4": "none"},
    "CopyMetaData": {"merge_mode": "Image + Meta", "keys_mode": "keys and values", "image_keys": "",
                     "image_keys_mode": "keys and values"},
    "Keyer": {"key_input": "rgb", "key_output": "rgba.alpha", "key_combine": "replace"},
    "ChromaKeyer": {"screen_gain": 1.0, "screen_balance": 0.5, "chroma_gain": 0.0, "white_point": 1.0,
                    "black_point": 0.0, "key_view": "final result"},
    "Difference": {"output_enabled": 1, "output_channel": "rgba.alpha"},
}
OLD = {kind: dict(values) for kind, values in NEW.items()}
OLD["CopyMetaData"]["keys_mode"] = "exact keys"          # the key list always named whole keys

MERGE_MODES = ("Image + Meta", "Meta only", "Meta + Image", "Image only")
FILTER_MODES = ("keys and values", "keys only", "values only", "exact keys")
FROM_INPUT = ("B", "A", "All")
KEY_INPUTS = ("rgb", "rgba.alpha")
KEY_OUTPUTS = ("rgba.alpha", "rgba.red", "rgba.green", "rgba.blue", "none")
OUTPUT_CHANNELS = ("rgba.alpha", "rgba.red", "rgba.green", "rgba.blue")
KEY_COMBINES = ("replace", "intersect", "union")
KEY_VIEWS = ("final result", "intermediate result")
_OUTPUT_INDEX = {"rgba.red": 0, "rgba.green": 1, "rgba.blue": 2, "rgba.alpha": 3}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(OLD.get(kind, {}))


# ---------------------------------------------------------------------------------------------------------- Constant
def constant_color(params):
    """The premultiplied RGBA of a Constant. `channels` limits what it contains: a channel left out is 0."""
    alpha = float(params["alpha"])
    color = np.array([float(params["red"]) * alpha, float(params["green"]) * alpha,
                      float(params["blue"]) * alpha, alpha], dtype=np.float32)
    channels = str(params.get("channels", "rgba"))
    if channels != "rgba":
        from .ops2d_chrome import parse_channels
        color = color * np.array(parse_channels(channels).rgba, dtype=np.float32)
    return color


# ------------------------------------------------------------------------------------------------------------ Remove
def remove_names(params):
    """The layer names Remove acts on: `layers` (Nuke's `channels`) and `channels2` to `channels4`. `all` is every layer."""
    names = set()
    for key in ("layers", "channels2", "channels3", "channels4"):
        names.update(part.strip() for part in str(params.get(key, "")).split(",") if part.strip() and part.strip() != "none")
    return names


def remove_keeps(name, names, keep_mode):
    """True when the layer `name` stays after Remove."""
    hit = "all" in names or name in names
    return hit == keep_mode


# ------------------------------------------------------------------------------------------------------ Copy channels
def copy_layer_channels(a, b, channels):
    """Nuke's Copy `channels` ("Layer Copy"): the channels it names are copied from A to B wherever A has them. rgba words
    and `all` act on the four colour channels."""
    if str(channels) in ("", "none"):
        return b
    from .ops2d_chrome import parse_channels
    selected = parse_channels(channels).rgba
    out = b.copy()
    for index, chosen in enumerate(selected):
        if chosen:
            out[..., index] = a[..., index]
    return out


def copy_meta(params, images):
    """The metadata a Copy passes on (`metainput`): B's, A's, or both with B winning where a key is in both. Where the chosen
    input has none, the other's is passed, as Copy always did."""
    choice = str(params.get("metainput", "B"))
    a, b = (getattr(image, "meta", None) for image in images[:2])
    if choice == "All":
        merged = {**(a or {}), **(b or {})}
        return merged or None
    first, second = (a, b) if choice == "A" else (b, a)
    return first if first is not None else second


def copy_range(params):
    """Which inputs' frame ranges a Copy outputs (`rangeinput`): ('B',), ('A',) or both."""
    choice = str(params.get("rangeinput", "B"))
    return ("A", "B") if choice == "All" else (choice,)


# ----------------------------------------------------------------------------------------------------- CopyMetaData
def _terms(text):
    return [t for t in re.split(r"[,\s]+", str(text)) if t]


def filter_meta(meta, text, mode):
    """The items of `meta` that match the search `text` (any of its terms) in their keys, their values or either; every
    item when the text is empty. The `exact keys` mode names whole keys, which is what the key list did before."""
    meta = dict(meta or {})
    terms = _terms(text)
    if not terms:
        return meta
    if mode == "exact keys":
        return {k: v for k, v in meta.items() if k in terms}
    lowered = [t.lower() for t in terms]

    def hit(k, v):
        in_key = any(t in str(k).lower() for t in lowered)
        in_value = any(t in str(v).lower() for t in lowered)
        return {"keys only": in_key, "values only": in_value}.get(mode, in_key or in_value)
    return {k: v for k, v in meta.items() if hit(k, v)}


def merge_metadata(params, image_meta, meta_meta, has_meta_input):
    """CopyMetaData's result: the Image input's metadata (filtered by `image_keys`) and the Meta input's (filtered by `keys`)
    combined by `merge_mode`. With nothing wired to Meta, Image + Meta and Image only leave the image's metadata."""
    image_side = filter_meta(image_meta, params.get("image_keys", ""), params.get("image_keys_mode", "keys and values"))
    meta_side = filter_meta(meta_meta if has_meta_input else {}, params.get("keys", ""),
                            params.get("keys_mode", "exact keys"))
    mode = str(params.get("merge_mode", "Image + Meta"))
    if mode == "Meta only":
        out = meta_side
    elif mode == "Meta + Image":
        out = {**meta_side, **image_side}
    elif mode == "Image only":
        out = image_side
    else:
        out = {**image_side, **meta_side}
    return out


# --------------------------------------------------------------------------------------------------- keyer outputs
def write_matte(base, matte, channel, combine="replace"):
    """Lay `matte` ((h, w, 1)) into `channel` of `base`, combined with what that channel holds: replace, intersect (the
    smaller of the two) or union (the larger)."""
    index = _OUTPUT_INDEX[str(channel)]
    out = base.copy()
    existing = base[..., index:index + 1]
    if combine == "intersect":
        matte = np.minimum(existing, matte)
    elif combine == "union":
        matte = np.maximum(existing, matte)
    out[..., index:index + 1] = matte
    return out.astype(np.float32)
