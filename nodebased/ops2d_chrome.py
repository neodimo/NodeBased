"""Nuke's shared node chrome, built once and wrapped around the existing 2D kernels.

Nearly every Nuke filter and colour node carries the same strip of controls under its own knobs: a
`channels` selector, a mask (`maskChannelInput`, `invert_mask`, `inject`), `unpremult` with
`invert_unpremult`, and `mix`. Every Draw node carries a second strip: `output`, `premult`,
`cliptype`, `replace`, `invert` and `opacity`. NodeBased's kernels each did their own mask and mix
gating; this module is the one place the rest of the strip lives, so a node gains it by being listed
in `FILTER_CHROME` or `DRAW_CHROME` and nothing else.

How it stays out of the way. Each chrome knob has a default that reproduces what the node did
before the knob existed. `active(...)` is False for those defaults, and an inactive node runs its
kernel along the old path untouched, so every existing document and every golden image renders the
same bytes. Only a changed chrome knob sends a node through `run_evaluator` / `run_tile`, which run
the node's own kernel with the chrome switched off (`inner`) and then apply the strip around it:

    unpremult by  ->  kernel  ->  re-premult  ->  channels  ->  mask and mix  ->  inject

The evaluator (Raster level, named layers available) and the tile executor (TileArtifact level) both
call the array functions below, which is what keeps the two paths on the same pixels.

Knob names are Nuke's (docs/parity/nuke_nodes.json). `mix` and the mask input already existed on
these nodes; `channels` already existed, with the choices rgb / rgba / alpha, on the nodes in
`NATIVE_CHANNELS`, whose kernels keep honouring those three values themselves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------------- the tables
# kind -> (default of the `channels` knob it gains or None, takes the mask strip, takes (un)premult by).
# Defaults are the Nuke reference guide's, except that a guide default of "none" (a page that lists the
# knob's empty state rather than its default) is read as "all".
FILTER_CHROME = {
    "Grade": ("rgb", True, True), "ColorCorrect": ("rgb", True, True), "Invert": (None, True, True),
    "Saturation": ("all", True, True), "Multiply": (None, True, True), "Add": (None, True, True),
    "Gamma": (None, True, True), "Clamp": (None, True, True), "HueCorrect": ("rgb", True, True),
    "ColorLookup": ("rgba", True, True), "Exposure": (None, True, True), "ColorMatrix": ("rgb", True, True),
    "Colorspace": ("rgb", True, True), "OCIOColorspace": ("rgb", True, True),
    "OCIOLookTransform": ("rgb", True, True), "OCIOFileTransform": ("rgb", True, True),
    "OCIOLogConvert": (None, True, True), "Log2Lin": ("rgb", True, True), "PLogLin": ("all", True, True),
    "Expression": (None, True, True), "CrossTalk": (None, True, True), "Posterize": (None, True, True),
    "Toe": ("rgb", True, True), "Dither": (None, True, True),
    "Blur": ("all", True, False), "Sharpen": (None, True, False), "ErodeFilter": ("all", True, False),
    "Median": (None, True, False), "Glow": (None, True, False), "Soften": (None, True, False),
    "Defocus": (None, True, False), "ZDefocus": (None, True, False), "DirBlur": (None, True, False),
    "EdgeBlur": (None, True, False), "Erode": (None, True, False), "DegrainSimple": ("all", True, False),
    "Bilateral": ("all", True, False), "Matrix": ("all", True, False), "Emboss": ("all", True, False),
    "BumpBoss": ("all", True, False), "Laplacian": (None, True, False), "Convolve": ("all", True, False),
    "VectorBlur": ("all", True, False), "EdgeExtend": ("rgb", False, False), "EdgeDetect": ("all", False, False),
    "HueKeyer": ("rgb", False, False),
    "Merge": (None, True, False), "Dissolve": ("all", True, False), "ChannelMerge": (None, True, False),
    # Copy (fidelity step D) has the mask strip and (un)premult by; its own `channels` copies layers from A and is not the strip's.
    "Copy": (None, True, True),
    # The Draw nodes have Nuke's mask strip too; their background is the `image` input, or transparent black.
    "Ramp": (None, True, False), "Radial": (None, True, False), "Rectangle": (None, True, False),
    "Noise": (None, True, False), "Text": (None, True, False), "Grid": (None, True, False),
    # Roto's mask input and mask channel gate its result against the bg (or transparent black); `channels` limits which
    # channels it may write (the guide's Roto lists `channels` on its tracking tab, so its default here is "all").
    "Roto": ("all", True, False),
    # Nuke's Premult and Unpremult carry `channels` and nothing else of the strip; their kernels do rgb / rgba / alpha themselves.
    "Premult": ("rgb", False, False), "Unpremult": ("rgb", False, False),
}
# The kinds whose own kernel already reads `channels` (rgb / rgba / alpha). The wrapper leaves those three
# values to the kernel and handles every other value (all, none, red, green, blue, a layer) itself.
NATIVE_CHANNELS = frozenset({
    "Invert", "Multiply", "Add", "Gamma", "Clamp", "Exposure", "OCIOLogConvert", "Posterize", "Dither",
    "Sharpen", "Median", "Glow", "Soften", "Defocus", "ZDefocus", "DirBlur", "EdgeBlur", "Erode", "Laplacian",
    "Premult", "Unpremult",
})
NATIVE_VALUES = frozenset({"rgb", "rgba", "alpha"})
# Kinds with no `mix` (Nuke's Roto, Premult and Unpremult have none), which the shared tests therefore leave out of their mix checks.
MIXLESS = frozenset({"Roto", "Premult", "Unpremult"})
# Kinds whose own kernel does not read the mask input: the wrapper gates a wired mask for them, even with every chrome knob unchanged.
WRAPPER_GATED = frozenset({"Roto"})
# Merge-like nodes gate against input B (Nuke's background); everything else gates against its first input.
B_INPUT = frozenset({"Merge", "Dissolve", "ChannelMerge", "Copy"})

# The Draw generators' mask strip has no `fringe` in Nuke's guide (Rectangle, Radial, Ramp, Noise, Text and Grid), so they do not get one.
NO_FRINGE = frozenset({"Ramp", "Radial", "Rectangle", "Noise", "Text", "Grid"})
MASK_DEFAULTS = {"maskChannelInput": "none", "invert_mask": 0, "inject": 0, "fringe": 0}
PREMULT_DEFAULTS = {"unpremult": "none", "invert_unpremult": 0}
FILTER_PARAM_NAMES = ("channels", "maskChannelInput", "invert_mask", "inject", "fringe", "unpremult", "invert_unpremult")

# Draw nodes: kind -> {default of each knob it gains}. `invert` and `premultiply_mode` of Roto are its own.
DRAW_KINDS_WITH_CHROME = ("Ramp", "Radial", "Rectangle", "Noise", "Text", "Grid")
BLANK_BACKGROUND_KINDS = DRAW_KINDS_WITH_CHROME + ("Roto",)       # with no image wired they draw on transparent black
DRAW_DEFAULTS = {"output": "rgba", "premult": "none", "cliptype": "format", "replace": 0, "invert": 0, "opacity": 1.0}
DRAW_CHROME = {kind: dict(DRAW_DEFAULTS) for kind in DRAW_KINDS_WITH_CHROME}
DRAW_CHROME["Roto"] = {"output": "alpha", "cliptype": "format", "replace": 0, "opacity": 1.0}
DRAW_CHROME["RotoPaint"] = {"output": "rgba", "cliptype": "format", "replace": 0, "opacity": 1.0}
DRAW_PARAM_NAMES = ("output", "premult", "cliptype", "replace", "invert", "opacity")
CLIP_TYPES = ("no clip", "bbox", "format", "union bbox + format", "intersect bbox + format")
MAX_CLIP_PIXELS = 1 << 26        # a clip region larger than this (67 million pixels) is refused rather than allocated


def natural_box(kind, p):
    """The area a Rectangle or Radial occupies on its own (its soft edge falls inside the box): what `no clip` lets it draw
    into, past the format and past the incoming picture. None for the shapes with no finite extent of their own (Ramp, Noise,
    Grid, Text)."""
    from .tiers import Region
    if kind not in ("Rectangle", "Radial"):
        return None
    return Region(int(p["box_x"]), int(p["box_y"]), int(p["box_width"]), int(p["box_height"]))


def clip_region(cliptype, fmt, bbox, natural=None):
    """The rectangle a Draw node may draw into for Nuke's `clip to`: the format, the incoming bounding box `bbox` (the
    format when nothing is wired), their union or intersection, or with `no clip` the union widened to the shape's own area."""
    box = fmt if bbox is None else bbox
    if cliptype == "format":
        region = fmt
    elif cliptype == "bbox":
        region = box
    elif cliptype == "union bbox + format":
        region = box.union(fmt)
    elif cliptype == "intersect bbox + format":
        region = box.intersect(fmt)
    elif cliptype == "no clip":
        region = box.union(fmt)
        if natural is not None and not natural.is_empty:
            region = region.union(natural)
    else:
        raise ValueError(f"Unknown clip to {cliptype!r}; choose one of {', '.join(CLIP_TYPES)}")
    if region.width * region.height > MAX_CLIP_PIXELS:
        raise ValueError(f"The clip region {region.width} x {region.height} is too large to draw")
    return region

# Suggested values of the open-ended knobs. A layer name typed into one is also accepted.
CHANNEL_CHOICES = ["rgb", "rgba", "alpha", "all", "none", "red", "green", "blue"]
MASK_CHANNEL_CHOICES = ["none", "rgba.alpha", "rgba.red", "rgba.green", "rgba.blue", "rgba.luminance"]
UNPREMULT_CHOICES = ["none", "rgba.alpha", "rgba.red", "rgba.green", "rgba.blue"]
PREMULT_BY_CHOICES = ["rgba.alpha", "rgba.red", "rgba.green", "rgba.blue", "rgba.luminance"]       # Premult / Unpremult: there is always a divisor
MERGE_CHANNEL_CHOICES = ["rgba", "rgb", "alpha", "none", "red", "green", "blue"]
OUTPUT_CHOICES = ["rgba", "rgb", "alpha", "red", "green", "blue", "none"]
MERGE_OPEN_CHOICES = ("Achannels", "Bchannels", "merge_output", "also_merge")
OPEN_CHOICE_PARAMS = frozenset({"channels", "maskChannelInput", "unpremult", "output", "premult"})

_COMPONENT = {"R": 0, "X": 0, "U": 0, "G": 1, "Y": 1, "V": 1, "B": 2, "Z": 2, "A": 3, "W": 3}
_RGBA_WORDS = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2, "alpha": 3, "a": 3}


def spec_params(kind):
    """The chrome knobs `kind` gains, as {name: default}. Added to core.SPECS at import."""
    entry = FILTER_CHROME.get(kind)
    out = {}
    if entry is not None:
        channels, mask, unpremult = entry
        if channels is not None and kind not in NATIVE_CHANNELS:
            out["channels"] = channels
        if mask:
            out.update(MASK_DEFAULTS)
            if kind in NO_FRINGE:
                out.pop("fringe")
        if unpremult:
            out.update(PREMULT_DEFAULTS)
    if kind in DRAW_CHROME:
        out.update(DRAW_CHROME[kind])
    return out


def is_open_choice(kind, name):
    """Is `name` on `kind` one of the chrome knobs that takes a layer or channel name as well as a listed word?"""
    if kind == "Write" and name == "channels":
        return True              # Write's channels (fidelity step C) take a layer name too
    if name == "premult_by":
        return kind in ("Premult", "Unpremult")
    if kind == "Merge" and name in MERGE_OPEN_CHOICES:
        return True              # Merge's channel sets (fidelity step E) take a layer name too
    if name not in OPEN_CHOICE_PARAMS:
        return False
    if name in ("output", "premult"):
        return kind in DRAW_CHROME
    return True


def suggestions(name):
    """The words offered in a chrome knob's drop-down."""
    return {"channels": CHANNEL_CHOICES, "maskChannelInput": MASK_CHANNEL_CHOICES, "unpremult": UNPREMULT_CHOICES,
            "output": OUTPUT_CHOICES, "premult": UNPREMULT_CHOICES, "premult_by": PREMULT_BY_CHOICES,
            "Achannels": MERGE_CHANNEL_CHOICES, "Bchannels": MERGE_CHANNEL_CHOICES, "merge_output": MERGE_CHANNEL_CHOICES,
            "also_merge": ["none", "all"]}[name]


def step_c_params(kind):
    """The knobs fidelity step C added to Read, Write and Crop, at the values that leave an older document's node as it was."""
    if kind == "Read":
        from .readknobs import DEFAULTS
        return dict(DEFAULTS)
    if kind == "Write":
        from .writenode import DEFAULTS
        return dict(DEFAULTS)
    if kind == "Crop":
        return {"crop_softness": 0.0, "reformat": 0, "intersect": 1, "crop": 1}
    if kind in ("Premult", "Unpremult"):
        return {"channels": "rgb", "premult_by": "rgba.alpha", "invert": 0}
    return {}


def step_d_params(kind):
    """The knobs fidelity step D added (the time nodes, the keyers and the small channel nodes), at the values that leave an
    older document's node as it was. Where Nuke's default differs the new node gets Nuke's and an old one gets this."""
    from . import smallnodes, timeknobs
    out = timeknobs.upgrade_params(kind)
    out.update(smallnodes.upgrade_params(kind))
    return out


def step_e_params(kind):
    """The knobs fidelity step E added (Grade, Merge, Blur, Sharpen and Soften), at the values that leave an older document's
    node as it was."""
    from . import blurknobs, gradeknobs, mergeknobs
    out = gradeknobs.upgrade_params(kind)
    out.update(mergeknobs.upgrade_params(kind))
    out.update(blurknobs.upgrade_params(kind))
    return out


def step_f_params(kind):
    """The knobs fidelity step F added (Roto's node-level settings, Text, the time nodes' path mask, Read and Write), at the
    values that leave an older document's node as it was."""
    from . import rotoknobs
    out = rotoknobs.upgrade_params(kind)
    return out


def upgrade_nodes(nodes):
    """Give an older document's nodes the chrome knobs at their no-change defaults. Walks Group graphs."""
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        params = node.get("params")
        added = spec_params(node.get("type"))
        added = {**added, **step_c_params(node.get("type")), **step_d_params(node.get("type")), **step_e_params(node.get("type")), **step_f_params(node.get("type"))}
        if added and isinstance(params, dict):
            for name, default in added.items():
                params.setdefault(name, default)
        graph = node.get("graph")
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), dict):
            upgrade_nodes(graph["nodes"])


# ------------------------------------------------------------------------------------------------ channels
@dataclass(frozen=True)
class ChannelSet:
    """A parsed `channels` value: which of R, G, B, A, and which named layers (token strings)."""
    rgba: tuple
    layers: tuple

    @property
    def everything(self):
        return all(self.rgba) and not self.layers


def parse_channels(text):
    """`channels` / `output` text -> ChannelSet. Tokens are separated by commas, spaces or '+'."""
    rgba = [False] * 4
    layers = []
    for token in re.split(r"[,\s+]+", str(text).strip()):
        word = token.lower()
        if not word or word == "none":
            continue
        if word in ("all", "rgba"):
            rgba = [True] * 4
        elif word == "rgb":
            rgba[:3] = [True] * 3
        elif word in ("alpha", "red", "green", "blue", "r", "g", "b", "a"):
            rgba[_RGBA_WORDS[word]] = True
        elif word.startswith("rgba.") and word[5:] in _RGBA_WORDS:
            rgba[_RGBA_WORDS[word[5:]]] = True
        else:
            layers.append(token)
    return ChannelSet(tuple(rgba), tuple(layers))


def resolve_channel(token, pixels, layer_array=None):
    """One channel as an (h, w, 1) float32 array. `token` is a word (alpha, red, ...), `rgba.alpha`,
    `rgba.luminance`, `layer.X` (a component of a named layer) or a bare layer name (its first channel).
    `layer_array(name)` returns that layer as (h, w, c) or None."""
    word = str(token).strip()
    lower = word.lower()
    if lower.startswith("rgba."):
        lower = lower[5:]
    if lower in _RGBA_WORDS:
        index = _RGBA_WORDS[lower]
        return pixels[..., index:index + 1]
    if lower == "luminance":
        return (0.2126 * pixels[..., 0:1] + 0.7152 * pixels[..., 1:2] + 0.0722 * pixels[..., 2:3]).astype(np.float32)
    if layer_array is not None:
        array = layer_array(word)
        if array is not None:
            return array[..., 0:1]
        if "." in word:
            base, component = word.rsplit(".", 1)
            array = layer_array(base)
            index = _COMPONENT.get(component.upper(), -1)
            if array is not None and 0 <= index < array.shape[-1]:
                return array[..., index:index + 1]
    raise ValueError(f"No channel named {word!r}: use alpha, red, green, blue, luminance, or a layer the input carries")


# ------------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class FilterConfig:
    kind: str
    channels: str
    native: bool            # the kind's kernel handles rgb / rgba / alpha itself
    mask_channel: str
    invert_mask: bool
    inject: bool
    unpremult: str
    invert_unpremult: bool
    mix: float
    fringe: bool = False

    @property
    def channel_set(self):
        return parse_channels(self.channels)

    @property
    def restricts(self):
        """True when the wrapper (not the kernel) must restrict the result to some channels."""
        if self.native and self.channels in NATIVE_VALUES:
            return False
        default = FILTER_CHROME[self.kind][0]
        if not self.native and (default is None or self.channels == default):
            return False        # no channels knob on this node, or it is at its default
        return True

    @property
    def uses_layers(self):
        return bool(self.restricts and self.channel_set.layers)

    def active(self, has_mask):
        """Does anything in the strip change this node's result from what its kernel alone gives?"""
        if self.restricts or self.unpremult != "none":
            return True
        if self.mask_channel != "none":
            return True
        if has_mask and (self.invert_mask or self.inject or self.fringe or self.kind in WRAPPER_GATED):
            return True
        return False


def filter_config(kind, p):
    entry = FILTER_CHROME.get(kind)
    if entry is None:
        return None
    default = entry[0]
    channels = str(p.get("channels", default if default is not None else "rgba"))
    return FilterConfig(kind=kind, channels=channels, native=kind in NATIVE_CHANNELS,
                        mask_channel=str(p.get("maskChannelInput", "none")), invert_mask=bool(p.get("invert_mask", 0)),
                        inject=bool(p.get("inject", 0)), unpremult=str(p.get("unpremult", "none")),
                        invert_unpremult=bool(p.get("invert_unpremult", 0)), mix=float(p.get("mix", 1.0)),
                        fringe=bool(p.get("fringe", 0)))


def mask_slot(spec):
    """Index of the mask input among a kind's inputs and optional inputs, or None."""
    slots = list(spec["inputs"]) + list(spec.get("optional_inputs", []))
    return slots.index("mask") if "mask" in slots else None


def primary_slot(kind):
    return 1 if kind in B_INPUT else 0


def inner_params(cfg, p):
    """The kernel's own params: chrome off, mix 1, so the kernel returns its pure filter."""
    q = dict(p)
    q.update(MASK_DEFAULTS if "maskChannelInput" in p else {})
    q.update(PREMULT_DEFAULTS if "unpremult" in p else {})
    q["mix"] = 1.0
    if cfg.restricts:
        if cfg.native:
            q["channels"] = "rgba"
        elif "channels" in q:
            q["channels"] = FILTER_CHROME[cfg.kind][0]
    return q


def needs_full_frame(kind, p):
    """True when a node's chrome reads named layers, which tile artifacts do not take part in here:
    the graph then goes to the whole-image evaluator, like a Shuffle that names a layer."""
    if kind in ("Roto", "RotoPaint"):
        # Their tiles are rasterised straight from the shape list, so the Draw strip runs on the whole image.
        draw = draw_config(kind, p)
        if draw is not None and (draw.replace or draw.opacity != 1.0 or draw.output != draw.default_output or draw.cliptype != "format"):
            return True
        if output_mask_layer(kind, p):
            return True
        cfg = filter_config(kind, p)
        return bool(cfg is not None and cfg.active(False))
    if kind in ("Blur", "Sharpen", "Soften"):
        from . import blurknobs
        if blurknobs.needs_full_frame(kind, p):
            return True      # a window that grows, or a picture scaled down to fit `quality`, is not a tile's business
    if kind == "Merge":
        from . import mergeknobs
        if mergeknobs.needs_full_frame(p):
            return True      # a bounding box other than the union moves the output window, and layers are not in tiles
    if kind in DRAW_KINDS_WITH_CHROME and str(p.get("cliptype", "format")) != "format":
        return True      # a clip other than the format moves the output window, which tiles of the format cannot hold
    if kind in ("Premult", "Unpremult") and not _is_rgba_token(p.get("premult_by", "rgba.alpha")):
        return True      # divides or multiplies by a layer channel, which only the whole-image path can read
    cfg = filter_config(kind, p)
    if cfg is None:
        return False
    if cfg.uses_layers:
        return True
    for token in (cfg.mask_channel, cfg.unpremult):
        if token != "none" and not _is_rgba_token(token):
            return True
    return False


def _is_rgba_token(token):
    lower = str(token).lower()
    if lower.startswith("rgba."):
        lower = lower[5:]
    return lower in _RGBA_WORDS or lower == "luminance"


# ------------------------------------------------------------------------------------- the array arithmetic
def unpremult_factor(cfg, channel):
    """The (h, w, 1) divisor and whether it is usable pixelwise: zero channels are left alone."""
    safe = np.where(channel != 0, channel, np.float32(1.0)).astype(np.float32)
    return channel, safe


def apply_pre(rgba, channel, inverse):
    """Un-premultiply rgb by `channel` (or, inverted, premultiply it) before the kernel runs."""
    out = rgba.copy()
    _, safe = unpremult_factor(None, channel)
    out[..., :3] = rgba[..., :3] * channel if inverse else rgba[..., :3] / safe
    return out


def apply_post(rgba, channel, inverse):
    """Undo `apply_pre` on the kernel's result."""
    out = rgba.copy()
    _, safe = unpremult_factor(None, channel)
    out[..., :3] = rgba[..., :3] / safe if inverse else rgba[..., :3] * channel
    return out


def restrict(source, filtered, channel_set):
    """`filtered` in the selected R, G, B, A channels, `source` in the others."""
    keep = np.array(channel_set.rgba, dtype=bool)
    if keep.all():
        return filtered
    out = source.copy()
    out[..., keep] = filtered[..., keep]
    return out


def matte_of(channel, invert):
    matte = channel.astype(np.float32)
    return (np.float32(1.0) - matte) if invert else matte


def fringe_of(matte):
    """Nuke's `fringe`: the effect applies only on the soft edge of the mask. The weight is 0 where the mask is empty or full and
    peaks at 1 where it is half (a triangle, `1 - |2 m - 1|`, on the mask clamped to 0 .. 1), so a hard-edged mask has no edge
    to work on. The guide says only "only apply the effect to the edge of the mask"; the shape of the ramp is our reading."""
    m = np.clip(matte, np.float32(0.0), np.float32(1.0))
    return (np.float32(1.0) - np.abs(np.float32(2.0) * m - np.float32(1.0))).astype(np.float32)


def gate_matte(matte, cfg):
    """The matte the standard blend uses: the mask itself, or its soft edge when `fringe` is on."""
    return matte if matte is None or not cfg.fringe else fringe_of(matte)


def gate(source, filtered, matte, mix):
    """The standard Nuke blend `mix * matte`: the same arithmetic as Evaluator._apply_mask_mix."""
    if matte is None:
        if float(mix) == 1.0:
            return filtered
        g = np.float32(mix)
    else:
        g = (matte * np.float32(mix)).astype(np.float32)
    return (filtered * g + source * (1.0 - g)).astype(np.float32)


def injected_layer(matte):
    """The `mask` layer Nuke's inject writes: the matte in every channel, so `mask.a` is the matte."""
    return np.repeat(matte.astype(np.float32), 4, axis=-1)


# --------------------------------------------------------------------------------------- evaluator wrapper
def run_evaluator(kind, p, inputs, inner, spec, force=False):
    """Run a filter or colour node's kernel with the chrome strip around it (Raster level).

    `inner(kind, params, inputs)` is the node's own whole-image kernel with chrome switched off. Returns
    its Raster, or the stripped Raster the chrome produced."""
    from .raster import Raster

    cfg = filter_config(kind, p)
    index = primary_slot(kind)
    source = inputs[index] if len(inputs) > index else None
    slot = mask_slot(spec)
    mask = inputs[slot] if slot is not None and len(inputs) > slot else None
    blank = source is None and kind in BLANK_BACKGROUND_KINDS          # a Draw node with no image draws on transparent black
    if cfg is None or (source is None and not blank) or not (force or cfg.active(mask is not None)):
        return inner(kind, p, inputs)
    if blank:
        return _run_blank_background(kind, p, inputs, inner, cfg, slot, mask)
    if mask is not None and mask.display != source.display:
        raise ValueError(f"Mask display window {mask.display} does not match source {source.display}; "
                         "no silent resampling is performed")

    def pre(raster):
        """The raster the kernel should see: rgb divided by the (un)premult channel."""
        if cfg.unpremult == "none":
            return raster
        channel = resolve_channel(cfg.unpremult, raster.pixels, _layer_getter(raster, raster.data))
        return Raster(apply_pre(raster.pixels, channel, cfg.invert_unpremult), raster.data, raster.display,
                      raster.layers, raster.meta)

    kernel_inputs = list(inputs)
    kernel_inputs[index] = pre(source)
    if slot is not None and slot < len(kernel_inputs):
        kernel_inputs[slot] = None
    params = inner_params(cfg, p)
    result = inner(kind, params, kernel_inputs)
    matte_wanted = mask is not None or cfg.mask_channel != "none"
    gated = matte_wanted or cfg.mix != 1.0
    box = result.data.union(source.data) if gated else result.data
    base = source.fit(box)
    filtered = result.fit(box)
    if cfg.unpremult != "none":
        channel = resolve_channel(cfg.unpremult, base, _layer_getter(source, box))
        filtered = apply_post(filtered, channel, cfg.invert_unpremult)
    layers = dict(source.layers or {})
    if cfg.restricts:
        selected = cfg.channel_set
        filtered = restrict(base, filtered, selected)
        for name in selected.layers:
            layers_name, processed = _process_layer(kind, params, kernel_inputs, index, inner, source, name, box)
            if processed is not None:
                layers[layers_name] = processed
    matte = None
    if matte_wanted:
        if cfg.mask_channel == "none":
            channel = mask.fit(box)[..., 3:4]
        else:
            holder = mask if mask is not None else source
            channel = resolve_channel(cfg.mask_channel, holder.fit(box), _layer_getter(holder, box))
        matte = matte_of(channel, cfg.invert_mask)
    pixels = gate(base, filtered, gate_matte(matte, cfg), cfg.mix)
    if cfg.inject and matte is not None:
        layers["mask"] = Raster(injected_layer(matte), box, source.display)
    return Raster(pixels, box, source.display, layers or None, source.meta)


def _run_blank_background(kind, p, inputs, inner, cfg, slot, mask):
    """The chrome around a Draw node that has no image wired: the background is transparent black at its format."""
    from .raster import Raster

    kernel_inputs = list(inputs)
    if slot is not None and slot < len(kernel_inputs):
        kernel_inputs[slot] = None
    result = inner(kind, inner_params(cfg, p), kernel_inputs)
    if mask is not None and mask.display != result.display:
        raise ValueError(f"Mask display window {mask.display} does not match {kind}'s own format {result.display}; "
                         "no silent resampling is performed")
    box = result.data
    base = np.zeros((box.height, box.width, 4), np.float32)
    matte = None
    if mask is not None or cfg.mask_channel != "none":
        if cfg.mask_channel == "none":
            channel = mask.fit(box)[..., 3:4]
        else:
            holder = mask if mask is not None else Raster(base, box)
            channel = resolve_channel(cfg.mask_channel, holder.fit(box), _layer_getter(holder, box))
        matte = matte_of(channel, cfg.invert_mask)
    pixels = gate(base, result.pixels, gate_matte(matte, cfg), cfg.mix)
    layers = {"mask": Raster(injected_layer(matte), box, result.display)} if cfg.inject and matte is not None else None
    return Raster(pixels, box, result.display, layers, result.meta)


def _layer_getter(raster, box):
    layers = raster.layers or {}

    def get(name):
        layer = layers.get(name)
        return None if layer is None else layer.fit(box)
    return get


def _process_layer(kind, params, kernel_inputs, index, inner, source, token, box):
    """Run the kernel on one named layer (or one component of it) and return (layer name, Raster)."""
    from .raster import Raster
    layers = source.layers or {}
    name, component = token, None
    if token not in layers and "." in token:
        name, component = token.rsplit(".", 1)
    if name not in layers:
        raise ValueError(f"{kind}: channels names {token!r}, but the input has no such layer"
                         + (f" (it has {', '.join(sorted(layers))})" if layers else " (it carries no named layers)"))
    layer = layers[name]
    sub_inputs = list(kernel_inputs)
    sub_inputs[index] = layer
    done = inner(kind, params, sub_inputs)
    out = done.fit(box)
    base = layer.fit(box)
    if component is not None:
        which = _COMPONENT.get(component.upper(), -1)
        if not 0 <= which < 4:
            raise ValueError(f"{kind}: {token!r} is not a channel")
        merged = base.copy()
        merged[..., which] = out[..., which]
        out = merged
    return name, Raster(out, box, source.display)


# ------------------------------------------------------------------------------------------- tile wrapper
def run_tile(kind, p, inputs, inner, spec, buffered_region, force=False):
    """The chrome strip around a tile kernel. `inputs` are TileArtifacts; `inner(params, inputs)` is the
    tile kernel with chrome off. Returns the pixels (same shape the kernel itself would return)."""
    from dataclasses import replace

    cfg = filter_config(kind, p)
    index = primary_slot(kind)
    source = inputs[index] if len(inputs) > index else None
    slot = mask_slot(spec)
    mask = inputs[slot] if slot is not None and len(inputs) > slot else None
    blank = source is None and kind in BLANK_BACKGROUND_KINDS
    if cfg is None or (source is None and not blank) or not (force or cfg.active(mask is not None)):
        return inner(p, inputs)
    if blank:
        return _run_blank_tile(kind, p, inputs, inner, cfg, slot, mask, buffered_region)
    src_pixels = source.pixels
    kernel_inputs = list(inputs)
    if cfg.unpremult != "none":
        channel = resolve_channel(cfg.unpremult, src_pixels)
        kernel_inputs[index] = replace(source, pixels=apply_pre(src_pixels, channel, cfg.invert_unpremult))
    if slot is not None and slot < len(kernel_inputs):
        kernel_inputs[slot] = None
    params = inner_params(cfg, p)
    filtered = inner(params, kernel_inputs)
    if filtered.shape[:2] == src_pixels.shape[:2]:
        base = src_pixels
        reference = source
    else:
        from .tileexec import _align_artifact_to
        base = _align_artifact_to(source, buffered_region)
        reference = None
    if cfg.unpremult != "none":
        channel = resolve_channel(cfg.unpremult, base)
        filtered = apply_post(filtered, channel, cfg.invert_unpremult)
    if cfg.restricts:
        filtered = restrict(base, filtered, cfg.channel_set)
    matte = None
    if mask is not None or cfg.mask_channel != "none":
        if mask is not None:
            mask_pixels = mask.pixels
            if mask_pixels.shape[:2] != base.shape[:2] and reference is not None:
                mh, mw = mask_pixels.shape[:2]
                offset_x = max(0, mask.region.x - reference.region.x)
                offset_y = max(0, mask.region.y - reference.region.y)
                base = base[offset_y:offset_y + mh, offset_x:offset_x + mw]
                filtered = filtered[offset_y:offset_y + mh, offset_x:offset_x + mw]
            elif mask_pixels.shape[:2] != base.shape[:2]:
                from .tileexec import _align_artifact_to
                mask_pixels = _align_artifact_to(mask, buffered_region)
            holder = mask_pixels
        else:
            holder = base
        channel = holder[..., 3:4] if cfg.mask_channel == "none" else resolve_channel(cfg.mask_channel, holder)
        matte = matte_of(channel, cfg.invert_mask)
    return gate(base, filtered, gate_matte(matte, cfg), cfg.mix)


def _run_blank_tile(kind, p, inputs, inner, cfg, slot, mask, region):
    """Tile twin of `_run_blank_background`."""
    kernel_inputs = list(inputs)
    if slot is not None and slot < len(kernel_inputs):
        kernel_inputs[slot] = None
    filtered = inner(inner_params(cfg, p), kernel_inputs)
    base = np.zeros((region.height, region.width, 4), np.float32)
    matte = None
    if mask is not None or cfg.mask_channel != "none":
        from .tileexec import _align_artifact_to
        holder = _align_artifact_to(mask, region) if mask is not None else base
        channel = holder[..., 3:4] if cfg.mask_channel == "none" else resolve_channel(cfg.mask_channel, holder)
        matte = matte_of(channel, cfg.invert_mask)
    return gate(base, filtered, gate_matte(matte, cfg), cfg.mix)


def tile_inject_layer(kind, p, inputs, spec, region):
    """The `mask` layer array an `inject` writes on a tile, or None. Mirrors the matte `run_tile` used."""
    cfg = filter_config(kind, p)
    if cfg is None or not cfg.inject:
        return None
    slot = mask_slot(spec)
    mask = inputs[slot] if slot is not None and len(inputs) > slot else None
    source = inputs[primary_slot(kind)] if len(inputs) > primary_slot(kind) else None
    if mask is None and cfg.mask_channel == "none":
        return None
    from .tileexec import _align_artifact_to
    holder_artifact = mask if mask is not None else source
    holder = (_align_artifact_to(holder_artifact, region) if holder_artifact is not None
              else np.zeros((region.height, region.width, 4), np.float32))
    channel = holder[..., 3:4] if cfg.mask_channel == "none" else resolve_channel(cfg.mask_channel, holder)
    return injected_layer(matte_of(channel, cfg.invert_mask))


# --------------------------------------------------------------------------------------------- draw chrome
@dataclass(frozen=True)
class DrawConfig:
    output: str
    default_output: str
    premult: str
    cliptype: str
    replace: bool
    invert: bool
    opacity: float

    @property
    def active(self):
        return (self.output != self.default_output or self.premult != "none" or self.replace or self.invert
                or self.opacity != 1.0)


def draw_config(kind, p):
    if kind not in DRAW_CHROME:
        return None
    defaults = DRAW_CHROME[kind]
    return DrawConfig(output=str(p.get("output", defaults.get("output", "rgba"))),
                      default_output=defaults.get("output", "rgba"), premult=str(p.get("premult", "none")), cliptype=str(p.get("cliptype", "format")),
                      replace=bool(p.get("replace", 0)),
                      invert=bool(p.get("invert", 0)) if kind not in ("Roto", "RotoPaint") else False,
                      opacity=float(p.get("opacity", 1.0)))


def shape_fallback_color(p):
    """A straight colour for pixels the shape leaves uncovered (needed when `invert` uncovers them)."""
    if "red" in p:
        return np.array([p["red"], p["green"], p["blue"], p["alpha"]], np.float32)
    if "color0_red" in p:
        return np.array([p["color0_red"], p["color0_green"], p["color0_blue"], p["color0_alpha"]], np.float32)
    return np.ones(4, np.float32)


def shape_chrome(kind, p, shape):
    """Invert, premult-by and opacity applied to a freshly drawn premultiplied RGBA shape. Called by
    Evaluator._draw_shape, the one function both the evaluator and the tile path draw through."""
    cfg = draw_config(kind, p)
    if cfg is None or not (cfg.invert or cfg.premult != "none" or cfg.opacity != 1.0):
        return shape
    out = shape.astype(np.float32)
    if cfg.invert:
        alpha = out[..., 3:4]
        fallback = shape_fallback_color(p)
        straight = np.where(alpha > 1e-9, out[..., :3] / np.where(alpha > 1e-9, alpha, 1.0), fallback[:3])
        new_alpha = np.float32(1.0) - alpha
        out = np.concatenate([straight * new_alpha, new_alpha], axis=-1).astype(np.float32)
    if cfg.premult != "none":
        channel = resolve_channel(cfg.premult, out)
        out = out.copy()
        out[..., :3] = out[..., :3] * channel
    if cfg.opacity != 1.0:
        out = (out * np.float32(cfg.opacity)).astype(np.float32)
    return out


def compose_draw(kind, p, shape, background):
    """The shape over its background (or replacing it), written only to the `output` channels."""
    cfg = draw_config(kind, p)
    over = shape if (cfg is not None and cfg.replace) else (shape + background * (1.0 - shape[..., 3:4])).astype(np.float32)
    if cfg is None or cfg.output == "rgba":
        return over
    selected = parse_channels(cfg.output)
    return restrict(background, over, selected)


def output_mask_layer(kind, p):
    """The layer name Roto's `outputMask` writes the matte to, or None when it is "none" (or a node without the knob)."""
    if kind != "Roto":
        return None
    name = str(p.get("outputMask", "none")).strip()
    return None if name.lower() in ("", "none") else name


def run_roto(kind, p, inputs, inner):
    """Roto and RotoPaint's Draw chrome around the node's own result (Raster level). The background is the
    wired `bg` (Roto) or `image` (RotoPaint) input, or transparent black when Roto has none.

    `replace` draws the shapes without the background; `opacity` scales Roto's coverage and blends RotoPaint's result
    toward its image; `output` writes only the listed channels and passes the rest of the background through. The default
    `output` ("alpha" for Roto) keeps the node's established output, which is how every older document looks."""
    from .raster import Raster

    cfg = draw_config(kind, p)
    mask_layer = output_mask_layer(kind, p)
    if cfg is None or not (cfg.replace or cfg.opacity != 1.0 or cfg.output != cfg.default_output or cfg.cliptype != "format"
                           or mask_layer):
        return inner(kind, p, inputs)
    background = inputs[0] if inputs else None
    work = list(inputs)
    if cfg.replace and kind == "Roto" and work:
        work[0] = None
    result = inner(kind, p, work)
    box = result.data if background is None else result.data.union(background.data)
    base = np.zeros((box.height, box.width, 4), np.float32) if background is None else background.fit(box)
    pixels = result.fit(box)
    if cfg.replace and kind == "Roto":
        base = np.zeros_like(base)
    if cfg.opacity != 1.0:
        opacity = np.float32(cfg.opacity)
        if kind == "Roto":
            # Opacity scales the shapes' coverage: the alpha, and the colour where the node premultiplies by it.
            pixels = pixels.copy()
            pixels[..., 3] *= opacity
            if background is None or cfg.replace or str(p.get("premultiply_mode", "none")) in ("rgb", "rgba"):
                pixels[..., :3] *= opacity
        else:
            pixels = (base + opacity * (pixels - base)).astype(np.float32)
    layers = result.layers
    if mask_layer:
        # Output mask: the shapes' matte (after opacity) is also written, in all four channels, to the layer of that name.
        matte = np.repeat(pixels[..., 3:4], 4, axis=2).astype(np.float32)
        layers = {**(layers or {}), mask_layer: Raster(matte, box, result.display)}
    if cfg.output != cfg.default_output:
        pixels = restrict(base, pixels, parse_channels(cfg.output))
    if cfg.cliptype != "format":
        # Clip to: the output reaches only as far as the clip says; what lies beyond it is dropped.
        region = clip_region(cfg.cliptype, result.display, None if background is None else background.data)
        clipped = np.zeros((region.height, region.width, 4), np.float32)
        overlap = region.intersect(box)
        if not overlap.is_empty:
            clipped[overlap.y - region.y:overlap.bottom - region.y, overlap.x - region.x:overlap.right - region.x] = \
                pixels[overlap.y - box.y:overlap.bottom - box.y, overlap.x - box.x:overlap.right - box.x]
        if layers is not result.layers and mask_layer:
            clipped_layer = np.zeros((region.height, region.width, 4), np.float32)
            if not overlap.is_empty:
                clipped_layer[overlap.y - region.y:overlap.bottom - region.y, overlap.x - region.x:overlap.right - region.x] = \
                    layers[mask_layer].pixels[overlap.y - box.y:overlap.bottom - box.y, overlap.x - box.x:overlap.right - box.x]
            layers = {**layers, mask_layer: Raster(clipped_layer, region, result.display)}
        return Raster(clipped, region, result.display, layers, result.meta)
    return Raster(pixels, box, result.display, layers, result.meta)
