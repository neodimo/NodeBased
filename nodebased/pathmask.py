"""Nuke's path mask on the time nodes (fidelity step F): TimeOffset and FrameHold change the time of only the layers the patterns select.

With `enable_path_mask` on, the node reads its input twice: at the time its own knobs give (the time change) and at the frame it was
asked for. A layer whose name matches `mask_patterns` comes from the time-changed read; every other layer comes from the current
frame. The picture itself is the layer `rgba`; the other layers are the input's named layers (depth, mask, light groups, ...).

* `mask_patterns`: names or shell-style patterns (`*`, `?`, `[abc]`) separated by spaces, commas or new lines, matched without regard
  to case. A pattern that starts with `-` or `!` removes the layers it matches from the selection; patterns that are all removals
  start from every layer. An empty list selects nothing, so the node then passes the current frame through.
* `inject_mask`: also writes the layer `pathmask`, the alpha of the time-changed picture in all four channels (zero when `rgba`
  is not selected), so the part of the picture the time change contributes can be used downstream. This reading of the guide's
  one-line description is NodeBased's; it has not been compared with Nuke.

One module for both nodes: the shared code the brief asks for.
"""
from __future__ import annotations

import re
from fnmatch import fnmatchcase

import numpy as np

KINDS = ("TimeOffset", "FrameHold")
PARAMS = {"enable_path_mask": 0, "mask_patterns": "", "inject_mask": 0}
PICTURE = "rgba"
INJECTED = "pathmask"


def patterns_of(text):
    return [part for part in re.split(r"[\s,]+", str(text or "").strip()) if part]


def is_selected(name, patterns):
    """True when the layer `name` is chosen by `patterns` (see the module docstring for the rules)."""
    low = str(name).lower()
    includes = [p.lower() for p in patterns if not p.startswith(("-", "!"))]
    excludes = [p[1:].lower() for p in patterns if p.startswith(("-", "!")) and len(p) > 1]
    chosen = any(fnmatchcase(low, p) for p in includes) if includes else bool(excludes)
    return chosen and not any(fnmatchcase(low, p) for p in excludes)


def active(kind, params):
    return kind in KINDS and bool(params.get("enable_path_mask", 0))


def combine(changed, current, params):
    """The time nodes' output with a path mask: `changed` (the input read at the time-changed frame) for the selected layers,
    `current` (the input read at the requested frame) for the rest. Both are Rasters on the same data window."""
    from .raster import Raster

    if changed.pixels.shape != current.pixels.shape or changed.data != current.data or changed.display != current.display:
        raise ValueError("a path mask needs the time-changed frame and the current frame to share one format")
    patterns = patterns_of(params.get("mask_patterns", ""))
    picture_changed = is_selected(PICTURE, patterns)
    layers = {}
    for name in dict.fromkeys([*(changed.layers or {}), *(current.layers or {})]):
        take_changed = is_selected(name, patterns)
        source = (changed.layers or {}) if take_changed else (current.layers or {})
        other = (current.layers or {}) if take_changed else (changed.layers or {})
        layers[name] = source[name] if name in source else other[name]
    base = changed if picture_changed else current
    if params.get("inject_mask", 0):
        matte = changed.pixels[..., 3:4] if picture_changed else np.zeros_like(changed.pixels[..., 3:4])
        layers[INJECTED] = Raster(np.repeat(matte, 4, axis=2).astype(np.float32), changed.data, changed.display)
    return Raster(base.pixels, base.data, base.display, layers or None, base.meta)
