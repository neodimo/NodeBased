"""The OCIO nodes' remaining Nuke knobs (2D behaviour lane 2, step P): context variables, swap, gain, gamma, saturation, channel view, ignore errors, reload
and working space.

Every default reproduces what the node did before the knob existed, so a document saved earlier renders the same pixels and needs no migration:
no context variables, swap off, gain 1, gamma 1, saturation 1, channel view RGB, ignore errors off, working space scene_linear.
"""
from __future__ import annotations

import numpy as np

CONTEXT = {f"{part}{n}": "" for n in range(1, 5) for part in ("key", "value")}
DISPLAY_CHANNELS = ["RGB", "R", "G", "B", "A", "Luminance"]
NEW = {
    "OCIOColorspace": {**CONTEXT, "swap": 0},
    "OCIODisplay": {**CONTEXT, "gain": 1.0, "gamma": 1.0, "saturation": 1.0, "channel_selector": "RGB"},
    "OCIOLookTransform": {**CONTEXT, "reload": 0, "ignore_errors": 0},
    "OCIOFileTransform": {"working_space": "scene_linear"},
    "Colorspace": {"swap": 0},
}
LIMITS = {"swap": (0, 1), "ignore_errors": (0, 1)}
CHOICES = {"channel_selector": DISPLAY_CHANNELS}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def context_vars(p):
    """The (name, value) pairs of the node's key1/value1 ... key4/value4 slots that name a variable."""
    out = []
    for n in range(1, 5):
        key = str(p.get(f"key{n}", "") or "").strip()
        if key:
            out.append((key, str(p.get(f"value{n}", "") or "")))
    return out


def context_for(ocio, cfg, p):
    """A copy of the config's context with the node's variables set, or None when the node names none."""
    pairs = context_vars(p)
    if not pairs:
        return None
    base = cfg.getCurrentContext()
    ctx = ocio.Context()
    ctx.setSearchPath(base.getSearchPath())
    ctx.setWorkingDir(base.getWorkingDir())
    for name, value in base.getStringVars():
        ctx.setStringVar(name, value)
    for key, value in pairs:
        ctx.setStringVar(key, value)
    return ctx


def display_before(rgb, alpha, p):
    """What Nuke's OCIODisplay does to the colour before the display transform: pick the channel to view, then the exposure (gain) and the saturation."""
    rgb = np.asarray(rgb, np.float32)
    view = str(p.get("channel_selector", "RGB"))
    if view in ("R", "G", "B"):
        rgb = np.repeat(rgb[..., "RGB".index(view):"RGB".index(view) + 1], 3, axis=-1)
    elif view == "A":
        rgb = np.repeat(alpha, 3, axis=-1).astype(np.float32)
    elif view == "Luminance":
        rgb = np.repeat((rgb @ LUMA)[..., None], 3, axis=-1)
    gain = float(p.get("gain", 1.0))
    if gain != 1.0:
        rgb = rgb * np.float32(gain)
    saturation = float(p.get("saturation", 1.0))
    if saturation != 1.0:
        grey = (rgb @ LUMA)[..., None]
        rgb = grey + (rgb - grey) * np.float32(saturation)
    return rgb


def display_after(rgb, p):
    """Gamma correction after the display transform: the picture to the power of 1 / gamma (a gamma above 1 brightens, as in Nuke's viewer)."""
    gamma = float(p.get("gamma", 1.0))
    if gamma == 1.0:
        return rgb
    return (np.sign(rgb) * np.power(np.abs(rgb), np.float32(1.0 / gamma))).astype(np.float32)


def display_undo_after(rgb, p):
    """The inverse direction starts by undoing the gamma."""
    gamma = float(p.get("gamma", 1.0))
    if gamma == 1.0:
        return rgb
    return (np.sign(rgb) * np.power(np.abs(rgb), np.float32(gamma))).astype(np.float32)


def display_undo_before(rgb, p):
    """The inverse direction ends by undoing the saturation and the gain."""
    saturation = float(p.get("saturation", 1.0))
    if saturation not in (1.0, 0.0):
        grey = (rgb @ LUMA)[..., None]
        rgb = grey + (rgb - grey) / np.float32(saturation)
    gain = float(p.get("gain", 1.0))
    if gain not in (1.0, 0.0):
        rgb = rgb / np.float32(gain)
    return rgb
