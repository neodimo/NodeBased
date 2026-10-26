"""Nuke's Keylight and HueKeyer knobs that fidelity step H added to the ScreenKeyer and the HueKeyer.

Pure NumPy helpers the evaluator, the tile executor and the tests share, so a knob means the same on both paths.

ScreenKeyer (Keylight):

* `unpremultiply_result`: the finished colour is shown straight (the despilled colour; alpha is the matte). Off, it is
  multiplied by the matte, which is what the node always did.
* `use_alpha_bias_for_despill` (Keylight's `gangBiases`): the despill takes its bias from the Alpha bias colour instead of the
  Despill bias.
* `screen_despot_black` / `screen_despot_white`: after the softness, black holes of up to that many pixels inside white areas of
  the matte are filled (a closing), and white specks of up to that many pixels inside black areas are removed (an opening).
* `screen_replace` / `screen_replace_*`: what replaces the screen colour in the semi-transparent part of the matte. `none` leaves
  the despilled colour; `source` brings back the original pixel in proportion to the transparency; `hard color` adds the replace
  colour in the same proportion; `soft color` adds it scaled so the pixel keeps the original's luminance.
* `mid_point`, `shadow_gain`, `midtone_gain`, `highlight_gain`: the screen gain is scaled by a gain that follows the source's
  luminance: `shadow_gain` at 0, `midtone_gain` at `mid_point`, `highlight_gain` at 1 and above, straight lines between.
* `source_alpha`: `ignore` (as before) or `normal`, which multiplies the matte by the source's own alpha.
* `inside_replace` / `inside_replace_*`: the same replace choices for the pixels the inside mask turned opaque.
* `source_crop_*`, `source_x_method`, `source_y_method`, `source_edge_*`: the source is cropped to a rectangle given as fractions
  of the image (left, right, bottom, top). Past a cropped edge the pixels are the edge colour (`color`), the nearest pixel of the
  crop (`repeat`), the mirror image (`reflect`) or the opposite side (`wrap`); each axis has its own method.
* `inside_component` / `outside_component`: which part of the inside and outside inputs forms the mask: alpha, luminance, or
  either one inverted.

HueKeyer: `output_enabled` and `output_channel` choose whether and where the matte is written.
"""
from __future__ import annotations

import numpy as np

REPLACE_METHODS = ("none", "source", "hard color", "soft color")
SOURCE_ALPHA = ("ignore", "normal")
CROP_METHODS = ("color", "repeat", "reflect", "wrap")
COMPONENTS = ("alpha", "luminance", "inverted alpha", "inverted luminance")
KEYER_VIEWS = ("final", "status", "screen_matte", "intermediate", "source", "source_alpha", "inside_mask", "outside_mask",
               "composite")
# The views that read the new `bg` input or that show a plain mask rather than the keyed picture.
NEW_VIEWS = KEYER_VIEWS[4:]

_SCREEN = {
    "screen_despot_black": 0.0, "screen_despot_white": 0.0, "screen_replace_red": 0.5, "screen_replace_green": 0.5,
    "screen_replace_blue": 0.5, "mid_point": 0.5, "shadow_gain": 1.0, "midtone_gain": 1.0, "highlight_gain": 1.0,
    "source_alpha": "ignore", "inside_replace_red": 0.5, "inside_replace_green": 0.5, "inside_replace_blue": 0.5,
    "source_x_method": "color", "source_y_method": "color", "source_edge_red": 0.0, "source_edge_green": 0.0,
    "source_edge_blue": 0.0, "source_crop_left": 0.0, "source_crop_right": 1.0, "source_crop_bottom": 0.0,
    "source_crop_top": 1.0, "inside_component": "alpha", "outside_component": "alpha",
}
NEW = {
    # Nuke's defaults: the result is unpremultiplied, the despill follows the alpha bias, and the screen and the inside mask
    # are replaced with soft colour.
    "ScreenKeyer": {**_SCREEN, "unpremultiply_result": 1, "use_alpha_bias_for_despill": 1, "screen_replace": "soft color",
                    "inside_replace": "soft color"},
    "HueKeyer": {"output_enabled": 1, "output_channel": "rgba.alpha"},
}
OLD = {
    # What the node did before: premultiplied result, its own despill bias, no replacement.
    "ScreenKeyer": {**_SCREEN, "unpremultiply_result": 0, "use_alpha_bias_for_despill": 0, "screen_replace": "none",
                    "inside_replace": "none"},
    "HueKeyer": dict(NEW["HueKeyer"]),
}
KINDS = tuple(NEW)


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(OLD.get(kind, {}))


def _f(p, name):
    return float(p.get(name, NEW["ScreenKeyer"].get(name, 0.0)))


# --------------------------------------------------------------------------------------------------------- the matte
def tuning_gain(rgb, p):
    """The screen gain's multiplier at every pixel from its luminance: `shadow_gain` at 0, `midtone_gain` at `mid_point`,
    `highlight_gain` at 1 and above. None when the three gains are all 1 (the multiplier is 1 everywhere)."""
    low, mid, high = _f(p, "shadow_gain"), _f(p, "midtone_gain"), _f(p, "highlight_gain")
    if low == 1.0 and mid == 1.0 and high == 1.0:
        return None
    point = min(max(_f(p, "mid_point"), 1e-4), 1.0 - 1e-4)
    luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
    below = low + (mid - low) * np.clip(luma / point, 0.0, 1.0)
    above = mid + (high - mid) * np.clip((luma - point) / (1.0 - point), 0.0, 1.0)
    return np.where(luma <= point, below, above).astype(np.float32)


def despot(matte, black, white, box_extreme):
    """Fill black holes up to `black` pixels across inside the white of the matte (a closing), then remove white specks up to
    `white` pixels across inside the black (an opening). `box_extreme(image, radius, use_max)` is the evaluator's box filter."""
    if black >= 0.5:
        radius = int(np.ceil(black / 2.0))
        matte = box_extreme(box_extreme(matte, radius, True), radius, False)
    if white >= 0.5:
        radius = int(np.ceil(white / 2.0))
        matte = box_extreme(box_extreme(matte, radius, False), radius, True)
    return matte


def despot_reach(p):
    """The pixels the despot reads around an output pixel: two box filters of the same radius each."""
    reach = 0
    for name in ("screen_despot_black", "screen_despot_white"):
        size = abs(float(p.get(name, 0.0)))
        if size >= 0.5:
            reach += 2 * int(np.ceil(size / 2.0))
    return reach


def component(image, name):
    """The inside or outside mask an input forms: its alpha or luminance (rec. 709 weights), or one minus either."""
    if name in ("luminance", "inverted luminance"):
        value = 0.2126 * image[..., 0:1] + 0.7152 * image[..., 1:2] + 0.0722 * image[..., 2:3]
    else:
        value = image[..., 3:4]
    value = np.clip(value, 0.0, 1.0)
    if name.startswith("inverted"):
        value = 1.0 - value
    return value.astype(np.float32)


def replace(method, despilled, original, share, colour):
    """Put something back where the matte lets the screen through. `share` ((h, w, 1), 0..1) is how much of the pixel is
    replaced. `source` blends toward the original pixel, `hard color` adds the colour, `soft color` adds it and scales the
    result so the pixel keeps the original's luminance (as much as `share` says)."""
    if method == "none":
        return despilled
    if method == "source":
        return despilled + (original - despilled) * share
    hard = despilled + share * np.asarray(colour, np.float32)
    if method != "soft color":
        return hard
    want = 0.2126 * original[..., 0:1] + 0.7152 * original[..., 1:2] + 0.0722 * original[..., 2:3]
    have = 0.2126 * hard[..., 0:1] + 0.7152 * hard[..., 1:2] + 0.0722 * hard[..., 2:3]
    return hard * (1.0 + np.clip(share, 0.0, 1.0) * (want / np.maximum(have, 1e-6) - 1.0))


def replace_colour(p, prefix):
    return np.array([p.get(f"{prefix}_red", 0.5), p.get(f"{prefix}_green", 0.5), p.get(f"{prefix}_blue", 0.5)], np.float32)


# --------------------------------------------------------------------------------------------------------- the crop
def crop_active(p):
    return (_f(p, "source_crop_left") > 0.0 or _f(p, "source_crop_right") < 1.0 or _f(p, "source_crop_bottom") > 0.0
            or _f(p, "source_crop_top") < 1.0)


def needs_full_frame(kind, p):
    """A crop is a fraction of the whole image, which a tile does not hold."""
    return kind == "ScreenKeyer" and crop_active(p)


def _axis(n, low, high, method):
    """For one axis of `n` pixels cropped to [low, high): the pixel each one reads and whether it is outside the crop."""
    index = np.arange(n)
    outside = (index < low) | (index >= high)
    width = max(high - low, 1)
    if method == "repeat":
        source = np.clip(index, low, max(high - 1, low))
    elif method == "reflect":
        t = (index - low) % (2 * width)
        source = low + np.where(t < width, t, 2 * width - 1 - t)
    elif method == "wrap":
        source = low + (index - low) % width
    else:
        source = index
    return np.clip(source, 0, n - 1), outside


def crop_source(image, p):
    """The source cropped to Keylight's crop rectangle (fractions of the image, bottom and top counted from the bottom edge).
    Past an edge each axis follows its own method; `color` pixels become the edge colour with alpha 1."""
    if not crop_active(p):
        return image
    height, width = image.shape[:2]
    left = int(round(min(max(_f(p, "source_crop_left"), 0.0), 1.0) * width))
    right = int(round(min(max(_f(p, "source_crop_right"), 0.0), 1.0) * width))
    top_edge = int(round((1.0 - min(max(_f(p, "source_crop_top"), 0.0), 1.0)) * height))
    bottom_edge = int(round((1.0 - min(max(_f(p, "source_crop_bottom"), 0.0), 1.0)) * height))
    x_method = str(p.get("source_x_method", "color"))
    y_method = str(p.get("source_y_method", "color"))
    if right <= left:
        x_method, left, right = "color", 0, 0
    if bottom_edge <= top_edge:
        y_method, top_edge, bottom_edge = "color", 0, 0
    columns, x_out = _axis(width, left, right, x_method)
    rows, y_out = _axis(height, top_edge, bottom_edge, y_method)
    out = image[rows][:, columns]
    colour = (x_out[None, :] & (x_method == "color")) | (y_out[:, None] & (y_method == "color"))
    if colour.any():
        out = out.copy()
        edge = np.array([_f(p, "source_edge_red"), _f(p, "source_edge_green"), _f(p, "source_edge_blue"), 1.0], np.float32)
        out[colour] = edge
    return out.astype(np.float32)


# -------------------------------------------------------------------------------------------------------- HueKeyer
def hue_output(image, alpha, p):
    """Write the HueKeyer's matte to its output channel; with the output off the image passes unchanged."""
    from .smallnodes import write_matte
    if not p.get("output_enabled", 1):
        return image.astype(np.float32)
    return write_matte(image, alpha, p.get("output_channel", "rgba.alpha"))
