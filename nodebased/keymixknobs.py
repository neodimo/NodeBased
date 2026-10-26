"""Nuke's KeyMix and STMap knobs of fidelity step H.

KeyMix (`Keymix`): A is mixed over B where the matte is high; the matte is `maskChannelInput` of the `mask` input (the
alpha by default; with no mask wired, of input A, and the default alpha then leaves A whole, as it always did), `invert_mask`
turns it over, `mix` fades the mixing.

* `channels` (default `all`): the channels of A that are mixed; the others are B's. Colour channels and alpha only.
* `maskChannelInput` (default `rgba.alpha`): a word (`alpha`, `red`, `green`, `blue`, `luminance`), `rgba.<word>`, or a channel of a layer
  the mask input carries (such as `mask.a`).
* `merge_bbox` (Nuke's "set bbox to"; `union`, `intersection`, `A`, `B`): the data window of the result, as on Merge.

STMap (the uv remap):

* `channels` (default `all`): the channels of the remapped picture that replace the source's; the others stay as the source has them.
* `blur_channel` (Nuke's `blur`, default `none`) with `blur_scale` (default 1): a channel of the uv map whose value times the scale is
  a box radius in pixels, per pixel, applied to the remapped result. The radius is rounded to whole pixels and the window
  stays inside the picture (its mean is over the pixels that exist). The guide gives no more; that reading is NodeBased's.
* `maskChannelInput` (Nuke's `mask channel`, default `none` meaning the mask input's alpha) with `invert_mask`: the matte of the gating.
* `premultiplied`: the map is premultiplied, so its u and v are divided by its alpha (zero where it is zero) before they are used.
"""
from __future__ import annotations

import numpy as np

KEYMIX = {"channels": "all", "maskChannelInput": "rgba.alpha", "merge_bbox": "union"}
STMAP = {"channels": "all", "blur_channel": "none", "blur_scale": 1.0, "maskChannelInput": "none", "invert_mask": 0,
         "premultiplied": 0}
BY_KIND = {"Keymix": KEYMIX, "STMap": STMAP}
_ALPHA_WORDS = ("alpha", "rgba.alpha", "none", "")


def new_params(kind):
    return dict(BY_KIND.get(kind, {}))


def upgrade_params(kind):
    return dict(BY_KIND.get(kind, {}))


def _word(token):
    return str(token).strip().lower()


def is_alpha(token):
    return _word(token) in _ALPHA_WORDS


def _flags(text, name, kind):
    from .ops2d_chrome import parse_channels
    chosen = parse_channels(text)
    if chosen.layers:
        raise ValueError(f"{kind}: {name} takes R, G, B and A, not the layer {chosen.layers[0]!r}")
    return chosen.rgba


def needs_full_frame(kind, p):
    """True for what tiles cannot do: a result window other than the union, and a mask channel from a named layer."""
    if kind == "Keymix":
        from .ops2d_chrome import _is_rgba_token
        return (str(p.get("merge_bbox", "union")) != "union"
                or not (is_alpha(p.get("maskChannelInput", "rgba.alpha")) or _is_rgba_token(p.get("maskChannelInput"))))
    return False


# ------------------------------------------------------------------------------------------------------- KeyMix
def keymix_matte(p, a, mask, layer_getter=None):
    """The mask array KeyMix gates with: the chosen channel of the mask input (or of A when none is wired) as the alpha of a
    copy of that picture, or `mask` itself while the channel is the alpha. None when no mask is wired and the channel is alpha."""
    from .ops2d_chrome import resolve_channel
    token = p.get("maskChannelInput", "rgba.alpha")
    if is_alpha(token):
        return mask
    holder = mask if mask is not None else a
    channel = resolve_channel(token, holder, layer_getter)
    out = np.array(holder, np.float32, copy=True)
    out[..., 3:4] = channel
    return out


def keymix(p, a, b, mask, layer_getter=None):
    """KeyMix's result on (h, w, 4) arrays; `Evaluator._apply_mask_mix` does the gating."""
    from .imaging import Evaluator
    from .ops2d_chrome import parse_channels, restrict
    mixed = a
    if str(p.get("channels", "all")).strip().lower() not in ("all", "rgba"):
        _flags(p["channels"], "channels", "Keymix")
        mixed = restrict(b, a, parse_channels(p["channels"]))
    key = keymix_matte(p, a, mask, layer_getter)
    if key is not None and p.get("invert_mask"):
        key = np.concatenate([key[..., :3], 1 - key[..., 3:4]], axis=-1)
    return Evaluator._apply_mask_mix(b, mixed, mask=key, mix=p.get("mix", 1.0))


# ------------------------------------------------------------------------------------------------------- STMap
def unpremultiply_map(uv_pixels, uv_alpha):
    """u and v divided by the map's alpha, zero where the alpha is."""
    safe = np.where(uv_alpha != 0.0, uv_alpha, 1.0)
    return np.where(uv_alpha != 0.0, uv_pixels / safe, 0.0)


def variable_box_blur(image, radius):
    """`image` (h, w, c) with each pixel replaced by the mean of the box of whole-pixel radius `radius` (h, w) around it, clipped to the
    picture. Done with a summed-area table, so a large radius costs the same as a small one."""
    height, width = image.shape[:2]
    r = np.clip(np.rint(np.asarray(radius, np.float64)), 0, max(height, width)).astype(np.int64)
    if not r.any():
        return image
    table = np.zeros((height + 1, width + 1, image.shape[2]), np.float64)
    table[1:, 1:] = image.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    ys, xs = np.indices((height, width))
    y0, y1 = np.clip(ys - r, 0, height), np.clip(ys + r + 1, 0, height)
    x0, x1 = np.clip(xs - r, 0, width), np.clip(xs + r + 1, 0, width)
    total = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    count = ((y1 - y0) * (x1 - x0))[..., None]
    return (total / count).astype(np.float32)
