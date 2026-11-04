"""GridWarp and SplineWarp: Nuke's inputs, output choice, background and warp amounts (lane 9, 2D behaviour step H).

GridWarp (Nuke `src`, `dst` and `bg` inputs; NodeBased's `image`, `dst` and `bg`, the last two optional and after the mask so a saved
edge keeps its slot):

* `grid_output` (Nuke's `output`): `source` is the source picture as it is, `source warped` (the default) the source carried from its
  grid to the destination grid, `destination` the `dst` picture as it is (the source when none is wired), `destination warped` the
  `dst` picture carried from the destination grid back to the source grid. `morph` is not offered: the guide's page for the node
  is a video transcript that says only "if we turn on morph, we get access to the mix and warp values".
* `background` and `background_mix`: the warped picture goes over `on black` (opaque black), `on src` (the source), `on dst` (the
  `dst` picture, else the source) or `on bg` (the `bg` picture, else transparent black), and `background_mix` blends from the warped
  picture alone (0, the default, which is what the node always output) to that composite (1). The guide gives no formula; this
  reading is NodeBased's.

SplineWarp:

* `premultiply` (Nuke's `premultiply input`): the colour is multiplied by the alpha before it is warped.
* `layer_warp` and `pair_warp` (both 0 to 1, default 1): how far each pair of curves carries the picture from its source curve towards
  its destination curve; the two multiply, so 0 on either gives the input back. NodeBased has one layer and applies the amount to every
  pair; per-pair amounts would live in the pair's payload entry, which the schema keeps to name, source and destination.
* `boundary_bbox` (Nuke's `bbox boundary curve`, default on): with the bounding box set to `union`, the window grows to the destination
  curves. NodeBased has no separate boundary curve, so the destination curves stand in for it; off keeps the source's window.
* `crop_to_format` (default on for a new node): the output window is cut to the format. A saved node keeps off.
* `outputMask` (default `mask_splinwarp.a`; `none` writes nothing): a layer named by the part before the dot, whose alpha is 1 where the
  warp moved the picture (by more than a thousandth of a pixel) and 0 elsewhere.
* `key` and `curves` live in the node's payload (docs/SHAPES.md): `curves` is its `pairs` list and the keys are the animation keys of
  each point's scalars, so neither is a param.

Everything is pure NumPy so the evaluator and the tests read one arithmetic. Both nodes run on the full-frame path only, as before.
"""
from __future__ import annotations

import numpy as np

GRID_OUTPUTS = ["source", "source warped", "destination", "destination warped"]
BACKGROUNDS = ["on black", "on src", "on dst", "on bg"]
_GRID_TRANSFORM = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0,
                   "skew_x": 0.0, "skew_y": 0.0, "skew_order": "XY", "center_x": 0.0, "center_y": 0.0}
GRIDWARP = {"grid_output": "source warped", "warp": 1.0, "background": "on src", "background_mix": 0.0,
            **{f"source_grid_transform_{key}": value for key, value in _GRID_TRANSFORM.items()},
            "use_source_transform": 1,
            **{f"destination_grid_transform_{key}": value for key, value in _GRID_TRANSFORM.items()},
            "render_res": 10}
SPLINEWARP = {"premultiply": 0, "outputMask": "mask_splinwarp.a", "crop_to_format": 1, "boundary_bbox": 1,
              "layer_warp": 1.0, "pair_warp": 1.0,
              "translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0,
              "skew_x": 0.0, "skew_y": 0.0, "skew_order": "XY", "center_x": 0.0, "center_y": 0.0}
# A saved SplineWarp had no mask layer and no crop, so it keeps both off.
SPLINEWARP_SAVED = {**SPLINEWARP, "outputMask": "none", "crop_to_format": 0}
LIMITS = {"background_mix": (0.0, 1.0), "warp": (0.0, 10.0), "use_source_transform": (0, 1),
          "premultiply": (0, 1), "crop_to_format": (0, 1), "boundary_bbox": (0, 1),
          "layer_warp": (0.0, 1.0), "pair_warp": (0.0, 1.0), "render_res": (1.0, 100.0),
          **{f"{grid}_grid_transform_{name}": bounds
             for grid in ("source", "destination")
             for name, bounds in (("translate_x", (-1000000.0, 1000000.0)), ("translate_y", (-1000000.0, 1000000.0)),
                                  ("rotate", (-360.0, 360.0)), ("scale", (-1000.0, 1000.0)),
                                  ("skew_x", (-1000.0, 1000.0)), ("skew_y", (-1000.0, 1000.0)),
                                  ("center_x", (-1000000.0, 1000000.0)), ("center_y", (-1000000.0, 1000000.0)))}}
CHOICES = {"grid_output": GRID_OUTPUTS, "background": BACKGROUNDS, "skew_order": ("XY", "YX"),
           "source_grid_transform_skew_order": ("XY", "YX"),
           "destination_grid_transform_skew_order": ("XY", "YX")}
OPEN_CHOICES = ("outputMask",)


def new_params(kind):
    return dict({"GridWarp": GRIDWARP, "SplineWarp": SPLINEWARP}.get(kind, {}))


def upgrade_params(kind):
    return dict({"GridWarp": GRIDWARP, "SplineWarp": SPLINEWARP_SAVED}.get(kind, {}))


def premultiplied(pixels):
    """The colour multiplied by the alpha (the alpha as it is)."""
    out = np.array(pixels, np.float32, copy=True)
    out[..., :3] *= out[..., 3:4]
    return out


def amount(p):
    """How far SplineWarp carries the picture along each pair, 0 to 1."""
    return float(np.clip(float(p.get("layer_warp", 1.0)) * float(p.get("pair_warp", 1.0)), 0.0, 1.0))


def scaled_destination(source, destination, factor):
    """The destination controls pulled back towards the source ones by `factor` (1 leaves them)."""
    if factor == 1.0 or not len(source):
        return destination
    return source + (destination - source) * np.float32(factor)


def transform_points(points, p):
    """Apply SplineWarp's Nuke-style Transform tab to both curve sets; identity is bit-preserving."""
    tx, ty = float(p.get("translate_x", 0.0)), float(p.get("translate_y", 0.0))
    rotate, scale = float(p.get("rotate", 0.0)), float(p.get("scale", 1.0))
    kx, ky = float(p.get("skew_x", 0.0)), float(p.get("skew_y", 0.0))
    if tx == ty == rotate == kx == ky == 0.0 and scale == 1.0:
        return points
    cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
    angle = np.deg2rad(-rotate)
    c, s = np.cos(angle), np.sin(angle)
    sx = np.array([[1.0, kx], [0.0, 1.0]])
    sy = np.array([[1.0, 0.0], [ky, 1.0]])
    skew = sy @ sx if p.get("skew_order", "XY") == "XY" else sx @ sy
    linear = np.array([[c, -s], [s, c]]) @ skew * scale
    pivot = np.array([cx, cy])
    offset = pivot + np.array([tx, -ty]) - linear @ pivot
    return np.asarray(points) @ linear.T + offset


def transform_grid(points, p, grid):
    """Apply a GridWarp source or destination transform to its grid control points."""
    prefix = f"{grid}_grid_transform_"
    params = {name: p.get(prefix + name, default) for name, default in _GRID_TRANSFORM.items()}
    return transform_points(points, params)


def over(top, under):
    """Premultiplied RGBA `top` over `under`."""
    return (top + under * (1.0 - top[..., 3:4])).astype(np.float32)


def black_pixels(shape):
    out = np.zeros(shape, np.float32)
    out[..., 3] = 1.0
    return out


def background_pixels(mode, shape, source, destination, background):
    """The picture the warped one goes over, fitted to the output window by the caller."""
    if mode == "on black":
        return black_pixels(shape)
    if mode == "on dst":
        return destination if destination is not None else source
    if mode == "on bg":
        return background if background is not None else np.zeros(shape, np.float32)
    return source


def with_background(warped, backdrop, mix):
    """The warped picture blended towards its composite over the background by `mix` (0 gives the warped picture back)."""
    mix = float(mix)
    if mix <= 0.0:
        return warped
    return (warped + (over(warped, backdrop) - warped) * np.float32(min(mix, 1.0))).astype(np.float32)


def moved_mask(field):
    """1 where the warp moved the picture by more than a thousandth of a pixel, else 0, as the four channels of a layer."""
    moved = (np.abs(field).max(axis=-1, keepdims=True) > 1e-3).astype(np.float32)
    return np.repeat(moved, 4, axis=-1)


def mask_layer_name(p):
    """The layer SplineWarp's `outputMask` writes (the part before the dot), or None when it is `none`."""
    token = str(p.get("outputMask", "none")).strip()
    if token.lower() in ("", "none"):
        return None
    return token.split(".")[0]
