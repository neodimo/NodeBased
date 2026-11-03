"""GodRays as Nuke 17.0 describes it (lane 2, 2D behaviour step G).

"Averages together many samples of the image between its initial position and the translate and center indicators": the picture is
moved a fraction `t` of the way from where it is (t = 0) to the full translate, rotate, scale and skew about `center` (t = 1), sampled at
`steps + 1` equally spaced `t`, and the samples are averaged. Each sample is multiplied by a colour that runs from `from color` at the
first to `to color` at the last, interpolated in the gamma space `gamma` names (`(1 - t) * from^(1/gamma) + t * to^(1/gamma)`, raised to
`gamma`). With `max` on, only the brightest result of each channel is output rather than the average.

A sample at `t` is the picture scaled by `1 + (scale - 1) * t` about the centre (scale below one zooms out, so a point moves toward the
centre; above one it moves away), rotated by `rotate * t` degrees, skewed by `skew * t` and translated by `translate * t` pixels, read with
bilinear filtering and transparent black outside the picture. The guide notes that translate and skew do not merge correctly with the other
controls, so the order in which the four combine is ours (scale, skew, rotate, then translate); each is exact alone.

The picture's rows run down and `center` and `translate` are in pixels from the top-left, as the older GodRays read them. Mask and mix are
the node's own, applied by the caller. A GodRays node from an older document (no `legacy_godrays`, or 1) runs `Evaluator._godrays_legacy`.
"""
from __future__ import annotations

import math

import numpy as np

LEGACY = {"GodRays": {"legacy_godrays": 1}}
NEW = {"GodRays": {"legacy_godrays": 0, "rays_translate_x": 0.0, "rays_translate_y": 0.0, "rays_rotate": 0.0, "rays_scale": 1.0, "rays_skew": 0.0,
                   "rays_from_color": 1.0, "rays_to_color": 1.0, "rays_gamma": 1.0, "rays_max": 0}}
# An older document's node: the legacy flag, and the old step count (a saved node has its own; this is for one saved without).
UPGRADE = {"GodRays": {"legacy_godrays": 1, "rays_translate_x": 0.0, "rays_translate_y": 0.0, "rays_rotate": 0.0, "rays_scale": 1.0, "rays_skew": 0.0,
                       "rays_from_color": 1.0, "rays_to_color": 1.0, "rays_gamma": 1.0, "rays_max": 0}}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    return dict(UPGRADE.get(kind, {}))


def is_legacy(p):
    """True for what the older GodRays did: an older node (a missing flag) and VolumeRays, which runs the same smear with its matte."""
    return bool(int(p.get("legacy_godrays", 1)))


def _gamma_lerp(a, b, t, gamma):
    gamma = max(float(gamma), 1e-6)
    ga, gb = math.copysign(abs(a) ** (1.0 / gamma), a), math.copysign(abs(b) ** (1.0 / gamma), b)
    v = (1.0 - t) * ga + t * gb
    return math.copysign(abs(v) ** gamma, v)


def _bilinear(image, sx, sy):
    """`image` at the float pixel positions (sx, sy), bilinear, transparent black outside the picture."""
    h, w = image.shape[:2]
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    out = np.zeros(sx.shape + (image.shape[2],), np.float64)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            xi, yi = x0 + dx, y0 + dy
            inside = ((xi >= 0) & (xi < w) & (yi >= 0) & (yi < h))[..., None]
            taken = image[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)].astype(np.float64)
            out += np.where(inside, taken, 0.0) * wy * wx
    return out


def godrays(image, p, origin=(0, 0)):
    """Nuke's GodRays on `image`; `origin` is the array's top-left pixel on the canvas (the centre is a canvas position)."""
    steps = max(1, min(256, int(p.get("steps", 5))))
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    cx, cy = float(p.get("center_x", w / 2)) - origin[0], float(p.get("center_y", h / 2)) - origin[1]
    translate = (float(p.get("rays_translate_x", 0.0)), float(p.get("rays_translate_y", 0.0)))
    rotate, scale, skew = float(p.get("rays_rotate", 0.0)), float(p.get("rays_scale", 1.0)), float(p.get("rays_skew", 0.0))
    from_color, to_color, gamma = float(p.get("rays_from_color", 1.0)), float(p.get("rays_to_color", 1.0)), float(p.get("rays_gamma", 1.0))
    brightest = bool(int(p.get("rays_max", 0)))
    total = None
    for k in range(steps + 1):
        t = k / steps
        s = 1.0 + (scale - 1.0) * t
        angle = math.radians(rotate * t)
        # The picture's point q lands at c + R(angle) K(skew * t) S(s) (q - c) + translate * t; an output pixel reads the point that lands on it.
        m = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) @ np.array([[1.0, skew * t], [0.0, 1.0]]) \
            @ np.array([[s, 0.0], [0.0, s]])
        inv = np.linalg.inv(m) if abs(np.linalg.det(m)) > 1e-12 else np.zeros((2, 2))
        dx, dy = xx - cx - translate[0] * t, yy - cy - translate[1] * t
        sx, sy = cx + inv[0, 0] * dx + inv[0, 1] * dy, cy + inv[1, 0] * dx + inv[1, 1] * dy
        sample = _bilinear(image, sx, sy)
        sample[..., :3] *= _gamma_lerp(from_color, to_color, t, gamma)
        if total is None:
            total = sample
        elif brightest:
            total = np.maximum(total, sample)
        else:
            total = total + sample
    result = total if brightest else total / float(steps + 1)
    return result.astype(np.float32)
