"""Temporal coherence: carry a generated frame into the next one with the exported motion, and measure flicker.

Numpy only and free of NodeBased imports, because two sides use it: the verifier (the "temporal" line of the
verification report) and the SDXL provider's model process, which runs in its own interpreter and loads this
file by path to warp the previous picture before generating the next one. Both use the same warp and the same
mask, so the provider is guided by exactly what the verifier measures.

Conventions (ControlBundle): `motion_backward` of frame N is, per pixel of N, where its surface point was on
frame N-1 minus where it is now, in pixels (+X right, +Y down), measured at pixel centres. Pixel (x, y) has its
centre at index (x, y). Depth is camera-space Z in metres; zero, negative or not finite means no surface
(empty background). Object IDs are the exported Cryptomatte object values, 0 where there is no object.
"""
from __future__ import annotations

import numpy as np

# Two surfaces count as the same point when their depths on consecutive frames differ by at most this fraction.
DEPTH_TOLERANCE = 0.1
# Flicker pass threshold, in 8-bit display steps (0 to 255) of the per-pixel mean absolute RGB difference.
THRESHOLD_MEAN = 4.0
THRESHOLD_P95 = 12.0


def _plane(values):
    array = np.asarray(values, dtype=np.float32)
    return array[..., 0] if array.ndim == 3 else array


def sample_positions(backward_motion):
    """Where each pixel of frame N samples frame N-1: (sx, sy) float arrays, index coordinates."""
    motion = np.asarray(backward_motion, dtype=np.float32)
    h, w = motion.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return xx + motion[..., 0], yy + motion[..., 1]


def warp(previous, backward_motion):
    """Frame N-1 carried into frame N: bilinear samples of `previous` at each pixel's backward position.

    Returns (warped, inside): `inside` is False where the position falls outside frame N-1 (those pixels of
    `warped` are 0). A whole-pixel translation is copied exactly; a fractional one is exact on linear content.
    """
    src = np.asarray(previous, dtype=np.float32)
    flat = src.ndim == 2
    if flat:
        src = src[..., None]
    h, w = src.shape[:2]
    sx, sy = sample_positions(backward_motion)
    inside = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    sx = np.where(inside, sx, 0.0)
    sy = np.where(inside, sy, 0.0)
    x0 = np.clip(np.floor(sx).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(sy).astype(np.int64), 0, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (sx - x0)[..., None].astype(np.float32)
    fy = (sy - y0)[..., None].astype(np.float32)
    top = src[y0, x0] * (1 - fx) + src[y0, x1] * fx
    bottom = src[y1, x0] * (1 - fx) + src[y1, x1] * fx
    out = top * (1 - fy) + bottom * fy
    out[~inside] = 0.0
    return (out[..., 0] if flat else out), inside


def valid_mask(backward_motion, depth, ids, previous_depth, previous_ids, depth_tolerance=DEPTH_TOLERANCE):
    """Pixels of frame N whose warped value is trustworthy: inside frame N-1, not occluded, the same object.

    A pixel is kept when every one of its four bilinear neighbours on frame N-1 has the pixel's own object ID
    (so no colour is mixed across an object edge and an object that appears or disappears is left out), and
    their depth agrees with the pixel's depth within `depth_tolerance` (a surface that was hidden behind a
    nearer one on frame N-1 is an occlusion) or, for empty background, has no depth either.
    """
    depth, ids = _plane(depth), _plane(ids)
    previous_depth, previous_ids = _plane(previous_depth), _plane(previous_ids)
    h, w = depth.shape
    sx, sy = sample_positions(backward_motion)
    inside = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    sx = np.where(inside, sx, 0.0)
    sy = np.where(inside, sy, 0.0)
    x0 = np.clip(np.floor(sx).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(sy).astype(np.int64), 0, h - 1)
    # A neighbour with zero weight (a whole-pixel position) does not count.
    x1 = np.where(sx > x0, np.minimum(x0 + 1, w - 1), x0)
    y1 = np.where(sy > y0, np.minimum(y0 + 1, h - 1), y0)
    surface = np.isfinite(depth) & (depth > 0)
    keep = inside.copy()
    for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1)):
        other_depth = previous_depth[yy, xx]
        other_surface = np.isfinite(other_depth) & (other_depth > 0)
        same_depth = np.where(surface, other_surface & (np.abs(np.where(other_surface, other_depth, 0.0) - depth)
                                                       <= depth_tolerance * np.where(surface, depth, 1.0)),
                              ~other_surface)
        keep &= (previous_ids[yy, xx] == ids) & same_depth
    return keep


def display_code_values(rgb):
    """Linear RGB to 8-bit display code values (0 to 255, float): clip to 0..1, sRGB transfer, times 255."""
    v = np.clip(np.asarray(rgb, dtype=np.float32)[..., :3], 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055) * 255.0


def pair_difference(generated_previous, generated, backward_motion, depth, ids, previous_depth, previous_ids,
                    to_display=display_code_values):
    """Per-pixel colour difference between frame N and frame N-1 warped into it, over the valid pixels.

    Returns (differences, mask): `differences` is a 1-D array of the mean absolute RGB difference in 8-bit
    display steps for each valid pixel. `to_display` turns the pictures into display code values first.
    """
    warped, _ = warp(np.asarray(generated_previous, dtype=np.float32)[..., :3], backward_motion)
    mask = valid_mask(backward_motion, depth, ids, previous_depth, previous_ids)
    a = to_display(warped[mask])
    b = to_display(np.asarray(generated, dtype=np.float32)[..., :3][mask])
    return np.abs(a - b).mean(axis=-1), mask


def flicker(frames, generated, controls, to_display=display_code_values, threshold_mean=THRESHOLD_MEAN,
            threshold_p95=THRESHOLD_P95):
    """The shot's flicker measure: every consecutive pair, pooled over the valid pixels of all pairs.

    `frames` are frame numbers in order; `generated[frame]` an HxWxRGB(A) picture; `controls[frame]` a dict
    with `motion_backward`, `depth` and `ids` planes of the same size. Pairs whose frame numbers are not
    consecutive are skipped. Returns the report entry: shot mean and 95th percentile, the pass threshold and
    verdict, the share of pixels compared and one row per pair.
    """
    pairs, pooled, compared, total = [], [], 0, 0
    for before, after in zip(frames, frames[1:]):
        if after != before + 1:
            continue
        c, p = controls[after], controls[before]
        diffs, mask = pair_difference(generated[before], generated[after], np.asarray(c["motion_backward"])[..., :2],
                                      c["depth"], c["ids"], p["depth"], p["ids"], to_display)
        compared += int(mask.sum()); total += mask.size
        pooled.append(diffs)
        pairs.append({"frames": [int(before), int(after)], "pixels": int(mask.sum()),
                      "mean": round(float(diffs.mean()), 4) if diffs.size else None,
                      "p95": round(float(np.percentile(diffs, 95)), 4) if diffs.size else None})
    values = np.concatenate(pooled) if pooled else np.zeros(0, np.float32)
    mean = round(float(values.mean()), 4) if values.size else None
    p95 = round(float(np.percentile(values, 95)), 4) if values.size else None
    return {"unit": "8-bit display steps (mean absolute RGB difference per pixel)",
            "mean": mean, "p95": p95, "threshold_mean": threshold_mean, "threshold_p95": threshold_p95,
            "pass": None if mean is None else bool(mean <= threshold_mean and p95 <= threshold_p95),
            "pairs_measured": len(pairs), "compared_fraction": round(compared / total, 4) if total else 0.0,
            "pairs": pairs}
