"""SmartVector's estimate knobs: Strength, Flicker Compensation, the three luminance weights, Inpaint Matte Region and Matte Dilation.

Nuke 17 Reference Guide, SmartVector:

* Weight Red/Green/Blue (0.3, 0.6, 0.1): "much of the motion estimation is done on luminance only ... the tolerances allow you to tune the
  weight of each color channel when calculating the image luminance". The estimate reads `weight . rgb`. Saved nodes keep the Rec. 709
  weights the solver always used (the 30 to 31 migration writes them), so their vectors are bit for bit what they were.
* Flicker Compensation: "takes into account variations in luminance and overall flickering". Every frame is scaled, per channel, to the
  mean of the reference frame before the motion is estimated, so a brightness step between frames does not read as motion.
* Strength (1.5): "the strength in matching pixels between frames. Higher values ... concentrating on detail matching even if the resulting
  motion field is jagged. Lower values may miss local detail ... producing smoother results". The solver's data term and smoothness weigh
  against each other, so a strength lower than 1.5 raises the smoothness by 1.5 over strength and a higher one lowers it; 1.5 is the
  solver as it was.
* Inpaint Matte Region (off): "any areas where vectors are absent are infilled using the nearest available vectors". The absent side is the
  one Output does not pass on; with the control on its vectors are copied from the nearest present pixel and its coverage is 1.
* Matte Dilation (50): "the amount of dilation applied to the matte before inpainting the matte region", available only with Inpaint Matte
  Region. The absent region grows by that many pixels, so the vectors nearest the matte's edge, which the matte may not have cut cleanly,
  are infilled as well. The guide does not say which side of the matte grows; the absent side does, whichever Output keeps.
"""
import numpy as np

REC709 = (0.2126, 0.7152, 0.0722)
STRENGTH = 1.5
MAX_SMOOTHNESS = 8.0


def weights(params):
    return tuple(float(params.get(name, default)) for name, default in
                 zip(("weight_red", "weight_green", "weight_blue"), REC709))


def weighted(frames, params):
    """The frames with the estimate's luminance replaced by `weight . rgb`: every colour channel holds it, so the solver's own luminance
    (weights summing to one) reads it back, on the CPU and the GPU path alike. Rec. 709 weights return the frames untouched."""
    w = weights(params)
    if w == REC709:
        return frames
    vector = np.asarray(w, np.float32)
    out = []
    for frame in frames:
        frame = np.asarray(frame, np.float32)
        luma = frame[..., :3] @ vector
        copy = frame.copy()
        copy[..., :3] = luma[..., None] / np.float32(sum(w) or 1.0)
        out.append(copy)
    return out


def flicker_compensated(frames, reference_index):
    """Each frame scaled per channel to the reference frame's mean."""
    from .retimeknobs import estimation_gain
    reference = frames[reference_index]
    return [f if i == reference_index else estimation_gain(reference, f) for i, f in enumerate(frames)]


def effective_smoothness(smoothness, strength):
    strength = float(strength)
    if strength == STRENGTH:
        return float(smoothness)
    return float(min(MAX_SMOOTHNESS, float(smoothness) * STRENGTH / max(strength, 1e-3)))


def inpaint_plan(present, dilation, distance_transform):
    """(region to infill, row index, column index of the nearest kept pixel) for the 1/0 `present` plane, or None when nothing would
    change. `distance_transform` is `Evaluator._distance_transform`."""
    absent = present < 0.5
    if not absent.any():
        return None
    distance, _, _ = distance_transform(absent)
    region = distance <= float(dilation)
    kept = ~region
    if not kept.any():
        return None
    _, rows, cols = distance_transform(kept)
    return region, rows, cols


def infill(field, plan):
    """`field` (h x w x 2 or 4) with the region copied from the nearest kept pixel's value."""
    region, rows, cols = plan
    out = np.array(field, np.float32, copy=True)
    out[region] = field[rows[region], cols[region]]
    return out
