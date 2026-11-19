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


# Export Write (`export_write`, tab Export; lane 2 step X6). The guide's button "adds a Write node with EXR settings" for the vectors the node makes, which are
# written as EXR layers. Here it adds a Write on the node's output (a 32 bit float EXR of every channel, the node's frame range, zip, a lossless compression, because vectors must not be quantised), and counts the press on the node, as Reload's counter does, so the press is one change the
# document and the undo history hold. It moves no pixel of the SmartVector itself.
EXPORT_COUNTER = {"export_write": 0}


def upgrade_params(kind):
    """What a SmartVector saved before the Export Write button gets on load."""
    return dict(EXPORT_COUNTER) if kind == "SmartVector" else {}


def export_write_commands(key, nodes):
    """The commands of the Export Write button for SmartVector `key`: create a Write node with EXR settings connected to it, and count the press.
    Run as one batch they are one undo step. `nodes` is the document's node map (it holds the SmartVector and its position)."""
    node = nodes[key]
    ident, n = f"{key}-export", 1
    while ident in nodes:
        n += 1
        ident = f"{key}-export{n}"
    pos = list(node.get("pos") or [0, 0])
    params = node.get("params") or {}
    first, last = int(params.get("frame_start", 1)), int(params.get("frame_end", 100))
    return [{"op": "create", "id": ident, "type": "Write", "name": f"{node.get('name', 'SmartVector')} Write",
             "params": {"file_type": "exr", "bit_depth": "float", "channels": "all", "exr_compression": "zip", "first": first, "last": last, "use_limit": 1},
             "pos": [pos[0] + 160, pos[1]]},
            {"op": "connect", "id": ident, "input": "image", "source": key},
            {"op": "set", "id": key, "param": "export_write", "value": int(params.get("export_write", 0)) + 1}]
