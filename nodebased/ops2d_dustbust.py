"""Temporal spot replacement for the standalone DustBust node."""
import numpy as np


def dustbust(current, source, spots, frame, frame_offset=1, edge_hardness=0.5,
             view_source=False, maskchannel="none"):
    if view_source:
        return source.copy()
    out = current.copy()
    h, w = out.shape[:2]
    coverage = np.zeros((h, w), np.float32)
    hardness = float(np.clip(edge_hardness, 0, 1))
    for spot in spots or []:
        if int(spot.get("frame", -2**31)) != int(round(frame)):
            continue
        x, y, radius = float(spot["x"]), float(spot["y"]), max(0.01, float(spot["radius"]))
        x0, x1 = max(0, int(np.floor(x-radius))), min(w, int(np.ceil(x+radius+1)))
        y0, y1 = max(0, int(np.floor(y-radius))), min(h, int(np.ceil(y+radius+1)))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.sqrt((xx-x)**2 + (yy-y)**2)
        t = np.clip((radius-distance) / max(radius*(1-hardness), 1e-6), 0, 1)
        alpha = np.where(distance <= radius, t*t*(3-2*t), 0).astype(np.float32)
        if hardness >= 1:
            alpha = (distance <= radius).astype(np.float32)
        patch = out[y0:y1, x0:x1]
        patch[...] = patch * (1-alpha[..., None]) + source[y0:y1, x0:x1] * alpha[..., None]
        coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], alpha)
    if maskchannel and maskchannel.lower() != "none":
        # The evaluator attaches this as a named layer; the image remains the composited plate.
        return out, coverage
    return out
