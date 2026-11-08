"""Frame-global MarkerRemoval fill kernel."""
from __future__ import annotations

import math
import numpy as np


def marker_removal(image, control, mode="bilinear", linearsearch=100):
    src = np.asarray(image, dtype=np.float32)
    mask = np.asarray(control)[..., 0] > 0
    if not mask.any():
        return src.copy()
    h, w = mask.shape
    out = src.copy()
    limit = max(0, int(linearsearch))
    fallback = None
    rays = [(math.cos(2*math.pi*i/16), math.sin(2*math.pi*i/16)) for i in range(16)]
    for y, x in zip(*np.nonzero(mask)):
        estimates=[]
        if mode == "bilinear":
            horizontal=[]; vertical=[]
            for direction, (dy, dx, maximum) in enumerate(((0, -1, w), (0, 1, w), (-1, 0, limit), (1, 0, limit))):
                found=None
                for d in range(1, maximum+1):
                    yy, xx = y+dy*d, x+dx*d
                    if 0 <= yy < h and 0 <= xx < w and not mask[yy, xx]:
                        found=(d, src[yy, xx]); break
                if found is not None:
                    (horizontal if direction < 2 else vertical).append(found)
            groups=[g for g in (horizontal,vertical) if g]
            if groups:
                # Blend horizontal and vertical pairs independently, inverse-distance weighted.
                means=[sum(v/d for d,v in g)/sum(1/d for d,_ in g) for g in groups]
                weights=[sum(1/d for d,_ in g) for g in groups]
                out[y,x]=np.average(means,axis=0,weights=weights)
                continue
        else:
            for dy, dx in rays:
                for d in range(1, max(h,w)+1):
                    yy, xx = int(round(y+dy*d)), int(round(x+dx*d))
                    if not (0 <= yy < h and 0 <= xx < w): break
                    if not mask[yy,xx]:
                        estimates.append((math.hypot(yy-y,xx-x),src[yy,xx])); break
            if estimates:
                out[y,x]=sum(v/d for d,v in estimates)/sum(1/d for d,_ in estimates)
                continue
        from .flow_nodes import spatial_fill
        if fallback is None:
            fallback = spatial_fill(src, mask.astype(np.float32), "diffusion")
        out[y,x]=fallback[y,x]
    return out
