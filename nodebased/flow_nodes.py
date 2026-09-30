"""Full-frame helpers for SmartVector, vector-driven warps and flow-guided inpaint."""
from __future__ import annotations

import numpy as np

from .opticalflow import _sample, flow_pair


def vector_layers_to_motion(source_layers, forward_name="smartvector.forward",
                            backward_name="smartvector.backward"):
    """Expose SmartVector's two RG vector planes under VectorBlur's named-layer convention."""
    from .imaging import Raster
    output = dict(source_layers or {})
    for direction, name in (("forward", forward_name), ("backward", backward_name)):
        field = output.get(name)
        if field is None:
            raise ValueError(f"VectorToMotion: missing SmartVector layer {name!r}")
        rgba = np.zeros((*field.pixels.shape[:2], 4), np.float32)
        rgba[..., :2] = field.pixels[..., :2]
        rgba[..., 3] = 1.0
        motion = Raster.of(rgba, field.display)
        output[f"vector.{direction}"] = motion
        output[f"motion.{direction}"] = motion
    return output


def accumulated_vectors(frames, reference_index, *, vector_detail=4, smoothness=1.0,
                       reanchor_interval=5, backend="cpu"):
    """Return ref->each-frame and each-frame->ref fields, chaining pairs with periodic anchors."""
    if not frames or not 0 <= reference_index < len(frames):
        raise ValueError("reference_index must name a supplied frame")
    h, w = np.asarray(frames[reference_index]).shape[:2]
    forward = [np.zeros((h, w, 2), np.float32) for _ in frames]
    backward = [np.zeros((h, w, 2), np.float32) for _ in frames]
    for direction in (-1, 1):
        indices = range(reference_index + direction, len(frames), direction) if direction > 0 else range(reference_index - 1, -1, -1)
        prev = reference_index
        anchor = reference_index
        accum = np.zeros((h, w, 2), np.float32)
        for index in indices:
            if direction > 0:
                step, reverse, occ = flow_pair(frames[prev], frames[index], vector_detail=vector_detail,
                                               smoothness=smoothness, backend=backend)
                yy, xx = np.mgrid[:h, :w].astype(np.float32)
                accum = step + _sample(accum, xx - step[..., 0], yy - step[..., 1])
                # Re-anchor the chain over short intervals. A single reference-to-frame solve
                # becomes ambiguous after large motion; short anchor segments bound drift.
                if reanchor_interval and abs(index-anchor) >= reanchor_interval:
                    anchor_accum = (forward[anchor] if direction > 0 else backward[anchor])
                    direct, _, direct_occ = flow_pair(frames[anchor], frames[index],
                                                       vector_detail=vector_detail, smoothness=smoothness,
                                                       backend=backend)
                    qx, qy = xx + anchor_accum[..., 0], yy + anchor_accum[..., 1]
                    candidate = anchor_accum + _sample(direct, qx, qy)
                    visible = _sample(direct_occ.astype(np.float32), qx, qy) < .5
                    accum = np.where(visible[..., None], candidate, accum)
                    anchor = index
                forward[index] = accum.copy()
                previous_back = backward[prev]
                backward[index] = reverse + _sample(previous_back, xx + reverse[..., 0], yy + reverse[..., 1])
            else:
                step, reverse, occ = flow_pair(frames[prev], frames[index], vector_detail=vector_detail,
                                               smoothness=smoothness, backend=backend)
                yy, xx = np.mgrid[:h, :w].astype(np.float32)
                accum = step + _sample(accum, xx - step[..., 0], yy - step[..., 1])
                if reanchor_interval and abs(index-anchor) >= reanchor_interval:
                    anchor_accum = backward[anchor]
                    direct, _, direct_occ = flow_pair(frames[anchor], frames[index],
                                                       vector_detail=vector_detail, smoothness=smoothness,
                                                       backend=backend)
                    qx, qy = xx + anchor_accum[..., 0], yy + anchor_accum[..., 1]
                    candidate = anchor_accum + _sample(direct, qx, qy)
                    visible = _sample(direct_occ.astype(np.float32), qx, qy) < .5
                    accum = np.where(visible[..., None], candidate, accum)
                    anchor = index
                backward[index] = accum.copy()
                previous_forward = forward[prev]
                forward[index] = reverse + _sample(previous_forward, xx + reverse[..., 0], yy + reverse[..., 1])
            prev = index
    return forward, backward


def warp_by_flow(image, flow, *, blur_size=0.0):
    """Backward-sample a reference image using a ref->target displacement field."""
    image = np.asarray(image, np.float32); flow = np.asarray(flow, np.float32)
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x = xx - flow[..., 0]; y = yy - flow[..., 1]
    result = _sample(image, x, y)
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    if image.ndim == 3 and result.ndim == 2:
        result = result[..., None]
    result = np.where(valid[..., None] if result.ndim == 3 else valid, result, 0)
    if blur_size > 0:
        # Tiny separable box blur; only applied to the warped paint layer.
        radius = max(1, int(round(blur_size)))
        if result.ndim == 3:
            padded = np.pad(result, ((radius, radius), (0, 0), (0, 0)), mode="edge")
            result = np.mean([padded[radius+dy:radius+dy+h] for dy in range(-radius, radius+1)], axis=0)
            padded = np.pad(result, ((0, 0), (radius, radius), (0, 0)), mode="edge")
            result = np.mean([padded[:, radius+dx:radius+dx+w] for dx in range(-radius, radius+1)], axis=0)
    return result.astype(np.float32)


def homography_from_corners(source, target):
    """Solve a projective mapping from four source corners to four target corners."""
    src = np.asarray(source, np.float64); dst = np.asarray(target, np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("corner pin needs four source and four destination points")
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.extend(([x, y, 1, 0, 0, 0, -u*x, -u*y, -u], [0, 0, 0, x, y, 1, -v*x, -v*y, -v]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    return vt[-1].reshape(3, 3)


def warp_by_homography(image, matrix):
    """Inverse-project an RGBA image into the target canvas with bilinear sampling."""
    image = np.asarray(image, np.float32); h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    inv = np.linalg.inv(np.asarray(matrix, np.float64))
    den = inv[2, 0]*xx + inv[2, 1]*yy + inv[2, 2]
    sx = (inv[0, 0]*xx + inv[0, 1]*yy + inv[0, 2]) / np.where(abs(den) < 1e-10, 1e-10, den)
    sy = (inv[1, 0]*xx + inv[1, 1]*yy + inv[1, 2]) / np.where(abs(den) < 1e-10, 1e-10, den)
    out = _sample(image, sx.astype(np.float32), sy.astype(np.float32))
    valid = (sx >= -1e-4) & (sx <= w-1+1e-4) & (sy >= -1e-4) & (sy <= h-1+1e-4)
    return np.where(valid[..., None], out, 0).astype(np.float32)


def _patch_fill(image, unknown, radius=3):
    """Criminisi-style exemplar fill: confidence/data priority, known-patch SSD exemplars."""
    out = image.copy(); h, w = unknown.shape
    confidence = (~unknown).astype(np.float32)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    # A deterministic spatially distributed exemplar set bounds the search cost on large plates.
    stride = max(1, int(np.sqrt((h * w) / 10000)))
    while np.any(unknown):
        padded = np.pad(unknown, 1, mode="constant", constant_values=False)
        front = unknown & (padded[:-2,1:-1] | padded[2:,1:-1] |
                           padded[1:-1,:-2] | padded[1:-1,2:])
        ys, xs = np.nonzero(front)
        if not len(xs):
            break
        # Confidence and image-structure terms choose which boundary patch advances first.
        conf_pad = np.pad(confidence, radius, mode="edge")
        grad_x = np.zeros((h,w), np.float32); grad_y = np.zeros((h,w), np.float32)
        grad_x[:,1:-1] = np.mean(np.abs(out[:,2:] - out[:,:-2]), axis=2) * .5
        grad_y[1:-1] = np.mean(np.abs(out[2:] - out[:-2]), axis=2) * .5
        nx = np.pad(unknown[:,1:].astype(np.float32) - unknown[:,:-1], ((0,0),(0,1)))
        ny = np.pad(unknown[1:].astype(np.float32) - unknown[:-1], ((0,1),(0,0)))
        priorities = []
        for y,x in zip(ys,xs):
            y0,y1=max(0,y-radius),min(h,y+radius+1); x0,x1=max(0,x-radius),min(w,x+radius+1)
            c = float(confidence[y0:y1,x0:x1].mean())
            structure = abs(float(grad_x[y,x]*ny[y,x] - grad_y[y,x]*nx[y,x]))
            priorities.append(c * (1e-3 + structure))
        best = int(np.argmax(priorities)); cy,cx=int(ys[best]),int(xs[best])
        y0,y1=max(0,cy-radius),min(h,cy+radius+1); x0,x1=max(0,cx-radius),min(w,cx+radius+1)
        known = ~unknown[y0:y1,x0:x1]
        cand_y, cand_x = np.mgrid[radius:h-radius:stride, radius:w-radius:stride]
        cand_y, cand_x = cand_y.ravel(), cand_x.ravel()
        if not len(cand_x): cand_y, cand_x = np.array([h//2]), np.array([w//2])
        best_cost = np.inf; source = None
        for sy,sx in zip(cand_y,cand_x):
            py0,px0=sy-(cy-y0),sx-(cx-x0)
            if py0<0 or px0<0 or py0+(y1-y0)>h or px0+(x1-x0)>w: continue
            if np.any(unknown[py0:py0+y1-y0,px0:px0+x1-x0]): continue
            ref=out[py0:py0+y1-y0,px0:px0+x1-x0]
            diff=ref[known]-out[y0:y1,x0:x1][known]
            cost=float(np.mean(diff*diff)) if diff.size else 0.
            if cost < best_cost: best_cost=cost; source=(py0,px0)
        if source is None: # Tiny images: nearest known patch centre is a stable fallback.
            ky,kx=np.argwhere(~unknown)[np.argmin((np.argwhere(~unknown)[:,0]-cy)**2 + (np.argwhere(~unknown)[:,1]-cx)**2)]
            source=(max(0,min(h-(y1-y0),int(ky)-(cy-y0))), max(0,min(w-(x1-x0),int(kx)-(cx-x0))))
        sy,sx=source; patch_unknown=unknown[y0:y1,x0:x1].copy()
        target=out[y0:y1,x0:x1]; exemplar=out[sy:sy+y1-y0,sx:sx+x1-x0]
        target[patch_unknown]=exemplar[patch_unknown]; out[y0:y1,x0:x1]=target
        confidence[y0:y1,x0:x1][patch_unknown]=float(np.mean(priorities))
        unknown[y0:y1,x0:x1][patch_unknown]=False
    return out


def spatial_fill(image, matte, method="diffusion", iterations=64):
    """Fill masked RGBA pixels from known neighbours; diffusion keeps flat regions flat."""
    image = np.asarray(image, np.float32); matte = np.clip(np.asarray(matte, np.float32), 0, 1)
    if matte.ndim == 3: matte = matte[..., 0]
    unknown = matte > 1e-5
    if not np.any(unknown): return image.copy()
    if method == "patch":
        out = _patch_fill(image, unknown.copy(), radius=7)
        return image * (1 - matte[..., None]) + out * matte[..., None]
    if method != "diffusion":
        raise ValueError("fill_method must be diffusion or patch")
    out = image.copy(); known = ~unknown
    # Jacobi diffusion: average only known/currently filled 4-neighbours, preserving constants.
    for _ in range(max(1, iterations)):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        q = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * .25
        out[unknown] = q[unknown]
    return image * (1 - matte[..., None]) + out * matte[..., None]


def inpaint(current, matte, neighbours=(), method="diffusion", backend="cpu"):
    """Use aligned, warped temporal observations first, then fill residual holes spatially."""
    current = np.asarray(current, np.float32); matte = np.clip(np.asarray(matte, np.float32), 0, 1)
    if matte.ndim == 3: matte = matte[..., 0]
    h, w = current.shape[:2]
    if not neighbours: return spatial_fill(current, matte, method)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    candidates = []
    for neighbour in neighbours:
        neighbour = np.asarray(neighbour, np.float32)
        fwd, _, occ = flow_pair(current, neighbour, backend=backend)
        warped = _sample(neighbour, xx + fwd[..., 0], yy + fwd[..., 1])
        if warped.ndim == 2: warped = warped[..., None]
        candidates.append((warped, ~occ))
    accum = np.zeros_like(current); weight = np.zeros((h, w), np.float32)
    for pixels, valid in candidates:
        wt = matte * valid
        accum += pixels * wt[..., None]; weight += wt
    fillable = weight > 1e-5
    base = current.copy(); base[fillable] = accum[fillable] / weight[fillable, None]
    remaining = np.where(fillable, 0, matte)
    return spatial_fill(base, remaining, method)
