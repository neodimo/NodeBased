"""Dependency-free coarse-to-fine Lucas--Kanade optical flow.

Vectors are in full-resolution pixels per frame (x right, y down). The solver follows
Bouguet, *Pyramidal Implementation of the Lucas Kanade Feature Tracker* (Intel, 2000),
using dense local least-squares updates and a forward/backward consistency confidence.
"""
from __future__ import annotations

import numpy as np


def _luma(image, flow_on="luminance"):
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 2:
        return a
    rgb = a[..., :3]
    if flow_on == "rgb":
        return rgb
    return rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722


def _down(a):
    h, w = a.shape[:2]
    h2, w2 = max(1, h // 2), max(1, w // 2)
    padded = np.pad(a, ((0, h % 2), (0, w % 2)) + (((0, 0),) if a.ndim == 3 else ()), mode="edge")
    shape = (padded.shape[0] // 2, 2, padded.shape[1] // 2, 2) + (() if a.ndim == 2 else (a.shape[2],))
    return padded.reshape(shape).mean(axis=(1, 3)).astype(np.float32)


def _sample(a, x, y):
    h, w = a.shape[:2]
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    x0c = np.clip(x0, 0, w - 1); x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1); y1c = np.clip(y0 + 1, 0, h - 1)
    if a.ndim == 2:
        fx = fx[..., 0]; fy = fy[..., 0]
    return ((1-fx)*(1-fy)*a[y0c, x0c] + fx*(1-fy)*a[y0c, x1c]
            + (1-fx)*fy*a[y1c, x0c] + fx*fy*a[y1c, x1c])


def _box_sum(a, radius):
    r = max(1, int(radius))
    p = np.pad(a, ((r, r), (r, r)), mode="edge")
    integ = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integ[2*r+1:, 2*r+1:] - integ[:-2*r-1, 2*r+1:] - integ[2*r+1:, :-2*r-1] + integ[:-2*r-1, :-2*r-1]


def _grad(a):
    return (np.gradient(a, axis=1).astype(np.float32),
            np.gradient(a, axis=0).astype(np.float32))


def _solve_level(a, b, flow, smoothness, iterations):
    h, w = a.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for _ in range(iterations):
        warped = _sample(b, xx + flow[..., 0], yy + flow[..., 1])
        avg = (a + warped) * .5
        ix, iy = _grad(avg)
        it = warped - a
        # RGB gradients use a summed normal matrix; luminance inputs use the same equations.
        if ix.ndim == 3:
            gxx = np.sum(ix*ix, axis=2); gyy = np.sum(iy*iy, axis=2)
            gxy = np.sum(ix*iy, axis=2)
            gxt = np.sum(ix*it, axis=2); gyt = np.sum(iy*it, axis=2)
        else:
            gxx, gyy, gxy, gxt, gyt = ix*ix, iy*iy, ix*iy, ix*it, iy*it
        win = max(3, int(round(4 + smoothness * 3)))
        sxx, syy, sxy = (_box_sum(q, win) for q in (gxx, gyy, gxy))
        sxt, syt = (_box_sum(q, win) for q in (gxt, gyt))
        det = sxx*syy - sxy*sxy
        denom = np.maximum(det, 1e-9)
        du = (-syy*sxt + sxy*syt) / denom
        dv = (sxy*sxt - sxx*syt) / denom
        valid = det > 1e-7
        flow[..., 0] += np.where(valid, du, 0).astype(np.float32)
        flow[..., 1] += np.where(valid, dv, 0).astype(np.float32)
        if smoothness:
            radius = max(2, min(5, int(round(2 * smoothness))))
            area = float((2 * radius + 1) ** 2)
            averaged = np.stack((_box_sum(flow[..., 0], radius) / area,
                                 _box_sum(flow[..., 1], radius) / area), axis=2)
            flow = (flow * .15 + averaged * .85).astype(np.float32)
    return flow


def flow_pair(first, second, *, vector_detail=4, smoothness=1.0,
              flow_on="luminance", iterations=5):
    """Return dense forward/backward fields and a forward/backward occlusion mask."""
    a, b = _luma(first, flow_on), _luma(second, flow_on)
    if a.shape != b.shape:
        raise ValueError("Optical flow frames must have matching dimensions")
    if vector_detail not in (1, 2, 3, 4, 5, 6):
        raise ValueError("vector_detail must be in 1..6")
    if smoothness < 0:
        raise ValueError("smoothness must be non-negative")
    pa, pb = [a], [b]
    for _ in range(vector_detail - 1):
        if min(pa[-1].shape[:2]) < 12:
            break
        pa.append(_down(pa[-1])); pb.append(_down(pb[-1]))
    f = np.zeros((*pa[-1].shape[:2], 2), np.float32)
    for level in range(len(pa)-1, -1, -1):
        if f.shape[:2] != pa[level].shape[:2]:
            h, w = pa[level].shape[:2]
            f = _sample(f, np.broadcast_to(np.linspace(0, f.shape[1]-1, w), (h,w)),
                        np.broadcast_to(np.linspace(0, f.shape[0]-1, h)[:,None], (h,w))).astype(np.float32) * 2
        f = _solve_level(pa[level], pb[level], f, smoothness, iterations)
    backward = flow_pair_oneway(b, a, vector_detail, smoothness, iterations)
    h, w = a.shape[:2]; yy, xx = np.mgrid[:h, :w].astype(np.float32)
    back_at = _sample(backward, xx + f[..., 0], yy + f[..., 1])
    consistency = np.linalg.norm(f + back_at, axis=2)
    occluded = consistency > (0.5 + 0.01 * np.linalg.norm(f, axis=2))
    return f, backward, occluded


def flow_pair_oneway(a, b, detail, smoothness, iterations):
    pa, pb = [a], [b]
    for _ in range(detail - 1):
        if min(pa[-1].shape[:2]) < 12: break
        pa.append(_down(pa[-1])); pb.append(_down(pb[-1]))
    f = np.zeros((*pa[-1].shape[:2], 2), np.float32)
    for level in range(len(pa)-1, -1, -1):
        if f.shape[:2] != pa[level].shape[:2]:
            h,w=pa[level].shape[:2]
            f=_sample(f,np.broadcast_to(np.linspace(0,f.shape[1]-1,w),(h,w)),
                      np.broadcast_to(np.linspace(0,f.shape[0]-1,h)[:,None],(h,w))).astype(np.float32)*2
        f=_solve_level(pa[level],pb[level],f,smoothness,iterations)
    return f
