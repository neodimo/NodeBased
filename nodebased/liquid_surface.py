"""Liquid surface extraction for FLIP particles (Lane 6, step E): a level set, a mesh and foam.

`level_set` blends the particles into a signed distance field (Zhu and Bridson 2005): at a grid point x the
kernel-weighted mean position and radius of the particles within `support` give phi = |x - mean| - radius, negative
inside the liquid. The field is sampled at the cell centres of a grid of `shape` cells of `voxel` world units at
`origin`, like `scene3d.Volume`. `marching_tetrahedra` turns it into a closed, watertight `Geometry` with outward
smooth normals (each cube of cell centres is split into the six tetrahedra of the Kuhn triangulation, whose faces
match between neighbours, and the field is padded with air on every side so the surface always closes). `curvature`
is the mean curvature div(grad phi / |grad phi|) in 1 / world unit, and `foam_mask` tags the fast, sharply curved
particles that a splash throws off.
"""
from __future__ import annotations

import numpy as np

# Kuhn triangulation of a cube whose corners are numbered x + 2 y + 4 z: six tetrahedra sharing the 0-7 diagonal.
_TETS = ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7))
_CORNER = np.array([(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)], np.int64)
FAR = 1.0e3


def kernel_offsets(reach):
    r = np.arange(-reach + 1, reach + 1)
    return [(a, b, c) for a in r for b in r for c in r]


def level_set(positions, origin, voxel, shape, radius, support, smoothing=0):
    """(shape) float32 signed distance to the blended particle surface at the cell centres of the grid.
    `positions` are world units; `radius` and `support` (the kernel reach, at least twice `radius`) too."""
    shape = tuple(int(n) for n in shape)
    total = int(np.prod(shape))
    phi = np.full(total, FAR, np.float64)
    if len(positions):
        p = (np.asarray(positions, np.float64) - np.asarray(origin, np.float64)) / voxel - 0.5   # cell-centre coordinates
        reach = max(1, int(np.ceil(support / voxel)))
        base = np.floor(p).astype(np.int64)
        s = float(support / voxel)
        k_sum = np.zeros(total)
        mean = np.zeros((3, total))
        for off in kernel_offsets(reach):
            idx = base + np.array(off)
            d = idx - p
            d2 = (d * d).sum(axis=1)
            k = np.clip(1.0 - d2 / (s * s), 0.0, None) ** 3
            ok = (k > 0) & np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
            if not ok.any():
                continue
            flat = np.ravel_multi_index(tuple(idx[ok].T), shape)
            kk = k[ok]
            k_sum += np.bincount(flat, weights=kk, minlength=total)
            for c in range(3):
                mean[c] += np.bincount(flat, weights=kk * p[ok, c], minlength=total)
        have = k_sum > 1e-12
        grid = np.stack(np.unravel_index(np.arange(total), shape)).astype(np.float64)
        dist = np.sqrt(sum((grid[c][have] - mean[c][have] / k_sum[have]) ** 2 for c in range(3))) * voxel
        phi[have] = dist - radius
    phi = np.minimum(phi.reshape(shape), 2.0 * support)         # far air must not leak into the smoothing
    for _ in range(int(smoothing)):
        phi = box_smooth(phi)
    return phi.astype(np.float32)


def box_smooth(phi):
    """One 3 x 3 x 3 box average, edge-clamped."""
    pad = np.pad(phi, 1, mode="edge")
    out = np.zeros_like(phi)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            for c in (0, 1, 2):
                out += pad[a:a + phi.shape[0], b:b + phi.shape[1], c:c + phi.shape[2]]
    return out / 27.0


def gradient(phi, voxel):
    return np.stack(np.gradient(phi.astype(np.float64), voxel), axis=-1)


def curvature(phi, voxel):
    """Mean curvature of the level sets of `phi`, div(grad phi / |grad phi|), 1 / world unit."""
    g = gradient(phi, voxel)
    n = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    return sum(np.gradient(n[..., a], voxel, axis=a) for a in range(3))


def _sample(field, index_coords):
    """Trilinear sample of `field` at cell-index coordinates (N, 3)."""
    from .fluid3d import Stencil
    st = Stencil(field.shape, index_coords[:, 0], index_coords[:, 1], index_coords[:, 2])
    return st.sample(field)


def marching_tetrahedra(phi, origin, voxel):
    """(vertices (V, 3) float32, triangles (T, 3) int32, normals (V, 3) float32): the phi = 0 surface as a
    closed mesh, wound counter-clockwise seen from outside (normals along +grad phi)."""
    inner = np.asarray(phi, np.float64)
    # one layer of the edge value (so liquid touching a wall carries on past it), then one of air; the mesh is
    # clamped back onto the domain box below, which caps the liquid at the wall
    padded = np.pad(np.pad(inner, 1, mode="edge"), 1, mode="constant", constant_values=FAR)
    nx, ny, nz = padded.shape
    inside = padded < 0.0
    # cubes whose eight corners are mixed
    corner_in = [inside[c[0]:nx - 1 + c[0], c[1]:ny - 1 + c[1], c[2]:nz - 1 + c[2]] for c in _CORNER]
    total = sum(x.astype(np.int8) for x in corner_in)
    cubes = np.argwhere((total > 0) & (total < 8))
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))
    if not len(cubes):
        return empty
    node = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    corner_node = np.stack([node[c[0]:nx - 1 + c[0], c[1]:ny - 1 + c[1], c[2]:nz - 1 + c[2]][tuple(cubes.T)]
                            for c in _CORNER], axis=1)                     # (C, 8) node ids
    flat_phi = padded.reshape(-1)
    coord = np.stack(np.unravel_index(np.arange(nx * ny * nz), padded.shape), axis=1)
    tris = []                                                              # (T, 3, 2) node id pairs per vertex
    for tet in _TETS:
        ids = corner_node[:, list(tet)]                                    # (C, 4)
        ins = flat_phi[ids] < 0.0
        code = ins[:, 0] * 1 + ins[:, 1] * 2 + ins[:, 2] * 4 + ins[:, 3] * 8
        for c in range(1, 15):
            sel = code == c
            if not sel.any():
                continue
            inside_v = [i for i in range(4) if (c >> i) & 1]
            outside_v = [i for i in range(4) if not (c >> i) & 1]
            t = ids[sel]
            if len(inside_v) == 1 or len(outside_v) == 1:
                a = inside_v[0] if len(inside_v) == 1 else outside_v[0]
                others = [i for i in range(4) if i != a]
                tris.append(np.stack([np.stack((t[:, a], t[:, o]), axis=1) for o in others], axis=1))
            else:
                a, b = inside_v
                c1, d = outside_v
                e = [(a, c1), (a, d), (b, d), (b, c1)]                      # the quad, in cyclic order
                q = [np.stack((t[:, i], t[:, j]), axis=1) for i, j in e]
                tris.append(np.stack((q[0], q[1], q[2]), axis=1))
                tris.append(np.stack((q[0], q[2], q[3]), axis=1))
    if not tris:
        return empty
    pairs = np.concatenate(tris)                                           # (T, 3, 2)
    key = np.sort(pairs, axis=2)
    ekey = key[..., 0].astype(np.int64) * (nx * ny * nz) + key[..., 1]
    unique, inverse = np.unique(ekey.reshape(-1), return_inverse=True)
    triangles = inverse.reshape(-1, 3).astype(np.int64)
    first = unique // (nx * ny * nz)
    second = unique % (nx * ny * nz)
    pa, pb = flat_phi[first], flat_phi[second]
    t = pa / (pa - pb)
    grid_pos = coord[first] + (coord[second] - coord[first]) * t[:, None]         # padded cell-index coordinates
    vertices = (np.asarray(origin, np.float64) + (grid_pos - 2.0 + 0.5) * voxel)
    upper = np.asarray(origin, np.float64) + np.array(inner.shape) * voxel
    vertices = np.clip(vertices, np.asarray(origin, np.float64), upper)
    grad = gradient(padded, voxel)
    normals = np.stack([_sample(grad[..., a], grid_pos) for a in range(3)], axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    # wind every triangle so that its face normal follows the field gradient (outward)
    v = vertices[triangles]
    face = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    flip = (face * normals[triangles].sum(axis=1)).sum(axis=1) < 0
    triangles[flip] = triangles[flip][:, ::-1]
    return vertices.astype(np.float32), triangles.astype(np.int32), normals.astype(np.float32)


def signed_volume(vertices, triangles):
    v = np.asarray(vertices, np.float64)[triangles]
    return float(np.einsum("ij,ij->", v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6.0)


def is_closed(triangles):
    """True when every edge belongs to exactly two triangles (a watertight surface)."""
    t = np.asarray(triangles, np.int64)
    edges = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges[:, 0] * (edges.max() + 1) + edges[:, 1], return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def relative_velocity(positions, velocities, origin, voxel, shape, reach=1):
    """Each particle's velocity minus the mean velocity of the particles within `reach` cells of it: a falling
    blob moves as one and reads zero, a droplet thrown out of it does not."""
    idx = np.floor((np.asarray(positions, np.float64) - np.asarray(origin, np.float64)) / voxel).astype(np.int64)
    for a, n in enumerate(shape):
        np.clip(idx[:, a], 0, n - 1, out=idx[:, a])
    flat = np.ravel_multi_index(tuple(idx.T), shape)
    total = int(np.prod(shape))
    count = np.bincount(flat, minlength=total).reshape(shape).astype(np.float64)
    sums = [np.bincount(flat, weights=np.asarray(velocities, np.float64)[:, c], minlength=total).reshape(shape)
            for c in range(3)]

    def box(field):
        pad = np.pad(field, reach, mode="constant")
        out = np.zeros_like(field)
        span = 2 * reach + 1
        for a in range(span):
            for b in range(span):
                for c in range(span):
                    out += pad[a:a + shape[0], b:b + shape[1], c:c + shape[2]]
        return out
    n = box(count)
    mean = np.stack([box(s).reshape(-1)[flat] for s in sums], axis=1) / np.maximum(n.reshape(-1)[flat], 1.0)[:, None]
    return np.asarray(velocities, np.float64) - mean


def foam_mask(positions, velocities, phi, origin, voxel, speed, curvature_threshold, fps, shape=None):
    """Bool per particle: moving faster than `speed` (world units per second) relative to the liquid around it
    (when `shape`, the grid, is given; otherwise absolute) where the surface curves harder than
    `curvature_threshold` (1 / world unit). `velocities` are world units per frame."""
    if not len(positions):
        return np.zeros(0, bool)
    if shape is not None:
        velocities = relative_velocity(positions, velocities, origin, voxel, shape)
    kappa = curvature(phi, voxel)
    idx = (np.asarray(positions, np.float64) - np.asarray(origin, np.float64)) / voxel - 0.5
    k = _sample(kappa.astype(np.float32), idx)
    phi_at = _sample(phi.astype(np.float32), idx)
    fast = np.linalg.norm(np.asarray(velocities, np.float64), axis=1) * float(fps) > float(speed)
    near = phi_at > -1.5 * voxel                                  # at or near the surface
    return fast & near & (k > float(curvature_threshold))


def taubin(vertices, triangles, passes, lo, hi, lam=0.5, mu=-0.53):
    """`passes` rounds of Taubin smoothing (a shrinking then an inflating uniform Laplacian step, which softens the
    lumps of a particle surface without shrinking the volume). The topology is untouched, so a closed mesh stays
    closed; vertices are kept inside the box [lo, hi]."""
    v = np.asarray(vertices, np.float64).copy()
    if not passes or not len(triangles):
        return v.astype(np.float32)
    t = np.asarray(triangles, np.int64)
    edges = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = edges[:, 0], edges[:, 1]
    degree = np.bincount(np.concatenate((a, b)), minlength=len(v)).astype(np.float64)
    n = len(v)

    def laplacian(x):
        total = np.stack([np.bincount(a, weights=x[b, c], minlength=n) + np.bincount(b, weights=x[a, c], minlength=n)
                          for c in range(3)], axis=1)
        return total / np.maximum(degree, 1.0)[:, None] - x
    for _ in range(int(passes)):
        v += lam * laplacian(v)
        v += mu * laplacian(v)
    return np.clip(v, lo, hi).astype(np.float32)


def surface_from_particles(positions, origin, voxel, shape, radius, support, resolution=1, smoothing=1):
    """((vertices, triangles, normals), phi): the mesh of the particle surface and its level set. The surface grid
    is the solver's grid subdivided `resolution` times per axis; `smoothing` rounds of Taubin smoothing soften it."""
    res = max(1, int(resolution))
    fine = tuple(n * res for n in shape)
    phi = level_set(positions, origin, voxel / res, fine, radius, support)
    vertices, triangles, normals = marching_tetrahedra(phi, origin, voxel / res)
    lo = np.asarray(origin, np.float64)
    vertices = taubin(vertices, triangles, smoothing, lo, lo + np.array(fine) * (voxel / res))
    return (vertices, triangles, normals), phi
