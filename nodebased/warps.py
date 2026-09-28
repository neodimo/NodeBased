"""Smooth 2D displacement fields shared by SplineWarp and GridWarp.

The controls are paired source/destination samples. A Gaussian radial-basis interpolant
fits the displacement exactly at every control and decays smoothly to zero away from it.
This is preferable here to a global thin-plate spline: spline controls are local edits,
and extrapolating an affine component across the whole frame would move unrelated pixels.
"""
from __future__ import annotations

import numpy as np


def displacement_field(source_points, destination_points, x, y, radius=32.0):
    """Evaluate the inverse-sampling displacement at pixel coordinates ``(x, y)``.

    ``source_points`` and ``destination_points`` are Nx2 pixel-centre coordinates. The
    returned dx/dy are destination minus source, so a destination sample at p reads the
    source at p - displacement. The RBF kernel is exp(-r² / (2 sigma²)); solving for its
    coefficients makes every control exact while retaining a smooth, decaying field.
    """
    source = np.asarray(source_points, dtype=np.float64).reshape((-1, 2))
    destination = np.asarray(destination_points, dtype=np.float64).reshape((-1, 2))
    if source.shape != destination.shape or not len(source):
        raise ValueError("warp controls must be equally sized, non-empty Nx2 arrays")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("warp controls must be finite")
    sigma = float(radius)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("curve_resolution/radius must be positive and finite")
    delta = destination - source
    if np.max(np.abs(delta), initial=0.0) == 0.0:
        return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)) + (2,), np.float32)
    d2 = np.sum((destination[:, None, :] - destination[None, :, :]) ** 2, axis=2)
    gram = np.exp(-d2 / (2.0 * sigma * sigma))
    # A tiny diagonal regulariser makes coincident samples deterministic; the solve remains
    # effectively exact for ordinary, distinct editor handles.
    coeff = np.linalg.solve(gram + np.eye(len(gram)) * 1e-10, delta)
    xx, yy = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    query = np.stack((xx.ravel(), yy.ravel()), axis=1)
    # Bound the temporary query-by-control matrix: a full-HD frame with a dense warp grid
    # would otherwise allocate several gigabytes in one broadcast.
    flat = np.empty((len(query), 2), dtype=np.float64)
    chunk_size = max(1, 1_000_000 // len(destination))
    for start in range(0, len(query), chunk_size):
        points = query[start:start + chunk_size]
        qd2 = np.sum((points[:, None, :] - destination[None, :, :]) ** 2, axis=2)
        weights = np.exp(-qd2 / (2.0 * sigma * sigma))
        flat[start:start + len(points)] = weights @ coeff
    field = flat.reshape(xx.shape + (2,))
    return field.astype(np.float32)


def grid_controls(source_grid, destination_grid):
    """Flatten paired grids, sampling their horizontal/vertical Bezier-tangent cell edges."""
    source = np.asarray(source_grid, dtype=np.float64)
    destination = np.asarray(destination_grid, dtype=np.float64)
    if source.ndim != 3 or source.shape[-1] not in (2, 6) or destination.shape != source.shape:
        raise ValueError("source and destination grids must have matching MxNx2 or MxNx6 shape")
    if source.shape[0] < 2 or source.shape[1] < 2:
        raise ValueError("warp grids need at least 2 rows and 2 columns")
    if source.shape[-1] == 2:
        return source.reshape((-1, 2)), destination.reshape((-1, 2))
    src_samples, dst_samples = [], []
    for grid_src, grid_dst in ((source, destination), (source.transpose(1, 0, 2), destination.transpose(1, 0, 2))):
        for row_src, row_dst in zip(grid_src, grid_dst):
            if len(row_src) >= 2:
                src_samples.append(sample_curve(row_src, 4))
                dst_samples.append(sample_curve(row_dst, 4))
    src = np.concatenate(src_samples)
    dst = np.concatenate(dst_samples)
    # Horizontal and vertical patches share corner samples. Merge those exact duplicates so the
    # RBF matrix stays well-conditioned while retaining the common corner's displacement.
    unique, inverse = np.unique(dst, axis=0, return_inverse=True)
    delta = dst - src
    summed = np.zeros((len(unique), 2), np.float64)
    counts = np.zeros(len(unique), np.float64)
    np.add.at(summed, inverse, delta)
    np.add.at(counts, inverse, 1.0)
    return unique - summed / counts[:, None], unique


def default_grid(width, height, rows=5, columns=5):
    """Return a regular pixel-centre grid spanning the display window."""
    rows, columns = int(rows), int(columns)
    if rows < 2 or columns < 2:
        raise ValueError("warp grids need at least 2 rows and 2 columns")
    xs = np.linspace(0.5, max(0.5, float(width) - 0.5), columns)
    ys = np.linspace(0.5, max(0.5, float(height) - 0.5), rows)
    x, y = np.meshgrid(xs, ys)
    return np.stack((x, y), axis=-1).tolist()


def sample_curve(points, resolution=12):
    """Sample an open Bezier curve into matching parameter-space control points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 6 or len(pts) < 2:
        raise ValueError("a spline curve requires at least two six-value points")
    resolution = max(2, int(resolution))
    samples = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        p0, p3 = a[:2], b[:2]
        p1, p2 = p0 + a[4:6], p3 + b[2:4]
        for j in range(resolution):
            t = j / resolution
            u = 1.0 - t
            samples.append(u**3*p0 + 3*u*u*t*p1 + 3*u*t*t*p2 + t**3*p3)
    samples.append(pts[-1, :2])
    return np.asarray(samples, dtype=np.float64)
