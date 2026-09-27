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
    qd2 = np.sum((query[:, None, :] - destination[None, :, :]) ** 2, axis=2)
    weights = np.exp(-qd2 / (2.0 * sigma * sigma))
    field = (weights @ coeff).reshape(xx.shape + (2,))
    return field.astype(np.float32)


def grid_controls(source_grid, destination_grid):
    """Flatten matching MxNx2 grids into paired control arrays."""
    source = np.asarray(source_grid, dtype=np.float64)
    destination = np.asarray(destination_grid, dtype=np.float64)
    if source.ndim != 3 or source.shape[-1] != 2 or destination.shape != source.shape:
        raise ValueError("source and destination grids must have matching MxNx2 shape")
    if source.shape[0] < 2 or source.shape[1] < 2:
        raise ValueError("warp grids need at least 2 rows and 2 columns")
    return source.reshape((-1, 2)), destination.reshape((-1, 2))
