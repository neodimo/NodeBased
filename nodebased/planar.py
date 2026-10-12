"""Deterministic robust homography fitting primitives for planar tracking."""
from __future__ import annotations

import itertools

import numpy as np


class PlanarTrackingError(ValueError):
    """Input correspondences cannot define a reliable projective transform."""


def project_points(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix, dtype=np.float64).T
    divisor = homogeneous[:, 2]
    if np.any(np.abs(divisor) < 1e-12):
        raise PlanarTrackingError("homography maps a point to infinity")
    return homogeneous[:, :2] / divisor[:, None]


def _normalise(points):
    center = points.mean(axis=0)
    distances = np.linalg.norm(points-center, axis=1)
    mean_distance = float(distances.mean())
    if mean_distance < 1e-10:
        raise PlanarTrackingError("feature points have insufficient spread")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1]], dtype=np.float64)
    return (points-center)*scale, transform


def _fit(source, destination):
    src, ts = _normalise(source)
    dst, td = _normalise(destination)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        rows.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    _, singular, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=True)
    if singular[-1] < 1e-12 and len(source) == 4:
        raise PlanarTrackingError("degenerate feature geometry")
    normalized = vh[-1].reshape(3, 3)
    result = np.linalg.inv(td) @ normalized @ ts
    if abs(result[2, 2]) < 1e-12:
        raise PlanarTrackingError("unstable homography")
    result /= result[2, 2]
    return result


def fit_homography_ransac(reference_points, current_points, threshold=2.0, max_trials=512):
    """Fit reference-to-current homography; deterministic four-point RANSAC plus refit.

    Returns ``(3x3 matrix, inlier mask, RMS inlier error)``. Candidate sampling is exhaustive
    for small sets and a fixed-seed sample for larger sets, making repeated analyses identical.
    """
    source = np.asarray(reference_points, dtype=np.float64)
    destination = np.asarray(current_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or destination.shape != source.shape:
        raise PlanarTrackingError("point arrays must have matching Nx2 shapes")
    if len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise PlanarTrackingError("at least four finite point pairs are required")
    if not np.isfinite(threshold) or threshold <= 0:
        raise PlanarTrackingError("threshold must be positive and finite")
    total = len(source)*(len(source)-1)*(len(source)-2)*(len(source)-3)//24
    if total <= max_trials:
        samples = itertools.combinations(range(len(source)), 4)
    else:
        rng = np.random.default_rng(0)
        samples = (tuple(sorted(rng.choice(len(source), 4, replace=False))) for _ in range(max_trials))
    best = None
    for sample in samples:
        try:
            candidate = _fit(source[list(sample)], destination[list(sample)])
            errors = np.linalg.norm(project_points(candidate, source)-destination, axis=1)
        except (np.linalg.LinAlgError, PlanarTrackingError):
            continue
        inliers = errors <= threshold
        score = (int(inliers.sum()), -float(np.sum(np.minimum(errors, threshold))))
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[1].sum() < 4:
        raise PlanarTrackingError("RANSAC found fewer than four consistent feature matches")
    inliers = best[1]
    matrix = _fit(source[inliers], destination[inliers])
    errors = np.linalg.norm(project_points(matrix, source[inliers])-destination[inliers], axis=1)
    return matrix, inliers, float(np.sqrt(np.mean(errors*errors)))
