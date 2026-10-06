"""Projection and triangulation helpers shared by the 2D/3D tracking nodes."""
from __future__ import annotations

import numpy as np

from . import scene3d


MIN_RAY_ANGLE_DEGREES = 1.0


def pixel_ray(camera, width, height, xy):
    """Return a normalized world-space ray for a top-left-origin pixel coordinate."""
    eye, view = scene3d._view_basis(camera)
    x, y = map(float, xy)
    focal = 1.0 / np.tan(np.deg2rad(camera.fov) / 2.0)
    aspect = float(width) / max(float(height), 1.0)
    local = np.array(((2*x/max(width, 1)-1) * aspect/focal,
                      (1-2*y/max(height, 1)) / focal, -1.0), dtype=np.float64)
    direction = view.astype(np.float64).T @ local
    direction /= np.linalg.norm(direction)
    return eye.astype(np.float64), direction


def triangulate(cameras, points, width, height, min_angle_degrees=MIN_RAY_ANGLE_DEGREES):
    """Least-squares closest intersection of camera rays; return point and per-view px residuals.

    At least two distinct views are required. The minimum pairwise ray angle is a stable,
    explicit conditioning check: below it, the input is rejected as ``too_parallel_rays``.
    """
    if len(cameras) != len(points) or len(points) < 2:
        raise ValueError("triangulation_requires_two_or_more_views")
    rays = [pixel_ray(camera, width, height, point) for camera, point in zip(cameras, points)]
    dirs = [ray[1] for ray in rays]
    max_angle = max(np.rad2deg(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0)))
                    for i, a in enumerate(dirs) for b in dirs[i+1:])
    if max_angle < float(min_angle_degrees):
        raise ValueError(f"too_parallel_rays: baseline angle {max_angle:.6g} degrees is below {min_angle_degrees:g}")
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ origin
    try:
        xyz = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("too_parallel_rays: least-squares system is singular") from exc
    projected, depths = scene3d.project(cameras[0], width, height, xyz[None, :])
    # Residuals use the actual camera per frame, not one reference camera.
    residuals = []
    for camera, point in zip(cameras, points):
        pixels, _ = scene3d.project(camera, width, height, xyz[None, :])
        residuals.append(float(np.linalg.norm(pixels[0].astype(np.float64) - np.asarray(point))))
    return xyz, residuals


def reconcile(camera, point, width, height):
    """Project one world-space point, retaining off-screen and behind-camera status."""
    pixels, depth = scene3d.project(camera, width, height, np.asarray(point, dtype=np.float64)[None, :])
    x, y = map(float, pixels[0])
    z = float(depth[0])
    return {"x": x, "y": y, "depth": z,
            "behind_camera": z <= float(camera.near),
            "off_screen": x < 0.0 or y < 0.0 or x >= width or y >= height}
