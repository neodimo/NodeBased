"""Axis-aligned slice through a `scene3d.Volume`, for the artist's slice viewer.

The field list is built from what the `Volume` actually carries (never a hard-coded list): density
is always present; temperature, velocity and flame are optional, and appear only when the solver
that produced the volume filled them in. `speed`, `vorticity` and `divergence` are derived from
`velocity` when it is present. A future channel (fuel, whitewater, a liquid SDF, a collider mask)
falls in the same way, once a `Volume` actually carries it: nothing here names it ahead of time.
"""
from __future__ import annotations

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}
DERIVED_FROM_VELOCITY = ("speed", "vorticity", "divergence")


def available_fields(volume) -> list[str]:
    """The field names this `Volume` actually carries, in a stable display order."""
    fields = ["density"]
    if volume.temperature is not None:
        fields.append("temperature")
    if volume.velocity is not None:
        fields.extend(DERIVED_FROM_VELOCITY)
    if volume.flame is not None:
        fields.append("flame")
    return fields


def _divergence(velocity, voxel_size):
    u, v, w = velocity[..., 0], velocity[..., 1], velocity[..., 2]
    du = np.gradient(u, voxel_size, axis=0)
    dv = np.gradient(v, voxel_size, axis=1)
    dw = np.gradient(w, voxel_size, axis=2)
    return (du + dv + dw).astype(np.float32)


def _vorticity_magnitude(velocity, voxel_size):
    u, v, w = velocity[..., 0], velocity[..., 1], velocity[..., 2]
    dwdy, dvdz = np.gradient(w, voxel_size, axis=1), np.gradient(v, voxel_size, axis=2)
    dudz, dwdx = np.gradient(u, voxel_size, axis=2), np.gradient(w, voxel_size, axis=0)
    dvdx, dudy = np.gradient(v, voxel_size, axis=0), np.gradient(u, voxel_size, axis=1)
    cx, cy, cz = dwdy - dvdz, dudz - dwdx, dvdx - dudy
    return np.sqrt(cx * cx + cy * cy + cz * cz).astype(np.float32)


def scalar_field(volume, field_name: str) -> np.ndarray:
    """The full (nx, ny, nz) array for `field_name`. Raises `ValueError` for a field this volume
    does not carry (see `available_fields`)."""
    if field_name not in available_fields(volume):
        raise ValueError(f"volume does not carry field {field_name!r}")
    if field_name == "density":
        return volume.density
    if field_name == "temperature":
        return volume.temperature
    if field_name == "flame":
        return volume.flame
    if field_name == "speed":
        return np.linalg.norm(volume.velocity, axis=-1).astype(np.float32)
    if field_name == "vorticity":
        return _vorticity_magnitude(volume.velocity, volume.voxel_size)
    if field_name == "divergence":
        return _divergence(volume.velocity, volume.voxel_size)
    raise ValueError(f"unknown field {field_name!r}")


def slice_index_count(volume, axis: str) -> int:
    return int(volume.density.shape[AXES[axis]])


def clamp_index(volume, axis: str, index: int) -> int:
    n = slice_index_count(volume, axis)
    return max(0, min(n - 1, int(index)))


def slice_plane(volume, axis: str, index: int, field_name: str) -> np.ndarray:
    """The 2D float32 plane of `field_name` at `index` along `axis` (rows, cols), where the two
    remaining axes, in their natural (x, y, z) order, become (rows, cols)."""
    data = scalar_field(volume, field_name)
    return np.take(data, clamp_index(volume, axis, index), axis=AXES[axis]).astype(np.float32)


def sample_value(volume, axis: str, index: int, field_name: str, row: int, col: int) -> float:
    """The scalar value under the cursor at (row, col) of the current slice's plane."""
    plane = slice_plane(volume, axis, index, field_name)
    row = max(0, min(plane.shape[0] - 1, int(row)))
    col = max(0, min(plane.shape[1] - 1, int(col)))
    return float(plane[row, col])


def in_plane_velocity(volume, axis: str, index: int, step: int = 4):
    """(rows, cols, 2) in-plane velocity components, subsampled every `step` cells for arrows, or
    `None` when the volume carries no `velocity`."""
    if volume.velocity is None:
        return None
    index = clamp_index(volume, axis, index)
    plane_vel = np.take(volume.velocity, index, axis=AXES[axis])
    in_plane_axes = [a for a in range(3) if a != AXES[axis]]
    return plane_vel[..., in_plane_axes][::step, ::step]


def colorize(plane: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """uint8 (rows, cols, 3) RGB colour ramp of `plane`: black at the low end, through blue and red,
    to yellow at the high end. `vmin`/`vmax` default to the plane's own range."""
    lo = float(np.min(plane)) if vmin is None else float(vmin)
    hi = float(np.max(plane)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(2.0 * t - 0.5, 0.0, 1.0)
    g = np.clip(np.clip(2.0 * t - 1.0, 0.0, 1.0) * 1.6, 0.0, 1.0)
    b = np.clip(1.5 - 2.0 * t, 0.0, 1.0)
    rgb = np.stack((r, g, b), axis=-1)
    return (rgb * 255.0).astype(np.uint8)
