"""Representation-specific Charts for E1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Fidelity, FootprintBatch, OrderedContributions, RayBatch, SampleBatch, Space
from .representations import (
    CircleSDF,
    GaussianSplats,
    GaussianVolume,
    Raster,
    TriangleSurface,
)


_GRID_4 = np.stack(
    np.meshgrid(np.array((-0.375, -0.125, 0.125, 0.375)),
                np.array((-0.375, -0.125, 0.125, 0.375))),
    axis=-1,
).reshape(-1, 2)


def _require_space(expected: Space, actual: Space) -> None:
    if expected != actual:
        raise ValueError(f"expected space {expected.name!r}, got {actual.name!r}")


def _bilinear(pixels: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = pixels.shape
    validity = np.all((points >= 0.0) & (points <= 1.0), axis=1).astype(np.float64)
    x = np.clip(points[:, 0] * width - 0.5, 0.0, width - 1.0)
    y = np.clip(points[:, 1] * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = pixels[y0, x0] * (1.0 - fx) + pixels[y0, x1] * fx
    bottom = pixels[y1, x0] * (1.0 - fx) + pixels[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy, validity


@dataclass(frozen=True)
class RasterField:
    raster: Raster
    filter_name: str

    @property
    def space(self) -> Space:
        return self.raster.space

    def sample(self, footprints: FootprintBatch) -> SampleBatch:
        _require_space(self.space, footprints.space)
        if self.filter_name == "point":
            values, validity = _bilinear(self.raster.pixels, footprints.centers)
            return SampleBatch(
                values, validity, Fidelity.HEURISTIC,
                ("point reconstruction ignores footprint area",),
            )
        if self.filter_name != "box16":
            raise ValueError(f"unknown raster reconstruction {self.filter_name!r}")
        points = footprints.centers[:, None, :] + np.einsum(
            "nij,sj->nsi", footprints.basis, _GRID_4
        )
        flat_values, flat_validity = _bilinear(
            self.raster.pixels, points.reshape(-1, 2)
        )
        values = flat_values.reshape(footprints.count, len(_GRID_4), 4).mean(axis=1)
        validity = flat_validity.reshape(footprints.count, len(_GRID_4)).mean(axis=1)
        return SampleBatch(values, validity, Fidelity.FILTERED, ("box16 filter",))


@dataclass(frozen=True)
class RasterChart:
    representation: Raster

    def reconstruct(self, filter_name: str = "box16") -> RasterField:
        return RasterField(self.representation, filter_name)


@dataclass(frozen=True)
class SDFField:
    representation: CircleSDF

    @property
    def space(self) -> Space:
        return self.representation.space

    def sample(self, footprints: FootprintBatch) -> SampleBatch:
        _require_space(self.space, footprints.space)
        center = np.asarray(self.representation.center)
        distance = np.linalg.norm(footprints.centers - center, axis=1)
        signed = self.representation.radius - distance
        # A conservative scalar width from the local differential basis.
        width = np.maximum(np.linalg.norm(footprints.basis, axis=(1, 2)) * 0.5, 1e-12)
        coverage = np.clip(0.5 + signed / width, 0.0, 1.0)
        color = np.asarray(self.representation.color)
        alpha = coverage * color[3]
        values = np.empty((footprints.count, 4), dtype=np.float64)
        values[:, :3] = color[:3] * alpha[:, None]
        values[:, 3] = alpha
        return SampleBatch(values, np.ones(footprints.count), Fidelity.APPROXIMATE,
                           ("SDF coverage uses first-order footprint width",))


@dataclass(frozen=True)
class SDFChart:
    representation: CircleSDF

    def reconstruct(self) -> SDFField:
        return SDFField(self.representation)


@dataclass(frozen=True)
class GaussianField:
    representation: GaussianSplats
    kernel_scale: float

    @property
    def space(self) -> Space:
        return self.representation.space

    def sample(self, footprints: FootprintBatch) -> SampleBatch:
        _require_space(self.space, footprints.space)
        delta = footprints.centers[:, None, :] - self.representation.centers[None, :, :]
        sigma = self.representation.sigmas[None, :] * self.kernel_scale
        weight = np.exp(-0.5 * np.sum(delta * delta, axis=2) / (sigma * sigma))
        density = weight * self.representation.colors[None, :, 3]
        total = density.sum(axis=1)
        alpha = 1.0 - np.exp(-total)
        straight_rgb = (
            density @ self.representation.colors[:, :3]
        ) / np.maximum(total[:, None], 1e-12)
        values = np.concatenate((straight_rgb * alpha[:, None], alpha[:, None]), axis=1)
        return SampleBatch(
            values,
            np.ones(footprints.count),
            Fidelity.HEURISTIC,
            ("splat reconstruction is point-evaluated and footprint-inconsistent",),
        )


@dataclass(frozen=True)
class GaussianSplatChart:
    representation: GaussianSplats

    def reconstruct(self, kernel_scale: float = 1.0) -> GaussianField:
        if kernel_scale <= 0.0:
            raise ValueError("kernel scale must be positive")
        return GaussianField(self.representation, kernel_scale)


@dataclass(frozen=True)
class TriangleSurfaceChart:
    representation: TriangleSurface

    @property
    def space(self) -> Space:
        return self.representation.space

    def ray_contributions(self, rays: RayBatch) -> OrderedContributions:
        _require_space(self.space, rays.space)
        triangles = self.representation.triangles
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        pvec = np.cross(rays.directions[:, None, :], edge2[None, :, :])
        determinant = np.einsum("tj,ntj->nt", edge1, pvec)
        usable = np.abs(determinant) > 1e-12
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
        tvec = rays.origins[:, None, :] - triangles[None, :, 0, :]
        u = np.einsum("ntj,ntj->nt", tvec, pvec) * inverse
        qvec = np.cross(tvec, edge1[None, :, :])
        v = np.einsum("nj,ntj->nt", rays.directions, qvec) * inverse
        depth = np.einsum("tj,ntj->nt", edge2, qvec) * inverse
        active = (
            usable
            & (u >= 0.0)
            & (v >= 0.0)
            & (u + v <= 1.0)
            & (depth >= rays.near[:, None])
            & (depth <= rays.far[:, None])
        )
        sortable_depth = np.where(active, depth, np.inf)
        order = np.argsort(sortable_depth, axis=1)
        sorted_depth = np.take_along_axis(sortable_depth, order, axis=1)
        sorted_active = np.take_along_axis(active, order, axis=1)
        colors = np.broadcast_to(
            self.representation.colors[None, :, :],
            (rays.count, len(triangles), 4),
        )
        colors = np.take_along_axis(colors, order[:, :, None], axis=1)
        values = colors.copy()
        values[:, :, :3] *= values[:, :, 3:4]
        values[~sorted_active] = 0.0
        return OrderedContributions(
            sorted_depth,
            values,
            sorted_active,
            np.ones(rays.count),
            Fidelity.EXACT,
            ("idealized double-sided triangle intersections",),
        )


def _intersect_bounds(
    rays: RayBatch, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = rays.near.copy()
    exit = rays.far.copy()
    hit = np.ones(rays.count, dtype=bool)
    for axis in range(3):
        direction = rays.directions[:, axis]
        origin = rays.origins[:, axis]
        parallel = np.abs(direction) < 1e-12
        hit &= ~(parallel & ((origin < bounds_min[axis]) | (origin > bounds_max[axis])))
        first = np.full(rays.count, -np.inf)
        second = np.full(rays.count, np.inf)
        np.divide(bounds_min[axis] - origin, direction, out=first, where=~parallel)
        np.divide(bounds_max[axis] - origin, direction, out=second, where=~parallel)
        entry = np.maximum(entry, np.minimum(first, second))
        exit = np.minimum(exit, np.maximum(first, second))
    hit &= exit > entry
    return entry, exit, hit


@dataclass(frozen=True)
class GaussianVolumeChart:
    representation: GaussianVolume
    integration_steps: int = 32

    def __post_init__(self) -> None:
        if self.integration_steps <= 0:
            raise ValueError("volume integration needs positive steps")

    @property
    def space(self) -> Space:
        return self.representation.space

    def ray_contributions(self, rays: RayBatch) -> OrderedContributions:
        _require_space(self.space, rays.space)
        entry, exit, hit = _intersect_bounds(
            rays,
            np.asarray(self.representation.bounds_min),
            np.asarray(self.representation.bounds_max),
        )
        length = np.where(hit, exit - entry, 0.0)
        offsets = (np.arange(self.integration_steps, dtype=np.float64) + 0.5)
        offsets /= self.integration_steps
        depth = np.where(
            hit[:, None],
            entry[:, None] + length[:, None] * offsets[None, :],
            np.inf,
        )
        safe_depth = np.where(hit[:, None], depth, 0.0)
        points = rays.origins[:, None, :] + safe_depth[:, :, None] * rays.directions[:, None, :]
        center = np.asarray(self.representation.center)
        sigma = np.asarray(self.representation.sigma)
        normalized = (points - center) / sigma
        density = self.representation.peak_density * np.exp(
            -0.5 * np.sum(normalized * normalized, axis=2)
        )
        step_length = length / self.integration_steps
        alpha = 1.0 - np.exp(-density * step_length[:, None])
        alpha *= hit[:, None]
        color = np.asarray(self.representation.color)
        values = np.empty(depth.shape + (4,), dtype=np.float64)
        values[:, :, :3] = color[None, None, :] * alpha[:, :, None]
        values[:, :, 3] = alpha
        active = np.broadcast_to(hit[:, None], depth.shape).copy()
        return OrderedContributions(
            depth,
            values,
            active,
            np.ones(rays.count),
            Fidelity.APPROXIMATE,
            (f"midpoint volume integration: {self.integration_steps} steps",),
        )
