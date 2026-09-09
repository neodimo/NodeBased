"""Representation-specific Charts for E1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Fidelity, FootprintBatch, SampleBatch, Space
from .representations import CircleSDF, GaussianSplats, Raster


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
