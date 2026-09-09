"""Opaque E1 payloads. They deliberately do not share a base class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Space


@dataclass(frozen=True)
class Raster:
    space: Space
    pixels: np.ndarray

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels, dtype=np.float64)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("raster pixels must have shape [height, width, 4]")
        object.__setattr__(self, "pixels", pixels)


@dataclass(frozen=True)
class CircleSDF:
    space: Space
    center: tuple[float, float]
    radius: float
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class GaussianSplats:
    space: Space
    centers: np.ndarray
    sigmas: np.ndarray
    colors: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        sigmas = np.asarray(self.sigmas, dtype=np.float64)
        colors = np.asarray(self.colors, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("splat centers must have shape [count, 2]")
        if sigmas.shape != (len(centers),) or np.any(sigmas <= 0.0):
            raise ValueError("splats need one positive sigma each")
        if colors.shape != (len(centers), 4):
            raise ValueError("splat colors must have shape [count, 4]")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "colors", colors)


@dataclass(frozen=True)
class TriangleSurface:
    space: Space
    triangles: np.ndarray
    colors: np.ndarray

    def __post_init__(self) -> None:
        triangles = np.asarray(self.triangles, dtype=np.float64)
        colors = np.asarray(self.colors, dtype=np.float64)
        if self.space.dimensions != 3 or triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("surface triangles must have shape [count, 3, 3] in 3D")
        if colors.shape != (len(triangles), 4):
            raise ValueError("surface colors must have shape [count, 4]")
        if np.any((colors[:, 3] < 0.0) | (colors[:, 3] > 1.0)):
            raise ValueError("surface alpha must lie in [0, 1]")
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "colors", colors)


@dataclass(frozen=True)
class GaussianVolume:
    space: Space
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    center: tuple[float, float, float]
    sigma: tuple[float, float, float]
    peak_density: float
    color: tuple[float, float, float]

    def __post_init__(self) -> None:
        bounds_min = np.asarray(self.bounds_min, dtype=np.float64)
        bounds_max = np.asarray(self.bounds_max, dtype=np.float64)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        if self.space.dimensions != 3:
            raise ValueError("volume requires a 3D space")
        if np.any(bounds_max <= bounds_min):
            raise ValueError("volume bounds must have positive extent")
        if np.any(sigma <= 0.0) or self.peak_density < 0.0:
            raise ValueError("volume sigma must be positive and density nonnegative")
