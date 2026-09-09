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
