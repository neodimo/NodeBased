"""Representation-independent values used by the E1 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


@dataclass(frozen=True)
class Space:
    """An explicitly identified coordinate space."""

    name: str
    dimensions: int
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dimensions <= 0 or len(self.units) != self.dimensions:
            raise ValueError("units must name every positive-dimensional axis")


class Fidelity(IntEnum):
    """Ordered E1 error classes; larger values are less trustworthy."""

    EXACT = 0
    FILTERED = 1
    APPROXIMATE = 2
    HEURISTIC = 3


@dataclass(frozen=True)
class FootprintBatch:
    """Sample centres and a local differential basis for each centre.

    ``basis[n]`` maps unit local offsets to coordinates in ``space``. This is
    intentionally batched; Carta has no scalar sampling protocol.
    """

    space: Space
    centers: np.ndarray
    basis: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        basis = np.asarray(self.basis, dtype=np.float64)
        expected_basis = (len(centers), self.space.dimensions, self.space.dimensions)
        if centers.ndim != 2 or centers.shape[1] != self.space.dimensions:
            raise ValueError("centers must have shape [batch, space.dimensions]")
        if basis.shape != expected_basis:
            raise ValueError(f"basis must have shape {expected_basis}")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(basis)):
            raise ValueError("footprints must be finite")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "basis", basis)

    @property
    def count(self) -> int:
        return self.centers.shape[0]


@dataclass(frozen=True)
class SampleBatch:
    """Premultiplied RGBA values plus validity and declared fidelity."""

    values: np.ndarray
    validity: np.ndarray
    fidelity: Fidelity
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        validity = np.asarray(self.validity, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("E1 values must have shape [batch, 4]")
        if validity.shape != (values.shape[0],):
            raise ValueError("validity must have shape [batch]")
        if np.any((validity < 0.0) | (validity > 1.0)):
            raise ValueError("validity must lie in [0, 1]")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


@dataclass(frozen=True)
class ProjectiveMap:
    """A 2D projective pullback with analytic local Jacobian transport."""

    source: Space
    target: Space
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.source.dimensions != 2 or self.target.dimensions != 2:
            raise ValueError("E1 projective maps are two-dimensional")
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be a finite 3x3 array")
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("projective map must be invertible for E1")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls, space: Space) -> ProjectiveMap:
        return cls(space, space, np.eye(3, dtype=np.float64))

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape [batch, 2]")
        homogeneous = np.concatenate(
            (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
        )
        mapped = homogeneous @ self.matrix.T
        denominator = mapped[:, 2:3]
        if np.any(np.abs(denominator) < 1e-12):
            raise ValueError("projective map crossed infinity")
        return mapped[:, :2] / denominator

    def jacobian(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        mapped = self.apply(points)
        denominator = (
            self.matrix[2, 0] * points[:, 0]
            + self.matrix[2, 1] * points[:, 1]
            + self.matrix[2, 2]
        )
        numerators = np.stack(
            (
                self.matrix[0, 0] - mapped[:, 0] * self.matrix[2, 0],
                self.matrix[0, 1] - mapped[:, 0] * self.matrix[2, 1],
                self.matrix[1, 0] - mapped[:, 1] * self.matrix[2, 0],
                self.matrix[1, 1] - mapped[:, 1] * self.matrix[2, 1],
            ),
            axis=1,
        )
        return (numerators / denominator[:, None]).reshape(-1, 2, 2)

    def pullback(self, footprints: FootprintBatch) -> FootprintBatch:
        if footprints.space != self.source:
            raise ValueError(
                f"map expects {self.source.name!r}, got {footprints.space.name!r}"
            )
        jacobian = self.jacobian(footprints.centers)
        basis = np.einsum("nij,njk->nik", jacobian, footprints.basis)
        return FootprintBatch(self.target, self.apply(footprints.centers), basis)

    def then(self, next_map: ProjectiveMap) -> ProjectiveMap:
        """Return ``next_map(self(x))`` with no intermediate resampling."""

        if self.target != next_map.source:
            raise ValueError("map spaces do not compose")
        return ProjectiveMap(self.source, next_map.target, next_map.matrix @ self.matrix)
