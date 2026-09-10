"""Representation-independent values earned by Carta experiments."""

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
class RayBatch:
    """Batched metric rays in one explicit three-dimensional space."""

    space: Space
    origins: np.ndarray
    directions: np.ndarray
    near: np.ndarray
    far: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=np.float64)
        directions = np.asarray(self.directions, dtype=np.float64)
        near = np.asarray(self.near, dtype=np.float64)
        far = np.asarray(self.far, dtype=np.float64)
        if self.space.dimensions != 3 or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("rays require origins shaped [batch, 3] in a 3D space")
        if directions.shape != origins.shape:
            raise ValueError("ray directions must match origins")
        if near.shape != (len(origins),) or far.shape != (len(origins),):
            raise ValueError("ray intervals must have shape [batch]")
        if not all(np.all(np.isfinite(value)) for value in (origins, directions, near, far)):
            raise ValueError("rays must be finite")
        if np.any(far <= near):
            raise ValueError("every ray interval must have far > near")
        if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-9):
            raise ValueError("ray directions must be normalized so depth is metric")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "near", near)
        object.__setattr__(self, "far", far)

    @property
    def count(self) -> int:
        return self.origins.shape[0]


@dataclass(frozen=True)
class OrderedContributions:
    """Operation-local IR for a front-to-back ray fold.

    It is deliberately not a Representation or Resource. Adapters produce it,
    the reduction op consumes it, and a Score must never store it.
    """

    depths: np.ndarray
    values: np.ndarray
    active: np.ndarray
    ray_validity: np.ndarray
    fidelity: Fidelity
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        depths = np.asarray(self.depths, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        active = np.asarray(self.active, dtype=bool)
        ray_validity = np.asarray(self.ray_validity, dtype=np.float64)
        if depths.ndim != 2:
            raise ValueError("contribution depths must have shape [rays, events]")
        if values.shape != depths.shape + (4,) or active.shape != depths.shape:
            raise ValueError("contribution values/active mask must match depths")
        if ray_validity.shape != (depths.shape[0],):
            raise ValueError("ray validity must have shape [rays]")
        if np.any((ray_validity < 0.0) | (ray_validity > 1.0)):
            raise ValueError("ray validity must lie in [0, 1]")
        if np.any((values[..., 3] < 0.0) | (values[..., 3] > 1.0)):
            raise ValueError("contribution alpha must lie in [0, 1]")
        effective_depth = np.where(active, depths, np.inf)
        if np.any(np.diff(effective_depth, axis=1) < 0.0):
            raise ValueError("active contributions must be ordered front-to-back")
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "ray_validity", ray_validity)

    @property
    def ray_count(self) -> int:
        return self.depths.shape[0]


@dataclass(frozen=True)
class ReductionBatch:
    """Result of a ray-family reduction, distinct from a field sample."""

    values: np.ndarray
    validity: np.ndarray
    transmittance: np.ndarray
    fidelity: Fidelity
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        validity = np.asarray(self.validity, dtype=np.float64)
        transmittance = np.asarray(self.transmittance, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("reduction values must have shape [rays, 4]")
        if validity.shape != (len(values),) or transmittance.shape != (len(values),):
            raise ValueError("reduction metadata must have shape [rays]")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "transmittance", transmittance)


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
