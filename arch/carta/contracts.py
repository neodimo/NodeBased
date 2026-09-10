"""Small structural contracts; no representation types belong here."""

from __future__ import annotations

from typing import Protocol

from .core import FootprintBatch, OrderedContributions, RayBatch, SampleBatch, Space


class Sampleable(Protocol):
    @property
    def space(self) -> Space: ...

    def sample(self, footprints: FootprintBatch) -> SampleBatch: ...


class RayReducible(Protocol):
    @property
    def space(self) -> Space: ...

    def ray_contributions(self, rays: RayBatch) -> OrderedContributions: ...
