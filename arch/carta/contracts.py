"""Small structural contracts for E1; no representation types belong here."""

from __future__ import annotations

from typing import Protocol

from .core import FootprintBatch, SampleBatch, Space


class Sampleable(Protocol):
    @property
    def space(self) -> Space: ...

    def sample(self, footprints: FootprintBatch) -> SampleBatch: ...
