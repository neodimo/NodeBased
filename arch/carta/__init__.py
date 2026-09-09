"""Carta E1 reference implementation.

Only the contracts needed by the Chart Test live here. This is an experiment,
not a compatibility promise.
"""

from .core import Fidelity, FootprintBatch, ProjectiveMap, SampleBatch, Space
from .ops import over, output_footprints, render, warp_sample

__all__ = [
    "Fidelity",
    "FootprintBatch",
    "ProjectiveMap",
    "SampleBatch",
    "Space",
    "output_footprints",
    "over",
    "render",
    "warp_sample",
]
