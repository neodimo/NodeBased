"""Carta architecture experiment reference implementation.

Only contracts earned by completed experiments live here. This is not yet a
compatibility promise.
"""

from .core import (
    Fidelity,
    FootprintBatch,
    OrderedContributions,
    ProjectiveMap,
    RayBatch,
    ReductionBatch,
    SampleBatch,
    Space,
)
from .ops import front_to_back, over, output_footprints, reduce_rays, render, warp_sample

__all__ = [
    "Fidelity",
    "FootprintBatch",
    "OrderedContributions",
    "ProjectiveMap",
    "RayBatch",
    "ReductionBatch",
    "SampleBatch",
    "Space",
    "front_to_back",
    "output_footprints",
    "over",
    "reduce_rays",
    "render",
    "warp_sample",
]
