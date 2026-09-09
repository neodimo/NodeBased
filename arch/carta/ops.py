"""Representation-independent E1 operation algebra."""

from __future__ import annotations

import numpy as np

from .contracts import Sampleable
from .core import Fidelity, FootprintBatch, ProjectiveMap, SampleBatch, Space


def output_footprints(space: Space, width: int, height: int) -> FootprintBatch:
    if space.dimensions != 2 or width <= 0 or height <= 0:
        raise ValueError("output grid needs a 2D space and positive dimensions")
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    xx, yy = np.meshgrid(x, y)
    centers = np.stack((xx.ravel(), yy.ravel()), axis=1)
    basis = np.broadcast_to(
        np.diag((1.0 / width, 1.0 / height)), (len(centers), 2, 2)
    ).copy()
    return FootprintBatch(space, centers, basis)


def warp_sample(
    source: Sampleable, pullback: ProjectiveMap, footprints: FootprintBatch
) -> SampleBatch:
    mapped = pullback.pullback(footprints)
    if mapped.space != source.space:
        raise ValueError("pullback target and sample source space differ")
    return source.sample(mapped)


def render(
    source: Sampleable,
    pullback: ProjectiveMap,
    output_space: Space,
    width: int,
    height: int,
) -> SampleBatch:
    return warp_sample(source, pullback, output_footprints(output_space, width, height))


def over(foreground: SampleBatch, background: SampleBatch) -> SampleBatch:
    if foreground.values.shape != background.values.shape:
        raise ValueError("over operands must have equal batch shapes")
    fg = foreground.values
    bg = background.values
    inverse_alpha = 1.0 - fg[:, 3:4]
    values = fg + bg * inverse_alpha
    validity = foreground.validity * background.validity
    fidelity = Fidelity(max(foreground.fidelity, background.fidelity))
    notes = tuple(dict.fromkeys(foreground.notes + background.notes))
    return SampleBatch(values, validity, fidelity, notes)
