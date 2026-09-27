"""Viewport inspection of solved particles (particle artist tools, DiMo 9/27, deliverable 4):
picking a particle under the cursor, a plain-dict readout for its tooltip, and "colour by
attribute" -- a display-only viewport mode independent of the particle's own stored colour.

Pure numpy/`scene3d` logic, no Qt: a widget calls `pick_particle` on a mouse click and
`colorize_by_attribute` when it rebuilds its particle draw call, exactly the split
`artisttools.py`'s widgets already keep between "what changed" (a module like this one) and "how
it is painted" (the widget shell).
"""
from __future__ import annotations

import numpy as np

from . import scene3d

ATTRIBUTES = ("age", "speed", "id", "lifetime_fraction")

# Blue -> green -> red, matching the cool-to-hot ramp `slice3d.colorize` already uses for scalar
# fields, so a "colour by attribute" viewport mode reads consistently with the slice viewer.
_RAMP = ((0.0, (0.0, 0.0, 1.0)), (0.5, (0.0, 1.0, 0.0)), (1.0, (1.0, 0.0, 0.0)))


def pick_particle(camera, width, height, positions, click_x, click_y, tolerance=8.0):
    """The index into `positions` (N,3) nearest `(click_x, click_y)` in screen space, within
    `tolerance` pixels, ties broken by whichever is closest to the camera; `None` when nothing
    is within tolerance, in front of the camera, or `positions` is empty."""
    positions = np.asarray(positions)
    if len(positions) == 0:
        return None
    pixels, depth = scene3d.project(camera, width, height, positions)
    in_front = depth > 1e-6
    if not in_front.any():
        return None
    distance = np.linalg.norm(pixels - np.array([click_x, click_y]), axis=1)
    distance = np.where(in_front, distance, np.inf)
    within = distance <= tolerance
    if not within.any():
        return None
    candidates = np.flatnonzero(within)
    best = candidates[np.argmin(depth[candidates])]
    return int(best)


def particle_readout(instance, index: int) -> dict:
    """A plain dict of id/age/velocity/size/colour for the particle at `index`, ready for a
    tooltip: `{"id", "age", "velocity", "speed", "size", "color"}`."""
    velocity = instance.velocities[index]
    return {
        "id": int(instance.ids[index]),
        "age": float(instance.ages[index]),
        "velocity": tuple(float(v) for v in velocity),
        "speed": float(np.linalg.norm(velocity)),
        "size": float(instance.sizes[index]),
        "color": tuple(float(c) for c in instance.colors[index]),
    }


def attribute_values(instance, attribute: str) -> np.ndarray:
    """The per-particle scalar `attribute` (one of `ATTRIBUTES`) as a float64 array."""
    if attribute == "age":
        return instance.ages.astype(np.float64)
    if attribute == "speed":
        return np.linalg.norm(instance.velocities.astype(np.float64), axis=1)
    if attribute == "id":
        return instance.ids.astype(np.float64)
    if attribute == "lifetime_fraction":
        lifetimes = instance.lifetimes.astype(np.float64)
        ages = instance.ages.astype(np.float64)
        return np.divide(ages, lifetimes, out=np.zeros_like(ages), where=lifetimes > 0)
    raise ValueError(f"unknown attribute {attribute!r}, expected one of {ATTRIBUTES}")


def _ramp_color(t: float):
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(_RAMP, _RAMP[1:]):
        if t0 <= t <= t1:
            fraction = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(c0[i] + fraction * (c1[i] - c0[i]) for i in range(3))
    return _RAMP[-1][1]


def colorize_by_attribute(instance, attribute: str, vmin=None, vmax=None) -> np.ndarray:
    """RGB (N,3) float32, one colour per particle, mapping `attribute` through the ramp above,
    normalised to `[vmin, vmax]` (the attribute's own min/max across `instance` when either is
    omitted). Display-only: it never touches `instance.colors`, the particle's own stored colour."""
    values = attribute_values(instance, attribute)
    if not len(values):
        return np.zeros((0, 3), np.float32)
    low = float(values.min()) if vmin is None else float(vmin)
    high = float(values.max()) if vmax is None else float(vmax)
    span = high - low if high > low else 1.0
    t = (values - low) / span
    return np.array([_ramp_color(float(value)) for value in t], np.float32)


def particle_count_per_emitter(counts: dict) -> str:
    """One "<label>: <count> particles" line per entry of `counts` (emitter label -> particle
    count), for the A1 sim stats overlay (see `simstats.SimStatsSnapshot.particle_counts`)."""
    return "\n".join(f"{label}: {count} particles" for label, count in counts.items())
