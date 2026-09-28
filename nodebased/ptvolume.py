"""Smoke and fire volumes inside the path tracer (lane L4, step R4).

A `scene3d.Volume` is a participating medium with the density the raymarch reads (`volumerender`): the
extinction of a point is `(absorption + scattering) * density_scale * density(p)` (zero-padded trilinear on
the cell-centred grid), split into absorption and scattering by the two knobs. The path tracer samples the
medium instead of marching it:

* Free flight is delta tracking. Each volume has a majorant (its largest density times the largest extinction
  factor); tentative collisions are drawn at that rate along the ray inside the volume's box, and each is real
  with probability `sigma_t / majorant` (otherwise null and the ray goes on). A path that survives to the
  surface behind the volume passes unattenuated; the surviving fraction is the transmittance. Volumes are
  independent Poisson processes, so the first real collision of the set is the nearest of each volume's own.
* A real collision absorbs the path with probability `1 - scattering / (absorption + scattering)` and otherwise
  scatters it: the smoke colour tints it, and the new direction is drawn from a Henyey-Greenstein phase
  function of anisotropy `g` (`volume_anisotropy`). Light is sampled at the collision like at a surface: every
  analytic light, the environment (luminance-CDF sampled) and the ambient sky, weighted against the phase
  function's own sampling by the same power heuristic, with a shadow ray that meets meshes, splats and every
  volume. Scattering goes on for as many events as the bounce limits allow, so smoke is lit by the dome, by
  light bounced off meshes and splats, and by other smoke, without the multiple-scattering approximation the
  raymarch uses (`volume_multi_scatter` and `volume_fire_light` are ignored here: the path tracer computes
  what they approximate).
* Fire emits `fire_intensity * Le(K) * sigma` per unit length where `temperature * temperature_scale`
  exceeds `fire_threshold` (`Le` the raymarch's blackbody or ramp table). It is added at every tentative
  collision as `emission / majorant` (a track-length estimator), so a path that reaches the fire from a mesh or
  from a splat picks its light up and fire lights everything its paths touch, the smoke included.
* Shadow rays (light sampling) take the deterministic transmittance `exp(-tau)` of the raymarch's shadow rays,
  `tau = shadow_density * sigma_t_unit * density_scale * integral(density)` by the midpoint rule with
  `shadow_steps` segments per volume; `shadow_density` 0 makes the smoke cast no shadow.

The phase function is the physical one (integrates to 1 over the sphere), against the raymarch's phase 1: a
single scattering event lit by a light of intensity `I` gives `I / 4` per unit optical depth here, and the rest
of a thick cloud's brightness comes from real multiple scattering. A scattering-only cloud of any shape in a
uniform sky of radiance 1 renders as 1.

Limits, stated: a volume box is the whole majorant, so a large box holding a small plume costs many null
collisions; motion blur, the fire-light knob and the multiple-scattering knobs are not used; smoke does not
light itself through `volume_multi_scatter` but through the tracer's real bounces, so it needs `max_bounces`
above 1 to show more than single scattering; the data passes other than `depth` do not see volumes, and `depth` is the
raymarch's first sample whose scaled density reaches `VolumeSettings.depth_threshold` (the rule `scene3d` merges).
"""
import math
from dataclasses import dataclass

import numpy as np

from . import raytrace, volumerender as vr
from .volumerender import _trilinear

PI = math.pi
MAX_COLLISIONS = 4096       # tentative collisions one ray may take through one volume before it is let through


@dataclass
class VolumeLayer:
    preps: list                # volumerender._Volume
    settings: object           # volumerender.VolumeSettings, quality-resolved
    sigma_t_unit: float
    scatter_fraction: float    # scattering / (absorption + scattering)
    majorant: list             # per volume
    max_density: list
    fire_table: object
    color: np.ndarray
    lo: np.ndarray             # world bounds of every box
    hi: np.ndarray

    def __len__(self):
        return len(self.preps)


def build(scene, settings=None):
    """The `VolumeLayer` of `scene.volumes`, or None when there are none."""
    volumes = tuple(getattr(scene, "volumes", ()) or ())
    if not volumes:
        return None
    settings = (settings or vr.VolumeSettings()).validated().resolved(volumes)
    sa, ss = float(settings.absorption), float(settings.scattering)
    sigma_t = sa + ss
    preps, majorant, peaks = [], [], []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for volume in volumes:
        prep = vr._Volume(volume, False)
        preps.append(prep)
        peak = float(volume.density.max()) if volume.density.size else 0.0
        peaks.append(peak)
        majorant.append(max(settings.density_scale * peak * max(sigma_t, 1.0), 1e-9))
        corners = np.array([[x, y, z] for x in (prep.box_min[0], prep.box_max[0]) for y in (prep.box_min[1], prep.box_max[1])
                            for z in (prep.box_min[2], prep.box_max[2])])
        world = corners @ prep.m[:3, :3].T + prep.m[:3, 3]
        lo, hi = np.minimum(lo, world.min(axis=0)), np.maximum(hi, world.max(axis=0))
    table = vr.fire_table(settings) if settings.fire_intensity > 0 and any(v.temperature is not None for v in volumes) else None
    return VolumeLayer(preps, settings, sigma_t, (ss / sigma_t) if sigma_t > 0 else 0.0, majorant, peaks, table,
                       np.asarray(settings.color, np.float64), lo, hi)


def _clip(prep, o, d):
    """Per-ray world-distance interval `(t_enter >= 0, t_exit)` of the rays against the volume's box."""
    o_obj = o @ prep.inv[:3, :3].T + prep.inv[:3, 3]
    d_obj = d @ prep.inv[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (prep.box_min - o_obj) / d_obj, (prep.box_max - o_obj) / d_obj
    near, far = np.minimum(a, b), np.maximum(a, b)
    parallel = np.abs(d_obj) < 1e-12
    inside = (o_obj >= prep.box_min) & (o_obj <= prep.box_max)
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), near).max(axis=1)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), far).min(axis=1)
    return np.maximum(near, 0.0), far


# --- free flight ---------------------------------------------------------------------------------------------

def _uniform(keys, turn, volume, step, slot, pcg, rand):
    """Uniform number `slot` of collision `step` of `volume` at path vertex `turn`: hashed from the path key."""
    key = pcg(keys + np.uint32((turn * 7919 + volume * 104729 + 12345) & 0xFFFFFFFF))
    return rand(key, step * 4 + slot)


def free_flight(layer, o, d, t_end, keys, turn, cancel=None):
    """Delta tracking of unit-direction rays from `o` up to distance `t_end` (per ray; may be inf).

    Returns `(t_event, emission)`: the distance of each ray's first real collision (inf for none) and the fire's
    contribution along the ray up to it, (N, 3), before the path throughput multiplies it. Absorption and
    scattering are the caller's."""
    from .pathtrace import pcg, rand
    n = len(o)
    t_event = np.full(n, np.inf)
    emission = np.zeros((n, 3))
    if not n:
        return t_event, emission
    settings = layer.settings
    glow_points = []          # (ray, t, weight) of every emission sample, filtered against the final events
    for vi, prep in enumerate(layer.preps):
        raytrace._cancel(cancel)
        mu = layer.majorant[vi]
        t0, t1 = _clip(prep, o, d)
        t1 = np.minimum(np.minimum(t1, t_end), t_event)
        alive = np.flatnonzero(t1 > t0)
        position = t0.copy()
        volume = prep.volume
        glows = layer.fire_table is not None and volume.temperature is not None
        for step in range(MAX_COLLISIONS):
            if not len(alive):
                break
            u_step = _uniform(keys[alive], turn, vi, step, 0, pcg, rand)
            position[alive] += -np.log1p(-np.minimum(u_step, 1 - 1e-12)) / mu
            inside = position[alive] < t1[alive]
            alive = alive[inside]
            if not len(alive):
                break
            p = o[alive] + d[alive] * position[alive][:, None]
            g = prep.to_grid(prep.to_object(p))
            sigma = settings.density_scale * _trilinear(volume.density, g)
            if glows:
                kelvin = settings.temperature_scale * _trilinear(volume.temperature, g)
                glow = (kelvin > settings.fire_threshold) & (sigma > 0)
                if glow.any():
                    weight = vr.fire_radiance(layer.fire_table, kelvin[glow]) * (settings.fire_intensity * sigma[glow] / mu)[:, None]
                    glow_points.append((alive[glow], position[alive][glow], weight))
            real = _uniform(keys[alive], turn, vi, step, 1, pcg, rand) < layer.sigma_t_unit * sigma / mu
            hit = alive[real]
            t_event[hit] = np.minimum(t_event[hit], position[hit])
            alive = alive[~real]
    for ray, t, weight in glow_points:
        keep = t <= t_event[ray]
        np.add.at(emission, ray[keep], weight[keep])
    return t_event, emission


# --- shadow rays ---------------------------------------------------------------------------------------------

def transmittance(layer, origin, direction, dist):
    """`exp(-tau)` through every volume along shadow rays (unit `direction`, up to `dist`, which may be inf)."""
    n = len(origin)
    out = np.ones(n)
    settings = layer.settings
    if not n or settings.shadow_density <= 0:
        return out
    steps = int(settings.shadow_steps)
    dist = np.broadcast_to(np.asarray(dist, np.float64), (n,))
    for prep in layer.preps:
        near, far = _clip(prep, origin, direction)
        length = np.clip(np.minimum(far, dist) - near, 0.0, None)
        start = np.where(length > 0, near, 0.0)
        o_obj = origin @ prep.inv[:3, :3].T + prep.inv[:3, 3]
        d_obj = direction @ prep.inv[:3, :3].T
        total = np.zeros(n)
        active = np.flatnonzero(length > 0)
        for j in range(steps):
            if not len(active):
                break
            p = o_obj[active] + d_obj[active] * (start[active] + (j + 0.5) / steps * length[active])[:, None]
            total[active] += _trilinear(prep.volume.density, prep.to_grid(p))
        tau = settings.shadow_density * layer.sigma_t_unit * settings.density_scale * total * (length / steps)
        out *= np.exp(-tau)
    return out


# --- phase function ------------------------------------------------------------------------------------------

def phase(g, cosine):
    """Henyey-Greenstein density over solid angle (integrates to 1 over the sphere); `cosine` is the cosine
    between the direction the light travels and the direction from the collision to the eye, so 1 is forward."""
    denom = 1 + g * g - 2 * g * cosine
    return (1 - g * g) / (4 * PI * np.maximum(denom, 1e-12) ** 1.5)


def sample_phase(g, d, u1, u2):
    """New unit directions for paths arriving along `d` (N,3): the collision's continuing ray, with the cosine
    against `d` distributed as the phase function's (forward peaked for positive g). Returns `(directions, pdf)`."""
    if abs(g) < 1e-3:
        cosine = 1 - 2 * u1
    else:
        s = (1 - g * g) / (1 - g + 2 * g * u1)
        cosine = (1 + g * g - s * s) / (2 * g)
    cosine = np.clip(cosine, -1, 1)
    sine = np.sqrt(np.maximum(0, 1 - cosine * cosine))
    phi = 2 * PI * u2
    helper = np.where((np.abs(d[:, 0]) > 0.9)[:, None], np.array((0.0, 1.0, 0.0)), np.array((1.0, 0.0, 0.0)))
    t = np.cross(helper, d)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-30)
    b = np.cross(d, t)
    out = cosine[:, None] * d + (sine * np.cos(phi))[:, None] * t + (sine * np.sin(phi))[:, None] * b
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30), phase(g, cosine)
