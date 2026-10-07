"""Smoke and fire volumes inside the path tracer (lane L4, step R4).

A `scene3d.Volume` is a participating medium with the density the raymarch reads (`volumerender`): the
extinction of a point is `(absorption + scattering) * density_scale * density(p)` (zero-padded trilinear on
the cell-centred grid), split into absorption and scattering by the two knobs. The path tracer samples the
medium instead of marching it:

* Free flight is delta tracking. Each volume has a majorant (its largest density times the largest extinction
  factor), refined by a two-level grid of local bounds (`majorant_levels`: cells of `MAJORANT_TILE` voxels, and blocks
  of `MAJORANT_RATIO` cubed cells above them) so empty space is crossed in a stride and thin space against a tight bound;
  tentative collisions are drawn at the local rate along the ray inside the volume's box, and each is real
  with probability `sigma_t / bound` (otherwise null and the ray goes on). A path that survives to the
  surface behind the volume passes unattenuated; the surviving fraction is the transmittance. Volumes are
  independent Poisson processes, so the first real collision of the set is the nearest of each volume's own.
* A real collision absorbs the path with probability `1 - scattering / (absorption + scattering)` and otherwise
  scatters it: the smoke colour tints it, and the new direction is drawn from a Henyey-Greenstein phase
  function of anisotropy `g` (`volume_anisotropy`). Light is sampled at the collision like at a surface: every
  analytic light, the environment (luminance-CDF sampled) and the ambient sky, weighted against the phase
  function's own sampling by the same power heuristic, with a shadow ray that meets meshes, splats and every
  volume. Scattering goes on for as many events as the bounce limits allow, so smoke is lit by the dome, by
  light bounced off meshes and splats, and by other smoke, without the multiple-scattering approximation the
  raymarch uses. `volume_multi_scatter` m and `volume_fire_light` are gains on what the tracer computes: the path's
  throughput is multiplied by `1 + m` once, at its second smoke collision in a row, and fire seen from a smoke vertex is
  multiplied by `fire_light`; 0 and 1 leave the render as it was.
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

Limits, stated: smoke needs `max_bounces` above 1 to show more than single scattering; the data passes find the smoke by
the raymarch's first sample whose scaled density reaches `VolumeSettings.depth_threshold` (`volumerender.first_hit_data`,
merged by `pathtrace.merge_volume_data`); motion blur is the scene's (`motionblur.advect_volume`, one scene per shutter time).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import raytrace, volumerender as vr
from .volumerender import _trilinear

PI = math.pi
MAX_COLLISIONS = 4096       # tentative collisions one ray may take through one volume before it is let through
ENABLE_SKIP = True          # False keeps the box as the whole majorant (the baseline the measurements compare against)
MAJORANT_TILE = 16          # voxels along one edge of a cell of the fine majorant grid
MAJORANT_RATIO = 4          # fine cells along one edge of a cell of the coarse level above it


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
    excl: object = None        # light linking: per volume, the bit mask of the lights it excludes (None: no volume excludes any)
    counts: dict = field(default_factory=lambda: {"tentative": 0, "real": 0, "hops": 0})   # free flight's tallies
    _levels: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.preps)

    def levels(self, vi):
        """The `MajorantLevels` of volume `vi` (built on first use), or None when skipping is off or the volume is empty."""
        if not ENABLE_SKIP or self.max_density[vi] <= 0:
            return None
        if vi not in self._levels:
            self._levels[vi] = majorant_levels(self.preps[vi].volume, self.majorant[vi])
        return self._levels[vi]


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


def _window_max(a, axis, tile):
    """Max over each coarse cell's window along `axis`: the cell's `tile` voxels and one more voxel on each side."""
    n = a.shape[axis]
    cells = -(-n // tile)
    first = np.arange(cells) * tile
    bounds = np.empty(2 * cells, np.intp)
    bounds[0::2], bounds[1::2] = np.maximum(first - 1, 0), np.minimum(first + tile + 1, n)
    pad = list(a.shape)
    pad[axis] = 1
    padded = np.concatenate((a, np.zeros(pad, a.dtype)), axis=axis)       # reduceat wants every index below the length
    return np.take(np.maximum.reduceat(padded, bounds, axis=axis), np.arange(0, 2 * cells, 2), axis=axis)


def coarse_majorants(density, global_mu, tile=None):
    """Conservative extinction bound per coarse cell of `tile` voxels (default `MAJORANT_TILE`): the global bound
    `global_mu` scaled by the largest density in the cell grown by trilinear's one-voxel halo, over the peak."""
    tile = tile or MAJORANT_TILE
    density = np.asarray(density)
    peak = max(float(np.max(density)), 1e-20)
    best = density
    for axis in range(3):
        best = _window_max(best, axis, tile)
    out = (global_mu * best.astype(np.float64) / peak).astype(np.float32)
    # Round upward so float32 packing cannot underbound the source's peak.
    return np.where(out > 0, np.nextafter(out, np.float32(np.inf)), out)      # an empty cell stays exactly zero


def coarse_majorants_sparse(grid, name, global_mu, tile=None):
    """`_coarse_volume_majorants` of a sparse field, from its stored tiles alone: the same bound per coarse cell (the cell
    grown by trilinear's one-voxel halo), with the rest value counted wherever the halo window reaches an empty tile."""
    tile = tile or MAJORANT_TILE
    shape = grid.shape
    edge = grid.tile
    coarse = tuple((n + tile - 1) // tile for n in shape)
    peak = max(grid.max(name), 1e-20)
    best = np.full(coarse, -np.inf)
    covered = np.zeros(coarse, np.int64)
    window = np.ones(coarse, np.int64)
    for axis, n in enumerate(shape):       # the voxels each cell's window holds along one axis
        starts = np.arange(coarse[axis]) * tile
        length = np.minimum(n, starts + tile + 1) - np.maximum(0, starts - 1)
        window = window * length.reshape([-1 if a == axis else 1 for a in range(3)])
    block = grid.data[name]
    for index, coord in enumerate(grid.coords):
        lo = coord.astype(np.int64) * edge
        reach = []
        for axis in range(3):
            first = max(0, int(np.ceil((lo[axis] - tile - 1) / tile)))        # cells whose window can touch this tile
            last = min(coarse[axis] - 1, int((lo[axis] + edge) // tile))
            reach.append(range(first, last + 1))
        for kx in reach[0]:
            for ky in reach[1]:
                for kz in reach[2]:
                    k = (kx, ky, kz)
                    w_lo = [max(0, c * tile - 1) for c in k]
                    w_hi = [min(shape[a], c * tile + tile + 1) for a, c in enumerate(k)]
                    a_lo = [max(int(lo[a]), w_lo[a]) for a in range(3)]
                    a_hi = [min(int(lo[a]) + edge, w_hi[a]) for a in range(3)]
                    if any(h <= l for l, h in zip(a_lo, a_hi)):
                        continue
                    part = block[index, a_lo[0] - lo[0]:a_hi[0] - lo[0], a_lo[1] - lo[1]:a_hi[1] - lo[1],
                                 a_lo[2] - lo[2]:a_hi[2] - lo[2]]
                    best[k] = max(best[k], float(np.max(part)))
                    covered[k] += part.size
    best = np.where(covered < window, np.maximum(best, grid.rest[name]), best)
    out = (global_mu * best / peak).astype(np.float32)
    return np.where(out > 0, np.nextafter(out, np.float32(np.inf)), out)      # an empty cell stays exactly zero


@dataclass
class MajorantLevels:
    """A two-level bound on a volume's extinction. `fine[i, j, k]` bounds every trilinear sample whose point lies in the
    cube of `tile` voxels at cell (i, j, k) (the cube grown by the one-voxel halo); `coarse` bounds the `ratio` cubed
    fine cells each of its cells holds, so a ray crosses empty space in strides of `tile * ratio` voxels and thin or
    patchy space in the tight steps of the fine grid. Both are exact upper bounds (rounded up in float32)."""
    fine: np.ndarray
    tile: int
    coarse: np.ndarray
    ratio: int


def majorant_levels(volume, global_mu, tile=None, ratio=None):
    """The `MajorantLevels` of `volume`, built from its stored tiles when it is sparse (the dense field is not
    materialised)."""
    tile, ratio = tile or MAJORANT_TILE, ratio or MAJORANT_RATIO
    if getattr(volume, "sparse", None) is not None and hasattr(volume.density, "sparse_sample"):
        fine = coarse_majorants_sparse(volume.sparse, "density", global_mu, tile)
    else:
        fine = coarse_majorants(volume.density, global_mu, tile)
    cells = tuple(-(-n // ratio) for n in fine.shape)
    padded = np.zeros(tuple(c * ratio for c in cells), np.float32)
    padded[:fine.shape[0], :fine.shape[1], :fine.shape[2]] = fine
    coarse = padded.reshape(cells[0], ratio, cells[1], ratio, cells[2], ratio).max(axis=(1, 3, 5))
    return MajorantLevels(fine, tile, np.ascontiguousarray(coarse), ratio)


def _boundary_exit(cell, edge, q, dq, position):
    """World distance at which each ray leaves its cell of `edge` voxels: `q` is the grid-space point (voxels from the
    box minimum) it is at, `dq` the grid-space step per world unit. The same rule as the shader (a face already
    behind the ray, or within 1e-6 of it, is ignored)."""
    out = np.full(len(q), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis in range(3):
            moving = np.abs(dq[:, axis]) > 1e-12
            side = np.where(dq[:, axis] > 0, cell[:, axis] + 1, cell[:, axis])
            reach = position + (side * edge - q[:, axis]) / dq[:, axis]
            out = np.where(moving & (reach > position + 1e-6), np.minimum(out, reach), out)
    return out


def local_majorant(levels, q, dq, position):
    """`(mu, exit)` for rays at grid-space points `q` heading along `dq` (per world unit) at world distance `position`:
    the bound of the cell they are in (0 for an empty one) and the distance at which they leave the region it holds.
    An empty coarse cell is crossed in one stride; otherwise the fine cell sets both."""
    cells = np.asarray(levels.fine.shape)
    cell = np.clip(np.floor(q / levels.tile).astype(np.int64), 0, cells - 1)
    mu = levels.fine[cell[:, 0], cell[:, 1], cell[:, 2]].astype(np.float64)
    big = levels.tile * levels.ratio
    sup = cell // levels.ratio
    empty = levels.coarse[sup[:, 0], sup[:, 1], sup[:, 2]] <= 0
    exit_ = np.where(empty, _boundary_exit(sup, big, q, dq, position), _boundary_exit(cell, levels.tile, q, dq, position))
    return mu, exit_


# --- free flight ---------------------------------------------------------------------------------------------

def _uniform(keys, turn, volume, step, slot, pcg, rand):
    """Uniform number `slot` of collision `step` of `volume` at path vertex `turn`: hashed from the path key."""
    key = pcg(keys + np.uint32((turn * 7919 + volume * 104729 + 12345) & 0xFFFFFFFF))
    return rand(key, step * 4 + slot)


def free_flight(layer, o, d, t_end, keys, turn, cancel=None):
    """Delta tracking of unit-direction rays from `o` up to distance `t_end` (per ray; may be inf).

    Returns `(t_event, emission, owner)`: the distance of each ray's first real collision (inf for none), the fire's
    contribution along the ray up to it, (N, 3), before the path throughput multiplies it, and the index of the volume
    that collision happened in (-1 for none). Absorption and scattering are the caller's."""
    from .pathtrace import pcg, rand
    n = len(o)
    t_event = np.full(n, np.inf)
    emission = np.zeros((n, 3))
    owner = np.full(n, -1, np.int64)
    if not n:
        return t_event, emission, owner
    settings = layer.settings
    glow_points = []          # (ray, t, weight) of every emission sample, filtered against the final events
    counts = layer.counts
    for vi, prep in enumerate(layer.preps):
        raytrace._cancel(cancel)
        levels = layer.levels(vi)
        t0, t1 = _clip(prep, o, d)
        t1 = np.minimum(np.minimum(t1, t_end), t_event)
        alive = np.flatnonzero(t1 > t0)
        position = t0.copy()
        volume = prep.volume
        glows = layer.fire_table is not None and volume.temperature is not None
        if levels is not None:
            o_cells = (prep.to_object(o) - prep.box_min) / volume.voxel_size      # grid-space origin and step per world unit
            d_cells = (d @ prep.inv[:3, :3].T) / volume.voxel_size
        for step in range(MAX_COLLISIONS):
            if not len(alive):
                break
            here = position[alive]
            if levels is None:
                mu, leave = np.full(len(alive), layer.majorant[vi]), t1[alive]
            else:
                mu, leave = local_majorant(levels, o_cells[alive] + d_cells[alive] * here[:, None], d_cells[alive], here)
                leave = np.minimum(leave, t1[alive])
            u_step = _uniform(keys[alive], turn, vi, step, 0, pcg, rand)
            candidate = here - np.log1p(-np.minimum(u_step, 1 - 1e-12)) / np.where(mu > 0, mu, 1.0)
            tentative = (mu > 0) & (candidate < leave)
            hop = ~tentative
            # a ray that reaches the end of its region (or sits in empty space) goes on from just past it
            position[alive[hop]] = leave[hop] + 1e-5
            carry = np.zeros(len(alive), bool)
            carry[hop] = position[alive[hop]] < t1[alive[hop]]
            counts["hops"] += int(hop.sum()) if levels is not None else 0
            ray = alive[tentative]
            counts["tentative"] += len(ray)
            if len(ray):
                position[ray] = candidate[tentative]
                g = prep.to_grid(prep.to_object(o[ray] + d[ray] * position[ray][:, None]))
                sigma = settings.density_scale * _trilinear(volume.density, g)
                mu_here = mu[tentative]
                if glows:
                    kelvin = settings.temperature_scale * _trilinear(volume.temperature, g)
                    glow = (kelvin > settings.fire_threshold) & (sigma > 0)
                    if glow.any():
                        weight = vr.fire_radiance(layer.fire_table, kelvin[glow]) * (settings.fire_intensity * sigma[glow] / mu_here[glow])[:, None]
                        glow_points.append((ray[glow], position[ray][glow], weight))
                real = _uniform(keys[ray], turn, vi, step, 1, pcg, rand) < layer.sigma_t_unit * sigma / mu_here
                counts["real"] += int(real.sum())
                hit = ray[real]
                owner[hit[position[hit] < t_event[hit]]] = vi
                t_event[hit] = np.minimum(t_event[hit], position[hit])
                carry[tentative] = ~real
            alive = alive[carry]
    for ray, t, weight in glow_points:
        keep = t <= t_event[ray]
        np.add.at(emission, ray[keep], weight[keep])
    return t_event, emission, owner


# --- shadow rays ---------------------------------------------------------------------------------------------

def transmittance(layer, origin, direction, dist, bit=None):
    """`exp(-tau)` through every volume along shadow rays (unit `direction`, up to `dist`, which may be inf). A shadow ray
    to light `bit` skips a volume that excludes it (light linking)."""
    n = len(origin)
    out = np.ones(n)
    settings = layer.settings
    if not n or settings.shadow_density <= 0:
        return out
    steps = int(settings.shadow_steps)
    dist = np.broadcast_to(np.asarray(dist, np.float64), (n,))
    for vi, prep in enumerate(layer.preps):
        if bit is not None and layer.excl is not None and int(layer.excl[vi]) >> bit & 1:
            continue
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
