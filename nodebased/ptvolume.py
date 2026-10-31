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
  from a splat picks its light up and fire lights everything its paths touch, the smoke included. At a smoke vertex the fire is also
  a sampled light (one volume, `volume_fire_light` above 0, no splats): a point is drawn from the coarse emission grid
  (`FireEmitter`), shaded through the smoke by the midpoint rule, and shares the fire with the phase-sampled continuation ray by the balance
  heuristic (`fire_nee_sample`, `fire_nee_value`, `expected_emission`'s `mis`), so the fire's light on smoke has the mean it had and less noise.
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
ENABLE_EXPECTED_EMISSION = True   # False sums the fire's glow at the tentative collisions again (the baseline the tests compare against)
EMIT_CELLS = 4.0            # emission quadrature: voxels per jittered sample on a flight from the camera or a surface ...
EMIT_CELLS_DEEP = 16.0      # ... and on a flight that starts at a smoke vertex (its glow is weighted down); the shader's twins
EMIT_TAU = 0.5              # a step's optical depth at the region's extinction bound never exceeds this: dense smoke is sampled finer
EMIT_TAU_CAP = 6.0          # a flight whose optical depth passes this adds nothing more (exp(-6) < 2.5e-3)
EMIT_SELF = 0.75            # share of the sample's own extinction in the optical depth back to the step's start; the rest is the previous one's
ENABLE_FIRE_NEE = True      # False leaves the fire reaching smoke only by phase-sampled continuation rays (the baseline the measurements compare against)
NEE_STREAM = 31             # hashed-number stream ids of the fire sampling at a smoke vertex (cell and point) and of its ratio tracking
NEE_RATIO_STREAM = 32      # (only `ratio_transmittance`, the unbiased reference, draws from it)
NEE_DELTA_CELLS = 2.0       # the step the multiple-importance weights assume for the phase-sampled glow, in voxels (any step keeps the sum unbiased; set by the variance sweep)
NEE_NEAR_CELLS = 4.0        # inside this many voxels of a smoke vertex the sampled fire's share fades in as (r / this)^2, the continuation keeping the rest (0: no fade)
NEE_STEP_CELLS = 4.0        # the fire shadow ray's quadrature: one midpoint step per this many voxels ...
NEE_TAU = 1.0               # ... and per this much optical depth at the volume's extinction bound, at most NEE_MAX_STEPS in all
NEE_MAX_STEPS = 48
NEE_HALO = 0.05             # share of the neighbouring cells' emission a cell's sampling weight also holds (trilinear reads one voxel beyond a cell)
NEE_FLOOR = 1e-3            # share of the samples drawn uniformly over the grid, so no cell the fine field could glow in has zero probability
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
    _emitter: object = field(default=False, repr=False)

    def __len__(self):
        return len(self.preps)

    def levels(self, vi):
        """The `MajorantLevels` of volume `vi` (built on first use), or None when skipping is off or the volume is empty."""
        if (not ENABLE_SKIP or self.max_density[vi] <= 0
                or (self.settings.flame_intensity > 0 and self.preps[vi].volume.flame is not None)):
            return None
        if vi not in self._levels:
            self._levels[vi] = majorant_levels(self.preps[vi].volume, self.majorant[vi])
        return self._levels[vi]


    def emitter(self, ignore_switch=False):
        """The `FireEmitter` that smoke vertices draw next-event fire samples from, or None: next-event sampling needs one volume
        (`_expected`), a fire-light gain above zero and a fire; its multiple-importance partner is the glow `expected_emission` gathers.
        `ENABLE_FIRE_NEE` False gives None unless `ignore_switch` (the GPU packs the table and applies its own switch)."""
        if self._emitter is False:
            self._emitter = None
            if _expected(self) and self.settings.fire_light > 0:
                self._emitter = FireEmitter.build(self.preps[0], self.settings)
        return self._emitter if (ENABLE_FIRE_NEE or ignore_switch) else None

    def smoke_majorant(self, vi):
        """A bound on the extinction alone (the glow-carrying `majorant` also holds the flame's peak), for ratio tracking."""
        return self.settings.density_scale * self.max_density[vi] * self.sigma_t_unit


class FireEmitter:
    """The fire as a light for next-event sampling at smoke vertices (lane 6, fire-in-smoke step).

    The distribution is the coarse emission grid `volumerender.fire_coarse_emission` (the grid `fire_light_grid` blurs), unblurred:
    each cell's weight is its emission plus a small share of its 3 x 3 x 3 neighbourhood's (trilinear reads one voxel past a cell)
    plus a uniform floor, so every cell the fine emission could be positive in has a probability. A point is a cell drawn by that
    probability and a uniform position in it; its density per world volume is `density_at`."""

    def __init__(self, prep, cells, factor, pmf):
        self.prep = prep
        self.cells = np.asarray(cells, np.int64)
        self.factor = factor
        self.pmf = pmf                                    # (cells,) float64 flat, x-major, sums to 1
        self.cdf = np.cumsum(pmf)
        self.cdf[-1] = 1.0
        self.cell_object = factor * prep.volume.voxel_size
        self.cell_world = self.cell_object * abs(np.linalg.det(prep.m[:3, :3])) ** (1 / 3)
        self.delta = NEE_DELTA_CELLS * prep.volume.voxel_size      # the step in the multiple-importance weights (see NEE_DELTA_CELLS)
        self.near = NEE_NEAR_CELLS * prep.volume.voxel_size        # world distance the sampled share fades in over (0: no fade)

    @classmethod
    def build(cls, prep, settings):
        got = vr.fire_coarse_emission(prep.volume, settings)
        if got is None:
            return None
        coarse, factor = got
        lum = coarse.sum(axis=-1)
        padded = np.pad(lum, 1)
        halo = sum(padded[i:i + lum.shape[0], j:j + lum.shape[1], k:k + lum.shape[2]]
                   for i in range(3) for j in range(3) for k in range(3)) / 27.0
        weight = lum + NEE_HALO * halo
        weight = weight + NEE_FLOOR * weight.mean()
        return cls(prep, lum.shape, factor, (weight / weight.sum()).reshape(-1))

    def density_at(self, q_object):
        """Probability density per world volume of drawing object-space points `q_object` (N, 3); 0 outside the grid."""
        cell = np.floor((q_object - self.prep.box_min) / self.cell_object).astype(np.int64)
        inside = ((cell >= 0) & (cell < self.cells)).all(axis=1)
        flat = (np.clip(cell[:, 0], 0, self.cells[0] - 1) * self.cells[1] + np.clip(cell[:, 1], 0, self.cells[1] - 1)) * self.cells[2] \
            + np.clip(cell[:, 2], 0, self.cells[2] - 1)
        return np.where(inside, self.pmf[flat], 0.0) / self.cell_world ** 3


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
        flame_peak = (float(volume.flame.max()) if settings.flame_intensity > 0 and volume.flame is not None else 0.0)
        majorant.append(max(settings.density_scale * peak * max(sigma_t, 1.0), flame_peak, 1e-9))
        corners = np.array([[x, y, z] for x in (prep.box_min[0], prep.box_max[0]) for y in (prep.box_min[1], prep.box_max[1])
                            for z in (prep.box_min[2], prep.box_max[2])])
        world = corners @ prep.m[:3, :3].T + prep.m[:3, 3]
        lo, hi = np.minimum(lo, world.min(axis=0)), np.maximum(hi, world.max(axis=0))
    table = vr.fire_table(settings) if any(settings.glows(v) for v in volumes) else None
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


def free_flight(layer, o, d, t_end, keys, turn, cancel=None, deep=None, mis=None):
    """Delta tracking of unit-direction rays from `o` up to distance `t_end` (per ray; may be inf).

    Returns `(t_event, emission, owner)`: the distance of each ray's first real collision (inf for none), the fire's
    contribution along the ray, (N, 3), before the path throughput multiplies it, and the index of the volume
    that collision happened in (-1 for none). Absorption and scattering are the caller's. With one volume the fire's
    contribution is `expected_emission` over the whole flight (`deep` marks the rays that start at a smoke vertex); with several
    it is summed at the tentative collisions up to the first real one. `mis` is `(on, pdf)` per ray: the rays that leave a smoke
    vertex whose fire was also sampled by next-event estimation (`fire_nee`), with the phase pdf their direction was drawn with;
    their glow is weighted against that sampling (`expected_emission`)."""
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
        glows = layer.fire_table is not None and not _expected(layer)
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
                if glows and settings.fire_intensity > 0 and volume.temperature is not None:
                    kelvin = settings.temperature_scale * _trilinear(volume.temperature, g)
                    glow = (kelvin > settings.fire_threshold) & (sigma > 0)
                    if glow.any():
                        weight = vr.fire_radiance(layer.fire_table, kelvin[glow]) * (settings.fire_intensity * sigma[glow] / mu_here[glow])[:, None]
                        glow_points.append((ray[glow], position[ray][glow], weight))
                if glows and settings.flame_intensity > 0 and volume.flame is not None:
                    flame = np.maximum(_trilinear(volume.flame, g), 0.0)
                    if flame.any():
                        colour = vr.flame_colour(settings, layer.fire_table, _trilinear(volume.temperature, g)
                                                 if settings.flame_follow and volume.temperature is not None else None)
                        weight = colour * (settings.flame_intensity * flame / mu_here)[:, None]
                        glow_points.append((ray, position[ray].copy(), weight))
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
    if _expected(layer):
        emission = expected_emission(layer, 0, o, d, t_end, keys, turn, np.zeros(n, bool) if deep is None else deep, cancel, mis)
    return t_event, emission, owner


def _expected(layer):
    """Whether the fire's glow along a flight is integrated (`expected_emission`) rather than summed at collisions."""
    return ENABLE_EXPECTED_EMISSION and len(layer.preps) == 1 and layer.fire_table is not None


def expected_emission(layer, vi, o, d, t_end, keys, turn, deep, cancel=None, mis=None):
    """The fire's emission along rays of one volume up to `t_end`, weighted by the smoke's transmittance from the ray's start
    (N, 3): the expected value of what delta tracking's tentative collisions add, found by jittered quadrature over the same
    regions (a region with a zero majorant holds neither collision nor glow). One sample per `EMIT_CELLS` voxels (`EMIT_CELLS_DEEP`
    where `deep`), stratified with the same hashed numbers as the shader's `vol_emission`, which this mirrors line for line.
    Used for one volume; with several the other volumes' transmittance is unknown here and `free_flight` sums the glow at the
    collisions as before. `mis` (`on`, `pdf`) marks the rays that leave a smoke vertex which also sampled the fire as a light
    (`fire_nee`): a sample at distance t is weighted `pdf / (pdf + p_V delta t^2)` against that sampling, `pdf` being the phase pdf the
    ray's direction was drawn with, `p_V` the emitter's density per world volume at the sample and `delta` the quadrature step."""
    from .pathtrace import pcg, rand
    n = len(o)
    out = np.zeros((n, 3))
    emitter = layer.emitter() if mis is not None else None
    mis_on = np.zeros(n, bool) if emitter is None else np.asarray(mis[0], bool)
    mis_pdf = None if emitter is None else np.asarray(mis[1], np.float64)
    settings = layer.settings
    prep = layer.preps[vi]
    volume = prep.volume
    has_temp = settings.fire_intensity > 0 and volume.temperature is not None
    has_flame = settings.flame_intensity > 0 and volume.flame is not None
    if not n or layer.fire_table is None or not (has_temp or has_flame) or layer.majorant[vi] <= 1e-8:
        return out
    levels = layer.levels(vi)
    t0, t1 = _clip(prep, o, d)
    t1 = np.minimum(t1, t_end)
    alive = np.flatnonzero(t1 > t0)
    position = t0.copy()
    tau = np.zeros(n)
    sigma_prev = np.zeros(n)
    taken = np.zeros(n, np.int64)                       # samples drawn so far: the hashed number's step
    wstep = volume.voxel_size * np.where(np.asarray(deep, bool), EMIT_CELLS_DEEP, EMIT_CELLS)
    flame_colour = vr.flame_colour(settings, layer.fire_table)
    follow = bool(settings.flame_follow) and volume.temperature is not None
    if levels is not None:
        o_cells = (prep.to_object(o) - prep.box_min) / volume.voxel_size
        d_cells = (d @ prep.inv[:3, :3].T) / volume.voxel_size
    for _ in range(MAX_COLLISIONS):
        if not len(alive):
            break
        raytrace._cancel(cancel)
        here = position[alive]
        if levels is None:
            mu, leave = np.full(len(alive), layer.majorant[vi]), t1[alive]
        else:
            mu, leave = local_majorant(levels, o_cells[alive] + d_cells[alive] * here[:, None], d_cells[alive], here)
            leave = np.minimum(leave, t1[alive])
        end = np.minimum(leave, t1[alive])
        sampled = (mu > 0) & (end > here)
        if sampled.any():
            rays = alive[sampled]
            start, stop = here[sampled], end[sampled]
            count = np.maximum(1, np.ceil((stop - start) / np.minimum(wstep[rays], EMIT_TAU / mu[sampled])).astype(np.int64))
            h = (stop - start) / count
            for j in range(int(count.max())):
                live = j < count
                r, hh = rays[live], h[live]
                ju = _uniform(keys[r], turn, vi, taken[r], 2, pcg, rand)
                taken[r] += 1
                t = start[live] + (j + ju) * hh
                g = prep.to_grid(prep.to_object(o[r] + d[r] * t[:, None]))
                sigma = settings.density_scale * _trilinear(volume.density, g)
                e = np.zeros((len(r), 3))
                if has_temp:
                    kelvin = settings.temperature_scale * _trilinear(volume.temperature, g)
                    glow = (kelvin > settings.fire_threshold) & (sigma > 0)
                    if glow.any():
                        e[glow] += vr.fire_radiance(layer.fire_table, kelvin[glow]) * (settings.fire_intensity * sigma[glow])[:, None]
                if has_flame:
                    tint = vr.flame_colour(settings, layer.fire_table, _trilinear(volume.temperature, g)) if follow else flame_colour
                    e += tint * (settings.flame_intensity * np.maximum(_trilinear(volume.flame, g), 0.0))[:, None]
                sigma_t = layer.sigma_t_unit * sigma
                blend = EMIT_SELF * sigma_t + (1.0 - EMIT_SELF) * sigma_prev[r]
                if mis_on[r].any():
                    on = mis_on[r]
                    pv = emitter.density_at(prep.to_object(o[r] + d[r] * t[:, None]))
                    ph = mis_pdf[r]
                    e = e * np.where(on, 1.0 - sampled_share(emitter, pv, t, ph), 1.0)[:, None]
                out[r] += e * (hh * np.exp(-(tau[r] + blend * hh * ju)))[:, None]
                tau[r] += sigma_t * hh
                sigma_prev[r] = sigma_t
            done = np.zeros(len(alive), bool)
            done[sampled] = tau[rays] > EMIT_TAU_CAP
        else:
            done = np.zeros(len(alive), bool)
        position[alive] = np.maximum(leave, here) + 1e-5
        alive = alive[~done & (position[alive] < t1[alive])]
    return out


def _emission_at(layer, vi, g):
    """The fire's emission (N, 3) at grid-space points `g` of volume `vi`: what `expected_emission` weights at a sample."""
    settings = layer.settings
    volume = layer.preps[vi].volume
    sigma = settings.density_scale * _trilinear(volume.density, g)
    e = np.zeros((len(g), 3))
    if settings.fire_intensity > 0 and volume.temperature is not None:
        kelvin = settings.temperature_scale * _trilinear(volume.temperature, g)
        glow = (kelvin > settings.fire_threshold) & (sigma > 0)
        if glow.any():
            e[glow] += vr.fire_radiance(layer.fire_table, kelvin[glow]) * (settings.fire_intensity * sigma[glow])[:, None]
    if settings.flame_intensity > 0 and volume.flame is not None:
        follow = bool(settings.flame_follow) and volume.temperature is not None
        tint = vr.flame_colour(settings, layer.fire_table, _trilinear(volume.temperature, g) if follow else None)
        e += tint * (settings.flame_intensity * np.maximum(_trilinear(volume.flame, g), 0.0))[:, None]
    return e


def fire_nee_sample(layer, pos, keys, turn):
    """One fire point per smoke vertex `pos` (N, 3): a cell by the emitter's probability, a uniform position in it.
    Returns `(point, wi, dist, e, pv)`: the world point, the unit direction from the vertex to it, the distance, the fire's
    emission there (N, 3) and the point's density per world volume. `e` is zero for points outside the volume's box.
    The numbers are hashed from the path key like the shader's `vol_fire_sample`."""
    from .pathtrace import pcg, rand
    emitter = layer.emitter()
    prep = layer.preps[0]
    n = len(pos)
    u = [_uniform(keys, turn, NEE_STREAM, 0, slot, pcg, rand) for slot in range(4)]
    cell = np.minimum(np.searchsorted(emitter.cdf, u[0], side="right"), len(emitter.cdf) - 1)
    gy, gz = int(emitter.cells[1]), int(emitter.cells[2])
    index = np.stack((cell // (gy * gz), (cell // gz) % gy, cell % gz), axis=1).astype(np.float64)
    q = prep.box_min + emitter.cell_object * (index + np.stack(u[1:], axis=1))
    point = q @ prep.m[:3, :3].T + prep.m[:3, 3]
    inside = ((q >= prep.box_min) & (q <= prep.box_max)).all(axis=1)
    e = np.zeros((n, 3))
    if inside.any():
        e[inside] = _emission_at(layer, 0, prep.to_grid(q[inside]))
    to = point - pos
    dist = np.linalg.norm(to, axis=1)
    return point, to / np.maximum(dist, 1e-12)[:, None], dist, e, emitter.pmf[cell] / emitter.cell_world ** 3


def fire_transmittance(layer, vi, o, d, dist):
    """`exp(-tau)` through volume `vi` along rays (unit `d`, up to `dist`) by the midpoint rule: one step per `NEE_STEP_CELLS`
    voxels, more where the extinction bound would put more than `NEE_TAU` of optical depth in a step. Deterministic, like the
    shadow rays of the lights; the shader's `vol_fire_transmittance` mirrors it. (Ratio tracking, `ratio_transmittance`, has the
    same mean but ran 2.8 to 3.7 times noisier per fire sample through a dense flame: see docs/FLUIDS_SPIKE.md.)"""
    n = len(o)
    out = np.ones(n)
    prep = layer.preps[vi]
    volume = prep.volume
    mu = layer.smoke_majorant(vi)
    if not n or mu <= 1e-12:
        return out
    near, far = _clip(prep, o, d)
    length = np.clip(np.minimum(far, dist) - near, 0.0, None)
    steps = np.clip(np.ceil(np.maximum(length / (NEE_STEP_CELLS * volume.voxel_size), length * mu / NEE_TAU)), 1, NEE_MAX_STEPS)
    total = np.zeros(n)
    for j in range(int(steps.max()) if n else 0):
        live = (j < steps) & (length > 0)
        if not live.any():
            continue
        t = near[live] + (j + 0.5) / steps[live] * length[live]
        g = prep.to_grid(prep.to_object(o[live] + d[live] * t[:, None]))
        total[live] += _trilinear(volume.density, g) * (length[live] / steps[live])
    return np.exp(-layer.sigma_t_unit * layer.settings.density_scale * total)


def ratio_transmittance(layer, vi, o, d, dist, keys, turn, cancel=None):
    """Transmittance of volume `vi` along rays (unit `d`, up to `dist`) by ratio tracking: delta tracking's tentative collisions
    against a bound on the extinction (the local bound where there is a majorant grid, otherwise the smoke's alone) each multiply
    the estimate by `1 - sigma_t / bound`. Unbiased, with a mean of `exp(-tau)`; it is the reference `fire_transmittance` is tested
    against, not what the tracers use (no shader twin)."""
    from .pathtrace import pcg, rand
    n = len(o)
    out = np.ones(n)
    prep = layer.preps[vi]
    volume = prep.volume
    settings = layer.settings
    mu_all = layer.smoke_majorant(vi)
    if not n or mu_all <= 1e-12:
        return out
    levels = layer.levels(vi)
    t0, t1 = _clip(prep, o, d)
    t1 = np.minimum(t1, dist)
    alive = np.flatnonzero(t1 > t0)
    position = t0.copy()
    if levels is not None:
        o_cells = (prep.to_object(o) - prep.box_min) / volume.voxel_size
        d_cells = (d @ prep.inv[:3, :3].T) / volume.voxel_size
    for step in range(MAX_COLLISIONS):
        if not len(alive):
            break
        raytrace._cancel(cancel)
        here = position[alive]
        if levels is None:
            mu, leave = np.full(len(alive), mu_all), t1[alive]
        else:
            mu, leave = local_majorant(levels, o_cells[alive] + d_cells[alive] * here[:, None], d_cells[alive], here)
            leave = np.minimum(leave, t1[alive])
        u_step = _uniform(keys[alive], turn, NEE_RATIO_STREAM, step, 0, pcg, rand)
        candidate = here - np.log1p(-np.minimum(u_step, 1 - 1e-12)) / np.where(mu > 0, mu, 1.0)
        tentative = (mu > 0) & (candidate < leave)
        hop = ~tentative
        position[alive[hop]] = leave[hop] + 1e-5
        carry = np.zeros(len(alive), bool)
        carry[hop] = position[alive[hop]] < t1[alive[hop]]
        ray = alive[tentative]
        if len(ray):
            position[ray] = candidate[tentative]
            g = prep.to_grid(prep.to_object(o[ray] + d[ray] * position[ray][:, None]))
            sigma_t = layer.sigma_t_unit * settings.density_scale * _trilinear(volume.density, g)
            out[ray] *= np.clip(1.0 - sigma_t / mu[tentative], 0.0, 1.0)
            carry[tentative] = True
        alive = alive[carry]
    return out


def near_fade(emitter, dist):
    """How much of the sampled fire a point at distance `dist` from a smoke vertex keeps: 1, faded to 0 over the first
    `NEE_NEAR_CELLS` voxels as (r / that)^2 (a point beside the vertex is better found by the continuation ray, which integrates the
    glow along its short flight, than by a draw from the whole fire)."""
    if emitter.near <= 0:
        return np.ones_like(np.asarray(dist, np.float64))
    return np.minimum(1.0, (dist / emitter.near) ** 2)


def sampled_share(emitter, pv, dist, cosine_pdf):
    """The share of a fire point at distance `dist` that goes to the sampled fire, the continuation ray keeping the rest: the balance
    heuristic `pv delta r^2 / (pv delta r^2 + phase)` times `near_fade`."""
    balance = pv * emitter.delta * dist * dist
    return balance / np.maximum(balance + cosine_pdf, 1e-300) * near_fade(emitter, dist)


def fire_nee_value(layer, wi, dist, e, pv, cosine_pdf, transmittance):
    """What a fire point adds at a smoke vertex, before the path throughput, the smoke colour and the fire-light gain:
    `phase * e * T / (pv r^2)` times `sampled_share`, which reduces to `phase * e * T * delta * fade / (pv delta r^2 + phase)`.
    `cosine_pdf` is the phase function at the direction to the point."""
    emitter = layer.emitter()
    denominator = pv * emitter.delta * dist * dist + cosine_pdf
    scale = np.where(denominator > 0, cosine_pdf * emitter.delta * near_fade(emitter, dist) * transmittance / np.maximum(denominator, 1e-300), 0.0)
    return e * scale[:, None]


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
