"""3D smoke and fire solver on a MAC grid, CPU reference (Lane 6, step C). See docs/FLUIDS_SPIKE.md.

Layout and units
    A grid of `nx` by `ny` by `nz` unit cells indexed [i, j, k] = [x, y, z] (the order of `scene3d.Volume`);
    +y is up. Cell-centred fields (`density`, `temperature`, `fuel`, `burn`, `pressure`) are (nx, ny, nz); the
    face-centred velocity is `u` (nx + 1, ny, nz), `v` (nx, ny + 1, nz) and `w` (nx, ny, nz + 1). Cell
    (i, j, k) has its centre at (i + .5, j + .5, k + .5); u[i, j, k] sits at (i, j + .5, k + .5), v at
    (i + .5, j, k + .5), w at (i + .5, j + .5, k). Lengths are cells and time is frame units as in
    docs/SIMULATION.md: a substep advances dt = 1 / substeps of a frame and velocity is cells per frame. The
    node layer below maps world units onto cells with `origin` and `voxel_size`.

One substep (`Smoke3D.step`)
    1. Emit density, temperature, fuel and velocity from the sources.
    2. Advect velocity (semi-Lagrangian, midpoint backtrace, trilinear) and the scalars (semi-Lagrangian or
       MacCormack with a min/max clamp to the neighbours of the backtrace, faded out above one cell of
       travel per substep). A backtrace that leaves the
       domain through an open boundary brings in fresh air (zero density and fuel, ambient temperature).
    3. Combustion (`fire`): where fuel is present and temperature is at or above `ignition_temperature`, a
       fraction 1 - exp(-burn_rate dt) of the fuel burns; it releases `burn_heat` per unit of fuel as
       temperature, `burn_smoke` as density, and an expansion source `burn_expansion * burn` (the burn rate,
       fuel per frame) that becomes a positive velocity divergence in the projection. `burn` is the flame
       channel.
    4. Cooling (`cooling_rate`, exponential toward ambient) and dissipation (`dissipation`, exponential loss).
    5. Forces: buoyancy `v += dt (-alpha density + beta (T - ambient))` (unless a buoyancy force node
       replaces it), then the force list (gravity, wind, turbulence, drag).
    6. Vector vorticity confinement, f = epsilon (N x w) with N the unit gradient of |w| (Fedkiw, Stam and
       Jensen 2001), on the cell-centred curl.
    7. Project. Faces of a solid cell take the solid's velocity, faces on a closed boundary are zero; the
       pressure system is the 7-point Laplacian over the fluid cells (a solid or closed-wall neighbour drops
       out, an open-boundary neighbour is a p = 0 cell), solved through the `pressure_solver` hook, by default
       conjugate gradient warm-started from the last pressure, until the largest cell residual is at most
       `tolerance` or `max_iterations`; then the pressure gradient is subtracted from the faces.

Boundaries: `boundary_x`, `boundary_y`, `boundary_z` are "closed" (wall on both ends, free-slip) or "open" (both
ends open: outflow to p = 0). A closed box has a singular pressure system, so its right-hand side is made
zero-mean over the fluid cells; with expansion the box compensates for the net expansion.

Determinism: the solver draws no random numbers of its own (noise is a hash of the seed, frame and cell) and
holds no state outside the `State` it is handed, so the same parameters and seed give bit-identical grids on the
same machine however the frames were reached. Reductions are single-threaded einsum, as in fluid2d.

The simcache forward solve calls `initial_state(seed)` and `step(state, frame, substep, seed)`; `checkpoint`
and `restore` copy a State in and out of a cache.
"""
from __future__ import annotations

import math

import numpy as np

from .cancellation import Cancelled
from . import simcache
from .simcache import State

ARRAYS = ("u", "v", "w", "density", "temperature", "fuel", "burn", "pressure")
CANCEL_POLL = 8          # CG iterations between cancellation checks
# Solver-call counter for tests and benchmarks: one tick per substep actually solved.
SOLVER_STATS = {"steps": 0}
ADVECTIONS = ("semi_lagrangian", "maccormack")
BOUNDARIES = ("closed", "open")

DEFAULTS = {
    "nx": 32, "ny": 48, "nz": 32,
    "auto_resize": 0, "padding": 8, "max_size": 256,
    "substeps": 1,
    "advection": "maccormack",
    "buoyancy_density": 0.05,       # alpha: downward pull of density, cells / frame^2 per unit density
    "buoyancy_temperature": 0.8,    # beta: upward push of temperature, cells / frame^2 per unit
    "ambient_temperature": 0.0,
    "vorticity": 0.3,               # epsilon, vorticity confinement strength
    "dissipation": 0.0,             # density loss per frame (exponential)
    "cooling_rate": 0.0,            # temperature relaxation toward ambient per frame (exponential)
    "boundary_x": "closed", "boundary_y": "closed", "boundary_z": "closed",
    "tolerance": 1.0e-3,            # largest allowed |divergence residual| per cell after projection
    "max_iterations": 1500,         # conjugate-gradient cap per substep
    # fire
    "fire": 0,
    "ignition_temperature": 0.5, "burn_rate": 0.6, "burn_heat": 2.0, "burn_smoke": 0.3, "burn_expansion": 0.0,
    "fuel_inefficiency": 0.0, "temperature_output": 2.0, "smoke_output": 0.3,
    "gas_release": 0.0, "flame_lifespan": 1.0,
    # world mapping (the node layer sets these; the standalone solver works in cells)
    "origin_x": 0.0, "origin_y": 0.0, "origin_z": 0.0, "voxel_size": 1.0,
    # the built-in source: a sphere at these fractions of the grid (set default_source to 0 to turn it off)
    "default_source": 1,
    "source_x": 0.5, "source_y": 0.12, "source_z": 0.5,
    "source_radius": 0.08,               # fraction of nx
    "source_density": 1.0, "source_temperature": 1.0, "source_fuel": 0.0,
    # Shape tab (Houdini Pyro vocabulary, docs/FLUIDS_SPIKE.md "Shape controls"). Every strength below is
    # 0 ("off") by default so a stream built before this step solves bit-identically. `dissipation` above gains
    # an optional control field; `vorticity` above ("confinement") is unchanged.
    "dissipation_field": "none", "dissipation_range_lo": 0.0, "dissipation_range_hi": 1.0, "dissipation_ramp": 0.0,
    "disturbance": 0.0,               # block-size random velocity kicks, cells / frame^2
    "disturbance_size": 4.0,          # block edge, cells
    "disturbance_field": "none", "disturbance_range_lo": 0.0, "disturbance_range_hi": 1.0, "disturbance_ramp": 0.0,
    "shredding": 0.0,                 # (v . grad) v self-advection, extra stretch along the velocity gradient
    "turbulence": 0.0,                # curl-noise velocity forcing, cells / frame^2
    "swirl_size": 1.0,                # noise lattice cell, in cells (nodebased.particles.turbulence_field's `size`)
    "grain": 2,                       # fbm octaves (nodebased.particles.turbulence_field's `octaves`)
    "pulse_length": 30.0,             # frames between one noise pattern and the next it blends toward
    "turbulence_field": "none", "turbulence_range_lo": 0.0, "turbulence_range_hi": 1.0, "turbulence_ramp": 0.0,
}

CONTROL_FIELDS = ("none", "density", "temperature", "speed", "vorticity")


# --- sampling -------------------------------------------------------------------------------------

class Stencil:
    """Trilinear sampling weights for index-space points, shared by every field sampled at them."""
    __slots__ = ("base", "tx", "ty", "tz", "sy", "sz")

    def __init__(self, shape, x, y, z):
        nx, ny, nz = shape
        x = np.clip(x, 0.0, nx - 1.0)
        y = np.clip(y, 0.0, ny - 1.0)
        z = np.clip(z, 0.0, nz - 1.0)
        i0 = np.minimum(x.astype(np.intp), nx - 2)
        j0 = np.minimum(y.astype(np.intp), ny - 2)
        k0 = np.minimum(z.astype(np.intp), nz - 2)
        self.tx = x - i0
        self.ty = y - j0
        self.tz = z - k0
        self.sy = ny * nz
        self.sz = nz
        self.base = i0 * self.sy + j0 * nz + k0

    def corners(self, field):
        flat = field.reshape(-1)
        b, sy, sz = self.base, self.sy, self.sz
        return (flat.take(b), flat.take(b + 1), flat.take(b + sz), flat.take(b + sz + 1),
                flat.take(b + sy), flat.take(b + sy + 1), flat.take(b + sy + sz), flat.take(b + sy + sz + 1))

    def sample(self, field, corners=None):
        c000, c001, c010, c011, c100, c101, c110, c111 = self.corners(field) if corners is None else corners
        tz, ty, tx = self.tz, self.ty, self.tx
        c00 = c000 + (c001 - c000) * tz
        c01 = c010 + (c011 - c010) * tz
        c10 = c100 + (c101 - c100) * tz
        c11 = c110 + (c111 - c110) * tz
        c0 = c00 + (c01 - c00) * ty
        c1 = c10 + (c11 - c10) * ty
        return c0 + (c1 - c0) * tx

    def sample_bounded(self, field):
        """(value, low, high): the trilinear value and the min and max of its eight corners."""
        corners = self.corners(field)
        lo = corners[0]
        hi = corners[0]
        for c in corners[1:]:
            lo = np.minimum(lo, c)
            hi = np.maximum(hi, c)
        return self.sample(field, corners), lo, hi


def trilerp(field, x, y, z):
    """Trilinear sample of `field` at index-space points, edge-clamped."""
    return Stencil(field.shape, x, y, z).sample(field)


# --- the pressure system --------------------------------------------------------------------------

def divergence(u, v, w):
    return (u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])


def _dot(a, b):
    """Single-threaded, fixed-order dot product (BLAS threads change the order and stall under load)."""
    return float(np.einsum("ijk,ijk->", a, b))


class Poisson3D:
    """The negative 7-point Laplacian over the fluid cells: A q = rhs.

    A face between two fluid cells couples them; a face to a solid cell or a closed wall drops out
    (Neumann); a face to the outside of an open boundary adds one to the diagonal (a p = 0 neighbour).
    Rows of solid cells are zero (their pressure stays zero). `solid` is a bool array or None.
    """

    def __init__(self, shape, solid=None, open_axes=(False, False, False)):
        self.shape = tuple(shape)
        self.solid = None if solid is None or not np.any(solid) else np.asarray(solid, bool)
        self.open_axes = tuple(bool(a) for a in open_axes)
        fluid = None if self.solid is None else ~self.solid
        self.fluid = fluid
        self.cx = self.cy = self.cz = None
        if fluid is not None:
            self.cx = (fluid[:-1] & fluid[1:]).astype(np.float64)
            self.cy = (fluid[:, :-1] & fluid[:, 1:]).astype(np.float64)
            self.cz = (fluid[:, :, :-1] & fluid[:, :, 1:]).astype(np.float64)
        # open faces: (low face coefficient, high face coefficient) per axis, or None
        self.ends = [None, None, None]
        for axis in range(3):
            if not self.open_axes[axis]:
                continue
            lo = np.take(fluid, 0, axis=axis) if fluid is not None else np.ones(_face_shape(shape, axis), bool)
            hi = np.take(fluid, -1, axis=axis) if fluid is not None else np.ones(_face_shape(shape, axis), bool)
            self.ends[axis] = (lo.astype(np.float64), hi.astype(np.float64))
        self.singular = not any(self.open_axes)
        self._diag = None

    def apply(self, q, out):
        out[...] = 0.0
        for axis, coef in enumerate((self.cx, self.cy, self.cz)):
            lo = _sl(axis, slice(None, -1))
            hi = _sl(axis, slice(1, None))
            d = q[lo] - q[hi]
            if coef is not None:
                d *= coef
            out[lo] += d
            out[hi] -= d
            ends = self.ends[axis]
            if ends is not None:
                first, last = _sl(axis, 0), _sl(axis, -1)
                out[first] += ends[0] * q[first]
                out[last] += ends[1] * q[last]
        return out

    def diagonal(self):
        """Per-cell diagonal (the number of fluid or open neighbours), float64; zero on solid cells."""
        if self._diag is None:
            diag = np.zeros(self.shape, np.float64)
            for axis, coef in enumerate((self.cx, self.cy, self.cz)):
                lo = _sl(axis, slice(None, -1))
                hi = _sl(axis, slice(1, None))
                one = 1.0 if coef is None else coef
                diag[lo] += one
                diag[hi] += one
                ends = self.ends[axis]
                if ends is not None:
                    diag[_sl(axis, 0)] += ends[0]
                    diag[_sl(axis, -1)] += ends[1]
            self._diag = diag
        return self._diag

    def neighbour_bits(self):
        """uint32 per cell: bit 0..5 set when the -x, +x, -y, +y, -z, +z neighbour is a coupled fluid cell."""
        bits = np.zeros(self.shape, np.uint32)
        for axis, coef in enumerate((self.cx, self.cy, self.cz)):
            lo = _sl(axis, slice(None, -1))
            hi = _sl(axis, slice(1, None))
            on = np.ones(_pair_shape(self.shape, axis), bool) if coef is None else coef > 0
            bits[hi] |= (on.astype(np.uint32) << np.uint32(2 * axis))
            bits[lo] |= (on.astype(np.uint32) << np.uint32(2 * axis + 1))
        return bits


def _sl(axis, index):
    key = [slice(None)] * 3
    key[axis] = index
    return tuple(key)


def _face_shape(shape, axis):
    return tuple(n for a, n in enumerate(shape) if a != axis)


def _pair_shape(shape, axis):
    return tuple(n - 1 if a == axis else n for a, n in enumerate(shape))


def conjugate_gradient(rhs, x0, tolerance, max_iterations, cancel=None, system=None):
    """Solve A x = rhs for `system` (a Poisson3D). Stops when max |rhs - A x| <= tolerance.

    Returns (x, iterations, residual). Deterministic: fixed operation order, no random start.
    """
    x = x0.copy()
    ap = np.empty_like(x)
    r = rhs - system.apply(x, ap)
    residual = float(np.abs(r).max())
    if residual <= tolerance:
        return x, 0, residual
    p = r.copy()
    rs = _dot(r, r)
    iterations = 0
    while iterations < max_iterations:
        if cancel is not None and iterations % CANCEL_POLL == 0 and cancel.is_set():
            raise Cancelled()
        system.apply(p, ap)
        denom = _dot(p, ap)
        if denom <= 0.0:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * ap
        iterations += 1
        residual = float(np.abs(r).max())
        if residual <= tolerance:
            break
        rs_new = _dot(r, r)
        p *= rs_new / rs
        p += r
        rs = rs_new
    return x, iterations, residual


# --- voxelising geometry --------------------------------------------------------------------------

def _triangle_box_overlap(tri, centres):
    """Separating-axis test of one triangle (3, 3) against unit boxes centred at `centres` (M, 3)."""
    v = tri[None, :, :] - centres[:, None, :]                      # (M, 3 vertices, 3)
    alive = np.ones(len(centres), bool)
    for a in range(3):
        alive &= ~((v[:, :, a].min(axis=1) > 0.5) | (v[:, :, a].max(axis=1) < -0.5))
    e = (tri[1] - tri[0], tri[2] - tri[1], tri[0] - tri[2])
    n = np.cross(e[0], e[1])
    d = v[:, 0, :] @ n
    alive &= np.abs(d) <= 0.5 * np.abs(n).sum() + 1e-12
    for edge in e:
        for a in range(3):
            axis = np.cross(edge, np.eye(3)[a])
            if not axis.any():
                continue
            p = v @ axis
            r = 0.5 * np.abs(axis).sum()
            alive &= ~((p.min(axis=1) > r + 1e-12) | (p.max(axis=1) < -r - 1e-12))
    return alive


def voxelize_surface(triangles, shape, values=None):
    """Conservative surface voxelisation: every cell a triangle touches (triangles in cell coordinates, cell
    (i, j, k) spans [i, i + 1]). Returns a bool mask, and with `values` (T, 3 per triangle) also the
    per-cell mean of the values of the triangles that touch it as (nx, ny, nz, 3)."""
    mask = np.zeros(shape, bool)
    total = None if values is None else np.zeros(tuple(shape) + (3,), np.float64)
    count = None if values is None else np.zeros(shape, np.int32)
    hi_limit = np.array(shape) - 1
    for t, tri in enumerate(np.asarray(triangles, np.float64)):
        lo = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(axis=0)).astype(int), hi_limit)
        if np.any(hi < lo):
            continue
        ii, jj, kk = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                                 np.arange(lo[2], hi[2] + 1), indexing="ij")
        cells = np.stack((ii.ravel(), jj.ravel(), kk.ravel()), axis=1)
        keep = _triangle_box_overlap(tri, cells + 0.5)
        cells = cells[keep]
        if not len(cells):
            continue
        mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        if total is not None:
            total[cells[:, 0], cells[:, 1], cells[:, 2]] += values[t]
            count[cells[:, 0], cells[:, 1], cells[:, 2]] += 1
    if total is None:
        return mask
    return mask, total / np.maximum(count, 1)[..., None]


def fill_interior(surface):
    """`surface` plus every cell it encloses: the cells not reachable from outside without crossing it.
    Conservative voxel surfaces are 6-connected barriers, so a closed mesh always encloses its interior;
    an open mesh encloses nothing and the result is the surface itself."""
    if not surface.any():
        return surface
    idx = np.nonzero(surface)
    lo = np.maximum(np.array([a.min() for a in idx]) - 1, 0)
    hi = np.minimum(np.array([a.max() for a in idx]) + 2, surface.shape)
    crop = surface[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    free = ~crop
    outside = np.zeros_like(crop)
    for axis in range(3):
        outside[_sl(axis, 0)] = free[_sl(axis, 0)]
        outside[_sl(axis, -1)] |= free[_sl(axis, -1)]
    while True:
        grown = outside.copy()
        for axis in range(3):
            lower, upper = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            grown[upper] |= outside[lower]
            grown[lower] |= outside[upper]
        grown &= free
        if np.array_equal(grown, outside):
            break
        outside = grown
    result = surface.copy()
    result[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= ~outside
    return result


# --- sources, colliders and forces (solver-side objects; the node layer builds them) -----------------

class GeometryTrack:
    """World-space triangles of a geometry input, per frame. Static tracks are sampled once, at `start_frame`."""

    def __init__(self, provider, animated=False, start_frame=1, digest=None):
        self.provider = provider
        self.animated = bool(animated)
        self.start_frame = int(start_frame)
        self.digest = digest
        self._cache = {}

    def at(self, frame):
        key = int(frame) if self.animated else self.start_frame
        tri = self._cache.get(key)
        if tri is None:
            tri = np.asarray(self.provider(key), np.float64)
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = tri
        return tri

    def motion(self, frame):
        """Per-triangle displacement over the last frame in world units per frame, (T, 3); zeros when static
        or when the triangle count changed."""
        now = self.at(frame)
        if not self.animated:
            return np.zeros((len(now), 3), np.float64)
        before = self.at(int(frame) - 1)
        if before.shape != now.shape:
            return np.zeros((len(now), 3), np.float64)
        return (now - before).mean(axis=1)


class Source:
    """A place to emit density, temperature, fuel and velocity. All lengths are world units; the solver's
    `origin` and `voxel_size` map them to cells."""

    def __init__(self, emit_from="sphere", center=(0.0, 0.0, 0.0), radius=1.0, falloff=0.0, density=1.0,
                 temperature=1.0, fuel=0.0, velocity=(0.0, 0.0, 0.0), inherit_velocity=0.0,
                 noise_amount=0.0, noise_scale=1.0, start_frame=1, end_frame=1000000, track=None, seed=0,
                 params_at=None, fluid_type="smoke"):
        self.fluid_type = fluid_type      # "smoke" feeds FluidSolver3D, "liquid" seeds FluidLiquidSolver3D (flip3d)
        self.emit_from = emit_from
        self.center = np.asarray(center, np.float64)
        self.radius = float(radius)
        self.falloff = float(falloff)
        self.density, self.temperature, self.fuel = float(density), float(temperature), float(fuel)
        self.velocity = np.asarray(velocity, np.float64)
        self.inherit_velocity = float(inherit_velocity)
        self.noise_amount, self.noise_scale = float(noise_amount), float(noise_scale)
        self.start_frame, self.end_frame = int(start_frame), int(end_frame)
        self.track = track
        self.seed = int(seed)
        self.params_at = params_at        # optional callable frame -> dict of animated overrides
        self._footprints = {}

    def _knobs(self, frame):
        knobs = {name: getattr(self, name) for name in
                 ("radius", "falloff", "density", "temperature", "fuel", "inherit_velocity", "noise_amount",
                  "center")}
        knobs["velocity"] = self.velocity
        if self.params_at is not None:
            knobs.update(self.params_at(frame))
        return knobs

    def footprint(self, solver, frame):
        """(flat cell indices, weights, motion (M, 3) in cells per frame or None) on the solver's grid."""
        geo = self.emit_from in ("surface", "volume")
        animated = geo and self.track is not None and self.track.animated
        knobs = self._knobs(frame)
        radius = knobs["radius"]
        key = (int(frame) if animated else 0, radius, tuple(np.asarray(knobs["center"], float)), knobs["falloff"],
               tuple(solver.shape), tuple(np.asarray(solver.origin, float)))
        cached = self._footprints.get(key)
        if cached is not None:
            return cached
        shape = solver.shape
        if geo:
            if self.track is None:
                result = (np.zeros(0, np.intp), np.zeros(0), None)
            else:
                tri = (self.track.at(frame) - solver.origin) / solver.voxel
                if not len(tri):
                    result = (np.zeros(0, np.intp), np.zeros(0), None)
                else:
                    motion = self.track.motion(frame) / solver.voxel if self.inherit_velocity else None
                    if motion is not None:
                        surface, velocity = voxelize_surface(tri, shape, motion)
                    else:
                        surface, velocity = voxelize_surface(tri, shape), None
                    mask = fill_interior(surface) if self.emit_from == "volume" else surface
                    if velocity is not None:
                        mean = velocity[surface].mean(axis=0) if surface.any() else np.zeros(3)
                        velocity = np.where(surface[..., None], velocity, mean)
                    flat = np.flatnonzero(mask.reshape(-1))
                    moved = None if velocity is None else velocity.reshape(-1, 3)[flat]
                    result = (flat, np.ones(len(flat)), moved)
        else:
            centre = (np.asarray(knobs["center"], np.float64) - solver.origin) / solver.voxel
            r = radius / solver.voxel
            if self.emit_from == "point" or r < 0.5:
                ijk = np.clip(np.floor(centre).astype(int), 0, np.array(shape) - 1)
                flat = np.array([np.ravel_multi_index(tuple(ijk), shape)], np.intp)
                result = (flat, np.ones(1), None)
            else:
                lo = np.maximum(np.floor(centre - r).astype(int), 0)
                hi = np.minimum(np.ceil(centre + r).astype(int), np.array(shape))
                if np.any(hi <= lo):
                    result = (np.zeros(0, np.intp), np.zeros(0), None)
                else:
                    ii, jj, kk = np.meshgrid(*(np.arange(lo[a], hi[a]) for a in range(3)), indexing="ij")
                    dist = np.sqrt((ii + .5 - centre[0]) ** 2 + (jj + .5 - centre[1]) ** 2 + (kk + .5 - centre[2]) ** 2)
                    inside = dist <= r
                    t = dist[inside] / r
                    falloff = float(knobs["falloff"])
                    weight = np.clip(1.0 - falloff * t, 0.0, 1.0)
                    flat = np.ravel_multi_index((ii[inside], jj[inside], kk[inside]), shape)
                    result = (flat.astype(np.intp), weight, None)
        if len(self._footprints) > 8:
            self._footprints.clear()
        self._footprints[key] = result
        return result

    def emit(self, solver, arrays, frame, dt):
        if self.fluid_type != "smoke" or not (self.start_frame <= frame <= self.end_frame):
            return
        knobs = self._knobs(frame)
        flat, weight, motion = self.footprint(solver, frame)
        if not len(flat):
            return
        if knobs["noise_amount"]:
            from .particles import _value_noise
            ijk = np.stack(np.unravel_index(flat, solver.shape), axis=1) + 0.5
            drift = np.array((0.11 * frame, 0.0, 0.0))
            noise = _value_noise((ijk * solver.voxel) / max(self.noise_scale, 1e-9) + drift, self.seed, 41)
            weight = weight * np.clip(1.0 + knobs["noise_amount"] * noise, 0.0, None)
        wf = weight.astype(arrays["density"].dtype)
        for name, amount in (("density", knobs["density"]), ("temperature", knobs["temperature"]),
                             ("fuel", knobs["fuel"])):
            if amount:
                arrays[name].reshape(-1)[flat] += wf * arrays[name].dtype.type(amount * dt)
        target = np.asarray(knobs["velocity"], np.float64) / solver.voxel
        inherit = float(knobs["inherit_velocity"])
        if not (target.any() or (inherit and motion is not None)):
            return
        ijk = np.stack(np.unravel_index(flat, solver.shape), axis=1)
        for axis, face in enumerate(("u", "v", "w")):
            goal = np.full(len(flat), target[axis])
            if inherit and motion is not None:
                goal = goal + inherit * motion[:, axis]
            blend = np.minimum(weight, 1.0).astype(np.float32)
            goal = goal.astype(np.float32)
            array = arrays[face]
            for side in (0, 1):
                index = ijk.copy()
                index[:, axis] += side
                old = array[index[:, 0], index[:, 1], index[:, 2]]
                array[index[:, 0], index[:, 1], index[:, 2]] = old + blend * (goal - old)


class Collider:
    """A solid: cells covered by a closed (or surface-only) geometry. `animated` makes the solid move
    with the animated geometry, resampled between the frame either side of the current one and
    linearly interpolated per substep (mirroring ParticleBounce3D's `animated`, docs/SIMULATION.md
    "Bounce and collisions (animated)"), with its own velocity imposed on the boundary cells so a
    moving object pushes the fluid; otherwise it is frozen at the track's start frame."""

    def __init__(self, track, animated=False):
        self.track = track
        self.animated_flag = bool(animated)
        self._cache = {}

    @property
    def animated(self):
        return self.animated_flag and self.track.animated

    def mask(self, solver, frame, substep=0, substeps=1):
        """(solid bool (nx, ny, nz), velocity (nx, ny, nz, 3) cells per frame or None) voxelised at
        the fractional time of `substep` of `substeps` within `frame`. Substep 0 of 1 (the default)
        samples exactly the frame's own position, bit-identical to the old once-per-frame behaviour."""
        domain = (tuple(solver.shape), tuple(np.asarray(solver.origin, float)))
        key = ((int(frame), int(substep)) if self.animated else (0,)) + domain
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not self.animated:
            tri = (self.track.at(frame) - solver.origin) / solver.voxel
            result = ((np.zeros(solver.shape, bool), None) if not len(tri)
                      else (fill_interior(voxelize_surface(tri, solver.shape)), None))
        else:
            now, nxt = self.track.at(frame), self.track.at(int(frame) + 1)
            if now.shape != nxt.shape:
                world, motion_cells = now, np.zeros((len(now), 3))
            else:
                t = float(substep) / max(1, substeps)
                world = now + t * (nxt - now)
                motion_cells = (nxt - now).mean(axis=1) / solver.voxel
            tri = (world - solver.origin) / solver.voxel
            if not len(tri):
                result = (np.zeros(solver.shape, bool), None)
            else:
                surface, velocity = voxelize_surface(tri, solver.shape, motion_cells)
                solid = fill_interior(surface)
                mean = motion_cells.mean(axis=0) if len(motion_cells) else np.zeros(3)
                result = (solid, np.where(surface[..., None], velocity, mean).astype(np.float32))
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = result
        return result


def _smooth(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Force:
    """A velocity force, one of buoyancy, gravity, wind, turbulence, drag. Lengths in world units."""

    def __init__(self, kind, params, params_at=None, seed=0):
        self.kind = kind
        self.params = dict(params)
        self.params_at = params_at
        self.seed = int(seed)
        self._lattice = {}

    def _p(self, frame):
        if self.params_at is None:
            return self.params
        return {**self.params, **self.params_at(frame)}

    def active(self, frame):
        p = self._p(frame)
        return int(p.get("from_frame", -1000000)) <= frame <= int(p.get("to_frame", 1000000))

    def apply(self, solver, arrays, frame, substep, dt):
        if not self.active(frame):
            return
        p = self._p(frame)
        kind = self.kind
        u, v, w = arrays["u"], arrays["v"], arrays["w"]
        if kind == "drag":
            k = arrays["u"].dtype.type(math.exp(-float(p["drag"]) * dt))
            for face in (u, v, w):
                face *= k
        elif kind in ("gravity", "wind"):
            d = np.array((p["dir_x"], p["dir_y"], p["dir_z"]), np.float64)
            length = np.linalg.norm(d)
            if length < 1e-12:
                return
            accel = d / length * float(p["strength"]) / solver.voxel * dt
            if kind == "wind":
                # a uniform acceleration: only visible where the air can leave (an open boundary), as a body force
                # on a closed box is absorbed by the pressure
                for axis, face in enumerate((u, v, w)):
                    if accel[axis]:
                        face += face.dtype.type(accel[axis])
            else:
                # gravity weighs the smoke: the acceleration scales with the local density, so dense smoke sinks
                # and thin smoke does not (uniform gravity on all the air changes nothing but the pressure)
                density = arrays["density"]
                solver.add_cell_force(arrays, np.stack([density * density.dtype.type(a) for a in accel]))
        elif kind == "buoyancy":
            solver.buoyancy(arrays, float(p["buoyancy_settle"]) / solver.voxel, float(p["buoyancy_lift"]) / solver.voxel,
                            float(p["ambient_temperature"]), dt)
        elif kind == "turbulence":
            t = frame + substep * dt
            field = self._turbulence(solver.shape, float(p["turbulence_scale"]) / solver.voxel,
                                     t * float(p["turbulence_speed"]))
            scale = float(p["strength"]) / solver.voxel * dt
            solver.add_cell_force(arrays, field * np.float32(scale))

    def _slice(self, shape, scale, index):
        key = (shape, scale, index)
        cached = self._lattice.get(key)
        if cached is None:
            size = tuple(int(math.ceil(n / scale)) + 3 for n in shape)
            rng = np.random.default_rng((self.seed, index & 0x7FFFFFFF, 7))
            cached = rng.random(size=(3,) + size).astype(np.float32) * 2.0 - 1.0
            if len(self._lattice) > 6:
                self._lattice.clear()
            self._lattice[key] = cached
        return cached

    @staticmethod
    def _to_grid(lattice, shape, scale):
        out = lattice
        for axis, n in enumerate(shape):
            p = (np.arange(n, dtype=np.float64) + 0.5) / scale
            i0 = np.floor(p).astype(np.intp)
            wgt = _smooth(p - i0).astype(np.float32)
            a = np.take(out, i0, axis=axis + 1)
            b = np.take(out, i0 + 1, axis=axis + 1)
            wshape = [1, 1, 1, 1]
            wshape[axis + 1] = n
            wgt = wgt.reshape(wshape)
            out = a + (b - a) * wgt
        return out

    def _turbulence(self, shape, scale, t):
        """Curl of a smooth noise potential on the grid, (3, nx, ny, nz) float32, divergence free."""
        scale = max(scale, 1e-3)
        s0 = int(math.floor(t))
        tw = np.float32(_smooth(t - s0))
        a = self._to_grid(self._slice(shape, scale, s0), shape, scale)
        b = self._to_grid(self._slice(shape, scale, s0 + 1), shape, scale)
        pot = a + (b - a) * tw
        gx = [np.gradient(pot[c], axis=0) for c in range(3)]
        gy = [np.gradient(pot[c], axis=1) for c in range(3)]
        gz = [np.gradient(pot[c], axis=2) for c in range(3)]
        curl = np.stack((gy[2] - gz[1], gz[0] - gx[2], gx[1] - gy[0]))
        return (curl * np.float32(scale)).astype(np.float32)


# --- the solver -----------------------------------------------------------------------------------

class Smoke3D:
    """The solver. Holds parameters only; every substep takes and returns a State."""

    def __init__(self, params=None, pressure_solver=None, cancel=None, dtype=np.float32,
                 sources=None, forces=(), colliders=(), replace_buoyancy=False):
        self.params = {**DEFAULTS, **(params or {})}
        p = self.params
        self.nx, self.ny, self.nz = int(p["nx"]), int(p["ny"]), int(p["nz"])
        if min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("the grid must be at least 4 cells on every side")
        self.shape = (self.nx, self.ny, self.nz)
        self.substeps = max(1, int(p["substeps"]))
        self.dt = 1.0 / self.substeps
        self.dtype = np.dtype(dtype)
        if p["advection"] not in ADVECTIONS:
            raise ValueError(f"advection must be one of {ADVECTIONS}")
        self.open_axes = tuple(p[f"boundary_{a}"] == "open" for a in "xyz")
        self.origin = np.array((p["origin_x"], p["origin_y"], p["origin_z"]), np.float64)
        self.voxel = float(p["voxel_size"])
        # A callable (rhs, x0, tolerance, max_iterations, cancel, system) -> (x, iterations, residual)
        # replaces the NumPy conjugate gradient; `system` is the Poisson3D of this substep. The NumPy one
        # stays the reference (see tools/benchmark_fluid3d.py).
        self.pressure_solver = pressure_solver or conjugate_gradient
        self.cancel = cancel
        self.pressure_seconds = 0.0
        if sources is None:
            sources = []
            if int(p["default_source"]):
                sources.append(Source(
                    "sphere", center=(p["source_x"] * self.nx, p["source_y"] * self.ny, p["source_z"] * self.nz),
                    radius=max(1.0, p["source_radius"] * self.nx), density=p["source_density"],
                    temperature=p["source_temperature"], fuel=p["source_fuel"]))
        self.sources = list(sources)
        self.forces = list(forces)
        self.colliders = list(colliders)
        self.replace_buoyancy = bool(replace_buoyancy)
        self._systems = {}

    # -- simcache API -------------------------------------------------------------------------------
    def initial_state(self, seed=0) -> State:
        d, (nx, ny, nz) = self.dtype, self.shape
        arrays = {"u": np.zeros((nx + 1, ny, nz), d), "v": np.zeros((nx, ny + 1, nz), d),
                  "w": np.zeros((nx, ny, nz + 1), d)}
        for name in ("density", "temperature", "fuel", "burn", "pressure"):
            arrays[name] = np.zeros(self.shape, d)
        arrays["temperature"][...] = d.type(self.params["ambient_temperature"])
        return State(arrays, {"substep_count": 0, "cg_iterations": 0, "cg_residual": 0.0,
                              "domain_shape": list(self.shape), "domain_origin": self.origin.tolist()}, copy=False)

    def checkpoint(self, state) -> State:
        return State(state.arrays, state.meta, copy=True)

    def restore(self, state) -> State:
        arrays = {name: np.asarray(state.arrays[name], dtype=self.dtype) for name in ARRAYS}
        return State(arrays, state.meta, copy=True)

    def step(self, state, frame=0, substep=0, seed=0) -> State:
        """One substep of dt = 1 / substeps frames. Pure: the input state is not modified."""
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        self._sync_domain(state)
        if int(self.params.get("auto_resize", 0)):
            state = self._resize_active_domain(state, frame=frame, include_sources=True)
            self._sync_domain(state)
        SOLVER_STATS["steps"] += 1
        dt, p, dtype = self.dt, self.params, self.dtype
        a = {name: state.arrays[name].astype(dtype) for name in ARRAYS if name != "pressure"}
        pressure = state.arrays["pressure"]
        solid, solid_velocity, system = self._solid_for(frame, substep)
        ambient = float(p["ambient_temperature"])

        for source in self.sources:
            source.emit(self, a, frame, dt)
        self._advect(a, dt, ambient)
        expansion = self._combust(a, dt, dtype) if int(p["fire"]) else None
        if not int(p["fire"]):
            a["burn"][...] = 0.0
        self._decay(a, dt, ambient)
        if solid is not None:
            for name in ("density", "fuel", "burn"):
                a[name][solid] = 0.0
            a["temperature"][solid] = dtype.type(ambient)
        if not self.replace_buoyancy:
            self.buoyancy(a, p["buoyancy_density"], p["buoyancy_temperature"], ambient, dt)
        for force in self.forces:
            force.apply(self, a, frame, substep, dt)
        self._disturb(a, frame, substep, dt)
        self._shred(a, dt)
        self._shape_turbulence(a, frame, substep, dt)
        self._confine(a, dt, solid)
        pressure, iterations, residual = self._project(a, pressure, expansion, solid, solid_velocity, system)

        meta = dict(state.meta)
        meta.update(substep_count=int(meta.get("substep_count", 0)) + 1, cg_iterations=int(iterations),
                    cg_residual=float(residual))
        a["pressure"] = pressure
        result = State(a, meta, copy=False)
        return self._resize_active_domain(result) if int(p.get("auto_resize", 0)) else result

    def _sync_domain(self, state):
        """Adopt the per-checkpoint box before replaying a frame after a resize."""
        shape = state.meta.get("domain_shape")
        if shape is None:
            return
        shape = tuple(int(v) for v in shape)
        changed = shape != self.shape
        if changed:
            self.shape = shape
            self.nx, self.ny, self.nz = self.shape
            self.params.update(nx=self.nx, ny=self.ny, nz=self.nz)
        origin = state.meta.get("domain_origin")
        if origin is not None and tuple(origin) != tuple(self.origin):
            self.origin = np.asarray(origin, np.float64)
            for axis, value in zip("xyz", self.origin):
                self.params[f"origin_{axis}"] = float(value)
            changed = True
        if changed:
            self._systems.clear()

    def _resize_active_domain(self, state, frame=None, include_sources=False):
        """Fit smoke/fuel to whole 8-cell tiles while preserving every world-space sample."""
        active = (state.arrays["density"] > 1.0e-6) | (state.arrays["fuel"] > 1.0e-6)
        padding = max(0, int(self.params.get("padding", 8)))
        cap = max(8, (int(self.params.get("max_size", 256)) // 8) * 8)
        if active.any():
            points = np.where(active)
            lo = np.min(points, axis=1).astype(np.float64)
            hi = np.max(points, axis=1).astype(np.float64) + 1
            has_bounds = True
        else:
            center = np.asarray(self.shape, np.float64) * 0.5
            lo, hi = center - 4.0, center + 4.0
            has_bounds = False
        if include_sources and frame is not None:
            for source in self.sources:
                if source.fluid_type != "smoke" or int(frame) < source.start_frame or int(frame) > source.end_frame:
                    continue
                knobs = source._knobs(frame)
                if source.emit_from in ("surface", "volume") and source.track is not None:
                    tri = np.asarray(source.track.at(frame), np.float64)
                    if not tri.size:
                        continue
                    points_world = tri.reshape(-1, 3)
                    slo = (points_world.min(axis=0) - self.origin) / self.voxel
                    shi = (points_world.max(axis=0) - self.origin) / self.voxel + 1
                else:
                    center = (np.asarray(knobs["center"], np.float64) - self.origin) / self.voxel
                    radius = max(0.0, float(knobs["radius"]) / self.voxel)
                    slo, shi = center - radius - 1, center + radius + 2
                lo, hi = np.minimum(lo, slo), np.maximum(hi, shi)
                has_bounds = True
        if has_bounds:
            lo -= padding
            hi += padding
        # Tile-aligned bounds may extend past the current box. Negative starts grow the
        # low faces and are reflected in domain_origin; high stops grow the opposite faces.
        start = np.floor(lo / 8.0).astype(int) * 8
        stop = np.maximum(8, np.ceil(hi / 8.0).astype(int) * 8)
        for axis in range(3):
            if stop[axis] - start[axis] > cap:
                middle = 0.5 * (lo[axis] + hi[axis])
                start[axis] = int(np.floor((middle - cap * 0.5) / 8.0)) * 8
                stop[axis] = start[axis] + cap
        new_shape = tuple(int(v) for v in (stop - start))
        if new_shape == self.shape and not np.any(start):
            return state
        arrays = {}
        for name, source in state.arrays.items():
            if source.ndim < 3:
                continue  # sparse-GPU tile masks are rebuilt for the resized allocation
            target_shape = list(new_shape)
            face_axis = {"u": 0, "v": 1, "w": 2}.get(name)
            if face_axis is not None:
                target_shape[face_axis] += 1
            target = np.zeros(tuple(target_shape), dtype=source.dtype)
            src_lo = np.maximum(0, start)
            dst_lo = np.maximum(0, -start)
            lengths = np.minimum(np.array(source.shape[:3]) - src_lo,
                                 np.array(target.shape[:3]) - dst_lo)
            lengths = np.maximum(0, lengths)
            src = tuple(slice(int(src_lo[a]), int(src_lo[a] + lengths[a])) for a in range(3))
            dst = tuple(slice(int(dst_lo[a]), int(dst_lo[a] + lengths[a])) for a in range(3))
            if source.ndim == 3:
                target[dst] = source[src]
            elif source.ndim > 3:
                target[dst + (slice(None),)] = source[src + (slice(None),)]
            arrays[name] = target
        meta = dict(state.meta)
        new_origin = self.origin + start * self.voxel
        meta.update(domain_shape=list(new_shape), domain_origin=new_origin.tolist())
        return State(arrays, meta, copy=False)

    # -- colliders and boundaries -------------------------------------------------------------------
    def _solid_for(self, frame, substep=0):
        if not self.colliders:
            cached = self._systems.get("none")
            if cached is None:
                cached = self._systems["none"] = (None, None, self._system(None, "none"))
            return cached
        animated = any(c.animated for c in self.colliders)
        key = (int(frame), int(substep)) if animated else 0
        cached = self._systems.get(key)
        if cached is not None:
            return cached
        solid = np.zeros(self.shape, bool)
        velocity = None
        for collider in self.colliders:
            mask, vel = collider.mask(self, frame, substep, self.substeps)
            solid |= mask
            if vel is not None:
                if velocity is None:
                    velocity = np.zeros(self.shape + (3,), np.float32)
                velocity[mask] = vel[mask]
        if not solid.any():
            solid = None
        result = (solid, velocity, self._system(solid, key))
        if len(self._systems) > 8:
            self._systems.clear()
        self._systems[key] = result
        return result

    def _system(self, solid, key):
        return Poisson3D(self.shape, solid, self.open_axes)

    # -- physics ------------------------------------------------------------------------------------
    def _grid(self, ox, oy, oz, shape):
        i = (np.arange(shape[0], dtype=np.float32) + np.float32(ox))[:, None, None]
        j = (np.arange(shape[1], dtype=np.float32) + np.float32(oy))[None, :, None]
        k = (np.arange(shape[2], dtype=np.float32) + np.float32(oz))[None, None, :]
        return (np.broadcast_to(i, shape).reshape(-1), np.broadcast_to(j, shape).reshape(-1),
                np.broadcast_to(k, shape).reshape(-1))

    @staticmethod
    def _velocity_at(u, v, w, x, y, z):
        return (trilerp(u, x, y - .5, z - .5), trilerp(v, x - .5, y, z - .5), trilerp(w, x - .5, y - .5, z))

    def _backtrace(self, u, v, w, x, y, z, dt):
        """Midpoint backtrace; returns the unclamped departure points."""
        vx, vy, vz = self._velocity_at(u, v, w, x, y, z)
        nx, ny, nz = self.shape
        xm = np.clip(x - 0.5 * dt * vx, 0.0, nx)
        ym = np.clip(y - 0.5 * dt * vy, 0.0, ny)
        zm = np.clip(z - 0.5 * dt * vz, 0.0, nz)
        mx, my, mz = self._velocity_at(u, v, w, xm, ym, zm)
        return x - dt * mx, y - dt * my, z - dt * mz

    def _advect(self, a, dt, ambient):
        u, v, w = a["u"], a["v"], a["w"]
        dtype = self.dtype
        nx, ny, nz = self.shape
        # face velocities first, all three from the old field
        new = {}
        for name, offset, field in (("u", (0, .5, .5), u), ("v", (.5, 0, .5), v), ("w", (.5, .5, 0), w)):
            x, y, z = self._grid(*offset, field.shape)
            bx, by, bz = self._backtrace(u, v, w, x, y, z, dt)
            off = np.array(offset, np.float32)
            new[name] = trilerp(field, bx - off[0], by - off[1], bz - off[2]).reshape(field.shape).astype(dtype)
        # scalars share one departure point and one stencil
        x, y, z = self._grid(.5, .5, .5, self.shape)
        bx, by, bz = self._backtrace(u, v, w, x, y, z, dt)
        st = Stencil(self.shape, bx - .5, by - .5, bz - .5)
        outside = np.zeros(bx.shape, bool)
        for axis, (pos, n) in enumerate(((bx, nx), (by, ny), (bz, nz))):
            if self.open_axes[axis]:
                outside |= (pos < 0.0) | (pos > n)
        maccormack = self.params["advection"] == "maccormack"
        if maccormack:
            # The error correction is only trustworthy while the reverse trace is: it is full strength below
            # one cell of travel per substep, fades out linearly and is off from two cells, where the scheme
            # stops conserving mass and the field turns to semi-Lagrangian.
            travel = np.sqrt((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2)
            trust = np.clip(2.0 - travel, 0.0, 1.0).astype(dtype)
        fresh = {"density": 0.0, "temperature": ambient, "fuel": 0.0}
        for name in ("density", "temperature", "fuel"):
            field = a[name]
            if maccormack:
                predicted, lo, hi = st.sample_bounded(field)
                # the forward step from the arrival cell by the same displacement, then the error correction
                fx, fy, fz = 2.0 * x - bx, 2.0 * y - by, 2.0 * z - bz
                back = Stencil(self.shape, fx - .5, fy - .5, fz - .5).sample(predicted.reshape(self.shape))
                result = predicted + trust * dtype.type(0.5) * (field.reshape(-1) - back)
                result = np.minimum(np.maximum(result, lo), hi)
                # MacCormack is not conservative (measured: a plume gains a third of its mass), so the field
                # is rescaled to the total the semi-Lagrangian result has, which keeps the sharper structure.
                base = dtype.type(fresh[name])
                gained = float(np.sum(result - base, dtype=np.float64))
                if gained > 1e-12:
                    result = (base + (result - base) * dtype.type(min(2.0, max(0.5, float(np.sum(
                        predicted - base, dtype=np.float64)) / gained)))).astype(dtype)
            else:
                result = st.sample(field)
            if outside.any():
                result = np.where(outside, dtype.type(fresh[name]), result)
            new[name] = result.reshape(self.shape).astype(dtype)
        a.update(new)

    def _combust(self, a, dt, dtype):
        p = self.params
        fuel, temperature = a["fuel"], a["temperature"]
        hot = (temperature >= dtype.type(p["ignition_temperature"])) & (fuel > 0.0)
        # Lifespan is the duration over which the reactant flame is consumed. Legacy output knobs remain
        # readable for saved graphs and override their modern alias only when their old value was changed.
        lifespan = max(float(p.get("flame_lifespan", 1.0)), 1e-6)
        fraction = dtype.type(1.0 - math.exp(-float(p["burn_rate"]) * dt / lifespan))
        fraction *= dtype.type(1.0 - min(1.0, max(0.0, float(p.get("fuel_inefficiency", 0.0)))))
        consumed = np.where(hot, fuel * fraction, dtype.type(0.0)).astype(dtype)
        fuel -= consumed
        heat = float(p.get("temperature_output", 2.0))
        smoke = float(p.get("smoke_output", 0.3))
        if heat == 2.0 and float(p["burn_heat"]) != 2.0:
            heat = float(p["burn_heat"])
        if smoke == 0.3 and float(p["burn_smoke"]) != 0.3:
            smoke = float(p["burn_smoke"])
        temperature += consumed * dtype.type(heat)
        a["density"] += consumed * dtype.type(smoke)
        a["burn"] = (consumed / dtype.type(dt)).astype(dtype)
        expansion = float(p.get("gas_release", 0.0))
        if expansion == 0.0:
            expansion = float(p["burn_expansion"])
        return None if expansion == 0.0 else (a["burn"] * dtype.type(expansion)).astype(np.float64)

    def _decay(self, a, dt, ambient):
        p, dtype = self.params, self.dtype
        if p["dissipation"]:
            k = math.exp(-float(p["dissipation"]) * dt)
            weight = self._field_weight(a, p["dissipation_field"], p["dissipation_range_lo"],
                                        p["dissipation_range_hi"], p["dissipation_ramp"])
            if weight is None:
                a["density"] *= dtype.type(k)
            else:
                a["density"] *= (1.0 - weight * (1.0 - k)).astype(dtype)
        if p["cooling_rate"]:
            k = dtype.type(math.exp(-float(p["cooling_rate"]) * dt))
            a["temperature"] -= dtype.type(ambient)
            a["temperature"] *= k
            a["temperature"] += dtype.type(ambient)

    def buoyancy(self, a, alpha, beta, ambient, dt):
        dtype = self.dtype
        force = dtype.type(-alpha) * a["density"] + dtype.type(beta) * (a["temperature"] - dtype.type(ambient))
        v = a["v"]
        v[:, 1:-1, :] += dtype.type(0.5 * dt) * (force[:, :-1, :] + force[:, 1:, :])

    def add_cell_force(self, a, field):
        """Add a cell-centred force (3, nx, ny, nz), already scaled by dt, to the faces (average of the two cells)."""
        a["u"][1:-1] += 0.5 * (field[0][:-1] + field[0][1:])
        a["v"][:, 1:-1] += 0.5 * (field[1][:, :-1] + field[1][:, 1:])
        a["w"][:, :, 1:-1] += 0.5 * (field[2][:, :, :-1] + field[2][:, :, 1:])

    def _cell_velocity(self, a):
        """Cell-centred (u, v, w), each (nx, ny, nz)."""
        u, v, w = a["u"], a["v"], a["w"]
        return (0.5 * (u[:-1] + u[1:]), 0.5 * (v[:, :-1] + v[:, 1:]), 0.5 * (w[:, :, :-1] + w[:, :, 1:]))

    def _curl(self, uc, vc, wc):
        """Cell-centred curl (wx, wy, wz) of a cell-centred velocity."""
        wx = np.gradient(wc, axis=1) - np.gradient(vc, axis=2)
        wy = np.gradient(uc, axis=2) - np.gradient(wc, axis=0)
        wz = np.gradient(vc, axis=0) - np.gradient(uc, axis=1)
        return wx, wy, wz

    # -- the shape tab: dissipation's field, disturbance, shredding, turbulence, confinement --------
    def _field_source(self, a, field):
        """The named control field, or None for "none" (every shape control applies everywhere)."""
        if field == "density":
            return a["density"]
        if field == "temperature":
            return a["temperature"]
        if field == "speed":
            uc, vc, wc = self._cell_velocity(a)
            return np.sqrt(uc * uc + vc * vc + wc * wc)
        if field == "vorticity":
            wx, wy, wz = self._curl(*self._cell_velocity(a))
            return np.sqrt(wx * wx + wy * wy + wz * wz)
        return None

    def _field_weight(self, a, field, lo, hi, ramp):
        """1 where `field` is in [lo, hi], 0 outside, a linear falloff `ramp` (a fraction of hi - lo) wide on
        each side; None (apply everywhere, the pre-shape-tab behaviour) when `field` is "none"."""
        source = self._field_source(a, field)
        if source is None:
            return None
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1e-9
        width = max(float(ramp), 0.0) * (hi - lo)
        value = source.astype(np.float64)
        if width <= 1e-12:
            return ((value >= lo) & (value <= hi)).astype(np.float64)
        rise = np.clip((value - (lo - width)) / width, 0.0, 1.0)
        fall = np.clip(((hi + width) - value) / width, 0.0, 1.0)
        return np.minimum(rise, fall)

    def _disturb(self, a, frame, substep, dt):
        """Block-size random velocity kicks: a hashed lattice of `disturbance_size`-cell blocks, one random
        direction each, refreshed every substep so consecutive substeps see independent kicks."""
        amount = float(self.params["disturbance"])
        if amount == 0.0:
            return
        size = max(1.0, float(self.params["disturbance_size"]))
        shape = self.shape
        blocks = tuple(max(1, int(math.ceil(n / size))) for n in shape)
        rng = np.random.default_rng((int(self.params.get("seed", 0)), int(frame), int(substep), 0x4453))
        kicks = rng.random(blocks + (3,)) * 2.0 - 1.0
        idx = [np.minimum(np.arange(n) // int(size), blocks[axis] - 1) for axis, n in enumerate(shape)]
        field = kicks[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
        weight = self._field_weight(a, self.params["disturbance_field"], self.params["disturbance_range_lo"],
                                    self.params["disturbance_range_hi"], self.params["disturbance_ramp"])
        if weight is not None:
            field = field * weight[..., None]
        scale = amount * dt
        self.add_cell_force(a, np.moveaxis(field, -1, 0).astype(np.float32) * np.float32(scale))

    def _shred(self, a, dt):
        """Vortex stretching: S . omega_hat, the symmetric strain-rate tensor (the velocity gradient's
        symmetric half) applied to the local vorticity *direction*. This is the mechanism that thins a
        smooth vortex sheet into filaments in real 3D turbulence (and is exactly zero for a flow with no
        z-variation, since 2D flows have no vortex stretching). Using the direction rather than the raw
        vorticity, as `_confine` does for its own gradient, keeps this a single power of the velocity
        gradient rather than a quadratic one: S . omega (not normalised) doubles as a positive feedback on
        any grid-scale noise the advection leaves behind and diverges within a few frames."""
        amount = float(self.params["shredding"])
        if amount == 0.0:
            return
        dtype = self.dtype
        uc, vc, wc = self._cell_velocity(a)
        vel = (uc, vc, wc)
        grad = [[np.gradient(vel[i], axis=j) for j in range(3)] for i in range(3)]
        wx, wy, wz = self._curl(uc, vc, wc)
        norm = np.sqrt(wx * wx + wy * wy + wz * wz) + dtype.type(1e-9)
        omega_hat = (wx / norm, wy / norm, wz / norm)
        force = []
        for i in range(3):
            component = None
            for j in range(3):
                term = 0.5 * (grad[i][j] + grad[j][i]) * omega_hat[j]
                component = term if component is None else component + term
            force.append(component)
        delta = np.stack(force) * np.float32(amount * dt)
        # A per-cell speed cap: shredding feeds its own output back in (a bigger velocity gradient makes a
        # bigger stretch next substep), so an uncapped strength is a positive feedback that diverges within a
        # handful of frames. Capping the induced change to the local speed (floored, so a still cell can still
        # start moving) keeps every strength stable while still scaling the effect below the cap.
        speed = np.sqrt(uc * uc + vc * vc + wc * wc)
        cap = np.maximum(speed, dtype.type(0.05))
        delta_mag = np.sqrt((delta * delta).sum(axis=0)) + dtype.type(1e-9)
        delta = delta * np.minimum(1.0, cap / delta_mag)
        self.add_cell_force(a, delta.astype(np.float32))

    def _shape_turbulence(self, a, frame, substep, dt):
        """Curl-noise velocity forcing (nodebased.particles.turbulence_field), blended between two lattice
        draws every `pulse_length` frames so the pattern keeps changing rather than looping."""
        amount = float(self.params["turbulence"])
        if amount == 0.0:
            return
        from .particles import turbulence_field
        swirl = max(float(self.params["swirl_size"]), 1e-3)
        grain = max(1, int(self.params["grain"]))
        pulse = max(float(self.params["pulse_length"]), 1e-3)
        seed = int(self.params.get("seed", 0))
        t = (float(frame) + substep * dt) / pulse
        s0 = math.floor(t)
        tw = _smooth(t - s0)
        nx, ny, nz = self.shape
        xs, ys, zs = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, np.arange(nz) + 0.5, indexing="ij")
        positions = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        field_a = turbulence_field(positions, "curl", swirl, grain, seed + 1013 * s0)
        field_b = turbulence_field(positions, "curl", swirl, grain, seed + 1013 * (s0 + 1))
        field = (field_a + (field_b - field_a) * tw).reshape(nx, ny, nz, 3)
        weight = self._field_weight(a, self.params["turbulence_field"], self.params["turbulence_range_lo"],
                                    self.params["turbulence_range_hi"], self.params["turbulence_ramp"])
        if weight is not None:
            field = field * weight[..., None]
        scale = amount * dt
        self.add_cell_force(a, np.moveaxis(field, -1, 0).astype(np.float32) * np.float32(scale))

    def _confine(self, a, dt, solid):
        eps = float(self.params["vorticity"])
        if eps == 0.0:
            return
        dtype = self.dtype
        uc, vc, wc = self._cell_velocity(a)
        wx, wy, wz = self._curl(uc, vc, wc)
        mag = np.sqrt(wx * wx + wy * wy + wz * wz)
        gx, gy, gz = np.gradient(mag, axis=0), np.gradient(mag, axis=1), np.gradient(mag, axis=2)
        norm = np.sqrt(gx * gx + gy * gy + gz * gz) + dtype.type(1e-9)
        nx_, ny_, nz_ = gx / norm, gy / norm, gz / norm
        scale = dtype.type(eps * dt)
        fx = scale * (ny_ * wz - nz_ * wy)
        fy = scale * (nz_ * wx - nx_ * wz)
        fz = scale * (nx_ * wy - ny_ * wx)
        if solid is not None:
            fx[solid] = fy[solid] = fz[solid] = 0.0
        self.add_cell_force(a, np.stack((fx, fy, fz)))

    def _face_constraints(self, a, solid, solid_velocity):
        """Set boundary and solid faces to their prescribed normal velocity."""
        u, v, w = a["u"], a["v"], a["w"]
        for axis, face in enumerate((u, v, w)):
            if not self.open_axes[axis]:
                face[_sl(axis, 0)] = 0.0
                face[_sl(axis, -1)] = 0.0
            if solid is None:
                continue
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            inner = _sl(axis, slice(1, -1))
            near = solid[lo] | solid[hi]
            if solid_velocity is None:
                face[inner][near] = 0.0
            else:
                sv = solid_velocity[..., axis]
                value = np.where(solid[lo], sv[lo], np.where(solid[hi], sv[hi], 0.0)).astype(face.dtype)
                block = face[inner]
                block[near] = value[near]
                face[inner] = block
            if self.open_axes[axis]:
                for index, cells in ((0, solid[_sl(axis, 0)]), (-1, solid[_sl(axis, -1)])):
                    edge = face[_sl(axis, index)]
                    edge[cells] = 0.0
                    face[_sl(axis, index)] = edge

    def _project(self, a, pressure, expansion, solid, solid_velocity, system):
        p = self.params
        dtype = self.dtype
        self._face_constraints(a, solid, solid_velocity)
        u, v, w = a["u"], a["v"], a["w"]
        rhs = -divergence(u, v, w).astype(np.float64)
        if expansion is not None:
            rhs += expansion
        if solid is not None:
            rhs[solid] = 0.0
        x0 = np.asarray(pressure, dtype=np.float64)
        if solid is not None:
            x0 = np.where(solid, 0.0, x0)
        if system.singular:
            fluid = None if solid is None else ~solid
            mean = rhs.mean() if fluid is None else rhs[fluid].mean()
            rhs -= mean
            if fluid is not None:
                rhs[solid] = 0.0
        started = self._now()
        q, iterations, residual = self.pressure_solver(rhs, x0, float(p["tolerance"]), int(p["max_iterations"]),
                                                       self.cancel, system)
        self.pressure_seconds += self._now() - started
        if system.singular:
            fluid = None if solid is None else ~solid
            q = q - (q.mean() if fluid is None else q[fluid].mean())
            if solid is not None:
                q = np.where(solid, 0.0, q)
        q32 = q.astype(dtype)
        for axis, face in enumerate((u, v, w)):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            inner = _sl(axis, slice(1, -1))
            grad = q32[hi] - q32[lo]
            coef = (None, system.cx, system.cy, system.cz)[axis + 1]
            if coef is not None:
                grad = grad * coef.astype(dtype)
            face[inner] -= grad
            ends = system.ends[axis]
            if ends is not None:
                face[_sl(axis, 0)] -= q32[_sl(axis, 0)] * ends[0].astype(dtype)
                face[_sl(axis, -1)] += q32[_sl(axis, -1)] * ends[1].astype(dtype)
        return q32, iterations, residual

    @staticmethod
    def _now():
        import time
        return time.perf_counter()


# --- diagnostics ----------------------------------------------------------------------------------

def centre_of_mass(density):
    """(x, y, z) of the density-weighted centre in cell units, or None for an empty grid."""
    total = float(density.sum(dtype=np.float64))
    if total <= 0.0:
        return None
    nx, ny, nz = density.shape
    xs = (np.arange(nx) + 0.5)[:, None, None]
    ys = (np.arange(ny) + 0.5)[None, :, None]
    zs = (np.arange(nz) + 0.5)[None, None, :]
    return tuple(float((density * axis).sum(dtype=np.float64)) / total for axis in (xs, ys, zs))


# --- the nodes ------------------------------------------------------------------------------------
# FluidSource3D, FluidForce3D and FluidCollide3D build a `FluidChain`; FluidSolver3D turns the chain and its own
# knobs into a `FluidStream` (one deterministic run, identified by a digest) and outputs a `scene3d.Volume` per
# frame; FluidCache3D keeps solved frames through a `simcache.SimCache` like ParticleCache3D. The run identity is
# the chain of node identities (knobs, curves, expressions, geometry digests) plus the solver's knobs and the
# resolved pressure backend, so any edit abandons the old frames exactly like an emitter edit does.

CHAIN_KINDS = ("FluidSource3D", "FluidForce3D", "FluidCollide3D")
MAX_CELLS = 16_777_216              # 256 cubed: the CPU reference is a bake-and-scrub tool, and memory is the wall
GPU_AUTO_CELLS = 1_000_000          # `pressure = auto` uses the wgpu solve at or above this many cells (and only when an adapter exists)
NODE_LIFT = 0.08                    # built-in buoyancy of a solver node, world units per frame squared per unit temperature
NODE_SETTLE = 0.005                 # and per unit density (downward)
CHANNEL_SETS = {"density": ("density",), "density_temperature": ("density", "temperature"),
                "density_temperature_velocity": ("density", "temperature", "velocity"),
                "all": ("density", "temperature", "velocity", "flame", "fuel")}
MAX_TRACK_FRAMES = 2000             # frames hashed to identify an animated geometry input


def resolution(params):
    """(nx, ny, nz) of a FluidSolver3D's grid: the bounds divided by `division_size`, rounded up, at least 4."""
    d = float(params["division_size"])
    lo = np.array([params[f"bounds_min_{a}"] for a in "xyz"], np.float64)
    hi = np.array([params[f"bounds_max_{a}"] for a in "xyz"], np.float64)
    if np.any(hi <= lo):
        raise ValueError("FluidSolver3D: bounds max must be above bounds min on every axis")
    n = np.maximum(4, np.ceil((hi - lo) / d - 1e-9)).astype(int)
    return tuple(int(v) for v in n)


class FluidChain:
    """What flows through the "fluid" wires: sources, forces and colliders, with the run digest so far."""

    def __init__(self, sources=(), forces=(), colliders=(), replace_buoyancy=False, run=None):
        self.sources = tuple(sources)
        self.forces = tuple(forces)
        self.colliders = tuple(colliders)
        self.replace_buoyancy = bool(replace_buoyancy)
        self.run = run

    def then(self, identity, *, source=None, force=None, collider=None, replace_buoyancy=False):
        return FluidChain(self.sources + ((source,) if source else ()), self.forces + ((force,) if force else ()),
                          self.colliders + ((collider,) if collider else ()),
                          self.replace_buoyancy or replace_buoyancy, simcache.run_key(self.run, identity))


def _resolver(kind, node, curves):
    """frame -> the node's knobs with animation curves applied (memoised)."""
    memo = {}

    def at(frame):
        got = memo.get(frame)
        if got is None:
            from .core import SPECS, LIMITS
            from .animation import resolve_params
            got = resolve_params({"params": node["params"]}, curves, frame, SPECS[kind]["params"], LIMITS)
            if len(memo) > 4096:
                memo.clear()
            memo[frame] = got
        return got
    return at


def _geo_track(evaluator, doc, source_key, cancel, start, animated):
    """(GeometryTrack, digest) of the geometry wired into a fluid node. A static track is sampled once at
    `start` (like ParticleBounce3D); an animated one is sampled per frame and identified by the digests over the
    document's time range."""
    from .particles import collider_triangles

    def provider(frame):
        geometry = evaluator.evaluate_raster(doc, source_key, cancel, frame=int(frame), typed=True)
        return collider_triangles(geometry)
    if not animated:
        _, digest = evaluator.evaluate_raster(doc, source_key, cancel, frame=start, typed=True, return_digest=True)
        return GeometryTrack(provider, False, start, digest), digest
    import hashlib
    time = doc.get("time", {})
    first = int(time.get("first", start))
    last = min(int(time.get("last", first)), first + MAX_TRACK_FRAMES)
    h = hashlib.sha256()
    for frame in range(first, last + 1):
        _, digest = evaluator.evaluate_raster(doc, source_key, cancel, frame=frame, typed=True, return_digest=True)
        h.update(str(digest).encode())
    digest = h.hexdigest()
    return GeometryTrack(provider, True, start, digest), digest


def chain_for(evaluator, doc, key, node, incoming, cancel=None):
    """The FluidChain a FluidSource3D, FluidForce3D or FluidCollide3D node produces. `incoming` is the chain
    wired into it (None for a source or an unwired slot)."""
    from .core import SPECS
    from . import scene3d
    kind, params = node["type"], node["params"]
    curves = doc.get("animation", {}).get("curves", {}).get(key)
    expressions = doc.get("expressions", {}).get(key)
    base = incoming if incoming is not None else FluidChain()
    if node["disabled"]:
        return FluidChain() if kind == "FluidSource3D" else base
    resolve = _resolver(kind, node, curves)
    identity = {"kind": kind, "params": params, "curves": curves, "expressions": expressions, "format": 1}
    if kind == "FluidSource3D":
        emit_from = params["fluid_emit_from"]
        track = None
        if emit_from in ("surface", "volume") and node["inputs"].get("geo") is not None:
            track, digest = _geo_track(evaluator, doc, node["inputs"]["geo"], cancel, int(params["start_frame"]),
                                       bool(params["src_inherit_velocity"]))
            identity["geo"] = digest

        def params_at(frame):
            p = resolve(frame)
            m = scene3d._transform_from(p).matrix().astype(np.float64)
            scale = abs(np.linalg.det(m[:3, :3])) ** (1.0 / 3.0)
            centre = m @ np.array((p["src_center_x"], p["src_center_y"], p["src_center_z"], 1.0))
            return {"center": centre[:3], "radius": float(p["src_radius"]) * scale, "falloff": p["src_falloff"],
                    "density": p["src_density"], "temperature": p["src_temperature"], "fuel": p["src_fuel"],
                    "velocity": m[:3, :3] @ np.array((p["src_vel_x"], p["src_vel_y"], p["src_vel_z"])),
                    "inherit_velocity": p["src_inherit_velocity"], "noise_amount": p["src_noise_amount"]}
        source = Source(emit_from, start_frame=params["start_frame"], end_frame=params["end_frame"], track=track,
                        fluid_type=params.get("fluid_type", "smoke"),
                        noise_scale=params["src_noise_scale"],
                        inherit_velocity=params["src_inherit_velocity"], params_at=params_at)
        return base.then(identity, source=source)
    if kind == "FluidForce3D":
        fkind = params["force_kind"]

        def force_at(frame):
            p = resolve(frame)
            return {"dir_x": p["force_dir_x"], "dir_y": p["force_dir_y"], "dir_z": p["force_dir_z"],
                    "strength": p["strength"], "drag": p["drag"], "buoyancy_lift": p["buoyancy_lift"],
                    "buoyancy_settle": p["buoyancy_settle"], "ambient_temperature": p["ambient_temperature"],
                    "turbulence_scale": p["turbulence_scale"], "turbulence_speed": p["turbulence_speed"],
                    "from_frame": p["from_frame"], "to_frame": p["to_frame"]}
        force = Force(fkind, force_at(int(params["from_frame"])), force_at, seed=int(params["seed"]))
        return base.then(identity, force=force, replace_buoyancy=(fkind == "buoyancy"))
    if kind == "FluidCollide3D":
        geo = node["inputs"].get("geometry")
        if geo is None:
            return base.then(identity)
        moving = bool(params["animated"])
        # a frozen collider is sampled at the first frame of the document's range (a source is frozen at its own
        # start frame); an animated one is sampled per frame and interpolated per substep (Collider.mask), and its
        # digest hashes the whole document time range so any edited keyframe abandons the run, as for particles'
        # ParticleBounce3D.animated (docs/SIMULATION.md, "Bounce and collisions (animated)")
        track, digest = _geo_track(evaluator, doc, geo, cancel, int(doc.get("time", {}).get("first", 1)), moving)
        identity["geo"], identity["animated"] = digest, moving
        return base.then(identity, collider=Collider(track, moving))
    raise ValueError(f"not a fluid chain node: {kind}")


class FluidStream:
    """One deterministic fluid run: enough to solve any frame of it. Built by `build_stream`."""

    def __init__(self, chain, params, run, fps):
        self.chain = chain
        self.params = dict(params)
        self.run = run
        self.fps = float(fps)
        self.start_frame = int(params["start_frame"])
        self.substeps = int(params["substeps"])
        self.seed = int(params["seed"])
        self.shape = resolution(params)
        self.origin = tuple(float(params[f"bounds_min_{a}"]) for a in "xyz")
        self.voxel = float(params["division_size"])
        self.backend = None          # "cpu" or "gpu", set by build_stream
        self._solver = None

    def solver(self, cancel=None):
        """The Smoke3D of this run (built once, then reused: its voxelised footprints and colliders are cached)."""
        if self._solver is None:
            p = self.params
            nx, ny, nz = self.shape
            for source in self.chain.sources:
                source.seed = self.seed
            shape_keys = ("auto_resize", "padding", "max_size", "dissipation_field", "dissipation_range_lo", "dissipation_range_hi", "dissipation_ramp",
                          "disturbance", "disturbance_size", "disturbance_field", "disturbance_range_lo",
                          "disturbance_range_hi", "disturbance_ramp", "shredding", "turbulence", "swirl_size",
                          "grain", "pulse_length", "turbulence_field", "turbulence_range_lo",
                          "turbulence_range_hi", "turbulence_ramp")
            params = {"nx": nx, "ny": ny, "nz": nz, "substeps": self.substeps, "advection": p["advection"],
                      "buoyancy_density": NODE_SETTLE / self.voxel, "buoyancy_temperature": NODE_LIFT / self.voxel,
                      "ambient_temperature": 0.0, "vorticity": p["vorticity"], "dissipation": p["dissipation"],
                      "cooling_rate": p["cooling_rate"], "boundary_x": p["boundary_x"], "boundary_y": p["boundary_y"],
                      "boundary_z": p["boundary_z"], "tolerance": p["tolerance"],
                      "max_iterations": p["max_iterations"], "fire": p["fire"],
                      "ignition_temperature": p["ignition_temperature"], "burn_rate": p["burn_rate"],
                      "burn_heat": p["burn_heat"], "burn_smoke": p["burn_smoke"],
                      "burn_expansion": p["burn_expansion"], "fuel_inefficiency": p["fuel_inefficiency"],
                      "temperature_output": p["temperature_output"], "smoke_output": p["smoke_output"],
                      "gas_release": p["gas_release"], "flame_lifespan": p["flame_lifespan"],
                      "origin_x": self.origin[0], "origin_y": self.origin[1],
                      "origin_z": self.origin[2], "voxel_size": self.voxel, "default_source": 0,
                      **{key: p[key] for key in shape_keys}}
            smoke_sources = [s for s in self.chain.sources if s.fluid_type == "smoke"]
            solver_fn = None
            if self.backend == "gpu":
                solver_fn = _gpu_solver().solve
            if self.backend in ("resident", "resident_sparse"):
                from .fluid_gpu_solver import GpuSmoke3D
                self._solver = GpuSmoke3D(params, sources=smoke_sources, forces=self.chain.forces,
                                          colliders=self.chain.colliders, replace_buoyancy=self.chain.replace_buoyancy,
                                          sparse=self.backend == "resident_sparse")
            else:
                self._solver = Smoke3D(params, pressure_solver=solver_fn, sources=smoke_sources,
                                       forces=self.chain.forces, colliders=self.chain.colliders,
                                       replace_buoyancy=self.chain.replace_buoyancy)
        self._solver.cancel = cancel
        return self._solver


_GPU = {}


def _gpu_solver():
    if "solver" not in _GPU:
        from .fluid_gpu3d import GpuPressure3D
        _GPU["solver"] = GpuPressure3D()
    return _GPU["solver"]


def resolve_backend(params, cells, shape=None):
    """`pressure` resolved to "cpu", "gpu" (the wgpu SOR pressure hook), "resident" or "resident_sparse" (the whole
    substep on the GPU, nodebased/fluid_gpu_solver.py). Auto picks the resident solver for big grids (where it wins,
    see the benchmarks in docs/FLUIDS_SPIKE.md) when it fits the card, else the GPU hook, else the CPU; the resolved
    name is part of the run identity because the solves agree to the tolerance, not bit for bit."""
    choice = params["pressure"]
    shape = shape or resolution(params)
    if choice == "cpu":
        return "cpu"
    from . import fluid_gpu3d, fluid_gpu_solver
    if choice in ("resident", "resident_sparse"):
        ok, reason = fluid_gpu_solver.fits(shape)
        if not ok:
            raise ValueError(f"FluidSolver3D: pressure is {choice} but the GPU solver cannot run here: {reason}")
        return choice
    if choice == "auto" and cells >= GPU_AUTO_CELLS and fluid_gpu_solver.fits(shape)[0]:
        return "resident"
    if choice == "gpu":
        if not fluid_gpu3d.available():
            raise ValueError("FluidSolver3D: pressure is gpu but no wgpu adapter can be opened here")
        return "gpu"
    return "gpu" if cells >= GPU_AUTO_CELLS and fluid_gpu3d.available() else "cpu"


def build_stream(doc, key, node, chain):
    """The FluidStream of one FluidSolver3D node, from the chain wired into it."""
    params = node["params"]
    shape = resolution(params)
    cells = shape[0] * shape[1] * shape[2]
    if cells > MAX_CELLS:
        raise ValueError(f"FluidSolver3D: {shape[0]} x {shape[1]} x {shape[2]} is {cells:,} cells; the CPU "
                         f"reference solver stops at {MAX_CELLS:,} (raise division_size or shrink the bounds)")
    backend = resolve_backend(params, cells, shape)
    if int(params.get("auto_resize", 0)) and backend == "resident":
        backend = "resident_sparse"
    base = chain if chain is not None else FluidChain()
    fps = float(doc.get("time", {}).get("fps", 24.0))
    identity = {"kind": "FluidSolver3D", "params": params, "backend": backend, "fps": fps,
                "format": 2 if int(params.get("auto_resize", 0)) else 1}
    run = simcache.run_key(base.run, identity)
    stream = FluidStream(base, params, run, fps)
    stream.backend = backend
    return stream


def solve_frame(stream, frame, cache, cancel=None):
    """The solved `simcache.State` at `frame`, from `cache` where possible."""
    solver = stream.solver(cancel)
    return simcache.solve_to_frame(cache, stream.run, int(frame), stream.start_frame, stream.substeps,
                                   stream.seed, solver.initial_state, solver.step, cancel)


def volume_from_state(state, stream, frame):
    """A `scene3d.Volume` (full precision) for one solved frame: density, temperature, a cell-centred velocity in
    world units per second and the flame (burn rate) channel. The volume's transform is the identity; the grid
    sits at the solver's bounds."""
    from .scene3d import Volume
    a = state.arrays
    scale = np.float32(stream.voxel * stream.fps)
    velocity = np.stack((0.5 * (a["u"][:-1] + a["u"][1:]), 0.5 * (a["v"][:, :-1] + a["v"][:, 1:]),
                         0.5 * (a["w"][:, :, :-1] + a["w"][:, :, 1:])), axis=-1) * scale
    origin = tuple(state.meta.get("domain_origin", stream.origin))
    return Volume(a["density"], voxel_size=stream.voxel, origin=origin, temperature=a["temperature"],
                  velocity=velocity, flame=a["burn"], fuel=a["fuel"], stream=stream, frame=int(frame))


def _sparse_of(vol, names):
    from .sparsevol import SparseGrid
    return SparseGrid.from_dense({name: getattr(vol, name) for name in names})


def placeholder_volume(stream, frame):
    """A one-voxel empty Volume that only names the run, for a FluidCache3D to solve."""
    from .scene3d import Volume
    return Volume(np.zeros((1, 1, 1), np.float32), voxel_size=stream.voxel, origin=stream.origin,
                  stream=stream, frame=int(frame))


def cached_volume(stream, frame, store, cancel, precision, channels):
    """The volume of `frame` through `store` (a SimCache): solver checkpoints under the run, and the served
    channels at `precision` under a derived run, so a scrub is a cache read. What is served is always the quantised
    copy, so a fresh solve and a cache hit are identical. A `resident_sparse` run serves sparse tiles (only the
    tiles that hold smoke, heat or flow are stored) and builds the dense arrays here, when the volume is asked for."""
    names = CHANNEL_SETS[channels]
    sparse = getattr(stream, "backend", None) == "resident_sparse"
    out_run = simcache.run_key(stream.run, {"out": [precision, channels] + (["sparse"] if sparse else [])})
    got = store.get(out_run, int(frame))
    if got is None:
        state = solve_frame(stream, frame, store, cancel)
        vol = volume_from_state(state, stream, frame)
        dtype = np.float16 if precision == "float16" else np.float32
        if sparse:
            grid = _sparse_of(vol, names)
            arrays = {"coords": grid.coords, **{name: grid.data[name].astype(dtype) for name in names}}
        else:
            arrays = {name: np.asarray(getattr(vol, name)).astype(dtype) for name in names}
        meta = {"frame": int(frame)}
        for key in ("domain_shape", "domain_origin"):
            if key in state.meta:
                meta[key] = state.meta[key]
        got = simcache.State(arrays, meta, copy=False)
        store.put(out_run, int(frame), got)
    from .scene3d import Volume
    a = got.arrays
    domain_shape = tuple(got.meta.get("domain_shape", stream.shape))
    domain_origin = tuple(got.meta.get("domain_origin", stream.origin))
    if sparse:
        from .sparsevol import SparseGrid
        grid = SparseGrid.from_arrays(domain_shape, {k: (v if k == "coords" else v.astype(np.float32)) for k, v in a.items()})
        return Volume.from_sparse(grid, voxel_size=stream.voxel, origin=domain_origin, stream=stream, frame=int(frame))
    return Volume(a["density"].astype(np.float32), voxel_size=stream.voxel, origin=domain_origin,
                  temperature=a.get("temperature"), velocity=a.get("velocity"), flame=a.get("flame"), fuel=a.get("fuel"),
                  stream=stream, frame=int(frame))
