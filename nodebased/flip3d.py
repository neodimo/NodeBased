"""FLIP/PIC liquid solver on the MAC grid of fluid3d.py, CPU reference (Lane 6, step E). See docs/FLUIDS_SPIKE.md.

Particles carry position and velocity; a MAC grid does the pressure step (Zhu and Bridson 2005, Bridson's
"Fluid Simulation for Computer Graphics"). Layout, units, cell and face indexing are those of `fluid3d`: a grid of
`nx` by `ny` by `nz` unit cells [i, j, k] = [x, y, z], +y up, faces `u` (nx + 1, ny, nz), `v`, `w`, lengths in cells
and time in frames inside the solver. The State stores particles in WORLD units (position, and velocity in world
units per frame) so that `particles.instance_from_state` and ParticleCache3D handle a liquid unchanged; the node
layer maps them with `origin` and `voxel_size`.

One substep (`Liquid3D.step`)
    1. Emit: a liquid `fluid3d.Source` seeds `particles_per_cell` jittered particles into every free cell of its
       footprint at the start frame (a still source fills once); a source with a velocity keeps pouring, topping
       up the footprint cells that hold fewer than half the target count.
    2. Maintain: cells over the maximum (12, or 1.5 times the target) lose their highest-id particles; a cell
       with fewer than 3 whose six neighbours all hold particles (an interior gap) is topped up to 3 with the
       mean velocity of its neighbours. Ids grow by one per new particle.
    3. Particle to grid: trilinear weights onto the three face grids (`Stencil` of fluid3d), velocity divided by
       weight, the grid velocity before forces kept as `old`.
    4. Forces: gravity (`gravity`, cells per frame squared, along -y), the chain's forces (gravity, wind, drag,
       turbulence: `fluid3d.Force` applied to a density of one in the liquid cells), viscosity (implicit
       diffusion) if `viscosity` is above zero.
    5. Classify: a cell is liquid where it holds a particle and is not solid, solid where a collider or the
       domain wall is, air otherwise. Faces of solid cells and the domain walls take the solid velocity.
    6. Project: the pressure system is the 7-point Laplacian over the liquid cells with p = 0 in air (the free
       surface) and Neumann at solids, solved by the `pressure_solver` hook (conjugate gradient by default), then
       the gradient is subtracted from every face that touches a liquid cell.
    7. Extrapolate the projected and the old grid velocity a few layers into the air (same valid masks), so the
       particles near the surface see defined values.
    8. Grid to particle: `flip_ratio` blends FLIP (`v_p + interp(new - old)`) with PIC (`interp(new)`).
    9. Advect the particles with midpoint (RK2) sub-steps, at most one cell per sub-step; a particle that would
       enter a solid cell stays where it was, and one that leaves the domain is put back on the wall.

Determinism: the only randomness is `np.random.default_rng((seed, frame, substep, salt))`; reductions are
`np.bincount` and single-threaded einsum, so the same parameters and seed give bit-identical states however the
frames were reached. The simcache forward solve calls `initial_state(seed)` and `step(state, frame, substep,
seed)`; `checkpoint` and `restore` copy a State in and out of a cache.
"""
from __future__ import annotations

import hashlib
import math
import copy

import numpy as np

from .cancellation import Cancelled
from .fluid3d import Poisson3D, Smoke3D, Stencil, conjugate_gradient, fill_interior, _sl
from .simcache import State

ARRAYS = ("position", "velocity", "age", "life", "size", "color", "id", "temperature")
SOLVER_STATS = {"steps": 0}
MIN_PER_CELL = 3
MAX_PER_CELL = 12
EXTRAPOLATE_LAYERS = 4
MAX_ADVECT_SUBSTEPS = 6
LIFE = 2 ** 30                       # "lifetime" of a liquid particle: it never dies of age
LIQUID_COLOR = (0.30, 0.55, 0.85, 1.0)

DEFAULTS = {
    "nx": 32, "ny": 32, "nz": 32, "substeps": 1, "flip_ratio": 0.95, "particles_per_cell": 8,
    "gravity": 0.03,                # cells per frame squared, along -y (the node layer converts from world units)
    "viscosity": 0.0,               # cells squared per frame; implicit diffusion, off at zero
    "surface_tension": 0.0,         # curvature acceleration in cells / frame^2; off keeps legacy runs bit-identical
    "viscosity_by_attribute": "none", # optional particle temperature multiplier
    "narrow_band": 0.0,             # retain particles this many cells from the interface; zero is full FLIP
    "boundary_x_min": "closed", "boundary_x_max": "closed",
    "boundary_y_min": "closed", "boundary_y_max": "closed",
    "boundary_z_min": "closed", "boundary_z_max": "closed",
    "tolerance": 1.0e-3, "max_iterations": 1500,
    "origin_x": 0.0, "origin_y": 0.0, "origin_z": 0.0, "voxel_size": 1.0,
    "start_frame": 1, "seed": 0,
    "max_per_cell": MAX_PER_CELL,
}


def empty_arrays():
    return {"position": np.zeros((0, 3), np.float32), "velocity": np.zeros((0, 3), np.float32),
            "age": np.zeros(0, np.int32), "life": np.zeros(0, np.int32), "size": np.zeros(0, np.float32),
            "color": np.zeros((0, 4), np.float32), "id": np.zeros(0, np.int64),
            "temperature": np.ones(0, np.float32)}


class LiquidPoisson(Poisson3D):
    """The negative 7-point Laplacian over the liquid cells with p = 0 in air: A q = rhs.

    A face between two liquid cells couples them; a face to a solid cell or the domain wall drops out
    (Neumann); a face to an air cell adds one to the diagonal (a p = 0 neighbour). Rows of non-liquid cells
    are zero. `singular` is true when no liquid cell touches air (a full container), where the right-hand
    side is made zero-mean like a closed smoke box.
    """

    def __init__(self, shape, liquid, solid=None, open_faces=(False,) * 6):
        self.shape = tuple(shape)
        self.solid = None if solid is None or not np.any(solid) else np.asarray(solid, bool)
        self.open_axes = (False, False, False)
        self.open_faces = tuple(bool(v) for v in open_faces)
        self.fluid = liquid
        self.cx = (liquid[:-1] & liquid[1:]).astype(np.float64)
        self.cy = (liquid[:, :-1] & liquid[:, 1:]).astype(np.float64)
        self.cz = (liquid[:, :, :-1] & liquid[:, :, 1:]).astype(np.float64)
        self.ends = [None, None, None]
        air = ~liquid if self.solid is None else ~(liquid | self.solid)
        extra = np.zeros(self.shape, np.float64)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            extra[lo] += liquid[lo] & air[hi]
            extra[hi] += liquid[hi] & air[lo]
            low_face, high_face = self.open_faces[axis * 2:axis * 2 + 2]
            if low_face:
                edge = [slice(None)] * 3; edge[axis] = 0
                extra[tuple(edge)] += liquid[tuple(edge)]
            if high_face:
                edge = [slice(None)] * 3; edge[axis] = -1
                extra[tuple(edge)] += liquid[tuple(edge)]
        self.extra = extra
        self.air_neighbours = extra
        self.singular = not extra.any()
        self._diag = None

    def apply(self, q, out):
        super().apply(q, out)
        out += self.extra * q
        return out

    def diagonal(self):
        if self._diag is None:
            self._diag = super().diagonal() + self.extra
        return self._diag


def extrapolate(field, valid, layers):
    """Fill invalid entries next to valid ones with the mean of their valid 6-neighbours, `layers` times.
    Returns (field, valid); the input is not modified."""
    f = np.where(valid, field, 0.0)
    valid = valid.copy()
    for _ in range(layers):
        total = np.zeros_like(f)
        count = np.zeros(f.shape, np.float64)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            total[hi] += f[lo] * valid[lo]
            count[hi] += valid[lo]
            total[lo] += f[hi] * valid[hi]
            count[lo] += valid[hi]
        new = (count > 0) & ~valid
        if not new.any():
            break
        f[new] = total[new] / count[new]
        valid |= new
    return f, valid


class Liquid3D:
    """The FLIP/PIC liquid solver. Holds parameters only; every substep takes and returns a State."""

    # Smoke3D's boundary and solid-face handling applies unchanged (closed domain, moving or still solids).
    _face_constraints = Smoke3D._face_constraints
    _solid_for = Smoke3D._solid_for

    def __init__(self, params=None, pressure_solver=None, cancel=None, sources=None, forces=(), colliders=()):
        self.params = {**DEFAULTS, **(params or {})}
        p = self.params
        self.nx, self.ny, self.nz = int(p["nx"]), int(p["ny"]), int(p["nz"])
        if min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("the grid must be at least 4 cells on every side")
        self.shape = (self.nx, self.ny, self.nz)
        self.substeps = max(1, int(p["substeps"]))
        self.dt = 1.0 / self.substeps
        self.open_faces = tuple(str(p.get(f"boundary_{a}_{side}", "closed")) == "open"
                                for a in "xyz" for side in ("min", "max"))
        self.open_axes = tuple(self.open_faces[2 * i] or self.open_faces[2 * i + 1] for i in range(3))
        self.origin = np.array((p["origin_x"], p["origin_y"], p["origin_z"]), np.float64)
        self.floor_y = float(self.origin[1])
        self.voxel = float(p["voxel_size"])
        self.ppc = max(1, int(p["particles_per_cell"]))
        self.max_per_cell = max(int(p["max_per_cell"]), int(math.ceil(1.5 * self.ppc)))
        self.flip_ratio = float(p["flip_ratio"])
        self.start_frame = int(p["start_frame"])
        self.pressure_solver = pressure_solver or conjugate_gradient
        self.backend = str(p.get("backend", "cpu"))
        self._flip_gpu = None
        self._viscosity_gpu = None
        self.cancel = cancel
        self.sources = [s for s in (sources or ()) if getattr(s, "fluid_type", "liquid") == "liquid"]
        self.forces = list(forces)
        self.colliders = list(colliders)
        self._systems = {}
        self.stats = {"cg_iterations": 0, "particles": 0}

    # -- simcache API -------------------------------------------------------------------------------
    def initial_state(self, seed=0) -> State:
        return State(empty_arrays(), {"next_id": 0, "substep_count": 0, "cg_iterations": 0, "cg_residual": 0.0,
                                      "domain_shape": list(self.shape), "domain_origin": self.origin.tolist()}, copy=False)

    def checkpoint(self, state) -> State:
        return State(state.arrays, state.meta, copy=True)

    def restore(self, state) -> State:
        return State(state.arrays, state.meta, copy=True)

    def _system(self, solid, key):
        return None

    # Force.apply and the turbulence force expect these of the solver they are handed
    def add_cell_force(self, a, field):
        Smoke3D.add_cell_force(self, a, field)

    def buoyancy(self, *args, **kwargs):
        return None

    def step(self, state, frame=0, substep=0, seed=0) -> State:
        """One substep of dt = 1 / substeps frames. Pure: the input state is not modified."""
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        self._sync_domain(state)
        if int(self.params.get("auto_resize", 0)):
            state = self._resize_domain(state, frame, include_sources=True)
            self._sync_domain(state)
        SOLVER_STATS["steps"] += 1
        p, dt = self.params, self.dt
        shape = self.shape
        arrays = state.arrays
        pos = (arrays["position"].astype(np.float64) - self.origin) / self.voxel
        vel = arrays["velocity"].astype(np.float64) / self.voxel
        ids = arrays["id"].copy()
        age = arrays["age"].copy()
        temperature = arrays.get("temperature", np.ones(len(pos), np.float32)).astype(np.float64).copy()
        next_id = int(state.meta.get("next_id", 0))
        rng = np.random.default_rng((int(seed) & 0x7FFFFFFF, int(frame) & 0x7FFFFFFF, int(substep), 1))
        solid, solid_velocity, _unused = self._solid_for(frame, substep)

        pos, vel, ids, age, temperature, next_id = self._emit(pos, vel, ids, age, temperature, next_id, solid, frame, substep, rng)
        pos, vel, ids, age, temperature, next_id = self._maintain(pos, vel, ids, age, temperature, next_id, solid, rng)

        # 3. particle to grid
        stencils = self._stencils(pos)
        gpu_liquid = None
        if self.backend == "gpu":
            if self._flip_gpu is None:
                from .flip_gpu_transfer import GpuFlipTransfers
                self._flip_gpu = GpuFlipTransfers(self.shape)
            (u, v, w), valid_old, gpu_liquid, old_values = self._flip_gpu.to_grid(pos, vel)
            old_grid = dict(zip("uvw", old_values))
        else:
            u, v, w, valid_old, old_grid = self._to_grid(vel, stencils)
        carried = [state.arrays.get("grid_" + name) for name in "uvw"]
        if float(p.get("narrow_band", 0.0)) > 0.0 and all(x is not None for x in carried):
            for name, field, valid, old in zip("uvw", (u, v, w), (valid_old[x] for x in "uvw"), carried):
                previous = np.asarray(old, np.float64)
                old_grid[name] = np.where(valid, old_grid[name], previous)
                np.copyto(field, np.where(valid, field, old))
                valid[...] = True
        # 4. forces
        liquid = self._classify(pos, solid) if gpu_liquid is None else gpu_liquid
        if solid is not None:
            liquid &= ~solid
        prior_liquid = state.arrays.get("liquid_mask")
        if float(p.get("narrow_band", 0.0)) > 0.0 and prior_liquid is not None:
            carried_liquid = self._advect_liquid_mask(np.asarray(prior_liquid, bool), {"u": u, "v": v, "w": w}, dt)
            if solid is not None:
                carried_liquid &= ~solid
            liquid |= carried_liquid
        a = {"u": u, "v": v, "w": w, "density": liquid.astype(np.float64)}
        v += self.params["gravity"] * (-dt)
        for force in self.forces:
            force.apply(self, a, frame, substep, dt)
        surface_tension = float(p.get("surface_tension", 0.0))
        if surface_tension != 0.0:
            self._apply_surface_tension(a, liquid, surface_tension * dt)
        if float(p["viscosity"]) > 0.0:
            coeff = None
            attribute = str(p.get("viscosity_by_attribute", "none"))
            if attribute == "temperature" and len(temperature):
                # A cooler particle is more viscous. The bounded tenfold ramp keeps the
                # artist-facing temperature channel useful without making coefficients singular.
                particle_mu = 1.0 + 9.0 * (1.0 - np.clip(temperature, 0.0, 1.0))
                coeff = self._scalar_to_grid(particle_mu, stencils)
            if self.backend == "gpu":
                from .fluid_gpu_viscosity import GpuViscosity3D
                if self._viscosity_gpu is None:
                    self._viscosity_gpu = GpuViscosity3D()
                for name in "uvw":
                    a[name][...] = self._viscosity_gpu.solve(
                        a[name], float(p["viscosity"]) * dt,
                        None if coeff is None else coeff[name], cancel=self.cancel)
            else:
                self._viscosity(a, float(p["viscosity"]) * dt, coeff)
        # 5 and 6. boundaries, projection
        self._face_constraints(a, solid, solid_velocity)
        for axis, face in enumerate((a["u"], a["v"], a["w"])):
            low_open, high_open = self.open_faces[axis * 2:axis * 2 + 2]
            if not low_open:
                edge = [slice(None)] * 3; edge[axis] = 0
                face[tuple(edge)] = 0.0
            if not high_open:
                edge = [slice(None)] * 3; edge[axis] = -1
                face[tuple(edge)] = 0.0
        system = LiquidPoisson(shape, liquid, solid, self.open_faces)
        iterations, residual = self._project(a, liquid, solid, system)
        # faces we can trust: those touching liquid, and the ones held by solids and walls
        touched = self._touched_faces(liquid, solid)
        # 7. extrapolate the new and the old velocity into the air
        new_f, old_f = {}, {}
        for axis, name in enumerate(("u", "v", "w")):
            if self.backend == "gpu":
                from .flip_gpu_extrapolate import extrapolate as gpu_extrapolate
                new_f[name], _ = gpu_extrapolate(a[name], touched[name], EXTRAPOLATE_LAYERS)
                old_f[name], _ = gpu_extrapolate(old_grid[name], valid_old[name] | touched[name], EXTRAPOLATE_LAYERS)
            else:
                new_f[name], _ = extrapolate(a[name], touched[name], EXTRAPOLATE_LAYERS)
                old_f[name], _ = extrapolate(old_grid[name], valid_old[name] | touched[name], EXTRAPOLATE_LAYERS)
        # 8. grid to particle
        if self.backend == "gpu":
            prior_pos = pos.copy()
            pos, vel = self._flip_gpu.from_grid(pos, vel, new_f, old_f, self.flip_ratio, dt, self.open_faces)
            # Preserve the reference path's collider and boundary behavior after GPU RK2 advection.
            if solid is not None and len(pos):
                blocked = solid.reshape(-1)[self._cell(pos)]
                pos[blocked] = prior_pos[blocked]
            for axis in range(3):
                low_open, high_open = self.open_faces[axis * 2:axis * 2 + 2]
                if not low_open:
                    pos[:, axis] = np.maximum(pos[:, axis], 1e-3)
                if not high_open:
                    pos[:, axis] = np.minimum(pos[:, axis], self.shape[axis] - 1e-3)
        else:
            vel = self._from_grid(vel, new_f, old_f, stencils)
            # 9. advect
            pos = self._advect(pos, vel, new_f, solid, dt)
        escaped = np.zeros(len(pos), bool)
        for axis in range(3):
            low, high = self.open_faces[axis * 2:axis * 2 + 2]
            if low:
                escaped |= pos[:, axis] < 0.0
            if high:
                escaped |= pos[:, axis] >= self.shape[axis]
        escaped_n = int(np.count_nonzero(escaped))
        if escaped_n:
            keep = ~escaped
            pos, vel, ids, age, temperature = (x[keep] for x in (pos, vel, ids, age, temperature))
        age = age + 1
        temperature *= 0.995 ** dt
        band = max(0.0, float(p.get("narrow_band", 0.0)))
        if band > 0.0:
            pos, vel, ids, age, temperature, next_id, liquid = self._narrow(
                pos, vel, ids, age, temperature, next_id, band, liquid)
        else:
            liquid = None
        self.stats.update(cg_iterations=int(iterations), particles=int(len(pos)))
        grids = {"grid_u": a["u"].astype(np.float32), "grid_v": a["v"].astype(np.float32),
                 "grid_w": a["w"].astype(np.float32)} if band > 0.0 else None
        meta = dict(state.meta)
        meta["escaped_mass"] = float(meta.get("escaped_mass", 0.0)) + escaped_n / self.ppc * self.voxel ** 3
        result = self._pack(pos, vel, ids, age, temperature, meta, next_id, iterations, residual, liquid, grids)
        return self._resize_domain(result, frame=frame) if int(p.get("auto_resize", 0)) else result

    def _sync_domain(self, state):
        shape = state.meta.get("domain_shape")
        if shape is None:
            return
        shape = tuple(int(v) for v in shape)
        changed = shape != self.shape
        if changed:
            self.shape = shape
            self.nx, self.ny, self.nz = shape
            self.params.update(nx=self.nx, ny=self.ny, nz=self.nz)
        origin = state.meta.get("domain_origin")
        if origin is not None and tuple(origin) != tuple(self.origin):
            self.origin = np.asarray(origin, np.float64)
            for axis, value in zip("xyz", self.origin):
                self.params[f"origin_{axis}"] = float(value)
            changed = True
        if changed:
            self._systems.clear()

    def _fit_box(self, bounds, frame=None, include_sources=False):
        """The tile-aligned (start, stop, new_shape) cell box that fits the liquid, or None when the current box
        already is it. `bounds` is the particles' cell-space (lo, hi) in the current box, or None without particles;
        the sources of `frame` and the colliders near the liquid widen it, and the lower world-space floor stays put."""
        padding = max(0, int(self.params.get("padding", 8)))
        cap = max(8, int(self.params.get("max_size", 256)) // 8 * 8)
        if bounds is not None:
            lo, hi = np.array(bounds[0], np.float64), np.array(bounds[1], np.float64)
            has_bounds = True
        else:
            center = np.asarray(self.shape, np.float64) * 0.5
            lo, hi = center - 4.0, center + 4.0
            has_bounds = False
        if include_sources and frame is not None:
            for source in self.sources:
                if int(frame) < source.start_frame or int(frame) > source.end_frame:
                    continue
                knobs = source._knobs(frame)
                if source.emit_from in ("surface", "volume") and source.track is not None:
                    tri = np.asarray(source.track.at(frame), np.float64).reshape(-1, 3)
                    if not len(tri):
                        continue
                    source_lo = (tri.min(axis=0) - self.origin) / self.voxel
                    source_hi = (tri.max(axis=0) - self.origin) / self.voxel + 1
                else:
                    center = (np.asarray(knobs["center"], np.float64) - self.origin) / self.voxel
                    radius = max(0.0, float(knobs["radius"]) / self.voxel)
                    source_lo, source_hi = center - radius - 1, center + radius + 2
                lo, hi = np.minimum(lo, source_lo), np.maximum(hi, source_hi)
                has_bounds = True
        if frame is not None:
            for collider_lo, collider_hi in Smoke3D._collider_cell_bounds(self, frame, (lo, hi) if has_bounds else None,
                                                                          padding):
                lo, hi = np.minimum(lo, collider_lo), np.maximum(hi, collider_hi)
                has_bounds = True
        if has_bounds:
            lo -= padding
            hi += padding
        start = np.floor(lo / 8.0).astype(int) * 8
        stop = np.maximum(8, np.ceil(hi / 8.0).astype(int) * 8)
        for axis in range(3):
            if stop[axis] - start[axis] > cap:
                middle = 0.5 * (lo[axis] + hi[axis])
                start[axis] = int(np.floor((middle - cap * 0.5) / 8.0)) * 8
                stop[axis] = start[axis] + cap
        # The liquid's closed lower wall is its floor. Let the free surface grow upward,
        # while keeping that world-space floor anchored as the rest of the box adapts.
        floor_start = int(math.ceil((self.floor_y - self.origin[1]) / self.voxel / 8.0) * 8)
        if start[1] < floor_start:
            start[1] = floor_start
            stop[1] = max(stop[1], start[1] + 8)
            if stop[1] - start[1] > cap:
                stop[1] = start[1] + cap
        new_shape = tuple(int(x) for x in stop - start)
        if new_shape == self.shape and not np.any(start):
            return None
        return start, stop, new_shape

    def _resize_domain(self, state, frame=None, include_sources=False):
        """Tile-align the liquid free-surface/particle bounds, retaining world-space particles."""
        positions = np.asarray(state.arrays.get("position", ()), np.float64).reshape(-1, 3)
        bounds = None
        if len(positions):
            cell = (positions - self.origin) / self.voxel
            bounds = (cell.min(axis=0), cell.max(axis=0) + 1.0)
        box = self._fit_box(bounds, frame, include_sources)
        if box is None:
            return state
        start, stop, new_shape = box
        spatial_axes = {"liquid_mask": None, "grid_u": 0, "grid_v": 1, "grid_w": 2}
        arrays = dict(state.arrays)
        for name, face_axis in spatial_axes.items():
            source = state.arrays.get(name)
            if source is None:
                continue
            target_shape = list(new_shape)
            if face_axis is not None:
                target_shape[face_axis] += 1
            target = np.zeros(tuple(target_shape), dtype=source.dtype)
            src_lo, dst_lo = np.maximum(0, start), np.maximum(0, -start)
            lengths = np.minimum(np.asarray(source.shape[:3]) - src_lo,
                                 np.asarray(target.shape[:3]) - dst_lo)
            lengths = np.maximum(0, lengths)
            src = tuple(slice(int(src_lo[a]), int(src_lo[a] + lengths[a])) for a in range(3))
            dst = tuple(slice(int(dst_lo[a]), int(dst_lo[a] + lengths[a])) for a in range(3))
            target[dst] = source[src]
            arrays[name] = target
        meta = dict(state.meta)
        new_origin = self.origin + start * self.voxel
        meta.update(domain_shape=list(new_shape), domain_origin=new_origin.tolist())
        return State(arrays, meta, copy=False)

    # -- pieces -------------------------------------------------------------------------------------
    def _pack(self, pos, vel, ids, age, temperature, meta, next_id, iterations, residual, liquid=None, grids=None):
        n = len(pos)
        spacing = 1.0 / self.ppc ** (1.0 / 3.0)
        size = np.full(n, spacing * self.voxel, np.float32)
        color = np.tile(np.asarray(LIQUID_COLOR, np.float32), (n, 1))
        out = {"position": (pos * self.voxel + self.origin).astype(np.float32),
               "velocity": (vel * self.voxel).astype(np.float32), "age": age.astype(np.int32),
               "life": np.full(n, LIFE, np.int32), "size": size, "color": color, "id": ids.astype(np.int64),
               "temperature": temperature.astype(np.float32)}
        if liquid is not None:
            out["liquid_mask"] = liquid
        if grids:
            out.update(grids)
        m = dict(meta)
        m.update(next_id=int(next_id), substep_count=int(m.get("substep_count", 0)) + 1,
                 cg_iterations=int(iterations), cg_residual=float(residual))
        return State(out, m, copy=False)

    def _cell(self, pos):
        idx = np.floor(pos).astype(np.intp)
        for a, n in enumerate(self.shape):
            np.clip(idx[:, a], 0, n - 1, out=idx[:, a])
        return np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), self.shape)

    def _classify(self, pos, solid):
        count = np.bincount(self._cell(pos), minlength=int(np.prod(self.shape))).reshape(self.shape) if len(pos) \
            else np.zeros(self.shape, np.int64)
        liquid = count > 0
        if solid is not None:
            liquid &= ~solid
        return liquid

    def _apply_surface_tension(self, arrays, liquid, amount):
        """Continuum-surface-force approximation, restricted to the liquid/air interface.

        The smoothed occupancy gradient supplies the outward normal; its divergence is
        curvature.  The force points inward and is interpolated to the MAC faces.
        This branch is never entered for the default zero knob.
        """
        from .fluid3d import _sl
        phi = liquid.astype(np.float64)
        # A compact 3x3x3 binomial smooth suppresses voxel-scale curvature noise.
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            p = np.pad(phi, [(1, 1)] * 3, mode="edge")
            center = [slice(1, -1)] * 3
            minus, plus = list(center), list(center)
            minus[axis] = slice(0, -2); plus[axis] = slice(2, None)
            phi = (p[tuple(minus)] + 2.0 * p[tuple(center)] + p[tuple(plus)]) * 0.25
        grad = np.stack(np.gradient(phi), axis=-1)
        mag = np.linalg.norm(grad, axis=-1)
        normal = grad / np.maximum(mag[..., None], 1e-8)
        curvature = sum(np.gradient(normal[..., axis], axis=axis) for axis in range(3))
        band = np.zeros(liquid.shape, bool)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            band[lo] |= liquid[lo] != liquid[hi]
            band[hi] |= liquid[lo] != liquid[hi]
        force = -float(amount) * curvature[..., None] * normal * band[..., None]
        for axis, name in enumerate("uvw"):
            face = arrays[name]
            left = [slice(None)] * 3; right = [slice(None)] * 3
            left[axis] = slice(None, -1); right[axis] = slice(1, None)
            middle = [slice(None)] * 3; middle[axis] = slice(1, -1)
            face[tuple(middle)] += 0.5 * (force[tuple(left) + (axis,)] + force[tuple(right) + (axis,)])

    def _seed_cells(self, flat, count_per_cell, rng):
        """Jittered particles, stratified when `count_per_cell` is a cube, `count_per_cell` in each cell of `flat`."""
        ppc = int(count_per_cell)
        cells = np.stack(np.unravel_index(flat, self.shape), axis=1).astype(np.float64)
        cells = np.repeat(cells, ppc, axis=0)
        n = len(cells)
        side = int(round(ppc ** (1.0 / 3.0)))
        offsets = rng.random((n, 3))
        if side ** 3 == ppc:
            sub = np.stack(np.unravel_index(np.arange(ppc), (side,) * 3), axis=1).astype(np.float64)
            sub = np.tile(sub, (len(flat), 1))
            return cells + (sub + offsets) / side
        return cells + offsets

    def _emit(self, pos, vel, ids, age, temperature, next_id, solid, frame, substep, rng):
        added = []
        count = None
        for source in self.sources:
            knobs = source._knobs(frame)
            velocity = np.asarray(knobs["velocity"], np.float64) / self.voxel
            pouring = bool(np.any(velocity))
            first = max(int(source.start_frame), self.start_frame)
            if frame < first or frame > int(source.end_frame):
                continue
            if not pouring and not (frame == first and substep == 0):
                continue
            flat, weight, _motion = source.footprint(self, frame)
            if not len(flat):
                continue
            flat = flat[weight > 0]
            if solid is not None:
                flat = flat[~solid.reshape(-1)[flat]]
            if pouring:
                if count is None:
                    count = np.bincount(self._cell(pos), minlength=int(np.prod(self.shape))) if len(pos) \
                        else np.zeros(int(np.prod(self.shape)), np.int64)
                flat = flat[count[flat] < max(1, self.ppc // 2)]
            if not len(flat):
                continue
            fresh = self._seed_cells(flat, self.ppc, rng)
            added.append((fresh, np.tile(velocity, (len(fresh), 1)), np.full(len(fresh), float(knobs.get("temperature", 1.0)))))
        if not added:
            return pos, vel, ids, age, temperature, next_id
        new_pos = np.concatenate([a for a, _, _ in added])
        new_vel = np.concatenate([b for _, b, _ in added])
        new_temperature = np.concatenate([c for _, _, c in added])
        n = len(new_pos)
        new_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        return (np.concatenate((pos, new_pos)), np.concatenate((vel, new_vel)), np.concatenate((ids, new_ids)),
                np.concatenate((age, np.zeros(n, age.dtype))), np.concatenate((temperature, new_temperature)), next_id + n)

    def _maintain(self, pos, vel, ids, age, temperature, next_id, solid, rng):
        if not len(pos):
            return pos, vel, ids, age, temperature, next_id
        cells = self._cell(pos)
        total = int(np.prod(self.shape))
        # delete over the maximum: within a cell keep the lowest ids
        order = np.lexsort((ids, cells))
        sorted_cells = cells[order]
        starts = np.r_[0, np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1]) + 1]
        group_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
        rank = np.arange(len(order)) - group_start
        drop = order[rank >= self.max_per_cell]
        if len(drop):
            keep = np.ones(len(pos), bool)
            keep[drop] = False
            pos, vel, ids, age, temperature, cells = pos[keep], vel[keep], ids[keep], age[keep], temperature[keep], cells[keep]
        count = np.bincount(cells, minlength=total).reshape(self.shape)
        # interior gaps: fewer than the minimum, every neighbour holds particles (walls and solids count as full)
        full = count > 0
        if solid is not None:
            full = full | solid
        neighbours = np.ones(self.shape, bool)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            neighbours[hi] &= full[lo]
            neighbours[lo] &= full[hi]
        gap = (count < MIN_PER_CELL) & neighbours & (count > 0)
        if solid is not None:
            gap &= ~solid
        if not gap.any():
            return pos, vel, ids, age, temperature, next_id
        # mean neighbour velocity per cell
        sums = np.zeros(self.shape + (3,))
        for c in range(3):
            sums[..., c] = np.bincount(cells, weights=vel[:, c], minlength=total).reshape(self.shape)
        temperature_sum = np.bincount(cells, weights=temperature, minlength=total).reshape(self.shape)
        nsum = np.zeros_like(sums)
        ncnt = np.zeros(self.shape)
        temperature_neighbours = np.zeros(self.shape)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            nsum[hi] += sums[lo]
            ncnt[hi] += count[lo]
            nsum[lo] += sums[hi]
            ncnt[lo] += count[hi]
            temperature_neighbours[hi] += temperature_sum[lo]
            temperature_neighbours[lo] += temperature_sum[hi]
        mean = nsum / np.maximum(ncnt, 1)[..., None]
        mean_temperature = temperature_neighbours / np.maximum(ncnt, 1)
        flat = np.flatnonzero(gap.reshape(-1))
        need = (MIN_PER_CELL - count.reshape(-1)[flat]).astype(np.intp)
        flat_rep = np.repeat(flat, need)
        base = np.stack(np.unravel_index(flat_rep, self.shape), axis=1).astype(np.float64)
        fresh = base + rng.random(base.shape)
        fresh_vel = mean.reshape(-1, 3)[flat_rep]
        n = len(fresh)
        fresh_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        return (np.concatenate((pos, fresh)), np.concatenate((vel, fresh_vel)), np.concatenate((ids, fresh_ids)),
                np.concatenate((age, np.zeros(n, age.dtype))),
                np.concatenate((temperature, mean_temperature.reshape(-1)[flat_rep])), next_id + n)

    def _stencils(self, pos):
        nx, ny, nz = self.shape
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        return (Stencil((nx + 1, ny, nz), x, y - .5, z - .5), Stencil((nx, ny + 1, nz), x - .5, y, z - .5),
                Stencil((nx, ny, nz + 1), x - .5, y - .5, z))

    @staticmethod
    def _corner_weights(st):
        tx, ty, tz = st.tx, st.ty, st.tz
        sx, sy, sz = 1.0 - tx, 1.0 - ty, 1.0 - tz
        b, sy_, sz_ = st.base, st.sy, st.sz
        return ((b, sx * sy * sz), (b + 1, sx * sy * tz), (b + sz_, sx * ty * sz), (b + sz_ + 1, sx * ty * tz),
                (b + sy_, tx * sy * sz), (b + sy_ + 1, tx * sy * tz), (b + sy_ + sz_, tx * ty * sz),
                (b + sy_ + sz_ + 1, tx * ty * tz))

    def _to_grid(self, vel, stencils):
        nx, ny, nz = self.shape
        shapes = ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))
        out, valid = [], {}
        old = {}
        for axis, (st, shape) in enumerate(zip(stencils, shapes)):
            size = int(np.prod(shape))
            num = np.zeros(size)
            den = np.zeros(size)
            for idx, weight in self._corner_weights(st):
                num += np.bincount(idx, weights=weight * vel[:, axis], minlength=size)
                den += np.bincount(idx, weights=weight, minlength=size)
            field = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0).reshape(shape)
            name = "uvw"[axis]
            valid[name] = (den > 1e-9).reshape(shape)
            old[name] = field.copy()
            out.append(field)
        return out[0], out[1], out[2], valid, old

    def _scalar_to_grid(self, values, stencils):
        """Transfer a particle scalar to the three staggered face grids with FLIP's weights."""
        shapes = ((self.nx + 1, self.ny, self.nz), (self.nx, self.ny + 1, self.nz),
                  (self.nx, self.ny, self.nz + 1))
        out = []
        for st, shape in zip(stencils, shapes):
            size = int(np.prod(shape))
            num = np.zeros(size)
            den = np.zeros(size)
            for idx, weight in self._corner_weights(st):
                num += np.bincount(idx, weights=weight * values, minlength=size)
                den += np.bincount(idx, weights=weight, minlength=size)
            out.append(np.divide(num, den, out=np.ones_like(num), where=den > 1e-9).reshape(shape))
        return dict(zip("uvw", out))

    def _advect_liquid_mask(self, mask, fields, dt):
        """Carry grid-only liquid cells with the MAC velocity and close rasterization gaps."""
        if not mask.any():
            return mask.copy()
        points = np.argwhere(mask).astype(np.float64) + 0.5
        velocity = self._sample(fields, self._stencils(points))
        moved = np.floor(np.clip(points + velocity * dt, 0.0, np.asarray(self.shape) - 1e-4)).astype(np.intp)
        advected = np.zeros(self.shape, bool)
        advected[tuple(moved.T)] = True
        shell = np.zeros_like(advected)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            shell[lo] |= advected[lo] & ~advected[hi]
            shell[hi] |= advected[hi] & ~advected[lo]
        return fill_interior(shell) | advected

    def _narrow(self, pos, vel, ids, age, temperature, next_id, width, mask_hint=None):
        """Keep particles near the closed interface; the interior and its velocities remain grid state."""
        cells = self._cell(pos) if len(pos) else np.zeros(0, np.intp)
        occupied = np.zeros(self.shape, bool)
        if len(cells):
            occupied.reshape(-1)[cells] = True
        # A cell is on the interface when at least one of its six neighbours is empty.
        surface = np.zeros_like(occupied)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            surface[lo] |= occupied[lo] & ~occupied[hi]
            surface[hi] |= occupied[hi] & ~occupied[lo]
        liquid = (np.asarray(mask_hint, bool).copy() if mask_hint is not None else fill_interior(surface)) | occupied
        # Reclose any rasterization gaps, then measure distance from this true interface. This avoids
        # mistaking the artificial inner edge of the particle band for another liquid surface.
        shell = np.zeros_like(liquid)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            shell[lo] |= liquid[lo] & ~liquid[hi]
            shell[hi] |= liquid[hi] & ~liquid[lo]
        liquid = fill_interior(shell) | liquid
        surface = shell
        distance = np.full(self.shape, np.inf, np.float32)
        distance[surface] = 0.0
        # Six-neighbour distance propagation suffices for a conservative cell-width band.
        for step in range(1, min(max(self.shape), int(math.ceil(width)) + 1) + 1):
            grown = distance.copy()
            for axis in range(3):
                lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
                grown[lo] = np.minimum(grown[lo], distance[hi] + 1.0)
                grown[hi] = np.minimum(grown[hi], distance[lo] + 1.0)
            grown[~occupied] = np.inf
            distance = grown
        keep = distance.reshape(-1)[cells] <= width
        return pos[keep], vel[keep], ids[keep], age[keep], temperature[keep], next_id, liquid

    def _touched_faces(self, liquid, solid):
        """Per axis, the bool mask of faces whose velocity is defined after the projection: faces next to a
        liquid cell and faces held by a solid or the domain wall."""
        held = np.zeros(self.shape, bool) if solid is None else solid
        out = {}
        nx, ny, nz = self.shape
        for axis, name in enumerate("uvw"):
            shape = list(self.shape)
            shape[axis] += 1
            mask = np.zeros(shape, bool)
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            inner = _sl(axis, slice(1, -1))
            mask[inner] = liquid[lo] | liquid[hi] | held[lo] | held[hi]
            mask[_sl(axis, 0)] = True
            mask[_sl(axis, -1)] = True
            mask[lo] |= liquid
            mask[hi] |= liquid
            out[name] = mask
        return out

    def _viscosity(self, a, k, coefficient=None):
        """Backward-Euler velocity diffusion on each staggered face grid.

        Solve (I + k L) u_new = u_old with deterministic Jacobi-preconditioned CG.
        Unlike the former clipped explicit step, this remains stable for arbitrarily large k.
        The wall/collider constraints are imposed again by the caller after this solve.
        """
        k = max(0.0, float(k))
        if k == 0.0:
            return
        for name in ("u", "v", "w"):
            rhs = np.asarray(a[name], np.float64)
            mu = np.ones(rhs.shape, np.float64) if coefficient is None else np.asarray(coefficient[name], np.float64)
            edges = []
            degree = np.zeros(rhs.shape, np.float64)
            for axis in range(3):
                lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
                edge = 0.5 * (mu[lo] + mu[hi])
                edges.append((lo, hi, edge))
                degree[lo] += edge
                degree[hi] += edge

            def apply(x):
                out = (1.0 + k * degree) * x
                for lo, hi, edge in edges:
                    out[lo] -= k * edge * x[hi]
                    out[hi] -= k * edge * x[lo]
                return out

            x = rhs.copy()
            residual = rhs - apply(x)
            inv_diag = 1.0 / (1.0 + k * degree)
            z = residual * inv_diag
            direction = z.copy()
            rz = float(np.vdot(residual, z))
            target = max(1e-12, float(np.linalg.norm(rhs)) * 1e-8)
            for _ in range(1000):
                if float(np.linalg.norm(residual)) <= target:
                    break
                ad = apply(direction)
                denom = float(np.vdot(direction, ad))
                if denom <= 0.0 or not math.isfinite(denom):
                    break
                alpha = rz / denom
                x += alpha * direction
                residual -= alpha * ad
                z = residual * inv_diag
                next_rz = float(np.vdot(residual, z))
                direction = z + (next_rz / rz) * direction if rz > 0.0 else z.copy()
                rz = next_rz
            a[name][...] = x

    def _project(self, a, liquid, solid, system):
        u, v, w = a["u"], a["v"], a["w"]
        div = (u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])
        rhs = np.where(liquid, -div, 0.0)
        if system.singular and liquid.any():
            rhs[liquid] -= rhs[liquid].mean()
        x0 = np.zeros(self.shape)
        if not liquid.any():
            return 0, 0.0
        q, iterations, residual = self.pressure_solver(rhs, x0, float(self.params["tolerance"]),
                                                       int(self.params["max_iterations"]), self.cancel, system)
        if system.singular:
            q = np.where(liquid, q - q[liquid].mean(), 0.0)
        held = np.zeros(self.shape, bool) if solid is None else solid
        for axis, face in enumerate((u, v, w)):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            inner = _sl(axis, slice(1, -1))
            grad = q[hi] - q[lo]
            ok = (liquid[lo] | liquid[hi]) & ~held[lo] & ~held[hi]
            block = face[inner]
            block -= np.where(ok, grad, 0.0)
            face[inner] = block
            low_open, high_open = system.open_faces[axis * 2:axis * 2 + 2]
            if low_open:
                edge = [slice(None)] * 3; edge[axis] = 0
                face[tuple(edge)] += q[tuple(edge)]
            if high_open:
                edge = [slice(None)] * 3; edge[axis] = -1
                face[tuple(edge)] -= q[tuple(edge)]
        return iterations, residual

    def _sample(self, fields, stencils, order=(0, 1, 2)):
        return np.stack([stencils[i].sample(fields[name]) for i, name in zip(order, "uvw")], axis=1)

    def _from_grid(self, vel, new_f, old_f, stencils):
        pic = self._sample(new_f, stencils)
        if self.flip_ratio <= 0.0:
            return pic
        delta = pic - self._sample(old_f, stencils)
        flip = vel + delta
        return self.flip_ratio * flip + (1.0 - self.flip_ratio) * pic

    def _advect(self, pos, vel, fields, solid, dt):
        if not len(pos):
            return pos
        vmax = float(np.abs(vel).max())
        k = int(min(MAX_ADVECT_SUBSTEPS, max(1, math.ceil(vmax * dt / 0.9))))
        h = dt / k
        upper = np.array(self.shape, np.float64) - 1e-3
        for _ in range(k):
            v1 = self._sample(fields, self._stencils(pos))
            mid = np.clip(pos + 0.5 * h * v1, 1e-3, upper)
            v2 = self._sample(fields, self._stencils(mid))
            raw = pos + h * v2
            new = np.clip(raw, 1e-3, upper)
            for axis in range(3):
                low_open, high_open = self.open_faces[axis * 2:axis * 2 + 2]
                if low_open:
                    new[:, axis] = np.where(raw[:, axis] < 0.0, raw[:, axis], new[:, axis])
                if high_open:
                    new[:, axis] = np.where(raw[:, axis] >= self.shape[axis], raw[:, axis], new[:, axis])
            if solid is not None:
                blocked = solid.reshape(-1)[self._cell(new)]
                new[blocked] = pos[blocked]
            pos = new
        return pos


def liquid_volume(state, stream=None, shape=None, voxel=None):
    """Liquid volume in world units cubed: cells' worth of particles times the cell volume."""
    if "liquid_mask" in state.arrays:
        scale = stream.voxel if stream is not None else float(voxel or 1.0)
        return float(np.count_nonzero(state.arrays["liquid_mask"]) * scale ** 3)
    n = len(state.arrays["position"])
    return n / stream.ppc * stream.voxel ** 3


# --- the nodes ------------------------------------------------------------------------------------
# FluidLiquidSolver3D takes the chain of FluidSource3D (fluid_type liquid), FluidForce3D and FluidCollide3D nodes that
# FluidSolver3D takes, and outputs a ParticleInstance per frame with the signed-distance Volume on `.surface`. Its
# stream quacks like a particles.ParticleStream (`emitter`, `run`, `start_frame`, `substeps`, `seed`), so
# ParticleCache3D and ParticleRender3D take a liquid unchanged. FluidSurface3D meshes the particles and
# FluidFoam3D tags the splash.

MAX_LIQUID_CELLS = 4_194_304        # 160 cubed: the CPU reference is a bake-and-scrub tool
GPU_AUTO_CELLS = 1_000_000
RADIUS_SPACINGS = 1.0               # auto particle radius, in particle spacings
SUPPORT_SPACINGS = 3.0              # Zhu-Bridson kernel reach, in particle spacings


class LiquidStream:
    """One deterministic liquid run: enough to solve any frame of it. Built by `build_stream`."""

    def __init__(self, chain, params, run, fps):
        from .fluid3d import resolution
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
        self.ppc = int(params["particles_per_cell"])
        self.spacing = self.voxel / self.ppc ** (1.0 / 3.0)
        self.backend = "cpu"
        self.fallback_reason = None
        self._solver = None

    @property
    def emitter(self):
        return self.solver()

    def solver(self, cancel=None):
        if self._solver is None:
            p = self.params
            nx, ny, nz = self.shape
            for source in self.chain.sources:
                source.seed = self.seed
            params = {"nx": nx, "ny": ny, "nz": nz, "substeps": self.substeps, "flip_ratio": p["flip_ratio"],
                      "auto_resize": p.get("auto_resize", 0), "padding": p.get("padding", 8),
                      "max_size": p.get("max_size", 256),
                      "particles_per_cell": self.ppc,
                      "gravity": p["liquid_gravity"] / (self.fps * self.fps) / self.voxel,
                      "surface_tension": p.get("surface_tension", 0.0) / (self.fps * self.fps) / self.voxel,
                      "viscosity": p["viscosity"], "viscosity_by_attribute": p.get("viscosity_by_attribute", "none"),
                      "narrow_band": p.get("narrow_band", 0.0), "tolerance": p["tolerance"], "max_iterations": p["max_iterations"],
                      "origin_x": self.origin[0], "origin_y": self.origin[1], "origin_z": self.origin[2],
                      "voxel_size": self.voxel, "start_frame": self.start_frame, "seed": self.seed,
                      "backend": self.backend}
            hook = None
            if self.backend == "resident":
                from .flip_gpu_resident import create_solver
                self._solver = create_solver(params, sources=self.chain.sources, forces=self.chain.forces,
                                             colliders=self.chain.colliders)
                self._solver.cancel = cancel
                return self._solver
            if self.backend == "gpu":
                from .fluid3d import _gpu_solver
                hook = _gpu_solver().solve
            self._solver = Liquid3D(params, pressure_solver=hook, sources=self.chain.sources,
                                    forces=self.chain.forces, colliders=self.chain.colliders)
        self._solver.cancel = cancel
        return self._solver

    @property
    def radius(self):
        return RADIUS_SPACINGS * self.spacing


def resolve_backend(params, cells):
    """`pressure` resolved to "cpu", "gpu" (the wgpu SOR hook of step B, which reads the liquid system's own diagonal)
    or "resident" (the whole substep on the card, nodebased/flip_gpu_resident.py; opt-in, never picked by auto). The
    sparse-tile smoke solver is refused for liquids."""
    from . import fluid_gpu3d, fluid_gpu_solver, flip_gpu_resident
    choice = params["pressure"]
    if choice == "resident_sparse":
        raise ValueError("FluidLiquidSolver3D: pressure resident_sparse is for the smoke solver; use cpu, gpu, resident or auto")
    if choice == "cpu":
        return "cpu"
    if choice == "resident":
        if not fluid_gpu_solver.available():
            raise ValueError("FluidLiquidSolver3D: pressure is resident but no wgpu adapter can be opened here")
        need = flip_gpu_resident.estimate_bytes((round(cells ** (1 / 3)),) * 3)
        if need > flip_gpu_resident.GPU_MEMORY_BUDGET:
            raise ValueError(f"FluidLiquidSolver3D: pressure is resident but the grid needs about {need / 2 ** 30:.1f} GiB "
                             f"on the card; the budget is {flip_gpu_resident.GPU_MEMORY_BUDGET / 2 ** 30:.1f} GiB")
        return "resident"
    if choice == "gpu":
        if not fluid_gpu3d.available():
            raise ValueError("FluidLiquidSolver3D: pressure is gpu but no wgpu adapter can be opened here")
        return "gpu"
    return "gpu" if fluid_gpu3d.available() else "cpu"


def build_stream(doc, key, node, chain):
    """The LiquidStream of one FluidLiquidSolver3D node, from the chain wired into it."""
    from . import simcache
    from .fluid3d import FluidChain, resolution
    params = node["params"]
    shape = resolution(params)
    cells = shape[0] * shape[1] * shape[2]
    if cells > MAX_LIQUID_CELLS:
        raise ValueError(f"FluidLiquidSolver3D: {shape[0]} x {shape[1]} x {shape[2]} is {cells:,} cells; the CPU "
                         f"reference solver stops at {MAX_LIQUID_CELLS:,} (raise division_size or shrink the bounds)")
    backend = resolve_backend(params, cells)
    base = chain if chain is not None else FluidChain()
    fallback = None
    if backend == "resident":
        from .flip_gpu_resident import unsupported_reason
        fallback = unsupported_reason(params, base.forces)
        if fallback:
            backend = "gpu"
    fps = float(doc.get("time", {}).get("fps", 24.0))
    identity = {"kind": "FluidLiquidSolver3D", "params": params, "backend": backend, "fps": fps,
                "format": 2 if int(params.get("auto_resize", 0)) else 1}
    stream = LiquidStream(base, params, simcache.run_key(base.run, identity), fps)
    stream.backend = backend
    stream.fallback_reason = fallback
    return stream


def solve_frame(stream, frame, cache, cancel=None):
    """The solved `simcache.State` at `frame`, from `cache` where possible."""
    from . import simcache
    solver = stream.solver(cancel)
    return simcache.solve_to_frame(cache, stream.run, int(frame), stream.start_frame, stream.substeps, stream.seed,
                                   solver.initial_state, solver.step, cancel)


def signed_distance(stream, positions, smoothing=0, resolution=1, radius=None):
    """(phi float32, voxel) of the particle surface on the stream's grid subdivided `resolution` times."""
    from .liquid_surface import level_set
    radius = stream.radius if not radius else float(radius)
    support = max(SUPPORT_SPACINGS * stream.spacing, 2.2 * radius)
    res = max(1, int(resolution))
    fine = tuple(n * res for n in stream.shape)
    if stream.backend == "resident":
        from . import flip_gpu_levelset
        if flip_gpu_levelset.fits(fine):
            return flip_gpu_levelset.level_set(positions, stream.origin, stream.voxel / res, fine, radius, support,
                                               smoothing), stream.voxel / res
    return level_set(positions, stream.origin, stream.voxel / res, fine, radius, support, smoothing), stream.voxel / res


def instance_from_state(state, stream, frame):
    """The frame's ParticleInstance, with the signed-distance Volume on `.surface` when `liquid_sdf` is on."""
    from dataclasses import replace
    from . import particles
    from .scene3d import Volume
    frame_stream = copy.copy(stream)
    frame_stream.shape = tuple(state.meta.get("domain_shape", stream.shape))
    frame_stream.origin = tuple(state.meta.get("domain_origin", stream.origin))
    inst = particles.instance_from_state(state, frame_stream, frame)
    if not int(stream.params.get("liquid_sdf", 1)):
        return inst
    phi, voxel = signed_distance(frame_stream, inst.positions)
    return replace(inst, surface=Volume(phi, voxel_size=voxel, origin=frame_stream.origin, frame=int(frame)))


def apply_rigid_feedback(stream, frame, cache, bodies, solver_key, *, gravity=9.81, strength=0.04):
    """Persist a rigid body's liquid reaction at ``frame`` and invalidate dependent FLIP frames.

    RigidSolver3D runs downstream of the liquid graph node. Its contact impulse therefore has to
    replace the already-cached checkpoint at the current frame; later frames are then replayed
    from that adjusted checkpoint. The original velocity is retained in the checkpoint so asking
    for the same frame again is idempotent, and changing the rigid solver's result replaces its
    old contribution instead of accumulating it.
    """
    from . import simcache
    state = solve_frame(stream, int(frame), cache)
    positions = np.asarray(state.arrays.get("position", ()), dtype=np.float64)
    if not len(positions):
        return state
    feedback = np.zeros_like(positions, dtype=np.float64)
    for body in bodies:
        if not body.dynamic:
            continue
        low, high = body.position - body.half_extent, body.position + body.half_extent
        mask = np.all((positions >= low) & (positions <= high), axis=1)
        if np.any(mask):
            feedback[mask, 1] -= abs(float(gravity)) * float(strength)
    if not np.any(feedback):
        return state

    impulse_hash = hashlib.sha256(feedback.tobytes()).hexdigest()
    previous = state.meta.get("rigid_feedback", {})
    if previous.get(str(solver_key)) == impulse_hash:
        return state
    arrays = dict(state.arrays)
    original = arrays.get("_rigid_feedback_base_velocity")
    if original is None:
        original = np.asarray(arrays["velocity"]).copy()
    else:
        original = np.asarray(original).copy()
    arrays["_rigid_feedback_base_velocity"] = original
    arrays["velocity"] = (original.astype(np.float64) + feedback).astype(original.dtype)
    metadata = dict(state.meta)
    feedback_map = dict(previous)
    feedback_map[str(solver_key)] = impulse_hash
    metadata["rigid_feedback"] = feedback_map
    updated = simcache.State(arrays, metadata, copy=False)
    cache.put(stream.run, int(frame), updated)
    cache.invalidate_from(stream.run, int(frame) + 1)
    return updated


def empty_instance():
    """A ParticleInstance with no particles and no stream: what a bypassed liquid node contributes."""
    from .scene3d import ParticleInstance
    e = empty_arrays()
    zero = np.zeros(0, np.float32)
    return ParticleInstance(positions=e["position"], sizes=e["size"], colors=e["color"], velocities=e["velocity"],
                            ages=zero, lifetimes=zero, ids=e["id"])


def surface_level_set(instance, params, temporal_instances=()):
    """The filtered signed-distance grid and voxel size used by FluidSurface3D."""
    stream = getattr(instance, "stream", None)
    if not isinstance(stream, LiquidStream) or not len(instance):
        return None, None
    resolution = min(4, max(1, int(params.get("surface_resolution", 1)) * int(params.get("detail_ratio", 1))))
    phi, voxel = signed_distance(stream, instance.positions, 0, resolution,
                                 float(params["particle_radius"]))
    # Average signed distances on the common grid. Averaging fields (rather than vertices) also
    # tolerates changing particle counts and preserves a single watertight extraction.
    fields = [phi]
    for sample in temporal_instances:
        if sample is not None and len(sample):
            other, other_voxel = signed_distance(stream, sample.positions, 0, resolution,
                                                  float(params["particle_radius"]))
            if other.shape == phi.shape and other_voxel == voxel:
                fields.append(other)
    if len(fields) > 1:
        phi = np.mean(np.stack(fields), axis=0, dtype=np.float32)
    # A half-particle-spacing SDF dilation closes sub-particle gaps in fast, thin splashes before
    # extraction. Since phi is signed, reducing it expands the inside set without changing topology.
    if params.get("thin_sheet_preservation", 0):
        phi = preserve_thin_sheet_gaps(phi, stream.spacing)
    return phi, voxel


def preserve_thin_sheet_gaps(phi, particle_spacing):
    """Expand the level set by half a particle spacing to bridge sub-particle gaps in splash sheets."""
    return np.asarray(phi, np.float32) - np.float32(0.5 * particle_spacing)


def surface_geometry(instance, params, temporal_instances=()):
    """The Geometry (closed mesh, smooth outward normals) of a liquid's particles for FluidSurface3D."""
    from . import scene3d
    from .liquid_surface import marching_tetrahedra, taubin
    stream = getattr(instance, "stream", None)
    if not isinstance(stream, LiquidStream) or not len(instance):
        return scene3d.empty_geometry()
    phi, voxel = surface_level_set(instance, params, temporal_instances)
    vertices, triangles, normals = marching_tetrahedra(phi, stream.origin, voxel)
    lo = np.asarray(stream.origin, np.float64)
    vertices = taubin(vertices, triangles, int(params["smoothing"]), lo, lo + np.array(phi.shape) * voxel)
    if not len(triangles):
        return scene3d.empty_geometry()
    from .motionblur import point_velocities
    velocities = point_velocities(vertices, instance.positions, instance.velocities, voxel) \
        if instance.velocities is not None else None
    return scene3d.Geometry(vertices, triangles, LIQUID_COLOR, normals=normals, velocities=velocities)


def foam_instance(instance, params):
    """The particles of a liquid that a splash throws off: fast where the surface curves hard. A ParticleInstance
    of the tagged subset (smaller, white), with no stream, so a cache downstream passes it through."""
    from dataclasses import replace
    from .liquid_surface import foam_mask
    stream = getattr(instance, "stream", None)
    if not isinstance(stream, LiquidStream) or not len(instance):
        return empty_instance()
    phi, voxel = signed_distance(stream, instance.positions)
    mask = foam_mask(instance.positions, instance.velocities, phi, stream.origin, stream.voxel, float(params["foam_speed"]),
                     float(params["foam_curvature"]), stream.fps, stream.shape)
    ages = None if instance.ages is None else instance.ages[mask]
    life = None if instance.lifetimes is None else instance.lifetimes[mask]
    white = np.tile(np.array((1.0, 1.0, 1.0, 1.0), np.float32), (int(mask.sum()), 1))
    return replace(instance, positions=instance.positions[mask], sizes=instance.sizes[mask] * np.float32(params["foam_size"]),
                   colors=white, velocities=instance.velocities[mask], ages=ages, lifetimes=life,
                   ids=None if instance.ids is None else instance.ids[mask], stream=None, surface=None)
