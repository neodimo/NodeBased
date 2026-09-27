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
       turbulence: `fluid3d.Force` applied to a density of one in the liquid cells), viscosity (explicit
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

import math

import numpy as np

from .cancellation import Cancelled
from .fluid3d import Poisson3D, Smoke3D, Stencil, conjugate_gradient, _sl
from .simcache import State

ARRAYS = ("position", "velocity", "age", "life", "size", "color", "id")
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
    "viscosity": 0.0,               # cells squared per frame; explicit diffusion, off at zero
    "tolerance": 1.0e-3, "max_iterations": 1500,
    "origin_x": 0.0, "origin_y": 0.0, "origin_z": 0.0, "voxel_size": 1.0,
    "start_frame": 1, "seed": 0,
    "max_per_cell": MAX_PER_CELL,
}


def empty_arrays():
    return {"position": np.zeros((0, 3), np.float32), "velocity": np.zeros((0, 3), np.float32),
            "age": np.zeros(0, np.int32), "life": np.zeros(0, np.int32), "size": np.zeros(0, np.float32),
            "color": np.zeros((0, 4), np.float32), "id": np.zeros(0, np.int64)}


class LiquidPoisson(Poisson3D):
    """The negative 7-point Laplacian over the liquid cells with p = 0 in air: A q = rhs.

    A face between two liquid cells couples them; a face to a solid cell or the domain wall drops out
    (Neumann); a face to an air cell adds one to the diagonal (a p = 0 neighbour). Rows of non-liquid cells
    are zero. `singular` is true when no liquid cell touches air (a full container), where the right-hand
    side is made zero-mean like a closed smoke box.
    """

    def __init__(self, shape, liquid, solid=None):
        self.shape = tuple(shape)
        self.solid = None if solid is None or not np.any(solid) else np.asarray(solid, bool)
        self.open_axes = (False, False, False)
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
        self.open_axes = (False, False, False)
        self.origin = np.array((p["origin_x"], p["origin_y"], p["origin_z"]), np.float64)
        self.voxel = float(p["voxel_size"])
        self.ppc = max(1, int(p["particles_per_cell"]))
        self.max_per_cell = max(int(p["max_per_cell"]), int(math.ceil(1.5 * self.ppc)))
        self.flip_ratio = float(p["flip_ratio"])
        self.start_frame = int(p["start_frame"])
        self.pressure_solver = pressure_solver or conjugate_gradient
        self.cancel = cancel
        self.sources = [s for s in (sources or ()) if getattr(s, "fluid_type", "liquid") == "liquid"]
        self.forces = list(forces)
        self.colliders = list(colliders)
        self._systems = {}
        self.stats = {"cg_iterations": 0, "particles": 0}

    # -- simcache API -------------------------------------------------------------------------------
    def initial_state(self, seed=0) -> State:
        return State(empty_arrays(), {"next_id": 0, "substep_count": 0, "cg_iterations": 0, "cg_residual": 0.0},
                     copy=False)

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
        SOLVER_STATS["steps"] += 1
        p, dt = self.params, self.dt
        shape = self.shape
        arrays = state.arrays
        pos = (arrays["position"].astype(np.float64) - self.origin) / self.voxel
        vel = arrays["velocity"].astype(np.float64) / self.voxel
        ids = arrays["id"].copy()
        age = arrays["age"].copy()
        next_id = int(state.meta.get("next_id", 0))
        rng = np.random.default_rng((int(seed) & 0x7FFFFFFF, int(frame) & 0x7FFFFFFF, int(substep), 1))
        solid, solid_velocity, _unused = self._solid_for(frame, substep)

        pos, vel, ids, age, next_id = self._emit(pos, vel, ids, age, next_id, solid, frame, substep, rng)
        pos, vel, ids, age, next_id = self._maintain(pos, vel, ids, age, next_id, solid, rng)

        # 3. particle to grid
        stencils = self._stencils(pos)
        u, v, w, valid_old, old_grid = self._to_grid(vel, stencils)
        # 4. forces
        liquid = self._classify(pos, solid)
        a = {"u": u, "v": v, "w": w, "density": liquid.astype(np.float64)}
        v += self.params["gravity"] * (-dt)
        for force in self.forces:
            force.apply(self, a, frame, substep, dt)
        if float(p["viscosity"]) > 0.0:
            self._viscosity(a, float(p["viscosity"]) * dt)
        # 5 and 6. boundaries, projection
        self._face_constraints(a, solid, solid_velocity)
        system = LiquidPoisson(shape, liquid, solid)
        iterations, residual = self._project(a, liquid, solid, system)
        # faces we can trust: those touching liquid, and the ones held by solids and walls
        touched = self._touched_faces(liquid, solid)
        # 7. extrapolate the new and the old velocity into the air
        new_f, old_f = {}, {}
        for axis, name in enumerate(("u", "v", "w")):
            new_f[name], _ = extrapolate(a[name], touched[name], EXTRAPOLATE_LAYERS)
            old_f[name], _ = extrapolate(old_grid[name], valid_old[name] | touched[name], EXTRAPOLATE_LAYERS)
        # 8. grid to particle
        vel = self._from_grid(vel, new_f, old_f, stencils)
        # 9. advect
        pos = self._advect(pos, vel, new_f, solid, dt)
        age = age + 1
        self.stats.update(cg_iterations=int(iterations), particles=int(len(pos)))
        return self._pack(pos, vel, ids, age, state.meta, next_id, iterations, residual)

    # -- pieces -------------------------------------------------------------------------------------
    def _pack(self, pos, vel, ids, age, meta, next_id, iterations, residual):
        n = len(pos)
        spacing = 1.0 / self.ppc ** (1.0 / 3.0)
        size = np.full(n, spacing * self.voxel, np.float32)
        color = np.tile(np.asarray(LIQUID_COLOR, np.float32), (n, 1))
        out = {"position": (pos * self.voxel + self.origin).astype(np.float32),
               "velocity": (vel * self.voxel).astype(np.float32), "age": age.astype(np.int32),
               "life": np.full(n, LIFE, np.int32), "size": size, "color": color, "id": ids.astype(np.int64)}
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

    def _emit(self, pos, vel, ids, age, next_id, solid, frame, substep, rng):
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
            added.append((fresh, np.tile(velocity, (len(fresh), 1))))
        if not added:
            return pos, vel, ids, age, next_id
        new_pos = np.concatenate([a for a, _ in added])
        new_vel = np.concatenate([b for _, b in added])
        n = len(new_pos)
        new_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        return (np.concatenate((pos, new_pos)), np.concatenate((vel, new_vel)), np.concatenate((ids, new_ids)),
                np.concatenate((age, np.zeros(n, age.dtype))), next_id + n)

    def _maintain(self, pos, vel, ids, age, next_id, solid, rng):
        if not len(pos):
            return pos, vel, ids, age, next_id
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
            pos, vel, ids, age, cells = pos[keep], vel[keep], ids[keep], age[keep], cells[keep]
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
            return pos, vel, ids, age, next_id
        # mean neighbour velocity per cell
        sums = np.zeros(self.shape + (3,))
        for c in range(3):
            sums[..., c] = np.bincount(cells, weights=vel[:, c], minlength=total).reshape(self.shape)
        nsum = np.zeros_like(sums)
        ncnt = np.zeros(self.shape)
        for axis in range(3):
            lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
            nsum[hi] += sums[lo]
            ncnt[hi] += count[lo]
            nsum[lo] += sums[hi]
            ncnt[lo] += count[hi]
        mean = nsum / np.maximum(ncnt, 1)[..., None]
        flat = np.flatnonzero(gap.reshape(-1))
        need = (MIN_PER_CELL - count.reshape(-1)[flat]).astype(np.intp)
        flat_rep = np.repeat(flat, need)
        base = np.stack(np.unravel_index(flat_rep, self.shape), axis=1).astype(np.float64)
        fresh = base + rng.random(base.shape)
        fresh_vel = mean.reshape(-1, 3)[flat_rep]
        n = len(fresh)
        fresh_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        return (np.concatenate((pos, fresh)), np.concatenate((vel, fresh_vel)), np.concatenate((ids, fresh_ids)),
                np.concatenate((age, np.zeros(n, age.dtype))), next_id + n)

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

    def _viscosity(self, a, k):
        k = min(k, 0.125)
        for name in ("u", "v", "w"):
            f = a[name]
            lap = np.zeros_like(f)
            for axis in range(3):
                lo, hi = _sl(axis, slice(None, -1)), _sl(axis, slice(1, None))
                lap[hi] += f[lo] - f[hi]
                lap[lo] += f[hi] - f[lo]
            f += k * lap

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
            new = np.clip(pos + h * v2, 1e-3, upper)
            if solid is not None:
                blocked = solid.reshape(-1)[self._cell(new)]
                new[blocked] = pos[blocked]
            pos = new
        return pos


def liquid_volume(state, stream=None, shape=None, voxel=None):
    """Liquid volume in world units cubed: cells' worth of particles times the cell volume."""
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
                      "particles_per_cell": self.ppc,
                      "gravity": p["liquid_gravity"] / (self.fps * self.fps) / self.voxel,
                      "viscosity": p["viscosity"], "tolerance": p["tolerance"], "max_iterations": p["max_iterations"],
                      "origin_x": self.origin[0], "origin_y": self.origin[1], "origin_z": self.origin[2],
                      "voxel_size": self.voxel, "start_frame": self.start_frame, "seed": self.seed}
            hook = None
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
    """`pressure` resolved to "cpu" or "gpu" (the wgpu SOR hook of step B, which reads the liquid system's own
    diagonal). The multigrid and resident GPU solvers assume a smoke system and are refused for liquids."""
    from . import fluid_gpu3d
    choice = params["pressure"]
    if choice in ("resident", "resident_sparse"):
        raise ValueError(f"FluidLiquidSolver3D: pressure {choice} is for the smoke solver; use cpu, gpu or auto")
    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        if not fluid_gpu3d.available():
            raise ValueError("FluidLiquidSolver3D: pressure is gpu but no wgpu adapter can be opened here")
        return "gpu"
    return "gpu" if cells >= GPU_AUTO_CELLS and fluid_gpu3d.available() else "cpu"


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
    fps = float(doc.get("time", {}).get("fps", 24.0))
    identity = {"kind": "FluidLiquidSolver3D", "params": params, "backend": backend, "fps": fps, "format": 1}
    stream = LiquidStream(base, params, simcache.run_key(base.run, identity), fps)
    stream.backend = backend
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
    return level_set(positions, stream.origin, stream.voxel / res, fine, radius, support, smoothing), stream.voxel / res


def instance_from_state(state, stream, frame):
    """The frame's ParticleInstance, with the signed-distance Volume on `.surface` when `liquid_sdf` is on."""
    from dataclasses import replace
    from . import particles
    from .scene3d import Volume
    inst = particles.instance_from_state(state, stream, frame)
    if not int(stream.params.get("liquid_sdf", 1)):
        return inst
    phi, voxel = signed_distance(stream, inst.positions)
    return replace(inst, surface=Volume(phi, voxel_size=voxel, origin=stream.origin, frame=int(frame)))


def empty_instance():
    """A ParticleInstance with no particles and no stream: what a bypassed liquid node contributes."""
    from .scene3d import ParticleInstance
    e = empty_arrays()
    zero = np.zeros(0, np.float32)
    return ParticleInstance(positions=e["position"], sizes=e["size"], colors=e["color"], velocities=e["velocity"],
                            ages=zero, lifetimes=zero, ids=e["id"])


def surface_geometry(instance, params):
    """The Geometry (closed mesh, smooth outward normals) of a liquid's particles for FluidSurface3D."""
    from . import scene3d
    from .liquid_surface import marching_tetrahedra, taubin
    stream = getattr(instance, "stream", None)
    if not isinstance(stream, LiquidStream) or not len(instance):
        return scene3d.empty_geometry()
    phi, voxel = signed_distance(stream, instance.positions, 0, int(params["surface_resolution"]),
                                 float(params["particle_radius"]))
    vertices, triangles, normals = marching_tetrahedra(phi, stream.origin, voxel)
    lo = np.asarray(stream.origin, np.float64)
    vertices = taubin(vertices, triangles, int(params["smoothing"]), lo, lo + np.array(phi.shape) * voxel)
    if not len(triangles):
        return scene3d.empty_geometry()
    return scene3d.Geometry(vertices, triangles, LIQUID_COLOR, normals=normals)


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
