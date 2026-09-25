"""2D smoke solver spike on a MAC grid (Lane 6, step 2). See docs/FLUIDS_SPIKE.md.

Layout and units
    Grid of `ny` rows by `nx` columns of unit cells; row index increases with +y (up). Cell-centred fields
    (`density`, `temperature`, `pressure`) are (ny, nx); the face-centred velocity is `u` (ny, nx + 1), the
    x-velocity on vertical faces, and `v` (ny + 1, nx), the y-velocity on horizontal faces. Cell (j, i)
    has its centre at (i + 0.5, j + 0.5), u[j, i] sits at (i, j + 0.5), v[j, i] at (i + 0.5, j).
    Lengths are in cells and time is in frame units, as in docs/SIMULATION.md: one substep advances
    dt = 1 / substeps of a frame and velocity is cells per frame.

One substep (`Smoke2D.step`)
    1. Emit density and temperature inside the source disc.
    2. Advect velocity, density and temperature semi-Lagrangian: midpoint (RK2) backtrace, bilinear
       interpolation (second-order path, first-order in the interpolated field, so it is diffusive but
       unconditionally stable). Backtraces are clamped to the domain, i.e. a zero-gradient boundary.
    3. Add buoyancy: v += dt * (-alpha * density + beta * (temperature - ambient)).
    4. Add vorticity confinement: f = epsilon * (N x w) with w the cell-centred curl and N the unit
       gradient of |w| (Fedkiw, Stam and Jensen 2001).
    5. Project: conjugate gradient on the 5-point Laplacian with Neumann walls, warm-started from the
       previous pressure, stopping when the largest cell divergence is at most `tolerance` or after
       `max_iterations`; then subtract the pressure gradient from the faces.

Boundaries: a closed box. Wall-normal velocity is zero on all four sides (free-slip: tangential velocity
is unconstrained), density and temperature see a zero-gradient wall.

Determinism: the solver draws no random numbers and holds no state outside the `State` it is handed, so
the same parameters and the same seed give bit-identical grids on the same machine and NumPy build,
however the frames were reached (one solve, or a restore from a checkpoint). This is the rule of
docs/SIMULATION.md ("Determinism"); the arithmetic is fixed-order NumPy with no threaded reductions of
our own.

The simcache forward-solve calls `initial_state(seed)` and `step(state, frame, substep, seed)`; `checkpoint`
and `restore` copy a State in and out of a cache so a solver never shares memory with cached frames.
"""
from __future__ import annotations

import numpy as np

from .cancellation import Cancelled
from .simcache import State

ARRAYS = ("u", "v", "density", "temperature", "pressure")
CANCEL_POLL = 8          # CG iterations between cancellation checks

DEFAULTS = {
    "nx": 128, "ny": 128,
    "substeps": 1,
    "buoyancy_density": 0.05,       # alpha: downward pull of density, cells / frame^2 per unit density
    "buoyancy_temperature": 0.8,    # beta: upward push of temperature, cells / frame^2 per unit
    "ambient_temperature": 0.0,
    "vorticity": 0.3,               # epsilon, vorticity confinement strength
    "tolerance": 1.0e-3,            # largest allowed |divergence| per cell after projection
    "max_iterations": 1500,         # conjugate-gradient cap per substep
    "source_x": 0.5, "source_y": 0.12,   # disc centre as a fraction of the grid
    "source_radius": 0.06,               # disc radius as a fraction of nx
    "source_density": 1.0,          # density added per frame inside the disc
    "source_temperature": 1.0,      # temperature added per frame inside the disc
}


def bilerp(field, x, y):
    """Bilinear sample of `field` at index-space points (x along columns, y along rows), edge-clamped."""
    ny, nx = field.shape
    x = np.clip(x, 0.0, nx - 1.0)
    y = np.clip(y, 0.0, ny - 1.0)
    i0 = np.minimum(x.astype(np.intp), nx - 2)
    j0 = np.minimum(y.astype(np.intp), ny - 2)
    tx = x - i0
    ty = y - j0
    flat = field.ravel()
    base = j0 * nx + i0
    a = flat.take(base)
    b = flat.take(base + 1)
    c = flat.take(base + nx)
    d = flat.take(base + nx + 1)
    bottom = a + (b - a) * tx
    top = c + (d - c) * tx
    return bottom + (top - bottom) * ty


def laplacian_apply(q, out):
    """out = A q, the negative 5-point Laplacian with Neumann walls (each missing neighbour drops out)."""
    out[...] = 0.0
    d = q[:, :-1] - q[:, 1:]
    out[:, :-1] += d
    out[:, 1:] -= d
    d = q[:-1, :] - q[1:, :]
    out[:-1, :] += d
    out[1:, :] -= d
    return out


def divergence(u, v):
    return (u[:, 1:] - u[:, :-1]) + (v[1:, :] - v[:-1, :])


def _dot(a, b):
    """Single-threaded, fixed-order dot product. np.vdot goes to BLAS, whose thread pool both varies the
    summation order with the machine and stalls badly when the machine is busy (measured: 10 to 20 times
    slower per iteration under load), so the solver keeps its reductions out of it."""
    return float(np.einsum("ij,ij->", a, b))


def conjugate_gradient(rhs, x0, tolerance, max_iterations, cancel=None):
    """Solve A x = rhs for the Neumann Laplacian. Stops when max |rhs - A x| <= tolerance.

    Returns (x, iterations, residual). A is singular (constant null space); `rhs` must sum to about zero,
    which a closed box guarantees. Deterministic: fixed operation order, no random start.
    """
    x = x0.copy()
    ap = np.empty_like(x)
    r = rhs - laplacian_apply(x, ap)
    residual = float(np.abs(r).max())
    if residual <= tolerance:
        return x, 0, residual
    p = r.copy()
    rs = _dot(r, r)
    iterations = 0
    while iterations < max_iterations:
        if cancel is not None and iterations % CANCEL_POLL == 0 and cancel.is_set():
            raise Cancelled()
        laplacian_apply(p, ap)
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


class Smoke2D:
    """The solver. Holds parameters only; every substep takes and returns a State."""

    def __init__(self, params=None, pressure_solver=None, cancel=None, dtype=np.float32):
        self.params = {**DEFAULTS, **(params or {})}
        self.nx = int(self.params["nx"])
        self.ny = int(self.params["ny"])
        if self.nx < 4 or self.ny < 4:
            raise ValueError("the grid must be at least 4 by 4")
        self.substeps = max(1, int(self.params["substeps"]))
        self.dt = 1.0 / self.substeps
        self.dtype = np.dtype(dtype)
        # A callable (rhs, x0, tolerance, max_iterations, cancel) -> (x, iterations, residual) replaces the
        # NumPy conjugate gradient; the NumPy one stays the reference (see tools/benchmark_fluid.py).
        self.pressure_solver = pressure_solver or conjugate_gradient
        self.cancel = cancel
        self.pressure_seconds = 0.0
        self._timer = None
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        cx = float(self.params["source_x"]) * self.nx
        cy = float(self.params["source_y"]) * self.ny
        radius = max(1.0, float(self.params["source_radius"]) * self.nx)
        self._source = ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= radius * radius)
        # cell-centre and face positions, computed once
        self._cx = (xx + 0.5).astype(self.dtype)
        self._cy = (yy + 0.5).astype(self.dtype)
        uy, ux = np.mgrid[0:self.ny, 0:self.nx + 1]
        self._ux, self._uy = ux.astype(self.dtype), (uy + 0.5).astype(self.dtype)
        vy, vx = np.mgrid[0:self.ny + 1, 0:self.nx]
        self._vx, self._vy = (vx + 0.5).astype(self.dtype), vy.astype(self.dtype)

    # -- simcache API -------------------------------------------------------------------------------
    def initial_state(self, seed=0) -> State:
        d, ny, nx = self.dtype, self.ny, self.nx
        arrays = {"u": np.zeros((ny, nx + 1), d), "v": np.zeros((ny + 1, nx), d),
                  "density": np.zeros((ny, nx), d), "temperature": np.zeros((ny, nx), d),
                  "pressure": np.zeros((ny, nx), d)}
        return State(arrays, {"substep_count": 0, "cg_iterations": 0, "cg_residual": 0.0}, copy=False)

    def step(self, state, frame=0, substep=0, seed=0) -> State:
        """One substep of dt = 1 / substeps frames. Pure: the input state is not modified."""
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        dt = self.dt
        p = self.params
        u = state.arrays["u"].astype(self.dtype)
        v = state.arrays["v"].astype(self.dtype)
        density = state.arrays["density"].astype(self.dtype)
        temperature = state.arrays["temperature"].astype(self.dtype)
        pressure = state.arrays["pressure"]

        density[self._source] += self.dtype.type(p["source_density"] * dt)
        temperature[self._source] += self.dtype.type(p["source_temperature"] * dt)

        u, v, density, temperature = self._advect(u, v, density, temperature, dt)
        self._buoyancy(v, density, temperature, dt)
        self._confine(u, v, dt)
        u, v, pressure, iterations, residual = self._project(u, v, pressure)

        meta = dict(state.meta)
        meta.update(substep_count=int(meta.get("substep_count", 0)) + 1, cg_iterations=int(iterations),
                    cg_residual=float(residual))
        return State({"u": u, "v": v, "density": density, "temperature": temperature,
                      "pressure": pressure}, meta, copy=False)

    def checkpoint(self, state) -> State:
        return State(state.arrays, state.meta, copy=True)

    def restore(self, state) -> State:
        arrays = {name: np.asarray(state.arrays[name], dtype=self.dtype) for name in ARRAYS}
        return State(arrays, state.meta, copy=True)

    # -- physics ------------------------------------------------------------------------------------
    def _velocity_at(self, u, v, x, y):
        """Velocity at cell-space points (x, y)."""
        return bilerp(u, x, y - 0.5), bilerp(v, x - 0.5, y)

    def _backtrace(self, u, v, x, y, dt):
        ux, vy = self._velocity_at(u, v, x, y)
        xm = np.clip(x - 0.5 * dt * ux, 0.0, self.nx)
        ym = np.clip(y - 0.5 * dt * vy, 0.0, self.ny)
        umid, vmid = self._velocity_at(u, v, xm, ym)
        return x - dt * umid, y - dt * vmid

    def _advect(self, u, v, density, temperature, dt):
        cx, cy = self._backtrace(u, v, self._cx, self._cy, dt)
        density = bilerp(density, cx - 0.5, cy - 0.5)
        temperature = bilerp(temperature, cx - 0.5, cy - 0.5)
        fx, fy = self._backtrace(u, v, self._ux, self._uy, dt)
        new_u = bilerp(u, fx, fy - 0.5)
        gx, gy = self._backtrace(u, v, self._vx, self._vy, dt)
        new_v = bilerp(v, gx - 0.5, gy)
        return new_u.astype(self.dtype), new_v.astype(self.dtype), density, temperature

    def _buoyancy(self, v, density, temperature, dt):
        p = self.params
        force = (self.dtype.type(-p["buoyancy_density"]) * density
                 + self.dtype.type(p["buoyancy_temperature"]) * (temperature - self.dtype.type(p["ambient_temperature"])))
        v[1:-1, :] += self.dtype.type(0.5 * dt) * (force[:-1, :] + force[1:, :])
        v[0, :] = 0.0
        v[-1, :] = 0.0

    def _centre_velocity(self, u, v):
        return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])

    def _confine(self, u, v, dt):
        eps = float(self.params["vorticity"])
        if eps == 0.0:
            return
        uc, vc = self._centre_velocity(u, v)
        w = np.zeros_like(uc)
        w[1:-1, 1:-1] = 0.5 * ((vc[1:-1, 2:] - vc[1:-1, :-2]) - (uc[2:, 1:-1] - uc[:-2, 1:-1]))
        mag = np.abs(w)
        gx = np.zeros_like(w)
        gy = np.zeros_like(w)
        gx[1:-1, 1:-1] = 0.5 * (mag[1:-1, 2:] - mag[1:-1, :-2])
        gy[1:-1, 1:-1] = 0.5 * (mag[2:, 1:-1] - mag[:-2, 1:-1])
        norm = np.sqrt(gx * gx + gy * gy) + self.dtype.type(1.0e-9)
        nx_, ny_ = gx / norm, gy / norm
        scale = self.dtype.type(eps * dt)
        fx = scale * (ny_ * w)
        fy = scale * (-nx_ * w)
        u[:, 1:-1] += 0.5 * (fx[:, :-1] + fx[:, 1:])
        v[1:-1, :] += 0.5 * (fy[:-1, :] + fy[1:, :])

    def _project(self, u, v, pressure):
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        rhs = -divergence(u, v).astype(np.float64)
        rhs -= rhs.mean()
        started = self._now()
        q, iterations, residual = self.pressure_solver(
            rhs, np.asarray(pressure, dtype=np.float64), float(self.params["tolerance"]),
            int(self.params["max_iterations"]), self.cancel)
        self.pressure_seconds += self._now() - started
        q = q - q.mean()
        q32 = q.astype(self.dtype)
        u[:, 1:-1] -= (q32[:, 1:] - q32[:, :-1])
        v[1:-1, :] -= (q32[1:, :] - q32[:-1, :])
        return u, v, q32, iterations, residual

    @staticmethod
    def _now():
        import time
        return time.perf_counter()


def centre_of_mass(density):
    """(x, y) of the density-weighted centre in cell units, or None for an empty grid."""
    total = float(density.sum(dtype=np.float64))
    if total <= 0.0:
        return None
    ny, nx = density.shape
    ys = (np.arange(ny) + 0.5)[:, None]
    xs = (np.arange(nx) + 0.5)[None, :]
    return (float((density * xs).sum(dtype=np.float64)) / total,
            float((density * ys).sum(dtype=np.float64)) / total)
