"""Measures for the 3D smoke solver that do not depend on the grid's cell size (Lane 6, Fluids 18 B).

The solver keeps velocity in cells per frame on a staggered grid (`fluid3d`). Everything here converts to world
units (metres per second) first, so a number read on a 64-cell grid and on a 128-cell grid of the same box means
the same thing:

* `ring_vortex_arrays` seeds one vortex ring in still air, exactly divergence-free on the staggered grid (the
  velocity is the discrete curl of a vector potential), with a size and a peak velocity given in metres.
* `world_vorticity` is the curl of the face velocity in 1/s at the cell centres, from centred differences in metres.
* `peak_vorticity` and `kinetic_energy` follow a vortex through a run.
* `curl_share` is the share of smoke cells whose world vorticity magnitude is above a threshold (30 per second in
  the Fluids 17 and 18 notes), optionally after a box blur of a fixed physical width so two grids see the same scale.
"""
from __future__ import annotations

import sys
import time

import numpy as np

BOX = 2.0           # metres, the closed test box of `follow_ring_vortex`
FPS = 24.0


def cell_centred_velocity(arrays, voxel, fps):
    """Cell-centred velocity (nx, ny, nz, 3) in metres per second from the solver's face arrays (cells per frame)."""
    u, v, w = arrays["u"], arrays["v"], arrays["w"]
    scale = np.float64(voxel) * np.float64(fps)
    return np.stack((0.5 * (u[:-1] + u[1:]), 0.5 * (v[:, :-1] + v[:, 1:]), 0.5 * (w[:, :, :-1] + w[:, :, 1:])),
                    axis=-1).astype(np.float64) * scale


def world_vorticity(velocity, voxel):
    """Curl of a cell-centred velocity field (metres per second) in 1/s, shape (nx, ny, nz, 3); centred differences
    inside, one-sided at the walls, spacing `voxel` metres."""
    d = float(voxel)
    gx = [np.gradient(velocity[..., c], d, axis=0) for c in range(3)]
    gy = [np.gradient(velocity[..., c], d, axis=1) for c in range(3)]
    gz = [np.gradient(velocity[..., c], d, axis=2) for c in range(3)]
    return np.stack((gy[2] - gz[1], gz[0] - gx[2], gx[1] - gy[0]), axis=-1)


def peak_vorticity(arrays, voxel, fps):
    """Largest |curl| of the velocity in 1/s."""
    omega = world_vorticity(cell_centred_velocity(arrays, voxel, fps), voxel)
    return float(np.sqrt((omega ** 2).sum(-1)).max())


def kinetic_energy(arrays, voxel, fps):
    """Kinetic energy per unit density in m^5/s^2 (0.5 * sum |u|^2 * voxel^3) of the face velocities."""
    scale = (float(voxel) * float(fps)) ** 2 * float(voxel) ** 3
    return float(0.5 * scale * sum(float((np.asarray(arrays[k], np.float64) ** 2).sum()) for k in ("u", "v", "w")))


def max_divergence(arrays, voxel, fps):
    """Largest |divergence| of the face velocity in 1/s over the cells (what the projection leaves behind)."""
    u, v, w = (np.asarray(arrays[k], np.float64) for k in ("u", "v", "w"))
    div = (u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])
    return float(np.abs(div).max() * float(fps))


def ring_vortex_arrays(shape, voxel, fps, centre, radius, core, peak_speed, axis=1, direction=1.0):
    """Face velocities (cells per frame) of one vortex ring in still air, for `Smoke3D.initial_state` arrays.

    shape: (nx, ny, nz); voxel: cell size in metres; centre: the ring's centre in metres from the grid's low corner;
    radius: ring radius (m); core: Gaussian core radius (m); peak_speed: the largest speed of the seeded field in m/s.
    The velocity is the staggered curl of the potential A = g (-z', 0, x') (for axis 1, y), g = exp(-((rho - R)^2 + h^2) / a^2),
    so the discrete divergence is zero to round-off. `direction` is +1 for a ring that travels towards +axis."""
    nx, ny, nz = shape
    d = float(voxel)
    ax = int(axis)
    others = [i for i in range(3) if i != ax]       # the two axes of the ring's plane, in increasing order
    c = [float(x) for x in centre]

    def potential(px, py, pz):
        pos = (px, py, pz)
        p1 = pos[others[0]] - c[others[0]]
        p2 = pos[others[1]] - c[others[1]]
        h = pos[ax] - c[ax]
        rho = np.sqrt(p1 ** 2 + p2 ** 2)
        g = np.exp(-(((rho - radius) ** 2 + h ** 2) / core ** 2))
        out = [np.zeros_like(rho) for _ in range(3)]
        # A = g * (axis x position-in-plane): in the plane, (-p2, p1) along (others[0], others[1])
        out[others[0]] = -g * p2
        out[others[1]] = g * p1
        return out

    def grid(i, j, k, off):
        return ((i + off[0]) * d, (j + off[1]) * d, (k + off[2]) * d)

    def edge_axes(n0, n1, n2):
        return np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2), indexing="ij")

    # A_x lives on x-edges (i + .5, j, k), A_y on y-edges (i, j + .5, k), A_z on z-edges (i, j, k + .5).
    ii, jj, kk = edge_axes(nx + 1, ny + 1, nz + 1)
    ax_x = potential(*grid(ii, jj, kk, (0.5, 0.0, 0.0)))[0]
    ax_y = potential(*grid(ii, jj, kk, (0.0, 0.5, 0.0)))[1]
    ax_z = potential(*grid(ii, jj, kk, (0.0, 0.0, 0.5)))[2]
    # u = dAz/dy - dAy/dz at x-faces (i, j + .5, k + .5), and so on.
    u = (ax_z[:, 1:, :nz] - ax_z[:, :-1, :nz]) / d - (ax_y[:, :ny, 1:] - ax_y[:, :ny, :-1]) / d
    v = (ax_x[:nx, :, 1:] - ax_x[:nx, :, :-1]) / d - (ax_z[1:, :, :nz] - ax_z[:-1, :, :nz]) / d
    w = (ax_y[1:, :ny, :] - ax_y[:-1, :ny, :]) / d - (ax_x[:nx, 1:, :] - ax_x[:nx, :-1, :]) / d
    u, v, w = u[:, :ny, :nz], v[:nx, :, :nz], w[:nx, :ny, :]
    u, v, w = [np.ascontiguousarray(a, np.float64) for a in (u, v, w)]
    faces = (u, v, w)
    # Scale so the largest speed (at the cell centres) is `peak_speed` m/s, and orient the ring's travel along +-axis.
    probe = np.stack((0.5 * (u[:-1] + u[1:]), 0.5 * (v[:, :-1] + v[:, 1:]), 0.5 * (w[:, :, :-1] + w[:, :, 1:])), -1)
    speed = np.sqrt((probe ** 2).sum(-1))
    k = float(peak_speed) / float(speed.max())
    # The potential's sign gives the travel direction; probe the speed along the axis at the ring's centre.
    ci = tuple(int(np.clip(round(c[a] / d - 0.5), 0, (nx, ny, nz)[a] - 1)) for a in range(3))
    along = probe[ci[0], ci[1], ci[2], ax]
    sign = 1.0 if along * float(direction) >= 0 else -1.0
    scale = k * sign / (d * float(fps))
    return {name: (a * scale).astype(np.float32) for name, a in zip(("u", "v", "w"), faces)}


def box_blur(field, voxel, width):
    """Mean over a cube of about `width` metres (at least one cell) around every cell, edges reflected."""
    n = max(1, int(round(float(width) / float(voxel))))
    if n <= 1:
        return np.asarray(field, np.float64)
    out = np.asarray(field, np.float64)
    pad = n // 2
    for axis in range(3):
        padded = np.pad(out, [(pad, n - 1 - pad) if a == axis else (0, 0) for a in range(out.ndim)], mode="reflect")
        c = np.cumsum(padded, axis=axis, dtype=np.float64)
        zero_shape = list(c.shape)
        zero_shape[axis] = 1
        c = np.concatenate((np.zeros(zero_shape), c), axis=axis)
        hi = np.take(c, np.arange(n, n + out.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, out.shape[axis]), axis=axis)
        out = (hi - lo) / n
    return out


def curl_share(density, velocity, voxel, threshold=30.0, floor=0.05, blur=0.0):
    """World-space curl measure: the share of smoke cells (density at least `floor` of the largest) whose vorticity
    magnitude (1/s) is above `threshold`, plus the mean, 90th percentile and rms. `velocity` is cell-centred, m/s.
    `blur` (metres) averages the velocity over a cube of that width first, so grids of different cell size compare
    at the same physical scale; 0 reads the grid at its own scale."""
    vel = np.stack([box_blur(velocity[..., c], voxel, blur) for c in range(3)], -1) if blur > 0 else velocity
    mag = np.sqrt((world_vorticity(vel, voxel) ** 2).sum(-1))
    live = np.asarray(density) >= floor * float(np.max(density))
    inner = live.copy()
    for a in range(3):
        inner &= np.roll(live, 1, a) & np.roll(live, -1, a)
    m = mag[inner]
    if not m.size:
        return {"cells": 0, "share_above": 0.0, "mean": 0.0, "p90": 0.0, "rms": 0.0}
    return {"cells": int(m.size), "share_above": round(float((m > threshold).mean()), 4), "mean": round(float(m.mean()), 3),
            "p90": round(float(np.quantile(m, 0.9)), 3), "rms": round(float(np.sqrt((m ** 2).mean())), 3)}


def follow_ring_vortex(cells, frames=48, backend="cpu", peak_speed=0.5, radius=0.3, core=0.1, substeps=1, vorticity=0.0, advection="maccormack",
        max_iterations=200, tolerance=1e-3, extra=None, every=1):
    """Follow the ring for `frames` frames; returns a list of per-frame rows and the run's timing."""
    from . import fluid3d
    diag = sys.modules[__name__]
    voxel = BOX / cells
    params = {"nx": cells, "ny": cells, "nz": cells, "substeps": substeps, "advection": advection, "vorticity": vorticity,
              "voxel_size": voxel, "buoyancy_density": 0.0, "buoyancy_temperature": 0.0, "default_source": 0, "fire": 0,
              "max_iterations": max_iterations, "tolerance": tolerance, **(extra or {})}
    if backend == "gpu":
        from nodebased import fluid_gpu_solver
        solver = fluid_gpu_solver.GpuSmoke3D(params)
    else:
        solver = fluid3d.Smoke3D(params)
    state = solver.initial_state(0)
    centre = (BOX / 2, 0.7 * BOX / 2 + 0.2, BOX / 2)
    seeded = diag.ring_vortex_arrays((cells,) * 3, voxel, FPS, centre, radius, core, peak_speed)
    arrays = dict(state.arrays)
    arrays.update({k: np.asarray(v, np.float32) for k, v in seeded.items()})
    state = fluid3d.State(arrays, state.meta, copy=True)
    state = solver.restore(state) if hasattr(solver, "restore") else state
    rows = []

    def measure(frame, st):
        a = {k: np.asarray(st.arrays[k], np.float64) for k in ("u", "v", "w")}
        rows.append({"frame": frame, "peak_vorticity": diag.peak_vorticity(a, voxel, FPS), "energy": diag.kinetic_energy(a, voxel, FPS),
                     "max_divergence": diag.max_divergence(a, voxel, FPS),
                     "pressure_iterations": int(st.meta.get("cg_iterations", 0)), "pressure_residual": float(st.meta.get("cg_residual", 0.0))})

    measure(0, state)
    started = time.perf_counter()
    for frame in range(1, frames + 1):
        for substep in range(substeps):
            state = solver.step(state, frame=frame, substep=substep, seed=0)
        if frame % every == 0 or frame == frames:
            measure(frame, state)
    seconds = time.perf_counter() - started
    return rows, seconds
