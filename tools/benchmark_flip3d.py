"""Milliseconds per substep of the FLIP liquid solver (nodebased/flip3d.py), particles per second, and the surface.

`python tools/benchmark_flip3d.py [--sizes 64 128] [--steps 3] [--warmup 2] [--surface] [--gpu]`

A dam break: a block filling half of each axis (an eighth of the grid), 8 particles per cell, gravity on, one substep
per frame. `--warmup` substeps let the column start to fall (the pressure system is warm and the free surface is
real), then `--steps` are timed. Reported per size: particles, total ms per substep, of which the pressure solve,
conjugate gradient iterations, particles per second (particles times steps over seconds), the bytes of one checkpoint
and the peak resident set. With `--surface` the level set and the marching-tetrahedra mesh of the last state are timed
too (`FluidSurface3D`'s work at resolution 1), and with `--gpu` particle transfers use sparse-tile wgpu kernels
alongside the wgpu SOR pressure hook of step B (take the exclusive lock: `flock /tmp/nb-gpu.lock python
tools/benchmark_flip3d.py --gpu`).
"""
from __future__ import annotations

import argparse
import platform
import resource
import time

import numpy as np

from nodebased import fluid3d, flip3d, liquid_surface


class Block(fluid3d.Source):
    def __init__(self, hi):
        super().__init__("sphere", fluid_type="liquid")
        self.hi = hi

    def footprint(self, solver, frame):
        ii, jj, kk = np.meshgrid(*(np.arange(self.hi[a]) for a in range(3)), indexing="ij")
        flat = np.ravel_multi_index((ii.ravel(), jj.ravel(), kk.ravel()), solver.shape).astype(np.intp)
        return flat, np.ones(len(flat)), None


def run(size, steps, warmup, surface, gpu):
    hook = None
    if gpu:
        from nodebased.fluid3d import _gpu_solver
        hook = _gpu_solver().solve
    seconds = {"pressure": 0.0}

    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return (hook or fluid3d.conjugate_gradient)(*args, **kwargs)
        finally:
            seconds["pressure"] += time.perf_counter() - started
    solver = flip3d.Liquid3D({"nx": size, "ny": size, "nz": size, "gravity": 0.05, "flip_ratio": 0.95,
                              "backend": "gpu" if gpu else "cpu"},
                             pressure_solver=timed, sources=[Block((size // 2,) * 3)])
    state = solver.initial_state()
    for frame in range(1, warmup + 1):
        state = solver.step(state, frame, 0, 0)
    seconds["pressure"] = 0.0
    iterations, started = [], time.perf_counter()
    for frame in range(warmup + 1, warmup + steps + 1):
        state = solver.step(state, frame, 0, 0)
        iterations.append(state.meta["cg_iterations"])
    total = time.perf_counter() - started
    n = len(state.arrays["position"])
    out = {"size": size, "particles": n, "ms_step": 1000 * total / steps, "ms_pressure": 1000 * seconds["pressure"] / steps,
           "iterations": float(np.mean(iterations)), "particles_per_second": n * steps / total,
           "checkpoint_mb": state.nbytes / 2 ** 20, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if surface:
        pos = state.arrays["position"]
        spacing = 1.0 / 8 ** (1.0 / 3.0)
        started = time.perf_counter()
        phi = liquid_surface.level_set(pos, (0.0, 0.0, 0.0), 1.0, (size,) * 3, spacing, 3 * spacing)
        out["ms_level_set"] = 1000 * (time.perf_counter() - started)
        started = time.perf_counter()
        v, t, _ = liquid_surface.marching_tetrahedra(phi, (0.0, 0.0, 0.0), 1.0)
        out["ms_mesh"] = 1000 * (time.perf_counter() - started)
        out["triangles"] = len(t)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--surface", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(f"{platform.processor() or platform.machine()}, numpy {np.__version__}, "
          f"pressure {'gpu SOR' if args.gpu else 'cpu CG'}")
    for size in args.sizes:
        r = run(size, args.steps, args.warmup, args.surface, args.gpu)
        line = (f"{r['size']}^3: {r['particles']:,} particles, {r['ms_step']:.0f} ms/substep "
                f"(pressure {r['ms_pressure']:.0f} ms, {r['iterations']:.0f} iterations), "
                f"{r['particles_per_second'] / 1e6:.2f} M particles/s, checkpoint {r['checkpoint_mb']:.0f} MB, "
                f"peak RSS {r['peak_rss_mb']:.0f} MB")
        if "ms_level_set" in r:
            line += f"; surface: level set {r['ms_level_set']:.0f} ms, mesh {r['ms_mesh']:.0f} ms, {r['triangles']:,} triangles"
        print(line, flush=True)


if __name__ == "__main__":
    main()
