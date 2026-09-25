"""Milliseconds per substep of the 2D smoke solver, with the pressure solve split out.

`python tools/benchmark_fluid.py [--sizes 256 512] [--steps 20] [--warmup 10] [--gpu]`

The plume is run for `--warmup` substeps (so the pressure field is warm-started as in real use) and then
`--steps` substeps are timed. Reported per size: total ms per substep, of which pressure (conjugate
gradient) ms, the mean CG iteration count, and the largest cell divergence left. With `--gpu` the same
substeps are timed with the wgpu pressure solve (tools/fluid_gpu.py) and its result is compared with the
NumPy one. GPU runs take the exclusive lock: `flock /tmp/nb-gpu.lock python tools/benchmark_fluid.py --gpu`.
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from nodebased import fluid2d


def run(size, steps, warmup, pressure_solver=None, params=None):
    solver = fluid2d.Smoke2D({"nx": size, "ny": size, **(params or {})}, pressure_solver=pressure_solver)
    state = solver.initial_state()
    for frame in range(1, warmup + 1):
        state = solver.step(state, frame, 0, 0)
    solver.pressure_seconds = 0.0
    iterations = []
    started = time.perf_counter()
    for frame in range(warmup + 1, warmup + steps + 1):
        state = solver.step(state, frame, 0, 0)
        iterations.append(state.meta["cg_iterations"])
    total = time.perf_counter() - started
    div = fluid2d.divergence(state.arrays["u"].astype(np.float64), state.arrays["v"].astype(np.float64))
    return {"size": size, "ms_step": 1000 * total / steps, "ms_pressure": 1000 * solver.pressure_seconds / steps,
            "iterations": float(np.mean(iterations)), "max_div": float(np.abs(div).max()), "state": state}


def parity(size, other, warmup=10):
    """Largest velocity difference after one projection of the same warmed-up state, NumPy CG versus `other`.

    One step, not a trajectory: the plume is chaotic, so two solves that each meet the tolerance still
    drift apart over many steps, which says nothing about either solver.
    """
    solver = fluid2d.Smoke2D({"nx": size, "ny": size})
    state = solver.initial_state()
    for frame in range(1, warmup + 1):
        state = solver.step(state, frame, 0, 0)
    a = solver.step(state, warmup + 1, 0, 0)
    solver.pressure_solver = other
    b = solver.step(state, warmup + 1, 0, 0)
    return max(float(np.abs(a.arrays[n] - b.arrays[n]).max()) for n in ("u", "v"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(f"{platform.processor() or platform.machine()}, numpy {np.__version__}, float32 fields, "
          f"tolerance {fluid2d.DEFAULTS['tolerance']}, cap {fluid2d.DEFAULTS['max_iterations']}")
    gpu = None
    if args.gpu:
        try:
            from fluid_gpu import GpuPressure
        except ImportError:
            from tools.fluid_gpu import GpuPressure
        gpu = GpuPressure()
        print("GPU:", gpu.adapter_name)
    for size in args.sizes:
        cpu = run(size, args.steps, args.warmup)
        print(f"{size}x{size} numpy: {cpu['ms_step']:.1f} ms/step, pressure {cpu['ms_pressure']:.1f} ms, "
              f"{cpu['iterations']:.0f} CG iterations, max divergence {cpu['max_div']:.2e}")
        if gpu is not None:
            other = run(size, args.steps, args.warmup, pressure_solver=gpu.solve)
            diff = parity(size, gpu.solve)
            print(f"{size}x{size} wgpu : {other['ms_step']:.1f} ms/step, pressure {other['ms_pressure']:.1f} ms, "
                  f"max divergence {other['max_div']:.2e}, max |velocity difference| after one step vs numpy {diff:.2e}")


if __name__ == "__main__":
    main()
