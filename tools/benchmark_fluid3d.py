"""Milliseconds per substep of the 3D smoke solver, with the pressure solve split out, and memory per frame.

`python tools/benchmark_fluid3d.py [--sizes 64 128] [--steps 5] [--warmup 8] [--advection maccormack] [--gpu]`

The plume (the solver's built-in sphere source, buoyancy on) is run for `--warmup` substeps so the pressure field
is warm-started as in real use, then `--steps` substeps are timed. Reported per size: total ms per substep, of
which pressure ms, the non-pressure remainder, the mean pressure iteration count (CG iterations, or SOR sweeps on
the GPU), the largest cell divergence left, the bytes of one solver checkpoint (`State.nbytes`), and the peak
resident set of the process. With `--gpu` the same substeps run with the wgpu SOR pressure solve
(nodebased/fluid_gpu3d.py) and its result is compared with the NumPy one after one projection of the same warmed-up
state. GPU runs take the exclusive lock: `flock /tmp/nb-gpu.lock python tools/benchmark_fluid3d.py --gpu`.
"""
from __future__ import annotations

import argparse
import platform
import resource
import time

import numpy as np

from nodebased import fluid3d


def run(size, steps, warmup, pressure_solver=None, params=None):
    solver = fluid3d.Smoke3D({"nx": size, "ny": size, "nz": size, **(params or {})}, pressure_solver=pressure_solver)
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
    a = state.arrays
    div = fluid3d.divergence(a["u"].astype(np.float64), a["v"].astype(np.float64), a["w"].astype(np.float64))
    return {"size": size, "ms_step": 1000 * total / steps, "ms_pressure": 1000 * solver.pressure_seconds / steps,
            "iterations": float(np.mean(iterations)), "max_div": float(np.abs(div).max()),
            "checkpoint_mb": state.nbytes / 2 ** 20, "state": state}


def parity(size, other, warmup=8, params=None):
    """Largest velocity difference after one projection of the same warmed-up state, NumPy CG versus `other`."""
    solver = fluid3d.Smoke3D({"nx": size, "ny": size, "nz": size, **(params or {})})
    state = solver.initial_state()
    for frame in range(1, warmup + 1):
        state = solver.step(state, frame, 0, 0)
    a = solver.step(state, warmup + 1, 0, 0)
    solver.pressure_solver = other
    b = solver.step(state, warmup + 1, 0, 0)
    return max(float(np.abs(a.arrays[n] - b.arrays[n]).max()) for n in ("u", "v", "w"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--advection", default="maccormack", choices=fluid3d.ADVECTIONS)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    params = {"advection": args.advection}
    print(f"{platform.processor() or platform.machine()}, numpy {np.__version__}, float32 fields, "
          f"tolerance {fluid3d.DEFAULTS['tolerance']}, cap {fluid3d.DEFAULTS['max_iterations']}, "
          f"advection {args.advection}")
    gpu = None
    if args.gpu:
        from nodebased.fluid_gpu3d import GpuPressure3D
        gpu = GpuPressure3D()
        print("GPU:", gpu.adapter_name)
    for size in args.sizes:
        cpu = run(size, args.steps, args.warmup, params=params)
        rest = cpu["ms_step"] - cpu["ms_pressure"]
        print(f"{size}^3 numpy: {cpu['ms_step']:.0f} ms/substep, pressure {cpu['ms_pressure']:.0f} ms, "
              f"other {rest:.0f} ms, {cpu['iterations']:.0f} CG iterations, max divergence {cpu['max_div']:.2e}, "
              f"checkpoint {cpu['checkpoint_mb']:.1f} MB, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
        if gpu is not None:
            other = run(size, args.steps, args.warmup, pressure_solver=gpu.solve, params=params)
            diff = parity(size, gpu.solve, params=params)
            rest = other["ms_step"] - other["ms_pressure"]
            print(f"{size}^3 wgpu : {other['ms_step']:.0f} ms/substep, pressure {other['ms_pressure']:.0f} ms, "
                  f"other {rest:.0f} ms, {other['iterations']:.0f} SOR sweeps, max divergence {other['max_div']:.2e}, "
                  f"max |velocity difference| after one step vs numpy {diff:.2e}")


if __name__ == "__main__":
    main()
