"""Milliseconds per substep of the GPU-resident 3D smoke solver, dense and sparse, with the pressure time split out.

`flock /tmp/nb-gpu.lock python tools/benchmark_fluid_gpu.py [--sizes 64 128 256] [--warmup 40] [--steps 20]
[--profile-steps 8] [--cpu 64]`

The plume is `Smoke3D`'s built-in sphere source with buoyancy and MacCormack advection in a closed box (the same
scene as tools/benchmark_fluid3d.py), one substep per frame. Per size and mode: `--warmup` substeps first (upload,
shader compilation, calibration of the multigrid cycle count, a developed plume), then `--steps` substeps timed
end to end as a bake would run them (one residual readback per substep, nothing else), then `--profile-steps` more
with a synchronisation between phases, which is what splits out the pressure time (weights, coarse operators, the
V-cycles and their residual, and the gradient subtraction). Also reported: the multigrid cycle count, the active tile
fraction, bytes on the card, the time to read the whole state back (the USB4 cost) and the time of a sparse readback.
`--cpu N` also times the NumPy solver at N cubed for a ratio. `--scene closed` is the step C scene (a closed box, where
the incompressible return flow soon reaches every tile); `--scene open` has open boundaries, cooling and dissipation, a
plume that stays a column in a big box, which is what sparse tiles are for.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time

import numpy as np

from nodebased import fluid3d, gpu3d
from nodebased import fluid_gpu_solver as fgs

PRESSURE_PHASES = ("setup", "pressure", "finish")
SCENES = {"closed": {}, "open": {"boundary_x": "open", "boundary_y": "open", "boundary_z": "open", "cooling_rate": 0.15,
                                 "dissipation": 0.03}}


def smi_used_mb():
    """MiB in use on the card according to nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=20).stdout.split()
        return float(out[0])
    except Exception:
        return None


def run(size, sparse, warmup, steps, profile_steps, params=None, sparse_velocity=fgs.DEFAULT_SPARSE_VELOCITY):
    p = {"nx": size, "ny": size, "nz": size, **(params or {})}
    live_before = fgs._ctx().live_bytes
    solver = fgs.GpuSmoke3D(p, sparse=sparse, sparse_velocity=sparse_velocity)
    state = solver.initial_state()
    frame = 0
    for frame in range(1, warmup + 1):
        state = solver.step(state, frame, 0, 0)
    solver._sync()
    started = time.perf_counter()
    for frame in range(warmup + 1, warmup + steps + 1):
        state = solver.step(state, frame, 0, 0)
    solver._sync()
    ms_step = 1000 * (time.perf_counter() - started) / steps
    cycles = state.meta["mg_cycles"]
    tiles = solver.active_tiles
    solver.profile = True
    solver.phase_seconds = {}
    for frame in range(warmup + steps + 1, warmup + steps + profile_steps + 1):
        state = solver.step(state, frame, 0, 0)
    solver._sync()
    phases = {k: 1000 * v / profile_steps for k, v in solver.phase_seconds.items()}
    read_started = time.perf_counter()
    arrays = state.arrays
    read_seconds = time.perf_counter() - read_started
    total_tiles = (-(-size // 8)) ** 3
    return {"size": size, "sparse": sparse, "ms_step": ms_step, "phases": phases,
            "ms_pressure": sum(phases.get(k, 0.0) for k in PRESSURE_PHASES), "ms_profiled": sum(phases.values()),
            "cycles": cycles, "tiles": tiles, "tile_fraction": None if tiles is None else tiles / total_tiles,
            "bytes_card": solver.ctx.live_bytes - live_before, "smi_mb": smi_used_mb(), "bytes_estimate": solver.estimated_bytes, "readback_ms": 1000 * read_seconds,
            "stored_tiles": solver.stored_tiles, "residual": state.meta["cg_residual"],
            "max_div": float(np.abs(fluid3d.divergence(arrays["u"].astype(np.float64), arrays["v"].astype(np.float64),
                                                       arrays["w"].astype(np.float64))).max())}


def cpu_step_ms(size, warmup=4, steps=2):
    solver = fluid3d.Smoke3D({"nx": size, "ny": size, "nz": size})
    state = solver.initial_state()
    for frame in range(1, warmup + 1):
        state = solver.step(state, frame, 0, 0)
    started = time.perf_counter()
    for frame in range(warmup + 1, warmup + steps + 1):
        state = solver.step(state, frame, 0, 0)
    return 1000 * (time.perf_counter() - started) / steps


def link_speed(size=128):
    """GB/s of one full-grid read from the card and one write to it."""
    ctx = fgs._ctx()
    n = size ** 3
    buf = ctx.buffer(4 * n)
    data = np.zeros(n, np.float32)
    ctx.write(buf, data)
    ctx.read(buf, 4)
    started = time.perf_counter()
    ctx.write(buf, data)
    ctx.read(buf, 4)
    up = time.perf_counter() - started
    started = time.perf_counter()
    ctx.read(buf, 4 * n)
    down = time.perf_counter() - started
    return 4 * n / up / 1e9, 4 * n / down / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--cpu", type=int, nargs="*", default=[])
    ap.add_argument("--sparse-velocity", type=float, default=fgs.DEFAULT_SPARSE_VELOCITY,
                    help="face speed (cells per frame) above which a tile is active")
    ap.add_argument("--scene", choices=sorted(SCENES), default="closed")
    args = ap.parse_args()
    print(gpu3d.adapter_report())
    up, down = link_speed()
    print(f"link: write {up:.2f} GB/s, read {down:.2f} GB/s (one 128 cubed float32 grid)")
    print(f"scene: {args.scene}")
    print("size  mode    ms/substep  pressure ms  other ms  cycles  tile fill  card MB (smi)  full readback ms  sparse tiles stored")
    for size in args.sizes:
        for sparse in (False, True):
            r = run(size, sparse, args.warmup, args.steps, args.profile_steps, SCENES[args.scene], args.sparse_velocity)
            fill = "-" if r["tile_fraction"] is None else f"{100 * r['tile_fraction']:.0f}%"
            print(f"{size:4d}  {'sparse' if sparse else 'dense ':6s} {r['ms_step']:10.1f} {r['ms_pressure']:12.1f} "
                  f"{r['ms_profiled'] - r['ms_pressure']:9.1f} {r['cycles']:7d} {fill:>9s} {r['bytes_card'] / 2 ** 20:6.0f} ({r['smi_mb'] or 0:.0f}) "
                  f"{r['readback_ms']:16.0f}  {r['stored_tiles'] if sparse else '-'}   phases "
                  + ", ".join(f"{k} {v:.1f}" for k, v in r["phases"].items()))
    for size in args.cpu:
        print(f"NumPy at {size} cubed: {cpu_step_ms(size):.0f} ms per substep")


if __name__ == "__main__":
    main()
