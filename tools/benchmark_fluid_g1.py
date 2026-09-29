"""Measure G1 up-res and resident Shape costs on the selected wgpu adapter."""
from __future__ import annotations

import argparse
import time

import numpy as np

from nodebased import fluid_upres, fluid_gpu_solver, scene3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=int, default=96)
    parser.add_argument("--shape-grid", type=int, default=32)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    print("adapter:", fluid_gpu_solver.adapter_name(), flush=True)
    n = args.coarse
    x, y, z = np.ogrid[:n, :n, :n]
    d = np.exp(-((x-n*.5)**2+(y-n*.4)**2+(z-n*.5)**2)/(n*n*.02)).astype(np.float32)
    v = np.empty((n,n,n,3), np.float32)
    v[...,0], v[...,1], v[...,2] = .025, .02, 0.0
    source = scene3d.Volume(d, voxel_size=.1, velocity=v, fuel=d*.3)
    for factor in (2,4):
        for backend in ("cpu", "gpu"):
            start = time.perf_counter()
            out = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS,
                                                    "upres_factor": factor,
                                                    "upres_backend": backend}, 1)
            elapsed = time.perf_counter()-start
            mass = float(out.density.sum(dtype=np.float64))/float(d.sum(dtype=np.float64)*factor**3)
            print(f"upres {n}^3 factor {factor} {backend}: {elapsed:.3f} s; mass ratio {mass:.6f}", flush=True)
            del out
    def shape_timing(change):
        params = {"nx": args.shape_grid, "ny": args.shape_grid, "nz": args.shape_grid,
                  "boundary_y": "open", "vorticity": 0.0, **change}
        solver = fluid_gpu_solver.GpuSmoke3D(params)
        state = solver.initial_state(0)
        for frame in (1, 2):
            state = solver.step(state, frame)
        state.arrays  # exclude pipeline compilation and GPU queue warm-up
        start = time.perf_counter()
        for frame in range(3,args.frames+3):
            state = solver.step(state, frame)
        state.arrays
        return (time.perf_counter()-start)/args.frames
    baseline = shape_timing({})
    print(f"shape baseline {args.shape_grid}^3: {baseline*1000:.2f} ms/frame", flush=True)
    for name, change in (("disturbance", {"disturbance":.8, "disturbance_size":3.0}),
                         ("shredding", {"shredding":.4}),
                         ("turbulence", {"turbulence":.3, "swirl_size":2.0}),
                         ("limited dissipation", {"dissipation":.2, "dissipation_field":"density"}),
                         ("confinement", {"vorticity":.5})):
        current = shape_timing(change)
        print(f"shape {name}: {current*1000:.2f} ms/frame; delta {(current-baseline)*1000:+.2f} ms/frame",
              flush=True)


if __name__ == "__main__":
    main()
