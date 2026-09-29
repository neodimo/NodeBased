"""One-process-per-case dense/sparse up-res timing and peak RSS.

Run with --coarse 64 or 128, --layout dense or sparse, --backend cpu or gpu.
The clipped column occupies approximately a tenth of the coarse domain.
"""
from __future__ import annotations

import argparse
import resource
import time

import numpy as np

from nodebased import fluid_upres, fluid_gpu_solver, scene3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=int, choices=(64, 128), required=True)
    parser.add_argument("--layout", choices=("dense", "sparse"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    n = args.coarse
    x, y, z = np.ogrid[:n, :n, :n]
    radius2 = ((x - n * .5) ** 2 + (z - n * .5) ** 2) / (n * .19) ** 2
    height2 = ((y - n * .46) ** 2) / (n * .45) ** 2
    density = np.exp(-(radius2 + height2)).astype(np.float32)
    density[density < .25] = 0
    source = scene3d.Volume(density)
    start = time.perf_counter()
    if args.layout == "sparse":
        out = fluid_upres.upres_sparse_grid(source, 4, args.backend)
        mean = float(np.sum(out.data["density"], dtype=np.float64) / (n * 4) ** 3)
        stored = out.nbytes
        tiles = out.tile_count
    else:
        out = fluid_upres.upres_volume(source, {**fluid_upres.DEFAULTS, "upres_factor": 4,
                                                "upres_backend": args.backend}, 1)
        mean = float(out.density.mean(dtype=np.float64))
        stored = out.density.nbytes
        tiles = (n * 4 // 8) ** 3
    elapsed = time.perf_counter() - start
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"{n}^3 to {n*4}^3 {args.layout} {args.backend}: {elapsed:.3f} s/frame; "
          f"peak RSS {rss_mib:.0f} MiB; stored {stored / 2**20:.2f} MiB; "
          f"tiles {tiles}; mean density {mean:.9f}; adapter {fluid_gpu_solver.adapter_name()}",
          flush=True)


if __name__ == "__main__":
    main()
