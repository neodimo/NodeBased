"""Compare warmed per-sample GPU path-trace cost on a 256-cubed narrow plume."""
from __future__ import annotations

import time

from nodebased import fluid_gpu_solver, gpupathtrace, pathtrace
from tests.test_pathtrace_volume_skip import plume


def main():
    scene, camera, volume = plume()
    print("adapter:", fluid_gpu_solver.adapter_name(), flush=True)
    for enabled in (False, True):
        gpupathtrace.ENABLE_VOLUME_SKIP = enabled
        timings = []
        for samples in (1, 8, 72):
            start = time.perf_counter()
            pathtrace.render(scene, camera, 128, 128, ambient=.7, volume=volume, backend="gpu",
                             settings=pathtrace.PathSettings(samples=samples, max_bounces=4, seed=4))
            timings.append(time.perf_counter() - start)
        slope = (timings[2] - timings[1]) / 64
        print(f"skip={enabled}: {slope * 1000:.3f} ms/sample; "
              f"8-sample {timings[1]:.3f} s; 72-sample {timings[2]:.3f} s", flush=True)


if __name__ == "__main__":
    main()
