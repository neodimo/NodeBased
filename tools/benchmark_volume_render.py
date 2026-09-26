"""Time the volume raymarch, CPU reference against the GPU shader (docs/3D_FOUNDATION.md, "GPU volume raymarch").

    python tools/benchmark_volume_render.py [--adapter default|discrete|integrated|cpu] [--cpu-scale N]

For a 128 cubed and a 256 cubed analytic plume, one light and three lights, at 1920 by 1080, with the
Render3D default march (step 0.05) and with the step at one voxel. The GPU is timed end to end (host
preparation, upload on the first frame, submission, readback); the first frame is reported apart from the
median of the next three. The CPU reference refuses frames over its sample budget, so it runs at 1/N of the
width and height and the row says so; the column scales that time by N squared.
"""
import argparse
import statistics
import time
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, gpuvolume, scene3d, volumerender

WIDTH, HEIGHT = 1920, 1080
CAMERA = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0.4, 0.7, 2.6)), scene3d.Vec3(0, 0.5, 0), 45.0)
LIGHTS = (
    scene3d.Light("Directional", position=scene3d.Vec3(4, 5, 3), target=scene3d.Vec3()),
    scene3d.Light("Point", color=(1.0, 0.6, 0.3), intensity=3.0, position=scene3d.Vec3(-1.5, 1.2, 1.0),
                  target=scene3d.Vec3(0, .5, 0), falloff_type="Quadratic"),
    scene3d.Light("Spot", color=(0.4, 0.6, 1.0), intensity=2.0, position=scene3d.Vec3(1.5, 2.5, 1.5),
                  target=scene3d.Vec3(0, .5, 0), cone_angle=40.0, cone_penumbra_angle=15.0, falloff_type="Linear"),
)


def timed(function):
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--cpu-scale", type=int, default=8, help="the CPU reference renders at 1/N size")
    parser.add_argument("--sizes", default="128,256")
    args = parser.parse_args()
    print(f"adapter: {gpu3d.describe() if args.adapter is None else args.adapter}")
    print("| grid | lights | step | GPU first frame | GPU steady | density lookups | GPU lookups/s | CPU (1/%d size) | CPU x%d est. |" % (
        args.cpu_scale, args.cpu_scale ** 2))
    print("|---|---|---|---|---|---|---|---|---|")
    for n in (int(v) for v in args.sizes.split(",")):
        volume = scene3d.analytic_plume(n, 1)
        for light_count in (1, 3):
            scene = scene3d.Scene(volumes=(volume,), lights=LIGHTS[:light_count])
            for label, step in (("0.05", 0.05), (f"1/{n}", 1.0 / n)):
                settings = volumerender.VolumeSettings(step_size=step, density_scale=8.0)
                work = gpuvolume.work_estimate(scene, CAMERA, WIDTH, HEIGHT, settings, light_count)
                try:
                    first, _ = timed(lambda: gpu3d.render(scene, CAMERA, WIDTH, HEIGHT, ambient=0.1, volume=settings,
                                                          adapter=args.adapter))
                    steady = statistics.median(timed(lambda: gpu3d.render(
                        scene, CAMERA, WIDTH, HEIGHT, ambient=0.1, volume=settings, adapter=args.adapter))[0]
                        for _ in range(3))
                    gpu = f"{first:.2f} s", f"{steady:.2f} s", f"{work / steady:.2e}"
                except Exception as exc:
                    gpu = f"refused: {str(exc)[:40]}", "-", "-"
                small_w, small_h = WIDTH // args.cpu_scale, HEIGHT // args.cpu_scale
                try:
                    cpu_time, _ = timed(lambda: scene3d.render(scene, CAMERA, small_w, small_h, ambient=0.1, volume=settings))
                    cpu = f"{cpu_time:.1f} s", f"{cpu_time * args.cpu_scale ** 2:.0f} s"
                except ValueError as exc:
                    cpu = "over the CPU budget", "-"
                print(f"| {n}^3 | {light_count} | {label} | {gpu[0]} | {gpu[1]} | {work:.2e} | {gpu[2]} | {cpu[0]} | {cpu[1]} |",
                      flush=True)


if __name__ == "__main__":
    main()
