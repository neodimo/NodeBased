"""Null collisions and wall time of the path tracers' volume majorant: a plume filling a tenth of its box.

    python tools/benchmark_volume_majorant.py [--sizes 128 256] [--adapter default|integrated|cpu] [--cpu-only|--gpu-only]

Three majorants are compared on the CPU reference and on the GPU: the whole box (`box`), the fine grid alone (`fine`,
`MAJORANT_RATIO` 1) and the two-level grid (`grid`). Per camera path it prints the tentative collisions, the real ones
and the hops from one region to the next, and the wall time per sample.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace, ptvolume, scene3d, volumerender

BOX = 2.56          # world units along each side of the box, whatever the grid size


def tenth_plume(n):
    """A smooth ellipsoid of density that fills about a tenth of the cube it sits in (radii .22, .5, .22 of the side)."""
    axis = ((np.arange(n) + .5) / n - .5).astype(np.float32)
    r2 = (axis[:, None, None] / .22) ** 2 + (axis[None, :, None] / .5) ** 2 + (axis[None, None, :] / .22) ** 2
    density = np.clip(1 - r2, 0, None) ** 2
    volume = scene3d.Volume(density, BOX / n, (-BOX / 2,) * 3)
    scene = scene3d.Scene(volumes=(volume,))
    camera = scene3d.Camera(scene3d.Transform3D(position=scene3d.Vec3(0, 0, 8)), scene3d.Vec3(0, 0, 0), 30)
    settings = volumerender.VolumeSettings(absorption=.6, scattering=.8, density_scale=100)
    return scene, camera, settings, float((density > 0).mean())


def setup(mode):
    ptvolume.ENABLE_SKIP = gpupathtrace.ENABLE_VOLUME_SKIP = mode != "box"
    ptvolume.MAJORANT_RATIO = 1 if mode == "fine" else 4


def counts_cpu(scene, camera, settings, side):
    """Collisions per camera path on the CPU reference (the counters of the layer the render itself built) and the time
    per sample of a `side` pixel square, the slope between 4 and 12 samples so the one-time grid build is left out."""
    layers = []
    real_build = pathtrace.build_scene

    def build(*args, **kwargs):
        ps = real_build(*args, **kwargs)
        layers.append(ps.volumes)
        return ps
    pathtrace.build_scene = build
    times = {4: [], 12: []}
    try:
        for _ in range(3):
            for samples in (4, 12):
                start = time.perf_counter()
                pathtrace.render(scene, camera, side, side, ambient=.7, volume=settings, backend="cpu",
                                 settings=pathtrace.PathSettings(samples=samples, max_bounces=4, seed=4))
                times[samples].append(time.perf_counter() - start)
    finally:
        pathtrace.build_scene = real_build
    c = layers[1].counts
    paths = side * side * 12
    return (min(times[12]) - min(times[4])) / 8, c["tentative"] / paths, c["real"] / paths, c["hops"] / paths


def gpu_row(scene, camera, settings, side=64):
    """Collisions per path from the counting shader, and the warmed per-sample time (slope of 8 and 72 samples)."""
    gpupathtrace.COUNT_COLLISIONS = True
    try:
        image = pathtrace.render(scene, camera, side, side, ambient=.7, volume=settings, backend="gpu",
                                 settings=pathtrace.PathSettings(samples=8, max_bounces=4, seed=4))
    finally:
        gpupathtrace.COUNT_COLLISIONS = False
    counts = image[..., :3].reshape(-1, 3).mean(axis=0)
    times = {8: [], 136: []}
    pathtrace.render(scene, camera, 256, 256, ambient=.7, volume=settings, backend="gpu",
                     settings=pathtrace.PathSettings(samples=1, max_bounces=4, seed=4))      # compile and warm
    for _ in range(4):           # the card is shared with other work: the quietest run of each length
        for samples in times:
            start = time.perf_counter()
            pathtrace.render(scene, camera, 256, 256, ambient=.7, volume=settings, backend="gpu",
                             settings=pathtrace.PathSettings(samples=samples, max_bounces=4, seed=4))
            times[samples].append(time.perf_counter() - start)
    return (min(times[136]) - min(times[8])) / 128, *counts


def force(kind):
    original = gpu3d._state
    gpu3d._state = lambda choice=None: original(kind if (choice or "default") == "default" else choice)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--adapter", default="default")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()
    if args.adapter != "default":
        force(args.adapter)
    from nodebased import fluid_gpu_solver
    print("adapter:", fluid_gpu_solver.adapter_name() if not args.cpu_only else "-", flush=True)
    print(f"{'backend':8}{'grid':>6}{'majorant':>10}{'tentative':>11}{'real':>8}{'null':>9}{'hops':>8}{'ms/sample':>11}", flush=True)
    for n in args.sizes:
        scene, camera, settings, fill = tenth_plume(n)
        print(f"-- {n} cubed, {fill * 100:.1f}% of the box holds density", flush=True)
        for mode in ("box", "fine", "grid"):
            setup(mode)
            if not args.gpu_only:
                seconds, tentative, real, hops = counts_cpu(scene, camera, settings, 24)
                print(f"{'cpu':8}{n:>6}{mode:>10}{tentative:>11.1f}{real:>8.2f}{tentative - real:>9.1f}{hops:>8.1f}{seconds * 1000:>11.1f}", flush=True)
            if not args.cpu_only:
                seconds, tentative, real, hops = gpu_row(scene, camera, settings)
                print(f"{'gpu':8}{n:>6}{mode:>10}{tentative:>11.1f}{real:>8.2f}{tentative - real:>9.1f}{hops:>8.1f}{seconds * 1000:>11.3f}", flush=True)


if __name__ == "__main__":
    main()
