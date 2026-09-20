"""Run with PYTHONPATH=. python tools/benchmark_3d_gpu_rt.py; no file I/O."""
import argparse
import math
import time
import numpy as np
from nodebased import gpu3d, gpurt, scene3d as s
from nodebased.raytrace import Bvh, TriangleSet


def make_scene(target):
    # Sphere latitude/longitude tessellation gives approximately target triangles.
    segments = max(4, math.ceil(math.sqrt(target)))
    sphere = s._sphere(1, segments, (1, 1, 1, 1), s.Transform3D())
    ground = s._card(8, 8, (1, 1, 1, 1),
                     s.Transform3D(s.Vec3(0, -1.2, 0), s.Vec3(-90, 0, 0)))
    vertices = []
    for geometry in (sphere, ground):
        m = geometry.world_matrix()
        world = (m[:3, :3] @ geometry.vertices.T + m[:3, 3:4]).T
        vertices.append(world[geometry.triangles])
    v = np.concatenate(vertices)
    return TriangleSet(v[:, 0], v[:, 1]-v[:, 0], v[:, 2]-v[:, 0], 1)


def timed(label, count, query):
    start = time.perf_counter(); result = query(); elapsed = time.perf_counter()-start
    print(f'  {label}: {count/elapsed:,.0f} rays/s, {elapsed:.4f} s measured', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--cpu-rays', type=int, default=4096)
    args = parser.parse_args()
    if min(args.width, args.height, args.cpu_rays) < 1:
        parser.error('dimensions and CPU subset must be positive')
    state = gpu3d._state(); print('Adapter:', state['info'], flush=True)
    reason = gpurt.check_capability(state)
    if reason:
        raise RuntimeError(reason)
    o, d, lo, hi = gpurt.primary_rays(s.Camera(), args.width, args.height)
    indices = np.linspace(0, len(o)-1, min(args.cpu_rays, len(o)), dtype=int)
    print(f'{args.width}x{args.height}; GPU timings include ray upload/readback; scene upload excluded.\n'
          f'CPU uses {len(indices):,} evenly sampled rays; throughput extrapolates that subset.', flush=True)
    for target in (1000, 10000, 100000):
        tri = make_scene(target); bvh = Bvh.build(*tri.aabbs())
        print(f'Target {target:,}; actual {len(tri.v0):,} triangles', flush=True)
        with gpurt.GpuTriangleScene(state, tri, bvh) as scene:
            gpurt.nearest_hits(scene, o[:64], d[:64], lo[:64], hi[:64], 1)
            timed('GPU closest', len(o), lambda: gpurt.nearest_hits(scene, o, d, lo, hi, 1))
            timed('GPU nearest K=8', len(o), lambda: gpurt.nearest_hits(scene, o, d, lo, hi, 8))
            timed('GPU peeled K=8 (all surfaces)', len(o), lambda: gpurt.all_hits(scene, o, d, lo, hi, k=8))
        timed('CPU closest (subset, scaled throughput)', len(indices), lambda:
              tri.nearest_hits(bvh, o[indices], d[indices], lo[indices], hi[indices], 1))


if __name__ == '__main__':
    main()
