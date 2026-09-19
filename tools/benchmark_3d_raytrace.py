"""Reproduce CPU shadow budget calibration (run with PYTHONPATH=. from repo root)."""
import time

import numpy as np

from nodebased import scene3d as s
from nodebased.raytrace import Bvh, TriangleSet


def main():
    x, z = np.meshgrid(np.linspace(-3, 3, 960), np.linspace(-3, 3, 540))
    origins = np.column_stack((x.ravel(), np.full(x.size, -1.19), z.ravel())).astype('f4')
    direction = np.array([.25, 1, .15], dtype='f4') / np.linalg.norm([.25, 1, .15])
    dirs = np.broadcast_to(direction, origins.shape)
    for segments in (100, 316):
        sphere = s._sphere(1, segments, (1, 1, 1, .25), s.Transform3D())
        ground = s._card(8, 8, (1, 1, 1, 1),
                         s.Transform3D(s.Vec3(0, -1.2, 0), s.Vec3(-90, 0, 0)))
        triangles, alphas = [], []
        for geometry in (sphere, ground):
            matrix = geometry.world_matrix()
            world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
            triangles.append(world[geometry.triangles])
            alphas.extend([geometry.color[3]] * len(geometry.triangles))
        vertices = np.concatenate(triangles)
        primitives = TriangleSet(vertices[:, 0], vertices[:, 1]-vertices[:, 0],
                                 vertices[:, 2]-vertices[:, 0], alphas)
        start = time.perf_counter()
        bvh = Bvh.build(*primitives.aabbs())
        build = time.perf_counter()-start
        stats = {}
        start = time.perf_counter()
        primitives.transmittance(bvh, origins, dirs, .0001, np.inf, stats=stats)
        query = time.perf_counter()-start
        # Calibrate equivalents with the same kernel on 1/512 of the rays.
        start = time.perf_counter()
        primitives.brute_transmittance(origins[::512], dirs[::512], .0001, np.inf)
        brute = time.perf_counter()-start
        count = len(vertices)
        brute_tests_per_second = len(origins[::512])*count/brute
        print(dict(triangles=count, build_ms=build*1000, query_ms=query*1000,
                   rays_per_second=len(origins)/query, stats=stats,
                   c1=query*brute_tests_per_second/(len(origins)*np.log2(count+2)),
                   build_coefficient=build*brute_tests_per_second/(count*np.log2(count+2))),
              flush=True)


if __name__ == '__main__':
    main()
