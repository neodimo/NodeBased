"""Lane 6 step N2: the GPU path tracer reads a sparse volume's tiles straight from its buffer, never a dense grid.

A sparse upload is compared with the dense upload of the same grid on the same seed, and the GPU with the CPU
reference. Run the module on the other adapters with `scratch/nb-lanes/run/force-adapter.py integrated|cpu`.
"""
import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace as pt, ptvolume, scene3d as s, volumerender as vr
from nodebased.sparsevol import SparseField, SparseGrid
from tests.test_3d_pathtrace_gpu_soft import READY, FLOOR, SUN, CAMERA
from tools import benchmark_sparse_gpu as bench

SMOKE = vr.VolumeSettings(absorption=0.2, scattering=0.9, step_size=0.05, shadow_steps=12, density_scale=6.0)


def blob(nx=40, ny=32, nz=24, fire=True, rest=0.0):
    """A turned, off-centre smoke blob on an odd-sized grid that is empty outside a radius, so whole tiles stay empty."""
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    r2 = ((x - 14) / 7.0) ** 2 + ((y - 20) / 6.0) ** 2 + ((z - 12) / 8.0) ** 2
    density = np.where(r2 < 1.0, 2.5 * (1.0 - r2), rest).astype(np.float32)
    fields = {"density": density}
    if fire:
        fields["temperature"] = np.where(r2 < 0.6, 1.0 + 0.5 * y / ny, 0.0).astype(np.float32)
    matrix = np.eye(4)
    c, sn = math.cos(0.6), math.sin(0.6)
    matrix[:3, :3] = ((c, 0, sn), (0, 1, 0), (-sn, 0, c))
    matrix[:3, 3] = (0.2, 0.1, -0.1)
    grid = SparseGrid.from_dense(fields, rest={"density": rest})
    kwargs = dict(voxel_size=0.075, origin=(-1.0, -0.7, -0.9), matrix=matrix)
    return s.Volume(**grid.to_dense(), **kwargs), s.Volume.from_sparse(grid, **kwargs)


class never_dense:
    def __enter__(self):
        error = AssertionError("a sparse volume was expanded to a dense grid")
        self.patches = [patch.object(SparseGrid, "to_dense", side_effect=error),
                        patch.object(SparseField, "__array__", side_effect=error)]
        for p in self.patches:
            p.start()

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def trace(scene, backend="gpu", size=(24, 24), samples=64, seed=5, **settings):
    settings.setdefault("max_bounces", 4)
    settings.setdefault("diffuse_bounces", 4)
    return pt.render(scene, CAMERA, *size, ambient=0.0, volume=SMOKE, backend=backend,
                     settings=pt.PathSettings(samples=samples, seed=seed, **settings))


class Majorants(unittest.TestCase):
    def test_the_coarse_bounds_of_stored_tiles_equal_the_dense_bounds(self):
        rng = np.random.default_rng(7)
        for shape, rest in (((37, 29, 21), 0.0), ((50, 33, 70), 0.0), ((33, 40, 17), 0.25)):
            density = np.full(shape, rest, np.float32)
            for _ in range(6):
                lo = [int(rng.integers(0, n - 4)) for n in shape]
                hi = [min(n, l + int(rng.integers(3, 22))) for n, l in zip(shape, lo)]
                density[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = rng.random([h - l for l, h in zip(lo, hi)]) + 0.1
            grid = SparseGrid.from_dense({"density": density}, rest={"density": rest})
            np.testing.assert_array_equal(
                ptvolume.coarse_majorants_sparse(grid, "density", 3.5),
                ptvolume.coarse_majorants(grid.to_dense()["density"], 3.5), err_msg=str(shape))


@unittest.skipUnless(READY, "no wgpu adapter cleared for splats and smoke in the path tracer")
class SparseOnTheGpuMatchesDense(unittest.TestCase):
    def scene(self, volume, **kwargs):
        return s.Scene((FLOOR,), lights=(SUN,), volumes=(volume,), **kwargs)

    def test_a_sun_lit_blob_traces_the_same_picture_from_tiles_and_from_the_dense_grid(self):
        dense, sparse = blob()
        self.assertLess(sparse.sparse.tile_count, 0.5 * 5 * 4 * 3 + 3)
        expected = trace(self.scene(dense))
        with never_dense():
            actual = trace(self.scene(sparse))
        self.assertGreater(float(expected[..., 3].max()), 0.3)
        np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_fire_in_a_sparse_volume_glows_like_the_dense_one(self):
        dense, sparse = blob()
        settings = vr.VolumeSettings(absorption=0.2, scattering=0.9, step_size=0.05, shadow_steps=12, fire_intensity=3.0,
                                     temperature_scale=900.0, fire_threshold=500.0, density_scale=6.0)
        scene_a, scene_b = (s.Scene(volumes=(v,)) for v in (dense, sparse))
        a = pt.render(scene_a, CAMERA, 24, 24, volume=settings, backend="gpu", settings=pt.PathSettings(samples=48, seed=3))
        with never_dense():
            b = pt.render(scene_b, CAMERA, 24, 24, volume=settings, backend="gpu", settings=pt.PathSettings(samples=48, seed=3))
        self.assertGreater(float(a[..., :3].max()), 0.05, "the fire must show")
        np.testing.assert_allclose(b, a, atol=2e-4)

    def test_the_gpu_trace_of_tiles_agrees_with_the_cpu_reference_of_the_same_tiles(self):
        _dense, sparse = blob(fire=False)
        scene = s.Scene(lights=(SUN,), volumes=(sparse,))
        cpu = trace(scene, "cpu", (16, 16), samples=96, seed=5)
        with never_dense():
            gpu = trace(scene, "gpu", (16, 16), samples=1536, seed=9)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(gpu[..., 3].mean()), float(cpu[..., 3].mean()), delta=0.01)
        np.testing.assert_allclose(gpu[..., 3], cpu[..., 3], atol=0.2)

    def test_a_nonzero_rest_value_reads_back_in_the_empty_tiles(self):
        dense, sparse = blob(fire=False, rest=0.15)
        self.assertLess(sparse.sparse.tile_count, 60)
        expected = trace(self.scene(dense))
        with never_dense():
            actual = trace(self.scene(sparse))
        np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_a_dense_and_a_sparse_volume_share_a_scene(self):
        dense, sparse = blob()
        other = s.analytic_plume(16, 2)
        scene = s.Scene(lights=(SUN,), volumes=(sparse, other))
        reference = s.Scene(lights=(SUN,), volumes=(dense, other))
        np.testing.assert_allclose(trace(scene), trace(reference), atol=2e-4)

    def test_an_all_empty_tile_set_packs_no_tile_and_traces_nothing(self):
        grid = SparseGrid.from_dense({"density": np.zeros((32, 32, 32), np.float32)})
        empty = s.Volume.from_sparse(grid, voxel_size=0.1, origin=(-1.6, -1.6, -1.6))
        scene = s.Scene((FLOOR,), lights=(SUN,), volumes=(empty,))
        with never_dense():
            image = trace(scene, pass_samples=2)       # the same samples per dispatch as the bare scene: the sums then add in the same order
        np.testing.assert_allclose(image, trace(s.Scene((FLOOR,), lights=(SUN,)), pass_samples=2), atol=0)
        packed = gpupathtrace.pack(pt.build_scene(scene, 0.0, volume=SMOKE))
        bare = gpupathtrace.pack(pt.build_scene(s.Scene((FLOOR,), lights=(SUN,)), 0.0, volume=SMOKE))
        # the header, the one-cell tile table and the coarse bounds: no density value at all
        self.assertLess(packed.env.nbytes - bare.env.nbytes, 8 * 1024)
        self.assertTrue(packed.sparse_volumes)

    def test_a_growing_tile_set_is_packed_afresh_for_every_frame(self):
        n = 32
        axis = (np.arange(8) + 0.5) / 8.0 - 0.5
        blob_tile = np.exp(-4.0 * (axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)).astype(np.float32)
        order = [tuple(c) for c in np.argwhere(np.ones((4, 4, 4), bool))]
        order.sort(key=lambda c: float(np.linalg.norm(np.array(c) - np.array((1.5, 1.5, 1.5)))))
        kwargs = dict(voxel_size=0.07, origin=(-1.1, -1.1, -1.1))
        layouts = set()
        for count in (7, 8, 9, 20, 28, 9):
            density = np.zeros((n, n, n), np.float32)
            for index, (x, y, z) in enumerate(order[:count]):
                density[x * 8:x * 8 + 8, y * 8:y * 8 + 8, z * 8:z * 8 + 8] = blob_tile * (1.5 + (index % 3) * 0.5)
            grid = SparseGrid.from_dense({"density": density})
            layouts.add(grid.atlas_layout())
            with never_dense():
                actual = trace(s.Scene(lights=(SUN,), volumes=(s.Volume.from_sparse(grid, **kwargs),)))
            expected = trace(s.Scene(lights=(SUN,), volumes=(s.Volume(density, **kwargs),)))
            np.testing.assert_allclose(actual, expected, atol=2e-4, err_msg=str(count))
        self.assertGreaterEqual(len(layouts), 3)

    def test_the_sparse_buffer_is_a_fraction_of_the_dense_one(self):
        for size in (128, 256):
            dense_volume, sparse_volume = bench.volumes(size)
            sizes = []
            for volume in (dense_volume, sparse_volume):
                scene = s.Scene(volumes=(volume,))
                sizes.append(gpupathtrace.pack(pt.build_scene(scene, 0.0, volume=SMOKE)).env.nbytes)
            bare = gpupathtrace.pack(pt.build_scene(s.Scene(volumes=(s.analytic_plume(4),)), 0.0, volume=SMOKE)).env.nbytes
            self.assertLess(sizes[1] - bare, 0.25 * (sizes[0] - bare), size)

    def test_unsupported_tile_layouts_stay_on_the_cpu(self):
        dense, sparse = blob()
        mixed = s.Volume(sparse.density, voxel_size=sparse.voxel_size, origin=sparse.origin, matrix=sparse.matrix,
                         temperature=np.asarray(sparse.temperature), sparse=sparse.sparse)
        with self.assertRaisesRegex(ValueError, "same tiles"):
            trace(s.Scene(volumes=(mixed,)))
        stats = {}
        image = pt.render(s.Scene(volumes=(mixed,)), CAMERA, 8, 8, volume=SMOKE, backend="auto", stats=stats,
                          settings=pt.PathSettings(samples=4))
        self.assertEqual(stats["backend"], "cpu")
        self.assertEqual(image.shape, (8, 8, 4))


if __name__ == "__main__":
    unittest.main()
