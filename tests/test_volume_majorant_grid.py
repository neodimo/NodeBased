"""The path tracers' two-level volume majorant (lane 4 Rendering, step T3 part 1): a plume filling a tenth of its box costs
a few null collisions instead of over a hundred, the picture is the one the single-box majorant makes, and the GPU
counts the same collisions as the CPU reference."""
import unittest
from unittest import mock

import numpy as np

from nodebased import gpupathtrace, pathtrace as pt, ptvolume
from tests.test_3d_pathtrace_gpu_soft import READY
from tools.benchmark_volume_majorant import tenth_plume


def flight(scene, settings, rays=4000, skip=True, seed=11, spread=1.2):
    """Delta tracking of `rays` rays straight through the plume's box: (fraction that pass, tentative per ray, layer)."""
    with mock.patch.object(ptvolume, "ENABLE_SKIP", skip):
        layer = ptvolume.build(scene, settings)
        rng = np.random.default_rng(seed)
        o = np.column_stack((rng.uniform(-spread, spread, rays), rng.uniform(-spread, spread, rays), np.full(rays, 5.0)))
        d = np.tile((0.0, 0.0, -1.0), (rays, 1))
        keys = pt.path_key(np.arange(rays), 0, seed)
        t_event, _, _ = ptvolume.free_flight(layer, o, d, np.full(rays, np.inf), keys, 0)
    return float(np.isinf(t_event).mean()), layer.counts["tentative"] / rays, layer


class Bounds(unittest.TestCase):
    def test_each_cell_bounds_its_trilinear_halo_and_a_coarse_cell_bounds_its_fine_cells(self):
        scene, _, _, _ = tenth_plume(40)
        volume = scene.volumes[0]
        levels = ptvolume.majorant_levels(volume, 42.0, tile=8, ratio=2)
        scale = 42.0 / float(volume.density.max())
        density = volume.density
        for index in np.ndindex(levels.fine.shape):
            window = tuple(slice(max(0, c * 8 - 1), min(40, c * 8 + 9)) for c in index)
            self.assertGreaterEqual(float(levels.fine[index]) + 1e-6, float(density[window].max()) * scale)
        for index in np.ndindex(levels.coarse.shape):
            block = tuple(slice(c * 2, c * 2 + 2) for c in index)
            self.assertEqual(float(levels.coarse[index]), float(levels.fine[block].max()))

    def test_empty_space_is_exactly_zero_so_a_ray_can_stride_over_it(self):
        scene, _, _, _ = tenth_plume(64)
        levels = ptvolume.majorant_levels(scene.volumes[0], 10.0, tile=8, ratio=2)
        self.assertEqual(float(levels.fine[0, 0, 0]), 0.0)
        self.assertEqual(float(levels.coarse[0, 0, 0]), 0.0)
        self.assertGreater(float(levels.fine[4, 4, 4]), 0.0)

    def test_the_vectorised_window_maximum_equals_the_slice_by_slice_one(self):
        rng = np.random.default_rng(3)
        density = rng.random((37, 20, 51)).astype(np.float32) * (rng.random((37, 20, 51)) > .7)
        got = ptvolume.coarse_majorants(density, 5.0, tile=8)
        scale = 5.0 / float(density.max())
        for index in np.ndindex(got.shape):
            window = tuple(slice(max(0, c * 8 - 1), min(n, c * 8 + 9)) for c, n in zip(index, density.shape))
            self.assertAlmostEqual(float(got[index]), float(density[window].max()) * scale, delta=1e-5)


class CpuFreeFlight(unittest.TestCase):
    def test_the_grid_lets_the_same_light_through_with_far_fewer_null_collisions(self):
        scene, _, settings, fill = tenth_plume(64)
        self.assertAlmostEqual(fill, 0.1, delta=0.01)
        through_box, tentative_box, _ = flight(scene, settings, skip=False, spread=.5)
        through_grid, tentative_grid, layer = flight(scene, settings, skip=True, spread=.5)
        self.assertLess(abs(through_box - through_grid), 0.03)       # 4,000 rays: the noise on a fraction is about 0.008
        self.assertGreater(0.5, through_grid)                        # the plume is thick enough that it matters
        self.assertLess(tentative_grid * 2, tentative_box)           # rays through the thick part still mostly collide for real
        _, wide_box, _ = flight(scene, settings, skip=False)         # rays over the whole box, most of it empty
        _, wide_grid, _ = flight(scene, settings, skip=True)
        self.assertLess(wide_grid * 5, wide_box)
        counts = layer.counts
        self.assertGreater(counts["hops"], 0)
        self.assertLess(counts["real"], counts["tentative"])

    def test_the_coarse_level_saves_hops_in_empty_space(self):
        scene, _, settings, _ = tenth_plume(128)
        with mock.patch.object(ptvolume, "MAJORANT_TILE", 8):
            with mock.patch.object(ptvolume, "MAJORANT_RATIO", 1):
                _, _, one = flight(scene, settings, rays=1500)
            _, _, two = flight(scene, settings, rays=1500)
        self.assertLess(two.counts["hops"], one.counts["hops"] * 0.8)
        self.assertLess(abs(two.counts["tentative"] - one.counts["tentative"]), one.counts["tentative"] * 0.1 + 20)

    def test_the_picture_is_the_box_majorants_within_noise(self):
        scene, camera, settings, _ = tenth_plume(48)

        def render(skip, seed):
            with mock.patch.object(ptvolume, "ENABLE_SKIP", skip):
                return pt.render(scene, camera, 20, 20, ambient=.7, volume=settings, backend="cpu",
                                 settings=pt.PathSettings(samples=48, max_bounces=4, seed=seed))[..., :3]
        baseline, repeat, grid = render(False, 4), render(False, 5), render(True, 4)
        noise = float(np.abs(baseline - repeat).mean())
        self.assertGreater(noise, 0)
        self.assertLess(float(np.abs(baseline - grid).mean()), noise * 1.25)
        self.assertLess(abs(float(grid.mean() / baseline.mean()) - 1), 0.05)


@unittest.skipUnless(READY, "no wgpu adapter cleared for splats and smoke in the path tracer")
class GpuAgrees(unittest.TestCase):
    def counts(self, scene, camera, settings, skip):
        with mock.patch.object(gpupathtrace, "ENABLE_VOLUME_SKIP", skip), \
             mock.patch.object(gpupathtrace, "COUNT_COLLISIONS", True), \
             mock.patch.object(ptvolume, "ENABLE_SKIP", skip):
            image = pt.render(scene, camera, 48, 48, ambient=.7, volume=settings, backend="gpu",
                              settings=pt.PathSettings(samples=8, max_bounces=4, seed=4))
        return image[..., :3].reshape(-1, 3).mean(axis=0)

    def test_the_card_counts_the_collisions_the_reference_counts(self):
        scene, camera, settings, _ = tenth_plume(64)
        for skip in (False, True):
            tentative, real, hops = self.counts(scene, camera, settings, skip)
            with mock.patch.object(ptvolume, "ENABLE_SKIP", skip):
                layers = []
                original = pt.build_scene
                with mock.patch.object(pt, "build_scene", lambda *a, **k: layers.append(original(*a, **k)) or layers[-1]):
                    pt.render(scene, camera, 48, 48, ambient=.7, volume=settings, backend="cpu",
                              settings=pt.PathSettings(samples=8, max_bounces=4, seed=4))
            c, paths = layers[0].volumes.counts, 48 * 48 * 8
            self.assertAlmostEqual(tentative, c["tentative"] / paths, delta=0.15 * c["tentative"] / paths + 0.2)
            self.assertAlmostEqual(real, c["real"] / paths, delta=0.15 * c["real"] / paths + 0.05)
            if skip:
                self.assertGreater(hops, 0)
                self.assertLess(tentative, 25)
            else:
                self.assertGreater(tentative, 60)

    def test_the_card_makes_the_box_majorants_picture_within_noise(self):
        scene, camera, settings, _ = tenth_plume(96)

        def render(skip, seed):
            with mock.patch.object(gpupathtrace, "ENABLE_VOLUME_SKIP", skip), mock.patch.object(ptvolume, "ENABLE_SKIP", skip):
                return pt.render(scene, camera, 32, 32, ambient=.7, volume=settings, backend="gpu",
                                 settings=pt.PathSettings(samples=64, max_bounces=4, seed=seed))[..., :3]
        baseline, repeat, grid = render(False, 4), render(False, 5), render(True, 4)
        noise = float(np.abs(baseline - repeat).mean())
        self.assertGreater(noise, 0)
        self.assertLess(float(np.abs(baseline - grid).mean()), noise * 1.25)
        self.assertLess(abs(float(grid.mean() / baseline.mean()) - 1), 0.05)

    def test_the_card_and_the_reference_agree_on_the_plume(self):
        scene, camera, settings, _ = tenth_plume(64)
        config = pt.PathSettings(samples=128, max_bounces=4, seed=9)
        gpu = pt.render(scene, camera, 24, 24, ambient=.7, volume=settings, backend="gpu", settings=config)[..., :3]
        cpu = pt.render(scene, camera, 24, 24, ambient=.7, volume=settings, backend="cpu", settings=config)[..., :3]
        self.assertLess(abs(float(gpu.mean() / cpu.mean()) - 1), 0.04)


if __name__ == "__main__":
    unittest.main()
