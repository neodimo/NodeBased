import unittest
from types import SimpleNamespace

import numpy as np

from nodebased import flip3d, scene3d
from nodebased.scene3d import ParticleInstance


def world_bounds(volume):
    shape = np.asarray(volume.density.shape, dtype=np.float64)
    corners = np.array([[x, y, z, 1.0] for x in (0.0, shape[0] * volume.voxel_size)
                        for y in (0.0, shape[1] * volume.voxel_size)
                        for z in (0.0, shape[2] * volume.voxel_size)])
    origin = np.eye(4)
    origin[:3, 3] = volume.origin
    points = (np.asarray(volume.matrix) @ origin @ corners.T).T[:, :3]
    return points.min(axis=0), points.max(axis=0)


class SceneProxyTests(unittest.TestCase):
    def setUp(self):
        density = np.arange(5 * 7 * 9, dtype=np.float32).reshape((5, 7, 9)) / 20
        matrix = np.eye(4)
        matrix[:3, 3] = (1.5, -2.0, 0.25)
        self.volume = scene3d.Volume(density, voxel_size=0.125, origin=(-0.5, 0.0, 1.0),
                                     matrix=matrix, temperature=density * 2,
                                     velocity=np.repeat(density[..., None], 3, axis=-1))
        self.scene = scene3d.Scene(volumes=(self.volume,))

    def test_preview_reduces_cached_volume_and_preserves_bounds_and_total_density(self):
        preview = scene3d.proxy_scene(self.scene, 4)
        coarse = preview.volumes[0]
        self.assertLess(np.prod(coarse.density.shape), np.prod(self.volume.density.shape))
        np.testing.assert_allclose(world_bounds(coarse), world_bounds(self.volume), atol=1e-6, rtol=0)
        total = lambda v: float(np.sum(v.density, dtype=np.float64) * v.voxel_size ** 3 *
                                abs(np.linalg.det(v.matrix[:3, :3])))
        self.assertAlmostEqual(total(coarse), total(self.volume), delta=abs(total(self.volume)) * 1e-6)
        self.assertEqual(coarse.temperature.shape, coarse.density.shape)
        self.assertEqual(coarse.velocity.shape, coarse.density.shape + (3,))

    def test_full_detail_keeps_original_scene_and_volume(self):
        self.assertIs(scene3d.proxy_scene(self.scene, 1), self.scene)

    def test_liquid_preview_thins_particles_without_changing_silhouette_bounds(self):
        axis = np.arange(12, dtype=np.float32) * 0.1
        positions = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        instance = ParticleInstance(positions=positions, sizes=np.ones(len(positions), np.float32),
                                    colors=np.ones((len(positions), 4), np.float32),
                                    stream=SimpleNamespace(spacing=0.1, origin=(0.0, 0.0, 0.0)))
        preview = flip3d.proxy_instance(instance, 4)
        self.assertLess(len(preview.positions), len(instance.positions) // 8)
        np.testing.assert_allclose(preview.positions.min(axis=0), positions.min(axis=0))
        np.testing.assert_allclose(preview.positions.max(axis=0), positions.max(axis=0))
