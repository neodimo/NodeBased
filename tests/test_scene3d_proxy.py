import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nodebased import flip3d, fluid3d, scene3d, simcache
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
        self.assertEqual(coarse.proxy_tier, 4)
        self.assertIs(scene3d.proxy_scene(scene3d.Scene(volumes=(coarse,)), 4).volumes[0], coarse)

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

    def test_preview_particle_state_is_a_compact_bounds_preserving_simcache_state(self):
        axis = np.arange(12, dtype=np.float32) * 0.1
        positions = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        state = simcache.State({"position": positions, "velocity": positions * 2,
                                "id": np.arange(len(positions), dtype=np.int64)}, {"tag": "kept"})
        stream = SimpleNamespace(spacing=0.1, origin=(0.0, 0.0, 0.0))
        proxy = flip3d.preview_state(state, stream, 4)
        self.assertLess(len(proxy.arrays["position"]), len(positions) // 8)
        np.testing.assert_allclose(proxy.arrays["position"].min(axis=0), positions.min(axis=0))
        np.testing.assert_allclose(proxy.arrays["position"].max(axis=0), positions.max(axis=0))
        self.assertEqual(proxy.meta["tag"], "kept")
        np.testing.assert_array_equal(proxy.arrays["velocity"], proxy.arrays["position"] * 2)
        self.assertIn("radius", proxy.arrays)

    def test_preview_particle_state_has_a_fixed_point_budget(self):
        count = flip3d.PREVIEW_POINT_BUDGET * 2
        positions = np.column_stack((np.arange(count), np.arange(count) % 19, np.arange(count) % 31)).astype(np.float32)
        state = simcache.State({"position": positions, "size": np.ones(count, np.float32)}, {})
        proxy = flip3d.preview_state(state, SimpleNamespace(spacing=1.0, origin=(0, 0, 0)), 4)
        self.assertLessEqual(len(proxy.arrays["position"]), flip3d.PREVIEW_POINT_BUDGET)
        self.assertEqual(len(proxy.arrays["radius"]), len(proxy.arrays["position"]))
        np.testing.assert_array_equal(proxy.arrays["position"].min(axis=0), positions.min(axis=0))
        np.testing.assert_array_equal(proxy.arrays["position"].max(axis=0), positions.max(axis=0))

    def test_cached_volume_prepopulates_and_reuses_the_proxy_side_cache(self):
        density = np.arange(8 * 8 * 8, dtype=np.float32).reshape((8, 8, 8)) / 100
        stream = SimpleNamespace(run="volume-run", backend="cpu", shape=density.shape,
                                 origin=(-0.5, 0.25, 1.0), voxel=0.125)
        store = simcache.SimCache()
        preview_store = simcache.SimCache()
        out_run = simcache.run_key(stream.run, {"out": ["float32", "density"]})
        store.put(out_run, 3, simcache.State({"density": density},
                                              {"domain_shape": density.shape, "domain_origin": stream.origin}))
        full = fluid3d.cached_volume(stream, 3, store, None, "float32", "density", preview_store=preview_store)
        preview = fluid3d.cached_volume(stream, 3, store, None, "float32", "density", tier=4,
                                        preview_store=preview_store)
        self.assertEqual(preview.proxy_tier, 4)
        self.assertLessEqual(np.prod(preview.density.shape), fluid3d.PREVIEW_VOLUME_EDGE ** 3)
        np.testing.assert_allclose(world_bounds(preview), world_bounds(full), atol=1e-6, rtol=0)
        total = lambda v: float(np.sum(v.density, dtype=np.float64) * v.voxel_size ** 3 *
                                abs(np.linalg.det(v.matrix[:3, :3])))
        self.assertAlmostEqual(total(preview), total(full), delta=abs(total(full)) * 1e-6)

    def test_fixed_preview_artifact_reloads_without_full_cache_access(self):
        density = np.zeros((48, 48, 48), np.float32)
        density[7:19, 9:24, 11:27] = np.linspace(0.1, 2, 12 * 15 * 16, dtype=np.float32).reshape(12, 15, 16)
        stream = SimpleNamespace(run="fixed-preview-run", backend="cpu", shape=density.shape,
                                 origin=(-0.5, 0.25, 1.0), voxel=0.125)
        with tempfile.TemporaryDirectory() as tmp:
            full_store = simcache.SimCache(root=Path(tmp) / "full", memory_budget=0, disk_budget=32 << 20)
            preview_store = simcache.SimCache(root=Path(tmp) / "preview", memory_budget=0, disk_budget=8 << 20)
            out_run = simcache.run_key(stream.run, {"out": ["float32", "density"]})
            full_store.put(out_run, 3, simcache.State({"density": density, "temperature": density * 3.0},
                                                       {"domain_shape": density.shape, "domain_origin": stream.origin}))
            full = fluid3d.cached_volume(stream, 3, full_store, None, "float32", "density",
                                         preview_store=preview_store)
            artifact_run = simcache.run_key(out_run, {"preview_artifact": 1,
                                                       "volume_edge": fluid3d.PREVIEW_VOLUME_EDGE})
            artifact = preview_store.get(artifact_run, 3)
            self.assertIsNotNone(artifact)
            self.assertEqual(artifact.arrays["density_u8"].dtype, np.uint8)
            self.assertEqual(artifact.arrays["density_u8"].shape, (48, 48, 48))
            self.assertTrue(artifact.meta["preview_artifact"])
            preview_store._memory.clear()
            with patch.object(full_store, "get", side_effect=AssertionError("preview touched full cache")):
                preview = fluid3d.cached_volume(stream, 3, full_store, None, "float32", "density", tier=4,
                                                preview_store=preview_store)
            self.assertEqual(preview.density.shape, (48, 48, 48))
            self.assertEqual(preview.temperature.shape, (48, 48, 48))
            self.assertLess(float(np.max(np.abs(preview.temperature - preview.density * 3.0))), 0.1)
            np.testing.assert_allclose(world_bounds(preview), world_bounds(full), atol=1e-5, rtol=0)
            total = lambda v: float(np.sum(v.density, dtype=np.float64) * v.voxel_size ** 3 *
                                    abs(np.linalg.det(v.matrix[:3, :3])))
            self.assertAlmostEqual(total(preview), total(full), delta=abs(total(full)) * 0.01)
