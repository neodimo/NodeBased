import unittest
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nodebased import flip3d, scene3d
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.simcache import SimCache
from tests.test_particles_nodes import make, wire


class FluidSurfaceCacheTests(unittest.TestCase):
    def test_mesh_state_roundtrip_preserves_geometry_attributes(self):
        geometry = scene3d.Geometry(
            vertices=np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            triangles=np.array(((0, 1, 2),), np.int32), color=(0.1, 0.2, 0.3, 1.0),
            normals=np.tile((0, 0, 1), (3, 1)).astype(np.float32),
            velocities=np.ones((3, 3), np.float32))

        restored = flip3d.surface_from_state(flip3d.surface_to_state(geometry))

        np.testing.assert_array_equal(restored.vertices, geometry.vertices)
        np.testing.assert_array_equal(restored.triangles, geometry.triangles)
        np.testing.assert_array_equal(restored.normals, geometry.normals)
        np.testing.assert_array_equal(restored.velocities, geometry.velocities)
        self.assertEqual(restored.color, geometry.color)

    def test_half_resolution_signed_distance_uses_half_voxel_dimensions(self):
        stream = SimpleNamespace(shape=(8, 8, 8), backend="cpu", origin=(0, 0, 0),
                                 voxel=0.1, spacing=0.1)
        positions = np.array(((0.4, 0.4, 0.4), (0.5, 0.4, 0.4)), np.float32)

        full, full_voxel = flip3d.signed_distance(stream, positions, resolution=1, radius=0.1)
        half, half_voxel = flip3d.signed_distance(stream, positions, resolution=0.5, radius=0.1)

        self.assertEqual(full.shape, (8, 8, 8))
        self.assertEqual(half.shape, (4, 4, 4))
        self.assertEqual(half_voxel, full_voxel * 2)

    def test_whitewater_reuses_the_upstream_particle_cache_for_liquid_frames(self):
        dispatcher = Dispatcher()
        grid = {"division_size": 0.25, "bounds_min_x": -1.0, "bounds_min_y": 0.0,
                "bounds_min_z": -1.0, "bounds_max_x": 1.0, "bounds_max_y": 2.0,
                "bounds_max_z": 1.0, "max_size": 8, "substeps": 1}
        source = {"fluid_type": "liquid", "src_center_y": 1.0, "src_radius": 0.45, "end_frame": 2}
        make(dispatcher, src=("FluidSource3D", source), solver=("FluidLiquidSolver3D", grid),
             cache=("ParticleCache3D", {"cache_memory_mb": 1024, "cache_disk_mb": 8192}),
             white=("FluidWhitewater3D", {"cache_memory_mb": 256, "cache_disk_mb": 2048}))
        wire(dispatcher, "solver", "fluid", "src")
        wire(dispatcher, "cache", "particles", "solver")
        wire(dispatcher, "white", "particles", "cache")

        with tempfile.TemporaryDirectory() as root:
            evaluator = Evaluator(sim=SimCache(root=root, memory_budget=64 << 20, disk_budget=2 << 30))
            original = flip3d.solve_frame
            used_stores = []

            def record_store(stream, frame, cache, cancel=None):
                used_stores.append(cache)
                return original(stream, frame, cache, cancel)

            with mock.patch.object(flip3d, "solve_frame", side_effect=record_store):
                evaluator.evaluate_raster(dispatcher.document, "white", frame=2, typed=True)

            self.assertTrue(used_stores)
            upstream_store = evaluator._sim_stores[(1024, 8192)]
            whitewater_store = evaluator._sim_stores[(256, 2048)]
            self.assertTrue(all(cache is upstream_store for cache in used_stores))
            self.assertIsNot(upstream_store, whitewater_store)


if __name__ == "__main__":
    unittest.main()
