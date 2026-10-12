import unittest
from types import SimpleNamespace

import numpy as np

from nodebased import flip3d, scene3d


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


if __name__ == "__main__":
    unittest.main()
