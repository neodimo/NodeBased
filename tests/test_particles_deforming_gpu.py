"""Lane 6 K1: GPU self-collision and vertex-deforming particle/fluid colliders."""
import unittest
from types import SimpleNamespace

import numpy as np

from nodebased import fluid3d, particles


class DeformingParticleColliderTests(unittest.TestCase):
    def test_a_bending_triangle_pushes_a_resting_particle_with_vertex_speed(self):
        from nodebased.scene3d import Geometry

        vertices = np.array([[-2, 0, -2], [2, 0, -2], [0, 0, 2]], np.float32)
        triangles = np.array([[0, 1, 2]], np.int32)

        def provider(frame):
            moved = vertices.copy()
            # The mesh bends upward around its left edge; the triangle's vertices have distinct
            # speeds while the object's world matrix stays fixed.
            moved[1:, 1] = float(frame - 1)
            return Geometry(moved, triangles, (1.0, 1.0, 1.0, 1.0))

        force = particles.ParticleCollider("ParticleBounce3D", {
            "animated": 1, "bounce": 0.9, "friction": 0.0, "probability": 1.0,
            "from_frame": -100, "to_frame": 100, "seed": 0, "kill_on_collision": 0,
        }, track=particles.GeometryTrack(provider, True, 1))
        t, normal, surface = force.first_hit(np.array([[0.0, 0.5, 0.0]]),
                                              np.zeros((1, 3)), 1, 0, 1)
        self.assertAlmostEqual(float(t[0]), 0.6875, delta=0.03)
        self.assertGreater(float(surface[0, 1]), 0.5)
        position, velocity, _ = particles.collide([force], 1, 0, 1, np.array([1]),
                                                   np.array([[0.0, 0.5, 0.0]]),
                                                   np.zeros((1, 3), np.float32),
                                                   np.array([[0.0, 0.5, 0.0]], np.float32), 1.0)
        self.assertGreater(float(velocity[0, 1]), 0.5)


class DeformingFluidColliderTests(unittest.TestCase):
    def test_fluid_collider_tracks_per_triangle_vertex_motion(self):
        triangle = np.array([[[1, 2, 1], [5, 2, 1], [3, 2, 5]]], np.float64)

        def provider(frame):
            result = triangle.copy()
            result[:, :, 1] += frame - 1
            return result

        track = fluid3d.GeometryTrack(provider, animated=True, start_frame=1)
        collider = fluid3d.Collider(track, animated=True)
        solver = SimpleNamespace(shape=(8, 8, 8), origin=np.zeros(3), voxel=1.0)
        solid, velocity = collider.mask(solver, 1, 0, 1)
        self.assertTrue(solid.any())
        self.assertGreater(float(velocity[solid][:, 1].mean()), 0.9)




if __name__ == "__main__":
    unittest.main()
