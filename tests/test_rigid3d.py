import unittest
import numpy as np
from nodebased.rigid3d import RigidBody3D, RigidSolver3D, liquid_reaction
from nodebased.scene3d import ParticleInstance


class RigidSolverReferenceTests(unittest.TestCase):
    def test_box_falls_to_floor_and_sleeps_at_half_height(self):
        body = RigidBody3D(size=(1, 1, 1), position=(0, 3, 0), restitution=0.0)
        solver = RigidSolver3D([body], substeps=8, sleep_frames=0.25)
        frame = solver.solve_frame(240)
        self.assertAlmostEqual(frame[0, 1], 0.5, places=6)
        self.assertTrue(body.sleeping)
        self.assertTrue(np.allclose(frame[0, 3:6], 0))

    def test_five_box_stack_stays_up_for_two_hundred_frames(self):
        bodies = [RigidBody3D(position=(0, 0.5 + i, 0), density=1000) for i in range(5)]
        solver = RigidSolver3D(bodies, substeps=6, iterations=12)
        result = solver.solve_frame(200)
        self.assertTrue(np.allclose(result[:, 1], np.arange(5) + 0.5, atol=0.03))
        self.assertTrue(np.all(np.abs(result[:, 4]) < 0.02))

    def test_two_sphere_collision_conserves_linear_momentum(self):
        a = RigidBody3D(shape="sphere", size=(1, 1, 1), position=(-1, 2, 0),
                        velocity=(2, 0, 0), density=1000, restitution=1)
        b = RigidBody3D(shape="sphere", size=(1, 1, 1), position=(1, 2, 0),
                        velocity=(-1, 0, 0), density=1000, restitution=1)
        initial = a.effective_mass * a.velocity + b.effective_mass * b.velocity
        solver = RigidSolver3D([a, b], gravity=(0, 0, 0), floor_y=None, substeps=12)
        solver.solve_frame(20)
        final = a.effective_mass * a.velocity + b.effective_mass * b.velocity
        self.assertLess(np.linalg.norm(final - initial) / np.linalg.norm(initial), 0.01)

    def test_incremental_frames_match_a_straight_cached_solve(self):
        first = RigidSolver3D([RigidBody3D(position=(0, 2, 0))], substeps=4)
        first.solve_frame(10)
        incremental = first.solve_frame(20)
        straight = RigidSolver3D([RigidBody3D(position=(0, 2, 0))], substeps=4).solve_frame(20)
        self.assertTrue(np.array_equal(incremental, straight))
        cached = first.solve_frame(15)
        target = RigidSolver3D([RigidBody3D(position=(0, 2, 0))], substeps=4).solve_frame(15)
        self.assertTrue(np.array_equal(cached, target))

    def test_box_floats_at_the_density_predicted_submerged_fraction(self):
        body = RigidBody3D(position=(0, 0.6, 0), density=500)
        solver = RigidSolver3D([body], floor_y=None, substeps=8, liquid_surface_y=1.0, liquid_density=1000)
        solver.solve_frame(240)
        submerged_depth = 1.0 - (body.position[1] - 0.5)
        self.assertAlmostEqual(submerged_depth, 0.5, delta=0.025)

    def test_body_transfers_a_bounded_reaction_to_overlapping_liquid_particles(self):
        p = np.array([[0, 0, 0], [2, 0, 0]], np.float32)
        zeros = np.zeros((2, 3), np.float32)
        liquid = ParticleInstance(positions=p, sizes=np.ones(2, np.float32),
                                  colors=np.ones((2, 4), np.float32), velocities=zeros)
        body = RigidBody3D(size=(1, 1, 1), position=(0, 0, 0))
        reacted = liquid_reaction(liquid, [body])
        self.assertLess(float(reacted.velocities[0, 1]), 0)
        self.assertEqual(float(reacted.velocities[1, 1]), 0)

    def test_body_mass_uses_density_and_shape_volume(self):
        box = RigidBody3D(size=(2, 3, 4), density=2)
        sphere = RigidBody3D(shape="sphere", size=(2, 2, 2), density=3)
        self.assertAlmostEqual(box.effective_mass, 48)
        self.assertAlmostEqual(sphere.effective_mass, 4 * np.pi)

    def test_constant_torque_changes_angular_velocity_and_orientation(self):
        body = RigidBody3D(position=(0, 2, 0), torque=(0, 0, 166.6666667))
        solver = RigidSolver3D([body], gravity=(0, 0, 0), floor_y=None, fps=24, substeps=4)
        solver.solve_frame(25)
        self.assertGreater(body.angular_velocity[2], 0.8)
        self.assertGreater(body.rotation[2], 20.0)

    def test_convex_hull_drops_interior_mesh_vertices_and_keeps_density_volume(self):
        from nodebased.rigid3d import convex_hull_triangles
        cube = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        points, faces = convex_hull_triangles(np.vstack((cube, (0, 0, 0))))
        body = RigidBody3D(shape="convex", size=(2, 2, 2), density=3,
                           collision_parts=((points, faces),))
        self.assertEqual(len(points), 8)
        self.assertAlmostEqual(body.volume, 8.0)
        self.assertAlmostEqual(body.effective_mass, 24.0)


if __name__ == "__main__":
    unittest.main()
