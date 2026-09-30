import unittest
import numpy as np
from nodebased.rigid3d import RigidBody3D, RigidSolver3D


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

    def test_body_mass_uses_density_and_shape_volume(self):
        box = RigidBody3D(size=(2, 3, 4), density=2)
        sphere = RigidBody3D(shape="sphere", size=(2, 2, 2), density=3)
        self.assertAlmostEqual(box.effective_mass, 48)
        self.assertAlmostEqual(sphere.effective_mass, 4 * np.pi)


if __name__ == "__main__":
    unittest.main()
