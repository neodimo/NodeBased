"""Lane L4 step E, part 1: gpu3d.render draws scene.particles and matches the CPU reference."""
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, scene3d as s
from tests import gpu_precision

W = H = 160
# Exact-pixel checks: a half-float rgba target (Windows CI's Microsoft Basic Render Driver) quantises
# 0.2 to 0.19995, so the exact bound is loosened there only.
EXACT = 1e-3 if gpu_precision.half_float_target() else 1e-6
CAMERA = s.Camera()


def cloud(kind, n=150, seed=1, texture=None, size=(0.1, 0.5)):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.5, 1.5, (n, 3)).astype('f4')
    positions[:, 2] *= 0.5
    colors = rng.uniform(0.2, 1.0, (n, 4)).astype('f4')
    colors[:, 3] = rng.uniform(0.3, 1.0, n)
    colors[:, :3] *= colors[:, 3:]
    return s.ParticleInstance(positions, rng.uniform(*size, n).astype('f4'), colors, render_as=kind, texture=texture)


def wall(z, color=(0.2, 0.5, 0.9, 1.0)):
    corners = np.array([[-3, -3, z], [3, -3, z], [3, 3, z], [-3, 3, z]], 'f4')
    return s.Geometry(corners, np.array([[0, 1, 2], [0, 2, 3]], 'i4'), color)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUParticleParity(unittest.TestCase):
    def check(self, scene, tolerance=2e-3, **kwargs):
        expected = s.render(scene, CAMERA, W, H, **kwargs)
        actual = gpu3d.render(scene, CAMERA, W, H, **kwargs)
        self.assertEqual(actual.shape, (H, W, 4))
        self.assertEqual(actual.dtype, np.float32)
        covered = int((expected[..., 3] > 0).sum())
        self.assertGreater(covered, 1000, 'the comparison scene must actually draw particles')
        wrong = (np.abs(actual - expected).max(axis=-1) > 0.02).sum()
        self.assertLessEqual(int(wrong), max(2, covered // 200), f'{wrong} of {covered} pixels differ')
        self.assertLess(float(np.abs(actual - expected).mean()), tolerance)
        return actual

    def test_points_match_the_cpu_render(self):
        self.check(s.Scene(particles=(cloud('points'),)))

    def test_spheres_match_the_cpu_render(self):
        image = self.check(s.Scene(particles=(cloud('spheres', seed=2),)))
        self.assertGreater(float(image[..., :3].max()), 0.3)

    def test_textured_cards_match_the_cpu_render(self):
        texture = np.random.default_rng(5).uniform(0, 1, (8, 16, 4)).astype('f4')
        self.check(s.Scene(particles=(cloud('cards', seed=3, texture=texture),)))

    def test_untextured_cards_and_two_textured_sets_match(self):
        one = np.random.default_rng(6).uniform(0, 1, (4, 4, 4)).astype('f4')
        two = np.random.default_rng(7).uniform(0, 1, (3, 9, 4)).astype('f4')
        self.check(s.Scene(particles=(cloud('cards', 60, 4), cloud('cards', 60, 5, one), cloud('cards', 60, 6, two),
                                      cloud('spheres', 60, 7))))

    def test_particles_are_depth_tested_against_the_mesh(self):
        # A wall at z = 0 hides the particles behind it (z < 0) and is covered by the ones in front (z > 0).
        behind = s.ParticleInstance(np.array([[0, 0, -1.0]], 'f4'), np.array([1.0], 'f4'),
                                    np.array([[1, 0, 0, 1]], 'f4'), render_as='cards')
        front = s.ParticleInstance(np.array([[0, 0, 1.0]], 'f4'), np.array([0.3], 'f4'),
                                   np.array([[0, 1, 0, 1]], 'f4'), render_as='cards')
        image = self.check_wall(s.Scene((wall(0.0),), particles=(behind, front)))
        np.testing.assert_allclose(image[H // 2, W // 2], [0, 1, 0, 1], atol=EXACT)
        np.testing.assert_allclose(image[H // 2, W // 2 + 40], [0.2, 0.5, 0.9, 1.0], atol=EXACT)  # behind-particle is hidden

    def check_wall(self, scene):
        expected = s.render(scene, CAMERA, W, H)
        actual = gpu3d.render(scene, CAMERA, W, H)
        self.assertLess(float(np.abs(actual - expected).mean()), 2e-3)
        return actual

    def test_particles_only_and_mesh_scenes_and_empty_sets(self):
        empty = s.ParticleInstance(np.zeros((0, 3), 'f4'), np.zeros(0, 'f4'), np.zeros((0, 4), 'f4'))
        image = gpu3d.render(s.Scene(particles=(empty,)), CAMERA, 32, 32, background=(0.1, 0.2, 0.3, 1))
        np.testing.assert_allclose(image, np.broadcast_to([0.1, 0.2, 0.3, 1], (32, 32, 4)), atol=EXACT)
        self.check(s.Scene((wall(-2.0),), particles=(cloud('points', seed=9),)))

    def test_background_shows_through_translucent_particles_and_data_outputs_skip_particles(self):
        scene = s.Scene(particles=(cloud('spheres', seed=11),))
        self.check(scene, background=(0.3, 0.1, 0.2, 1.0))
        depth = gpu3d.render(scene, CAMERA, 32, 32, output='depth')
        self.assertEqual(float(depth[..., 3].max()), 0.0)

    def test_the_sizes_are_scaled_and_out_of_range_particles_are_skipped(self):
        big = s.ParticleInstance(np.array([[0, 0, 0.0], [0, 0, 60.0]], 'f4'), np.array([0.5, 0.5], 'f4'),
                                 np.ones((2, 4), 'f4'), render_as='points', size_scale=2.0)
        image = self.check(s.Scene(particles=(big,)))
        self.assertGreater(int((image[..., 3] > 0).sum()), 1000)

    def test_unsupported_combinations_still_fall_back_to_the_cpu(self):
        scene = s.Scene(particles=(cloud('points', 5),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, CAMERA, 32, 32, mode='raytrace')
        with patch.object(gpu3d, 'particle_data', side_effect=gpu3d.Unsupported('x')):
            with self.assertRaises(gpu3d.Unsupported):
                gpu3d.render(scene, CAMERA, 32, 32)


if __name__ == '__main__':
    unittest.main()
