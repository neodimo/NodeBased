"""Particle artist tools 4: viewport inspection (picking, the tooltip readout and "colour by
attribute")."""
import unittest

import numpy as np

from nodebased import particleinspect, scene3d


def _instance(positions, velocities=None, ages=None, lifetimes=None, ids=None, sizes=None, colors=None):
    n = len(positions)
    return scene3d.ParticleInstance(
        positions=np.asarray(positions, np.float32),
        sizes=np.asarray(sizes if sizes is not None else np.full(n, 0.1), np.float32),
        colors=np.asarray(colors if colors is not None else np.tile([1, 1, 1, 1], (n, 1)), np.float32),
        velocities=np.asarray(velocities if velocities is not None else np.zeros((n, 3)), np.float32),
        ages=np.asarray(ages if ages is not None else np.zeros(n), np.float32),
        lifetimes=np.asarray(lifetimes if lifetimes is not None else np.full(n, 10.0), np.float32),
        ids=np.asarray(ids if ids is not None else np.arange(n), np.int64))


def _straight_on_camera(distance=5.0):
    return scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, distance)),
                          target=scene3d.Vec3(0, 0, 0), fov=45.0)


class PickParticleTests(unittest.TestCase):
    def test_picks_the_particle_at_a_known_screen_position(self):
        camera = _straight_on_camera()
        positions = np.array([(0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (-2.0, -2.0, 0.0)])
        pixels, _ = scene3d.project(camera, 400, 300, positions)
        index = particleinspect.pick_particle(camera, 400, 300, positions,
                                              float(pixels[1, 0]), float(pixels[1, 1]))
        self.assertEqual(index, 1)

    def test_returns_none_outside_tolerance(self):
        camera = _straight_on_camera()
        positions = np.array([(0.0, 0.0, 0.0)])
        index = particleinspect.pick_particle(camera, 400, 300, positions, 0.0, 0.0, tolerance=1.0)
        self.assertIsNone(index)
        far_index = particleinspect.pick_particle(camera, 400, 300, positions, 399.0, 299.0, tolerance=1.0)
        self.assertIsNone(far_index)

    def test_empty_positions_returns_none(self):
        camera = _straight_on_camera()
        self.assertIsNone(particleinspect.pick_particle(camera, 400, 300, np.zeros((0, 3)), 0, 0))

    def test_ties_broken_by_nearest_depth(self):
        camera = _straight_on_camera()
        # Both project to (approximately) the same screen point; the nearer one (smaller z) wins.
        positions = np.array([(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)])
        pixels, _ = scene3d.project(camera, 400, 300, positions)
        index = particleinspect.pick_particle(camera, 400, 300, positions,
                                              float(pixels[0, 0]), float(pixels[0, 1]), tolerance=50.0)
        self.assertEqual(index, 0)   # z=1 is closer to the eye at z=5 than z=-1


class ParticleReadoutTests(unittest.TestCase):
    def test_reads_id_age_velocity_and_speed(self):
        instance = _instance([(0, 0, 0)], velocities=[(3.0, 4.0, 0.0)], ages=[7.0], ids=[42])
        readout = particleinspect.particle_readout(instance, 0)
        self.assertEqual(readout["id"], 42)
        self.assertEqual(readout["age"], 7.0)
        self.assertEqual(readout["velocity"], (3.0, 4.0, 0.0))
        self.assertAlmostEqual(readout["speed"], 5.0)


class AttributeValuesTests(unittest.TestCase):
    def test_age_speed_id_lifetime_fraction(self):
        instance = _instance([(0, 0, 0), (1, 0, 0)], velocities=[(3, 4, 0), (0, 0, 0)],
                             ages=[5.0, 2.0], lifetimes=[10.0, 4.0], ids=[7, 8])
        np.testing.assert_allclose(particleinspect.attribute_values(instance, "age"), [5.0, 2.0])
        np.testing.assert_allclose(particleinspect.attribute_values(instance, "speed"), [5.0, 0.0])
        np.testing.assert_allclose(particleinspect.attribute_values(instance, "id"), [7.0, 8.0])
        np.testing.assert_allclose(particleinspect.attribute_values(instance, "lifetime_fraction"), [0.5, 0.5])

    def test_unknown_attribute_raises(self):
        instance = _instance([(0, 0, 0)])
        with self.assertRaises(ValueError):
            particleinspect.attribute_values(instance, "not-a-real-attribute")


class ColorizeByAttributeTests(unittest.TestCase):
    def test_speed_maps_to_the_expected_ramp_colour(self):
        # Two particles, speed 0 and 10: with vmin=0, vmax=10 explicit, a speed of 5 (the
        # midpoint) must land exactly on the ramp's green midpoint.
        instance = _instance([(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                             velocities=[(0, 0, 0), (5, 0, 0), (10, 0, 0)])
        rgb = particleinspect.colorize_by_attribute(instance, "speed", vmin=0.0, vmax=10.0)
        np.testing.assert_allclose(rgb[0], (0.0, 0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(rgb[1], (0.0, 1.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(rgb[2], (1.0, 0.0, 0.0), atol=1e-6)

    def test_empty_instance_returns_empty_array(self):
        instance = _instance([])
        rgb = particleinspect.colorize_by_attribute(instance, "age")
        self.assertEqual(rgb.shape, (0, 3))

    def test_default_range_is_the_attributes_own_min_and_max(self):
        instance = _instance([(0, 0, 0), (1, 0, 0)], ages=[2.0, 8.0])
        rgb = particleinspect.colorize_by_attribute(instance, "age")
        np.testing.assert_allclose(rgb[0], (0.0, 0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(rgb[1], (1.0, 0.0, 0.0), atol=1e-6)


class ParticleCountPerEmitterTests(unittest.TestCase):
    def test_one_line_per_emitter(self):
        text = particleinspect.particle_count_per_emitter({"Sparks": 340, "Debris": 12})
        self.assertEqual(text, "Sparks: 340 particles\nDebris: 12 particles")


if __name__ == "__main__":
    unittest.main()
