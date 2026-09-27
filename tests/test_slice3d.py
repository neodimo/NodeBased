import unittest

import numpy as np

from nodebased import slice3d
from nodebased.scene3d import Volume


def _analytic_volume(with_velocity=True, with_temperature=True, with_flame=False):
    nx, ny, nz = 4, 5, 6
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    density = (ii + 10 * jj + 100 * kk).astype(np.float32)
    kwargs = {}
    if with_temperature:
        kwargs["temperature"] = (density * 2.0).astype(np.float32)
    if with_velocity:
        velocity = np.zeros((nx, ny, nz, 3), np.float32)
        velocity[..., 0] = ii
        velocity[..., 1] = jj
        velocity[..., 2] = kk
        kwargs["velocity"] = velocity
    if with_flame:
        kwargs["flame"] = np.full((nx, ny, nz), 0.5, np.float32)
    return Volume(density, voxel_size=1.0, **kwargs)


class FieldListTests(unittest.TestCase):
    def test_only_carried_fields_are_listed(self):
        bare = Volume(np.zeros((2, 2, 2), np.float32))
        self.assertEqual(slice3d.available_fields(bare), ["density"])

    def test_temperature_and_velocity_add_their_derived_fields(self):
        volume = _analytic_volume(with_velocity=True, with_temperature=True, with_flame=True)
        fields = slice3d.available_fields(volume)
        self.assertEqual(fields, ["density", "temperature", "speed", "vorticity", "divergence", "flame"])


class SlicePlaneTests(unittest.TestCase):
    def setUp(self):
        self.volume = _analytic_volume()

    def test_slice_matches_known_voxel_values_on_every_axis(self):
        # density[i, j, k] = i + 10*j + 100*k, so a Z slice at k is a plane of i + 10*j + 100*k.
        plane = slice3d.slice_plane(self.volume, "z", 2, "density")
        expected = np.array([[i + 10 * j + 200 for j in range(5)] for i in range(4)], np.float32)
        np.testing.assert_array_equal(plane, expected)

        plane_x = slice3d.slice_plane(self.volume, "x", 1, "density")
        expected_x = np.array([[1 + 10 * j + 100 * k for k in range(6)] for j in range(5)], np.float32)
        np.testing.assert_array_equal(plane_x, expected_x)

        plane_y = slice3d.slice_plane(self.volume, "y", 3, "density")
        expected_y = np.array([[i + 30 + 100 * k for k in range(6)] for i in range(4)], np.float32)
        np.testing.assert_array_equal(plane_y, expected_y)

    def test_readout_matches_the_voxel_value(self):
        for (i, j, k) in [(0, 0, 0), (2, 3, 4), (3, 4, 5)]:
            value = slice3d.sample_value(self.volume, "z", k, "density", i, j)
            self.assertEqual(value, float(self.volume.density[i, j, k]))

    def test_moving_the_slice_position_changes_the_image(self):
        plane_a = slice3d.slice_plane(self.volume, "z", 0, "density")
        plane_b = slice3d.slice_plane(self.volume, "z", 5, "density")
        self.assertFalse(np.array_equal(plane_a, plane_b))

    def test_index_is_clamped_in_range(self):
        self.assertEqual(slice3d.clamp_index(self.volume, "z", -5), 0)
        self.assertEqual(slice3d.clamp_index(self.volume, "z", 999), 5)

    def test_unavailable_field_raises(self):
        bare = Volume(np.zeros((2, 2, 2), np.float32))
        with self.assertRaises(ValueError):
            slice3d.slice_plane(bare, "z", 0, "speed")


class DerivedFieldTests(unittest.TestCase):
    def test_speed_is_the_velocity_magnitude(self):
        volume = _analytic_volume()
        speed = slice3d.scalar_field(volume, "speed")
        expected = np.linalg.norm(volume.velocity, axis=-1)
        np.testing.assert_allclose(speed, expected, atol=1e-5)

    def test_divergence_of_a_uniform_field_is_zero(self):
        nx, ny, nz = 4, 4, 4
        density = np.zeros((nx, ny, nz), np.float32)
        velocity = np.zeros((nx, ny, nz, 3), np.float32)
        velocity[..., 0] = 1.0   # constant flow: zero divergence everywhere
        volume = Volume(density, velocity=velocity)
        divergence = slice3d.scalar_field(volume, "divergence")
        np.testing.assert_allclose(divergence, 0.0, atol=1e-5)


class VelocityArrowsTests(unittest.TestCase):
    def test_none_without_velocity(self):
        bare = Volume(np.zeros((2, 2, 2), np.float32))
        self.assertIsNone(slice3d.in_plane_velocity(bare, "z", 0))

    def test_arrows_are_the_in_plane_components(self):
        volume = _analytic_volume()
        arrows = slice3d.in_plane_velocity(volume, "z", 2, step=1)
        self.assertEqual(arrows.shape, (4, 5, 2))
        self.assertTrue(np.array_equal(arrows[..., 0], volume.velocity[:, :, 2, 0]))
        self.assertTrue(np.array_equal(arrows[..., 1], volume.velocity[:, :, 2, 1]))


class ColorizeTests(unittest.TestCase):
    def test_shape_and_dtype(self):
        plane = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
        rgb = slice3d.colorize(plane)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_low_and_high_ends_are_distinct_colours(self):
        plane = np.array([[0.0, 10.0]], np.float32)
        rgb = slice3d.colorize(plane)
        self.assertFalse(np.array_equal(rgb[0, 0], rgb[0, 1]))

    def test_flat_plane_does_not_divide_by_zero(self):
        plane = np.full((3, 3), 5.0, np.float32)
        rgb = slice3d.colorize(plane)
        self.assertEqual(rgb.shape, (3, 3, 3))


if __name__ == "__main__":
    unittest.main()
