import unittest

from tools.live_picture_bench import qt_global_to_x11


class QtGlobalToX11Tests(unittest.TestCase):
    def test_unscaled_pointer_coordinates_are_unchanged(self):
        self.assertEqual(qt_global_to_x11((321, 123), 1.0), (321, 123))

    def test_scaled_pointer_coordinates_target_the_same_physical_pixel(self):
        self.assertEqual(qt_global_to_x11((321, 123), 2.0), (642, 246))

    def test_fractional_scale_rounds_to_nearest_native_pixel(self):
        self.assertEqual(qt_global_to_x11((10.4, 10.6), 1.5), (16, 16))

    def test_invalid_device_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            qt_global_to_x11((10, 10), 0)


if __name__ == "__main__":
    unittest.main()
