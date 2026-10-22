import unittest

from tools.live_picture_bench import qt_global_to_x11, qt_horizontal_drag_path


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

    def test_slider_drag_stays_inside_the_widget_and_keeps_a_stable_row(self):
        start, end = qt_horizontal_drag_path((100, 40), 200, 24)
        self.assertEqual(start, (150, 52))
        self.assertEqual(end, (250, 52))

    def test_scaled_slider_drag_maps_both_endpoints_to_physical_pixels(self):
        start, end = qt_horizontal_drag_path((100, 40), 200, 24, 2.0)
        self.assertEqual(start, (300, 104))
        self.assertEqual(end, (500, 104))

    def test_slider_drag_rejects_bad_geometry_or_fractions(self):
        with self.assertRaises(ValueError):
            qt_horizontal_drag_path((0, 0), 0, 24)
        with self.assertRaises(ValueError):
            qt_horizontal_drag_path((0, 0), 100, 24, start_fraction=0.8, end_fraction=0.2)


if __name__ == "__main__":
    unittest.main()
