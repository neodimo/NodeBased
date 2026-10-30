"""Focused ColorCorrect coverage for the resident GPU island."""
import unittest

from tests.test_gpu2d import DeviceTestCase, params
from nodebased import colorcurves, gpu2d


class ColorCorrectTests(DeviceTestCase):
    def test_master_zones_alpha_mask_and_odd_frame(self):
        source, mask = self.data.image(37, 23, bands=True), self.data.matte(37, 23)
        p = params("ColorCorrect", saturation=1.3, contrast=1.15, gamma=.7, gain=1.2, offset=-.1,
                   shadows_saturation=.3, midtones_gamma=1.8, highlights_gain=.6, mix=.3)
        self.assert_matches("ColorCorrect", p, [source, mask], "zones mask mix")

    def test_linear_and_smooth_curves(self):
        source = self.data.image(37, 23, bands=True)
        points = [[0, .8], [.2, .7], [.5, .3], [.8, .1], [1, 0]]
        for interpolation in ("linear", "smooth"):
            p = params("ColorCorrect", shadows_curve=colorcurves.encode(points, interpolation),
                       highlights_curve=colorcurves.encode(points[::-1][::-1], interpolation), gain=1.1)
            self.assert_matches("ColorCorrect", p, [source], interpolation)

    def test_keyed_or_large_curve_stays_on_cpu(self):
        base = [[0, 1], [1, 0]]
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_color_correct(params("ColorCorrect", shadows_curve=colorcurves.encode(base, slopes=[[0, 0], [0, 0]])))
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_color_correct(params("ColorCorrect", shadows_curve=colorcurves.encode([[i / 8, 1 - i / 8] for i in range(9)])))


if __name__ == "__main__":
    unittest.main()
