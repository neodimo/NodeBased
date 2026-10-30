"""Focused ColorCorrect coverage for the resident GPU island."""
import unittest
import numpy as np

from tests import test_gpu2d as shared
from tests.test_gpu2d import DeviceTestCase, cpu_node, params
from nodebased import colorcurves, gpu2d
from nodebased.raster import Raster


def setUpModule():
    # DeviceTestCase's backend is installed by its defining module's fixture;
    # importing the class alone does not run that fixture under unittest.
    shared.setUpModule()


def tearDownModule():
    shared.tearDownModule()


class CurveEndpointTests(unittest.TestCase):
    def test_highlight_pixel_does_not_pick_up_a_rounded_shadow_endpoint(self):
        source = Raster(np.array([[[-1.1177609, .9303242, 1.7771034, .0017853]]], np.float32))
        p = params("ColorCorrect", saturation=2, contrast=1.5, gamma=.5,
                   shadows_saturation=0, shadows_gamma=.5)
        held = cpu_node("ColorCorrect", p, [source]).pixels
        p["shadows_curve"] = colorcurves.encode([[0, 1], [.125, 0]], "smooth")
        exact_endpoint = cpu_node("ColorCorrect", p, [source]).pixels
        np.testing.assert_allclose(held, exact_endpoint, rtol=1e-6, atol=0)


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

    def test_seeded_master_and_zone_sweep(self):
        rng = np.random.default_rng(610)
        source = self.data.image(37, 23, bands=True)
        gpu_cases = cpu_fallbacks = 0
        for i in range(200):
            knobs = {}
            for zone in ("", "shadows_", "midtones_", "highlights_"):
                knobs[zone + "saturation"] = float(rng.choice([0, .5, 1, 1.5, 2]))
                knobs[zone + "contrast"] = float(rng.choice([.7, 1, 1.5]))
                knobs[zone + "gamma"] = float(rng.choice([.5, 1, 2.2]))
                knobs[zone + "gain"] = float(rng.choice([.6, 1, 1.4]))
                knobs[zone + "offset"] = float(rng.choice([-.1, 0, .1]))
            knobs["mix"] = float(rng.choice([.3, 1]))
            p = params("ColorCorrect", **knobs)
            try:
                self.assert_matches("ColorCorrect", p, [source], f"seeded case {i}")
                gpu_cases += 1
            except gpu2d.Unsupported:
                cpu_fallbacks += 1
        self.assertGreaterEqual(gpu_cases, 150)
        self.assertLessEqual(cpu_fallbacks, 50)


if __name__ == "__main__":
    unittest.main()
