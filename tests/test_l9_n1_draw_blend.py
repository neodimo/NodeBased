"""Lane 9, step N1 (Fri 11/20): Rectangle, Ramp, Radial, Grid and Noise mix their colour into the picture the way Nuke 17.0v6 renders it.

The numbers are read from real Nuke renders (not in git). Every channel, the alpha too, goes from the input to the colour by the matte; the
colour's alpha is a value drawn into the alpha channel; a Ramp has the one colour and its ramp weight is the matte, sampled at the pixel's index; the
Ramp node's smooth0, smooth1 and plinear curves are 2 t^2 - t^3, its mirror and t^3, and Rectangle's softness is a width in pixels with a smoothstep
per edge. A node saved before keeps its pixels through `legacy_draw_blend` 1.
"""
import copy
import unittest

import numpy as np

from nodebased.core import SCHEMA_VERSION, SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster

from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels, tile_pixels


def flat(h=8, w=12, rgba=(0.2, 0.4, 0.6, 0.5)):
    img = np.zeros((h, w, 4), np.float32)
    img[...] = np.asarray(rgba, np.float32)
    return img


def draw(kind, pixels, **knobs):
    params = {**SPECS[kind]["params"], "width": pixels.shape[1], "height": pixels.shape[0], **knobs}
    assert params["legacy_draw_blend"] == 0
    return np.asarray(Evaluator._windowed_kernel(kind, params, [Raster.of(np.asarray(pixels, np.float32))]).pixels, np.float64)


RECT = dict(box_x=3.0, box_y=2.0, box_width=6.0, box_height=4.0, red=0.25, green=0.5, blue=0.75, alpha=0.8, softness=0.0)


class RectangleMixesChannels(unittest.TestCase):
    def test_inside_the_area_every_channel_is_the_colour(self):
        out = draw("Rectangle", flat(), **RECT)
        bottom = out[::-1]                                           # row 0 is the bottom
        np.testing.assert_allclose(bottom[3, 3:9], [[0.25, 0.5, 0.75, 0.8]] * 6, atol=1e-6)
        np.testing.assert_allclose(bottom[3, 2], [0.2, 0.4, 0.6, 0.5], atol=1e-6)         # outside
        np.testing.assert_allclose(bottom[3, 9], [0.2, 0.4, 0.6, 0.5], atol=1e-6)
        np.testing.assert_allclose(bottom[2, 4], [0.25, 0.5, 0.75, 0.8], atol=1e-6)          # the area covers rows 2 to 5 from the bottom
        np.testing.assert_allclose(bottom[5, 4], [0.25, 0.5, 0.75, 0.8], atol=1e-6)
        np.testing.assert_allclose(bottom[1, 4], [0.2, 0.4, 0.6, 0.5], atol=1e-6)
        np.testing.assert_allclose(bottom[6, 4], [0.2, 0.4, 0.6, 0.5], atol=1e-6)

    def test_opacity_and_invert_and_replace(self):
        half = draw("Rectangle", flat(), **RECT, opacity=0.5)[::-1]
        np.testing.assert_allclose(half[3, 3:9, 0], 0.225, atol=1e-6)
        np.testing.assert_allclose(half[3, 3:9, 3], 0.65, atol=1e-6)                    # Nuke: the alpha moves 0.5 -> 0.8 by the matte too
        inv = draw("Rectangle", flat(), **RECT, invert=1)[::-1]
        np.testing.assert_allclose(inv[3, :3, 0], 0.25, atol=1e-6)
        np.testing.assert_allclose(inv[3, 3:9, 0], 0.2, atol=1e-6)
        rep = draw("Rectangle", flat(), **RECT, replace=1)[::-1]
        np.testing.assert_allclose(rep[3, 3:9], [[0.25, 0.5, 0.75, 0.8]] * 6, atol=1e-6)
        np.testing.assert_allclose(rep[3, 0], 0.0, atol=1e-6)                             # replace clears the input before drawing

    def test_softness_is_pixels_with_a_smoothstep_per_edge(self):
        black = np.zeros((20, 40, 4), np.float32)
        area = dict(box_x=4.0, box_y=2.0, box_width=32.0, box_height=16.0, red=1.0, green=1.0, blue=1.0, alpha=1.0)
        row = draw("Rectangle", black, **area, softness=4.0)[::-1][10, :, 0]
        np.testing.assert_allclose(row[:10], [0, 0, 0, 0, 0.104, 0.352, 0.648, 0.896, 1.0, 1.0], atol=1e-3)
        wide = draw("Rectangle", black, **area, softness=10.0)[::-1]
        np.testing.assert_allclose(wide[10, 4:15, 0], [0.0174, 0.065, 0.1362, 0.2242, 0.3224, 0.4238, 0.5219, 0.61, 0.6812, 0.7288, 0.7462], atol=1e-3)
        np.testing.assert_allclose(wide[2:10, 20, 0], [0.0233, 0.0872, 0.1826, 0.3005, 0.432, 0.568, 0.6832, 0.7462], atol=1e-3)

    def test_the_ramp_menu_curves_are_the_squares_of_nukes_render(self):
        black = np.zeros((13, 3, 4), np.float32)
        base = dict(box_x=0.0, box_y=0.0, box_width=3.0, box_height=13.0, softness=0.0, p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=10.0,
                    color0_red=0.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0, red=1.0, green=1.0, blue=1.0, alpha=1.0)
        t = np.arange(13) / 10.0
        for kind, expected in (("linear", t), ("smooth0", np.minimum(t, 1) ** 2), ("smooth1", 1 - (1 - np.minimum(t, 1)) ** 2),
                               ("smooth", np.minimum(t, 1) ** 2 * (3 - 2 * np.minimum(t, 1)))):
            with self.subTest(kind=kind):
                col = draw("Rectangle", black, ramp=kind, **base)[::-1, 0, 0]
                np.testing.assert_allclose(col, np.minimum(expected, 1.0) if kind == "linear" else expected, atol=1e-6)


class RampIsOneColourSampledAtTheIndex(unittest.TestCase):
    P = dict(p0_x=0.0, p0_y=1.0, p1_x=0.0, p1_y=7.0, color1_red=0.9, color1_green=0.7, color1_blue=0.5, color1_alpha=0.4)

    def test_the_input_goes_to_the_colour_along_the_ramp(self):
        out = draw("Ramp", flat(), **self.P)[::-1]
        np.testing.assert_allclose(out[:, 0, 0], [0.2, 0.2, 0.31667, 0.43333, 0.55, 0.66667, 0.78333, 0.9], atol=1e-4)
        np.testing.assert_allclose(out[:, 0, 1], [0.4, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], atol=1e-4)
        np.testing.assert_allclose(out[:, 0, 3], [0.5, 0.5, 0.48333, 0.46667, 0.45, 0.43333, 0.41667, 0.4], atol=1e-4)
        half = draw("Ramp", flat(), **self.P, opacity=0.5)[::-1]
        np.testing.assert_allclose(half[:, 0, 0], [0.2, 0.2, 0.25833, 0.31667, 0.375, 0.43333, 0.49167, 0.55], atol=1e-4)

    def test_colour_zero_has_no_say(self):
        a = draw("Ramp", flat(), **self.P)
        b = draw("Ramp", flat(), **self.P, color0_red=0.9, color0_alpha=0.0)
        np.testing.assert_array_equal(a, b)

    def test_the_ramp_nodes_curves(self):
        black = np.zeros((13, 3, 4), np.float32)
        base = dict(p0_x=0.0, p0_y=0.0, p1_x=0.0, p1_y=10.0, color1_red=1.0, color1_green=1.0, color1_blue=1.0, color1_alpha=1.0)
        want = {
            "linear": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "plinear": [0.0, 0.001, 0.008, 0.027, 0.064, 0.125, 0.216, 0.343, 0.512, 0.729, 1.0],
            "smooth0": [0.0, 0.019, 0.072, 0.153, 0.256, 0.375, 0.504, 0.637, 0.768, 0.891, 1.0],
            "smooth1": [0.0, 0.109, 0.232, 0.363, 0.496, 0.625, 0.744, 0.847, 0.928, 0.981, 1.0],
            "smooth": [0.0, 0.028, 0.104, 0.216, 0.352, 0.5, 0.648, 0.784, 0.896, 0.972, 1.0],
        }
        for kind, expected in want.items():
            with self.subTest(kind=kind):
                col = draw("Ramp", black, ramp_type=kind, **base)[::-1, 0, 0]
                np.testing.assert_allclose(col[:11], expected, atol=1e-6)
                np.testing.assert_allclose(col[11:], 1.0, atol=1e-6)


class TilesEqualTheFrame(unittest.TestCase):
    CASES = {
        "Rectangle": dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=3.0, red=0.9, green=0.4, blue=0.2, alpha=0.7, opacity=0.8),
        "Rectangle ramp": dict(box_x=2.0, box_y=2.0, box_width=40.0, box_height=30.0, softness=2.0, ramp="smooth", p0_x=0.0, p0_y=4.0, p1_x=0.0,
                               p1_y=30.0, color0_red=0.1, color0_green=0.2, color0_blue=0.3, red=0.9, green=0.7, blue=0.5),
        "Ramp": dict(p0_x=3.0, p0_y=4.0, p1_x=40.0, p1_y=30.0, ramp_type="smooth0", color1_red=0.9, color1_green=0.7, color1_blue=0.5, color1_alpha=0.6),
        "Radial": dict(box_x=4.0, box_y=3.0, box_width=36.0, box_height=30.0, softness=0.7, alpha=0.9),
        "Grid": dict(spacing_x=9.0, spacing_y=8.0, number_x=0, number_y=0, line_width=2.0, alpha=0.8),
    }

    def test_tiles_of_7_16_and_32_equal_the_whole_frame_with_and_without_a_mask(self):
        for label, params in self.CASES.items():
            kind = label.split()[0]
            for masked in (False, True):
                g = Graph()
                g.add("src", "Checker", dict(width=45, height=37, size=5))
                inputs = {"image": "src"}
                if masked:
                    g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
                    inputs["mask"] = "matte"
                g.add("node", kind, dict(width=45, height=37, **params), **inputs)
                self.assertEqual(g.doc["nodes"]["node"]["params"]["legacy_draw_blend"], 0)
                whole = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=label, masked=masked, tile=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)


class OlderDocumentsKeepTheirPixels(unittest.TestCase):
    def saved_before(self, kind, params):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("n", kind, dict(width=45, height=37, **params), image="src")
        old = copy.deepcopy(g.doc)
        old["nodes"]["n"]["params"].pop("legacy_draw_blend")
        old["nodes"]["n"]["params"].pop("legacy_draw_axes", None)           # a node from before both Draw steps carries neither flag
        old["version"] = 38
        loaded = upgrade_document(old)
        validate(loaded)
        return loaded

    def test_the_flag_is_set_on_every_draw_kind_and_the_pixels_are_the_older_over(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 40)
        params = dict(box_x=9.0, box_y=6.0, box_width=25.0, box_height=18.0, softness=0.4, red=1.0, green=0.5, blue=0.25, alpha=0.8)
        loaded = self.saved_before("Rectangle", params)
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_draw_blend"], 1)
        for kind in ("Radial", "Ramp", "Grid", "Noise"):
            self.assertEqual(self.saved_before(kind, {})["nodes"]["n"]["params"]["legacy_draw_blend"], 1, kind)
        pixels = Evaluator().evaluate_raster(dict(loaded, view="n"), target="n").pixels
        plate = Evaluator().evaluate_raster(dict(loaded, view="src"), target="src").pixels
        p = loaded["nodes"]["n"]["params"]
        self.assertEqual(p["legacy_draw_axes"], 1)
        # the older reading: rows from the top, softness a fraction of half the short side, the colour premultiplied by its alpha over the plate
        ys, xs = np.mgrid[0:37, 0:45]
        dist = np.minimum(np.minimum(xs - 9.0, 9.0 + 25.0 - 1.0 - xs), np.minimum(ys - 6.0, 6.0 + 18.0 - 1.0 - ys))
        alpha = (np.clip(dist / (0.4 * 18.0 / 2.0), 0.0, 1.0) * 0.8)[..., None]
        want = np.concatenate([np.asarray([1.0, 0.5, 0.25]) * alpha, alpha], axis=-1) + plate * (1.0 - alpha)
        np.testing.assert_allclose(pixels, want, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
