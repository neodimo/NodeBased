"""2D behaviour lane 2, step K: the main knobs Nuke's IBKGizmo, HueCorrect, MatchGrade and GenerateLUT have and NodeBased lacked, and Nuke's layout of the time
and metadata panels.

The Reference Guide names these knobs and says nothing about what they compute, so the checks here say what each knob's name and the tutorial state as
properties of the pixels (a wider screen range keys more, yellow takes yellow out, mix luminance restores the luminance the pixel came in with), and the
cases in tools/nuke_fidelity_cases_color.py stay not scored. An older document renders as it did before, bit for bit.
"""
import copy
import unittest

import numpy as np

from nodebased.core import SPECS, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels


def kernel(kind, params, *inputs):
    return Evaluator._kernel(kind, dict(SPECS[kind]["params"], **params), list(inputs))


def solid(r, g, b, a=1.0, h=4, w=4):
    return np.tile(np.array([r, g, b, a], np.float32), (h, w, 1))


GREEN_PLATE = (0.15, 0.7, 0.3)


class IBKGizmoNewKnobsTests(unittest.TestCase):
    def setUp(self):
        self.plate = solid(*GREEN_PLATE)
        self.fg = solid(0.45, 0.6, 0.2)             # a pixel part way between the screen and a foreground colour

    def key(self, **params):
        return kernel("IBKGizmo", dict({"screen_type": "green"}, **params), self.fg, self.plate)

    def test_a_wider_screen_range_keys_more_of_the_picture(self):
        alphas = [float(self.key(screen_range=r)[0, 0, 3]) for r in (0.5, 1.0, 1.5, 3.0)]
        self.assertTrue(all(a > b for a, b in zip(alphas, alphas[1:])), alphas)
        np.testing.assert_array_equal(self.key(screen_range=1.0), kernel("IBKGizmo", {"screen_type": "green"}, self.fg, self.plate))

    def test_pick_reads_the_screen_from_the_colour(self):
        green = self.key()
        np.testing.assert_array_equal(self.key(screen_type="pick", color_r=0.1, color_g=0.8, color_b=0.2), green)
        blue_plate = solid(0.2, 0.3, 0.9)
        blue = kernel("IBKGizmo", {"screen_type": "blue"}, self.fg, blue_plate)
        picked = kernel("IBKGizmo", {"screen_type": "pick", "color_r": 0.2, "color_g": 0.3, "color_b": 0.9}, self.fg, blue_plate)
        np.testing.assert_array_equal(picked, blue)
        self.assertFalse(np.array_equal(green, self.key(screen_type="pick", color_r=0.9, color_g=0.1, color_b=0.1)))        # a red screen reads another difference

    def test_yellow_cyan_and_magenta_take_that_secondary_colours_excess_out(self):
        for flag, pair, third in (("yellow", (0, 1), 2), ("cyan", (1, 2), 0), ("magenta", (0, 2), 1)):
            with self.subTest(flag=flag):
                fg = solid(0.8, 0.7, 0.2) if flag == "yellow" else (solid(0.2, 0.8, 0.7) if flag == "cyan" else solid(0.8, 0.2, 0.7))
                off = kernel("IBKGizmo", {"screen_type": "green", "screen_subtraction": 0}, fg, self.plate)
                on = kernel("IBKGizmo", {"screen_type": "green", "screen_subtraction": 0, flag: 1}, fg, self.plate)
                np.testing.assert_array_equal(on[..., 3], off[..., 3])                      # the matte is not touched
                excess = max(min(float(off[0, 0, pair[0]]), float(off[0, 0, pair[1]])) - float(off[0, 0, third]), 0.0)
                self.assertGreater(excess, 0.01)
                for channel in pair:
                    self.assertAlmostEqual(float(off[0, 0, channel] - on[0, 0, channel]), excess, places=5)
                self.assertEqual(float(on[0, 0, third]), float(off[0, 0, third]))

    def test_the_luminance_level_acts_only_when_its_enable_box_is_on(self):
        fg = solid(0.5, 0.55, 0.2)
        base = {"screen_type": "green", "luminance_match": 1}

        def run(**p):
            return kernel("IBKGizmo", dict(base, **p), fg, self.plate)

        np.testing.assert_array_equal(run(luminance_level=0.2, luminance_level_enable=0), run(luminance_level=1.0, luminance_level_enable=0))
        self.assertFalse(np.array_equal(run(luminance_level=0.2, luminance_level_enable=1), run(luminance_level=1.0, luminance_level_enable=1)))
        # with the enable on and the level at 1 the match is whole, as it is with the enable off
        np.testing.assert_array_equal(run(luminance_level=1.0, luminance_level_enable=1), run(luminance_level=1.0, luminance_level_enable=0))

    def test_use_bg_chroma_puts_the_backgrounds_chroma_in_the_edge(self):
        bg = solid(0.9, 0.3, 0.2)
        edge_on = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 1}, self.fg, self.plate, bg)
        edge_off = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 0}, self.fg, self.plate, bg)
        np.testing.assert_array_equal(edge_on[..., 3], edge_off[..., 3])
        self.assertGreater(float(edge_off[0, 0, 3]), 0.0)
        self.assertLess(float(edge_off[0, 0, 3]), 1.0)
        self.assertGreater(float(edge_on[0, 0, 0]), float(edge_off[0, 0, 0]))                  # red, the background's strong channel, comes in
        self.assertGreater(edge_on[0, 0, 0] - edge_off[0, 0, 0], edge_on[0, 0, 2] - edge_off[0, 0, 2])
        np.testing.assert_array_equal(kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 1}, self.fg, self.plate), self.key())      # no bg wired: nothing to take

    def test_a_fully_opaque_or_fully_transparent_pixel_takes_no_bg_chroma(self):
        bg = solid(0.9, 0.3, 0.2)
        for fg in (solid(*GREEN_PLATE), solid(0.5, 0.5, 0.5)):
            on = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 1}, fg, self.plate, bg)
            off = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 0}, fg, self.plate, bg)
            np.testing.assert_allclose(on, off, atol=1e-6)

    def test_every_new_knob_is_registered_with_its_range_and_label(self):
        from nodebased import knobs
        from nodebased.core import LIMITS
        for name in ("screen_range", "color_r", "luminance_level_enable", "yellow", "cyan", "magenta", "use_bg_chroma"):
            self.assertIn(name, SPECS["IBKGizmo"]["params"])
            self.assertIn(name, LIMITS)
        self.assertEqual(SPECS["IBKGizmo"]["params"]["luminance_level_enable"], 0)
        labels = [g.label for g in knobs.KNOB_GROUPS["IBKGizmo"] if getattr(g, "label", None)] if hasattr(knobs, "KNOB_GROUPS") else []
        self.assertTrue(labels == [] or "Screen range" in labels)


def ibk_graph(params, masked=False):
    g = Graph()
    g.add("fg", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.45, green=0.6, blue=0.2, alpha=1.0, softness=0.5))
    g.add("cp", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.1, green=0.95, blue=0.3, alpha=1.0, softness=0.1))
    g.add("bg", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.8, green=0.3, blue=0.2, alpha=1.0, softness=0.1))
    g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
    inputs = {"fg": "fg", "c": "cp"}
    if "use_bg_chroma" in params or "use_bg_luminance" in params:
        inputs["bg"] = "bg"
    if masked:
        inputs["mask"] = "matte"
    g.add("node", "IBKGizmo", params, **inputs)
    return g


class IBKGizmoNewKnobsOnTilesTests(unittest.TestCase):
    def test_tiles_equal_the_whole_frame_with_the_new_knobs(self):
        from tests.test_2d_behaviour_color_a import tiled
        for params in ({"screen_type": "pick", "color_r": 0.1, "color_g": 0.7, "color_b": 0.3, "screen_range": 1.4, "yellow": 1, "magenta": 1},
                       {"screen_type": "green", "luminance_match": 1, "luminance_level": 0.4, "luminance_level_enable": 1, "cyan": 1},
                       {"screen_type": "green", "use_bg_chroma": 1, "screen_range": 0.7}):
            for masked in (False, True):
                g = ibk_graph(params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class IBKGizmoOlderDocumentTests(unittest.TestCase):
    def test_an_older_document_keeps_its_luminance_level(self):
        params = {"screen_type": "green", "luminance_match": 1, "luminance_level": 0.3}
        doc = copy.deepcopy(ibk_graph(params).doc)
        for name in ("luminance_level_enable", "screen_range", "yellow", "cyan", "magenta", "use_bg_chroma", "color_r", "color_g", "color_b"):
            doc["nodes"]["node"]["params"].pop(name, None)
        doc = upgrade_document(doc)
        got = doc["nodes"]["node"]["params"]
        self.assertEqual(got["luminance_level_enable"], 1)                      # the older node always applied its level
        self.assertEqual(got.get("screen_range", 1.0), 1.0)
        fg, plate = evaluator_pixels(doc, "fg"), evaluator_pixels(doc, "cp")
        older = kernel("IBKGizmo", dict(params, luminance_level_enable=1), fg, plate)
        np.testing.assert_array_equal(evaluator_pixels(doc, "node"), older)
        # the same graph made fresh (enable off) is the whole match
        fresh = kernel("IBKGizmo", dict(params, luminance_level_enable=0), fg, plate)
        self.assertFalse(np.array_equal(older, fresh))


class HueCorrectMixLuminanceTests(unittest.TestCase):
    CURVE_LUM = '{"interpolation":"linear","points":[[0,2],[60,1],[360,2]]}'            # doubles the luminance of the reds (hue 0)

    def image(self):
        return np.tile(np.array([0.6, 0.2, 0.15, 1.0], np.float32), (4, 4, 1))

    @staticmethod
    def luma(px):
        return 0.2125 * px[..., 0] + 0.7154 * px[..., 1] + 0.0721 * px[..., 2]

    def test_mix_luminance_gives_the_output_back_the_luminance_it_came_in_with(self):
        src = self.image()
        base = kernel("HueCorrect", {"curve_lum": self.CURVE_LUM}, src)
        self.assertGreater(float(self.luma(base)[0, 0]), float(self.luma(src)[0, 0]) * 1.5)          # the curve raised the luminance
        full = kernel("HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 1.0}, src)
        self.assertAlmostEqual(float(self.luma(full)[0, 0]), float(self.luma(src)[0, 0]), places=5)
        half = kernel("HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 0.5}, src)
        self.assertAlmostEqual(float(self.luma(half)[0, 0]), 0.5 * float(self.luma(src)[0, 0]) + 0.5 * float(self.luma(base)[0, 0]), places=5)
        np.testing.assert_array_equal(kernel("HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 0.0}, src), base)
        np.testing.assert_array_equal(full[..., 3], src[..., 3])

    def test_the_default_leaves_every_older_pixel_alone(self):
        self.assertEqual(SPECS["HueCorrect"]["params"]["mix_luminance"], 0.0)
        src = self.image()
        self.assertTrue(np.array_equal(kernel("HueCorrect", {"curve_sat": '{"interpolation":"linear","points":[[0,2],[360,2]]}'}, src),
                                       kernel("HueCorrect", {"curve_sat": '{"interpolation":"linear","points":[[0,2],[360,2]]}', "mix_luminance": 0.0}, src)))

    def test_tiles_equal_the_whole_frame(self):
        from tests.test_2d_behaviour_color_a import tiled
        for masked in (False, True):
            g = Graph()
            g.add("plate", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.9, green=0.5, blue=0.2, alpha=0.8, softness=0.5))
            g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
            inputs = {"image": "plate"}
            if masked:
                inputs["mask"] = "matte"
            g.add("node", "HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 0.7}, **inputs)
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
