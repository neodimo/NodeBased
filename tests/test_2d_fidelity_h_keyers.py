"""Fidelity step H, part 1: the ScreenKeyer (Keylight) takes unPreMultiply, gangBiases, the screen despot, screen and inside
replace, the tuning gains, source alpha, the crops, the inside and outside components and the `bg` input; the HueKeyer takes its
output switch and channel. Pixels by hand against the guide's wording, tiles against the evaluator, bypass, old documents."""
import copy
import unittest

import numpy as np

from nodebased import keyerknobs
from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_2d_parity_step_k1 import Graph, evaluator_pixels, screen_graph, tile_pixels

KEY = (0.1, 0.8, 0.2)
PARTIAL = (0.4, 0.5, 0.3)          # screen difference 0.15 of the screen's 0.65: matte 1 - 0.15 / 0.65
PARTIAL_MATTE = 1.0 - 0.15 / 0.65
PARTIAL_DESPILLED = (0.4, 0.35, 0.3)   # green limited to the mean of red and blue


def one(*rgba):
    return np.array([[list(rgba)]], np.float32)


def key(image, inside=None, outside=None, clean=None, mask=None, bg=None, **knobs):
    params = {**SPECS["ScreenKeyer"]["params"], **knobs}
    return Evaluator._kernel("ScreenKeyer", params, [image, inside, outside, clean, mask, bg])


def old_key(image, **knobs):
    """The keyer as an older document has it."""
    return key(image, **{**keyerknobs.upgrade_params("ScreenKeyer"), **knobs})


class DefaultsTests(unittest.TestCase):
    def test_a_new_keyer_has_keylights_defaults(self):
        params = SPECS["ScreenKeyer"]["params"]
        self.assertEqual((params["unpremultiply_result"], params["use_alpha_bias_for_despill"]), (1, 1))
        self.assertEqual((params["screen_replace"], params["inside_replace"]), ("soft color", "soft color"))
        self.assertEqual((params["mid_point"], params["shadow_gain"], params["midtone_gain"], params["highlight_gain"]),
                         (0.5, 1.0, 1.0, 1.0))
        self.assertEqual((params["source_crop_left"], params["source_crop_right"], params["source_crop_bottom"],
                          params["source_crop_top"]), (0.0, 1.0, 0.0, 1.0))
        self.assertEqual(SPECS["ScreenKeyer"]["optional_inputs"][-1], "bg")

    def test_an_old_document_loads_with_the_values_that_leave_the_keyer_as_it_was(self):
        document = {"version": 25, "settings": {}, "view": "k", "nodes": {
            "k": {"type": "ScreenKeyer", "params": {"screen_gain": 1.5}, "inputs": {"image": None}},
            "h": {"type": "HueKeyer", "params": {"hue_center": 100.0}, "inputs": {"image": None}}}}
        document = upgrade_document(copy.deepcopy(document))
        params = document["nodes"]["k"]["params"]
        for name, value in keyerknobs.OLD["ScreenKeyer"].items():
            self.assertEqual(params[name], value, name)
        self.assertEqual((params["unpremultiply_result"], params["screen_replace"], params["use_alpha_bias_for_despill"]),
                         (0, "none", 0))
        self.assertEqual(params["screen_gain"], 1.5)
        self.assertEqual(document["nodes"]["h"]["params"]["output_channel"], "rgba.alpha")
        self.assertEqual(document["nodes"]["h"]["params"]["output_enabled"], 1)

    def test_an_old_node_draws_what_the_keyer_drew_before(self):
        rng = np.random.default_rng(3)
        image = rng.random((9, 11, 4)).astype(np.float32)
        image[..., 1] += 0.3
        legacy = {k: v for k, v in SPECS["ScreenKeyer"]["params"].items() if k not in keyerknobs.NEW["ScreenKeyer"]}
        for view in CHOICES["keyer_view"][:4]:
            with self.subTest(view=view):
                np.testing.assert_array_equal(old_key(image, keyer_view=view),
                                              Evaluator._kernel("ScreenKeyer", {**legacy, "keyer_view": view}, [image]))

    def test_every_new_knob_is_registered(self):
        for kind in keyerknobs.KINDS:
            for name, value in keyerknobs.NEW[kind].items():
                self.assertIn(name, SPECS[kind]["params"])
                if isinstance(value, str):
                    self.assertIn(value, CHOICES[name], name)
                else:
                    self.assertIn(name, LIMITS, name)


class ResultTests(unittest.TestCase):
    def test_unpremultiply_result_shows_the_despilled_colour_and_the_matte(self):
        image = one(*PARTIAL, 1.0)
        straight = key(image, unpremultiply_result=1, screen_replace="none", use_alpha_bias_for_despill=0)[0, 0]
        premult = key(image, unpremultiply_result=0, screen_replace="none", use_alpha_bias_for_despill=0)[0, 0]
        np.testing.assert_allclose(straight[:3], PARTIAL_DESPILLED, atol=1e-6)
        np.testing.assert_allclose(straight[3], PARTIAL_MATTE, atol=1e-6)
        np.testing.assert_allclose(premult[:3], np.array(PARTIAL_DESPILLED) * PARTIAL_MATTE, atol=1e-6)
        np.testing.assert_allclose(premult[3], PARTIAL_MATTE, atol=1e-6)

    def test_gang_biases_takes_the_despill_bias_from_the_alpha_bias(self):
        # Green is the screen; the despill limits it to (1 - bias) * red + bias * blue. A despill bias of 0 uses red only; with
        # the biases ganged, an alpha bias colour weighing blue only (red 0, blue 1) makes the bias 1.
        image = one(0.2, 0.9, 0.6, 1.0)
        own = key(image, keyer_view="intermediate", screen_replace="none", use_alpha_bias_for_despill=0, despill_bias=0.0)[0, 0]
        ganged = key(image, keyer_view="intermediate", screen_replace="none", use_alpha_bias_for_despill=1, despill_bias=0.0,
                     alpha_bias_red=0.0, alpha_bias_blue=1.0)[0, 0]
        self.assertAlmostEqual(float(own[1]), 0.2, places=6)            # limited to red
        self.assertAlmostEqual(float(ganged[1]), 0.6, places=6)         # limited to blue: the alpha bias's
        grey = key(image, keyer_view="intermediate", screen_replace="none", use_alpha_bias_for_despill=1, despill_bias=0.0)[0, 0]
        self.assertAlmostEqual(float(grey[1]), 0.4, places=6)           # grey alpha bias: the mean of red and blue

    def test_ganged_with_bias_colours_the_alpha_bias_wins_over_the_despill_colours(self):
        image = one(0.2, 0.9, 0.6, 1.0)
        out = key(image, keyer_view="intermediate", screen_replace="none", use_alpha_bias_for_despill=1, bias_colours=1, alpha_bias_red=1.0,
                  alpha_bias_blue=0.0, despill_bias_red=0.0, despill_bias_blue=1.0)[0, 0]
        self.assertAlmostEqual(float(out[1]), 0.2, places=6)

    def test_source_alpha_normal_multiplies_the_matte_by_the_sources_alpha(self):
        image = one(0.9, 0.1, 0.1, 0.25)                       # a foreground pixel with a quarter alpha
        self.assertAlmostEqual(float(key(image, keyer_view="screen_matte", source_alpha="ignore")[0, 0, 3]), 1.0, places=6)
        self.assertAlmostEqual(float(key(image, keyer_view="screen_matte", source_alpha="normal")[0, 0, 3]), 0.25, places=6)


class TuningTests(unittest.TestCase):
    def matte(self, rgb, **knobs):
        return float(key(one(*rgb, 1.0), keyer_view="screen_matte", **knobs)[0, 0, 3])

    @staticmethod
    def luma(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    def test_the_shadow_gain_acts_on_dark_pixels_only(self):
        dark, bright = (0.02, 0.1, 0.04), (0.7, 0.85, 0.72)
        base_dark = self.matte(dark)
        self.assertGreater(base_dark, 0.0)
        self.assertAlmostEqual(self.matte(bright, shadow_gain=2.0), self.matte(bright), places=6)
        diff = dark[1] - 0.5 * (dark[0] + dark[2])
        gain = 2.0 + (1.0 - 2.0) * (self.luma(dark) / 0.5)
        self.assertAlmostEqual(self.matte(dark, shadow_gain=2.0), max(0.0, 1.0 - diff / 0.65 * gain), places=6)
        self.assertLess(self.matte(dark, shadow_gain=2.0), base_dark)

    def test_the_highlight_gain_acts_from_the_midpoint_up(self):
        bright = (0.95, 1.0, 0.97)
        diff = 1.0 - 0.5 * (0.95 + 0.97)
        gain = 1.0 + (3.0 - 1.0) * ((self.luma(bright) - 0.5) / 0.5)
        self.assertAlmostEqual(self.matte(bright, highlight_gain=3.0), max(0.0, 1.0 - diff / 0.65 * gain), places=6)

    def test_the_midtone_gain_peaks_at_the_midpoint(self):
        pixel = (0.4, 0.5, 0.3)
        gain = float(np.interp(self.luma(pixel), [0, 0.5, 1], [1, 0, 1]))
        self.assertAlmostEqual(self.matte(pixel, midtone_gain=0.0, mid_point=0.5), 1.0 - 0.15 / 0.65 * gain, places=6)
        # The same pixel with the midpoint moved up to its own luminance is gained fully.
        self.assertAlmostEqual(self.matte(pixel, midtone_gain=0.0, mid_point=self.luma(pixel)), 1.0, places=6)

    def test_all_gains_one_is_the_plain_gain(self):
        self.assertAlmostEqual(self.matte(PARTIAL, mid_point=0.2), PARTIAL_MATTE, places=6)


class DespotTests(unittest.TestCase):
    def frame(self):
        """Foreground (red) on the left, green screen on the right; one screen pixel inside the foreground and one
        foreground pixel inside the screen."""
        image = np.zeros((15, 21, 4), np.float32)
        image[...] = (*KEY, 1.0)
        image[:, :9] = (0.9, 0.1, 0.1, 1.0)
        image[7, 4] = (*KEY, 1.0)            # a hole in the foreground
        image[7, 16] = (0.9, 0.1, 0.1, 1.0)  # a speck in the screen
        return image

    def matte(self, **knobs):
        return key(self.frame(), keyer_view="screen_matte", **knobs)[..., 3]

    def test_without_despot_the_hole_and_the_speck_are_there(self):
        matte = self.matte()
        self.assertEqual(float(matte[7, 4]), 0.0)
        self.assertEqual(float(matte[7, 16]), 1.0)

    def test_despot_black_fills_the_hole_and_leaves_the_speck(self):
        matte = self.matte(screen_despot_black=3.0)
        self.assertEqual(float(matte[7, 4]), 1.0)
        self.assertEqual(float(matte[7, 16]), 1.0)
        np.testing.assert_array_equal(matte[[0, 14], 12:20], 0.0)

    def test_despot_white_removes_the_speck_and_leaves_the_hole(self):
        matte = self.matte(screen_despot_white=3.0)
        self.assertEqual(float(matte[7, 16]), 0.0)
        self.assertEqual(float(matte[7, 4]), 0.0)

    def test_a_size_smaller_than_the_dot_leaves_it(self):
        image = self.frame()
        image[6:9, 3:6] = (*KEY, 1.0)          # a hole 3 pixels across
        matte = key(image, keyer_view="screen_matte", screen_despot_black=1.0)[..., 3]
        self.assertEqual(float(matte[7, 4]), 0.0)
        matte = key(image, keyer_view="screen_matte", screen_despot_black=4.0)[..., 3]
        self.assertEqual(float(matte[7, 4]), 1.0)


class ReplaceTests(unittest.TestCase):
    base = dict(use_alpha_bias_for_despill=0, unpremultiply_result=1)

    def run_replace(self, method, **knobs):
        return key(one(*PARTIAL, 1.0), screen_replace=method, screen_replace_red=0.2, screen_replace_green=0.6,
                   screen_replace_blue=0.9, **self.base, **knobs)[0, 0]

    def test_none_leaves_the_despilled_colour(self):
        np.testing.assert_allclose(self.run_replace("none")[:3], PARTIAL_DESPILLED, atol=1e-6)

    def test_source_brings_back_the_original_in_proportion_to_the_transparency(self):
        share = 1.0 - PARTIAL_MATTE
        expect = np.array(PARTIAL_DESPILLED) + (np.array(PARTIAL) - np.array(PARTIAL_DESPILLED)) * share
        np.testing.assert_allclose(self.run_replace("source")[:3], expect, atol=1e-6)

    def test_hard_color_adds_the_colour_in_proportion_to_the_transparency(self):
        share = 1.0 - PARTIAL_MATTE
        expect = np.array(PARTIAL_DESPILLED) + share * np.array([0.2, 0.6, 0.9])
        np.testing.assert_allclose(self.run_replace("hard color")[:3], expect, atol=1e-6)

    def test_soft_color_keeps_the_sources_luminance_in_proportion(self):
        share = 1.0 - PARTIAL_MATTE
        hard = np.array(PARTIAL_DESPILLED) + share * np.array([0.2, 0.6, 0.9])

        def luma(c):
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

        expect = hard * (1.0 + share * (luma(PARTIAL) / luma(hard) - 1.0))
        np.testing.assert_allclose(self.run_replace("soft color")[:3], expect, atol=1e-6)

    def test_nothing_is_replaced_on_the_opaque_or_the_clear_pixels(self):
        for pixel in ((0.9, 0.1, 0.1), KEY):
            out = key(one(*pixel, 1.0), screen_replace="hard color", **self.base)[0, 0]
            plain = key(one(*pixel, 1.0), screen_replace="none", **self.base)[0, 0]
            np.testing.assert_allclose(out, plain, atol=1e-6)

    def test_inside_replace_colours_the_pixels_the_inside_mask_turned_opaque(self):
        screen_pixel = one(*KEY, 1.0)
        inside = one(0.0, 0.0, 0.0, 0.5)
        common = dict(inside=inside, screen_replace="none", use_alpha_bias_for_despill=0, unpremultiply_result=1)
        plain = key(screen_pixel, inside_replace="none", **common)[0, 0]
        self.assertAlmostEqual(float(plain[3]), 0.5, places=6)
        hard = key(screen_pixel, inside_replace="hard color", inside_replace_red=0.2, inside_replace_green=0.4,
                   inside_replace_blue=0.6, **common)[0, 0]
        np.testing.assert_allclose(hard[:3] - plain[:3], 0.5 * np.array([0.2, 0.4, 0.6]), atol=1e-6)
        source = key(screen_pixel, inside_replace="source", **common)[0, 0]
        np.testing.assert_allclose(source[:3] - plain[:3], 0.5 * (np.array(KEY) - plain[:3]), atol=1e-6)


class ComponentTests(unittest.TestCase):
    LUMA = float(np.float32(0.2126 * 0.5 + 0.7152 * 0.5 + 0.0722 * 0.5))

    def test_inside_components(self):
        screen = one(*KEY, 1.0)
        inside = one(0.5, 0.5, 0.5, 0.25)
        for name, expect in (("alpha", 0.25), ("inverted alpha", 0.75), ("luminance", self.LUMA),
                             ("inverted luminance", 1.0 - self.LUMA)):
            with self.subTest(name):
                out = key(screen, inside=inside, inside_component=name, keyer_view="inside_mask")[0, 0]
                np.testing.assert_allclose(out, [expect] * 4, atol=1e-6)
                matte = key(screen, inside=inside, inside_component=name, keyer_view="screen_matte")[0, 0, 3]
                self.assertAlmostEqual(float(matte), expect, places=6)

    def test_outside_components(self):
        foreground = one(0.9, 0.1, 0.1, 1.0)
        outside = one(0.5, 0.5, 0.5, 0.25)
        for name, expect in (("alpha", 0.75), ("inverted alpha", 0.25), ("luminance", 1.0 - self.LUMA),
                             ("inverted luminance", self.LUMA)):
            with self.subTest(name):
                matte = key(foreground, outside=outside, outside_component=name, keyer_view="screen_matte")[0, 0, 3]
                self.assertAlmostEqual(float(matte), expect, places=6)

    def test_the_views_of_an_unwired_mask_are_black(self):
        for view in ("inside_mask", "outside_mask"):
            np.testing.assert_array_equal(key(one(*KEY, 1.0), keyer_view=view), np.zeros((1, 1, 4), np.float32))


class ViewTests(unittest.TestCase):
    def test_source_and_source_alpha(self):
        image = one(0.3, 0.5, 0.7, 0.4)
        np.testing.assert_array_equal(key(image, keyer_view="source"), image)
        np.testing.assert_allclose(key(image, keyer_view="source_alpha")[0, 0], [0.4] * 4, atol=1e-7)

    def test_composite_lays_the_premultiplied_result_over_bg(self):
        image = one(*PARTIAL, 1.0)
        bg = one(0.2, 0.3, 0.4, 1.0)
        out = key(image, bg=bg, keyer_view="composite", screen_replace="none", use_alpha_bias_for_despill=0)[0, 0]
        expect = np.array(PARTIAL_DESPILLED) * PARTIAL_MATTE + np.array([0.2, 0.3, 0.4]) * (1.0 - PARTIAL_MATTE)
        np.testing.assert_allclose(out[:3], expect, atol=1e-6)
        self.assertAlmostEqual(float(out[3]), 1.0, places=6)

    def test_composite_without_a_bg_is_the_premultiplied_result(self):
        image = one(*PARTIAL, 1.0)
        np.testing.assert_allclose(key(image, keyer_view="composite")[0, 0], key(image, unpremultiply_result=0)[0, 0], atol=1e-6)

    def test_a_wrong_sized_bg_is_refused_on_the_whole_image_path(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("small", "Constant", dict(width=10, height=10, red=1, green=1, blue=1, alpha=1))
        g.add("k", "ScreenKeyer", dict(keyer_view="composite"), image=plate, bg="small")
        with self.assertRaises(ValueError):
            evaluator_pixels(g.doc, "k")


class CropTests(unittest.TestCase):
    def ramp(self, width=10, height=8):
        image = np.zeros((height, width, 4), np.float32)
        image[..., 0] = np.arange(width)[None, :] / 10.0
        image[..., 1] = np.arange(height)[:, None] / 10.0
        image[..., 3] = 1.0
        return image

    def crop(self, **knobs):
        return key(self.ramp(), keyer_view="source", **knobs)

    def test_a_full_crop_changes_nothing(self):
        np.testing.assert_array_equal(self.crop(), self.ramp())

    def test_colour_past_the_edge_is_the_edge_colour(self):
        out = self.crop(source_crop_left=0.2, source_crop_right=0.8, source_edge_red=0.5, source_edge_green=0.25,
                        source_edge_blue=0.125)
        np.testing.assert_array_equal(out[:, 2:8], self.ramp()[:, 2:8])
        for column in (0, 1, 8, 9):
            np.testing.assert_array_equal(out[:, column], np.tile([0.5, 0.25, 0.125, 1.0], (8, 1)).astype(np.float32))

    def test_bottom_and_top_count_from_the_bottom_edge(self):
        # The top row of the array is the top of the picture (rows 0..7); bottom 0.25 and top 0.75 keep rows 2..5.
        out = self.crop(source_crop_bottom=0.25, source_crop_top=0.75, source_edge_red=1.0)
        np.testing.assert_array_equal(out[2:6], self.ramp()[2:6])
        for row in (0, 1, 6, 7):
            np.testing.assert_array_equal(out[row, :, :3], np.tile([1.0, 0.0, 0.0], (10, 1)).astype(np.float32))

    def test_repeat_reflect_and_wrap_along_x(self):
        base = self.ramp()
        columns = {"repeat": [2, 2, 2, 3, 4, 5, 5, 5, 5, 5], "reflect": [3, 2, 2, 3, 4, 5, 5, 4, 3, 2],
                   "wrap": [4, 5, 2, 3, 4, 5, 2, 3, 4, 5]}
        for method, expected in columns.items():
            with self.subTest(method):
                out = self.crop(source_crop_left=0.2, source_crop_right=0.6, source_x_method=method)
                np.testing.assert_array_equal(out, base[:, expected])

    def test_each_axis_has_its_own_method(self):
        out = self.crop(source_crop_bottom=0.25, source_crop_top=0.75, source_y_method="repeat")
        np.testing.assert_array_equal(out, self.ramp()[[2, 2, 2, 3, 4, 5, 5, 5]])

    def test_the_crop_feeds_the_key(self):
        # A foreground pixel outside the crop becomes the (screen-coloured) edge colour, which keys out.
        image = np.zeros((4, 8, 4), np.float32)
        image[...] = (0.9, 0.1, 0.1, 1.0)
        out = key(image, keyer_view="screen_matte", source_crop_left=0.5, source_edge_red=KEY[0], source_edge_green=KEY[1],
                  source_edge_blue=KEY[2])
        np.testing.assert_array_equal(out[:, :4, 3], 0.0)
        np.testing.assert_array_equal(out[:, 4:, 3], 1.0)


class TileTests(unittest.TestCase):
    KNOBS = (
        dict(),
        dict(screen_despot_black=3.0, screen_despot_white=2.0, screen_softness=2.0),
        dict(shadow_gain=2.0, highlight_gain=0.5, mid_point=0.3, source_alpha="normal"),
        dict(screen_replace="hard color", inside_replace="soft color", unpremultiply_result=0),
        dict(inside_component="luminance", outside_component="inverted alpha"),
    )

    def test_every_option_matches_the_evaluator_across_seams(self):
        for knobs in self.KNOBS:
            for view in ("final", "composite", "inside_mask", "source_alpha"):
                with self.subTest(knobs=knobs, view=view):
                    g = Graph()
                    plate = screen_graph(g, 97, 71)
                    g.add("garbage", "Constant", dict(width=97, height=71, red=0.6, green=0.6, blue=0.6, alpha=0.5))
                    g.add("back", "Constant", dict(width=97, height=71, red=0.2, green=0.4, blue=0.8, alpha=1.0))
                    g.add("node", "ScreenKeyer", dict(knobs, keyer_view=view, mix=0.9), image=plate, inside="garbage",
                          outside="garbage", bg="back")
                    np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_a_crop_goes_to_the_whole_image_path(self):
        from nodebased.ops2d_chrome import needs_full_frame
        self.assertFalse(needs_full_frame("ScreenKeyer", SPECS["ScreenKeyer"]["params"]))
        self.assertTrue(needs_full_frame("ScreenKeyer", dict(SPECS["ScreenKeyer"]["params"], source_crop_left=0.25)))

    def test_the_whole_image_path_crops_the_plate(self):
        g = Graph()
        plate = screen_graph(g, 97, 71)
        knobs = dict(source_crop_left=0.25, source_crop_top=0.8, source_x_method="wrap")
        g.add("node", "ScreenKeyer", knobs, image=plate)
        picture = evaluator_pixels(g.doc, plate)
        expect = Evaluator._kernel("ScreenKeyer", dict(SPECS["ScreenKeyer"]["params"], **knobs), [picture])
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), expect, atol=1e-6)

    def test_the_halo_includes_the_despot(self):
        from nodebased.tiers import Region, input_regions
        plain = input_regions("ScreenKeyer", {}, Region(20, 20, 30, 30), 6)
        self.assertEqual((plain[0].x, plain[0].width), (20, 30))
        padded = input_regions("ScreenKeyer", dict(screen_despot_black=4.0, screen_despot_white=2.0), Region(20, 20, 30, 30), 6)
        self.assertEqual((padded[0].x, padded[0].width), (20 - 6, 30 + 12))      # radius 2 + radius 1, two box filters each
        self.assertEqual((padded[4].x, padded[5].x), (20, 20))                   # mask and bg at the output region

    def test_bypass_passes_the_image(self):
        self.assertEqual(bypass_slot({"type": "ScreenKeyer", "inputs": {"image": "a", "bg": "b"}}), "image")


class HueKeyerTests(unittest.TestCase):
    def run_hue(self, pixel, **knobs):
        params = {**SPECS["HueKeyer"]["params"], "hue_center": 0.0, "hue_width": 60.0, "hue_softness": 0.0, **knobs}
        return Evaluator._kernel("HueKeyer", params, [one(*pixel)])[0, 0]

    def test_by_default_the_matte_goes_to_alpha(self):
        np.testing.assert_allclose(self.run_hue((0.9, 0.1, 0.1, 0.5)), [0.9, 0.1, 0.1, 1.0], atol=1e-6)

    def test_output_off_passes_the_picture_unchanged(self):
        out = self.run_hue((0.9, 0.1, 0.1, 0.5), output_enabled=0)
        np.testing.assert_array_equal(out, np.array([0.9, 0.1, 0.1, 0.5], np.float32))

    def test_the_matte_can_go_to_another_channel(self):
        np.testing.assert_allclose(self.run_hue((0.9, 0.1, 0.1, 0.5), output_channel="rgba.blue"), [0.9, 0.1, 1.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(self.run_hue((0.1, 0.9, 0.1, 0.5), output_channel="rgba.red"), [0.0, 0.9, 0.1, 0.5], atol=1e-6)
        self.assertEqual(float(self.run_hue((0.1, 0.9, 0.1, 0.5), output_channel="rgba.red", invert=1)[0]), 1.0)

    def test_old_documents_key_into_alpha(self):
        legacy = {k: v for k, v in SPECS["HueKeyer"]["params"].items() if k not in keyerknobs.NEW["HueKeyer"]}
        pixels = np.random.default_rng(1).random((6, 7, 4)).astype(np.float32)
        np.testing.assert_array_equal(Evaluator._kernel("HueKeyer", {**SPECS["HueKeyer"]["params"]}, [pixels]),
                                      Evaluator._kernel("HueKeyer", legacy, [pixels]))

    def test_tiles_match_the_evaluator(self):
        for knobs in (dict(output_channel="rgba.green"), dict(output_enabled=0)):
            g = Graph()
            plate = screen_graph(g, 97, 71)
            g.add("node", "HueKeyer", dict(hue_center=120.0, hue_width=90.0, **knobs), image=plate)
            np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
