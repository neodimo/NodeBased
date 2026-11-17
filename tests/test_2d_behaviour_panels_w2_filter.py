"""2D behaviour, lane 9 step W2: the rest of the Filter and Draw panel rows the review book's worklist listed.

Text's mask strip sits after Opacity and DirBlur's before Mix as the guide's tables have them; Sparkles, Flare, MarkerRemoval and Denoise use the guide's
labels; DustBust's Enable is the effect's intensity from 0 to 1; VectorBlur starts at motion amount 0, offset -0.5 and no uv channels; Glint gets the
aspect ratio, w channel and gamma knobs. A node saved before keeps its pixels.
"""
import copy
import json
import os
import sys
import unittest

import numpy as np

from nodebased import core, knobs
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, kern, render, texture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def labels(kind):
    return [group.label for group in knobs.KNOB_LAYOUT[kind] if group.label]


def label_of(kind, param):
    return next(group.label for group in knobs.KNOB_LAYOUT[kind] if group.params[0] == param)


def position(kind, param):
    return next(i for i, group in enumerate(knobs.KNOB_LAYOUT[kind]) if param in group.params)


class LayoutTests(unittest.TestCase):
    def test_text_puts_the_mask_strip_after_opacity(self):
        opacity = position("Text", "opacity")
        strip = [position("Text", n) for n in ("maskChannelInput", "inject", "invert_mask")]
        self.assertEqual(strip, [opacity + 1, opacity + 2, opacity + 3])
        self.assertEqual(position("Text", "message"), opacity + 4)

    def test_dirblur_puts_the_mask_strip_before_mix(self):
        strip = [position("DirBlur", n) for n in ("maskChannelInput", "inject", "invert_mask", "fringe")]
        self.assertEqual(strip, sorted(strip))
        self.assertEqual(position("DirBlur", "mix"), strip[-1] + 1)
        self.assertEqual(position("DirBlur", "pixeloffset"), strip[0] - 1)

    def test_the_labels_are_the_guides(self):
        for kind, param, label in (("Sparkles", "spark_random_thick", "Random"), ("Sparkles", "random_broken_start", "Random"),
                                   ("Sparkles", "random_broken_holes", "Random"), ("Flare", "flare_chroma_angle_add", "Rel"),
                                   ("Flare", "corners_angle_add", "Rel"), ("MarkerRemoval", "x_offset", "X offset"),
                                   ("MarkerRemoval", "y_offset", "Y offset"), ("Denoise", "denoise_profile_low_gain", "Low gain"),
                                   ("Denoise", "denoise_profile_mid_gain", "Mid gain"), ("Denoise", "denoise_profile_high_gain", "High gain"),
                                   ("Glint", "glint_aspect", "Aspect ratio"), ("Glint", "glint_gamma", "Gamma")):
            with self.subTest(kind=kind, param=param):
                self.assertEqual(label_of(kind, param), label)

    def test_glint_follows_the_guides_order(self):
        order = [position("Glint", n) for n in ("rays", "length", "glint_aspect", "glint_oddlen", "rotation", "glint_effect_only",
                                                "glint_w_channel", "tolerance", "glint_gamma")]
        self.assertEqual(order, sorted(order))

    def test_use_input_channels_is_a_check_box_in_the_reference_data(self):
        with open(os.path.join(ROOT, "docs", "parity", "nuke_nodes.json"), encoding="utf-8") as handle:
            nodes = json.load(handle)["nodes"]
        for node in ("Convolve", "ZDefocus"):
            knob = next(k for k in nodes[node]["knobs"] if k["name"] == "use_input_channels")
            self.assertEqual(knob["type"], "bool")
            self.assertIs(knob["default"], False)

    def test_the_review_book_finds_no_panel_mismatch_in_these_nodes(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import parity_review_book as book
        with open(os.path.join(ROOT, "docs", "parity", "nuke_nodes.json"), encoding="utf-8") as handle:
            nuke = json.load(handle)["nodes"]
        with open(os.path.join(ROOT, "docs", "parity", "nodebased_to_nuke.json"), encoding="utf-8") as handle:
            pairs = json.load(handle)["pairs"]
        for node in ("Text", "DirBlur", "VectorBlur", "MarkerRemoval", "Sparkles", "Flare", "Denoise", "DustBust", "Convolve", "ZDefocus", "Glint"):
            with self.subTest(node=node):
                rows = book.alignment(nuke[node], pairs[node], pairs[node]["nodebased"], node)
                self.assertEqual([(r["name"], r["marks"]) for r in rows if r["marks"]], [])


class DustBustEnableTests(unittest.TestCase):
    """Nuke's enable is the effect's intensity, 0 to 1."""

    def document(self, **params):
        d = Dispatcher(empty_document())
        d.execute({"op": "time", "first": 1, "last": 2, "current": 2})
        d.execute({"op": "create", "id": "a", "type": "Constant", "params": {"width": 24, "height": 24, "red": 0, "green": 0, "blue": 0, "alpha": 1}})
        d.execute({"op": "set_curve", "id": "a", "param": "red",
                   "curve": {"interpolation": "linear", "keys": [{"frame": 1, "value": 0.2}, {"frame": 2, "value": 0.8}]}})
        d.execute({"op": "create", "id": "b", "type": "DustBust", "params": {"frame_offset": -1, "edge_hardness": 1, **params}})
        d.execute({"op": "set_dustbust_spots", "id": "b", "spots": [{"frame": 2, "x": 12, "y": 12, "radius": 4}]})
        d.execute({"op": "connect", "id": "b", "input": "image", "source": "a"})
        return d.document

    def pixels(self, document, key="b"):
        return Evaluator().evaluate_raster(document, key, frame=2).pixels

    def test_enable_is_a_zero_to_one_slider_on_by_default(self):
        self.assertEqual(SPECS["DustBust"]["params"]["enable"], 1.0)
        self.assertEqual(core.LIMITS["enable"], (0, 1))
        group = next(g for g in knobs.KNOB_LAYOUT["DustBust"] if g.params == ("enable",))
        self.assertEqual(group.kind, "float_slider")

    def test_zero_passes_the_picture_and_one_is_the_full_effect(self):
        plate = self.pixels(self.document(), "a")
        np.testing.assert_array_equal(self.pixels(self.document(enable=0.0)), plate)
        full = self.pixels(self.document())
        self.assertTrue(differs(full, plate))

    def test_a_middle_value_is_halfway_between(self):
        plate = self.pixels(self.document(), "a")
        full = self.pixels(self.document())
        half = self.pixels(self.document(enable=0.5))
        self.assertTrue(differs(full, plate))
        np.testing.assert_allclose(half, plate + (full - plate) * 0.5, atol=1e-6)

    def test_a_node_saved_with_the_old_switch_keeps_its_pixels(self):
        document = self.document()
        expected = self.pixels(document)
        for saved_value, reference in ((1, expected), (0, self.pixels(document, "a"))):
            with self.subTest(saved=saved_value):
                saved = copy.deepcopy(document)
                saved["nodes"]["b"]["params"]["enable"] = saved_value
                loaded = upgrade_document(saved)
                validate(loaded)
                np.testing.assert_array_equal(self.pixels(loaded), reference)


class VectorBlurDefaultTests(unittest.TestCase):
    def test_a_new_node_starts_as_the_guide_lists(self):
        params = SPECS["VectorBlur"]["params"]
        self.assertEqual((params["vector_scale"], params["vector_offset"], params["uv_layer"]), (0.0, -0.5, ""))

    def test_the_pairing_reads_an_empty_uv_layer_as_none(self):
        with open(os.path.join(ROOT, "docs", "parity", "nodebased_to_nuke.json"), encoding="utf-8") as handle:
            pair = json.load(handle)["pairs"]["VectorBlur"]
        self.assertEqual(pair["value_map"], {"uv": {"": "none"}})

    def vectors(self, **params):
        g = Graph()
        g.add("plate", "Checker", dict(width=40, height=30, size=4))
        g.add("vec", "Constant", dict(width=40, height=30, red=4.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("n", "VectorBlur", dict(uv_layer="rgba", **params), image="plate", uv="vec")
        return g

    def test_a_new_node_does_nothing_until_the_motion_amount_is_set(self):
        g = self.vectors()
        np.testing.assert_array_equal(render(g.doc, "n"), render(g.doc, "plate"))
        self.assertTrue(differs(render(self.vectors(vector_scale=1.0).doc, "n"), render(g.doc, "plate")))

    def test_a_node_saved_at_motion_amount_one_keeps_blurring(self):
        g = self.vectors(vector_scale=1.0, vector_offset=0.0)
        loaded = upgrade_document(copy.deepcopy(g.doc))
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["vector_scale"], 1.0)
        np.testing.assert_array_equal(render(loaded, "n"), render(g.doc, "n"))


class GlintKnobTests(unittest.TestCase):
    def impulse(self, n=61, value=2.0):
        pixels = np.zeros((n, n, 4), np.float32)
        pixels[n // 2, n // 2] = value
        pixels[..., 3] = 1.0
        return pixels

    def glint(self, pixels, **knobs_):
        return kern("Glint", pixels, rays=2, length=20.0, tolerance=0.5, rotation=90.0, glint_steps=20, **knobs_)

    def extent(self, picture, axis):
        lit = np.argwhere(picture[..., 0] > 0.2)
        return int(lit[:, axis].max() - lit[:, axis].min())

    def test_the_new_knobs_start_where_the_guide_lists_them(self):
        params = SPECS["Glint"]["params"]
        self.assertEqual((params["glint_aspect"], params["glint_gamma"], params["glint_w_channel"]), (1.0, 1.0, "none"))

    def test_aspect_ratio_stretches_the_star_sideways(self):
        pixels = self.impulse()
        plain, wide = self.glint(pixels), self.glint(pixels, glint_aspect=2.0)
        self.assertGreater(self.extent(wide, 1), self.extent(plain, 1) * 1.7)
        self.assertEqual(self.extent(wide, 0), self.extent(plain, 0))

    def test_gamma_corrects_the_highlights_before_the_star_is_made(self):
        pixels = self.impulse(value=0.9)                                                      # a highlight of 0.4 over the tolerance
        plain = self.glint(pixels, glint_effect_only=1)
        bright = self.glint(pixels, glint_gamma=2.0, glint_effect_only=1)
        dark = self.glint(pixels, glint_gamma=0.5, glint_effect_only=1)
        self.assertGreater(bright[..., 0].max(), plain[..., 0].max())
        self.assertLess(dark[..., 0].max(), plain[..., 0].max())
        np.testing.assert_allclose(bright[..., 0].max(), 0.4 ** 0.5, rtol=1e-4)            # the highlight, 0.9 - 0.5, to the power 1 / 2

    def test_the_w_channel_gates_which_highlights_are_sources(self):
        pixels = self.impulse()
        pixels[..., 3] = 0.0
        shut = self.glint(pixels, glint_w_channel="rgba.alpha", glint_effect_only=1)
        self.assertFalse(np.any(shut[..., :3]))
        pixels[30, 30, 3] = 1.0
        open_ = self.glint(pixels, glint_w_channel="rgba.alpha", glint_effect_only=1)
        self.assertTrue(np.any(open_[..., :3]))
        none = self.glint(pixels, glint_w_channel="none", glint_effect_only=1)
        self.assertTrue(differs(none, shut))

    def test_a_node_saved_before_keeps_its_pixels(self):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("n", "Glint", {"rays": 3, "length": 12.0, "tolerance": 0.1}, image="src")
        new = render(g.doc, "n")
        old = copy.deepcopy(g.doc)
        for name in ("glint_aspect", "glint_gamma", "glint_w_channel"):
            old["nodes"]["n"]["params"].pop(name)
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["glint_aspect"], 1.0)
        np.testing.assert_array_equal(render(loaded, "n"), new)


if __name__ == "__main__":
    unittest.main()
