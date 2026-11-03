"""2D behaviour, lane 9 step O: EdgeExtend, Defocus and the motion blurs get Nuke's missing knobs and defaults.

* Defocus quality smooths the disc's edge (1 to 5 is the hard disc the node always drew, higher gives each edge pixel the share the disc
  covers; the weights still sum to 1); method changes no pixels; a saved node keeps the hard disc bit for bit; tiles equal the full frame.
* EdgeExtend matte (none, source alpha, source inverted alpha), detail amount (0 leaves the smooth colour, 1 the colour as it was) and edge
  mask (the filled-in edge in the alpha); a saved node keeps its picture; tiles equal the full frame.
* MotionBlur2D and MotionBlur3D start at the guide's shutter 0.5 and start offset; saved nodes keep their own values.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, edgedefocusknobs, knobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import differs, evaluator_pixels, kern, old_document, source_graph, texture, tile_pixels


def impulse(n=31):
    picture = np.zeros((n, n, 4), np.float32)
    picture[n // 2, n // 2] = 1.0
    return picture


def soft_matte(h=14, w=18):
    """Colour on the left, a soft alpha edge across the middle columns, nothing on the right."""
    image = np.zeros((h, w, 4), np.float32)
    image[..., 0], image[..., 1], image[..., 2] = 0.8, 0.4, 0.2
    ramp = np.clip((np.arange(w) - 6) / 4.0, 0.0, 1.0)
    image[..., 3] = (1.0 - ramp)[None, :]
    image[..., :3] *= image[..., 3:4]
    return image


class DefocusQualityTests(unittest.TestCase):
    BASE = dict(defocus=6.0, legacy_defocus=0)

    def test_a_low_quality_is_the_hard_disc_of_a_saved_node(self):
        pic = np.clip(texture(), 0.0, 1.0)
        np.testing.assert_allclose(kern("Defocus", pic, quality=1.0, **self.BASE), kern("Defocus", pic, quality=1.0, legacy_defocus_disc=1, **self.BASE), atol=2e-6)

    def test_the_default_quality_gives_the_disc_a_soft_edge_and_the_weights_still_sum_to_one(self):
        hard = kern("Defocus", impulse(), legacy_defocus_disc=1, **self.BASE)[..., 0]
        soft = kern("Defocus", impulse(), quality=20.0, **self.BASE)[..., 0]
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(hard.sum()), 1.0, places=5)
        self.assertEqual(len(np.unique(np.round(hard[hard > 0], 7))), 1)                # a flat disc
        self.assertGreater(len(np.unique(np.round(soft[soft > 0], 7))), 2)               # edge pixels carry a share
        self.assertGreater(float(soft[15, 15]), 0.0)
        self.assertGreater(float(soft.max()), float(soft[soft > 0].min()))

    def test_a_higher_quality_gives_a_smoother_edge_up_to_the_sample_grid(self):
        low = kern("Defocus", impulse(), quality=10.0, **self.BASE)[..., 0]
        high = kern("Defocus", impulse(), quality=40.0, **self.BASE)[..., 0]
        self.assertTrue(differs(low, high))

    def test_quality_acts_and_method_does_not(self):
        pic = np.clip(texture(), 0.0, 1.0)
        self.assertTrue(differs(kern("Defocus", pic, quality=20.0), kern("Defocus", pic, quality=1.0)))
        np.testing.assert_array_equal(kern("Defocus", pic, defocus_method="accelerated"), kern("Defocus", pic, defocus_method="fullprecision"))

    def test_the_disc_is_still_a_disc_scaled_by_aspect_and_scale(self):
        wide = kern("Defocus", impulse(41), defocus=5.0, aspect=2.0, legacy_defocus=0)[..., 0]
        ys, xs = np.nonzero(wide > 1e-9)
        self.assertGreater(xs.max() - xs.min(), ys.max() - ys.min())

    def test_a_saved_node_keeps_its_hard_disc_bit_for_bit(self):
        loaded = old_document("Defocus", {"defocus": 4.0}, ("quality", "defocus_method", "legacy_defocus_disc", "scale"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual((params["legacy_defocus_disc"], params["quality"], params["defocus_method"]), (1, 20.0, "accelerated"))
        pic = np.clip(texture(), 0.0, 1.0)
        before = {k: v for k, v in params.items() if k not in ("quality", "defocus_method", "legacy_defocus_disc")}
        np.testing.assert_array_equal(Evaluator._defocus(pic, params), Evaluator._defocus(pic, before))

    def test_tiles_equal_the_full_frame_with_and_without_a_mask(self):
        for params in ({"defocus": 5.0}, {"defocus": 3.0, "aspect": 2.0, "quality": 35.0}):
            for masked in (False, True):
                doc = source_graph("Defocus", params, masked).doc
                full = evaluator_pixels(doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tile_pixels(doc, "node", edge), full, atol=1e-5)

    def test_main_tab_follows_the_guide(self):
        shown = [g.label for g in knobs.KNOB_LAYOUT["Defocus"] if g.kind != "legacy" and not g.params[0].startswith("legacy_")]
        self.assertEqual(shown[:6], ["Channels", "Defocus", "Aspect ratio", "Scaling", "Quality", "Method"])


class EdgeExtendTests(unittest.TestCase):
    BASE = dict(extend_size=4.0, extend_threshold=0.8, premultiply=0, source_premultiplied=1)

    def test_the_new_node_starts_at_the_guide_defaults(self):
        params = SPECS["EdgeExtend"]["params"]
        self.assertEqual((params["matte"], params["detail_amount"], params["edge_mask"]), ("Source Alpha", 1.0, 0))

    def test_no_matte_leaves_the_colour_alone_and_a_matte_extends_it(self):
        image = soft_matte()
        none = kern("EdgeExtend", image, matte="None", legacy_edge_extend=0, **self.BASE)
        alpha = kern("EdgeExtend", image, matte="Source Alpha", legacy_edge_extend=0, **self.BASE)
        straight = np.where(image[..., 3:4] > 0, image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0.0)
        np.testing.assert_allclose(none[..., :3], straight, atol=1e-6)
        self.assertTrue(differs(none, alpha))

    def test_the_inverted_alpha_trusts_the_other_side(self):
        image = soft_matte()
        straight = kern("EdgeExtend", image, matte="Source Alpha", legacy_edge_extend=0, **self.BASE)
        inverted = kern("EdgeExtend", image, matte="Source Inverted Alpha", legacy_edge_extend=0, **self.BASE)
        self.assertTrue(differs(straight, inverted))

    def test_detail_amount_zero_leaves_the_smooth_colour_and_one_the_colour_as_it_was(self):
        pic = np.clip(texture(h=16, w=20), 0.0, 1.0)
        pic[..., 3] = np.where(np.arange(20)[None, :] < 8, 1.0, 0.0)
        kept = kern("EdgeExtend", pic, detail_amount=1.0, legacy_edge_extend=0, **self.BASE)
        smooth = kern("EdgeExtend", pic, detail_amount=0.0, legacy_edge_extend=0, **self.BASE)
        half = kern("EdgeExtend", pic, detail_amount=0.5, legacy_edge_extend=0, **self.BASE)
        padded = np.pad(kept[..., :3].astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
        mean3 = sum(padded[dy:dy + 16, dx:dx + 20] for dy in range(3) for dx in range(3)) / 9.0
        np.testing.assert_allclose(smooth[..., :3], mean3, atol=1e-5)
        np.testing.assert_allclose(half[..., :3], 0.5 * (kept[..., :3] + smooth[..., :3]), atol=1e-5)

    def test_the_edge_mask_writes_the_filled_in_edge_to_alpha(self):
        image = soft_matte()
        off = kern("EdgeExtend", image, legacy_edge_extend=0, **self.BASE)
        on = kern("EdgeExtend", image, edge_mask=1, legacy_edge_extend=0, **self.BASE)
        np.testing.assert_allclose(off[..., 3], image[..., 3], atol=1e-7)
        self.assertTrue(set(np.unique(on[..., 3]).tolist()) <= {0.0, 1.0})
        self.assertGreater(float(on[..., 3].sum()), 0.0)
        filled = on[..., 3] > 0
        self.assertTrue(bool((image[..., 3][filled] < 0.8).all()))                  # only pixels the matte did not trust are filled
        np.testing.assert_allclose(on[..., :3], off[..., :3], atol=1e-7)

    def test_every_new_knob_acts_on_the_default_node(self):
        image = soft_matte()
        base = dict(legacy_edge_extend=0, extend_threshold=0.8)
        for knob, value in (("matte", "None"), ("detail_amount", 0.0), ("edge_mask", 1)):
            with self.subTest(knob=knob):
                self.assertTrue(differs(kern("EdgeExtend", image, **base), kern("EdgeExtend", image, **{**base, knob: value})))

    def test_a_saved_node_keeps_its_picture(self):
        loaded = old_document("EdgeExtend", {"extend_size": 3.0}, ("matte", "detail_amount", "edge_mask", "source_premultiplied", "premultiply", "erode",
                                                                   "legacy_edge_extend"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual((params["matte"], params["detail_amount"], params["edge_mask"], params["legacy_edge_extend"]), ("Source Alpha", 1.0, 0, 1))
        image = soft_matte()
        older = {k: v for k, v in params.items() if k not in ("matte", "detail_amount", "edge_mask")}
        np.testing.assert_array_equal(Evaluator._edge_extend(image, params), Evaluator._edge_extend(image, older))

    def test_tiles_equal_the_full_frame_with_and_without_a_mask(self):
        for params in ({"legacy_edge_extend": 0, "detail_amount": 0.3, "extend_threshold": 0.4}, {"legacy_edge_extend": 0, "matte": "Source Inverted Alpha"},
                       {"legacy_edge_extend": 0, "edge_mask": 1, "extend_threshold": 0.4}):
            for masked in (False, True):
                doc = source_graph("EdgeExtend", params, masked).doc
                full = evaluator_pixels(doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tile_pixels(doc, "node", edge), full, atol=1e-5)

    def test_main_tab_follows_the_guide(self):
        shown = [g.label for g in knobs.KNOB_LAYOUT["EdgeExtend"] if g.kind != "legacy" and not g.params[0].startswith("legacy_")]
        self.assertEqual(shown[:6], ["Channels", "Matte", "Source Is Premultiplied", "Erode", "Detail amount", "Edge mask"])


class MotionBlurDefaultTests(unittest.TestCase):
    def test_a_new_node_starts_at_the_guide_values(self):
        for kind in ("MotionBlur2D", "MotionBlur3D"):
            with self.subTest(kind=kind):
                params = SPECS[kind]["params"]
                self.assertEqual((params["shutter"], params["shutter_offset"]), (0.5, "start"))

    def test_a_saved_node_keeps_its_own_shutter(self):
        for kind in ("MotionBlur2D", "MotionBlur3D"):
            with self.subTest(kind=kind):
                loaded = old_document(kind, {"shutter": 1.0, "shutter_offset": "centred", "custom_offset": 0.25}, ())
                params = loaded["nodes"]["n"]["params"]
                self.assertEqual((params["shutter"], params["shutter_offset"], params["custom_offset"]), (1.0, "centred", 0.25))


class LimitsTests(unittest.TestCase):
    def test_every_new_param_has_limits_or_choices_and_is_laid_out_once(self):
        new = {"EdgeExtend": ("matte", "detail_amount", "edge_mask"), "Defocus": ("quality", "defocus_method", "legacy_defocus_disc")}
        for kind, names in new.items():
            placed = [name for g in knobs.KNOB_LAYOUT[kind] for name in g.params]
            for name in names:
                with self.subTest(kind=kind, name=name):
                    self.assertEqual(placed.count(name), 1)
                    self.assertIn(name, SPECS[kind]["params"])
                    self.assertTrue(name in core.NODE_LIMITS.get(kind, {}) or name in core.LIMITS or name in core.CHOICES)


if __name__ == "__main__":
    unittest.main()
