"""2D behaviour, lane 9 step O finish: the inputs and Emboss knobs step O left out.

* BumpBoss input B: A's slope lights B (the picture being embossed); a flat A leaves B as it was; B's alpha is kept; the mix dissolves toward B;
  B unconnected is the one-input node as before; tiles equal the frame with and without a mask.
* EdgeExtend Matte input: the four `Matte ...` choices read its luminance and alpha (and their inversions); unconnected it is black;
  a saved node (no Matte wire) renders as before; tiles equal the frame with and without a mask.
* Emboss Edge Channels (the channels the slope is read from), Blur size and Blur quality (a pre-blur): each changes pixels in the stated
  direction; the defaults are the picture a saved node drew; tiles equal the frame.
* MotionBlur2D 2D transf input: the picture is blurred along the motion of the Transform connected there, for the shutter the node states;
  a node that is not a Transform or CornerPin is refused.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs
from nodebased.core import CHOICES, SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_edge_defocus_o import soft_matte
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, old_document, source_graph, texture, tile_pixels


def kernel(kind, slots, **knob_values):
    return np.asarray(Evaluator._kernel(kind, {**SPECS[kind]["params"], **knob_values}, slots), np.float64)


def ramp_picture(h=24, w=30):
    """A grey ramp across the columns, alpha 1."""
    image = np.ones((h, w, 4), np.float32)
    image[..., :3] = np.linspace(0.2, 0.8, w, dtype=np.float32)[None, :, None]
    return image


def flat_picture(value=0.5, h=24, w=30):
    image = np.ones((h, w, 4), np.float32)
    image[..., :3] = value
    return image


class BumpBossInputBTests(unittest.TestCase):
    BASE = dict(legacy_bumpboss=0, height_channel="rgba.red")

    def test_the_node_has_a_b_slot_after_the_mask(self):
        self.assertEqual(SPECS["BumpBoss"]["optional_inputs"], ["mask", "b"])

    def test_a_flat_a_leaves_b_as_it_was(self):
        b = np.clip(texture(24, 30), 0.0, 1.0)
        lit = kernel("BumpBoss", [flat_picture(), None, b], **self.BASE)
        np.testing.assert_allclose(lit, b, atol=1e-6)

    def test_the_slope_of_a_lights_b_and_b_alpha_is_kept(self):
        b = flat_picture(0.6)
        b[..., 3] = 0.7
        lit = kernel("BumpBoss", [ramp_picture(), None, b], **self.BASE)
        self.assertTrue(differs(lit[..., :3], b[..., :3]))
        np.testing.assert_allclose(lit[..., 3], 0.7, atol=1e-6)
        factor = lit[10, 10, 0] / 0.6                                  # one light factor on all three channels
        np.testing.assert_allclose(lit[10, 10, :3], 0.6 * factor, atol=1e-6)

    def test_b_unconnected_is_a_lit_by_its_own_slope(self):
        a = np.clip(texture(24, 30), 0.0, 1.0)
        np.testing.assert_array_equal(kernel("BumpBoss", [a, None, None], **self.BASE), kernel("BumpBoss", [a, None, a], **self.BASE))

    def test_a_decides_the_shading_and_b_only_carries_it(self):
        a = ramp_picture()
        lit_dark = kernel("BumpBoss", [a, None, flat_picture(0.2)], **self.BASE)[..., 0] / 0.2
        lit_bright = kernel("BumpBoss", [a, None, flat_picture(0.8)], **self.BASE)[..., 0] / 0.8
        np.testing.assert_allclose(lit_dark, lit_bright, atol=1e-5)

    def test_the_mix_dissolves_toward_b(self):
        b = flat_picture(0.6)
        np.testing.assert_allclose(kernel("BumpBoss", [ramp_picture(), None, b], mix=0.0, **self.BASE), b, atol=1e-6)

    def test_a_saved_node_ignores_b_and_keeps_its_grey_emboss(self):
        a = np.clip(texture(24, 30), 0.0, 1.0)
        np.testing.assert_array_equal(kernel("BumpBoss", [a, None, None], legacy_bumpboss=1),
                                      kernel("BumpBoss", [a, None, flat_picture(0.3)], legacy_bumpboss=1))

    def graph(self, masked, **params):
        g = Graph()
        g.add("a", "Checker", dict(width=45, height=37, size=5))
        g.add("b", "Checker", dict(width=45, height=37, size=9))
        g.add("shade", "Grade", dict(exposure=-0.5), image="b")
        wires = dict(image="a", b="shade")
        if masked:
            g.add("gate", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
            wires["mask"] = "gate"
        g.add("node", "BumpBoss", dict(self.BASE, **params), **wires)
        return g

    def test_b_is_lit_on_the_evaluator_and_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for masked in (False, True):
            g = self.graph(masked, mix=0.9, bumpsize=3.0)
            whole = evaluator_pixels(g.doc, "node")
            self.assertTrue(differs(whole, evaluator_pixels(g.doc, "shade")))
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)


class EdgeExtendMatteInputTests(unittest.TestCase):
    BASE = dict(legacy_edge_extend=0, extend_size=4.0, extend_threshold=0.5)

    @staticmethod
    def matte_from(plane, as_alpha):
        """A picture whose alpha (or grey luminance) carries `plane`."""
        pic = np.zeros(plane.shape + (4,), np.float32)
        if as_alpha:
            pic[..., 3] = plane
        else:
            pic[..., :3] = plane[..., None]
            pic[..., 3] = 1.0
        return pic

    @staticmethod
    def wide_plane():
        plane = np.zeros(soft_matte().shape[:2], np.float32)
        plane[:, :11] = 1.0                                              # a wider matte than the source's own
        return plane

    def test_the_node_has_a_matte_slot_after_the_mask_and_offers_the_four_matte_choices(self):
        self.assertEqual(SPECS["EdgeExtend"]["optional_inputs"], ["mask", "matte"])
        for choice in ("Matte Luminance", "Matte Inverted Luminance", "Matte Alpha", "Matte Inverted Alpha"):
            self.assertIn(choice, CHOICES["matte"])

    def test_matte_alpha_and_matte_luminance_read_the_input(self):
        image, plane = soft_matte(), self.wide_plane()
        by_alpha = kernel("EdgeExtend", [image, None, self.matte_from(plane, True)], matte="Matte Alpha", **self.BASE)
        by_luma = kernel("EdgeExtend", [image, None, self.matte_from(plane, False)], matte="Matte Luminance", **self.BASE)
        own = kernel("EdgeExtend", [image, None, None], matte="Source Alpha", **self.BASE)
        np.testing.assert_allclose(by_alpha, by_luma, atol=1e-6)
        self.assertTrue(differs(by_alpha, own))

    def test_a_matte_equal_to_the_source_alpha_is_source_alpha(self):
        image = soft_matte()
        np.testing.assert_allclose(kernel("EdgeExtend", [image, None, self.matte_from(image[..., 3], True)], matte="Matte Alpha", **self.BASE),
                                   kernel("EdgeExtend", [image, None, None], matte="Source Alpha", **self.BASE), atol=1e-6)

    def test_the_inverted_choices_use_the_inverse_of_the_matte(self):
        image, plane = soft_matte(), self.wide_plane()
        for as_alpha, inverted, plain in ((True, "Matte Inverted Alpha", "Matte Alpha"), (False, "Matte Inverted Luminance", "Matte Luminance")):
            np.testing.assert_allclose(kernel("EdgeExtend", [image, None, self.matte_from(plane, as_alpha)], matte=inverted, **self.BASE),
                                       kernel("EdgeExtend", [image, None, self.matte_from(1.0 - plane, as_alpha)], matte=plain, **self.BASE), atol=1e-6)
        self.assertTrue(differs(kernel("EdgeExtend", [image, None, self.matte_from(plane, True)], matte="Matte Alpha", **self.BASE),
                                kernel("EdgeExtend", [image, None, self.matte_from(plane, True)], matte="Matte Inverted Alpha", **self.BASE)))

    def test_an_unconnected_matte_reads_as_black(self):
        image = soft_matte()
        black = self.matte_from(np.zeros(image.shape[:2], np.float32), True)
        np.testing.assert_array_equal(kernel("EdgeExtend", [image, None, None], matte="Matte Alpha", **self.BASE),
                                      kernel("EdgeExtend", [image, None, black], matte="Matte Alpha", **self.BASE))

    def test_the_source_choices_ignore_the_matte_input(self):
        image = soft_matte()
        any_matte = self.matte_from(np.full(image.shape[:2], 0.3, np.float32), True)
        for choice in ("Source Alpha", "Source Inverted Alpha", "None"):
            np.testing.assert_array_equal(kernel("EdgeExtend", [image, None, any_matte], matte=choice, **self.BASE),
                                          kernel("EdgeExtend", [image, None, None], matte=choice, **self.BASE))

    def test_a_saved_node_is_unchanged_by_the_slot(self):
        loaded = old_document("EdgeExtend", {"extend_size": 3.0}, ("matte", "detail_amount", "edge_mask", "source_premultiplied", "premultiply", "erode",
                                                                   "legacy_edge_extend"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual(params["matte"], "Source Alpha")
        image = texture()
        np.testing.assert_array_equal(Evaluator._edge_extend(image, params), Evaluator._edge_extend(image, params, None))

    def graph(self, masked, **params):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("shade", "Multiply", dict(channels="alpha", multiply=0.2), image="plate")                        # alpha below the threshold: nothing trusted
        g.add("pattern", "Checker", dict(width=45, height=37, size=11))
        wires = dict(image="shade", matte="pattern")
        if masked:
            g.add("gate", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
            wires["mask"] = "gate"
        g.add("node", "EdgeExtend", dict(self.BASE, **params), **wires)
        return g

    def test_the_matte_wire_changes_the_picture_and_tiles_equal_the_frame_with_and_without_a_mask(self):
        for masked in (False, True):
            g = self.graph(masked, matte="Matte Luminance", mix=0.9, extend_threshold=0.2)
            whole = evaluator_pixels(g.doc, "node")
            self.assertTrue(differs(whole, evaluator_pixels(self.graph(masked, matte="Source Alpha", mix=0.9, extend_threshold=0.2).doc, "node")))
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)


class EmbossEdgeKnobTests(unittest.TestCase):
    BASE = dict(emboss_type="effect", width=1.0, angle=0.0)

    def test_the_defaults_are_the_picture_a_saved_node_drew(self):
        pic = np.clip(texture(), 0.0, 1.0)
        loaded = old_document("Emboss", {"angle": 30.0, "width": 2.0}, ("emboss_type", "edge_detector", "edge_channels", "blur_size", "blur_quality"))
        params = loaded["nodes"]["n"]["params"]
        self.assertEqual((params["edge_channels"], params["blur_size"], params["blur_quality"]), ("rgb", 0.0, 15.0))
        older = {k: v for k, v in params.items() if k not in ("edge_channels", "blur_size", "blur_quality")}
        np.testing.assert_array_equal(Evaluator._emboss(pic, params), Evaluator._emboss(pic, older))

    def test_the_edge_channels_decide_which_brightness_gives_the_relief(self):
        pic = np.ones((20, 24, 4), np.float32)
        pic[..., :3] = 0.5
        pic[:, 12:, 0] = 0.9                                              # an edge in red only
        by_rgb = kernel("Emboss", [pic, None], edge_channels="rgb", **self.BASE)
        by_red = kernel("Emboss", [pic, None], edge_channels="red", **self.BASE)
        self.assertGreater(float(np.abs(by_red[..., 0] - 0.5).max()), float(np.abs(by_rgb[..., 0] - 0.5).max()))     # red alone sees the whole step
        self.assertTrue(differs(by_rgb, by_red))
        for choice in ("green", "blue", "alpha", "none"):
            with self.subTest(choice=choice):
                np.testing.assert_allclose(kernel("Emboss", [pic, None], edge_channels=choice, **self.BASE)[..., 0], 0.5, atol=1e-7)

    def test_the_blur_size_smooths_the_relief(self):
        pic = np.ones((24, 30, 4), np.float32)
        pic[..., :3] = 0.2
        pic[:, 15:, :3] = 0.8
        sharp = kernel("Emboss", [pic, None], blur_size=0.0, **self.BASE)[..., 0]
        soft = kernel("Emboss", [pic, None], blur_size=4.0, **self.BASE)[..., 0]
        self.assertTrue(differs(sharp, soft))
        self.assertLess(float(np.abs(soft - 0.5).max()), float(np.abs(sharp - 0.5).max()))
        self.assertGreater(int((np.abs(soft[10] - 0.5) > 1e-4).sum()), int((np.abs(sharp[10] - 0.5) > 1e-4).sum()))       # the relief spreads wider

    def test_the_blur_quality_changes_a_wide_blur(self):
        pic = np.clip(texture(40, 48), 0.0, 1.0)
        full = kernel("Emboss", [pic, None], blur_size=12.0, blur_quality=100.0, **self.BASE)
        scaled = kernel("Emboss", [pic, None], blur_size=12.0, blur_quality=4.0, **self.BASE)
        self.assertTrue(differs(full, scaled))

    def test_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for masked in (False, True):
            g = source_graph("Emboss", dict(self.BASE, blur_size=3.0, edge_channels="red", edge_detector="sobel", mix=0.9), masked)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)

    def test_the_panel_lists_the_new_knobs(self):
        shown = [g.params[0] for g in knobs.KNOB_LAYOUT["Emboss"]]
        for name in ("edge_channels", "blur_size", "blur_quality"):
            self.assertIn(name, shown)


class MotionBlur2DTransfTests(unittest.TestCase):
    SHIFT = 8.0

    def graph(self, with_transf=True, shutter=1.0, offset="start", **params):
        g = Graph()
        g.add("src", "Checker", dict(width=64, height=32, size=8))
        g.add("tex", "Transform", dict(translate_x=0.0, filter="bilinear"), image="src")
        g.d.execute(dict(op="set_key", id="tex", param="translate_x", frame=1, value=0.0))
        g.d.execute(dict(op="set_key", id="tex", param="translate_x", frame=2, value=self.SHIFT))
        g.d.execute(dict(op="set_key", id="tex", param="translate_x", frame=3, value=3 * self.SHIFT))     # the speed changes, so the two windows differ
        wires = dict(image="tex")
        if with_transf:
            wires["transf"] = "tex"
        g.add("node", "MotionBlur2D", dict(shutter=shutter, shutter_offset=offset, samples=8, **params), **wires)
        return g

    @staticmethod
    def render(g, frame=2):
        return Evaluator().evaluate(dict(g.doc, view="node"), "node", frame=frame)

    def test_the_node_has_a_transf_slot_after_the_mask(self):
        self.assertEqual(SPECS["MotionBlur2D"]["optional_inputs"], ["mask", "transf"])

    def test_the_picture_is_blurred_along_the_transforms_motion(self):
        blurred = self.render(self.graph())
        sharp = self.render(self.graph(shutter=0.0001))
        self.assertTrue(differs(blurred, sharp, 1e-3))
        # the checker's edges are vertical and the transform moves sideways: rows inside one checker band stay equal
        np.testing.assert_allclose(blurred[2], blurred[5], atol=1e-5)
        self.assertEqual(int((np.abs(blurred - sharp) > 1e-3).any(axis=(0, 2)).sum() > 0), 1)

    def test_the_shutter_offset_puts_the_blur_before_or_after_the_frame(self):
        sharp = self.render(self.graph(shutter=0.0001))
        start = self.render(self.graph(offset="start"))
        end = self.render(self.graph(offset="end"))
        self.assertTrue(differs(start, end, 1e-3))
        self.assertTrue(differs(start, sharp, 1e-3))
        self.assertTrue(differs(end, sharp, 1e-3))

    def test_a_longer_shutter_blurs_more(self):
        sharp = self.render(self.graph(shutter=0.0001))
        short = self.render(self.graph(shutter=0.25))
        long_ = self.render(self.graph(shutter=1.0))
        self.assertGreater(float(np.abs(long_ - sharp).mean()), float(np.abs(short - sharp).mean()))

    def test_a_transform_that_does_not_move_gives_the_picture(self):
        g = Graph()
        g.add("src", "Checker", dict(width=64, height=32, size=8))
        g.add("tex", "Transform", dict(translate_x=5.0, filter="bilinear"), image="src")
        g.add("node", "MotionBlur2D", dict(shutter=1.0, shutter_offset="centred", samples=8), image="tex", transf="tex")
        np.testing.assert_allclose(self.render(g, 1), evaluator_pixels(g.doc, "tex"), atol=1e-4)

    def test_the_matrix_blur_agrees_with_sampling_the_moving_picture(self):
        # Without the wire the node blends the animated picture across the shutter; with it the same motion comes from the transform's
        # matrix. For a pure translation the two give the same picture to within the resampling.
        # (The picture the wire carries is the already-moved one, so the strip along the left edge, where earlier frames still held picture, is
        # left out of the comparison.)
        np.testing.assert_allclose(self.render(self.graph(with_transf=False))[:, 24:], self.render(self.graph(with_transf=True))[:, 24:], atol=5e-2)

    def test_the_shutter_knobs_act_across_frames_without_the_wire_too(self):
        # shutter, shutter offset and custom offset are NOT_PIXEL on one frame: here the animated picture is read at several frames.
        sharp = self.render(self.graph(with_transf=False, shutter=0.0001))
        short = self.render(self.graph(with_transf=False, shutter=0.25))
        long_ = self.render(self.graph(with_transf=False, shutter=1.0))
        self.assertGreater(float(np.abs(long_ - sharp).mean()), float(np.abs(short - sharp).mean()))
        start = self.render(self.graph(with_transf=False, offset="start"))
        end = self.render(self.graph(with_transf=False, offset="end"))
        custom = self.render(self.graph(with_transf=False, offset="custom", custom_offset=-1.0))
        self.assertTrue(differs(start, end, 1e-3))
        self.assertTrue(differs(custom, start, 1e-3) or differs(custom, end, 1e-3))

    def test_a_node_that_is_not_a_transform_is_refused(self):
        g = Graph()
        g.add("src", "Checker", dict(width=32, height=16, size=4))
        g.add("soft", "Blur", dict(radius=2.0), image="src")
        g.add("node", "MotionBlur2D", dict(shutter=1.0, samples=4), image="src", transf="soft")
        with self.assertRaises(ValueError):
            self.render(g)


if __name__ == "__main__":
    unittest.main()
