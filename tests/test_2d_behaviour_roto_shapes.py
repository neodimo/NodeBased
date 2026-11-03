"""2D behaviour, lane 9 step I: Roto's per-shape controls act, shown by pixels.

The guide's Shape, Transform and Lifetime tabs (blending mode, colour, invert, visible, the transform, open splines, lifetime) live in each
shape's payload, so these tests build shapes in the payload and read the matte the evaluator and the tile path draw.

* Blending: `minus` is the guide's A - B (the shape taken from what is below); graphs saved before carried B - A, which is `from`, and keep
  it through `legacy_roto_shapes`.
* Lifetime: a shape is absent outside its range (the one-frame pixel probe cannot see this, so Roto stays NOT_PROBED there).
* The tile path equals the evaluator for every new mode, a skewed and rotated shape and an open spline, with and without a mask.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import shapes
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_behaviour_roto import N, box, pt


def build(shape_list, mask=False, **params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": N, "height": N, **params}})
    if mask:
        d.execute({"op": "create", "id": "m", "type": "Ramp", "params": {"width": N, "height": N}})
        d.execute({"op": "connect", "id": "r", "input": "mask", "source": "m"})
    d.execute({"op": "set_shapes", "id": "r", "shapes": shape_list})
    d.execute({"op": "view", "id": "r"})
    return d.document


def draw(doc, frame=None):
    return Evaluator().evaluate_raster(doc, target="r", frame=frame).pixels


def tiles(doc, edge):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
    region = executor.canvas_region(doc, "r", frame=1, tier=1)
    return executor.compose_region(doc, "r", region, frame=1, tier=1).pixels


def line(**extra):
    shape = {"name": "line", "mode": "union", "opacity": 1.0, "feather": 0.0, "feather_model": "edge", "closed": False,
             "stroke_width": 4.0, "points": [pt(6, 12), pt(18, 12)]}
    shape.update(extra)
    return shape


class BlendingTests(unittest.TestCase):
    below = box(2, 2, 14, 14, opacity=0.5)

    def two(self, mode, **params):
        return build([self.below, box(8, 8, 20, 20, name="top", blend_mode=mode)], **params)

    def test_every_mode_changes_the_picture_it_is_given(self):
        # With a shape that covers a pixel fully its matte is op(1, B), which for max and screen is 1 (the same as over); their colour differs.
        def picture(mode):
            return draw(build([box(2, 2, 14, 14, color=[0.8, 0.2, 0.4, 1.0], opacity=0.5),
                               box(8, 8, 20, 20, name="top", color=[0.3, 0.6, 0.5, 1.0], blend_mode=mode)], output="rgba"))
        base = picture("over")
        for mode in ("minus", "from", "difference", "exclusion", "max", "min", "screen"):
            with self.subTest(mode=mode):
                self.assertTrue((picture(mode) != base).any())

    def test_minus_is_the_shape_taken_from_what_is_below_and_from_the_reverse(self):
        # Inside the top shape and outside the one below, B = 0: minus gives A (1), from gives B - A (below zero, so 0).
        minus, from_ = draw(self.two("minus")), draw(self.two("from"))
        self.assertEqual(float(minus[17, 17, 3]), 1.0)
        self.assertEqual(float(from_[17, 17, 3]), 0.0)
        # Where B = 0.5 under the opaque shape: A - B = 0.5; B - A is below zero.
        self.assertAlmostEqual(float(minus[11, 11, 3]), 0.5, places=6)
        self.assertEqual(float(from_[11, 11, 3]), 0.0)

    def test_max_min_difference_exclusion_and_screen_at_the_overlap(self):
        want = {"max": 1.0, "min": 0.5, "difference": 0.5, "exclusion": 0.5, "screen": 1.0}
        for mode, value in want.items():
            with self.subTest(mode=mode):
                self.assertAlmostEqual(float(draw(self.two(mode))[11, 11, 3]), value, places=6)

    def test_an_older_document_keeps_the_old_minus_bit_for_bit(self):
        new = self.two("minus")
        old = copy.deepcopy(new)
        old["nodes"]["r"]["params"].pop("legacy_roto_shapes")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["r"]["params"]["legacy_roto_shapes"], 1)
        self.assertEqual(SPECS["Roto"]["params"]["legacy_roto_shapes"], 0)
        np.testing.assert_array_equal(draw(dict(loaded, view="r")), draw(self.two("from")))
        self.assertTrue((draw(dict(loaded, view="r")) != draw(new)).any())

    def test_the_tile_path_equals_the_evaluator(self):
        for mode in ("minus", "from", "difference", "exclusion", "max", "min", "screen"):
            for mask in (False, True):
                for legacy in (0, 1):
                    doc = self.two(mode, legacy_roto_shapes=legacy)
                    if mask:
                        doc = build([self.below, box(8, 8, 20, 20, name="top", blend_mode=mode)], mask=True, legacy_roto_shapes=legacy)
                    expected = draw(doc)
                    for edge in (7, 16, 32):
                        with self.subTest(mode=mode, mask=mask, legacy=legacy, tile=edge):
                            np.testing.assert_allclose(tiles(doc, edge), expected, atol=1e-6)

    def test_the_panel_offers_every_mode(self):
        for mode in ("minus", "from", "difference", "exclusion", "max", "min", "screen"):
            self.assertIn(mode, shapes.BLEND_MODES)


class ColourVisibleInvertTests(unittest.TestCase):
    def test_a_shapes_colour_is_the_colour_it_draws(self):
        pixels = draw(build([box(4, 4, 16, 16, color=[0.9, 0.3, 0.1, 1.0])], output="rgba"))
        np.testing.assert_allclose(pixels[10, 10, :3], [0.9, 0.3, 0.1], atol=1e-6)
        self.assertEqual(float(pixels[10, 10, 3]), 1.0)
        other = draw(build([box(4, 4, 16, 16, color=[0.1, 0.3, 0.9, 1.0])], output="rgba"))
        self.assertTrue((pixels[..., 0] != other[..., 0]).any())

    def test_the_colours_alpha_scales_the_matte(self):
        half = draw(build([box(4, 4, 16, 16, color=[1.0, 1.0, 1.0, 0.5])]))
        self.assertAlmostEqual(float(half[10, 10, 3]), 0.5, places=6)

    def test_invert_complements_the_matte(self):
        plain = draw(build([box(4, 4, 16, 16)]))[..., 3]
        inverted = draw(build([box(4, 4, 16, 16, invert=True)]))[..., 3]
        np.testing.assert_allclose(inverted, 1.0 - plain, atol=1e-6)

    def test_a_shape_that_is_not_visible_draws_nothing(self):
        self.assertTrue(draw(build([box(4, 4, 16, 16, visible=False)]))[..., 3].max() == 0.0)
        self.assertTrue(draw(build([box(4, 4, 16, 16)]))[..., 3].max() == 1.0)


class TransformTests(unittest.TestCase):
    def matte(self, **transform):
        return draw(build([box(8, 10, 16, 14, transform=transform)]))[..., 3]

    def test_translate_moves_the_pixels_by_the_amount(self):
        base, moved = self.matte(), self.matte(translate_x=3.0, translate_y=2.0)
        np.testing.assert_allclose(moved[2:, 3:], base[:-2, :-3], atol=1e-6)

    def test_rotate_turns_the_shape_about_the_centre(self):
        turned = self.matte(rotate=90.0, center_x=12.0, center_y=12.0)
        np.testing.assert_allclose(turned[10:14, 10:14], 1.0, atol=1e-6)         # the middle stays covered
        self.assertEqual(float(turned[12, 8]), 0.0)                                # the old wide box's ends are gone
        self.assertEqual(float(turned[8, 12]), 1.0)                                # and the box is now tall

    def test_scale_resizes_about_the_centre(self):
        grown = self.matte(scale_x=2.0, scale_y=2.0, center_x=12.0, center_y=12.0)
        self.assertAlmostEqual(float(grown.sum()), 4.0 * 32.0, places=3)

    def test_skew_keeps_the_area_and_moves_the_top_edge(self):
        base, skewed = self.matte(), self.matte(skew_x=30.0, center_x=12.0, center_y=12.0)
        self.assertAlmostEqual(float(skewed.sum()), float(base.sum()), delta=0.05)
        self.assertTrue((skewed != base).any())
        self.assertTrue((self.matte(skew_y=30.0, center_x=12.0, center_y=12.0) != base).any())

    def test_skew_order_changes_the_result_only_with_both_skews(self):
        both = dict(skew_x=30.0, skew_y=20.0, center_x=12.0, center_y=12.0)
        xy, yx = self.matte(**both, skew_order="XY"), self.matte(**both, skew_order="YX")
        self.assertTrue((xy != yx).any())
        one = dict(skew_x=30.0, center_x=12.0, center_y=12.0)
        np.testing.assert_array_equal(self.matte(**one, skew_order="XY"), self.matte(**one, skew_order="YX"))

    def test_the_centre_alone_does_not_move_the_shape(self):
        np.testing.assert_array_equal(self.matte(center_x=3.0, center_y=20.0), self.matte())

    def test_a_rotated_skewed_shape_is_the_same_on_the_tile_path(self):
        doc = build([box(8, 10, 16, 14, transform={"rotate": 33.0, "skew_x": 20.0, "skew_y": 10.0, "scale_x": 1.3, "scale_y": 0.8,
                                                   "translate_x": 2.0, "center_x": 12.0, "center_y": 12.0})])
        masked = build([box(8, 10, 16, 14, transform={"rotate": 33.0, "skew_x": 20.0, "skew_y": 10.0, "center_x": 12.0,
                                                      "center_y": 12.0})], mask=True)
        for d in (doc, masked):
            expected = draw(d)
            for edge in (7, 16, 32):
                np.testing.assert_allclose(tiles(d, edge), expected, atol=1e-6)


class OpenSplineTests(unittest.TestCase):
    def matte(self, **extra):
        return draw(build([line(**extra)]))[..., 3]

    def test_the_width_is_the_stroke_width(self):
        for width in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
            with self.subTest(width=width):
                column = self.matte(stroke_width=width, cap_start="square", cap_end="square")[:, 12]
                self.assertAlmostEqual(float(column.sum()), width, places=5)

    def test_an_older_document_keeps_the_hard_edged_stroke_bit_for_bit(self):
        doc = build([line(stroke_width=3.0)])
        old = copy.deepcopy(doc)
        old["nodes"]["r"]["params"].pop("legacy_roto_shapes")
        loaded = upgrade_document(old)
        column = draw(dict(loaded, view="r"))[:, 12, 3]
        self.assertEqual(float(column.sum()), 4.0)                     # a pixel wider than the 3 stated, with a hard edge
        self.assertTrue(set(np.unique(column)) <= {0.0, 1.0})
        self.assertAlmostEqual(float(draw(doc)[:, 12, 3].sum()), 3.0, places=5)

    def test_square_ends_reach_half_the_width_past_the_end_points(self):
        square = self.matte(cap_start="square", cap_end="square")
        self.assertAlmostEqual(float(square.sum()), (12 + 4) * 4.0, places=4)
        np.testing.assert_allclose(square[10:14, 4:20], 1.0, atol=1e-6)
        self.assertEqual(float(square[10:14, 3].max()), 0.0)

    def test_a_square_end_covers_more_than_a_rounded_one_at_either_end(self):
        rounded = self.matte()
        start_square = self.matte(cap_start="square")
        end_square = self.matte(cap_end="square")
        self.assertGreater(float(start_square[:, :6].sum()), float(rounded[:, :6].sum()))
        self.assertAlmostEqual(float(start_square[:, 12:].sum()), float(rounded[:, 12:].sum()), places=5)
        self.assertGreater(float(end_square[:, 18:].sum()), float(rounded[:, 18:].sum()))
        self.assertAlmostEqual(float(end_square[:, :12].sum()), float(rounded[:, :12].sum()), places=5)

    def test_a_rounded_end_adds_a_half_disc(self):
        butt = self.matte(cap_start="butt", cap_end="butt")
        rounded = self.matte()
        self.assertAlmostEqual(float(butt.sum()), 12 * 4.0, places=4)
        self.assertAlmostEqual(float(rounded.sum() - butt.sum()), np.pi * 4.0, delta=0.7)

    def test_the_older_and_the_faster_drawer_agree_on_the_soft_edge(self):
        from nodebased import roto
        resolved = shapes.resolve_shapes({"shapes": [line(stroke_width=3.0, feather=1.5), line(stroke_width=5.0, cap_end="square")]}, 1,
                                         {"legacy_roto_shapes": 0})
        self.assertTrue(all(shape.get("stroke_soft") for shape in resolved))
        before = roto.drawer()
        try:
            roto.set_drawer("legacy")
            slow = roto.rasterise(copy.deepcopy(resolved), N, N)
            roto.set_drawer("fast")
            fast = roto.rasterise(copy.deepcopy(resolved), N, N)
        finally:
            roto.set_drawer(before)
        np.testing.assert_allclose(fast, slow, atol=1e-5)

    def test_the_tile_path_equals_the_evaluator(self):
        for extra in ({}, {"cap_start": "square"}, {"cap_end": "square", "stroke_width": 7.0}):
            for mask in (False, True):
                doc = build([line(**extra)], mask=mask)
                expected = draw(doc)
                for edge in (7, 16, 32):
                    with self.subTest(extra=extra, mask=mask, tile=edge):
                        np.testing.assert_allclose(tiles(doc, edge), expected, atol=1e-6)


class LifetimeTests(unittest.TestCase):
    frames = range(1, 11)

    def present(self, life):
        doc = build([box(4, 4, 16, 16, lifetime=life)] if life else [box(4, 4, 16, 16)])
        return [bool(draw(doc, frame=f)[..., 3].max() > 0.0) for f in self.frames]

    def test_all_frames(self):
        self.assertEqual(self.present({"type": "all"}), [True] * 10)
        self.assertEqual(self.present(None), [True] * 10)

    def test_single_frame_is_on_the_from_frame_only(self):
        self.assertEqual(self.present({"type": "single", "start": 4}), [f == 4 for f in self.frames])

    def test_start_to_frame_runs_to_the_to_frame(self):
        self.assertEqual(self.present({"type": "start_to", "end": 5}), [f <= 5 for f in self.frames])

    def test_to_end_runs_from_the_from_frame(self):
        self.assertEqual(self.present({"type": "to_end", "start": 7}), [f >= 7 for f in self.frames])

    def test_range_runs_from_the_from_frame_to_the_to_frame(self):
        self.assertEqual(self.present({"type": "range", "start": 3, "end": 6}), [3 <= f <= 6 for f in self.frames])

    def test_a_shape_outside_its_lifetime_leaves_the_others_alone(self):
        both = build([box(2, 2, 10, 10), box(12, 12, 20, 20, name="late", lifetime={"type": "to_end", "start": 5})])
        early, late = draw(both, frame=2)[..., 3], draw(both, frame=6)[..., 3]
        self.assertEqual(float(early[15, 15]), 0.0)
        self.assertEqual(float(late[15, 15]), 1.0)
        np.testing.assert_array_equal(early[:11, :11], late[:11, :11])


if __name__ == "__main__":
    unittest.main()
