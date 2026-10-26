"""Fidelity step F, part 1: Roto takes Nuke's feather, transform, motion blur, lifetime, open-spline cap and output mask
knobs. Pixel assertions per option, the tile path against the evaluator, bypass of the shape settings and old documents."""
import copy
import unittest

import numpy as np

from nodebased import roto, shapes, tiers
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

N = 48


def point(x, y, **extra):
    return {"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0, **extra}


def box(**extra):
    shape = {"name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0,
             "points": [point(14.0, 12.0), point(34.0, 12.0), point(34.0, 32.0), point(14.0, 32.0)]}
    shape.update(extra)
    return shape


def wire(**extra):
    shape = {"name": "wire", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": False, "stroke_width": 8.0,
             "points": [point(14.5, 24.5), point(34.5, 24.5)]}
    shape.update(extra)
    return shape


def moving(**extra):
    """A box that slides 8 px right per frame on a linear curve (frames 1 to 5)."""
    def curve(first):
        return {"value": first, "curve": {"interpolation": "linear",
                                          "keys": [{"frame": 1, "value": first}, {"frame": 5, "value": first + 32.0}]}}
    shape = box(**extra)
    shape["points"] = [dict(p, x=curve(p["x"]) if i else curve(p["x"])) for i, p in enumerate(shape["points"])]
    return shape


def alpha(shape_list, frame=1, params=None):
    return roto.rasterise(shapes.resolve_shapes({"shapes": shape_list}, frame, params), N, N)[..., 3]


def document(shape_list, **params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": N, "height": N, **params}})
    d.execute({"op": "set_shapes", "id": "r", "shapes": shape_list})
    d.execute({"op": "view", "id": "r"})
    return d.document


def render(doc):
    return Evaluator().evaluate_raster(doc, target="r")


class FeatherTests(unittest.TestCase):
    def test_feather_off_is_a_hard_edge_even_with_a_feather_width(self):
        hard = alpha([box()])
        np.testing.assert_array_equal(alpha([box(feather=6.0, feather_on=False)]), hard)
        self.assertGreater(float(np.abs(alpha([box(feather=6.0)]) - hard).sum()), 5.0)
        self.assertGreater(float(np.abs(alpha([box(feather=6.0, feather_on=True)]) - hard).sum()), 5.0)

    def test_feather_off_clears_the_per_point_feather_too(self):
        feathered = box(feather_model="edge")
        feathered["points"] = [dict(p, feather_x=5.0, feather_y=5.0) for p in feathered["points"]]
        self.assertGreater(float(np.abs(alpha([feathered]) - alpha([box()])).sum()), 5.0)
        off = dict(feathered, feather_on=False)
        np.testing.assert_array_equal(alpha([off]), alpha([box(feather_model="edge")]))

    def test_falloff_power_shapes_the_feather_ramp(self):
        base = alpha([box(feather=6.0)])
        np.testing.assert_array_equal(alpha([box(feather=6.0, falloff_power=1.0)]), base)
        soft, hard, sharp = (alpha([box(feather=6.0, falloff_power=p)]) for p in (0.5, 0.0, 2.0))
        ramp = (base > 0.01) & (base < 0.99)
        self.assertTrue(ramp.any())
        self.assertTrue((soft[ramp] >= base[ramp] - 1e-6).all())
        self.assertTrue((sharp[ramp] <= base[ramp] + 1e-6).all())
        np.testing.assert_allclose(soft[ramp], np.sqrt(base[ramp]), atol=1e-5)
        np.testing.assert_allclose(sharp[ramp], base[ramp] ** 2, atol=1e-5)
        self.assertEqual(set(np.unique(hard)) - {0.0, 1.0}, set())              # 0 is a hard edge out to the feather's reach
        self.assertTrue((hard[base > 0] == 1.0).all())

    def test_the_default_resolved_shape_carries_none_of_the_new_settings(self):
        resolved = shapes.resolve_shapes({"shapes": [box()]}, 1)[0]
        for key in ("falloff_power", "extra_matrix", "cap_start", "cap_end", "_motion_only", "_motion_final"):
            self.assertNotIn(key, resolved)
        self.assertEqual(resolved["transform"]["skew_x"], 0.0)


class TransformTests(unittest.TestCase):
    def test_skew_x_shears_about_the_centre(self):
        skewed = alpha([box(transform={"skew_x": 45.0})])                       # tan 45 = 1: x moves by the offset from the centre row
        centre_y = 22.0
        explicit = box()
        explicit["points"] = [dict(p, x=p["x"] + (p["y"] - centre_y)) for p in explicit["points"]]
        np.testing.assert_allclose(skewed, alpha([explicit]), atol=1e-5)

    def test_skew_y_shears_the_other_way(self):
        skewed = alpha([box(transform={"skew_y": 45.0})])
        centre_x = 24.0
        explicit = box()
        explicit["points"] = [dict(p, y=p["y"] + (p["x"] - centre_x)) for p in explicit["points"]]
        np.testing.assert_allclose(skewed, alpha([explicit]), atol=1e-5)

    def test_skew_order_matters_only_when_both_skews_are_set(self):
        both = {"skew_x": 30.0, "skew_y": 20.0}
        xy = alpha([box(transform={**both, "skew_order": "XY"})])
        yx = alpha([box(transform={**both, "skew_order": "YX"})])
        self.assertGreater(float(np.abs(xy - yx).sum()), 1.0)
        one = {"skew_x": 30.0}
        np.testing.assert_allclose(alpha([box(transform={**one, "skew_order": "XY"})]),
                                   alpha([box(transform={**one, "skew_order": "YX"})]), atol=1e-6)

    def test_skew_order_xy_applies_x_first(self):
        t = {"skew_x": 30.0, "skew_y": 20.0}
        tx, ty = np.tan(np.radians(30.0)), np.tan(np.radians(20.0))
        x, y = 10.0, -6.0                                                      # offset from the centre
        # X first: (x + tx*y, y) then Y: (x', y + ty*x')
        want_xy = (x + tx * y, y + ty * (x + tx * y))
        got = shapes._skew_matrix({**t, "skew_order": "XY"})
        np.testing.assert_allclose((got[0] * x + got[1] * y, got[2] * x + got[3] * y), want_xy, atol=1e-9)
        want_yx = (x + tx * (y + ty * x), y + ty * x)
        got = shapes._skew_matrix({**t, "skew_order": "YX"})
        np.testing.assert_allclose((got[0] * x + got[1] * y, got[2] * x + got[3] * y), want_yx, atol=1e-9)

    def test_extra_matrix_translates_scales_and_shears_in_image_space(self):
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        np.testing.assert_array_equal(alpha([box(extra_matrix=identity)]), alpha([box()]))
        move = list(identity)
        move[3], move[7] = 5.0, 3.0
        moved = box()
        moved["points"] = [dict(p, x=p["x"] + 5.0, y=p["y"] + 3.0) for p in moved["points"]]
        np.testing.assert_allclose(alpha([box(extra_matrix=move)]), alpha([moved]), atol=1e-6)
        grow = list(identity)
        grow[0] = grow[5] = 0.5                                                 # scale about the image origin
        small = box()
        small["points"] = [dict(p, x=p["x"] * 0.5, y=p["y"] * 0.5) for p in small["points"]]
        np.testing.assert_allclose(alpha([box(extra_matrix=grow)]), alpha([small]), atol=1e-6)
        shear = list(identity)
        shear[1] = 0.5                                                          # x' = x + 0.5 y
        sheared = box()
        sheared["points"] = [dict(p, x=p["x"] + 0.5 * p["y"]) for p in sheared["points"]]
        np.testing.assert_allclose(alpha([box(extra_matrix=shear)]), alpha([sheared]), atol=1e-6)

    def test_extra_matrix_follows_the_proxy_tier(self):
        matrix = [1, 0, 0, 8.0, 0, 1, 0, 4.0, 0, 0, 1, 0, 0, 0, 0, 1]
        scaled = tiers.scale_node_data("Roto", {"shapes": [box(extra_matrix=matrix)]}, 2)["shapes"][0]["extra_matrix"]
        self.assertEqual((scaled[3], scaled[7], scaled[0], scaled[5]), (4.0, 2.0, 1, 1))

    def test_a_scaling_matrix_scales_a_strokes_width_with_it(self):
        double = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        plain = shapes.resolve_shapes({"shapes": [wire()]}, 1)[0]["stroke_width"]
        grown = shapes.resolve_shapes({"shapes": [wire(extra_matrix=double)]}, 1)[0]["stroke_width"]
        self.assertAlmostEqual(grown, plain * 2.0)


class LifetimeTests(unittest.TestCase):
    def drawn(self, life, frame):
        return float(alpha([box(lifetime=life)], frame).sum()) > 0

    def test_each_lifetime_type(self):
        cases = (({"type": "all"}, (1, 3, 9), ()),
                 ({"type": "single", "start": 4}, (4,), (3, 5)),
                 ({"type": "start_to", "end": 4}, (1, 4), (5, 9)),
                 ({"type": "to_end", "start": 4}, (4, 9), (1, 3)),
                 ({"type": "range", "start": 3, "end": 6}, (3, 5, 6), (2, 7)))
        for life, shown, hidden in cases:
            for frame in shown:
                with self.subTest(life=life, frame=frame):
                    self.assertTrue(self.drawn(life, frame))
            for frame in hidden:
                with self.subTest(life=life, frame=frame):
                    self.assertFalse(self.drawn(life, frame))

    def test_a_shape_with_no_lifetime_exists_on_every_frame(self):
        self.assertTrue(all(float(alpha([box()], f).sum()) > 0 for f in (-5, 1, 400)))

    def test_a_hidden_shape_leaves_the_others_alone(self):
        both = [box(), box(name="b", lifetime={"type": "single", "start": 9},
                           points=[point(2.0, 2.0), point(8.0, 2.0), point(8.0, 8.0), point(2.0, 8.0)])]
        np.testing.assert_array_equal(alpha(both, 1), alpha([box()], 1))
        self.assertGreater(float(alpha(both, 9)[4, 4]), 0.9)


class MotionBlurTests(unittest.TestCase):
    def average(self, times, **extra):
        return sum(alpha([moving(**extra)], t) for t in times) / len(times)

    def test_a_shutter_averages_the_shape_over_the_open_shutter(self):
        blurred = alpha([moving(motion_blur=True, motion_blur_samples=4, motion_blur_shutter=1.0, motion_blur_offset_type="start")], 2)
        want = self.average([2.0, 2.25, 2.5, 2.75, 3.0])
        np.testing.assert_allclose(blurred, want, atol=1e-5)
        self.assertGreater(float(np.abs(blurred - alpha([moving()], 2)).sum()), 5.0)

    def test_the_four_shutter_offsets(self):
        for kind, offset, times in (("centred", 0.0, [1.5, 2.0, 2.5]), ("start", 0.0, [2.0, 2.5, 3.0]),
                                    ("end", 0.0, [1.0, 1.5, 2.0]), ("custom", -0.5, [1.5, 2.0, 2.5]),
                                    ("custom", 1.0, [3.0, 3.5, 4.0])):
            with self.subTest(kind=kind, offset=offset):
                blurred = alpha([moving(motion_blur=True, motion_blur_samples=2, motion_blur_shutter=1.0,
                                        motion_blur_offset_type=kind, motion_blur_offset=offset)], 2)
                np.testing.assert_allclose(blurred, self.average(times), atol=1e-5)

    def test_a_longer_shutter_smears_further_and_zero_is_sharp(self):
        def spread(shutter):
            return float((alpha([moving(motion_blur=True, motion_blur_samples=8, motion_blur_shutter=shutter,
                                        motion_blur_offset_type="centred")], 3) > 0.01).sum())
        self.assertGreater(spread(1.0), spread(0.25))
        np.testing.assert_array_equal(
            alpha([moving(motion_blur=True, motion_blur_samples=8, motion_blur_shutter=0.0)], 3), alpha([moving()], 3))

    def test_more_samples_average_the_same_travel_more_smoothly(self):
        coarse = alpha([moving(motion_blur=True, motion_blur_samples=1, motion_blur_shutter=1.0, motion_blur_offset_type="centred")], 2)
        fine = alpha([moving(motion_blur=True, motion_blur_samples=16, motion_blur_shutter=1.0, motion_blur_offset_type="centred")], 2)
        self.assertEqual(float(coarse.sum()).__round__(2), float(fine.sum()).__round__(2))     # the same coverage in total
        self.assertGreater(len(np.unique(fine.round(4))), len(np.unique(coarse.round(4))))      # on more distinct levels

    def test_a_blurred_shape_blurs_its_transform_too(self):
        spinning = box(motion_blur=True, motion_blur_samples=4, motion_blur_shutter=2.0, motion_blur_offset_type="centred")
        spinning["transform"] = {"rotate": {"value": 0.0, "curve": {"interpolation": "linear",
                                                                      "keys": [{"frame": 1, "value": 0.0}, {"frame": 5, "value": 40.0}]}}}
        sharp = dict(spinning, motion_blur=False)
        want = sum(alpha([sharp], t) for t in (2.0, 2.5, 3.0, 3.5, 4.0)) / 5
        np.testing.assert_allclose(alpha([spinning], 3), want, atol=1e-5)

    def test_a_shape_saved_with_only_the_motion_blur_flag_keeps_its_three_frame_blur(self):
        legacy = moving(motion_blur=True)
        want = (alpha([moving()], 2) + alpha([moving()], 3) + alpha([moving()], 4)) / 3
        np.testing.assert_allclose(alpha([legacy], 3), want, atol=1e-5)
        resolved = shapes.resolve_shapes({"shapes": [legacy]}, 3)[0]
        self.assertNotIn("_motion_only", resolved)
        self.assertEqual(len(resolved["_motion_samples"]), 2)

    def test_global_settings_blur_every_shape_without_its_own(self):
        params = {"global_motionblur_on": 1, "global_motionblur": 4, "global_motionblur_shutter": 1.0,
                  "global_motionblur_shutter_offset_type": "start"}
        want = self.average([3.0, 3.25, 3.5, 3.75, 4.0])
        np.testing.assert_allclose(alpha([moving()], 3, params), want, atol=1e-5)
        np.testing.assert_array_equal(alpha([moving()], 3, {"global_motionblur_on": 0}), alpha([moving()], 3))

    def test_a_shapes_own_settings_beat_the_global_ones(self):
        params = {"global_motionblur_on": 1, "global_motionblur": 4, "global_motionblur_shutter": 2.0}
        own = moving(motion_blur=True, motion_blur_samples=2, motion_blur_shutter=1.0, motion_blur_offset_type="end")
        np.testing.assert_allclose(alpha([own], 3, params), self.average([2.0, 2.5, 3.0]), atol=1e-5)

    def test_shape_motion_blur_off_ignores_the_shapes_own_settings(self):
        own = moving(motion_blur=True, motion_blur_samples=4, motion_blur_shutter=1.0)
        np.testing.assert_array_equal(alpha([own], 3, {"shape_motionblur_on": 0}), alpha([moving()], 3))
        self.assertGreater(float(np.abs(alpha([own], 3, {"shape_motionblur_on": 1}) - alpha([moving()], 3)).sum()), 5.0)

    def test_the_node_reads_its_global_motion_blur_on_the_evaluator_and_the_tiles(self):
        options = dict(global_motionblur_on=1, global_motionblur=4, global_motionblur_shutter=1.0)
        doc = document([moving()], **options)
        got = render(doc)
        np.testing.assert_allclose(got.pixels[..., 3], self.average([1.0, 1.25, 1.5, 1.75, 2.0]), atol=1e-5)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        region = executor.canvas_region(doc, "r", frame=1, tier=1)
        tiled = executor.compose_region(doc, "r", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, got.pixels, atol=1e-6)


class CapTests(unittest.TestCase):
    def cover(self, **extra):
        return alpha([wire(**extra)])

    def test_rounded_square_and_butt_ends(self):
        rounded, square, butt = self.cover(), self.cover(cap_start="square", cap_end="square"), self.cover(cap_start="butt", cap_end="butt")
        np.testing.assert_array_equal(self.cover(cap_start="rounded", cap_end="rounded"), rounded)
        self.assertEqual(float(butt[24, 13]), 0.0)                         # left of the first point: nothing
        self.assertEqual(float(butt[24, 36]), 0.0)                         # right of the last point: nothing
        self.assertEqual(float(butt[24, 16]), 1.0)                         # on the line
        self.assertGreater(float(rounded[24, 12]), 0.9)                    # half a width (4 px) beyond the end, on the axis
        self.assertGreater(float(square[24, 12]), 0.9)
        self.assertGreater(float(square[20, 11]), 0.9)                     # a square end is as wide at its tip
        self.assertLess(float(rounded[20, 11]), 0.5)                       # a rounded one has narrowed there
        self.assertEqual(float(square[24, 8]), 0.0)                        # but runs on no further than half a width
        self.assertLess(float(butt.sum()), float(rounded.sum()))
        self.assertGreater(float(square.sum()), float(rounded.sum()))

    def test_the_start_and_end_caps_are_independent(self):
        one = self.cover(cap_start="butt")
        self.assertEqual(float(one[24, 13]), 0.0)
        self.assertGreater(float(one[24, 36]), 0.9)
        two = self.cover(cap_end="butt")
        self.assertGreater(float(two[24, 13]), 0.9)
        self.assertEqual(float(two[24, 36]), 0.0)

    def test_caps_do_not_touch_a_closed_shape(self):
        np.testing.assert_array_equal(alpha([box(cap_start="butt", cap_end="square")]), alpha([box()]))


class OutputMaskTests(unittest.TestCase):
    def test_the_matte_is_also_written_to_the_named_layer(self):
        doc = document([box()], outputMask="roto_matte", opacity=0.5)
        raster = render(doc)
        matte = raster.layers["roto_matte"].pixels
        np.testing.assert_allclose(matte[..., 3], raster.pixels[..., 3], atol=1e-7)
        for channel in range(3):
            np.testing.assert_array_equal(matte[..., channel], matte[..., 3])
        self.assertAlmostEqual(float(matte[22, 24, 3]), 0.5, places=6)

    def test_none_adds_no_layer_and_changes_nothing(self):
        plain = render(document([box()]))
        self.assertFalse(plain.layers and "none" in plain.layers)
        named = render(document([box()], outputMask="m"))
        np.testing.assert_array_equal(named.pixels, plain.pixels)

    def test_the_layer_survives_a_chosen_output_channel(self):
        raster = render(document([box()], outputMask="m", output="rgb"))
        self.assertIn("m", raster.layers)
        self.assertEqual(float(raster.pixels[22, 24, 3]), 0.0)             # alpha was not an output channel

    def test_a_node_with_an_output_mask_renders_whole_and_tiles_agree(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(document([box()]), "r"))
        doc = document([box()], outputMask="m")
        self.assertFalse(executor.supports_tiled(doc, "r"))
        region = executor.canvas_region(doc, "r", frame=1, tier=1)
        np.testing.assert_allclose(executor.compose_region(doc, "r", region, frame=1, tier=1).pixels, render(doc).pixels, atol=1e-6)


class TileTests(unittest.TestCase):
    def test_every_shape_setting_draws_the_same_through_tiles(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        variants = (box(feather=5.0, feather_on=False), box(feather=5.0, falloff_power=0.5), box(transform={"skew_x": 20.0, "skew_y": 10.0}),
                    box(extra_matrix=[1, 0.2, 0, 3, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1]), box(lifetime={"type": "single", "start": 1}),
                    moving(motion_blur=True, motion_blur_samples=3, motion_blur_shutter=1.0), wire(cap_start="butt", cap_end="square"))
        for shape in variants:
            with self.subTest(shape=sorted(set(shape) - {"name", "mode", "opacity", "feather", "points"})):
                doc = document([shape])
                region = executor.canvas_region(doc, "r", frame=1, tier=1)
                np.testing.assert_allclose(executor.compose_region(doc, "r", region, frame=1, tier=1).pixels,
                                           render(doc).pixels, atol=1e-6)


class ValidationAndOldDocumentTests(unittest.TestCase):
    def test_bad_settings_are_refused(self):
        for bad in (dict(feather_on=1), dict(falloff_power=-1.0), dict(falloff_power=99.0), dict(cap_start="pointy"),
                    dict(motion_blur_samples=0), dict(motion_blur_samples=1.5), dict(motion_blur_shutter=-1.0),
                    dict(motion_blur_offset_type="middle"), dict(lifetime={"type": "forever"}), dict(lifetime={"type": "range", "start": 5, "end": 2}),
                    dict(lifetime={"type": "single", "start": 1.5}), dict(extra_matrix=[1, 0, 0]), dict(extra_matrix=["a"] * 16),
                    dict(transform={"skew_order": "ZZ"}), dict(transform={"skew_x": 120.0})):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                shapes.validate_shape(box(**bad), "shape")

    def test_good_settings_validate(self):
        shapes.validate_shape(box(feather_on=False, falloff_power=0.0, cap_start="butt", motion_blur_samples=8, motion_blur_shutter=2.0,
                                   motion_blur_offset_type="custom", motion_blur_offset=-1.0, lifetime={"type": "range", "start": 2, "end": 9},
                                   extra_matrix=list(range(16)), transform={"skew_x": 10.0, "skew_order": "YX"}), "shape")
        validate(document([box(feather_on=False)], outputMask="m", global_motionblur_on=1))

    def test_a_roto_saved_before_these_knobs_loads_and_draws_the_same(self):
        doc = document([box(feather=4.0, motion_blur=True), wire()])
        old = copy.deepcopy(doc)
        for name in ("outputMask", "shape_motionblur_on", "global_motionblur_on", "global_motionblur", "global_motionblur_shutter",
                     "global_motionblur_shutter_offset_type", "global_motionblur_shutter_offset"):
            old["nodes"]["r"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        self.assertEqual(loaded["nodes"]["r"]["params"]["outputMask"], "none")
        self.assertEqual(loaded["nodes"]["r"]["params"]["shape_motionblur_on"], 1)
        self.assertEqual(loaded["nodes"]["r"]["params"]["global_motionblur_on"], 0)
        np.testing.assert_array_equal(render(loaded).pixels, render(doc).pixels)

    def test_the_new_params_have_their_defaults_on_a_fresh_roto(self):
        params = SPECS["Roto"]["params"]
        self.assertEqual((params["outputMask"], params["shape_motionblur_on"], params["global_motionblur_on"]), ("none", 1, 0))
        self.assertEqual((params["global_motionblur"], params["global_motionblur_shutter"],
                          params["global_motionblur_shutter_offset_type"]), (1, 0.5, "start"))


if __name__ == "__main__":
    unittest.main()
