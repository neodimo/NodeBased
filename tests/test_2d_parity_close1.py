import copy
import unittest

import numpy as np

from nodebased import colorcurves, paint
from nodebased.paint import apply_stroke
from nodebased.core import Dispatcher, SPECS, upgrade_document
from nodebased.dustbust import detect_specks, dustbust_items_for_specks
from nodebased.imaging import Evaluator, curve_tool_metrics
from tests.test_2d_parity_group_3b import kernel


class CurveToolAnalysisModesTests(unittest.TestCase):
    def test_min_and_max_luminance_pixel_and_value(self):
        image = np.zeros((4, 6, 4), np.float32)
        image[1, 4] = [0.9, 0.9, 0.9, 1.0]
        image[3, 0] = [0.05, 0.05, 0.05, 1.0]
        # Everywhere else is exactly zero luminance, so the minimum falls on the first zero pixel
        # scanned in row-major order (0, 0) rather than the planted low value at (0, 3).
        m = curve_tool_metrics(image)
        self.assertEqual((m["max_x"], m["max_y"]), (4.0, 1.0))
        self.assertAlmostEqual(m["max_value"], 0.9, places=6)
        self.assertEqual((m["min_x"], m["min_y"]), (0.0, 0.0))
        self.assertAlmostEqual(m["min_value"], 0.0, places=6)

    def test_average_luminance_is_the_mean_of_the_three_channel_averages(self):
        image = np.zeros((2, 2, 4), np.float32)
        image[...] = [0.3, 0.6, 0.9, 1.0]
        m = curve_tool_metrics(image)
        self.assertAlmostEqual(m["average_luminance"], 0.6, places=6)

    def test_exposure_difference_is_zero_with_no_previous_sample_and_signed_after(self):
        dim = np.full((2, 2, 4), 0.2, np.float32)
        bright = np.full((2, 2, 4), 0.5, np.float32)
        first = curve_tool_metrics(dim)
        self.assertEqual(first["exposure_diff"], 0.0)
        second = curve_tool_metrics(bright, previous_luminance=first["average_luminance"])
        self.assertAlmostEqual(second["exposure_diff"], 0.3, places=6)
        third = curve_tool_metrics(dim, previous_luminance=second["average_luminance"])
        self.assertAlmostEqual(third["exposure_diff"], -0.3, places=6)

    def test_curve_tool_analyze_writes_exposure_and_extrema_curves_in_one_undo_step(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"src", "type":"Constant",
                   "params":{"width":2,"height":2,"red":.2,"green":.2,"blue":.2,"alpha":1}})
        d.execute({"op":"create", "id":"curve", "type":"CurveTool",
                   "params":{"frame_start":1,"frame_end":2}})
        d.execute({"op":"connect", "id":"curve", "input":"image", "source":"src"})
        d.execute({"op":"set_key", "id":"src", "param":"red", "frame":1, "value":0.2})
        d.execute({"op":"set_key", "id":"src", "param":"red", "frame":2, "value":0.8})
        m1 = curve_tool_metrics(Evaluator().evaluate_raster(d.document, target="src", frame=1).to_display())
        m2 = curve_tool_metrics(Evaluator().evaluate_raster(d.document, target="src", frame=2).to_display(),
                                previous_luminance=m1["average_luminance"])
        undo_depth_before = len(d.undo_stack)
        d.execute({"op":"batch", "commands":
                   [{"op":"set_key","id":"curve","param":name,"frame":1,"value":value,"interpolation":"linear"}
                    for name, value in m1.items()] +
                   [{"op":"set_key","id":"curve","param":name,"frame":2,"value":value,"interpolation":"linear"}
                    for name, value in m2.items()]})
        self.assertEqual(len(d.undo_stack), undo_depth_before + 1)
        self.assertAlmostEqual(m2["exposure_diff"], m2["average_luminance"] - m1["average_luminance"], places=6)


class HSVToolColorReplaceTests(unittest.TestCase):
    def _pixels(self, *colours):
        image = np.ones((1, len(colours), 4), np.float32)
        image[0, :, :3] = np.array(colours, np.float32)
        return image

    def test_disabled_by_default_is_the_identity_even_with_colours_set(self):
        image = self._pixels((1, 0, 0), (0.2, 0.4, 0.6))
        out = kernel("HSVTool", image, srccolor_r=1.0, srccolor_g=0.0, srccolor_b=0.0,
                    dstcolor_r=0.0, dstcolor_g=0.0, dstcolor_b=1.0)
        np.testing.assert_array_equal(out, image)

    def test_pure_source_colour_becomes_destination_colour_at_full_weight(self):
        image = self._pixels((1, 0, 0))   # pure red, matches srccolor exactly
        out = kernel("HSVTool", image, color_replace=1,
                    srccolor_r=1.0, srccolor_g=0.0, srccolor_b=0.0,
                    dstcolor_r=0.0, dstcolor_g=0.0, dstcolor_b=1.0)   # pure blue
        np.testing.assert_allclose(out[0, 0, :3], (0.0, 0.0, 1.0), atol=1e-6)
        self.assertEqual(out[0, 0, 3], 1.0)   # alpha untouched (output_alpha is off)

    def test_identical_source_and_destination_forces_pixels_toward_that_one_colour(self):
        # A grey pixel (S 0) forced toward a fully saturated destination equal to the source still
        # moves: "force" targets an absolute value, it does not compare against the pixel's own.
        image = self._pixels((0.5, 0.5, 0.5))
        out = kernel("HSVTool", image, color_replace=1,
                    srccolor_r=1.0, srccolor_g=0.0, srccolor_b=0.0,
                    dstcolor_r=1.0, dstcolor_g=0.0, dstcolor_b=0.0)
        np.testing.assert_allclose(out[0, 0, :3], (1.0, 0.0, 0.0), atol=1e-6)

    def test_range_gating_still_applies_under_color_replace(self):
        image = self._pixels((1, 0, 0), (0, 0, 1))   # red matches srccolor's hue, blue does not
        out = kernel("HSVTool", image, color_replace=1,
                    hue_range_min=340.0, hue_range_max=20.0,
                    srccolor_r=1.0, srccolor_g=0.0, srccolor_b=0.0,
                    dstcolor_r=0.0, dstcolor_g=1.0, dstcolor_b=0.0)
        np.testing.assert_allclose(out[0, 0, :3], (0.0, 1.0, 0.0), atol=1e-6)
        np.testing.assert_array_equal(out[0, 1], image[0, 1])


class CrossTalkTangentCurveTests(unittest.TestCase):
    def _tangent_curve(self, points, slopes, modes):
        return colorcurves.encode(points, slopes=slopes, modes=modes, broken=[True] * len(points))

    def test_evaluate_array_matches_the_scalar_evaluate_on_a_broken_tangent_curve(self):
        raw = self._tangent_curve(((0, 0), (0.5, 0.8), (1, 1)),
                                  ((0, 2.0), (0.5, -1.0), (3.0, 0)), ("broken", "broken", "broken"))
        curve = colorcurves.decode(raw)
        samples = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, -0.2, 1.3], np.float64)
        vectorised = colorcurves.evaluate_array(curve, samples)
        scalar = np.array([colorcurves.evaluate(curve, float(v)) for v in samples], np.float32)
        np.testing.assert_allclose(vectorised, scalar, atol=1e-5)

    def test_evaluate_array_matches_scalar_on_a_smooth_auto_tangent_curve(self):
        raw = colorcurves.encode(((0, 0), (0.3, 0.9), (0.7, 0.2), (1, 1)),
                                 slopes=[[0, 0]] * 4, modes=["smooth"] * 4, broken=[False] * 4)
        curve = colorcurves.decode(raw)
        samples = np.linspace(0.0, 1.0, 21)
        vectorised = colorcurves.evaluate_array(curve, samples)
        scalar = np.array([colorcurves.evaluate(curve, float(v)) for v in samples], np.float32)
        np.testing.assert_allclose(vectorised, scalar, atol=1e-5)

    def test_evaluate_array_matches_scalar_on_a_constant_step_curve(self):
        raw = colorcurves.encode(((0, 0.2), (0.5, 0.9), (1, 0.4)),
                                 slopes=[[0, 0]] * 3, modes=["constant"] * 3, broken=[True] * 3)
        curve = colorcurves.decode(raw)
        samples = np.array([0.0, 0.2, 0.49999, 0.5, 0.75, 1.0])
        vectorised = colorcurves.evaluate_array(curve, samples)
        scalar = np.array([colorcurves.evaluate(curve, float(v)) for v in samples], np.float32)
        np.testing.assert_allclose(vectorised, scalar, atol=1e-5)

    def test_crosstalk_pixel_output_follows_a_tangent_handle_not_just_a_straight_line(self):
        # A steep outgoing tangent at the first key bows the curve well above the straight chord
        # between (0, 0) and (0.5, 0.5) near its start: the "full curve UI" gap this closes.
        p = dict(SPECS["CrossTalk"]["params"])
        p["xt_curve_r_r"] = self._tangent_curve(((0, 0), (0.5, 0.5), (1, 1)),
                                                 ((0, 3.0), (3.0, 3.0), (0, 0)), ("broken",) * 3)
        image = np.array([[[0.1, 0.0, 0.0, 1.0]]], np.float32)
        out = Evaluator._crosstalk(image, p)
        straight_line_value = 0.1   # what a plain linear/smooth chord would give here
        self.assertGreater(float(out[0, 0, 0]), straight_line_value + 0.05)
        np.testing.assert_array_equal(out[..., 1:], image[..., 1:])


class DustBustSpeckDetectionTests(unittest.TestCase):
    def _frame(self, value=0.2, speck=None, speck_size=2, size=8):
        frame = np.full((size, size, 4), value, np.float32)
        frame[..., 3] = 1.0
        if speck is not None:
            y, x = speck
            frame[y:y + speck_size, x:x + speck_size, :3] = 0.9
        return frame

    def test_a_single_frame_speck_is_found_and_bounded_to_its_frame(self):
        frames = {1: self._frame(), 2: self._frame(speck=(3, 3)), 3: self._frame()}
        specks = detect_specks(frames, sensitivity=0.5)
        self.assertEqual(len(specks), 1)
        speck = specks[0]
        self.assertEqual(speck["frame"], 2)
        self.assertEqual((speck["x"], speck["y"]), (3.0, 3.0))
        self.assertEqual((speck["width"], speck["height"]), (2.0, 2.0))

    def test_a_persistent_difference_across_every_frame_is_not_a_speck(self):
        frames = {1: self._frame(speck=(3, 3)), 2: self._frame(speck=(3, 3)), 3: self._frame(speck=(3, 3))}
        self.assertEqual(detect_specks(frames, sensitivity=0.5), [])

    def test_a_frame_wide_change_is_excluded_as_a_real_moving_object(self):
        big = self._frame()
        big[:, :, :3] = 0.9   # the entire frame changes: far larger than any speck
        frames = {1: self._frame(), 2: big, 3: self._frame()}
        self.assertEqual(detect_specks(frames, sensitivity=0.5, max_size=4), [])

    def test_raising_sensitivity_finds_a_fainter_speck_a_stricter_pass_misses(self):
        faint = self._frame()
        faint[3:5, 3:5, :3] = 0.35   # a moderate step above the 0.2 background
        frames = {1: self._frame(), 2: faint, 3: self._frame()}
        self.assertEqual(detect_specks(frames, sensitivity=0.1), [])
        self.assertEqual(len(detect_specks(frames, sensitivity=0.9)), 1)

    def test_only_frames_with_a_neighbour_on_both_sides_are_analysed(self):
        frames = {1: self._frame(), 2: self._frame(speck=(0, 0)), 3: self._frame(speck=(5, 5))}
        specks = detect_specks(frames, sensitivity=0.5)
        self.assertEqual([s["frame"] for s in specks], [2])

    def test_dustbust_items_for_specks_appends_single_point_clone_strokes(self):
        specks = [{"frame": 5, "x": 2.0, "y": 3.0, "width": 2.0, "height": 2.0, "magnitude": 0.5}]
        items = dustbust_items_for_specks([], specks)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["tool"], "clone")
        self.assertEqual(item["lifetime"], {"mode": "single", "first": 5})
        self.assertEqual(item["source_frame"], "relative")
        self.assertEqual(len(item["points"]), 1)
        self.assertAlmostEqual(item["points"][0]["x"], 3.0)
        self.assertAlmostEqual(item["points"][0]["y"], 4.0)

    def test_dustbust_items_for_specks_avoids_name_collisions_with_existing_items(self):
        existing = [{"name": "stroke1"}, {"name": "stroke2"}]
        items = dustbust_items_for_specks(existing, [{"frame": 1, "x": 0.0, "y": 0.0,
                                                       "width": 1.0, "height": 1.0, "magnitude": 1.0}])
        self.assertEqual(items[-1]["name"], "stroke3")

    def test_an_accepted_speck_clones_from_the_previous_frame_when_rasterised(self):
        plate2 = self._frame(speck=(3, 3))
        plate1 = self._frame()
        items = dustbust_items_for_specks([], [{"frame": 2, "x": 3.0, "y": 3.0,
                                                "width": 2.0, "height": 2.0, "magnitude": 0.7}])
        out = paint.rasterise(plate2, items, frame=2, source=plate1)
        np.testing.assert_allclose(out[3:5, 3:5, :3], 0.2, atol=1e-5)

    def test_accepted_specks_land_through_set_paint_items_as_one_undo_step(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant",
                   "params": {"width": 8, "height": 8, "red": .2, "green": .2, "blue": .2, "alpha": 1}})
        d.execute({"op": "create", "id": "rp", "type": "RotoPaint"})
        d.execute({"op": "connect", "id": "rp", "input": "image", "source": "src"})
        specks = [{"frame": 1, "x": 2.0, "y": 2.0, "width": 2.0, "height": 2.0, "magnitude": 0.6}]
        items = dustbust_items_for_specks([], specks)
        undo_depth_before = len(d.undo_stack)
        d.execute({"op": "set_paint_items", "id": "rp", "items": items})
        self.assertEqual(len(d.undo_stack), undo_depth_before + 1)
        self.assertEqual(d.document["node_data"]["rp"]["items"], items)
        d.execute({"op": "undo"})
        self.assertEqual(d.document["node_data"].get("rp", {}).get("items", []), [])

    def test_rotopaint_node_accepts_the_new_detection_params(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "rp", "type": "RotoPaint",
                   "params": {"dustbust_frame_start": 1, "dustbust_frame_end": 10,
                             "dustbust_sensitivity": 0.7}})
        params = d.document["nodes"]["rp"]["params"]
        self.assertEqual(params["dustbust_frame_start"], 1)
        self.assertAlmostEqual(params["dustbust_sensitivity"], 0.7)


class RotoPaintRetouchQualityTests(unittest.TestCase):
    def _stroke(self, tool, points, size=6.0, opacity=1.0, strength=0.2):
        return {"points": points,
                "brush": {"size": size, "hardness": 0.0, "opacity": 1.0, "spacing": 0.25, "strength": strength},
                "tool": tool, "lifetime": {"mode": "all"}, "color": [0.0, 0.0, 0.0, 1.0],
                "source_offset": [0.0, 0.0], "source_frame": "relative",
                "opacity": opacity, "blend": "over", "visible": True}

    def _delta(self, at=(10, 10), size=24):
        image = np.zeros((size, size, 4), np.float32)
        image[at[1], at[0]] = [1.0, 1.0, 1.0, 1.0]
        return image

    def test_a_bigger_brush_spreads_the_blur_over_a_wider_area(self):
        image = self._delta()
        small = apply_stroke(image, self._stroke("blur", [{"x": 10, "y": 10, "pressure": 1.0}], size=4.0), 1)
        big = apply_stroke(image, self._stroke("blur", [{"x": 10, "y": 10, "pressure": 1.0}], size=24.0), 1)
        spread_small = int((small[..., 0] > 0.01).sum())
        spread_big = int((big[..., 0] > 0.01).sum())
        self.assertGreater(spread_big, spread_small)

    def test_sharpen_increases_local_contrast_at_an_edge(self):
        image = np.full((16, 16, 4), 0.3, np.float32)
        image[:, 8:, :3] = 0.7; image[..., 3] = 1.0
        out = apply_stroke(image, self._stroke("sharpen", [{"x": 8, "y": 8, "pressure": 1.0}], size=8.0), 1)
        # Right at the edge, sharpening should push the bright side brighter and the dark side darker.
        self.assertGreater(float(out[8, 8, 0]), float(image[8, 8, 0]))
        self.assertLess(float(out[8, 7, 0]), float(image[8, 7, 0]))

    def test_smear_drags_colour_from_behind_the_stroke_direction(self):
        image = np.zeros((16, 24, 4), np.float32)
        image[1:4, 6:9, :3] = 1.0; image[..., 3] = 1.0   # a bright patch, no colour at x=10
        stroke = self._stroke("smear", [{"x": 10, "y": 2, "pressure": 1.0}, {"x": 14, "y": 2, "pressure": 1.0}],
                              size=10.0)
        out = apply_stroke(image, stroke, 1)
        self.assertGreater(float(out[2, 10, 0]), 0.3)   # colour pulled in from the bright patch
        np.testing.assert_array_equal(out[2, 1], image[2, 1])   # untouched, outside the stroke

    def test_smear_is_no_longer_an_alias_for_blur(self):
        image = self._delta()
        stroke_points = [{"x": 10, "y": 10, "pressure": 1.0}, {"x": 16, "y": 10, "pressure": 1.0}]
        blurred = apply_stroke(image, self._stroke("blur", stroke_points, size=10.0), 1)
        smeared = apply_stroke(image, self._stroke("smear", stroke_points, size=10.0), 1)
        self.assertFalse(np.array_equal(blurred, smeared))


class DodgeBurnStrengthTests(unittest.TestCase):
    def _stroke(self, tool, strength):
        return {"kind": "stroke", "name": "stroke1", "points": [{"x": 8, "y": 8, "pressure": 1.0}],
                "brush": {"size": 8.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.25, "strength": strength},
                "tool": tool, "lifetime": {"mode": "all"}, "color": [0.0, 0.0, 0.0, 1.0],
                "source_offset": [0.0, 0.0], "source_frame": "relative",
                "opacity": 1.0, "blend": "over", "visible": True, "follow_track": None, "patch_blend": 0.0}

    def test_default_strength_matches_the_old_fixed_dodge_and_burn_constants(self):
        image = np.full((16, 16, 4), 0.4, np.float32)
        dodged = apply_stroke(image, self._stroke("dodge", 0.2), 1)
        burned = apply_stroke(image, self._stroke("burn", 0.2), 1)
        np.testing.assert_allclose(dodged[8, 8], image[8, 8] + (1 - image[8, 8]) * 0.2, atol=1e-5)
        np.testing.assert_allclose(burned[8, 8], image[8, 8] * 0.8, atol=1e-5)

    def test_a_higher_strength_dodges_and_burns_further(self):
        image = np.full((16, 16, 4), 0.4, np.float32)
        weak_dodge = apply_stroke(image, self._stroke("dodge", 0.1), 1)
        strong_dodge = apply_stroke(image, self._stroke("dodge", 0.8), 1)
        self.assertGreater(float(strong_dodge[8, 8, 0]), float(weak_dodge[8, 8, 0]))
        weak_burn = apply_stroke(image, self._stroke("burn", 0.1), 1)
        strong_burn = apply_stroke(image, self._stroke("burn", 0.8), 1)
        self.assertLess(float(strong_burn[8, 8, 0]), float(weak_burn[8, 8, 0]))

    def test_zero_strength_leaves_the_pixel_unchanged(self):
        image = np.full((16, 16, 4), 0.4, np.float32)
        dodged = apply_stroke(image, self._stroke("dodge", 0.0), 1)
        burned = apply_stroke(image, self._stroke("burn", 0.0), 1)
        np.testing.assert_allclose(dodged[8, 8], image[8, 8], atol=1e-6)
        np.testing.assert_allclose(burned[8, 8], image[8, 8], atol=1e-6)

    def test_old_document_migrates_the_brush_strength_and_renders_identically(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant",
                   "params": {"width": 8, "height": 8, "red": .4, "green": .4, "blue": .4, "alpha": 1}})
        d.execute({"op": "create", "id": "rp", "type": "RotoPaint"})
        d.execute({"op": "connect", "id": "rp", "input": "image", "source": "src"})
        d.execute({"op": "set_paint_items", "id": "rp", "items": [self._stroke("dodge", 0.2)]})
        before = Evaluator().evaluate_raster(d.document, "rp").pixels
        old = copy.deepcopy(d.document)
        old["version"] = 16
        del old["node_data"]["rp"]["items"][0]["brush"]["strength"]
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["node_data"]["rp"]["items"][0]["brush"]["strength"], 0.2)
        after = Evaluator().evaluate_raster(upgraded, "rp").pixels
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
