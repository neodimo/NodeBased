import unittest

import numpy as np

from nodebased import colorcurves
from nodebased.core import Dispatcher, SPECS
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


if __name__ == "__main__":
    unittest.main()
