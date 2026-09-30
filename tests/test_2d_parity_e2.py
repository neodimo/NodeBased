"""Lane L2 (2D parity) plan 13, step E2 of 2: close the last partial rows named in
docs/PARITY_2D.md -- CrossTalk fringe and unpremult, CurveTool AutoCrop by colour, DustBust patch
synthesis, and Inpaint on the tile path (tiles equal to full-frame, the same "solve once, slice
many" shape step E1 gave TimeBlur/TimeEcho, see tests/test_2d_parity_time_e1.py)."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS
from nodebased.dustbust import dustbust_items_for_specks
from nodebased.imaging import Evaluator, curve_tool_metrics
from nodebased import paint
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS
from tests.test_2d_parity_group_3b import kernel


class CrossTalkFringeAndUnpremultTests(unittest.TestCase):
    def test_unpremult_by_alpha_divides_the_curve_input_and_multiplies_back(self):
        # A half-covered pixel: straight red is 1.0, premultiplied red is 0.5. A curve that maps
        # r -> r*0.5 must halve the STRAIGHT value (giving premultiplied 0.25), not the already
        # premultiplied 0.5 (which would give 0.125). This is the observable difference unpremult
        # by alpha makes on a semi-transparent pixel.
        image = np.array([[[0.5, 0.0, 0.0, 0.5]]], dtype=np.float32)
        curve = '{"interpolation":"linear","points":[[0,0],[1,0.5]]}'
        out = kernel("CrossTalk", image, xt_curve_r_r=curve, xt_unpremult="alpha")
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.25, places=5)
        # Without unpremult, the same curve halves the premultiplied value directly.
        out_plain = kernel("CrossTalk", image, xt_curve_r_r=curve)
        self.assertAlmostEqual(float(out_plain[0, 0, 0]), 0.25, places=5)

    def test_unpremult_leaves_fully_opaque_pixels_unchanged_from_the_plain_path(self):
        image = np.array([[[0.4, 0.2, 0.1, 1.0]]], dtype=np.float32)
        curve = '{"interpolation":"linear","points":[[0,0],[1,0.6]]}'
        plain = kernel("CrossTalk", image, xt_curve_r_r=curve)
        unpremult = kernel("CrossTalk", image, xt_curve_r_r=curve, xt_unpremult="alpha")
        np.testing.assert_allclose(plain, unpremult, atol=1e-6)

    def test_unpremult_by_zero_alpha_passes_that_pixel_through_instead_of_dividing_by_zero(self):
        image = np.array([[[0.3, 0.0, 0.0, 0.0]]], dtype=np.float32)
        curve = '{"interpolation":"linear","points":[[0,0],[1,0.9]]}'
        out = kernel("CrossTalk", image, xt_curve_r_r=curve, xt_unpremult="alpha")
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.3, places=5)

    def test_fringe_limits_the_effect_to_partial_alpha_edge_pixels(self):
        image = np.zeros((1, 3, 4), np.float32)
        image[0, 0] = (0.5, 0.0, 0.0, 1.0)   # solid interior
        image[0, 1] = (0.5, 0.0, 0.0, 0.5)   # partial-alpha edge
        image[0, 2] = (0.5, 0.0, 0.0, 0.0)   # fully transparent
        curve = '{"interpolation":"linear","points":[[0,0],[1,0]]}'   # maps red to 0
        out = kernel("CrossTalk", image, xt_curve_r_r=curve, xt_fringe=True)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.5, places=5)   # solid: untouched
        self.assertAlmostEqual(float(out[0, 1, 0]), 0.0, places=5)   # edge: curve applied
        self.assertAlmostEqual(float(out[0, 2, 0]), 0.5, places=5)   # transparent: untouched

    def test_fringe_off_applies_everywhere_as_before(self):
        image = np.zeros((1, 2, 4), np.float32)
        image[0, 0] = (0.5, 0.0, 0.0, 1.0)
        image[0, 1] = (0.5, 0.0, 0.0, 0.5)
        curve = '{"interpolation":"linear","points":[[0,0],[1,0]]}'
        out = kernel("CrossTalk", image, xt_curve_r_r=curve, xt_fringe=False)
        np.testing.assert_allclose(out[..., 0], 0.0, atol=1e-6)

    def test_registered_params_and_choices(self):
        self.assertIn("xt_fringe", SPECS["CrossTalk"]["params"])
        self.assertIn("xt_unpremult", SPECS["CrossTalk"]["params"])
        self.assertEqual(SPECS["CrossTalk"]["params"]["xt_fringe"], False)
        self.assertEqual(SPECS["CrossTalk"]["params"]["xt_unpremult"], "none")
        self.assertEqual(CHOICES["xt_unpremult"], ["none", "red", "green", "blue", "alpha"])

    def test_old_documents_load_and_render_unchanged(self):
        # A document without the new params (as an old save file would decode to via SPECS
        # defaults) renders identically to one with them explicitly set to their defaults.
        image = np.random.default_rng(4).random((3, 3, 4)).astype(np.float32)
        params = dict(SPECS["CrossTalk"]["params"])
        del params["xt_fringe"]; del params["xt_unpremult"]
        legacy = Evaluator._kernel("CrossTalk", {**dict(SPECS["CrossTalk"]["params"]), **params,
                                                 "xt_fringe": False, "xt_unpremult": "none"}, [image])
        current = kernel("CrossTalk", image)
        np.testing.assert_array_equal(legacy, current)


class CurveToolAutoCropByColourTests(unittest.TestCase):
    def test_alpha_mode_is_unchanged_from_before(self):
        image = np.zeros((6, 6, 4), np.float32)
        image[1:4, 2:5, 3] = 1.0
        measured = curve_tool_metrics(image, autocrop_mode="alpha")
        self.assertEqual((measured["crop_x"], measured["crop_y"]), (2.0, 1.0))
        self.assertEqual((measured["crop_width"], measured["crop_height"]), (3.0, 3.0))

    def test_color_mode_bounds_pixels_that_are_not_the_target_colour(self):
        image = np.zeros((6, 6, 4), np.float32)
        image[..., :3] = (0.1, 0.1, 0.1)   # background colour everywhere
        image[1:4, 2:5, :3] = (0.9, 0.9, 0.9)   # foreground block
        measured = curve_tool_metrics(image, autocrop_mode="color",
                                      autocrop_color=(0.1, 0.1, 0.1), autocrop_tolerance=0.05)
        self.assertEqual((measured["crop_x"], measured["crop_y"]), (2.0, 1.0))
        self.assertEqual((measured["crop_width"], measured["crop_height"]), (3.0, 3.0))

    def test_color_mode_tolerance_absorbs_a_near_match(self):
        image = np.zeros((4, 4, 4), np.float32)
        image[..., :3] = (0.5, 0.5, 0.5)
        image[2, 2, :3] = (0.52, 0.5, 0.5)   # within a 0.05 tolerance of the target
        measured = curve_tool_metrics(image, autocrop_mode="color",
                                      autocrop_color=(0.5, 0.5, 0.5), autocrop_tolerance=0.05)
        self.assertEqual((measured["crop_width"], measured["crop_height"]), (0.0, 0.0))
        image[3, 3, :3] = (0.9, 0.5, 0.5)   # well outside tolerance
        measured2 = curve_tool_metrics(image, autocrop_mode="color",
                                       autocrop_color=(0.5, 0.5, 0.5), autocrop_tolerance=0.05)
        self.assertEqual((measured2["crop_x"], measured2["crop_y"]), (3.0, 3.0))

    def test_registered_params_choices_and_limits(self):
        params = SPECS["CurveTool"]["params"]
        for name in ("autocrop_mode", "autocrop_color_r", "autocrop_color_g", "autocrop_color_b", "autocrop_tolerance"):
            self.assertIn(name, params)
        self.assertEqual(params["autocrop_mode"], "alpha")
        self.assertEqual(CHOICES["autocrop_mode"], ["alpha", "color"])
        self.assertIn("autocrop_tolerance", LIMITS)

    def test_bypass_and_old_documents_still_report_alpha_bounds_by_default(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "ct", "type": "CurveTool"})
        params = d.document["nodes"]["ct"]["params"]
        self.assertEqual(params["autocrop_mode"], "alpha")


def _small_constant_graph_with_matte(node_id="node", params=None):
    """A small plate feeding Inpaint through a Constant matte, small enough that a small
    `tile_edge` still produces several tiles quickly."""
    d = Dispatcher()
    d.execute({"op": "create", "id": "plate", "type": "Constant",
               "params": {"width": 48, "height": 48, "red": 0.4, "green": 0.3, "blue": 0.6, "alpha": 1.0}})
    d.execute({"op": "create", "id": "matte", "type": "Constant",
               "params": {"width": 48, "height": 48, "red": 0.0, "green": 0.0, "blue": 0.0, "alpha": 0.0}})
    d.execute({"op": "set", "id": "matte", "param": "alpha", "value": 1.0})
    d.execute({"op": "create", "id": node_id, "type": "Inpaint", "params": params or {}})
    d.execute({"op": "connect", "id": node_id, "input": "image", "source": "plate"})
    d.execute({"op": "connect", "id": node_id, "input": "matte", "source": "matte"})
    return d


class InpaintTilePathTests(unittest.TestCase):
    """Inpaint moves onto the tile path this step, the same "solve once, slice many" shape as
    TimeBlur/TimeEcho (`_temporal_tile` asks `Evaluator` for the node's own already-blended full
    result and slices tiles from it), so a tiled render must be pixel-identical to the full-frame
    evaluator -- tiles equal to full-frame."""

    def test_inpaint_is_a_supported_tiled_kind(self):
        self.assertIn("Inpaint", SUPPORTED_TILED_KINDS)

    def test_inpaint_tiled_result_matches_full_frame(self):
        d = _small_constant_graph_with_matte(params={"fill_method": "diffusion"})
        doc = dict(d.document, view="node")
        expected = Evaluator().evaluate(doc, "node", frame=1)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(doc, "node"))
        result = executor.compose(doc, "node", frame=1, tier=1)
        self.assertTrue(result.tiled)
        np.testing.assert_allclose(result.pixels, expected, atol=1e-6)

    def test_inpaint_bypass_on_the_tile_path_matches_the_evaluator(self):
        d = _small_constant_graph_with_matte()
        d.execute({"op": "disable", "id": "node", "value": True})
        doc = dict(d.document, view="node")
        expected = Evaluator().evaluate(doc, "node", frame=1)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        result = executor.compose(doc, "node", frame=1, tier=1)
        self.assertTrue(result.tiled)
        np.testing.assert_allclose(result.pixels, expected, atol=1e-6)


class DustBustPatchSynthesisTests(unittest.TestCase):
    """DustBust / MarkerRemoval: patch synthesis for accepted specks -- fill the speck region from
    the surrounding pixels of the same frame (a border-in diffusion, `flow_nodes.spatial_fill`),
    blended with the previous-frame clone by `patch_blend`, so a speck on a moving background no
    longer leaves a ghost of the stale previous frame."""

    @staticmethod
    def _gradient_frame(size, shift):
        column = np.clip((np.arange(size, dtype=np.float32) + shift) / size, 0.0, 1.0)
        rgb = np.tile(column[None, :, None], (size, 1, 3))
        alpha = np.ones((size, size, 1), np.float32)
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    def test_patch_blend_default_is_zero_and_backward_compatible(self):
        specks = [{"frame": 2, "x": 3.0, "y": 3.0, "width": 2.0, "height": 2.0, "magnitude": 0.7}]
        items = dustbust_items_for_specks([], specks)
        self.assertEqual(items[0]["patch_blend"], 0.0)
        plate1 = np.full((8, 8, 4), 0.2, np.float32); plate1[..., 3] = 1.0
        plate2 = plate1.copy(); plate2[3:5, 3:5, :3] = 0.9
        out = paint.rasterise(plate2, items, frame=2, source=plate1)
        np.testing.assert_allclose(out[3:5, 3:5, :3], 0.2, atol=1e-5)

    def test_full_patch_blend_fills_from_the_current_frame_not_a_stale_previous_frame(self):
        size = 40
        clean_frame2 = self._gradient_frame(size, shift=1.0)
        frame1_stale = self._gradient_frame(size, shift=0.0)   # the moving background one step back
        speck_frame = clean_frame2.copy()
        speck_frame[17:20, 17:20, :3] = (1.0, 0.0, 0.0)   # a bright two-frame speck
        specks = [{"frame": 2, "x": 17.0, "y": 17.0, "width": 3.0, "height": 3.0, "magnitude": 1.0}]
        items = dustbust_items_for_specks([], specks, patch_blend=1.0)
        self.assertEqual(items[0]["patch_blend"], 1.0)
        result = paint.rasterise(speck_frame, items, frame=2, source=frame1_stale)
        centre = result[18, 18, :3]
        expected = clean_frame2[18, 18, :3]
        self.assertLess(float(np.max(np.abs(centre - expected))), 0.02,
                        "patch-synthesised centre pixel must stay within 2%% of the clean plate")
        # And it must actually differ from a pure previous-frame clone, proving the fill used the
        # current frame's own border rather than the stale, shifted gradient one frame back.
        cloned_only = dustbust_items_for_specks([], specks, patch_blend=0.0)
        cloned_result = paint.rasterise(speck_frame, cloned_only, frame=2, source=frame1_stale)
        self.assertGreater(float(np.max(np.abs(cloned_result[18, 18, :3] - expected))), 0.02)

    def test_partial_patch_blend_mixes_clone_and_patch(self):
        size = 40
        clean_frame2 = self._gradient_frame(size, shift=1.0)
        frame1_stale = self._gradient_frame(size, shift=0.0)
        speck_frame = clean_frame2.copy()
        speck_frame[17:20, 17:20, :3] = (1.0, 0.0, 0.0)
        specks = [{"frame": 2, "x": 17.0, "y": 17.0, "width": 3.0, "height": 3.0, "magnitude": 1.0}]
        half = dustbust_items_for_specks([], specks, patch_blend=0.5)
        self.assertAlmostEqual(half[0]["patch_blend"], 0.5)
        full = dustbust_items_for_specks([], specks, patch_blend=1.0)
        none = dustbust_items_for_specks([], specks, patch_blend=0.0)
        out_half = paint.rasterise(speck_frame, half, frame=2, source=frame1_stale)
        out_full = paint.rasterise(speck_frame, full, frame=2, source=frame1_stale)
        out_none = paint.rasterise(speck_frame, none, frame=2, source=frame1_stale)
        # The half-blend centre pixel must lie strictly between the pure-clone and pure-patch
        # results (or match one of them, never overshoot past either).
        lo = min(out_full[18, 18, 0], out_none[18, 18, 0])
        hi = max(out_full[18, 18, 0], out_none[18, 18, 0])
        self.assertGreaterEqual(out_half[18, 18, 0] + 1e-5, lo)
        self.assertLessEqual(out_half[18, 18, 0] - 1e-5, hi)

    def test_rotopaint_registers_the_patch_blend_knob(self):
        self.assertIn("dustbust_patch_blend", SPECS["RotoPaint"]["params"])
        self.assertEqual(SPECS["RotoPaint"]["params"]["dustbust_patch_blend"], 0.5)
        self.assertIn("dustbust_patch_blend", LIMITS)


if __name__ == "__main__":
    unittest.main()
