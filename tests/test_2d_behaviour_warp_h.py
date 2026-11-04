"""2D behaviour, lane 9 step H: GridWarp and SplineWarp take the inputs, mask controls and knobs Nuke's Reference Guide lists.

* GridWarp: `dst` and `bg` inputs after the mask; the output chooses the source, the warped source, the destination or the warped destination; the background
  and its mix composite the warped picture over black, the source, the destination or bg; the mask strip (channels, mask channel, invert, inject, fringe) acts;
  a grid with one interior key moved moves the picture under it by that offset; mix 0 is the source; older graphs render as before;
* SplineWarp: premultiply, layer warp, pair warp, the boundary curve box, crop to format and the output mask layer act; a curve paired with one shifted
  by a known offset moves the pixels on it; mix 0 is the input; older graphs keep their window and add no layer; the main tab keeps the knobs in order.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, knobs, ops2d_chrome, shapes, warpknobs
from nodebased.core import Dispatcher, SPECS, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tiers import Region

import nuke_fidelity_cases as ruler  # noqa: E402

W, H = 20, 16


def ramp(offset=0.0, alpha=1.0):
    yy, xx = np.mgrid[:H, :W]
    out = np.zeros((H, W, 4), np.float32)
    out[..., 0] = xx / W + offset
    out[..., 1] = yy / H + offset
    out[..., 2] = 0.25 + offset
    out[..., 3] = alpha
    return out


def raster(pixels):
    region = Region(0, 0, pixels.shape[1], pixels.shape[0])
    return Raster(pixels, region, region)


def point(x, y, in_y=0.0, out_y=0.0):
    return {"x": float(x), "y": float(y), "in_x": 0.0, "in_y": float(in_y), "out_x": 0.0, "out_y": float(out_y)}


def line(x, top, bottom):
    """A straight curve at column x: handles a third of the length along it make the Bezier evenly sampled (zero handles bunch the samples at the ends)."""
    third = (bottom - top) / 3.0
    return [point(x, top, out_y=third), point(x, bottom, in_y=-third)]


XS, YS = (0.5, 9.5, 19.5), (0.5, 7.5, 15.5)
GRID = [[[x, y] for x in XS] for y in YS]


def moved_grid(dx=3.0, dy=0.0):
    out = copy.deepcopy(GRID)
    out[1][1] = [GRID[1][1][0] + dx, GRID[1][1][1] + dy]
    return {"source": GRID, "destination": out}


def grid_params(**knob):
    return {**SPECS["GridWarp"]["params"], "rows": 3, "columns": 3, "filter": "bilinear", **knob}


def run_grid(params, source, data, dst=None, bg=None, mask=None):
    inputs = [raster(source), None if mask is None else raster(mask), None if dst is None else raster(dst), None if bg is None else raster(bg)]
    return Evaluator._windowed_kernel("GridWarp", params, inputs, 1, data)


CURVE_PAIRS = [{"name": "edge", "source": line(9.5, 0.5, 15.5), "destination": line(12.5, 0.5, 15.5)}]


def spline_params(**knob):
    return {**SPECS["SplineWarp"]["params"], "curve_resolution": 15, "filter": "bilinear", **knob}


def run_spline(params, source, pairs=None, mask=None):
    inputs = [raster(source), None if mask is None else raster(mask)]
    resolved = shapes.resolve_warp_data("SplineWarp", {"pairs": CURVE_PAIRS if pairs is None else pairs}, 1)
    return Evaluator._windowed_kernel("SplineWarp", params, inputs, 1, resolved)


def labels(kind):
    return [g.label for g in knobs.KNOB_LAYOUT[kind] if not g.params[0].startswith("legacy_") and g.params != ("mix",)]


class GridWarpTests(unittest.TestCase):
    def test_interior_key_moves_the_pixel_under_it_and_corners_stay(self):
        src = ramp()
        out = run_grid(grid_params(), src, moved_grid()).pixels
        np.testing.assert_allclose(out[7, 12], src[7, 9], atol=1e-5)
        for row, col in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
            np.testing.assert_allclose(out[row, col], src[row, col], atol=1e-4)

    def test_mix_zero_is_the_source(self):
        src = ramp()
        np.testing.assert_array_equal(run_grid(grid_params(mix=0.0), src, moved_grid()).pixels, src)

    def test_warp_scales_the_known_interior_displacement(self):
        src = ramp()
        two_thirds = run_grid(grid_params(warp=2.0 / 3.0), src, moved_grid()).pixels
        expected = run_grid(grid_params(), src, moved_grid(dx=2.0)).pixels
        np.testing.assert_allclose(two_thirds, expected, atol=1e-6)
        np.testing.assert_array_equal(run_grid(grid_params(warp=0.0), src, moved_grid()).pixels, src)

    def test_source_and_destination_transform_translation_moves_the_known_grid(self):
        src = ramp()
        data = moved_grid()
        source_transform = run_grid(grid_params(source_grid_transform_translate_x=2.0), src, data).pixels
        np.testing.assert_allclose(source_transform[7, 14], src[7, 11], atol=1e-5)
        destination_transform = run_grid(grid_params(use_source_transform=0, destination_grid_transform_translate_x=2.0), src, data).pixels
        np.testing.assert_allclose(destination_transform[7, 14], src[7, 9], atol=1e-5)

    def test_each_source_and_destination_transform_control_changes_grid_pixels(self):
        src, data = ramp(), moved_grid()
        base = run_grid(grid_params(), src, data).pixels
        variants = (
            {"translate_x": 1.0}, {"translate_y": 1.0}, {"rotate": 12.0}, {"scale": 1.15},
            {"skew_x": 0.2}, {"skew_y": 0.2}, {"skew_x": 0.2, "skew_y": 0.35, "skew_order": "YX"},
            {"center_x": 4.0, "rotate": 12.0}, {"center_y": 4.0, "rotate": 12.0},
        )
        for grid in ("source", "destination"):
            for variant in variants:
                params = {f"{grid}_grid_transform_{name}": value for name, value in variant.items()}
                if grid == "destination":
                    params["use_source_transform"] = 0
                with self.subTest(grid=grid, transform=variant):
                    changed = run_grid(grid_params(**params), src, data).pixels
                    self.assertGreater(float(np.abs(changed - base).max()), 1e-4)

    def test_use_source_transform_couples_the_destination_transform(self):
        src, data = ramp(), moved_grid()
        inherited = run_grid(grid_params(source_grid_transform_translate_x=2.0, use_source_transform=1), src, data).pixels
        independent = run_grid(grid_params(source_grid_transform_translate_x=2.0, use_source_transform=0), src, data).pixels
        self.assertGreater(float(np.abs(inherited - independent).max()), 1e-4)

    def test_submesh_resolution_changes_warp_interpolation(self):
        src, data = ramp(), moved_grid()
        regular = run_grid(grid_params(render_res=10.0), src, data).pixels
        smoother = run_grid(grid_params(render_res=40.0), src, data).pixels
        self.assertGreater(float(np.abs(smoother - regular).max()), 1e-4)

    def test_output_choices(self):
        src, dst = ramp(), ramp(0.3)
        data = moved_grid()
        self.assertEqual(run_grid(grid_params(grid_output="source"), src, data, dst).pixels.tobytes(), src.tobytes())
        self.assertEqual(run_grid(grid_params(grid_output="destination"), src, data, dst).pixels.tobytes(), dst.tobytes())
        self.assertEqual(run_grid(grid_params(grid_output="destination"), src, data).pixels.tobytes(), src.tobytes())   # no dst wired: the source
        warped = run_grid(grid_params(), src, data, dst).pixels
        self.assertGreater(float(np.abs(warped - src).max()), 0.05)
        # the destination warped goes from the destination grid back to the source grid: its key pixel (9, 7) reads the dst picture at (12, 7)
        back = run_grid(grid_params(grid_output="destination warped"), src, data, dst).pixels
        np.testing.assert_allclose(back[7, 9], dst[7, 12], atol=1e-5)
        # with no dst wired the source is what is warped back
        alone = run_grid(grid_params(grid_output="destination warped"), src, data).pixels
        np.testing.assert_allclose(alone[7, 9], src[7, 12], atol=1e-5)

    def test_background_composites_the_warped_picture(self):
        src = ramp(alpha=0.5)
        src[..., :3] *= 0.5                                  # premultiplied half cover
        dst, bg = ramp(0.2), ramp(0.4)
        data = moved_grid()
        plain = run_grid(grid_params(), src, data, dst, bg).pixels
        np.testing.assert_array_equal(run_grid(grid_params(background_mix=0.0, background="on bg"), src, data, dst, bg).pixels, plain)
        results = {}
        for mode in warpknobs.BACKGROUNDS:
            results[mode] = run_grid(grid_params(background=mode, background_mix=1.0), src, data, dst, bg).pixels
            expected_under = {"on black": np.zeros_like(src) + np.array([0, 0, 0, 1], np.float32), "on src": src, "on dst": dst, "on bg": bg}[mode]
            np.testing.assert_allclose(results[mode], plain + expected_under * (1 - plain[..., 3:4]), atol=1e-5)
        half = run_grid(grid_params(background="on bg", background_mix=0.5), src, data, dst, bg).pixels
        np.testing.assert_allclose(half, (plain + results["on bg"]) / 2, atol=1e-5)
        no_bg = run_grid(grid_params(background="on bg", background_mix=1.0), src, data, dst).pixels     # no bg wired: transparent black
        np.testing.assert_allclose(no_bg, plain, atol=1e-6)

    def test_mask_strip_acts(self):
        g = Dispatcher(empty_document())
        g.execute({"op": "create", "id": "plate", "type": "Ramp", "params": {"width": 16, "height": 12}})
        g.execute({"op": "create", "id": "matte", "type": "Constant", "params": {"width": 16, "height": 12, "red": 1, "green": 1, "blue": 1, "alpha": 0.5}})
        g.execute({"op": "create", "id": "n", "type": "GridWarp", "params": {"rows": 2, "columns": 2}})
        g.execute({"op": "connect", "id": "n", "input": "image", "source": "plate"})
        corners = [[{"x": 0.5, "y": 0.5}, {"x": 15.5, "y": 0.5}], [{"x": 0.5, "y": 11.5}, {"x": 15.5, "y": 11.5}]]
        shifted = [[{"x": 3.5, "y": 0.5}, {"x": 15.5, "y": 0.5}], [{"x": 0.5, "y": 11.5}, {"x": 15.5, "y": 11.5}]]
        g.execute({"op": "set_warp_data", "id": "n", "data": {"source": corners, "destination": shifted}})

        def render(**knob):
            for name, value in knob.items():
                g.execute({"op": "set", "id": "n", "param": name, "value": value})
            return Evaluator().evaluate_raster(dict(g.document, view="n"), target="n")
        base = render()
        g.execute({"op": "connect", "id": "n", "input": "mask", "source": "matte"})
        masked = render()
        np.testing.assert_allclose(masked.pixels, 0.5 * base.pixels + 0.5 * Evaluator().evaluate_raster(dict(g.document, view="plate"), target="plate").pixels, atol=1e-5)
        inverted = render(invert_mask=1)
        np.testing.assert_allclose(inverted.pixels, masked.pixels, atol=1e-5)           # 0.5 inverted is 0.5
        render(invert_mask=0)
        self.assertNotIn("mask", render().layers or {})
        injected = render(inject=1)
        self.assertIn("mask", injected.layers)
        red_only = render(inject=0, channels="red")
        self.assertTrue(np.array_equal(red_only.pixels[..., 1], Evaluator().evaluate_raster(dict(g.document, view="plate"), target="plate").pixels[..., 1]))

    def test_dst_and_bg_follow_the_mask_so_saved_edges_keep_their_slot(self):
        self.assertEqual(SPECS["GridWarp"]["optional_inputs"], ["mask", "dst", "bg"])

    def test_graphs_saved_before_render_as_before(self):
        src = ramp()
        old = {k: v for k, v in grid_params().items() if k not in ("grid_output", "background", "background_mix", "channels", "maskChannelInput",
                                                                 "invert_mask", "inject", "fringe")}
        doc = {"nodes": {"n": {"type": "GridWarp", "params": old}}}
        ops2d_chrome.upgrade_nodes(doc["nodes"])
        self.assertEqual(doc["nodes"]["n"]["params"]["grid_output"], "source warped")
        self.assertEqual(doc["nodes"]["n"]["params"]["background_mix"], 0.0)
        np.testing.assert_array_equal(run_grid(doc["nodes"]["n"]["params"], src, moved_grid()).pixels,
                                      run_grid(grid_params(), src, moved_grid()).pixels)

    def test_gridwarptracker_still_reads_its_vectors_at_slot_two(self):
        self.assertEqual(SPECS["GridWarpTracker"]["optional_inputs"], ["mask", "vectors"])

    def test_main_tab_order(self):
        order = labels("GridWarp")
        self.assertEqual(order[:2], ["Channels", "Mask"])
        for earlier, later in (("Output", "Background"), ("Background", "Background mix"), ("Background mix", "Set bbox to")):
            self.assertLess(order.index(earlier), order.index(later), order)


class SplineWarpTests(unittest.TestCase):
    def test_curve_moves_the_pixels_on_it_and_mix_zero_is_the_input(self):
        src = ramp()
        out = run_spline(spline_params(), src).pixels
        for row in range(H):
            np.testing.assert_allclose(out[row, 12], src[row, 9], atol=1e-4)
        np.testing.assert_array_equal(run_spline(spline_params(mix=0.0), src).pixels, src)

    def test_layer_and_pair_warp_scale_the_move(self):
        src = ramp()
        third = 2.0 / 3.0
        half = run_spline(spline_params(layer_warp=third), src).pixels
        np.testing.assert_allclose(half[8, 11], src[8, 9], atol=1e-4)         # two thirds of 3 pixels is 2: the curve sits at 11.5, a pixel centre
        also = run_spline(spline_params(pair_warp=third), src).pixels
        np.testing.assert_allclose(also, half, atol=1e-6)
        both = run_spline(spline_params(layer_warp=third, pair_warp=third), src).pixels
        self.assertGreater(float(np.abs(both - half).max()), 1e-3)
        np.testing.assert_allclose(run_spline(spline_params(layer_warp=0.0), src).pixels, src, atol=1e-6)

    def test_transform_translate_moves_the_curve_and_the_pixels_under_it(self):
        src = ramp()
        out = run_spline(spline_params(translate_x=2.0), src).pixels
        np.testing.assert_allclose(out[8, 14], src[8, 11], atol=1e-5)
        np.testing.assert_allclose(run_spline(spline_params(), src).pixels[8, 12], src[8, 9], atol=1e-5)

    def test_transform_scale_and_rotation_move_curve_points(self):
        point = np.array([[3.0, 0.0]])
        np.testing.assert_allclose(warpknobs.transform_points(point, {"scale": 2.0}), [[6.0, 0.0]])
        np.testing.assert_allclose(warpknobs.transform_points(point, {"rotate": 90.0}), [[0.0, -3.0]], atol=1e-12)

    def test_transform_skews_and_center_are_pixel_coordinates(self):
        point = np.array([[3.0, 2.0]])
        np.testing.assert_allclose(warpknobs.transform_points(point, {"skew_x": 0.5}), [[4.0, 2.0]])
        np.testing.assert_allclose(warpknobs.transform_points(point, {"skew_y": 0.5}), [[3.0, 3.5]])
        np.testing.assert_allclose(warpknobs.transform_points(point, {"scale": 2.0, "center_x": 1.0, "center_y": 2.0}), [[5.0, 2.0]])

    def test_skew_order_changes_the_combined_transform(self):
        point = np.array([[1.0, 1.0]])
        xy = warpknobs.transform_points(point, {"skew_x": 1.0, "skew_y": 1.0, "skew_order": "XY"})
        yx = warpknobs.transform_points(point, {"skew_x": 1.0, "skew_y": 1.0, "skew_order": "YX"})
        self.assertFalse(np.array_equal(xy, yx))

    def test_premultiply_scales_the_colour_by_alpha(self):
        src = ramp(alpha=0.5)
        plain = run_spline(spline_params(), src).pixels
        pre = run_spline(spline_params(premultiply=1), src).pixels
        np.testing.assert_allclose(pre[..., :3], plain[..., :3] * 0.5, atol=1e-5)
        np.testing.assert_allclose(pre[..., 3], plain[..., 3], atol=1e-6)

    def test_boundary_box_and_crop_to_format(self):
        src = ramp()
        away = [{"name": "out", "source": line(15.5, 4.5, 10.5), "destination": line(26.5, 4.5, 10.5)}]
        grown = run_spline(spline_params(bbox="union", crop_to_format=0), src, away)
        self.assertGreater(grown.data.width, W)
        no_boundary = run_spline(spline_params(bbox="union", crop_to_format=0, boundary_bbox=0), src, away)
        self.assertEqual((no_boundary.data.width, no_boundary.data.height), (W, H))
        cropped = run_spline(spline_params(bbox="union", crop_to_format=1), src, away)
        self.assertEqual((cropped.data.x, cropped.data.y, cropped.data.width, cropped.data.height), (0, 0, W, H))

    def test_output_mask_layer(self):
        src = ramp()
        out = run_spline(spline_params(), src)
        layer = out.layers["mask_splinwarp"].pixels
        self.assertEqual(layer[8, 12, 3], 1.0)
        self.assertEqual(layer[8, 0, 3], 0.0)
        named = run_spline(spline_params(outputMask="warpmask.a"), src)
        self.assertIn("warpmask", named.layers)
        none = run_spline(spline_params(outputMask="none"), src)
        self.assertFalse(none.layers)

    def test_mask_strip_and_chrome_defaults(self):
        self.assertIn("channels", SPECS["SplineWarp"]["params"])          # the guide's table lists channels only; the mask input gates through mix
        self.assertNotIn("maskChannelInput", SPECS["SplineWarp"]["params"])
        src = ramp()
        matte = ramp()
        matte[..., 3] = 0.5
        half = run_spline(spline_params(), src, mask=matte).pixels
        full = run_spline(spline_params(), src).pixels
        np.testing.assert_allclose(half, 0.5 * full + 0.5 * src, atol=1e-5)

    def test_graphs_saved_before_keep_their_window_and_add_no_layer(self):
        doc = {"nodes": {"n": {"type": "SplineWarp", "params": {"mix": 1.0, "curve_resolution": 12, "bbox": "union", "root_warp": "A",
                                                                  "filter": "bilinear", "output": "image"}}}}
        ops2d_chrome.upgrade_nodes(doc["nodes"])
        params = doc["nodes"]["n"]["params"]
        self.assertEqual((params["crop_to_format"], params["outputMask"], params["boundary_bbox"]), (0, "none", 1))
        away = [{"name": "out", "source": line(15.5, 4.5, 10.5), "destination": line(26.5, 4.5, 10.5)}]
        out = run_spline(params, ramp(), away)
        self.assertGreater(out.data.width, W)
        self.assertFalse(out.layers)
        self.assertEqual((SPECS["SplineWarp"]["params"]["crop_to_format"], SPECS["SplineWarp"]["params"]["outputMask"]), (1, "mask_splinwarp.a"))

    def test_payload_holds_the_curves_and_keys(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "w", "type": "SplineWarp"})
        d.execute({"op": "set_warp_data", "id": "w", "data": {"pairs": CURVE_PAIRS}})
        validate(d.document)
        self.assertEqual(set(d.document["node_data"]["w"]), {"pairs"})

    def test_main_tab_order(self):
        order = labels("SplineWarp")
        self.assertEqual(order[:5], ["Channels", "Premultiply input", "Output mask", "Crop to format", "Bbox boundary curve"])
        for earlier, later in (("Bbox boundary curve", "Output"), ("Output", "Root warp"), ("Root warp", "Layer warp"), ("Layer warp", "Pair warp")):
            self.assertLess(order.index(earlier), order.index(later), order)


class FidelityCaseTests(unittest.TestCase):
    def test_cases_pass(self):
        found = [c for c in ruler.CASES if c.nb_kind in ("GridWarp", "SplineWarp")]
        self.assertGreaterEqual(len(found), 6)
        for c in found:
            out = ruler.run_nodebased(c)
            ref, valid = c.expected(ruler.make_fixtures())
            self.assertLess(float(np.abs(out - ref)[valid].max()), 2e-3, c.id)


if __name__ == "__main__":
    unittest.main()
