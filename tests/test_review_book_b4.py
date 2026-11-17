"""Lane 11 step B4: the Transform group's review-book rows (SplineWarp's root warp, Tile, STMap and IDistort panels)."""
import unittest

import numpy as np

from nodebased import knobs, ops2d_chrome
from nodebased.core import CHOICES, LIMITS, SPECS
from tests.test_2d_behaviour_filter_g import FRAME, H, W, pin_params, pinned, plate
from tests.test_2d_behaviour_warp_h import ramp, run_spline, spline_params


def first_params(kind):
    return [g.params[0] for g in knobs.knob_layout(kind)]


class SplineWarpRootWarpTests(unittest.TestCase):
    """Nuke 17.0 guide, SplineWarp: 'root warp (root_warp, default 1): Sets the global warp multiplier for all shape pairs and layers'."""

    def test_root_warp_is_the_global_multiplier(self):
        self.assertEqual(SPECS["SplineWarp"]["params"]["root_warp"], 1.0)
        self.assertEqual(LIMITS["root_warp"], (0.0, 1.0))
        self.assertEqual(knobs.resolve_kind("SplineWarp", "root_warp", 1.0), "float_slider")
        src = ramp()
        np.testing.assert_allclose(run_spline(spline_params(root_warp=0.0), src).pixels, src, atol=1e-6)
        full = run_spline(spline_params(), src).pixels
        half = run_spline(spline_params(root_warp=0.5), src).pixels
        self.assertGreater(float(np.abs(full - src).max()), 0.05)
        # Half the multiplier is the same as half the layer warp: the amounts multiply.
        np.testing.assert_allclose(half, run_spline(spline_params(layer_warp=0.5), src).pixels, atol=1e-6)
        self.assertFalse(np.allclose(half, full, atol=1e-4))

    def test_saved_direction_moves_to_warp_direction_and_renders_as_before(self):
        self.assertEqual(CHOICES["warp_direction"], ("A", "B"))
        src = ramp()
        for direction in ("A", "B"):
            saved = {k: v for k, v in spline_params().items() if k != "warp_direction"}
            saved["root_warp"] = direction
            nodes = {"n": {"type": "SplineWarp", "params": saved}}
            ops2d_chrome.upgrade_nodes(nodes)
            params = nodes["n"]["params"]
            self.assertEqual((params["root_warp"], params["warp_direction"]), (1.0, direction))
            np.testing.assert_array_equal(run_spline(params, src).pixels,
                                          run_spline(spline_params(warp_direction=direction), src).pixels)
        b = run_spline(spline_params(warp_direction="B"), src).pixels
        self.assertFalse(np.allclose(b, run_spline(spline_params(), src).pixels, atol=1e-4))


class TransformPanelTests(unittest.TestCase):
    def test_tile_mirror_checkboxes_read_mirror(self):
        labels = {g.params[0]: g.label for g in knobs.knob_layout("Tile")}
        self.assertEqual((labels["mirror_y"], labels["mirror_x"]), ("Mirror", "Mirror"))
        self.assertEqual(first_params("Tile")[:4], ["rows", "mirror_y", "columns", "mirror_x"])

    def test_stmap_and_idistort_follow_the_guide_order(self):
        self.assertEqual(first_params("STMap")[:2], ["channels", "uv_layer"])
        order = first_params("IDistort")
        self.assertEqual(order[:12], ["channels", "uv_layer", "u_channel", "v_channel", "uv_offset_x", "uv_scale_x", "blur",
                                      "blur_scale", "maskChannel", "invert_mask", "premultiplied", "filter"])
        self.assertEqual(SPECS["IDistort"]["params"]["maskChannel"], "none")


class VectorCornerPinFilterTests(unittest.TestCase):
    """Nuke 17.0 guide, VectorCornerPin: imageFilter (Cubic) 'Select the filtering algorithm to use when remapping pixels', blurSize (0)
    'Controls the amount of blur applied to the internally calculated STMap'."""

    def test_new_and_saved_defaults(self):
        from nodebased import vectorcornerpinknobs as vcp
        params = SPECS["VectorCornerPin"]["params"]
        self.assertEqual((params["image_filter"], params["blur_size"]), ("cubic", 0.0))
        self.assertEqual((vcp.UPGRADE["image_filter"], vcp.UPGRADE["blur_size"]), ("bilinear", 0.0))
        self.assertEqual(knobs.resolve_kind("VectorCornerPin", "image_filter", "cubic"), "enum")

    def test_filter_changes_a_fractional_warp(self):
        half = tuple((x + 2.5, y + 1.5) for x, y in FRAME)
        bilinear = pinned(**pin_params(half), image_filter="bilinear")
        cubic = pinned(**pin_params(half), image_filter="cubic")
        impulse = pinned(**pin_params(half), image_filter="impulse")
        self.assertFalse(np.allclose(impulse, bilinear, atol=1e-4))
        # A ramp is linear inside, so bilinear and cubic agree away from the edges and differ only where cubic reaches past them.
        np.testing.assert_allclose(cubic[3:-3, 4:-4], bilinear[3:-3, 4:-4], atol=1e-5)
        # Impulse takes whole pixels of the plate, no new values.
        self.assertTrue(set(np.round(impulse[..., 0].ravel(), 6)) <= set(np.round(plate()[..., 0].ravel(), 6)) | {0.0})

    def test_bilinear_with_no_blur_is_the_saved_warp(self):
        moved = tuple((x + 3, y + 2) for x, y in FRAME)
        shifted = pinned(**pin_params(moved), image_filter="bilinear")
        src = plate()
        np.testing.assert_allclose(shifted[:H - 2, 3:], src[2:, :W - 3], atol=1e-6)

    def test_blur_size_softens_the_st_map(self):
        corner = ((0, 0), (W, 0), (W, H), (4, H - 3))     # one pin pulled in: the st-map bends near it
        sharp = pinned(**pin_params(corner), image_filter="bilinear")
        soft = pinned(**pin_params(corner), image_filter="bilinear", blur_size=4.0)
        self.assertGreater(float(np.abs(soft - sharp).max()), 1e-3)
        np.testing.assert_array_equal(pinned(**pin_params(corner), image_filter="bilinear", blur_size=0.4), sharp)   # under half a pixel: no blur


if __name__ == "__main__":
    unittest.main()
