"""Lane 11 step B4: the Transform group's review-book rows (SplineWarp's root warp, Tile, STMap and IDistort panels)."""
import unittest

import numpy as np

from nodebased import knobs, ops2d_chrome
from nodebased.core import CHOICES, LIMITS, SPECS
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


if __name__ == "__main__":
    unittest.main()
