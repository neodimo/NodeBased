"""2D behaviour, lane 9 step X3: the two Bokeh knobs the fidelity report listed as missing.

Source: the Nuke 17.0 Reference Guide, Bokeh page (learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/bokeh.html).
* Integrate Deep Front and Back Separately (`integrateFrontAndBackSeperately`, Bokeh tab, after Normalize Visualization, default off): acts on the Deep
  input only. NodeBased has no Deep image, so it changes no pixels, and the test says so.
* Expand Bounding Box To Max Kernel Size (`expandBBoxToMaxKernelSize`, Kernel tab, after Maximum Kernel Size, default off): "when enabled, the bounding
  box is increased to accommodate the kernel size if the kernel extends beyond the bounding box. When disabled, the parts of the kernel extending past
  the bounding box are ignored."
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import bokehknobs, core, knobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor
from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels, tile_pixels
from tests.test_l9_x1_bokeh import H, W, differs, labels, texture

REACH = 4                                    # a maximum kernel size of 8 reaches 4 pixels from a pixel
BASE = dict(multiplier=0.4, max_kernelsize=8.0)


def white_graph(**params):
    g = Graph()
    g.add("plate", "Constant", dict(width=W, height=H, red=1.0, green=1.0, blue=1.0, alpha=1.0))
    g.add("node", "Bokeh", {**BASE, **params}, image="plate")
    return g


def raster(g):
    return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=1)


class ExpandBoundingBoxTests(unittest.TestCase):
    def test_off_the_window_is_the_sources_and_on_it_grows_by_the_filters_reach(self):
        self.assertEqual(bokehknobs.reach({"max_kernelsize": 8.0}), REACH)
        off, on = raster(white_graph()), raster(white_graph(expandBBoxToMaxKernelSize=1))
        self.assertEqual((off.data.x, off.data.y, off.data.width, off.data.height), (0, 0, W, H))
        self.assertEqual((on.data.x, on.data.y, on.data.width, on.data.height), (-REACH, -REACH, W + 2 * REACH, H + 2 * REACH))
        self.assertEqual((on.display.width, on.display.height), (W, H))             # the format does not change

    def test_the_blur_that_spills_past_the_picture_lands_in_the_ring_and_nothing_is_lost(self):
        on = raster(white_graph(expandBBoxToMaxKernelSize=1)).pixels
        self.assertGreater(float(on[REACH + H // 2, 0, 0]), 0.0)                    # just outside the left edge, the spill
        self.assertEqual(float(on[0, 0, 0]), 0.0)                                   # a ring corner is further than the filter reaches
        self.assertAlmostEqual(float(on[..., 0].sum()), float(H * W), delta=0.02 * H * W)          # a disc gathers and keeps its energy

    def test_off_the_parts_of_the_kernel_past_the_box_are_ignored_so_the_edge_keeps_its_colour(self):
        off = raster(white_graph()).pixels
        on = raster(white_graph(expandBBoxToMaxKernelSize=1)).pixels
        np.testing.assert_allclose(off[..., :3], 1.0, atol=1e-5)                    # a white plate stays white to its edge
        edge = on[REACH:REACH + H, REACH:REACH + W]
        self.assertLess(float(edge[H // 2, 0, 0]), 0.9)                              # on, the edge softens into the ring
        np.testing.assert_allclose(edge[H // 2, W // 2 - 2:W // 2 + 2, :3], 1.0, atol=1e-5)    # the middle is the same either way

    def test_the_knob_changes_the_pixels_a_viewer_shows_even_when_the_window_is_the_format(self):
        g = white_graph()
        g.doc["nodes"]["node"]["params"]["expandBBoxToMaxKernelSize"] = 1
        on = evaluator_pixels(g.doc, "node")
        self.assertEqual(on.shape[:2], (H, W))
        self.assertTrue(differs(on, evaluator_pixels(white_graph().doc, "node")))

    def test_a_cropped_plate_shows_the_ring_inside_the_format(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=W, height=H, red=1.0, green=1.0, blue=1.0, alpha=1.0))
        g.add("crop", "Crop", dict(x=4, y=3, width=8, height=6, reformat=0, intersect=0), image="plate")
        g.add("node", "Bokeh", {**BASE, "expandBBoxToMaxKernelSize": 1}, image="crop")
        g.add("plain", "Bokeh", dict(BASE), image="crop")
        on = Evaluator().evaluate(dict(g.doc, view="node"))
        off = Evaluator().evaluate(dict(g.doc, view="plain"))
        self.assertEqual(float(off[0, 0, 0]), 0.0)                                   # off, nothing is drawn outside the cropped window
        self.assertGreater(float(on[3 + 3, 4 - 2, 0]), 0.0)                          # on, the blur reaches two pixels outside it
        self.assertGreater(float(on[3 - 2, 4 + 4, 0]), 0.0)
        self.assertEqual(float(on[0, 0, 0]), 0.0)

    def test_the_mask_and_mix_apply_on_the_grown_window(self):
        g = white_graph(expandBBoxToMaxKernelSize=1, mix=0.5)
        half = raster(g).pixels
        full = raster(white_graph(expandBBoxToMaxKernelSize=1)).pixels
        source = np.zeros_like(full)
        source[REACH:REACH + H, REACH:REACH + W] = 1.0
        np.testing.assert_allclose(half[..., 0], source[..., 0] + (full[..., 0] - source[..., 0]) * 0.5, atol=1e-5)

    def test_the_depth_ring_takes_the_nearest_edge_depth(self):
        depth = np.tile(np.linspace(0.5, 6.0, W, dtype=np.float32), (H, 1))
        plain = Evaluator._windowed_kernel("Bokeh", {**SPECS["Bokeh"]["params"], **BASE, "focalPlane": 2.0, "focusRegionSize": 0.5},
                                           [Raster(texture()), Raster(np.repeat(depth[..., None], 4, axis=2))])
        grown = Evaluator._windowed_kernel("Bokeh", {**SPECS["Bokeh"]["params"], **BASE, "focalPlane": 2.0, "focusRegionSize": 0.5,
                                                     "expandBBoxToMaxKernelSize": 1},
                                           [Raster(texture()), Raster(np.repeat(depth[..., None], 4, axis=2))])
        self.assertEqual((grown.data.width, grown.data.height), (W + 2 * REACH, H + 2 * REACH))
        self.assertTrue(np.isfinite(grown.pixels).all())
        self.assertEqual(plain.pixels.shape[:2], (H, W))

    def test_tiles_cannot_do_a_growing_window_so_the_executor_hands_it_to_the_whole_frame(self):
        self.assertTrue(bokehknobs.needs_full_frame({**BASE, "expandBBoxToMaxKernelSize": 1}))
        self.assertFalse(bokehknobs.needs_full_frame(BASE))
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=7)
        self.assertFalse(executor.supports_tiled(dict(white_graph(expandBBoxToMaxKernelSize=1).doc, view="node"), "node"))
        self.assertTrue(executor.supports_tiled(dict(white_graph().doc, view="node"), "node"))


class OldDocumentsTests(unittest.TestCase):
    def test_a_node_saved_without_the_two_params_loads_and_renders_the_same_on_both_paths(self):
        picture = Graph()
        picture.add("plate", "Checker", dict(width=45, height=37, size=5))
        picture.add("node", "Bokeh", dict(multiplier=0.5, max_kernelsize=12.0), image="plate")
        saved = {key: value for key, value in picture.doc["nodes"]["node"]["params"].items()
                 if key not in ("expandBBoxToMaxKernelSize", "integrateFrontAndBackSeperately")}
        picture.doc["nodes"]["node"]["params"] = saved
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "old.nodebased")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(dict(picture.doc, view="node"), handle)
            loaded = core.load_document(path)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["expandBBoxToMaxKernelSize"], params["integrateFrontAndBackSeperately"]), (0, 0))
        full = evaluator_pixels(loaded, "node")
        for edge in (7, 16, 32):
            np.testing.assert_allclose(tile_pixels(loaded, "node", edge), full, atol=1e-5, err_msg=f"tile {edge}")


class IntegrateDeepTests(unittest.TestCase):
    def test_integrate_front_and_back_separately_acts_on_the_deep_input_only_so_no_pixel_changes(self):
        depth = np.tile(np.linspace(0.5, 6.0, W, dtype=np.float32), (H, 1))
        wired = [Raster(texture()), Raster(np.repeat(depth[..., None], 4, axis=2))]
        base = {**SPECS["Bokeh"]["params"], "focalPlane": 2.0, "focusRegionSize": 0.5, "multiplier": 0.4, "max_kernelsize": 12.0}
        one = Evaluator._windowed_kernel("Bokeh", base, wired).pixels
        two = Evaluator._windowed_kernel("Bokeh", {**base, "integrateFrontAndBackSeperately": 1}, wired).pixels
        np.testing.assert_array_equal(one, two)
        self.assertTrue(differs(one, texture()))                                        # the defocus itself is running


class LayoutTests(unittest.TestCase):
    def test_the_two_knobs_sit_where_the_guide_puts_them_with_its_labels_and_defaults(self):
        shown = labels("Bokeh")
        self.assertEqual(shown[shown.index("Normalize Visualization") + 1], "Integrate Deep Front and Back Separately")
        self.assertEqual(shown[shown.index("Real World Lens Simulation") - 1], "Integrate Deep Front and Back Separately")
        self.assertEqual(shown[shown.index("Maximum Kernel Size") + 1], "Expand Bounding Box To Max Kernel Size")
        self.assertEqual(shown[shown.index("Expand Bounding Box To Max Kernel Size") + 1], "Missing Kernel Channel")
        self.assertEqual((SPECS["Bokeh"]["params"]["integrateFrontAndBackSeperately"], SPECS["Bokeh"]["params"]["expandBBoxToMaxKernelSize"]), (0, 0))
        for name in ("integrateFrontAndBackSeperately", "expandBBoxToMaxKernelSize"):
            self.assertEqual(core.LIMITS[name], (0, 1))
        by_name = {g.params[0]: g for g in knobs.KNOB_LAYOUT["Bokeh"]}
        self.assertEqual(by_name["integrateFrontAndBackSeperately"].kind, "bool")
        self.assertEqual(by_name["expandBBoxToMaxKernelSize"].kind, "bool")


if __name__ == "__main__":
    unittest.main()
