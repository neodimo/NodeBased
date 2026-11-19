"""2D behaviour, lane 9 step X5: Text's `group_animation` ("animation layers", Groups tab; the guide lists no default) and MarkerRemoval's panel order.

* Text: the knob is held on the node (empty by default), listed after Centre offset where the guide has it, changes no pixel (NodeBased's Text draws one
  block with no per-character groups), and a node saved without it loads with the empty entry;
* MarkerRemoval: Paint type, Bilinear Y search, Control channel (the switch and the channel on one row), then the Pixel offset pair, as the guide's table lists them.
"""
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import old_document
from tests.test_2d_fidelity_f_text import H, W, draw, params


class TextGroupAnimationTests(unittest.TestCase):
    def test_default_is_empty_and_the_knob_sits_after_the_centre(self):
        self.assertEqual(SPECS["Text"]["params"]["group_animation"], "")
        names = [g.label for g in knobs.KNOB_LAYOUT["Text"]]
        self.assertEqual(names[names.index("Animation layers") - 1], "Centre offset")

    def test_it_changes_no_pixel(self):
        np.testing.assert_array_equal(draw(group_animation="1 0 0 1"), draw())

    def test_a_saved_node_loads_with_the_empty_entry(self):
        doc = old_document("Text", {"message": "Ab"}, drop=["group_animation"])
        self.assertEqual(doc["nodes"]["n"]["params"]["group_animation"], "")


class MarkerRemovalPanelTests(unittest.TestCase):
    def test_the_panel_follows_the_guides_order(self):
        names = [g.label for g in knobs.KNOB_LAYOUT["MarkerRemoval"]]
        self.assertEqual(names, ["Paint type", "Bilinear Y search", "Control channel", "X offset", "Y offset"])

    def test_the_knobs_still_act(self):
        image = np.zeros((5, 9, 4), np.float32)
        image[..., 0] = np.arange(9, dtype=np.float32)[None, :]
        image[2, 3, 3] = 1
        moved = Evaluator._kernel("MarkerRemoval", {**SPECS["MarkerRemoval"]["params"], "linearsearch": 20, "x_offset": 1}, [image])
        self.assertAlmostEqual(float(moved[2, 4, 0]), 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
