"""Lane 2 step X4: Reformat's pixel aspect (`box_pixel_aspect` in the Nuke 17.0 Reference Guide) is a knob of the to-box block.

Reference: Nuke 17.0 Reference Guide, Transform nodes > Reformat (learn.foundry.com/nuke/17.0/content/reference_guide/transform_nodes/reformat.html):
box_width/box_height ("width/height"), box_fixed ("force this shape"), box_pixel_aspect ("pixel aspect", default 1), then scale and resize type.
The guide says only that it "sets the pixel aspect ratio for the output image". NodeBased works in square pixels and carries the value with the format,
so it changes no pixel (the fidelity case for it is unscored for that reason); here it is the node's pixel_aspect, paired with the guide's knob, laid
out straight after Force this shape, kept by a named-format pick and by a saved document.
"""
import json
import os
import unittest

from nodebased import knobs
from nodebased.core import SPECS, Dispatcher
from tests.test_2d_behaviour_filter_f import layout_order


def mapping():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity", "nodebased_to_nuke.json")
    return json.load(open(path))["pairs"]["Reformat"]["knobs"]


class ReformatPixelAspect(unittest.TestCase):
    def test_the_knob_is_paired_with_the_guides_name(self):
        entry = next(e for e in mapping() if e.get("nuke") == ["box_pixel_aspect"])
        self.assertEqual((entry["kind"], entry["nodebased"]), ("renamed", ["pixel_aspect"]))
        self.assertIn("changes no pixel", entry["reason"])
        self.assertEqual(SPECS["Reformat"]["params"]["pixel_aspect"], 1.0)

    def test_it_sits_in_the_to_box_block_after_force_this_shape(self):
        labels = [g.label for g in knobs.KNOB_LAYOUT["Reformat"] if not g.params[0].startswith("legacy_")]
        box = labels.index("Width/height")
        self.assertEqual(labels[box:box + 3], ["Width/height", "Force this shape", "Pixel aspect"])
        kept, total = layout_order("Reformat")[0]
        self.assertEqual(kept, total)                                   # the paired knobs, pixel aspect now among them, are all in the guide's order

    def test_a_to_box_reformat_keeps_its_pixel_aspect_and_a_named_format_sets_it(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="r", type="Reformat", params={"reformat_type": "to_box", "width": 40, "height": 40, "pixel_aspect": 2.0}))
        self.assertEqual(d.document["nodes"]["r"]["params"]["pixel_aspect"], 2.0)
        d.execute(dict(op="set", id="r", param="format", value="HD_1080"))
        self.assertEqual(d.document["nodes"]["r"]["params"]["pixel_aspect"], 1.0)      # a named format brings its own


if __name__ == "__main__":
    unittest.main()
