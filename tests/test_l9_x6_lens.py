"""2D behaviour, lane 9 step X6: LensDistortion's nine remaining secondary knobs (Nuke 17.0 Reference Guide, LensDistortion, Analysis tab).

Detect, Reject, Solve and the six line drawing tools are buttons of the grid-calibration workflow: Detect finds a grid's features and links them into lines,
Solve estimates the distortion model from those links, Reject drops links whose residual is over the threshold, and the drawing tools select, add and remove
features and lines in the viewer. NodeBased's Undistort takes typed coefficients (and holds the detection settings of step X5); it has no detector, no solver
and no interactive viewer tool, so every one of the nine stays paired as unsupported with the guide's sentence and what it would need. None is a param.
"""
import json
import os
import unittest

from nodebased.core import SPECS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pairs():
    with open(os.path.join(HERE, "docs", "parity", "nodebased_to_nuke.json")) as handle:
        return json.load(handle)
NINE = ("detect", "deleteOutliers", "solveDistortion", "lineDrawingModeSelectAll", "lineDrawingModeSelectFeatures", "lineDrawingModeSelectLines",
        "lineDrawingModeAddFeatures", "lineDrawingModeRemoveFeatures", "lineDrawingModeAddLines")
TOOLS = tuple(name for name in NINE if name.startswith("lineDrawingMode"))


def entries():
    pairs = _load_pairs()["pairs"]["LensDistortion"]["knobs"]
    return {name: e for e in pairs for name in e.get("nuke", [])}


class RemainingKnobTests(unittest.TestCase):
    def test_all_nine_are_paired_as_unsupported_with_the_guides_sentence(self):
        found = entries()
        for name in NINE:
            with self.subTest(name=name):
                self.assertEqual(found[name]["kind"], "unsupported")
                self.assertEqual(found[name]["nodebased"], [])
                self.assertIn("the guide", found[name]["reason"])
                self.assertIn("\"Click to", found[name]["reason"])      # the sentence is quoted

    def test_the_six_drawing_tools_say_they_need_a_viewer_tool(self):
        found = entries()
        for name in TOOLS:
            with self.subTest(name=name):
                self.assertIn("interactive viewer tool", found[name]["reason"])

    def test_the_three_calibration_buttons_say_what_they_would_need(self):
        found = entries()
        self.assertIn("grid detector", found["detect"]["reason"])
        self.assertIn("residual error", found["deleteOutliers"]["reason"])
        self.assertIn("solver", found["solveDistortion"]["reason"])

    def test_undistort_has_no_param_for_them(self):
        for kind in ("Undistort", "Redistort"):
            params = SPECS[kind]["params"]
            for name in NINE:
                self.assertNotIn(name, params, (kind, name))


if __name__ == "__main__":
    unittest.main()
