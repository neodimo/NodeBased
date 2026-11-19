"""2D behaviour, lane 9 step X5: LensDistortion's Analysis tab settings (Nuke 17.0 Reference Guide, LensDistortion: Grid Detect, Settings, Feature Linking, Solve, Overlay).

Twelve of the detector and solver knobs have a value and a default in the guide: Detection frames (Current Frame), Preview (off), Number of Features (5000), Patch Size (9),
Feature Separation (15), Detection Threshold (100), Angle Threshold (8), Distance Threshold (30), Peak Threshold (20), Solve Error (0), Threshold (10) and Show (All).
NodeBased has no grid detector or distortion solver, so they are held on the node at those defaults and change no pixel. The buttons (Detect, Solve, Reject and the
line drawing modes) have nothing to run and are not built. A node saved before the knobs loads with the defaults and draws what it drew.
"""
import copy
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import CHOICES, SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels

GUIDE = {"detectionFrameType": "Current Frame", "preview": 0, "numFeatures": 5000, "patchSize": 9, "featureSeparation": 15, "detectionThreshold": 100,
         "angleThreshold": 8.0, "distanceThreshold": 30.0, "peakThreshold": 20.0, "solveError": 0.0, "errorThreshold": 10.0, "overlayType": "All"}
CHANGED = {"detectionFrameType": "Every N Frames", "preview": 1, "numFeatures": 900, "patchSize": 21, "featureSeparation": 40, "detectionThreshold": 7.0,
           "angleThreshold": 30.0, "distanceThreshold": 90.0, "peakThreshold": 3.0, "solveError": 2.5, "errorThreshold": 1.0, "overlayType": "Links"}


def lens(kind="Undistort", **knob_values):
    g = Graph()
    g.add("plate", "Checker", dict(width=64, height=48, size=8))
    g.add("lens", kind, {"distortionModelPreset": "Custom", "distortionDenominator0": 0.05, **knob_values}, image="plate")
    return g


class LensAnalysisKnobTests(unittest.TestCase):
    def test_defaults_are_the_guides_on_both_kinds(self):
        for kind in ("Undistort", "Redistort"):
            params = SPECS[kind]["params"]
            for name, value in GUIDE.items():
                self.assertEqual(params[name], value, (kind, name))
        self.assertEqual(CHOICES["overlayType"], ["None", "Features", "Links", "All"])
        self.assertEqual(CHOICES["detectionFrameType"], ["Current Frame", "Every N Frames"])

    def test_they_change_no_pixel(self):
        for kind in ("Undistort", "Redistort"):
            base = evaluator_pixels(lens(kind).doc, "lens")
            changed = evaluator_pixels(lens(kind, **CHANGED).doc, "lens")
            np.testing.assert_array_equal(getattr(changed, "pixels", changed), getattr(base, "pixels", base))

    def test_the_panel_lists_them_in_the_guides_order(self):
        for kind in ("Undistort", "Redistort"):
            names = [g.label for g in knobs.KNOB_LAYOUT[kind]]
            wanted = ["Detection frames", "Preview", "Number of Features", "Patch Size", "Feature Separation", "Detection Threshold",
                      "Angle Threshold", "Distance Threshold", "Peak Threshold", "Solve Error", "Threshold", "Show"]
            at = names.index("Detection frames")
            self.assertEqual(names[at:at + len(wanted)], wanted)

    def test_a_saved_node_loads_with_the_defaults(self):
        g = lens()
        saved = copy.deepcopy(g.doc)
        for name in GUIDE:
            saved["nodes"]["lens"]["params"].pop(name, None)
        loaded = upgrade_document(saved)
        validate(loaded)
        for name, value in GUIDE.items():
            self.assertEqual(loaded["nodes"]["lens"]["params"][name], value, name)
        before = evaluator_pixels(g.doc, "lens")
        after = evaluator_pixels(loaded, "lens")
        np.testing.assert_array_equal(getattr(after, "pixels", after), getattr(before, "pixels", before))


if __name__ == "__main__":
    unittest.main()
