"""2D behaviour, lane 9 step X3: LensDistortion's Reset button (the Solve section of the Nuke 17.0 Reference Guide's Analysis tab).

The guide lists `resetDistortion` ("Reset") beside Solve and says nothing more; ours puts every solved term back to no distortion (nodebased/lensdistortion.py
`reset_edits`). The detector and solver knobs (Detect, the feature and linking thresholds, the line drawing modes, Solve, Reject, Solve Error, the overlay) stay
unsupported: NodeBased has no grid detector or distortion solver.
"""
from __future__ import annotations

import sys
import types
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import lensdistortion
from nodebased.core import SPECS, Dispatcher
from nodebased.imaging import Evaluator

APP = QApplication.instance() or QApplication(sys.argv[:1])
from nodebased.app import Window  # noqa: E402


def graph(kind="Undistort", **params):
    d = Dispatcher()
    d.execute(dict(op="create", id="plate", type="Checker", params=dict(width=64, height=48, size=8)))
    d.execute(dict(op="create", id="lens", type=kind, params=params))
    d.execute(dict(op="connect", id="lens", input="image", source="plate"))
    return d


def pixels(d, view="lens"):
    return Evaluator().evaluate(dict(d.document, view=view))


class ResetTests(unittest.TestCase):
    SOLVED = dict(distortionModelPreset="Custom", distortionDenominator0=0.1, distortionDenominator1=-0.02, distortionNumerator0=0.01,
                  distortionNumerator1=0.005, centre_x=3.0, centre_y=-2.0, anamorphicTwist=1.5, anamorphicScale_x=1.2, anamorphicScale_y=0.9)

    def test_reset_writes_only_the_terms_that_differ_and_leaves_the_choices(self):
        d = graph(**self.SOLVED)
        params = d.document["nodes"]["lens"]["params"]
        edits = lensdistortion.reset_edits("Undistort", params)
        self.assertEqual(set(edits), set(self.SOLVED) - {"distortionModelPreset"})
        self.assertTrue(all(value == SPECS["Undistort"]["params"][name] for name, value in edits.items()))
        self.assertEqual(lensdistortion.reset_edits("Undistort", SPECS["Undistort"]["params"]), {})

    def test_the_button_puts_the_picture_back_to_no_distortion_in_one_undo_step(self):
        d = graph(**self.SOLVED)
        warped = pixels(d)
        untouched = pixels(graph(distortionModelPreset="Custom"))
        self.assertTrue(np.abs(warped - untouched).max() > 1e-3)                         # the solved terms bend the picture
        window = types.SimpleNamespace(dispatcher=d)
        window.command = lambda command: d.execute(command)
        undo_before = len(d.undo_stack)
        self.assertTrue(Window._lens_reset(window, "lens"))
        self.assertEqual(len(d.undo_stack), undo_before + 1)
        np.testing.assert_allclose(pixels(d), untouched, atol=1e-6)                      # no distortion: the same picture as a fresh Custom node
        self.assertEqual(d.document["nodes"]["lens"]["params"]["distortionModelPreset"], "Custom")
        self.assertFalse(Window._lens_reset(window, "lens"))                              # nothing left to reset
        d.execute({"op": "undo"})
        np.testing.assert_allclose(pixels(d), warped, atol=1e-6)

    def test_it_resets_the_older_brown_conrady_lens_too_and_a_redistort_node(self):
        d = graph("Redistort", distortionModelPreset="Brown-Conrady (NodeBased)", lens_k1=0.05, lens_p1=0.01)
        window = types.SimpleNamespace(dispatcher=d)
        window.command = lambda command: d.execute(command)
        self.assertTrue(Window._lens_reset(window, "lens"))
        params = d.document["nodes"]["lens"]["params"]
        self.assertEqual((params["lens_k1"], params["lens_p1"]), (0.0, 0.0))
        self.assertFalse(Window._lens_reset(window, "plate"))                             # not a lens node


if __name__ == "__main__":
    unittest.main()
