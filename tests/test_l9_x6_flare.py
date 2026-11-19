"""2D behaviour, lane 9 step X6: Flare's corners angle `rel` (Nuke 17.0 Reference Guide, Flare, Shape tab: `corners_angle_add`, disabled by default).

The guide: "rel: When enabled, angle is relative to the light angle (the angle between position and the center point of the image)." The knob is wired:
the light angle is added to the corners angle. The audit listed it as changing no pixels because its fixture put Position at the image centre, where the light
angle is zero, and a circle has no corners to turn; the probe now starts the flare off-centre (tools/nuke_fidelity_requirements.py `PROBE_BASE`) and moves the
flattening first, as it does for every knob that needs another knob moved. The preset buttons and Speedup stay unsupported with their recorded reasons.
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pairs():
    with open(os.path.join(HERE, "docs", "parity", "nodebased_to_nuke.json")) as handle:
        return json.load(handle)
sys.path.insert(0, os.path.join(HERE, "tools"))

import nuke_fidelity_cases as ruler  # noqa: E402
import nuke_fidelity_requirements as req  # noqa: E402
from tests.test_2d_behaviour_extra_draw import flare  # noqa: E402

CENTRE = 15.0                      # the helper's picture is 30 x 30
OFF = dict(position_x=24.0, position_y=6.0)       # nine pixels right of the centre and nine up: a light angle of 45 degrees


class RelTests(unittest.TestCase):
    def test_at_the_centre_the_light_angle_is_zero_so_the_knob_shows_nothing(self):
        for flattening in (0.0, 1.0):
            with self.subTest(flattening=flattening):
                a = flare(position_x=CENTRE, position_y=CENTRE, flattening=flattening, corners_angle_add=0)
                b = flare(position_x=CENTRE, position_y=CENTRE, flattening=flattening, corners_angle_add=1)
                np.testing.assert_array_equal(a, b)

    def test_off_centre_a_polygon_turns_by_the_light_angle(self):
        plain = flare(**OFF, flattening=1.0, corners_angle_add=0)
        relative = flare(**OFF, flattening=1.0, corners_angle_add=1)
        self.assertGreater(float(np.abs(plain - relative).max()), 1e-4)

    def test_rel_adds_exactly_the_angle_between_position_and_the_centre(self):
        angle = math.degrees(math.atan2(-(OFF["position_y"] - CENTRE), OFF["position_x"] - CENTRE))
        self.assertAlmostEqual(angle, 45.0)
        for typed in (0.0, 20.0, -35.0):
            with self.subTest(corners_angle=typed):
                relative = flare(**OFF, flattening=1.0, corners_angle=typed, corners_angle_add=1)
                turned = flare(**OFF, flattening=1.0, corners_angle=typed + angle, corners_angle_add=0)
                np.testing.assert_allclose(relative, turned, atol=1e-6)

    def test_a_circle_has_no_corner_for_it_to_turn(self):
        a = flare(**OFF, flattening=0.0, corners_angle_add=0)
        b = flare(**OFF, flattening=0.0, corners_angle_add=1)
        np.testing.assert_array_equal(a, b)

    def test_the_audit_probe_now_sees_it_act(self):
        acts = req.probe_acts("Flare", ["corners_angle_add", "flattening", "flare_chroma_angle_add", "flare_chroma_shift", "vis_in"], ruler.make_fixtures())
        self.assertTrue(acts["corners_angle_add"])
        self.assertTrue(acts["flare_chroma_angle_add"])
        self.assertTrue(acts["vis_in"])


class StillUnsupportedTests(unittest.TestCase):
    def test_the_presets_and_speedup_keep_their_recorded_reasons(self):
        pairs = _load_pairs()["pairs"]["Flare"]["knobs"]
        by_name = {name: e for e in pairs for name in e.get("nuke", [])}
        for name in ("speedup", "MultiPenta", "LgRainbow", "Bright", "ManyBright", "Symmetry4"):
            with self.subTest(name=name):
                self.assertEqual(by_name[name]["kind"], "unsupported")
                self.assertGreater(len(by_name[name]["reason"]), 30)
        for name in ("MultiPenta", "LgRainbow", "Bright", "ManyBright", "Symmetry4"):
            self.assertIn(name, req.NOT_PIXEL["Flare"])


if __name__ == "__main__":
    unittest.main()
