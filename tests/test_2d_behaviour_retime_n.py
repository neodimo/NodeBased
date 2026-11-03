"""2D behaviour lane 2, step N (Time group): TimeWarp's input length, Kronos's and OFlow's remaining retime knobs and inputs.

The cases live in tools/nuke_fidelity_cases_time.py and the audit runs them. This module asserts their statuses and that each new
knob changes the picture the way the source states.
"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as fidelity  # noqa: E402
from nodebased.core import SPECS, upgrade_document  # noqa: E402
from nodebased.imaging import Evaluator, timewarp_box  # noqa: E402
from nodebased.knobs import knob_layout  # noqa: E402
from tests.test_2d_behaviour_retime_vectors import stamp_plate  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402


def clip_red(kind, params, frame, last=20):
    """The red value of `kind` at `frame` on the stamped plate cut to a clip of frames 1 to `last` (it holds the last frame past the end)."""
    g = Graph()
    stamp_plate(g)
    g.add("clip", "FrameRange", dict(first_frame=1, last_frame=last), image="plate")
    g.add("node", kind, params, image="clip")
    g.d.execute(dict(op="time", first=1, last=last, current=1))
    return float(Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame).pixels[0, 0, 0])


class TimeWarpLengthTests(unittest.TestCase):
    def test_the_length_cases_pass(self):
        cases = [c for c in fidelity.CASES if c.id.startswith("timewarp_length")]
        self.assertGreaterEqual(len(cases), 4)
        self.assertEqual([(c.id, audit.run_case(c)["status"]) for c in cases if audit.run_case(c)["status"] != "pass"], [])

    def test_box_weights_cover_the_window_and_sum_to_one(self):
        for centre, length in ((10.0, 3.0), (10.0, 2.0), (10.25, 1.0), (7.5, 4.0), (3.0, 0.5), (3.3, 6.5)):
            frames, weights = timewarp_box(centre, length)
            self.assertAlmostEqual(sum(weights), 1.0, places=9, msg=(centre, length))
            if float(centre).is_integer():
                self.assertAlmostEqual(sum(w * f for f, w in zip(frames, weights)), centre, places=9, msg=(centre, length))   # the window is centred
        self.assertEqual(timewarp_box(10.0, 3.0), ([9, 10, 11], [1 / 3, 1 / 3, 1 / 3]))
        self.assertEqual(timewarp_box(10.0, 0.0), ([10], [1.0]))

    def test_a_wider_box_averages_in_the_neighbours_a_held_frame_makes_visible(self):
        self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter="blend", length=1.0), 20), 20.0, places=5)
        self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter="blend", length=3.0), 20), (19 + 20 + 20) / 3, places=4)
        self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter="blend", length=5.0), 20), (18 + 19 + 20 + 20 + 20) / 5, places=4)

    def test_filters_none_and_nearest_ignore_the_length(self):
        for filter_name in ("none", "nearest"):
            self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter=filter_name, length=5.0), 20), 20.0, places=5, msg=filter_name)

    def test_an_older_timewarp_has_no_length_and_shows_the_picture_it_always_did(self):
        g = Graph()
        stamp_plate(g)
        g.add("node", "TimeWarp", dict(lookup_filter="blend"), image="plate")
        g.d.execute(dict(op="set_curve", id="node", param="lookup", curve=dict(interpolation="linear", keys=[dict(frame=1, value=1), dict(frame=5, value=10)])))
        document = copy.deepcopy(g.doc)
        del document["nodes"]["node"]["params"]["length"]
        upgrade_document(document)
        before = Evaluator().evaluate_raster(dict(document, view="node"), target="node", frame=3).pixels
        after = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=3).pixels
        self.assertTrue(np.array_equal(before, after))
        self.assertAlmostEqual(float(after[0, 0, 0]), 5.5, places=5)
        self.assertEqual(SPECS["TimeWarp"]["params"]["length"], 1.0)

    def test_the_panel_follows_the_guide_input_frame_input_length_filter(self):
        labels = [group.label for group in knob_layout("TimeWarp")]
        self.assertEqual(labels, ["Input frame", "Input length", "Filter"])


if __name__ == "__main__":
    unittest.main()
