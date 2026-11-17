"""Panels that work like Nuke (lane 10, 2026-11-16): NodeBased's real knob widgets against Nuke's documented behaviours.

docs/parity/nuke_knob_interactions.json lists what Nuke 17.0's User Guide says each knob widget does; the probes in
tools/nuke_panel_interactions.py drive the real panels with Qt key and mouse events. Every behaviour NodeBased already
matches is asserted here, so a regression fails by name; the behaviours still open are listed in KNOWN_GAPS and reported
in docs/PARITY_2D_FIDELITY.md ("Panels that work like Nuke"). Closing a gap means moving it out of KNOWN_GAPS.
"""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import json
import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

from nodebased.app import evaluate_field_formula  # noqa: E402

KNOWN_GAPS = {
    "channel.layer_check_boxes": "the channels knob is one menu; a layer's channels have no check boxes to leave one out",
    "channel.single_channel": "with channels set to none there is no right-most channel field to pick one channel",
}


class ReferenceTests(unittest.TestCase):
    def test_every_entry_is_sourced_or_marked_undocumented(self):
        ref = json.loads(interactions.REFERENCE.read_text())
        widgets = set(ref["widget_order"])
        seen = set()
        for entry in ref["behaviours"]:
            self.assertNotIn(entry["id"], seen)
            seen.add(entry["id"])
            self.assertIn(entry["widget"], widgets)
            self.assertIn(entry["status"], ("documented", "undocumented"))
            if entry["status"] == "documented":
                self.assertIn(entry["source"], ref["sources"])
                self.assertTrue(entry["quote"])
                self.assertTrue(ref["sources"][entry["source"]]["url"].startswith("https://learn.foundry.com/nuke/17.0/"))

    def test_every_known_gap_names_a_documented_behaviour(self):
        documented = {e["id"] for e in interactions.reference()["behaviours"] if e["status"] == "documented"}
        self.assertLessEqual(set(KNOWN_GAPS), documented)


class ReportTests(unittest.TestCase):
    def test_the_fidelity_report_carries_the_table_beside_the_score(self):
        text = (ROOT / "docs" / "PARITY_2D_FIDELITY.md").read_text()
        self.assertIn("## Panels that work like Nuke", text)
        self.assertIn("| Widget | Passing / documented | Not yet like Nuke | Undocumented |", text)
        self.assertIn("not part of its formula", text)


class FormulaTests(unittest.TestCase):
    def test_arithmetic_is_worked_out_as_nuke_does(self):
        self.assertEqual(evaluate_field_formula("378/2"), 189.0)
        self.assertEqual(evaluate_field_formula("(1+2)*-3"), -9.0)
        self.assertEqual(evaluate_field_formula("2**3"), 8.0)

    def test_anything_but_arithmetic_is_refused(self):
        for text in ("", "frame", "__import__('os')", "1/0", "2**999", "abs(1)", "1e400*1e400"):
            self.assertIsNone(evaluate_field_formula(text), text)


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = interactions.probe_all()

    def test_each_documented_behaviour_matches_nuke_or_is_a_named_gap(self):
        for behaviour, row in self.results.items():
            if row["status"] != "documented":
                continue
            with self.subTest(behaviour=behaviour):
                if behaviour in KNOWN_GAPS:
                    self.assertNotEqual(row["result"], "pass", f"{behaviour} now passes: move it out of KNOWN_GAPS")
                else:
                    self.assertEqual(row["result"], "pass", row["detail"])

    def test_the_table_counts_every_documented_behaviour_once(self):
        rows = interactions.table(self.results)
        documented = sum(1 for r in self.results.values() if r["status"] == "documented")
        self.assertEqual(sum(r[2] for r in rows), documented)
        self.assertEqual(sum(r[1] for r in rows) + sum(len(r[3]) for r in rows), documented)


if __name__ == "__main__":
    unittest.main()
