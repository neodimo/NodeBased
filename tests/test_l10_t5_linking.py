"""Knob linking finished against Nuke 17.0's User Guide (lane 10, step T5, 2026-11-17): Ctrl+drag between animation buttons
links a whole knob at once, and the Animation menu's Link to > Tracker links a knob to a Tracker's track."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

PASSING = ("link.add_expression", "link.reference_other", "link.ctrl_drag", "link.turns_blue", "link.graph_arrow",
           "link.edit_expression", "link.ctrl_drag_animation_button", "link.tracker_link_to")


class LinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["widget"] == "linking"]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            cls.results = interactions.probe_all()
        finally:
            interactions.reference = original

    def test_linking_behaviours_work_like_nuke(self):
        for behaviour in PASSING:
            with self.subTest(behaviour=behaviour):
                self.assertEqual(self.results[behaviour]["result"], "pass", self.results[behaviour]["detail"])

    def test_link_menu_is_still_open(self):
        self.assertNotEqual(self.results["link.link_menu"]["result"], "pass")


if __name__ == "__main__":
    unittest.main()
