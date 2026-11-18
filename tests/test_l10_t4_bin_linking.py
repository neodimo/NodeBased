"""The Properties bin and knob linking against Nuke 17.0's User Guide (lane 10, step T4, 2026-11-17).

The probes live in tools/nuke_panel_interactions.py (widgets "properties_bin" and "linking"); this module runs only those
and asserts each one that now works like Nuke's by name, plus what the document stores for a Ctrl-drag link.
"""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

from nodebased.app import EXPRESSION_COLOR, Window  # noqa: E402

PASSING = ("bin.insert_opens", "bin.double_click_opens", "bin.ctrl_click_opens", "bin.return_opens", "bin.max_panels",
           "bin.remove_all", "bin.close", "bin.alt_close_all", "bin.ctrl_close_others", "bin.hide_pages",
           "link.add_expression", "link.reference_other", "link.ctrl_drag", "link.turns_blue", "link.edit_expression")


class BinAndLinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["widget"] in ("properties_bin", "linking")]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            cls.results = interactions.probe_all()
        finally:
            interactions.reference = original

    def test_bin_and_linking_behaviours_work_like_nuke(self):
        for behaviour in PASSING:
            with self.subTest(behaviour=behaviour):
                self.assertEqual(self.results[behaviour]["result"], "pass", self.results[behaviour]["detail"])

    def test_every_documented_entry_has_a_probe(self):
        for behaviour, row in self.results.items():
            if row["status"] == "documented":
                self.assertNotEqual(row["result"], "not probed", behaviour)


class LinkTextTests(unittest.TestCase):
    def test_a_plain_id_links_by_name_and_any_other_id_by_knob(self):
        self.assertEqual(Window.knob_reference("grade1", "gamma"), "grade1.gamma")
        self.assertEqual(Window.knob_reference("1a2b3c4d5e6f", "gamma"), 'knob("1a2b3c4d5e6f", "gamma")')
        self.assertEqual(Window.knob_reference("if", "gamma"), 'knob("if", "gamma")')

    def test_linked_knobs_are_a_blue_lighter_than_the_key_blue(self):
        from PySide6.QtGui import QColor
        from nodebased.timeline import KEY_COLOR
        hue = QColor(EXPRESSION_COLOR).hslHue()
        self.assertTrue(190 <= hue <= 250, hue)
        self.assertGreater(QColor(EXPRESSION_COLOR).lightness(), KEY_COLOR.lightness())


if __name__ == "__main__":
    unittest.main()
