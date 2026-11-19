"""The Curve Editor and the Dope Sheet against Nuke 17.0's User Guide (lane 10, step T7, 2026-11-18): adding, moving and
deleting keys, the interpolation menu, tangents, framing, zoom and pan in the curve canvas; key markers, selection, delete, add
and moving keys in time in the Dope Sheet. Each probe in tools/nuke_panel_interactions.py drives the built window with mouse and
key events and checks the stored curve, the undo step and the selection; the gaps the guide states and NodeBased lacks are named."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import json
import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

PASSING = ("curve.add_key", "curve.move_one_axis", "curve.move_ctrl_free", "curve.numeric_xy", "curve.tangent_handles",
           "curve.interpolation_menu", "curve.interp_constant_linear", "curve.interp_smooth", "curve.interp_catmull_cubic",
           "curve.interp_horizontal", "curve.interp_break", "curve.zoom", "curve.pan", "curve.frame_all",
           "curve.frame_selected", "curve.select_several", "curve.select_all", "curve.several_curves", "curve.move_dialog",
           "curve.expression_field", "curve.frame_snap",
           "dope.key_markers", "dope.current_frame", "dope.select_key", "dope.name_selects_keys", "dope.delete_key",
           "dope.add_key", "dope.no_passing", "dope.project_range", "dope.marquee", "dope.move_field", "dope.copy_paste",
           "dope.clip_bars")
OPEN = ()


class CurveEditorAndDopeSheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["widget"] in ("curve_editor", "dope_sheet")]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            cls.results = interactions.probe_all()
        finally:
            interactions.reference = original

    def test_every_documented_entry_is_named_here(self):
        documented = {k for k, v in self.results.items() if v["status"] == "documented"}
        self.assertEqual(documented, set(PASSING) | set(OPEN))

    def test_the_curve_editor_and_dope_sheet_work_like_nuke(self):
        for behaviour in PASSING:
            with self.subTest(behaviour=behaviour):
                self.assertEqual(self.results[behaviour]["result"], "pass", self.results[behaviour]["detail"])

    def test_the_named_gaps_are_still_open(self):
        for behaviour in OPEN:
            with self.subTest(behaviour=behaviour):
                self.assertEqual(self.results[behaviour]["result"], "fail", self.results[behaviour]["detail"])

    def test_each_documented_entry_quotes_the_animation_page(self):
        ref = json.loads(interactions.REFERENCE.read_text())
        for entry in ref["behaviours"]:
            if entry["widget"] in ("curve_editor", "dope_sheet") and entry["status"] == "documented":
                self.assertEqual(entry["source"], "animation")
                self.assertTrue(entry["quote"])


class OldCurvesLoadUnchanged(unittest.TestCase):
    def test_a_legacy_lookup_curve_evaluates_as_before_after_opening_the_editor(self):
        from nodebased.colorcurves import decode, encode, evaluate
        from nodebased.curveeditor import CurveCanvas
        for interp in ("linear", "smooth"):
            value = encode(((0, 0), (0.3, 0.6), (1, 1)), interp)
            canvas = CurveCanvas(value)
            seen = []
            canvas.changed.connect(seen.append)
            canvas.frame_all()
            canvas.frame_selected()
            self.assertEqual(seen, [])           # framing never edits the curve
            original = decode(value)
            for x in (0.1, 0.3, 0.55, 0.9):
                self.assertEqual(evaluate(canvas.curve, x), evaluate(original, x))


if __name__ == "__main__":
    unittest.main()
