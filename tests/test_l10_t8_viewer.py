"""The Viewer works like Nuke's (lane 10, step T8, 2026-11-18): NodeBased's real Viewer, driven with Qt key and mouse
events, against the Nuke 17.0 User Guide's Viewers and Using the Viewer Controls pages (the "viewer" entries of
docs/parity/nuke_knob_interactions.json). Every documented behaviour passes or is a named gap in OPEN; closing one means
moving it out of OPEN (tests/test_panel_interactions.py lists the same ids in KNOWN_GAPS)."""
import tests.isolation  # noqa: F401

import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

OPEN = {
    "viewer.alpha_overlay": "M is not a Viewer key in NodeBased yet",
    "viewer.recenter_middle_click": "a middle click without a drag does not re-centre",
    "viewer.zoom_drag": "middle plus left drag does not zoom",
    "viewer.prevent_auto_zoom": "no Alt+Z lock on the zoom",
    "viewer.toolbars": "{ and } do not hide the Viewer's toolbars",
    "viewer.gain_gamma_toggle": "not probed in this step",
    "viewer.refresh": "not probed in this step",
    "viewer.goto_frame": "not probed in this step",
    "viewer.frame_field_math": "not probed in this step",
}


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = [e for e in interactions.reference()["behaviours"] if e["widget"] == "viewer"]
        bench = interactions.Bench()
        cls.results = {}
        try:
            for entry in cls.entries:
                if entry["status"] == "documented" and entry["id"] in interactions.PROBES:
                    cls.results[entry["id"]] = interactions.PROBES[entry["id"]](bench)
        finally:
            bench.close()

    def test_every_documented_viewer_behaviour_passes_or_is_open(self):
        for entry in self.entries:
            if entry["status"] != "documented":
                continue
            with self.subTest(behaviour=entry["id"]):
                passed, detail = self.results.get(entry["id"], (False, "not probed"))
                if entry["id"] in OPEN:
                    self.assertFalse(passed, f"{entry['id']} now passes: move it out of OPEN")
                else:
                    self.assertTrue(passed, detail)

    def test_the_reference_cites_both_viewer_pages(self):
        sources = {e["source"] for e in self.entries if e["status"] == "documented"}
        self.assertEqual(sources, {"viewers", "viewer_controls"})


if __name__ == "__main__":
    unittest.main()
