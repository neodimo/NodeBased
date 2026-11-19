"""On-screen controls that work like Nuke (lane 10, step T3, 2026-11-17): NodeBased's Viewer handles against Nuke's
documented on-screen controls in docs/parity/nuke_viewer_controls.json. Every documented control passes, or is listed in
KNOWN_GAPS with its reason; closing a gap means moving it out of KNOWN_GAPS. The audit prints the same results as the table
"On-screen controls that work like Nuke" in docs/PARITY_2D_FIDELITY.md."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import nuke_viewer_controls as controls  # noqa: E402

KNOWN_GAPS = {
}


class ReferenceTests(unittest.TestCase):
    def test_every_entry_is_sourced(self):
        data = controls.reference()
        for entry in data["controls"]:
            with self.subTest(entry=entry["id"]):
                self.assertIn(entry["source"], data["sources"])
                self.assertIn(entry["status"], ("documented", "undocumented"))
                self.assertIn(entry["node"], data["node_order"])
                if entry["status"] == "documented":
                    self.assertTrue(entry.get("quote"))


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = controls.probe_all()

    def test_documented_controls_pass_or_are_known_gaps(self):
        for control, result in self.results.items():
            if result["result"] == "undocumented":
                continue
            with self.subTest(control=control, detail=result["detail"]):
                if control in KNOWN_GAPS:
                    self.assertNotEqual(result["result"], "pass", "a known gap now passes: move it out of KNOWN_GAPS")
                else:
                    self.assertEqual(result["result"], "pass")

    def test_transform_jack_translates_rotates_scales_and_moves_its_centre(self):
        for control in ("Transform.translate", "Transform.translate_shift", "Transform.center_ctrl",
                        "Transform.scale_uniform", "Transform.rotate"):
            self.assertEqual(self.results[control]["result"], "pass", self.results[control]["detail"])

    def test_jack_scales_one_axis_snaps_skews_and_shows_the_invert_handle(self):
        for control in ("Transform.scale_x", "Transform.scale_y", "Transform.rotate_snap", "Transform.skew",
                        "Transform.invert_handle"):
            self.assertEqual(self.results[control]["result"], "pass", self.results[control]["detail"])

    def test_crop_and_text_boxes(self):
        for control in ("Crop.box_edges", "Text.box_move_resize", "Text.center_ctrl"):
            self.assertEqual(self.results[control]["result"], "pass", self.results[control]["detail"])

    def test_colortransfer_roi_box(self):
        for control in ("ColorTransfer.roi_box", "ColorTransfer.roi_on_input_a"):
            self.assertEqual(self.results[control]["result"], "pass", self.results[control]["detail"])

    def test_tracker_anchor_and_gridwarp_overlays(self):
        for control in ("Tracker.anchor", "GridWarp.grid_handles", "GridWarp.grid_transform"):
            self.assertEqual(self.results[control]["result"], "pass", self.results[control]["detail"])

    def test_report_has_the_table(self):
        text = (ROOT / "docs" / "PARITY_2D_FIDELITY.md").read_text()
        self.assertIn("## On-screen controls that work like Nuke", text)


if __name__ == "__main__":
    unittest.main()
