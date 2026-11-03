"""2D behaviour lane 2, step F, part 3: Write's missing knobs are all on the QuickTime and MXF tabs, which NodeBased has no writer for.

Step E named five to take first: the QuickTime `mov64` Color Primaries and Transfer Function (the first pair of twelve codec copies) and MXF Codec, Pattern
(OP pattern) and FPS. A knob can only act where the writer honours it, and Write's file types are the ones in `core.WRITE_FILE_TYPES`; none is a QuickTime
or MXF container, so each of the 29 counted knobs is paired as unsupported with that reason rather than as a param that would change nothing.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
from nodebased import core  # noqa: E402

FIRST_FIVE = ("mov64_color_primaries", "mov64_transfer_function", "mxf_video_codec_knob", "mxf_op_pattern_knob", "mxf_edit_rate_knob")


class WriteTabTests(unittest.TestCase):
    def setUp(self):
        self.nuke = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]["Write"]
        self.pairs = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]["Write"]["knobs"]

    def test_write_has_no_quicktime_or_mxf_file_type(self):
        for kind in core.WRITE_FILE_TYPES:
            self.assertNotIn(kind.lower(), ("mov", "mxf", "mp4", "mov64"))
        self.assertEqual(set(core.WRITE_FILE_TYPES), {"Auto", "exr", "png", "tiff", "jpeg", "dpx"})

    def test_the_five_knobs_are_unpaired_with_a_reason(self):
        by_nuke = {n: e for e in self.pairs for n in e.get("nuke", [])}
        for name in FIRST_FIVE:
            with self.subTest(knob=name):
                self.assertEqual(by_nuke[name]["kind"], "unsupported")
                self.assertEqual(by_nuke[name]["nodebased"], [])
                self.assertIn("no writer", by_nuke[name]["reason"])

    def test_every_counted_write_knob_is_paired_or_unpaired_with_a_reason(self):
        by_nuke = {n: e for e in self.pairs for n in e.get("nuke", [])}
        for knob in self.nuke["knobs"]:
            if knob["counted"]:
                with self.subTest(knob=knob["name"]):
                    self.assertIn(knob["name"], by_nuke)
                    if by_nuke[knob["name"]]["kind"] == "unsupported":
                        self.assertTrue(by_nuke[knob["name"]]["reason"])


if __name__ == "__main__":
    unittest.main()
