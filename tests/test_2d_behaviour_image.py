"""2D behaviour lane 2, step E (Image group): Read, Write and Constant read like Nuke's panels, and Constant fills the format with its colour.

The main tab keeps the order and the wording of the Nuke 17.0 Reference Guide's table for the node (docs/parity/nuke_nodes.json); knobs only
NodeBased has follow Nuke's. Viewer's controls are viewer state, not node params (the mapping pairs none of them), so there is no panel order
to compare for it. Constant's pixels are in tests/test_2d_behaviour_merge.py.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402
from nodebased import core  # noqa: E402
from nodebased.knobs import knob_layout  # noqa: E402

MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]
NUKE = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]

# The main tab, in the guide's order, as the Properties panel shows it: (NodeBased param, label).
READ = [("path", "File"), ("localizationPolicy", "Localization policy"), ("read_format", "Format"), ("proxy", "Proxy"), ("proxy_format", "Proxy format"),
        ("first", "First frame"), ("before", "Before first"), ("last", "Last frame"), ("after", "After last"), ("frame_mode", "Frame mode"),
        ("frame_expr", "Frame"), ("origfirst", "Original first"), ("origlast", "Original last"), ("missing", "Missing frames"), ("reload", "Reload"),
        ("colorspace", "Input transform"), ("premultiplied", "Premultiplied"), ("raw", "Raw data"), ("auto_alpha", "Auto alpha")]
WRITE = [("channels", "Channels"), ("path", "File"), ("proxy", "Proxy"), ("frame_mode", "Frame mode"), ("frame_expr", "Frame"),
         ("colorspace", "Output transform"), ("premultiplied", "Premultiplied"), ("raw", "Raw data"), ("views", "Views"), ("file_type", "File type")]
WRITE_AFTER_FORMAT = [("create_directories", "Create directories"), ("render_order", "Render order"), ("first", "First frame"), ("last", "Last frame"),
                      ("use_limit", "Limit to range"), ("reading", "Read file"), ("on_error", "Missing frames"), ("reload", "Reload"),
                      ("read_all_lines", "Read all lines")]
CONSTANT = [("channels", "Channels"), ("red", "Color"), ("width", "Width"), ("height", "Height"), ("first", "First frame"), ("last", "Last frame")]


def layout(kind):
    return [(g.params[0], g.label) for g in knob_layout(kind)]


def labelled(kind, wanted):
    """The kind's panel rows for the params in `wanted`, with each row's label as shown."""
    from nodebased.knobs import knob_label
    return [(p, knob_label(kind, p)) if False else (p, label) for p, label in layout(kind) if p in {w for w, _ in wanted}]


def requirement_layout(nuke_key, kind):
    """What the audit measures for the node: (knobs in order, paired knobs, label matches, labels compared, mismatches)."""
    entries = MAPPING[nuke_key]["knobs"]
    knobs = [k for k in NUKE[nuke_key]["knobs"] if k["counted"] and not k["role"].startswith("chrome:")]
    frac, detail = requirements.layout_component(kind, entries, knobs)
    return frac, detail


class MainTabOrderTests(unittest.TestCase):
    def test_read_main_tab_follows_the_guide(self):
        rows = [row for row in layout("Read") if row[0] in {p for p, _ in READ}]
        self.assertEqual(rows, READ)

    def test_read_keeps_its_own_knobs_after_nukes(self):
        params = [p for p, _ in layout("Read")]
        last_nuke = max(params.index(p) for p, _ in READ)
        for own in ("alpha_mode", "layer", "subimage", "frame_offset"):
            self.assertGreater(params.index(own), last_nuke, own)

    def test_write_main_tab_follows_the_guide(self):
        params = layout("Write")
        names = [p for p, _ in params]
        head = params[:len(WRITE)]
        self.assertEqual(head, WRITE)
        tail = [row for row in params if row[0] in {p for p, _ in WRITE_AFTER_FORMAT}]
        self.assertEqual(tail, WRITE_AFTER_FORMAT)
        # The chosen file type's own knobs sit between the file type and the create-directories row, where Nuke shows them.
        file_type, create = names.index("file_type"), names.index("create_directories")
        for name in ("bit_depth", "exr_compression", "tiff_datatype", "jpeg_quality", "dpx_datatype"):
            self.assertTrue(file_type < names.index(name) < create, name)

    def test_constant_main_tab_follows_the_guide(self):
        self.assertEqual(layout("Constant"), CONSTANT)

    def test_viewer_has_no_paired_param_to_order(self):
        self.assertEqual([p for e in MAPPING["Viewer"]["knobs"] if e["kind"] in requirements.PAIRED for p in e["nodebased"]], [])


class AuditMeasuresTheLayoutTests(unittest.TestCase):
    def test_read_write_and_constant_have_no_layout_difference(self):
        for key, kind in (("Read", "Read"), ("Write", "Write"), ("Constant", "Constant")):
            with self.subTest(node=key):
                frac, detail = requirement_layout(key, kind)
                kept, total = detail["layout_order"]
                self.assertEqual(kept, total, detail)
                self.assertEqual(detail["label_mismatches"], [], detail)
                self.assertEqual(frac, 1.0)

    def test_every_param_is_still_laid_out_once(self):
        for kind in ("Read", "Write", "Constant"):
            flat = [p for g in knob_layout(kind) for p in g.params]
            self.assertEqual(len(flat), len(set(flat)), kind)
            self.assertEqual(set(flat), set(core.SPECS[kind]["params"]), kind)


class ConstantRangeTests(unittest.TestCase):
    def test_frame_range_is_listed_as_acting_across_frames(self):
        for knob in ("first", "last"):
            self.assertIn("across frames", requirements.not_pixel_reason("Constant", knob))


if __name__ == "__main__":
    unittest.main()
