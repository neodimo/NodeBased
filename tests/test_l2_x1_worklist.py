"""Lane 2 step X1, part 2: the review book's label and widget-type marks follow the audit's own rules.

The audit (tools/audit_nuke_fidelity.py, tools/nuke_fidelity_requirements.py) already reads a text knob whose default is on or off as a check box,
a number list held by a row of number fields as matched, and a guide label that several knobs share (or that sits on a knob holding several
controls) as the row's label, not one control's. The review book compared each knob alone, so it listed as open work what the audit scores as
done. These tests pin the shared rules, and that a real difference is still marked.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import parity_review_book as book  # noqa: E402


def marks(node, name, **overrides):
    """The marks of one knob of a real node's alignment rows."""
    nuke_doc, map_doc = book.load_json(book.NUKE_JSON), book.load_json(book.MAP_JSON)
    pair = map_doc["pairs"][node]
    rows = book.alignment(nuke_doc["nodes"][node], pair, pair["nodebased"], node)
    return next(r for r in rows if r["name"] == name)["marks"]


def open_rows(nodes, kinds):
    items = book.worklist(nodes=nodes, with_cases=False)
    return [(i["node"], i["kind"], i["knob"]) for i in items if i["status"] == "open" and i["kind"] in kinds]


class SharedRulesTests(unittest.TestCase):
    def test_a_text_knob_whose_default_is_on_or_off_is_a_check_box(self):
        for node, knob in (("TimeOffset", "enable_path_mask"), ("TimeOffset", "inject_mask"), ("TimeOffset", "reverse_input"),
                           ("FrameHold", "enable_path_mask"), ("FrameHold", "inject_mask")):
            with self.subTest(node=node, knob=knob):
                self.assertNotIn("widget", marks(node, knob))

    def test_a_row_of_number_fields_is_a_number_list(self):
        for knob in ("in", "out"):
            self.assertNotIn("widget", marks("Histogram", knob))

    def test_a_single_slider_is_still_not_a_number_list(self):
        """IBKColor's darks and lights hold one number where Nuke's hold red, green and blue: that is real work, and stays listed."""
        self.assertIn("widget", marks("IBKColor", "off"))
        self.assertIn("widget", marks("IBKColor", "mult"))

    def test_a_shared_guide_label_is_the_rows_not_one_knobs(self):
        for knob in ("to0", "to1", "to2", "to3"):
            self.assertNotIn("label", marks("Copy", knob))
        for knob in ("temp_name0", "temp_name1", "temp_name2", "temp_name3"):
            self.assertNotIn("label", marks("Expression", knob))

    def test_a_knob_that_holds_two_guide_knobs_has_no_single_label(self):
        self.assertNotIn("label", marks("HSVTool", "Conversion"))

    def test_a_real_label_difference_is_still_marked(self):
        """The rule must not go blind: a knob with its own guide label and a different one on our panel is listed."""
        nuke_doc, map_doc = book.load_json(book.NUKE_JSON), book.load_json(book.MAP_JSON)
        node = dict(nuke_doc["nodes"]["Blur"])
        pair = map_doc["pairs"]["Blur"]
        rows = book.alignment(node, pair, pair["nodebased"], "Blur")
        laid_out = next(r for r in rows if r["nb_label"] and r["status"] in ("same", "renamed") and "label" not in r["marks"]
                        and book._norm_label(r["nb_label"]) != book._norm_label(r["name"]) and len(r["nodebased"].split(", ")) == 1)
        knobs = [dict(k) for k in node["knobs"]]
        next(k for k in knobs if k["name"] == laid_out["name"])["label"] = "A label nobody gave this knob"
        node["knobs"] = knobs
        rows = book.alignment(node, pair, pair["nodebased"], "Blur")
        self.assertIn("label", next(r for r in rows if r["name"] == laid_out["name"])["marks"])


class WorklistTests(unittest.TestCase):
    def test_the_rows_that_were_the_audits_done_work_are_gone_from_lane_2(self):
        rows = open_rows(["TimeOffset", "FrameHold", "Histogram", "Copy", "Expression", "HSVTool"], ("label", "widget type"))
        self.assertEqual(rows, [])

    def test_the_real_ones_remain(self):
        rows = open_rows(["IBKColor", "MatchGrade"], ("label", "widget type"))
        self.assertIn(("IBKColor", "widget type", "mult"), rows)
        self.assertIn(("IBKColor", "widget type", "off"), rows)
        self.assertIn(("MatchGrade", "widget type", "sourceRefFrames"), rows)

    def test_color_transfer_is_paired_so_it_is_not_a_missing_node(self):
        items = book.worklist(nodes=["ColorTransfer"], with_cases=True)
        self.assertEqual([i for i in items if i["status"] == "open"], [])


if __name__ == "__main__":
    unittest.main()
