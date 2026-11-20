"""Lane 11 step B5: the Nuke reference pack writes scripts Nuke 17 Non-commercial runs.

These run where Nuke is not installed: they read the scripts' text and the class/knob table recorded from Nuke 17.0v6
(tools/nuke17_knobs.json, written by tools/nuke17_knob_table.py).
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_nuke_reference_pack as pack  # noqa: E402
import nuke_fidelity_cases as cases  # noqa: E402
from nuke17_knob_table import load_table  # noqa: E402

# Classes Nuke 17's Python refuses ("nuke.nodes.X: Unknown command"), read from the first render pass on 11/20.
UNKNOWN_IN_NUKE17 = {"OFlow", "CrossTalk", "ScreenKeyer", "IBKColor", "DirBlur2", "Exposure", "Denoise", "Toe", "AdjustBBox",
                     "Stabilize", "Link", "Backdrop", "TVIScale", "UDIM_Import"}


def _case(case_id):
    return next(c for c in cases.CASES if c.id == case_id)


class Nuke17TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_table()

    def test_every_class_the_pack_creates_was_probed_in_nuke_17(self):
        for cls in pack.nuke_classes_used():
            if cls in ("Primatte", "Ultimatte", "GenerateLUT", "Link", "UDIM_Import", "Root"):
                continue
            with self.subTest(cls=cls):
                self.assertIn(cls, self.table)
                self.assertIn("knobs", self.table[cls], self.table[cls].get("error"))

    def test_renamed_classes_are_the_ones_nuke_17_creates(self):
        for guide, real in pack.NUKE17_CLASS.items():
            with self.subTest(guide=guide):
                self.assertIn("knobs", self.table[real])
        self.assertEqual(pack.NUKE17_CLASS["ScreenKeyer"], "Keylight")
        self.assertEqual(pack.NUKE17_CLASS["OFlow"], "OFlow2")

    def test_renamed_knobs_exist_on_the_nuke_17_node(self):
        for (guide_cls, _), knob in pack.NUKE17_KNOB.items():
            real = pack.NUKE17_CLASS.get(guide_cls, guide_cls)
            with self.subTest(cls=real, knob=knob):
                self.assertIn(knob, self.table[real]["knobs"])

    def test_knob_statements_follow_the_knob_kind(self):
        t = self.table
        self.assertRegex(pack.knob_statement("Merge2", "operation", "plus", t), r"setValue\(\d+\)  # plus$")
        self.assertEqual(pack.knob_statement("Blur", "size", "12", t), "node['size'].fromScript('12')")        # text to a number knob
        self.assertEqual(pack.knob_statement("Grade", "black_clamp", False, t), "node['black_clamp'].setValue(False)")
        self.assertIn("fromScript('{{0.5 1 2}", pack.knob_statement("ColorMatrix", "matrix", "{{0.5 1 2} {0 1 0} {0.25 0 1}}", t))
        self.assertRegex(pack.knob_statement("Saturation", "mode", "average", t), r"setValue\(2\)  # Average$")
        self.assertRegex(pack.knob_statement("Defocus", "method", "fullprecision", t), r"# full precision$")   # spaces and case ignored
        self.assertRegex(pack.knob_statement("Colorspace", "colorspace_out", "Gamma2.2", t), r"# 2\.20$")
        self.assertEqual(pack.knob_statement("Keyer", "range", "{0.05 0.4 0.75 1.5}", t),           # Keyer_Knob holds numbers
                         "node['range'].fromScript('{0.05 0.4 0.75 1.5}')")
        with self.assertRaisesRegex(ValueError, "has no knob"):
            pack.knob_statement("Bilateral", "filter", "median", t)
        with self.assertRaisesRegex(ValueError, "no menu entry 'lin'"):
            pack.knob_statement("OCIOColorSpace", "in_colorspace", "lin", t)

    def test_integer_knobs_get_ints(self):
        int_knobs = [(cls, name) for cls, v in self.table.items() for name, kind in v.get("knobs", {}).items() if kind == "Int_Knob"]
        cls, name = int_knobs[0]
        self.assertEqual(pack.knob_statement(cls, name, 5.0, self.table), f"node[{name!r}].setValue(5)")


class PackScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        pack.write_pack(cls.tmp.name)
        cls.root = Path(cls.tmp.name)
        cls.manifest = json.loads((cls.root / "MANIFEST.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def script(self, case_id):
        return (self.root / "scripts" / f"{case_id}.py").read_text()

    def test_no_script_creates_a_class_nuke_17_refuses(self):
        for path in (self.root / "scripts").glob("*.py"):
            created = set(re.findall(r"nuke\.nodes\.(\w+)\(", path.read_text()))
            with self.subTest(case=path.stem):
                self.assertFalse(created & UNKNOWN_IN_NUKE17)

    def test_every_write_keeps_alpha_and_crops_to_the_format(self):
        for path in (self.root / "scripts").glob("*.py"):
            text = path.read_text()
            with self.subTest(case=path.stem):
                self.assertIn("channels='rgba'", text)          # the Write's default, rgb, drops alpha
                self.assertIn("write.setInput(0, crop)", text)
                self.assertIn("crop['box'].setExpression('input.format.r', 2)", text)

    def test_cases_that_are_not_pixel_behaviours_are_left_out_and_listed(self):
        skipped = {s["id"] for s in self.manifest["skipped"]}
        self.assertIn("backdrop_not_a_pixel_behaviour", skipped)
        self.assertIn("link_not_in_nodebased", skipped)
        self.assertFalse(any((self.root / "scripts" / f"{s}.py").exists() for s in skipped))

    def test_unrenderable_cases_carry_their_reason(self):
        reasons = {s["id"]: s["reason"] for s in self.manifest["not_renderable"]}
        self.assertTrue(all(reasons.values()))
        self.assertIn("Non-commercial does not offer Primatte", " ".join(reasons.values()))
        self.assertIn("no menu entry 'lin'", reasons["ocio_colorspace_gain"])
        self.assertIn("SmartVector input over a frame range", reasons["vectordistort_follows_the_vectors"])
        self.assertFalse(any((self.root / "scripts" / f"{s}.py").exists() for s in reasons))

    def test_time_cases_build_the_frame_stamp_in_nuke_and_render_their_frame(self):
        text = self.script("timewarp_lookup_key_at_5")
        self.assertIn("nuke.nodes.Expression(expr0='clamp(frame, -90, 100)', expr1='0.25', expr2='0.5', expr3='1')", text)
        self.assertNotIn("stamp.exr", text)
        self.assertIn("node['lookup'].setAnimated()", text)
        self.assertIn("node['lookup'].setValueAt(10.0, 5.0)", text)
        self.assertIn("nuke.execute(write, 5, 5)", text)
        clip = self.script("appendclip_head_to_tail_at_1")
        self.assertIn("expr2='0.2'", clip)
        self.assertIn("expr2='0.3'", clip)
        self.assertIn("nuke.nodes.FrameRange(first_frame=11, last_frame=15)", clip)
        self.assertLessEqual(clip.count("nuke.nodes."), 10)

    def test_files_formats_and_mask_channels_are_written_the_way_nuke_reads_them(self):
        self.assertTrue((self.root / "fixtures" / "affine.cube").exists())     # the LUT the Vectorfield case made, beside the fixtures
        self.assertIn("node['vfield_file'].setValue(os.path.join(fixtures, 'affine.cube')",
                      self.script("vectorfield_trilinear_applies_the_lut_0.2_0.5_0.8"))
        self.assertIn("nuke.addFormat('10 7 1 nb_10x7')\nnode['format'].setValue('nb_10x7')", self.script("constant_format"))
        self.assertIn("node['maskChannelInput'].setValue('rgba.alpha')", self.script("copy_mask"))

    def test_renamed_classes_and_knobs_reach_the_scripts(self):
        self.assertIn("nuke.nodes.Keylight()", self.script(next(c.id for c in cases.CASES if c.nuke_class == "ScreenKeyer"
                                                                and (self.root / "scripts" / f"{c.id}.py").exists())))
        self.assertIn("node['frmaesfade']", self.script("timeecho_average_4_frames_at_10"))
        self.assertIn("node['mode'].setValue(2)  # Average", self.script("saturation_average_half"))


class DisplayWindowTests(unittest.TestCase):
    def test_a_smaller_data_window_lands_in_the_format(self):
        data = np.ones((2, 3, 4), np.float32)
        out = pack.place_in_display_window(data, (4, 5), (0, 0, 16, 12))
        self.assertEqual(out.shape, (12, 16, 4))
        self.assertEqual(float(out.sum()), 24.0)
        self.assertTrue((out[5:7, 4:7] == 1).all())

    def test_a_larger_data_window_is_cut_to_the_format(self):
        data = np.arange(20 * 30 * 4, dtype=np.float32).reshape(20, 30, 4)
        out = pack.place_in_display_window(data, (-2, -3), (0, 0, 16, 12))
        np.testing.assert_array_equal(out, data[3:15, 2:18])


if __name__ == "__main__":
    unittest.main()


class PixelMismatchTests(unittest.TestCase):
    """tools/nuke_pixel_mismatches.py on renders made here: one matching Nuke, one off by 0.5 at a known pixel."""

    def test_rows_sort_by_lane_and_mark_where_nuke_disagrees_with_the_guide(self):
        import nuke_pixel_mismatches as mism
        from nodebased.media import write_exr
        plus, over = _case("merge_plus"), _case("merge_over")
        with tempfile.TemporaryDirectory() as tmp:
            write_exr(Path(tmp) / "merge_plus.exr", cases.run_nodebased(plus).astype(np.float32), bits="float", compression="none")
            off = cases.run_nodebased(over).astype(np.float32)
            off[3, 5, 1] += 0.5
            write_exr(Path(tmp) / "merge_over.exr", off, bits="float", compression="none")
            data = mism.build(tmp)
        rows = {r["id"]: r for r in data["rows"]}
        self.assertEqual(set(rows), {"merge_plus", "merge_over"})
        self.assertEqual(rows["merge_plus"]["status"], "pass")
        bad = rows["merge_over"]
        self.assertEqual((bad["status"], bad["lane"], bad["channel"], bad["worst_pixel"]), ("fail", 11, "G", [3, 5]))
        self.assertAlmostEqual(bad["max_error"], 0.5, places=4)
        self.assertTrue(bad["nuke_disagrees_with_guide"])            # NodeBased does what the guide states, the 'render' does not
        self.assertFalse(rows["merge_plus"]["nuke_disagrees_with_guide"])
        self.assertEqual(data["summary"]["mismatch_by_lane"], {"11": 1})
        import parity_review_book as book
        text = "\n".join(book.nuke_pixel_section({11: "Image, Merge"}, data))
        self.assertIn("### Lane 11 (Image, Merge): 1 cases", text)
        self.assertIn("| Merge | `merge_over` | 0.5 | G | 3, 5 |", text)
        self.assertIn("**guide**", text)
