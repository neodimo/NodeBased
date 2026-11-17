"""The Nuke parity review book (tools/parity_review_book.py) and the Nuke script export (tools/nuke_script_export.py)."""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import parity_review_book as book  # noqa: E402
import nuke_script_export as nk  # noqa: E402
from nuke_fidelity_cases import CASES, make_fixtures  # noqa: E402


class ReviewBookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="nb-review-book-test-")
        cls.out = Path(cls._tmp.name)
        cls.result = book.build(cls.out, book.TEN)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_ten_node_has_a_page_linked_from_the_index(self):
        self.assertEqual(set(self.result["pages"]), set(book.TEN))
        index = (self.out / "index.html").read_text(encoding="utf-8")
        for name, info in self.result["pages"].items():
            self.assertIn(f"href='{info['page']}'", index, name)
            self.assertTrue((self.out / info["page"]).exists(), name)

    def test_every_page_links_back_and_to_the_guide_and_its_images_exist(self):
        for name, info in self.result["pages"].items():
            text = (self.out / info["page"]).read_text(encoding="utf-8")
            self.assertIn("href='index.html'", text, name)
            self.assertIn("learn.foundry.com/nuke/17.0", text, name)
            for src in re.findall(r"src='([^']+)'", text):
                self.assertTrue((self.out / src).is_file(), f"{name}: {src}")

    def test_every_panel_png_is_non_empty(self):
        for name, info in self.result["pages"].items():
            self.assertTrue(info["panels"], name)
            for path in info["panels"]:
                self.assertGreater(Path(path).stat().st_size, 1000, f"{name}: {path}")

    def test_pages_show_both_halves(self):
        text = (self.out / self.result["pages"]["Grade"]["page"]).read_text(encoding="utf-8")
        self.assertIn("1. Does the panel look like Nuke's?", text)
        self.assertIn("2. Does it compute what Nuke computes?", text)
        self.assertIn("Knobs that act", text)
        self.assertIn("review plate", text)
        self.assertIn("<code>multiply</code>", text)

    def test_alignment_marks_a_default_the_guide_disagrees_with(self):
        rows = book.alignment({"knobs": [{"name": "size", "label": "size", "tab": "Blur", "primary": True, "role": "knob",
                                          "counted": True, "type": "number", "default": 123.0}]},
                              {"knobs": [{"nodebased": ["radius"], "nuke": ["size"], "kind": "renamed"}]}, "Blur")
        self.assertIn("default", rows[0]["marks"])
        self.assertEqual(rows[0]["nodebased"], "radius")


    def test_pages_carry_a_worklist_badge_and_the_index_counts_lanes(self):
        index = (self.out / "index.html").read_text(encoding="utf-8")
        self.assertIn("<h2>Worklist</h2>", index)
        for name, info in self.result["pages"].items():
            text = (self.out / info["page"]).read_text(encoding="utf-8")
            badge = "clean" if info["open"] == 0 else f"{info['open']} open"
            self.assertIn(badge, text.split("</h1>")[0], name)


class WorklistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = book.worklist()

    def test_items_go_to_the_lane_that_owns_the_group_and_are_sorted_by_weight(self):
        self.assertTrue(self.items)
        for i in self.items:
            if i["kind"] != "widget behaviour":
                self.assertEqual(i["lane"], book.LANE_OF_GROUP.get(i["group"], 11), i)
        open_w = [i["weight"] for i in self.items if i["status"] == "open"]
        self.assertEqual(open_w, sorted(open_w, reverse=True))

    def test_a_skipped_default_is_accepted_with_its_reason(self):
        acc = [i for i in self.items if i["status"] == "accepted"]
        self.assertTrue(acc)
        self.assertTrue(all(i["reason"] for i in acc))
        multiply = [i for i in acc if i["node"] == "Multiply" and i["knob"] == "value"]
        for i in multiply:
            self.assertIn("Nuke's own default is 1", i["reason"])

    def test_an_open_default_shows_both_values_and_the_pairing_note(self):
        blur = [i for i in self.items if i["node"] == "Blur" and i["knob"] == "size" and i["kind"] == "default"]
        if blur:  # lane 9 may have closed it
            self.assertEqual(blur[0]["lane"], 9)
            self.assertIn("radius", blur[0]["reason"])

    def test_markdown_and_json_agree_on_the_counts(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            md, js = Path(tmp) / "W.md", Path(tmp) / "w.json"
            book.write_worklist(self.items, md, js)
            data = json.loads(js.read_text(encoding="utf-8"))
            self.assertEqual(len(data["items"]), len(self.items))
            text = md.read_text(encoding="utf-8")
            for lane, c in book.lane_counts(self.items).items():
                self.assertEqual(data["counts"][str(lane)], c)
                self.assertRegex(text, rf"\| {lane} \| [^|]+ \| {c['open']} \| {c['accepted']} \|")

    def test_failing_behaviours_go_to_lane_10(self):
        fake = {"nodes": {"Blur": [{"knob": "size", "interaction": "drag scrubs", "status": "fail"},
                                    {"knob": "filter", "interaction": "wheel", "status": "pass"}, {"knob": "x"}]}}
        self.assertEqual([b["status"] for b in book.node_behaviours("Blur", fake)], ["failing", "passing", "undocumented"])
        import json
        pairs = json.loads(book.MAP_JSON.read_text(encoding="utf-8"))["pairs"]
        nodes = json.loads(book.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]
        items = book.node_items("Blur", {"group": "Filter"}, nodes["Blur"], pairs["Blur"], 1.0, fake)
        beh = [i for i in items if i["kind"] == "widget behaviour"]
        self.assertEqual([(i["lane"], i["knob"]) for i in beh], [(10, "size")])


class NukeRenderColumnTests(unittest.TestCase):
    def test_a_nuke_render_adds_the_column_and_counts_as_verified(self):
        from nodebased.media import write_exr
        from nuke_fidelity_cases import run_nodebased
        case = next(c for c in CASES if c.id == "grade_affine")
        with tempfile.TemporaryDirectory() as tmp:
            renders = Path(tmp) / "renders"
            renders.mkdir()
            # A stand-in for Nuke's render: NodeBased's own output, so the comparison passes.
            write_exr(renders / "grade_affine.exr", run_nodebased(case, make_fixtures()).astype("float32"), bits="float", compression="none")
            saved = book.RENDERS_DIR
            book.RENDERS_DIR = renders
            try:
                result = book.build(Path(tmp) / "out", ["Grade"])
            finally:
                book.RENDERS_DIR = saved
            page = (Path(tmp) / "out" / result["pages"]["Grade"]["page"]).read_text(encoding="utf-8")
            self.assertIn("Nuke rendered", page)
            self.assertIn("|NodeBased - Nuke| x10", page)
            self.assertTrue(result["pages"]["Grade"]["nuke_verified"])
            self.assertIn("verified against Nuke's own pixels: 1", (Path(tmp) / "out" / "index.html").read_text(encoding="utf-8"))


class NukeScriptExportTests(unittest.TestCase):
    def test_every_exported_script_names_the_nuke_knobs_from_the_pairing(self):
        fixtures = make_fixtures()
        checked = 0
        for case in CASES:
            if nk.input_plates(case, fixtures)[0] is None or not case.nb_params:
                continue
            text = nk.nk_script(case)
            body = text.split(f"\n{case.nuke_class} {{\n", 1)[1].split("\n name NodeUnderTest", 1)[0]
            set_knobs = {line.split()[0] for line in body.splitlines() if line.startswith(" ")}
            for knob, _ in case.nuke_knobs:
                self.assertIn(knob, set_knobs, case.id)
            entries = nk._pairs().get(case.nuke, {}).get("knobs", [])
            nuke_names = {n for e in entries for n in (e.get("nuke") or [])}
            for param in case.nb_params:
                e = next((x for x in entries if param in (x.get("nodebased") or [])), None)
                if param.startswith("_") or e is None or e["kind"] != "same" or len(e.get("nuke") or []) != 1 \
                        or len(e["nodebased"]) != 1 or not isinstance(case.nb_params[param], (bool, int, float, str)):
                    continue
                self.assertIn(e["nuke"][0], set_knobs, f"{case.id}: {param}")
                self.assertNotIn(param, set_knobs - nuke_names - {k for k, _ in case.nuke_knobs}, f"{case.id}: NodeBased name {param}")
                checked += 1
        self.assertGreater(checked, 100)

    def test_script_reads_its_plates_and_writes_a_float_exr_beside_itself(self):
        case = next(c for c in CASES if c.id == "merge_over")
        text = nk.nk_script(case)
        self.assertIn("version 17.0", text)
        self.assertEqual(text.count("Read {"), 2)
        self.assertIn("push $read_A\npush $read_B\nMerge2 {\n inputs 2", text)   # B on top: input 0
        self.assertIn("nuke_render.exr", text)
        self.assertIn("datatype {32 bit float}", text)
        self.assertEqual(text.count("{"), text.count("}"))

    def test_export_writes_plates_scripts_and_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            written, skipped = nk.export(tmp, {"Grade", "Merge"})
            self.assertGreater(written, 10)
            folder = Path(tmp) / "grade_affine"
            self.assertTrue((folder / "grade_affine.nk").is_file())
            self.assertGreater((folder / "A.exr").stat().st_size, 0)
            self.assertTrue((Path(tmp) / "MANIFEST.json").is_file())
            self.assertTrue((Path(tmp) / "render_all.py").is_file())
            (folder / nk.RENDER_NAME).write_bytes(b"x")
            renders = Path(tmp) / "renders"
            self.assertEqual(nk.collect(tmp, renders), 1)
            self.assertTrue((renders / "grade_affine.exr").is_file())

    def test_tcl_values(self):
        self.assertEqual(nk.tcl(True), "true")
        self.assertEqual(nk.tcl(1.5), "1.5")
        self.assertEqual(nk.tcl("over"), "over")
        self.assertEqual(nk.tcl("to box"), "{to box}")
        self.assertEqual(nk.tcl("{0.5 1 2}"), "{0.5 1 2}")


if __name__ == "__main__":
    unittest.main()
