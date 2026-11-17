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
