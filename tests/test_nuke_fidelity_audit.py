"""Lane L2, fidelity ruler: the committed report matches a fresh run, the pairing file is complete, and the reference loader works.

`tools/audit_nuke_fidelity.py` scores each 2D node against the Nuke 17.0 Reference Guide's tables, an explicit knob pairing and
formula tests; `docs/PARITY_2D_FIDELITY.md` is its output. This test fails when the report drifts from the code, the same way
`tests/test_2d_parity_step_w3.py` pins PARITY_2D.md."""
import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import make_nuke_reference_pack as pack  # noqa: E402
import nuke_fidelity_cases as cases  # noqa: E402
from nodebased.core import SPECS  # noqa: E402

NUKE = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))
MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))


class ReportDriftTests(unittest.TestCase):
    def test_committed_report_matches_a_fresh_run(self):
        fresh = audit.build_report()
        committed = audit.REPORT.read_text(encoding="utf-8")
        self.assertEqual(committed, fresh, "docs/PARITY_2D_FIDELITY.md drifted; run tools/audit_nuke_fidelity.py --write")

    def test_bundled_copy_matches(self):
        bundled = ROOT / "nodebased" / "data" / "docs" / "PARITY_2D_FIDELITY.md"
        self.assertTrue(bundled.exists(), "copy docs/PARITY_2D_FIDELITY.md into nodebased/data/docs")
        self.assertEqual(bundled.read_text(encoding="utf-8"), audit.REPORT.read_text(encoding="utf-8"))

    def test_report_names_its_formula_and_weights(self):
        text = audit.REPORT.read_text(encoding="utf-8")
        for name, weight in audit.WEIGHTS.items():
            self.assertIn(f"| {name} | {weight} |", text)
        self.assertIn("fidelity = 100 * sum(weight * fraction) / sum(weight)", text)
        self.assertIn("weighs `1 / r`", text)
        gaps = text.split("## The 30 largest gaps")[1].split("## Formula cases")[0]
        self.assertEqual([int(n) for n in re.findall(r"^\| (\d+) \| ", gaps, re.M)], list(range(1, 31)))

    def test_parity_note_points_at_the_report(self):
        head = (ROOT / "docs" / "PARITY_2D.md").read_text(encoding="utf-8").split("\n## ")[0]
        self.assertIn("PARITY_2D_FIDELITY.md", head)
        self.assertIn("does not mean the node behaves like the Nuke node", head)


class ReferenceDataTests(unittest.TestCase):
    def test_every_scored_row_node_has_guide_data_and_a_pairing(self):
        for group, rank, cell, status in audit.doc_rows():
            for name in audit.names_of(cell):
                if name == "LightMixer":
                    continue
                key = audit.pair_key(group, name)
                with self.subTest(node=key):
                    self.assertIn(key, NUKE["nodes"])
                    if status != "n/a":
                        self.assertIn(key, MAPPING["pairs"])

    def test_the_data_is_facts_only(self):
        banned = {"function", "description", "text", "prose"}
        for key, node in NUKE["nodes"].items():
            self.assertTrue(node["page"] is None or node["page"].startswith(NUKE["source"]["base_url"]), key)
            for knob in node["knobs"]:
                self.assertFalse(banned & set(knob), (key, knob["name"]))
                self.assertTrue(knob["name"] or not knob["counted"], key)
                self.assertIn(knob["type"], {"bool", "number", "number_list", "string", "enum", "channels", "unknown"})
                if knob["default"] is None and knob["counted"]:
                    self.assertIn("default_note", knob)
                self.assertLess(len(json.dumps(knob)), 600)

    def test_each_node_has_one_main_tab(self):
        for key, node in NUKE["nodes"].items():
            tabs = {k["tab"] for k in node["knobs"] if k["primary"]}
            self.assertLessEqual(len(tabs), 1, key)

    def test_guide_defaults_use_null_when_unstated(self):
        grade = {k["name"]: k for k in NUKE["nodes"]["Grade"]["knobs"]}
        self.assertEqual(grade["multiply"]["default"], 1)
        self.assertIs(grade["black_clamp"]["default"], True)
        self.assertIs(grade["white_clamp"]["default"], False)
        self.assertEqual(grade["blackpoint"]["default"], 0)
        self.assertTrue(grade["black"]["primary"])
        self.assertEqual(grade["mix"]["role"], "chrome:mix")
        self.assertIsNone({k["name"]: k for k in NUKE["nodes"]["Crop"]["knobs"]}["box"]["default"])


class MappingTests(unittest.TestCase):
    def test_every_nodebased_knob_and_socket_is_listed_exactly_once(self):
        for key, pair in MAPPING["pairs"].items():
            kind = pair["nodebased"]
            if kind is None:
                self.assertTrue(pair["reason"], key)
                continue
            spec = SPECS[kind]
            with self.subTest(pair=key):
                listed = [p for e in pair["knobs"] for p in e["nodebased"]]
                self.assertEqual(set(listed), set(spec["params"]))
                for p in spec["params"]:
                    self.assertGreaterEqual(listed.count(p), 1, p)
                slots = list(spec["inputs"]) + list(spec.get("optional_inputs", []))
                self.assertEqual(sorted(i["nodebased"] for i in pair["inputs"]), sorted(slots))

    def test_every_pairing_names_a_real_nuke_knob_and_gives_a_reason_when_not_identical(self):
        for key, pair in MAPPING["pairs"].items():
            if pair["nodebased"] is None:
                continue
            nuke_knobs = {k["name"] for k in NUKE["nodes"][key]["knobs"]}
            nuke_sockets = {i["name"] for i in NUKE["nodes"][key]["inputs"]}
            with self.subTest(pair=key):
                for e in pair["knobs"]:
                    self.assertIn(e["kind"], {"same", "renamed", "components", "panel", "extra"})
                    if e["kind"] == "same":
                        self.assertEqual(e["nodebased"], e["nuke"])
                    if e["kind"] != "same":
                        self.assertTrue(e["reason"], e)
                    for n in e.get("nuke", []):
                        self.assertIn(n, nuke_knobs)
                for i in pair["inputs"]:
                    if i["kind"] == "mapped":
                        self.assertIn(i["nuke"], nuke_sockets)
                    else:
                        self.assertTrue(i["reason"])

    def test_renames_the_parity_document_records_are_paired(self):
        by_name = {e["nodebased"][0]: e for e in MAPPING["pairs"]["Exposure"]["knobs"] if len(e["nodebased"]) == 1}
        self.assertEqual((by_name["exposure_mode"]["kind"], by_name["exposure_mode"]["nuke"]), ("renamed", ["mode"]))
        light = {e["nodebased"][0]: e for e in MAPPING["pairs"]["LightWrap"]["knobs"] if len(e["nodebased"]) == 1}
        self.assertEqual(light["wrap_diffuse"]["nuke"], ["size"])
        grade = {e["nodebased"][0]: e for e in MAPPING["pairs"]["Grade"]["knobs"]}
        self.assertEqual(grade["offset"]["nuke"], ["add"])
        self.assertEqual(grade["exposure"]["kind"], "extra")


class ScoringTests(unittest.TestCase):
    def fake(self, knobs, inputs=(), pair_knobs=(), pair_inputs=(), skip=()):
        nuke = {"inputs": [{"name": n, "mask": n == "mask", "type": "Input"} for n in inputs], "knobs": knobs}
        pair = {"nodebased": "Grade", "inputs": list(pair_inputs), "knobs": list(pair_knobs)}
        if skip:
            pair["skip_default"] = list(skip)
        return audit.score_pair("Fake", pair, nuke)

    @staticmethod
    def knob(name, primary=True, default=None, role="knob"):
        return {"name": name, "label": name, "tab": "T", "primary": primary, "role": role, "counted": not role.startswith("ui"),
                "type": "number", "default": default}

    def test_primary_knobs_weigh_two_and_secondary_one(self):
        knobs = [self.knob("multiply"), self.knob("blackpoint"), self.knob("gamma", primary=False)]
        pair = [{"nodebased": ["multiply"], "nuke": ["multiply"], "kind": "same"}]
        res = self.fake(knobs, pair_knobs=pair)
        self.assertAlmostEqual(res["components"]["knobs"], 2 / 5)
        self.assertEqual(res["detail"]["missing_primary"], ["blackpoint"])
        self.assertEqual(res["detail"]["missing_secondary"], ["gamma"])
        res = self.fake(knobs, pair_knobs=pair + [{"nodebased": ["offset"], "nuke": ["gamma"], "kind": "renamed", "reason": "x"}])
        self.assertAlmostEqual(res["components"]["knobs"], 3 / 5)

    def test_chrome_and_sockets_are_their_own_components(self):
        knobs = [self.knob("mix", default=1, role="chrome:mix"), self.knob("channels", default="rgb", role="chrome:channels")]
        pair = [{"nodebased": ["mix"], "nuke": ["mix"], "kind": "same"}]
        res = self.fake(knobs, inputs=("unnamed", "mask"), pair_knobs=pair,
                        pair_inputs=[{"nodebased": "image", "nuke": "unnamed", "kind": "mapped"}])
        self.assertNotIn("knobs", res["components"])
        self.assertAlmostEqual(res["components"]["chrome"], 0.5)
        self.assertAlmostEqual(res["components"]["sockets"], 0.5)
        self.assertEqual(res["detail"]["missing_sockets"], ["mask"])
        self.assertEqual(res["detail"]["missing_chrome"], ["channels"])

    def test_defaults_compare_only_paired_knobs_with_a_known_default(self):
        knobs = [self.knob("multiply", default=1), self.knob("offset", default=0.5), self.knob("gamma"), self.knob("unpaired", default=3)]
        pair = [{"nodebased": ["multiply"], "nuke": ["multiply"], "kind": "same"}, {"nodebased": ["offset"], "nuke": ["offset"], "kind": "same"},
                {"nodebased": ["exposure"], "nuke": ["gamma"], "kind": "renamed", "reason": "x"}]
        res = self.fake(knobs, pair_knobs=pair)
        self.assertEqual(res["detail"]["default_counts"], (1, 2))               # multiply 1 == 1; Grade's offset 0.0 != 0.5; gamma has no guide default
        res = self.fake(knobs, pair_knobs=pair, skip=("offset",))
        self.assertEqual(res["detail"]["default_counts"], (1, 1))

    def test_a_node_with_nothing_to_count_is_unscored_and_a_missing_node_scores_zero(self):
        self.assertIsNone(self.fake([], ())["score"])
        res = audit.score_pair("Fake", {"nodebased": None, "reason": "none"}, {"inputs": [], "knobs": [self.knob("a")]})
        self.assertEqual(res["score"], 0.0)

    def test_fidelity_is_the_weighted_mean_of_the_live_components(self):
        knobs = [self.knob("multiply", default=1)]
        res = self.fake(knobs, inputs=("unnamed",), pair_knobs=[{"nodebased": ["multiply"], "nuke": ["multiply"], "kind": "same"}],
                        pair_inputs=[{"nodebased": "image", "nuke": "unnamed", "kind": "mapped"}])
        w = audit.WEIGHTS
        self.assertAlmostEqual(res["score"], 100 * (w["knobs"] + w["defaults"] + w["sockets"]) / (w["knobs"] + w["defaults"] + w["sockets"]))

    def test_default_comparison_rules(self):
        c = audit.compare_default
        self.assertTrue(c(1.0, 1)); self.assertFalse(c(0.5, 1)); self.assertTrue(c(0, False)); self.assertFalse(c(1, False))
        self.assertTrue(c("RGB", "rgb")); self.assertTrue(c("rgba", "all")); self.assertFalse(c("box", "gaussian"))
        self.assertTrue(c([1.0, 1.0, 1.0, 1.0], 1)); self.assertTrue(c([0, 0], [0, 0])); self.assertFalse(c([0, 1], [0, 0]))
        self.assertIsNone(c("rgb", 3)); self.assertIsNone(c([1, 2], 1.5 if False else [1, 2, 3]))
        self.assertTrue(c("log to lin", "log2lin", {"log to lin": "log2lin"}))

    def test_weights_split_a_combined_row_and_rank_one_weighs_most(self):
        units, skipped = audit.build_units(NUKE, MAPPING)
        by = {(u["group"], u["name"]): u for u in units}
        self.assertAlmostEqual(by[("Draw", "Grain")]["weight"], 1 / 9 / 2)
        self.assertAlmostEqual(by[("Draw", "ScannedGrain")]["weight"], 1 / 9 / 2)
        self.assertAlmostEqual(by[("Color", "Grade")]["weight"], 1.0)
        self.assertGreater(by[("Color", "Grade")]["weight"], by[("Color", "ColorCorrect")]["weight"])
        self.assertIn(("Merge", 13, "Absminus", "n/a in PARITY_2D.md"), skipped)

    def test_overall_is_the_rank_weighted_mean(self):
        units, _ = audit.build_units(NUKE, MAPPING)
        overall, with_nodes, groups = audit.summarize(units)
        scored = [u for u in units if u["score"] is not None]
        self.assertAlmostEqual(overall, sum(u["weight"] * u["score"] for u in scored) / sum(u["weight"] for u in scored))
        self.assertLess(overall, with_nodes + 5)
        self.assertTrue(0 < overall < 100)
        self.assertEqual(set(groups), set(g for g in audit.GROUPS))


class FormulaCaseTests(unittest.TestCase):
    def test_fixtures_are_deterministic_hdr_with_alpha(self):
        a1, a2 = cases.make_fixtures(), cases.make_fixtures()
        for name in a1:
            np.testing.assert_array_equal(a1[name], a2[name])
            self.assertEqual(a1[name].shape, (cases.H, cases.W, 4))
            self.assertEqual(a1[name].dtype, np.float32)
        a, b = a1["A"], a1["B"]
        self.assertLess(float(a[..., :3].min()), 0)
        self.assertGreater(float(a[..., :3].max()), 8)
        self.assertEqual({float(v) for v in np.unique(a[..., 3])}, {float(np.float32(v)) for v in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)})
        pairs = {(float(x), float(y)) for x, y in zip(a[..., 0].ravel(), b[..., 0].ravel())}
        self.assertEqual(len(pairs), 11 * 11)                                    # every (A, B) value pair appears in the red channel

    def test_merge_reference_matches_hand_computed_values(self):
        a = np.array([[[0.5, 0.25, 2.0, 0.5]]], np.float32)
        b = np.array([[[0.2, 1.0, 0.0, 0.8]]], np.float32)
        over, _ = cases.merge_reference("over", a, b)
        np.testing.assert_allclose(over[0, 0], [0.5 + 0.2 * 0.5, 0.25 + 1.0 * 0.5, 2.0, 0.5 + 0.8 * 0.5], atol=1e-6)
        atop, _ = cases.merge_reference("atop", a, b)
        np.testing.assert_allclose(atop[0, 0, :3], [0.5 * 0.8 + 0.2 * 0.5, 0.25 * 0.8 + 1.0 * 0.5, 2.0 * 0.8], atol=1e-6)
        mult, _ = cases.merge_reference("multiply", np.array([[[-1.0, -1.0, 2.0, 1.0]]], np.float32), np.array([[[-2.0, 3.0, 4.0, 1.0]]], np.float32))
        np.testing.assert_allclose(mult[0, 0], [-1.0, -3.0, 8.0, 1.0], atol=1e-6)     # A when both negative, else AB
        div, _ = cases.merge_reference("divide", np.array([[[1.0, 1.0, 1.0, 1.0]]], np.float32), np.array([[[2.0, -2.0, 0.0, 1.0]]], np.float32))
        np.testing.assert_allclose(div[0, 0], [0.5, 1.0, 1.0, 1.0], atol=1e-6)        # A/B if B > 0, otherwise A

    def test_every_guide_operation_has_a_case_and_nodebased_knows_the_same_list(self):
        from nodebased.core import MERGE_OPERATIONS
        ids = {c.id for c in cases.CASES if c.nuke == "Merge"}
        self.assertEqual(ids, {f"merge_{op}" for op in cases.MERGE_OPERATIONS_GUIDE})
        self.assertEqual(set(cases.MERGE_OPERATIONS_GUIDE), set(MERGE_OPERATIONS))
        self.assertEqual({n for n in NUKE["nodes"]["Merge"]["knobs"][0]["choices"]}, set(cases.MERGE_OPERATIONS_GUIDE))

    def test_a_correct_node_passes_and_a_known_difference_fails(self):
        results = {c.id: audit.run_case(c) for c in cases.CASES}
        for ok in ("merge_over", "merge_plus", "merge_xor", "invert_rgb", "premult_default", "keymix_default", "colormatrix_rows"):
            self.assertEqual(results[ok]["status"], "pass", ok)
        self.assertEqual(results["merge_divide"]["status"], "fail")        # NodeBased divides to 0 where B <= 0; the guide says A
        self.assertEqual(results["clamp_to_values"]["status"], "inexpressible")
        self.assertEqual(results["grade_levels"]["status"], "pass")
        self.assertEqual(results["merge_matte"]["status"], "not scored")

    def test_cases_are_expressible_on_the_nuke_side(self):
        nuke_names = {}
        for c in cases.CASES:
            knobs = {k["name"] for k in NUKE["nodes"][c.nuke]["knobs"]}
            for knob, _ in c.nuke_knobs:
                self.assertIn(knob, knobs, (c.id, knob))
            sockets = {i["name"] for i in NUKE["nodes"][c.nuke]["inputs"]}
            self.assertTrue(c.nuke_inputs)
            for name in c.nb_inputs:
                self.assertIn(name, cases.make_fixtures())


class ReferencePackTests(unittest.TestCase):
    def test_pack_writes_a_python_script_per_case_and_a_render_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = pack.write_pack(tmp)
            root = Path(tmp)
            self.assertEqual(n, len(cases.CASES))
            self.assertEqual(len(list((root / "scripts").glob("*.py"))), n)
            self.assertEqual(len(list((root / "scripts").glob("*.nk"))), 0)
            for name in ("A", "B", "mask"):
                self.assertTrue((root / "fixtures" / f"{name}.exr").exists())
            render = root / "render_all.sh"
            if os.name != "nt":                              # Windows keeps no execute bit
                self.assertTrue(render.stat().st_mode & stat.S_IXUSR)
            text = render.read_text()
            self.assertIn("-t", text)                       # terminal mode: the only render path Non-commercial Nuke allows
            self.assertIn("scripts/*.py", text)
            self.assertNotIn(" -x ", text)
            self.assertIn("NB_RENDER_DIR", text)
            self.assertNotIn("/home/", text)
            manifest = json.loads((root / "MANIFEST.json").read_text())
            self.assertEqual({c["id"] for c in manifest["cases"]}, {c.id for c in cases.CASES})
            script = (root / "scripts" / "merge_plus.py").read_text()
            self.assertIn("nuke.nodes.Merge2()", script)
            self.assertIn("node['operation'].setValue('plus')", script)
            self.assertLess(script.index("node.setInput(0, read_B)"), script.index("node.setInput(1, read_A)"))   # B is Nuke input 0
            self.assertIn("32 bit float", script)
            self.assertIn("raw=True", script)
            self.assertIn("nuke.execute(write, 1, 1)", script)
            self.assertNotIn("/home/", script)

    def test_every_case_script_compiles_and_stays_inside_the_non_commercial_node_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack.write_pack(tmp)
            for path in sorted((Path(tmp) / "scripts").glob("*.py")):
                text = path.read_text()
                with self.subTest(case=path.stem):
                    compile(text, str(path), "exec")
                    self.assertLessEqual(text.count("nuke.nodes."), 10)
                    self.assertNotIn("toNode", text.replace("nuke.toNode or nuke.allNodes", ""))
                    self.assertNotIn("allNodes(", text)
                    self.assertIn("nuke.execute(write, 1, 1)", text)

    def test_array_knobs_go_through_fromScript_and_plain_ones_through_setValue(self):
        script = pack.py_script(next(c for c in cases.CASES if c.nuke_class == "ColorMatrix"))
        self.assertIn("node['matrix'].fromScript('{{0.5 1 2} {0 1 0} {0.25 0 1}}')", script)
        script = pack.py_script(next(c for c in cases.CASES if c.nuke_class == "Grade"))
        self.assertIn("node['black_clamp'].setValue(False)", script)

    def test_loader_scores_a_perfect_render_a_perturbed_one_and_a_missing_one(self):
        from nodebased.media import write_exr
        plus = next(c for c in cases.CASES if c.id == "merge_plus")
        over = next(c for c in cases.CASES if c.id == "merge_over")
        clamp = next(c for c in cases.CASES if c.id == "clamp_to_values")
        with tempfile.TemporaryDirectory() as tmp:
            good = cases.run_nodebased(plus).astype(np.float32)
            write_exr(Path(tmp) / "merge_plus.exr", good, bits="float", compression="none")
            bad = cases.run_nodebased(over).astype(np.float32)
            bad[3, 5, 2] += 0.5
            write_exr(Path(tmp) / "merge_over.exr", bad, bits="float", compression="none")
            write_exr(Path(tmp) / "clamp_to_values.exr", good, bits="float", compression="none")
            loaded = pack.load_render(tmp, "merge_plus")
            np.testing.assert_array_equal(loaded, good)
            self.assertEqual(pack.compare_case(plus, tmp)["status"], "pass")
            res = pack.compare_case(over, tmp)
            self.assertEqual(res["status"], "fail")
            self.assertAlmostEqual(res["channels"]["B"]["max_abs_error"], 0.5, places=5)
            self.assertEqual(res["channels"]["B"]["worst_pixel"], [3, 5])
            self.assertLess(res["channels"]["R"]["max_abs_error"], 1e-6)
            self.assertEqual(pack.compare_case(clamp, tmp)["status"], "fail")        # NodeBased cannot express the case at all
            self.assertIsNone(pack.compare_case(next(c for c in cases.CASES if c.id == "merge_in"), tmp))
            self.assertIsNone(pack.load_render(tmp, "merge_in"))

    def test_reference_dir_turns_the_pixel_column_on(self):
        from nodebased.media import write_exr
        with tempfile.TemporaryDirectory() as tmp:
            for c in cases.cases_for("Clamp"):
                out = cases.run_nodebased(c)
                if out is not None:
                    write_exr(Path(tmp) / f"{c.id}.exr", out.astype(np.float32), bits="float", compression="none")
            pair = MAPPING["pairs"]["Clamp"]
            res = audit.score_pair("Clamp", pair, NUKE["nodes"]["Clamp"], tmp)
            self.assertEqual(res["pixel_column"], "measured")
            self.assertIn("pixels", res["components"])
            self.assertEqual(len(res["detail"]["pixels"]), 2)
            plain = audit.score_pair("Clamp", pair, NUKE["nodes"]["Clamp"])
            self.assertEqual(plain["pixel_column"], "unmeasured")
            self.assertNotIn("pixels", plain["components"])


if __name__ == "__main__":
    unittest.main()
