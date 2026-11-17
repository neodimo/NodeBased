"""Lane 11 step B3: behaviour cases for the Image, Merge, Metadata and Other nodes, and Blend's invert beside the mask."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases_other as other_cases  # noqa: E402
from nodebased import core, knobs  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from tests.test_2d_behaviour_color_a import tiled  # noqa: E402
from tests.test_2d_behaviour_keyer import two_plate_graph  # noqa: E402
from tests.test_2d_parity_group_3b import evaluator_pixels  # noqa: E402


class OtherCasesTests(unittest.TestCase):
    def test_scored_cases_pass_and_unscored_ones_give_a_reason(self):
        scored = unscored = 0
        for case in other_cases.cases():
            with self.subTest(case=case.id):
                if case.unscored:
                    unscored += 1
                    self.assertTrue(case.unscored.strip())
                    continue
                scored += 1
                self.assertEqual(audit.run_case(case)["status"], "pass")
        self.assertEqual((scored, unscored), (19, 15))

    def test_every_lane_11_node_the_worklist_found_unchecked_has_a_case(self):
        nodes = {c.nuke for c in other_cases.cases()}
        for name in ("Dot", "Read", "ViewMetaData", "ModifyMetaData", "NoOp", "Write", "Backdrop", "CopyMetaData", "Viewer", "AddTimeCode",
                     "PostageStamp", "CompareMetaData", "UDIM Import", "BlinkScript", "CurveTool", "BurnIn", "Profile", "ContactSheet",
                     "Precomp", "Root", "ZMerge", "Assert", "AudioRead", "Group", "Input", "Output", "CopyBBox", "Link", "LiveGroup",
                     "LiveInput", "VariableGroup"):
            self.assertIn(name, nodes)

    def test_zmerge_cases_tell_the_settings_apart(self):
        from nuke_fidelity_cases import make_fixtures
        fx = make_fixtures()
        by_id = {c.id: c for c in other_cases.cases()}
        near = by_id["zmerge_smaller_z_is_nearer"].expected(fx)[0]
        back = by_id["zmerge_backwards_larger_z_is_nearer"].expected(fx)[0]
        self.assertGreater(float(np.abs(near - back).max()), 0.1)


def blend(params, a, b, mask=None):
    inputs = [a, b] + [None] * 14 + [mask]
    return Evaluator._kernel("Blend", dict(core.SPECS["Blend"]["params"], **params), inputs)


class BlendInvertMaskTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.a, self.b = rng.random((6, 8, 4)).astype(np.float32), rng.random((6, 8, 4)).astype(np.float32)
        self.mask = np.zeros((6, 8, 4), np.float32)
        self.mask[:, :4, 3] = 1.0

    def test_invert_swaps_where_the_blend_shows(self):
        plain = blend({}, self.a, self.b, self.mask)
        inverted = blend({"invert_mask": 1}, self.a, self.b, self.mask)
        np.testing.assert_array_equal(plain[:, 4:], self.a[:, 4:])          # outside the matte: the first input
        np.testing.assert_array_equal(inverted[:, :4], self.a[:, :4])       # inverted: inside the matte is untouched
        np.testing.assert_allclose(inverted[:, 4:], blend({}, self.a, self.b)[:, 4:], atol=1e-6)  # and the blend shows outside it

    def test_invert_without_a_mask_changes_nothing(self):
        np.testing.assert_array_equal(blend({"invert_mask": 1}, self.a, self.b), blend({}, self.a, self.b))

    def test_inject_writes_the_inverted_matte(self):
        out = blend({"invert_mask": 1, "inject": 1}, self.a, self.b, self.mask)
        np.testing.assert_array_equal(out[:, :4, 3], 0.0)
        np.testing.assert_array_equal(out[:, 4:, 3], 1.0)

    def test_tiles_equal_the_full_frame_with_invert(self):
        for masked in (False, True):
            g = two_plate_graph("Blend", {"invert_mask": 1, "weight0": 0.3}, masked, second="in1", first="in0")
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)

    def test_a_blend_saved_without_invert_loads_with_it_off(self):
        doc = {"version": core.SCHEMA_VERSION, "view": "b", "nodes": {"b": {"type": "Blend", "params": dict(core.SPECS["Blend"]["params"])}}}
        del doc["nodes"]["b"]["params"]["invert_mask"]
        loaded = core.upgrade_document(json.loads(json.dumps(doc)))
        self.assertEqual(loaded["nodes"]["b"]["params"]["invert_mask"], 0)

    def test_panel_puts_invert_after_inject_and_pairs_nukes_knob(self):
        order = knobs.NUKE_LAYOUT_LANE11["Blend"]
        self.assertLess(order.index("inject"), order.index("invert_mask"))
        self.assertLess(order.index("invert_mask"), order.index("fringe"))
        pair = json.loads((ROOT / "docs" / "parity" / "nodebased_to_nuke.json").read_text(encoding="utf-8"))["pairs"]["Blend"]
        self.assertIn({"nodebased": ["invert_mask"], "nuke": ["invert_mask"], "kind": "same"}, pair["knobs"])


if __name__ == "__main__":
    unittest.main()
