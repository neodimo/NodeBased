"""2D behaviour lane 2, step E (Merge group): Switch, Dissolve, AddMix, Blend, CopyRectangle and TimeDissolve compute what the guide says, and
every knob the mapping pairs with a NodeBased param moves the picture.

The cases themselves live in tools/nuke_fidelity_cases_merge.py; the audit runs them. This module asserts their statuses, the clamp that
Switch gained (a `which` past either end shows the nearest input; it raised before), the Blend edge the guide leaves open, the paired knobs
acting on the evaluator, the tile path agreeing with the full frame, and TimeDissolve over several frames.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases_merge as merge_cases  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402
from nodebased import core  # noqa: E402
from nodebased.core import SPECS, switch_index  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from tests.test_2d_behaviour_color_a import tiled  # noqa: E402
from tests.test_2d_behaviour_keyer import two_plate_graph  # noqa: E402
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels  # noqa: E402

import nuke_fidelity_cases as fidelity  # noqa: E402

FX = fidelity.make_fixtures()
KINDS = ("Switch", "Dissolve", "TimeDissolve", "AddMix", "Blend", "CopyRectangle", "Constant")
# Cases listed `not scored`, each with the reason in the case itself.
NOT_SCORED = {"switch_which_past_last", "switch_which_fraction", "addmix_nonlinear", "blend_all_zero_weights", "copyrectangle_softness", "constant_color_alpha"}


def kernel(kind, inputs, **params):
    return Evaluator._kernel(kind, dict(SPECS[kind]["params"], **params), list(inputs))


class CasesTests(unittest.TestCase):
    def test_every_case_passes_or_says_why_it_is_not_scored(self):
        found = {c.id: c for c in merge_cases.cases()}
        self.assertEqual({i for i, c in found.items() if c.unscored}, NOT_SCORED)
        for case_id, case in found.items():
            with self.subTest(case=case_id):
                res = audit.run_case(case)
                if case_id in NOT_SCORED:
                    self.assertEqual(res["status"], "not scored")
                    self.assertGreater(len(case.unscored), 40)
                else:
                    self.assertEqual(res["status"], "pass", res)
                self.assertIn(case.source, ("guide", "convention"))

    def test_every_node_of_the_step_has_a_scored_case(self):
        for kind in KINDS:
            scored = [c for c in fidelity.cases_for(kind) if not c.unscored]
            self.assertTrue(scored, kind)


class SwitchTests(unittest.TestCase):
    def test_switch_index_holds_to_the_ends(self):
        self.assertEqual([switch_index(w, 2) for w in (-3, -1, 0, 1, 2, 9)], [0, 0, 0, 1, 1, 1])
        self.assertEqual(switch_index(7, 4), 3)

    def test_kernel_shows_the_nearest_input_past_either_end(self):
        a, b = FX["A"], FX["B"]
        for which, want in ((0, a), (1, b), (2, b), (5, b), (-1, a)):
            np.testing.assert_array_equal(kernel("Switch", [a, b], which=which), want)

    def test_graph_and_source_walk_agree_past_the_end(self):
        for which, plate in ((5, "plate_a"), (-2, "plate_b"), (1, "plate_a"), (0, "plate_b")):
            g = Graph()
            g.add("plate_a", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.9, alpha=0.8))
            g.add("plate_b", "Ramp", dict(width=61, height=47))
            g.add("sw", "Switch", {}, **{"0": "plate_b", "1": "plate_a"})
            g.add("node", "Multiply", {"multiply": 1.0}, image="sw")
            g.doc["nodes"]["sw"]["params"]["which"] = which       # past the document's 0..1 range, as an animated curve can reach
            full = evaluator_pixels(g.doc, "node")
            np.testing.assert_allclose(full, evaluator_pixels(g.doc, plate), atol=1e-6, err_msg=f"which {which}")
            # Switch is not a tiled kind (tests/test_tiles.py); the tile executor's source walks pick the same input for the canvas.
            from nodebased.tileexec import TileExecutor
            region = TileExecutor(evaluator=Evaluator()).canvas_region(dict(g.doc, view="node"), "node", frame=1, tier=1)
            self.assertEqual((region.width, region.height), (61, 47))


class BlendEdgeTests(unittest.TestCase):
    def test_all_zero_weights_with_normalize_on_give_black(self):
        out = kernel("Blend", [FX["A"], FX["B"]], weight0=0.0, weight1=0.0)
        np.testing.assert_array_equal(out, np.zeros_like(FX["A"]))

    def test_all_zero_weights_with_normalize_off_give_black_too(self):
        out = kernel("Blend", [FX["A"], FX["B"]], weight0=0.0, weight1=0.0, normalize=0)
        np.testing.assert_array_equal(out, np.zeros_like(FX["A"]))

    def test_normalize_off_is_the_plain_weighted_sum(self):
        # The guide's table names the checkbox weight1 (its label is normalize), so it cannot be written as a Nuke knob in a case; it is checked here.
        out = kernel("Blend", [FX["A"], FX["B"]], weight0=0.5, weight1=0.25, normalize=0)
        np.testing.assert_allclose(out, 0.5 * FX["A"] + 0.25 * FX["B"], atol=1e-6)
        normalized = kernel("Blend", [FX["A"], FX["B"]], weight0=0.5, weight1=0.25)
        np.testing.assert_allclose(normalized, (0.5 * FX["A"] + 0.25 * FX["B"]) / 0.75, atol=1e-6)

    def test_a_third_input_takes_its_share(self):
        c = FX["C"]
        out = kernel("Blend", [FX["A"], FX["B"], c], weight0=1.0, weight1=1.0, weight2=2.0)
        want = (FX["A"] + FX["B"] + 2.0 * c) / 4.0
        np.testing.assert_allclose(out, want, atol=1e-5)


# What each paired knob is moved to (from its default) for the pixel check, per kind: {NodeBased param: params to set}.
MOVES = {
    "Switch": {"which": {"which": 1}},
    "Dissolve": {"which": {"which": 0.5}, "mix": {"which": 0.5, "mix": 0.5}},
    "AddMix": {"mix": {"mix": 0.5}, "Achannels": {"Achannels": "rgb"}, "Bchannels": {"Bchannels": "rgb"}, "merge_output": {"merge_output": "rgb"},
               "premultiplied": {"premultiplied": 1}, "nonlinear": {"nonlinear": 1}, "lut_a": {"lut_a": "0:0 0.5:1 1:1"},
               "lut_b": {"lut_b": "0:1 0.5:0.25 1:0"}},
    "Blend": {"weight0": {"weight0": 3.0}, "weight1": {"weight1": 3.0}, "normalize": {"normalize": 0}, "channels": {"channels": "rgb"},
              "mix": {"mix": 0.5}},
    "CopyRectangle": {"channels": {"channels": "rgb"}, "area_x": {"area_x": 5.0}, "area_y": {"area_y": 4.0}, "area_r": {"area_r": 9.0},
                      "area_t": {"area_t": 7.0}, "softness": {"softness": 0.5}, "mix": {"mix": 0.5}},
}
BASE = {"CopyRectangle": {"area_x": 3.0, "area_y": 2.0, "area_r": 11.0, "area_t": 9.0}, "Dissolve": {"which": 0.25}}
# Blend's `channels` is read from the lower-case alias in the mapping; the Nuke chrome knobs (mask, inject, invert, fringe) are the shared chrome tests' job.
# Blend's `channels` default is rgba: with rgb only the alpha stays the first input's, which differs from the blend on the fixture.


class PairedKnobsActTests(unittest.TestCase):
    def test_every_paired_param_of_the_group_is_listed(self):
        pairs = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]
        chrome = {"maskChannelInput", "invert_mask", "inject", "fringe", "mask_channel"}
        for kind, moves in MOVES.items():
            paired = {p for e in pairs[kind]["knobs"] if e["kind"] in requirements.PAIRED for p in e["nodebased"]} - chrome
            if kind == "Dissolve":
                paired -= {"channels"}           # Dissolve's channels is the shared chrome (tests/test_2d_fidelity_b_chrome.py)
            extra_weights = {f"weight{i}" for i in range(2, 16)} if kind == "Blend" else set()     # the same knob as weight0 and weight1, for inputs 2 to 15
            self.assertEqual(paired - set(moves) - extra_weights, set(), kind)

    def test_each_knob_moved_off_its_default_changes_the_pixels(self):
        inputs = [FX["A"], FX["B"]]
        for kind, moves in MOVES.items():
            base = dict(BASE.get(kind, {}))
            ref = kernel(kind, inputs, **base)
            for name, change in moves.items():
                with self.subTest(kind=kind, knob=name):
                    moved = kernel(kind, inputs, **dict(base, **change))
                    self.assertGreater(float(np.abs(moved - ref).max()), 1e-6)

    def test_constant_knobs_move_the_picture(self):
        size = dict(width=16, height=12)
        ref = kernel("Constant", [], **size)
        for name, change in (("red", {"red": 0.9}), ("green", {"green": 0.9}), ("blue", {"blue": 0.1}), ("alpha", {"alpha": 0.5}),
                             ("channels", {"channels": "r"})):
            with self.subTest(knob=name):
                self.assertGreater(float(np.abs(kernel("Constant", [], **size, **change) - ref).max()), 1e-6)
        self.assertEqual(kernel("Constant", [], width=10, height=7).shape, (7, 10, 4))

    def test_the_probe_start_for_copyrectangle_puts_the_box_inside_the_fixture(self):
        # Nuke's default box (512 389 1536 1167) lies off a 16 x 12 picture, which is why the old probe saw five inert knobs.
        default = SPECS["CopyRectangle"]["params"]
        self.assertGreater(default["area_x"], 12)
        acts = requirements.probe_acts("CopyRectangle", ["area_x", "area_y", "area_r", "area_t", "softness"], FX)
        self.assertEqual(acts, {k: True for k in ("area_x", "area_y", "area_r", "area_t", "softness")})

    def test_copyrectangle_default_box_is_unchanged(self):
        self.assertEqual({k: SPECS["CopyRectangle"]["params"][k] for k in ("area_x", "area_y", "area_r", "area_t")},
                         {"area_x": 512.0, "area_y": 389.0, "area_r": 1536.0, "area_t": 1167.0})


class TilesMatchTests(unittest.TestCase):
    def test_merge_group_nodes_equal_the_full_frame_in_tiles(self):
        cases = (("Dissolve", {"which": 0.4}), ("AddMix", {"lut_a": "0:0 0.5:1 1:1"}), ("AddMix", {"premultiplied": 1}),
                 ("CopyRectangle", {"area_x": 8.0, "area_y": 6.0, "area_r": 50.0, "area_t": 40.0, "softness": 0.4}),
                 ("CopyRectangle", {"area_x": 8.0, "area_y": 6.0, "area_r": 50.0, "area_t": 40.0, "channels": "rgb"}))
        for kind, params in cases:
            for masked in (False, True):
                g = two_plate_graph(kind, params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class TimeDissolveTests(unittest.TestCase):
    """The dissolve starts at `in` and ends at `out`; the curve (the ease) shapes the mix between."""

    def build(self, **params):
        g = Graph()
        g.add("a", "Constant", dict(width=2, height=2, red=1.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("b", "Constant", dict(width=2, height=2, red=0.0, green=0.0, blue=1.0, alpha=1.0))
        g.add("mix", "TimeDissolve", dict({"in": 3, "out": 7}, **params), A="a", B="b")
        return g

    def blue(self, g, frame):
        return float(Evaluator().evaluate(dict(g.doc, view="mix"), frame=frame)[0, 0, 2])

    def test_start_middle_and_end_of_the_range(self):
        g = self.build()
        shown = {f: self.blue(g, f) for f in (1, 2, 3, 4, 5, 6, 7, 8, 12)}
        self.assertEqual([shown[1], shown[2], shown[3]], [0.0, 0.0, 0.0])               # before and at `in`: all A
        self.assertAlmostEqual(shown[5], 0.5, places=6)                                   # the middle
        self.assertEqual([shown[7], shown[8], shown[12]], [1.0, 1.0, 1.0])               # at `out` and after: all B
        self.assertEqual(sorted(shown.values()), [shown[f] for f in sorted(shown)])      # never goes back
        self.assertAlmostEqual(shown[4], 0.25, places=6)
        self.assertAlmostEqual(shown[6], 0.75, places=6)

    def test_in_and_out_move_the_range(self):
        wide, narrow = self.build(**{"in": 1, "out": 9}), self.build(**{"in": 4, "out": 6})
        self.assertAlmostEqual(self.blue(wide, 5), 0.5, places=6)
        self.assertAlmostEqual(self.blue(wide, 3), 0.25, places=6)
        self.assertAlmostEqual(self.blue(narrow, 5), 0.5, places=6)
        self.assertEqual(self.blue(narrow, 3), 0.0)
        self.assertEqual(self.blue(narrow, 6), 1.0)

    def test_the_curve_shapes_the_mix(self):
        linear, smooth = self.build(ease="linear"), self.build(ease="smooth")
        self.assertAlmostEqual(self.blue(linear, 4), 0.25, places=6)
        self.assertAlmostEqual(self.blue(smooth, 4), 0.25 * 0.25 * (3.0 - 2.0 * 0.25), places=6)    # smoothstep(0.25) = 0.15625
        self.assertAlmostEqual(self.blue(smooth, 5), 0.5, places=6)
        self.assertEqual((self.blue(smooth, 3), self.blue(smooth, 7)), (0.0, 1.0))

    def test_an_animated_curve_replaces_the_ramp(self):
        g = self.build(ease="animation curve")
        g.doc["animation"]["curves"]["mix"] = {"which": {"interpolation": "linear", "keys": [{"frame": 3, "value": 0.0}, {"frame": 7, "value": 0.2}]}}
        self.assertAlmostEqual(self.blue(g, 5), 0.1, places=6)
        self.assertAlmostEqual(self.blue(g, 7), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
