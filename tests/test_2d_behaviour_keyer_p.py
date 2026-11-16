"""2D behaviour lane 2, step P (11/16): behaviour checks for ChromaKeyer, Keylight (ScreenKeyer), IBKColor, IBKGizmo, ShuffleCopy, Remove and Posterize.

The cases live in tools/nuke_fidelity_cases_color.py and are scored by tools/audit_nuke_fidelity.py. This module runs them as tests and adds what the
cases cannot say: the cases are not trivial (the reference differs from the plain input), every paired knob of ShuffleCopy and Remove changes the
output on the probe's fixture, the tile path equals the whole frame for the keyers and Posterize, and Posterize's main tab is in Nuke's order.
"""
import os
import sys
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import nuke_fidelity_cases as ruler  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402
from nodebased import knobs  # noqa: E402
from nodebased.tileexec import SUPPORTED_TILED_KINDS  # noqa: E402
from tests.test_2d_behaviour_color_a import TilesMatchTests, evaluator_pixels, tiled  # noqa: E402
from tests.test_2d_parity_group_3b import Graph  # noqa: E402

# Identity by design: the default routing of ShuffleCopy hands input 1 on, and Remove with no layer named removes nothing.
IDENTITY_BY_DESIGN = {"shufflecopy_default_passes_input_one", "remove_nothing_named"}
KEYS = {"ChromaKeyer", "Keylight", "IBKColor", "IBKGizmo", "ShuffleCopy", "Remove", "Posterize"}


class StepPCaseTests(unittest.TestCase):
    def test_every_node_has_a_scored_case(self):
        for key in KEYS:
            scored = [c for c in ruler.CASES if c.nuke == key and not c.unscored]
            self.assertTrue(scored, f"{key} has no scored behaviour case")

    def test_every_scored_case_passes(self):
        fx = ruler.make_fixtures()
        for case in (c for c in ruler.CASES if c.nuke in KEYS and not c.unscored):
            with self.subTest(case=case.id):
                got = ruler.run_nodebased(case, fx)
                ref, valid = case.expected(fx)
                self.assertTrue(valid.any(), "nothing is compared")
                self.assertLessEqual(float((np.abs(got - ref) * valid).max()), 1e-5)

    def test_the_cases_would_catch_a_pass_through(self):
        """A kernel that handed its first input back unchanged must fail each keyer, channel and posterize case (the reference is not the input)."""
        fx = ruler.make_fixtures()
        for case in (c for c in ruler.CASES if c.nuke in KEYS and not c.unscored):
            ref, valid = case.expected(fx)
            first = fx[case.nb_inputs[0]].astype(np.float64)
            if case.nb_params and "_remove" in case.nb_params:
                first = fx["A"].astype(np.float64).copy()
                first[-1, :3, :] = 1.0                      # nothing removed: every layer marker present
            if case.id in IDENTITY_BY_DESIGN:
                continue
            moved = int((valid & (np.abs(ref - first) > 1e-9)).sum())
            with self.subTest(case=case.id):
                self.assertGreater(moved, 0, "the reference equals the input everywhere it is compared")

    def test_the_keyer_fixture_has_the_stated_colours(self):
        k = ruler.make_fixtures()["K"]
        np.testing.assert_allclose(k[0, 0, :3], ruler.KEY_SCREEN)
        np.testing.assert_allclose(k[0, 4, :3], ruler.KEY_SKIN)
        np.testing.assert_allclose(k[0, 8, :3], ruler.KEY_HALF)
        np.testing.assert_allclose(k[0, 12, :3], ruler.KEY_GREY)
        np.testing.assert_allclose(k[-1, 12, :3], ruler.KEY_SPILL)
        np.testing.assert_allclose(k[..., 3], 1.0)


class ShuffleCopyAndRemoveKnobsActTests(unittest.TestCase):
    """Every paired knob of the two layer nodes changes the probe's output, given the state it needs: named layers on the inputs (the probe adds a `depth`
    layer to each ShuffleCopy input, and `depth`, `mask` and `id` to Remove's)."""

    def test_shufflecopy_every_param_acts(self):
        fx = ruler.make_fixtures()
        names = [f"out{o}_{c}" for o in (1, 2) for c in "rgba"] + ["layer1", "layer2"]
        acts = requirements.probe_acts("ShuffleCopy", names, fx)
        self.assertEqual({n for n, ok in acts.items() if not ok}, set())
        self.assertEqual(set(acts), set(names))

    def test_remove_every_param_acts(self):
        fx = ruler.make_fixtures()
        names = ["remove_operation", "layers", "channels2", "channels3", "channels4"]
        acts = requirements.probe_acts("Remove", names, fx)
        self.assertEqual({n for n, ok in acts.items() if not ok}, set())
        self.assertEqual(set(acts), set(names))


class StepPTilesTests(TilesMatchTests):
    TILE_PARAMS = {
        "ScreenKeyer": [{}, {"screen_gain": 2.0, "clip_black": 0.2, "keyer_view": "screen_matte"}, {"keyer_view": "intermediate", "screen_replace": "none"}],
        "Posterize": [{"colors": 2}, {"colors": 5, "channels": "rgba"}],
    }

    def test_screen_keyer_tiles_equal_the_whole_frame(self):
        self._check("ScreenKeyer")

    def test_ibk_color_tiles_equal_the_whole_frame(self):
        """IBKColor reads its plate on the `c` slot: a green plate with a skin-tone island, filled across tiles of 7, 16 and 32 as across the whole frame."""
        for params in ({"screen_type": "green", "screen_erode": 0.0, "patch_black": 0, "fill_size": 3}, {"screen_type": "green", "screen_erode": 1.0, "fill_size": 6}):
            g = Graph()
            g.add("screen", "Constant", dict(width=61, height=47, red=0.1, green=0.8, blue=0.2, alpha=1.0))
            g.add("island", "Rectangle", dict(width=61, height=47, box_x=20, box_y=15, box_width=18, box_height=14, red=0.8, green=0.5, blue=0.4, alpha=1.0, softness=0.0))
            g.add("plate", "Merge", dict(operation="over"), A="island", B="screen")
            g.add("node", "IBKColor", params, c="plate")
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(params=params, edge=edge):
                    np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)

    def test_posterize_tiles_equal_the_whole_frame(self):
        self._check("Posterize")

    def test_every_kind_runs_on_tiles(self):
        for kind in ("ChromaKeyer", "ScreenKeyer", "IBKColor", "IBKGizmo", "Posterize", "ShuffleCopy", "Remove"):
            self.assertIn(kind, SUPPORTED_TILED_KINDS)


class ShuffleCopyConstantsTests(unittest.TestCase):
    """ShuffleCopy's black and white sources ("0" and "1") on the whole frame, on the second output layer and across tiles."""

    PARAMS = {"out1_r": "in2.g", "out1_g": "0", "out1_b": "1", "out1_a": "in1.a", "out2_r": "1", "out2_g": "in1.b", "out2_b": "0", "out2_a": "in2.r"}

    def graph(self):
        g = Graph()
        g.add("a", "Ramp", dict(width=61, height=47))
        g.add("b", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.9, green=0.5, blue=0.2, alpha=0.8, softness=0.5))
        g.add("node", "ShuffleCopy", dict(self.PARAMS), in1="a", in2="b")
        return g

    def test_both_outputs_follow_the_routes(self):
        from nodebased.imaging import Evaluator
        g = self.graph()
        doc = dict(g.doc, view="node")
        first = np.asarray(Evaluator().evaluate_raster(doc, target="node").pixels, np.float64)
        raster = Evaluator().evaluate_raster(doc, target="node")
        a = np.asarray(Evaluator().evaluate_raster(dict(g.doc, view="a"), target="a").pixels, np.float64)
        b = np.asarray(Evaluator().evaluate_raster(dict(g.doc, view="b"), target="b").pixels, np.float64)
        np.testing.assert_allclose(first[..., 0], b[..., 1], atol=1e-6)
        np.testing.assert_allclose(first[..., 1], 0.0, atol=0)
        np.testing.assert_allclose(first[..., 2], 1.0, atol=0)
        np.testing.assert_allclose(first[..., 3], a[..., 3], atol=1e-6)
        second = np.asarray(raster.layers["out2"].pixels, np.float64)
        np.testing.assert_allclose(second[..., 0], 1.0, atol=0)
        np.testing.assert_allclose(second[..., 1], a[..., 2], atol=1e-6)
        np.testing.assert_allclose(second[..., 2], 0.0, atol=0)
        np.testing.assert_allclose(second[..., 3], b[..., 0], atol=1e-6)

    def test_tiles_equal_the_whole_frame(self):
        g = self.graph()
        full = evaluator_pixels(g.doc, "node")
        for edge in (7, 16, 32):
            with self.subTest(edge=edge):
                np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class PosterizeLayoutTests(unittest.TestCase):
    def test_main_tab_follows_nukes_order(self):
        """Nuke's Posterize tab: channels, Colors, then the mask and (un)premult rows every node has."""
        labels = [g.label for g in knobs.KNOB_LAYOUT["Posterize"] if g.params != ("mix",)]
        self.assertEqual(labels[:3], ["Channels", "Colors", "Mask"])
        by_param = {g.params[0]: g.kind for g in knobs.KNOB_LAYOUT["Posterize"]}
        self.assertEqual(by_param["colors"], "int")      # Colors counts levels: a whole number, as in Nuke


if __name__ == "__main__":
    unittest.main()
