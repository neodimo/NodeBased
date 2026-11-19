"""Lane 2 step X1: the ColorTransfer node (Nuke 17.0, Color menu).

Input A holds the colours to match, measured inside the ROI (x, y, r, t from A's bottom left; all zero is the whole of A, the guide's default);
input B is the picture that takes them. The guide documents only the ROI, so the match is the Reinhard et al. (2001) per-channel mean and
standard deviation transfer, run on MatchGrade's own statistics helper.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as fidelity  # noqa: E402
from nodebased import core  # noqa: E402
from nodebased.core import MASK_MIX_KINDS, SPECS  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.raster import Raster  # noqa: E402
from nodebased.tiles import SUPPORTED_TILED_KINDS  # noqa: E402
from tests.test_2d_behaviour_knobs_color import assert_in_order  # noqa: E402
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels  # noqa: E402


def picture(seed, h=12, w=16, scale=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    out = (rng.random((h, w, 4)) * scale + shift).astype(np.float32)
    out[..., 3] = 0.25 + 0.5 * rng.random((h, w))
    return out


def transfer(a, b, **params):
    """The node on two pictures, through the windowed kernel the evaluator and the audit use."""
    p = dict(SPECS["ColorTransfer"]["params"], **params)
    return Evaluator._windowed_kernel("ColorTransfer", p, [Raster.of(a), Raster.of(b)]).pixels


def mean_std(pixels, c):
    channel = pixels[..., c].astype(np.float64)
    return channel.mean(), channel.std()


class CaseTests(unittest.TestCase):
    def test_every_case_is_scored_and_passes(self):
        mine = [c for c in fidelity.CASES if c.nuke == "ColorTransfer"]
        self.assertGreaterEqual(len(mine), 4)
        for case in mine:
            with self.subTest(case=case.id):
                self.assertFalse(case.unscored)
                self.assertEqual(audit.run_case(case)["status"], "pass")


class BehaviourTests(unittest.TestCase):
    def setUp(self):
        self.a = picture(1, scale=0.5, shift=0.4)       # the colours to match: a bright, narrow spread
        self.b = picture(2, scale=1.5, shift=-0.2)      # the picture: darker, wider

    def test_b_takes_the_mean_and_spread_of_a_in_every_channel(self):
        out = transfer(self.a, self.b)
        for c in range(3):
            got_mean, got_std = mean_std(out, c)
            want_mean, want_std = mean_std(self.a, c)
            self.assertAlmostEqual(got_mean, want_mean, places=5)
            self.assertAlmostEqual(got_std, want_std, places=5)

    def test_alpha_is_bs_own(self):
        np.testing.assert_array_equal(transfer(self.a, self.b)[..., 3], self.b[..., 3])

    def test_identical_inputs_give_the_picture_back(self):
        np.testing.assert_allclose(transfer(self.a, self.a), self.a, atol=1e-6)

    def test_b_keeps_its_own_pattern(self):
        """Only mean and spread move: the picture's pixels stay in the same order (a straight line through them)."""
        out = transfer(self.a, self.b)
        for c in range(3):
            self.assertGreater(np.corrcoef(out[..., c].ravel(), self.b[..., c].ravel())[0, 1], 0.999999)

    def test_a_flat_b_takes_a_means(self):
        flat = np.full((12, 16, 4), 0.3, np.float32)
        out = transfer(self.a, flat)
        for c in range(3):
            np.testing.assert_allclose(out[..., c], self.a[..., c].mean(), atol=1e-6)

    def test_the_two_inputs_may_differ_in_size(self):
        small = picture(3, h=5, w=7, scale=0.5, shift=0.4)
        out = transfer(small, self.b)
        self.assertEqual(out.shape, self.b.shape)
        for c in range(3):
            self.assertAlmostEqual(mean_std(out, c)[0], mean_std(small, c)[0], places=5)


class RoiTests(unittest.TestCase):
    def setUp(self):
        # A: its left half dark and flat-ish, its right half bright, so the region read decides the answer.
        self.a = np.empty((12, 16, 4), np.float32)
        self.a[:, :8] = picture(4, scale=0.2, shift=0.1)[:, :8]
        self.a[:, 8:] = picture(5, scale=0.2, shift=0.7)[:, 8:]
        self.b = picture(6, scale=1.0, shift=0.0)

    def test_the_default_roi_is_the_whole_of_a_and_so_is_every_degenerate_box(self):
        whole = transfer(self.a, self.b)
        np.testing.assert_array_equal(transfer(self.a, self.b, roi_x=0, roi_y=0, roi_r=16, roi_t=12), whole)
        for box in ({"roi_x": 5, "roi_y": 0, "roi_r": 5, "roi_t": 12},      # no width
                    {"roi_x": 9, "roi_y": 0, "roi_r": 3, "roi_t": 12},      # reversed
                    {"roi_x": 40, "roi_y": 0, "roi_r": 50, "roi_t": 12}):   # entirely off A
            with self.subTest(box=box):
                np.testing.assert_array_equal(transfer(self.a, self.b, **box), whole)

    def test_the_roi_limits_where_a_is_measured(self):
        left = transfer(self.a, self.b, roi_x=0, roi_y=0, roi_r=8, roi_t=12)
        right = transfer(self.a, self.b, roi_x=8, roi_y=0, roi_r=16, roi_t=12)
        whole = transfer(self.a, self.b)
        self.assertFalse(np.array_equal(left, right))
        self.assertFalse(np.array_equal(left, whole))
        self.assertFalse(np.array_equal(right, whole))
        for c in range(3):
            self.assertAlmostEqual(mean_std(left, c)[0], self.a[:, :8, c].mean(), places=5)
            self.assertAlmostEqual(mean_std(left, c)[1], self.a[:, :8, c].std(), places=5)
            self.assertAlmostEqual(mean_std(right, c)[0], self.a[:, 8:, c].mean(), places=5)
        self.assertLess(mean_std(left, 0)[0], mean_std(right, 0)[0])        # the dark half asks for a darker result

    def test_y_and_t_count_up_from_the_bottom_edge(self):
        bottom = transfer(self.a, self.b, roi_x=0, roi_y=0, roi_r=16, roi_t=4)
        top = transfer(self.a, self.b, roi_x=0, roi_y=8, roi_r=16, roi_t=12)
        for c in range(3):
            self.assertAlmostEqual(mean_std(bottom, c)[0], self.a[8:, :, c].mean(), places=5)     # the last four rows of the array
            self.assertAlmostEqual(mean_std(top, c)[0], self.a[:4, :, c].mean(), places=5)        # the first four rows

    def test_every_roi_value_changes_the_picture(self):
        base = transfer(self.a, self.b, roi_x=2, roi_y=2, roi_r=14, roi_t=10)
        for name, value in (("roi_x", 5), ("roi_y", 5), ("roi_r", 9), ("roi_t", 6)):
            with self.subTest(knob=name):
                changed = transfer(self.a, self.b, **{"roi_x": 2, "roi_y": 2, "roi_r": 14, "roi_t": 10, name: value})
                self.assertFalse(np.array_equal(changed, base))


class NodeTests(unittest.TestCase):
    def test_registered_with_a_mask_and_mix_and_filed(self):
        spec = SPECS["ColorTransfer"]
        self.assertEqual(spec["inputs"], ["A", "B"])
        self.assertEqual(spec["optional_inputs"], ["mask"])
        self.assertIn("ColorTransfer", MASK_MIX_KINDS)
        self.assertEqual(core.OUTPUT_TYPES["ColorTransfer"], "image")

    def test_mix_and_mask(self):
        a, b = picture(1, scale=0.5, shift=0.4), picture(2, scale=1.5, shift=-0.2)
        full = transfer(a, b)
        np.testing.assert_array_equal(transfer(a, b, mix=0.0), b)
        np.testing.assert_allclose(transfer(a, b, mix=0.5), (b + full) / 2, atol=1e-6)
        mask = np.zeros((12, 16, 4), np.float32)
        mask[:, :8] = 1.0
        p = dict(SPECS["ColorTransfer"]["params"])
        out = Evaluator._windowed_kernel("ColorTransfer", p, [Raster.of(a), Raster.of(b), Raster.of(mask)]).pixels
        np.testing.assert_allclose(out[:, :8], full[:, :8], atol=1e-6)
        np.testing.assert_array_equal(out[:, 8:], b[:, 8:])

    def test_a_document_renders_it_and_bypass_passes_input_zero_on(self):
        g = Graph()
        g.add("a", "Constant", {"red": 0.8, "green": 0.6, "blue": 0.2, "alpha": 1.0})
        g.add("b", "Constant", {"red": 0.1, "green": 0.1, "blue": 0.5, "alpha": 1.0})
        g.add("ct", "ColorTransfer", {}, A="a", B="b")
        out = evaluator_pixels(g.doc, "ct")
        np.testing.assert_allclose(out[0, 0, :3], [0.8, 0.6, 0.2], atol=1e-5)   # a flat B has no spread, so it takes A's mean
        g.bypass("ct")      # a disabled Nuke node passes input 0, which the guide lists as A; core.bypass_slot says the same
        self.assertEqual(core.bypass_slot({"type": "ColorTransfer", "inputs": {"A": "a", "B": "b"}}), "A")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "ct")[0, 0, :3], [0.8, 0.6, 0.2], atol=1e-6)

    def test_global_statistics_keep_it_off_the_tile_path_like_matchgrade(self):
        self.assertNotIn("ColorTransfer", SUPPORTED_TILED_KINDS)
        self.assertNotIn("MatchGrade", SUPPORTED_TILED_KINDS)

    def test_main_tab_reads_roi_then_mix(self):
        assert_in_order(self, "ColorTransfer", ["ROI", "Mix"])

    def test_mapping_pairs_the_node_and_its_roi(self):
        entry = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]["ColorTransfer"]
        self.assertEqual(entry["nodebased"], "ColorTransfer")
        paired = [k for k in entry["knobs"] if k.get("nuke") == ["ROI"]]
        self.assertEqual(paired[0]["nodebased"], ["roi_x", "roi_y", "roi_r", "roi_t"])


class MatchGradeUnchangedTests(unittest.TestCase):
    """MatchGrade now reads its statistics from the helper ColorTransfer shares; its pixels do not move."""

    def test_the_live_match_equals_the_formula(self):
        source, target = picture(7, scale=1.5, shift=-0.2), picture(8, scale=0.5, shift=0.4)
        p = dict(SPECS["MatchGrade"]["params"])
        got = Evaluator._match_grade(source, target, p)
        for c in range(3):
            s, t = source[..., c].astype(np.float64), target[..., c].astype(np.float64)
            want = (s - s.mean()) * (t.std() / s.std()) + t.mean()
            np.testing.assert_allclose(got[..., c], want, atol=2e-6)
        np.testing.assert_array_equal(got[..., 3], source[..., 3])

    def test_the_masked_statistics_are_the_weighted_ones(self):
        values = np.arange(12, dtype=np.float64).reshape(3, 4)
        weights = np.array([[1, 1, 0, 0]] * 3, np.float64)
        mean, std = Evaluator._channel_stats(values, weights)
        self.assertAlmostEqual(mean, np.mean([0, 1, 4, 5, 8, 9]))
        self.assertAlmostEqual(std, np.std([0, 1, 4, 5, 8, 9]))
        self.assertEqual(Evaluator._channel_stats(values), (float(values.mean()), float(values.std())))
        self.assertEqual(Evaluator._std_ratio(2.0, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
