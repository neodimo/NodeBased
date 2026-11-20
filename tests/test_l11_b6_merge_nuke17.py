"""Lane 11 step B6: Merge's blend modes match Nuke 17's own render, with the guide's formulas kept behind `legacy_merge_blend`.

The expected values are Nuke 17.0v6 Non-commercial's output, read on 11/20 from a Merge2 of two Read nodes (A along x, B along y,
values -0.5 to 12, A's alpha 0.5, B's 0.8) written to a 32-bit float EXR. A few rows of that grid are quoted here as numbers.
"""
import json
import unittest

import numpy as np

from nodebased import core
from nodebased.imaging import Evaluator

V = np.array([-0.5, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.5, 2, 4, 12], np.float32)


def _grid():
    n = len(V)
    a = np.zeros((n, n, 4), np.float32)
    b = np.zeros((n, n, 4), np.float32)
    a[..., :3] = V[None, :, None]
    b[..., :3] = V[:, None, None]
    a[..., 3] = 0.5
    b[..., 3] = 0.8
    return a, b


# Nuke's red channel: {operation: {row of B's value: the row across A's values}}.
NUKE = {
    "overlay": {0.25: [0, 0, 0.05, 0.125, 0.25, 0.375, 0.45, 0.5, 0.75, 1, 2, 6], 0.75: [0.5, 0.5, 0.55, 0.625, 0.75, 0.875, 0.95, 1, 1, 1, 1, 1],
                12: [1] * 12, -0.5: [0] * 12},
    "hard-light": {12: [0, 0, 2.4, 6, 1, 1, 1, 1, 1, 1, 1, 1], 0.1: [0, 0, 0.02, 0.05, 0.1, 0.55, 0.82, 1, 1, 1, 1, 1]},
    "soft-light": {12: [144, 144, 2.4, 6, 12, 18, 21.6, 24, 36, 48, 96, 288], 0.5: [0.25, 0.25, 0.3375, 0.46875, 0.6875, 0.90625, 1.0375, 1.125,
                                                                               1.5625, 2, 4, 12]},
    "color-dodge": {0.1: [0.1, 0.1, 0.11111, 0.13333, 0.2, 0.4, 1, 1, 1, 1, 1, 1], 1.5: [1] * 12},
    "color-burn": {12: [0, 0, 111, 45, 23, 15.6667, 13.2222, 12, 8.3333, 6.5, 3.75, 1.9167], -0.5: [0] * 8 + [0.3333, 0.5, 0.75, 0.9167]},
    "exclusion": {-0.5: [-1, -0.5, -0.4, -0.25, 0, 0.25, 0.4, 0.5, 1, 1.5, 3.5, 11.5], 12: [11.5, 12, 9.7, 6.25, 0.5, -5.25, -8.7, -11, -22.5, -34, -80, -264]},
}


class MergeMatchesNuke17Tests(unittest.TestCase):
    def test_rows_of_nukes_render(self):
        a, b = _grid()
        for op, rows in NUKE.items():
            out = Evaluator._merge_op_nuke17(op, a, b)
            for bv, row in rows.items():
                r = int(np.flatnonzero(V == np.float32(bv))[0])
                with self.subTest(op=op, b=bv):
                    np.testing.assert_allclose(out[r, :, 0], row, atol=1e-3)

    def test_overlay_of_twelve_and_twelve_is_one(self):
        a = np.full((1, 1, 4), 12, np.float32)
        self.assertEqual(float(Evaluator._merge_op_nuke17("overlay", a, a)[0, 0, 0]), 1.0)
        self.assertEqual(float(Evaluator._merge_op("overlay", a, a)[0, 0, 0]), -241.0)     # the guide's formula, kept as legacy

    def test_matte_alpha_is_overs(self):
        a, b = _grid()
        out = Evaluator._merge_op_nuke17("matte", a, b)
        np.testing.assert_allclose(out[..., 3], 0.9, atol=1e-6)
        np.testing.assert_allclose(out[..., :3], a[..., :3] * 0.5 + b[..., :3] * 0.5, atol=1e-6)

    def test_other_operations_are_unchanged(self):
        a, b = _grid()
        for op in ("over", "screen", "multiply", "difference", "conjoint-over", "geometric", "divide", "plus"):
            with self.subTest(op=op):
                np.testing.assert_array_equal(Evaluator._merge_op_nuke17(op, a, b), Evaluator._merge_op(op, a, b))


class MergeNodeFlagTests(unittest.TestCase):
    def _merge(self, **params):
        a, b = _grid()
        p = dict(core.SPECS["Merge"]["params"], operation="overlay", **params)
        return Evaluator._merge_node(p, a, b), a, b

    def test_a_new_merge_follows_nuke_and_the_legacy_flag_the_guide(self):
        self.assertEqual(core.SPECS["Merge"]["params"]["legacy_merge_blend"], 0)
        new, a, b = self._merge()
        np.testing.assert_array_equal(new, Evaluator._merge_op_nuke17("overlay", a, b))
        old, a, b = self._merge(legacy_merge_blend=1)
        np.testing.assert_array_equal(old, Evaluator._merge_op("overlay", a, b))

    def test_the_channel_knobs_path_uses_the_same_choice(self):
        new, a, b = self._merge(sRGB=0, Bchannels="rgb")
        self.assertEqual(float(new[11, 11, 0]), 1.0)

    def test_a_saved_merge_loads_with_the_guides_formulas(self):
        doc = core.empty_document()
        doc["nodes"]["m"] = {"type": "Merge", "params": dict(core.SPECS["Merge"]["params"], operation="overlay"),
                             "inputs": {"A": None, "B": None, "mask": None}}
        saved = json.loads(json.dumps(doc))
        del saved["nodes"]["m"]["params"]["legacy_merge_blend"]          # a Merge saved before this step
        upgraded = core.upgrade_document(saved)
        self.assertEqual(upgraded["nodes"]["m"]["params"]["legacy_merge_blend"], 1)
        fresh = core.upgrade_document(json.loads(json.dumps(doc)))
        self.assertEqual(fresh["nodes"]["m"]["params"]["legacy_merge_blend"], 0)

if __name__ == "__main__":
    unittest.main()
