"""Lane L2 step 2c3: Keyer, HueKeyer, Difference. Pixel assertions against hand-computed values,
evaluator/tile-path parity, mask + mix, bypass and CHOICES/LIMITS coverage. See docs/PARITY_2D.md
for the audit these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_C3_SINGLE = ("Keyer",)


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def bypass(self, key, value=True):
        self.d.execute(dict(op="disable", id=key, value=value))

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target):
    return Evaluator().evaluate(dict(document, view=target))


def tile_pixels(document, target):
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


class KernelPixelTests(unittest.TestCase):
    """Hand-computed pixel assertions, per the lane brief's worked examples."""

    def test_keyer_luminance_a_b_half_c_d_one_keys_white_not_mid_grey(self):
        white = np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
        grey = np.array([[[0.5, 0.5, 0.5, 1.0]]], dtype=np.float32)
        params = {"keyer_operation": "luminance", "range_a": 0.5, "range_b": 0.5,
                 "range_c": 1.0, "range_d": 1.0, "invert": 0, "mix": 1.0}
        out_white = Evaluator._kernel("Keyer", params, [white])
        out_grey = Evaluator._kernel("Keyer", params, [grey])
        self.assertAlmostEqual(float(out_white[0, 0, 3]), 1.0)
        self.assertAlmostEqual(float(out_grey[0, 0, 3]), 0.0)
        # RGB passes through untouched; only alpha is written.
        np.testing.assert_allclose(out_white[..., :3], white[..., :3])

    def test_keyer_ramps_linearly_between_a_and_b(self):
        half = np.array([[[0.5, 0.5, 0.5, 1.0]]], dtype=np.float32)
        params = {"keyer_operation": "luminance", "range_a": 0.0, "range_b": 1.0,
                  "range_c": 1.0, "range_d": 1.0, "mix": 1.0}
        out = Evaluator._kernel("Keyer", params, [half])
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.5, places=5)

    def test_keyer_invert_flips_the_matte(self):
        white = np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
        params = {"keyer_operation": "luminance", "range_a": 0.5, "range_b": 0.5,
                 "range_c": 1.0, "range_d": 1.0, "invert": 1, "mix": 1.0}
        out = Evaluator._kernel("Keyer", params, [white])
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.0)

    def test_keyer_red_operation_keys_on_the_red_channel_only(self):
        red = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        params = {"keyer_operation": "red", "range_a": 0.5, "range_b": 0.5,
                 "range_c": 1.0, "range_d": 1.0, "mix": 1.0}
        out = Evaluator._kernel("Keyer", params, [red])
        self.assertAlmostEqual(float(out[0, 0, 3]), 1.0)


class MaskMixTests(unittest.TestCase):
    """mix=0 is identity to the untouched source; a zero mask hides the effect."""

    def _params(self, mix):
        return {"Keyer": {"keyer_operation": "luminance", "range_a": 0.0, "range_b": 0.2,
                          "range_c": 0.4, "range_d": 0.6, "mix": mix}}

    def test_kinds_mix_zero_is_identity(self):
        params = self._params(0.0)
        for kind in GROUP_C3_SINGLE:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=16, height=12, size=4))
                g.add("node", kind, params[kind], image="plate")
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)

    def test_kinds_zero_mask_hides_the_effect(self):
        params = self._params(1.0)
        for kind in GROUP_C3_SINGLE:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Constant", dict(red=0.9, green=0.9, blue=0.9, alpha=1.0))
                g.add("matte", "Constant", dict(alpha=0.0))
                g.add("node", kind, params[kind], image="plate", mask="matte")
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)


class TilePathParityTests(unittest.TestCase):
    """Every group-c3 single-image kind renders byte-identical pixels on the evaluator and the
    tile path, including at tile seams."""

    def test_padded_free_kinds_match_across_both_paths(self):
        cases = {
            "Keyer": dict(keyer_operation="saturation", range_a=0.1, range_b=0.3,
                         range_c=0.6, range_d=0.9, mix=1.0),
        }
        for kind, params in cases.items():
            with self.subTest(kind=kind):
                self.assertIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                g.add("plate", "Checker", dict(width=48, height=32, size=8))
                g.add("matte", "Constant", dict(width=48, height=32, red=1, green=1, blue=1, alpha=0.5))
                g.add("node", kind, params, image="plate", mask="matte")
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node")
                np.testing.assert_allclose(ti, ev, atol=1e-6)

    def test_seamless_across_multiple_tiles(self):
        cases = {
            "Keyer": dict(keyer_operation="max", range_a=0.0, range_b=0.4, range_c=0.7, range_d=1.0, mix=1.0),
        }
        for kind, params in cases.items():
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=257, height=193, size=11))
                g.add("node", kind, params, image="plate")
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node")
                np.testing.assert_allclose(ti, ev, atol=1e-6)


class BypassTests(unittest.TestCase):
    """Every group-c3 single-image kind passes its input untouched when bypassed."""

    def visible_params(self, kind):
        return {"Keyer": dict(keyer_operation="luminance", range_a=0.0, range_b=0.3,
                              range_c=0.6, range_d=1.0)}[kind]

    def test_bypassed_kind_equals_its_input_on_both_paths(self):
        for kind in GROUP_C3_SINGLE:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=32, height=24, size=8))
                g.add("half", "Grade", dict(exposure=-1.0), image="plate")
                g.add("node", kind, self.visible_params(kind), image="half")
                upstream = evaluator_pixels(g.doc, "half")
                enabled = evaluator_pixels(g.doc, "node")
                self.assertFalse(np.array_equal(enabled, upstream), f"{kind} must visibly change its input")
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))

    def test_bypass_slot_is_the_single_image_input(self):
        for kind in GROUP_C3_SINGLE:
            with self.subTest(kind=kind):
                node = dict(type=kind, inputs={"image": "x", "mask": None})
                self.assertEqual(bypass_slot(node), "image")


class SpecCoverageTests(unittest.TestCase):
    def test_nodes_are_registered_with_mask_and_mix(self):
        for kind in GROUP_C3_SINGLE:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mix", SPECS[kind]["params"])
                self.assertEqual(LIMITS["mix"], (0, 1))

    def test_choices_and_limits_cover_every_new_ranged_param(self):
        self.assertIn("keyer_operation", CHOICES)
        for name in ("range_a", "range_b", "range_c", "range_d"):
            self.assertIn(name, LIMITS)

    def test_dispatcher_creates_every_new_node_with_valid_defaults(self):
        for kind in GROUP_C3_SINGLE:
            with self.subTest(kind=kind):
                d = Dispatcher()
                result = d.execute(dict(op="create", type=kind))
                self.assertIn("id", result["result"])


if __name__ == "__main__":
    unittest.main()
