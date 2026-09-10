"""Phase A: full merge operation set, real Transform with filters, Premult/Unpremult.

Numerical coverage for every new op/filter, including HDR (>1) and negative values,
and the v0.3.0 -> v3 document upgrade path.
"""
import copy
import unittest
import numpy as np

from nodebased.core import (CHOICES, SCHEMA_VERSION, Dispatcher, LIMITS, SPECS, empty_document,
                             upgrade_document, validate)
from nodebased.imaging import Evaluator


def _op(op, a, b, **mix):
    p = {"operation": op, "mix": mix.get("mix", 1.0)}
    return Evaluator._kernel("Merge", p, [a, b])


class MergeOperationTests(unittest.TestCase):
    """Each operation is verified against its Nuke-documented premultiplied formula."""

    def _assert_formula(self, op, result, expected, **kwargs):
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7,
                                    err_msg=f"{op}: {kwargs}")

    def test_over_standard_compositing(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 1]]], np.float32)
        self._assert_formula("over", _op("over", a, b), [[[0.5, 0, 0.5, 1]]])

    def test_under_swaps_role_of_alphas(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 1]]], np.float32)
        # under = b + a*(1 - b.a); with b.a=1, the contribution of A vanishes.
        self._assert_formula("under", _op("under", a, b), b)

    def test_plus_carries_hdr_and_negative(self):
        a = np.array([[[1.5, -2.0, 0.5, 0.7]]], np.float32)
        b = np.array([[[0.4, 3.0, -1.0, 0.3]]], np.float32)
        self._assert_formula("plus", _op("plus", a, b), a + b)

    def test_minus_keeps_negative_rgb(self):
        a = np.array([[[1.0, 0.0, 3.0, 0.3]]], np.float32)  # alpha 0.3
        b = np.array([[[0.4, 0.2, 2.5, 0.7]]], np.float32)  # alpha 0.7 — subtract yields -0.4
        self._assert_formula("minus", _op("minus", a, b), a - b)
        # Premultiplied minus can yield negative alpha when foreground alpha < background alpha.
        self.assertTrue((_op("minus", a, b)[..., 3] < 0).any())

    def test_multiply_keys_off_both_channels(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 1]]], np.float32)
        self._assert_formula("multiply", _op("multiply", a, b), [[[0, 0, 0, 0.5]]])

    def test_screen_one_minus_product(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 1]]], np.float32)
        self._assert_formula("screen", _op("screen", a, b), [[[0.5, 0, 1, 1]]])

    def test_max_uses_alpha_comparison(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 0.3]]], np.float32)
        self._assert_formula("max", _op("max", a, b), [[[0.5, 0, 1, 0.5]]])

    def test_min_uses_alpha_comparison(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 0.3]]], np.float32)
        self._assert_formula("min", _op("min", a, b), [[[0, 0, 0, 0.3]]])

    def test_difference_is_absolute(self):
        a = np.array([[[1.0, -2.0, 3.0, 0.7]]], np.float32)
        b = np.array([[[0.4, 1.0, 5.0, 0.3]]], np.float32)
        self._assert_formula("difference", _op("difference", a, b), np.abs(a - b))

    def test_divide_guards_against_zero(self):
        a = np.array([[[1.0, 0.5, 0.25, 0.5]]], np.float32)
        b = np.array([[[0.0, 0.0, 2.0, 2.0]]], np.float32)
        result = _op("divide", a, b)
        # Where |B| == 0, output is exactly 0 (no NaN/inf).
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[0, 0, :2], np.zeros((2,), np.float32))
        # Where |B| > 0, normal division.
        np.testing.assert_allclose(result[0, 0, 2:], [0.125, 0.25], rtol=1e-6)

    def test_mask_modulates_b_with_a_alpha(self):
        a = np.array([[[0.5, 0, 0, 0.4]]], np.float32)  # alpha 0.4
        b = np.array([[[1, 1, 1, 1]]], np.float32)
        self._assert_formula("mask", _op("mask", a, b), b * 0.4)

    def test_stencil_is_inverse_mask(self):
        a = np.array([[[0.5, 0, 0, 0.4]]], np.float32)
        b = np.array([[[1, 1, 1, 1]]], np.float32)
        self._assert_formula("stencil", _op("stencil", a, b), b * 0.6)

    def test_in_clips_a_with_b_alpha(self):
        a = np.array([[[1, 1, 1, 0.7]]], np.float32)
        b = np.array([[[0, 0, 0, 0.3]]], np.float32)
        self._assert_formula("in", _op("in", a, b), a * 0.3)

    def test_out_is_inverse_of_in(self):
        a = np.array([[[1, 1, 1, 0.7]]], np.float32)
        b = np.array([[[0, 0, 0, 0.3]]], np.float32)
        self._assert_formula("out", _op("out", a, b), a * 0.7)

    def test_atop_inside_b_outside_a(self):
        a = np.array([[[1, 1, 1, 0.7]]], np.float32)
        b = np.array([[[0.4, 0.4, 0.4, 0.5]]], np.float32)
        self._assert_formula("atop", _op("atop", a, b), a * 0.5 + b * 0.3)

    def test_xor_is_symmetric_exclusion(self):
        a = np.array([[[1, 1, 1, 0.7]]], np.float32)
        b = np.array([[[0.4, 0.4, 0.4, 0.5]]], np.float32)
        self._assert_formula("xor", _op("xor", a, b), a * 0.5 + b * 0.3)

    def test_mix_zero_returns_b_for_every_op(self):
        rng = np.random.RandomState(42)
        a = rng.rand(4, 4, 4).astype(np.float32)
        b = rng.rand(4, 4, 4).astype(np.float32)
        # except over, where v0.3.0 semantics reduce to b anyway (a*0 + b*(1-0) = b)
        for op in CHOICES["operation"]:
            out = Evaluator._kernel("Merge", {"operation": op, "mix": 0.0}, [a, b])
            np.testing.assert_allclose(out, b, atol=1e-6, err_msg=f"mix=0 {op}")

    def test_mix_one_returns_full_op_for_every_op(self):
        rng = np.random.RandomState(7)
        a = rng.rand(4, 4, 4).astype(np.float32) + 0.5  # avoid degenerate alpha=0
        b = rng.rand(4, 4, 4).astype(np.float32) + 0.5
        for op in CHOICES["operation"]:
            full = Evaluator._merge_op(op, a, b)
            out = Evaluator._kernel("Merge", {"operation": op, "mix": 1.0}, [a, b])
            np.testing.assert_allclose(out, full, atol=1e-5, err_msg=f"mix=1 {op}")

    def test_hdr_values_pass_through_plus_minus_screen(self):
        a = np.full((1, 1, 4), 2.5, np.float32)  # premultiplied HDR
        b = np.full((1, 1, 4), 3.0, np.float32)
        a[0, 0, 3] = b[0, 0, 3] = 1.5  # both alphas > 1
        np.testing.assert_allclose(_op("plus", a, b), a + b)
        np.testing.assert_allclose(_op("minus", a, b), a - b)
        np.testing.assert_allclose(_op("screen", a, b), a + b - a * b)
        self.assertTrue((_op("plus", a, b) > 1.0).all())

    def test_negative_premultiplied_inputs(self):
        a = np.array([[[0.5, -0.5, 0.2, 0.6]]], np.float32)
        b = np.array([[[-0.3, 0.4, 0.1, 0.4]]], np.float32)
        # No operation should ever produce NaN/inf for finite premultiplied inputs.
        for op in CHOICES["operation"]:
            with self.subTest(op=op):
                result = _op(op, a, b)
                self.assertTrue(np.isfinite(result).all(), msg=op)


class TransformTests(unittest.TestCase):
    """Real 2D transform with translate/rotate/scale and three filters."""

    PARAMS = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
              "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest"}

    def test_identity_all_filters_reproduce_source(self):
        # Gradient + diagonal so sub-pixel filters are exercised even at integer sample points.
        idx = np.indices((4, 6), dtype=np.float32).transpose(1, 2, 0).astype(np.float32)
        alpha = np.full((4, 6, 1), 0.7, np.float32)
        src = np.concatenate([idx, np.zeros_like(idx[..., :1]), alpha], axis=2)
        for f in ("nearest", "bilinear", "cubic"):
            with self.subTest(filter=f):
                p = {**self.PARAMS, "filter": f}
                out = Evaluator._kernel("Transform", p, [src])
                np.testing.assert_array_equal(out, src)

    def test_integer_translate_matches_v030_no_wrap(self):
        src = np.ones((2, 3, 4), np.float32)
        out = Evaluator._kernel("Transform", {**self.PARAMS, "translate_x": 1.0, "translate_y": -1.0}, [src])
        self.assertEqual(out[:, 0].sum(), 0)
        self.assertEqual(out[1].sum(), 0)
        self.assertEqual(out[0, 1:].sum(), 8)

    def test_fractional_translate_uses_filter(self):
        src = np.zeros((1, 4, 4), np.float32)
        src[0, :, 0] = [1, 0, 0, 0]
        # Sub-pixel shift by 0.5: nearest snaps to the first 1.0 column at dest[0]; bilinear blends.
        nearest = Evaluator._kernel("Transform", {**self.PARAMS, "translate_x": 0.5}, [src])
        bilinear = Evaluator._kernel("Transform", {**self.PARAMS, "translate_x": 0.5, "filter": "bilinear"}, [src])
        # Bilinear at dest 0 reads between src columns 0 and 1 (values 1 and 0) => 0.5.
        np.testing.assert_allclose(bilinear[0, 0, 0], 0.5, atol=1e-6)
        # Nearest at dest 0 with shift 0.5 reads src column 0 => 1.0.
        np.testing.assert_allclose(nearest[0, 0, 0], 1.0, atol=1e-6)
        # Cubic is continuous with bilinear at integer sample points, sharper than bilinear otherwise.
        cubic = Evaluator._kernel("Transform", {**self.PARAMS, "translate_x": 0.5, "filter": "cubic"}, [src])
        self.assertTrue(np.isfinite(cubic).all())

    def test_rotation_swaps_axes(self):
        src = np.zeros((3, 3, 4), np.float32)
        src[0, 0, 0] = 1  # mark the top-left pixel
        # Rotate 90° about the canvas centre.
        center = (1.5, 1.5)
        out = Evaluator._kernel("Transform", {**self.PARAMS, "rotate": 90.0,
                                              "center_x": center[0], "center_y": center[1]}, [src])
        # The red pixel should now be at the top-right after a 90° clockwise rotation (in Nuke's
        # screen-coordinate convention, where +y is down so a +90° rotation sends (0,0) -> (h, 0)).
        self.assertGreater(out[0, -1, 0], 0.5)
        self.assertLess(out[0, 0, 0], 0.5)

    def test_scale_halves_and_doubles(self):
        src = np.zeros((2, 2, 4), np.float32)
        src[0, 0, 0] = 1
        # Scale 2x: the source pixel still falls inside the same canvas size, but its energy spreads.
        out = Evaluator._kernel("Transform", {**self.PARAMS, "scale": 2.0}, [src])
        self.assertEqual(out.shape, (2, 2, 4))
        self.assertGreater(out.sum(), 0)
        # Scale 0.5 with center at the corner: most of the source falls outside the canvas.
        out_half = Evaluator._kernel(
            "Transform", {**self.PARAMS, "scale": 0.5, "center_x": 0.0, "center_y": 0.0}, [src])
        self.assertTrue(np.isfinite(out_half).all())
        self.assertEqual(out_half.shape, (2, 2, 4))

    def test_out_of_bounds_is_transparent_black(self):
        src = np.ones((2, 2, 4), np.float32)
        # A large translate pushes everything outside the destination canvas.
        out = Evaluator._kernel("Transform", {**self.PARAMS, "translate_x": 100.0}, [src])
        np.testing.assert_array_equal(out, np.zeros_like(src))

    def test_negative_translate_uses_correct_side(self):
        src = np.ones((2, 2, 4), np.float32)
        # translate (-1, -1) shifts the source up-and-left; only src (1, 1) lands inside the canvas,
        # at destination (0, 0).
        out = Evaluator._kernel("Transform",
                                  {**self.PARAMS, "translate_x": -1.0, "translate_y": -1.0}, [src])
        np.testing.assert_array_equal(out[0, 0], src[1, 1])
        np.testing.assert_array_equal(out[0, 1], np.zeros((4,), np.float32))
        np.testing.assert_array_equal(out[1, 0], np.zeros((4,), np.float32))
        np.testing.assert_array_equal(out[1, 1], np.zeros((4,), np.float32))

    def test_hdr_and_negative_values_preserved(self):
        src = np.full((3, 3, 4), -1.0, np.float32)
        src[..., 3] = 1.0  # full alpha
        src[0, 0] = [3.0, 0.5, -2.0, 1.0]  # HDR + negative premultiplied
        # Identity transform should pass everything through bit-exact for nearest.
        out = Evaluator._kernel("Transform", {**self.PARAMS}, [src])
        np.testing.assert_array_equal(out, src)
        # Rotation must not produce NaN/inf for HDR inputs.
        out = Evaluator._kernel("Transform", {**self.PARAMS, "rotate": 45.0,
                                              "center_x": 1.0, "center_y": 1.0}, [src])
        self.assertTrue(np.isfinite(out).all())

    def test_filters_differ_on_a_subpixel_edge(self):
        # A 2x2 impulse at the source corner, sampled with a half-pixel translate distinguishes
        # the three filters' smoothing characteristics.
        src = np.zeros((4, 4, 4), np.float32)
        src[1:3, 1:3] = 1.0
        src[..., 3] = 1.0
        params = {**self.PARAMS, "translate_x": 0.25}
        nearest = Evaluator._kernel("Transform", params, [src])
        bilinear = Evaluator._kernel("Transform", {**params, "filter": "bilinear"}, [src])
        cubic = Evaluator._kernel("Transform", {**params, "filter": "cubic"}, [src])
        # All three filters produce finite output.
        self.assertTrue(np.isfinite(bilinear).all())
        self.assertTrue(np.isfinite(cubic).all())
        self.assertTrue(np.isfinite(nearest).all())
        # The bilinear output must contain a fractional value somewhere (i.e. > 0 and < 1),
        # proving it's actually interpolating rather than snapping like nearest does.
        interior = (bilinear[..., 0] > 0.05) & (bilinear[..., 0] < 0.95)
        self.assertTrue(interior.any())


class PremultUnpremultTests(unittest.TestCase):

    def test_premult_scales_rgb_by_alpha(self):
        src = np.array([[[2.0, 3.0, 4.0, 0.5]]], np.float32)
        out = Evaluator._kernel("Premult", {}, [src])
        np.testing.assert_allclose(out, [[[1.0, 1.5, 2.0, 0.5]]])

    def test_unpremult_recovers_straight_color(self):
        premultiplied = np.array([[[1.0, 1.5, 2.0, 0.5]]], np.float32)
        out = Evaluator._kernel("Unpremult", {}, [premultiplied])
        np.testing.assert_allclose(out, [[[2.0, 3.0, 4.0, 0.5]]])

    def test_premult_unpremult_round_trip_for_alpha_above_zero(self):
        rng = np.random.RandomState(2025)
        straight = rng.rand(8, 8, 3).astype(np.float32) * 2 - 0.5  # includes negative
        alpha = np.clip(rng.rand(8, 8, 1).astype(np.float32), 0.1, 1.0)
        premultiplied = np.concatenate([straight * alpha, alpha], axis=2).astype(np.float32)
        recovered = Evaluator._kernel("Unpremult", {}, [premultiplied])
        round_trip = Evaluator._kernel("Premult", {}, [recovered])
        np.testing.assert_allclose(round_trip, premultiplied, rtol=1e-6, atol=1e-7)

    def test_unpremult_premult_round_trip_is_identity(self):
        # Any image (including premultiplied) → Unpremult → Premult reproduces it.
        rng = np.random.RandomState(11)
        src = rng.rand(5, 5, 4).astype(np.float32)
        out = Evaluator._kernel("Premult", {}, [Evaluator._kernel("Unpremult", {}, [src])])
        np.testing.assert_allclose(out, src, rtol=1e-6, atol=1e-7)

    def test_alpha_zero_is_safe_in_both_directions(self):
        # Premult: RGB -> 0, alpha -> 0; no NaN/inf.
        prem_in = np.array([[[1.0, -2.0, 3.0, 0.0]]], np.float32)
        out = Evaluator._kernel("Premult", {}, [prem_in])
        np.testing.assert_array_equal(out, np.zeros_like(prem_in))
        self.assertTrue(np.isfinite(out).all())
        # Unpremult: RGB left untouched, alpha preserved; no NaN/inf.
        unprem_in = np.array([[[1.0, -2.0, 3.0, 0.0]]], np.float32)
        out = Evaluator._kernel("Unpremult", {}, [unprem_in])
        np.testing.assert_array_equal(out, unprem_in)
        self.assertTrue(np.isfinite(out).all())

    def test_hdr_and_negative_round_trip(self):
        straight = np.array([[[5.0, -3.0, 0.5], [0.0, 2.0, -1.0]]], np.float32)
        alpha = np.array([[[0.8], [1.5]]], np.float32)  # alpha > 1 in HDR space
        premultiplied = np.concatenate([straight * alpha, alpha], axis=2).astype(np.float32)
        round_trip = Evaluator._kernel("Premult", {},
                                         [Evaluator._kernel("Unpremult", {}, [premultiplied])])
        np.testing.assert_allclose(round_trip, premultiplied, rtol=1e-6, atol=1e-7)


class DocumentUpgradeTests(unittest.TestCase):
    """v0.3.0 (.nbcomp version=2) documents load into v3 with the right defaults."""

    def test_v2_transform_upgrades_to_translate_rotate_scale(self):
        # Real v0.3.0 documents always wire Transform's "image" input to a source node; the upgrade
        # preserves that wiring verbatim while renaming the integer x/y params to the float form.
        old = {"version": 2, "view": "t", "nodes": {
            "src": {"type": "Constant", "name": "Src", "pos": [0, 0], "disabled": False,
                    "inputs": {}, "params": {"width": 2, "height": 2,
                                              "red": 0.5, "green": 0.5, "blue": 0.5, "alpha": 1.0}},
            "t": {"type": "Transform", "name": "T", "pos": [0, 0], "disabled": False,
                  "inputs": {"image": "src"}, "params": {"x": 12, "y": -7}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        # Input wiring is preserved as-is by the upgrade (the upgrade does not invent edges).
        self.assertEqual(upgraded["nodes"]["t"]["inputs"]["image"], "src")
        params = upgraded["nodes"]["t"]["params"]
        self.assertEqual(params["translate_x"], 12.0)
        self.assertEqual(params["translate_y"], -7.0)
        self.assertEqual(params["rotate"], 0.0)
        self.assertEqual(params["scale"], 1.0)
        self.assertEqual(params["center_x"], 0.0)
        self.assertEqual(params["center_y"], 0.0)
        self.assertEqual(params["filter"], "nearest")

    def test_v2_merge_gains_operation_default(self):
        old = {"version": 2, "view": "m", "nodes": {
            "fa": {"type": "Constant", "name": "Fg", "pos": [0, 0], "disabled": False,
                   "inputs": {}, "params": {"width": 2, "height": 2,
                                             "red": 0.0, "green": 0.0, "blue": 0.0, "alpha": 1.0}},
            "bg": {"type": "Constant", "name": "Bg", "pos": [0, 0], "disabled": False,
                   "inputs": {}, "params": {"width": 2, "height": 2,
                                             "red": 1.0, "green": 0.5, "blue": 0.0, "alpha": 1.0}},
            "m": {"type": "Merge", "name": "M", "pos": [0, 0], "disabled": False,
                  "inputs": {"A": "fa", "B": "bg"}, "params": {"mix": 0.6}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["nodes"]["m"]["inputs"], {"A": "fa", "B": "bg"})
        self.assertEqual(upgraded["nodes"]["m"]["params"]["operation"], "over")
        self.assertEqual(upgraded["nodes"]["m"]["params"]["mix"], 0.6)

    def test_v1_chained_upgrade_reaches_v3_with_all_defaults(self):
        old = {"version": 1, "view": "r", "nodes": {
            "r": {"type": "Read", "name": "R", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {"path": "/tmp/a.png"}},
            "t": {"type": "Transform", "name": "T", "pos": [0, 0], "disabled": False,
                  "inputs": {"image": "r"}, "params": {"x": 0, "y": 0}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        # The chain now reaches v4 because Phase B added the v3 -> v4 step that injects the
        # optional `mask` input and `mix` param on image-filter nodes. The v1 -> v2 -> v3 history
        # of the chain is still verified by the per-step checks elsewhere; this just confirms the
        # final landing version.
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["nodes"]["r"]["params"]["colorspace"], "Auto")
        self.assertEqual(upgraded["nodes"]["t"]["params"]["translate_x"], 0.0)
        self.assertEqual(upgraded["nodes"]["t"]["params"]["filter"], "nearest")
        # v3 -> v4 step also adds the optional mask input slot and mix=1.0 default.
        self.assertIn("mask", upgraded["nodes"]["t"]["inputs"])
        self.assertEqual(upgraded["nodes"]["t"]["params"]["mix"], 1.0)

    def test_v2_transform_upgrade_renders_identically_to_v030(self):
        # A v2 integer-translate (x=1, y=-1) upgraded to v3 must produce the same pixels as the
        # v0.3.0 forward-mapping integer-translate kernel did on the same input.
        from nodebased.core import demo_document
        d = Dispatcher(demo_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "shifted", "type": "Constant", "name": "Plate",
             "params": {"width": 4, "height": 4, "red": 0.8, "green": 0.6, "blue": 0.4, "alpha": 1.0}},
            {"op": "create", "id": "xform", "type": "Transform",
             "params": {"translate_x": 1.0, "translate_y": -1.0}},
            {"op": "connect", "id": "xform", "input": "image", "source": "shifted"},
            {"op": "create", "id": "shifted_view", "type": "Viewer"},
            {"op": "connect", "id": "shifted_view", "input": "image", "source": "xform"},
            {"op": "view", "id": "shifted_view"}]})
        out = Evaluator().evaluate(d.document)
        # translate (+1, -1): dest(dy, dx) reads src(dy + 1, dx - 1) — matches the v0.3.0 forward-
        # mapping kernel, which scanned `frame[top:bottom, left:right] = src[top - y:bottom - y, left - x:right - x]`.
        # Col 0 is always OOB (dx - 1 == -1), row 3 is always OOB (dy + 1 == 4).
        expected = np.zeros((4, 4, 4), np.float32)
        src_val = np.array([0.8, 0.6, 0.4, 1.0], np.float32)
        for row in (0, 1, 2):
            expected[row, 1] = src_val
            expected[row, 2] = src_val
            expected[row, 3] = src_val
        np.testing.assert_array_equal(out, expected)


class SpecAndChoicesTests(unittest.TestCase):
    """The agent/UI surface picks up new params automatically — SPECS/LIMITS/CHOICES must agree."""

    def test_merge_has_operation_and_mix(self):
        self.assertEqual(SPECS["Merge"]["params"]["operation"], "over")
        self.assertEqual(SPECS["Merge"]["params"]["mix"], 1.0)
        self.assertIn("operation", CHOICES)
        self.assertEqual(set(CHOICES["operation"]),
                         {"over", "under", "plus", "minus", "multiply", "screen",
                          "max", "min", "difference", "divide", "mask", "stencil",
                          "in", "out", "atop", "xor"})

    def test_transform_has_all_new_params_with_filter_choice(self):
        params = SPECS["Transform"]["params"]
        self.assertEqual(set(params),
                         {"translate_x", "translate_y", "rotate", "scale",
                          "center_x", "center_y", "filter", "mix"})
        self.assertEqual(params["filter"], "nearest")
        self.assertEqual(set(CHOICES["filter"]), {"nearest", "bilinear", "cubic"})
        for name in ("translate_x", "translate_y", "rotate", "scale", "center_x", "center_y", "mix"):
            self.assertIn(name, LIMITS)

    def test_premult_and_unpremult_are_single_input_nodes(self):
        self.assertEqual(SPECS["Premult"]["inputs"], ["image"])
        self.assertEqual(SPECS["Unpremult"]["inputs"], ["image"])
        self.assertEqual(SPECS["Premult"]["params"], {})
        self.assertEqual(SPECS["Unpremult"]["params"], {})

    def test_dispatcher_creates_premult_unpremult_with_valid_params(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "src", "type": "Constant",
             "params": {"width": 2, "height": 2, "alpha": 0.5}},
            {"op": "create", "id": "p", "type": "Premult"},
            {"op": "create", "id": "u", "type": "Unpremult"},
            {"op": "connect", "id": "p", "input": "image", "source": "src"},
            {"op": "connect", "id": "u", "input": "image", "source": "p"}]})
        self.assertEqual(d.document["nodes"]["p"]["type"], "Premult")
        self.assertEqual(d.document["nodes"]["u"]["type"], "Unpremult")


if __name__ == "__main__":
    unittest.main()
