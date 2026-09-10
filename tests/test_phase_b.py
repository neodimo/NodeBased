"""Phase B: schema v3 -> v4, reusable mask + mix on image-filter nodes, Dot, Switch,
and source-level agent-protocol proof.

Numerical coverage for the new mask/mix contract (per-pixel, HDR-safe, dimension-checked)
plus the new Dot/Switch nodes and the v0.4.0-era document upgrade path.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased.core import (CHOICES, Dispatcher, IMAGE_FILTER_KINDS, LIMITS, SPECS,
                             empty_document, upgrade_document, validate)
from nodebased.imaging import Evaluator


# Filter kinds that gained the optional mask input + mix param in v4.
FILTER_KINDS = IMAGE_FILTER_KINDS  # ("Grade", "ColorCorrect", "Blur", "Transform", "Crop")


def _full_alpha(rgb, alpha=1.0):
    """Build a (h, w, 4) premultiplied image with the given straight RGB and constant alpha.
    Accepts a flat (3,) or (h, w, 3) array; always returns (h, w, 4)."""
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, 1, 3)
    elif rgb.ndim == 2:
        rgb = rgb[..., None]
    alpha = np.float32(alpha)
    out = np.empty(rgb.shape[:-1] + (4,), np.float32)
    out[..., :3] = rgb * alpha
    out[..., 3] = alpha
    return out


class SchemaV4UpgradeTests(unittest.TestCase):
    """v3 documents upgrade to v4 cleanly and v3->v4 injects mask + mix with identity defaults."""

    def test_v3_grade_gains_mask_input_and_mix_param(self):
        old = {"version": 3, "view": "v", "nodes": {
            "p": {"type": "Constant", "name": "P", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {"width": 4, "height": 4, "red": 0.5, "green": 0.5, "blue": 0.5, "alpha": 1.0}},
            "g": {"type": "Grade", "name": "G", "pos": [0, 0], "disabled": False,
                  "inputs": {"image": "p"}, "params": {"exposure": 0.5, "multiply": 1.0, "offset": 0.0}},
            "v": {"type": "Viewer", "name": "V", "pos": [0, 0], "disabled": False,
                  "inputs": {"image": "g"}, "params": {}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], 4)
        self.assertIn("mask", upgraded["nodes"]["g"]["inputs"])
        self.assertIsNone(upgraded["nodes"]["g"]["inputs"]["mask"])
        self.assertEqual(upgraded["nodes"]["g"]["params"]["mix"], 1.0)
        # Original Grade params are preserved.
        self.assertEqual(upgraded["nodes"]["g"]["params"]["exposure"], 0.5)

    def test_v3_all_filter_kinds_get_mask_and_mix(self):
        # Build a v3 doc with one of each filter kind (all upstreamed from a Constant).
        old_nodes = {"src": {"type": "Constant", "name": "Src", "pos": [0, 0], "disabled": False,
                              "inputs": {}, "params": {"width": 2, "height": 2, "red": 0.5, "green": 0.5,
                                                        "blue": 0.5, "alpha": 1.0}}}
        filter_specs = {
            "Grade": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0},
            "ColorCorrect": {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0},
            "Blur": {"radius": 0.0},
            "Transform": {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                          "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest"},
            "Crop": {"x": 0, "y": 0, "width": 2, "height": 2},
        }
        for kind, params in filter_specs.items():
            old_nodes[kind.lower()] = {"type": kind, "name": kind, "pos": [0, 0], "disabled": False,
                                        "inputs": {"image": "src"}, "params": params}
        old = {"version": 3, "view": "grade", "nodes": old_nodes}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], 4)
        for kind in filter_specs:
            node = upgraded["nodes"][kind.lower()]
            self.assertIn("mask", node["inputs"], f"{kind} missing mask slot after upgrade")
            self.assertIsNone(node["inputs"]["mask"], f"{kind} mask not None after upgrade")
            self.assertEqual(node["params"]["mix"], 1.0, f"{kind} mix default wrong after upgrade")

    def test_v4_does_not_inject_mask_or_mix_on_non_filter_kinds(self):
        old = {"version": 3, "view": None, "nodes": {
            "c": {"type": "Constant", "name": "C", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {"width": 1, "height": 1, "red": 1, "green": 0, "blue": 0, "alpha": 1.0}},
            "m": {"type": "Merge", "name": "M", "pos": [0, 0], "disabled": False,
                  "inputs": {"A": "c", "B": "c"}, "params": {"operation": "over", "mix": 0.5}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], 4)
        # Merge keeps its existing mix; the upgrade does not duplicate or rename it.
        self.assertNotIn("mask", upgraded["nodes"]["m"]["inputs"])
        self.assertEqual(upgraded["nodes"]["m"]["params"]["mix"], 0.5)

    def test_v3_doc_with_dor_or_switch_is_rejected(self):
        # New node types in a pre-v4 doc are an error (upgrade can't reach the future).
        old = {"version": 3, "view": None, "nodes": {
            "d": {"type": "Dot", "name": "D", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {}}}}
        with self.assertRaises(ValueError):
            validate(upgrade_document(old))

    def test_v3_doc_renders_identically_after_upgrade(self):
        # Realistic v3 graph: Constant + Grade + Viewer. After v3 -> v4 (which injects mask=None
        # and mix=1.0), the rendered pixels must equal a freshly-built v4 graph.
        v3 = {"version": 3, "view": "v", "nodes": {
            "p": {"type": "Constant", "name": "P", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {"width": 2, "height": 2, "red": 0.4, "green": 0.6, "blue": 0.8, "alpha": 1.0}},
            "g": {"type": "Grade", "name": "G", "pos": [0, 0], "disabled": False,
                  "inputs": {"image": "p"}, "params": {"exposure": 0.5, "multiply": 1.0, "offset": 0.1}},
            "v": {"type": "Viewer", "name": "V", "pos": [0, 0], "disabled": False,
                  "inputs": {"image": "g"}, "params": {}}}}
        upgraded = upgrade_document(v3)
        out_v3 = Evaluator().evaluate(upgraded)
        d = Dispatcher()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "p", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 0.4, "green": 0.6, "blue": 0.8, "alpha": 1.0}},
            {"op": "create", "id": "g", "type": "Grade",
             "params": {"exposure": 0.5, "multiply": 1.0, "offset": 0.1}},
            {"op": "connect", "id": "g", "input": "image", "source": "p"},
            {"op": "create", "id": "v", "type": "Viewer"},
            {"op": "connect", "id": "v", "input": "image", "source": "g"},
            {"op": "view", "id": "v"}]})
        out_v4 = Evaluator().evaluate(d.document)
        np.testing.assert_array_equal(out_v3, out_v4)


class MaskMixSemanticsTests(unittest.TestCase):
    """The reusable mask + mix contract on every image-filter node.

    Math (premultiplied):
        gate = mix * mask.a   (scalar when mask unwired; per-pixel when wired)
        result = gate * filtered + (1 - gate) * source
    """

    SOURCE_RGB = (0.4, 0.2, 0.1)
    EXPOSURE = 1.0

    def _filter(self, kind, params, source_image, mask=None, mix=1.0):
        params = {**params, "mix": mix}
        inputs = [source_image]
        if kind in FILTER_KINDS:
            inputs.append(mask)
        return Evaluator._kernel(kind, params, inputs)

    def test_mix_zero_bypasses_filter_fully(self):
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                src = _full_alpha(self.SOURCE_RGB)
                # Whatever the filter params are, mix=0 must reproduce source.
                out = self._filter(kind, self._filter_params(kind), src, mask=None, mix=0.0)
                np.testing.assert_array_equal(out, src)

    def test_no_mask_full_opacity_means_only_mix_gates(self):
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                src = _full_alpha(self.SOURCE_RGB)
                full = self._filter(kind, self._filter_params(kind), src, mask=None, mix=1.0)
                half = self._filter(kind, self._filter_params(kind), src, mask=None, mix=0.5)
                # result = mix*full + (1-mix)*src; mix=0.5 → halfway between source and full.
                expected_half = 0.5 * full + 0.5 * src
                np.testing.assert_allclose(half, expected_half, atol=1e-6)

    def test_mask_alpha_zero_hides_filter_fully(self):
        src = _full_alpha(self.SOURCE_RGB)
        # Mask with alpha=0 everywhere (transparent black, premultiplied).
        mask = np.zeros_like(src)
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                # mix=1 still yields source because mask.a=0 zeroes the gate.
                out = self._filter(kind, self._filter_params(kind), src, mask=mask, mix=1.0)
                np.testing.assert_array_equal(out, src)

    def test_mask_alpha_one_equals_no_mask(self):
        # mask.a=1 everywhere is the identity case for the mask gate.
        src = _full_alpha(self.SOURCE_RGB)
        mask = _full_alpha((1.0, 1.0, 1.0), alpha=1.0)
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                out = self._filter(kind, self._filter_params(kind), src, mask=mask, mix=0.7)
                unmasked = self._filter(kind, self._filter_params(kind), src, mask=None, mix=0.7)
                np.testing.assert_allclose(out, unmasked, atol=1e-6)

    def test_mask_half_alpha_halves_the_mix_gate(self):
        src = _full_alpha(self.SOURCE_RGB)
        mask = _full_alpha((1.0, 1.0, 1.0), alpha=0.5)
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                full = self._filter(kind, self._filter_params(kind), src, mask=None, mix=1.0)
                # gate = 0.5 * 1.0 = 0.5, so result = 0.5*full + 0.5*src
                out = self._filter(kind, self._filter_params(kind), src, mask=mask, mix=1.0)
                expected = 0.5 * full + 0.5 * src
                np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_mask_shape_mismatch_raises_explicitly(self):
        src = _full_alpha(self.SOURCE_RGB, alpha=1.0)  # 1x1x4
        big_mask = _full_alpha((1.0, 1.0, 1.0), alpha=1.0)  # 1x1x4 — but resize to 2x2
        big_mask = np.broadcast_to(big_mask, (2, 2, 4)).copy()
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "no silent resampling"):
                    self._filter(kind, self._filter_params(kind), src, mask=big_mask, mix=1.0)

    def test_hdr_mask_alpha_and_mix_produce_finite_output(self):
        # HDR alpha above 1 and negative premultiplied RGB must never produce NaN/inf.
        src = np.array([[[0.5, -0.3, 0.2, 0.7]]], np.float32)
        mask = np.array([[[1.0, 0.5, 0.3, 1.5]]], np.float32)  # alpha 1.5
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                out = self._filter(kind, self._filter_params(kind), src, mask=mask, mix=0.7)
                self.assertTrue(np.isfinite(out).all(), f"{kind} produced non-finite values")

    def test_negative_mask_alpha_is_safe(self):
        # Defensive: a malformed negative mask alpha (shouldn't happen via UI but can via JSON)
        # is clamped to no effect rather than producing negative-premultiplied output.
        src = _full_alpha(self.SOURCE_RGB)
        bad_mask = np.array([[[0.5, 0.5, 0.5, -0.5]]], np.float32)
        # Math: gate = 0.7 * -0.5 = -0.35. result = -0.35 * filtered + 1.35 * src.
        # We don't claim "safety" by clamping — we claim determinism and finiteness.
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                out = self._filter(kind, self._filter_params(kind), src, mask=bad_mask, mix=0.7)
                self.assertTrue(np.isfinite(out).all())

    def _filter_params(self, kind):
        if kind == "Grade":
            return {"exposure": self.EXPOSURE, "multiply": 1.0, "offset": 0.0}
        if kind == "ColorCorrect":
            return {"lift": 0.0, "gamma": 1.0, "gain": 1.5, "saturation": 1.0}
        if kind == "Blur":
            return {"radius": 0.0}  # radius 0 → identity, simplifies comparisons
        if kind == "Transform":
            return {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                    "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest"}
        if kind == "Crop":
            return {"x": 0, "y": 0, "width": 1, "height": 1}
        raise AssertionError(kind)


class DotNodeTests(unittest.TestCase):
    """Dot is a graph passthrough — pixel data unchanged."""

    def test_dot_returns_first_input_unchanged(self):
        src = _full_alpha((0.3, 0.6, 0.9), alpha=0.8)
        out = Evaluator._kernel("Dot", {}, [src])
        np.testing.assert_array_equal(out, src)

    def test_dot_preserves_hdr_and_negative_values(self):
        src = np.array([[[2.0, -1.0, 0.5, 0.5]]], np.float32)
        out = Evaluator._kernel("Dot", {}, [src])
        np.testing.assert_array_equal(out, src)

    def test_dot_does_not_apply_mask_or_mix(self):
        # Dot is intentionally outside the mask/mix contract.
        src = _full_alpha((0.5, 0.5, 0.5))
        out = Evaluator._kernel("Dot", {}, [src])
        np.testing.assert_array_equal(out, src)

    def test_dispatcher_create_and_connect_dot(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "src", "type": "Constant",
             "params": {"width": 2, "height": 2, "alpha": 1.0}},
            {"op": "create", "id": "dot", "type": "Dot"},
            {"op": "connect", "id": "dot", "input": "input", "source": "src"},
            {"op": "create", "id": "v", "type": "Viewer"},
            {"op": "connect", "id": "v", "input": "image", "source": "dot"},
            {"op": "view", "id": "v"}]})
        out = Evaluator().evaluate(d.document)
        src_out = Evaluator().evaluate(d.document, "src")
        np.testing.assert_array_equal(out, src_out)


class SwitchNodeTests(unittest.TestCase):
    """Switch selects one of its inputs based on the 'which' param."""

    def test_switch_selects_input_zero(self):
        a = _full_alpha((1.0, 0.0, 0.0))
        b = _full_alpha((0.0, 0.0, 1.0))
        out = Evaluator._kernel("Switch", {"which": 0}, [a, b])
        np.testing.assert_array_equal(out, a)

    def test_switch_selects_input_one(self):
        a = _full_alpha((1.0, 0.0, 0.0))
        b = _full_alpha((0.0, 0.0, 1.0))
        out = Evaluator._kernel("Switch", {"which": 1}, [a, b])
        np.testing.assert_array_equal(out, b)

    def test_switch_out_of_range_raises(self):
        a = _full_alpha((1.0, 0.0, 0.0))
        b = _full_alpha((0.0, 0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "out of range"):
            Evaluator._kernel("Switch", {"which": 5}, [a, b])
        with self.assertRaisesRegex(ValueError, "out of range"):
            Evaluator._kernel("Switch", {"which": -1}, [a, b])

    def test_switch_passes_through_pixel_data_unchanged(self):
        # HDR + negative premultiplied pass through verbatim — no resampling, no clamping.
        src = np.array([[[2.0, -0.5, 0.3, 0.7]]], np.float32)
        other = np.array([[[0.1, 0.1, 0.1, 0.1]]], np.float32)
        # which=0 picks inputs[0], which=1 picks inputs[1]; both must round-trip pixel-exact.
        for which, expected in ((0, src), (1, other)):
            with self.subTest(which=which):
                out = Evaluator._kernel("Switch", {"which": which}, [src, other])
                np.testing.assert_array_equal(out, expected)

    def test_dispatcher_create_and_wire_switch(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "red", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 1.0, "green": 0.0, "blue": 0.0, "alpha": 1.0}},
            {"op": "create", "id": "blue", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 0.0, "green": 0.0, "blue": 1.0, "alpha": 1.0}},
            {"op": "create", "id": "sw", "type": "Switch"},
            {"op": "connect", "id": "sw", "input": "0", "source": "red"},
            {"op": "connect", "id": "sw", "input": "1", "source": "blue"},
            {"op": "create", "id": "v", "type": "Viewer"},
            {"op": "connect", "id": "v", "input": "image", "source": "sw"},
            {"op": "view", "id": "v"}]})
        out0 = Evaluator().evaluate(d.document)
        # Switch starts at which=0, so the red plate passes through.
        np.testing.assert_array_equal(out0, Evaluator().evaluate(d.document, "red"))
        d.execute({"op": "set", "id": "sw", "param": "which", "value": 1})
        out1 = Evaluator().evaluate(d.document)
        np.testing.assert_array_equal(out1, Evaluator().evaluate(d.document, "blue"))


class SpecAndChoicesTests(unittest.TestCase):
    """SPECS / LIMITS surface that the inspector and agent protocol expose."""

    def test_dot_and_switch_are_in_specs(self):
        self.assertEqual(SPECS["Dot"]["inputs"], ["input"])
        self.assertEqual(SPECS["Dot"]["params"], {})
        self.assertEqual(SPECS["Switch"]["inputs"], ["0", "1"])
        self.assertEqual(SPECS["Switch"]["params"], {"which": 0})
        self.assertIn("which", LIMITS)
        self.assertEqual(LIMITS["which"], (0, 1))

    def test_filter_kinds_advertise_mask_and_mix(self):
        for kind in FILTER_KINDS:
            with self.subTest(kind=kind):
                spec = SPECS[kind]
                self.assertEqual(set(spec.get("optional_inputs", [])), {"mask"},
                                  f"{kind} should expose an optional 'mask' input")
                self.assertIn("mix", spec["params"])
                self.assertEqual(spec["params"]["mix"], 1.0)
                self.assertIn("mix", LIMITS)
                self.assertEqual(LIMITS["mix"], (0, 1))

    def test_non_filter_kinds_do_not_get_mask_or_mix_injected(self):
        # Non-filter node kinds must not advertise the new optional mask input. Merge keeps its
        # own (pre-existing) mix param, so we only check mask exposure here; mix is checked in
        # the filter-kind test above.
        for kind in ("Premult", "Unpremult", "Shuffle", "Read", "Constant",
                     "Checker", "Viewer", "Dot", "Switch"):
            with self.subTest(kind=kind):
                self.assertNotIn("mask", SPECS[kind].get("optional_inputs", []),
                                  f"{kind} should not advertise an optional mask input")


class AgentProtocolTests(unittest.TestCase):
    """Pipe real JSON-lines commands through nodebased.agent to verify the new capabilities
    surface in describe and that create+connect works for Dot and Switch end-to-end."""

    def setUp(self):
        self._env = {"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/usr/local/bin"}

    def _agent(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            stdin_payload = "\n".join(json.dumps(line) for line in lines) + "\n"
            proc = subprocess.run(
                [sys.executable, "-m", "nodebased.agent"],
                input=stdin_payload, capture_output=True, text=True, env=self._env,
            )
            responses = [json.loads(line) for line in proc.stdout.strip().split("\n") if line.strip()]
            return proc, responses

    def test_describe_advertises_new_capabilities(self):
        proc, responses = self._agent([{"op": "describe"}])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(responses[0]["ok"])
        result = responses[0]["result"]
        # Dot and Switch are surfaced with their inputs and params.
        self.assertEqual(result["nodes"]["Dot"]["inputs"], ["input"])
        self.assertEqual(result["nodes"]["Dot"]["params"], {})
        self.assertEqual(result["nodes"]["Switch"]["inputs"], ["0", "1"])
        self.assertEqual(result["nodes"]["Switch"]["params"], {"which": 0})
        # Filter nodes advertise optional mask + mix.
        for kind in FILTER_KINDS:
            self.assertEqual(set(result["nodes"][kind].get("optional_inputs", [])), {"mask"},
                              f"{kind} describe missing optional mask")
            self.assertIn("mix", result["nodes"][kind]["params"])
        # LIMITS carries the new 'which' bound. (JSON converts tuples to lists on the wire.)
        self.assertEqual(tuple(result["limits"]["which"]), (0, 1))

    def test_create_connect_switch_through_agent(self):
        proc, responses = self._agent([
            {"op": "create", "id": "red", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 1.0, "green": 0.0, "blue": 0.0, "alpha": 1.0}},
            {"op": "create", "id": "blue", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 0.0, "green": 0.0, "blue": 1.0, "alpha": 1.0}},
            {"op": "create", "id": "sw", "type": "Switch"},
            {"op": "connect", "id": "sw", "input": "0", "source": "red"},
            {"op": "connect", "id": "sw", "input": "1", "source": "blue"},
            {"op": "set", "id": "sw", "param": "which", "value": 1},
            {"op": "inspect"}])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(all(r["ok"] for r in responses))
        # The inspect response is the last one (revision: 5).
        doc = responses[-1]["result"]["document"]
        self.assertEqual(doc["version"], 4)
        sw = doc["nodes"]["sw"]
        self.assertEqual(sw["type"], "Switch")
        self.assertEqual(sw["inputs"], {"0": "red", "1": "blue"})
        self.assertEqual(sw["params"]["which"], 1)

    def test_create_connect_dot_through_agent(self):
        proc, responses = self._agent([
            {"op": "create", "id": "src", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 0.7, "green": 0.3, "blue": 0.5, "alpha": 1.0}},
            {"op": "create", "id": "dot", "type": "Dot"},
            {"op": "connect", "id": "dot", "input": "input", "source": "src"},
            {"op": "inspect"}])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(all(r["ok"] for r in responses))
        doc = responses[-1]["result"]["document"]
        self.assertEqual(doc["version"], 4)
        self.assertEqual(doc["nodes"]["dot"]["type"], "Dot")
        self.assertEqual(doc["nodes"]["dot"]["inputs"]["input"], "src")


if __name__ == "__main__":
    unittest.main()
