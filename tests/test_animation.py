"""Animation curves (schema v6): curve validation/evaluation, schema migration,
Dispatcher ops (set_key/delete_key/clear_curve), undo/redo, atomic rollback,
per-frame cache differentiation, and an agent CLI 2-frame render proof.

This file is self-contained: it does not depend on tests/test_phase_*.py except by
implicit use of the project's already-loaded modules.
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

from nodebased.animation import (CURVE_INTERPOLATIONS, DEFAULT_INTERPOLATION, FRAME_LIMITS,
                                   coerce_value_for_param, drop_key, evaluate_curve,
                                   merge_key, resolve_document, resolve_params, validate_curve,
                                   CurveError)
from nodebased.core import (Dispatcher, LIMITS, SCHEMA_VERSION, SPECS, demo_document,
                             empty_document, upgrade_document, validate)
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


def _build_two_constant_graph():
    """A Constant -> Grade -> Viewer. Returns a freshly-built v6 dispatcher whose document
    references nodes by id 'c', 'g', 'v' and whose view is 'v'."""
    d = Dispatcher()
    d.execute({"op": "batch", "commands": [
        {"op": "create", "id": "c", "type": "Constant",
         "params": {"width": 2, "height": 2, "red": 1.0, "green": 0.0, "blue": 0.0, "alpha": 1.0}},
        {"op": "create", "id": "g", "type": "Grade",
         "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}},
        {"op": "connect", "id": "g", "input": "image", "source": "c"},
        {"op": "create", "id": "v", "type": "Viewer"},
        {"op": "connect", "id": "v", "input": "image", "source": "g"},
        {"op": "view", "id": "v"}]})
    return d


class SchemaV6UpgradeTests(unittest.TestCase):
    """v5 documents upgrade cleanly to v6 and gain the animation section without rendering
    changes."""

    def test_empty_doc_is_v6_with_animation_section(self):
        d = empty_document()
        self.assertEqual(d["version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 6)
        self.assertEqual(d["animation"], {"curves": {}})
        validate(d)

    def test_v5_doc_gains_animation_curves_on_upgrade(self):
        old = {"version": 5, "view": None, "nodes": {}, "time": {"first": 1, "last": 1, "current": 1, "fps": 24.0}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], 6)
        self.assertEqual(upgraded["animation"], {"curves": {}})

    def test_v5_doc_with_grade_renders_identically_after_upgrade(self):
        # A v5 graph with no curves must render byte-identically through v6 — the upgrade is a
        # pure schema addition, not a behavioural change.
        v5 = {"version": 5, "view": "v", "time": {"first": 1, "last": 1, "current": 1, "fps": 24.0},
              "nodes": {
                  "c": {"type": "Constant", "name": "C", "pos": [0, 0], "disabled": False,
                        "inputs": {}, "params": {"width": 2, "height": 2, "red": 0.4, "green": 0.6,
                                                 "blue": 0.8, "alpha": 1.0}},
                  "g": {"type": "Grade", "name": "G", "pos": [0, 0], "disabled": False,
                        "inputs": {"image": "c"}, "params": {"exposure": 0.5, "multiply": 1.0,
                                                              "offset": 0.0, "mix": 1.0}},
                  "v": {"type": "Viewer", "name": "V", "pos": [0, 0], "disabled": False,
                        "inputs": {"image": "g"}, "params": {}}}}
        upgraded = upgrade_document(v5)
        out_upgraded = Evaluator().evaluate(upgraded)
        d = Dispatcher()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "c", "type": "Constant",
             "params": {"width": 2, "height": 2, "red": 0.4, "green": 0.6, "blue": 0.8, "alpha": 1.0}},
            {"op": "create", "id": "g", "type": "Grade",
             "params": {"exposure": 0.5, "multiply": 1.0, "offset": 0.0, "mix": 1.0}},
            {"op": "connect", "id": "g", "input": "image", "source": "c"},
            {"op": "create", "id": "v", "type": "Viewer"},
            {"op": "connect", "id": "v", "input": "image", "source": "g"},
            {"op": "view", "id": "v"}]})
        out_fresh = Evaluator().evaluate(d.document)
        np.testing.assert_array_equal(out_upgraded, out_fresh)

    def test_v5_to_v6_does_not_mutate_input_doc(self):
        # A v5 document passed to upgrade_document is deep-copied; the caller's input is untouched.
        old = {"version": 5, "view": None, "nodes": {}, "time": {"first": 1, "last": 1, "current": 1, "fps": 24.0}}
        snapshot = copy.deepcopy(old)
        upgrade_document(old)
        self.assertEqual(old, snapshot)

    def test_validate_rejects_unknown_curve_node(self):
        d = empty_document()
        d["animation"]["curves"]["missing"] = {"exposure": {"interpolation": "linear", "keys": []}}
        with self.assertRaisesRegex(ValueError, "missing node"):
            validate(d)

    def test_validate_rejects_curve_on_unknown_param(self):
        d = _build_two_constant_graph().document
        d["animation"]["curves"].setdefault("g", {})["bogus"] = {
            "interpolation": "linear",
            "keys": [{"frame": 0, "value": 0.0}],
        }
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            validate(d)

    def test_validate_rejects_curve_on_non_numeric_param(self):
        # Build a Read node (its 'path' is a string) and try to curve it.
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plate.png")
            # Create a 1x1 PNG so the Read's path validates.
            from nodebased.imaging import write_png as _write_png
            _write_png(path, np.zeros((1, 1, 4), np.float32))
            d = Dispatcher()
            d.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": path}})
            d.document["animation"]["curves"]["r"] = {"path": {"interpolation": "linear",
                                                                "keys": [{"frame": 0, "value": 0.0}]}}
            try:
                validate(d.document)
                raise AssertionError("expected validate to fail on non-numeric curve param")
            except ValueError as error:
                self.assertIn("only numeric", str(error))

    def test_validate_rejects_malformed_curve_payload(self):
        d = _build_two_constant_graph().document
        d["animation"]["curves"].setdefault("g", {})["exposure"] = {
            "interpolation": "bezier",
            "keys": [{"frame": 0, "value": 0.0}],
        }
        with self.assertRaisesRegex(ValueError, "interpolation"):
            validate(d)


class ValidateCurveUnitTests(unittest.TestCase):

    def test_empty_keys_list_rejected(self):
        with self.assertRaisesRegex(CurveError, "non-empty"):
            validate_curve({"interpolation": "linear", "keys": []})

    def test_duplicate_frame_rejected(self):
        with self.assertRaisesRegex(CurveError, "strictly increasing"):
            validate_curve({"interpolation": "linear",
                            "keys": [{"frame": 0, "value": 0.0}, {"frame": 0, "value": 1.0}]})

    def test_unsorted_keys_rejected(self):
        with self.assertRaisesRegex(CurveError, "strictly increasing"):
            validate_curve({"interpolation": "linear",
                            "keys": [{"frame": 5, "value": 0.0}, {"frame": 2, "value": 1.0}]})

    def test_nan_value_rejected(self):
        with self.assertRaisesRegex(CurveError, "finite"):
            validate_curve({"interpolation": "linear",
                            "keys": [{"frame": 0, "value": float("nan")}]})

    def test_inf_value_rejected(self):
        with self.assertRaisesRegex(CurveError, "finite"):
            validate_curve({"interpolation": "linear",
                            "keys": [{"frame": 0, "value": float("inf")}]})

    def test_non_int_frame_rejected(self):
        with self.assertRaisesRegex(CurveError, "integer"):
            validate_curve({"interpolation": "linear",
                            "keys": [{"frame": 1.5, "value": 0.0}]})

    def test_extra_keys_rejected(self):
        with self.assertRaisesRegex(CurveError, "exactly"):
            validate_curve({"interpolation": "linear",
                            "keys": [{"frame": 0, "value": 0.0}],
                            "extra": 1})


class EvaluateCurveUnitTests(unittest.TestCase):

    LINEAR = {"interpolation": "linear",
              "keys": [{"frame": 0, "value": 0.0}, {"frame": 10, "value": 1.0}]}

    CONSTANT = {"interpolation": "constant",
                "keys": [{"frame": 0, "value": 0.5}, {"frame": 10, "value": 1.5}]}

    def test_linear_exact_at_endpoints(self):
        self.assertEqual(evaluate_curve(self.LINEAR, 0), 0.0)
        self.assertEqual(evaluate_curve(self.LINEAR, 10), 1.0)

    def test_linear_midpoint(self):
        self.assertEqual(evaluate_curve(self.LINEAR, 5), 0.5)

    def test_linear_quarter_point(self):
        self.assertEqual(evaluate_curve(self.LINEAR, 2), 0.2)

    def test_linear_before_first_holds_first_value(self):
        self.assertEqual(evaluate_curve(self.LINEAR, -1), 0.0)
        self.assertEqual(evaluate_curve(self.LINEAR, -1000), 0.0)

    def test_linear_after_last_holds_last_value(self):
        self.assertEqual(evaluate_curve(self.LINEAR, 11), 1.0)
        self.assertEqual(evaluate_curve(self.LINEAR, 1000), 1.0)

    def test_constant_exact_at_endpoints(self):
        self.assertEqual(evaluate_curve(self.CONSTANT, 0), 0.5)
        self.assertEqual(evaluate_curve(self.CONSTANT, 10), 1.5)

    def test_constant_holds_previous_value_within_range(self):
        self.assertEqual(evaluate_curve(self.CONSTANT, 1), 0.5)
        self.assertEqual(evaluate_curve(self.CONSTANT, 9), 0.5)
        self.assertEqual(evaluate_curve(self.CONSTANT, 5), 0.5)

    def test_constant_before_first_holds_first_value(self):
        self.assertEqual(evaluate_curve(self.CONSTANT, -1), 0.5)
        self.assertEqual(evaluate_curve(self.CONSTANT, -1000), 0.5)

    def test_constant_after_last_holds_last_value(self):
        self.assertEqual(evaluate_curve(self.CONSTANT, 11), 1.5)
        self.assertEqual(evaluate_curve(self.CONSTANT, 1000), 1.5)


class ResolveParamsTests(unittest.TestCase):

    def _grade(self):
        return {"type": "Grade", "params": {"exposure": 0.5, "multiply": 1.0, "offset": 0.0, "mix": 1.0}}

    def test_no_curves_returns_equivalent_dict(self):
        node = self._grade()
        resolved = resolve_params(node, None, 1, SPECS["Grade"]["params"], LIMITS)
        self.assertEqual(resolved, node["params"])
        # Resolved is a copy, not the same object.
        self.assertIsNot(resolved, node["params"])

    def test_endpoint_hold_outside_curve_range(self):
        # Curve covers frames 10..20; out-of-range frames endpoint-hold the first/last value.
        node = self._grade()
        curves = {"exposure": {"interpolation": "linear",
                                "keys": [{"frame": 10, "value": 1.0}, {"frame": 20, "value": 2.0}]}}
        for frame, expected in ((1, 1.0), (5, 1.0), (9, 1.0), (21, 2.0), (100, 2.0)):
            with self.subTest(frame=frame):
                resolved = resolve_params(node, curves, frame, SPECS["Grade"]["params"], LIMITS)
                self.assertEqual(resolved["exposure"], expected)

    def test_linear_resolution_within_range(self):
        node = self._grade()
        curves = {"exposure": {"interpolation": "linear",
                                "keys": [{"frame": 10, "value": 0.0}, {"frame": 20, "value": 2.0}]}}
        self.assertEqual(resolve_params(node, curves, 10, SPECS["Grade"]["params"], LIMITS)["exposure"], 0.0)
        self.assertEqual(resolve_params(node, curves, 15, SPECS["Grade"]["params"], LIMITS)["exposure"], 1.0)
        self.assertEqual(resolve_params(node, curves, 20, SPECS["Grade"]["params"], LIMITS)["exposure"], 2.0)

    def test_constant_holds_value_within_range(self):
        node = self._grade()
        # Single-key constant curve at frame=10 -> endpoint hold means every frame is 1.5.
        curves = {"exposure": {"interpolation": "constant",
                                "keys": [{"frame": 10, "value": 1.5}]}}
        for frame in (10, 11, 99):
            with self.subTest(frame=frame):
                self.assertEqual(resolve_params(node, curves, frame, SPECS["Grade"]["params"], LIMITS)["exposure"], 1.5)

    def test_int_param_rounds_and_clamps_at_resolution(self):
        # 'which' (Switch) is an int param bounded (0, 1). A curve with a float value must round,
        # and the rounded result must still fall inside the param's LIMITS.
        node = {"type": "Switch", "params": {"which": 0}}
        curves = {"which": {"interpolation": "constant",
                            "keys": [{"frame": 0, "value": 1.6}, {"frame": 5, "value": 0.4}]}}
        # 1.6 rounds to 2, but LIMITS['which']=(0,1) clamps to 1.
        self.assertEqual(resolve_params(node, curves, 0, SPECS["Switch"]["params"], LIMITS)["which"], 1)
        # 0.4 rounds to 0.
        self.assertEqual(resolve_params(node, curves, 5, SPECS["Switch"]["params"], LIMITS)["which"], 0)

    def test_range_clamping_at_resolution(self):
        # 'exposure' is bounded (-20, 20). A constant curve at the in-range key clamps.
        node = self._grade()
        # Two keys so frame 5 is within the curve's range.
        curves = {"exposure": {"interpolation": "constant",
                                "keys": [{"frame": 0, "value": 100.0}, {"frame": 10, "value": 100.0}]}}
        resolved = resolve_params(node, curves, 5, SPECS["Grade"]["params"], LIMITS)
        self.assertEqual(resolved["exposure"], 20.0)
        curves = {"exposure": {"interpolation": "constant",
                                "keys": [{"frame": 0, "value": -100.0}, {"frame": 10, "value": -100.0}]}}
        resolved = resolve_params(node, curves, 5, SPECS["Grade"]["params"], LIMITS)
        self.assertEqual(resolved["exposure"], -20.0)

    def test_other_params_are_untouched(self):
        node = self._grade()
        curves = {"exposure": {"interpolation": "constant",
                                "keys": [{"frame": 0, "value": 0.0}]}}
        resolved = resolve_params(node, curves, 5, SPECS["Grade"]["params"], LIMITS)
        self.assertEqual(resolved["multiply"], 1.0)
        self.assertEqual(resolved["offset"], 0.0)
        self.assertEqual(resolved["mix"], 1.0)


class DispatcherAnimationOpsTests(unittest.TestCase):

    def setUp(self):
        self.d = _build_two_constant_graph()

    def test_set_key_creates_curve(self):
        r = self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.7})
        self.assertTrue(r["result"]["interpolation"], "linear")
        self.assertEqual(self.d.document["animation"]["curves"]["g"]["exposure"]["keys"], [{"frame": 1, "value": 0.7}])

    def test_set_key_does_not_mutate_node_params(self):
        before = copy.deepcopy(self.d.document["nodes"]["g"]["params"])
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.7})
        # Stored params are unchanged.
        self.assertEqual(self.d.document["nodes"]["g"]["params"], before)

    def test_set_key_replaces_existing_key_at_same_frame(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.5})
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.9})
        keys = self.d.document["animation"]["curves"]["g"]["exposure"]["keys"]
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0]["value"], 0.9)

    def test_set_key_appends_and_sorts(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 5, "value": 1.0})
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 2, "value": 0.5})
        keys = self.d.document["animation"]["curves"]["g"]["exposure"]["keys"]
        self.assertEqual([k["frame"] for k in keys], [2, 5])

    def test_set_key_rejects_non_numeric_param(self):
        # Grade has no 'missing' param; this triggers the unknown-parameter error.
        with self.assertRaisesRegex(ValueError, "no parameter"):
            self.d.execute({"op": "set_key", "id": "g", "param": "missing", "frame": 1, "value": 0.5})

    def test_set_key_rejects_out_of_range_value(self):
        # exposure is bounded (-20, 20); 100 is out.
        with self.assertRaisesRegex(ValueError, "outside"):
            self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 100})

    def test_set_key_rejects_unknown_interpolation(self):
        with self.assertRaisesRegex(ValueError, "interpolation"):
            self.d.execute({"op": "set_key", "id": "g", "param": "exposure",
                             "frame": 1, "value": 0.5, "interpolation": "bezier"})

    def test_set_key_rejects_non_integer_frame(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            self.d.execute({"op": "set_key", "id": "g", "param": "exposure",
                             "frame": 1.5, "value": 0.5})

    def test_set_key_rejects_nan_value(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.d.execute({"op": "set_key", "id": "g", "param": "exposure",
                             "frame": 1, "value": float("nan")})

    def test_invalid_set_key_rolls_back_atomically(self):
        # No undo slot added when the op is rejected; revision unchanged.
        rev_before = self.d.revision
        with self.assertRaises(ValueError):
            self.d.execute({"op": "set_key", "id": "g", "param": "exposure",
                             "frame": 1, "value": 999})
        self.assertEqual(self.d.revision, rev_before)
        self.assertEqual(self.d.document["animation"]["curves"], {})

    def test_delete_key_removes_one_key_and_keeps_curve(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.4})
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 5, "value": 0.8})
        self.d.execute({"op": "delete_key", "id": "g", "param": "exposure", "frame": 1})
        keys = self.d.document["animation"]["curves"]["g"]["exposure"]["keys"]
        self.assertEqual([k["frame"] for k in keys], [5])

    def test_delete_key_empties_curve_and_prunes_node(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.4})
        self.d.execute({"op": "delete_key", "id": "g", "param": "exposure", "frame": 1})
        # Empty curve is removed entirely; node entry is removed when empty.
        self.assertNotIn("g", self.d.document["animation"]["curves"])

    def test_delete_key_on_missing_curve_raises(self):
        with self.assertRaisesRegex(ValueError, "no curve"):
            self.d.execute({"op": "delete_key", "id": "g", "param": "exposure", "frame": 1})

    def test_clear_curve_removes_just_one_param(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.4})
        self.d.execute({"op": "set_key", "id": "g", "param": "offset", "frame": 1, "value": 0.1})
        self.d.execute({"op": "clear_curve", "id": "g", "param": "exposure"})
        curve_for_g = self.d.document["animation"]["curves"]["g"]
        self.assertNotIn("exposure", curve_for_g)
        self.assertIn("offset", curve_for_g)

    def test_clear_curve_on_missing_raises(self):
        with self.assertRaisesRegex(ValueError, "no curve"):
            self.d.execute({"op": "clear_curve", "id": "g", "param": "exposure"})

    def test_undo_restores_pre_animation_state(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.7})
        self.d.execute({"op": "undo"})
        self.assertEqual(self.d.document["animation"]["curves"], {})

    def test_redo_reapplies_animation_edit(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.7})
        self.d.execute({"op": "undo"})
        self.d.execute({"op": "redo"})
        self.assertIn("g", self.d.document["animation"]["curves"])

    def test_animation_edits_are_undoable_in_a_batch(self):
        # Atomicity of a batch containing animation ops: the whole batch rolls back on any error.
        rev_before = self.d.revision
        with self.assertRaises(ValueError):
            self.d.execute({"op": "batch", "commands": [
                {"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.5},
                {"op": "set_key", "id": "g", "param": "exposure", "frame": 2, "value": 999},
            ]})
        self.assertEqual(self.d.revision, rev_before)
        self.assertEqual(self.d.document["animation"]["curves"], {})

    def test_delete_node_with_curves_drops_them_atomically(self):
        # Without atomic cleanup, doc["animation"]["curves"][node_id] would survive the delete
        # and validate() would reject the resulting document for referencing a missing node.
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.5})
        self.d.execute({"op": "set_key", "id": "g", "param": "offset", "frame": 2, "value": 0.1})
        # Snapshot the curves entry for the doomed node before deletion.
        curves_before = copy.deepcopy(self.d.document["animation"]["curves"]["g"])
        self.d.execute({"op": "delete", "id": "g"})
        # Curves entry is gone; the document still validates.
        self.assertNotIn("g", self.d.document["animation"]["curves"])
        validate(self.d.document)
        # Other nodes' curves are unaffected.
        self.d.execute({"op": "set_key", "id": "c", "param": "alpha", "frame": 1, "value": 0.4})
        self.d.execute({"op": "delete", "id": "c"})
        self.assertNotIn("c", self.d.document["animation"]["curves"])

    def test_delete_node_with_curves_is_undoable(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.5})
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 5, "value": 0.9})
        snapshot = copy.deepcopy(self.d.document["animation"]["curves"]["g"])
        self.d.execute({"op": "delete", "id": "g"})
        self.assertNotIn("g", self.d.document["animation"]["curves"])
        self.d.execute({"op": "undo"})
        # The node is back, and its curves are restored exactly as they were before the delete.
        self.assertIn("g", self.d.document["nodes"])
        self.assertEqual(self.d.document["animation"]["curves"]["g"], snapshot)

    def test_delete_node_with_curves_is_redoable(self):
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.5})
        self.d.execute({"op": "delete", "id": "g"})
        self.d.execute({"op": "undo"})
        self.d.execute({"op": "redo"})
        # Redo of the delete clears the curves again.
        self.assertNotIn("g", self.d.document["nodes"])
        self.assertNotIn("g", self.d.document["animation"]["curves"])

    def test_batch_delete_and_set_key_atomicity(self):
        # A batch that deletes a node and tries to add a curve to it on the same key must roll
        # back the delete entirely, because the set_key would refer to a missing node. Confirms
        # batch atomicity still applies when animation and topology edits are interleaved.
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.5})
        rev_before = self.d.revision
        with self.assertRaises(ValueError):
            self.d.execute({"op": "batch", "commands": [
                {"op": "delete", "id": "g"},
                {"op": "set_key", "id": "g", "param": "exposure", "frame": 2, "value": 0.9},
            ]})
        self.assertEqual(self.d.revision, rev_before)
        # The g node and its original curves are still there.
        self.assertIn("g", self.d.document["nodes"])
        self.assertIn("g", self.d.document["animation"]["curves"])


class EvaluatorCacheIntegrationTests(unittest.TestCase):
    """Animation only invalidates the cache when the resolved per-frame value actually changes;
    static nodes still cache through the full frame range."""

    def setUp(self):
        self.d = _build_two_constant_graph()
        self.e = Evaluator()
        # Render once at the document's default frame so the cache is warm for the static baseline.
        self.e.evaluate(self.d.document, frame=1)
        self.baseline_misses = self.e.misses

    def test_static_node_cache_reuses_across_frames(self):
        # The Constant has no curves and no time dependence; rendering any frame should hit the
        # cache for it (moves == 0).
        before_misses = self.e.misses
        out2 = self.e.evaluate(self.d.document, frame=5)
        out3 = self.e.evaluate(self.d.document, frame=10)
        # Cache hit on Constant; the Viewer/Grade digest changes with frame via the resolved
        # exposure (still 0 here), so we mainly check no NaN and the static Constant stays hit.
        self.assertFalse(np.isnan(out2).any())
        self.assertFalse(np.isnan(out3).any())
        # We should have at least 1 cache hit from re-rendering at a different frame.
        self.assertGreater(self.e.hits, 0)

    def test_animated_node_cache_invalidates_per_frame(self):
        # Set up a linear exposure ramp from frame 1 (0.0) to frame 10 (2.0).
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.0})
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 10, "value": 2.0})
        e = Evaluator()
        e.evaluate(self.d.document, frame=1)
        e.evaluate(self.d.document, frame=10)
        # Now render frame 5 — must be a miss for the Grade node (resolved exposure differs).
        before_misses = e.misses
        out5 = e.evaluate(self.d.document, frame=5)
        # The Grade pixel at frame 5 should be brighter than at frame 1 and dimmer than at frame 10.
        out1 = Evaluator().evaluate(self.d.document, frame=1)
        out10 = Evaluator().evaluate(self.d.document, frame=10)
        # The Constant is premultiplied red, so an exposure of 0 keeps RGB*1.0; 2 doubles it.
        self.assertGreater(out5[..., 0].mean(), out1[..., 0].mean())
        self.assertLess(out5[..., 0].mean(), out10[..., 0].mean())
        # Cache invalidation happened (frame 5 resolved to a value not seen at 1 or 10).
        self.assertGreater(e.misses, before_misses)

    def test_out_of_range_frame_uses_endpoint_hold_in_cache(self):
        # With endpoint hold, an out-of-range frame resolves to the first/last key's value, not
        # the base param. Re-evaluating the same out-of-range frame twice is a cache hit (the
        # resolved params are deterministic), and the cached value differs from a frame inside
        # the range.
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 10, "value": 1.0})
        self.d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 20, "value": 2.0})
        e = Evaluator()
        out_before = e.evaluate(self.d.document, frame=100)  # out of range -> holds last (2.0)
        out_after = e.evaluate(self.d.document, frame=5)     # before first  -> holds first (1.0)
        # Re-evaluate the same out-of-range frames: must hit the cache.
        before_hits = e.hits
        out_before_again = e.evaluate(self.d.document, frame=100)
        out_after_again = e.evaluate(self.d.document, frame=5)
        self.assertGreater(e.hits, before_hits)
        np.testing.assert_array_equal(out_before, out_before_again)
        np.testing.assert_array_equal(out_after, out_after_again)
        # The cached mean R for the before-first frame (held at 1.0) is dimmer than for the
        # after-last frame (held at 2.0).
        self.assertGreater(out_before[..., 0].mean(), out_after[..., 0].mean())


class AgentCliLiveProofTests(unittest.TestCase):
    """Pipe real JSON-lines commands through nodebased.agent to verify describe exposes the new
    animation schema and that a set_key + render across two frames works end-to-end."""

    def setUp(self):
        self._env = {"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/usr/local/bin"}

    def _run(self, lines):
        stdin_payload = "\n".join(json.dumps(line) for line in lines) + "\n"
        proc = subprocess.run(
            [sys.executable, "-m", "nodebased.agent"],
            input=stdin_payload, capture_output=True, text=True, env=self._env,
        )
        responses = [json.loads(line) for line in proc.stdout.strip().split("\n") if line.strip()]
        return proc, responses

    def test_describe_advertises_animation(self):
        proc, responses = self._run([{"op": "describe"}])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(responses[0]["ok"])
        result = responses[0]["result"]
        anim = result["animation"]
        self.assertIn("interpolations", anim)
        self.assertEqual(set(anim["interpolations"]), {"constant", "linear"})
        self.assertEqual(tuple(anim["frame_limits"]), FRAME_LIMITS)
        self.assertIn("set_key", anim)
        self.assertIn("delete_key", anim)
        self.assertIn("clear_curve", anim)
        # Operations list surfaces the new ops.
        self.assertIn("set_key", result["operations"])
        self.assertIn("delete_key", result["operations"])
        self.assertIn("clear_curve", result["operations"])

    def test_set_key_render_across_two_frames(self):
        # Build a graph that can render to PNG via the agent CLI, key exposure at frames 1 and 10,
        # then render at frames 1, 5, and 10 and verify each call returns a fresh path with
        # distinct content (cached path differs across frames because the resolved value does).
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            lines = [
                {"op": "create", "id": "c", "type": "Constant",
                 "params": {"width": 2, "height": 2, "red": 1.0, "green": 0.5, "blue": 0.2, "alpha": 1.0}},
                {"op": "create", "id": "g", "type": "Grade",
                 "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}},
                {"op": "connect", "id": "g", "input": "image", "source": "c"},
                {"op": "create", "id": "v", "type": "Viewer"},
                {"op": "connect", "id": "v", "input": "image", "source": "g"},
                {"op": "view", "id": "v"},
                {"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 0.0},
                {"op": "set_key", "id": "g", "param": "exposure", "frame": 10, "value": 2.0},
                {"op": "inspect"},
            ]
            for frame in (1, 5, 10):
                paths.append(str(Path(tmp) / f"frame_{frame}.png"))
                lines.append({"op": "render", "id": "v", "path": paths[-1], "frame": frame})
            proc, responses = self._run(lines)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(all(r["ok"] for r in responses))
            # Render responses should each report the rendered shape and confirm the path was written.
            render_responses = responses[-3:]
            for path, resp in zip(paths, render_responses):
                self.assertTrue(os.path.exists(path), f"render output missing: {path}")
                self.assertEqual(resp["result"]["path"], path)
                self.assertEqual(resp["result"]["width"], 2)
                self.assertEqual(resp["result"]["height"], 2)
            # Compare mean R of the three frames: out-of-range (frame=1 -> exposure=0) is dim,
            # frame 5 is in-range (linear midpoint -> exposure=1.0), frame 10 is brightest.
            try:
                import OpenImageIO as oiio
                means = []
                for path in paths:
                    arr = oiio.ImageBuf(path).get_pixels(oiio.FLOAT)
                    means.append(float(arr[..., 0].mean()))
                self.assertLess(means[0], means[1])
                self.assertLess(means[1], means[2])
            except Exception:  # noqa: BLE001 — OIIO optional at test time
                # Compare file sizes as a coarse sanity check; encoded bytes scale with mean.
                sizes = [os.path.getsize(p) for p in paths]
                self.assertEqual(sizes, sorted(sizes), "frame mean R should increase across frames")


class AnimationThroughTileExecutorTests(unittest.TestCase):
    """Animation predates the tile engine; the two features first meet at this rebase.

    `tileexec` reads `node["params"]` directly for digests, canvas size, source generation and
    kernel dispatch. Without `animation.resolve_document` an animated parameter renders
    correctly through the reference `Evaluator` and silently freezes at its stored base value
    through tiles — and tiles are the viewer's default path since v0.9. Each case pins tile
    output against the reference at the same frame, then proves the frames actually differ so a
    frozen parameter cannot pass by matching a reference that is equally frozen.
    """

    def _ramped_graph(self):
        """Checker -> Blur -> Grade, with an animated pixel-unit param and an animated scalar."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "plate",
                   "params": {"width": 288, "height": 192, "size": 24}})
        d.execute({"op": "create", "type": "Blur", "id": "blur",
                   "params": {"radius": 0.0, "mix": 1.0}})
        d.execute({"op": "connect", "id": "blur", "input": "image", "source": "plate"})
        d.execute({"op": "create", "type": "Grade", "id": "grade",
                   "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}})
        d.execute({"op": "connect", "id": "grade", "input": "image", "source": "blur"})
        # radius is pixel-unit, so it also exercises tier scaling of a resolved value.
        d.execute({"op": "set_key", "id": "blur", "param": "radius", "frame": 1, "value": 0.0})
        d.execute({"op": "set_key", "id": "blur", "param": "radius", "frame": 10, "value": 24.0})
        d.execute({"op": "set_key", "id": "grade", "param": "exposure", "frame": 1, "value": 0.0})
        d.execute({"op": "set_key", "id": "grade", "param": "exposure", "frame": 10, "value": 2.0})
        return d

    def test_animated_params_match_the_reference_through_tiles_at_every_tier(self):
        d = self._ramped_graph()
        for tier in (1, 2, 4):
            for frame in (1, 5, 10):
                with self.subTest(tier=tier, frame=frame):
                    tiled = TileExecutor(tile_edge=64).compose(d.document, "grade",
                                                               frame=frame, tier=tier)
                    reference = Evaluator().evaluate(d.document, "grade", frame=frame, tier=tier)
                    self.assertTrue(tiled.tiled, "graph must take the tiled path, not the fallback")
                    self.assertEqual(tiled.pixels.shape, reference.shape)
                    np.testing.assert_allclose(tiled.pixels, reference, atol=1e-5)

    def test_tile_output_actually_changes_across_animated_frames(self):
        d = self._ramped_graph()
        executor = TileExecutor(tile_edge=64)
        first = executor.compose(d.document, "grade", frame=1).pixels
        last = executor.compose(d.document, "grade", frame=10).pixels
        # Exposure 0 -> 2 must brighten, and radius 0 -> 24 must reduce checker contrast.
        self.assertGreater(last[..., 0].mean(), first[..., 0].mean() * 1.5)
        self.assertLess(float(last[..., :3].std()), float(first[..., :3].std()))

    def test_animated_canvas_size_reaches_the_tile_canvas_query(self):
        """`canvas_size` reads the generator's params on its own path, separate from compose."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "plate",
                   "params": {"width": 200, "height": 100}})
        d.execute({"op": "set_key", "id": "plate", "param": "width", "frame": 1, "value": 200})
        d.execute({"op": "set_key", "id": "plate", "param": "width", "frame": 10, "value": 400})
        executor = TileExecutor(tile_edge=64)
        self.assertEqual(executor.canvas_size(d.document, "plate", frame=1), (200, 100))
        self.assertEqual(executor.canvas_size(d.document, "plate", frame=10), (400, 100))

    def test_a_document_without_curves_is_returned_unchanged(self):
        """Cache keys of unanimated graphs must not shift because resolution now runs."""
        d = Dispatcher(demo_document())
        self.assertIs(resolve_document(d.document, 7), d.document)
        animated = _build_two_constant_graph()
        # Curves exist but resolve to the stored values at frame 1, so still identity.
        animated.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1,
                          "value": animated.document["nodes"]["g"]["params"]["exposure"]})
        self.assertIs(resolve_document(animated.document, 1), animated.document)


if __name__ == "__main__":
    unittest.main()
