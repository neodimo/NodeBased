"""2D parity plan 15, step F2: TVIScale, Precomp and Assert.

TVIScale is Nuke's legacy power-of-two up/down scaler, changing the display window like Reformat
but -- unlike Reformat -- on the tile path (`tileexec._temporal_tile`'s "solve once, slice many"
shape). Precomp sources its picture from another saved document entirely, a generator like Read.
Assert is a pass-through QA node that raises, naming itself, when a restricted expression over the
input's own pixel statistics is false.
"""
import os
import tempfile
import unittest

import numpy as np

from nodebased.core import Dispatcher, atomic_save, bypass_slot
from nodebased.expressions import ExpressionError
from nodebased.imaging import Evaluator
from nodebased.ops2d_assert import evaluate_condition
from nodebased.tileexec import TileExecutor


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(evaluator, document, target, frame=1):
    return evaluator.evaluate(dict(document, view=target), frame=frame)


def tile_pixels(document, target, tile_edge=None, frame=1):
    kwargs = {} if tile_edge is None else {"tile_edge": tile_edge}
    executor = TileExecutor(evaluator=Evaluator(), **kwargs)
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=frame, tier=1)
    return executor.compose_region(document, target, region, frame=frame, tier=1).pixels


def build_checker_tviscale(power, filter="nearest"):
    g = Graph()
    g.add("plate", "Checker", dict(width=8, height=8, size=2))
    g.add("node", "TVIScale", dict(power=power, filter=filter), image="plate")
    return g


class TVIScaleTests(unittest.TestCase):
    def test_power_zero_is_the_identity(self):
        g = build_checker_tviscale(0)
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        np.testing.assert_array_equal(evaluator_pixels(Evaluator(), g.doc, "node"), plate)

    def test_hand_worked_2x_up_with_nearest_is_pixel_replication(self):
        g = build_checker_tviscale(1)
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        expected = np.repeat(np.repeat(plate, 2, axis=0), 2, axis=1)
        result = evaluator_pixels(Evaluator(), g.doc, "node")
        self.assertEqual(result.shape[:2], (16, 16))
        np.testing.assert_array_equal(result, expected)

    def test_hand_worked_2x_down_with_nearest_is_every_other_pixel(self):
        g = build_checker_tviscale(-1)
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        expected = plate[::2, ::2]
        result = evaluator_pixels(Evaluator(), g.doc, "node")
        self.assertEqual(result.shape[:2], (4, 4))
        np.testing.assert_array_equal(result, expected)

    def test_tiles_equal_full_frame_at_default_and_small_tile_edge(self):
        for power in (1, -1, 0):
            with self.subTest(power=power):
                g = build_checker_tviscale(power)
                full = evaluator_pixels(Evaluator(), g.doc, "node")
                np.testing.assert_array_equal(tile_pixels(g.doc, "node"), full)
                np.testing.assert_array_equal(tile_pixels(g.doc, "node", tile_edge=3), full)

    def test_bypassed_passes_the_unscaled_input_on_both_paths(self):
        g = build_checker_tviscale(2)
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        g.d.execute(dict(op="disable", id="node", value=True))
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
        np.testing.assert_array_equal(evaluator_pixels(Evaluator(), g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node"), plate)

    def test_mask_and_mix_gate_the_scale_like_reformat(self):
        g = build_checker_tviscale(1)
        # The mask is checked against the SOURCE's own display window (8x8), like Reformat's own
        # mask check -- gating happens before the resample, not against the scaled-up target.
        g.add("mask", "Constant", dict(width=8, height=8, red=0.5, green=0.5, blue=0.5, alpha=1.0))
        g.d.execute(dict(op="connect", id="node", input="mask", source="mask"))
        g.d.execute(dict(op="set", id="node", param="mix", value=0.5))
        scaled = evaluator_pixels(Evaluator(), g.doc, "node")
        # mix=0.5 blends the scaled result halfway back toward the (aligned) unscaled source; the
        # result must differ from a pure mix=1 scale wherever source and scaled pixels disagree.
        full_mix = build_checker_tviscale(1)
        full = evaluator_pixels(Evaluator(), full_mix.doc, "node")
        self.assertFalse(np.array_equal(scaled, full))


class PrecompTests(unittest.TestCase):
    def _write_reference(self, path, exposure=0.0):
        g = Graph()
        g.add("plate", "Checker", dict(width=8, height=8, size=2))
        g.add("grade", "Grade", dict(exposure=exposure), image="plate")
        g.d.execute(dict(op="view", id="grade"))
        atomic_save(path, g.doc)
        return g.doc

    def test_reproduces_the_referenced_output_pixel_for_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.nbp")
            ref_doc = self._write_reference(ref_path, exposure=0.7)
            expected = evaluator_pixels(Evaluator(), ref_doc, "grade")

            g = Graph()
            g.add("node", "Precomp", dict(file=ref_path))
            result = evaluator_pixels(Evaluator(), g.doc, "node")
            np.testing.assert_array_equal(result, expected)

    def test_explicit_output_node_overrides_the_referenced_documents_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.nbp")
            ref_doc = self._write_reference(ref_path, exposure=0.7)
            plate_only = evaluator_pixels(Evaluator(), ref_doc, "plate")

            g = Graph()
            g.add("node", "Precomp", dict(file=ref_path, output_node="plate"))
            result = evaluator_pixels(Evaluator(), g.doc, "node")
            np.testing.assert_array_equal(result, plate_only)

    def test_reload_picks_up_an_edited_and_saved_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.nbp")
            self._write_reference(ref_path, exposure=0.0)

            g = Graph()
            g.add("node", "Precomp", dict(file=ref_path))
            evaluator = Evaluator()
            before = evaluator_pixels(evaluator, g.doc, "node")

            self._write_reference(ref_path, exposure=2.0)
            # Same node params (reload unchanged): the outer evaluator's own per-node cache may
            # still serve the earlier result -- this is the documented reason `reload` exists.
            still_cached = evaluator_pixels(evaluator, g.doc, "node")
            np.testing.assert_array_equal(still_cached, before)

            g.d.execute(dict(op="set", id="node", param="reload", value=1))
            after = evaluator_pixels(evaluator, g.doc, "node")
            self.assertFalse(np.array_equal(after, before))

    def test_missing_file_raises_naming_the_path(self):
        g = Graph()
        missing_path = "/no/such/directory/nowhere.nbp"
        g.add("node", "Precomp", dict(file=missing_path))
        with self.assertRaises(ValueError) as ctx:
            evaluator_pixels(Evaluator(), g.doc, "node")
        self.assertIn(missing_path, str(ctx.exception))

    def test_empty_file_raises_asking_for_a_document(self):
        g = Graph()
        g.add("node", "Precomp", {})
        with self.assertRaises(ValueError):
            evaluator_pixels(Evaluator(), g.doc, "node")


class AssertTests(unittest.TestCase):
    def build(self, condition, message="Assertion failed"):
        g = Graph()
        g.add("plate", "Checker", dict(width=8, height=8, size=2))
        g.add("node", "Assert", dict(condition=condition, message=message), image="plate")
        return g

    def test_true_condition_passes_pixels_unchanged(self):
        g = self.build("avg >= 0")
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        np.testing.assert_array_equal(evaluator_pixels(Evaluator(), g.doc, "node"), plate)

    def test_false_condition_raises_with_the_node_name_and_message(self):
        g = self.build("avg > 1000000", message="plate is too dark")
        g.d.execute(dict(op="rename", id="node", name="QA_check"))
        with self.assertRaises(ValueError) as ctx:
            evaluator_pixels(Evaluator(), g.doc, "node")
        self.assertIn("QA_check", str(ctx.exception))
        self.assertIn("plate is too dark", str(ctx.exception))

    def test_bypassed_passes_the_input_regardless_of_the_condition(self):
        g = self.build("avg > 1000000")
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        g.d.execute(dict(op="disable", id="node", value=True))
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
        np.testing.assert_array_equal(evaluator_pixels(Evaluator(), g.doc, "node"), plate)

    def test_condition_reads_frame_and_per_channel_stats(self):
        g = self.build("frame == 1 and r_avg >= 0 and r_avg <= 1 and width == 8 and height == 8")
        evaluator_pixels(Evaluator(), g.doc, "node", frame=1)  # does not raise

    def test_off_the_tile_path(self):
        from nodebased.tiles import SUPPORTED_TILED_KINDS
        self.assertNotIn("Assert", SUPPORTED_TILED_KINDS)


class AssertConditionEvaluatorTests(unittest.TestCase):
    """Direct coverage of `ops2d_assert.evaluate_condition`'s safety and statistics."""

    def _pixels(self):
        pixels = np.zeros((2, 2, 4), np.float32)
        pixels[0, 0] = (0.0, 0.0, 0.0, 1.0)
        pixels[0, 1] = (1.0, 0.0, 0.0, 1.0)
        pixels[1, 0] = (0.0, 1.0, 0.0, 1.0)
        pixels[1, 1] = (0.0, 0.0, 1.0, 1.0)
        return pixels

    def test_channel_average_matches_numpy(self):
        pixels = self._pixels()
        self.assertTrue(evaluate_condition("r_avg == 0.25", pixels, frame=1))
        self.assertTrue(evaluate_condition("g_avg == 0.25", pixels, frame=1))
        self.assertTrue(evaluate_condition("b_avg == 0.25", pixels, frame=1))

    def test_min_and_max(self):
        pixels = self._pixels()
        self.assertTrue(evaluate_condition("max == 1 and min == 0", pixels, frame=1))

    def test_unknown_variable_is_rejected(self):
        with self.assertRaises(ExpressionError):
            evaluate_condition("unknown_thing > 0", self._pixels(), frame=1)

    def test_attribute_and_import_are_rejected(self):
        with self.assertRaises(ExpressionError):
            evaluate_condition("__import__('os').system('echo hi')", self._pixels(), frame=1)
        with self.assertRaises(ExpressionError):
            evaluate_condition("avg.__class__", self._pixels(), frame=1)

    def test_boolop_and_ifexp(self):
        pixels = self._pixels()
        self.assertTrue(evaluate_condition("avg >= 0 and avg <= 1", pixels, frame=1))
        self.assertTrue(evaluate_condition("1 if avg >= 0 else 0", pixels, frame=1))


if __name__ == "__main__":
    unittest.main()
