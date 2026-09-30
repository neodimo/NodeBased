"""Profile (2D parity plan 15, step F1 part 2): Nuke's in-graph performance probe.

A pass-through node that times its own nested `Evaluator.evaluate_raster` call on the subgraph
wired into it, recording wall time and the delta against `Evaluator.hits` (the cache-hit counter)
into `Evaluator.profile_log`, session-only like `hits`/`misses` themselves.
"""
import time
import unittest
from unittest import mock

import numpy as np

from nodebased.core import Dispatcher, bypass_slot
from nodebased.imaging import Evaluator
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


def tile_pixels(document, target, tile_edge=None):
    kwargs = {} if tile_edge is None else {"tile_edge": tile_edge}
    executor = TileExecutor(evaluator=Evaluator(), **kwargs)
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def build_checker_profile():
    g = Graph()
    g.add("plate", "Checker", dict(width=64, height=48, size=8))
    g.add("node", "Profile", {}, image="plate")
    return g


class ProfilePassthroughTests(unittest.TestCase):
    def test_output_equals_input_on_both_paths(self):
        g = build_checker_profile()
        evaluator = Evaluator()
        plate = evaluator_pixels(evaluator, g.doc, "plate")
        np.testing.assert_array_equal(evaluator_pixels(Evaluator(), g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node", tile_edge=16), plate)

    def test_bypassed_and_enabled_both_pass_the_input(self):
        g = build_checker_profile()
        plate = evaluator_pixels(Evaluator(), g.doc, "plate")
        g.d.execute(dict(op="disable", id="node", value=True))
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
        np.testing.assert_array_equal(evaluator_pixels(Evaluator(), g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node"), plate)


class ProfileMeasurementTests(unittest.TestCase):
    def build(self, exposure=0.0):
        g = Graph()
        g.add("plate", "Checker", dict(width=64, height=48, size=8))
        g.add("grade", "Grade", dict(exposure=exposure), image="plate")
        g.add("node", "Profile", {}, image="grade")
        return g

    def test_a_deliberately_slow_upstream_node_shows_the_larger_time(self):
        fast_doc = self.build(exposure=0.1).doc
        slow_doc = self.build(exposure=0.2).doc

        fast_evaluator = Evaluator()
        fast_evaluator.evaluate_raster(fast_doc, "node", frame=1)
        fast_ms = fast_evaluator.profile_log["node"][-1]["wall_time_ms"]

        real_grade = Evaluator._grade

        def slow_grade(image, p):
            time.sleep(0.05)
            return real_grade(image, p)

        slow_evaluator = Evaluator()
        with mock.patch.object(Evaluator, "_grade", staticmethod(slow_grade)):
            slow_evaluator.evaluate_raster(slow_doc, "node", frame=1)
        slow_ms = slow_evaluator.profile_log["node"][-1]["wall_time_ms"]

        self.assertGreater(slow_ms, fast_ms + 20)

    def test_a_cache_hit_on_the_second_evaluation_is_counted(self):
        doc = self.build(exposure=0.3).doc
        evaluator = Evaluator()

        evaluator.evaluate_raster(doc, "node", frame=1)
        first = evaluator.profile_log["node"][-1]
        self.assertEqual(first["cache_hits"], 0)

        evaluator.evaluate_raster(doc, "node", frame=1)
        second = evaluator.profile_log["node"][-1]
        self.assertGreater(second["cache_hits"], 0)

    def test_each_evaluation_appends_one_row_per_frame(self):
        doc = self.build().doc
        evaluator = Evaluator()
        evaluator.evaluate_raster(doc, "node", frame=1)
        evaluator.evaluate_raster(doc, "node", frame=1)
        rows = evaluator.profile_log["node"]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["frame"] for row in rows], [1, 1])

    def test_profile_log_is_not_populated_before_any_evaluation(self):
        evaluator = Evaluator()
        self.assertEqual(evaluator.profile_log, {})


if __name__ == "__main__":
    unittest.main()
