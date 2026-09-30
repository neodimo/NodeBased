"""Lane L2 (2D parity) plan 14, step E1: TimeWarp, and TimeBlur/TimeEcho on the tile path with
cached fractional shutter samples. See docs/PARITY_2D.md's Time table for the audit this flips.

Reuses the Graph/animated_plate/evaluator_pixels/tile_fallback_pixels helpers from step 2c4's
test module (package-qualified import, per COMMON.md: the integrator runs modules by dotted
name and a bare `from test_2d_parity_group_c4 import ...` fails there)."""
import unittest

import numpy as np

from nodebased.core import LIMITS, CHOICES, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS
from tests.test_2d_parity_group_c4 import Graph, animated_plate, evaluator_pixels, tile_fallback_pixels


def keyed(g, node_id, param, frame, value, interpolation="linear"):
    """Like `Graph.key`, but exposes the curve interpolation (needed for TimeWarp's "hold" case,
    a single constant-interpolation key held everywhere by the curve's own endpoint-hold rule)."""
    g.d.execute(dict(op="set_key", id=node_id, param=param, frame=frame, value=value,
                     interpolation=interpolation))


class TimeWarpWorkedExampleTests(unittest.TestCase):
    """The brief's own worked examples: a 0.5 speed change over 20 frames, a reverse, a hold, and
    a fractional lookup with blend -- each against an animated upstream plate whose keyed red
    channel proves *which* input frame TimeWarp actually sampled."""

    def test_no_curve_at_all_is_identity(self):
        g = Graph()
        animated_plate(g, keys=((1, 0.1), (5, 0.5), (9, 0.9)))
        g.add("node", "TimeWarp", dict(), image="plate")
        for frame in (1, 5, 9):
            with self.subTest(frame=frame):
                out = evaluator_pixels(g.doc, "node", frame=frame)
                plate = evaluator_pixels(g.doc, "plate", frame=frame)
                np.testing.assert_array_equal(out, plate)

    def test_speed_half_over_twenty_frames(self):
        # lookup(1) = 1, lookup(21) = 11: output frame advances twice as fast as input frame.
        g = Graph()
        animated_plate(g, keys=((1, 0.1), (6, 0.6), (11, 0.9)))
        g.add("node", "TimeWarp", dict(), image="plate")
        keyed(g, "node", "lookup", 1, 1.0)
        keyed(g, "node", "lookup", 21, 11.0)
        # output frame 11 -> lookup 1 + (11-1)*0.5 = 6 -> the plate's keyed value at frame 6.
        out = evaluator_pixels(g.doc, "node", frame=11)
        expected = evaluator_pixels(g.doc, "plate", frame=6)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_reverse(self):
        # lookup(1) = 20, lookup(20) = 1: output plays the input backwards.
        g = Graph()
        animated_plate(g, keys=((1, 0.05), (12, 0.77), (20, 0.4)))
        g.add("node", "TimeWarp", dict(), image="plate")
        keyed(g, "node", "lookup", 1, 20.0)
        keyed(g, "node", "lookup", 20, 1.0)
        # output frame 9 -> lookup 20 - (9-1) = 12 -> the plate's keyed value at frame 12.
        out = evaluator_pixels(g.doc, "node", frame=9)
        expected = evaluator_pixels(g.doc, "plate", frame=12)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_hold(self):
        # A single constant-interpolation key: the curve's own endpoint hold (animation.py) means
        # every output frame reads the same input frame, exactly like Nuke freezing a TimeWarp.
        g = Graph()
        animated_plate(g, keys=((1, 0.15), (7, 0.65), (30, 0.95)))
        g.add("node", "TimeWarp", dict(), image="plate")
        keyed(g, "node", "lookup", 1, 7.0, interpolation="constant")
        expected = evaluator_pixels(g.doc, "plate", frame=7)
        for frame in (1, 2, 15, 30):
            with self.subTest(frame=frame):
                out = evaluator_pixels(g.doc, "node", frame=frame)
                np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_fractional_lookup_with_filter_none_passes_the_exact_fractional_frame(self):
        # A constant-interpolation key holds lookup at 5.3 everywhere; "none" must sample the
        # plate at exactly frame 5.3 (its own curve is linear, so this is a real fractional read,
        # not a rounded one), matching a direct evaluation at that same fractional frame.
        g = Graph()
        animated_plate(g, keys=((5, 0.2), (6, 0.8)))
        g.add("node", "TimeWarp", dict(lookup_filter="none"), image="plate")
        keyed(g, "node", "lookup", 1, 5.3, interpolation="constant")
        out = evaluator_pixels(g.doc, "node", frame=1)
        expected = evaluator_pixels(g.doc, "plate", frame=5.3)
        np.testing.assert_array_equal(out, expected)
        # And it must actually differ from the nearest-integer neighbours, proving genuine
        # fractional sampling rather than a silent round.
        self.assertFalse(np.array_equal(out, evaluator_pixels(g.doc, "plate", frame=5)))
        self.assertFalse(np.array_equal(out, evaluator_pixels(g.doc, "plate", frame=6)))

    def test_fractional_lookup_with_filter_blend_matches_the_hand_worked_mix(self):
        # lookup held at 5.3: "blend" must be 0.7 * frame(5) + 0.3 * frame(6).
        g = Graph()
        animated_plate(g, keys=((5, 0.2), (6, 0.8)))
        g.add("node", "TimeWarp", dict(lookup_filter="blend"), image="plate")
        keyed(g, "node", "lookup", 1, 5.3, interpolation="constant")
        out = evaluator_pixels(g.doc, "node", frame=1)
        frame5 = evaluator_pixels(g.doc, "plate", frame=5)
        frame6 = evaluator_pixels(g.doc, "plate", frame=6)
        expected = frame5 * np.float32(0.7) + frame6 * np.float32(0.3)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_integer_lookup_is_the_same_under_either_filter(self):
        g = Graph()
        animated_plate(g, keys=((1, 0.1), (5, 0.9)))
        for lookup_filter in ("none", "blend"):
            with self.subTest(lookup_filter=lookup_filter):
                g2 = Graph()
                animated_plate(g2, keys=((1, 0.1), (5, 0.9)))
                g2.add("node", "TimeWarp", dict(lookup_filter=lookup_filter), image="plate")
                keyed(g2, "node", "lookup", 1, 5.0, interpolation="constant")
                out = evaluator_pixels(g2.doc, "node", frame=1)
                expected = evaluator_pixels(g2.doc, "plate", frame=5)
                np.testing.assert_array_equal(out, expected)


class TimeWarpTilePathAndBypassTests(unittest.TestCase):
    """TimeWarp evaluates its input at a different frame, exactly like TimeOffset/FrameHold/
    Retime before it: excluded from the tile path for the same documented reason, falling back
    to the full-frame evaluator, which the tile executor's own fallback asserts equal to."""

    def test_excluded_from_the_tile_path_and_falls_back_to_the_evaluator(self):
        self.assertNotIn("TimeWarp", SUPPORTED_TILED_KINDS)
        g = Graph()
        animated_plate(g, keys=((1, 0.1), (2, 0.2), (5, 0.5), (10, 0.9)))
        g.add("node", "TimeWarp", dict(), image="plate")
        keyed(g, "node", "lookup", 1, 1.0)
        keyed(g, "node", "lookup", 11, 6.0)
        ev = evaluator_pixels(g.doc, "node", frame=6)
        ti = tile_fallback_pixels(g.doc, "node", frame=6)
        np.testing.assert_allclose(ti, ev, atol=1e-6)

    def test_bypass_passes_the_current_frame_unremapped(self):
        g = Graph()
        animated_plate(g, keys=((1, 0.1), (5, 0.5), (20, 0.2)))
        g.add("node", "TimeWarp", dict(), image="plate")
        keyed(g, "node", "lookup", 1, 20.0)
        keyed(g, "node", "lookup", 20, 1.0)
        current = evaluator_pixels(g.doc, "plate", frame=5)
        remapped = evaluator_pixels(g.doc, "node", frame=5)
        self.assertFalse(np.array_equal(remapped, current), "TimeWarp must visibly remap")
        g.bypass("node")
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node", frame=5), current))
        self.assertTrue(np.array_equal(tile_fallback_pixels(g.doc, "node", frame=5), current))

    def test_bypass_slot_is_the_single_image_input(self):
        node = dict(type="TimeWarp", inputs={"image": "x"})
        self.assertEqual(bypass_slot(node), "image")


class TimeWarpSpecCoverageTests(unittest.TestCase):
    def test_registered_with_a_single_image_input_and_no_mask_or_mix(self):
        self.assertIn("TimeWarp", SPECS)
        self.assertEqual(SPECS["TimeWarp"]["inputs"], ["image"])
        self.assertNotIn("mask", SPECS["TimeWarp"].get("optional_inputs", []))
        self.assertNotIn("mix", SPECS["TimeWarp"]["params"])

    def test_limits_and_choices_cover_the_new_params(self):
        self.assertIn("lookup", LIMITS)
        self.assertEqual(CHOICES["lookup_filter"], ["none", "blend"])


def _small_constant_graph(node_id="node", kind="TimeBlur", params=None, keys=((1, 0.1), (2, 0.5), (3, 0.9), (4, 0.2))):
    """A small-canvas animated plate feeding `kind`, small enough that a small `tile_edge` still
    produces several tiles quickly."""
    g = Graph()
    g.add("plate", "Constant", dict(width=64, height=64, red=keys[0][1], green=0.4, blue=0.6, alpha=1.0))
    for frame, value in keys:
        g.key("plate", "red", frame, value)
    g.add(node_id, kind, params or {}, image="plate")
    return g


class TemporalTilePathTests(unittest.TestCase):
    """TimeBlur/TimeEcho move onto the tile path this step: `_temporal_tile` solves the node's
    already-blended result once (the same call the old full-frame fallback made for the whole
    graph) and slices tiles from it, so a tiled render must be pixel-identical to the full-frame
    evaluator (tiles equal to full-frame)."""

    def test_timeblur_and_timeecho_are_supported_tiled_kinds(self):
        self.assertIn("TimeBlur", SUPPORTED_TILED_KINDS)
        self.assertIn("TimeEcho", SUPPORTED_TILED_KINDS)

    def test_timeblur_tiled_result_matches_full_frame(self):
        g = _small_constant_graph(kind="TimeBlur",
                                  params=dict(shutter=1.0, divisions=6, shutter_offset="centred", custom_offset=0.0))
        doc = dict(g.doc, view="node")
        expected = Evaluator().evaluate(doc, "node", frame=3)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(doc, "node"))
        result = executor.compose(doc, "node", frame=3, tier=1)
        self.assertTrue(result.tiled)
        np.testing.assert_allclose(result.pixels, expected, atol=1e-6)

    def test_timeecho_tiled_result_matches_full_frame(self):
        g = _small_constant_graph(kind="TimeEcho", params=dict(frames=3, method="average", falloff=1.0))
        doc = dict(g.doc, view="node")
        expected = Evaluator().evaluate(doc, "node", frame=4)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(doc, "node"))
        result = executor.compose(doc, "node", frame=4, tier=1)
        self.assertTrue(result.tiled)
        np.testing.assert_allclose(result.pixels, expected, atol=1e-6)

    def test_timeblur_bypass_on_the_tile_path_matches_the_evaluator(self):
        g = _small_constant_graph(kind="TimeBlur",
                                  params=dict(shutter=1.0, divisions=4, shutter_offset="centred", custom_offset=0.0))
        g.bypass("node")
        doc = dict(g.doc, view="node")
        expected = Evaluator().evaluate(doc, "node", frame=3)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        result = executor.compose(doc, "node", frame=3, tier=1)
        self.assertTrue(result.tiled)
        np.testing.assert_allclose(result.pixels, expected, atol=1e-6)


class TemporalCacheTests(unittest.TestCase):
    """docs/TIME_MODEL.md: TimeBlur's shutter subframes are fractional (a centred, one-frame
    shutter at an integer outer frame samples none of its own integer neighbours), and were
    deliberately never cached -- this step retains them, scoped to TimeBlur alone, so a second
    evaluation of the same frame reads them from cache instead of re-walking upstream."""

    def test_timeblur_second_evaluation_of_the_same_frame_hits_the_cache(self):
        g = _small_constant_graph(kind="TimeBlur",
                                  params=dict(shutter=1.0, divisions=4, shutter_offset="centred", custom_offset=0.0))
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        first = ev.evaluate(doc, "node", frame=3)
        misses_after_first = ev.misses
        second = ev.evaluate(doc, "node", frame=3)
        self.assertEqual(ev.misses, misses_after_first,
                         "a second TimeBlur evaluation at the same frame must not re-walk its "
                         "shutter subframes")
        np.testing.assert_array_equal(first, second)

    def test_motion_blur2d_keeps_the_old_ephemeral_behaviour(self):
        # The scoping is TimeBlur-only (`cache_fractional=(kind == "TimeBlur")`): a sibling
        # temporal kind sharing the same sampling code must still re-walk every time, proving
        # this is a deliberate, narrow change and not a global loosening of the fractional gate.
        g = _small_constant_graph(kind="MotionBlur2D",
                                  params=dict(shutter=1.0, shutter_offset="centred", custom_offset=0.0,
                                             samples=4, mix=1.0))
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        ev.evaluate(doc, "node", frame=3)
        misses_after_first = ev.misses
        ev.evaluate(doc, "node", frame=3)
        self.assertGreater(ev.misses, misses_after_first,
                           "MotionBlur2D's shutter subframes must stay ephemeral")

    def test_timeblur_tile_path_second_compose_reuses_the_shared_evaluator_cache(self):
        g = _small_constant_graph(kind="TimeBlur",
                                  params=dict(shutter=1.0, divisions=4, shutter_offset="centred", custom_offset=0.0))
        doc = dict(g.doc, view="node")
        ev = Evaluator()
        executor = TileExecutor(evaluator=ev, tile_edge=16)
        first = executor.compose(doc, "node", frame=3, tier=1)
        self.assertTrue(first.tiled)
        misses_after_first = ev.misses
        second = executor.compose(doc, "node", frame=3, tier=1)
        self.assertTrue(second.tiled)
        self.assertEqual(ev.misses, misses_after_first,
                         "a second compose at the same frame must not re-walk TimeBlur's "
                         "shutter subframes, even though `_source_cache` itself was cleared")
        np.testing.assert_array_equal(first.pixels, second.pixels)


if __name__ == "__main__":
    unittest.main()
