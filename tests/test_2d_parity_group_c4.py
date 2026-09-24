"""Lane L2 step 2c4: TimeOffset, FrameHold, Retime. Worked-example assertions against an animated
upstream, evaluator/tile-path parity (via the documented full-frame fallback), cache-digest proof,
bypass and CHOICES/LIMITS coverage. See docs/PARITY_2D.md for the audit these flip."""
import unittest

import numpy as np

from nodebased.core import Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_C4 = ("TimeOffset", "FrameHold", "Retime")


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def key(self, node_id, param, frame, value):
        self.d.execute(dict(op="set_key", id=node_id, param=param, frame=frame, value=value))

    def bypass(self, key, value=True):
        self.d.execute(dict(op="disable", id=key, value=value))

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target, frame):
    return Evaluator().evaluate(dict(document, view=target), target, frame=frame)


def tile_fallback_pixels(document, target, frame):
    """These three kinds evaluate their input at a different frame, which the tile executor has
    no per-tile notion of (like Transform/Crop/Mirror before them), so they fall back to the
    full-frame evaluator. Asserts the fallback actually happened."""
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert not executor.supports_tiled(document, target), target
    result = executor.compose(document, target, frame=frame, tier=1)
    assert result.tiled is False
    return result.pixels


def animated_plate(g, node_id="plate", first_value=0.2, keys=()):
    """A Constant whose red channel is keyed per frame -- so a time-remapping node's output at a
    given timeline frame proves *which* source frame it actually sampled, not just that it copied
    pixels through."""
    g.add(node_id, "Constant", dict(red=first_value, green=0.4, blue=0.6, alpha=1.0))
    for frame, value in keys:
        g.key(node_id, "red", frame, value)
    return node_id


class WorkedExampleTests(unittest.TestCase):
    """The brief's own worked examples, against an animated upstream."""

    def test_time_offset_two_at_frame_ten_shows_frame_eight(self):
        g = Graph()
        animated_plate(g, keys=((8, 0.81), (10, 0.42)))
        g.add("node", "TimeOffset", dict(time_offset=2), image="plate")
        out = evaluator_pixels(g.doc, "node", frame=10)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.81, places=5)

    def test_time_offset_reverse_flips_the_applied_direction(self):
        g = Graph()
        animated_plate(g, keys=((8, 0.81), (12, 0.33)))
        g.add("node", "TimeOffset", dict(time_offset=2, reverse=1), image="plate")
        # reverse: effective = frame + time_offset, so frame 10 samples frame 12.
        out = evaluator_pixels(g.doc, "node", frame=10)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.33, places=5)

    def test_frame_hold_at_five_shows_frame_five_at_five_six_and_twenty(self):
        g = Graph()
        animated_plate(g, keys=((5, 0.55), (6, 0.66), (20, 0.99)))
        g.add("node", "FrameHold", dict(first_frame=5, increment=0), image="plate")
        for frame in (5, 6, 20):
            with self.subTest(frame=frame):
                out = evaluator_pixels(g.doc, "node", frame=frame)
                self.assertAlmostEqual(float(out[0, 0, 0]), 0.55, places=5)

    def test_frame_hold_with_increment_steps_forward(self):
        g = Graph()
        animated_plate(g, keys=((5, 0.1), (10, 0.5), (15, 0.9)))
        g.add("node", "FrameHold", dict(first_frame=5, increment=5), image="plate")
        # frame 12 holds at the nearest first_frame + k*increment at or before it: 5 + 1*5 = 10.
        out = evaluator_pixels(g.doc, "node", frame=12)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.5, places=5)
        # frame 3, before first_frame, clamps to first_frame itself.
        out = evaluator_pixels(g.doc, "node", frame=3)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.1, places=5)

    def test_retime_speed_two_at_output_frame_four_shows_input_frame_eight(self):
        g = Graph()
        animated_plate(g, keys=((8, 0.77), (4, 0.11)))
        g.add("node", "Retime", dict(speed=2.0), image="plate")
        out = evaluator_pixels(g.doc, "node", frame=4)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.77, places=5)

    def test_retime_range_starts_shift_the_anchor(self):
        g = Graph()
        animated_plate(g, keys=((11, 0.42),))
        g.add("node", "Retime", dict(input_range_start=1, output_range_start=1, speed=2.0),
             image="plate")
        # effective = 1 + (frame - 1) * 2; frame 6 -> 1 + 5*2 = 11.
        out = evaluator_pixels(g.doc, "node", frame=6)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.42, places=5)


class TilePathParityTests(unittest.TestCase):
    """Excluded from the tile path (per-tile frame remapping has no meaning here, like
    Transform/Crop/Mirror), so both paths still must agree via the full-frame fallback."""

    def test_all_three_kinds_fall_back_and_match_the_evaluator(self):
        for kind, params in (("TimeOffset", dict(time_offset=3)),
                             ("FrameHold", dict(first_frame=2, increment=0)),
                             ("Retime", dict(speed=0.5))):
            with self.subTest(kind=kind):
                self.assertNotIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                animated_plate(g, keys=((1, 0.15), (2, 0.25), (5, 0.5), (10, 0.9)))
                g.add("node", kind, params, image="plate")
                ev = evaluator_pixels(g.doc, "node", frame=6)
                ti = tile_fallback_pixels(g.doc, "node", frame=6)
                np.testing.assert_allclose(ti, ev, atol=1e-6)


class BypassTests(unittest.TestCase):
    """Bypassing any of the three kinds passes the input through at the *current*, un-remapped
    frame -- not the frame the node would otherwise have requested."""

    def visible_params(self, kind):
        return {"TimeOffset": dict(time_offset=4), "FrameHold": dict(first_frame=1, increment=0),
               "Retime": dict(speed=3.0)}[kind]

    def test_bypassed_kind_passes_the_current_frame_not_the_remapped_one(self):
        for kind in GROUP_C4:
            with self.subTest(kind=kind):
                g = Graph()
                animated_plate(g, keys=((1, 0.1), (5, 0.5), (20, 0.2)))
                g.add("node", kind, self.visible_params(kind), image="plate")
                current = evaluator_pixels(g.doc, "plate", frame=5)
                remapped = evaluator_pixels(g.doc, "node", frame=5)
                self.assertFalse(np.array_equal(remapped, current), f"{kind} must visibly remap")
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node", frame=5), current))
                self.assertTrue(np.array_equal(tile_fallback_pixels(g.doc, "node", frame=5), current))

    def test_bypass_slot_is_the_single_image_input(self):
        for kind in GROUP_C4:
            with self.subTest(kind=kind):
                node = dict(type=kind, inputs={"image": "x"})
                self.assertEqual(bypass_slot(node), "image")


class CacheTests(unittest.TestCase):
    """docs/TIME_MODEL.md: the cache key must change when the effective source frame changes,
    and must not change when it does not (proven, not asserted)."""

    def test_static_source_produces_no_extra_misses_while_scrubbed(self):
        g = Graph()
        g.add("plate", "Constant", dict(red=0.4, green=0.4, blue=0.4, alpha=1.0))
        g.add("node", "TimeOffset", dict(time_offset=3), image="plate")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        ev.evaluate(doc, "node", frame=1)
        misses_after_first = ev.misses
        for frame in (5, 20, -4, 100):
            ev.evaluate(doc, "node", frame=frame)
        self.assertEqual(ev.misses, misses_after_first,
                         "scrubbing a still through TimeOffset must not miss the cache again")

    def test_frame_hold_revisiting_the_same_effective_frame_hits_the_cache(self):
        g = Graph()
        animated_plate(g, keys=((5, 0.3), (10, 0.7)))
        g.add("node", "FrameHold", dict(first_frame=5, increment=0), image="plate")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        ev.evaluate(doc, "node", frame=5)
        misses_after_five = ev.misses
        ev.evaluate(doc, "node", frame=6)   # same effective frame (5): no new miss
        ev.evaluate(doc, "node", frame=20)  # still the same effective frame
        self.assertEqual(ev.misses, misses_after_five)

    def test_effective_frame_change_with_different_content_does_miss_then_hits_on_revisit(self):
        g = Graph()
        animated_plate(g, keys=((8, 0.2), (12, 0.9)))
        g.add("node", "TimeOffset", dict(time_offset=2), image="plate")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        first = ev.evaluate(doc, "node", frame=10)   # effective 8
        misses_at_ten = ev.misses
        second = ev.evaluate(doc, "node", frame=14)  # effective 12, different content
        misses_at_fourteen = ev.misses
        self.assertGreater(misses_at_fourteen, misses_at_ten)
        self.assertFalse(np.array_equal(first, second))
        third = ev.evaluate(doc, "node", frame=10)   # effective 8 again: cache hit
        self.assertEqual(ev.misses, misses_at_fourteen)
        np.testing.assert_array_equal(first, third)


class SpecCoverageTests(unittest.TestCase):
    def test_nodes_are_registered_with_a_single_image_input_and_no_mask_or_mix(self):
        for kind in GROUP_C4:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertEqual(SPECS[kind]["inputs"], ["image"])
                self.assertNotIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertNotIn("mix", SPECS[kind]["params"])

    def test_limits_cover_every_new_param(self):
        for name in ("time_offset", "reverse", "first_frame", "increment",
                    "input_range_start", "input_range_end",
                    "output_range_start", "output_range_end", "speed"):
            self.assertIn(name, LIMITS)

    def test_dispatcher_creates_every_new_node_with_valid_defaults(self):
        for kind in GROUP_C4:
            with self.subTest(kind=kind):
                d = Dispatcher()
                result = d.execute(dict(op="create", type=kind))
                self.assertIn("id", result["result"])


if __name__ == "__main__":
    unittest.main()
