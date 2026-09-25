"""Lane L2 step 4b: TimeClip, FrameRange, AppendClip. Hand-computed frame mappings against an
animated upstream (a plate whose red channel encodes the frame it was sampled at), the tile-path
full-frame fallback, cache behaviour and bypass. See docs/PARITY_2D.md for the audit these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_4B = ("TimeClip", "FrameRange", "AppendClip")


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
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert not executor.supports_tiled(document, target), target
    result = executor.compose(document, target, frame=frame, tier=1)
    assert result.tiled is False
    return result.pixels


def frame_plate(g, node_id, base=0.0, scale=0.05, frames=range(1, 21)):
    """A Constant whose red is keyed to ``base + scale * frame`` at every frame in ``frames``, so a
    pixel's red says exactly which source frame was sampled."""
    g.add(node_id, "Constant", dict(red=base, green=0.4, blue=0.6, alpha=1.0))
    for frame in frames:
        g.key(node_id, "red", frame, base + scale * frame)
    return node_id


def red_at(g, target, frame):
    return float(evaluator_pixels(g.doc, target, frame)[0, 0, 0])


class TimeClipTests(unittest.TestCase):
    def clip(self, **params):
        g = Graph()
        frame_plate(g, "plate")
        g.add("node", "TimeClip", dict(first=5, last=8, **params), image="plate")
        return g

    def test_inside_the_range_passes_the_frame_through(self):
        g = self.clip()
        for frame in (5, 6, 8):
            self.assertAlmostEqual(red_at(g, "node", frame), 0.05 * frame, places=5)

    def test_hold_repeats_the_last_and_first_frame_outside_the_range(self):
        g = self.clip()
        self.assertAlmostEqual(red_at(g, "node", 20), 0.4, places=5)   # after: frame 8
        self.assertAlmostEqual(red_at(g, "node", 2), 0.25, places=5)   # before: frame 5

    def test_black_gives_a_transparent_frame_of_the_same_size(self):
        g = self.clip(before="black", after="black")
        inside = evaluator_pixels(g.doc, "node", 8)
        self.assertAlmostEqual(float(inside[0, 0, 0]), 0.4, places=5)
        for frame in (9, 20, 4, 1):
            with self.subTest(frame=frame):
                out = evaluator_pixels(g.doc, "node", frame)
                self.assertEqual(out.shape, inside.shape)
                self.assertFalse(out.any(), "black must be all zeros, alpha included")

    def test_before_and_after_are_independent(self):
        g = self.clip(before="black", after="hold")
        self.assertFalse(evaluator_pixels(g.doc, "node", 3).any())
        self.assertAlmostEqual(red_at(g, "node", 12), 0.4, places=5)

    def test_loop_wraps_over_the_four_frame_range(self):
        g = self.clip(before="loop", after="loop")
        # range 5..8 has period 4: frame 9 -> 5, 10 -> 6, 12 -> 8, 13 -> 5; frame 4 -> 8, 1 -> 5.
        for frame, source in ((9, 5), (10, 6), (12, 8), (13, 5), (4, 8), (1, 5)):
            with self.subTest(frame=frame):
                self.assertAlmostEqual(red_at(g, "node", frame), 0.05 * source, places=5)

    def test_bounce_runs_forward_then_back_without_repeating_the_end_frames(self):
        g = self.clip(before="bounce", after="bounce")
        # 5 6 7 8 7 6 5 6 7 8 ... : frame 9 -> 7, 10 -> 6, 11 -> 5, 12 -> 6, 14 -> 8, 15 -> 7.
        for frame, source in ((9, 7), (10, 6), (11, 5), (12, 6), (14, 8), (15, 7),
                              (4, 6), (3, 7)):   # before the range it mirrors the other way
            with self.subTest(frame=frame):
                self.assertAlmostEqual(red_at(g, "node", frame), 0.05 * source, places=5)

    def test_time_offset_shifts_before_the_range_is_applied(self):
        g = Graph()
        frame_plate(g, "plate")
        g.add("node", "TimeClip", dict(frame_range_type="all", time_offset=2), image="plate")
        self.assertAlmostEqual(red_at(g, "node", 10), 0.4, places=5)   # frame 10 - 2 = 8
        g.d.execute(dict(op="set", id="node", param="frame_range_type", value="custom"))
        g.d.execute(dict(op="set", id="node", param="first", value=1))
        g.d.execute(dict(op="set", id="node", param="last", value=6))
        # 10 - 2 = 8 is past the 1..6 range, so hold gives frame 6.
        self.assertAlmostEqual(red_at(g, "node", 10), 0.3, places=5)
        self.assertAlmostEqual(red_at(g, "node", 4), 0.1, places=5)   # 4 - 2 = 2 -> frame 2

    def test_frame_range_type_all_ignores_the_range(self):
        g = self.clip(frame_range_type="all", after="black")
        self.assertAlmostEqual(red_at(g, "node", 15), 0.75, places=5)


class FrameRangeTests(unittest.TestCase):
    def test_clamps_to_the_range(self):
        g = Graph()
        frame_plate(g, "plate")
        g.add("node", "FrameRange", dict(first_frame=3, last_frame=6), image="plate")
        for frame, source in ((1, 3), (3, 3), (4, 4), (6, 6), (10, 6)):
            with self.subTest(frame=frame):
                self.assertAlmostEqual(red_at(g, "node", frame), 0.05 * source, places=5)

    def test_black_outside_and_loop_use_the_same_policies(self):
        g = Graph()
        frame_plate(g, "plate")
        g.add("node", "FrameRange", dict(first_frame=3, last_frame=6, before="black", after="loop"),
              image="plate")
        self.assertFalse(evaluator_pixels(g.doc, "node", 2).any())
        self.assertAlmostEqual(red_at(g, "node", 7), 0.15, places=5)   # loops to frame 3
        self.assertAlmostEqual(red_at(g, "node", 6), 0.30, places=5)


class AppendClipTests(unittest.TestCase):
    def two_clips(self, first_frame=1):
        """A: frames 1..4 of plate a (red = f/20). B: frames 10..12 of plate b (red = 0.5 + f/100).
        The clips' lengths come from the FrameRange nodes directly upstream."""
        g = Graph()
        frame_plate(g, "plate_a")
        frame_plate(g, "plate_b", base=0.5, scale=0.01)
        g.add("a", "FrameRange", dict(first_frame=1, last_frame=4), image="plate_a")
        g.add("b", "FrameRange", dict(first_frame=10, last_frame=12), image="plate_b")
        g.add("node", "AppendClip", dict(first_frame=first_frame), clip0="a", clip1="b")
        return g

    def test_plays_clip_a_then_clip_b_at_the_computed_boundary(self):
        g = self.two_clips()
        # A takes timeline frames 1..4 (source 1..4); B takes 5..7 (source 10..12).
        for frame, expected in ((1, 0.05), (4, 0.20), (5, 0.60), (6, 0.61), (7, 0.62)):
            with self.subTest(frame=frame):
                self.assertAlmostEqual(red_at(g, "node", frame), expected, places=5)

    def test_holds_the_first_and_last_frame_outside_the_timeline(self):
        g = self.two_clips()
        self.assertAlmostEqual(red_at(g, "node", 8), 0.62, places=5)
        self.assertAlmostEqual(red_at(g, "node", -30), 0.05, places=5)

    def test_first_frame_moves_the_whole_sequence(self):
        g = self.two_clips(first_frame=10)
        self.assertAlmostEqual(red_at(g, "node", 10), 0.05, places=5)
        self.assertAlmostEqual(red_at(g, "node", 13), 0.20, places=5)
        self.assertAlmostEqual(red_at(g, "node", 14), 0.60, places=5)

    def test_length_knobs_serve_clips_with_no_known_range(self):
        g = Graph()
        frame_plate(g, "a")
        frame_plate(g, "b", base=0.5, scale=0.01)
        g.add("node", "AppendClip", dict(length0=3, length1=2), clip0="a", clip1="b")
        # Unknown ranges sample from frame 1: A is timeline 1..3 (source 1..3), B is 4..5 (source 1..2).
        for frame, expected in ((3, 0.15), (4, 0.51), (5, 0.52), (9, 0.52)):
            with self.subTest(frame=frame):
                self.assertAlmostEqual(red_at(g, "node", frame), expected, places=5)

    def test_empty_slots_are_skipped_and_length_zero_drops_a_clip(self):
        g = Graph()
        frame_plate(g, "a")
        frame_plate(g, "b", base=0.5, scale=0.01)
        frame_plate(g, "c", base=0.9, scale=0.0)
        g.add("node", "AppendClip", dict(length0=2, length1=0, length4=2), clip0="a", clip1="b", clip4="c")
        self.assertAlmostEqual(red_at(g, "node", 2), 0.10, places=5)
        self.assertAlmostEqual(red_at(g, "node", 3), 0.9, places=5)   # b was skipped

    def test_dissolve_cross_fades_the_overlap_with_hand_computed_weights(self):
        g = Graph()
        g.add("a", "Constant", dict(red=0.2, green=0.4, blue=0.6, alpha=1.0))
        g.add("b", "Constant", dict(red=0.8, green=0.4, blue=0.6, alpha=1.0))
        g.add("node", "AppendClip", dict(length0=4, length1=4, dissolve=2), clip0="a", clip1="b")
        # B starts at 1 + 4 - 2 = 3. Frames 3 and 4 overlap: incoming weight (k + 1) / 3.
        expected = {1: 0.2, 2: 0.2, 3: 0.2 * (2 / 3) + 0.8 / 3, 4: 0.2 / 3 + 0.8 * (2 / 3), 5: 0.8, 6: 0.8}
        for frame, red in expected.items():
            with self.subTest(frame=frame):
                self.assertAlmostEqual(red_at(g, "node", frame), red, places=5)

    def test_dissolve_between_animated_clips_samples_both_clips_at_their_own_frames(self):
        g = Graph()
        frame_plate(g, "a")
        frame_plate(g, "b", base=0.5, scale=0.01)
        g.add("node", "AppendClip", dict(length0=4, length1=4, dissolve=1), clip0="a", clip1="b")
        # B starts at 4. Frame 4: A source 4 (0.20), B source 1 (0.51), weight 1/2.
        self.assertAlmostEqual(red_at(g, "node", 4), 0.5 * 0.20 + 0.5 * 0.51, places=5)

    def test_a_still_stays_one_cache_entry_per_clip(self):
        g = Graph()
        g.add("a", "Constant", dict(red=0.2, green=0.4, blue=0.6, alpha=1.0))
        g.add("b", "Constant", dict(red=0.8, green=0.4, blue=0.6, alpha=1.0))
        g.add("node", "AppendClip", dict(length0=3, length1=2), clip0="a", clip1="b")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        ev.evaluate(doc, "node", frame=1)
        self.assertEqual(ev.misses, 2)   # plate a and the AppendClip
        for frame in (2, 3, 1):
            ev.evaluate(doc, "node", frame=frame)
        self.assertEqual(ev.misses, 2, "scrubbing inside clip A must not miss the cache again")
        ev.evaluate(doc, "node", frame=4)
        self.assertEqual(ev.misses, 4)   # plate b and the AppendClip re-keyed on b
        for frame in (5, 4, 3, 2):
            ev.evaluate(doc, "node", frame=frame)
        self.assertEqual(ev.misses, 4)

    def test_nothing_wired_reports_an_error_enabled_or_bypassed(self):
        g = Graph()
        g.add("node", "AppendClip")
        with self.assertRaisesRegex(ValueError, "connect at least one clip"):
            evaluator_pixels(g.doc, "node", 1)
        g.bypass("node")
        with self.assertRaisesRegex(ValueError, "connect at least one clip"):
            evaluator_pixels(g.doc, "node", 1)


class TilePathTests(unittest.TestCase):
    def test_all_three_fall_back_to_the_full_frame_evaluator_and_match_it(self):
        for kind in GROUP_4B:
            self.assertNotIn(kind, SUPPORTED_TILED_KINDS)
        cases = []
        g = Graph()
        frame_plate(g, "plate")
        g.add("node", "TimeClip", dict(first=5, last=8, after="bounce", time_offset=1), image="plate")
        cases.append((g, 12))
        g = Graph()
        frame_plate(g, "plate")
        g.add("node", "FrameRange", dict(first_frame=3, last_frame=6), image="plate")
        cases.append((g, 9))
        g = Graph()
        frame_plate(g, "a")
        frame_plate(g, "b", base=0.5, scale=0.01)
        g.add("node", "AppendClip", dict(length0=4, length1=4, dissolve=2), clip0="a", clip1="b")
        cases.append((g, 3))
        for g, frame in cases:
            with self.subTest(kind=g.doc["nodes"]["node"]["type"]):
                np.testing.assert_allclose(tile_fallback_pixels(g.doc, "node", frame),
                                           evaluator_pixels(g.doc, "node", frame), atol=1e-6)


class BypassTests(unittest.TestCase):
    def test_bypassed_single_input_kinds_pass_the_current_frame(self):
        for kind, params in (("TimeClip", dict(first=5, last=8, time_offset=3)),
                             ("FrameRange", dict(first_frame=5, last_frame=8))):
            with self.subTest(kind=kind):
                g = Graph()
                frame_plate(g, "plate")
                g.add("node", kind, params, image="plate")
                current = evaluator_pixels(g.doc, "plate", 12)
                self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node", 12), current))
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node", 12), current))
                self.assertTrue(np.array_equal(tile_fallback_pixels(g.doc, "node", 12), current))

    def test_bypassed_append_clip_passes_the_first_wired_input(self):
        g = Graph()
        frame_plate(g, "a")
        frame_plate(g, "b", base=0.5, scale=0.01)
        g.add("node", "AppendClip", dict(length1=2, length5=2), clip1="a", clip5="b")
        enabled = evaluator_pixels(g.doc, "node", 12)
        g.bypass("node")
        current = evaluator_pixels(g.doc, "a", 12)   # clip1 is the first wired slot, at frame 12
        self.assertFalse(np.array_equal(enabled, current))
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node", 12), current))
        self.assertTrue(np.array_equal(tile_fallback_pixels(g.doc, "node", 12), current))

    def test_bypass_slot_names(self):
        self.assertEqual(bypass_slot(dict(type="AppendClip", inputs={"clip3": "x", "clip6": "y"})), "clip3")
        self.assertEqual(bypass_slot(dict(type="AppendClip", inputs={})), "clip0")
        for kind in ("TimeClip", "FrameRange"):
            self.assertEqual(bypass_slot(dict(type=kind, inputs={"image": "x"})), "image")

    def test_a_bypassed_remap_node_follows_edits_upstream(self):
        """The bypassed digest must include the passed-through input's own digest: with a shared
        evaluator an edit upstream has to show through a bypassed time node."""
        g = Graph()
        g.add("plate", "Constant", dict(red=0.3, green=0.4, blue=0.6, alpha=1.0))
        g.add("node", "TimeClip", dict(first=5, last=8), image="plate")
        g.bypass("node")
        ev = Evaluator()
        doc = dict(g.doc, view="node")
        self.assertAlmostEqual(float(ev.evaluate(doc, "node", frame=1)[0, 0, 0]), 0.3, places=5)
        g.d.execute(dict(op="set", id="plate", param="red", value=0.7))
        doc = dict(g.doc, view="node")
        self.assertAlmostEqual(float(ev.evaluate(doc, "node", frame=1)[0, 0, 0]), 0.7, places=5)


class SpecCoverageTests(unittest.TestCase):
    def test_registration(self):
        for kind in ("TimeClip", "FrameRange"):
            self.assertEqual(SPECS[kind]["inputs"], ["image"])
            self.assertNotIn("mix", SPECS[kind]["params"])
        self.assertEqual(SPECS["AppendClip"]["inputs"], [])
        self.assertEqual(SPECS["AppendClip"]["optional_inputs"], [f"clip{i}" for i in range(8)])

    def test_every_param_has_limits_and_enums_have_choices(self):
        for kind in GROUP_4B:
            for name in SPECS[kind]["params"]:
                with self.subTest(kind=kind, param=name):
                    self.assertTrue(name in LIMITS or name in CHOICES)
        for name in ("before", "after"):
            self.assertEqual(CHOICES[name], ["hold", "loop", "bounce", "black"])

    def test_knob_layout_covers_every_param_once(self):
        for kind in GROUP_4B:
            params = [p for group in knob_layout(kind) for p in group.params]
            self.assertEqual(sorted(params), sorted(SPECS[kind]["params"]), kind)

    def test_dispatcher_creates_validates_and_rejects_bad_choices(self):
        for kind in GROUP_4B:
            d = Dispatcher()
            self.assertIn("id", d.execute(dict(op="create", type=kind))["result"])
        d = Dispatcher()
        node = d.execute(dict(op="create", type="TimeClip"))["result"]["id"]
        d.execute(dict(op="set", id=node, param="before", value="bounce"))
        self.assertEqual(d.document["nodes"][node]["params"]["before"], "bounce")
        with self.assertRaises(ValueError):
            d.execute(dict(op="set", id=node, param="before", value="spin"))
        with self.assertRaises(ValueError):
            d.execute(dict(op="set", id=node, param="dissolve", value=-1))


if __name__ == "__main__":
    unittest.main()
