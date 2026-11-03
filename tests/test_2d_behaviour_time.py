"""2D behaviour lane 2, step I (Time group, retiming and frame nodes): TimeOffset, FrameHold, Retime, FrameRange, TimeClip, AppendClip,
TimeEcho, TimeBlur and TimeWarp pick the frames the guide says, shown on a plate whose red channel is the frame number.

The cases live in tools/nuke_fidelity_cases_time.py and the audit runs them. This module asserts their statuses, the three kernels that
disagreed with the guide (TimeBlur's shutter window, TimeEcho's missing fade, FrameRange cutting fractional frames to whole ones) and the
TimeWarp filter that was missing, the older documents that keep their frames bit for bit, every paired knob moving the picture, and the
tile path agreeing with the full frame for the nodes that have one.
"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as fidelity  # noqa: E402
import nuke_fidelity_cases_time as time_cases  # noqa: E402
from nodebased import timeknobs  # noqa: E402
from nodebased.core import CHOICES, LIMITS, SPECS, upgrade_document, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.tileexec import TileExecutor  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph, evaluator_pixels  # noqa: E402

KINDS = ("TimeOffset", "FrameHold", "Retime", "FrameRange", "TimeClip", "AppendClip", "TimeEcho", "TimeBlur", "TimeWarp")
TIME_CASES = [c for c in fidelity.CASES if c.nuke in KINDS]


def stamp_plate(g, name="plate", first=-60, last=100):
    """A Constant whose red channel is the frame number on every frame from `first` to `last`."""
    g.add(name, "Constant", dict(width=16, height=12, red=0.0, green=0.25, blue=0.5, alpha=1.0))
    g.d.execute(dict(op="set_curve", id=name, param="red",
                     curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(f)) for f in range(first, last + 1)])))
    return name


def picked(kind, params, frame, chain=(), curves=None):
    """The red value (the input frame the node asked for) of `kind` at `frame` on the stamped plate, after the `chain` of (kind, params) nodes."""
    g = Graph()
    last = stamp_plate(g)
    for index, (chain_kind, chain_params) in enumerate(chain):
        g.add(f"c{index}", chain_kind, chain_params, image=last)
        last = f"c{index}"
    g.add("node", kind, params, image=last)
    for param, curve in (curves or {}).items():
        g.d.execute(dict(op="set_curve", id="node", param=param, curve=curve))
    return float(evaluator_pixels(g.doc, "node", frame)[0, 0, 0])


class CasesTests(unittest.TestCase):
    def test_every_time_node_has_cases_and_the_scored_ones_pass(self):
        self.assertEqual({c.nuke for c in TIME_CASES}, set(KINDS))
        results = {c.id: audit.run_case(c) for c in TIME_CASES}
        failing = [(i, r["status"], r["note"]) for i, r in results.items() if r["status"] not in ("pass", "not scored")]
        self.assertEqual(failing, [])

    def test_a_case_names_its_source_and_states_its_reading(self):
        for c in TIME_CASES:
            with self.subTest(case=c.id):
                self.assertIn(c.source, ("guide", "convention"))
                self.assertGreater(len(c.statement), 40)
                if c.unscored:
                    self.assertGreater(len(c.unscored), 20)

    def test_the_unscored_cases_are_the_ones_the_guide_is_silent_on(self):
        unscored = {c.id for c in TIME_CASES if c.unscored}
        for expected in ("timeoffset_reverse_with_offset", "framehold_negative_increment", "retime_nearest_half_way", "retime_filter_box",
                         "retime_shutter_blending", "timeclip_before_bounce", "timeclip_fade_in_curve", "appendclip_dissolve_weights",
                         "appendclip_fade_in_curve", "timeecho_fade_longer_than_the_frames", "timewarp_nearest_half_way", "timewarp_box_filter"):
            self.assertIn(expected, unscored)
        self.assertTrue(all(c.unscored for c in TIME_CASES if c.id.startswith("framehold_increment_outside")))

    def test_the_case_runner_reads_the_frame_back_as_a_number(self):
        case = next(c for c in TIME_CASES if c.id == "timeoffset_offset_3_at_10")
        out = fidelity.run_nodebased(case)
        self.assertAlmostEqual(float(out[0, 0, 0]), 7.0)
        self.assertAlmostEqual(float(out[0, 0, 1]), 0.25)
        self.assertEqual(out.shape, (12, 16, 4))


class FramesTests(unittest.TestCase):
    """The guide's own worked examples, one per node, through the evaluator."""

    clip = [("FrameRange", dict(first_frame=1, last_frame=20))]

    def test_timeoffset_holds_the_edges_of_the_clip_it_shifts(self):
        self.assertEqual(picked("TimeOffset", dict(time_offset=5), 3, self.clip), 1.0)
        self.assertEqual(picked("TimeOffset", dict(time_offset=-5), 18, self.clip), 20.0)

    def test_framehold_lists_the_frames_the_guide_names(self):
        seen = {picked("FrameHold", dict(first_frame=1, increment=5), f) for f in (6, 11, 16)}
        self.assertEqual(seen, {6.0, 11.0, 16.0})

    def test_retime_loops_bounces_and_holds_before_the_first_output_frame(self):
        before = dict(retime_filter="none", output_range_start=5, output_range_start_lock=1)
        self.assertEqual([picked("Retime", dict(before, retime_before="loop"), f, self.clip) for f in (1, 2, 3, 4, 5, 6)], [17, 18, 19, 20, 1, 2])
        self.assertEqual([picked("Retime", dict(before, retime_before="bounce"), f, self.clip) for f in range(1, 9)], [5, 4, 3, 2, 1, 2, 3, 4])
        self.assertEqual([picked("Retime", dict(before, retime_before="hold"), f, self.clip) for f in range(1, 9)], [1, 1, 1, 1, 1, 2, 3, 4])

    def test_retime_after_the_last_frame(self):
        none = dict(retime_filter="none")
        self.assertEqual([picked("Retime", dict(none, retime_after="loop"), f, self.clip) for f in (19, 20, 21, 22, 23, 24)], [19, 20, 1, 2, 3, 4])
        self.assertEqual([picked("Retime", dict(none, retime_after="bounce"), f, self.clip) for f in (19, 20, 21, 22, 23)], [19, 20, 19, 18, 17])
        self.assertEqual([picked("Retime", dict(none, retime_after="hold"), f, self.clip) for f in (19, 20, 21, 22, 23)], [19, 20, 20, 20, 20])

    def test_retime_passes_a_fractional_frame_through_a_framerange_below_it(self):
        self.assertAlmostEqual(picked("Retime", dict(retime_filter="none", speed=0.5), 4, self.clip), 2.5)

    def test_timeclip_loop_and_bounce_after_the_range(self):
        base = dict(first=1, last=20, fade_in=0, fade_out=0)
        self.assertEqual([picked("TimeClip", dict(base, after="loop"), f) for f in (20, 21, 22, 23)], [20, 1, 2, 3])
        self.assertEqual([picked("TimeClip", dict(base, after="bounce"), f) for f in (20, 21, 22, 23)], [20, 19, 18, 17])

    def test_timeecho_merges_the_current_frame_and_the_ones_before_it(self):
        for method, red in (("plus", 7 + 8 + 9 + 10), ("max", 10), ("average", (7 + 8 + 9 + 10) / 4)):
            self.assertAlmostEqual(picked("TimeEcho", dict(frames=4, method=method, falloff=1.0), 10), red, places=4, msg=method)

    def test_timeecho_fade_is_twenty_forty_sixty_eighty_and_full(self):
        self.assertEqual([round(w, 6) for w in timeknobs.echo_weights(5, 5)], [0.2, 0.4, 0.6, 0.8, 1.0])
        plus = picked("TimeEcho", dict(frames=5, method="plus", falloff=1.0, frames_fade=5), 10)
        self.assertAlmostEqual(plus, 0.2 * 6 + 0.4 * 7 + 0.6 * 8 + 0.8 * 9 + 10, places=4)
        self.assertEqual(timeknobs.echo_weights(4, 1), [1.0] * 4)
        self.assertEqual(timeknobs.echo_weights(4, 0.0), [1.0] * 4)
        self.assertEqual([round(w, 6) for w in timeknobs.echo_weights(6, 3)], [round(1 / 3, 6), round(2 / 3, 6), 1, 1, 1, 1])

    def test_timeblur_window_follows_the_guide(self):
        # frame 30, shutter 1: the stamp is a straight line, so the average of the samples is the centre of the window
        for offset, custom, centre in (("centred", 0, 30.0), ("start", 0, 30.5), ("end", 0, 29.5), ("custom", -0.5, 30.0), ("custom", 2.0, 32.5)):
            got = picked("TimeBlur", dict(shutter=1.0, divisions=8, shutter_offset=offset, custom_offset=custom), 30)
            self.assertAlmostEqual(got, centre, places=4, msg=f"{offset} {custom}")

    def test_timewarp_filters(self):
        curves = {"lookup": dict(interpolation="linear", keys=[dict(frame=1, value=1), dict(frame=5, value=10)])}
        self.assertAlmostEqual(picked("TimeWarp", dict(lookup_filter="none"), 3, curves=curves), 5.5, places=4)
        self.assertEqual(picked("TimeWarp", dict(lookup_filter="nearest"), 4, curves=curves), 8.0)    # 7.75
        self.assertEqual(picked("TimeWarp", dict(lookup_filter="nearest"), 2, curves=curves), 3.0)    # 3.25
        self.assertAlmostEqual(picked("TimeWarp", dict(lookup_filter="blend"), 4, curves=curves), 7.75, places=4)
        self.assertEqual(CHOICES["lookup_filter"], ["none", "blend", "nearest"])
        held = {"lookup": dict(interpolation="constant", keys=[dict(frame=1, value=3), dict(frame=10, value=8)])}
        self.assertEqual([picked("TimeWarp", dict(lookup_filter="none"), f, curves=held) for f in (1, 9, 10, 14)], [3, 3, 8, 8])


class OlderDocumentsTests(unittest.TestCase):
    """A document saved before this step keeps the frames it had: the load migration sets the new flags."""

    def load(self, kind, params, names, chain=()):
        g = Graph()
        last = stamp_plate(g)
        for index, (chain_kind, chain_params) in enumerate(chain):
            g.add(f"c{index}", chain_kind, chain_params, image=last)
            last = f"c{index}"
        g.add("node", kind, params, image=last)
        document = copy.deepcopy(g.doc)
        for name in names:
            document["nodes"]["node"]["params"].pop(name)
        loaded = upgrade_document(document)
        validate(loaded)
        return loaded

    def test_a_new_node_follows_the_guide_and_an_older_one_keeps_its_flag(self):
        for kind, flag in (("TimeBlur", "legacy_shutter_window"), ("FrameRange", "legacy_fractional_frames")):
            self.assertEqual(SPECS[kind]["params"][flag], 0)
            self.assertEqual(timeknobs.OLD[kind][flag], 1)
            self.assertIn(flag, LIMITS)
        self.assertEqual(SPECS["TimeEcho"]["params"]["frames_fade"], 1.0)
        self.assertEqual(timeknobs.OLD["TimeEcho"]["frames_fade"], 1.0)

    def test_an_older_timeblur_keeps_the_window_it_had_bit_for_bit(self):
        names = list(timeknobs.NEW["TimeBlur"])
        for offset, custom, low, high in (("start", 0.0, 29.0, 30.0), ("end", 0.0, 30.0, 31.0), ("custom", 2.0, 31.5, 32.5), ("centred", 0.0, 29.5, 30.5)):
            params = dict(shutter=1.0, divisions=4, shutter_offset=offset, custom_offset=custom)
            loaded = self.load("TimeBlur", params, names)
            self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_shutter_window"], 1)
            samples = [np.float32(low + (i + 0.5) * (high - low) / 4) for i in range(4)]
            self.assertEqual(np.float32(evaluator_pixels(loaded, "node", 30)[0, 0, 0]), np.mean(np.stack(samples), axis=0, dtype=np.float32), offset)
            self.assertEqual(self.load("TimeBlur", params, [])["nodes"]["node"]["params"]["legacy_shutter_window"], 0)

    def test_an_older_framerange_still_cuts_a_fractional_frame_to_a_whole_one(self):
        g = Graph()
        stamp_plate(g)
        g.add("clip", "FrameRange", dict(first_frame=1, last_frame=20), image="plate")
        g.add("node", "Retime", dict(retime_filter="none", speed=0.5), image="clip")
        document = copy.deepcopy(g.doc)
        for name in timeknobs.NEW["FrameRange"]:
            document["nodes"]["clip"]["params"].pop(name)
        loaded = upgrade_document(document)
        self.assertEqual(loaded["nodes"]["clip"]["params"]["legacy_fractional_frames"], 1)
        self.assertEqual(float(evaluator_pixels(loaded, "node", 4)[0, 0, 0]), 2.0)                 # 2.5 cut down, as it always was
        self.assertAlmostEqual(float(evaluator_pixels(g.doc, "node", 4)[0, 0, 0]), 2.5)

    def test_an_older_timeecho_is_unchanged_by_the_fade(self):
        loaded = self.load("TimeEcho", dict(frames=4, method="average", falloff=0.5), list(timeknobs.NEW["TimeEcho"]))
        self.assertEqual(loaded["nodes"]["node"]["params"]["frames_fade"], 1.0)
        weights = np.array([0.5 ** i for i in range(4)], np.float32)
        expected = np.float32(sum(np.float32(f) * w for f, w in zip((10, 9, 8, 7), weights)) / np.sum(weights))
        self.assertAlmostEqual(float(evaluator_pixels(loaded, "node", 10)[0, 0, 0]), float(expected), places=4)

    def test_a_whole_valued_float_frame_is_the_same_request_as_the_integer(self):
        evaluator = Evaluator()
        g = Graph()
        stamp_plate(g)
        g.add("node", "FrameRange", dict(first_frame=1, last_frame=20), image="plate")
        document = dict(g.doc, view="node")
        first = evaluator.evaluate(document, "node", frame=5.0)
        second = evaluator.evaluate(document, "node", frame=5)
        self.assertTrue(np.array_equal(first, second))


class KnobsActTests(unittest.TestCase):
    """Every main-tab knob the mapping pairs with a NodeBased param moves the picture off its default."""

    clip = [("FrameRange", dict(first_frame=1, last_frame=20))]
    none = dict(retime_filter="none")
    flat = dict(fade_in=0, fade_out=0)

    def cases(self):
        clip, none, flat = self.clip, self.none, self.flat
        return (
            ("TimeOffset", dict(), dict(time_offset=3), 10, clip),
            ("TimeOffset", dict(), dict(reverse_input=1), 10, clip),
            ("TimeOffset", dict(time_offset=0.5), dict(time_offset=0.5, rounding_mode="Fractional time"), 10, ()),
            ("FrameHold", dict(), dict(first_frame=7), 10, ()),
            ("FrameHold", dict(first_frame=1, increment=0), dict(first_frame=1, increment=5), 12, ()),
            ("FrameHold", dict(first_frame=7.5), dict(first_frame=7.5, rounding_mode="Fractional time"), 12, ()),
            ("Retime", none, dict(none, speed=2.0), 5, clip),
            ("Retime", none, dict(none, reverse=1), 5, clip),
            ("Retime", none, dict(none, input_range_start=5, input_range_start_lock=1, output_range_start=1, output_range_start_lock=1), 7, clip),
            ("Retime", none, dict(none, input_range_end=10, input_range_end_lock=1, output_range_end=5, output_range_end_lock=1,
                                  output_range_start_lock=1), 3, clip),
            ("Retime", none, dict(none, output_range_start=3, output_range_start_lock=1), 5, clip),
            ("Retime", dict(none, speed=0.7), dict(retime_filter="nearest", speed=0.7), 4, clip),
            # on a straight-line stamp a box average of a whole frame's neighbours is the frame itself, so the box shows at the clip's edge
            ("Retime", dict(retime_filter="nearest", speed=2.0), dict(retime_filter="box", speed=2.0), 1, clip),
            ("Retime", dict(retime_filter="box", speed=2.0), dict(retime_filter="box", speed=2.0, shutter=0.5), 1, clip),
            ("Retime", dict(none, output_range_start=5, output_range_start_lock=1),
             dict(none, output_range_start=5, output_range_start_lock=1, retime_before="loop"), 2, clip),
            ("Retime", none, dict(none, retime_after="loop"), 22, clip),
            ("FrameRange", dict(first_frame=1, last_frame=100), dict(first_frame=30, last_frame=100), 10, ()),
            ("FrameRange", dict(first_frame=1, last_frame=100), dict(first_frame=1, last_frame=5), 10, ()),
            ("TimeClip", dict(first=1, last=100, **flat), dict(first=1, last=100, frame_mode="offset", frame_expr="5", **flat), 10, ()),
            ("TimeClip", dict(first=1, last=100, **flat), dict(first=1, last=100, fade_in=5, fade_out=0), 2, ()),
            ("TimeClip", dict(first=1, last=100, **flat), dict(first=1, last=100, fade_in=0, fade_out=5), 99, ()),
            ("TimeClip", dict(first=5, last=20, **flat), dict(first=5, last=20, reverse=1, **flat), 10, ()),
            ("TimeClip", dict(first=5, last=20, **flat), dict(first=5, last=20, before="loop", **flat), 2, ()),
            ("TimeClip", dict(first=1, last=20, **flat), dict(first=1, last=20, after="loop", **flat), 22, ()),
            ("TimeClip", dict(first=1, last=100, **flat), dict(first=1, last=100, origfirst=20, origlast=40, **flat), 10, ()),
            ("TimeEcho", dict(frames=1, method="max", falloff=1.0), dict(frames=3, method="plus", falloff=1.0), 10, ()),
            ("TimeEcho", dict(frames=4, method="plus", falloff=1.0), dict(frames=4, method="max", falloff=1.0), 10, ()),
            ("TimeEcho", dict(frames=4, method="plus", falloff=1.0), dict(frames=4, method="plus", falloff=1.0, frames_fade=4), 10, ()),
            ("TimeBlur", dict(shutter=1.0, divisions=4, shutter_offset="start"), dict(shutter=2.0, divisions=4, shutter_offset="start"), 30, ()),
            ("TimeBlur", dict(shutter=1.0, divisions=4, shutter_offset="centred"), dict(shutter=1.0, divisions=4, shutter_offset="start"), 30, ()),
            ("TimeBlur", dict(shutter=1.0, divisions=4, shutter_offset="custom", custom_offset=0),
             dict(shutter=1.0, divisions=4, shutter_offset="custom", custom_offset=3), 30, ()),
            ("TimeWarp", dict(lookup_filter="none"), dict(lookup_filter="nearest"), 4, ()),
        )

    def test_each_paired_knob_changes_the_frame_picked(self):
        curves = {"lookup": dict(interpolation="linear", keys=[dict(frame=1, value=1), dict(frame=5, value=10)])}
        for kind, base, moved, frame, chain in self.cases():
            with self.subTest(kind=kind, moved=moved):
                use = curves if kind == "TimeWarp" else None
                self.assertNotEqual(picked(kind, base, frame, chain, use), picked(kind, moved, frame, chain, use))

    def test_timeblur_divisions_and_shutter_change_a_picture_that_is_not_a_straight_line(self):
        seen = set()
        for shutter, divisions in ((1.0, 4), (2.0, 4), (3.0, 4), (2.0, 1)):
            h = Graph()
            h.add("plate", "Constant", dict(width=16, height=12, red=0.1, green=0.4, blue=0.6, alpha=1.0))
            for frame, value in ((1, 0.1), (2, 0.9), (3, 0.2), (4, 0.8), (5, 0.1), (6, 0.7)):
                h.key("plate", "red", frame, value)
            h.add("node", "TimeBlur", dict(shutter=shutter, divisions=divisions, shutter_offset="start"), image="plate")
            seen.add(round(float(evaluator_pixels(h.doc, "node", 2)[0, 0, 0]), 5))
        self.assertEqual(len(seen), 4)

    def test_the_timewarp_curve_and_the_appendclip_knobs_move_the_picture(self):
        flat = picked("TimeWarp", dict(lookup_filter="none"), 5)
        steep = picked("TimeWarp", dict(lookup_filter="none"), 5,
                       curves={"lookup": dict(interpolation="linear", keys=[dict(frame=1, value=1), dict(frame=5, value=10)])})
        self.assertEqual((flat, steep), (5.0, 10.0))
        pictures = {}
        for name, params in (("plain", dict()), ("slip", dict(first_frame=5)), ("dissolve", dict(dissolve=2)), ("fade", dict(fade_in=3, fade_out=3))):
            g = Graph()
            stamp_plate(g, "a")
            stamp_plate(g, "b")
            g.add("ra", "FrameRange", dict(first_frame=1, last_frame=5), image="a")
            g.add("rb", "FrameRange", dict(first_frame=11, last_frame=15), image="b")
            g.add("node", "AppendClip", params, clip0="ra", clip1="rb")
            pictures[name] = [round(float(evaluator_pixels(g.doc, "node", f)[0, 0, 0]), 4) for f in range(1, 10)]
        self.assertEqual(len({tuple(v) for v in pictures.values()}), 4)
        self.assertEqual(pictures["plain"], [1, 2, 3, 4, 5, 11, 12, 13, 14])


class TilePathTests(unittest.TestCase):
    """TimeBlur and TimeEcho are tile-native: a tiled render equals the full frame at tiles of 7, 16 and 32."""

    def small(self, kind, params):
        g = Graph()
        g.add("plate", "Constant", dict(width=64, height=64, red=0.1, green=0.4, blue=0.6, alpha=1.0))
        for frame, value in ((1, 0.1), (2, 0.5), (3, 0.9), (4, 0.2), (5, 0.7), (6, 0.3)):
            g.key("plate", "red", frame, value)
        g.add("node", kind, params, image="plate")
        return dict(g.doc, view="node")

    def test_tiles_equal_the_full_frame(self):
        for kind, params, frame in (
                ("TimeBlur", dict(shutter=2.0, divisions=5, shutter_offset="start", custom_offset=0.0), 3),
                ("TimeBlur", dict(shutter=2.0, divisions=5, shutter_offset="end", custom_offset=0.0), 4),
                ("TimeBlur", dict(shutter=1.5, divisions=5, shutter_offset="custom", custom_offset=1.0), 2),
                ("TimeEcho", dict(frames=4, method="plus", falloff=1.0, frames_fade=3), 5),
                ("TimeEcho", dict(frames=4, method="average", falloff=0.8, frames_fade=2), 5),
                ("TimeEcho", dict(frames=3, method="max", falloff=1.0, frames_fade=3), 4)):
            doc = self.small(kind, params)
            full = Evaluator().evaluate(doc, "node", frame=frame)
            for edge in (7, 16, 32):
                with self.subTest(kind=kind, params=params, edge=edge):
                    result = TileExecutor(evaluator=Evaluator(), tile_edge=edge).compose(doc, "node", frame=frame, tier=1)
                    self.assertTrue(result.tiled)
                    np.testing.assert_allclose(result.pixels, full, atol=1e-6)


class RegistrationTests(unittest.TestCase):
    def test_the_case_module_is_registered_in_the_case_hook(self):
        self.assertIn('"nuke_fidelity_cases_time"', (ROOT / "tools" / "nuke_fidelity_cases.py").read_text())
        self.assertTrue(callable(time_cases.cases))


if __name__ == "__main__":
    unittest.main()
