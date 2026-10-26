"""Fidelity step D, part 1: TimeOffset, FrameHold, Retime and TimeClip take Nuke's knobs.

The upstream is a Constant whose red channel is keyed to `frame / 100` on every frame from -10 to 60, so a pixel proves
which input frame (whole or fractional) the node asked for. The four nodes evaluate their input at other frames, which the
tile executor has no per-tile notion of, so they fall back to the evaluator, like TimeWarp before them (asserted)."""
import copy
import unittest

import numpy as np

from nodebased import timeknobs
from nodebased.core import SCHEMA_VERSION, SPECS, bypass_slot, upgrade_document, validate
from tests.test_2d_parity_group_c4 import Graph, evaluator_pixels, tile_fallback_pixels

LEGACY_RETIME = dict(input_range_start_lock=1, output_range_start_lock=1, retime_before="continue",
                     retime_after="continue", retime_filter="nearest")


def plate(g, first=-10, last=60):
    g.add("plate", "Constant", dict(red=0.0, green=0.4, blue=0.6, alpha=1.0))
    for frame in range(first, last + 1):
        g.key("plate", "red", frame, frame / 100.0)
    return "plate"


def source_frame(kind, params, frame, upstream=None, **inputs):
    """The input frame a node asked for at `frame`: its red channel times 100 (None when the frame is black)."""
    g = Graph()
    plate(g)
    if upstream:
        g.add("clip", upstream[0], upstream[1], image="plate")
        g.add("node", kind, params, image="clip")
    else:
        g.add("node", kind, params, image="plate")
    out = evaluator_pixels(g.doc, "node", frame=frame)
    return round(float(out[0, 0, 0]) * 100.0, 4), float(out[0, 0, 3])


class TimeOffsetTests(unittest.TestCase):
    def test_the_offset_moves_the_clip_later(self):
        self.assertEqual(source_frame("TimeOffset", dict(time_offset=3), 10)[0], 7)
        self.assertEqual(source_frame("TimeOffset", dict(time_offset=-3), 10)[0], 13)

    def test_whole_frames_round_down(self):
        self.assertEqual(source_frame("TimeOffset", dict(time_offset=0.5), 10)[0], 9)
        self.assertEqual(source_frame("TimeOffset", dict(time_offset=-0.5), 10)[0], 10)
        self.assertEqual(source_frame("TimeOffset", dict(time_offset=0.25), 10)[0], 9)

    def test_fractional_time_keeps_the_exact_time(self):
        fractional = dict(rounding_mode="Fractional time")
        self.assertAlmostEqual(source_frame("TimeOffset", dict(time_offset=0.5, **fractional), 10)[0], 9.5, places=3)
        self.assertAlmostEqual(source_frame("TimeOffset", dict(time_offset=-0.25, **fractional), 10)[0], 10.25, places=3)

    def test_reverse_input_flips_the_clip_about_its_range(self):
        clip = ("FrameRange", dict(first_frame=1, last_frame=20))
        self.assertEqual(source_frame("TimeOffset", dict(reverse_input=1), 5, upstream=clip)[0], 16)
        self.assertEqual(source_frame("TimeOffset", dict(reverse_input=1), 1, upstream=clip)[0], 20)
        # offset first, then flip: the reversed clip keeps the span an offset clip would have
        self.assertEqual(source_frame("TimeOffset", dict(reverse_input=1, time_offset=3), 5, upstream=clip)[0], 19)

    def test_with_no_range_upstream_the_timeline_is_the_clip(self):
        g = Graph()
        plate(g)
        g.add("node", "TimeOffset", dict(reverse_input=1), image="plate")
        first, last = g.doc["time"]["first"], g.doc["time"]["last"]
        out = evaluator_pixels(g.doc, "node", frame=4)
        self.assertAlmostEqual(float(out[0, 0, 0]) * 100.0, first + last - 4, places=3)

    def test_the_legacy_reverse_flag_still_flips_the_offsets_direction(self):
        self.assertEqual(source_frame("TimeOffset", dict(time_offset=3, reverse=1), 10)[0], 13)


class FrameHoldTests(unittest.TestCase):
    def test_the_first_frame_is_held_at_every_frame(self):
        for frame in (1, 5, 40):
            self.assertEqual(source_frame("FrameHold", dict(first_frame=7), frame)[0], 7)

    def test_the_guides_increment_example(self):
        # first frame 1, increment 5: the input's frames 1, 6, 11, 16 and so on
        seen = [source_frame("FrameHold", dict(first_frame=1, increment=5), f)[0] for f in range(1, 17)]
        self.assertEqual(seen, [1] * 5 + [6] * 5 + [11] * 5 + [16])

    def test_whole_frames_round_the_held_time_down(self):
        self.assertEqual(source_frame("FrameHold", dict(first_frame=2.75), 9)[0], 2)
        self.assertEqual(source_frame("FrameHold", dict(first_frame=2.5, increment=2.5), 9)[0], 7)

    def test_fractional_time_keeps_it(self):
        fractional = dict(rounding_mode="Fractional time")
        self.assertAlmostEqual(source_frame("FrameHold", dict(first_frame=2.75, **fractional), 9)[0], 2.75, places=3)
        self.assertAlmostEqual(source_frame("FrameHold", dict(first_frame=1.5, increment=2.5, **fractional), 8)[0],
                               6.5, places=3)

    def test_set_to_current_frame_is_a_panel_action_on_first_frame(self):
        from nodebased.knobs import knob_layout
        self.assertIn("first_frame", [p for group in knob_layout("FrameHold") for p in group.params])


class RetimePlanTests(unittest.TestCase):
    """The guide's own 20 frame examples, with output.first and output.last at 5 and 24 (both locked)."""

    base = dict(input_range_start=1, input_range_end=20, output_range_start=5, output_range_end=24, speed=1.0,
                input_range_start_lock=1, input_range_end_lock=1, output_range_start_lock=1, output_range_end_lock=1,
                retime_filter="nearest")

    def frames(self, span, **over):
        params = {**self.base, **over}
        return [timeknobs.plan("Retime", params, f, (1, 20)).samples[0][0] for f in span]

    def test_before_loop_bounce_hold(self):
        self.assertEqual(self.frames(range(1, 9), retime_before="loop"), [17, 18, 19, 20, 1, 2, 3, 4])
        self.assertEqual(self.frames(range(1, 9), retime_before="bounce"), [5, 4, 3, 2, 1, 2, 3, 4])
        self.assertEqual(self.frames(range(1, 9), retime_before="hold"), [1, 1, 1, 1, 1, 2, 3, 4])

    def test_after_loop_bounce_hold(self):
        self.assertEqual(self.frames(range(22, 30), retime_after="loop"), [18, 19, 20, 1, 2, 3, 4, 5])
        self.assertEqual(self.frames(range(22, 30), retime_after="bounce"), [18, 19, 20, 19, 18, 17, 16, 15])
        self.assertEqual(self.frames(range(22, 30), retime_after="hold"), [18, 19, 20, 20, 20, 20, 20, 20])

    def test_continue_keeps_the_line_going(self):
        self.assertEqual(self.frames(range(1, 5), retime_before="continue"), [-3, -2, -1, 0])
        self.assertEqual(self.frames(range(25, 28), retime_after="continue"), [21, 22, 23])

    def test_black_has_no_frame(self):
        plan = timeknobs.plan("Retime", {**self.base, "retime_before": "black"}, 2, (1, 20))
        self.assertTrue(plan.black)
        self.assertFalse(timeknobs.plan("Retime", {**self.base, "retime_before": "black"}, 6, (1, 20)).black)

    def test_both_output_ends_locked_set_the_speed(self):
        # 20 input frames over output 1 to 10: two input frames per output frame
        params = {**self.base, "output_range_start": 1, "output_range_end": 11}
        self.assertEqual([timeknobs.plan("Retime", params, f, (1, 20)).samples[0][0] for f in (1, 2, 3, 11)],
                         [1, 3, 5, 20])
        self.assertAlmostEqual(timeknobs.retime_numbers(params, (1, 20))[4], 1.9)

    def test_an_unlocked_range_is_the_input_clips(self):
        params = {**self.base, "input_range_start_lock": 0, "input_range_end_lock": 0, "output_range_start_lock": 0,
                  "output_range_end_lock": 0, "input_range_start": 50, "input_range_end": 60}
        self.assertEqual(timeknobs.retime_numbers(params, (3, 12)), (3.0, 12.0, 3.0, 12.0, 1.0))
        self.assertEqual(timeknobs.retime_numbers({**params, "speed": 2.0}, (3, 12)), (3.0, 12.0, 3.0, 7.5, 2.0))

    def test_reverse_runs_the_input_backwards(self):
        params = {**self.base, "reverse": 1, "retime_before": "hold", "retime_after": "hold"}
        self.assertEqual([timeknobs.plan("Retime", params, f, (1, 20)).samples[0][0] for f in (5, 6, 24, 30, 1)],
                         [20, 19, 1, 1, 20])


class RetimePixelTests(unittest.TestCase):
    def test_the_nearest_filter_rounds_the_input_frame(self):
        # ties go to the even frame, as Retime always did
        for speed, frame, want in ((0.5, 5, 2), (0.5, 7, 4), (0.5, 4, 2), (2.0, 4, 8), (1.5, 3, 4), (1.5, 4, 6)):
            got = source_frame("Retime", dict(speed=speed, input_range_start=0, output_range_start=0, **LEGACY_RETIME),
                               frame)[0]
            self.assertEqual(got, want, (speed, frame))

    def test_the_none_filter_passes_the_fractional_frame_on(self):
        got = source_frame("Retime", dict(speed=0.5, input_range_start=0, output_range_start=0, **{
            **LEGACY_RETIME, "retime_filter": "none"}), 5)[0]
        self.assertAlmostEqual(got, 2.5, places=3)

    def test_the_box_filter_averages_the_frames_under_the_box(self):
        params = dict(speed=2.0, input_range_start=0, output_range_start=0, **{**LEGACY_RETIME, "retime_filter": "box"})
        # speed 2: output frame 5 is input time 10; a box two frames wide takes 9 (half), 10 and 11 (half), weights 1/4, 1/2, 1/4
        got = source_frame("Retime", params, 5)[0]
        self.assertAlmostEqual(got, (9 * 0.25 + 10 * 0.5 + 11 * 0.25), places=3)
        # half speed with a full shutter looks at half an input frame: a whole frame stays whole
        slow = dict(speed=0.5, input_range_start=0, output_range_start=0, **{**LEGACY_RETIME, "retime_filter": "box"})
        self.assertAlmostEqual(source_frame("Retime", slow, 4)[0], 2.0, places=3)
        self.assertAlmostEqual(source_frame("Retime", slow, 5)[0], 2.5, places=3)

    def test_the_shutter_narrows_the_box(self):
        params = dict(speed=2.0, shutter=0.5, input_range_start=0, output_range_start=0,
                      **{**LEGACY_RETIME, "retime_filter": "box"})
        self.assertAlmostEqual(source_frame("Retime", params, 5)[0], 10.0, places=3)

    def test_box_at_speed_one_is_the_plain_frame(self):
        params = dict(speed=1.0, input_range_start=0, output_range_start=0, **{**LEGACY_RETIME, "retime_filter": "box"})
        self.assertEqual(source_frame("Retime", params, 7)[0], 7)

    def test_before_black_is_a_transparent_frame(self):
        params = dict(speed=1.0, input_range_start=5, output_range_start=5, input_range_end=20,
                      input_range_start_lock=1, input_range_end_lock=1, output_range_start_lock=1,
                      retime_before="black", retime_after="hold", retime_filter="nearest")
        value, alpha = source_frame("Retime", params, 3)
        self.assertEqual((value, alpha), (0.0, 0.0))
        self.assertEqual(source_frame("Retime", params, 8)[0], 8)

    def test_the_tile_path_falls_back_to_the_evaluator(self):
        g = Graph()
        plate(g)
        g.add("node", "Retime", dict(speed=2.0, retime_filter="box", input_range_start=0, output_range_start=0,
                                      input_range_start_lock=1, output_range_start_lock=1), image="plate")
        self.assertTrue(np.array_equal(tile_fallback_pixels(g.doc, "node", 5), evaluator_pixels(g.doc, "node", 5)))

    def test_a_bypassed_retime_passes_the_current_frame(self):
        g = Graph()
        plate(g)
        g.add("node", "Retime", dict(speed=2.0, input_range_start=0, output_range_start=0), image="plate")
        g.bypass("node")
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
        self.assertAlmostEqual(float(evaluator_pixels(g.doc, "node", 9)[0, 0, 0]) * 100.0, 9.0, places=3)

    def test_an_older_document_loads_and_draws_the_same(self):
        g = Graph()
        plate(g)
        g.add("node", "Retime", dict(speed=2.0, input_range_start=3, output_range_start=1), image="plate")
        document = copy.deepcopy(g.doc)
        for name in timeknobs.NEW["Retime"]:
            document["nodes"]["node"]["params"].pop(name)
        loaded = upgrade_document(document)
        validate(loaded)
        for name, value in timeknobs.OLD["Retime"].items():
            self.assertEqual(loaded["nodes"]["node"]["params"][name], value)
        for frame in (1, 4, 20):
            self.assertAlmostEqual(float(evaluator_pixels(loaded, "node", frame)[0, 0, 0]) * 100.0,
                                   float(3 + (frame - 1) * 2), places=3)


class TimeClipTests(unittest.TestCase):
    base = dict(first=5, last=8, fade_in=0, fade_out=0)

    def clip(self, frame, **over):
        return source_frame("TimeClip", {**self.base, **over}, frame)

    def test_the_range_with_before_and_after(self):
        self.assertEqual([self.clip(f)[0] for f in (3, 5, 7, 8, 12)], [5, 5, 7, 8, 8])
        self.assertEqual([self.clip(f, after="loop")[0] for f in (8, 9, 10, 12)], [8, 5, 6, 8])
        self.assertEqual([self.clip(f, after="bounce")[0] for f in (8, 9, 10, 11)], [8, 7, 6, 5])

    def test_start_at_puts_the_first_frame_at_that_timeline_frame(self):
        self.assertEqual([self.clip(f, frame_mode="start at", frame_expr="20")[0] for f in (20, 21, 23, 24)], [5, 6, 8, 8])
        self.assertEqual(self.clip(10, frame_mode="start at", frame_expr="20", before="black")[1], 0.0)

    def test_offset_adds_to_the_frame(self):
        self.assertEqual([self.clip(f, frame_mode="offset", frame_expr="3")[0] for f in (2, 3, 4, 6)], [5, 6, 7, 8])
        self.assertEqual(self.clip(7, frame_mode="offset", frame_expr="-3")[0], 5)

    def test_expression_is_the_value_of_the_formula(self):
        self.assertEqual([self.clip(f, frame_mode="expression", frame_expr="frame / 2")[0] for f in (10, 12, 14)], [5, 6, 7])
        self.assertEqual(self.clip(11, frame_mode="expression", frame_expr="")[0], 8)

    def test_the_legacy_offset_still_shifts_the_clip(self):
        self.assertEqual([self.clip(f, time_offset=2)[0] for f in (7, 8, 9)], [5, 6, 7])

    def test_reverse_plays_the_range_backwards(self):
        self.assertEqual([self.clip(f, reverse=1)[0] for f in (5, 6, 7, 8)], [8, 7, 6, 5])
        self.assertEqual([self.clip(f, reverse=1)[0] for f in (3, 12)], [8, 5])      # hold shows the reversed clip's ends

    def test_the_original_range_cuts_the_clip(self):
        self.assertEqual([self.clip(f, first=2, last=10, origfirst=4, origlast=6)[0] for f in (2, 4, 5, 6, 9)], [4, 4, 5, 6, 6])

    def test_before_black_is_transparent(self):
        self.assertEqual(self.clip(3, before="black")[1], 0.0)
        self.assertEqual(self.clip(6, before="black")[1], 1.0)

    def test_fades_dim_the_first_and_last_frames(self):
        gains = [self.clip(f, first=10, last=19, fade_in=3, fade_out=2)[1] for f in range(10, 20)]
        expected = [1 / 4, 2 / 4, 3 / 4, 1, 1, 1, 1, 1, 2 / 3, 1 / 3]
        for got, want in zip(gains, expected):
            self.assertAlmostEqual(got, want, places=5)
        # the colour is dimmed with the alpha: premultiplied
        g = Graph()
        plate(g)
        g.add("node", "TimeClip", dict(first=10, last=19, fade_in=3, fade_out=0), image="plate")
        out = evaluator_pixels(g.doc, "node", 10)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.10 * 0.25, places=5)
        self.assertAlmostEqual(float(out[0, 0, 2]), 0.6 * 0.25, places=5)

    def test_fades_leave_the_held_frames_alone(self):
        self.assertEqual(self.clip(30, first=10, last=19, fade_in=3, fade_out=3)[1], 1.0)

    def test_the_defaults_are_nukes(self):
        params = SPECS["TimeClip"]["params"]
        self.assertEqual((params["fade_in"], params["fade_out"], params["frame_mode"], params["reverse"]),
                         (5.0, 5.0, "expression", 0))

    def test_an_older_document_loads_with_no_fades(self):
        g = Graph()
        plate(g)
        g.add("node", "TimeClip", dict(first=5, last=8, time_offset=2), image="plate")
        document = copy.deepcopy(g.doc)
        for name in timeknobs.NEW["TimeClip"]:
            document["nodes"]["node"]["params"].pop(name)
        loaded = upgrade_document(document)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["node"]["params"]["fade_in"], 0.0)
        self.assertEqual([round(float(evaluator_pixels(loaded, "node", f)[0, 0, 0]) * 100) for f in (7, 8, 9)], [5, 6, 7])

    def test_the_tile_path_falls_back_and_bypass_passes_the_frame(self):
        g = Graph()
        plate(g)
        g.add("node", "TimeClip", dict(first=5, last=8, reverse=1), image="plate")
        self.assertTrue(np.array_equal(tile_fallback_pixels(g.doc, "node", 6), evaluator_pixels(g.doc, "node", 6)))
        g.bypass("node")
        self.assertAlmostEqual(float(evaluator_pixels(g.doc, "node", 12)[0, 0, 0]) * 100.0, 12.0, places=3)


class FrameHoldPanelTests(unittest.TestCase):
    def test_set_to_current_frame_button_writes_first_frame(self):
        from PySide6.QtWidgets import QPushButton
        from tests.test_desktop import APP, Window, release_window
        window = Window()
        self.addCleanup(release_window, self)
        window.show()
        APP.processEvents()
        window.dispatcher.execute({"op": "create", "id": "hold", "type": "FrameHold", "pos": [0, 0]})
        window.dispatcher.document["time"]["current"] = 17
        panel = window.build_node_panel("hold")
        button = panel.findChild(QPushButton, "framehold-set-to-current-frame")
        self.assertIsNotNone(button)
        button.click()
        APP.processEvents()
        self.assertEqual(window.graph_nodes()["hold"]["params"]["first_frame"], 17.0)
        window.saved_document = window.dispatcher.document
        window.close()


if __name__ == "__main__":
    unittest.main()
