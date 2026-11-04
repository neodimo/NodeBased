"""2D behaviour lane 2, step J (Time group, Kronos and OFlow): the two retimers pick the source frames their timing knobs state.

The cases live in tools/nuke_fidelity_cases_time.py and the audit runs them, on a plate whose red channel is the frame number (so the red
value is the source frame the node picked) and on a Noise plate moving a whole 4 pixels a frame (so the in-between picture of the Motion
method can be compared with half the translation). This module asserts their statuses, the Kronos rule that changed (the output speed
counts from the input range's start, "frame" now shows the nearest original frame, and the input range, input speed and retimed channels
exist), the older Kronos documents that keep their frames bit for bit, every paired knob moving the picture, and the panel's knob order.
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
from nodebased import retimeknobs  # noqa: E402
from nodebased.core import CHOICES, LIMITS, SPECS, upgrade_document, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.knobs import knob_layout  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402

KINDS = ("Kronos", "OFlow")
RETIME_CASES = [c for c in fidelity.CASES if c.nuke in KINDS]


def stamp_plate(g, name="plate", first=-60, last=100, green=False):
    """A Constant whose red channel is the frame number on every frame from `first` to `last` (green: the frame times 0.01)."""
    g.add(name, "Constant", dict(width=16, height=12, red=0.0, green=0.25, blue=0.5, alpha=1.0))
    for channel, scale in (("red", 1.0), *((("green", 0.01),) if green else ())):
        g.d.execute(dict(op="set_curve", id=name, param=channel,
                         curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(f) * scale) for f in range(first, last + 1)])))
    return name


def retimed(kind, params, frame, last=20, green=False, curves=None, project=None):
    """The picture of `kind` at `frame` on the stamped plate, in a project of 1 to `last` frames."""
    g = Graph()
    stamp_plate(g, green=green)
    g.add("node", kind, params, image="plate")
    for param, curve in (curves or {}).items():
        g.d.execute(dict(op="set_curve", id="node", param=param, curve=curve))
    first, end = project or (1, last)
    g.d.execute(dict(op="time", first=first, last=end, current=first))
    return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame).pixels


def red(kind, params, frame, **extra):
    return float(retimed(kind, params, frame, **extra)[0, 0, 0])


def moving_plate_picture(kind, params, frame, shift=4):
    """The picture of `kind` at `frame` on Noise moving `shift` whole pixels to the right every frame (project 1 to 8), cut to the format."""
    g = Graph()
    g.add("plate", "Noise", dict(width=96, height=64, size=16, octaves=3))
    g.add("move", "Transform", dict(filter="nearest"), image="plate")
    g.d.execute(dict(op="set_curve", id="move", param="translate_x",
                     curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(shift * (f - 1))) for f in range(-5, 30)])))
    g.add("node", kind, dict(input_start=1.0, input_end=8.0, **params), image="move")
    g.d.execute(dict(op="time", first=1, last=8, current=1))
    raster = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame)
    canvas = np.zeros((raster.display.height, raster.display.width, 4), np.float32)
    both = raster.display.intersect(raster.data)
    canvas[both.y - raster.display.y:both.bottom - raster.display.y, both.x - raster.display.x:both.right - raster.display.x] = \
        raster.pixels[both.y - raster.data.y:both.bottom - raster.data.y, both.x - raster.data.x:both.right - raster.data.x]
    return canvas


class CasesTests(unittest.TestCase):
    def test_both_nodes_have_cases_and_the_scored_ones_pass(self):
        self.assertEqual({c.nuke for c in RETIME_CASES}, set(KINDS))
        results = {c.id: audit.run_case(c) for c in RETIME_CASES}
        failing = [(i, r["status"], r["note"]) for i, r in results.items() if r["status"] not in ("pass", "not scored")]
        self.assertEqual(failing, [])
        for kind in KINDS:
            scored = [r for i, r in results.items() if r["nuke"] == kind and r["status"] == "pass"]
            self.assertGreaterEqual(len(scored), 30, kind)

    def test_a_case_names_its_source_and_a_not_scored_one_says_why(self):
        for case in RETIME_CASES:
            self.assertIn(case.source, ("guide", "convention"), case.id)
            self.assertTrue(case.statement, case.id)
            if case.unscored:
                self.assertGreater(len(case.unscored), 30, case.id)

    def test_the_guide_cases_quote_the_guide(self):
        for case in RETIME_CASES:
            if case.source == "guide":
                self.assertIn("reference_guide/time_nodes/", case.statement, case.id)

    def test_the_motion_case_states_its_tolerance_and_is_not_exact(self):
        motion = [c for c in RETIME_CASES if "motion_method_half_translation" in c.id]
        self.assertEqual(len(motion), 2)
        for case in motion:
            self.assertEqual(case.tolerance, 0.01)
            self.assertIn("tolerance", case.statement)

    def test_the_motion_case_tells_motion_from_blend(self):
        """The 0.01 tolerance would pass a plain blend if a blend were close to half the translation; it is not (its error is several times larger)."""
        case = [c for c in RETIME_CASES if c.id == "oflow_motion_method_half_translation"][0]
        reference, valid = case.expected(None)
        for method, passes in (("motion", True), ("blend", False)):
            got = moving_plate_picture("OFlow", dict(interpolation=method, speed=0.5, output_start=1.0), 2)
            error = float(np.max(np.abs(got - reference)[valid]))
            self.assertEqual(error <= 0.01, passes, (method, error))


class PlanTests(unittest.TestCase):
    """retimeknobs.plan, the frame arithmetic both nodes share."""

    def params(self, **kw):
        return dict(dict(speed=1.0, frame=1.0, timing="speed", input_start=1.0, input_end=20.0, input_speed=1.0, interpolation="motion",
                         legacy_retime=0), **kw)

    def test_output_speed_counts_from_the_input_range_start(self):
        for speed, frame, picks in ((0.5, 1, 1.0), (0.5, 3, 2.0), (2.0, 4, 7.0), (0.4, 4, 2.2)):
            self.assertAlmostEqual(retimeknobs.plan("Kronos", self.params(speed=speed), frame, 1, 20).target, picks)
        self.assertAlmostEqual(retimeknobs.plan("Kronos", self.params(speed=0.5, input_start=5.0), 9, 1, 20).target, 7.0)

    def test_oflow_counts_from_its_output_start(self):
        base = self.params(speed=0.5, output_start=1.0, input_start=5.0)
        self.assertAlmostEqual(retimeknobs.plan("OFlow", base, 9, 1, 20).target, 9.0)       # 5 + (9 - 1) x 0.5
        self.assertAlmostEqual(retimeknobs.plan("OFlow", dict(base, output_start=5.0), 9, 1, 20).target, 7.0)

    def test_the_frame_timing_is_the_frame_knob_and_the_input_speed_is_read_as_not_negative(self):
        self.assertEqual(retimeknobs.plan("Kronos", self.params(timing="frame", frame=7.0), 15, 1, 20).target, 7.0)
        self.assertEqual(retimeknobs.plan("Kronos", self.params(timing="input_speed", input_speed=-2.0), 9, 1, 20).target, 1.0)
        self.assertAlmostEqual(retimeknobs.plan("Kronos", self.params(timing="input_speed", input_speed=0.5), 3, 1, 20).target, 2.0)

    def test_the_picked_frame_stays_inside_the_range_and_the_project(self):
        self.assertEqual(retimeknobs.plan("Kronos", self.params(speed=2.0, input_end=6.0), 20, 1, 20).target, 6.0)
        self.assertEqual(retimeknobs.plan("Kronos", self.params(speed=2.0), 20, 1, 12).target, 12.0)
        self.assertEqual(retimeknobs.plan("Kronos", self.params(speed=-1.0), 5, 1, 20).target, 1.0)
        with self.assertRaises(ValueError):
            retimeknobs.plan("OFlow", self.params(input_start=9.0, input_end=4.0), 5, 1, 20)

    def test_method_frame_rounds_to_the_nearest_frame_and_a_tie_goes_to_the_even_one(self):
        base = self.params(interpolation="frame", speed=0.4)
        self.assertEqual([retimeknobs.plan("Kronos", base, f, 1, 20).target for f in range(1, 6)], [1.0, 1.0, 2.0, 2.0, 3.0])
        half = self.params(interpolation="frame", speed=0.5)
        self.assertEqual([retimeknobs.plan("Kronos", half, f, 1, 20).target for f in (2, 4, 6)], [2.0, 2.0, 4.0])


class FramesTests(unittest.TestCase):
    def test_kronos_method_frame_shows_a_whole_frame_and_blend_a_mix(self):
        base = dict(speed=0.4, timing="speed", input_start=1.0, input_end=20.0)
        self.assertEqual(red("Kronos", dict(base, interpolation="frame"), 3), 2.0)
        self.assertAlmostEqual(red("Kronos", dict(base, interpolation="blend"), 3), 1.8, places=5)

    def test_the_guide_example_for_the_frame_knob_keyed_over_a_fifty_frame_clip(self):
        keys = {"frame": dict(interpolation="linear", keys=[dict(frame=1, value=1), dict(frame=100, value=50)])}
        for kind in KINDS:
            params = dict(timing="frame", interpolation="blend", input_start=1.0, input_end=50.0)
            self.assertAlmostEqual(red(kind, params, 1, curves=keys, project=(1, 100)), 1.0, places=5)
            self.assertAlmostEqual(red(kind, params, 100, curves=keys, project=(1, 100)), 50.0, places=4)

    def test_retimed_channels_leaves_the_others_as_the_source_at_the_current_frame(self):
        base = dict(speed=0.4, interpolation="blend", input_start=1.0, input_end=20.0)
        for kind in KINDS:
            params = dict(base, output_start=1.0) if kind == "OFlow" else base
            everything = retimed(kind, dict(params, retimed_channels="all"), 3, green=True)[0, 0]
            red_only = retimed(kind, dict(params, retimed_channels="red"), 3, green=True)[0, 0]
            green_only = retimed(kind, dict(params, retimed_channels="green"), 3, green=True)[0, 0]
            self.assertAlmostEqual(float(everything[0]), 1.8, places=5)
            self.assertAlmostEqual(float(everything[1]), 0.018, places=5)
            self.assertAlmostEqual(float(red_only[0]), 1.8, places=5)
            self.assertAlmostEqual(float(red_only[1]), 0.03, places=5)           # green is the source's own at frame 3
            self.assertAlmostEqual(float(green_only[0]), 3.0, places=5)           # red is the source's own at frame 3
            self.assertAlmostEqual(float(green_only[1]), 0.018, places=5)

    def test_a_retimed_channels_value_outside_the_choices_is_refused(self):
        self.assertIn("rgb", CHOICES["retimed_channels"])
        g = Graph()
        g.add("node", "Kronos")
        with self.assertRaises(Exception):
            g.d.execute(dict(op="set", id="node", param="retimed_channels", value="no such channel"))

    def test_input_speed_takes_no_negative_value(self):
        self.assertEqual(LIMITS["input_speed"][0], 0.0)
        g = Graph()
        g.add("node", "OFlow")
        with self.assertRaises(Exception):
            g.d.execute(dict(op="set", id="node", param="input_speed", value=-1.0))


class OlderDocumentsTests(unittest.TestCase):
    """A Kronos saved before this step keeps the frames it had: the load migration sets legacy_retime and the old frame knob."""

    def load(self, kind, params, frame=None):
        g = Graph()
        stamp_plate(g)
        g.add("node", kind, params, image="plate")
        g.d.execute(dict(op="time", first=1, last=20, current=1))
        document = copy.deepcopy(g.doc)
        for name in retimeknobs.NEW[kind]:
            document["nodes"]["node"]["params"].pop(name)
        if frame is not None:
            document["nodes"]["node"]["params"]["frame"] = frame
        loaded = upgrade_document(document)
        validate(loaded)
        return loaded

    def pixel(self, document, frame):
        return Evaluator().evaluate_raster(dict(document, view="node"), target="node", frame=frame).pixels[0, 0, 0]

    def test_a_new_kronos_follows_the_guide_and_an_older_one_keeps_its_flag(self):
        self.assertEqual(SPECS["Kronos"]["params"]["legacy_retime"], 0)
        self.assertEqual(retimeknobs.OLD["Kronos"]["legacy_retime"], 1)
        self.assertIn("legacy_retime", LIMITS)
        self.assertNotIn("legacy_retime", SPECS["OFlow"]["params"])
        loaded = self.load("Kronos", dict(speed=0.5, interpolation="motion"), frame=-1.0)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_retime"], 1)
        self.assertEqual(loaded["nodes"]["node"]["params"]["frame"], -1.0)

    def test_an_older_kronos_counts_from_zero_and_blends_for_method_frame_bit_for_bit(self):
        loaded = self.load("Kronos", dict(speed=0.4, interpolation="frame"), frame=-1.0)
        for frame, low in ((3, 1.0), (4, 1.0), (7, 2.0)):
            alpha = np.float32(frame * 0.4 - np.floor(frame * 0.4))
            self.assertEqual(np.float32(self.pixel(loaded, frame)), np.float32(np.float32(low) * (1 - alpha) + np.float32(low + 1) * alpha), frame)
        self.assertEqual(np.float32(self.pixel(loaded, 5)), np.float32(2.0))                  # 5 x 0.4, a whole frame
        new = self.load("Kronos", dict(speed=0.4, interpolation="frame"), frame=-1.0)
        new["nodes"]["node"]["params"].update(legacy_retime=0, timing="speed", frame=1.0)
        self.assertEqual(float(self.pixel(new, 3)), 2.0)                                     # the guide: the nearest original frame

    def test_an_older_kronos_with_a_frame_at_or_above_zero_shows_that_frame(self):
        loaded = self.load("Kronos", dict(speed=0.5, interpolation="motion"), frame=7.0)
        self.assertEqual(float(self.pixel(loaded, 3)), 7.0)

    def test_an_older_kronos_is_limited_to_the_project_range_not_an_input_range(self):
        loaded = self.load("Kronos", dict(speed=2.0, interpolation="motion"), frame=-1.0)
        self.assertEqual(float(self.pixel(loaded, 15)), 20.0)                                # 30 held at the project's last frame
        self.assertEqual(loaded["nodes"]["node"]["params"]["input_end"], 100.0)

    def test_an_older_oflow_keeps_its_frames_and_gets_the_new_knobs_at_what_it_did(self):
        loaded = self.load("OFlow", dict(speed=0.5, input_start=1.0, input_end=20.0, output_start=1.0))
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["interpolation"], params["retimed_channels"], params["input_speed"]), ("motion", "all", 1.0))
        self.assertEqual(float(self.pixel(loaded, 3)), 2.0)
        self.assertEqual(float(self.pixel(loaded, 5)), 3.0)


class KnobsActTests(unittest.TestCase):
    """Every main-tab knob the mapping pairs with a NodeBased param moves the picture off its default."""

    def base(self, kind):
        base = dict(interpolation="blend", speed=0.4, input_start=1.0, input_end=20.0, timing="speed")
        return dict(base, output_start=1.0) if kind == "OFlow" else base

    def test_each_timing_knob_changes_the_frame_picked(self):
        moves = (("input_start", 2.0, 3), ("input_end", 1.0, 3), ("retimed_channels", "green", 3), ("interpolation", "frame", 3),
                 ("speed", 2.0, 3), ("timing", "frame", 3), ("timing", "input_speed", 3))
        for kind in KINDS:
            for name, value, frame in moves:
                params = self.base(kind)
                if name == "timing" and value == "input_speed":
                    params.update(input_speed=1.0)
                if name == "timing" and value == "frame":
                    params.update(frame=9.0)
                moved = dict(params, **{name: value})
                self.assertNotEqual(red(kind, params, frame), red(kind, moved, frame), (kind, name))

    def test_frame_and_input_speed_act_only_under_their_own_timing(self):
        for kind in KINDS:
            base = self.base(kind)
            self.assertEqual(red(kind, base, 3), red(kind, dict(base, frame=9.0), 3), kind)
            self.assertEqual(red(kind, base, 3), red(kind, dict(base, input_speed=1.0), 3), kind)
            frame_timing = dict(base, timing="frame", frame=4.0)
            self.assertNotEqual(red(kind, frame_timing, 3), red(kind, dict(frame_timing, frame=9.0), 3), kind)
            speed_timing = dict(base, timing="input_speed", input_speed=0.4)
            self.assertNotEqual(red(kind, speed_timing, 3), red(kind, dict(speed_timing, input_speed=1.0), 3), kind)

    def test_oflow_output_start_moves_the_frame(self):
        params = self.base("OFlow")
        self.assertNotEqual(red("OFlow", params, 5), red("OFlow", dict(params, output_start=3.0), 5))

    def test_the_motion_knobs_change_a_moving_picture(self):
        """Shutter, vector detail and smoothness only show on a picture with motion in it."""
        for kind in KINDS:
            base = dict(interpolation="motion", speed=0.5, output_start=1.0) if kind == "OFlow" else dict(interpolation="motion", speed=0.5)
            reference = moving_plate_picture(kind, base, 2)
            moves = [("vector_detail", 1), ("smoothness", 20.0), ("shutter_time", 1.0)] if kind == "OFlow" else [("shutter_time", 1.0), ("vector_detail", 1)]
            for name, value in moves:
                params = dict(base, **{name: value})
                if name == "shutter_time":
                    params["shutter_samples"] = 4
                self.assertFalse(np.array_equal(moving_plate_picture(kind, params, 2), reference), (kind, name))
        # OFlow's samples only matter once the shutter has a width: the same shutter with 2 and with 8 samples differs
        shutter = dict(interpolation="motion", speed=0.5, output_start=1.0, shutter_time=1.0)
        self.assertFalse(np.array_equal(moving_plate_picture("OFlow", dict(shutter, shutter_samples=2), 2),
                                        moving_plate_picture("OFlow", dict(shutter, shutter_samples=8), 2)))


class LayoutTests(unittest.TestCase):
    def labels(self, kind):
        return [(g.params[0], g.label) for g in knob_layout(kind)]

    def test_kronos_main_tab_follows_the_guide_order(self):
        names = [n for n, _ in self.labels("Kronos")]
        # input_start and input_end share one "Input range" row (panel sweep S4), listed under the first
        self.assertEqual(names[:7], ["input_start", "retimed_channels", "interpolation", "timing", "speed", "input_speed", "frame"])
        labels = dict(self.labels("Kronos"))
        self.assertEqual((labels["interpolation"], labels["timing"], labels["speed"], labels["input_speed"], labels["frame"]),
                         ("Method", "Timing", "Output Speed", "Input Speed", "Frame"))

    def test_oflow_main_tab_follows_the_guide_order(self):
        names = [n for n, _ in self.labels("OFlow")]
        paired = [n for n in names if n != "output_start" and n != "flow_backend"]
        self.assertEqual(paired, ["input_start", "retimed_channels", "interpolation", "timing", "speed", "input_speed", "frame",
                                  "vector_detail", "smoothness", "shutter_samples", "shutter_time", "auto_shutter_time", "flicker_compensation"])
        labels = dict(self.labels("OFlow"))
        self.assertEqual((labels["interpolation"], labels["speed"], labels["frame"]), ("Method", "Output Speed", "Frame"))


class RegistrationTests(unittest.TestCase):
    def test_the_case_module_is_registered_in_the_case_hook(self):
        self.assertIn('"nuke_fidelity_cases_time"', (ROOT / "tools" / "nuke_fidelity_cases.py").read_text())
        self.assertTrue(callable(time_cases.cases))

    def test_the_new_knobs_are_paired_in_the_mapping(self):
        import json
        mapping = json.loads((ROOT / "docs" / "parity" / "nodebased_to_nuke.json").read_text())["pairs"]
        for kind in KINDS:
            paired = {n for entry in mapping[kind]["knobs"] for n in entry.get("nodebased", [])}
            for name in ("input_start", "input_end", "retimed_channels", "timing", "input_speed", "speed", "frame", "interpolation"):
                self.assertIn(name, paired, (kind, name))
        nuke = {n for entry in mapping["Kronos"]["knobs"] for n in entry.get("nuke", [])}
        self.assertTrue({"input.first", "input.last", "retimedChannels", "timing2", "timingInputSpeed"} <= nuke)


if __name__ == "__main__":
    unittest.main()
