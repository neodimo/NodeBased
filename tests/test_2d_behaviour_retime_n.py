"""2D behaviour lane 2, step N (Time group): TimeWarp's input length, Kronos's and OFlow's remaining retime knobs and inputs.

The cases live in tools/nuke_fidelity_cases_time.py and the audit runs them. This module asserts their statuses and that each new
knob changes the picture the way the source states.
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
from nodebased.core import SPECS, upgrade_document  # noqa: E402
from nodebased.imaging import Evaluator, timewarp_box  # noqa: E402
from nodebased.knobs import knob_layout  # noqa: E402
from tests.test_2d_behaviour_retime_vectors import stamp_plate  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402


def clip_red(kind, params, frame, last=20):
    """The red value of `kind` at `frame` on the stamped plate cut to a clip of frames 1 to `last` (it holds the last frame past the end)."""
    g = Graph()
    stamp_plate(g)
    g.add("clip", "FrameRange", dict(first_frame=1, last_frame=last), image="plate")
    g.add("node", kind, params, image="clip")
    g.d.execute(dict(op="time", first=1, last=last, current=1))
    return float(Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame).pixels[0, 0, 0])


class TimeWarpLengthTests(unittest.TestCase):
    def test_the_length_cases_pass(self):
        cases = [c for c in fidelity.CASES if c.id.startswith("timewarp_length")]
        self.assertGreaterEqual(len(cases), 4)
        self.assertEqual([(c.id, audit.run_case(c)["status"]) for c in cases if audit.run_case(c)["status"] != "pass"], [])

    def test_box_weights_cover_the_window_and_sum_to_one(self):
        for centre, length in ((10.0, 3.0), (10.0, 2.0), (10.25, 1.0), (7.5, 4.0), (3.0, 0.5), (3.3, 6.5)):
            frames, weights = timewarp_box(centre, length)
            self.assertAlmostEqual(sum(weights), 1.0, places=9, msg=(centre, length))
            if float(centre).is_integer():
                self.assertAlmostEqual(sum(w * f for f, w in zip(frames, weights)), centre, places=9, msg=(centre, length))   # the window is centred
        self.assertEqual(timewarp_box(10.0, 3.0), ([9, 10, 11], [1 / 3, 1 / 3, 1 / 3]))
        self.assertEqual(timewarp_box(10.0, 0.0), ([10], [1.0]))

    def test_a_wider_box_averages_in_the_neighbours_a_held_frame_makes_visible(self):
        self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter="blend", lookup_length=1.0), 20), 20.0, places=5)
        self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter="blend", lookup_length=3.0), 20), (19 + 20 + 20) / 3, places=4)
        self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter="blend", lookup_length=5.0), 20), (18 + 19 + 20 + 20 + 20) / 5, places=4)

    def test_filters_none_and_nearest_ignore_the_length(self):
        for filter_name in ("none", "nearest"):
            self.assertAlmostEqual(clip_red("TimeWarp", dict(lookup_filter=filter_name, lookup_length=5.0), 20), 20.0, places=5, msg=filter_name)

    def test_an_older_timewarp_has_no_length_and_shows_the_picture_it_always_did(self):
        g = Graph()
        stamp_plate(g)
        g.add("node", "TimeWarp", dict(lookup_filter="blend"), image="plate")
        g.d.execute(dict(op="set_curve", id="node", param="lookup", curve=dict(interpolation="linear", keys=[dict(frame=1, value=1), dict(frame=5, value=10)])))
        document = copy.deepcopy(g.doc)
        del document["nodes"]["node"]["params"]["lookup_length"]
        upgrade_document(document)
        before = Evaluator().evaluate_raster(dict(document, view="node"), target="node", frame=3).pixels
        after = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=3).pixels
        self.assertTrue(np.array_equal(before, after))
        self.assertAlmostEqual(float(after[0, 0, 0]), 5.5, places=5)
        self.assertEqual(SPECS["TimeWarp"]["params"]["lookup_length"], 1.0)

    def test_the_panel_follows_the_guide_input_frame_input_length_filter(self):
        labels = [group.label for group in knob_layout("TimeWarp")]
        self.assertEqual(labels, ["Input frame", "Input length", "Filter"])


SHIFT, WIDTH, HEIGHT = 4, 96, 64


def moving(g, name, shift=SHIFT, seed_params=None, gain=None):
    """A Noise plate moving `shift` whole pixels to the right every frame (frame 1 is the unmoved one); `gain` brightens frame f by gain(f)."""
    g.add(name + "_noise", "Noise", dict(width=WIDTH, height=HEIGHT, size=16, octaves=3, **(seed_params or {})))
    g.add(name, "Transform", dict(filter="nearest"), image=name + "_noise")
    g.d.execute(dict(op="set_curve", id=name, param="translate_x",
                     curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(shift * (f - 1))) for f in range(-5, 30)])))
    return name


def picture(g, frame=2, last=8):
    g.d.execute(dict(op="time", first=1, last=last, current=1))
    raster = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame)
    out = np.zeros((raster.display.height, raster.display.width, 4), np.float32)
    both = raster.display.intersect(raster.data)
    out[both.y - raster.display.y:both.bottom - raster.display.y, both.x - raster.display.x:both.right - raster.display.x] = \
        raster.pixels[both.y - raster.data.y:both.bottom - raster.data.y, both.x - raster.data.x:both.right - raster.data.x]
    return out


INTERIOR = (slice(8, -8), slice(16, -16), slice(0, 3))


def half_shift_error(pixels):
    """The largest difference from frame 1's Noise moved half the translation (2 pixels), away from the edges, on red, green and blue."""
    g = Graph()
    g.add("p", "Noise", dict(width=WIDTH, height=HEIGHT, size=16, octaves=3))
    base = Evaluator().evaluate_raster(dict(g.doc, view="p"), target="p", frame=1).pixels
    ref = np.zeros_like(base)
    ref[:, SHIFT // 2:] = base[:, :-(SHIFT // 2)]
    return float(np.max(np.abs(pixels - ref)[INTERIOR]))


def kronos(g, params, **inputs):
    g.add("node", "Kronos", dict(input_start=1.0, input_end=8.0, speed=0.5, interpolation="motion", **params), **inputs)
    return g


class ShutterTests(unittest.TestCase):
    def test_widths(self):
        from nodebased import retimeknobs
        new = dict(retimeknobs.new_params("Kronos"), speed=2.0, timing="speed", shutter_time=0.5)
        self.assertEqual(retimeknobs.shutter_width("Kronos", new), 0.5)
        self.assertEqual(retimeknobs.shutter_width("Kronos", dict(new, auto_shutter_time=1)), 1.0)                     # twice the original speed
        self.assertEqual(retimeknobs.shutter_width("Kronos", dict(new, auto_shutter_time=1, speed=0.5)), 0.0)          # slower: no blur
        self.assertEqual(retimeknobs.shutter_width("Kronos", dict(new, legacy_shutter=1)), 2.0)                       # the older shutter: the speed
        self.assertEqual(retimeknobs.shutter_width("OFlow", dict(retimeknobs.new_params("OFlow"), shutter_time=0.5, speed=2.0)), 0.5)
        self.assertEqual(retimeknobs.shutter_width("OFlow", dict(retimeknobs.new_params("OFlow"), auto_shutter_time=1, speed=3.0)), 2.0)

    def test_samples_without_a_shutter_time_show_the_sharp_frame_and_a_shutter_time_blurs(self):
        for kind in ("Kronos", "OFlow"):
            def shown(**extra):
                g = Graph()
                stamp_plate(g)
                params = dict(input_start=1.0, input_end=20.0, interpolation="blend", speed=1.0, shutter_samples=5, **extra)
                g.add("node", kind, dict(params, output_start=1.0) if kind == "OFlow" else params, image="plate")
                g.d.execute(dict(op="time", first=1, last=20, current=1))
                return float(Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=1).pixels[0, 0, 0])
            self.assertEqual(shown(), 1.0, kind)
            self.assertAlmostEqual(shown(shutter_time=1.0), (1 + 1 + 1 + 1.25 + 1.5) / 5, places=5, msg=kind)      # 0.5 to 1.5, held at the range's first frame
            self.assertAlmostEqual(shown(shutter_time=1.0, auto_shutter_time=1), 1.0, places=5, msg=kind)          # speed 1: no automatic blur

    def test_a_kronos_saved_before_the_shutter_keeps_its_output_speed_wide_shutter(self):
        from nodebased import retimeknobs
        self.assertEqual(retimeknobs.OLD["Kronos"]["legacy_shutter"], 1)
        self.assertEqual(SPECS["Kronos"]["params"]["legacy_shutter"], 0)
        g = Graph()
        stamp_plate(g)
        base = dict(input_start=1.0, input_end=20.0, interpolation="blend", speed=1.0, shutter_samples=5)
        g.add("node", "Kronos", dict(base, legacy_shutter=1), image="plate")
        g.d.execute(dict(op="time", first=1, last=20, current=1))
        # the old shutter was the output speed wide (1 frame) and averaged its samples only for the motion method
        old = float(Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=1).pixels[0, 0, 0])
        self.assertEqual(old, 1.0)                           # blend method: one sample, as before
        g.d.execute(dict(op="set", id="node", param="interpolation", value="motion"))
        self.assertAlmostEqual(float(Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=1).pixels[0, 0, 0]), 1.15, places=5)


class KronosInputsTests(unittest.TestCase):
    def test_the_fg_and_bg_vectors_given_are_used_without_estimating(self):
        g = Graph()
        moving(g, "plate")
        g.add("fg", "Constant", dict(width=WIDTH, height=HEIGHT, red=float(SHIFT), green=0.0, blue=0.0, alpha=1.0))
        kronos(g, {}, image="plate", fgvecs="fg")
        self.assertLessEqual(half_shift_error(picture(g)), 1e-4)
        g2 = Graph()
        moving(g2, "plate")
        g2.add("bg", "Constant", dict(width=WIDTH, height=HEIGHT, red=float(-SHIFT), green=0.0, blue=0.0, alpha=1.0))
        kronos(g2, {}, image="plate", bgvecs="bg")
        self.assertLessEqual(half_shift_error(picture(g2)), 1e-4)

    def test_wrong_vectors_give_a_different_picture(self):
        g = Graph()
        moving(g, "plate")
        g.add("fg", "Constant", dict(width=WIDTH, height=HEIGHT, red=0.0, green=0.0, blue=0.0, alpha=1.0))
        kronos(g, {}, image="plate", fgvecs="fg")
        self.assertGreater(half_shift_error(picture(g)), 0.05)

    def test_the_motion_source_decides_the_motion_the_source_gets_moved_by(self):
        g = Graph()
        g.add("still", "Noise", dict(width=WIDTH, height=HEIGHT, size=16, octaves=3))
        moving(g, "mo")
        kronos(g, {}, image="still", mosrc="mo")
        shifted = picture(g)
        plain = Graph()
        plain.add("still", "Noise", dict(width=WIDTH, height=HEIGHT, size=16, octaves=3))
        kronos(plain, {}, image="still")
        unmoved = picture(plain)
        self.assertFalse(np.array_equal(shifted, unmoved))
        # the still source carried 2 pixels forward by the forward field and 2 back by the backward one, half of each
        base = unmoved
        expected = np.zeros_like(base)
        expected[:, 2:] += 0.5 * base[:, :-2]
        expected[:, :-2] += 0.5 * base[:, 2:]
        # Mean, not max: the vectors here are estimated from the motion source, and the CPU estimator (CI has no GPU) misreads a
        # few Noise pixels (max 0.14, mean 0.002 on 8ab38cf; the GPU one gives max 0.003). The claim is which input the motion
        # comes from, and on either estimator the result sits several times closer to the moved picture than the unmoved one is.
        self.assertLessEqual(float(np.mean(np.abs(shifted - expected)[INTERIOR])), 0.005)
        self.assertGreater(float(np.mean(np.abs(unmoved - expected)[INTERIOR])), 0.01)
        self.assertGreater(float(np.max(np.abs(unmoved - expected)[INTERIOR])), 0.05)

class MatteAndOutputTests(unittest.TestCase):
    def plates(self, method="blend", **params):
        """A stamped source (red = frame) and a matte whose alpha is 0.05 x frame, retimed at speed 0.5: frame 2 shows source time 1.5."""
        g = Graph()
        stamp_plate(g)
        g.add("matte", "Constant", dict(width=16, height=12, red=0.0, green=0.0, blue=0.0, alpha=0.0))
        g.d.execute(dict(op="set_curve", id="matte", param="alpha",
                         curve=dict(interpolation="linear", keys=[dict(frame=f, value=0.05 * f) for f in range(1, 21)])))
        g.add("node", "Kronos", dict(dict(input_start=1.0, input_end=20.0, speed=0.5, interpolation=method, motion_matte_channel="Matte Alpha"), **params),
              image="plate", matte="matte")
        g.d.execute(dict(op="time", first=1, last=20, current=1))
        return g

    def at2(self, g):
        return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=2).pixels[0, 0]

    def test_output_matte_shows_the_retimed_matte(self):
        got = self.at2(self.plates(retime_output="matte"))
        self.assertTrue(np.allclose(got, 0.075, atol=1e-5), got)

    def test_foreground_and_background_are_the_retimed_picture_inside_and_outside_the_matte(self):
        result = self.at2(self.plates(retime_output="result"))
        fg = self.at2(self.plates(retime_output="foreground"))
        bg = self.at2(self.plates(retime_output="background"))
        self.assertTrue(np.allclose(result[:3], [1.5, 0.25, 0.5], atol=1e-5), result)
        # each frame is cut by its own matte (frame 1: 0.05, frame 2: 0.1) before the two are mixed, a premultiplied foreground retimed
        self.assertTrue(np.allclose(fg[:3], 0.5 * np.array([1, 0.25, 0.5]) * 0.05 + 0.5 * np.array([2, 0.25, 0.5]) * 0.1, atol=1e-5), fg)
        self.assertTrue(np.allclose(bg[:3], 0.5 * np.array([1, 0.25, 0.5]) * 0.95 + 0.5 * np.array([2, 0.25, 0.5]) * 0.9, atol=1e-5), bg)
        self.assertTrue(np.allclose(fg + bg, result, atol=1e-5))

    def test_the_matte_channel_picks_the_matte_and_a_matte_never_alters_a_blended_result(self):
        for channel, value in (("Matte Alpha", 0.075), ("Matte Inverted Alpha", 0.925), ("Source Alpha", 1.0), ("Source Inverted Alpha", 0.0)):
            got = self.at2(self.plates(retime_output="matte", motion_matte_channel=channel))
            self.assertTrue(np.allclose(got, value, atol=1e-5), (channel, got))
        self.assertTrue(np.allclose(self.at2(self.plates(motion_matte_channel="Matte Alpha"))[:3], self.at2(self.plates(motion_matte_channel="None"))[:3]))

    def test_no_matte_means_a_black_matte_and_a_whole_background(self):
        g = self.plates(retime_output="matte", motion_matte_channel="None")
        self.assertTrue(np.allclose(self.at2(g), 0.0))

    def test_a_matte_separates_the_motion_of_two_layers(self):
        """On a moving plate, estimating the foreground and the background apart gives another picture than one estimate for the whole frame."""
        def moved(matte_channel):
            g = Graph()
            moving(g, "plate")
            g.add("m", "Noise", dict(width=WIDTH, height=HEIGHT, size=24, octaves=2, seed=3))
            kronos(g, {"motion_matte_channel": matte_channel}, image="plate", matte="m")
            return picture(g)
        with_matte, without = moved("Matte Luminance"), moved("None")
        self.assertFalse(np.array_equal(with_matte, without))
        self.assertTrue(np.isfinite(with_matte).all())

    def test_the_matte_choices_and_inputs_are_registered(self):
        from nodebased.core import CHOICES
        self.assertEqual(CHOICES["retime_output"], ["result", "matte", "foreground", "background"])
        self.assertEqual(SPECS["Kronos"]["optional_inputs"], ["matte", "mosrc", "fgvecs", "bgvecs"])


class FlickerTests(unittest.TestCase):
    def test_flicker_compensation_estimates_the_motion_through_a_brightness_change(self):
        def moved(compensate, gain):
            g = Graph()
            moving(g, "move")
            g.add("plate", "Multiply", dict(multiply=1.0), image="move")
            g.d.execute(dict(op="set_curve", id="plate", param="multiply",
                             curve=dict(interpolation="linear", keys=[dict(frame=f, value=1.0 + gain * (f - 1)) for f in range(-5, 30)])))
            kronos(g, {"flicker_compensation": int(compensate)}, image="plate")
            return picture(g)
        flat = half_shift_error(moved(False, 0.0))
        self.assertLess(flat, 0.01)
        steady = moved(True, 0.0)
        self.assertEqual(float(np.max(np.abs(steady - moved(False, 0.0)))), 0.0)           # no flicker, nothing to compensate
        off, on = moved(False, 0.5), moved(True, 0.5)
        self.assertFalse(np.array_equal(off, on))


if __name__ == "__main__":
    unittest.main()
