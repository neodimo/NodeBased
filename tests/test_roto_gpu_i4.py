"""Roto mattes drawn on the graphics card (nodebased/rotogpu.py) against the processor drawer, step I4.

Every case draws the same resolved shapes with the processor drawer (the reference, the card path switched off) and with
the card, and compares the whole RGBA frame. The measured largest and mean absolute differences go to the JSON file named
by NODEBASED_ROTO_GPU_REPORT when it is set, so the docs quote what the tests saw. The cases that do not need a card
(the switch, the fall back to the processor, the cases the kernel declines) run anywhere.
"""
import copy
import json
import math
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, roto, rotogpu, shapes
from tests.test_rotofast import blob, point, shape, smooth

APP = QApplication.instance() or QApplication([])
TOL = 5e-5          # the card fills in 32-bit floats, the drawer in 64-bit; measured worst case 1.2e-5
REPORT = {}


def card_available():
    try:
        gpu3d._state("default")
        return True
    except Exception:
        return False


def resolve(shape_list, frame=1):
    return shapes.resolve_shapes({"shapes": copy.deepcopy(shape_list)}, frame, None)


def processor(resolved, width, height, **kwargs):
    saved = rotogpu._wanted, rotogpu._wanted_pid
    rotogpu.set_enabled(False)
    try:
        with mock.patch.dict(os.environ, {rotogpu.ENV: "0"}):
            return roto.rasterise(resolved, width, height, **kwargs)
    finally:
        rotogpu._wanted, rotogpu._wanted_pid = saved


def regular(cx, cy, radius, count, step=1, **extra):
    """A star polygon: `step` 2 over an odd count crosses itself."""
    return [point(cx + radius * math.cos(2 * math.pi * i * step / count),
                  cy + radius * math.sin(2 * math.pi * i * step / count), **extra) for i in range(count)]


def comb(x0, y0, teeth=9, width=18, height=70):
    top, pts = y0, []
    for i in range(teeth):
        left = x0 + i * width * 2
        pts += [point(left, y0 + height), point(left, top), point(left + width, top), point(left + width, y0 + height)]
    pts += [point(x0 + teeth * width * 2, y0 + height + 20), point(x0, y0 + height + 20)]
    return pts


CASES = {
    "convex quad": ([shape([point(20, 20), point(180, 30), point(150, 150), point(30, 140)])], 200, 200),
    "concave curve": ([shape(blob(100, 100, 60))], 200, 200),
    "concave comb": ([shape(comb(10, 30))], 360, 160),
    "self-crossing bowtie": ([shape([point(20, 20), point(180, 150), point(180, 20), point(20, 150)])], 200, 200),
    "self-crossing star": ([shape(regular(100, 100, 85, 5, step=2))], 200, 200),
    "tiny triangle": ([shape([point(50.2, 50.3), point(52.9, 50.9), point(51.4, 53.1)])], 100, 100),
    "sub-pixel sliver": ([shape([point(10.1, 10.45), point(40.9, 10.55), point(40.9, 10.7), point(10.1, 10.6)])], 60, 40),
    "huge feather 90": ([shape(blob(160, 160, 70), feather=90.0)], 320, 320),
    "feather 12 linear": ([shape(blob(100, 100, 60), feather=12.0)], 200, 200),
    "feather 12 smooth": ([shape(blob(100, 100, 60), feather=12.0, feather_falloff="smooth")], 200, 200),
    "feather 12 gaussian": ([shape(blob(100, 100, 60), feather=12.0, feather_falloff="gaussian")], 200, 200),
    "feather 3 (small radius path)": ([shape(blob(100, 100, 60), feather=3.0)], 200, 200),
    "feather 0.4 (below the blur)": ([shape(blob(100, 100, 60), feather=0.4)], 200, 200),
    "falloff power 0.5": ([shape(blob(100, 100, 60), feather=14.0, falloff_power=0.5)], 200, 200),
    "falloff power 2": ([shape(blob(100, 100, 60), feather=14.0, falloff_power=2.0)], 200, 200),
    "rotated, scaled, moved": ([shape(blob(100, 100, 50), feather=8.0, transform={
        "center_x": 100.0, "center_y": 100.0, "rotate": 37.0, "scale_x": 1.4, "scale_y": 0.7,
        "translate_x": 12.0, "translate_y": -9.0})], 220, 220),
    "b-spline": ([shape([point(40, 40), point(160, 50), point(150, 150), point(50, 140), point(30, 90)],
                        curve_type="bspline", feather=6.0)], 200, 200),
    "half off the frame": ([shape(blob(10, 10, 60), feather=10.0)], 120, 120),
    "opacity and colour": ([shape(blob(100, 100, 60), opacity=0.5, color=[1.0, 0.4, 0.1, 0.8], feather=6.0)], 200, 200),
    "two shapes over": ([shape(blob(80, 100, 55), color=[1.0, 0.2, 0.2, 1.0], feather=4.0),
                         shape(blob(120, 100, 55), color=[0.2, 0.3, 1.0, 1.0], opacity=0.7, feather=4.0)], 200, 200),
    "subtract mode": ([shape(blob(100, 100, 70), feather=5.0),
                       shape(blob(100, 100, 30), mode="subtract", feather=5.0)], 200, 200),
    "plus blend": ([shape(blob(80, 100, 55), color=[0.6, 0.2, 0.1, 1.0]),
                    shape(blob(120, 100, 55), color=[0.5, 0.5, 0.1, 1.0], blend_mode="plus")], 200, 200),
    "multiply blend": ([shape(blob(80, 100, 55), color=[0.9, 0.5, 0.2, 1.0]),
                        shape(blob(120, 100, 55), color=[0.4, 0.8, 1.0, 1.0], blend_mode="multiply", feather=5.0)], 200, 200),
    "difference blend": ([shape(blob(80, 100, 55), color=[0.9, 0.5, 0.2, 1.0], feather=3.0),
                          shape(blob(120, 100, 55), color=[0.4, 0.8, 1.0, 1.0], blend_mode="difference", feather=3.0)], 200, 200),
    "screen and max blends": ([shape(blob(70, 100, 50), color=[0.5, 0.2, 0.2, 1.0]),
                               shape(blob(110, 100, 50), color=[0.2, 0.6, 0.2, 1.0], blend_mode="screen"),
                               shape(blob(100, 120, 50), color=[0.3, 0.3, 0.7, 1.0], blend_mode="max")], 200, 200),
    "inverted shape": ([shape(blob(100, 100, 50), invert=True, feather=8.0)], 200, 200),
    "no shapes": ([], 64, 48),
    "hidden shape": ([shape(blob(100, 100, 50), visible=False)], 200, 200),
}


@unittest.skipUnless(card_available(), "no graphics adapter")
class ParityTests(unittest.TestCase):
    def check(self, name, shape_list, width, height, invert=False):
        resolved = resolve(shape_list)
        expected = processor(resolved, width, height, invert=invert)
        got = rotogpu.rasterise(resolved, width, height, invert)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, np.float32)
        difference = np.abs(got - expected)
        REPORT[name + (" (frame inverted)" if invert else "")] = {
            "max": float(difference.max()), "mean": float(difference.mean())}
        self.assertLess(float(difference.max()), TOL, name)
        return got, expected

    def test_every_case_matches_the_processor(self):
        for name, (shape_list, width, height) in CASES.items():
            with self.subTest(name):
                got, expected = self.check(name, shape_list, width, height)
                if name in ("convex quad", "concave curve", "self-crossing star"):
                    self.assertGreater(float(got[..., 3].max()), 0.99)      # the matte is drawn, not just equal and empty

    def test_a_self_crossing_star_fills_its_centre_by_the_non_zero_rule(self):
        got = rotogpu.rasterise(resolve([shape(regular(100, 100, 85, 5, step=2))]), 200, 200)
        self.assertEqual(float(got[100, 100, 3]), 1.0)

    def test_the_whole_frame_inverted(self):
        for name in ("concave curve", "feather 12 smooth", "no shapes"):
            shape_list, width, height = CASES[name]
            with self.subTest(name):
                self.check(name, shape_list, width, height, invert=True)

    def test_thirty_shapes_at_1080p(self):
        shape_list = []
        for index in range(30):
            cx, cy = 100 + (index % 10) * 180, 150 + (index // 10) * 320
            shape_list.append(shape(blob(cx, cy, 70 + index, count=10 + index % 7), feather=float(index % 5 * 4),
                                    opacity=0.6 + 0.4 * (index % 2), color=[1.0, 0.5 + 0.01 * index, 0.2, 1.0],
                                    feather_falloff=("linear", "smooth", "gaussian")[index % 3]))
        self.check("30 shapes at 1080p", shape_list, 1920, 1080)

    def test_a_shape_with_more_than_the_crossing_limit_on_one_line_runs_on_the_processor(self):
        teeth = rotogpu.MAX_CROSSINGS // 2 + 20
        saw = [shape(comb(2, 20, teeth=teeth, width=2, height=30))]
        with self.assertRaises(rotogpu.Unsupported):
            rotogpu.rasterise(resolve(saw), teeth * 4 + 8, 90)

    def test_the_result_does_not_alias_between_calls(self):
        resolved = resolve(CASES["convex quad"][0])
        first = rotogpu.rasterise(resolved, 200, 200)
        keep = first.copy()
        rotogpu.rasterise(resolve(CASES["concave curve"][0]), 200, 200)
        np.testing.assert_array_equal(first, keep)


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.saved = rotogpu._wanted, rotogpu._wanted_pid, rotogpu._failures
        rotogpu.set_enabled(False)
        rotogpu.reset_counters()
        self.resolved = resolve(CASES["feather 12 smooth"][0])

    def tearDown(self):
        rotogpu._wanted, rotogpu._wanted_pid, rotogpu._failures = self.saved

    def test_it_is_off_by_default_and_the_card_is_never_touched(self):
        self.assertFalse(rotogpu.enabled(environ={}))
        with mock.patch.object(rotogpu, "rasterise", side_effect=AssertionError("the card ran")), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(rotogpu.ENV, None)
            result = roto.rasterise(self.resolved, 200, 200)
        self.assertGreater(float(result[..., 3].max()), 0.9)
        self.assertEqual(rotogpu.counters["gpu_mattes"], 0)

    def test_the_preference_and_the_environment(self):
        self.assertFalse(rotogpu.enabled(environ={}))
        rotogpu.set_enabled(True)
        self.assertTrue(rotogpu.enabled(environ={}))
        self.assertFalse(rotogpu.enabled(environ={rotogpu.ENV: "0"}))
        rotogpu.set_enabled(False)
        self.assertTrue(rotogpu.enabled(environ={rotogpu.ENV: "1"}))
        self.assertTrue(rotogpu.forced({rotogpu.ENV: "0"}))
        self.assertFalse(rotogpu.forced({}))

    def test_a_child_process_never_uses_the_card(self):
        rotogpu.set_enabled(True)
        with mock.patch("multiprocessing.parent_process", return_value=object()):
            self.assertFalse(rotogpu.enabled(environ={rotogpu.ENV: "1"}))

    def test_a_graphics_error_falls_back_to_the_processor_matte(self):
        expected = processor(self.resolved, 200, 200)
        rotogpu.set_enabled(True)
        with mock.patch.object(rotogpu, "rasterise", side_effect=RuntimeError("device lost")):
            result = roto.rasterise(self.resolved, 200, 200)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(rotogpu.counters["errors"], 1)
        self.assertEqual(rotogpu.counters["cpu_fallbacks"], 1)
        self.assertIn("device lost", rotogpu.counters["last_error"])
        self.assertTrue(rotogpu.enabled(environ={}))

    def test_three_errors_in_a_row_turn_the_path_off_for_the_session(self):
        rotogpu.set_enabled(True)
        with mock.patch.object(rotogpu, "rasterise", side_effect=RuntimeError("device lost")) as call:
            for _ in range(5):
                roto.rasterise(self.resolved, 200, 200)
        self.assertEqual(call.call_count, rotogpu.FAIL_LIMIT)
        self.assertFalse(rotogpu.enabled(environ={}))
        rotogpu.set_enabled(True)                        # asking again in Settings tries once more
        self.assertTrue(rotogpu.enabled(environ={}))

    def test_no_adapter_is_a_graphics_error_and_the_processor_draws(self):
        expected = processor(self.resolved, 200, 200)
        rotogpu.set_enabled(True)
        with mock.patch.object(gpu3d, "_state", side_effect=RuntimeError("wgpu unavailable (default)")):
            result = roto.rasterise(self.resolved, 200, 200)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(rotogpu.counters["errors"], 1)

    def test_what_the_kernel_does_not_cover_runs_on_the_processor_without_counting_an_error(self):
        declined = {
            "an open stroke": shape([point(10, 10), point(100, 50), point(150, 120)], closed=False, stroke_width=8.0),
            "the edge feather": shape(blob(100, 100, 50), feather=10.0, feather_model="edge"),
            "a Merge-style blend": shape(blob(100, 100, 50), blend_mode="overlay"),
            "a picture source": shape(blob(100, 100, 50), source="background"),
        }
        rotogpu.set_enabled(True)
        for name, odd in declined.items():
            with self.subTest(name):
                resolved = resolve([shape(blob(60, 60, 30)), odd])
                with self.assertRaises(rotogpu.Unsupported):
                    rotogpu.plan(resolved, 200, 200)
                with mock.patch.object(rotogpu, "rasterise", wraps=rotogpu.rasterise):
                    result = roto.rasterise(resolved, 200, 200)
                np.testing.assert_array_equal(result, processor(resolved, 200, 200))
        self.assertEqual(rotogpu.counters["errors"], 0)
        self.assertEqual(rotogpu.counters["gpu_mattes"], 0)
        self.assertEqual(rotogpu.counters["unsupported"], len(declined))

    def test_motion_blur_and_a_wide_feather_are_declined(self):
        resolved = resolve([shape(blob(100, 100, 50), feather=float(rotogpu.MAX_BLUR_RADIUS + 10))])
        with self.assertRaises(rotogpu.Unsupported):
            rotogpu.plan(resolved, 600, 600)
        sampled = copy.deepcopy(self.resolved)
        sampled[0]["_motion_samples"] = [sampled[0]["points"]]
        with self.assertRaises(rotogpu.Unsupported):
            rotogpu.plan(sampled, 200, 200)

    def test_a_tile_request_stays_on_the_processor(self):
        rotogpu.set_enabled(True)
        with mock.patch.object(rotogpu, "rasterise", side_effect=AssertionError("the card ran")):
            tile = roto.rasterise(self.resolved, 200, 200, region=(20, 20, 64, 64))
        np.testing.assert_array_equal(tile, processor(self.resolved, 200, 200, region=(20, 20, 64, 64)))

    @unittest.skipUnless(card_available(), "no graphics adapter")
    def test_with_the_switch_on_a_covered_matte_comes_from_the_card_and_matches(self):
        rotogpu.set_enabled(True)
        result = roto.rasterise(self.resolved, 200, 200)
        self.assertEqual(rotogpu.counters["gpu_mattes"], 1)
        self.assertEqual(rotogpu.counters["cpu_fallbacks"], 0)
        self.assertLess(float(np.abs(result - processor(self.resolved, 200, 200)).max()), TOL)

    @unittest.skipUnless(card_available(), "no graphics adapter")
    def test_the_evaluator_draws_a_roto_node_on_the_card_when_switched_on(self):
        from nodebased.core import Dispatcher, empty_document
        from nodebased.imaging import Evaluator
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto", "params": {"width": 200, "height": 200}})
        d.execute({"op": "set_shapes", "id": "r", "shapes": copy.deepcopy(CASES["feather 12 smooth"][0])})
        d.execute({"op": "view", "id": "r"})
        document, node_id = d.document, "r"
        rotogpu.set_enabled(False)
        expected = Evaluator().evaluate_raster(document, target=node_id, frame=1).pixels.copy()
        rotogpu.set_enabled(True)
        rotogpu.reset_counters()
        got = Evaluator().evaluate_raster(document, target=node_id, frame=1).pixels
        self.assertGreaterEqual(rotogpu.counters["gpu_mattes"], 1)
        self.assertLess(float(np.abs(got - expected).max()), TOL)


class SettingTests(unittest.TestCase):
    """Settings -> Interface -> the Roto GPU row: off by default, the environment wins, applied at once."""

    def setUp(self):
        from nodebased import playbackprefs
        self.prefs = playbackprefs
        self.saved = rotogpu._wanted, rotogpu._wanted_pid, rotogpu._failures
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(rotogpu.ENV, None)

    def tearDown(self):
        rotogpu._wanted, rotogpu._wanted_pid, rotogpu._failures = self.saved

    def test_the_default_is_off_and_the_environment_wins(self):
        self.assertFalse(self.prefs.DEFAULT_ROTO_GPU)
        self.assertFalse(self.prefs.roto_gpu(None))
        self.assertTrue(self.prefs.roto_gpu(1))
        self.assertTrue(self.prefs.roto_gpu("1"))
        self.assertFalse(self.prefs.roto_gpu("nonsense"))
        self.assertTrue(self.prefs.roto_gpu(False, environ={self.prefs.ROTO_GPU_ENV: "1"}))
        self.assertFalse(self.prefs.roto_gpu(True, environ={self.prefs.ROTO_GPU_ENV: "0"}))
        self.assertTrue(self.prefs.roto_gpu_forced({self.prefs.ROTO_GPU_ENV: "0"}))
        self.assertFalse(self.prefs.roto_gpu_forced({}))
        self.assertEqual(self.prefs.ROTO_GPU_ENV, rotogpu.ENV)

    def test_the_choice_round_trips_through_the_saved_preferences(self):
        from nodebased.app import Preferences
        preferences = Preferences()
        self.addCleanup(preferences.set_roto_gpu, False)
        preferences.set_roto_gpu(False)
        self.assertFalse(Preferences().roto_gpu())
        preferences.set_roto_gpu(True)
        self.assertTrue(Preferences().roto_gpu())
        preferences.set_roto_gpu(False)                # off is stored as no choice at all
        self.assertFalse(Preferences().roto_gpu())

    def test_the_settings_dialog_shows_and_returns_the_switch(self):
        from nodebased.app import ProjectSettingsDialog
        from nodebased.core import empty_document
        dialog = ProjectSettingsDialog(empty_document()["settings"], roto_gpu=True, roto_gpu_note="fixed by the environment")
        try:
            self.assertTrue(dialog.roto_gpu.isChecked())
            self.assertFalse(dialog.roto_gpu_note.isHidden())
            dialog.roto_gpu.setChecked(False)
            self.assertFalse(dialog.roto_gpu.isChecked())
        finally:
            dialog.close()
        plain = ProjectSettingsDialog(empty_document()["settings"])
        try:
            self.assertFalse(plain.roto_gpu.isChecked())
            self.assertTrue(plain.roto_gpu_note.isHidden())
        finally:
            plain.close()

    def test_a_running_window_switches_the_path_and_redraws(self):
        from nodebased.app import Window
        window = Window()
        self.addCleanup(window.close)
        self.addCleanup(window.preferences.set_roto_gpu, False)
        self.assertFalse(rotogpu.enabled(environ={}))                 # a new window starts with it off
        window.evaluator.cache["stale"] = object()
        window.preferences.set_roto_gpu(True)
        self.assertTrue(window.apply_roto_gpu())
        self.assertTrue(rotogpu.enabled(environ={}))
        self.assertNotIn("stale", window.evaluator.cache)            # the mattes of the other drawer are dropped
        window.preferences.set_roto_gpu(False)
        self.assertFalse(window.apply_roto_gpu())
        self.assertFalse(rotogpu.enabled(environ={}))


def tearDownModule():
    path = os.environ.get("NODEBASED_ROTO_GPU_REPORT")
    if path and REPORT:
        with open(path, "w") as handle:
            json.dump(REPORT, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
