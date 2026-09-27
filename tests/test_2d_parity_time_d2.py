import threading
import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.cancellation import Cancelled


class TimeParityD2Tests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, inputs=None):
        self.d.execute({"op": "create", "id": key, "type": kind, "params": params or {}})
        for slot, source in (inputs or {}).items():
            self.d.execute({"op": "connect", "id": key, "input": slot, "source": source})
        return key

    def evaluate(self, key, frame=1, cancel=None):
        return Evaluator().evaluate_raster(self.d.document, key, frame=frame, cancel=cancel).to_display()

    def test_timeblur_static_is_identical_and_does_not_cache_fractional_samples(self):
        self.add("src", "Constant", {"width": 4, "height": 3, "red": 0.3, "green": 0.2, "blue": 0.1})
        self.add("blur", "TimeBlur", {"shutter": 1, "divisions": 12}, {"image": "src"})
        original = Evaluator().evaluate_raster(self.d.document, "src", frame=1).pixels
        evaluator = Evaluator()
        actual = evaluator.evaluate_raster(self.d.document, "blur", frame=1).pixels
        np.testing.assert_array_equal(actual, original)
        self.assertEqual(len(evaluator.cache), 1, "fractional subframe results must not enter the persistent cache")

    def test_timeblur_samples_animated_transform_fractionally_with_conserved_energy(self):
        self.add("src", "Constant", {"width": 24, "height": 3, "alpha": 0, "red": 0, "green": 0, "blue": 0})
        self.add("dot", "Rectangle", {"width": 24, "height": 3, "box_x": 7, "box_y": 1,
                                         "box_width": 1, "box_height": 1, "red": 1, "green": 1, "blue": 1})
        self.add("move", "Transform", {"filter": "nearest"}, {"image": "dot"})
        self.add("blur", "TimeBlur", {"shutter": 1, "divisions": 10, "shutter_offset": "centred"}, {"image": "move"})
        self.d.document["animation"]["curves"]["move"] = {"translate_x": {"interpolation": "linear", "keys": [
            {"frame": 0, "value": 0}, {"frame": 2, "value": 20}]}}
        pixels = self.evaluate("blur", frame=1)
        alpha = pixels[..., 3]
        self.assertEqual(np.count_nonzero(alpha), 10)
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)

    def test_timeecho_max_collects_current_and_two_previous_dots(self):
        self.add("src", "Rectangle", {"width": 8, "height": 1, "box_x": 0, "box_y": 0,
                                        "box_width": 1, "box_height": 1, "red": 1, "green": 1, "blue": 1})
        self.add("move", "Transform", {"filter": "nearest"}, {"image": "src"})
        self.add("echo", "TimeEcho", {"frames": 3, "method": "max", "falloff": 1}, {"image": "move"})
        self.d.document["animation"]["curves"]["move"] = {"translate_x": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 0}, {"frame": 3, "value": 2}]}}
        alpha = self.evaluate("echo", frame=3)[0, :, 3]
        self.assertEqual(np.flatnonzero(alpha).tolist(), [0, 1, 2])


    def test_timeecho_plus_and_average_apply_the_falloff_weights(self):
        self.add("src", "Constant", {"width": 1, "height": 1, "red": 0.25, "green": 0, "blue": 0})
        for method, expected in (("plus", 0.4375), ("average", 0.25)):
            self.add(method, "TimeEcho", {"frames": 3, "method": method, "falloff": 0.5}, {"image": "src"})
            self.assertAlmostEqual(float(self.evaluate(method, frame=5)[0, 0, 0]), expected, places=6)

    def test_timedissolve_linear_boundaries_and_midpoint(self):
        self.add("a", "Constant", {"width": 1, "height": 1, "red": 0, "green": 0, "blue": 0})
        self.add("b", "Constant", {"width": 1, "height": 1, "red": 1, "green": 1, "blue": 1})
        self.add("mix", "TimeDissolve", {"in": 2, "out": 6, "ease": "linear"}, {"A": "a", "B": "b"})
        self.assertEqual(float(self.evaluate("mix", 1)[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(self.evaluate("mix", 4)[0, 0, 0]), 0.5, places=6)
        self.assertEqual(float(self.evaluate("mix", 7)[0, 0, 0]), 1.0)

    def test_timedissolve_can_follow_an_animated_which_curve(self):
        self.add("a", "Constant", {"width": 1, "height": 1, "red": 0, "green": 0, "blue": 0})
        self.add("b", "Constant", {"width": 1, "height": 1, "red": 1, "green": 1, "blue": 1})
        self.add("mix", "TimeDissolve", {"in": 1, "out": 10, "ease": "animation curve"},
                 {"A": "a", "B": "b"})
        self.d.document["animation"]["curves"]["mix"] = {"which": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 0}, {"frame": 10, "value": 1}]}}
        self.assertAlmostEqual(float(self.evaluate("mix", 5)[0, 0, 0]), 4 / 9, places=6)

    def test_timeblur_cancel_bypass_and_tiles_fall_back(self):
        self.add("src", "Constant", {"width": 3, "height": 2, "red": 0.25})
        self.add("blur", "TimeBlur", {"shutter": 8, "divisions": 200}, {"image": "src"})
        cancelled = threading.Event(); cancelled.set()
        with self.assertRaises(Cancelled):
            Evaluator().evaluate_raster(self.d.document, "blur", frame=1, cancel=cancelled)
        self.d.document["nodes"]["blur"]["disabled"] = True
        expected = Evaluator().evaluate_raster(self.d.document, "src", frame=1).pixels
        np.testing.assert_array_equal(Evaluator().evaluate_raster(self.d.document, "blur", frame=1).pixels, expected)
        from nodebased.tileexec import TileExecutor
        self.assertFalse(TileExecutor().supports_tiled(self.d.document, "blur"))


if __name__ == "__main__":
    unittest.main()
