"""Roto B-Spline tension: a per-point scalar (Ctrl+Alt-drag), default 0 = the plain uniform cubic B-spline, animatable under autokey."""
import copy
import os
import tempfile
import unittest

import numpy as np

import tests.isolation
from tests.test_roto_walkthrough import load_walkthrough
from nodebased import roto, shapes

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def square(tension=None):
    points = [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
              for x, y in ((10, 10), (50, 10), (50, 50), (10, 50))]
    if tension is not None:
        points[1]["tension"] = tension
    return {"name": "s", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "curve_type": "bspline", "points": points}


class TensionMathTests(unittest.TestCase):
    def test_absent_and_zero_tension_render_identically(self):
        plain = roto.rasterise(shapes.resolve_shapes({"shapes": [square()]}, 1), 64, 64)
        zero = roto.rasterise(shapes.resolve_shapes({"shapes": [square(0.0)]}, 1), 64, 64)
        self.assertTrue(np.array_equal(plain, zero))
        resolved = shapes.resolve_shapes({"shapes": [square()]}, 1)
        self.assertNotIn("tension", resolved[0]["points"][1])

    def test_tension_one_puts_the_curve_through_the_point(self):
        points = square(1.0)["points"]
        controls = roto.tensioned_controls(points, True)
        # The uniform cubic B-spline reaches (P[i-1] + 4 P[i] + P[i+1]) / 6 at a control: the effective control c makes that equal P.
        before, after = points[0], points[2]
        reached = [(before[k] + 4 * controls[1][k] + after[k]) / 6.0 for k in ("x", "y")]
        self.assertAlmostEqual(reached[0], points[1]["x"], places=9)
        self.assertAlmostEqual(reached[1], points[1]["y"], places=9)

    def test_negative_tension_loosens_and_positive_tightens(self):
        area = {t: float(roto.coverage(square(t)["points"], 64, 64, curve_type="bspline").sum()) for t in (-1.0, 0.0, 1.0)}
        self.assertLess(area[-1.0], area[0.0])
        self.assertLess(area[0.0], area[1.0])

    def test_only_the_tensioned_point_changes_its_control(self):
        controls = roto.tensioned_controls(square(0.6)["points"], True)
        original = square(0.6)["points"]
        for index in (0, 2, 3):
            self.assertEqual((controls[index]["x"], controls[index]["y"]), (original[index]["x"], original[index]["y"]))
        self.assertNotEqual((controls[1]["x"], controls[1]["y"]), (original[1]["x"], original[1]["y"]))

    def test_open_spline_stroke_with_tension_renders(self):
        pts = square(1.0)["points"]
        for p in pts:
            p["feather_x"] = p["feather_y"] = 0.0
        flat = roto.stroke_coverage(pts, 64, 64, 4.0, 0.0, "linear", curve_type="bspline")
        none = roto.stroke_coverage([{k: v for k, v in p.items() if k != "tension"} for p in pts], 64, 64, 4.0, 0.0, "linear",
                                    curve_type="bspline")
        self.assertEqual(flat.shape, (64, 64))
        self.assertFalse(np.array_equal(flat, none))

    def test_validation_and_animation(self):
        shape = square(0.5)
        shapes.validate_shape(shape, "s")
        animated = copy.deepcopy(shape)
        animated["points"][1]["tension"] = {"value": 0.0, "curve": {"interpolation": "linear",
                                                                      "keys": [{"frame": 1, "value": 0.0}, {"frame": 11, "value": 1.0}]}}
        shapes.validate_shape(animated, "s")
        self.assertAlmostEqual(shapes.resolve_shapes({"shapes": [animated]}, 6)[0]["points"][1]["tension"], 0.5)
        bad = square(2.0)
        with self.assertRaises(ValueError):
            shapes.validate_shape(bad, "s")


class TensionGestureTests(unittest.TestCase):
    def test_ctrl_alt_drag_sets_tension_with_one_undo_and_autokey(self):
        walk_module = load_walkthrough()
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            result = walk_module.Result(0, "t", [], "t")
            w = walk_module.Walk(app, folder, result)
            try:
                w.fresh("roto", shapes=[square()], last=24, current=1)
                w.tool("Select", "Select Points")
                depth = w.undo_depth()
                w.drag((50, 10), (110, 10), mods=walk_module.CTRL | Qt.KeyboardModifier.AltModifier)
                tension = w.shape_entries()[0]["points"][1]["tension"]
                self.assertEqual(w.undo_depth(), depth + 1)
                # Autokey is on: the value is keyed at the current frame.
                value = tension["curve"]["keys"][0] if isinstance(tension, dict) else None
                self.assertTrue(isinstance(tension, dict) and value["frame"] == 1 and value["value"] > 0)
                w.goto(10)
                w.drag((50, 10), (20, 10), mods=walk_module.CTRL | Qt.KeyboardModifier.AltModifier)
                keys = w.shape_entries()[0]["points"][1]["tension"]["curve"]["keys"]
                self.assertEqual([k["frame"] for k in keys], [1, 10])
                self.assertTrue(keys[1]["value"] < keys[0]["value"])
            finally:
                w.close()


if __name__ == "__main__":
    unittest.main()
