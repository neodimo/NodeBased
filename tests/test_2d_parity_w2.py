"""W2: curve editor tangents, HueCorrect output curves, and viewer-drawn analysis regions."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased import colorcurves
from nodebased.colorcurves import decode, encode, evaluate
from nodebased.core import Dispatcher, SPECS
from nodebased.curveeditor import CurveCanvas, CurveEditorDialog
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

APP = QApplication.instance() or QApplication([])


def keyed(points, slopes, modes=None, broken=None):
    n = len(points)
    return decode(encode(points, "linear", slopes, modes or ["broken"] * n, broken or [False] * n))


class TangentTests(unittest.TestCase):
    def test_a_tangent_change_moves_the_midpoint_by_the_hand_computed_amount(self):
        # Hermite midpoint (t = 1/2, h = 1): y = (y0 + y1)/2 + (m0 - m1)/8.
        curve = keyed(((0, 0), (1, 1)), [[1, 1], [1, 1]])
        self.assertAlmostEqual(evaluate(curve, .5), .5, places=12)
        colorcurves.set_tangent(curve, 0, 1, 3.0)
        self.assertAlmostEqual(evaluate(curve, .5), .5 + (3 - 1) / 8, places=12)
        colorcurves.set_tangent(curve, 1, 0, -1.0)
        self.assertAlmostEqual(evaluate(curve, .5), .5 + (3 - -1) / 8, places=12)

    def test_a_broken_tangent_leaves_the_other_side_unchanged(self):
        curve = keyed(((0, 0), (1, 1), (2, 0)), [[0, 0], [0.5, 0.5], [0, 0]])
        before_left = [evaluate(curve, x / 20) for x in range(0, 11)]
        colorcurves.set_tangent(curve, 1, 1, -4.0, break_tangent=True)
        self.assertTrue(curve["broken"][1])
        self.assertEqual(curve["slopes"][1], [0.5, -4.0])
        self.assertEqual([evaluate(curve, x / 20) for x in range(0, 11)], before_left)
        self.assertNotAlmostEqual(evaluate(curve, 1.5), 0.5, places=3)
        # A joined key moves both sides with one handle.
        joined = keyed(((0, 0), (1, 1), (2, 0)), [[0, 0], [0.5, 0.5], [0, 0]])
        colorcurves.set_tangent(joined, 1, 1, 2.0)
        self.assertEqual(joined["slopes"][1], [2.0, 2.0])

    def test_a_broken_tangent_beside_a_smooth_segment_keeps_that_segment(self):
        curve = decode(encode(((0, 0), (1, 1), (2, 0)), "linear"))
        colorcurves.ensure_keyed(curve)
        curve["modes"] = ["smooth", "smooth", "smooth"]
        before, right = [evaluate(curve, x / 20) for x in range(0, 21)], evaluate(curve, 1.5)
        colorcurves.set_tangent(curve, 1, 1, 5.0, break_tangent=True)
        self.assertEqual([evaluate(curve, x / 20) for x in range(0, 21)], before)  # left half untouched
        self.assertNotAlmostEqual(evaluate(curve, 1.5), right, places=3)

    def test_adding_a_key_on_a_curve_keeps_its_shape(self):
        for interpolation in ("linear", "smooth"):
            original = decode(encode(((0, 0), (.3, .7), (1, .2)), interpolation))
            curve = copy.deepcopy(original)
            index = colorcurves.insert_key(curve, .5)
            self.assertEqual(index, 2)
            self.assertAlmostEqual(curve["points"][index][1], evaluate(original, .5), places=12)
            for i in range(101):
                self.assertAlmostEqual(evaluate(curve, i / 100), evaluate(original, i / 100), places=9)
        hermite = keyed(((0, 0), (1, 2), (2, 1)), [[0, 3], [-1, 0.5], [0, 0]],
                        ["smooth", "broken", "broken"])
        curve = copy.deepcopy(hermite)
        colorcurves.insert_key(curve, 1.4)
        colorcurves.insert_key(curve, .2)
        for i in range(201):
            self.assertAlmostEqual(evaluate(curve, i / 100), evaluate(hermite, i / 100), places=9)
        np.testing.assert_allclose(colorcurves.evaluate_array(curve, np.linspace(-.2, 2.2, 50)),
                                   [evaluate(curve, x) for x in np.linspace(-.2, 2.2, 50)], atol=1e-6)


class CurveDialogTests(unittest.TestCase):
    def test_numeric_fields_follow_and_edit_the_selected_point_and_reset_restores_the_default(self):
        default = encode(((0, 0), (1, 1)))
        seen = []
        dialog = CurveEditorDialog("t", encode(((0, 0), (.5, .8), (1, 1))), seen.append, default=default)
        dialog.show()
        dialog.canvas.select(1)
        self.assertAlmostEqual(dialog.point_x.value(), .5)
        self.assertAlmostEqual(dialog.point_y.value(), .8)
        dialog.point_y.setValue(.6); dialog.point_y.editingFinished.emit()
        self.assertAlmostEqual(decode(seen[-1])["points"][1][1], .6)
        dialog.slope_out.setValue(2.0); dialog.slope_out.editingFinished.emit()
        curve = decode(seen[-1])
        self.assertEqual(curve["slopes"][1], [2.0, 2.0])
        self.assertEqual(dialog.slope_in.value(), 2.0)
        dialog.reset_curve()
        self.assertEqual(decode(seen[-1]), decode(default))
        dialog.close()

    def test_ctrl_dragging_a_handle_breaks_it_and_a_plain_drag_keeps_both_sides_joined(self):
        for ctrl, expect_broken in ((True, True), (False, False)):
            canvas = CurveCanvas(encode(((0, 0), (.5, .5), (1, 1))))
            canvas.resize(440, 280); canvas.show(); APP.processEvents()
            handle = canvas._handle_pos(1, 1).toPoint()
            destination = canvas._xy(.5 + .1, .5 + .4).toPoint()
            modifier = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, modifier, handle)
            QTest.mouseMove(canvas, destination, delay=20)
            canvas.mouseMoveEvent(type("E", (), {"position": lambda s: canvas._xy(.6, .9),
                                                  "modifiers": lambda s: modifier})())
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, modifier, destination)
            slopes = canvas.curve["slopes"][1]
            self.assertEqual(canvas.curve["broken"][1], expect_broken)
            self.assertEqual(slopes[0] == slopes[1], not expect_broken)
            canvas.close()


class HueCorrectOutputTests(unittest.TestCase):
    def test_the_sat_curve_changes_saturation_and_leaves_luminance_alone(self):
        rng = np.random.default_rng(3)
        image = np.concatenate([rng.random((9, 13, 3), dtype=np.float32), np.ones((9, 13, 1), np.float32)], 2)
        p = SPECS["HueCorrect"]["params"].copy()
        p["curve_sat"] = encode(((0, .5), (360, .5)))
        out = Evaluator._hue_correct(image, p)
        weights = np.array([.2126, .7152, .0722], np.float32)
        np.testing.assert_allclose(out[..., :3] @ weights, image[..., :3] @ weights, atol=1e-5)
        spread = lambda a: a[..., :3].max(-1) - a[..., :3].min(-1)
        np.testing.assert_allclose(spread(out), spread(image) * .5, atol=1e-5)

    def test_the_suppress_curves_are_separate_from_the_channel_response(self):
        image = np.array([[[.8, .4, .3, 1], [.2, .6, .1, 1]]], np.float32)
        p = SPECS["HueCorrect"]["params"].copy()
        p["curve_red"] = encode(((0, .5), (360, .5)))
        response = Evaluator._hue_correct(image, p)
        np.testing.assert_allclose(response[0, 0, :3], [.4, .4, .3], atol=1e-6)
        p = SPECS["HueCorrect"]["params"].copy()
        p["curve_r_sup"] = encode(((0, 1), (360, 1)))
        out = Evaluator._hue_correct(image, p)
        # Full suppression limits red to the larger of green and blue; a pixel whose red is
        # already lowest is untouched, and the other channels never change.
        np.testing.assert_allclose(out[0, 0], [.4, .4, .3, 1], atol=1e-6)
        np.testing.assert_array_equal(out[0, 1], image[0, 1])
        p["curve_r_sup"] = encode(((0, .5), (360, .5)))
        np.testing.assert_allclose(Evaluator._hue_correct(image, p)[0, 0, 0], .6, atol=1e-6)

    def test_every_output_curve_matches_the_tile_path(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {
            "width": 16, "height": 8, "red": .8, "green": .5, "blue": .4, "alpha": 1}})
        params = SPECS["HueCorrect"]["params"].copy()
        for i, name in enumerate(("sat", "lum", "red", "green", "blue")):
            params[f"curve_{name}"] = encode(((0, .6 + i / 10), (180, 1.2), (360, .6 + i / 10)), "smooth")
        for name in ("r_sup", "g_sup", "b_sup"):
            params[f"curve_{name}"] = encode(((0, .3), (360, .3)))
        d.execute({"op": "create", "id": "fx", "type": "HueCorrect", "params": params})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        whole = Evaluator().evaluate(dict(d.document, view="fx"))
        tile = TileExecutor(tile_edge=5)
        self.assertTrue(tile.supports_tiled(d.document, "fx"))
        region = tile.canvas_region(d.document, "fx", 1, 1)
        np.testing.assert_allclose(tile.compose_region(d.document, "fx", region, 1, 1).pixels, whole, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
