"""W2: curve editor tangents, HueCorrect output curves, and viewer-drawn analysis regions."""
import copy
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased import analysisregion, colorcurves
from nodebased.animation import resolve_document
from nodebased.app import STYLE, Window
from nodebased.imaging import curve_tool_metrics
from tests.waiting import wait_until
from nodebased.colorcurves import decode, encode, evaluate
from nodebased.core import Dispatcher, SPECS
from nodebased.curveeditor import CurveCanvas, CurveEditorDialog
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


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


class CurveTileTests(unittest.TestCase):
    def test_keyed_tangent_curves_match_the_tile_path_on_color_lookup_and_crosstalk(self):
        bent = encode(((0, 0), (.5, .5), (1, 1)), "linear", [[2, 2], [-.5, -.5], [3, 3]],
                      ["broken", "broken", "broken"], [False] * 3)
        for kind, knobs in (("ColorLookup", {"curve_master": bent, "curve_green": bent}),
                            ("CrossTalk", {"xt_curve_r_g": bent, "xt_curve_b_b": bent})):
            d = Dispatcher()
            d.execute({"op": "create", "id": "src", "type": "Ramp", "params": {"width": 16, "height": 8, "p1_x": 15.0}})
            d.execute({"op": "create", "id": "fx", "type": kind, "params": dict(SPECS[kind]["params"], **knobs)})
            d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
            whole = Evaluator().evaluate(dict(d.document, view="fx"))
            tile = TileExecutor(tile_edge=5)
            self.assertTrue(tile.supports_tiled(d.document, "fx"))
            region = tile.canvas_region(d.document, "fx", 1, 1)
            np.testing.assert_allclose(tile.compose_region(d.document, "fx", region, 1, 1).pixels, whole, atol=1e-6)
            plain = Evaluator().evaluate(dict(d.document, view="src"))
            self.assertGreater(float(np.abs(whole[..., :3] - plain[..., :3]).max()), .05)


class BoxGeometryTests(unittest.TestCase):
    def test_corner_edge_and_body_drags_move_the_right_sides(self):
        box = {"box_x": 20.0, "box_y": 20.0, "box_width": 100.0, "box_height": 50.0}
        self.assertEqual(analysisregion.box_after_drag(box, "nw", 10, 5),
                         {"box_x": 30, "box_y": 25, "box_width": 90, "box_height": 45})
        self.assertEqual(analysisregion.box_after_drag(box, "e", -30, 99)["box_width"], 70)
        self.assertEqual(analysisregion.box_after_drag(box, "s", 99, 10)["box_height"], 60)
        self.assertEqual(analysisregion.box_after_drag(box, "body", 7, -3),
                         {"box_x": 27, "box_y": 17, "box_width": 100, "box_height": 50})
        self.assertEqual(analysisregion.box_after_drag(box, "w", 500, 0)["box_width"], 1.0)
        empty = dict(box, box_width=0.0, box_height=0.0)
        self.assertEqual(analysisregion.box_after_drag(empty, "se", 6, 4)["box_width"], 70.0)


def ramp_document(kind):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "create", "id": "src", "type": "Ramp",
                        "params": {"width": 320, "height": 240, "p1_x": 319.0}})
    dispatcher.execute({"op": "create", "id": "analysis", "type": kind})
    dispatcher.execute({"op": "connect", "id": "analysis", "input": "image", "source": "src"})
    dispatcher.execute({"op": "view", "id": "analysis"})
    return dispatcher.document


class RegionDragTests(unittest.TestCase):
    def open(self, kind):
        self.window = Window(ramp_document(kind))
        self.window.resize(1600, 1000)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.viewer.format_rect is not None))
        # Let the default dock split land first: the viewer refits on resize until the artist zooms, so a late
        # split would undo the transform set below between the drag and the check (10/10 full suite).
        last = [None, 0.0]

        def layout_settled():
            size = self.window.viewer.size()
            if size != last[0]:
                last[:] = [size, time.monotonic()]
                return False
            return time.monotonic() - last[1] >= 0.3
        self.assertTrue(wait_until(layout_settled))
        self.window.viewer.resetTransform()
        self.window.viewer.centerOn(130, 36)  # the default layout leaves the viewer only ~60 px tall
        viewport = self.window.viewer.viewport().rect()
        for corner in ((32, 32), (200, 62)):
            self.assertTrue(viewport.contains(self.scene(*corner)), "viewport too small for the drag points")
        self.window.graph.items_by_id["analysis"].setSelected(True)
        self.window.properties_dock.show()
        APP.processEvents()
        self.assertTrue(wait_until(lambda: self.window.viewer._analysis_context() is not None))
        self.window.dispatcher.undo_stack.clear()

    def tearDown(self):
        window = getattr(self, "window", None)
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close()
            APP.processEvents()

    def scene(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def drag(self, start, end):
        viewport = self.window.viewer.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, end, 20)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
        APP.processEvents()

    def params(self, frame=None):
        document = self.window.dispatcher.document
        frame = int(document["time"]["current"]) if frame is None else frame
        return resolve_document(document, frame)["nodes"]["analysis"]["params"]

    def test_a_mincolor_box_drag_moves_its_knobs_and_the_analysed_minimum(self):
        self.open("MinColor")
        self.assertAlmostEqual(self.params()["mincolor_r"], 0.0)
        self.drag(self.scene(32, 32), self.scene(132, 32))
        p = self.params()
        self.assertAlmostEqual(p["box_x"], 100, delta=2)
        self.assertAlmostEqual(p["box_width"], 64, delta=1)
        # The ramp runs 0..1 over 319 pixels, so the darkest pixel of the box is its left edge.
        self.assertAlmostEqual(p["mincolor_r"], p["box_x"] / 319, delta=.01)
        self.assertGreater(p["mincolor_r"], .25)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        p = self.params()
        self.assertEqual((p["box_x"], p["mincolor_r"]), (0.0, 0.0))

    def test_a_mincolor_edge_grip_resizes_only_that_side(self):
        self.open("MinColor")
        self.drag(self.scene(64, 32), self.scene(144, 32))
        p = self.params()
        self.assertAlmostEqual(p["box_x"], 0, delta=1)
        self.assertAlmostEqual(p["box_width"], 144, delta=2)
        self.assertAlmostEqual(p["box_height"], 64, delta=1)
        self.assertAlmostEqual(p["mincolor_r"], 0.0, delta=.01)

    def test_a_curvetool_region_is_drawn_dragged_and_measured_on_release(self):
        self.open("CurveTool")
        self.assertEqual(self.window.viewer._analysis_context()[1]["type"], "CurveTool")
        self.drag(self.scene(32, 32), self.scene(132, 62))
        p = self.params()
        self.assertAlmostEqual(p["box_x"], 100, delta=2)
        self.assertAlmostEqual(p["box_y"], 30, delta=2)
        document = self.window.graph_document()
        raster = self.window.evaluator.evaluate_raster(document, target="src", frame=int(document["time"]["current"]))
        expected = curve_tool_metrics(raster.to_display(), (p["box_x"], p["box_y"], p["box_width"], p["box_height"]))
        self.assertGreater(expected["average_r"], .2)
        self.assertAlmostEqual(p["average_r"], expected["average_r"], places=5)
        self.assertAlmostEqual(p["average_luminance"], expected["average_luminance"], places=5)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_the_region_is_keyframable(self):
        self.open("MinColor")
        self.window.command({"op": "set_key", "id": "analysis", "param": "box_x", "frame": 1, "value": 10.0})
        self.window.command({"op": "set_key", "id": "analysis", "param": "box_x", "frame": 10, "value": 50.0})
        self.window.command({"op": "set", "id": "analysis", "param": "box_width", "value": 64.0})
        self.window.command({"op": "set", "id": "analysis", "param": "box_height", "value": 64.0})
        frame = int(self.window.dispatcher.document["time"]["current"])
        before_other = self.params(10)["box_x"]
        self.window.dispatcher.undo_stack.clear()
        start = self.params(frame)["box_x"]
        self.drag(self.scene(start + 32, 32), self.scene(start + 62, 32))
        self.assertAlmostEqual(self.params(frame)["box_x"], start + 30, delta=2)
        self.assertEqual(self.params(10)["box_x"], before_other)
        curve = self.window.node_curve("analysis", "box_x")
        self.assertIsNotNone(curve)
        self.assertIsNone(self.window.node_curve("analysis", "box_y"))
        self.assertAlmostEqual(self.params(frame)["mincolor_r"], self.params(frame)["box_x"] / 319, delta=.01)


if __name__ == "__main__":
    unittest.main()
