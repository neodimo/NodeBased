"""Offscreen interaction tests for the Viewer Transform foreground handle."""
import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document
from nodebased import handles2d


APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        QTest.qWait(10)
    return False


def transform_document():
    document = empty_document()
    document["nodes"] = {}
    dispatcher = Dispatcher(document)
    dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0]})
    dispatcher.execute({"op": "create", "id": "t", "type": "Transform", "pos": [150, 0]})
    dispatcher.execute({"op": "connect", "id": "t", "input": "image", "source": "c"})
    dispatcher.execute({"op": "view", "id": "t"})
    return dispatcher.document


class TransformHandleTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(transform_document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None and self.window.viewer.format_rect is not None))
        self.window.graph.items_by_id["t"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def scene_pos(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def drag(self, start, target, modifiers=Qt.KeyboardModifier.NoModifier):
        viewer = self.window.viewer
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, modifiers, start)
        QTest.mouseMove(viewer.viewport(), target, 20)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, modifiers, target)

    def test_visibility_rule(self):
        viewer = self.window.viewer
        self.assertIsNotNone(viewer._transform_context())
        self.window.command({"op": "create", "id": "g", "type": "Grade", "pos": [300, 0]})
        self.window.command({"op": "connect", "id": "g", "input": "image", "source": "t"})
        self.window.command({"op": "view", "id": "g"})
        self.assertTrue(wait_until(lambda: viewer.format_rect is not None))
        self.assertIsNotNone(viewer._transform_context())
        self.window.command({"op": "create", "id": "s", "type": "Constant", "pos": [0, 150]})
        self.window.command({"op": "view", "id": "s"})
        self.assertIsNone(viewer._transform_context())
        self.window.command({"op": "view", "id": "g"})
        self.window.graph.items_by_id["t"].setSelected(False)
        self.window.graph.items_by_id["g"].setSelected(True)
        self.assertIsNone(viewer._transform_context())

    def test_translate_drag_and_undo(self):
        viewer, params = self.window.viewer, self.window.dispatcher.document["nodes"]["t"]["params"]
        before = (params["translate_x"], params["translate_y"])
        start, target = self.scene_pos(300, 250), self.scene_pos(340, 280)
        self.drag(start, target)
        drop, press = viewer.mapToScene(target), viewer.mapToScene(start)
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        self.assertAlmostEqual(p["translate_x"], before[0] + drop.x() - press.x(), delta=.01)
        self.assertAlmostEqual(p["translate_y"], before[1] + drop.y() - press.y(), delta=.01)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        self.assertEqual((p["translate_x"], p["translate_y"]), before)

    def test_corner_drag_scales(self):
        viewer = self.window.viewer
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        corner = handles2d.box_corners(viewer.format_rect.width(), viewer.format_rect.height(),
                                       p["translate_x"], p["translate_y"], p["rotate"], p["scale"],
                                       p["center_x"], p["center_y"])[2]
        pivot = handles2d.pivot_point(p["translate_x"], p["translate_y"], p["center_x"], p["center_y"])
        start = self.scene_pos(*corner)
        target = self.scene_pos(corner[0] + 80, corner[1] + 60)
        self.drag(start, target)
        expected = handles2d.scale_from_drag(1, pivot, viewer._transform_data_point(viewer.mapToScene(start)),
                                             viewer._transform_data_point(viewer.mapToScene(target)))
        self.assertAlmostEqual(self.window.dispatcher.document["nodes"]["t"]["params"]["scale"], expected, delta=.01)

    def test_ring_drag_rotates(self):
        viewer = self.window.viewer
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        # Default centre is a corner; move it so the ring has a useful radius.
        self.window.command({"op": "set", "id": "t", "param": "center_x", "value": 300.0})
        self.window.command({"op": "set", "id": "t", "param": "center_y", "value": 250.0})
        self.window.dispatcher.undo_stack.clear()
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        pvt = handles2d.pivot_point(p["translate_x"], p["translate_y"], p["center_x"], p["center_y"])
        corners = handles2d.box_corners(viewer.format_rect.width(), viewer.format_rect.height(), 0, 0, 0, 1, 300, 250)
        radius = 1.15 * np.hypot(corners[0][0] - pvt[0], corners[0][1] - pvt[1])
        start, target = self.scene_pos(pvt[0] + radius, pvt[1]), self.scene_pos(pvt[0], pvt[1] + radius)
        self.drag(start, target)
        expected = handles2d.rotate_from_drag(0, pvt, viewer._transform_data_point(viewer.mapToScene(start)),
                                              viewer._transform_data_point(viewer.mapToScene(target)))
        self.assertAlmostEqual(self.window.dispatcher.document["nodes"]["t"]["params"]["rotate"], expected, delta=.1)

    def test_ctrl_drag_pivot_preserves_image(self):
        generation = self.window.frame_generation
        for param, value in (("rotate", 30.0), ("scale", 1.5), ("center_x", 300.0), ("center_y", 250.0)):
            self.window.command({"op": "set", "id": "t", "param": param, "value": value})
        self.window.dispatcher.undo_stack.clear()
        self.assertTrue(wait_until(lambda: self.window.frame_generation != generation))
        before, generation = np.array(self.window.frame, copy=True), self.window.frame_generation
        start = self.window.viewer.mapFromScene(self.window.viewer._transform_scene_point(300, 250))
        target = self.window.viewer.mapFromScene(self.window.viewer._transform_scene_point(340, 280))
        self.drag(start, target, Qt.KeyboardModifier.ControlModifier)
        self.assertTrue(wait_until(lambda: self.window.frame_generation != generation))
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        self.assertNotEqual((p["center_x"], p["center_y"]), (300, 250))
        self.assertNotEqual((p["translate_x"], p["translate_y"]), (0, 0))
        self.assertTrue(np.array_equal(before, self.window.frame))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_click_does_not_edit(self):
        viewer = self.window.viewer
        before = dict(self.window.dispatcher.document["nodes"]["t"]["params"])
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.scene_pos(300, 250))
        p = self.window.dispatcher.document["nodes"]["t"]["params"]
        self.assertEqual((p["translate_x"], p["translate_y"]), (before["translate_x"], before["translate_y"]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 0)

    def test_animated_translate_is_keyed(self):
        self.window.command({"op": "set_key", "id": "t", "param": "translate_x", "frame": 1, "value": 0.0})
        self.window.command({"op": "set_key", "id": "t", "param": "translate_x", "frame": 2, "value": 10.0})
        self.window.dispatcher.undo_stack.clear()
        self.drag(self.scene_pos(300, 250), self.scene_pos(340, 250))
        curve = self.window.dispatcher.document["animation"]["curves"]["t"]["translate_x"]
        self.assertTrue(any(key["frame"] == 1 and key["value"] != 0 for key in curve["keys"]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_tracker_and_roto_clicks_keep_priority(self):
        # The properties panel always mirrors graph selection (Graph.selection_changed calls
        # Window.inspect), so the Tracker's "Add track point" button only exists once the
        # Tracker itself is selected -- a Transform can never stay selected while a different
        # node's tracker pick is in progress. That makes the two states structurally exclusive;
        # this is a regression guard that basic tracker/roto interaction still works once the
        # transform-handle hit test is appended after them in mousePressEvent, not a real
        # conflict resolution test.
        viewer = self.window.viewer
        self.window.command({"op": "create", "id": "k", "type": "Tracker", "pos": [300, 0]})
        self.window.command({"op": "connect", "id": "k", "input": "image", "source": "t"})
        self.window.graph.items_by_id["t"].setSelected(False)
        self.window.graph.items_by_id["k"].setSelected(True)
        self.assertTrue(self.window.begin_tracker_pick("k"))
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.scene_pos(300, 250))
        self.assertIsNotNone(self.window._tracker_seed)
        self.assertIsNone(viewer.transform_drag)

        self.window.command({"op": "create", "id": "r", "type": "Roto", "pos": [0, 150]})
        self.window.command({"op": "set", "id": "r", "param": "width", "value": 100})
        self.window.command({"op": "set", "id": "r", "param": "height", "value": 100})
        shape = {"name": "shape1", "mode": "union", "opacity": 1.0, "feather": 0.0,
                 "points": [{"x": 10., "y": 10., "in_x": 0., "in_y": 0., "out_x": 0., "out_y": 0.},
                            {"x": 90., "y": 10., "in_x": 0., "in_y": 0., "out_x": 0., "out_y": 0.},
                            {"x": 50., "y": 90., "in_x": 0., "in_y": 0., "out_x": 0., "out_y": 0.}]}
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [shape]})
        self.window.command({"op": "view", "id": "r"})
        self.window.graph.items_by_id["k"].setSelected(False)
        self.window.graph.items_by_id["r"].setSelected(True)
        self.assertTrue(wait_until(lambda: viewer._roto_context() is not None))
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.scene_pos(10, 10))
        self.assertIsNotNone(viewer.roto_drag)
        QTest.keyClick(viewer, Qt.Key.Key_Escape)


if __name__ == "__main__":
    unittest.main()
