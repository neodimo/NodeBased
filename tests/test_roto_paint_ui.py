"""Offscreen viewer gesture tests for RotoPaint stroke capture and DustBust."""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document


APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition(): return True
        QTest.qWait(10)
    return False


def document():
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Constant", "params": {"width": 100, "height": 100}},
        {"op": "create", "id": "paint", "type": "RotoPaint"},
        {"op": "connect", "id": "paint", "input": "image", "source": "plate"},
        {"op": "view", "id": "paint"}]})
    return dispatcher.document


class RotoPaintViewerTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["paint"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def position(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def gesture(self):
        viewer = self.window.viewer
        start, end = self.position(20, 20), self.position(30, 20)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer.viewport(), end, 20)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
        APP.processEvents()

    def test_mouse_gesture_records_pressure_points_as_one_undoable_stroke(self):
        before = len(self.window.dispatcher.undo_stack)
        self.gesture()
        items = self.window.dispatcher.document["node_data"]["paint"]["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["tool"], "paint")
        self.assertGreaterEqual(len(items[0]["points"]), 2)
        self.assertEqual(items[0]["points"][0]["pressure"], 1.0)
        self.assertEqual(len(self.window.dispatcher.undo_stack), before + 1)
        self.window.command({"op": "undo"})
        self.assertNotIn("paint", self.window.dispatcher.document["node_data"])

    def test_dustbust_click_records_a_single_frame_previous_frame_clone(self):
        viewer = self.window.viewer
        viewer.dustbust_preset = True
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.position(40, 40))
        item = self.window.dispatcher.document["node_data"]["paint"]["items"][0]
        self.assertEqual(item["tool"], "clone")
        self.assertEqual(item["lifetime"], {"mode": "single", "first": 1})
        self.assertEqual(item["source_frame"], "relative")

    def test_paint_node_keeps_roto_shapes_in_the_same_ordered_payload(self):
        viewer = self.window.viewer
        self.assertTrue(viewer.begin_roto_draw("paint"))
        for x, y in ((10, 10), (90, 10), (50, 90)):
            QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.position(x, y))
        viewer.setFocus(); QTest.keyClick(viewer, Qt.Key.Key_Return)
        items = self.window.dispatcher.document["node_data"]["paint"]["items"]
        self.assertEqual(items[0]["kind"], "shape")
        self.assertEqual(items[0]["name"], "shape1")
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)


if __name__ == "__main__":
    unittest.main()
