"""Offscreen drag tests for MinColor and Sampler viewer controls."""
import os
import unittest

from tests.waiting import wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def document(kind):
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "create", "id": "src", "type": "Constant", "params":
                        {"width": 320, "height": 240, "red": .2, "green": .3, "blue": .4, "alpha": 1}})
    dispatcher.execute({"op": "create", "id": "analysis", "type": kind})
    dispatcher.execute({"op": "connect", "id": "analysis", "input": "image", "source": "src"})
    dispatcher.execute({"op": "view", "id": "analysis"})
    return dispatcher.document


class AnalysisHandleTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(document("Sampler"))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.viewer.format_rect is not None))
        self.window.graph.items_by_id["analysis"].setSelected(True)
        self.window.properties_dock.show()
        APP.processEvents()
        self.window.dispatcher.undo_stack.clear()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def scene(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def drag(self, start, end):
        viewer = self.window.viewer
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer.viewport(), end, 20)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)

    def test_sampler_endpoint_drag_tracks_canvas_distance_at_two_zooms_and_undoes(self):
        viewer = self.window.viewer
        for zoom in (1.0, 2.0):
            viewer.resetTransform(); viewer.scale(zoom, zoom)
            before = dict(self.window.dispatcher.document["nodes"]["analysis"]["params"])
            start = self.scene(before["sample_x0"], before["sample_y0"])
            end = self.scene(before["sample_x0"] + 24, before["sample_y0"] + 16)
            viewer = self.window.viewer
            QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
            QTest.mouseMove(viewer.viewport(), end, 20)
            APP.processEvents()
            field = next(control for control in self.window.properties_dock.findChildren(QDoubleSpinBox)
                         if control.property("nodebased_node_id") == "analysis"
                         and control.property("nodebased_param") == "sample_x0")
            self.assertAlmostEqual(field.value(), before["sample_x0"] + 24, delta=1.5)
            QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
            p = self.window.dispatcher.document["nodes"]["analysis"]["params"]
            self.assertAlmostEqual(p["sample_x0"], before["sample_x0"] + 24, delta=1.5)
            self.assertAlmostEqual(p["sample_y0"], before["sample_y0"] + 16, delta=1.5)
            self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
            self.window.command({"op": "undo"})
            p = self.window.dispatcher.document["nodes"]["analysis"]["params"]
            self.assertEqual((p["sample_x0"], p["sample_y0"]),
                             (before["sample_x0"], before["sample_y0"]))
            self.window.dispatcher.undo_stack.clear()

    def test_mincolor_region_grip_resizes_and_closed_panel_hides_handle(self):
        self.window.command({"op": "create", "id": "min", "type": "MinColor", "pos": [300, 0]})
        self.window.command({"op": "connect", "id": "min", "input": "image", "source": "src"})
        self.window.command({"op": "view", "id": "min"})
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id["min"].setSelected(True)
        self.window.pin_panel("min")
        self.window.properties_dock.show()
        self.window.dispatcher.undo_stack.clear()
        self.assertTrue(wait_until(lambda: self.window.viewer._analysis_context() is not None))
        params = self.window.dispatcher.document["nodes"]["min"]["params"]
        start, end = self.scene(64, 64), self.scene(100, 96)
        self.drag(start, end)
        p = self.window.dispatcher.document["nodes"]["min"]["params"]
        self.assertGreaterEqual(p["box_width"], 30)
        self.assertGreaterEqual(p["box_height"], 25)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        p = self.window.dispatcher.document["nodes"]["min"]["params"]
        self.assertEqual((p["box_width"], p["box_height"]), (0.0, 0.0))
        self.window.properties_dock.hide(); APP.processEvents()
        self.assertIsNone(self.window.viewer._analysis_context())


if __name__ == "__main__":
    unittest.main()
