"""Offscreen artist-facing Roto overlay tests.

These deliberately drive the Viewer widget, while checking the resulting document through the
same Dispatcher-owned state that an agent sees.  The overlay itself remains outside the graphics
scene so format/data-window bounds cannot be changed by its handles.
"""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import empty_document, Dispatcher


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


def roto_document(shapes=None, frame=1):
    document = empty_document()
    document["nodes"] = {}
    document["view"] = None
    document["time"].update(first=1, last=2, current=frame)
    dispatcher = Dispatcher(document)
    dispatcher.execute({"op": "create", "id": "r", "type": "Roto", "pos": [0, 0]})
    dispatcher.execute({"op": "set", "id": "r", "param": "width", "value": 100})
    dispatcher.execute({"op": "set", "id": "r", "param": "height", "value": 100})
    dispatcher.execute({"op": "view", "id": "r"})
    if shapes:
        dispatcher.execute({"op": "set_shapes", "id": "r", "shapes": shapes})
    return dispatcher.document


def triangle():
    return {"name": "triangle", "mode": "union", "opacity": 1.0, "feather": 0.0,
            "points": [{"x": 10.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0},
                       {"x": 90.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0},
                       {"x": 50.0, "y": 90.0, "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0}]}


class RotoOverlayTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(roto_document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def scene_pos(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def test_drawing_commits_one_validated_payload_and_keeps_scene_bounds(self):
        viewer = self.window.viewer
        original_bounds = viewer.scene().itemsBoundingRect()
        self.assertTrue(self.window.begin_roto_draw("r"))
        for x, y in ((10, 10), (90, 10), (50, 90)):
            QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton,
                              pos=self.scene_pos(x, y))
        viewer.setFocus()
        QTest.keyClick(viewer, Qt.Key.Key_Return)
        self.assertTrue(wait_until(lambda: "r" in self.window.dispatcher.document["node_data"]))
        shapes = self.window.dispatcher.document["node_data"]["r"]["shapes"]
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["name"], "shape1")
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.assertEqual(viewer.scene().itemsBoundingRect(), original_bounds)

    def test_dragging_a_point_commits_once_and_updates_animated_coordinates_at_current_frame(self):
        shape = triangle()
        for point in shape["points"][:1]:
            for field, value in (("x", 10.0), ("y", 10.0)):
                point[field] = {"value": value,
                                "curve": {"interpolation": "linear",
                                          "keys": [{"frame": 1, "value": value},
                                                   {"frame": 2, "value": value + 5.0}]}}
        self.window.close()
        APP.processEvents()
        self.window = Window(roto_document([shape], frame=2))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        viewer = self.window.viewer
        start = self.scene_pos(15, 15)
        target = self.scene_pos(30, 40)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer.viewport(), target, 20)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=target)
        # The drop lands on a whole viewport pixel, so the exact value depends on the viewer's
        # zoom, which follows the window's size. Compare against where that pixel really is.
        dropped = viewer.mapToScene(target)
        self.assertAlmostEqual(dropped.x(), 30.0, delta=1.0)
        self.assertAlmostEqual(dropped.y(), 40.0, delta=1.0)
        self.assertTrue(wait_until(lambda: abs(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]["x"]["curve"]["keys"][-1]["value"] - dropped.x()) < 0.01))
        point = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]
        self.assertEqual(point["x"]["curve"]["keys"][-1]["frame"], 2)
        self.assertAlmostEqual(point["y"]["curve"]["keys"][-1]["value"], dropped.y(), delta=0.01)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_overlay_is_not_editable_when_viewing_a_downstream_node(self):
        shape = triangle()
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [shape]})
        self.window.command({"op": "create", "id": "c", "type": "Constant", "pos": [150, 0]})
        self.window.command({"op": "view", "id": "c"})
        self.window.graph.items_by_id["r"].setSelected(True)
        self.assertIsNone(self.window.viewer._roto_context())
        self.assertFalse(self.window.viewer.begin_roto_draw("r"))

    def test_proxy_display_keeps_full_format_coordinate_mapping(self):
        viewer = self.window.viewer
        self.window.proxy.setCurrentIndex(1)  # 1/2 preview is upscaled to the full scene rectangle.
        self.assertTrue(wait_until(lambda: viewer.format_rect is not None))
        context = viewer._roto_context()
        self.assertEqual(viewer._roto_data_point(QPointF(30, 40), context[1], 2), [30.0, 40.0])


if __name__ == "__main__":
    unittest.main()
