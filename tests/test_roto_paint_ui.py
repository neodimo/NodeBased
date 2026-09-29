"""Offscreen viewer gesture tests for RotoPaint stroke capture and DustBust."""
import os
import time
import unittest
from tests.waiting import wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QListWidget, QPushButton

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document


APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)




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


class RotoPaintLayerEditorTests(unittest.TestCase):
    """A dedicated per-layer editor: fixing an already-drawn shape or stroke without redrawing it."""

    def setUp(self):
        self.window = Window(document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["paint"].setSelected(True)
        APP.processEvents()
        self.shape_item = {"kind": "shape", "name": "shape1", "mode": "union", "opacity": 1.0,
                           "feather": 0.0, "points": [
                               {"x": 10.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                               {"x": 90.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                               {"x": 50.0, "y": 90.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}],
                           "blend": "over", "visible": True}
        self.stroke_item = {"kind": "stroke", "name": "stroke1", "points": [{"x": 20.0, "y": 20.0, "pressure": 1.0}],
                            "brush": {"size": 12.0, "hardness": 0.8, "opacity": 1.0, "spacing": 0.2, "strength": 0.2},
                            "tool": "dodge", "lifetime": {"mode": "all"}, "color": [1.0, 0.0, 0.0, 1.0],
                            "source_offset": [0.0, 0.0], "source_frame": "relative", "opacity": 1.0,
                            "blend": "over", "visible": True, "follow_track": None}
        self.window.command({"op": "set_paint_items", "id": "paint", "items": [self.shape_item, self.stroke_item]})

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def panel(self):
        panel = self.window.build_node_panel("paint")
        self.window.set_properties_widget(panel)
        return panel

    def items(self):
        return self.window.dispatcher.document["node_data"]["paint"]["items"]

    def test_layer_list_shows_every_item_in_order(self):
        layer_list = self.panel().findChild(QListWidget, "rotopaint-layer-list")
        self.assertIsNotNone(layer_list)
        self.assertEqual(layer_list.count(), 2)
        self.assertIn("shape1", layer_list.item(0).text())
        self.assertIn("stroke1", layer_list.item(1).text())

    def test_selecting_a_stroke_shows_its_brush_and_dodge_burn_strength(self):
        self.window.viewer.paint_selected_item_index = 1
        panel = self.panel()
        strength_field = panel.findChild(QDoubleSpinBox, "rotopaint-layer-brush-strength")
        self.assertIsNotNone(strength_field)
        self.assertAlmostEqual(strength_field.value(), 0.2)
        self.assertIsNone(panel.findChild(QDoubleSpinBox, "rotopaint-layer-feather"))

    def test_selecting_a_shape_shows_its_mode_and_feather_instead_of_brush_fields(self):
        self.window.viewer.paint_selected_item_index = 0
        panel = self.panel()
        self.assertIsNotNone(panel.findChild(QDoubleSpinBox, "rotopaint-layer-feather"))
        self.assertIsNone(panel.findChild(QDoubleSpinBox, "rotopaint-layer-brush-strength"))

    def test_editing_the_layer_opacity_updates_the_document_in_one_undo_step(self):
        self.window.viewer.paint_selected_item_index = 1
        panel = self.panel()
        before = len(self.window.dispatcher.undo_stack)
        field = panel.findChild(QDoubleSpinBox, "rotopaint-layer-opacity")
        field.setValue(0.5)
        self.assertAlmostEqual(self.items()[1]["opacity"], 0.5)
        self.assertEqual(self.items()[0], self.shape_item)
        self.assertEqual(len(self.window.dispatcher.undo_stack), before + 1)

    def test_editing_the_dodge_burn_strength_updates_only_that_stroke(self):
        self.window.viewer.paint_selected_item_index = 1
        panel = self.panel()
        field = panel.findChild(QDoubleSpinBox, "rotopaint-layer-brush-strength")
        field.setValue(0.6)
        self.assertAlmostEqual(self.items()[1]["brush"]["strength"], 0.6)
        self.assertEqual(self.items()[0], self.shape_item)

    def test_move_up_swaps_the_selected_item_earlier_in_one_undo_step(self):
        self.window.viewer.paint_selected_item_index = 1
        panel = self.panel()
        before = len(self.window.dispatcher.undo_stack)
        up_button = panel.findChild(QPushButton, "rotopaint-layer-move-up")
        self.assertTrue(up_button.isEnabled())
        up_button.click()
        items = self.items()
        self.assertEqual(items[0]["name"], "stroke1")
        self.assertEqual(items[1]["name"], "shape1")
        self.assertEqual(len(self.window.dispatcher.undo_stack), before + 1)
        self.assertEqual(self.window.viewer.paint_selected_item_index, 0)

    def test_delete_layer_removes_the_selected_item_in_one_undo_step(self):
        self.window.viewer.paint_selected_item_index = 1
        panel = self.panel()
        before = len(self.window.dispatcher.undo_stack)
        delete_button = panel.findChild(QPushButton, "rotopaint-layer-delete")
        delete_button.click()
        items = self.items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["name"], "shape1")
        self.assertEqual(len(self.window.dispatcher.undo_stack), before + 1)
        self.assertIsNone(self.window.viewer.paint_selected_item_index)

    def test_move_and_delete_buttons_are_disabled_with_nothing_selected(self):
        self.window.viewer.paint_selected_item_index = None
        panel = self.panel()
        self.assertFalse(panel.findChild(QPushButton, "rotopaint-layer-move-up").isEnabled())
        self.assertFalse(panel.findChild(QPushButton, "rotopaint-layer-move-down").isEnabled())
        self.assertFalse(panel.findChild(QPushButton, "rotopaint-layer-delete").isEnabled())


if __name__ == "__main__":
    unittest.main()
