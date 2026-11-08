"""R4 timing and key-control behavior checks."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document, upgrade_document
from nodebased import shapes
from nodebased.imaging import Evaluator
from nodebased import rotopanel
from tests.test_roto_ui import roto_document

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)


class RotoTimingR4Tests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(roto_document())
        self.window.show()
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def test_autokey_on_preserves_old_point_key_and_adds_current_frame(self):
        viewer = self.window.viewer
        keyed = {"value": 10.0, "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 10.0}]}}
        result = viewer._roto_scalar_at_frame(keyed, 5, 24.0)
        self.assertEqual(shapes.resolve_scalar(result, 1, "x"), 10.0)
        self.assertEqual(shapes.resolve_scalar(result, 5, "x"), 24.0)

    def test_autokey_off_edits_unkeyed_frame_as_static_value(self):
        viewer = self.window.viewer
        viewer.roto_toolbar.set_setting("r", "Select Points", "autokey", False)
        keyed = {"value": 10.0, "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 10.0}]}}
        result = viewer._roto_scalar_at_frame(keyed, 5, 24.0)
        self.assertEqual(result, 24.0)
        self.assertEqual(shapes.resolve_scalar(result, 1, "x"), 24.0)

    def test_old_document_migrates_with_autokey_off(self):
        document = copy.deepcopy(roto_document())
        document["version"] = 33
        document["node_data"]["r"].pop("tool_settings", None)
        migrated = upgrade_document(document)
        self.assertFalse(migrated["node_data"]["r"]["tool_settings"]["__global__"]["autokey"])

    def test_old_document_migration_preserves_rendered_pixels(self):
        before = copy.deepcopy(self.window.dispatcher.document)
        before["node_data"]["r"]["shapes"] = [{
            "name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0,
            "points": [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0}
                       for x, y in ((10, 10), (90, 10), (90, 90), (10, 90))]}]
        expected = Evaluator().evaluate_raster(before, target="r").pixels.copy()
        before["version"] = 33
        before["node_data"]["r"].pop("tool_settings", None)
        migrated = upgrade_document(before)
        actual = Evaluator().evaluate_raster(migrated, target="r").pixels
        self.assertTrue((expected == actual).all())

    def test_new_roto_node_defaults_autokey_on_and_strip_has_toggle(self):
        node_data = self.window.dispatcher.document["node_data"]["r"]
        self.assertTrue(node_data["tool_settings"]["__global__"]["autokey"])
        self.window.viewer.roto_toolbar.refresh()
        toggle = self.window.viewer.roto_toolbar.settings_strip.findChild(QCheckBox,
                                                                           "roto-setting-autokey")
        self.assertIsNotNone(toggle)
        self.assertTrue(toggle.isChecked())

    def test_spline_point_key_controls_and_navigation(self):
        points = [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                   "out_x": 0.0, "out_y": 0.0} for x, y in ((10, 10), (80, 10), (80, 80))]
        shape = {"name": "keys", "mode": "union", "opacity": 1.0, "feather": 0.0, "points": points}
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [shape]}, render=False)
        self.window.viewer.roto_selected_points = {(0, 1)}
        panel = rotopanel.build_panel(self.window, "r")
        panel.findChild(QPushButton, "roto-point-set-keys").click()
        entries = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"]
        self.assertIsInstance(entries[1]["x"], dict)
        self.assertEqual(entries[0]["x"], 10.0)
        undo_count = len(self.window.dispatcher.undo_stack)
        panel.findChild(QPushButton, "roto-point-delete-keys").click()
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo_count + 1)
        entries = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"]
        self.assertEqual(entries[1]["x"], 80.0)

        panel = rotopanel.build_panel(self.window, "r")
        panel.findChild(QPushButton, "roto-spline-set-key").click()
        self.window.command({"op": "time", "last": 10})
        self.window.set_time(current=5)
        panel = rotopanel.build_panel(self.window, "r")
        panel.findChild(QPushButton, "roto-spline-set-key").click()
        panel = rotopanel.build_panel(self.window, "r")
        self.assertEqual(panel.findChild(QLabel, "roto-spline-key-count").text(), "Key 2 / 2")
        panel.findChild(QPushButton, "roto-spline-previous-key").click()
        self.assertEqual(self.window.dispatcher.document["time"]["current"], 1)
        panel = rotopanel.build_panel(self.window, "r")
        panel.findChild(QPushButton, "roto-spline-delete-key").click()
        curve = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]["x"]["curve"]
        self.assertEqual([key["frame"] for key in curve["keys"]], [5])

