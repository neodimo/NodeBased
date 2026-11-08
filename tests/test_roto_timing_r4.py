"""R4 timing and key-control behavior checks."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QCheckBox

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document, upgrade_document
from nodebased import shapes
from nodebased.imaging import Evaluator
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

