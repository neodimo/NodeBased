"""R4 timing and key-control behavior checks."""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton, QSpinBox, QComboBox

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document, upgrade_document
from nodebased import shapes
from nodebased.imaging import Evaluator
from nodebased import paint, rotopanel
from tests.test_roto_ui import roto_document
from tests.test_roto_paint import stroke

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
        self.assertEqual(result["value"], 24.0)
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
        document["node_data"]["r"] = {"shapes": []}
        migrated = upgrade_document(document)
        self.assertFalse(migrated["node_data"]["r"]["tool_settings"]["__global__"]["autokey"])

    def test_old_document_migration_preserves_rendered_pixels(self):
        before = copy.deepcopy(self.window.dispatcher.document)
        before["node_data"]["r"] = {"shapes": [{
            "name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0,
            "points": [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0}
                       for x, y in ((10, 10), (90, 10), (90, 90), (10, 90))]}]}
        expected = Evaluator().evaluate_raster(before, target="r").pixels.copy()
        before["version"] = 33
        migrated = upgrade_document(before)
        actual = Evaluator().evaluate_raster(migrated, target="r").pixels
        self.assertTrue((expected == actual).all())

    def test_new_roto_node_defaults_autokey_on_and_strip_has_toggle(self):
        node_data = self.window.dispatcher.document["node_data"].get("r", {})
        self.assertNotIn("tool_settings", node_data)
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

        panel = rotopanel.build_panel(self.window, "r")
        scope = panel.findChild(QComboBox, "roto-key-scope")
        scope.setCurrentIndex(scope.findData("transform"))
        before = len(self.window.dispatcher.undo_stack)
        panel.findChild(QPushButton, "roto-item-set-key").click()
        transform = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["transform"]
        self.assertEqual(transform["rotate"]["curve"]["keys"], [{"frame": 1, "value": 0.0}])
        self.assertEqual(len(self.window.dispatcher.undo_stack), before + 1)

    def test_lifetime_modes_and_nuke_endpoint_visibility(self):
        expected = {
            "all": [True] * 7,
            "single": [False, False, True, False, False, False, False],
            "start_to": [True, True, True, True, True, False, False],
            "to_end": [False, False, True, True, True, True, True],
            "range": [False, False, True, True, True, False, False],
        }
        for kind, outcomes in expected.items():
            life = {"type": kind, "start": 3, "end": 5}
            self.assertEqual([shapes.in_lifetime(life, frame) for frame in range(1, 8)], outcomes)
        shape = {"name": "life", "mode": "union", "opacity": 1.0, "feather": 0.0,
                 "lifetime": {"type": "all"}, "points": [
                     {"x": 1.0, "y": 1.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                     {"x": 2.0, "y": 1.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                     {"x": 2.0, "y": 2.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}]}
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [shape]}, render=False)
        panel = rotopanel.build_panel(self.window, "r")
        self.assertTrue(panel.findChild(QSpinBox, "roto-shape-lifetime-start").isHidden())
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [{**shape, "lifetime": {"type": "range", "start": 3, "end": 5}}]}, render=False)
        panel = rotopanel.build_panel(self.window, "r")
        self.assertFalse(panel.findChild(QSpinBox, "roto-shape-lifetime-start").isHidden())
        self.assertFalse(panel.findChild(QSpinBox, "roto-shape-lifetime-end").isHidden())
        self.assertEqual(self.window.viewer.paint_lifetime, "single")
        stroke_lifetimes = {
            "all": {"mode": "all"},
            "single": {"mode": "single", "first": 3},
            "start_to": {"mode": "start_to", "last": 5},
            "to_end": {"mode": "to_end", "first": 3},
            "range": {"mode": "range", "first": 3, "last": 5},
        }
        for life in stroke_lifetimes.values():
            outcomes = [paint.active({"lifetime": life}, frame) for frame in range(1, 8)]
            if life["mode"] == "all": self.assertEqual(outcomes, [True] * 7)
            elif life["mode"] == "single": self.assertEqual(outcomes, [False, False, True, False, False, False, False])
            elif life["mode"] == "start_to": self.assertEqual(outcomes, [True, True, True, True, True, False, False])
            elif life["mode"] == "to_end": self.assertEqual(outcomes, [False, False, True, True, True, True, True])
            else: self.assertEqual(outcomes, [False, False, True, True, True, False, False])

    def test_rotopaint_stroke_lifetime_modes_validate(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "paint", "type": "RotoPaint"})
        lives = ({"mode": "all"}, {"mode": "start_to", "last": 5},
                 {"mode": "single", "first": 3}, {"mode": "to_end", "first": 3},
                 {"mode": "range", "first": 3, "last": 5})
        for life in lives:
            dispatcher.execute({"op": "set_paint_items", "id": "paint",
                                "items": [stroke(lifetime=life)]})

    def test_rotopaint_shape_lifetime_panel_has_nuke_modes_and_endpoints(self):
        self.window.command({"op": "create", "id": "p", "type": "RotoPaint"})
        self.window.command({"op": "connect", "id": "p", "input": "image", "source": "r"})
        self.window.command({"op": "set_paint_items", "id": "p", "items": [{
            "kind": "shape", "name": "paint-shape", "mode": "union", "opacity": 1.0,
            "feather": 0.0, "points": [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                                         "out_x": 0.0, "out_y": 0.0}
                                        for x, y in ((10, 10), (90, 10), (50, 90))],
            "blend": "over", "visible": True, "lifetime": {"type": "all"}}]})
        self.window.command({"op": "view", "id": "p"})
        self.window.viewer.paint_selected_item_index = 0
        panel = self.window.build_node_panel("p")
        self.window.set_properties_widget(panel)
        combo = panel.findChild(QComboBox, "rotopaint-layer-lifetime")
        self.assertIsNotNone(combo)
        self.assertEqual(combo.findData("all"), combo.currentIndex())
        self.assertEqual([combo.itemData(i) for i in range(combo.count())],
                         ["all", "start_to", "single", "to_end", "range"])
        combo.setCurrentIndex(combo.findData("range"))
        life = self.window.dispatcher.document["node_data"]["p"]["items"][0]["lifetime"]
        self.assertEqual(life, {"type": "range", "start": 1, "end": 1})
        panel = self.window.build_node_panel("p")
        self.window.set_properties_widget(panel)
        before = len(self.window.dispatcher.undo_stack)
        panel.findChild(QPushButton, "rotopaint-layer-key-transform-scale_x").click()
        item = self.window.dispatcher.document["node_data"]["p"]["items"][0]
        self.assertEqual(item["transform"]["scale_x"]["curve"]["keys"], [{"frame": 1, "value": 1.0}])
        self.assertEqual(len(self.window.dispatcher.undo_stack), before + 1)
    def test_rotopaint_stroke_lifetime_panel_edits_from_and_to(self):
        self.window.command({"op": "create", "id": "p", "type": "RotoPaint"})
        self.window.command({"op": "connect", "id": "p", "input": "image", "source": "r"})
        self.window.command({"op": "set_paint_items", "id": "p", "items": [stroke(lifetime={"mode": "all"})]})
        self.window.command({"op": "view", "id": "p"})
        self.window.viewer.paint_selected_item_index = 0
        panel = self.window.build_node_panel("p")
        self.window.set_properties_widget(panel)
        combo = panel.findChild(QComboBox, "rotopaint-layer-lifetime")
        self.assertEqual([combo.itemData(i) for i in range(combo.count())],
                         ["all", "start_to", "single", "to_end", "range"])
        combo.setCurrentIndex(combo.findData("start_to"))
        panel = self.window.build_node_panel("p")
        self.window.set_properties_widget(panel)
        endpoint = panel.findChild(QSpinBox, "rotopaint-layer-lifetime-last")
        self.assertIsNotNone(endpoint)
        endpoint.setValue(5)
        self.assertEqual(self.window.dispatcher.document["node_data"]["p"]["items"][0]["lifetime"],
                         {"mode": "start_to", "last": 5})

