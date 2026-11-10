"""Ripple editing (Nuke's "Deleting or Rippling Keyframes"): one point edit applied as the same offset to the other keys."""
import copy
import os
import unittest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
from nodebased import rotoripple, shapes as shape_model
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document
from nodebased.roto_settings import GLOBAL_SETTINGS, TOOL_CONTROLS
from tests.test_roto_ui import roto_document, triangle
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)

KEYS = {1: 10.0, 3: 20.0, 5: 30.0, 7: 40.0}


def curve(keys):
    return {"value": keys[min(keys)], "curve": {"interpolation": "linear",
            "keys": [{"frame": f, "value": v} for f, v in sorted(keys.items())]}}


def key_values(scalar):
    return {k["frame"]: k["value"] for k in scalar["curve"]["keys"]}


def keyed_triangle():
    shape = triangle()
    shape["points"][0]["x"] = curve(KEYS)
    return shape


class RippleMathTests(unittest.TestCase):
    def test_windows_follow_the_four_choices(self):
        self.assertEqual(rotoripple.frame_window("all", 5, 1, 10), (1, 10))
        self.assertEqual(rotoripple.frame_window("from start", 5, 1, 10), (1, 5))
        self.assertEqual(rotoripple.frame_window("to end", 5, 1, 10), (5, 10))
        self.assertEqual(rotoripple.frame_window("range", 5, 1, 10, 7, 3), (3, 7))

    def test_window_is_none_with_ripple_off(self):
        time = {"first": 1, "last": 10, "current": 3}
        self.assertIsNone(rotoripple.settings_window({}, time))
        self.assertIsNone(rotoripple.settings_window({"ripple_edit": False, "ripple_mode": "all"}, time))
        self.assertEqual(rotoripple.settings_window({"ripple_edit": True, "ripple_mode": "to end"}, time), (3, 10))
        self.assertEqual(rotoripple.settings_window({"ripple_edit": True, "ripple_mode": "bogus"}, time), (1, 10))

    def test_scalar_ripple_moves_keys_in_window_by_the_same_offset(self):
        old = curve(KEYS)
        new = copy.deepcopy(old)
        new["curve"]["keys"][1]["value"] = 25.0                      # the edit at frame 3: +5
        rippled = rotoripple.ripple_scalar(old, new, 3, 3, 5)
        self.assertEqual(key_values(rippled), {1: 10.0, 3: 25.0, 5: 35.0, 7: 40.0})

    def test_static_scalar_takes_the_new_value_without_keys(self):
        rippled = rotoripple.ripple_scalar(12.0, {"value": 12.0, "curve": {"interpolation": "smooth", "keys": [
            {"frame": 1, "value": 12.0}, {"frame": 3, "value": 15.0}]}}, 3, 1, 10)
        self.assertEqual(rippled, 15.0)

    def test_unchanged_scalar_is_untouched(self):
        old = curve(KEYS)
        self.assertIs(rotoripple.ripple_scalar(old, old, 3, 1, 10), old)

    def test_edit_off_a_key_lands_exactly_on_the_new_value(self):
        old = curve(KEYS)
        # Frame 2 reads 15 on the linear curve; the artist sets 18 there: +3, a key at 2 is added only if the shift misses 18.
        new = copy.deepcopy(old); new["curve"]["keys"].append({"frame": 2, "value": 18.0})
        new["curve"]["keys"].sort(key=lambda k: k["frame"])
        rippled = rotoripple.ripple_scalar(old, new, 2, 1, 10)
        self.assertAlmostEqual(shape_model.resolve_scalar(rippled, 2), 18.0)
        self.assertEqual(key_values(rippled)[7], 43.0)

    def test_shapes_with_changed_point_count_are_left_alone(self):
        old = [keyed_triangle()]
        new = copy.deepcopy(old); new[0]["points"].pop()
        self.assertEqual(rotoripple.ripple_shapes(old, new, 3, 1, 10), new)

    def test_controls_and_defaults_are_registered(self):
        for tool in ("Select All", "Select Splines", "Select Points", "Select Feather Points"):
            labels = [c["label"] for c in TOOL_CONTROLS[tool]]
            for label in ("Ripple edit", "Ripple mode", "Ripple from", "Ripple to"):
                self.assertIn(label, labels)
        self.assertEqual(GLOBAL_SETTINGS["ripple_edit"], False)
        self.assertEqual(GLOBAL_SETTINGS["ripple_mode"], "all")


class RippleViewerTests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        document = roto_document([keyed_triangle()])
        document["time"].update(first=1, last=10, current=3)
        self.window = Window(document)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        self.viewer = self.window.viewer

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def shapes(self):
        return self.window.dispatcher.document["node_data"]["r"]["shapes"]

    def x_keys(self):
        return key_values(self.shapes()[0]["points"][0]["x"])

    def set_global(self, **settings):
        for name, value in settings.items():
            self.viewer.roto_toolbar.set_setting("r", "Select Points", name, value)

    def nudge(self):
        self.viewer.setFocus()
        self.viewer.roto_selected_points = {(0, 0)}
        self.viewer.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.KeypadModifier))

    def test_ripple_off_changes_one_key(self):
        self.assertEqual(self.x_keys(), KEYS)
        self.nudge()
        self.assertEqual(self.x_keys(), {1: 10.0, 3: 21.0, 5: 30.0, 7: 40.0})

    def test_ripple_all_moves_every_key_by_the_same_offset(self):
        self.set_global(ripple_edit=True, ripple_mode="all")
        self.nudge()
        self.assertEqual(self.x_keys(), {1: 11.0, 3: 21.0, 5: 31.0, 7: 41.0})

    def test_ripple_to_end_leaves_earlier_keys(self):
        self.set_global(ripple_edit=True, ripple_mode="to end")
        self.nudge()
        self.assertEqual(self.x_keys(), {1: 10.0, 3: 21.0, 5: 31.0, 7: 41.0})

    def test_ripple_from_start_leaves_later_keys(self):
        self.set_global(ripple_edit=True, ripple_mode="from start")
        self.nudge()
        self.assertEqual(self.x_keys(), {1: 11.0, 3: 21.0, 5: 30.0, 7: 40.0})

    def test_ripple_range_moves_only_keys_inside_it(self):
        self.set_global(ripple_edit=True, ripple_mode="range", ripple_from=3.0, ripple_to=5.0)
        self.nudge()
        self.assertEqual(self.x_keys(), {1: 10.0, 3: 21.0, 5: 31.0, 7: 40.0})

    def test_one_undo_restores_every_key(self):
        self.set_global(ripple_edit=True, ripple_mode="all")
        depth = len(self.window.dispatcher.undo_stack)
        self.nudge()
        self.assertEqual(len(self.window.dispatcher.undo_stack), depth + 1)
        self.window.command({"op": "undo"})
        self.assertEqual(self.x_keys(), KEYS)

    def test_other_points_and_fields_are_not_rippled(self):
        self.set_global(ripple_edit=True, ripple_mode="all")
        self.nudge()
        point = self.shapes()[0]["points"][1]
        self.assertEqual(point["x"], 90.0)

    def test_the_settings_are_stored_globally_and_defaults_are_not(self):
        self.set_global(ripple_edit=True, ripple_mode="to end")
        stored = self.window.dispatcher.document["node_data"]["r"]["tool_settings"]["__global__"]
        self.assertEqual((stored["ripple_edit"], stored["ripple_mode"]), (True, "to end"))
        self.set_global(ripple_edit=False, ripple_mode="all")
        stored = self.window.dispatcher.document["node_data"]["r"].get("tool_settings", {}).get("__global__", {})
        self.assertNotIn("ripple_edit", stored)
        self.assertNotIn("ripple_mode", stored)

    def test_strip_shows_the_ripple_controls_and_viewer_shows_the_red_border(self):
        self.viewer.roto_toolbar.choose("Select", "Select All")
        for name in ("ripple_edit", "ripple_mode", "ripple_from", "ripple_to"):
            self.assertIn(name, self.viewer.roto_toolbar.settings_widgets)
        self.assertIsNone(self.viewer.roto_ripple_window())
        self.viewer.roto_toolbar.settings_widgets["ripple_edit"].setChecked(True)
        self.assertEqual(self.viewer.roto_ripple_window(), (1, 10))
        combo = self.viewer.roto_toolbar.settings_widgets["ripple_mode"]
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], list(rotoripple.MODES))
        combo.setCurrentIndex(combo.findData("to end"))
        self.assertEqual(self.viewer.roto_ripple_window(), (3, 10))


if __name__ == "__main__":
    unittest.main()
