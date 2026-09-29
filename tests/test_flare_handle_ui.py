"""Offscreen Viewer interaction checks for Flare positioning and Tracker linking."""
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def wait_until(condition, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        time.sleep(.01)
    return False


def flare_document():
    dispatcher = Dispatcher(empty_document())
    dispatcher.execute({"op": "create", "id": "source", "type": "Constant"})
    dispatcher.execute({"op": "create", "id": "flare", "type": "Flare",
                        "params": {"position_x": 300.0, "position_y": 250.0}})
    dispatcher.execute({"op": "connect", "id": "flare", "input": "image", "source": "source"})
    dispatcher.execute({"op": "view", "id": "flare"})
    return dispatcher.document


class FlareHandleTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(flare_document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None and
                                   self.window.viewer.format_rect is not None))
        self.window.graph.items_by_id["flare"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def scene_pos(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def test_drag_moves_flare_and_undoes_as_one_edit(self):
        viewer = self.window.viewer
        before = self.window.dispatcher.document["nodes"]["flare"]["params"]
        start = self.scene_pos(before["position_x"], before["position_y"])
        target = self.scene_pos(before["position_x"] + 35, before["position_y"] - 20)
        start_scene = viewer.mapToScene(start)
        target_scene = viewer.mapToScene(target)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer.viewport(), target, 20)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=target)
        params = self.window.dispatcher.document["nodes"]["flare"]["params"]
        self.assertAlmostEqual(params["position_x"], before["position_x"] + target_scene.x() - start_scene.x(), delta=.01)
        self.assertAlmostEqual(params["position_y"], before["position_y"] + target_scene.y() - start_scene.y(), delta=.01)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_position_drag_keys_animated_knobs(self):
        self.window.command({"op": "set_key", "id": "flare", "param": "position_x", "frame": 1, "value": 300.0})
        self.window.command({"op": "set_key", "id": "flare", "param": "position_x", "frame": 2, "value": 310.0})
        self.window.dispatcher.undo_stack.clear()
        start, target = self.scene_pos(300, 250), self.scene_pos(325, 250)
        QTest.mousePress(self.window.viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.window.viewer.viewport(), target, 20)
        QTest.mouseRelease(self.window.viewer.viewport(), Qt.MouseButton.LeftButton, pos=target)
        curve = self.window.dispatcher.document["animation"]["curves"]["flare"]["position_x"]
        self.assertNotEqual(next(key["value"] for key in curve["keys"] if key["frame"] == 1), 300.0)

    def test_tracker_link_picker_exposes_track_and_updates_flare(self):
        d = self.window.dispatcher
        self.window.command({"op": "create", "id": "tracker", "type": "Tracker"})
        self.window.command({"op": "connect", "id": "tracker", "input": "image", "source": "source"})
        self.window.command({"op": "set_tracks", "id": "tracker", "tracks": [
            {"name": "track1", "enabled": 1, "x": 100.0, "y": 120.0}]})
        self.window.graph.items_by_id["tracker"].setSelected(False)
        self.window.graph.items_by_id["flare"].setSelected(True)
        self.window.rebuild_properties_dock()
        link = self.window.properties_dock.findChild(QComboBox, "flare-tracker-link")
        self.assertIsNotNone(link)
        self.assertEqual(link.count(), 2)
        link.setCurrentIndex(1)
        self.assertEqual(d.document["nodes"]["flare"]["params"]["tracker_id"], "tracker")
        self.assertEqual(d.document["nodes"]["flare"]["params"]["track_index"], 0)
        values = self.window.viewer._flare_values()
        self.assertEqual((values["position_x"], values["position_y"]), (300.0, 250.0))


if __name__ == "__main__":
    unittest.main()
