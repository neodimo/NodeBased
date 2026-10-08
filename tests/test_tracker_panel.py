"""Tracker point list, rename propagation, and Roto link choices."""
import os
import unittest

from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QListWidget, QPushButton, QWidget

import tests.isolation
from tests.test_roto_ui import APP, triangle
from tests.waiting import wait_until
from nodebased.app import Window
from nodebased.core import Dispatcher, empty_document

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TrackerPointListTests(unittest.TestCase):
    def test_list_select_rename_and_delete_update_roto_link(self):
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        for key, kind in (("plate", "Checker"), ("t", "Tracker"), ("r", "Roto")):
            dispatcher.execute({"op": "create", "id": key, "type": kind})
        dispatcher.execute({"op": "connect", "id": "t", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "view", "id": "t"})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document),
                                 window.close(), window.executor.shutdown(wait=True, cancel_futures=True),
                                 APP.processEvents()))
        window.show()
        window.graph.items_by_id["t"].setSelected(True)
        window.inspect("t")
        self.assertTrue(wait_until(lambda: window.properties.widget().findChild(
            QWidget, "tracker-points-panel") is not None))
        panel = window.properties.widget().findChild(QWidget, "tracker-points-panel")
        listing = panel.findChild(QListWidget, "tracker-track-list")
        self.assertEqual(listing.count(), 0)
        panel.findChild(QPushButton, "tracker-track-add").click()
        self.assertTrue(window.viewer.tracker_picking)
        point = window.viewer.mapFromScene(QPointF(20.0, 22.0))
        QTest.mouseClick(window.viewer.viewport(), Qt.MouseButton.LeftButton, pos=point)
        self.assertIsNotNone(window._tracker_seed)
        self.assertTrue(window.analyse_tracker("t", "forward", first_frame=1, last_frame=1))
        self.assertTrue(wait_until(lambda: len(window.dispatcher.document.get("node_data", {})
                                    .get("t", {}).get("tracks", [])) == 1, timeout=10))
        panel = window.properties.widget().findChild(QWidget, "tracker-points-panel")
        listing = panel.findChild(QListWidget, "tracker-track-list")
        self.assertEqual(listing.count(), 1)
        self.assertIn("Analyzed", [label.text() for label in panel.findChildren(QLabel)])
        listing.setCurrentRow(0)
        self.assertEqual(window._tracker_selected_index, 0)
        name = panel.findChild(QLineEdit, "tracker-track-name-0")
        name.setText("face")
        name.editingFinished.emit()
        self.assertEqual(window.dispatcher.document["node_data"]["t"]["tracks"][0]["name"], "face")
        shape = triangle()
        shape["track_link"] = {"tracker_id": "t", "track_name": "face"}
        window.command({"op": "set_shapes", "id": "r", "shapes": [shape]}, render=False)
        self.assertEqual(window.dispatcher.document["node_data"]["r"]["shapes"][0]["track_link"]["track_name"], "face")

        window.graph.items_by_id["r"].setSelected(True)
        window.inspect("r")
        combo = window.properties.widget().findChild(QComboBox, "roto-shape-track-link-0")
        self.assertIsNotNone(combo)
        self.assertTrue(any("face" in combo.itemText(i) for i in range(combo.count())))

        window.graph.items_by_id["t"].setSelected(True)
        window.inspect("t")
        panel = window.properties.widget().findChild(QWidget, "tracker-points-panel")
        panel.findChild(QListWidget, "tracker-track-list").setCurrentRow(0)
        delete = panel.findChild(QPushButton, "tracker-track-delete")
        delete.click()
        self.assertEqual(window.dispatcher.document.get("node_data", {}).get("t", {}).get("tracks", []), [])
        self.assertIsNone(window.dispatcher.document["node_data"]["r"]["shapes"][0].get("track_link"))


if __name__ == "__main__":
    unittest.main()
