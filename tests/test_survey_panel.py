"""Rendering 13 step L2: the CameraSolve3D panel's Survey list and the viewer pick behind it."""
import json
import os
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QLineEdit, QListWidget, QPushButton, QWidget

import tests.isolation  # noqa: F401
from tests.test_roto_ui import APP
from tests.waiting import wait_until
from nodebased.app import Window
from nodebased.core import Dispatcher, empty_document

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class SurveyPanelTests(unittest.TestCase):
    def setUp(self):
        dispatcher = Dispatcher(empty_document())
        for key, kind in (("plate", "Checker"), ("solve", "CameraSolve3D")):
            dispatcher.execute({"op": "create", "id": key, "type": kind})
        dispatcher.execute({"op": "connect", "id": "solve", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "time", "first": 1, "last": 24, "current": 3})
        self.window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(self.window, "saved_document", self.window.dispatcher.document),
                                 self.window.close(), self.window.executor.shutdown(wait=True, cancel_futures=True),
                                 APP.processEvents()))
        self.window.show()
        self.window.graph.items_by_id["solve"].setSelected(True)
        self.window.inspect("solve")

    def panel(self):
        self.assertTrue(wait_until(lambda: self.window.properties.widget().findChild(QWidget, "survey-panel") is not None))
        return self.window.properties.widget().findChild(QWidget, "survey-panel")

    def survey(self):
        return json.loads(self.window.dispatcher.document["nodes"]["solve"]["params"]["survey"])

    def test_an_empty_survey_says_up_to_scale_and_the_panel_summarises_the_solve(self):
        panel = self.panel()
        self.assertIn("Up to scale (no picks yet)", panel.findChild(QLabel, "survey-status").text())
        summary = self.window.properties.widget().findChild(QLabel, "cameraSolveSummary")
        self.assertEqual(summary.text(), "No solve yet.")
        self.assertFalse(panel.findChild(QPushButton, "survey-pick-delete").isEnabled())

    def test_picks_are_added_on_the_current_frame_and_positions_and_distances_edit_the_survey(self):
        for x, y in ((10.0, 20.0), (30.5, 40.25), (50.0, 5.0)):
            self.assertTrue(self.window.survey_pick("solve", x, y))
        picks = self.survey()["picks"]
        self.assertEqual([(p["name"], p["frame"], p["x"], p["y"]) for p in picks],
                         [("S1", 3, 10.0, 20.0), ("S2", 3, 30.5, 40.25), ("S3", 3, 50.0, 5.0)])
        panel = self.panel()
        self.assertEqual(panel.findChild(QListWidget, "survey-pick-list").count(), 3)
        self.assertIn("Up to scale (3 picks)", panel.findChild(QLabel, "survey-status").text())
        # a distance between two picks fixes the scale
        panel.findChild(QPushButton, "survey-distance-add").click()
        self.assertEqual(self.survey()["distances"], [{"a": "S1", "b": "S2", "distance": 1.0}])
        self.assertIn("Scale from 1 surveyed distance", self.panel().findChild(QLabel, "survey-status").text())
        # positions for three picks fix the placement
        positions = ("0 0 0", "3 4 0", "0, 0, 2")
        for index, text in enumerate(positions):
            field = self.panel().findChild(QLineEdit, f"survey-pick-position-{index}")
            field.setText(text)
            field.editingFinished.emit()
        self.assertEqual([p["position"] for p in self.survey()["picks"]], [[0, 0, 0], [3, 4, 0], [0, 0, 2]])
        self.assertIn("Aligned to 3 surveyed points", self.panel().findChild(QLabel, "survey-status").text())
        # deleting a pick takes its distances with it
        self.panel().findChild(QListWidget, "survey-pick-list").setCurrentRow(0)
        self.panel().findChild(QPushButton, "survey-pick-delete").click()
        self.assertEqual([p["name"] for p in self.survey()["picks"]], ["S2", "S3"])
        self.assertEqual(self.survey()["distances"], [])

    def test_the_viewer_pick_views_the_node_and_escape_ends_it(self):
        self.assertTrue(self.window.begin_survey_pick("solve"))
        self.assertEqual(self.window.viewer.survey_picking, "solve")
        self.assertEqual(self.window.dispatcher.document["view"], "solve")
        QTest.keyClick(self.window.viewer, Qt.Key.Key_Escape)
        self.assertIsNone(self.window.viewer.survey_picking)

    def test_a_node_without_a_plate_cannot_be_picked_on(self):
        self.window.command({"op": "connect", "id": "solve", "input": "image", "source": None})
        shown = []
        self.window._show_command_error = shown.append
        self.assertFalse(self.window.begin_survey_pick("solve"))
        self.assertIn("Connect the plate", str(shown[0]))


if __name__ == "__main__":
    unittest.main()
