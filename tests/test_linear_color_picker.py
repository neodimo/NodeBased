"""The linear float colour picker that replaced QColorDialog on colour knobs (DiMo 10/18: "not
0-255 based", Nuke's colour picking, not Photoshop's)."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import unittest

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDoubleSpinBox, QFrame

from nodebased.app import Window
from nodebased.colorpicker import (LinearColorPicker, hsv_to_rgb, rgb_to_hsv, rgb_to_tmi, tmi_to_rgb)
from tests.test_desktop import APP, release_window
from tests.waiting import wait_until


class PickerModelTests(unittest.TestCase):
    def picker(self, rgba):
        picker = LinearColorPicker(rgba, None, "Test", view="Linear")
        self.addCleanup(picker.deleteLater)
        return picker

    def test_untouched_values_round_trip_exactly(self):
        values = [0.123456789, 2.5, -0.25, 0.7071067811865476]
        picker = self.picker(values)
        accepted = []
        picker.colorAccepted.connect(accepted.append)
        picker.show()
        APP.processEvents()
        picker.accept()
        self.assertEqual(accepted, [values])
        self.assertEqual(self.picker(values[:3]).color(), values[:3])

    def test_a_value_of_two_and_a_half_survives_every_row(self):
        picker = self.picker([0.1, 0.2, 0.3])
        picker.rows["R"].field.setValue(2.5)
        self.assertEqual(picker.color()[0], 2.5)
        self.assertAlmostEqual(picker.rows["V"].field.value(), 2.5)
        picker.rows["V"].field.setValue(2.5)            # re-entering V keeps the colour
        self.assertAlmostEqual(picker.color()[0], 2.5, places=12)
        picker.rows["I"].field.setValue(picker.rows["I"].field.value())
        self.assertAlmostEqual(picker.color()[0], 2.5, places=4)

    def test_fields_have_no_eight_bit_or_unit_ceiling(self):
        picker = self.picker([0.5, 0.5, 0.5, 1.0])
        for name, row in picker.rows.items():
            self.assertLess(row.field.minimum(), -1000, name)
            self.assertGreater(row.field.maximum(), 1000, name)
        picker.rows["G"].field.setValue(-0.75)
        self.assertEqual(picker.color()[1], -0.75)

    def test_hsv_and_tmi_maps_invert(self):
        for rgb in ([0.2, 0.4, 0.8], [3.0, 1.5, 0.25], [0.0, 0.0, 0.0]):
            np.testing.assert_allclose(hsv_to_rgb(*rgb_to_hsv(rgb)), rgb, atol=1e-12)
            np.testing.assert_allclose(tmi_to_rgb(*rgb_to_tmi(rgb)), rgb, atol=1e-12)

    def test_ctrl_drag_on_the_swatch_scrubs_intensity(self):
        picker = self.picker([0.25, 0.5, 1.0])
        picker.resize(420, 640)
        picker.show()
        APP.processEvents()
        swatch = picker.swatch
        start = QPointF(40, 20)
        QTest.mousePress(swatch, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, start.toPoint())
        move = QMouseEvent(QEvent.Type.MouseMove, QPointF(100, 20), swatch.mapToGlobal(QPointF(100, 20)),
                           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
        APP.sendEvent(swatch, move)
        QTest.mouseRelease(swatch, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, QPoint(100, 20))
        np.testing.assert_allclose(picker.color(), [0.5, 1.0, 2.0], rtol=1e-9)


class PickerInWindowTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.resize(1400, 900)
        self.window.show()
        APP.processEvents()
        wait_until(lambda: self.window.frame is not None)

    def tearDown(self):
        self.addCleanup(release_window, self)
        picker = getattr(self.window, "color_picker", None)
        if picker is not None:
            picker.close()
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def constant_key(self):
        return next(k for k, n in self.window.graph_nodes().items() if n["type"] == "Constant")

    def test_colour_knob_swatch_opens_the_linear_picker_and_commits_floats(self):
        w = self.window
        key = self.constant_key()
        w.inspect(key)
        APP.processEvents()
        swatch = w.properties.widget().findChild(QFrame, "color-swatch")
        QTest.mouseClick(swatch, Qt.MouseButton.LeftButton)
        picker = w.color_picker
        self.assertIsInstance(picker, LinearColorPicker)
        params = w.graph_nodes()[key]["params"]
        self.assertEqual(picker.color(), [params["red"], params["green"], params["blue"], params["alpha"]])
        picker.set_color([2.5, -0.125, 0.333333, 1.0])
        picker.accept()
        self.assertTrue(wait_until(lambda: w.graph_nodes()[key]["params"]["red"] == 2.5))
        params = w.graph_nodes()[key]["params"]
        self.assertEqual((params["green"], params["blue"]), (-0.125, 0.333333))
        w.command({"op": "undo"})
        self.assertNotEqual(w.graph_nodes()[key]["params"]["red"], 2.5)   # one undo step

    def test_eyedropper_returns_the_linear_float_under_the_cursor(self):
        w = self.window
        key = self.constant_key()
        w.command({"op": "batch", "commands": [
            {"op": "set", "id": key, "param": "red", "value": 3.25},
            {"op": "set", "id": key, "param": "green", "value": -0.5},
            {"op": "set", "id": key, "param": "blue", "value": 0.0625},
            {"op": "set", "id": key, "param": "alpha", "value": 1.0}]})
        w.command({"op": "view", "id": key})
        self.assertTrue(wait_until(lambda: w.frame is not None and abs(float(w.frame[0, 0, 0]) - 3.25) < 1e-6))
        picker = w.open_color_picker([0.0, 0.0, 0.0], "Sample", lambda rgba: None)
        picker.eyedropper.click()
        self.assertIsNotNone(w.viewer.color_sampling)
        viewer = w.viewer
        rect = viewer.format_rect if viewer.format_rect is not None else viewer.sceneRect()
        point = viewer.mapFromScene(rect.center())
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        APP.processEvents()
        scene = viewer.mapToScene(point)
        expected = viewer.linear_pixel_at(QPointF(scene))
        self.assertIsNotNone(expected)
        self.assertEqual(picker.color(), [float(v) for v in expected[:3]])
        self.assertAlmostEqual(picker.color()[0], 3.25, places=5)
        self.assertAlmostEqual(picker.color()[1], -0.5, places=5)
        self.assertIsNone(w.viewer.color_sampling)


if __name__ == "__main__":
    unittest.main()
