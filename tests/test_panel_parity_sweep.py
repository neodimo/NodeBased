"""Lane 2 panel sweep (2026-11-04): knobs the Nuke guide gives one label share one row, and number knobs that count are integer fields.

Each test builds the node's Properties panel and asserts the shared rows and widget kinds the ruler (tools/audit_nuke_fidelity.py)
measures: Shuffle, Transform, FrameHold, Text, Difference, Retime, Constant and HueKeyer."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import uuid

from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSlider, QSpinBox, QWidget)

from nodebased.app import Window
from nodebased.knobfit import WrappingRow
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class PanelSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Window(agent_name="panel-sweep-" + uuid.uuid4().hex)
        cls.window.resize(1200, 1200)
        cls.window.show()
        assert wait_until(lambda: cls.window.frame is not None), "first frame did not arrive"

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document
        cls.window.close(); cls.window.deleteLater(); APP.processEvents()

    def panel(self, kind):
        key = "sweep-" + kind
        self.window.command({"op": "create", "id": key, "type": kind, "pos": [0, 0]})
        self.window.inspect(key)
        QTest.qWait(120)
        return self.window.properties.widget()

    def widget(self, panel, name, cls):
        found = panel.findChild(cls, name)
        self.assertIsNotNone(found, name)
        return found

    def assertOneRow(self, panel, *names):
        widgets = [panel.findChild(QWidget, name) for name in names]
        for name, found in zip(names, widgets):
            self.assertIsNotNone(found, name)
        row = widgets[0]
        while row is not None and not isinstance(row, WrappingRow):
            row = row.parentWidget()
        self.assertIsNotNone(row, names)
        for name, found in zip(names, widgets):
            self.assertTrue(row.isAncestorOf(found), f"{name} is not on the row of {names[0]}")

    def assertNoSlider(self, panel, name):
        field = panel.findChild(QWidget, name)
        self.assertFalse(any(isinstance(w, QSlider) for w in field.parentWidget().findChildren(QSlider)), name)

    def test_shuffle_input_row(self):
        panel = self.panel("Shuffle")
        self.assertOneRow(panel, "fromInput1-choice", "layer-field", "fromInput2-choice", "in2-field")
        for name in ("fromInput1-choice", "fromInput2-choice"):
            self.widget(panel, name, QComboBox)
        for name in ("layer-field", "in2-field"):
            self.assertTrue(isinstance(panel.findChild(QWidget, name), (QLineEdit, QComboBox)), name)

    def test_transform_shutter_offset_row(self):
        panel = self.panel("Transform")
        self.assertOneRow(panel, "shutter_offset-choice", "custom_offset-field")
        self.widget(panel, "shutter_offset-choice", QComboBox)
        self.widget(panel, "custom_offset-field", QDoubleSpinBox)
        self.assertOneRow(panel, "translate_x-field", "translate_y-field")
        self.widget(panel, "shutter-field", QDoubleSpinBox)
        # The guide's motionblur is a number; NodeBased keeps its on/off switch (a kernel change to give it a quality value is out of scope).
        self.widget(panel, "motionblur-switch", QCheckBox)

    def test_framehold_first_frame_is_an_integer_field(self):
        panel = self.panel("FrameHold")
        self.widget(panel, "first_frame-field", QSpinBox)
        self.assertNoSlider(panel, "first_frame-field")
        self.widget(panel, "increment-field", QDoubleSpinBox)
        self.widget(panel, "enable_path_mask-switch", QCheckBox)
        self.widget(panel, "inject_mask-switch", QCheckBox)

    def test_text_justify_and_font_rows(self):
        panel = self.panel("Text")
        self.assertOneRow(panel, "justify-choice", "yjustify-choice")
        self.widget(panel, "justify-choice", QComboBox)
        self.widget(panel, "yjustify-choice", QComboBox)
        self.assertOneRow(panel, "font-field", "index-field")
        self.widget(panel, "font-field", QLineEdit)
        self.widget(panel, "index-field", QLineEdit)
        self.widget(panel, "enable_shadows-switch", QCheckBox)

    def test_difference_output_row(self):
        panel = self.panel("Difference")
        self.assertOneRow(panel, "output_enabled-switch", "output_channel-choice")
        self.widget(panel, "output_enabled-switch", QCheckBox)
        self.widget(panel, "output_channel-choice", QComboBox)

    def test_hue_keyer_output_row(self):
        panel = self.panel("HueKeyer")
        self.assertOneRow(panel, "output_enabled-switch", "output_channel-choice")
        self.widget(panel, "output_enabled-switch", QCheckBox)
        self.widget(panel, "output_channel-choice", QComboBox)
        self.widget(panel, "invert-switch", QCheckBox)

    def test_retime_range_rows(self):
        panel = self.panel("Retime")
        self.assertOneRow(panel, "input_range_start-field", "input_range_start_lock-switch", "input_range_end-field", "input_range_end_lock-switch")
        self.assertOneRow(panel, "output_range_start-field", "output_range_start_lock-switch", "output_range_end-field", "output_range_end_lock-switch")
        for name in ("input_range_start-field", "input_range_end-field", "output_range_start-field", "output_range_end-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)
        for name in ("input_range_start_lock-switch", "output_range_end_lock-switch", "reverse-switch"):
            self.widget(panel, name, QCheckBox)

    def test_constant_frame_range_row(self):
        panel = self.panel("Constant")
        self.assertOneRow(panel, "first-field", "last-field")
        for name in ("first-field", "last-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)

    def test_row_edits_reach_the_document(self):
        panel = self.panel("Constant")
        field = self.widget(panel, "last-field", QSpinBox)
        field.setValue(9)
        field.editingFinished.emit()
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-Constant"]["params"]["last"] == 9))
        panel = self.panel("Difference")
        self.widget(panel, "output_enabled-switch", QCheckBox).click()
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-Difference"]["params"]["output_enabled"] == 0))


if __name__ == "__main__":
    unittest.main()
