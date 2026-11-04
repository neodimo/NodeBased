"""Lane 2 panel sweep (2026-11-04): knobs the Nuke guide gives one label share one row, and number knobs that count are integer fields.

Each test builds the node's Properties panel and asserts the shared rows and widget kinds the ruler (tools/audit_nuke_fidelity.py)
measures: Shuffle, Transform, FrameHold, Text, Difference, Retime, Constant and HueKeyer (step S1); Tracker, Viewer, Switch, TimeClip, FrameRange, Glow, ScreenKeyer and Reformat (step S2)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import uuid

from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSlider, QSpinBox, QWidget)

from nodebased.app import Window
from nodebased.knobfit import WrappingRow
from nodebased.viewersettings import ViewerSettings
from tests.waiting import wait_until
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_nuke_fidelity as audit  # noqa: E402

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

    # ---- step S2 ----
    def test_tracker_shutter_row_and_integer_counts(self):
        panel = self.panel("Tracker")
        self.assertOneRow(panel, "shutter_offset-choice", "custom_offset-field")
        self.widget(panel, "shutter_offset-choice", QComboBox)
        self.widget(panel, "custom_offset-field", QDoubleSpinBox)
        for name in ("grab_interval-field", "jitter_period-field", "smoothing-field", "pattern_radius-field", "search_radius-field", "reference_frame-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)
        # The guide's motionblur is a number; NodeBased keeps its on/off switch (a kernel change to give it a quality value is out of scope).
        self.widget(panel, "motionblur-switch", QCheckBox)

    def test_viewer_knob_controls_match_the_audit_table(self):
        window = self.window
        settings = ViewerSettings(window)
        want = {"frame_range": QLineEdit, "frame_range_lock": QCheckBox, "frame_increment": QSpinBox, "rgb_only": QCheckBox,
                "ignore_pixel_aspect": QCheckBox, "zoom_lock": QCheckBox, "show_overscan": QCheckBox, "overscan": QSpinBox,
                "full_frame_processing": QCheckBox, "safe_zone": QComboBox, "format_center": QCheckBox, "center_fstop": QDoubleSpinBox,
                "shutter_angle": QDoubleSpinBox, "use_gpu_for_viewer": QCheckBox}
        for name, cls in want.items():
            self.assertIs(type(settings.controls[name]), cls, name)
        table = audit._VIEWER_WIDGETS
        kinds = {QLineEdit: "string", QCheckBox: "bool", QSpinBox: "int", QDoubleSpinBox: "float", QComboBox: "enum"}
        guide = {"frame_range": "frame_range", "frame_range_lock": "frame_range_lock", "frame_increment": "frame_increment", "rgb_only": "rgb_only",
                 "ignore_pixel_aspect": "ignore_pixel_aspect", "zoom_lock": "zoom_lock", "show_overscan": "show_overscan", "overscan": "overscan",
                 "full_frame_processing": "full_frame_processing", "safe_zone": "safe_zone", "format_center": "format_center",
                 "center_fstop": "center_fstop", "shutter_angle": "shutter_angle", "use_gpu_for_viewer": "useGPUForViewer"}
        for name, nuke_name in guide.items():
            self.assertEqual(table[nuke_name], kinds[want[name]], name)
        # the strip: gain and gamma spin boxes, the zebra check box, the proxy tier and the two mask dropdowns
        self.assertIsInstance(window.exposure, QDoubleSpinBox)
        self.assertIsInstance(window.gamma, QDoubleSpinBox)
        self.assertIsInstance(window.zebra, QCheckBox)
        self.assertIsInstance(window.proxy, QComboBox)
        self.assertIsInstance(window.mask_choice, QComboBox)
        self.assertIsInstance(window.mask_mode, QComboBox)
        self.assertEqual((table["gain"], table["gamma"], table["clip_warning"], table["downrez"], table["masking_ratio"], table["masking_mode"]),
                         ("float", "float", "bool", "enum", "enum", "enum"))
        settings.deleteLater()

    def test_switch_which_is_an_integer_field(self):
        panel = self.panel("Switch")
        self.widget(panel, "which-field", QSpinBox)
        self.assertNoSlider(panel, "which-field")

    def test_timeclip_range_rows(self):
        panel = self.panel("TimeClip")
        self.assertOneRow(panel, "first-field", "before-choice", "last-field", "after-choice")
        self.assertOneRow(panel, "origfirst-field", "origlast-field")
        for name in ("first-field", "last-field", "origfirst-field", "origlast-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)
        for name in ("before-choice", "after-choice"):
            self.widget(panel, name, QComboBox)
        self.widget(panel, "reverse-switch", QCheckBox)

    def test_framerange_frames_share_a_row(self):
        panel = self.panel("FrameRange")
        self.assertOneRow(panel, "first_frame-field", "last_frame-field")
        for name in ("first_frame-field", "last_frame-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)
        self.assertEqual(len([c for c in panel.findChildren(QComboBox) if c.isVisible()]), 2)   # before and after

    def test_glow_tint_is_a_colour_row(self):
        panel = self.panel("Glow")
        self.assertOneRow(panel, "red-field", "green-field", "blue-field")
        for name in ("red-field", "green-field", "blue-field"):
            self.widget(panel, name, QDoubleSpinBox)
        self.widget(panel, "nonlinear-switch", QCheckBox)
        self.widget(panel, "crop-switch", QCheckBox)

    def test_screenkeyer_edge_colour_is_a_colour_row(self):
        panel = self.panel("ScreenKeyer")
        self.assertOneRow(panel, "source_edge_red-field", "source_edge_green-field", "source_edge_blue-field")
        self.assertOneRow(panel, "screen_red-field", "screen_green-field", "screen_blue-field")
        self.widget(panel, "unpremultiply_result-switch", QCheckBox)

    def test_reformat_size_is_one_row(self):
        panel = self.panel("Reformat")
        self.assertOneRow(panel, "width-field", "height-field")
        for name in ("center-switch", "flip-switch", "flop-switch", "turn-switch", "box_fixed-switch"):
            self.widget(panel, name, QCheckBox)

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
