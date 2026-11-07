"""Lane 2 panel sweep (2026-11-04): knobs the Nuke guide gives one label share one row, and number knobs that count are integer fields.

Each test builds the node's Properties panel and asserts the shared rows and widget kinds the ruler (tools/audit_nuke_fidelity.py)
measures: Shuffle, Transform, FrameHold, Text, Difference, Retime, Constant and HueKeyer (step S1); Tracker, Viewer, Switch, TimeClip, FrameRange, Glow, ScreenKeyer and Reformat (step S2); Remove, AppendClip, CornerPin2D, BurnIn, Clamp, Blend, DirBlur and ContactSheet (step S3); TimeBlur, TimeEcho, ZDefocus, IBKGizmo, Encryptomatte, Kronos and IBKColor, with OFlow, CopyRectangle and TimeDissolve in place of RotoPaint (step S4); Grid, MotionBlur2D, MotionBlur3D, Colorspace, SmartVector, VectorBlur, OCIODisplay and Sparkles, with OCIOFileTransform and OCIOLookTransform (step S5)."""
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
        """Some row (a WrappingRow around the first widget; the outermost one when rows nest) holds every named widget."""
        widgets = [panel.findChild(QWidget, name) for name in names]
        for name, found in zip(names, widgets):
            self.assertIsNotNone(found, name)
        rows, row = [], widgets[0]
        while row is not None:
            if isinstance(row, WrappingRow):
                rows.append(row)
            row = row.parentWidget()
        self.assertTrue(rows, names)
        self.assertTrue(any(all(r.isAncestorOf(found) for found in widgets) for r in rows), f"{names} do not share a row")

    def assertOneSplitRow(self, panel, param, *names):
        """The split knob's own row (`<param>-split-row`) holds the named widgets: its value (or channel fields) and its check boxes."""
        row = panel.findChild(QWidget, f"{param}-split-row")
        self.assertIsNotNone(row, param)
        for name in names:
            found = panel.findChild(QWidget, name)
            self.assertIsNotNone(found, name)
            self.assertTrue(row.isAncestorOf(found), f"{name} is not on the row of {param}")

    def has_menu(self, panel, *items):
        return any([c.itemText(i) for i in range(c.count())] == list(items) for c in panel.findChildren(QComboBox))

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

    # ---- step S3 ----
    def test_remove_channels_share_the_and_row(self):
        panel = self.panel("Remove")
        self.assertOneRow(panel, "channels2-field", "channels3-field", "channels4-field")
        for name in ("channels2-field", "channels3-field", "channels4-field", "layers-field"):
            self.assertTrue(isinstance(panel.findChild(QWidget, name), (QLineEdit, QComboBox)), name)
        self.assertTrue(self.has_menu(panel, "keep", "remove"))

    def test_appendclip_dissolve_is_an_integer_field(self):
        panel = self.panel("AppendClip")
        self.widget(panel, "dissolve-field", QSpinBox)
        self.assertNoSlider(panel, "dissolve-field")
        self.widget(panel, "first_frame-field", QSpinBox)
        self.widget(panel, "meta_from_first-switch", QCheckBox)
        self.assertEqual(audit.number_knob_kind("dissolve", 0), "float")      # the name alone does not say it counts frames
        self.assertIn(("AppendClip", "dissolve"), audit._COUNT_KNOBS)

    def test_cornerpin_shutter_offset_row(self):
        panel = self.panel("CornerPin")
        self.assertOneRow(panel, "shutter_offset-choice", "custom_offset-field")
        self.widget(panel, "shutter_offset-choice", QComboBox)
        self.widget(panel, "custom_offset-field", QDoubleSpinBox)
        for name in ("enable1-switch", "clamp-switch", "black_outside-switch", "motionblur-switch"):
            self.widget(panel, name, QCheckBox)
        # Invert is a check box over the stored direction text. The guide gives motionblur as a number; NodeBased keeps its on/off
        # switch beside the samples count (as Tracker and Transform do), because one number would change what is stored and rendered.
        self.assertFalse(self.has_menu(panel, "forward", "inverse"))
        box = self.widget(panel, "direction-switch", QCheckBox)
        self.assertFalse(box.isChecked())
        box.click()
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-CornerPin"]["params"]["direction"] == "inverse"))
        box = self.widget(self.panel("CornerPin"), "direction-switch", QCheckBox)
        self.assertTrue(box.isChecked())
        box.click()
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-CornerPin"]["params"]["direction"] == "forward"))

    def test_burnin_font_and_style_share_a_row(self):
        panel = self.panel("BurnIn")
        self.assertOneRow(panel, "font-field", "font_style-field")
        for name in ("font-field", "font_style-field", "top_left-field", "bottom_right-field"):
            self.widget(panel, name, QLineEdit)
        self.widget(panel, "bar-switch", QCheckBox)

    def test_clamp_enable_check_boxes_sit_beside_their_values(self):
        panel = self.panel("Clamp")
        self.assertOneSplitRow(panel, "minimum", "minimum-field", "minimum-split", "clamp_min-switch")
        self.assertOneSplitRow(panel, "maximum", "maximum-field", "maximum-split", "clamp_max-switch")
        self.assertOneRow(panel, "MinClampTo-field", "MinClampTo_enable-switch")
        self.assertOneRow(panel, "MaxClampTo-field", "MaxClampTo_enable-switch")
        for name in ("clamp_min-switch", "clamp_max-switch", "MinClampTo_enable-switch", "MaxClampTo_enable-switch"):
            self.widget(panel, name, QCheckBox)
        self.widget(panel, "minimum-split", QWidget)          # the per-channel split button is still there

    def test_clamp_split_still_edits_per_channel_and_the_check_box_writes(self):
        panel = self.panel("Clamp")
        self.widget(panel, "clamp_min-switch", QCheckBox).click()
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-Clamp"]["params"]["clamp_min"] == 0))
        self.widget(panel, "minimum-split", QWidget).click()
        self.assertTrue(wait_until(lambda: isinstance(self.window.graph_nodes()["sweep-Clamp"]["params"]["minimum"], list)))
        panel = self.panel("Clamp")        # opens the node's panel again, now with the split entries
        self.assertOneSplitRow(panel, "minimum", "minimum-r-field", "minimum-a-field", "clamp_min-switch")

    def test_blend_pairs_the_normalize_check_box(self):
        panel = self.panel("Blend")
        self.widget(panel, "normalize-switch", QCheckBox)
        self.widget(panel, "weight0-field", QDoubleSpinBox)
        self.widget(panel, "weight1-field", QDoubleSpinBox)
        # the guide's table calls the normalize check box `weight1`; the mapping pairs it with `normalize`
        import json
        pairs = json.loads((Path(audit.__file__).resolve().parents[1] / "docs" / "parity" / "nodebased_to_nuke.json").read_text())["pairs"]
        paired = {name: e["nodebased"] for e in pairs["Blend"]["knobs"] for name in e.get("nuke", [])}
        self.assertEqual(paired["weight1"], ["normalize"])

    def test_dirblur_counts_are_integer_fields(self):
        panel = self.panel("DirBlur")
        self.widget(panel, "Samples-field", QSpinBox)
        self.assertNoSlider(panel, "Samples-field")
        self.widget(panel, "UseTarget-switch", QCheckBox)
        # Target is an x, y pair of fields over the text "x,y"; Quality is a whole-number factor with a slider over low/normal/high.
        self.assertFalse(self.has_menu(panel, "low", "normal", "high"))
        self.assertOneRow(panel, "target-x-field", "target-y-field")
        self.assertEqual((self.widget(panel, "target-x-field", QDoubleSpinBox).value(), self.widget(panel, "target-y-field", QDoubleSpinBox).value()),
                         (480.0, 240.0))
        quality = self.widget(panel, "Quality-field", QDoubleSpinBox)
        self.assertEqual(quality.value(), 1.0)          # "high"
        self.assertEqual(quality.decimals(), 0)
        self.assertTrue(any(isinstance(w, QSlider) for w in quality.parentWidget().findChildren(QSlider)))

    def test_dirblur_target_and_quality_write_the_stored_text(self):
        panel = self.panel("DirBlur")
        x, y = self.widget(panel, "target-x-field", QDoubleSpinBox), self.widget(panel, "target-y-field", QDoubleSpinBox)
        x.setValue(100.5)
        x.editingFinished.emit()
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-DirBlur"]["params"]["target"] == "100.5,240"))
        panel = self.panel("DirBlur")
        self.assertEqual(self.widget(panel, "target-x-field", QDoubleSpinBox).value(), 100.5)
        for typed, stored in ((2, "normal"), (4, "low"), (3, "3"), (1, "high")):
            field = self.widget(self.panel("DirBlur"), "Quality-field", QDoubleSpinBox)
            field.setValue(typed)
            field.editingFinished.emit()
            self.assertTrue(wait_until(lambda: self.window.graph_nodes()["sweep-DirBlur"]["params"]["Quality"] == stored), (typed, stored))
        self.window.command({"op": "set", "id": "sweep-DirBlur", "param": "Quality", "value": "low"})
        self.assertEqual(self.widget(self.panel("DirBlur"), "Quality-field", QDoubleSpinBox).value(), 4.0)

    def test_dirblur_quality_factor_reads_names_and_whole_numbers(self):
        from nodebased import filterbehaviour
        for stored, factor in (("high", 1), ("normal", 2), ("low", 4), ("3", 3), ("8", 8), ("0", 1), ("junk", 1), ("1000", 64)):
            self.assertEqual(filterbehaviour.dirblur_quality_factor({"Quality": stored, "legacy_dirblur_knobs": 0}), factor, stored)
        self.assertEqual(filterbehaviour.dirblur_quality_factor({"Quality": "low", "legacy_dirblur_knobs": 1}), 1)

    def test_dirblur_whole_number_quality_scales_the_picture_down(self):
        import numpy as np
        from nodebased.core import Dispatcher
        from tests.test_2d_behaviour_filter_a import differs, kern, texture
        pixels = texture()
        blurred = {q: kern("DirBlur", pixels, blur_type="linear", length=11.0, angle=20.0, Quality=q) for q in ("high", "normal", "3", "low", "8")}
        self.assertTrue(differs(blurred["high"], blurred["3"]))
        self.assertTrue(differs(blurred["normal"], blurred["3"]))
        self.assertTrue(differs(blurred["3"], blurred["low"]))
        self.assertTrue(differs(blurred["low"], blurred["8"]))
        flat = np.full((16, 20, 4), 0.4, np.float32)
        np.testing.assert_allclose(kern("DirBlur", flat, length=9.0, Quality="3"), flat, atol=1e-6)
        # a node holding a whole-number Quality is valid; a name outside the list or a stray number is refused
        d = Dispatcher()
        d.execute({"op": "create", "id": "q", "type": "DirBlur", "pos": [0, 0]})
        for good in ("3", "64", "high", "low"):
            d.execute({"op": "set", "id": "q", "param": "Quality", "value": good})
            self.assertEqual(d.document["nodes"]["q"]["params"]["Quality"], good)
        for bad in ("1", "65", "3.5", "medium"):
            with self.assertRaises(ValueError, msg=bad):
                d.execute({"op": "set", "id": "q", "param": "Quality", "value": bad})
            self.assertEqual(d.document["nodes"]["q"]["params"]["Quality"], "low")

    def test_contactsheet_rows_and_integer_pixels(self):
        panel = self.panel("ContactSheet")
        self.assertOneRow(panel, "width-field", "height-field")
        self.assertOneRow(panel, "rows-field", "columns-field")
        self.assertOneRow(panel, "startframe-field", "endframe-field")
        for name in ("width-field", "height-field", "rows-field", "columns-field", "gap-field", "startframe-field", "endframe-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)
        for name in ("center-switch", "splitinputs-switch"):
            self.widget(panel, name, QCheckBox)

    def test_timeblur_shutter_offset_row(self):
        panel = self.panel("TimeBlur")
        self.assertOneRow(panel, "shutter_offset-choice", "custom_offset-field")
        self.widget(panel, "shutter_offset-choice", QComboBox)
        self.widget(panel, "custom_offset-field", QDoubleSpinBox)
        self.widget(panel, "divisions-field", QSpinBox)
        self.assertNoSlider(panel, "divisions-field")

    def test_timeecho_frames_are_whole_numbers(self):
        panel = self.panel("TimeEcho")
        self.widget(panel, "frames-field", QSpinBox)
        self.assertNoSlider(panel, "frames-field")

    def test_zdefocus_blades_are_a_whole_number(self):
        panel = self.panel("ZDefocus")
        self.widget(panel, "blade_count-field", QSpinBox)
        self.assertNoSlider(panel, "blade_count-field")

    def test_ibkgizmo_screen_range_is_a_slider_number(self):
        panel = self.panel("IBKGizmo")
        field = self.widget(panel, "screen_range-field", QDoubleSpinBox)
        self.assertTrue(any(isinstance(w, QSlider) for w in field.parentWidget().findChildren(QSlider)))

    def test_encryptomatte_matte_name_row(self):
        panel = self.panel("Encryptomatte")
        self.assertOneRow(panel, "matte_name_type-choice", "id0-field")
        self.widget(panel, "matte_name_type-choice", QComboBox)

    def test_kronos_input_range_row_and_frame_is_a_number(self):
        panel = self.panel("Kronos")
        self.assertOneRow(panel, "input_start-field", "input_end-field")
        self.widget(panel, "frame-field", QDoubleSpinBox)
        self.widget(panel, "shutter_samples-field", QSpinBox)
        self.assertNoSlider(panel, "shutter_samples-field")

    def test_ibkcolor_size_is_a_whole_number_and_patch_black_a_check_box(self):
        panel = self.panel("IBKColor")
        self.widget(panel, "fill_size-field", QSpinBox)
        self.assertNoSlider(panel, "fill_size-field")
        self.widget(panel, "patch_black-switch", QCheckBox)

    def test_oflow_input_range_row_and_frame_is_a_number(self):
        panel = self.panel("OFlow")
        self.assertOneRow(panel, "input_start-field", "input_end-field")
        self.widget(panel, "frame-field", QDoubleSpinBox)

    def test_copyrectangle_area_is_one_row(self):
        panel = self.panel("CopyRectangle")
        self.assertOneRow(panel, "area_x-field", "area_y-field", "area_r-field", "area_t-field")

    def test_timedissolve_frames_are_whole_numbers(self):
        panel = self.panel("TimeDissolve")
        for name in ("in-field", "out-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)

    def test_s4_nodes_measure_as_the_ruler_says(self):
        def metrics(key):
            pair = audit.load_json(audit.MAP_JSON)["pairs"][key]
            nuke = audit.load_json(audit.NUKE_JSON)["nodes"][key]
            return audit.panel_shape_metrics(nuke, pair, pair["nodebased"], key)
        rows, types = metrics("Kronos")
        self.assertEqual((rows["matched"], rows["total"]), (2, 2))
        self.assertEqual([m[0] for m in types["missed"]], ["vectorDetail"])      # an integer 1 to 6 here, a 0 to 1 fraction in Nuke
        rows, types = metrics("IBKColor")
        self.assertEqual([m[0] for m in types["missed"]], ["off", "mult"])      # single numbers here, RGB triples in Nuke
        self.assertEqual(types["unknown"], 3)

    def test_grid_number_is_one_row_of_whole_numbers(self):
        panel = self.panel("Grid")
        self.assertOneRow(panel, "number_x-field", "number_y-field")
        for name in ("number_x-field", "number_y-field"):
            self.widget(panel, name, QSpinBox)
            self.assertNoSlider(panel, name)

    def test_motionblur_shutter_offset_row(self):
        for kind in ("MotionBlur2D", "MotionBlur3D"):
            panel = self.panel(kind)
            self.assertOneRow(panel, "shutter_offset-choice", "custom_offset-field")
            self.widget(panel, "shutter_offset-choice", QComboBox)
            self.widget(panel, "custom_offset-field", QDoubleSpinBox)
            self.widget(panel, "samples-field", QSpinBox)      # the sample count stays a whole number; the node renders as before

    def test_colorspace_input_and_output_rows(self):
        panel = self.panel("Colorspace")
        self.assertOneRow(panel, "transfer_in-choice", "whitepoint_in-choice", "primaries_in-choice")
        self.assertOneRow(panel, "transfer_out-choice", "whitepoint_out-choice", "primaries_out-choice")
        for name in ("transfer_in-choice", "whitepoint_out-choice"):
            self.widget(panel, name, QComboBox)

    def test_smartvector_matte_dilation_is_a_whole_number(self):
        panel = self.panel("SmartVector")
        self.widget(panel, "matte_dilation-field", QSpinBox)
        self.assertNoSlider(panel, "matte_dilation-field")
        # vector_detail (an integer 1 to 6 here, a fraction in the guide) stays: the ruler still lists it as the one mismatch
        rows, types = self.measure("SmartVector")
        self.assertEqual([m[0] for m in types["missed"]], ["vectorDetailReg"])

    def test_vectorblur_expand_bbox_is_a_whole_number(self):
        panel = self.panel("VectorBlur")
        self.widget(panel, "grow_bbox-field", QSpinBox)
        self.assertNoSlider(panel, "grow_bbox-field")
        rows, types = self.measure("VectorBlur")
        self.assertEqual([m[0] for m in types["missed"]], ["uv"])      # a layer picker in the guide, a layer name field here

    def test_sparkles_ray_count_is_a_whole_number(self):
        panel = self.panel("Sparkles")
        self.widget(panel, "spark_ray_nb-field", QSpinBox)
        self.assertNoSlider(panel, "spark_ray_nb-field")

    def test_ocio_invert_direction_is_a_check_box_over_the_direction_text(self):
        for kind in ("OCIODisplay", "OCIOFileTransform", "OCIOLookTransform"):
            key = "sweep-" + kind
            panel = self.panel(kind)
            self.assertFalse(self.has_menu(panel, "forward", "inverse"), kind)
            box = self.widget(panel, "transform_direction-switch", QCheckBox)
            self.assertFalse(box.isChecked())
            self.assertEqual(self.window.graph_nodes()[key]["params"]["transform_direction"], "forward")
            box.click()
            self.assertTrue(wait_until(lambda: self.window.graph_nodes()[key]["params"]["transform_direction"] == "inverse"), kind)
            self.assertTrue(self.widget(self.panel(kind), "transform_direction-switch", QCheckBox).isChecked())
            self.widget(self.panel(kind), "transform_direction-switch", QCheckBox).click()
            self.assertTrue(wait_until(lambda: self.window.graph_nodes()[key]["params"]["transform_direction"] == "forward"), kind)

    def measure(self, key):
        pair = audit.load_json(audit.MAP_JSON)["pairs"][key]
        nuke = audit.load_json(audit.NUKE_JSON)["nodes"][key]
        return audit.panel_shape_metrics(nuke, pair, pair["nodebased"], key)

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
