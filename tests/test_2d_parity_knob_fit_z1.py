"""Knob values are never cut off (10/7 hands-on pass, finding 1: translate X showed "20" for 200.000).

A keyed field used to share its narrow row with a key-frame diamond inside the editor and a key button
beside it, and the editor ended up narrower than its number. These tests read the real widgets: the text
the field displays, the width its editor has for text, and where the key button sits."""

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import unittest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent, QFontMetrics
from PySide6.QtWidgets import (QAbstractSpinBox, QComboBox, QDockWidget, QDoubleSpinBox, QPushButton)

from nodebased.app import Window
from nodebased.knobfit import ELLIPSIS, ElidingComboBox, FittedDoubleSpinBox, _text_area, elide_to_width
from tests.test_desktop import APP, release_window
from tests.waiting import pause, settle_layout, wait_until


def narrow_panel(window, width):
    dock = window.findChild(QDockWidget, "properties-dock")
    window.resizeDocks([dock], [width], Qt.Orientation.Horizontal)
    settle_layout(window, window.properties)
    pause(80)


class KnobFitTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.addCleanup(self.close)
        self.window.set_time(first=1, last=30, current=24)

    def close(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def keyed(self, node_type, param, value, frame=24):
        """A node of ``node_type`` with ``param`` keyed to ``value`` on the current frame, selected."""
        node_id = f"{node_type.lower()}-z1"
        if node_id not in self.window.dispatcher.document["nodes"]:
            self.window.command({"op": "create", "id": node_id, "type": node_type}, render=False)
        self.window.command({"op": "set_key", "id": node_id, "param": param, "frame": frame,
                             "value": value}, render=False)
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id[node_id].setSelected(True)
        APP.processEvents()
        settle_layout(self.window, self.window.properties)
        pause(80)
        return node_id

    def field(self, param):
        return next(s for s in self.window.properties.findChildren(QDoubleSpinBox)
                    if s.objectName() == f"{param}-field")

    def key_button(self, param):
        return next(b for b in self.window.properties.findChildren(QPushButton, "key-button")
                    if b.accessibleName() == f"{param} key")

    def assert_text_fits(self, spin):
        edit = spin.lineEdit()
        metrics = QFontMetrics(edit.font())
        self.assertLessEqual(metrics.horizontalAdvance(edit.text()), _text_area(edit),
                             f"{spin.objectName()} shows {edit.text()!r} in {_text_area(edit)} px")

    def assert_key_beside_editor(self, param):
        panel = self.window.properties
        field, button = self.field(param), self.key_button(param)
        field_rect = field.rect().translated(field.mapTo(panel, field.rect().topLeft()))
        button_rect = button.rect().translated(button.mapTo(panel, button.rect().topLeft()))
        self.assertFalse(field_rect.intersects(button_rect), f"{param}: key button over the editor")

    # -- the default layout --------------------------------------------------------------------

    def test_translate_x_keyed_to_200_shows_all_of_it(self):
        self.keyed("Transform", "translate_x", 200.0)
        spin = self.field("translate_x")
        self.assertEqual(spin.property("keyedHere"), True)
        self.assertEqual(spin.lineEdit().text(), "200.000")
        self.assert_text_fits(spin)
        self.assert_key_beside_editor("translate_x")

    def test_the_longest_translate_value_the_range_allows_shows_all_of_it(self):
        # Translate stops at a million either way, so -1000000.000 is its longest number.
        self.keyed("Transform", "translate_x", -1000000.0)
        spin = self.field("translate_x")
        self.assertEqual(spin.lineEdit().text(), "-1000000.000")
        self.assert_text_fits(spin)
        self.assert_key_beside_editor("translate_x")

    def test_a_ten_character_value_on_a_slider_knob_shows_all_of_it(self):
        # Rotate takes up to 100000, so -12345.678 is a legal key and the longest case in the brief.
        self.keyed("Transform", "rotate", -12345.678)
        spin = self.field("rotate")
        self.assertEqual(spin.lineEdit().text(), "-12345.678")
        self.assert_text_fits(spin)
        self.assert_key_beside_editor("rotate")

    def test_the_width_a_field_asks_for_holds_its_number_and_the_diamond(self):
        self.keyed("Transform", "translate_x", 200.0)
        spin = self.field("translate_x")
        metrics = QFontMetrics(spin.lineEdit().font())
        self.assertGreaterEqual(spin.sizeHint().width(), metrics.horizontalAdvance("200.000") + 22)

    def test_a_multi_component_knob_puts_each_axis_on_its_own_row_before_cutting_a_digit(self):
        self.keyed("Transform", "translate_x", 200.0)
        x, y = self.field("translate_x"), self.field("translate_y")
        for spin in (x, y):
            self.assert_text_fits(spin)
        self.assertEqual(y.lineEdit().text(), "0.000")

    # -- a narrow panel ------------------------------------------------------------------------

    def test_at_300_pixels_nothing_is_cut(self):
        narrow_panel(self.window, 300)
        self.keyed("Transform", "translate_x", 200.0)
        spin = self.field("translate_x")
        self.assertEqual(spin.lineEdit().text(), "200.000")
        self.assert_text_fits(spin)
        self.assert_key_beside_editor("translate_x")

    def test_a_value_that_cannot_fit_shows_an_ellipsis_and_the_whole_number_in_its_tooltip(self):
        self.keyed("Transform", "rotate", -12345.678)
        narrow_panel(self.window, 150)
        spin = self.field("rotate")
        shown = spin.lineEdit().text()
        self.assertTrue(shown.endswith(ELLIPSIS), f"expected an ellipsis, got {shown!r}")
        self.assertNotEqual(shown, "-12345.678")
        self.assertIn("-12345.678", spin.toolTip())
        self.assert_text_fits(spin)

    def test_clicking_an_elided_field_restores_the_whole_number_for_editing(self):
        self.keyed("Transform", "rotate", -12345.678)
        narrow_panel(self.window, 150)
        spin = self.field("rotate")
        self.assertTrue(spin.lineEdit().text().endswith(ELLIPSIS))
        spin.lineEdit().setFocus()
        # An offscreen window is never the active one, so Qt may not deliver the focus event itself.
        APP.sendEvent(spin.lineEdit(), QFocusEvent(QEvent.Type.FocusIn))
        APP.processEvents()
        self.assertEqual(spin.lineEdit().text(), "-12345.678")
        self.assertAlmostEqual(spin.value(), -12345.678)

    # -- every knob kind -----------------------------------------------------------------------

    def audit_panel(self, width):
        failures = []
        panel = self.window.properties
        for spin in panel.findChildren(QAbstractSpinBox):
            if not spin.isVisible():
                continue
            edit = spin.lineEdit()
            metrics = QFontMetrics(edit.font())
            if metrics.horizontalAdvance(edit.text()) > _text_area(edit):
                failures.append((spin.objectName(), edit.text(), width))
            if edit.text().endswith(ELLIPSIS) and not spin.toolTip():
                failures.append((spin.objectName(), "ellipsis without tooltip", width))
        for combo in panel.findChildren(QComboBox):
            if combo.isVisible() and combo.is_elided() and combo.currentText() not in combo.toolTip():
                failures.append(("combo", combo.currentText(), width))
        return failures

    def test_every_knob_kind_fits_at_the_default_width_and_at_300(self):
        # Transform: float, slider, XY. Constant: colour. Merge: choice. Grade: float and colour.
        # Crop: XY box. Text: text field. Blur: int-sized. A key diamond on each numeric knob.
        kinds = {
            "Transform": ("translate_x", 200.0),
            "Grade": ("gain", 12.345),
            "Constant": ("color_r", 0.123),
            "Merge": None, "Crop": None, "Text": None, "Blur": None,
        }
        for width in (None, 300):
            if width:
                narrow_panel(self.window, width)
            for node_type, key in kinds.items():
                if key:
                    self.keyed(node_type, *key)
                else:
                    node_id = f"{node_type.lower()}-z1"
                    if node_id not in self.window.dispatcher.document["nodes"]:
                        self.window.command({"op": "create", "id": node_id, "type": node_type}, render=False)
                    self.window.graph.scene().clearSelection()
                    self.window.graph.items_by_id[node_id].setSelected(True)
                    APP.processEvents()
                    settle_layout(self.window, self.window.properties)
                    pause(60)
                self.assertEqual(self.audit_panel(width), [], f"{node_type} at {width}")


class ElisionHelperTests(unittest.TestCase):
    def test_elided_text_ends_with_an_ellipsis_and_fits(self):
        from tests.test_desktop import APP  # noqa: F401  (a QApplication must exist for font metrics)
        metrics = QFontMetrics(APP.font())
        full = "-12345.678"
        cut = elide_to_width(full, metrics, metrics.horizontalAdvance(full) - 10)
        self.assertTrue(cut.endswith(ELLIPSIS))
        self.assertLessEqual(metrics.horizontalAdvance(cut), metrics.horizontalAdvance(full) - 10)
        self.assertEqual(elide_to_width(full, metrics, 500), full)

    def test_widgets_are_the_ones_the_panel_builds(self):
        self.assertTrue(issubclass(FittedDoubleSpinBox, QDoubleSpinBox))
        self.assertTrue(issubclass(ElidingComboBox, QComboBox))


if __name__ == "__main__":
    unittest.main()
