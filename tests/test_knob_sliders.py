"""Properties overhaul (DiMo, 10/18): every float knob shows a slider that draws.

From 10/9 to 10/18 RulerSlider painted with a QStyleOptionSlider whose subControls were left at
SC_None, so every knob slider in the app was an empty dark box beside its value field. These
tests build every node kind's panel and look at the pixels: a slider that does not draw fails."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from nodebased.app import FloatSliderControl, RulerSlider, Window
from nodebased.core import SPECS
from nodebased.knobs import knob_layout, slider_range
from tests.test_desktop import APP, release_window


def drawn_colors(widget):
    """The distinct colours along the slider's middle row."""
    image = widget.grab().toImage()
    y = image.height() // 2
    return {image.pixelColor(x, y).name() for x in range(image.width())}


class SliderDrawTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.resize(1400, 900)
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_every_float_knob_builds_a_visible_slider_that_draws(self):
        w = self.window
        missing, blank = [], []
        for index, node_type in enumerate(sorted(SPECS)):
            params = [group.params[0] for group in knob_layout(node_type)
                      if group.kind in ("float", "float_slider")]
            if not params:
                continue
            key = f"slider_scan_{index}"
            w.dispatcher.execute({"op": "create", "id": key, "type": node_type, "pos": [0, 0]})
            panel = w.build_node_panel(key)
            w.set_properties_widget(panel)
            APP.processEvents()
            for param in params:
                pairs = [pair for pair in panel.findChildren(FloatSliderControl)
                         if pair.spin.objectName() == f"{param}-field"]
                if not pairs:
                    missing.append((node_type, param))
                    continue
                slider = pairs[0].slider
                if slider.isHidden() or pairs[0].isHidden():
                    missing.append((node_type, param))
                    continue
                if not slider.isVisibleTo(panel):
                    continue                     # inside a section the artist has to open (RotoPaint's Detect)
                if slider.width() <= 0:
                    missing.append((node_type, param))
                    continue
                if len(drawn_colors(slider)) < 3:      # background only: the groove did not draw
                    blank.append((node_type, param))
            w.dispatcher.execute({"op": "delete", "id": key})
        self.assertEqual(missing, [], f"float knobs without a visible slider: {missing[:20]}")
        self.assertEqual(blank, [], f"sliders that drew nothing: {blank[:20]}")

    def test_soft_ranges_sit_inside_the_hard_limits(self):
        from nodebased.core import parameter_limits
        for node_type in SPECS:
            for group in knob_layout(node_type):
                if group.kind not in ("float", "float_slider"):
                    continue
                lo, hi = parameter_limits(node_type, group.params[0])
                soft = slider_range(node_type, group.params[0], group.soft_range)
                self.assertLess(soft[0], soft[1], (node_type, group.params))
                self.assertGreaterEqual(soft[0], lo, (node_type, group.params))
                self.assertLessEqual(soft[1], hi, (node_type, group.params))


class SliderInputTests(unittest.TestCase):
    def make(self, value=0.5):
        pair = FloatSliderControl((-100.0, 100.0), (0.0, 1.0), value)
        pair.resize(320, 28)
        pair.show()
        APP.processEvents()
        self.addCleanup(pair.deleteLater)
        return pair

    def groove_x(self, slider, fraction):
        left, width = slider._groove_span()
        return round(left + width * fraction)

    def test_a_press_on_the_groove_jumps_there_and_commits_on_release(self):
        pair = self.make(0.5)
        released = []
        pair.slider.sliderReleased.connect(lambda: released.append(pair.value()))
        QTest.mouseClick(pair.slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(self.groove_x(pair.slider, 0.25), 12))
        self.assertAlmostEqual(pair.value(), 0.25, delta=0.01)
        self.assertEqual(len(released), 1)

    def test_ctrl_drags_fine_and_shift_drags_coarse(self):
        moved = {}
        for name, modifier in (("plain", Qt.KeyboardModifier.NoModifier),
                               ("fine", Qt.KeyboardModifier.ControlModifier),
                               ("coarse", Qt.KeyboardModifier.ShiftModifier)):
            pair = self.make(0.1 if name == "coarse" else 0.5)
            start = pair.value()
            slider = pair.slider
            x = self.groove_x(slider, start)
            QTest.mousePress(slider, Qt.MouseButton.LeftButton, modifier, QPoint(x, 12))
            QTest.mouseMove(slider, QPoint(x + 10, 12))
            # QTest.mouseMove carries no modifiers; send the move the way a held key arrives.
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtCore import QEvent, QPointF
            event = QMouseEvent(QEvent.Type.MouseMove, QPointF(x + 10, 12), slider.mapToGlobal(QPointF(x + 10, 12)),
                                Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, modifier)
            APP.sendEvent(slider, event)
            QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, modifier, QPoint(x + 10, 12))
            moved[name] = pair.value() - start
        self.assertGreater(moved["plain"], 0)
        self.assertAlmostEqual(moved["fine"], moved["plain"] / 10, delta=0.002)
        self.assertGreater(moved["coarse"], moved["plain"] * 5)

    def test_typed_value_outside_the_soft_range_is_kept(self):
        pair = self.make(0.5)
        pair.spin.setValue(37.125)
        self.assertEqual(pair.value(), 37.125)
        self.assertEqual(pair.slider.value(), RulerSlider.STEPS)

    def test_slider_draws_its_groove(self):
        pair = self.make(0.5)
        self.assertGreaterEqual(len(drawn_colors(pair.slider)), 3)


class KnobSizingTests(unittest.TestCase):
    """Fields take the width their content needs; only sliders and file paths use the dock's spare
    width (DiMo 10/18: every field ran to the right edge of the panel)."""

    def setUp(self):
        self.window = Window()
        self.window.resize(1800, 900)
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def panel_for(self, node_type, width=700):
        w = self.window
        w.resizeDocks([w.properties_dock], [width], Qt.Orientation.Horizontal)
        key = f"size_{node_type.lower()}_{width}"
        if key not in w.dispatcher.document["nodes"]:
            w.dispatcher.execute({"op": "create", "id": key, "type": node_type, "pos": [0, 0]})
        panel = w.build_node_panel(key)
        w.set_properties_widget(panel)
        for _ in range(3):
            APP.processEvents()
        return panel

    def test_numbers_and_choices_keep_their_content_width_in_a_wide_dock(self):
        from PySide6.QtWidgets import QComboBox, QSpinBox
        panel = self.panel_for("Constant")
        self.assertGreater(panel.width(), 500)
        width_field = panel.findChild(QSpinBox, "width-field") or next(
            f for f in panel.findChildren(QSpinBox) if f.value() == 960)
        digits = width_field.fontMetrics().horizontalAdvance("-00000000")
        self.assertLess(width_field.width(), digits + 80)
        self.assertGreaterEqual(width_field.width(), digits)
        panel = self.panel_for("Merge")
        combo = next(c for c in panel.findChildren(QComboBox) if c.currentText() == "over")
        longest = max(combo.fontMetrics().horizontalAdvance(combo.itemText(i)) for i in range(combo.count()))
        self.assertGreaterEqual(combo.width(), longest)
        self.assertLess(combo.width(), longest + 90)

    def test_colour_row_keeps_swatch_and_four_fields_on_one_line(self):
        from PySide6.QtWidgets import QDoubleSpinBox, QFrame
        for width in (700, 380):
            panel = self.panel_for("Constant", width)
            fields = [panel.findChild(QDoubleSpinBox, f"{c}-field") for c in ("red", "green", "blue", "alpha")]
            swatch = panel.findChild(QFrame, "color-swatch")
            tops = {field.mapTo(panel, QPoint(0, 0)).y() for field in fields}
            self.assertEqual(len(tops), 1, width)
            self.assertLess(abs(swatch.mapTo(panel, QPoint(0, 0)).y() - min(tops)), swatch.height(), width)

    def test_slider_rows_use_the_spare_width(self):
        panel = self.panel_for("Grade")
        pair = next(p for p in panel.findChildren(FloatSliderControl) if p.spin.objectName() == "exposure-field")
        self.assertGreater(pair.slider.width(), 150)


if __name__ == "__main__":
    unittest.main()
