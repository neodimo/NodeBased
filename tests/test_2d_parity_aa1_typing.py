"""Lane L2 (2D parity) plan 23, step AA1 part 3: numeric fields keep every digit typed (hands-on
pass 10/7, finding 7). Qt refused a digit once the number could no longer fit the range's own
digit count, so 123456 into translate x (range +-8192) became 1234. These tests type with real
key events into the real editors."""
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

import tests.isolation  # noqa: F401
from nodebased.app import Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class TypingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Window()
        cls.window.show()
        cls.window.activateWindow()
        QTest.qWaitForWindowActive(cls.window)
        assert wait_until(lambda: cls.window.frame is not None)
        w = cls.window
        w.command({"op": "batch", "commands": [{"op": "delete", "id": k} for k in list(w.graph_nodes())]})
        for key, kind in (("xf", "Transform"), ("gr", "Grade"), ("cc", "ColorCorrect"), ("cn", "Constant"),
                          ("bl", "Blur"), ("cam", "Camera3D"), ("rd", "Read")):
            w.command({"op": "create", "id": key, "type": kind, "pos": [0, 0], "params": {}})
        APP.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document
        cls.window.close()
        cls.window.deleteLater()
        APP.processEvents()

    def panel(self, key):
        w = self.window
        w.graph.scene().clearSelection()
        w.graph.items_by_id[key].setSelected(True)
        APP.processEvents()
        return w.properties.widget()

    def field(self, key, name):
        # A committed value rebuilds the panel, so the field is looked up again each time.
        panel = self.panel(key)
        return panel.findChild(QDoubleSpinBox, name) or panel.findChild(QSpinBox, name)

    def type_into(self, spin, text):
        spin.setFocus()
        edit = spin.lineEdit()
        edit.selectAll()
        QTest.keyClicks(edit, text)
        # The offscreen platform never gives the window focus, so Tab alone does not leave the
        # field; Return finishes the edit the same way the real Tab does (editingFinished).
        QTest.keyClick(edit, Qt.Key.Key_Return)
        QTest.keyClick(edit, Qt.Key.Key_Tab)
        APP.processEvents()

    def stored(self, key, param):
        return self.window.dispatcher.document["nodes"][key]["params"][param]

    def test_translate_x_keeps_every_digit_typed_then_return_and_tab(self):
        for text, expected in (("123456", 123456.0), ("12345.678", 12345.678),
                               ("-99999.5", -99999.5), ("1e6", 1000000.0)):
            with self.subTest(text=text):
                self.type_into(self.field("xf", "translate_x-field"), text)
                self.assertTrue(wait_until(lambda: self.stored("xf", "translate_x") == expected),
                                f"{text} stored {self.stored('xf', 'translate_x')}")
                self.assertEqual(self.field("xf", "translate_x-field").value(), expected)

    def test_a_number_past_the_range_clamps_to_the_range_and_shows_it(self):
        self.type_into(self.field("xf", "translate_y-field"), "99999999")
        self.assertTrue(wait_until(lambda: self.stored("xf", "translate_y") == 1000000.0))
        self.assertEqual(self.field("xf", "translate_y-field").lineEdit().text(), "1000000.000")

    def test_every_numeric_field_of_each_knob_kind_takes_its_extreme_values_with_all_digits(self):
        # float, int, XY (Transform), colour components (Constant, ColorCorrect), XYZ (Camera3D)
        checked = 0
        for key in ("xf", "gr", "cc", "cn", "bl", "cam", "rd"):
            panel = self.panel(key)
            names = [spin.objectName() for spin in panel.findChildren(QDoubleSpinBox) + panel.findChildren(QSpinBox)
                     if spin.objectName().endswith("-field") and spin.isEnabled()]
            for name in names:
                spin = self.field(key, name)
                low, high = spin.minimum(), spin.maximum()
                if abs(low) > 1e9 or abs(high) > 1e9:
                    continue
                for target in (high, low):
                    text = f"{target:.3f}" if isinstance(spin, QDoubleSpinBox) else str(int(target))
                    with self.subTest(node=key, field=name, typed=text):
                        # Typing only: some knobs (a camera's near plane) refuse the committed
                        # extreme on their own terms, which is not what this test is about.
                        spin = self.field(key, name)
                        spin.setFocus()
                        spin.lineEdit().selectAll()
                        QTest.keyClicks(spin.lineEdit(), text)
                        spin.interpretText()
                        self.assertEqual(spin.value(), float(target))
                        checked += 1
        self.assertGreater(checked, 30)


if __name__ == "__main__":
    unittest.main()
