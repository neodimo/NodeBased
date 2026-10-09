"""Lane L2 (2D parity) plan 21, step Y1: a time row an artist reads at a glance (real-display QA
10/6, finding 1). Current frame first and largest, a labelled In and Out, one edit per typed value,
an inverted range refused, Home/End/Left/Right on the playhead."""
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QSpinBox

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased import app as nodebased_app_module
from nodebased.app import Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


def time_state(w):
    return dict(w.dispatcher.document["time"])


class TimeRowTests(unittest.TestCase):
    def open_window(self, size=(1440, 920)):
        original = nodebased_app_module.DEFAULT_WINDOW_SIZE
        nodebased_app_module.DEFAULT_WINDOW_SIZE = size
        try:
            self.window = Window()
        finally:
            nodebased_app_module.DEFAULT_WINDOW_SIZE = original
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        w = self.window
        w.set_time(first=1, last=48, current=5, fps=24.0)
        APP.processEvents()
        return w

    def tearDown(self):
        window = self.__dict__.pop("window", None)
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
            APP.processEvents()

    def type_into(self, spin, text):
        line = spin.lineEdit()
        spin.setFocus()
        line.selectAll()
        QTest.keyClicks(line, text)
        QTest.keyClick(line, Qt.Key.Key_Return)
        APP.processEvents()

    def left_edge(self, widget):
        return widget.mapTo(self.window.time_row, widget.rect().topLeft()).x()

    def test_field_order_and_labels_at_1440x920(self):
        w = self.open_window()
        order = [w.frame_current_label, w.frame_current, w.frame_slider, w.frame_first_label,
                 w.frame_first, w.frame_last_label, w.frame_last, w.frame_fps, w.fps_presets]
        edges = [self.left_edge(widget) for widget in order]
        self.assertEqual(edges, sorted(edges), "the time row is out of reading order")
        self.assertEqual(len(set(edges)), len(edges))
        self.assertEqual([w.frame_current_label.text(), w.frame_first_label.text(),
                          w.frame_last_label.text()], ["Frame", "In", "Out"])
        for label in (w.frame_current_label, w.frame_first_label, w.frame_last_label):
            self.assertTrue(label.isVisible())
        spins = (w.frame_first, w.frame_last, w.frame_fps)
        self.assertTrue(all(w.frame_current.width() > spin.width() for spin in spins),
                        "the current frame is the largest field: frame, in, out, fps = "
                        f"{[s.width() for s in (w.frame_current, *spins)]}")
        for spin in (w.frame_current, w.frame_first, w.frame_last, w.frame_fps):
            self.assertFalse(spin.keyboardTracking())

    def test_typing_24_into_the_current_frame_moves_only_the_playhead(self):
        w = self.open_window()
        revision = w.dispatcher.revision
        self.type_into(w.frame_current, "24")
        after = time_state(w)
        self.assertEqual((after["first"], after["last"], after["current"]), (1, 48, 24))
        self.assertEqual(w.dispatcher.revision, revision + 1, "typing 24 is one edit")
        self.assertEqual((w.frame_first.value(), w.frame_last.value()), (1, 48))

    def test_relative_input_in_the_current_frame_field(self):
        w = self.open_window()
        self.type_into(w.frame_current, "+10")
        self.assertEqual(time_state(w)["current"], 15)
        self.type_into(w.frame_current, "-5")
        self.assertEqual(time_state(w)["current"], 10)
        self.type_into(w.frame_current, "-99")
        self.assertEqual(time_state(w)["current"], 1, "relative moves stop at In")
        self.assertEqual(w.frame_current.value(), 1)

    def test_in_after_out_is_refused_and_reverts(self):
        w = self.open_window()
        revision = w.dispatcher.revision
        self.type_into(w.frame_first, "60")
        after = time_state(w)
        self.assertEqual((after["first"], after["last"], after["current"]), (1, 48, 5))
        self.assertEqual(w.frame_first.value(), 1, "the field reverts")
        self.assertEqual(w.dispatcher.revision, revision)
        self.assertIn("In 60", w.statusBar().currentMessage())

    def test_out_before_in_is_refused_and_reverts(self):
        w = self.open_window()
        w.set_time(first=10, last=48, current=12)
        self.type_into(w.frame_last, "3")
        after = time_state(w)
        self.assertEqual((after["first"], after["last"]), (10, 48))
        self.assertEqual(w.frame_last.value(), 48)
        self.assertIn("Out 3", w.statusBar().currentMessage())

    def test_a_range_edit_is_one_undo_step(self):
        w = self.open_window()
        self.type_into(w.frame_last, "30")
        self.assertEqual(time_state(w)["last"], 30)
        self.type_into(w.frame_first, "8")
        state = time_state(w)
        self.assertEqual((state["first"], state["last"], state["current"]), (8, 30, 8))
        w.command({"op": "undo"})
        state = time_state(w)
        self.assertEqual((state["first"], state["last"], state["current"]), (1, 30, 5))
        self.assertEqual((w.frame_first.value(), w.frame_last.value()), (1, 30))
        w.command({"op": "undo"})
        self.assertEqual(time_state(w)["last"], 48)

    def test_ctrl_z_restores_a_range_change(self):
        w = self.open_window()
        self.type_into(w.frame_last, "30")
        w.viewer.setFocus()
        QTest.keyClick(w.viewer, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        APP.processEvents()
        self.assertEqual(time_state(w)["last"], 48)
        self.assertEqual(w.frame_last.value(), 48)

    def test_home_end_left_right_move_the_playhead_from_viewer_and_graph(self):
        w = self.open_window()
        w.set_time(first=10, last=40, current=20)
        for target in (w.viewer, w.graph):
            with self.subTest(target=type(target).__name__):
                target.setFocus()
                w.set_time(current=20)
                QTest.keyClick(target, Qt.Key.Key_Right)
                self.assertEqual(time_state(w)["current"], 21)
                QTest.keyClick(target, Qt.Key.Key_Left)
                QTest.keyClick(target, Qt.Key.Key_Left)
                self.assertEqual(time_state(w)["current"], 19)
                QTest.keyClick(target, Qt.Key.Key_End)
                self.assertEqual(time_state(w)["current"], 40)
                QTest.keyClick(target, Qt.Key.Key_Home)
                # In the node graph Home frames every node (step Y2); elsewhere it is the first frame.
                self.assertEqual(time_state(w)["current"], 40 if target is w.graph else 10)

    def test_arrow_keys_stay_in_a_text_field(self):
        w = self.open_window()
        w.frame_current.setFocus()
        line = w.frame_current.lineEdit()
        line.setText("5")
        line.setCursorPosition(1)
        QTest.keyClick(line, Qt.Key.Key_Left)
        QTest.keyClick(line, Qt.Key.Key_Home)
        self.assertEqual(line.cursorPosition(), 0)
        self.assertEqual(time_state(w)["current"], 5, "the playhead did not move")

    def assert_unclipped(self, size):
        w = self.open_window(size)
        row = w.time_row
        bounds = row.rect()
        panel = w.viewer_panel
        self.assertTrue(panel.rect().contains(row.mapTo(panel, row.rect().topLeft())),
                        "the time row starts inside the viewer panel")
        self.assertLessEqual(row.width(), panel.width())
        checked = 0
        for widget in row.findChildren(QLabel) + row.findChildren(QSpinBox) + [w.frame_fps, w.fps_presets, w.play_button]:
            if widget.parentWidget() is not row or widget is w.frame_info:
                continue
            checked += 1
            self.assertTrue(widget.isVisible(), widget.objectName() or type(widget).__name__)
            self.assertTrue(bounds.contains(widget.geometry()), f"{widget.geometry()} outside {bounds}")
            self.assertGreaterEqual(widget.width(), widget.minimumSizeHint().width(),
                                    f"{type(widget).__name__} is squeezed below its minimum")
        # Nine: the TIME caption is gone and the transport buttons sit in their own group (new look, step 5).
        self.assertGreaterEqual(checked, 9)
        self.assertGreaterEqual(w.frame_slider.width(), 140, "the timeline keeps usable width")

    def test_no_time_row_widget_is_clipped_at_1280x720(self):
        self.assert_unclipped((1280, 720))

    def test_no_time_row_widget_is_clipped_at_1440x920(self):
        self.assert_unclipped((1440, 920))


if __name__ == "__main__":
    unittest.main()
