"""New look, step 5 (Lane 2), parts 1 and 2: the viewer's floating control strip and its quiet corner
readouts, and that nothing clips at 1280x720 and 1440x920 (also on wide fonts)."""
import unittest

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QFont, QImage, QMouseEvent
from PySide6.QtWidgets import QApplication, QToolButton, QWidget

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased import app as nodebased_app_module
from nodebased.app import Window
from nodebased.core import viewer_look, viewer_masks, viewer_proxy, viewer_roi
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class Base(unittest.TestCase):
    def open_window(self, size=(1440, 920), scale=None):
        original = nodebased_app_module.DEFAULT_WINDOW_SIZE
        nodebased_app_module.DEFAULT_WINDOW_SIZE = size
        try:
            self.window = Window()
        finally:
            nodebased_app_module.DEFAULT_WINDOW_SIZE = original
        w = self.window
        self._base_font = w.font()
        if scale:
            font = QFont(self._base_font)
            font.setPixelSize(max(1, round(self._base_font.pixelSize() * scale)))
            w.setFont(font)
        w.show()
        self.assertTrue(wait_until(lambda: w.frame is not None))
        for _ in range(5):
            APP.processEvents()
        return w

    def tearDown(self):
        window = self.__dict__.pop("window", None)
        if window is not None:
            overlay = getattr(window, "shortcut_overlay", None)
            if overlay is not None:
                overlay.close()
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
            APP.processEvents()

    def hover(self, w, x, y):
        point = w.viewer.viewportTransform().map(QPointF(x, y))
        event = QMouseEvent(QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton,
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        w.viewer.mouseMoveEvent(event)

    def show_frame(self, w, frame, scale=1):
        w.frame = frame
        image = QImage(frame.shape[1], frame.shape[0], QImage.Format.Format_RGBA8888)
        image.fill(0)
        w._show_image(image, scale, None)
        APP.processEvents()

    def menu_action(self, menu, *path):
        """The action reached through submenu titles `path[:-1]` and the action text `path[-1]`."""
        for title in path[:-1]:
            menu = next(a.menu() for a in menu.actions() if a.menu() is not None and a.text() == title)
        return next(a for a in menu.actions() if a.text() == path[-1])


class StripControlTests(Base):
    def test_the_strip_floats_at_the_top_of_the_viewer_and_holds_the_mockup_controls(self):
        w = self.open_window()
        strip = w.viewer_strip
        self.assertIs(strip.parentWidget(), w.view_stack)
        self.assertTrue(strip.isVisible())
        centre = strip.geometry().center().x()
        self.assertAlmostEqual(centre, w.view_stack.width() / 2, delta=2)
        self.assertLess(strip.geometry().top(), 24)
        for name in ("2d", "3d"):
            self.assertTrue(strip.isAncestorOf(w.viewer_mode_buttons[name]))
        self.assertEqual([w.channels.itemText(i) for i in range(w.channels.count())], ["RGB", "R", "G", "B", "A"])
        for widget in (w.channels, w.exposure, w.gamma, w.roi_button, strip.display_button, strip.fit_button,
                       strip.more_button):
            self.assertTrue(strip.isAncestorOf(widget) and widget.isVisible(), widget.objectName())
        # The old two rows are gone.
        self.assertEqual([c for c in w.viewer_panel.children() if getattr(c, "objectName", lambda: "")()
                          == "viewer-controls-row"], [])

    def test_channel_buttons_drive_the_channel_the_shortcuts_and_exports_read(self):
        w = self.open_window()
        seen = []
        w.channels.currentTextChanged.connect(seen.append)
        for name in ("R", "G", "B", "A", "RGB"):
            w.channels.buttons[name].click()
            self.assertEqual(w.channels.currentText(), name)
            self.assertTrue(w.channels.buttons[name].isChecked())
        self.assertEqual(seen, ["R", "G", "B", "A", "RGB"])
        # The graph's own shortcut path writes through the same object.
        w.channels.setCurrentText("G")
        self.assertTrue(w.channels.buttons["G"].isChecked())
        self.assertEqual(w.channels.currentIndex(), 2)

    def test_the_two_three_d_buttons_switch_the_view_stack(self):
        w = self.open_window()
        w.viewer_mode_buttons["3d"].click()
        self.assertEqual(w.viewer_mode(), "3d")
        self.assertTrue(w.viewer_strip.isVisible())
        w.viewer_mode_buttons["2d"].click()
        self.assertEqual(w.viewer_mode(), "2d")

    def test_gain_gamma_and_roi_edit_the_documents_viewer_look(self):
        w = self.open_window()
        w.exposure.setValue(1.5)
        w.gamma.setValue(2.0)
        look = viewer_look(w.dispatcher.document)
        self.assertEqual((look["gain"], look["gamma"]), (1.5, 2.0))
        w.roi_button.click()
        self.assertTrue(viewer_roi(w.dispatcher.document)["on"])
        w.roi_button.click()
        self.assertFalse(viewer_roi(w.dispatcher.document)["on"])

    def test_the_display_menu_lists_the_project_views_and_the_viewers_own_displays(self):
        w = self.open_window()
        button = w.viewer_strip.display_button
        self.assertEqual(button.text(), w.display_view.currentText())
        button.menu().aboutToShow.emit()
        menu = button.menu()
        titles = [a.text() for a in menu.actions() if not a.isSeparator()]
        for index in range(w.display_view.count()):
            self.assertIn(w.display_view.itemText(index), titles)
        for index in range(w.viewer_display.count()):
            self.assertIn(w.viewer_display.itemText(index), titles)
        # A project view pick is the combo's pick.
        other = next(w.display_view.itemText(i) for i in range(w.display_view.count())
                     if w.display_view.itemText(i) != w.display_view.currentText())
        self.menu_action(menu, other).trigger()
        self.assertEqual(w.display_view.currentText(), other)
        self.assertEqual(button.text(), other)
        # A viewer display pick is saved in the document's viewer look.
        raw = next(w.viewer_display.itemText(i) for i in range(w.viewer_display.count())
                   if w.viewer_display.itemText(i) not in ("Project view", ""))
        button.menu().aboutToShow.emit()
        self.menu_action(button.menu(), raw).trigger()
        self.assertEqual(viewer_look(w.dispatcher.document)["display"], raw)
        self.assertEqual(button.text(), raw)

    def test_fit_returns_the_picture_to_the_free_band(self):
        w = self.open_window()
        w.viewer.scale(3, 3)
        w.viewer_strip.fit_button.click()
        APP.processEvents()
        rect = w.viewer.mapFromScene(w.viewer.sceneRect()).boundingRect()
        self.assertGreaterEqual(rect.top(), w.viewer.insets[0] - 30,
                                "the fitted picture stays below the strip")
        self.assertLessEqual(rect.bottom(), w.viewer.viewport().height() - w.viewer.insets[1] + 30)

    def test_the_more_menu_holds_every_control_the_old_rows_had(self):
        w = self.open_window()
        strip = w.viewer_strip
        strip.fill_more_menu()
        menu = strip.more_menu
        top = [a.text() for a in menu.actions() if not a.isSeparator()]
        for expected in ("Proxy", "Proxy while playing", w.zebra.text(), "Format mask", "Mask mode", "1:1 pixels",
                         "Reset gain", "Reset gamma", "Viewer inputs"):
            self.assertIn(expected, top)
        # Proxy: picking 1/2 is the combo's own pick, saved with the document.
        self.menu_action(menu, "Proxy", "1/2").trigger()
        self.assertEqual(w.proxy.currentData(), 2)
        self.assertEqual(viewer_proxy(w.dispatcher.document), 2)
        # Zebra and proxy-while-playing are the check boxes' state.
        zebra = self.menu_action(menu, w.zebra.text())
        zebra.setChecked(True)
        self.assertTrue(w.zebra.isChecked())
        self.assertTrue(viewer_look(w.dispatcher.document)["zebra"])
        playing = self.menu_action(menu, "Proxy while playing")
        playing.setChecked(False)
        self.assertFalse(w.playback_proxy.isChecked())
        # Format mask and mask mode.
        mask = w.mask_choice.itemText(2)
        self.menu_action(menu, "Format mask", mask).trigger()
        self.assertEqual(viewer_masks(w.dispatcher.document)["mask"], mask)
        # Resets.
        w.exposure.setValue(2.0)
        self.menu_action(menu, "Reset gain").trigger()
        self.assertEqual(viewer_look(w.dispatcher.document)["gain"], 0.0)
        w.gamma.setValue(2.0)
        self.menu_action(menu, "Reset gamma").trigger()
        self.assertEqual(viewer_look(w.dispatcher.document)["gamma"], 1.0)
        # 1:1.
        w.viewer.scale(2, 2)
        self.menu_action(menu, "1:1 pixels").trigger()
        self.assertAlmostEqual(w.viewer.transform().m11(), 1.0)

    def test_the_viewer_inputs_stay_reachable_from_the_menu(self):
        w = self.open_window()
        strip = w.viewer_strip
        strip.fill_more_menu()
        self.menu_action(strip.more_menu, "Viewer inputs", "B buffer", "Input 2").trigger()
        self.assertEqual(w.viewer.input_state()["b"], 2)
        self.menu_action(strip.more_menu, "Viewer inputs", "B buffer", "None").trigger()
        self.assertIsNone(w.viewer.input_state()["b"])
        self.menu_action(strip.more_menu, "Viewer inputs", "Compare", "wipe").trigger()
        self.assertEqual(w.viewer.input_state()["compare"], "wipe")


class CornerReadoutTests(Base):
    def test_the_left_corner_names_resolution_colour_space_and_proxy_state(self):
        w = self.open_window()
        corners = w.viewer_corners
        self.assertEqual(corners.resolution.text(), "960 × 540")
        self.assertIn("ACEScg", corners.space.text())
        self.assertTrue(corners.proxy.text())
        frame = np.zeros((10, 20, 4), dtype=np.float32)
        self.show_frame(w, frame, scale=2)
        self.assertEqual(corners.resolution.text(), "40 × 20")
        self.assertEqual(corners.proxy.text(), "proxy 1/2")
        self.show_frame(w, frame, scale=1)
        w.playback_proxy.setChecked(False)
        self.assertEqual(corners.proxy.text(), "full resolution")

    def test_cursor_position_value_swatch_and_zoom_update_on_cursor_move(self):
        w = self.open_window()
        frame = np.zeros((4, 5, 4), dtype=np.float32)
        frame[1, 2] = (0.5, 0.25, 0.125, 1.0)
        self.show_frame(w, frame)
        readout = w.viewer.pixel_readout
        self.assertFalse(readout.isVisible())
        self.hover(w, 2.2, 1.2)
        self.assertTrue(readout.isVisible())
        self.assertEqual(readout.position.text(), "x 2  y 2")
        self.assertEqual(readout.values.text(), "0.500 0.250 0.125")
        self.assertEqual(readout.label.text(), "2, 2  0.50000 0.25000 0.12500 1.00000",
                         "the full machine-readable line is unchanged")
        color = readout.swatch.styleSheet()
        self.assertIn("rgb(128, 64, 32)", color)
        self.assertTrue(w.viewer_corners.right.isAncestorOf(readout))
        self.hover(w, 3.2, 3.2)
        self.assertEqual(readout.position.text(), "x 3  y 0")
        self.assertEqual(readout.values.text(), "0.000 0.000 0.000")
        # Off the picture the readout goes, the zoom stays.
        self.hover(w, 100, 100)
        self.assertFalse(readout.isVisible())
        w.viewer.fit()
        w.viewer.viewport().repaint()
        APP.processEvents()
        percent = round(w.viewer.transform().m11() * 100)
        self.assertEqual(w.viewer_corners.zoom.text(), f"{percent}%")
        w.viewer.scale(2, 2)
        w.viewer.viewport().repaint()
        APP.processEvents()
        self.assertEqual(w.viewer_corners.zoom.text(), f"{round(w.viewer.transform().m11() * 100)}%")

    def test_the_readouts_sit_in_the_bottom_corners_outside_the_picture(self):
        w = self.open_window()
        stack = w.view_stack
        left, right = w.viewer_corners.left.geometry(), w.viewer_corners.right.geometry()
        self.assertGreaterEqual(left.left(), 0)
        self.assertLessEqual(right.right(), stack.width())
        self.assertLess(left.right(), right.left(), "the corners do not overlap")
        self.assertGreater(left.top(), stack.height() * 0.8)
        picture = w.viewer.mapFromScene(w.viewer.sceneRect()).boundingRect()
        self.assertLess(picture.bottom(), left.top(), "the picture ends above the readouts")
        self.assertGreater(picture.top(), w.viewer_strip.geometry().bottom(), "and starts below the strip")


class NoClippingTests(Base):
    def assert_fits(self, size, scale=None):
        w = self.open_window(size, scale)
        stack = w.view_stack
        strip = w.viewer_strip
        self.assertTrue(stack.rect().contains(strip.geometry()), f"strip {strip.geometry()} outside {stack.rect()}")
        inner = strip.rect()
        checked = 0
        for widget in strip.findChildren(QWidget):
            if widget.parentWidget() is not strip or not widget.isVisible():
                continue
            checked += 1
            self.assertTrue(inner.contains(widget.geometry()), f"{widget.objectName() or type(widget)} clipped")
            if isinstance(widget, QToolButton):
                width = widget.fontMetrics().horizontalAdvance(widget.text()) if widget.text() else 0
                if widget.toolButtonStyle() != Qt.ToolButtonStyle.ToolButtonIconOnly:
                    self.assertGreaterEqual(widget.width(), width, f"{widget.text()} narrower than its text")
        self.assertGreaterEqual(checked, 10)
        for corner in (w.viewer_corners.left, w.viewer_corners.right):
            self.assertTrue(stack.rect().contains(corner.geometry()), f"{corner.geometry()} outside {stack.rect()}")
        self.assertLess(w.viewer_corners.left.geometry().right(), w.viewer_corners.right.geometry().left())
        row = w.time_row
        bounds = row.rect()
        for widget in row.findChildren(QWidget):
            if widget.parentWidget() is not row or not widget.isVisibleTo(row):
                continue
            self.assertTrue(bounds.contains(widget.geometry()), f"{widget.objectName() or type(widget)} clipped")
            self.assertGreaterEqual(widget.width(), widget.minimumSizeHint().width(),
                                    f"{widget.objectName() or type(widget)} squeezed below its minimum")
        self.assertGreaterEqual(w.frame_slider.width(), 140)
        for field in (w.frame_current, w.frame_first, w.frame_last, w.frame_fps):
            self.assertTrue(field.isVisibleTo(row), "an edited field is never the part that goes")
        # The strip and the time row are inside the VIEWER dock.
        self.assertTrue(w.viewer_dock.isAncestorOf(strip) and w.viewer_dock.isAncestorOf(row))

    def test_nothing_clips_at_1280x720(self):
        self.assert_fits((1280, 720))

    def test_nothing_clips_at_1440x920(self):
        self.assert_fits((1440, 920))

    def test_nothing_clips_at_1280x720_on_wide_fonts(self):
        self.assert_fits((1280, 720), scale=1.25)

    def test_nothing_clips_at_1440x920_on_wide_fonts(self):
        self.assert_fits((1440, 920), scale=1.25)

    def test_a_narrow_viewer_turns_roi_and_fit_into_icons_before_clipping(self):
        w = self.open_window((1280, 720), scale=1.25)
        strip = w.viewer_strip
        full = strip.sizeHint().width()
        w.view_stack.setFixedWidth(full - 30)
        for _ in range(5):
            APP.processEvents()
        self.assertTrue(strip.compact)
        self.assertTrue(w.view_stack.rect().contains(strip.geometry()))
        self.assertEqual(w.roi_button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonIconOnly)


if __name__ == "__main__":
    unittest.main()
