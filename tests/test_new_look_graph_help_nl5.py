"""New look, step 5 (Lane 2), part 4: the graph's shortcut cheat-line became the "?" overlay, and the dock
title bars are slim and quiet (the docks still float, close and rearrange)."""
import unittest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased.shortcuthelp import GRAPH_SHORTCUTS, graph_shortcut_items, graph_shortcut_text
from tests.test_new_look_viewer_strip_nl5 import Base

APP = QApplication.instance() or QApplication([])

# The graph's old hint line, word for word (the part after its "NODE GRAPH" caption).
OLD_HINT_LINE = ("Tab search/add  ·  R/G/M/T/B/C/S/O/P/U/W create  ·  Period Dot  ·  1 view  ·  D bypass  ·  F frame  "
                 "·  Ctrl+A select all  ·  Ctrl+C/X/V copy/cut/paste  ·  Alt+C duplicate  ·  Ctrl+G group / "
                 "Ctrl+Shift+G ungroup  ·  MMB or Alt+drag pan  ·  drag output ↔ input to wire  ·  Ctrl-drag noodle "
                 "midpoint inserts Dot  ·  click a wired input to rewire")


class GraphShortcutOverlayTests(Base):
    def test_the_cheat_line_is_gone(self):
        w = self.open_window()
        self.assertIsNone(w.findChild(QLabel, "graph-shortcuts-hint"))
        for label in w.graph_panel.findChildren(QLabel):
            self.assertFalse(label.text().strip().startswith("NODE GRAPH"), label.text())
            self.assertNotIn("Tab search/add", label.text())

    def test_the_overlay_lists_every_shortcut_the_line_had(self):
        self.assertEqual(graph_shortcut_text(), OLD_HINT_LINE)
        self.assertEqual(len(graph_shortcut_items()), len(GRAPH_SHORTCUTS))
        w = self.open_window()
        w.toggle_graph_shortcuts()
        overlay = w.shortcut_overlay
        self.assertTrue(overlay.isVisible())
        self.assertEqual(overlay.text(), OLD_HINT_LINE)
        for item in OLD_HINT_LINE.split("  ·  "):
            self.assertIn(item, overlay.text())
        overlay.close()

    def test_the_question_mark_key_and_the_corner_button_open_and_close_it(self):
        w = self.open_window()
        w.graph.setFocus()
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Question, Qt.KeyboardModifier.ShiftModifier, "?")
        w.graph.keyPressEvent(event)
        self.assertTrue(w.shortcut_overlay.isVisible())
        w.shortcut_overlay.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                                   Qt.KeyboardModifier.NoModifier))
        self.assertFalse(w.shortcut_overlay.isVisible())
        button = w.graph.corner.help.button
        self.assertTrue(button.isVisible())
        self.assertEqual(button.text(), "?")
        button.click()
        self.assertTrue(w.shortcut_overlay.isVisible())
        button.click()
        self.assertFalse(w.shortcut_overlay.isVisible())

    def test_the_overlay_opens_beside_the_graph_corner(self):
        w = self.open_window()
        w.toggle_graph_shortcuts()
        overlay = w.shortcut_overlay
        anchor = w.graph.corner.help
        corner_top = anchor.mapToGlobal(anchor.rect().topLeft()).y()
        self.assertLessEqual(overlay.geometry().bottom(), corner_top)
        overlay.close()


class DockTitleTests(Base):
    def test_title_bars_are_slim_and_the_docks_still_float_close_and_move(self):
        w = self.open_window()
        docks = (w.viewer_dock, w.graph_dock)
        for dock in docks:
            self.assertIsInstance(dock, QDockWidget)
            features = dock.features()
            for feature in (QDockWidget.DockWidgetFeature.DockWidgetClosable,
                            QDockWidget.DockWidgetFeature.DockWidgetMovable,
                            QDockWidget.DockWidgetFeature.DockWidgetFloatable):
                self.assertTrue(features & feature)
            title_height = dock.widget().geometry().top()
            self.assertLessEqual(title_height, dock.fontMetrics().height() + 14,
                                 f"{dock.windowTitle()} title bar is {title_height}px")
        self.assertLessEqual(w.viewer_dock.widget().geometry().top(), 30)

    def test_a_dock_floats_and_comes_back(self):
        w = self.open_window()
        dock = w.graph_dock
        dock.setFloating(True)
        APP.processEvents()
        self.assertTrue(dock.isFloating())
        dock.setFloating(False)
        APP.processEvents()
        self.assertFalse(dock.isFloating())


if __name__ == "__main__":
    unittest.main()
