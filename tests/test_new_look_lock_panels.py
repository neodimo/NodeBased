"""New look, final pass: docked panels have no title bars by default (the approved mockup has none).
Workspace → Lock panels brings them back for rearranging; floating panels always keep theirs."""
import unittest

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from tests.test_new_look_viewer_strip_nl5 import Base

APP = QApplication.instance() or QApplication([])


class LockPanelsTests(Base):
    def test_docked_panels_have_no_title_bar_by_default(self):
        w = self.open_window()
        self.assertTrue(w.lock_panels_action.isChecked())
        for dock in (w.viewer_dock, w.graph_dock, w.properties_dock):
            self.assertEqual(dock.widget().geometry().top(), 0, dock.windowTitle())

    def test_unlocking_brings_the_title_bars_back(self):
        w = self.open_window()
        w.lock_panels_action.setChecked(False)
        APP.processEvents()
        self.assertFalse(w.preferences.lock_panels())
        for dock in (w.viewer_dock, w.graph_dock, w.properties_dock):
            self.assertIsNone(dock.titleBarWidget(), dock.windowTitle())
            self.assertGreater(dock.widget().geometry().top(), 0, dock.windowTitle())
        w.lock_panels_action.setChecked(True)
        APP.processEvents()
        self.assertTrue(w.preferences.lock_panels())
        self.assertEqual(w.graph_dock.widget().geometry().top(), 0)

    def test_a_floating_panel_keeps_its_title_bar_while_locked(self):
        w = self.open_window()
        dock = w.graph_dock
        dock.setFloating(True)
        APP.processEvents()
        self.assertIsNone(dock.titleBarWidget())
        dock.setFloating(False)
        APP.processEvents()
        self.assertIsNotNone(dock.titleBarWidget())
        self.assertEqual(dock.widget().geometry().top(), 0)


if __name__ == "__main__":
    unittest.main()
