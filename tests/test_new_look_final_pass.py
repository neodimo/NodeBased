"""New look, final pass (Gonzo 10/10). Docked panels have no title bars by default (the approved mockup has
none); Workspace → Lock panels brings them back for rearranging, and floating panels always keep theirs.
A colour's four fields and an xyz row sit on one line in the default Properties width, sized for the
number they show rather than the widest number their range allows."""
import unittest

from pathlib import Path

from PySide6.QtWidgets import QApplication

from nodebased.knobfit import WrappingRow

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



class OneLineKnobRowTests(Base):
    def open_cube(self):
        w = self.open_window((1920, 1080))
        scene = Path(__file__).resolve().parents[1] / "examples" / "mixed_scene" / "mixed_scene.nbcomp"
        w.command({"op": "load", "path": str(scene)})
        APP.processEvents()
        cube = next(k for k, n in w.dispatcher.document["nodes"].items() if n["type"] == "Cube3D")
        w.graph.items_by_id[cube].setSelected(True)
        for _ in range(5):
            APP.processEvents()
        return w

    def test_colour_and_xyz_rows_fit_on_one_line(self):
        w = self.open_cube()
        rows = w.properties.findChildren(WrappingRow)
        colour = next(r for r in rows if any(u.objectName() == "red-field" for u in r.units))
        self.assertEqual(colour._columns, len(colour.units))
        for unit in colour.units[1:]:
            self.assertGreaterEqual(unit.width(),
                                    unit.fontMetrics().horizontalAdvance(unit.text()) + 6, unit.objectName())
        xyz = [r for r in rows if len(r.units) == 3 and not any(u.objectName().endswith("-field") for u in r.units)]
        self.assertTrue(xyz)
        for row in xyz:
            self.assertEqual(row._columns, 3)


if __name__ == "__main__":
    unittest.main()
