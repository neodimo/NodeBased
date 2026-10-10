"""End-to-end sprint stage 5: the new look's punch list from the 10/10 real-display QA
(scratch/nb-qa/1010-e2e/FINDINGS.md), and real layouts for the Simulate and Render tabs."""
import unittest

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QWidget

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from tests.test_new_look_viewer_strip_nl5 import Base

APP = QApplication.instance() or QApplication([])


def settle(n=6):
    for _ in range(n):
        APP.processEvents()


class FamilyPanelGhostTests(Base):
    """Finding 1: a green block sat over the list rows when a family panel opened. It was the
    timeline's cache bar and playhead, blurred and showing through the panel's translucent
    backdrop. The panel is opaque now and nothing is drawn over its rows."""

    def test_no_family_panel_has_a_widget_or_item_over_another_row(self):
        w = self.open_window((1600, 960))
        shelf = w.node_shelf
        for key in list(shelf.rail.buttons):
            shelf.open_family(key)
            settle()
            panel = shelf.panel
            self.assertTrue(panel.isVisible(), key)
            list_view = panel.list
            rects = []
            for row in range(list_view.count()):
                rect = list_view.visualItemRect(list_view.item(row))
                if rect.intersects(list_view.viewport().rect()):
                    rects.append(rect)
            for i, a in enumerate(rects):
                for b in rects[i + 1:]:
                    self.assertFalse(a.intersects(b), f"{key}: two rows overlap")
            # No child of the panel other than the list's own viewport sits over the list.
            list_rect = QRect(list_view.mapTo(panel, list_view.rect().topLeft()), list_view.size())
            for child in panel.findChildren(QWidget):
                if (child is list_view or child is list_view.viewport() or not child.isVisible()
                        or list_view.isAncestorOf(child)):
                    continue
                if child.parent() is not panel and not panel.isAncestorOf(child):
                    continue
                rect = QRect(child.mapTo(panel, child.rect().topLeft()), child.size())
                self.assertFalse(rect.intersects(list_rect), f"{key}: {child.objectName()!r} over the list")
            shelf.close_panel()

    def test_the_panel_ignores_what_is_behind_it(self):
        w = self.open_window((1600, 960))
        shelf = w.node_shelf
        shelf.open_family("Color")
        settle()
        panel = shelf.panel
        spot = panel.list.mapTo(w, panel.list.rect().bottomLeft())
        spot.setX(spot.x() + 4)
        spot.setY(spot.y() - 2)
        before = w.grab().toImage().pixelColor(spot)
        w.viewer_dock.hide()
        w.graph_dock.hide()
        settle()
        after = w.grab().toImage().pixelColor(spot)
        self.assertEqual(before.rgb(), after.rgb())
        self.assertEqual(panel.palette().window().color().alpha(), 255)


if __name__ == "__main__":
    unittest.main()
