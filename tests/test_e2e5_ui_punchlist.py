"""End-to-end sprint stage 5: the new look's punch list from the 10/10 real-display QA
(scratch/nb-qa/1010-e2e/FINDINGS.md), and real layouts for the Simulate and Render tabs."""
import unittest
from pathlib import Path

from PySide6.QtCore import QRect, QSettings, Qt
from PySide6.QtWidgets import QApplication, QWidget

from nodebased.app import Preferences
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


class ViewerLayersTests(Base):
    """Finding 2: on the Simulate tab the 3D viewport's help line painted under the viewer's floating
    strip and its bottom note over the 2D corner readouts."""

    def rects(self, w):
        stack = w.view_stack
        strip = w.viewer_strip
        corners = [c for c in (w.viewer_corners.left, w.viewer_corners.right) if c.isVisible()]
        return stack, strip, corners

    def test_the_two_sets_of_widgets_never_intersect_on_any_tab(self):
        w = self.open_window((1600, 960))
        for tab in ("Composite", "3D", "Simulate", "Render", "Composite", "Simulate"):
            w.select_workspace_tab(tab)
            settle()
            stack, strip, corners = self.rects(w)
            shown = [strip.geometry()] + [c.geometry() for c in corners]
            for i, a in enumerate(shown):
                for b in shown[i + 1:]:
                    self.assertFalse(a.intersects(b), tab)
            # the strip is above the page the stack shows, so the viewport never paints over it
            order = stack.children()
            self.assertGreater(order.index(strip), order.index(w.viewport), tab)
            self.assertGreater(order.index(strip), order.index(w.viewer), tab)
            if w.viewer_mode() == "3d":
                self.assertEqual(corners, [], f"{tab}: 2D readouts shown over the 3D viewport")
                # the viewport's top lines start below the strip
                self.assertGreaterEqual(w.viewport.top_inset, strip.geometry().bottom(), tab)
            else:
                self.assertEqual(len(corners), 2, tab)


SCENE = Path(__file__).resolve().parents[1] / "examples" / "mixed_scene" / "mixed_scene.nbcomp"


def docked(w, dock):
    return dock.isVisible() and not dock.isFloating() and not w.tabifiedDockWidgets(dock)


class TabLayoutTests(Base):
    """Finding 3: Simulate and Render have layouts of their own, and Composite comes back exactly."""

    def open_scene(self):
        store = QSettings("NodeBased", "NodeBased")
        store.remove("tab_layouts")
        w = self.open_window((1920, 1080))
        w.command({"op": "load", "path": str(SCENE)})
        settle()
        return w

    def geometry_of(self, w):
        # a hidden dock's rectangle is meaningless; what matters is what is shown and where
        return {d.objectName(): (d.isVisible(), d.isFloating(),
                                 d.geometry().getRect() if d.isVisible() else None)
                for d in w.workspace_docks}

    def test_simulate_layout(self):
        w = self.open_scene()
        w.select_workspace_tab("Simulate")
        settle()
        viewer, cache, slice_ = w.viewer_dock, w.cache_inspector_dock, w.slice_dock
        for dock in (viewer, cache, slice_, w.properties_dock):
            self.assertTrue(docked(w, dock) or dock is w.properties_dock, dock.objectName())
        self.assertEqual(w.viewer_mode(), "3d")
        self.assertFalse(w.graph_dock.isVisible())
        # viewport (with the timeline) on top; the cache inspector and slice viewer side by side under it
        self.assertLess(viewer.geometry().bottom(), cache.geometry().top())
        self.assertEqual(cache.geometry().top(), slice_.geometry().top())
        self.assertLess(cache.geometry().right(), slice_.geometry().left())
        self.assertGreater(viewer.height(), cache.height())
        # Properties at its normal width, not widened by tabbed docks
        self.assertLessEqual(w.properties_dock.width(), 520)
        self.assertFalse(w.tabifiedDockWidgets(w.properties_dock) and w.slice_dock in w.tabifiedDockWidgets(w.properties_dock))

    def test_render_layout(self):
        w = self.open_scene()
        w.select_workspace_tab("Render")
        settle()
        self.assertEqual(w.viewer_mode(), "2d")
        self.assertFalse(w.graph_dock.isVisible())
        queue, preview = w.queue_dock, w.output_preview_dock
        self.assertTrue(docked(w, queue) and docked(w, preview))
        self.assertLess(w.viewer_dock.geometry().bottom(), queue.geometry().top())
        self.assertEqual(queue.geometry().top(), preview.geometry().top())
        self.assertLess(queue.geometry().right(), preview.geometry().left())
        # the viewer shows a Write node, whose Properties are on the right
        document = w.dispatcher.document
        self.assertEqual(document["nodes"][document["view"]]["type"], "Write")
        self.assertEqual(w.graph.selected_id(), document["view"])
        self.assertTrue(w.properties_dock.isVisible())
        self.assertGreater(w.properties_dock.geometry().left(), w.viewer_dock.geometry().right() - 1)
        self.assertIn("exr", w.output_preview.details.text())
        self.assertLessEqual(w.properties_dock.width(), 520)

    def test_composite_comes_back_exactly(self):
        w = self.open_scene()
        w.select_workspace_tab("Composite")
        settle()
        before = self.geometry_of(w)
        for tab in ("Simulate", "Render", "3D", "Simulate", "Composite"):
            w.select_workspace_tab(tab)
            settle()
        self.assertEqual(self.geometry_of(w), before)
        w.select_workspace_tab("Render")
        settle()
        w.select_workspace_tab("Composite")
        settle()
        self.assertEqual(self.geometry_of(w), before)

    def test_each_tab_layout_is_saved_in_settings_and_restored_by_a_new_window(self):
        w = self.open_scene()
        w.select_workspace_tab("Simulate")
        settle()
        w.select_workspace_tab("Composite")
        settle()
        store = QSettings("NodeBased", "NodeBased")
        self.assertFalse(store.value("tab_layouts/Simulate") is None or store.value("tab_layouts/Composite") is None)
        w.select_workspace_tab("Render")
        settle()
        w.resizeDocks([w.queue_dock, w.output_preview_dock], [900, 300], Qt.Orientation.Horizontal)
        settle()
        wide_queue = w.queue_dock.width()
        w.select_workspace_tab("Composite")
        settle()
        w.select_workspace_tab("Render")
        settle()
        self.assertEqual(w.queue_dock.width(), wide_queue)
        # a window opened later finds the saved Render layout in the settings
        w.select_workspace_tab("Composite")
        settle()
        other = self.__class__.open_window(self, (1920, 1080))
        other.select_workspace_tab("Render")
        settle()
        self.assertTrue(docked(other, other.queue_dock) and docked(other, other.output_preview_dock))
        self.assertGreater(other.queue_dock.width(), other.output_preview_dock.width())

    def test_the_saved_workspace_is_always_the_composite_arrangement(self):
        w = self.open_scene()
        w.select_workspace_tab("Composite")
        settle()
        composite = w.saveState(Preferences.WORKSPACE_VERSION)
        w.select_workspace_tab("Simulate")
        settle()
        self.assertEqual(w._workspace_state_for_save(), composite)

    def test_viewer_buttons_follow_the_layout(self):
        w = self.open_scene()
        w.select_workspace_tab("Simulate")
        settle()
        w.set_viewer_mode("2d")
        settle()
        self.assertEqual(w.workspace_tab, "Composite")
        self.assertTrue(w.graph_dock.isVisible())
        self.assertFalse(w.cache_inspector_dock.isVisible())


if __name__ == "__main__":
    unittest.main()
