"""The default and the restored workspace give the viewer the room (QA 10/5, finding 1).

The 10/2 default laid the left column out at a fixed 600px beside a 400px Properties dock and left
the rest of a 1440px window to the empty central placeholder: ~440px of blank space, a thumbnail
viewer. These tests state the rules that replaced it, as ratios and relations that hold at any
window size, not as the pixels of one screenshot:

* the placeholder owns no width (the viewer takes every pixel Properties does not),
* a window that grows gives the new width to the viewer; one that shrinks keeps Properties usable,
* a layout saved by the cramped default (before WORKSPACE_LAYOUT 3) is repaired once, and only
  when it matches that default's signature; any layout the artist arranged is kept.
"""
import unittest
import unittest.mock

from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtWidgets import QApplication

from tests.waiting import wait_until

import nodebased.app as nodebased_app_module
from nodebased.app import (DEAD_CENTRAL_SLACK, DEFAULT_PROPERTIES_RANGE, LEGACY_CRAMPED_VIEWER_SHARE,
                           PROPERTIES_USABLE_WIDTH, Preferences, Window, default_properties_width)
from nodebased.nodecatalog import NODE_CATEGORIES

APP = QApplication.instance() or QApplication([])
REAL_WORKSPACE = getattr(Preferences.workspace, "real_workspace", Preferences.workspace)


class DefaultPropertiesWidthTests(unittest.TestCase):
    def test_a_share_of_the_window_inside_the_stated_range(self):
        low, high = DEFAULT_PROPERTIES_RANGE
        widths = [default_properties_width(width) for width in (800, 1280, 1440, 1920, 3840)]
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(widths[0], low)
        self.assertEqual(widths[-1], high)
        self.assertTrue(all(low <= width <= high for width in widths))
        self.assertTrue(low < default_properties_width(1440) < high)


class WorkspaceLayoutTests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").remove("workspace")
        self.patch = unittest.mock.patch.object(Preferences, "workspace", REAL_WORKSPACE)
        self.patch.start()
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
        self.windows = []
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.patch.stop()
        QSettings("NodeBased", "NodeBased").remove("workspace")

    def open_window(self, size=(1440, 920)):
        original = nodebased_app_module.DEFAULT_WINDOW_SIZE
        nodebased_app_module.DEFAULT_WINDOW_SIZE = size
        try:
            window = Window()
        finally:
            nodebased_app_module.DEFAULT_WINDOW_SIZE = original
        self.windows.append(window)
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None))
        # The post-show split and any restore repair run from a zero-delay timer.
        wait_until(lambda: not getattr(window, "_default_split_pending", False)
                   and getattr(window, "_pending_geometry", None) is None, timeout=5.0)
        APP.processEvents()
        return window

    def close_window(self, window):
        window.saved_document = window.dispatcher.document
        window.close()
        window.deleteLater()
        self.windows.remove(window)
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()

    def settle(self, window, condition=None, timeout=5.0):
        wait_until(lambda: window._central_gap() <= DEAD_CENTRAL_SLACK and (condition is None or condition()),
                   timeout=timeout)
        APP.processEvents()

    def stack_the_left_column(self, window, viewer_share):
        """The pre-10/5 topology: viewer, Node Graph and NODES stacked in one column, with the
        left column and Properties at the old fixed widths, which left the central gap."""
        window.splitDockWidget(window.viewer_dock, window.graph_dock, Qt.Orientation.Vertical)
        window.splitDockWidget(window.graph_dock, window.nodes_dock, Qt.Orientation.Vertical)
        window.layout().activate()
        APP.processEvents()
        column = sum(dock.height() for dock in (window.viewer_dock, window.graph_dock, window.nodes_dock))
        nodes = window.nodes_dock.minimumSizeHint().height()
        viewer = round(column * viewer_share)
        graph = column - viewer - nodes
        window.resizeDocks([window.viewer_dock, window.graph_dock, window.nodes_dock],
                           [viewer, graph, nodes], Qt.Orientation.Vertical)
        window.resizeDocks([window.viewer_dock, window.properties_dock], [600, 400], Qt.Orientation.Horizontal)
        window.layout().activate()
        APP.processEvents()

    def close_as_legacy(self, window):
        """Close `window` (which saves its layout) and strip the layout revision, as a build from
        before WORKSPACE_LAYOUT 3 saved it: those never wrote the key."""
        self.close_window(window)
        store = QSettings("NodeBased", "NodeBased")
        store.remove("workspace/layout")
        store.sync()
        self.assertEqual(Preferences().workspace()["layout"], 2)

    def is_stacked(self, window):
        # A hidden dock keeps the position it last had, so only the docks on screen are compared
        # (the left column of node families moved every visible dock 60px right; a hidden one
        # stayed where it was).
        docks = (window.viewer_dock, window.graph_dock, window.nodes_dock)
        return len({dock.x() for dock in docks if dock.isVisible()}) == 1

    # -- the default ------------------------------------------------------------------------

    def test_a_fresh_window_leaves_the_central_placeholder_no_width(self):
        window = self.open_window()
        self.settle(window)
        self.assertLessEqual(window._central_gap(), DEAD_CENTRAL_SLACK)
        self.assertGreater(window.viewer_dock.width(), 2 * window.properties_dock.width())
        self.assertGreaterEqual(window.properties_dock.width(), PROPERTIES_USABLE_WIDTH)

    def test_graph_fills_the_row_below_viewer_and_node_families_live_in_the_left_column(self):
        window = self.open_window()
        self.assertFalse(window.nodes_dock.isVisible())
        self.assertGreater(window.graph_dock.y(), window.viewer_dock.y())
        self.assertEqual(window.graph_dock.width(), window.viewer_dock.width())
        # Favourites, Recent, the thirteen families and Other, then Settings
        self.assertEqual(len(window.node_rail.buttons), 17)
        self.assertEqual(window.toolBarArea(window.node_rail_toolbar), Qt.ToolBarArea.LeftToolBarArea)
        self.assertLessEqual(window.node_rail.geometry().right(), window.viewer_dock.geometry().left())

    def test_category_icon_reveals_its_nodes_and_creates_the_selected_kind(self):
        window = self.open_window()
        for category, kinds in NODE_CATEGORIES.items():
            window.node_shelf.open_family(category)
            self.assertCountEqual(window.node_panel.visible_kinds(), list(kinds))
        window.node_shelf.open_family("Color")
        panel = window.node_panel
        before = set(window.dispatcher.document["nodes"])
        item = next(panel.list.item(i) for i in range(panel.list.count()) if panel.list.item(i).text() == "Grade")
        panel.list.setCurrentItem(item)
        panel.add_current()
        added = set(window.dispatcher.document["nodes"]) - before
        self.assertEqual(len(added), 1)
        self.assertEqual(window.dispatcher.document["nodes"][added.pop()]["type"], "Grade")

    def test_search_opens_the_full_browser_without_restoring_a_permanent_nodes_panel(self):
        window = self.open_window()
        self.assertFalse(window.nodes_dock.isVisible())
        window.focus_node_search()
        self.assertTrue(window.nodes_dock.isVisible())

    # -- the window changing size ----------------------------------------------------------

    def test_a_window_that_grows_gives_the_new_width_to_the_viewer(self):
        window = self.open_window((1280, 720))
        self.settle(window)
        before = window.viewer_dock.width()
        window.resize(1800, 900)
        self.assertTrue(wait_until(lambda: window.viewer_dock.width() > before + 300))
        self.assertLessEqual(window._central_gap(), DEAD_CENTRAL_SLACK)
        self.assertGreater(window.viewer_dock.width(), before + 400)

    def test_a_window_that_shrinks_keeps_properties_usable(self):
        window = self.open_window((1920, 1080))
        self.settle(window)
        window.resize(1100, 800)
        self.settle(window, lambda: window.properties_dock.width() >= PROPERTIES_USABLE_WIDTH)
        self.assertGreaterEqual(window.properties_dock.width(), PROPERTIES_USABLE_WIDTH)
        self.assertLessEqual(window._central_gap(), DEAD_CENTRAL_SLACK)

    # -- saved layouts ---------------------------------------------------------------------

    def test_the_old_workspace_version_is_replaced_with_the_new_shelf_layout(self):
        first = self.open_window()
        first.save_workspace()
        self.close_window(first)
        store = QSettings("NodeBased", "NodeBased")
        store.setValue("workspace/version", Preferences.WORKSPACE_VERSION - 1)
        store.sync()

        second = self.open_window()
        self.assertFalse(second.nodes_dock.isVisible())
        self.assertLessEqual(second._central_gap(), DEAD_CENTRAL_SLACK)

    def test_a_tall_viewer_keeps_its_arrangement_under_the_repair_rule(self):
        # The rule itself, on a live window: the end-to-end path reshuffles the saved heights
        # while Qt restores them, so the share a test sets is not the share the repair reads.
        window = self.open_window()
        self.stack_the_left_column(window, viewer_share=0.6)
        self.assertGreaterEqual(window.viewer_dock.height(), LEGACY_CRAMPED_VIEWER_SHARE * (
            window.viewer_dock.height() + window.graph_dock.height() + window.nodes_dock.height()))
        self.assertFalse(window._repair_legacy_layout())
        self.assertTrue(self.is_stacked(window))

    def test_the_repair_rule_leaves_a_floating_dock_alone(self):
        window = self.open_window()
        self.stack_the_left_column(window, viewer_share=0.3)
        window.graph_dock.setFloating(True)
        APP.processEvents()
        self.assertFalse(window._repair_legacy_layout())
        self.assertTrue(window.graph_dock.isFloating())

    def test_a_layout_saved_by_this_build_is_never_rebuilt(self):
        # The same short stacked viewer, saved with the current layout revision: the artist
        # chose it, so only the dead width (which nobody can choose) is reclaimed.
        first = self.open_window()
        self.stack_the_left_column(first, viewer_share=0.3)
        first.save_workspace()
        self.assertEqual(Preferences().workspace()["layout"], Preferences.WORKSPACE_LAYOUT)
        self.close_window(first)

        second = self.open_window()
        self.settle(second)
        self.assertTrue(self.is_stacked(second))
        self.assertLessEqual(second._central_gap(), DEAD_CENTRAL_SLACK)

    def test_the_layout_revision_is_saved_and_defaults_to_the_old_one(self):
        window = self.open_window()
        window.save_workspace()
        self.assertEqual(Preferences().workspace()["layout"], Preferences.WORKSPACE_LAYOUT)
        QSettings("NodeBased", "NodeBased").remove("workspace/layout")
        self.assertLess(Preferences().workspace()["layout"], Preferences.WORKSPACE_LAYOUT)


if __name__ == "__main__":
    unittest.main()
