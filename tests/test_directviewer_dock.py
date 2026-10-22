"""The direct picture as a real dock: in the window's saved layout, in the Viewer's place with one switch, and around it
the Viewer's strip and timeline while it covers the Viewer.

The card itself is not started here (these run on offscreen Qt, which has no native window): `set_direct_viewer` is
replaced by a stand-in that only shows and hides the dock, so what is tested is the layout, the switch and the controls.
`tests.isolation` keeps QSettings away from the real user file."""
import os
import unittest
import unittest.mock

import tests.isolation  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QSettings
from PySide6.QtWidgets import QApplication, QWidget

from nodebased.app import STYLE, Preferences, Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)
REAL_WORKSPACE = getattr(Preferences.workspace, "real_workspace", Preferences.workspace)


class StubWidget(QWidget):
    """What stands in for the card's widget: the calls the Window makes on it."""

    def fit(self):
        pass

    def release(self):
        pass


def front(window, dock):
    """Whether `dock` is the tab showing in its place: its tab is the current one in its tab bar (a dock behind another
    tab still reports itself visible), or it is on its own and shown."""
    from PySide6.QtWidgets import QTabBar
    if not dock.isVisible():
        return False
    title = dock.windowTitle()
    for bar in window.findChildren(QTabBar):
        if not bar.isVisible() or not window.rect().contains(bar.geometry().center()):
            continue                        # Qt keeps the bars of groups that no longer exist, parked out of sight
        titles = [bar.tabText(i) for i in range(bar.count())]
        if title in titles:
            return bar.tabText(bar.currentIndex()) == title
    return True


def stand_in(self, on):
    """What `Window.set_direct_viewer` does to the dock, without a card behind it."""
    if on:
        if self.direct_widget is None:
            self.direct_widget = StubWidget()
            self.direct_panel.set_picture(self.direct_widget)
        if self.direct_dock.isHidden() and not self.direct_dock.isFloating() and not self._direct_placed:
            self._place_direct_in_viewer_place()
        self._direct_placed = True
        self._bring_forward(self.direct_dock)
        self._sync_direct_controls()
    else:
        if not self._direct_release_only:
            self.direct_dock.hide()
        self._sync_direct_controls()


class DockCase(unittest.TestCase):
    def setUp(self):
        # Saving and restoring the layout are the real ones here (tests.isolation stubs the restore for every other
        # test), against the private settings file.
        QSettings("NodeBased", "NodeBased").remove("workspace")
        QSettings("NodeBased", "NodeBased").remove(Preferences.DIRECT_MAIN)
        for patch in (unittest.mock.patch.object(Preferences, "workspace", REAL_WORKSPACE),
                      unittest.mock.patch.object(Window, "set_direct_viewer", stand_in)):
            patch.start()
            self.addCleanup(patch.stop)
        self.windows = []
        self.addCleanup(self.close_all)

    def close_all(self):
        for window in self.windows:
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
        self.windows = []
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        QSettings("NodeBased", "NodeBased").remove("workspace")
        QSettings("NodeBased", "NodeBased").remove(Preferences.DIRECT_MAIN)

    def window(self):
        window = Window()
        self.windows.append(window)
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None))
        wait_until(lambda: not getattr(window, "_default_split_pending", False)
                   and getattr(window, "_pending_geometry", None) is None, timeout=5.0)
        self.settle()
        return window

    def settle(self):
        for _ in range(6):
            APP.processEvents()

    def reopen(self, window):
        """Close the window (which saves its layout) and open a new one, the way a quit and a launch do."""
        window.saved_document = window.dispatcher.document
        window.close()
        window.deleteLater()
        self.windows.remove(window)
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        return self.window()


class DirectDockLayoutTests(DockCase):
    def test_the_dock_exists_hidden_and_is_part_of_the_workspace(self):
        window = self.window()
        self.assertEqual(window.direct_dock.objectName(), "direct-viewer-dock")
        self.assertIn(window.direct_dock, window.workspace_docks)
        self.assertFalse(front(window, window.direct_dock))
        self.assertTrue(front(window, window.viewer_dock))
        self.assertFalse(window.preferences.direct_main())      # off by default

    def test_opening_it_puts_a_tab_in_front_of_the_viewer(self):
        window = self.window()
        window.direct_action.setChecked(True)
        self.settle()
        self.assertTrue(front(window, window.direct_dock))
        self.assertFalse(window.direct_dock.isFloating())
        self.assertIn(window.direct_dock, window.tabifiedDockWidgets(window.viewer_dock))
        self.assertFalse(front(window, window.viewer_dock))

    def test_a_tab_in_front_comes_back_in_front_after_a_restart(self):
        window = self.window()
        window.direct_action.setChecked(True)
        self.settle()
        fresh = self.reopen(window)
        self.assertTrue(front(fresh, fresh.direct_dock))
        self.assertFalse(fresh.direct_dock.isFloating())
        self.assertEqual(fresh.dockWidgetArea(fresh.direct_dock), fresh.dockWidgetArea(fresh.viewer_dock))
        self.assertFalse(front(fresh, fresh.viewer_dock))

    def test_a_dock_left_closed_is_still_closed_after_a_restart(self):
        window = self.window()
        window.direct_action.setChecked(True)
        window.direct_action.setChecked(False)
        self.settle()
        fresh = self.reopen(window)
        self.assertFalse(front(fresh, fresh.direct_dock))
        self.assertTrue(front(fresh, fresh.viewer_dock))

    def test_a_dock_beside_the_viewer_stays_beside_it(self):
        from PySide6.QtCore import Qt
        window = self.window()
        window.direct_action.setChecked(True)
        window.splitDockWidget(window.viewer_dock, window.direct_dock, Qt.Orientation.Horizontal)
        window.viewer_dock.show()
        window.direct_dock.show()
        self.settle()
        self.assertTrue(front(window, window.viewer_dock) and front(window, window.direct_dock))
        self.assertNotIn(window.direct_dock, window.tabifiedDockWidgets(window.viewer_dock))
        fresh = self.reopen(window)
        self.assertTrue(front(fresh, fresh.viewer_dock) and front(fresh, fresh.direct_dock))
        self.assertNotIn(fresh.direct_dock, fresh.tabifiedDockWidgets(fresh.viewer_dock))
        self.assertLess(fresh.viewer_dock.geometry().left(), fresh.direct_dock.geometry().left())

    def test_a_floating_dock_stays_floating(self):
        window = self.window()
        window.direct_action.setChecked(True)
        window.direct_dock.setFloating(True)
        window.direct_dock.resize(700, 400)
        window.direct_dock.move(60, 80)
        self.settle()
        self.assertTrue(window.direct_dock.isFloating())
        fresh = self.reopen(window)
        self.assertTrue(front(fresh, fresh.direct_dock))
        self.assertTrue(fresh.direct_dock.isFloating())
        self.assertTrue(front(fresh, fresh.viewer_dock))

    def test_the_viewer_dock_and_the_others_keep_their_places(self):
        window = self.window()
        others = lambda w: {dock.objectName(): front(w, dock) for dock in w.workspace_docks
                            if dock not in (w.direct_dock, w.viewer_dock)}
        before = others(window)
        window.direct_action.setChecked(True)
        self.settle()
        fresh = self.reopen(window)
        self.assertEqual(before, others(fresh))
        self.assertTrue(front(fresh, fresh.direct_dock))
        self.assertFalse(front(fresh, fresh.viewer_dock))


class DirectMainSwitchTests(DockCase):
    def test_the_switch_puts_the_direct_picture_in_the_viewers_place_and_back(self):
        window = self.window()
        self.assertFalse(window.direct_main_action.isChecked())
        window.direct_main_action.setChecked(True)
        self.settle()
        self.assertTrue(window.preferences.direct_main())
        self.assertTrue(front(window, window.direct_dock))
        self.assertIn(window.direct_dock, window.tabifiedDockWidgets(window.viewer_dock))
        self.assertFalse(front(window, window.viewer_dock))
        self.assertTrue(window.direct_action.isChecked())
        window.direct_main_action.setChecked(False)
        self.settle()
        self.assertFalse(window.preferences.direct_main())
        self.assertTrue(front(window, window.viewer_dock))
        self.assertFalse(front(window, window.direct_dock))
        self.assertFalse(window.direct_action.isChecked())

    def test_the_switch_round_trips_through_a_saved_layout_and_a_restart(self):
        window = self.window()
        window.direct_main_action.setChecked(True)
        self.settle()
        fresh = self.reopen(window)
        self.assertTrue(fresh.preferences.direct_main())
        self.assertTrue(fresh.direct_main_action.isChecked())
        self.assertTrue(wait_until(lambda: front(fresh, fresh.direct_dock) and not front(fresh, fresh.viewer_dock)))
        self.assertEqual(fresh.dockWidgetArea(fresh.direct_dock), fresh.dockWidgetArea(fresh.viewer_dock))
        fresh.direct_main_action.setChecked(False)
        self.settle()
        again = self.reopen(fresh)
        self.assertFalse(again.preferences.direct_main())
        self.assertTrue(front(again, again.viewer_dock))
        self.assertFalse(front(again, again.direct_dock))

    def test_the_switch_starts_a_launch_with_the_direct_picture_even_without_a_saved_layout(self):
        Preferences().set_direct_main(True)
        window = self.window()
        self.assertTrue(wait_until(lambda: front(window, window.direct_dock) and not front(window, window.viewer_dock)))

    def test_a_docked_picture_that_was_floated_comes_back_into_the_viewers_place(self):
        window = self.window()
        window.direct_action.setChecked(True)
        window.direct_dock.setFloating(True)
        self.settle()
        window.direct_main_action.setChecked(True)
        self.settle()
        self.assertFalse(window.direct_dock.isFloating())
        self.assertIn(window.direct_dock, window.tabifiedDockWidgets(window.viewer_dock))

    def test_the_3d_view_brings_the_viewer_forward_and_2d_takes_the_picture_back(self):
        window = self.window()
        window.direct_main_action.setChecked(True)
        self.settle()
        window.set_viewer_mode("3d")
        self.settle()
        self.assertTrue(front(window, window.viewer_dock))
        self.assertFalse(front(window, window.direct_dock))
        window.set_viewer_mode("2d")
        self.settle()
        self.assertTrue(front(window, window.direct_dock))
        self.assertFalse(front(window, window.viewer_dock))

    def test_the_default_is_unchanged(self):
        window = self.window()
        self.assertFalse(Preferences().direct_main())
        self.assertFalse(front(window, window.direct_dock))
        self.assertTrue(front(window, window.viewer_dock))


class DirectControlsTests(DockCase):
    """The Viewer's strip and timeline are laid out around the picture while it covers the Viewer."""

    def under(self, widget, parent):
        while widget is not None:
            if widget is parent:
                return True
            widget = widget.parentWidget()
        return False

    def test_the_strip_and_the_timeline_move_to_the_picture_while_it_covers_the_viewer(self):
        window = self.window()
        self.assertTrue(self.under(window.timeline_panel, window.viewer_panel))
        self.assertTrue(self.under(window.viewer_strip, window.view_stack))
        window.direct_action.setChecked(True)
        self.settle()
        self.assertTrue(self.under(window.timeline_panel, window.direct_panel))
        self.assertTrue(self.under(window.viewer_strip, window.direct_panel))
        self.assertTrue(window.viewer_strip.docked)
        self.assertTrue(window.timeline_panel.isVisibleTo(window.direct_panel))
        window.direct_action.setChecked(False)
        self.settle()
        self.assertTrue(self.under(window.timeline_panel, window.viewer_panel))
        self.assertTrue(self.under(window.viewer_strip, window.view_stack))
        self.assertFalse(window.viewer_strip.docked)
        self.assertTrue(window.timeline_panel.isVisibleTo(window.viewer_panel))

    def test_beside_the_viewer_the_controls_stay_with_the_viewer(self):
        from PySide6.QtCore import Qt
        window = self.window()
        window.direct_action.setChecked(True)
        window.splitDockWidget(window.viewer_dock, window.direct_dock, Qt.Orientation.Horizontal)
        window.viewer_dock.show()
        window.direct_dock.show()
        self.settle()
        self.assertTrue(self.under(window.timeline_panel, window.viewer_panel))
        self.assertTrue(self.under(window.viewer_strip, window.view_stack))

    def test_clicking_the_viewers_tab_gives_the_controls_back(self):
        window = self.window()
        window.direct_action.setChecked(True)
        self.settle()
        window.viewer_dock.raise_()
        self.settle()
        self.assertTrue(front(window, window.viewer_dock))
        self.assertTrue(self.under(window.timeline_panel, window.viewer_panel))
        window.direct_dock.raise_()
        self.settle()
        self.assertTrue(self.under(window.timeline_panel, window.direct_panel))

    def test_the_timeline_still_scrubs_from_the_direct_panel(self):
        window = self.window()
        window.direct_action.setChecked(True)
        self.settle()
        before = window.dispatcher.document["time"]["current"]
        window.step_frame(1)
        self.settle()
        self.assertEqual(window.frame_current.value(), window.dispatcher.document["time"]["current"])
        self.assertNotEqual(before, window.dispatcher.document["time"]["current"])
        self.assertTrue(window.frame_slider.isVisibleTo(window.direct_panel))


if __name__ == "__main__":
    unittest.main()
