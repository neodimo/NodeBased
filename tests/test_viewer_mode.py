"""The 2D viewer and the 3D viewport share one VIEWER panel; Tab over it switches (Lane 2, V2)."""
import base64
import unittest
import unittest.mock

from nodebased.app import PROPERTIES_USABLE_WIDTH, VIEWER_3D_MIN_HEIGHT, VIEWER_3D_MIN_WIDTH
from tests.test_desktop import (APP, Preferences, ProjectSettingsDialog, QSettings, Window,
                                release_window, wait_until, _REAL_WORKSPACE)
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QCursor, QKeyEvent

# Window.saveState(2) bytes from the build before the 3D VIEWPORT dock left the default layout,
# captured with that dock open ("viewport3d-dock" is named inside it).
OLD_STATE_WITH_3D_DOCK = base64.b64decode(
    "AAAA/wAAAAL9AAAAAgAAAAAAAAJYAAADPPwCAAAAA/sAAAAWAHYAaQBlAHcAZQByAC0AZABvAGMAawEAAAA+AAABXAAAAJ0A////+wAAAB4AbgBvAGQAZQAtAGcAcgBhAHAAaAAtAGQAbwBjAGsBAAABoAAAALYAAACYAP////wAAAJcAAABHgAAAP8A////+gAAAAECAAAAAvsAAAAUAG4AbwBkAGUAcwAtAGQAbwBjAGsAAAAAAP////8AAADeAP////sAAAAeAHYAaQBlAHcAcABvAHIAdAAzAGQALQBkAG8AYwBrAQAAAAD/////AAAA/wD///8AAAABAAABkAAAAzz8AgAAAAH8AAAAPgAAAzwAAABcAP////oAAAAAAgAAAAX7AAAAHgBwAHIAbwBwAGUAcgB0AGkAZQBzAC0AZABvAGMAawEAAAAA/////wAAAFwA////+wAAACIAcwBsAGkAYwBlAC0AdgBpAGUAdwBlAHIALQBkAG8AYwBrAAAAAAD/////AAAApwD////7AAAAKABjAGEAYwBoAGUALQBpAG4AcwBwAGUAYwB0AG8AcgAtAGQAbwBjAGsAAAAAAP////8AAACWAP////sAAAAUAGEAZwBlAG4AdAAtAGQAbwBjAGsAAAAAAP////8AAADYAP////sAAAAiAGMAdQByAHYAZQAtAGUAZABpAHQAbwByAC0AZABvAGMAawAAAAAA/////wAAADQA////AAABrAAAAzwAAAAEAAAABAAAAAgAAAAI/AAAAAEAAAACAAAAAQAAACIAdwBvAHIAawBzAHAAYQBjAGUALQB0AG8AbwBsAGIAYQByAQAAAAD/////AAAAAAAAAAA=")


def tab_press():
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier)


class ViewerModeTests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").remove(Preferences.VIEWER_AUTO_3D)
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        QSettings("NodeBased", "NodeBased").remove(Preferences.VIEWER_AUTO_3D)
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def _point_at(self, widget):
        QCursor.setPos(widget.mapToGlobal(widget.rect().center()))

    def _add_render3d(self):
        w = self.window
        w.command({"op": "batch", "commands": [
            {"op": "create", "id": "scene", "type": "Scene3D", "pos": [0, 300]},
            {"op": "create", "id": "cam", "type": "Camera3D", "pos": [100, 300]},
            {"op": "create", "id": "render", "type": "Render3D", "pos": [50, 400]},
            {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "render", "input": "camera", "source": "cam"}]})

    def test_the_two_views_share_one_panel_that_starts_on_2d(self):
        w = self.window
        self.assertIs(w.viewer.parentWidget(), w.view_stack)
        self.assertIs(w.viewport.parentWidget(), w.view_stack)
        self.assertEqual(w.viewer_mode(), "2d")
        self.assertTrue(w.viewer.isVisible())
        self.assertFalse(w.viewport.isVisible())
        self.assertTrue(w.viewer_mode_buttons["2d"].isChecked())
        self.assertFalse(w.viewer_mode_buttons["3d"].isChecked())

    def test_tab_over_the_viewer_switches_between_2d_and_3d(self):
        w = self.window
        self._point_at(w.view_stack)
        with unittest.mock.patch.object(w, "node_search") as search:
            self.assertTrue(w.eventFilter(w.properties, tab_press()))
            self.assertEqual(w.viewer_mode(), "3d")
            self.assertTrue(w.viewport.isVisible())
            self.assertFalse(w.viewer.isVisible())
            self.assertTrue(w.viewer_mode_buttons["3d"].isChecked())
            self.assertTrue(w.eventFilter(w.properties, tab_press()))
            self.assertEqual(w.viewer_mode(), "2d")
            self.assertTrue(w.viewer_mode_buttons["2d"].isChecked())
            search.assert_not_called()

    def test_tab_over_the_graph_still_opens_node_search(self):
        w = self.window
        self._point_at(w.graph.viewport())
        with unittest.mock.patch.object(w, "node_search") as search:
            self.assertTrue(w.eventFilter(w.properties, tab_press()))
            search.assert_called_once()
        self.assertEqual(w.viewer_mode(), "2d", "Tab over the graph must not switch the viewer")

    def test_tab_elsewhere_is_left_alone(self):
        w = self.window
        self._point_at(w.properties)
        self.assertFalse(w.eventFilter(w.properties, tab_press()))
        self.assertEqual(w.viewer_mode(), "2d")

    def test_the_3d_viewport_toolbar_button_switches_the_shared_panel(self):
        w = self.window
        action = w.topbar_actions["3D viewport"]
        action.trigger()
        APP.processEvents()
        self.assertEqual(w.viewer_mode(), "3d")
        self.assertTrue(w.viewport.isVisible())
        self.assertTrue(w.viewer_dock.isVisible())
        action.trigger()
        self.assertEqual(w.viewer_mode(), "3d", "the button switches to 3D; it does not hide it again")

    def test_the_3d_button_brings_back_a_closed_viewer_dock(self):
        w = self.window
        w.viewer_dock.hide()
        APP.processEvents()
        w.show_viewport_3d()
        APP.processEvents()
        self.assertTrue(w.viewer_dock.isVisible())
        self.assertTrue(w.viewport.isVisible())

    def test_the_toolbar_toggle_switches_and_shows_the_active_view(self):
        w = self.window
        w.viewer_mode_buttons["3d"].click()
        self.assertEqual(w.viewer_mode(), "3d")
        self.assertTrue(w.viewer_mode_buttons["3d"].isChecked())
        self.assertFalse(w.viewer_mode_buttons["2d"].isChecked())
        w.viewer_mode_buttons["2d"].click()
        self.assertEqual(w.viewer_mode(), "2d")
        self.assertTrue(w.viewer_mode_buttons["2d"].isChecked())
        w.toggle_viewer_mode()          # Tab's path: the buttons follow it
        self.assertTrue(w.viewer_mode_buttons["3d"].isChecked())
        with self.assertRaises(ValueError):
            w.set_viewer_mode("4d")

    def test_each_view_keeps_its_state_over_a_2d_3d_2d_round_trip(self):
        w = self.window
        viewer, viewport = w.viewer, w.viewport
        # 2D: zoom and pan, the ROI box and a compare mode.
        viewer.scale(2.0, 2.0)
        viewer.centerOn(300, 200)
        w.roi_button.setChecked(True)
        w.command({"op": "viewer_input", "slot": 2, "id": "plate", "activate": False})
        w.command({"op": "viewer_compare", "b": 2, "mode": "wipe"})
        zoom = viewer.transform().m11()
        center = viewer.mapToScene(viewer.viewport().rect().center())
        compare = viewer.compare_mode()
        self.assertEqual(compare, "wipe")
        # 3D: camera, the progressive render mode and a selection.
        w.show_viewport_3d()
        viewport.azimuth, viewport.elevation, viewport.distance = 33.0, 12.0, 7.5
        viewport.render_mode = True
        viewport.selected_key = "grade"
        camera = (viewport.azimuth, viewport.elevation, viewport.distance)
        w.set_viewer_mode("2d")
        self.assertAlmostEqual(viewer.transform().m11(), zoom)
        now = viewer.mapToScene(viewer.viewport().rect().center())
        self.assertAlmostEqual(now.x(), center.x(), delta=1.0)
        self.assertAlmostEqual(now.y(), center.y(), delta=1.0)
        self.assertTrue(w.roi_button.isChecked())
        self.assertEqual(viewer.compare_mode(), "wipe")
        w.set_viewer_mode("3d")
        self.assertEqual((viewport.azimuth, viewport.elevation, viewport.distance), camera)
        self.assertTrue(viewport.render_mode)
        self.assertEqual(viewport.selected_key, "grade")

    def test_switching_never_asks_for_another_2d_render(self):
        w = self.window
        with unittest.mock.patch.object(w, "request_preview") as preview:
            w.set_viewer_mode("3d")
            w.set_viewer_mode("2d")
            w.toggle_viewer_mode()
            w.toggle_viewer_mode()
            preview.assert_not_called()

    def test_a_hidden_3d_viewport_does_not_paint_while_2d_is_edited(self):
        w = self.window
        with unittest.mock.patch.object(w.viewport, "update") as update:
            w.command({"op": "set", "id": "grade", "param": "gain", "value": 1.2})
            update.assert_not_called()
        with unittest.mock.patch.object(w.viewer, "update") as update2:
            w.show_viewport_3d()
            w.command({"op": "set", "id": "grade", "param": "gain", "value": 1.3})
            update2.assert_not_called()

    def test_viewing_a_3d_node_stays_put_unless_the_preference_is_on(self):
        w = self.window
        self._add_render3d()
        self.assertFalse(w.preferences.viewer_auto_3d())
        w.command({"op": "view", "id": "render"})
        w.command({"op": "viewer_input", "slot": 1, "id": "scene"})
        self.assertEqual(w.viewer_mode(), "2d")
        w.preferences.set_viewer_auto_3d(True)
        w.command({"op": "view", "id": "grade"})
        self.assertEqual(w.viewer_mode(), "2d", "a 2D node never switches the panel")
        w.command({"op": "viewer_input", "slot": 1, "id": "render"})
        self.assertEqual(w.viewer_mode(), "3d")
        w.set_viewer_mode("2d")
        w.command({"op": "view", "id": "cam"})
        self.assertEqual(w.viewer_mode(), "3d")

    def test_the_settings_dialog_carries_the_preference(self):
        w = self.window
        dialog = ProjectSettingsDialog(w.dispatcher.document["settings"], w)
        self.assertFalse(dialog.auto_3d.isChecked())
        dialog.deleteLater()
        dialog = ProjectSettingsDialog(w.dispatcher.document["settings"], w, auto_3d=True)
        self.assertTrue(dialog.auto_3d.isChecked())
        dialog.deleteLater()

    def test_the_two_monitor_workspace_floats_one_viewer_panel(self):
        w = self.window
        w.apply_two_monitor_workspace()
        APP.processEvents()
        self.assertTrue(w.viewer_dock.isFloating())
        self.assertIs(w.viewport.parentWidget(), w.view_stack)


class ViewportButtonTests(unittest.TestCase):
    """The main toolbar's "3D viewport" button has a visible result every time (Lane 2, X3)."""

    def setUp(self):
        QSettings("NodeBased", "NodeBased").remove(Preferences.VIEWER_AUTO_3D)
        self.window = Window()
        self.window.resize(1440, 920)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        w = self.window
        self.action = w.topbar_actions["3D viewport"]

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def _sizes(self):
        w = self.window
        return {d.objectName(): (d.width(), d.height()) for d in w.workspace_docks}

    def test_repeated_clicks_reuse_one_viewport_and_leave_a_roomy_layout_alone(self):
        w = self.window
        before = self._sizes()
        self.action.trigger()
        APP.processEvents()
        self.assertTrue(w.viewport.isVisible())
        self.assertTrue(w.viewer_mode_buttons["3d"].isChecked())
        self.action.trigger()
        APP.processEvents()
        self.assertEqual(len(w.findChildren(type(w.viewport))), 1, "no second 3D viewport is created")
        self.assertEqual(sum(1 for d in w.findChildren(type(w.viewer_dock)) if "3D" in d.windowTitle()), 0,
                         "and no dock of its own")
        self.assertEqual(self._sizes(), before, "a viewer that is already big enough is not resized")

    def test_the_viewport_comes_forward_from_behind_another_dock(self):
        w = self.window
        w.nodes_dock.show()    # the category bar starts it hidden
        w.tabifyDockWidget(w.nodes_dock, w.viewer_dock)
        w.nodes_dock.raise_()
        APP.processEvents()
        self.assertFalse(w.viewer_dock.isVisible(), "setup: the viewer is a hidden tab")
        self.action.trigger()
        APP.processEvents()
        self.assertTrue(w.viewer_dock.isVisible())
        self.assertTrue(w.viewport.isVisible())
        self.assertFalse(w.nodes_dock.isVisible())

    def test_a_floating_viewer_stays_floating_and_shows_the_viewport(self):
        w = self.window
        w.viewer_dock.setFloating(True)
        w.viewer_dock.hide()
        APP.processEvents()
        self.action.trigger()
        APP.processEvents()
        self.assertTrue(w.viewer_dock.isFloating())
        self.assertTrue(w.viewport.isVisible())

    def test_a_squeezed_viewer_is_given_room(self):
        w = self.window
        w.resizeDocks([w.viewer_dock, w.graph_dock], [150, w.viewer_dock.height() + w.graph_dock.height() - 150],
                      Qt.Orientation.Vertical)
        w.resizeDocks([w.viewer_dock, w.properties_dock], [380, w.viewer_dock.width() + w.properties_dock.width() - 380],
                      Qt.Orientation.Horizontal)
        APP.processEvents()
        self.assertLess(w.view_stack.height(), VIEWER_3D_MIN_HEIGHT, "setup: the viewer is cramped")
        self.assertLess(w.view_stack.width(), VIEWER_3D_MIN_WIDTH, "setup: the viewer is narrow")
        self.action.trigger()
        APP.processEvents()
        self.assertGreaterEqual(w.view_stack.height(), VIEWER_3D_MIN_HEIGHT)
        self.assertGreaterEqual(w.view_stack.width(), VIEWER_3D_MIN_WIDTH)
        self.assertGreaterEqual(w.properties_dock.width(), PROPERTIES_USABLE_WIDTH)
        self.assertTrue(w.viewport.isVisible())

    def test_switching_back_to_2d_keeps_the_layout(self):
        w = self.window
        self.action.trigger()
        APP.processEvents()
        after_3d = self._sizes()
        w.viewer_mode_buttons["2d"].click()
        APP.processEvents()
        self.assertEqual(w.viewer_mode(), "2d")
        self.assertTrue(w.viewer.isVisible())
        self.assertEqual(self._sizes(), after_3d)

    def test_the_active_mode_button_is_styled_as_active(self):
        w = self.window
        self.assertIn("#viewer-mode-3d:checked", APP.styleSheet() or w.styleSheet())


class OldWorkspaceTests(unittest.TestCase):
    """A workspace saved while the 3D VIEWPORT dock existed loads into the shared panel."""

    def setUp(self):
        QSettings("NodeBased", "NodeBased").remove("workspace")
        self.patch = unittest.mock.patch.object(Preferences, "workspace", _REAL_WORKSPACE)
        self.patch.start()
        self.window = None

    def tearDown(self):
        self.addCleanup(release_window, self)
        if self.window is not None:
            self.window.saved_document = self.window.dispatcher.document
            self.window.close()
        APP.processEvents()
        self.patch.stop()
        QSettings("NodeBased", "NodeBased").remove("workspace")

    def _open(self):
        self.window = Window()
        self.window.show()
        APP.processEvents()
        return self.window

    def test_the_fixture_really_names_the_old_dock(self):
        self.assertIn("viewport3d-dock".encode("utf-16-be"), OLD_STATE_WITH_3D_DOCK)

    def test_a_saved_workspace_naming_the_3d_dock_loads(self):
        reference = Window()
        geometry = reference.saveGeometry()
        reference.saved_document = reference.dispatcher.document
        reference.close()
        reference.deleteLater()
        Preferences().set_workspace(geometry, OLD_STATE_WITH_3D_DOCK)
        w = self._open()
        self.assertTrue(w.viewer_dock.isVisible())
        self.assertTrue(w.graph_dock.isVisible())
        # The category bar replaced the always-open NODES dock (it starts hidden); the dock itself
        # must still exist so Ctrl+F, presets and the Workspace menu keep working.
        self.assertIn(w.nodes_dock, w.workspace_docks, "NODES was tabbed with the old dock and must survive")
        self.assertTrue(w.properties_dock.isVisible())
        self.assertFalse(hasattr(w, "viewport_dock"))
        w.set_viewer_mode("3d")
        self.assertTrue(w.viewport.isVisible())

    def test_a_named_default_workspace_naming_the_3d_dock_loads(self):
        reference = Window()
        geometry = reference.saveGeometry()
        reference.saved_document = reference.dispatcher.document
        reference.close()
        reference.deleteLater()
        store = QSettings("NodeBased", "NodeBased")
        store.setValue("workspaces/default/geometry", geometry)
        store.setValue("workspaces/default/state", OLD_STATE_WITH_3D_DOCK)
        store.setValue("workspaces/default/metrics", {"size": [1300, 800], "docks": {
            "viewport3d-dock": [500, 300], "viewer-dock": [600, 400]}})
        try:
            w = self._open()
            w.restore_default_workspace()
            APP.processEvents()
            self.assertTrue(w.viewer_dock.isVisible())
            self.assertIn(w.nodes_dock, w.workspace_docks)
        finally:
            store.remove("workspaces/default")


if __name__ == "__main__":
    unittest.main()
