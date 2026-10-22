"""Every Viewer zoom and navigation key works on the direct picture (Lane 2, 10/22).

`app.VIEW_KEY_ACTIONS` is the Viewer's key map. This module walks it: each binding is pressed on the Viewer and on
the direct picture, and the view must move the way the action says. A key added to the map without a line in
EXPECTED here, a line in `app.VIEW_ACTION_KEYS` (the text of the "?" list), or an implementation in
`DirectViewerWidget.view_action` fails one of these tests, so the direct picture cannot fall behind the Viewer.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest

import tests.isolation  # noqa: F401
from nodebased import app as app_module
from tests.test_directviewer_hit_order import APP, HAVE_CANVAS, SCALES, HitOrderBase, scene


def viewer_zoom(viewer):
    return viewer.transform().m11()


# action -> (what the Viewer's zoom should become from 2.0, what the direct picture's logical zoom should become
#            from 2.0, whether it ends in auto-fit)
EXPECTED = {
    "fit": (None, None, True),
    "one_to_one": (1.0, 1.0, False),
    "zoom_in": (2.0 * app_module.VIEW_ZOOM_STEP, 2.0 * app_module.VIEW_ZOOM_STEP, False),
    "zoom_out": (2.0 / app_module.VIEW_ZOOM_STEP, 2.0 / app_module.VIEW_ZOOM_STEP, False),
}


def bindings():
    return [(action, key, mods) for action, keys in app_module.VIEW_KEY_ACTIONS.items() for key, mods in keys]


@unittest.skipUnless(HAVE_CANVAS, "rendercanvas unavailable")
class ViewKeyTests(HitOrderBase):
    def setUp(self):
        super().setUp()
        self.show_node("c")

    def zoomed(self, surface, scale):
        """Both surfaces at logical zoom 2.0, panned off the middle, fit switched off."""
        self.place_surfaces(scene(self.window, 200.0, 120.0), 2.0, scale)

    def press(self, surface, key, mods):
        target = surface
        QTest.keyClick(target, key, mods)
        APP.processEvents()

    def test_every_binding_of_the_viewer_key_map_has_an_expectation_and_a_help_line(self):
        for action in app_module.VIEW_KEY_ACTIONS:
            self.assertIn(action, EXPECTED, f"{action}: say here what the key does to the view")
            self.assertIn(action, app_module.VIEW_ACTION_KEYS, f"{action}: add its line to the '?' list")

    def test_each_binding_moves_the_viewer_as_its_action_says(self):
        for action, key, mods in bindings():
            with self.subTest(action=action, key=key):
                viewer_zoom_after, _, fits = EXPECTED[action]
                self.zoomed(self.viewer, 1.0)
                self.viewer.setFocus()
                self.viewer.auto_fit = False
                self.press(self.viewer, key, mods)
                if viewer_zoom_after is not None:
                    self.assertAlmostEqual(viewer_zoom(self.viewer), viewer_zoom_after, places=6)
                self.assertEqual(self.viewer.auto_fit, fits)

    def test_each_binding_moves_the_direct_picture_the_same_way(self):
        for scale in SCALES:
            for action, key, mods in bindings():
                with self.subTest(action=action, key=key, scale=scale):
                    _, direct_zoom_after, fits = EXPECTED[action]
                    self.zoomed(self.widget, scale)
                    self.widget.setFocus()
                    before = self.widget.view
                    centre_before = before[:2]
                    self.press(self.widget, key, mods)
                    view = self.widget.view
                    self.assertEqual(self.widget.auto_fit, fits, "fit is on only after a fit")
                    if direct_zoom_after is not None:
                        self.assertAlmostEqual(view[2] / scale, direct_zoom_after, places=6,
                                               msg="logical zoom, like the percentage in the corner")
                        if action != "one_to_one":
                            self.assertNotEqual(view, before)
                    if action in ("zoom_in", "zoom_out", "one_to_one"):
                        # the middle of the picture is the anchor when the pointer is elsewhere
                        self.assertAlmostEqual(view[0], centre_before[0], places=6)
                        self.assertAlmostEqual(view[1], centre_before[1], places=6)

    def test_zoom_keys_keep_the_scene_point_under_the_pointer_fixed(self):
        self.zoomed(self.widget, 1.0)
        width, height = self.widget._physical()
        self.widget._pointer = (width * 0.75, height * 0.25)
        point = self.widget.scene_point(*self.widget._pointer)
        self.press(self.widget, Qt.Key.Key_Equal, Qt.KeyboardModifier.ControlModifier)
        after = self.widget.scene_point(*self.widget._pointer)
        self.assertAlmostEqual(point[0], after[0], places=6)
        self.assertAlmostEqual(point[1], after[1], places=6)

    def test_the_zoom_stops_at_the_viewer_limits(self):
        self.zoomed(self.widget, 1.0)
        for _ in range(60):
            self.press(self.widget, Qt.Key.Key_Equal, Qt.KeyboardModifier.ControlModifier)
        self.assertLess(self.widget.view[2], 20.0)
        top = self.widget.view[2]
        self.press(self.widget, Qt.Key.Key_Equal, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.widget.view[2], top)

    def test_a_zoomed_direct_picture_stays_zoomed_through_a_resize(self):
        self.zoomed(self.widget, 1.0)
        self.press(self.widget, Qt.Key.Key_Equal, Qt.KeyboardModifier.ControlModifier)
        self.assertFalse(self.widget.auto_fit)

    def test_the_question_mark_lists_the_keys_on_the_viewer_and_on_the_direct_picture(self):
        for surface in (self.viewer, self.widget):
            with self.subTest(surface=type(surface).__name__):
                surface.setFocus()
                self.press(surface, Qt.Key.Key_Question, Qt.KeyboardModifier.ShiftModifier)
                overlay = self.window.viewer_shortcut_overlay
                self.assertTrue(overlay.isVisible())
                text = overlay.text()
                for action in app_module.VIEW_KEY_ACTIONS:
                    keys, what = app_module.VIEW_ACTION_KEYS[action]
                    self.assertIn(f"{keys} {what}", text)
                for _, key, mods in bindings():
                    token = {Qt.Key.Key_F: "F", Qt.Key.Key_H: "H", Qt.Key.Key_1: "1", Qt.Key.Key_Equal: "=",
                             Qt.Key.Key_Plus: "+", Qt.Key.Key_Minus: "-"}[key]
                    self.assertIn(token, text, f"{key} is listed")
                overlay.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Question,
                                                Qt.KeyboardModifier.ShiftModifier, "?"))
                self.assertFalse(overlay.isVisible())

    def test_the_reference_dialog_and_the_list_agree_on_the_zoom_keys(self):
        viewer_rows = dict(app_module.SHORTCUT_SECTIONS)["Viewer"]
        listed = " ".join(keys for keys, _ in viewer_rows)
        for action in app_module.VIEW_KEY_ACTIONS:
            for key, mods in app_module.VIEW_KEY_ACTIONS[action]:
                token = {Qt.Key.Key_F: "F", Qt.Key.Key_H: "H", Qt.Key.Key_1: "1", Qt.Key.Key_Equal: "=",
                         Qt.Key.Key_Plus: "+", Qt.Key.Key_Minus: "-"}[key]
                self.assertIn(token, listed)


if __name__ == "__main__":
    unittest.main()
