"""Depth of field (lane L4 step R5, finish 1): the viewer's pick-focus click for Camera3D (offscreen Qt).

The click lands on the viewer showing a Render3D that uses the camera; the focus distance is the view depth of
the surface under the clicked pixel (`lens.pick_focus_distance`, tested in tests/test_3d_lens.py)."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document
from tests.test_transform_handle_ui import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)

SIZE = 128


class PickFocusUiTests(unittest.TestCase):
    def setUp(self):
        document = empty_document()
        document["nodes"] = {}
        dispatcher = Dispatcher(document)
        for key, kind, params in (("ball", "Sphere3D", dict(sphere_radius=1.0, tz=-20.0)),
                                  ("camera", "Camera3D", dict(tz=0.0, target_z=-1.0)),
                                  ("scene", "Scene3D", {}),
                                  ("render", "Render3D", dict(width=SIZE, height=SIZE))):
            dispatcher.execute(dict(op="create", id=key, type=kind, params=params))
        dispatcher.execute(dict(op="connect", id="scene", input="object0", source="ball"))
        dispatcher.execute(dict(op="connect", id="render", input="scene", source="scene"))
        dispatcher.execute(dict(op="connect", id="render", input="camera", source="camera"))
        dispatcher.execute({"op": "view", "id": "render"})
        self.window = Window(dispatcher.document)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None and self.window.viewer.format_rect is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def focus(self):
        return self.window.dispatcher.document["nodes"]["camera"]["params"]["focus_distance"]

    def click(self, x, y):
        viewer = self.window.viewer
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         viewer.mapFromScene(QPointF(x + 0.5, y + 0.5)))

    def test_a_click_focuses_on_the_surface_and_escape_ends_picking(self):
        viewer = self.window.viewer
        before = self.focus()
        self.assertTrue(self.window.begin_focus_pick("camera"))
        self.assertEqual(viewer.focus_picking, "camera")
        self.click(SIZE // 2, SIZE // 2)          # the front of the ball, 19 units from the camera (radius 1 at 20)
        self.assertAlmostEqual(self.focus(), 19.0, delta=0.05)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)      # one undo step
        self.click(1, 1)                          # empty space: unchanged
        self.assertAlmostEqual(self.focus(), 19.0, delta=0.05)
        QTest.keyClick(viewer, Qt.Key.Key_Escape)
        self.assertIsNone(viewer.focus_picking)
        self.window.dispatcher.execute({"op": "set", "id": "camera", "param": "focus_distance", "value": before})
        self.click(SIZE // 2, SIZE // 2)          # picking is off: a click sets nothing
        self.assertEqual(self.focus(), before)

    def test_picking_needs_a_camera_that_a_render3d_uses(self):
        self.assertFalse(self.window.begin_focus_pick("ball"))
        self.assertIsNone(self.window.viewer.focus_picking)
        self.window.command({"op": "create", "id": "spare", "type": "Camera3D", "pos": [0, 0]})
        self.assertFalse(self.window.begin_focus_pick("spare"))
        self.assertIsNone(self.window.viewer.focus_picking)

    def test_the_camera_panel_has_the_pick_button(self):
        from PySide6.QtWidgets import QPushButton
        self.window.inspect("camera")
        labels = [b.text() for b in self.window.properties.findChildren(QPushButton)]
        self.assertIn("Pick focus from viewer…", labels)


if __name__ == "__main__":
    unittest.main()
