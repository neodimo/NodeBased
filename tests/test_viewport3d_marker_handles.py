"""Offscreen interaction tests for Camera3D and Light3D marker picking and their position/
target handles (lane L1, step 4 of 4)."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher
from nodebased import handles3d as h, scene3d as s

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def light_and_camera_document():
    """A loose Light3D ('sun') and Camera3D ('cam'), both unwired: the viewport's loose-marker
    fallback, the same candidate set `_marker_candidates` uses when nothing is wired up yet."""
    d = Dispatcher()
    d.execute({"op": "create", "id": "sun", "type": "Light3D"})
    d.execute({"op": "create", "id": "cam", "type": "Camera3D"})
    return d.document


class MarkerHandleTestsBase(unittest.TestCase):
    def setUp(self):
        self.window = Window(light_and_camera_document())
        self.window.show()
        self.window.viewport_dock.show()
        self.window.resize(1000, 800)
        self.viewport = self.window.viewport
        self.viewport.resize(640, 360)
        # Angled off every principal axis (as test_handles3d's ANGLED_CAMERA is) so neither
        # marker's gizmo arrows foreshorten to a degenerate on-screen point.
        self.viewport.azimuth, self.viewport.elevation, self.viewport.distance = 35.0, 25.0, 20.0
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def _screen(self, world_point):
        xy, _z = s.project(self.viewport._camera(), self.viewport.width(), self.viewport.height(),
                           np.asarray(world_point, np.float64)[None])
        return QPointF(*xy[0]).toPoint()

    def _click(self, world_point):
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=self._screen(world_point))

    def _drag(self, start_world, target_world):
        """Drag between two world points; returns the tolerance a drag of that length can meet.

        Both ends snap to whole pixels, up to about 1.4 px of error together, so the world-space
        slack is that many pixels' worth at this viewport size and zoom (Windows CI's viewport
        comes out smaller than Linux's and missed a fixed 0.02 by 0.004)."""
        start, target = self._screen(start_world), self._screen(target_world)
        xy, _z = s.project(self.viewport._camera(), self.viewport.width(), self.viewport.height(),
                           np.asarray([start_world, target_world], np.float64))
        pixels = float(np.linalg.norm(xy[1] - xy[0]))
        world = float(np.linalg.norm(np.asarray(target_world) - np.asarray(start_world)))
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.viewport, target)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)
        return max(0.02, 1.5 * world / pixels)

    def _select_by_click(self, key):
        params = self.window.dispatcher.document["nodes"][key]["params"]
        self._click((params["tx"], params["ty"], params["tz"]))
        self.assertEqual(self.viewport.selected_key, key)

    def _position_anchor(self, key, axis="x"):
        origin, _parent, _params, scale = self.viewport._gizmo_info(key)
        start, end = h.gizmo_arrows(origin, scale)[axis]
        return (np.asarray(start) + np.asarray(end)) / 2  # anywhere on the shaft is a valid hit

    def _target_anchor(self, key, axis="x"):
        origin, _parent, _params, scale = self.viewport._target_gizmo_info(key)
        start, end = h.gizmo_arrows(origin, scale)[axis]
        return (np.asarray(start) + np.asarray(end)) / 2


class MarkerPickingTests(MarkerHandleTestsBase):
    def test_click_on_the_light_marker_selects_the_light_node(self):
        self._select_by_click("sun")
        self.assertEqual(self.window.graph.selected_id(), "sun")

    def test_click_on_the_camera_marker_selects_the_camera_node(self):
        self._select_by_click("cam")
        self.assertEqual(self.window.graph.selected_id(), "cam")

    def test_click_on_empty_space_selects_nothing(self):
        self._select_by_click("sun")
        self._click((-50.0, -50.0, -50.0))  # far off both markers and the origin
        self.assertIsNone(self.viewport.selected_key)


class PositionHandleTests(MarkerHandleTestsBase):
    def test_dragging_the_lights_position_handle_changes_its_position(self):
        self._select_by_click("sun")
        before = dict(self.window.dispatcher.document["nodes"]["sun"]["params"])
        anchor = self._position_anchor("sun")
        tolerance = self._drag(anchor, anchor + h.X_AXIS)
        after = self.window.dispatcher.document["nodes"]["sun"]["params"]
        self.assertAlmostEqual(after["tx"], before["tx"] + 1.0, delta=tolerance)
        self.assertEqual(after["ty"], before["ty"])
        self.assertEqual(after["tz"], before["tz"])
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_position_drag_is_one_undo_step_and_undo_restores_everything(self):
        self._select_by_click("sun")
        before = dict(self.window.dispatcher.document["nodes"]["sun"]["params"])
        anchor = self._position_anchor("sun")
        self._drag(anchor, anchor + h.X_AXIS)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        after = self.window.dispatcher.document["nodes"]["sun"]["params"]
        self.assertEqual((after["tx"], after["ty"], after["tz"]),
                         (before["tx"], before["ty"], before["tz"]))

    def test_animated_tx_gets_a_key(self):
        self._select_by_click("cam")
        self.window.command({"op": "set_key", "id": "cam", "param": "tx", "frame": 1, "value": 0.0})
        self.window.command({"op": "set_key", "id": "cam", "param": "tx", "frame": 2, "value": 5.0})
        self.window.dispatcher.undo_stack.clear()
        anchor = self._position_anchor("cam")
        self._drag(anchor, anchor + h.X_AXIS)
        curve = self.window.dispatcher.document["animation"]["curves"]["cam"]["tx"]
        frame = self.window.dispatcher.document["time"]["current"]
        self.assertTrue(any(key["frame"] == frame for key in curve["keys"]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)


class TargetHandleTests(MarkerHandleTestsBase):
    def test_dragging_the_lights_target_handle_changes_only_the_target(self):
        self._select_by_click("sun")
        before = dict(self.window.dispatcher.document["nodes"]["sun"]["params"])
        anchor = self._target_anchor("sun")
        tolerance = self._drag(anchor, anchor + h.X_AXIS * 0.8)
        after = self.window.dispatcher.document["nodes"]["sun"]["params"]
        self.assertAlmostEqual(after["target_x"], before["target_x"] + 0.8, delta=tolerance)
        self.assertEqual(after["target_y"], before["target_y"])
        self.assertEqual(after["target_z"], before["target_z"])
        self.assertEqual(after["tx"], before["tx"])  # the light itself does not move
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_dragging_the_cameras_target_handle_changes_aim_and_the_forward_vector_follows(self):
        self._select_by_click("cam")
        before = dict(self.window.dispatcher.document["nodes"]["cam"]["params"])
        anchor = self._target_anchor("cam")
        tolerance = self._drag(anchor, anchor + h.X_AXIS * 1.5)
        after = self.window.dispatcher.document["nodes"]["cam"]["params"]
        self.assertAlmostEqual(after["target_x"], before["target_x"] + 1.5, delta=tolerance)
        self.assertEqual(after["target_y"], before["target_y"])
        self.assertEqual(after["target_z"], before["target_z"])
        self.assertEqual((after["tx"], after["ty"], after["tz"]), (before["tx"], before["ty"], before["tz"]))
        camera = s.camera_from_node(self.window.dispatcher.document["nodes"]["cam"])
        eye = np.array((after["tx"], after["ty"], after["tz"]), np.float64)
        target = np.array((after["target_x"], after["target_y"], after["target_z"]), np.float64)
        expected_forward = (target - eye) / np.linalg.norm(target - eye)
        _eye, view = s._view_basis(camera)
        actual_forward = -view[2]  # _view_basis stacks (right, up, -forward)
        np.testing.assert_allclose(actual_forward, expected_forward, atol=1e-4)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_target_drag_is_one_undo_step_and_undo_restores_everything(self):
        self._select_by_click("sun")
        before = dict(self.window.dispatcher.document["nodes"]["sun"]["params"])
        anchor = self._target_anchor("sun")
        self._drag(anchor, anchor + h.X_AXIS * 0.5)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        after = self.window.dispatcher.document["nodes"]["sun"]["params"]
        self.assertEqual((after["target_x"], after["target_y"], after["target_z"]),
                         (before["target_x"], before["target_y"], before["target_z"]))

    def test_escape_cancels_a_target_drag(self):
        self._select_by_click("sun")
        before = dict(self.window.dispatcher.document["nodes"]["sun"]["params"])
        anchor = self._target_anchor("sun")
        start, target = self._screen(anchor), self._screen(anchor + h.X_AXIS)
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.viewport, target)
        self.assertIsNotNone(self.viewport._gizmo_drag)
        QTest.keyClick(self.viewport, Qt.Key.Key_Escape)
        self.assertIsNone(self.viewport._gizmo_drag)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)
        after = self.window.dispatcher.document["nodes"]["sun"]["params"]
        self.assertEqual(before["target_x"], after["target_x"])
        self.assertEqual(len(self.window.dispatcher.undo_stack), 0)


class LookedThroughCameraTests(MarkerHandleTestsBase):
    def test_looked_through_camera_shows_no_handles(self):
        self._select_by_click("cam")
        anchor = self._position_anchor("cam")
        self.viewport.look_through = True
        self.assertIsNone(self.viewport._gizmo_hit(self._screen(anchor)))
        self._drag(anchor, anchor + h.X_AXIS)
        after = self.window.dispatcher.document["nodes"]["cam"]["params"]
        self.assertEqual(after["tx"], 0.0)
        self.assertIsNone(self.viewport._gizmo_drag)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 0)


class OrbitStillWorksTests(MarkerHandleTestsBase):
    def test_orbit_still_works_after_picking_a_marker(self):
        self._select_by_click("sun")
        azimuth_before = self.viewport.azimuth
        corner = QPointF(self.viewport.width() - 5, 5).toPoint()  # well clear of both gizmos
        target = QPointF(corner.x() - 60, corner.y() + 40).toPoint()
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=corner)
        QTest.mouseMove(self.viewport, target)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)
        self.assertNotEqual(self.viewport.azimuth, azimuth_before)
        self.assertEqual(self.viewport.selected_key, "sun")  # the earlier click's pick survives


if __name__ == "__main__":
    unittest.main()
