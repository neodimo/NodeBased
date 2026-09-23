"""Offscreen interaction tests for the 3D viewport translate gizmo and pivot mode (lane L1,
step 3a)."""
import os
import time
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


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        QTest.qWait(10)
    return False


def single_card_document():
    """One Card3D, unwired (no Render3D): the viewport's loose-geometry fallback, the same
    candidate set `_pick_candidates` picking already exercises in test_viewport3d_picking."""
    d = Dispatcher()
    d.execute({"op": "create", "id": "obj", "type": "Card3D"})
    d.execute({"op": "set", "id": "obj", "param": "tz", "value": 2.0})
    return d.document


class GizmoDragTestsBase(unittest.TestCase):
    def setUp(self):
        self.window = Window(single_card_document())
        self.window.show()
        self.window.viewport_dock.show()
        self.window.resize(1000, 800)
        self.viewport = self.window.viewport
        self.viewport.resize(640, 360)
        # Front-on, matching test_viewport3d_picking: camera at (0, 0, 10) looking at the
        # origin. The card sits directly on the boresight, so the X arrow and the XY plane
        # square both face the camera undistorted; only Z foreshortens to a point (see
        # test_handles3d's degenerate-axis test), which these tests do not use.
        self.viewport.azimuth, self.viewport.elevation, self.viewport.distance = 0.0, 0.0, 10.0
        APP.processEvents()
        center = QPointF(self.viewport.width() / 2, self.viewport.height() / 2).toPoint()
        QTest.mouseClick(self.viewport, Qt.MouseButton.LeftButton, pos=center)
        self.assertEqual(self.viewport.selected_key, "obj")

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def _screen(self, world_point):
        xy, _z = s.project(self.viewport._camera(), self.viewport.width(), self.viewport.height(),
                           np.asarray(world_point, np.float64)[None])
        return QPointF(*xy[0]).toPoint()

    def _drag(self, start_world, target_world):
        start, target = self._screen(start_world), self._screen(target_world)
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.viewport, target)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)

    def _x_arrow_anchor(self):
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        start, end = h.gizmo_arrows(origin, scale)["x"]
        return (np.asarray(start) + np.asarray(end)) / 2  # anywhere on the shaft is a valid hit


class TranslateGizmoTests(GizmoDragTestsBase):
    def test_dragging_the_x_arrow_changes_only_tx_by_the_expected_world_distance(self):
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        anchor = self._x_arrow_anchor()
        self._drag(anchor, anchor + h.X_AXIS)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertAlmostEqual(after["tx"], before["tx"] + 1.0, delta=0.02)
        self.assertAlmostEqual(after["ty"], before["ty"], delta=1e-6)
        self.assertAlmostEqual(after["tz"], before["tz"], delta=1e-6)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_dragging_the_xy_square_changes_tx_and_ty_not_tz(self):
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        anchor = np.mean(np.asarray(h.gizmo_planes(origin, scale)["xy"]), axis=0)
        self._drag(anchor, anchor + h.X_AXIS * 0.6 + h.Y_AXIS * 0.4)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertAlmostEqual(after["tx"], before["tx"] + 0.6, delta=0.02)
        self.assertAlmostEqual(after["ty"], before["ty"] + 0.4, delta=0.02)
        self.assertAlmostEqual(after["tz"], before["tz"], delta=1e-6)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_a_drag_is_one_undo_step_and_undo_restores_everything(self):
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        anchor = self._x_arrow_anchor()
        self._drag(anchor, anchor + h.X_AXIS)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertEqual((after["tx"], after["ty"], after["tz"]), (before["tx"], before["ty"], before["tz"]))

    def test_animated_tx_gets_a_key(self):
        self.window.command({"op": "set_key", "id": "obj", "param": "tx", "frame": 1, "value": 0.0})
        self.window.command({"op": "set_key", "id": "obj", "param": "tx", "frame": 2, "value": 5.0})
        self.window.dispatcher.undo_stack.clear()
        anchor = self._x_arrow_anchor()
        self._drag(anchor, anchor + h.X_AXIS)
        curve = self.window.dispatcher.document["animation"]["curves"]["obj"]["tx"]
        frame = self.window.dispatcher.document["time"]["current"]
        self.assertTrue(any(key["frame"] == frame for key in curve["keys"]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_gizmo_does_not_respond_when_nothing_is_selected(self):
        anchor = self._x_arrow_anchor()
        self.viewport._select(None)
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        self._drag(anchor, anchor + h.X_AXIS)  # would have hit the arrow, had it still been drawn
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertEqual(before, after)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 0)
        self.assertIsNone(self.viewport._gizmo_drag)

    def test_orbit_still_works_when_the_drag_starts_off_the_gizmo(self):
        azimuth_before = self.viewport.azimuth
        corner = QPointF(self.viewport.width() - 5, 5).toPoint()  # well clear of the gizmo
        target = QPointF(corner.x() - 60, corner.y() + 40).toPoint()
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=corner)
        QTest.mouseMove(self.viewport, target)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)
        self.assertNotEqual(self.viewport.azimuth, azimuth_before)
        self.assertEqual(self.viewport.selected_key, "obj")  # the earlier click's pick survives
        self.assertIsNone(self.viewport._gizmo_drag)

    def test_escape_cancels_a_drag(self):
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        anchor = self._x_arrow_anchor()
        start, target = self._screen(anchor), self._screen(anchor + h.X_AXIS)
        QTest.mousePress(self.viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(self.viewport, target)
        self.assertIsNotNone(self.viewport._gizmo_drag)
        QTest.keyClick(self.viewport, Qt.Key.Key_Escape)
        self.assertIsNone(self.viewport._gizmo_drag)
        QTest.mouseRelease(self.viewport, Qt.MouseButton.LeftButton, pos=target)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertEqual(before["tx"], after["tx"])
        self.assertEqual(len(self.window.dispatcher.undo_stack), 0)


class PivotModeTests(GizmoDragTestsBase):
    def test_q_toggles_pivot_mode(self):
        self.assertFalse(self.viewport.pivot_mode)
        QTest.keyClick(self.viewport, Qt.Key.Key_Q)
        self.assertTrue(self.viewport.pivot_mode)
        QTest.keyClick(self.viewport, Qt.Key.Key_Q)
        self.assertFalse(self.viewport.pivot_mode)

    def test_pivot_drag_moves_the_pivot_and_leaves_a_vertex_fixed_in_world_space(self):
        # A non-trivial rotation makes `linear != I`, so a sign error in the translate
        # compensation ((linear - I) @ pivot_delta) would show up as vertex drift.
        self.window.command({"op": "set", "id": "obj", "param": "ry", "value": 40.0})
        self.window.dispatcher.undo_stack.clear()
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        sample_local = np.array((1.0, 1.0, 0.0, 1.0))  # a corner of the default 2x2 card
        world_before = s._transform_from(before).matrix() @ sample_local
        self.viewport.pivot_mode = True
        anchor = self._x_arrow_anchor()
        self._drag(anchor, anchor + h.X_AXIS * 0.7)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertNotEqual(after["pivot_x"], before["pivot_x"])
        self.assertNotEqual(after["tx"], before["tx"])
        world_after = s._transform_from(after).matrix() @ sample_local
        np.testing.assert_allclose(world_after, world_before, atol=0.02)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_pivot_drag_is_one_undo_step_and_undo_restores_everything(self):
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        self.viewport.pivot_mode = True
        anchor = self._x_arrow_anchor()
        self._drag(anchor, anchor + h.X_AXIS * 0.5)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        for param in ("tx", "ty", "tz", "pivot_x", "pivot_y", "pivot_z"):
            self.assertEqual(after[param], before[param])


if __name__ == "__main__":
    unittest.main()
