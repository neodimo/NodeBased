"""Offscreen interaction tests for the 3D viewport rotate and scale gizmos and the gizmo-mode
hotkeys (lane L1, step 3b). Reuses the drag fixture from step 3a's own test module."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nodebased import handles3d as h
from tests.test_viewport3d_gizmo import APP, GizmoDragTestsBase


class GizmoModeHotkeyTests(GizmoDragTestsBase):
    def test_w_e_r_switch_the_gizmo_mode(self):
        self.assertEqual(self.viewport.gizmo_mode, "translate")
        QTest.keyClick(self.viewport, Qt.Key.Key_E)
        self.assertEqual(self.viewport.gizmo_mode, "rotate")
        QTest.keyClick(self.viewport, Qt.Key.Key_R)
        self.assertEqual(self.viewport.gizmo_mode, "scale")
        QTest.keyClick(self.viewport, Qt.Key.Key_W)
        self.assertEqual(self.viewport.gizmo_mode, "translate")


class RotateGizmoTests(GizmoDragTestsBase):
    def _ring_anchor(self, name, fraction):
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        points = h.gizmo_rings(origin, scale)[name]
        return np.asarray(points[int(len(points) * fraction)])

    def test_dragging_the_z_ring_changes_only_rz_by_the_expected_angle(self):
        self.viewport.gizmo_mode = "rotate"
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        start = self._ring_anchor("z", 0.0)     # t = 0 degrees
        target = self._ring_anchor("z", 0.25)   # t = 90 degrees
        self._drag(start, target)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertAlmostEqual(after["rz"], before["rz"] + 90.0, delta=1.0)
        self.assertAlmostEqual(after["rx"], before["rx"], delta=1e-6)
        self.assertAlmostEqual(after["ry"], before["ry"], delta=1e-6)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_a_ring_drag_is_one_undo_step_and_undo_restores_everything(self):
        self.viewport.gizmo_mode = "rotate"
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        start, target = self._ring_anchor("z", 0.0), self._ring_anchor("z", 0.25)
        self._drag(start, target)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertEqual((after["rx"], after["ry"], after["rz"]), (before["rx"], before["ry"], before["rz"]))

    def test_animated_rz_gets_a_key(self):
        self.viewport.gizmo_mode = "rotate"
        self.window.command({"op": "set_key", "id": "obj", "param": "rz", "frame": 1, "value": 0.0})
        self.window.command({"op": "set_key", "id": "obj", "param": "rz", "frame": 2, "value": 5.0})
        self.window.dispatcher.undo_stack.clear()
        start, target = self._ring_anchor("z", 0.0), self._ring_anchor("z", 0.25)
        self._drag(start, target)
        curve = self.window.dispatcher.document["animation"]["curves"]["obj"]["rz"]
        frame = self.window.dispatcher.document["time"]["current"]
        self.assertTrue(any(key["frame"] == frame for key in curve["keys"]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_rotation_about_a_moved_pivot_leaves_the_pivots_world_position_unchanged(self):
        self.window.command({"op": "set", "id": "obj", "param": "pivot_x", "value": 0.7})
        self.window.command({"op": "set", "id": "obj", "param": "pivot_y", "value": -0.4})
        self.window.dispatcher.undo_stack.clear()
        self.viewport.gizmo_mode = "rotate"
        parent = self.viewport._gizmo_info("obj")[1]
        before_params = self.window.dispatcher.document["nodes"]["obj"]["params"]
        pivot_before = h.pivot_world_position(before_params, parent)
        start, target = self._ring_anchor("z", 0.0), self._ring_anchor("z", 0.25)
        self._drag(start, target)
        after_params = self.window.dispatcher.document["nodes"]["obj"]["params"]
        pivot_after = h.pivot_world_position(after_params, parent)
        np.testing.assert_allclose(pivot_after, pivot_before, atol=1e-6)


class ScaleGizmoTests(GizmoDragTestsBase):
    def test_dragging_the_x_cube_changes_only_sx_proportionally(self):
        self.viewport.gizmo_mode = "scale"
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        start = h.gizmo_cubes(origin, scale)["x"]
        offset = float(np.linalg.norm(start - origin))
        target = origin + h.X_AXIS * (offset * 2.0)  # same depth as origin: screen ratio == world ratio
        self._drag(start, target)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertAlmostEqual(after["sx"], before["sx"] * 2.0, delta=0.05)
        self.assertAlmostEqual(after["sy"], before["sy"], delta=1e-6)
        self.assertAlmostEqual(after["sz"], before["sz"], delta=1e-6)
        self.assertAlmostEqual(after["uscale"], before["uscale"], delta=1e-6)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_dragging_the_centre_cube_changes_only_uscale(self):
        self.viewport.gizmo_mode = "scale"
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        start = h.gizmo_cubes(origin, scale)["center"]
        target = origin + h.X_AXIS * (scale * 1.5)
        self._drag(start, target)
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertGreater(after["uscale"], before["uscale"])
        self.assertAlmostEqual(after["sx"], before["sx"], delta=1e-6)
        self.assertAlmostEqual(after["sy"], before["sy"], delta=1e-6)
        self.assertAlmostEqual(after["sz"], before["sz"], delta=1e-6)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_a_scale_drag_is_one_undo_step_and_undo_restores_everything(self):
        self.viewport.gizmo_mode = "scale"
        before = dict(self.window.dispatcher.document["nodes"]["obj"]["params"])
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        start = h.gizmo_cubes(origin, scale)["x"]
        target = origin + h.X_AXIS * (float(np.linalg.norm(start - origin)) * 2.0)
        self._drag(start, target)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.window.command({"op": "undo"})
        after = self.window.dispatcher.document["nodes"]["obj"]["params"]
        self.assertEqual(after["sx"], before["sx"])

    def test_animated_sx_gets_a_key(self):
        self.viewport.gizmo_mode = "scale"
        self.window.command({"op": "set_key", "id": "obj", "param": "sx", "frame": 1, "value": 1.0})
        self.window.command({"op": "set_key", "id": "obj", "param": "sx", "frame": 2, "value": 2.0})
        self.window.dispatcher.undo_stack.clear()
        origin, _parent, _params, scale = self.viewport._gizmo_info("obj")
        start = h.gizmo_cubes(origin, scale)["x"]
        target = origin + h.X_AXIS * (float(np.linalg.norm(start - origin)) * 2.0)
        self._drag(start, target)
        curve = self.window.dispatcher.document["animation"]["curves"]["obj"]["sx"]
        frame = self.window.dispatcher.document["time"]["current"]
        self.assertTrue(any(key["frame"] == frame for key in curve["keys"]))
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)


if __name__ == "__main__":
    unittest.main()
