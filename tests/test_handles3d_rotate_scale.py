"""Qt-free tests for the 3D viewport rotate and scale gizmo geometry (lane L1, step 3b):
ring/cube placement, hit-testing, and the drag math handles3d.py adds on top of step 3a's
translate gizmo."""
import math
import unittest

import numpy as np

from nodebased import handles3d as h
from nodebased import scene3d as s

CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 10)), s.Vec3(), 45.0, 0.1, 1000.0)
# Off-axis, for hit tests where a front-on camera would degenerate: the Z ring and the X/Y rings
# all pass through the +X and +Y cardinal points, and the Z axis cube foreshortens to the same
# screen pixel as the centre cube when the camera looks straight down -Z (see test_viewport3d_gizmo's
# fixture comment on the same degeneracy for the translate gizmo's Z arrow).
CAMERA_ANGLED = s.Camera(s.Transform3D(s.Vec3(6.0, 5.0, 8.0)), s.Vec3(), 45.0, 0.1, 1000.0)
WIDTH, HEIGHT = 640, 360
ORIGIN = np.zeros(3)
SCALE = 1.5


class GizmoRingsTests(unittest.TestCase):
    def test_ring_points_sit_at_the_expected_radius_in_the_right_plane(self):
        rings = h.gizmo_rings(ORIGIN, SCALE)
        radius = SCALE * h.RING_RADIUS_FACTOR
        for name, axis in h.AXIS_VECS.items():
            points = np.asarray(rings[name])
            self.assertEqual(len(points), h.RING_SEGMENTS)
            np.testing.assert_allclose(np.linalg.norm(points, axis=1), radius, atol=1e-9)
            np.testing.assert_allclose(points @ axis, 0.0, atol=1e-9)  # every point lies in the ring's plane

    def test_ring_start_and_quarter_point_match_the_cyclic_basis(self):
        rings = h.gizmo_rings(ORIGIN, SCALE)
        radius = SCALE * h.RING_RADIUS_FACTOR
        for name, (a, b) in h.RING_AXES.items():
            points = np.asarray(rings[name])
            np.testing.assert_allclose(points[0], a * radius, atol=1e-9)
            np.testing.assert_allclose(points[len(points) // 4], b * radius, atol=1e-9)


class GizmoRingHitTests(unittest.TestCase):
    def test_a_point_on_the_z_ring_hits_z_and_nothing_else(self):
        # t = 45 degrees: off both cardinal axes, so it cannot coincide with where ring x or
        # ring y crosses the same plane (each of those has one coordinate pinned to zero).
        radius = SCALE * h.RING_RADIUS_FACTOR
        point = ORIGIN + (h.X_AXIS + h.Y_AXIS) / math.sqrt(2) * radius
        xy, _z = s.project(CAMERA, WIDTH, HEIGHT, np.array((point,)))
        self.assertEqual(h.gizmo_ring_hit(CAMERA, WIDTH, HEIGHT, ORIGIN, SCALE, xy[0]), "z")

    def test_a_point_far_from_every_ring_misses(self):
        self.assertIsNone(h.gizmo_ring_hit(CAMERA, WIDTH, HEIGHT, ORIGIN, SCALE, (5, 5)))


class RingDragAngleTests(unittest.TestCase):
    def test_a_quarter_turn_on_the_camera_facing_z_ring_is_90_degrees(self):
        # The camera looks straight down -Z at the origin, so the XY-plane "z" ring is face-on:
        # a screen point at a ring position projects back to (very nearly) that exact world point.
        radius = SCALE * h.RING_RADIUS_FACTOR
        start_xy, _z = s.project(CAMERA, WIDTH, HEIGHT, np.array((ORIGIN + h.X_AXIS * radius,)))
        current_xy, _z = s.project(CAMERA, WIDTH, HEIGHT, np.array((ORIGIN + h.Y_AXIS * radius,)))
        angle = h.ring_drag_angle(CAMERA, WIDTH, HEIGHT, ORIGIN, h.Z_AXIS, h.RING_AXES["z"],
                                  start_xy[0], current_xy[0])
        self.assertAlmostEqual(angle, 90.0, delta=0.1)

    def test_dragging_back_to_the_start_is_zero(self):
        radius = SCALE * h.RING_RADIUS_FACTOR
        point_xy, _z = s.project(CAMERA, WIDTH, HEIGHT, np.array((ORIGIN + h.X_AXIS * radius,)))
        angle = h.ring_drag_angle(CAMERA, WIDTH, HEIGHT, ORIGIN, h.Z_AXIS, h.RING_AXES["z"],
                                  point_xy[0], point_xy[0])
        self.assertAlmostEqual(angle, 0.0, delta=1e-6)


class GizmoCubesTests(unittest.TestCase):
    def test_axis_cubes_sit_where_the_arrow_tips_do_and_centre_sits_on_the_pivot(self):
        cubes = h.gizmo_cubes(ORIGIN, SCALE)
        arrows = h.gizmo_arrows(ORIGIN, SCALE)
        for name in ("x", "y", "z"):
            np.testing.assert_allclose(cubes[name], arrows[name][1], atol=1e-9)
        np.testing.assert_allclose(cubes["center"], ORIGIN, atol=1e-9)


class GizmoScaleHitTests(unittest.TestCase):
    def test_a_point_on_the_x_cube_hits_x(self):
        xy, _z = s.project(CAMERA, WIDTH, HEIGHT, np.array((h.gizmo_cubes(ORIGIN, SCALE)["x"],)))
        self.assertEqual(h.gizmo_scale_hit(CAMERA, WIDTH, HEIGHT, ORIGIN, SCALE, xy[0]), "x")

    def test_a_point_on_the_pivot_hits_the_centre_cube(self):
        # The front-on CAMERA would put the Z cube at the same screen pixel as the centre cube
        # (Z is the boresight), so this needs the off-axis camera to be unambiguous.
        xy, _z = s.project(CAMERA_ANGLED, WIDTH, HEIGHT, np.array((ORIGIN,)))
        self.assertEqual(h.gizmo_scale_hit(CAMERA_ANGLED, WIDTH, HEIGHT, ORIGIN, SCALE, xy[0]), "center")


class ScaleFromDragTests(unittest.TestCase):
    def test_doubling_the_distance_doubles_the_value(self):
        value = h.scale_from_drag(2.0, (0.0, 0.0), (20.0, 0.0), (40.0, 0.0))
        self.assertAlmostEqual(value, 4.0, places=6)

    def test_a_start_distance_below_the_floor_uses_the_floor(self):
        # Pressing exactly on the origin (the centre cube's own screen position) would divide by
        # zero without the floor; with it, the ratio is measured against min_reference_px instead.
        value = h.scale_from_drag(1.0, (0.0, 0.0), (0.0, 0.0), (h.GIZMO_HIT_PIXELS * 3, 0.0))
        self.assertAlmostEqual(value, 3.0, places=6)

    def test_never_goes_below_the_minimum(self):
        value = h.scale_from_drag(1.0, (0.0, 0.0), (100.0, 0.0), (0.0, 0.0), minimum=0.01)
        self.assertAlmostEqual(value, 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
