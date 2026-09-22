import unittest

from nodebased.handles2d import (
    box_corners,
    edge_midpoints,
    forward_point,
    pivot_drag_result,
    pivot_point,
    rotate_from_drag,
    scale_from_drag,
)


class Handles2dTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected, places=9):
        self.assertAlmostEqual(actual[0], expected[0], places=places)
        self.assertAlmostEqual(actual[1], expected[1], places=places)

    def test_forward_point_identity_ignores_center(self):
        for x, y, center_x, center_y in ((0, 0, 0, 0), (12.5, -7.25, 100, 50), (-30, 80, -4, 9)):
            self.assertPointAlmostEqual(
                forward_point(x, y, 0, 0, 0, 1, center_x, center_y), (x, y))

    def test_forward_point_translation_ignores_center(self):
        for center_x, center_y in ((0, 0), (50, -20), (-3.5, 41)):
            self.assertPointAlmostEqual(
                forward_point(10, -4, 7.5, -3.25, 0, 1, center_x, center_y), (17.5, -7.25))

    def test_box_corners_identity(self):
        self.assertEqual(box_corners(100, 50, 0, 0, 0, 1, 17, -9),
                         [(0, 0), (100, 0), (100, 50), (0, 50)])

    def test_box_corners_rotate_90_about_own_center(self):
        corners = box_corners(100, 50, 0, 0, 90, 1, 50, 25)
        expected = [(75, -25), (75, 75), (25, 75), (25, -25)]
        for actual, target in zip(corners, expected):
            self.assertPointAlmostEqual(actual, target)

    def test_pivot_point(self):
        for tx, ty, cx, cy in ((0, 0, 1, 2), (10.5, -4, 3.5, 9), (-8, 12, -2, -6)):
            self.assertEqual(pivot_point(tx, ty, cx, cy), (cx + tx, cy + ty))

    def test_edge_midpoints(self):
        self.assertEqual(edge_midpoints([(0, 0), (8, 0), (10, 6), (2, 6)]),
                         [(4, 0), (9, 3), (6, 6), (1, 3)])

    def test_pivot_drag_preserves_every_sampled_point(self):
        parameter_sets = [
            (4, -3, 0, 1, 10, 20),
            (-12, 8, 45, 1, -5, 7),
            (6, 11, 0, 2.5, 30, -4),
            (-9, 2, -30, 0.6, 12, 18),
        ]
        deltas = [(0, 0), (3.5, -2), (-7, 4.25)]
        sample_points = [(0, 0), (12.75, -8.5), (-100, 43)]
        for tx, ty, rotate, scale, cx, cy in parameter_sets:
            for dx, dy in deltas:
                new_cx, new_cy, new_tx, new_ty = pivot_drag_result(
                    tx, ty, rotate, scale, cx, cy, dx, dy)
                for x, y in sample_points:
                    old = forward_point(x, y, tx, ty, rotate, scale, cx, cy)
                    new = forward_point(x, y, new_tx, new_ty, rotate, scale, new_cx, new_cy)
                    self.assertPointAlmostEqual(old, new)

    def test_scale_from_drag(self):
        pivot = (1, 1)
        self.assertEqual(scale_from_drag(2, pivot, (4, 1), (7, 1)), 4)
        self.assertEqual(scale_from_drag(2, pivot, (5, 1), (3, 1)), 1)
        self.assertEqual(scale_from_drag(3, pivot, pivot, (10, 10)), 3)
        self.assertEqual(scale_from_drag(2, pivot, (4, 1), pivot, minimum=0.25), 0.25)

    def test_rotate_from_drag(self):
        pivot = (0, 0)
        # atan2(1, 0) - atan2(0, 1) is +90 degrees in this coordinate system.
        self.assertAlmostEqual(rotate_from_drag(10, pivot, (1, 0), (0, 1)), 100)
        self.assertAlmostEqual(rotate_from_drag(10, pivot, (1, 0), (-1, 0)), 190)
        self.assertAlmostEqual(rotate_from_drag(10, pivot, (1, 0), (1, 0)), 10)


if __name__ == "__main__":
    unittest.main()
