import unittest

from nodebased.handles2d import (
    box_corners,
    edge_midpoints,
    forward_point,
    pivot_drag_result,
    pivot_point,
    point_in_quad,
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

    def test_point_in_quad_for_axis_aligned_and_rotated_boxes(self):
        square = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.assertTrue(point_in_quad(50, 50, square))
        self.assertTrue(point_in_quad(0.001, 50, square))
        self.assertFalse(point_in_quad(-1, 50, square))
        rotated = [(75, -25), (75, 75), (25, 75), (25, -25)]
        self.assertTrue(point_in_quad(50, 25, rotated))
        self.assertTrue(point_in_quad(74.999, 25, rotated))
        self.assertFalse(point_in_quad(76, 25, rotated))


if __name__ == "__main__":
    unittest.main()


class AxisSkewSnapBoxTests(unittest.TestCase):
    """Lane 10 step T3 finish: the jack's one-axis scale, skew bar, rotate snap, and Crop's and Text's box geometry."""

    def test_split_scale_and_skew_match_the_render_matrix(self):
        import numpy as np
        from nodebased import handles2d
        from nodebased.imaging import Evaluator
        p = {"translate_x": 12.0, "translate_y": -7.0, "rotate": 25.0, "scale": [1.5, 0.75], "skew_x": 0.2, "skew_y": -0.1,
             "skew_order": "XY", "center_x": 40.0, "center_y": 30.0}
        matrix = Evaluator._transform_forward_matrix(p)
        for x, y in ((0, 0), (100, 0), (100, 60), (5, 77)):
            expected = matrix @ np.array([x, y, 1.0])
            got = handles2d.forward_point(x, y, p["translate_x"], p["translate_y"], p["rotate"], p["scale"],
                                          p["center_x"], p["center_y"], skew_x=0.2, skew_y=-0.1, skew_order="XY")
            self.assertAlmostEqual(got[0], expected[0], places=9)
            self.assertAlmostEqual(got[1], expected[1], places=9)

    def test_axis_scale_ignores_the_other_axis(self):
        from nodebased import handles2d
        self.assertAlmostEqual(handles2d.axis_scale_from_drag(1.0, (0, 0), 0.0, 0, (10, 0), (20, 50)), 2.0)
        self.assertAlmostEqual(handles2d.axis_scale_from_drag(1.0, (0, 0), 90.0, 0, (0, 10), (40, 30)), 3.0)

    def test_skew_bar_follows_the_pointer(self):
        from nodebased import handles2d
        skew = handles2d.skew_from_drag((0, 0), 0.0, (25, -100))
        end = handles2d.skew_bar_end((0, 0), 0.0, skew, 100)
        self.assertAlmostEqual(end[0] / -end[1], 0.25)

    def test_rotate_snaps_near_typical_values_only(self):
        from nodebased import handles2d
        self.assertEqual(handles2d.snap_rotate(88.5), 90.0)
        self.assertEqual(handles2d.snap_rotate(-44.0), -45.0)
        self.assertEqual(handles2d.snap_rotate(80.0), 80.0)

    def test_box_hit_and_drag(self):
        from nodebased import handles2d
        self.assertEqual(handles2d.box_hit((100, 50), 0, 0, 100, 100, 5), "e")
        self.assertEqual(handles2d.box_hit((0, 0), 0, 0, 100, 100, 5), "nw")
        self.assertEqual(handles2d.box_hit((50, 50), 0, 0, 100, 100, 5), "body")
        self.assertIsNone(handles2d.box_hit((50, 50), 0, 0, 100, 100, 5, body=False))
        self.assertEqual(handles2d.box_drag((0, 0, 100, 100), "nw", 30, 20), (30, 20, 70, 80))
        self.assertEqual(handles2d.box_drag((0, 0, 100, 100), "w", 150, 0), (99, 0, 1, 100))
        self.assertEqual(handles2d.box_drag((0, 0, 100, 100), "body", 5, 6), (5, 6, 100, 100))
