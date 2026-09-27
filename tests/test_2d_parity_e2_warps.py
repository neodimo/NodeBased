import unittest

import numpy as np

from nodebased.warps import displacement_field, grid_controls


class WarpFieldTests(unittest.TestCase):
    def test_identical_controls_make_an_identity_field(self):
        points = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
        x, y = np.meshgrid(np.arange(24), np.arange(24))
        field = displacement_field(points, points, x, y)
        np.testing.assert_allclose(field, 0.0, atol=1e-7)

    def test_grid_control_displacement_is_exact_and_decays(self):
        src = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
        dst = src.copy()
        dst[0] += (3.0, -2.0)
        x = np.array([[3.0, 40.0]])
        y = np.array([[-2.0, 40.0]])
        field = displacement_field(src, dst, x, y, radius=8)
        np.testing.assert_allclose(field[0, 0], (3.0, -2.0), atol=1e-5)
        self.assertLess(float(np.linalg.norm(field[0, 1])), 1e-4)

    def test_grid_controls_reject_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching MxNx2"):
            grid_controls(np.zeros((3, 3, 2)), np.zeros((4, 3, 2)))


if __name__ == "__main__":
    unittest.main()
