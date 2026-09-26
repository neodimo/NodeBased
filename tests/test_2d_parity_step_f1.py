import unittest

import numpy as np

from nodebased.imaging import Evaluator


class MatrixAndLaplacianTests(unittest.TestCase):
    def test_identity_matrix_preserves_pixels(self):
        image = np.arange(60, dtype=np.float32).reshape(3, 5, 4)
        params = {f"weight{i}": float(i == 4) for i in range(9)}
        np.testing.assert_array_equal(Evaluator._matrix(image, params), image)

    def test_known_kernel_on_checker_values(self):
        image = np.zeros((3, 3, 4), np.float32)
        image[1, 1] = 1.0
        params = {f"weight{i}": 0.0 for i in range(9)}
        params["weight4"] = 2.0
        params["weight1"] = -1.0
        result = Evaluator._matrix(image, params)
        self.assertEqual(float(result[1, 1, 0]), 2.0)
        self.assertEqual(float(result[1, 0, 0]), 0.0)

    def test_laplacian_identity_on_flat_image(self):
        image = np.full((4, 5, 4), 0.5, np.float32)
        result = Evaluator._matrix(image, {}, laplacian=True)
        np.testing.assert_array_equal(result[..., :3], 0.0)


if __name__ == "__main__":
    unittest.main()
