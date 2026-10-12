import unittest

import numpy as np

from nodebased.planar import fit_homography_ransac, project_points


class PlanarFitTests(unittest.TestCase):
    def test_projective_fit_recovers_card_corners_and_is_deterministic(self):
        source = np.array([[12, 15], [90, 12], [95, 78], [10, 82],
                           [30, 25], [65, 27], [70, 60], [25, 63]], dtype=float)
        known = np.array([[1.08, .04, 13], [-.025, .96, 7], [.0007, -.0004, 1.]])
        destination = project_points(known, source)
        first = fit_homography_ransac(source, destination, threshold=.01)
        second = fit_homography_ransac(source, destination, threshold=.01)
        np.testing.assert_allclose(project_points(first[0], source[:4]), destination[:4], atol=1e-7)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_allclose(first[0], second[0], atol=0)
        self.assertAlmostEqual(first[2], 0.0, places=8)

    def test_quarter_occlusion_outliers_do_not_pull_the_plane(self):
        rng = np.random.default_rng(41)
        source = rng.uniform([0, 0], [100, 80], (32, 2))
        known = np.array([[.93, -.08, 11], [.06, 1.04, -4], [-.0003, .0006, 1.]])
        destination = project_points(known, source)
        destination[:8] = rng.uniform(0, 100, (8, 2))
        matrix, inliers, rms = fit_homography_ransac(source, destination, threshold=.05)
        self.assertEqual(int(inliers.sum()), 24)
        self.assertLess(rms, 1e-7)
        corners = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=float)
        np.testing.assert_allclose(project_points(matrix, corners), project_points(known, corners), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
