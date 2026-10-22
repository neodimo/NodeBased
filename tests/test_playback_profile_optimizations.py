import unittest

import numpy as np

from nodebased.imaging import Evaluator


class SmallRadiusBlurTests(unittest.TestCase):
    @staticmethod
    def reference(frame, radius, axis):
        n = frame.shape[axis]
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(frame, pad, mode="edge").astype(np.float64)
        zero_shape = list(padded.shape)
        zero_shape[axis] = 1
        summed = np.concatenate(
            [np.zeros(zero_shape, dtype=np.float64), np.cumsum(padded, axis=axis)], axis=axis
        )
        window = 2 * radius + 1
        hi = [slice(None)] * frame.ndim
        lo = [slice(None)] * frame.ndim
        hi[axis] = slice(window, window + n)
        lo[axis] = slice(0, n)
        return ((summed[tuple(hi)] - summed[tuple(lo)]) / window).astype(np.float32)

    def test_small_radius_path_matches_edge_padded_reference(self):
        rng = np.random.default_rng(917)
        frame = rng.random((17, 23, 4), dtype=np.float32)
        for radius in (1, 2, 4):
            for axis in (0, 1):
                with self.subTest(radius=radius, axis=axis):
                    actual = Evaluator._box_blur_axis(frame, radius, axis)
                    expected = self.reference(frame, radius, axis)
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(actual[[0, -1], :, :], expected[[0, -1], :, :],
                                               rtol=2e-6, atol=2e-7)

    def test_wide_radius_keeps_the_reference_path(self):
        rng = np.random.default_rng(27)
        frame = rng.random((13, 11, 4), dtype=np.float32)
        for axis in (0, 1):
            with self.subTest(axis=axis):
                np.testing.assert_array_equal(Evaluator._box_blur_axis(frame, 5, axis),
                                              self.reference(frame, 5, axis))


class TranslationTransformTests(unittest.TestCase):
    def test_separable_cubic_translation_matches_general_resampler(self):
        rng = np.random.default_rng(224)
        source = rng.random((19, 27, 4), dtype=np.float32)
        for clamp in (False, True):
            params = {"translate_x": 1.37, "translate_y": -2.21, "rotate": 0.0,
                      "scale": 1.0, "center_x": 0.0, "center_y": 0.0,
                      "scale_mode": "uniform", "skew_x": 0.0, "skew_y": 0.0,
                      "clamp": int(clamp), "black_outside": 0}
            forward = Evaluator._transform_forward_matrix(params)
            inverse = np.linalg.inv(forward)
            gx, gy = np.meshgrid(np.arange(source.shape[1], dtype=np.float32) + 0.5,
                                 np.arange(source.shape[0], dtype=np.float32) + 0.5)
            sx = inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2] - 0.5
            sy = inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2] - 0.5
            expected = Evaluator._resample(source, sx, sy, "cubic", clamp=clamp)
            actual = Evaluator._transform(source, 1.37, -2.21, 0.0, 1.0, 0.0, 0.0,
                                         "cubic", params=params)
            with self.subTest(clamp=clamp):
                # Separating the tensor-product sum changes float32 accumulation order by less
                # than 1e-6 while retaining the same Catmull-Rom weights and edge treatment.
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
