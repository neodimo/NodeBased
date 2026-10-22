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
            for translate_y in (0.0, -2.21):
                params = {"translate_x": 1.37, "translate_y": translate_y, "rotate": 0.0,
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
                actual = Evaluator._transform(source, 1.37, translate_y, 0.0, 1.0, 0.0, 0.0,
                                             "cubic", params=params)
                with self.subTest(clamp=clamp, translate_y=translate_y):
                    # Separating the tensor-product sum changes float32 accumulation order by less
                    # than 1e-6 while retaining the same Catmull-Rom weights and edge treatment.
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


class ColorCorrectIdentityTests(unittest.TestCase):
    def test_default_controls_return_the_input_without_pixel_work(self):
        from nodebased.core import SPECS
        frame = np.random.default_rng(802).random((7, 9, 4), dtype=np.float32)
        self.assertIs(Evaluator._color_correct(frame, SPECS["ColorCorrect"]["params"]), frame)

    def test_nondefault_control_keeps_the_color_correct_path(self):
        from nodebased.core import SPECS
        frame = np.random.default_rng(803).random((7, 9, 4), dtype=np.float32)
        params = dict(SPECS["ColorCorrect"]["params"], saturation=0.8)
        result = Evaluator._color_correct(frame, params)
        self.assertIsNot(result, frame)
        self.assertFalse(np.array_equal(result, frame))


class FullMixFastPathTests(unittest.TestCase):
    def test_unmasked_full_mix_returns_the_filtered_result(self):
        source = np.zeros((3, 4, 4), np.float32)
        filtered = np.ones_like(source)
        result = Evaluator._apply_mask_mix(source, filtered, None, 1.0)
        self.assertIs(result, filtered)

    def test_masked_full_mix_still_applies_the_mask(self):
        source = np.zeros((2, 2, 4), np.float32)
        filtered = np.ones_like(source)
        mask = np.zeros_like(source)
        result = Evaluator._apply_mask_mix(source, filtered, mask, 1.0)
        np.testing.assert_array_equal(result, source)


class GradeInPlaceTests(unittest.TestCase):
    def test_reduced_temporaries_preserve_grade_pixels(self):
        source = np.random.default_rng(804).random((11, 13, 4), dtype=np.float32)
        before = source.copy()
        for params in ({"exposure": 0.3, "multiply": 1.2, "offset": 0.0},
                       {"exposure": -0.2, "multiply": 0.8, "offset": 0.1}):
            expected = source.copy()
            expected[..., :3] = (source[..., :3] * (2.0 ** params["exposure"])
                                 * params["multiply"] + params["offset"] * source[..., 3:4])
            actual = Evaluator._grade(source, params)
            with self.subTest(params=params):
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, before)


class CheckerGenerationTests(unittest.TestCase):
    def test_checker_outputs_match_on_full_frame_and_tiled_source_paths(self):
        from nodebased.tileexec import _full_image
        params = {"width": 9, "height": 7, "size": 3}
        expected_pattern = ((np.arange(9)[None, :] // 3 + np.arange(7)[:, None] // 3) % 2)
        expected = np.empty((7, 9, 4), np.float32)
        expected[..., :3] = (0.06 + expected_pattern.astype(np.float32) * 0.24)[..., None]
        expected[..., 3] = 1.0
        actual = Evaluator._kernel("Checker", params, [])
        tiled_source = _full_image({"type": "Checker", "params": params}, frame=1)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(tiled_source, expected)

    def test_checker_preserves_fractional_animated_size_rounding(self):
        params = {"width": 9, "height": 7, "size": 3.5}
        yy, xx = np.ogrid[:7, :9]
        pattern = ((xx // 3.5 + yy // 3.5) % 2).astype(np.float32)
        expected = np.empty((7, 9, 4), np.float32)
        expected[..., :3] = (0.06 + pattern * 0.24)[..., None]
        expected[..., 3] = 1.0
        np.testing.assert_array_equal(Evaluator._kernel("Checker", params, []), expected)


if __name__ == "__main__":
    unittest.main()
