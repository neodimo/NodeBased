import unittest
from unittest.mock import patch

import numpy as np

from nodebased.imaging import Evaluator, Region


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

    def test_wide_radius_prefix_sum_does_not_concatenate_a_second_full_buffer(self):
        frame = np.random.default_rng(29).random((13, 17, 4), dtype=np.float32)
        expected = self.reference(frame, 5, axis=1)
        with patch("nodebased.imaging.np.concatenate", side_effect=AssertionError(
                "wide blur should write its prefix sum into one buffer")):
            actual = Evaluator._box_blur_axis(frame, 5, axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_small_radius_path_does_not_zero_fill_its_accumulator(self):
        frame = np.random.default_rng(28).random((13, 17, 4), dtype=np.float32)
        expected = self.reference(frame, 2, axis=1)
        with patch("nodebased.imaging.np.zeros_like", side_effect=AssertionError(
                "small-radius blur should seed from its first edge-padded sample")):
            actual = Evaluator._box_blur_axis(frame, 2, axis=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


class TranslationTransformTests(unittest.TestCase):
    def test_integer_translation_copies_pixels_without_resampling(self):
        rng = np.random.default_rng(1207)
        source = rng.random((13, 17, 4), dtype=np.float32)
        for dx, dy in ((4, -3), (-5, 2), (0, 0)):
            for filt in ("nearest", "bilinear", "cubic"):
                params = {"translate_x": dx, "translate_y": dy, "rotate": 0.0,
                          "scale": 1.0, "center_x": 0.0, "center_y": 0.0,
                          "scale_mode": "uniform", "skew_x": 0.0, "skew_y": 0.0,
                          "clamp": 0, "black_outside": 0}
                forward = Evaluator._transform_forward_matrix(params)
                inverse = np.linalg.inv(forward)
                gx, gy = np.meshgrid(np.arange(source.shape[1], dtype=np.float32) + 0.5,
                                     np.arange(source.shape[0], dtype=np.float32) + 0.5)
                sx = inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2] - 0.5
                sy = inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2] - 0.5
                expected = Evaluator._resample(source, sx, sy, filt)
                actual = Evaluator._transform(source, dx, dy, 0.0, 1.0, 0.0, 0.0, filt,
                                              params=params)
                with self.subTest(dx=dx, dy=dy, filter=filt):
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)

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

    def test_fractional_horizontal_fast_path_matches_the_full_animation(self):
        rng = np.random.default_rng(225)
        source = rng.random((19, 27, 4), dtype=np.float32)
        for frame in range(1, 49, 3):
            translate_x = -18.0 + 36.0 * (frame - 1) / 47.0
            params = {"translate_x": translate_x, "translate_y": 0.0, "rotate": 0.0,
                      "scale": 1.0, "center_x": 0.0, "center_y": 0.0,
                      "scale_mode": "uniform", "skew_x": 0.0, "skew_y": 0.0,
                      "clamp": 0, "black_outside": 0}
            forward = Evaluator._transform_forward_matrix(params)
            inverse = np.linalg.inv(forward)
            gx, gy = np.meshgrid(np.arange(source.shape[1], dtype=np.float32) + 0.5,
                                 np.arange(source.shape[0], dtype=np.float32) + 0.5)
            sx = inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2] - 0.5
            sy = inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2] - 0.5
            expected = Evaluator._resample(source, sx, sy, "cubic")
            with patch("nodebased.imaging.np.take", side_effect=AssertionError(
                    "horizontal translation should use contiguous tap slices")):
                actual = Evaluator._transform(source, translate_x, 0.0, 0.0, 1.0,
                                              0.0, 0.0, "cubic", params=params)
            with self.subTest(frame=frame, translate_x=translate_x):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

    def test_horizontal_transform_roi_skips_repeated_full_frame_resampling(self):
        rng = np.random.default_rng(226)
        source = rng.random((19, 27, 4), dtype=np.float32)
        src_box = Region(0, 0, 27, 19)
        # Blur asks its upstream Transform for a 2-pixel halo on every side. That made the
        # vertical coordinate vector longer than the source and disabled the contiguous x path.
        dst_box = Region(-2, -2, 31, 23)
        params = {"translate_x": 1.37, "translate_y": 0.0, "rotate": 0.0,
                  "scale": 1.0, "center_x": 0.0, "center_y": 0.0,
                  "scale_mode": "uniform", "skew_x": 0.0, "skew_y": 0.0,
                  "clamp": 0, "black_outside": 0}
        forward = Evaluator._transform_forward_matrix(params)
        inverse = np.linalg.inv(forward)
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float32) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float32) + dst_box.y + 0.5)
        sx = inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2] - 0.5 - src_box.x
        sy = inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2] - 0.5 - src_box.y
        expected = Evaluator._resample(source, sx, sy, "cubic")
        calls = []
        original = Evaluator._resample_cubic_separable

        def count_resamples(*args, **kwargs):
            calls.append((args[0].shape, args[1].shape, args[2].shape))
            return original(*args, **kwargs)

        with patch("nodebased.imaging.Evaluator._resample_cubic_separable",
                   side_effect=count_resamples), patch(
                       "nodebased.imaging.np.take", side_effect=AssertionError(
                           "horizontal ROI transform should use contiguous cubic slices")):
            actual = Evaluator._transform(source, 1.37, 0.0, 0.0, 1.0, 0.0, 0.0, "cubic",
                                          src_box=src_box, dst_box=dst_box, params=params)
        self.assertEqual(calls, [((19, 27, 4), (31,), (23,))])
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


class MaskedMergeMemoryTrafficTests(unittest.TestCase):
    def test_over_reuses_output_buffers_and_preserves_pixels(self):
        rng = np.random.default_rng(806)
        a = rng.random((7, 9, 4), dtype=np.float32)
        b = rng.random((7, 9, 4), dtype=np.float32)
        mask = rng.random((7, 9, 4), dtype=np.float32)
        expected = a * mask[..., 3:4] + b * (1.0 - a[..., 3:4] * mask[..., 3:4])
        with patch("nodebased.imaging.np.multiply", wraps=np.multiply) as multiply, patch(
                "nodebased.imaging.np.add", wraps=np.add) as add:
            actual = Evaluator._merge_gated("over", a, b, 1.0, mask)
        self.assertEqual(multiply.call_count, 3)
        self.assertEqual(add.call_count, 1)
        np.testing.assert_array_equal(actual, expected)


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

    def test_neutral_multiply_does_not_stream_the_rgb_frame_again(self):
        source = np.random.default_rng(805).random((11, 13, 4), dtype=np.float32)
        with patch("nodebased.imaging.np.multiply", wraps=np.multiply) as multiply:
            actual = Evaluator._grade(source, {"exposure": 0.3, "multiply": 1.0, "offset": 0.0})
        expected = source.copy()
        expected[..., :3] *= 2.0 ** 0.3
        self.assertEqual(multiply.call_count, 1)
        np.testing.assert_array_equal(actual, expected)


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


class PlaybackDiskSpillTests(unittest.TestCase):
    class DiskSpy:
        def __init__(self):
            self.puts = []

        def put_raster(self, digest, raster):
            self.puts.append((digest, raster))

    def test_playback_policy_drops_evicted_frames_without_disk_io(self):
        from nodebased.raster import Raster
        disk = self.DiskSpy()
        frame = Raster(np.ones((2, 2, 4), np.float32))
        evaluator = Evaluator(cache_bytes=frame.nbytes, disk=disk)
        evaluator._store("one", frame)
        evaluator.disk_spills_enabled = False
        evaluator._store("two", frame)
        self.assertEqual(disk.puts, [])
        evaluator.disk_spills_enabled = True
        evaluator._store("three", frame)
        self.assertEqual([item[0] for item in disk.puts], ["two"])


if __name__ == "__main__":
    unittest.main()
