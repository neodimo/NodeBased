"""Correctness for the display-transform speedups: threaded CPU chunking and the GPU path.

`nodebased.gpudisplay` needs a live QApplication (a QOffscreenSurface built without one
segfaults instead of raising, see the module docstring), so this file follows the same
`APP = QApplication.instance() or QApplication([])` convention as test_desktop.py /
test_timeline.py rather than being Qt-free like most color-pipeline tests would be.
"""
import os
import threading
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
import numpy as np

from nodebased import gpudisplay
from nodebased.color import apply_threaded, display_processor, display_rgb

APP = QApplication.instance() or QApplication([])


def hdr_gamut_image(height=64, width=96):
    """Wide-gamut/HDR test data: >1 highlights, negatives, saturated ACEScg primaries,
    near-zero and exactly-zero pixels -- the value classes a naive clamp-first transform
    or an 8-bit-only GPU path would get wrong."""
    rng = np.random.default_rng(20260914)
    image = rng.random((height, width, 3), dtype=np.float32) * 2.0 - 0.4
    # Saturated ACEScg primaries and secondaries in fixed blocks, plus pure black/near-zero.
    stripes = {
        0: (4.0, -0.05, -0.05), 1: (-0.05, 4.0, -0.05), 2: (-0.05, -0.05, 4.0),
        3: (6.0, 6.0, -0.1), 4: (0.0, 0.0, 0.0), 5: (1e-7, 1e-7, 1e-7),
        6: (16.0, 0.02, 0.02), 7: (-0.3, -0.3, -0.3),
    }
    for row, rgb in stripes.items():
        image[row % height, :, :] = rgb
    return image


class ApplyThreadedTests(unittest.TestCase):
    def test_matches_single_call_reference_exactly(self):
        cpu_processor = display_processor('ACES 2.0')
        for shape in [(1, 1, 3), (3, 5, 3), (37, 53, 3), (128, 256, 3)]:
            with self.subTest(shape=shape):
                rng = np.random.default_rng(1)
                image = rng.random(shape, dtype=np.float32) * 3 - 0.5
                reference = image.copy()
                cpu_processor.applyRGB(reference)
                threaded = image.copy()
                apply_threaded(cpu_processor, threaded)
                np.testing.assert_array_equal(threaded, reference)

    def test_display_processor_is_cached(self):
        self.assertIs(display_processor('sRGB'), display_processor('sRGB'))
        self.assertIs(display_processor('ACES 2.0'), display_processor('ACES 2.0'))


class DisplayRgbTests(unittest.TestCase):
    def test_linear_is_passthrough(self):
        image = np.array([[[2.0, -1.0, 0.5]]], dtype=np.float32)
        self.assertIs(display_rgb(image, 'Linear'), image)

    def test_unknown_view_raises(self):
        with self.assertRaises(ValueError):
            display_rgb(np.zeros((1, 1, 3), np.float32), 'Nonexistent View')

    def test_srgb_and_aces_match_cpu_reference(self):
        image = hdr_gamut_image()
        for view in ('sRGB', 'ACES 2.0'):
            with self.subTest(view=view):
                result = display_rgb(image, view)
                reference = image.copy()
                display_processor(view).applyRGB(reference)
                np.testing.assert_allclose(result, reference, atol=2e-4)

    def test_forcing_cpu_still_produces_correct_pixels(self):
        previous = os.environ.get('NODEBASED_DISPLAY_GPU')
        os.environ['NODEBASED_DISPLAY_GPU'] = '0'
        try:
            image = hdr_gamut_image()
            result = display_rgb(image, 'ACES 2.0')
            reference = image.copy()
            display_processor('ACES 2.0').applyRGB(reference)
            np.testing.assert_array_equal(result, reference)
        finally:
            if previous is None:
                del os.environ['NODEBASED_DISPLAY_GPU']
            else:
                os.environ['NODEBASED_DISPLAY_GPU'] = previous


class GpuDisplayTests(unittest.TestCase):
    """Every GPU test skips cleanly when this process cannot get a working GL context --
    exactly the QT_QPA_PLATFORM=offscreen-on-a-GPU-less-runner case the module falls back
    for. See docs/BENCHMARKS-v0.16-display.md for where this was actually exercised on
    real hardware."""

    def setUp(self):
        gpudisplay.reset_for_testing()
        self.addCleanup(gpudisplay.reset_for_testing)
        self.display = gpudisplay.get_display(force=True)
        if self.display is None:
            self.skipTest(f'no GPU context available: {gpudisplay.status()}')

    def test_gpu_matches_cpu_within_one_8bit_code_value(self):
        image = hdr_gamut_image()
        for view in ('sRGB', 'ACES 2.0'):
            with self.subTest(view=view):
                gpu_result = self.display.render(image, view)
                cpu_result = image.copy()
                display_processor(view).applyRGB(cpu_result)

                def quantize(rgb):
                    return np.clip(np.round(np.clip(rgb, 0, 1) * 255), 0, 255).astype(np.int32)

                gpu_codes = quantize(gpu_result)
                cpu_codes = quantize(cpu_result)
                max_error = int(np.max(np.abs(gpu_codes - cpu_codes)))
                self.assertLessEqual(max_error, 1,
                                      f'{view}: max 8-bit code-value error {max_error}')

    def test_odd_size_and_repeat_calls_are_stable(self):
        for shape in [(1, 1), (5, 3), (37, 53)]:
            with self.subTest(shape=shape):
                image = hdr_gamut_image(*shape)
                first = self.display.render(image, 'ACES 2.0')
                second = self.display.render(image, 'ACES 2.0')
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, (shape[0], shape[1], 3))

    def test_wrong_thread_raises_gpu_unavailable(self):
        error = []

        def call_from_other_thread():
            try:
                self.display.render(np.zeros((2, 2, 3), np.float32), 'sRGB')
            except gpudisplay.GpuUnavailable as exc:
                error.append(exc)

        thread = threading.Thread(target=call_from_other_thread)
        thread.start()
        thread.join(timeout=10)
        self.assertEqual(len(error), 1)

    def test_status_reports_gpu_when_active(self):
        self.display.render(np.zeros((2, 2, 3), np.float32), 'sRGB')
        self.assertEqual(gpudisplay.status(), 'GPU')

    def test_forced_gpu_failure_falls_back_to_correct_cpu_pixels(self):
        """A GPU failure mid-session (driver hiccup, lost context, anything) must fall
        back cleanly rather than crash or return wrong pixels -- simulated here since
        provoking a real driver fault is not reproducible in a test."""
        def always_fails(rgb, view):
            raise gpudisplay.GpuUnavailable('simulated mid-session failure')

        original = self.display.render
        self.display.render = always_fails
        try:
            image = hdr_gamut_image()
            result = display_rgb(image, 'ACES 2.0')
            reference = image.copy()
            display_processor('ACES 2.0').applyRGB(reference)
            np.testing.assert_array_equal(result, reference)
        finally:
            self.display.render = original


class GpuDisplayDisabledTests(unittest.TestCase):
    def test_env_var_disables_gpu(self):
        previous = os.environ.get('NODEBASED_DISPLAY_GPU')
        os.environ['NODEBASED_DISPLAY_GPU'] = '0'
        gpudisplay.reset_for_testing()
        try:
            self.assertIsNone(gpudisplay.get_display(force=True))
            self.assertEqual(gpudisplay.status(), 'CPU (forced)')
        finally:
            if previous is None:
                del os.environ['NODEBASED_DISPLAY_GPU']
            else:
                os.environ['NODEBASED_DISPLAY_GPU'] = previous
            gpudisplay.reset_for_testing()


if __name__ == '__main__':
    unittest.main()
