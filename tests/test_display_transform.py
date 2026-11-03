"""Correctness for the display-transform speedups: threaded CPU chunking and the GPU path.

`nodebased.gpudisplay` needs a live QApplication (a QOffscreenSurface built without one
segfaults instead of raising, see the module docstring), so this file follows the same
`APP = QApplication.instance() or QApplication([])` convention as test_desktop.py /
test_timeline.py rather than being Qt-free like most color-pipeline tests would be.
"""
import os
import threading
import unittest
import unittest.mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtGui import QOffscreenSurface
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


class GpuDisplayOwnershipTests(unittest.TestCase):
    def test_non_owner_thread_never_builds_the_context(self):
        # A GUI-thread display call made before the first preview must not claim the context;
        # otherwise the preview worker would be locked onto CPU for the whole session.
        previous = (gpudisplay._surface, gpudisplay._owner_prefix)
        gpudisplay.reset_for_testing()
        gpudisplay.configure_surface(None, owner_thread_prefix='nodebased-preview')
        try:
            self.assertIsNone(gpudisplay.get_display())
            self.assertIsNone(gpudisplay._instance)
            self.assertIsNone(gpudisplay._failed_reason)
        finally:
            gpudisplay.configure_surface(*previous)
            gpudisplay.reset_for_testing()


class GpuDisplayWorkerSurfaceTests(unittest.TestCase):
    """11/2: a preview worker built its own QOffscreenSurface (create() waits for the GUI thread) while the GUI thread was blocked in
    agentloop.request waiting for that worker, and the full suite sat 40 minutes. A worker never waits on the GUI thread for a surface."""

    def _call_from_worker(self, name, getter, seconds=4):
        """Run `getter` on a thread called `name` while this (GUI) thread blocks without processing events. Qt's own
        QOffscreenSurface.create() waits for the GUI thread on some platforms and not on this offscreen one, so a stand-in
        surface that blocks off the GUI thread (until released) makes the wait reproducible anywhere."""
        gui = threading.current_thread()
        release = threading.Event()
        real = QOffscreenSurface

        class WaitingSurface(real):
            def create(self):
                if threading.current_thread() is not gui:
                    release.wait(30)
                return super().create()

        out = []
        worker = threading.Thread(target=lambda: out.append(getter()), name=name, daemon=True)
        with unittest.mock.patch('PySide6.QtGui.QOffscreenSurface', WaitingSurface):
            worker.start()
            worker.join(seconds)         # no event processing here: the GUI thread is "busy"
            returned = not worker.is_alive()
            release.set()
            worker.join(10)
        return returned, out

    def setUp(self):
        self._previous = (gpudisplay._surface, gpudisplay._owner_prefix, getattr(gpudisplay, '_convert_surface', None))
        gpudisplay.reset_for_testing()

    def tearDown(self):
        gpudisplay._surface, gpudisplay._owner_prefix = self._previous[:2]
        gpudisplay._convert_surface = self._previous[2]
        gpudisplay.reset_for_testing()

    def test_worker_without_a_premade_surface_returns_while_the_gui_thread_is_blocked(self):
        gpudisplay._surface = gpudisplay._owner_prefix = None
        returned, out = self._call_from_worker('worker-no-surface', gpudisplay.get_display)
        self.assertTrue(returned, 'get_display waited on the blocked GUI thread')
        self.assertEqual(out, [None])
        self.assertIsNone(gpudisplay._failed_reason, 'the card was not at fault: later GUI-thread calls must still work')

    def test_convert_thread_without_a_premade_surface_returns_while_the_gui_thread_is_blocked(self):
        gpudisplay._surface, gpudisplay._owner_prefix = None, 'nodebased-preview'
        gpudisplay._convert_surface = None
        returned, out = self._call_from_worker(gpudisplay.CONVERT_THREAD_PREFIX + '-test', gpudisplay.get_display)
        self.assertTrue(returned, 'the conversion thread waited on the blocked GUI thread')
        self.assertEqual(out, [None])

    def test_worker_with_a_premade_surface_still_gets_the_card(self):
        surface = QOffscreenSurface()
        surface.setFormat(gpudisplay.context_format())
        surface.create()
        if not surface.isValid():
            self.skipTest('no offscreen GL surface on this platform')
        gpudisplay._surface, gpudisplay._owner_prefix = surface, 'worker-premade'
        returned, out = self._call_from_worker('worker-premade', gpudisplay.get_display)
        self.assertTrue(returned)
        if out == [None]:
            self.skipTest(f'no GL context on this platform: {gpudisplay._failed_reason}')
        self.assertIsNotNone(out[0])
        gpudisplay.reset_for_testing()


if __name__ == '__main__':
    unittest.main()
