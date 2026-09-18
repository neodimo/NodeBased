"""Real offscreen-GL coverage for the direct viewport renderer."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat

from nodebased import gpudisplay
from nodebased.color import display_rgb


APP = QApplication.instance() or QApplication([])


class ViewportRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 0)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        cls.surface = QOffscreenSurface()
        cls.surface.setFormat(fmt)
        cls.surface.create()
        cls.context = QOpenGLContext()
        cls.context.setFormat(fmt)
        if not cls.surface.isValid() or not cls.context.create():
            raise unittest.SkipTest('OpenGL 4.0 offscreen context unavailable')

    def setUp(self):
        if not self.context.makeCurrent(self.surface):
            self.skipTest('could not make offscreen context current')
        self.renderer = gpudisplay.ViewportRenderer()

    def tearDown(self):
        self.renderer.release()
        self.context.doneCurrent()

    def _render(self, frame, view='sRGB', exposure=0.0, channel='RGB',
                background='black'):
        height, width = frame.shape[:2]
        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setInternalTextureFormat(0x8814)  # GL_RGBA32F
        fbo = QOpenGLFramebufferObject(width, height, fmt)
        self.assertTrue(fbo.isValid())
        fbo.bind()
        self.renderer.draw(frame, view, exposure, channel, background,
                           ((0, 0), (width, 0), (0, height), (width, height)),
                           (width, height))
        raw = bytearray(width * height * 4 * 4)
        self.context.extraFunctions().glReadPixels(0, 0, width, height, 0x1908,
                                                    0x1406, raw)
        fbo.release()
        # glReadPixels is bottom-up; convert to this project's top-down image convention.
        return np.frombuffer(raw, dtype=np.float32).reshape(height, width, 4)[::-1].copy()

    def test_draw_matches_to_qimage_reference_for_srgb_view(self):
        frame = np.array([
            [[1.4, 0.2, 0.1, 1.0], [0.1, 0.8, 0.2, 0.5], [-0.2, 0.4, 1.6, 0.2]],
            [[0.0, 0.0, 0.0, 0.0], [0.7, 0.3, 0.1, 0.75], [2.0, -0.2, 0.5, 1.0]],
        ], dtype=np.float32)
        result = self._render(frame)
        alpha = frame[..., 3:4]
        weight = np.clip(alpha, 0, 1)
        straight = np.divide(frame[..., :3], weight, out=np.zeros_like(frame[..., :3]),
                             where=weight > 1e-8)
        expected = display_rgb(straight, 'sRGB') * weight
        expected = np.concatenate((np.clip(expected, 0, 1), np.ones_like(alpha)), axis=2)
        np.testing.assert_allclose(result, expected, atol=3e-4, rtol=3e-4)

    def test_orientation_preserves_top_down_input_rows(self):
        frame = np.zeros((8, 8, 4), dtype=np.float32)
        frame[:4, :, 0] = 1.0
        frame[4:, :, 1] = 1.0
        frame[..., 3] = 1.0
        result = self._render(frame, view='sRGB')
        self.assertGreater(float(result[:4, :, 0].mean()), 0.8)
        self.assertLess(float(result[:4, :, 1].mean()), 0.1)
        self.assertGreater(float(result[4:, :, 1].mean()), 0.8)
        self.assertLess(float(result[4:, :, 0].mean()), 0.1)

    def test_channel_solo_a_bypasses_exposure_and_view_transform(self):
        frame = np.zeros((2, 2, 4), dtype=np.float32)
        frame[..., :3] = (3.0, 0.1, 0.7)
        frame[..., 3] = ((0.3, 0.6), (0.3, 0.6))
        result = self._render(frame, exposure=4.0, channel='A', view='sRGB')
        expected = np.concatenate((np.repeat(frame[..., 3:4], 3, axis=2),
                                   np.ones_like(frame[..., 3:4])), axis=2)
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)

    def test_background_checker_matches_partial_alpha_compositing(self):
        frame = np.zeros((8, 32, 4), dtype=np.float32)
        frame[:, 16:, :3] = (0.2, 0.3, 0.4)
        frame[:, 16:, 3] = 1.0
        result = self._render(frame, background='checker')
        level0, level1 = gpudisplay._checker_levels('sRGB')
        self.assertTrue(np.allclose(result[:, :16, :3], level0, atol=3e-4) |
                        np.allclose(result[:, :16, :3], level1, atol=3e-4))
        self.assertTrue(np.allclose(result[:, 16:, :3], display_rgb(
            np.full((8, 16, 3), (0.2, 0.3, 0.4), np.float32), 'sRGB'), atol=3e-4))

    def test_unavailable_after_a_failure_reports_status_via_available(self):
        with self.assertRaises(gpudisplay.GpuUnavailable):
            self.renderer.draw(np.zeros((2, 2, 4), np.float32), 'not-a-real-view',
                               0.0, 'RGB', 'black', ((0, 0), (2, 0), (0, 2), (2, 2)), (2, 2))
        self.assertFalse(self.renderer.available())


if __name__ == '__main__':
    unittest.main()
