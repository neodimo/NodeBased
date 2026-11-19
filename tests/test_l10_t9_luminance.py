"""Lane 10 step T9 finish: the Viewer's luminance display (Nuke 17.0 User Guide, Keyboard Shortcuts, 2D Compositing Viewer:
"Y Toggles between the Luminance channel and RBG"). The picture shows the Rec. 709 luminance of the colour in grey, through the
same gain, view transform and alpha as RGB, and Y on the Viewer flips between it and RGB."""
import tests.isolation  # noqa: F401

import unittest

import numpy as np

from tests.test_desktop import APP  # noqa: F401
from nodebased.imaging import luminance_frame, to_qimage


def _pixel(image, x=0, y=0):
    colour = image.pixelColor(x, y)
    return colour.red(), colour.green(), colour.blue()


class LuminanceDisplayTests(unittest.TestCase):
    def test_luminance_is_rec709_with_alpha_kept(self):
        frame = np.zeros((2, 2, 4), np.float32)
        frame[..., :3] = (0.2, 0.5, 0.9)
        frame[..., 3] = 0.75
        luma = luminance_frame(frame)
        expected = 0.2126 * 0.2 + 0.7152 * 0.5 + 0.0722 * 0.9
        np.testing.assert_allclose(luma[..., :3], expected, rtol=1e-6)
        np.testing.assert_array_equal(luma[..., 3], frame[..., 3])

    def test_y_display_is_grey_and_matches_rgb_of_the_luminance(self):
        frame = np.zeros((64, 96, 4), np.float32)
        frame[..., 0], frame[..., 1], frame[..., 2], frame[..., 3] = 0.8, 0.1, 0.3, 1.0
        for view in ("sRGB", "Linear"):
            with self.subTest(view=view):
                shown = to_qimage(frame, 0.5, "Y", view=view)
                r, g, b = _pixel(shown, 10, 10)
                self.assertEqual(r, g)
                self.assertEqual(g, b)
                reference = to_qimage(luminance_frame(frame), 0.5, "RGB", view=view)
                self.assertEqual((r, g, b), _pixel(reference, 10, 10))
                self.assertNotEqual(_pixel(to_qimage(frame, 0.5, "RGB", view=view), 10, 10), (r, g, b))

    def test_y_key_toggles_the_viewer_between_luminance_and_rgb(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import nuke_panel_interactions as interactions
        bench = interactions.Bench()
        try:
            passed, detail = interactions.PROBES["viewer.luminance_y"](bench)
            self.assertTrue(passed, detail)
            self.assertIn("Y", bench.window.channels.buttons)
        finally:
            bench.close()


if __name__ == "__main__":
    unittest.main()
