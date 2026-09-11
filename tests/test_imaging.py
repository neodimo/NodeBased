import os
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
from nodebased.core import Dispatcher, demo_document
from nodebased.imaging import Evaluator, Cancelled, read_image, to_qimage, write_png
from nodebased.color import display_rgb


class ImageTests(unittest.TestCase):
    def test_premultiplied_over(self):
        a = np.array([[[0.5, 0, 0, 0.5]]], np.float32)
        b = np.array([[[0, 0, 1, 1]]], np.float32)
        np.testing.assert_allclose(Evaluator._kernel('Merge', {'mix': 1}, [a, b]), [[[0.5, 0, 0.5, 1]]])
        np.testing.assert_allclose(Evaluator._kernel('Merge', {'mix': 0}, [a, b]), b)

    def test_grade_preserves_hdr_negative_and_alpha(self):
        src = np.array([[[2, -1, 0.2, 0.5]]], np.float32)
        result = Evaluator._kernel('Grade', {'exposure': 1, 'multiply': 1, 'offset': 0.2}, [src])
        np.testing.assert_allclose(result, [[[4.1, -1.9, 0.5, 0.5]]], rtol=1e-6)
        np.testing.assert_array_equal(src, np.array([[[2, -1, 0.2, 0.5]]], np.float32))

    def test_translation_does_not_wrap(self):
        src = np.ones((2, 3, 4), np.float32)
        result = Evaluator._kernel('Transform', {'translate_x': 1.0, 'translate_y': -1.0, 'rotate': 0.0,
                                                  'scale': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'filter': 'nearest'}, [src])
        self.assertEqual(result[:, 0].sum(), 0)
        self.assertEqual(result[1].sum(), 0)
        self.assertEqual(result[0, 1:].sum(), 8)

    def test_cache_reuse_and_upstream_invalidation(self):
        d = Dispatcher(demo_document())
        e = Evaluator()
        original = e.evaluate(d.document)
        misses = e.misses
        d.execute({'op': 'move', 'id': 'grade', 'pos': [12, 40]})
        self.assertIs(e.evaluate(d.document), original)
        self.assertEqual(e.misses, misses)
        d.execute({'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': 2})
        self.assertFalse(np.array_equal(e.evaluate(d.document), original))
        self.assertFalse(original.flags.writeable)

    def test_cache_retention_budget_and_cancel(self):
        # The budget bounds the retained *set*, not one entry: a single result larger than the whole
        # budget is still kept, because discarding it left the cache inert at 4K and above.
        e = Evaluator(cache_bytes=100)
        doc = demo_document()
        e.evaluate(doc)
        self.assertEqual(len(e.cache), 1)
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(Cancelled):
            e.evaluate(doc, cancel=cancel)

    def test_png_roundtrip_and_changed_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'image.png'
            src = np.array([[[0.1, 0.2, 0.3, 0.5], [0, 0, 0, 0]]], np.float32)
            write_png(path, src)
            np.testing.assert_allclose(read_image(str(path)), src, atol=0.004)
            d = Dispatcher()
            d.execute({'op': 'create', 'type': 'Read', 'id': 'r', 'params': {'path': str(path)}})
            e = Evaluator()
            first = e.evaluate(d.document, 'r')
            before = path.stat().st_mtime_ns
            write_png(path, np.ones((1, 2, 4), np.float32))
            os.utime(path, ns=(before + 1000000000, before + 1000000000))
            self.assertFalse(np.array_equal(first, e.evaluate(d.document, 'r')))

    def test_transparent_regions_display_over_pure_black_by_default(self):
        # Transparent span covers two 16px checker cells, so both phases are observable.
        src = np.zeros((1, 48, 4), np.float32)
        src[0, :16, :] = 1
        black = to_qimage(src)
        for x in range(16, 48):
            self.assertEqual(black.pixelColor(x, 0).getRgb()[:3], (0, 0, 0),
                             f'transparent pixel {x} is not pure black')
        checker = to_qimage(src, background='checker')
        tinted = {checker.pixelColor(x, 0).red() for x in range(16, 48)}
        self.assertGreater(len(tinted), 1, 'checker background should still alternate when asked for')
        self.assertNotIn(0, tinted)

    def test_view_transform_runs_on_unpremultiplied_colour(self):
        """A semi-transparent pixel must display the colour the exporter writes for it.

        The view transform is nonlinear, so applying it to premultiplied RGB bends every
        partially transparent pixel. `write_png` already unpremultiplies before encoding,
        which makes the exporter the negative control: if the viewer disagrees with it on
        the same pixel, the viewer is wrong. Reported by DiMo 2026-09-10 alongside the
        request to get the ACES pipeline right.
        """
        for view in ('sRGB', 'ACES 2.0'):
            for alpha in (0.25, 0.5, 0.75):
                src = np.zeros((1, 1, 4), np.float32)
                src[0, 0] = [0.18 * alpha, 0.18 * alpha, 0.18 * alpha, alpha]
                shown = to_qimage(src, view=view).pixelColor(0, 0).red()
                # Straight colour through the same view, then composited over black.
                straight = display_rgb(np.full((1, 1, 3), 0.18, np.float32), view)[0, 0, 0]
                expected = round(float(np.clip(straight, 0, 1)) * alpha * 255 + 0.5)
                self.assertAlmostEqual(
                    shown, expected, delta=1,
                    msg=f'{view} at alpha {alpha}: viewer {shown}, composited straight {expected}')

    def test_opaque_pixels_are_unaffected_by_the_unpremultiply(self):
        """Guard the common path: alpha 1 must match a plain view transform."""
        src = np.zeros((1, 3, 4), np.float32)
        src[0, :, :3] = [0.05, 0.18, 0.6]
        src[0, :, 3] = 1.0
        for view in ('sRGB', 'ACES 2.0', 'Linear'):
            shown = to_qimage(src, view=view).pixelColor(1, 0).getRgb()[:3]
            reference = display_rgb(np.array([[[0.05, 0.18, 0.6]]], np.float32), view)[0, 0]
            expected = tuple(int(np.clip(c, 0, 1) * 255 + 0.5) for c in reference)
            self.assertEqual(shown, expected, f'{view} changed an opaque pixel')

    def test_fully_transparent_pixels_stay_black_under_every_view(self):
        """Unpremultiplying must not divide by zero into garbage at alpha 0."""
        src = np.zeros((1, 2, 4), np.float32)
        for view in ('sRGB', 'ACES 2.0', 'Linear'):
            image = to_qimage(src, view=view)
            self.assertEqual(image.pixelColor(0, 0).getRgb()[:3], (0, 0, 0), view)

    def test_merge_rejects_format_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'matching formats'):
            Evaluator._kernel('Merge', {'mix': 1}, [np.ones((2, 2, 4)), np.ones((3, 2, 4))])

    def test_blur_spreads_an_impulse_and_zero_radius_is_identity(self):
        src = np.zeros((1, 5, 4), np.float32)
        src[0, 2, :] = 1
        result = Evaluator._kernel('Blur', {'radius': 1}, [src])
        np.testing.assert_allclose(result[0, :, 0], [0, 1 / 3, 1 / 3, 1 / 3, 0], atol=1e-6)
        np.testing.assert_array_equal(Evaluator._kernel('Blur', {'radius': 0}, [src]), src)

    def test_color_correct_lift_gamma_gain(self):
        src = np.array([[[0.2, 0.3, 0.4, 1.0]]], np.float32)
        result = Evaluator._kernel('ColorCorrect', {'lift': 0.1, 'gamma': 1.0, 'gain': 2.0, 'saturation': 1.0}, [src])
        np.testing.assert_allclose(result, [[[0.48, 0.67, 0.86, 1.0]]], atol=1e-6)

    def test_color_correct_is_sign_safe_under_gamma(self):
        src = np.array([[[0.2, 0.5, 0.5, 1.0]]], np.float32)
        result = Evaluator._kernel('ColorCorrect', {'lift': -5.0, 'gamma': 2.0, 'gain': 1.0, 'saturation': 1.0}, [src])
        self.assertFalse(np.isnan(result).any())
        self.assertLess(result[0, 0, 0], 0)

    def test_crop_masks_without_resizing_canvas(self):
        src = np.ones((4, 4, 4), np.float32)
        result = Evaluator._kernel('Crop', {'x': 1, 'y': 1, 'width': 2, 'height': 2}, [src])
        self.assertEqual(result.shape, src.shape)
        np.testing.assert_array_equal(result[1:3, 1:3], 1)
        self.assertEqual(result[0].sum(), 0)
        self.assertEqual(result[3].sum(), 0)

    def test_shuffle_remaps_channels_and_constants(self):
        src = np.array([[[0.1, 0.2, 0.3, 0.4]]], np.float32)
        params = {'red_from': 'G', 'green_from': '0', 'blue_from': '1', 'alpha_from': 'R'}
        np.testing.assert_allclose(Evaluator._kernel('Shuffle', params, [src]), [[[0.2, 0.0, 1.0, 0.1]]])
