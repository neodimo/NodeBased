"""Exact half-float ingest and resident-island widening."""
import pathlib
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import OpenImageIO as oiio

from nodebased import gpu2d
from nodebased.core import SPECS
from nodebased.media import read_media_raster
from nodebased.raster import Raster


class HalfEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        enabled = patch.dict(os.environ, {"NODEBASED_GPU2D_HALF_UPLOAD": "1"})
        enabled.start()
        self.addCleanup(enabled.stop)

    def write(self, name, pixels, fmt, channels=("R", "G", "B", "A"), space="ACEScg",
              x=0, y=0, full=None):
        path = pathlib.Path(self.directory.name, name)
        spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], pixels.shape[2], fmt)
        spec.channelnames = list(channels)
        spec.attribute("oiio:ColorSpace", space)
        spec.x, spec.y = x, y
        if full:
            spec.full_x, spec.full_y, spec.full_width, spec.full_height = full
        out = oiio.ImageOutput.create(str(path))
        self.assertTrue(out and out.open(str(path), spec), oiio.geterror())
        self.assertTrue(out.write_image(np.ascontiguousarray(pixels, np.float32)), out.geterror())
        self.assertTrue(out.close(), out.geterror())
        return path

    def test_only_exact_working_space_exr_gets_compact_hint(self):
        pixels = np.array([[[.3, .5, .75, 1.]]], np.float32)
        half = self.write("half.exr", pixels, oiio.HALF)
        raster = read_media_raster(str(half))
        self.assertIsNotNone(raster.half)
        np.testing.assert_array_equal(raster.half.astype(np.float32), raster.pixels)
        self.assertIsNone(read_media_raster(str(self.write("float.exr", pixels, oiio.FLOAT))).half)
        self.assertIsNone(read_media_raster(str(self.write("converted.exr", pixels, oiio.HALF,
                                                       space="Linear Rec.709 (sRGB)"))).half)
        self.assertIsNone(read_media_raster(str(self.write("plain.png", pixels, oiio.UINT8))).half)

    def test_compact_hint_is_off_by_default_after_playback_regression(self):
        pixels = np.array([[[.3, .5, .75, 1.]]], np.float32)
        half = self.write("default.exr", pixels, oiio.HALF)
        with patch.dict(os.environ, {"NODEBASED_GPU2D_HALF_UPLOAD": "0"}):
            self.assertIsNone(read_media_raster(str(half)).half)

    def test_alpha_math_never_sets_an_inexact_compact_hint(self):
        pixels = np.array([[[.3, .5, .75, .3]]], np.float32)
        path = self.write("alpha.exr", pixels, oiio.HALF)
        self.assertIsNone(read_media_raster(str(path)).half)  # associated-alpha round trip
        straight = read_media_raster(str(path), alpha_mode="Straight")
        self.assertIsNone(straight.half)  # .3 times .3 cannot be recovered from half exactly

    def test_layer_only_half_exr_has_no_beauty_upload_hint(self):
        pixels = np.array([[[.3, .5, .75, 1.]]], np.float32)
        path = self.write("layers.exr", pixels, oiio.HALF,
                          channels=("crypto.R", "crypto.G", "crypto.B", "crypto.A"))
        self.assertIsNone(read_media_raster(str(path)).half)

    def test_native_half_decode_is_bit_equal_to_float_decode_for_safe_and_unsafe_cases(self):
        """Native OIIO half decode changes acquisition, never the Raster contract."""
        opaque = np.array([[[.3, .5, .75, 1.], [-.25, 2., .125, 1.], [4., .01, .9, 1.]],
                           [[.2, .4, .6, 1.], [.9, .7, .5, 1.], [.1, .8, .3, 1.]]], np.float16)
        partial = opaque.copy(); partial[..., 3] = np.float16(.3)
        cases = [
            ("beauty.exr", opaque, oiio.HALF, ("R", "G", "B", "A"), "ACEScg", {}, True),
            ("reordered.exr", opaque[..., [1, 0, 2, 3]], oiio.HALF, ("G", "R", "B", "A"), "ACEScg", {}, True),
            ("missing-alpha.exr", opaque[..., :3], oiio.HALF, ("R", "G", "B"), "ACEScg", {}, True),
            ("associated.exr", partial, oiio.HALF, ("R", "G", "B", "A"), "ACEScg", {}, False),
            ("straight.exr", partial, oiio.HALF, ("R", "G", "B", "A"), "ACEScg", {"alpha_mode": "Straight"}, False),
            ("float.exr", opaque.astype(np.float32), oiio.FLOAT, ("R", "G", "B", "A"), "ACEScg", {}, False),
            ("converted.exr", opaque, oiio.HALF, ("R", "G", "B", "A"), "Linear Rec.709 (sRGB)", {}, False),
            ("overscan.exr", opaque, oiio.HALF, ("R", "G", "B", "A"), "ACEScg",
             {"x": -1, "y": 2, "full": (0, 0, 5, 6)}, True),
        ]
        for name, pixels, fmt, channels, space, kwargs, expect_hint in cases:
            with self.subTest(name=name):
                read_kwargs = {k: v for k, v in kwargs.items() if k == "alpha_mode"}
                write_kwargs = {k: v for k, v in kwargs.items() if k != "alpha_mode"}
                path = self.write(name, pixels, fmt, channels, space, **write_kwargs)
                native = read_media_raster(str(path), **read_kwargs)
                with patch.dict(os.environ, {"NODEBASED_GPU2D_HALF_UPLOAD": "0"}):
                    floating = read_media_raster(str(path), **read_kwargs)
                np.testing.assert_array_equal(native.pixels.view(np.uint32), floating.pixels.view(np.uint32))
                self.assertEqual(native.data, floating.data)
                self.assertEqual(native.display, floating.display)
                self.assertEqual(native.half is not None, expect_hint)


class HalfUploadTests(unittest.TestCase):
    def setUp(self):
        self.backend = gpu2d.GPU2D.open("default")
        if self.backend is None:
            self.skipTest(gpu2d.GPU2D.open_error)
        self.addCleanup(self.backend.release_all)

    def test_widened_upload_is_bit_exact_and_counts_half_the_bus_bytes(self):
        pixels = np.array([[[.5, .25, .125, 1.], [.1, .2, .3, 1.]]], np.float32)
        compact = Raster(pixels)
        compact.half = pixels.astype(np.float16)
        plain = Raster(compact.half.astype(np.float32))
        p = dict(SPECS["Grade"]["params"], exposure=.5)
        before = self.backend.counters.snapshot()
        got = self.backend.run(self.backend.stage("Grade", p, [compact]))
        sent = self.backend.counters.delta(self.backend.counters.snapshot(), before)["upload_bytes"]
        self.backend.release_all()
        self.backend = gpu2d.GPU2D.open("default")
        expected = self.backend.run(self.backend.stage("Grade", p, [plain]))
        np.testing.assert_array_equal(got.pixels, expected.pixels)
        self.assertEqual(sent, pixels.nbytes // 2)

    def test_derived_raster_drops_the_compact_hint(self):
        raster = Raster(np.ones((2, 3, 4), np.float32)); raster.half = raster.pixels.astype(np.float16)
        self.assertIsNone(raster.with_pixels(raster.pixels.copy()).half)
        self.assertIsNone(raster.cropped_to_display().half)

    def test_grade_merge_colorcorrect_chain_matches_plain_upload(self):
        rng = np.random.default_rng(610)
        values = rng.uniform(-.5, 2.0, (23, 37, 4)).astype(np.float16)
        values[..., 3] = 1
        background = rng.uniform(0, 1, (23, 37, 4)).astype(np.float16)
        background[..., 3] = 1

        def run(compact):
            source = Raster(values.astype(np.float32))
            back = Raster(background.astype(np.float32))
            if compact:
                source.half, back.half = values, background
            grade = self.backend.stage("Grade", dict(SPECS["Grade"]["params"], exposure=.5), [source])
            merged = self.backend.stage("Merge", dict(SPECS["Merge"]["params"], operation="over"), [grade, back])
            corrected = self.backend.stage("ColorCorrect", dict(SPECS["ColorCorrect"]["params"],
                                                            gain=1.2, shadows_saturation=.8), [merged])
            result = self.backend.run(corrected).pixels.copy()
            self.backend.release_all()
            return result

        np.testing.assert_array_equal(run(True), run(False))


if __name__ == "__main__":
    unittest.main()
