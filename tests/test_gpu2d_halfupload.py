"""Exact half-float ingest and resident-island widening."""
import unittest

import numpy as np

from nodebased import gpu2d
from nodebased.core import SPECS
from nodebased.raster import Raster


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


if __name__ == "__main__":
    unittest.main()
