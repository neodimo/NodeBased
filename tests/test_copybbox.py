"""Nuke CopyBBox: pixels from A, output data window from B."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.raster import Region
from nodebased.tileexec import TileExecutor


def write_window(path, display, data, pixels):
    spec = oiio.ImageSpec(data.width, data.height, 4, oiio.FLOAT)
    spec.x, spec.y = data.x, data.y
    spec.full_x, spec.full_y = display.x, display.y
    spec.full_width, spec.full_height = display.width, display.height
    spec.channelnames = ["R", "G", "B", "A"]
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise AssertionError("Could not create EXR test image: " + oiio.geterror())
    output.write_image(pixels)
    output.close()


class CopyBBoxTests(unittest.TestCase):
    def graph(self, directory):
        display = Region(0, 0, 16, 12)
        a_data, b_data = Region(3, 2, 8, 6), Region(-2, -1, 20, 14)
        a_pixels = np.zeros((6, 8, 4), np.float32)
        a_pixels[..., 0] = np.arange(8, dtype=np.float32)[None, :] / 8
        a_pixels[..., 1] = np.arange(6, dtype=np.float32)[:, None] / 6
        a_pixels[..., 3] = 1
        b_pixels = np.ones((14, 20, 4), np.float32)
        b_pixels[..., 3] = 1
        a_path, b_path = Path(directory) / "a.exr", Path(directory) / "b.exr"
        write_window(a_path, display, a_data, a_pixels)
        write_window(b_path, display, b_data, b_pixels)
        d = Dispatcher()
        d.execute({"op": "create", "id": "a", "type": "Read", "params": {"path": str(a_path)}})
        d.execute({"op": "create", "id": "b", "type": "Read", "params": {"path": str(b_path)}})
        d.execute({"op": "create", "id": "copy", "type": "CopyBBox"})
        d.execute({"op": "connect", "id": "copy", "input": "A", "source": "a"})
        d.execute({"op": "connect", "id": "copy", "input": "B", "source": "b"})
        return d.document, a_data, b_data, a_pixels

    def test_copies_a_pixels_into_b_data_window_on_full_and_tile_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            document, a_data, b_data, a_pixels = self.graph(directory)
            full = Evaluator().evaluate_raster(document, "copy")
            source = Evaluator().evaluate_raster(document, "a")
            self.assertEqual(full.data, b_data)
            np.testing.assert_array_equal(full.pixels[3:9, 5:13], source.pixels)
            # New B window area outside A is transparent black, including the overscan.
            self.assertEqual(float(full.pixels[0, 0].sum()), 0.0)
            tiled = TileExecutor(tile_edge=5).compose(document, "copy")
            self.assertTrue(tiled.tiled)
            self.assertEqual(tiled.region.x, b_data.x)
            self.assertEqual(tiled.region.y, b_data.y)
            self.assertEqual(tiled.region.width, b_data.width)
            self.assertEqual(tiled.region.height, b_data.height)
            np.testing.assert_array_equal(tiled.pixels, full.pixels)


if __name__ == "__main__":
    unittest.main()
