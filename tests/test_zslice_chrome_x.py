import unittest
import numpy as np

from nodebased import ops2d_chrome
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from nodebased.raster import Raster


def image():
    pixels = np.array([[[0.2, 0.4, 0.6, 0.8]]], dtype=np.float32)
    depth = np.array([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
    return Raster(pixels, layers={"depth": Raster(depth)})


def kernel(_kind, params, inputs):
    from nodebased.tiers import Region
    return Evaluator._windowed_kernel("ZSlice", params, inputs, frame=1, data=Region(0, 0, 1, 1))


class ZSliceChromeTests(unittest.TestCase):
    def run_zslice(self, params, inputs):
        return ops2d_chrome.run_evaluator("ZSlice", params, inputs, kernel, SPECS["ZSlice"])

    def test_alpha_channel_restriction_preserves_rgb(self):
        source = image()
        out = self.run_zslice({**SPECS["ZSlice"]["params"], "zslice_output": "matte",
                               "channels": "alpha", "near": 0.0, "far": 0.5}, [source, None, None])
        np.testing.assert_array_equal(out.pixels[..., :3], source.pixels[..., :3])
        np.testing.assert_array_equal(out.pixels[..., 3], 0.0)

    def test_selected_mask_channel_and_invert_gate_result_once(self):
        source = image()
        mask = Raster(np.array([[[0.25, 0.0, 0.0, 1.0]]], dtype=np.float32))
        out = self.run_zslice({**SPECS["ZSlice"]["params"], "zslice_output": "matte", "near": 0.0, "far": 0.5,
                               "maskChannelInput": "rgba.red", "invert_mask": 1}, [source, None, mask])
        np.testing.assert_allclose(out.pixels[0, 0], [0.05, 0.10, 0.15, 0.20], atol=1e-7)

    def test_default_chrome_matches_the_legacy_masked_kernel(self):
        source = image()
        mask = Raster(np.array([[[0.37, 0.37, 0.37, 0.37]]], dtype=np.float32))
        inputs = [source, None, mask]
        params = dict(SPECS["ZSlice"]["params"])
        legacy = kernel("ZSlice", params, inputs)
        current = self.run_zslice(params, inputs)
        np.testing.assert_array_equal(current.pixels, legacy.pixels)


if __name__ == "__main__":
    unittest.main()
