"""Lane 8 D1: depth compositing, depth-band mattes and layer removal."""
import unittest
import numpy as np
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tiers import Region


def layered(colour, depth):
    pixels = np.zeros((2, 2, 4), np.float32); pixels[:] = colour
    z = np.zeros_like(pixels); z[..., :3] = depth; z[..., 3] = 1
    return Raster(pixels, layers={"depth": Raster(z)})


class DepthAndLayersD1Tests(unittest.TestCase):
    def run_node(self, kind, params, inputs):
        return Evaluator._windowed_kernel(kind, params, inputs, frame=1, data=Region(0, 0, 2, 2))

    def test_zmerge_selects_nearer_per_pixel_and_depth_minimum(self):
        a, b = layered((1, 0, 0, 1), 2), layered((0, 0, 1, 1), 5)
        a.layers["depth"].pixels[0, 0, 2] = 8
        out = self.run_node("ZMerge", {"depth_layer": "depth.Z", "depth_math": "depth", "smoothing": 0, "mix": 1}, [a, b])
        np.testing.assert_array_equal(out.pixels[0, 0], b.pixels[0, 0])
        np.testing.assert_array_equal(out.pixels[1, 1], a.pixels[1, 1])
        self.assertEqual(float(out.layers["depth"].pixels[0, 0, 2]), 5.0)
        a.layers["depth"].pixels[..., 2] = 9
        b.layers["depth"].pixels[..., 2] = 2
        swapped = self.run_node("ZMerge", {"depth_layer": "depth.Z", "depth_math": "depth", "smoothing": 0, "mix": 1}, [a, b])
        np.testing.assert_array_equal(swapped.pixels, b.pixels)
        inverse = self.run_node("ZMerge", {"depth_layer": "depth.Z", "depth_math": "1/depth", "smoothing": 0, "mix": 1}, [a, b])
        np.testing.assert_array_equal(inverse.pixels, b.pixels)

    def test_zslice_matte_band_falloff_and_missing_depth_error(self):
        image = layered((0.5, 0.25, 0.1, 1), 1)
        image.layers["depth"].pixels[..., 2] = [[0.5, 1.5], [2.5, 4.0]]
        matte = self.run_node("ZSlice", {"depth_layer": "depth.Z", "near": 1, "far": 3, "falloff": 0, "depth_math": "depth", "zslice_output": "matte", "mix": 1}, [image])
        np.testing.assert_array_equal(matte.pixels[..., 2], [[0, 1], [1, 0]])
        ramp = self.run_node("ZSlice", {"depth_layer": "depth.Z", "near": 1, "far": 3, "falloff": 1, "depth_math": "depth", "zslice_output": "matte", "mix": 1}, [image])
        self.assertAlmostEqual(float(ramp.pixels[0, 0, 2]), 0.5)
        with self.assertRaisesRegex(ValueError, "ZSlice: no layer"):
            self.run_node("ZSlice", {"depth_layer": "depth.Z"}, [Raster.of(image.pixels)])
        with self.assertRaisesRegex(ValueError, "ZMerge: no layer"):
            self.run_node("ZMerge", {"depth_layer": "depth.Z"}, [Raster.of(image.pixels), Raster.of(image.pixels)])

    def test_remove_and_keep_named_layers_and_rgba_passthrough(self):
        image = layered((1, 1, 1, 1), 3); image.layers["normals"] = Raster.of(np.ones((2, 2, 4), np.float32))
        removed = self.run_node("Remove", {"remove_operation": "remove", "layers": "depth"}, [image])
        self.assertEqual(set(removed.layers), {"normals"})
        kept = self.run_node("Remove", {"remove_operation": "keep", "layers": "depth"}, [image])
        self.assertEqual(set(kept.layers), {"depth"})
        rgba = Raster.of(image.pixels)
        self.assertIs(self.run_node("Remove", {"remove_operation": "remove", "layers": "depth"}, [rgba]), rgba)

    def test_zmerge_and_zslice_bypass_without_depth(self):
        # Disabled nodes are resolved before kernels, so their bypass never demands layer data.
        from nodebased.core import Dispatcher
        for kind, inputs in (("ZMerge", {"A": "src", "B": "src"}), ("ZSlice", {"image": "src"}),
                             ("Remove", {"image": "src"})):
            d = Dispatcher(); d.execute({"op": "create", "id": "src", "type": "Checker", "params": {"width": 2, "height": 2}})
            d.execute({"op": "create", "id": "fx", "type": kind})
            for slot, source in inputs.items(): d.execute({"op": "connect", "id": "fx", "input": slot, "source": source})
            d.execute({"op": "disable", "id": "fx", "value": True})
            d.document["view"] = "fx"
            np.testing.assert_array_equal(Evaluator().evaluate(d.document), Evaluator().evaluate(dict(d.document, view="src")))
