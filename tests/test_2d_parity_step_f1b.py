import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


class FilterF1bTests(unittest.TestCase):
    def graph(self, kind, params, width=16, height=16):
        d = Dispatcher()
        for key, node_type, values in (("src", "Checker", {"width": width, "height": height, "size": 1}),
                                       ("fx", kind, params)):
            d.execute({"op": "create", "id": key, "type": node_type, "params": values})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        return d.document

    def both_paths(self, document):
        full = Evaluator().evaluate(dict(document, view="fx"))
        tiles = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        self.assertTrue(tiles.supports_tiled(dict(document, view="fx"), "fx"))
        region = tiles.canvas_region(document, "fx", frame=1, tier=1)
        tiled = tiles.compose_region(document, "fx", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)
        return full

    def test_matrix_5x5_identity_and_tile_parity(self):
        params = {"matrix_size": "5", "mix": 1.0}
        params.update({f"weight{i}": float(i == 12) for i in range(25)})
        document = self.graph("Matrix", params)
        out = self.both_paths(document)
        expected = Evaluator().evaluate(dict(document, view="src"))
        np.testing.assert_array_equal(out, expected)

    def test_edge_detect_sobel_vertical_step_magnitude_and_tiles(self):
        image = np.zeros((9, 9, 4), np.float32)
        image[:, 4:, :3] = 1.0
        image[..., 3] = 1.0
        result = Evaluator._edge_detect(image, {"edge_type": "Sobel", "threshold": 0.0})
        self.assertEqual(float(result[4, 3, 0]), 4.0)
        self.assertEqual(float(result[4, 4, 0]), 4.0)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        self.both_paths(self.graph("EdgeDetect", {"edge_type": "Sobel"}))

    def test_emboss_flat_is_neutral_grey_and_tiles_match(self):
        image = np.full((8, 8, 4), 0.7, np.float32)
        result = Evaluator._emboss(image, {"angle": 45.0, "width": 3.0})
        np.testing.assert_array_equal(result[..., :3], 0.5)
        self.both_paths(self.graph("Emboss", {"angle": 45.0, "width": 3.0}))

    def test_filter_erode_disc_shrinks_alpha_by_size(self):
        yy, xx = np.mgrid[:21, :21]
        alpha = (((xx - 10) ** 2 + (yy - 10) ** 2) <= 25).astype(np.float32)
        image = np.zeros((21, 21, 4), np.float32); image[..., 3] = alpha
        result = Evaluator._erode_filter(image, {"filter_size": 2.0, "filter_type": "box"})
        # The square structuring element contracts the horizontal radius by exactly two pixels.
        self.assertEqual(float(result[10, 13, 3]), 1.0)
        self.assertEqual(float(result[10, 14, 3]), 0.0)
        self.both_paths(self.graph("ErodeFilter", {"filter_size": 2.0, "filter_type": "box"}, 21, 21))

    def test_matrix_kernel_has_hand_computed_checker_sample(self):
        image = np.zeros((5, 5, 4), np.float32)
        image[..., :3] = ((np.indices((5, 5)).sum(axis=0) % 2))[..., None]
        image[..., 3] = 1.0
        weights = {f"weight{i}": 0.0 for i in range(49)}
        weights.update(weight4=2.0, weight3=-1.0)
        out = Evaluator._matrix(image, weights)
        self.assertEqual(float(out[2, 2, 0]), -1.0)  # 2*black centre - white left neighbour

    def test_bypass_and_mask_mix_are_respected(self):
        document = self.graph("EdgeDetect", {"edge_type": "Sobel", "mix": 0.0})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(document, view="fx")),
                                      Evaluator().evaluate(dict(document, view="src")))
        d = Dispatcher(); document = self.graph("EdgeDetect", {"edge_type": "Sobel"})
        d.document = document
        d.execute({"op": "create", "id": "mask", "type": "Constant",
                   "params": {"width": 16, "height": 16, "alpha": 0.0}})
        d.execute({"op": "connect", "id": "fx", "input": "mask", "source": "mask"})
        masked = Evaluator().evaluate(dict(d.document, view="fx"))
        source = Evaluator().evaluate(dict(d.document, view="src"))
        np.testing.assert_array_equal(masked, source)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        region = executor.canvas_region(d.document, "fx", frame=1, tier=1)
        np.testing.assert_array_equal(executor.compose_region(d.document, "fx", region, 1, 1).pixels, source)

    def test_convolve_single_white_kernel_is_identity_on_both_paths(self):
        d = Dispatcher()
        for key, kind, params in (("src", "Checker", {"width": 12, "height": 10, "size": 1}),
                                  ("k", "Constant", {"width": 1, "height": 1, "red": 1.0,
                                                       "green": 1.0, "blue": 1.0, "alpha": 1.0}),
                                  ("fx", "Convolve", {"kernel_size": "1"})):
            d.execute({"op": "create", "id": key, "type": kind, "params": params})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "fx", "input": "kernel", "source": "k"})
        full = Evaluator().evaluate(dict(d.document, view="fx"))
        expected = Evaluator().evaluate(dict(d.document, view="src"))
        np.testing.assert_array_equal(full, expected)
        executor = TileExecutor(evaluator=Evaluator())
        self.assertTrue(executor.supports_tiled(dict(d.document, view="fx"), "fx"))
        region = executor.canvas_region(d.document, "fx", frame=1, tier=1)
        tiled = executor.compose_region(d.document, "fx", region, frame=1, tier=1).pixels
        np.testing.assert_array_equal(tiled, expected)
        d.execute({"op": "disable", "id": "fx", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="fx")), expected)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        region = executor.canvas_region(d.document, "fx", frame=1, tier=1)
        np.testing.assert_array_equal(executor.compose_region(d.document, "fx", region, 1, 1).pixels, expected)


if __name__ == "__main__":
    unittest.main()
