import time
import unittest
import numpy as np

from nodebased.core import Dispatcher, SPECS
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


class ColorC1Tests(unittest.TestCase):
    def source_graph(self, kind, params, pixels=None):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {
            "width": 16, "height": 8, "red": 0.4, "green": 0.5, "blue": 0.6, "alpha": 1}})
        d.execute({"op": "create", "id": "fx", "type": kind, "params": params})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        doc = dict(d.document, view="fx")
        return d, doc

    def assert_tile_match(self, doc):
        evaluator = Evaluator()
        full = evaluator.evaluate(doc)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=5)
        self.assertTrue(executor.supports_tiled(doc, "fx"))
        region = executor.canvas_region(doc, "fx", 1, 1)
        tiled = executor.compose_region(doc, "fx", region, 1, 1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)

    def test_log2lin_cineon_white_and_roundtrip(self):
        image = np.full((2, 2, 4), 685, np.float32); image[..., 3] = 1
        out = Evaluator._log2lin(image, {"black": 95, "white": 685, "gamma": 1, "log_direction": "log to lin"})
        np.testing.assert_allclose(out[..., :3], 1.0, atol=1e-7)
        codes = np.array([95, 200, 445, 685], np.float32)
        a = np.zeros((1, 4, 4), np.float32); a[..., :3] = codes[None, :, None]; a[..., 3] = 1
        lin = Evaluator._log2lin(a, {"black": 95, "white": 685, "gamma": 1, "log_direction": "log to lin"})
        back = Evaluator._log2lin(lin, {"black": 95, "white": 685, "gamma": 1, "log_direction": "lin to log"})
        np.testing.assert_allclose(back, a, atol=1e-5)

    def test_ploglin_hand_values_and_tiles(self):
        image = np.zeros((1, 3, 4), np.float32); image[0, :, :3] = np.array([445, 545, 345], np.float32)[:, None]; image[..., 3] = 1
        p = {"linear_reference": .18, "log_reference": 445, "density_per_code_value": .002, "negative_gamma": 1}
        out = Evaluator._ploglin(image, p)
        expected = .18 * 10 ** (np.array([0, .2, -.2], np.float32))
        np.testing.assert_allclose(out[0, :, 0], expected, atol=2e-6)
        _, doc = self.source_graph("PLogLin", p)
        self.assert_tile_match(doc)

    def test_crosstalk_identity_and_toe_knee(self):
        image = np.random.default_rng(1).uniform(-.2, 1.4, (4, 5, 4)).astype(np.float32)
        image[..., 3] = 1
        p = SPECS["CrossTalk"]["params"].copy()
        np.testing.assert_allclose(Evaluator._crosstalk(image, p), image, atol=1e-6)
        toe = Evaluator._toe(image, {"toe": .2, "toe_lift": .05})
        np.testing.assert_array_equal(toe[image[..., 0] >= .2, 0], image[image[..., 0] >= .2, 0])
        dark = np.zeros((1, 1, 4), np.float32)
        self.assertGreater(float(Evaluator._toe(dark, {"toe": .2, "toe_lift": .05})[0, 0, 0]), 0)
        _, doc = self.source_graph("CrossTalk", {**p, "mix": 1})
        self.assert_tile_match(doc)
        _, doc = self.source_graph("Toe", {"toe": .2, "toe_lift": .05, "mix": 1})
        self.assert_tile_match(doc)

    def test_mask_and_mix_gate_the_node(self):
        d, doc = self.source_graph("Toe", {"toe": .2, "toe_lift": .05, "mix": 0})
        d.execute({"op": "create", "id": "mask", "type": "Constant", "params": {
            "width": 16, "height": 8, "red": 1, "green": 1, "blue": 1, "alpha": 0}})
        d.execute({"op": "connect", "id": "fx", "input": "mask", "source": "mask"})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="fx")),
                                      Evaluator().evaluate(dict(d.document, view="src")))

    def test_expression_channels_coordinates_error_and_performance(self):
        _, doc = self.source_graph("Expression", {"expr_r": "r*2", "expr_g": "x/width",
            "expr_b": "b", "expr_a": "a", "mix": 1})
        out = Evaluator().evaluate(doc)
        self.assertAlmostEqual(float(out[0, 0, 0]), .8, places=6)
        self.assertAlmostEqual(float(out[0, 8, 1]), .5, places=6)
        self.assert_tile_match(doc)
        bad = dict(doc); bad["nodes"] = {k: dict(v) for k, v in doc["nodes"].items()}
        bad["nodes"]["fx"] = dict(bad["nodes"]["fx"])
        bad["nodes"]["fx"]["params"] = dict(bad["nodes"]["fx"]["params"], expr_r="r+__import__('os')")
        with self.assertRaisesRegex(ValueError, "Expression|unsupported"):
            Evaluator().evaluate(bad)
        image = np.ones((1080, 2048, 4), np.float32)
        p = {"expr_r": "x/width", "expr_g": "y/height", "expr_b": "r+g", "expr_a": "a"}
        start = time.perf_counter(); result = Evaluator._expression(image, None, p, 1); elapsed = time.perf_counter()-start
        self.assertEqual(result.shape, image.shape); self.assertLess(elapsed, 1.0)

    def test_every_node_bypass(self):
        for kind, params in (("Log2Lin", {}), ("PLogLin", {}), ("CrossTalk", {}), ("Toe", {}), ("Expression", {})):
            d, _ = self.source_graph(kind, params)
            d.execute({"op": "disable", "id": "fx", "value": True})
            np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="fx")),
                                          Evaluator().evaluate(dict(d.document, view="src")))


if __name__ == "__main__": unittest.main()
