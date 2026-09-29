"""T2 free-form colour curves, legacy HueCorrect conversion and ShuffleCopy routing."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from nodebased.colorcurves import decode, encode, evaluate
from nodebased.core import Dispatcher, SPECS, atomic_save, load_document, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


class CurveNodeTests(unittest.TestCase):
    def make_graph(self, kind, params, width=16, height=8):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {
            "width": width, "height": height, "red": .8, "green": .5, "blue": .4, "alpha": 1}})
        d.execute({"op": "create", "id": "fx", "type": kind, "params": params})
        d.execute({"op": "connect", "id": "fx", "input": "image", "source": "src"})
        return d

    def test_hue_defaults_are_identity_and_green_sat_dip_is_local(self):
        p = SPECS["HueCorrect"]["params"].copy()
        image = np.array([[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]], np.float32)
        np.testing.assert_array_equal(Evaluator._hue_correct(image, p), image)
        p["curve_sat"] = encode(((0, 1), (60, 1), (120, 0), (180, 1), (240, 1), (300, 1), (360, 1)))
        out = Evaluator._hue_correct(image, p)
        np.testing.assert_allclose(out[0, 1, :3], [.7152, .7152, .7152], atol=1e-6)
        np.testing.assert_array_equal(out[0, 0], image[0, 0])
        np.testing.assert_array_equal(out[0, 2], image[0, 2])

    def test_v12_hue_anchors_upgrade_to_equivalent_curve(self):
        image = np.random.default_rng(7).random((11, 17, 4), dtype=np.float32)
        legacy = SPECS["HueCorrect"]["params"].copy()
        legacy.update({f"sat_{band}": (i + 1) / 6 for i, band in enumerate(("red", "yellow", "green", "cyan", "blue", "magenta"))})
        legacy.update({f"lum_{band}": 1.3 - i / 10 for i, band in enumerate(("red", "yellow", "green", "cyan", "blue", "magenta"))})
        expected = Evaluator._hue_correct(image, legacy)
        d = Dispatcher()
        d.execute({"op": "create", "id": "h", "type": "HueCorrect"})
        doc = d.document
        doc["version"] = 12
        node = doc["nodes"]["h"]
        node["params"] = {k: v for k, v in legacy.items() if not k.startswith("curve_")}
        upgraded = upgrade_document(doc)
        np.testing.assert_allclose(Evaluator._hue_correct(image, upgraded["nodes"]["h"]["params"]), expected, atol=1e-6)

    def test_color_lookup_curve_evaluator_and_tiles(self):
        p = SPECS["ColorLookup"]["params"].copy()
        p["curve_red"] = encode(((0, 0), (1, .5)))
        d = self.make_graph("ColorLookup", p)
        whole = Evaluator().evaluate(dict(d.document, view="fx"))
        self.assertAlmostEqual(float(whole[0, 0, 0]), .4, places=6)
        tile = TileExecutor(tile_edge=5)
        self.assertTrue(tile.supports_tiled(d.document, "fx"))
        region = tile.canvas_region(d.document, "fx", 1, 1)
        tiled = tile.compose_region(d.document, "fx", region, 1, 1).pixels
        np.testing.assert_allclose(tiled, whole, atol=1e-6)

    def test_crosstalk_arbitrary_curve(self):
        p = SPECS["CrossTalk"]["params"].copy()
        p["xt_curve_r_r"] = encode(((0, 0), (.25, .8), (1, 1)))
        image = np.array([[[.25, .3, .4, .7]]], np.float32)
        out = Evaluator._crosstalk(image, p)
        self.assertAlmostEqual(float(out[0, 0, 0]), .8, places=6)
        np.testing.assert_array_equal(out[..., 1:], image[..., 1:])

    def test_curve_data_roundtrip_and_dispatcher_undo(self):
        raw = encode(((0, 0), (.25, .8), (1, 1)), "smooth")
        self.assertEqual(encode(decode(raw)["points"], decode(raw)["interpolation"]), raw)
        d = self.make_graph("ColorLookup", SPECS["ColorLookup"]["params"].copy())
        changed = encode(((0, 0), (.5, .25), (1, 1)))
        d.execute({"op": "set", "id": "fx", "param": "curve_master", "value": changed})
        self.assertEqual(d.document["nodes"]["fx"]["params"]["curve_master"], changed)
        d.execute({"op": "undo"})
        self.assertEqual(d.document["nodes"]["fx"]["params"]["curve_master"], SPECS["ColorLookup"]["params"]["curve_master"])

    def test_tangent_curve_roundtrip_and_hermite_evaluation(self):
        raw = encode(((0, 0), (1, 1)), slopes=((0, 0), (0, 0)),
                     modes=("broken", "broken"), broken=(True, True))
        curve = decode(raw)
        self.assertEqual(decode(encode(curve["points"], curve["interpolation"], curve["slopes"],
                                       curve["modes"], curve["broken"])), curve)
        self.assertAlmostEqual(evaluate(curve, .25), .15625, places=7)
        d = self.make_graph("ColorLookup", SPECS["ColorLookup"]["params"].copy())
        d.execute({"op": "set", "id": "fx", "param": "curve_master", "value": raw})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "curve.nbp"
            atomic_save(path, d.document)
            loaded = load_document(path)
        self.assertEqual(loaded["nodes"]["fx"]["params"]["curve_master"], raw)
        d.execute({"op": "undo"})
        self.assertEqual(d.document["nodes"]["fx"]["params"]["curve_master"],
                         SPECS["ColorLookup"]["params"]["curve_master"])

    def test_legacy_curve_shape_and_evaluation_stay_unchanged(self):
        legacy = '{"interpolation":"smooth","points":[[0.0,0.0],[1.0,1.0]]}'
        curve = decode(legacy)
        self.assertEqual(set(curve), {"interpolation", "points"})
        self.assertAlmostEqual(evaluate(curve, .25), .15625, places=7)
        self.assertEqual(encode(curve["points"], curve["interpolation"]), legacy)

    def test_shufflecopy_routes_second_input(self):
        d = Dispatcher()
        for key, color in (("a", (.1, .2, .3)), ("b", (.7, .8, .9))):
            d.execute({"op": "create", "id": key, "type": "Constant", "params": {
                "width": 4, "height": 3, "red": color[0], "green": color[1], "blue": color[2], "alpha": 1}})
        p = SPECS["ShuffleCopy"]["params"].copy()
        p["out1_r"] = "in2.r"
        d.execute({"op": "create", "id": "sc", "type": "ShuffleCopy", "params": p})
        d.execute({"op": "connect", "id": "sc", "input": "in1", "source": "a"})
        d.execute({"op": "connect", "id": "sc", "input": "in2", "source": "b"})
        raster = Evaluator().evaluate_raster(d.document, "sc")
        self.assertAlmostEqual(float(raster.pixels[0, 0, 0]), .7, places=6)
        self.assertIn("out2", raster.layers)
        self.assertAlmostEqual(float(raster.layers["out2"].pixels[0, 0, 1]), .8, places=6)


if __name__ == "__main__":
    unittest.main()
