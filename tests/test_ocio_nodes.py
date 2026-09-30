import copy
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor


class OcioNodeTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        self.d.execute({"op": "create", "id": "src", "type": "Constant",
                        "params": {"width": 3, "height": 2, "alpha": 1}})
        self.pixels = np.array([[[.18, .27, .42, 1], [.5, .2, .1, 1], [.03, .6, .2, 1]],
                                [[.1, .12, .13, 1], [.9, .2, .3, 1], [.4, .5, .6, 1]]], np.float32)

    def add(self, key, kind, params):
        self.d.execute({"op": "create", "id": key, "type": kind, "params": params})
        self.d.execute({"op": "connect", "id": key, "input": "image", "source": "src"})

    def eval(self, key):
        ev = Evaluator()
        ev._lut_roots = {"src": Raster.of(self.pixels)}
        return ev.evaluate_raster(self.d.document, key).pixels

    def test_display_transform_matches_viewer(self):
        from nodebased.color import display_rgb
        self.add("n", "OCIODisplay", {})
        got = self.eval("n")[..., :3]
        expected = display_rgb(self.pixels[..., :3].copy(), "ACES 2.0")
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=0)

    def test_display_transform_applies_a_look(self):
        """`DisplayViewTransform.setLooks` does not exist on OCIO 2.x; a non-empty `look` must
        route through `LegacyViewingPipeline` instead of raising `AttributeError` (found while
        walking every 2D node's params for the M1 cache-correctness gate: this param had no
        test at all and crashed on first use)."""
        self.add("n", "OCIODisplay", {"look": "ACES 1.3 Reference Gamut Compression"})
        looked = self.eval("n")[..., :3]
        self.d.execute({"op": "set", "id": "n", "param": "look", "value": ""})
        plain = self.eval("n")[..., :3]
        self.assertFalse(np.allclose(looked, plain))

    def test_aces2065_to_acescg_roundtrip(self):
        self.d.execute({"op": "create", "id": "to_working", "type": "OCIOColorspace",
                        "params": {"src": "ACES2065-1", "dst": "ACEScg"}})
        self.d.execute({"op": "connect", "id": "to_working", "input": "image", "source": "src"})
        self.d.execute({"op": "create", "id": "back", "type": "OCIOColorspace",
                        "params": {"src": "ACEScg", "dst": "ACES2065-1"}})
        self.d.execute({"op": "connect", "id": "back", "input": "image", "source": "to_working"})
        ev = Evaluator(); ev._lut_roots = {"src": Raster.of(self.pixels)}
        got = ev.evaluate_raster(self.d.document, "back").pixels
        np.testing.assert_allclose(got, self.pixels, atol=1e-5, rtol=0)

    def test_file_transform_identity_and_gain_cube(self):
        fixtures = Path(__file__).parent / "fixtures"
        self.add("identity", "OCIOFileTransform", {"path": str(fixtures / "identity-2.cube")})
        np.testing.assert_allclose(self.eval("identity"), self.pixels, atol=1e-6)
        self.add("gain", "OCIOFileTransform", {"path": str(fixtures / "red-gain-2.cube")})
        got = self.eval("gain")
        d2 = Dispatcher()
        d2.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": 3, "height": 2, "alpha": 1}})
        d2.execute({"op": "create", "id": "lut", "type": "Vectorfield", "params": {"cube_path": str(fixtures / "red-gain-2.cube")}})
        d2.execute({"op": "connect", "id": "lut", "input": "image", "source": "src"})
        ev = Evaluator(); ev._lut_roots = {"src": Raster.of(self.pixels)}
        np.testing.assert_allclose(got, ev.evaluate_raster(d2.document, "lut").pixels, atol=1e-6)

    def test_srgb_to_linear_curve(self):
        encoded = np.array([[[.0, .04, .5, 1], [.2, .8, 1, 1], [.03, .1, .9, 1]],
                            [[.4, .7, .2, 1], [.1, .2, .3, 1], [.9, .8, .7, 1]]], np.float32)
        self.pixels = encoded
        self.add("n", "Colorspace", {"transfer_in": "sRGB", "transfer_out": "Linear"})
        got = self.eval("n")[..., :3]
        x = encoded[..., :3]
        expected = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=0)

    def test_bad_config_path_is_explained(self):
        self.add("n", "OCIOColorspace", {"config": "/definitely/missing/config.ocio"})
        with self.assertRaisesRegex(ValueError, "Could not load OCIO config.*does not exist"):
            self.eval("n")

    def test_config_roles_drive_log_linear_roundtrip(self):
        self.d.execute({"op": "create", "id": "log", "type": "OCIOLogConvert", "params": {}})
        self.d.execute({"op": "connect", "id": "log", "input": "image", "source": "src"})
        self.d.execute({"op": "create", "id": "linear", "type": "OCIOLogConvert",
                        "params": {"ocio_log_operation": "lin to log"}})
        self.d.execute({"op": "connect", "id": "linear", "input": "image", "source": "log"})
        ev = Evaluator(); ev._lut_roots = {"src": Raster.of(self.pixels)}
        got = ev.evaluate_raster(self.d.document, "linear").pixels
        np.testing.assert_allclose(got, self.pixels, atol=1e-5, rtol=0)

    def test_config_selector_lists_follow_the_config(self):
        from nodebased.ocio_nodes import parameter_choices
        config = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
        spaces = parameter_choices(config, "src")
        self.assertIn("ACEScg", spaces)
        self.assertIn("ACES2065-1", spaces)
        displays = parameter_choices(config, "display")
        self.assertIn("sRGB - Display", displays)
        views = parameter_choices(config, "view", "sRGB - Display")
        self.assertIn("ACES 2.0 - SDR 100 nits (Rec.709)", views)

    def test_logconvert_tile_and_evaluator_paths_agree(self):
        self.d.execute({"op": "set", "id": "src", "param": "red", "value": .18})
        self.d.execute({"op": "set", "id": "src", "param": "green", "value": .27})
        self.d.execute({"op": "set", "id": "src", "param": "blue", "value": .42})
        self.add("n", "OCIOLogConvert", {})
        reference = Evaluator().evaluate_raster(self.d.document, "n").pixels
        tiled = TileExecutor(tile_edge=2).compose(self.d.document, "n")
        self.assertTrue(tiled.tiled)
        np.testing.assert_allclose(tiled.pixels, reference, atol=1e-5, rtol=0)

    def test_tile_path_agrees_with_evaluator(self):
        self.d.execute({"op": "set", "id": "src", "param": "red", "value": .18})
        self.d.execute({"op": "set", "id": "src", "param": "green", "value": .27})
        self.d.execute({"op": "set", "id": "src", "param": "blue", "value": .42})
        self.add("n", "OCIOColorspace", {"src": "ACES2065-1", "dst": "ACEScg"})
        reference = Evaluator().evaluate_raster(self.d.document, "n").pixels
        ex = TileExecutor(tile_edge=2)
        tiled = ex.compose(self.d.document, "n")
        self.assertTrue(tiled.tiled)
        np.testing.assert_allclose(tiled.pixels, reference, atol=1e-5, rtol=0)

    def test_schema_six_graph_still_loads_and_renders_same_pixels(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "checker", "type": "Checker", "params": {"width": 4, "height": 3}})
        d.execute({"op": "create", "id": "grade", "type": "Grade", "params": {"exposure": .5}})
        d.execute({"op": "connect", "id": "grade", "input": "image", "source": "checker"})
        current = copy.deepcopy(d.document)
        old = copy.deepcopy(current)
        for key in ("settings", "node_data", "expressions"):
            old.pop(key, None)
        old["version"] = 6
        upgraded = Dispatcher(old).document
        np.testing.assert_array_equal(Evaluator().evaluate_raster(upgraded, "grade").pixels,
                                      Evaluator().evaluate_raster(current, "grade").pixels)


if __name__ == "__main__":
    unittest.main()
