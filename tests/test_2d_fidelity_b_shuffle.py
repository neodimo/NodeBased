"""Fidelity step B, part 3: Shuffle as Nuke's two-input Shuffle. Inputs B and A, two routing slots (`layer` with
`fromInput1`, `in2` with `fromInput2`), per-output-channel routing from either slot, the constants 0 and 1,
the evaluator and tile paths on the same pixels, and older one-input documents that load and render the same."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import SPECS, Dispatcher, bypass_slot, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.raster import Raster
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor

H, W = 20, 28


def picture(seed):
    rng = np.random.default_rng(seed)
    return rng.random((H, W, 4), dtype=np.float32) * 1.5


def shuffle(params, b, a=None):
    p = dict(SPECS["Shuffle"]["params"])
    p.update(params)
    return Evaluator._windowed_kernel("Shuffle", p, [b, a])


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.b, self.a = picture(1), picture(2)

    def test_one_input_shuffle_is_what_it_was(self):
        out = shuffle(dict(red_from="B", green_from="A", blue_from="R", alpha_from="G"), Raster.of(self.b)).pixels
        np.testing.assert_array_equal(out[..., 0], self.b[..., 2])
        np.testing.assert_array_equal(out[..., 1], self.b[..., 3])
        np.testing.assert_array_equal(out[..., 2], self.b[..., 0])
        np.testing.assert_array_equal(out[..., 3], self.b[..., 1])
        constants = shuffle(dict(red_from="0", green_from="1", blue_from="R", alpha_from="1"), Raster.of(self.b)).pixels
        np.testing.assert_array_equal(constants[..., :2], np.stack([np.zeros((H, W)), np.ones((H, W))], -1))
        np.testing.assert_array_equal(constants[..., 3], np.ones((H, W)))

    def test_the_second_slot_reads_input_a(self):
        out = shuffle(dict(in2="rgba", red_from="R2", green_from="B", blue_from="A2", alpha_from="G2"),
                      Raster.of(self.b), Raster.of(self.a)).pixels
        np.testing.assert_array_equal(out[..., 0], self.a[..., 0])
        np.testing.assert_array_equal(out[..., 1], self.b[..., 2])
        np.testing.assert_array_equal(out[..., 2], self.a[..., 3])
        np.testing.assert_array_equal(out[..., 3], self.a[..., 1])

    def test_every_output_channel_can_come_from_either_slot_or_a_constant(self):
        names = ["R", "G", "B", "A", "0", "1", "R2", "G2", "B2", "A2"]
        for channel in ("red_from", "green_from", "blue_from", "alpha_from"):
            for name in names:
                with self.subTest(channel=channel, name=name):
                    out = shuffle({channel: name, "in2": "rgba"}, Raster.of(self.b), Raster.of(self.a)).pixels
                    index = ("red_from", "green_from", "blue_from", "alpha_from").index(channel)
                    if name in ("0", "1"):
                        expected = np.full((H, W), float(name), np.float32)
                    elif name.endswith("2"):
                        expected = self.a[..., "RGBA".index(name[0])]
                    else:
                        expected = self.b[..., "RGBA".index(name)]
                    np.testing.assert_array_equal(out[..., index], expected)

    def test_the_first_slot_can_read_input_a_and_the_second_can_read_b(self):
        out = shuffle(dict(fromInput1="A", fromInput2="B", in2="rgba", red_from="R", green_from="R2"),
                      Raster.of(self.b), Raster.of(self.a)).pixels
        np.testing.assert_array_equal(out[..., 0], self.a[..., 0])
        np.testing.assert_array_equal(out[..., 1], self.b[..., 0])

    def test_a_named_layer_in_either_slot(self):
        layer_b, layer_a = Raster.of(picture(5)), Raster.of(picture(6))
        b = Raster(self.b, layers={"extra": layer_b})
        a = Raster(self.a, layers={"extra": layer_a, "other": Raster.of(picture(7))})
        out = shuffle(dict(layer="extra", fromInput2="A", in2="other", red_from="R", green_from="G2", blue_from="B", alpha_from="1"),
                      b, a).pixels
        np.testing.assert_array_equal(out[..., 0], layer_b.pixels[..., 0])
        np.testing.assert_array_equal(out[..., 1], a.layers["other"].pixels[..., 1])
        np.testing.assert_array_equal(out[..., 2], layer_b.pixels[..., 2])

    def test_a_component_path_selects_a_channel_of_a_layer(self):
        depth = Raster.of(np.stack([np.full((H, W), 2.5, np.float32)] * 3 + [np.ones((H, W), np.float32)], -1))
        out = shuffle(dict(layer="depth.Z", red_from="R", green_from="R", blue_from="R", alpha_from="1"),
                      Raster(self.b, layers={"depth": depth})).pixels
        np.testing.assert_allclose(out[..., :3], 2.5)

    def test_a_routed_second_slot_without_a_layer_says_so(self):
        with self.assertRaisesRegex(ValueError, "in2 is none"):
            shuffle(dict(red_from="R2"), Raster.of(self.b), Raster.of(self.a))

    def test_an_unwired_input_that_a_slot_reads_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "input A is not connected"):
            shuffle(dict(in2="rgba", red_from="R2"), Raster.of(self.b), None)
        with self.assertRaisesRegex(ValueError, "input A is not connected"):
            shuffle(dict(fromInput1="A"), Raster.of(self.b), None)

    def test_the_second_slot_is_fitted_to_the_first_slots_window(self):
        from nodebased.tiers import Region
        shifted = Raster(self.a, Region(4, 3, W, H), Region(0, 0, W + 8, H + 6))
        wide_b = Raster(np.pad(self.b, ((0, 3), (0, 4), (0, 0))), Region(0, 0, W + 4, H + 3), Region(0, 0, W + 8, H + 6))
        out = shuffle(dict(in2="rgba", red_from="R2"), wide_b, shifted)
        np.testing.assert_array_equal(out.pixels[3:3 + H, 4:4 + W, 0], self.a[..., 0])
        self.assertEqual(float(out.pixels[0, 0, 0]), 0.0)
        self.assertEqual(out.data, wide_b.data)


class DocumentTests(unittest.TestCase):
    def test_the_registration(self):
        spec = SPECS["Shuffle"]
        self.assertEqual((spec["inputs"], spec["optional_inputs"]), (["image"], ["A"]))
        self.assertEqual(spec["params"]["fromInput1"], "B")
        self.assertEqual(spec["params"]["in2"], "none")
        self.assertEqual(spec["params"]["fromInput2"], "A")
        self.assertIn("Shuffle", SUPPORTED_TILED_KINDS)

    def test_an_old_one_input_document_loads_and_renders_the_same(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "plate", "type": "Rectangle",
                   "params": {"width": 64, "height": 48, "box_x": 4, "box_y": 4, "box_width": 40, "box_height": 30,
                              "red": 0.9, "green": 0.4, "blue": 0.1}})
        d.execute({"op": "create", "id": "s", "type": "Shuffle",
                   "params": {"red_from": "B", "green_from": "R", "blue_from": "G", "alpha_from": "1"}})
        d.execute({"op": "connect", "id": "s", "input": "image", "source": "plate"})
        before = Evaluator().evaluate(dict(d.document, view="s"))
        old = copy.deepcopy(d.document)
        node = old["nodes"]["s"]
        del node["inputs"]["A"]
        for name in ("fromInput1", "in2", "fromInput2"):
            del node["params"][name]
        with self.assertRaises(ValueError):
            validate(old)
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["nodes"]["s"]["inputs"], {"image": "plate", "A": None})
        self.assertEqual(upgraded["nodes"]["s"]["params"]["fromInput1"], "B")
        np.testing.assert_array_equal(Evaluator().evaluate(dict(upgraded, view="s")), before)

    def test_bypass_passes_input_b(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "b", "type": "Constant", "params": {"width": 8, "height": 8, "red": 0.2}})
        d.execute({"op": "create", "id": "a", "type": "Constant", "params": {"width": 8, "height": 8, "red": 0.9}})
        d.execute({"op": "create", "id": "s", "type": "Shuffle", "params": {"in2": "rgba", "red_from": "R2"}})
        d.execute({"op": "connect", "id": "s", "input": "image", "source": "b"})
        d.execute({"op": "connect", "id": "s", "input": "A", "source": "a"})
        shuffled = Evaluator().evaluate(dict(d.document, view="s"))
        self.assertAlmostEqual(float(shuffled[0, 0, 0]), 0.9, places=5)
        d.execute({"op": "disable", "id": "s", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="s")), Evaluator().evaluate(dict(d.document, view="b")))
        self.assertEqual(bypass_slot(d.document["nodes"]["s"]), "image")


class TileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        h, w = 32, 40
        yy, xx = np.mgrid[:h, :w]
        beauty = np.stack((xx / w, yy / h, (xx + yy) / (w + h), np.ones_like(xx)), -1).astype(np.float32)
        depth = np.stack((np.ones_like(xx) * 2, np.ones_like(xx) * 2, np.ones_like(xx) * 2, np.ones_like(xx)), -1).astype(np.float32)
        self.path_b, self.path_a = Path(self.temp.name) / "b.exr", Path(self.temp.name) / "a.exr"
        write_exr(self.path_b, beauty, bits="float", layers={"depth": depth})
        write_exr(self.path_a, beauty[::-1, ::-1].copy() * 0.5, bits="float", layers={"depth": depth * 3})

    def document(self, params, wire_a=True):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Read", "id": "b", "params": {"path": str(self.path_b)}})
        d.execute({"op": "create", "type": "Read", "id": "a", "params": {"path": str(self.path_a)}})
        d.execute({"op": "create", "type": "Shuffle", "id": "n", "params": params})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "b"})
        if wire_a:
            d.execute({"op": "connect", "id": "n", "input": "A", "source": "a"})
        return d.document

    def test_tiles_match_the_evaluator(self):
        cases = (dict(red_from="G", green_from="B", blue_from="0", alpha_from="1"),
                 dict(in2="rgba", red_from="R2", green_from="G", blue_from="B2", alpha_from="A"),
                 dict(fromInput1="A", red_from="B", green_from="R", blue_from="G", alpha_from="A"),
                 dict(layer="depth", in2="rgba", red_from="R", green_from="R2", blue_from="G2", alpha_from="1"),
                 dict(fromInput2="A", in2="depth", red_from="R2", green_from="G", blue_from="G2", alpha_from="A"),
                 dict(layer="depth.Z", fromInput2="B", in2="rgba", red_from="R", green_from="G2", blue_from="B2", alpha_from="1"))
        for params in cases:
            for tier in (1, 2):
                with self.subTest(params=params, tier=tier):
                    doc = self.document(params)
                    expected = Evaluator().evaluate_raster(doc, "n", tier=tier)
                    executor = TileExecutor(tile_edge=8)
                    self.assertTrue(executor.supports_tiled(doc, "n"))
                    result = executor.compose(doc, "n", tier=tier)
                    self.assertTrue(result.tiled)
                    self.assertEqual(result.full_frame_fallbacks, 0)
                    np.testing.assert_allclose(result.pixels, expected.pixels, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
