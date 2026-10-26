"""Fidelity step C, part 5: Premult and Unpremult take Nuke's channels, `by` channel and invert. Pixels by channel, by
divisor and by invert, tiles against the evaluator, bypass, and old documents."""
import copy
import unittest

import numpy as np

from nodebased import ops2d_chrome as chrome
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor

H, W = 14, 22
KINDS = ("Premult", "Unpremult")


def image():
    rng = np.random.default_rng(9)
    pixels = (rng.random((H, W, 4)) * 0.8 + 0.1).astype(np.float32)
    pixels[0, :4, 3] = 0.0                      # a few pixels with nothing behind them
    pixels[1, :4, 3] = 1.0
    return pixels


def run(kind, pixels, **knobs):
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs}, [Raster.of(pixels)]).pixels


def old_premult(pixels):
    out = pixels.copy()
    out[..., :3] = pixels[..., :3] * pixels[..., 3:4]
    return out


def old_unpremult(pixels):
    alpha = pixels[..., 3:4]
    safe = np.where(alpha > 0, alpha, np.float32(1.0))
    return np.concatenate([np.where(alpha > 0, pixels[..., :3] / safe, pixels[..., :3]), alpha], axis=2).astype(np.float32)


class DefaultsTests(unittest.TestCase):
    def test_the_specs_carry_the_nuke_defaults(self):
        for kind in KINDS:
            self.assertEqual(SPECS[kind]["params"], {"channels": "rgb", "premult_by": "rgba.alpha", "invert": 0})
            self.assertEqual(chrome.FILTER_CHROME[kind], ("rgb", False, False))

    def test_defaults_are_exactly_what_the_nodes_always_did(self):
        pixels = image()
        np.testing.assert_array_equal(run("Premult", pixels), old_premult(pixels))
        np.testing.assert_array_equal(run("Unpremult", pixels), old_unpremult(pixels))


class ChannelTests(unittest.TestCase):
    def test_each_channel_set_multiplies_or_divides_only_those_channels(self):
        pixels = image()
        alpha = pixels[..., 3:4]
        safe = np.where(alpha > 0, alpha, 1)
        cases = {"rgb": (0, 1, 2), "rgba": (0, 1, 2, 3), "alpha": (3,), "red": (0,), "green": (1,), "blue": (2,),
                 "all": (0, 1, 2, 3), "none": ()}
        for text, picked in cases.items():
            with self.subTest(channels=text):
                expected_premult = pixels.copy()
                expected_unpremult = pixels.copy()
                for index in picked:
                    expected_premult[..., index] = pixels[..., index] * alpha[..., 0]
                    expected_unpremult[..., index] = np.where(alpha[..., 0] > 0, pixels[..., index] / safe[..., 0], pixels[..., index])
                np.testing.assert_allclose(run("Premult", pixels, channels=text), expected_premult, atol=1e-7)
                np.testing.assert_allclose(run("Unpremult", pixels, channels=text), expected_unpremult, atol=1e-7)

    def test_two_channels_named_together(self):
        pixels = image()
        out = run("Premult", pixels, channels="red blue")
        np.testing.assert_allclose(out[..., 0], pixels[..., 0] * pixels[..., 3], atol=1e-7)
        np.testing.assert_allclose(out[..., 2], pixels[..., 2] * pixels[..., 3], atol=1e-7)
        np.testing.assert_array_equal(out[..., 1], pixels[..., 1])


class ByChannelTests(unittest.TestCase):
    def test_the_colour_is_multiplied_by_the_chosen_channel(self):
        pixels = image()
        luminance = (0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2])[..., None]
        for token, divisor in (("rgba.alpha", pixels[..., 3:4]), ("rgba.red", pixels[..., 0:1]), ("rgba.green", pixels[..., 1:2]),
                               ("rgba.blue", pixels[..., 2:3]), ("rgba.luminance", luminance), ("alpha", pixels[..., 3:4])):
            with self.subTest(by=token):
                out = run("Premult", pixels, premult_by=token)
                np.testing.assert_allclose(out[..., :3], pixels[..., :3] * divisor, atol=1e-6)
                np.testing.assert_array_equal(out[..., 3], pixels[..., 3])
                undone = run("Unpremult", pixels, premult_by=token)
                safe = np.where(divisor > 0, divisor, 1)
                np.testing.assert_allclose(undone[..., :3], np.where(divisor > 0, pixels[..., :3] / safe, pixels[..., :3]), rtol=1e-5, atol=1e-6)

    def test_a_channel_of_a_named_layer(self):
        pixels = image()
        matte = np.zeros((H, W, 4), np.float32)
        matte[..., 0] = np.linspace(0.2, 0.9, W)[None, :]
        raster = Raster(pixels, layers={"matte": Raster(matte)})
        out = Evaluator._windowed_kernel("Premult", {**SPECS["Premult"]["params"], "premult_by": "matte.R"}, [raster]).pixels
        np.testing.assert_allclose(out[..., :3], pixels[..., :3] * matte[..., 0:1], atol=1e-6)
        out = Evaluator._windowed_kernel("Premult", {**SPECS["Premult"]["params"], "premult_by": "matte"}, [raster]).pixels
        np.testing.assert_allclose(out[..., :3], pixels[..., :3] * matte[..., 0:1], atol=1e-6)
        with self.assertRaises(ValueError):
            Evaluator._windowed_kernel("Premult", {**SPECS["Premult"]["params"], "premult_by": "nothere"}, [raster])

    def test_dividing_by_zero_or_less_leaves_the_pixel(self):
        pixels = image()
        pixels[2, :3, 0] = 0.0                       # red = 0
        pixels[2, 3:5, 0] = -0.3
        out = run("Unpremult", pixels, premult_by="rgba.red")
        np.testing.assert_array_equal(out[2, :5, :3], pixels[2, :5, :3])
        self.assertTrue(np.all(np.isfinite(out)))


class InvertTests(unittest.TestCase):
    def test_invert_swaps_the_two_operations(self):
        pixels = image()
        np.testing.assert_array_equal(run("Premult", pixels, invert=1), old_unpremult(pixels))
        np.testing.assert_array_equal(run("Unpremult", pixels, invert=1), old_premult(pixels))

    def test_invert_follows_the_channels_and_the_divisor(self):
        pixels = image()
        out = run("Premult", pixels, invert=1, channels="rgba", premult_by="rgba.green")
        green = pixels[..., 1:2]
        safe = np.where(green > 0, green, 1)
        np.testing.assert_allclose(out, np.where(green > 0, pixels / safe, pixels), atol=1e-6)

    def test_premult_then_unpremult_comes_back_where_alpha_is_above_zero(self):
        pixels = image()
        back = run("Unpremult", run("Premult", pixels))
        visible = pixels[..., 3] > 0
        np.testing.assert_allclose(back[visible], pixels[visible], atol=1e-6)


class PathTests(unittest.TestCase):
    def document(self, kind, **knobs):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "src", "type": "Radial", "params": {"width": W, "height": H, "box_x": 2, "box_y": 2, "box_width": 18,
                                                                              "box_height": 10, "softness": 0.6, "red": 0.9, "green": 0.5, "blue": 0.2, "alpha": 0.8}})
        d.execute({"op": "create", "id": "n", "type": kind, "params": knobs})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
        d.execute({"op": "view", "id": "n"})
        return d.document

    def test_tiles_equal_the_evaluator_for_every_option(self):
        options = (dict(), dict(channels="rgba"), dict(channels="alpha"), dict(channels="red"), dict(channels="all"), dict(invert=1),
                   dict(premult_by="rgba.red"), dict(premult_by="rgba.luminance", channels="rgba", invert=1))
        for kind in KINDS:
            for option in options:
                with self.subTest(kind=kind, option=option):
                    document = self.document(kind, **option)
                    executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                    self.assertTrue(executor.supports_tiled(document, "n"))
                    expected = Evaluator().evaluate_raster(document, target="n").pixels
                    region = executor.canvas_region(document, "n", frame=1, tier=1)
                    got = executor.compose_region(document, "n", region, frame=1, tier=1).pixels
                    np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_a_layer_divisor_is_rendered_whole(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        self.assertTrue(executor.supports_tiled(self.document("Premult"), "n"))
        self.assertFalse(executor.supports_tiled(self.document("Premult", premult_by="matte.R"), "n"))
        self.assertTrue(chrome.needs_full_frame("Unpremult", dict(SPECS["Unpremult"]["params"], premult_by="matte.R")))
        self.assertFalse(chrome.needs_full_frame("Unpremult", dict(SPECS["Unpremult"]["params"], premult_by="rgba.red")))

    def test_bypass_passes_the_image(self):
        for kind in KINDS:
            document = self.document(kind, channels="rgba", invert=1, premult_by="rgba.red")
            plain = Evaluator().evaluate_raster(document, target="src").pixels
            self.assertFalse(np.array_equal(Evaluator().evaluate_raster(document, target="n").pixels, plain))
            document["nodes"]["n"]["disabled"] = True
            np.testing.assert_array_equal(Evaluator().evaluate_raster(document, target="n").pixels, plain)
            self.assertEqual(bypass_slot(document["nodes"]["n"]), "image")


class DocumentTests(unittest.TestCase):
    def test_a_premult_saved_without_params_loads_with_the_defaults_and_renders_the_same(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                document = PathTests().document(kind)
                old = copy.deepcopy(document)
                old["nodes"]["n"]["params"] = {}
                loaded = upgrade_document(old)
                validate(loaded)
                self.assertEqual(loaded["nodes"]["n"]["params"], {"channels": "rgb", "premult_by": "rgba.alpha", "invert": 0})
                np.testing.assert_array_equal(Evaluator().evaluate_raster(loaded, target="n").pixels,
                                              Evaluator().evaluate_raster(document, target="n").pixels)


if __name__ == "__main__":
    unittest.main()
