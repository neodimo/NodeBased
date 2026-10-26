"""Fidelity step D, part 2: the Keyer takes Nuke's input, output, combine and screen operations; the ChromaKeyer takes screen
gain, screen balance, chroma gain, black and white points and its view; the Difference takes its output. Pixels by hand
against the guide's wording, tiles against the evaluator, bypass, and old documents."""
import copy
import unittest

import numpy as np

from nodebased import smallnodes
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tileexec import TileExecutor

H, W = 12, 20
KEY = (0.1, 0.8, 0.2)


def run(kind, pixels, *more, **knobs):
    inputs = [Raster.of(np.asarray(p, np.float32)) for p in (pixels, *more)]
    return Evaluator._windowed_kernel(kind, {**SPECS[kind]["params"], **knobs}, inputs).pixels


def one(*rgba):
    return np.array([[list(rgba)]], np.float32)


class KeyerTests(unittest.TestCase):
    ramp = dict(range_a=0.0, range_b=0.5, range_c=2.0, range_d=3.0)        # alpha = min(value / 0.5, 1) for 0 <= value <= 2

    def key(self, rgba, **knobs):
        return run("Keyer", one(*rgba), **{**self.ramp, **knobs})[0, 0]

    def test_redscreen_greenscreen_and_bluescreen_subtract_the_larger_other_channel(self):
        pixel = (0.9, 0.3, 0.5, 1.0)
        self.assertAlmostEqual(float(self.key(pixel, keyer_operation="redscreen")[3]), 0.8, places=6)     # 0.9 - 0.5 = 0.4
        self.assertAlmostEqual(float(self.key(pixel, keyer_operation="greenscreen")[3]), 0.0, places=6)   # 0.3 - 0.9 < 0
        self.assertAlmostEqual(float(self.key(pixel, keyer_operation="bluescreen")[3]), 0.0, places=6)    # 0.5 - 0.9 < 0
        self.assertAlmostEqual(float(self.key((0.2, 0.7, 0.4, 1.0), keyer_operation="greenscreen")[3]), 0.6, places=6)
        self.assertAlmostEqual(float(self.key((0.2, 0.45, 0.4, 1.0), keyer_operation="greenscreen")[3]), 0.1, places=6)

    def test_the_colour_passes_through_and_only_the_alpha_is_written(self):
        out = self.key((0.9, 0.3, 0.5, 0.7), keyer_operation="redscreen")
        np.testing.assert_array_equal(out[:3], np.array([0.9, 0.3, 0.5], np.float32))

    def test_input_can_be_the_alpha_channel(self):
        out = self.key((0.9, 0.1, 0.1, 0.25), key_input="rgba.alpha")
        self.assertAlmostEqual(float(out[3]), 0.5, places=6)           # a grey image of 0.25: luminance 0.25 -> 0.25 / 0.5
        self.assertAlmostEqual(float(self.key((0.9, 0.1, 0.1, 0.25), key_input="rgb", keyer_operation="red")[3]), 1.0, places=6)

    def test_output_can_be_another_channel_and_the_alpha_stays(self):
        out = self.key((0.2, 0.3, 0.4, 0.6), keyer_operation="red", key_output="rgba.green")
        self.assertAlmostEqual(float(out[1]), 0.4, places=6)       # red 0.2 -> 0.2 / 0.5 = 0.4 written to green
        self.assertEqual([float(out[0]), float(out[2]), float(out[3])], [float(np.float32(v)) for v in (0.2, 0.4, 0.6)])

    def test_output_none_writes_nothing(self):
        pixel = one(0.2, 0.3, 0.4, 0.6)
        np.testing.assert_array_equal(run("Keyer", pixel, key_output="none", **self.ramp), pixel)

    def test_combine_replace_intersect_and_union_use_what_the_channel_holds(self):
        # red 0.2 keys to 0.4 into alpha, which holds 0.6
        replace = self.key((0.2, 0.3, 0.4, 0.6), keyer_operation="red")
        intersect = self.key((0.2, 0.3, 0.4, 0.6), keyer_operation="red", key_combine="intersect")
        union = self.key((0.2, 0.3, 0.4, 0.6), keyer_operation="red", key_combine="union")
        self.assertEqual([float(replace[3]), float(intersect[3]), float(union[3])],
                         [float(np.float32(0.4)), float(np.float32(0.4)), float(np.float32(0.6))])
        low = self.key((0.2, 0.3, 0.4, 0.3), keyer_operation="red", key_combine="intersect")
        high = self.key((0.2, 0.3, 0.4, 0.3), keyer_operation="red", key_combine="union")
        self.assertEqual([float(low[3]), float(high[3])], [float(np.float32(0.3)), float(np.float32(0.4))])

    def test_defaults_are_the_keyer_of_before(self):
        rng = np.random.default_rng(5)
        pixels = rng.random((H, W, 4)).astype(np.float32)
        legacy = {k: v for k, v in SPECS["Keyer"]["params"].items() if k not in smallnodes.NEW["Keyer"]}
        old = Evaluator._keyer(pixels, legacy)
        np.testing.assert_array_equal(run("Keyer", pixels), old)
        np.testing.assert_array_equal(old[..., :3], pixels[..., :3])


def chroma_pixel(t, luma=0.5, key=KEY):
    """A pixel `t` of the way from the screen colour to grey, on the colour's own brightness: its chroma distance is `t`."""
    key = np.array(key, np.float32)
    key_y = 0.2126 * key[0] + 0.7152 * key[1] + 0.0722 * key[2]
    n_key = key / key_y
    normalised = (1 - t) * n_key + t
    return one(*(normalised * luma), 1.0)


class ChromaKeyerTests(unittest.TestCase):
    base = dict(key_red=KEY[0], key_green=KEY[1], key_blue=KEY[2], key_tolerance=0.35, key_softness=0.3, shadow_level=0.0,
                despill=0, premultiply=0)

    def alpha(self, pixel, **knobs):
        return float(run("ChromaKeyer", pixel, **{**self.base, **knobs})[0, 0, 3])

    def test_the_screen_is_transparent_and_grey_is_opaque(self):
        self.assertAlmostEqual(self.alpha(chroma_pixel(0.0)), 0.0, places=5)
        self.assertAlmostEqual(self.alpha(chroma_pixel(1.0)), 1.0, places=5)
        self.assertAlmostEqual(self.alpha(chroma_pixel(0.5)), 0.5, places=4)          # (0.5 - 0.35) / 0.3

    def test_screen_gain_divides_the_distance_so_more_gain_keys_more(self):
        grey = chroma_pixel(1.0)
        self.assertAlmostEqual(self.alpha(grey, screen_gain=2.0), 0.5, places=5)       # 1 / 2 = 0.5 -> (0.5 - 0.35) / 0.3
        self.assertAlmostEqual(self.alpha(grey, screen_gain=4.0), 0.0, places=5)       # 0.25 is under the tolerance
        values = [self.alpha(chroma_pixel(0.7), screen_gain=g) for g in (0.5, 1.0, 2.0, 4.0)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], values[-1])

    def test_screen_balance_weighs_the_two_other_channels(self):
        # a pixel that differs from the screen in red only
        key = np.array(KEY, np.float32)
        key_y = 0.2126 * key[0] + 0.7152 * key[1] + 0.0722 * key[2]
        n_key = key / key_y
        for balance in (0.0, 0.25, 0.5, 0.8, 1.0):
            normalised = n_key.copy()
            normalised[0] += 0.4
            luma = 0.2126 * normalised[0] + 0.7152 * normalised[1] + 0.0722 * normalised[2]
            pixel = one(*(normalised / luma * 0.5), 1.0)
            n = normalised / luma
            weights = np.array([2 * (1 - balance), 1.0, 2 * balance])
            scale = np.sqrt(np.sum(weights * (n_key - 1.0) ** 2))
            dist = np.sqrt(np.sum(weights * (n - n_key) ** 2)) / scale
            expected = float(np.clip((dist - 0.35) / 0.3, 0, 1))
            with self.subTest(balance=balance):
                self.assertAlmostEqual(self.alpha(pixel, screen_balance=balance), expected, places=4)
        self.assertNotAlmostEqual(self.alpha(pixel, screen_balance=0.0), self.alpha(pixel, screen_balance=1.0), places=2)

    def test_chroma_gain_returns_opacity_to_pixels_away_from_the_screen(self):
        half = chroma_pixel(0.5)                                                        # alpha 0.5, chroma 0.5
        self.assertAlmostEqual(self.alpha(half, chroma_gain=1.0), 0.5 + 0.5 * 0.5, places=4)
        self.assertAlmostEqual(self.alpha(half, chroma_gain=0.5), 0.5 + 0.5 * 0.25, places=4)
        self.assertAlmostEqual(self.alpha(chroma_pixel(0.0), chroma_gain=1.0), 0.0, places=5)   # the screen stays clear

    def test_black_and_white_points_cut_and_stretch_the_matte(self):
        half = chroma_pixel(0.5)
        self.assertAlmostEqual(self.alpha(half, black_point=0.1, white_point=0.6), 0.8, places=4)   # (0.5 - 0.1) / 0.5
        self.assertAlmostEqual(self.alpha(half, white_point=0.4), 1.0, places=5)                     # above the white point: 1
        self.assertAlmostEqual(self.alpha(half, black_point=0.6), 0.0, places=5)                     # below the black point: 0

    def test_the_intermediate_view_is_the_source_with_the_matte(self):
        pixel = chroma_pixel(0.3)
        final = run("ChromaKeyer", pixel, **{**self.base, "despill": 1, "premultiply": 1})[0, 0]
        mid = run("ChromaKeyer", pixel, **{**self.base, "despill": 1, "premultiply": 1, "key_view": "intermediate result"})[0, 0]
        np.testing.assert_array_equal(mid[:3], pixel[0, 0, :3])
        self.assertAlmostEqual(float(mid[3]), float(final[3]), places=6)
        self.assertFalse(np.allclose(final[:3], pixel[0, 0, :3]))

    def test_defaults_are_the_chroma_keyer_of_before(self):
        rng = np.random.default_rng(6)
        pixels = rng.random((H, W, 4)).astype(np.float32)
        legacy = {k: v for k, v in SPECS["ChromaKeyer"]["params"].items() if k not in smallnodes.NEW["ChromaKeyer"]}
        np.testing.assert_array_equal(run("ChromaKeyer", pixels), Evaluator._chroma_keyer(pixels, legacy))


class DifferenceTests(unittest.TestCase):
    def test_the_alpha_is_the_default_output(self):
        a, b = one(0.2, 0.2, 0.2, 1.0), one(0.5, 0.2, 0.3, 0.9)
        out = run("Difference", a, b)[0, 0]
        np.testing.assert_allclose(out, [0.5, 0.2, 0.3, 0.3], atol=1e-6)

    def test_a_disabled_output_is_b_unchanged(self):
        a, b = one(0.2, 0.2, 0.2, 1.0), one(0.5, 0.2, 0.3, 0.9)
        np.testing.assert_array_equal(run("Difference", a, b, output_enabled=0), b)

    def test_the_second_output_picks_the_channel_the_key_goes_into(self):
        a, b = one(0.2, 0.2, 0.2, 1.0), one(0.5, 0.2, 0.3, 0.9)
        out = run("Difference", a, b, output_channel="rgba.blue")[0, 0]
        np.testing.assert_allclose(out, [0.5, 0.2, 0.3, 0.9], atol=1e-6)            # blue 0.3 is the key; alpha stays B's
        out = run("Difference", a, b, output_channel="rgba.red", gain=2.0)[0, 0]
        np.testing.assert_allclose(out, [0.6, 0.2, 0.3, 0.9], atol=1e-6)


class PathTests(unittest.TestCase):
    def document(self, kind, **knobs):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "a", "type": "Ramp", "params": {
            "width": W, "height": H, "p0_x": 0.0, "p0_y": 0.0, "p1_x": W - 1.0, "p1_y": 0.0, "color0_red": 0.1, "color0_green": 0.8,
            "color0_blue": 0.2, "color0_alpha": 0.9, "color1_red": 0.8, "color1_green": 0.4, "color1_blue": 0.5, "color1_alpha": 0.4}})
        d.execute({"op": "create", "id": "b", "type": "Ramp", "params": {
            "width": W, "height": H, "p0_x": 0.0, "p0_y": 0.0, "p1_x": 0.0, "p1_y": H - 1.0, "color0_red": 0.3, "color0_green": 0.2,
            "color0_blue": 0.6, "color0_alpha": 1.0, "color1_red": 0.6, "color1_green": 0.7, "color1_blue": 0.1, "color1_alpha": 0.6}})
        d.execute({"op": "create", "id": "n", "type": kind, "params": knobs})
        if SPECS[kind]["inputs"] == ["image"]:
            d.execute({"op": "connect", "id": "n", "input": "image", "source": "a"})
        else:
            d.execute({"op": "connect", "id": "n", "input": "A", "source": "a"})
            d.execute({"op": "connect", "id": "n", "input": "B", "source": "b"})
        d.execute({"op": "view", "id": "n"})
        return d.document

    options = {
        "Keyer": (dict(), dict(keyer_operation="redscreen", range_a=0.0, range_b=0.2, range_c=0.6, range_d=0.9),
                  dict(key_input="rgba.alpha", range_a=0.2, range_b=0.7), dict(key_output="rgba.red", keyer_operation="green"),
                  dict(key_combine="union", keyer_operation="blue", range_b=0.4), dict(key_combine="intersect", key_output="rgba.blue")),
        "ChromaKeyer": (dict(), dict(screen_gain=2.0), dict(screen_balance=0.2), dict(chroma_gain=0.7), dict(black_point=0.2, white_point=0.8),
                        dict(key_view="intermediate result", despill=1, premultiply=1), dict(screen_gain=0.5, screen_balance=0.9, chroma_gain=0.3)),
        "Difference": (dict(), dict(output_enabled=0), dict(output_channel="rgba.green", gain=2.0), dict(output_channel="rgba.red")),
    }

    def test_tiles_equal_the_evaluator_for_every_option(self):
        for kind, variants in self.options.items():
            for knobs in variants:
                with self.subTest(kind=kind, knobs=knobs):
                    document = self.document(kind, **knobs)
                    executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
                    self.assertTrue(executor.supports_tiled(document, "n"))
                    expected = Evaluator().evaluate_raster(document, target="n").pixels
                    region = executor.canvas_region(document, "n", frame=1, tier=1)
                    got = executor.compose_region(document, "n", region, frame=1, tier=1).pixels
                    np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_a_changed_option_changes_the_pixels(self):
        for kind, variants in self.options.items():
            plain = Evaluator().evaluate_raster(self.document(kind), target="n").pixels
            for knobs in variants[1:]:
                with self.subTest(kind=kind, knobs=knobs):
                    changed = Evaluator().evaluate_raster(self.document(kind, **knobs), target="n").pixels
                    self.assertFalse(np.array_equal(changed, plain))

    def test_bypass_passes_the_image(self):
        for kind, variants in self.options.items():
            document = self.document(kind, **variants[1])
            slot = bypass_slot(document["nodes"]["n"])
            document["nodes"]["n"]["disabled"] = True
            passed = Evaluator().evaluate_raster(document, target=document["nodes"]["n"]["inputs"][slot]).pixels
            np.testing.assert_array_equal(Evaluator().evaluate_raster(document, target="n").pixels, passed)

    def test_an_older_document_loads_and_renders_the_same(self):
        for kind in self.options:
            with self.subTest(kind=kind):
                document = self.document(kind)
                old = copy.deepcopy(document)
                for name in smallnodes.NEW[kind]:
                    old["nodes"]["n"]["params"].pop(name)
                loaded = upgrade_document(old)
                validate(loaded)
                self.assertEqual(loaded["nodes"]["n"]["params"], document["nodes"]["n"]["params"])
                np.testing.assert_array_equal(Evaluator().evaluate_raster(loaded, target="n").pixels,
                                              Evaluator().evaluate_raster(document, target="n").pixels)


if __name__ == "__main__":
    unittest.main()
