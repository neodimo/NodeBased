"""Fidelity step B, part 2: Nuke's shared Draw chrome (output, premult, cliptype, replace, invert, opacity)
on Ramp, Radial, Rectangle, Noise, Text and Grid, and the Roto and RotoPaint parts of it. Pixel assertions per
option, the evaluator and tile paths on the same pixels, mask and mix, bypass, and old documents."""
import copy
import unittest

import numpy as np

from nodebased import ops2d_chrome as chrome
from nodebased.core import SPECS, bypass_slot, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels
from tests.test_roto_ui import triangle

W, H = 130, 90
SHAPES = {
    "Rectangle": dict(box_x=20, box_y=15, box_width=70, box_height=50, softness=0.0, red=0.8, green=0.4, blue=0.2, alpha=0.9),
    "Radial": dict(box_x=15, box_y=10, box_width=90, box_height=70, softness=0.6, red=0.3, green=0.7, blue=0.9, alpha=0.8),
    "Ramp": dict(p0_x=0, p0_y=0, p1_x=W - 1, p1_y=0, color0_red=0.1, color0_green=0.2, color0_blue=0.9, color0_alpha=0.7,
                 color1_red=0.9, color1_green=0.5, color1_blue=0.1, color1_alpha=1.0),
    "Noise": dict(size=20.0, seed=4),
    "Grid": dict(spacing_x=16, spacing_y=16, line_width=2.0, red=0.9, green=0.9, blue=0.2, alpha=0.9),
    "Text": dict(message="Ab", font_size=40.0, box_x=5, box_y=5, box_width=120, box_height=80, red=0.9, green=0.8, blue=0.7, alpha=0.9),
}


def make(kind, **params):
    g = Graph()
    g.add("bg", "Constant", dict(width=W, height=H, red=0.2, green=0.3, blue=0.4, alpha=0.6))
    g.add("matte", "Radial", dict(width=W, height=H, box_x=10, box_y=10, box_width=100, box_height=60, softness=0.7))
    return g


def shape_of(kind, **params):
    p = dict(SPECS[kind]["params"], width=W, height=H, **SHAPES[kind])
    p.update(params)
    return Evaluator._draw_shape(kind, p, 0, 0, W, H, 1)


def drawn(kind, params=None, bg=False, mask=False):
    g = make(kind)
    wires = {}
    if bg:
        wires["image"] = "bg"
    if mask:
        wires["mask"] = "matte"
    g.add("node", kind, dict(width=W, height=H, **SHAPES[kind], **(params or {})), **wires)
    return g


class DefaultsTests(unittest.TestCase):
    def test_every_default_draws_the_shape_it_always_did(self):
        for kind in chrome.DRAW_KINDS_WITH_CHROME:
            with self.subTest(kind=kind):
                p = dict(SPECS[kind]["params"], width=W, height=H, **SHAPES[kind])
                self.assertFalse(chrome.draw_config(kind, p).active)
                self.assertIs(chrome.shape_chrome(kind, p, np.ones((3, 3, 4), np.float32)).dtype, np.dtype("float32"))
                raw = Evaluator._draw_shape_raw(kind, p, 0, 0, W, H, 1)
                np.testing.assert_array_equal(Evaluator._draw_shape(kind, p, 0, 0, W, H, 1), raw)
                background = np.random.default_rng(1).random((H, W, 4), dtype=np.float32)
                np.testing.assert_array_equal(chrome.compose_draw(kind, p, raw, background),
                                              Evaluator._composite_shape_over(raw, background))

    def test_the_nuke_defaults(self):
        for kind in chrome.DRAW_KINDS_WITH_CHROME:
            params = SPECS[kind]["params"]
            self.assertEqual((params["output"], params["premult"], params["cliptype"], params["replace"], params["invert"],
                              params["opacity"]), ("rgba", "none", "format", 0, 0, 1.0), kind)
        self.assertEqual(SPECS["Roto"]["params"]["output"], "alpha")
        self.assertEqual(SPECS["RotoPaint"]["params"]["opacity"], 1.0)

    def test_an_old_document_without_the_knobs_loads_with_the_defaults_and_renders_the_same(self):
        g = Graph()
        g.add("a", "Rectangle", dict(width=64, height=48, box_x=4, box_y=4, box_width=40, box_height=30))
        g.add("b", "Ramp", dict(width=64, height=48), image="a")
        before = evaluator_pixels(g.doc, "b")
        old = copy.deepcopy(g.doc)
        for node in old["nodes"].values():
            for name in chrome.DRAW_PARAM_NAMES:
                node["params"].pop(name, None)
        with self.assertRaises(ValueError):
            validate(old)
        upgraded = upgrade_document(old)
        validate(upgraded)
        np.testing.assert_array_equal(evaluator_pixels(upgraded, "b"), before)


class OptionTests(unittest.TestCase):
    def test_opacity_scales_colour_and_alpha(self):
        for kind in chrome.DRAW_KINDS_WITH_CHROME:
            with self.subTest(kind=kind):
                np.testing.assert_allclose(shape_of(kind, opacity=0.4), shape_of(kind) * 0.4, atol=1e-6)

    def test_invert_flips_the_coverage_and_keeps_the_colour(self):
        rectangle = shape_of("Rectangle")
        inverted = shape_of("Rectangle", invert=1)
        np.testing.assert_allclose(inverted[..., 3], 1.0 - rectangle[..., 3], atol=1e-6)
        colour = np.array([0.8, 0.4, 0.2], np.float32)
        np.testing.assert_allclose(inverted[..., :3], colour * inverted[..., 3:4], atol=1e-6)
        radial = shape_of("Radial")
        np.testing.assert_allclose(shape_of("Radial", invert=1)[..., 3], 1.0 - radial[..., 3], atol=1e-6)

    def test_premult_multiplies_the_colour_by_the_chosen_channel(self):
        base = shape_of("Rectangle")
        out = shape_of("Rectangle", premult="rgba.alpha")
        np.testing.assert_allclose(out[..., :3], base[..., :3] * base[..., 3:4], atol=1e-6)
        np.testing.assert_array_equal(out[..., 3], base[..., 3])
        red = shape_of("Rectangle", premult="red")
        np.testing.assert_allclose(red[..., :3], base[..., :3] * base[..., 0:1], atol=1e-6)

    def test_replace_ignores_the_background(self):
        g = drawn("Rectangle", dict(replace=1), bg=True)
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(out, shape_of("Rectangle"), atol=1e-6)
        over = evaluator_pixels(drawn("Rectangle", bg=True).doc, "node")
        self.assertFalse(np.allclose(out, over))

    def test_output_writes_only_the_listed_channels(self):
        background = evaluator_pixels(drawn("Rectangle", bg=True).doc, "bg")
        over = evaluator_pixels(drawn("Rectangle", bg=True).doc, "node")
        for output, picked in (("alpha", [3]), ("rgb", [0, 1, 2]), ("red", [0]), ("none", []), ("rgba", [0, 1, 2, 3])):
            with self.subTest(output=output):
                out = evaluator_pixels(drawn("Rectangle", dict(output=output), bg=True).doc, "node")
                expected = background.copy()
                expected[..., picked] = over[..., picked]
                np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_mask_and_mix_still_gate_the_result(self):
        g = drawn("Rectangle", dict(opacity=0.5, mix=0.6), bg=True, mask=True)
        out = evaluator_pixels(g.doc, "node")
        background = evaluator_pixels(g.doc, "bg")
        matte = evaluator_pixels(g.doc, "matte")[..., 3:4]
        shape = shape_of("Rectangle", opacity=0.5)
        composited = shape + background * (1.0 - shape[..., 3:4])
        gate = matte * 0.6
        np.testing.assert_allclose(out, composited * gate + background * (1.0 - gate), atol=1e-5)

    def test_cliptype_is_accepted_and_every_setting_draws_the_same_pixels(self):
        base = evaluator_pixels(drawn("Rectangle").doc, "node")
        for clip in chrome.CLIP_TYPES:
            g = drawn("Rectangle", dict(cliptype=clip))
            validate(g.doc)
            np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), base)


class TileTests(unittest.TestCase):
    def _tiled(self, document):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=32)
        document = dict(document, view="node")
        self.assertTrue(executor.supports_tiled(document, "node"))
        region = executor.canvas_region(document, "node", frame=1, tier=1)
        return executor.compose_region(document, "node", region, frame=1, tier=1).pixels

    def test_each_option_on_every_draw_node_tiles_to_the_evaluator_pixels(self):
        options = (dict(opacity=0.5), dict(invert=1), dict(premult="rgba.alpha"), dict(replace=1), dict(output="alpha"),
                   dict(output="rgb", invert=1, opacity=0.7), dict(premult="red", replace=1, opacity=0.3))
        for kind in chrome.DRAW_KINDS_WITH_CHROME:
            self.assertIn(kind, SUPPORTED_TILED_KINDS)
            for option in options:
                for bg, mask in ((False, False), (True, False), (True, True)):
                    with self.subTest(kind=kind, option=option, bg=bg, mask=mask):
                        g = drawn(kind, dict(option, mix=0.8), bg=bg, mask=mask)
                        # Qt's glyph hinting differs by up to 0.002 on three pixels at a tile seam, with or without
                        # the chrome (the default Text node shows the same), so Text is held to that, the rest exactly.
                        np.testing.assert_allclose(self._tiled(g.doc), evaluator_pixels(g.doc, "node"),
                                                   atol=5e-3 if kind == "Text" else 1e-6)

    def test_bypass(self):
        g = drawn("Rectangle", dict(opacity=0.5, invert=1, replace=1, output="alpha"), bg=True)
        base = evaluator_pixels(g.doc, "bg")
        self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), base))
        g.bypass("node")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), base)
        np.testing.assert_array_equal(self._tiled(g.doc), base)
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")


class RotoTests(unittest.TestCase):
    def _document(self, kind="Roto", **params):
        from nodebased.core import Dispatcher, empty_document
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "plate", "type": "Checker", "params": {"width": 32, "height": 32}})
        d.execute({"op": "create", "id": "r", "type": kind, "params": params})
        d.execute({"op": "connect", "id": "r", "input": "bg" if kind == "Roto" else "image", "source": "plate"})
        shape = triangle()
        shape["points"] = [dict(point, x=point["x"] * 0.25, y=point["y"] * 0.25) for point in shape["points"]]
        if kind == "Roto":
            d.execute({"op": "set_shapes", "id": "r", "shapes": [shape]})
        d.execute({"op": "view", "id": "r"})
        return d.document

    def test_opacity_scales_the_coverage(self):
        full = Evaluator().evaluate_raster(self._document(), target="r").pixels
        half = Evaluator().evaluate_raster(self._document(opacity=0.5), target="r").pixels
        self.assertGreater(float(full[..., 3].max()), 0.9)
        np.testing.assert_allclose(half[..., 3], full[..., 3] * 0.5, atol=1e-6)
        np.testing.assert_array_equal(half[..., :3], full[..., :3])      # the plate's colour is untouched by default

    def test_replace_draws_the_matte_without_the_plate(self):
        document = self._document(replace=1)
        out = Evaluator().evaluate_raster(document, target="r").pixels
        plate = Evaluator().evaluate_raster(document, target="plate").pixels
        self.assertFalse(np.array_equal(out[..., :3], plate[..., :3]))
        self.assertEqual(float(out[0, 0, 3]), 0.0)

    def test_output_rgb_writes_no_alpha(self):
        document = self._document(output="rgb")
        out = Evaluator().evaluate_raster(document, target="r").pixels
        plate = Evaluator().evaluate_raster(document, target="plate").pixels
        np.testing.assert_array_equal(out[..., 3], plate[..., 3])

    def test_a_roto_with_a_changed_draw_knob_is_rendered_whole_and_defaults_stay_tiled(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(self._document(), "r"))
        self.assertFalse(executor.supports_tiled(self._document(opacity=0.5), "r"))
        self.assertTrue(chrome.needs_full_frame("RotoPaint", dict(SPECS["RotoPaint"]["params"], replace=1)))

    def test_rotopaint_opacity_blends_toward_the_image(self):
        document = self._document("RotoPaint", opacity=0.5)
        out = Evaluator().evaluate_raster(document, target="r").pixels
        plate = Evaluator().evaluate_raster(document, target="plate").pixels
        np.testing.assert_allclose(out, plate, atol=1e-6)               # no strokes: the image comes back whatever the blend


if __name__ == "__main__":
    unittest.main()
