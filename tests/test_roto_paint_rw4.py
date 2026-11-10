"""RotoPaint shapes and strokes: their own colour, Invert and Roto's whole blending list, checked in pixels."""
import copy
import unittest

import numpy as np

from nodebased import paint, roto, shapes as shape_model
from nodebased.core import Dispatcher, empty_document

W = H = 64


def square(x0, y0, x1, y1):
    return [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def paint_shape(points=None, **extra):
    return {"kind": "shape", "name": "s", "mode": "union", "opacity": 1.0, "feather": 0.0, "blend": "over", "visible": True,
            "points": points or square(16, 16, 48, 48), **extra}


def paint_stroke(**extra):
    return {"kind": "stroke", "name": "k", "points": [{"x": 32.0, "y": 32.0, "pressure": 1.0}],
            "brush": {"size": 16.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2},
            "tool": "paint", "lifetime": {"mode": "all"}, "color": [0.2, 0.2, 0.2, 1.0], "source_offset": [0.0, 0.0],
            "source_frame": "relative", "opacity": 1.0, "blend": "over", "visible": True, "follow_track": None,
            "patch_blend": 0.0, **extra}


def render(items, base=None, frame=1):
    image = np.zeros((H, W, 4), np.float32) if base is None else base.copy()
    resolved = shape_model.resolve_paint_items({"items": items}, frame)
    return paint.rasterise(image, resolved, frame)


def grey(value):
    image = np.zeros((H, W, 4), np.float32)
    image[..., :3] = value
    image[..., 3] = 1.0
    return image


class RotoPaintShapeColourAndInvertTests(unittest.TestCase):
    def test_a_shape_without_a_colour_still_paints_white(self):
        out = render([paint_shape()])
        np.testing.assert_allclose(out[32, 32], [1, 1, 1, 1])
        np.testing.assert_allclose(out[2, 2], [0, 0, 0, 0])

    def test_the_shape_paints_its_own_colour(self):
        out = render([paint_shape(color=[1.0, 0.25, 0.0, 1.0])])
        np.testing.assert_allclose(out[32, 32], [1.0, 0.25, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out[2, 2], [0, 0, 0, 0])

    def test_the_colours_alpha_scales_the_coverage(self):
        out = render([paint_shape(color=[1.0, 0.0, 0.0, 0.5])])
        np.testing.assert_allclose(out[32, 32], [0.5, 0.0, 0.0, 0.5], atol=1e-6)

    def test_invert_paints_everything_but_the_shape(self):
        out = render([paint_shape(color=[0.0, 1.0, 0.0, 1.0], invert=True)])
        np.testing.assert_allclose(out[32, 32], [0, 0, 0, 0])
        np.testing.assert_allclose(out[2, 2], [0.0, 1.0, 0.0, 1.0], atol=1e-6)

    def test_an_inverted_stroke_paints_everything_but_its_footprint(self):
        plain = render([paint_stroke()], base=grey(0.5))
        inverted = render([paint_stroke(invert=True)], base=grey(0.5))
        np.testing.assert_allclose(plain[32, 32, :3], 0.2, atol=1e-6)
        np.testing.assert_allclose(plain[2, 2, :3], 0.5, atol=1e-6)
        np.testing.assert_allclose(inverted[32, 32, :3], 0.5, atol=1e-6)
        np.testing.assert_allclose(inverted[2, 2, :3], 0.2, atol=1e-6)

    def test_the_items_validate_with_the_new_fields(self):
        document = empty_document()
        dispatcher = Dispatcher(document)
        dispatcher.execute({"op": "create", "id": "p", "type": "RotoPaint", "pos": [0, 0]})
        items = [paint_shape(color=[1.0, 0.0, 0.0, 1.0], invert=True, blend="difference"),
                 paint_stroke(invert=True, blend="hypot")]
        dispatcher.execute({"op": "set_paint_items", "id": "p", "items": items})
        self.assertEqual(dispatcher.document["node_data"]["p"]["items"][1]["blend"], "hypot")
        with self.assertRaises(Exception):
            dispatcher.execute({"op": "set_paint_items", "id": "p", "items": [paint_shape(blend="nonsense")]})
        with self.assertRaises(Exception):
            dispatcher.execute({"op": "set_paint_items", "id": "p", "items": [paint_stroke(invert="yes")]})


class RotoPaintTileTests(unittest.TestCase):
    def test_tiles_match_the_whole_frame(self):
        """The tile path draws a tile from its own origin; the halves joined equal the frame drawn at once."""
        rng = np.random.default_rng(7)
        base = rng.random((H, W, 4)).astype(np.float32)
        items = [paint_shape(color=[0.9, 0.4, 0.1, 1.0], blend="difference"),
                 paint_shape(square(30, 6, 60, 40), color=[0.1, 0.8, 0.5, 1.0], blend="overlay", invert=True),
                 paint_stroke(blend="hypot", invert=True), paint_stroke(blend="atop", color=[0.9, 0.9, 0.2, 1.0])]
        resolved = shape_model.resolve_paint_items({"items": items}, 1)
        whole = paint.rasterise(base.copy(), resolved, 1)
        left = paint.rasterise(base[:, :W // 2].copy(), resolved, 1, origin=(0, 0), canvas_size=(W, H))
        right = paint.rasterise(base[:, W // 2:].copy(), resolved, 1, origin=(W // 2, 0), canvas_size=(W, H))
        np.testing.assert_allclose(np.concatenate((left, right), axis=1), whole, atol=1e-6)


class RotoPaintBlendingTests(unittest.TestCase):
    MODES = ("plus", "minus", "from", "difference", "exclusion", "max", "min", "under", "divide", "mask", "stencil", "in",
             "out", "atop", "xor", "average", "hypot", "matte", "disjoint-over", "conjoint-over", "copy", "geometric",
             "overlay", "hard-light", "soft-light", "color-dodge", "color-burn")

    def test_the_list_is_roto_s_with_the_four_classic_modes_first(self):
        self.assertEqual(shape_model.PAINT_BLEND_MODES[:4], ("over", "add", "multiply", "screen"))
        self.assertTrue(set(shape_model.BLEND_MODES) <= set(shape_model.PAINT_BLEND_MODES))
        self.assertTrue(set(roto.MERGE_BLEND_MODES) <= set(shape_model.PAINT_BLEND_MODES))

    def test_every_mode_reads_the_same_in_rotopaint_and_roto(self):
        """A RotoPaint shape laid over a picture equals Roto's second shape with that mode over the first."""
        first = {"name": "a", "mode": "union", "opacity": 1.0, "feather": 0.0, "points": square(8, 8, 40, 40),
                 "color": [0.2, 0.6, 0.9, 1.0]}
        base = roto.rasterise([first], W, H)
        points = square(24, 24, 56, 56)
        for mode in self.MODES:
            with self.subTest(mode=mode):
                second = {"name": "b", "mode": "union", "opacity": 1.0, "feather": 0.0, "points": points,
                          "color": [0.8, 0.3, 0.5, 1.0], "blend_mode": mode}
                expected = roto.rasterise([first, second], W, H)
                actual = render([paint_shape(points, color=[0.8, 0.3, 0.5, 1.0], blend=mode)], base=base)
                np.testing.assert_allclose(actual, expected, atol=2e-5)

    def test_modes_change_the_pixels(self):
        base = grey(0.5)
        results = {mode: render([paint_shape(color=[0.2, 0.2, 0.2, 1.0], blend=mode)], base=base)[32, 32, :3].tolist()
                   for mode in ("over", "plus", "minus", "difference", "max", "min", "average", "overlay")}
        np.testing.assert_allclose(results["over"], [0.2, 0.2, 0.2], atol=1e-6)
        self.assertAlmostEqual(results["plus"][0], 0.7, places=5)
        self.assertAlmostEqual(results["minus"][0], 0.0, places=5)
        self.assertAlmostEqual(results["difference"][0], 0.3, places=5)
        self.assertAlmostEqual(results["max"][0], 0.5, places=5)
        self.assertAlmostEqual(results["min"][0], 0.2, places=5)
        self.assertGreater(len({tuple(v) for v in results.values()}), 5)

    def test_a_stroke_blends_through_the_same_formulas(self):
        base = grey(0.5)
        for mode, expected in (("difference", 0.3), ("plus", 0.7), ("min", 0.2), ("max", 0.5)):
            with self.subTest(mode=mode):
                out = render([paint_stroke(blend=mode)], base=base)
                self.assertAlmostEqual(float(out[32, 32, 0]), expected, places=5)
                np.testing.assert_allclose(out[2, 2], base[2, 2], atol=1e-6)

    def test_the_four_classic_modes_keep_their_formulas(self):
        base = grey(0.5)
        shape = dict(color=[1.0, 1.0, 1.0, 1.0])
        for mode, rgb in (("over", 1.0), ("add", 1.0), ("multiply", 0.5), ("screen", 1.0)):
            with self.subTest(mode=mode):
                out = render([paint_shape(blend=mode, **shape)], base=base)
                self.assertAlmostEqual(float(out[32, 32, 0]), rgb, places=6)
        out = render([paint_stroke(blend="add", color=[0.2, 0.2, 0.2, 1.0])], base=base)
        self.assertAlmostEqual(float(out[32, 32, 0]), 0.7, places=6)

    def test_a_document_saved_before_this_loads_and_renders_the_same(self):
        document = {"items": [paint_shape(blend="screen"), paint_stroke(blend="multiply")]}
        self.assertEqual(shape_model.validate_payload("RotoPaint", copy.deepcopy(document), "p"), None)


if __name__ == "__main__":
    unittest.main()
