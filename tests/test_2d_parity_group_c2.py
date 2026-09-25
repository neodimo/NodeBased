"""Lane L2 step 2c2: Ramp, Radial, Rectangle, Noise, Text (Draw menu). Pixel assertions against
hand-computed values, evaluator/tile-path parity (including at tile seams), the optional-image
composite-over contract, mask + mix, bypass (wired and unwired) and CHOICES/LIMITS coverage. See
docs/PARITY_2D.md for the audit these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, DRAW_KINDS, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

GROUP_C2 = DRAW_KINDS  # ("Ramp", "Radial", "Rectangle", "Noise", "Text")


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def bypass(self, key, value=True):
        self.d.execute(dict(op="disable", id=key, value=value))

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target):
    return Evaluator().evaluate(dict(document, view=target))


def tile_pixels(document, target, tile_edge=None):
    kwargs = {} if tile_edge is None else {"tile_edge": tile_edge}
    executor = TileExecutor(evaluator=Evaluator(), **kwargs)
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


class KernelPixelTests(unittest.TestCase):
    """Hand-computed pixel assertions, per the lane brief's worked examples."""

    def test_ramp_horizontal_black_to_white_is_half_at_the_middle_column(self):
        g = Graph()
        g.add("node", "Ramp", dict(width=11, height=3, p0_x=0.0, p0_y=0.0, p1_x=10.0, p1_y=0.0,
                                   color0_red=0.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0,
                                   color1_red=1.0, color1_green=1.0, color1_blue=1.0, color1_alpha=1.0))
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(out[:, 0, :3], 0.0)
        np.testing.assert_allclose(out[:, 10, :3], 1.0)
        np.testing.assert_allclose(out[:, 5, :3], 0.5)

    def test_radial_is_one_at_centre_and_zero_outside_the_box(self):
        g = Graph()
        g.add("node", "Radial", dict(width=40, height=30, box_x=10.0, box_y=5.0, box_width=20.0,
                                     box_height=20.0, softness=0.3, red=1.0, green=1.0, blue=1.0, alpha=1.0))
        out = evaluator_pixels(g.doc, "node")
        cx, cy = 10 + int((20 - 1) / 2), 5 + int((20 - 1) / 2)
        np.testing.assert_allclose(out[cy, cx], [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.all(out[0, :] == 0.0))
        self.assertTrue(np.all(out[:, 0] == 0.0))

    def test_rectangle_is_one_inside_and_zero_outside_with_softness_zero(self):
        g = Graph()
        g.add("node", "Rectangle", dict(width=30, height=20, box_x=10.0, box_y=5.0, box_width=8.0,
                                        box_height=6.0, softness=0.0, red=1.0, green=1.0, blue=1.0, alpha=1.0))
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(out[5:11, 10:18], 1.0)
        np.testing.assert_allclose(out[0, 0], 0.0)
        np.testing.assert_allclose(out[19, 29], 0.0)
        np.testing.assert_allclose(out[4, 10], 0.0)   # one row above the box
        np.testing.assert_allclose(out[5, 18], 0.0)   # one column right of the box

    def test_noise_is_identical_across_two_runs_with_the_same_seed_and_differs_with_another(self):
        def render(seed):
            g = Graph()
            g.add("node", "Noise", dict(width=24, height=18, seed=seed))
            return evaluator_pixels(g.doc, "node")
        first = render(7)
        second = render(7)
        third = render(8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_text_sets_some_alpha_inside_its_box_and_none_far_outside(self):
        g = Graph()
        g.add("node", "Text", dict(width=200, height=100, message="Hi", font_size=48.0,
                                   box_x=10.0, box_y=10.0, box_width=180.0, box_height=80.0,
                                   justify="left", red=1.0, green=1.0, blue=1.0, alpha=1.0))
        out = evaluator_pixels(g.doc, "node")
        self.assertGreater(out[..., 3].sum(), 0.0, "text must set some alpha")
        # Two rows at the very top of the frame, above the text box and its glyph ascenders, are
        # untouched by any font's rendering of a short message at this size.
        np.testing.assert_allclose(out[0:2, :, 3], 0.0)


class MaskMixTests(unittest.TestCase):
    """mix=0 is identity to the (possibly transparent) background; a zero mask hides the shape."""

    def _visible_params(self, kind, mix):
        common = dict(width=16, height=12, mix=mix)
        if kind == "Ramp":
            return dict(common, color0_red=0.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0,
                       color1_red=1.0, color1_green=1.0, color1_blue=1.0, color1_alpha=1.0)
        if kind in ("Radial", "Rectangle"):
            return dict(common, box_x=2.0, box_y=2.0, box_width=12.0, box_height=8.0, softness=0.0,
                       red=1.0, green=0.0, blue=0.0, alpha=1.0)
        if kind == "Noise":
            return dict(common, seed=3)
        if kind == "Grid":
            return dict(common, spacing_x=4.0, spacing_y=4.0, red=1.0, green=0.0, blue=0.0, alpha=1.0)
        return dict(common, message="Hi", box_x=0.0, box_y=0.0, box_width=16.0, box_height=12.0)

    def test_kinds_mix_zero_is_identity_to_the_background(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Constant", dict(width=16, height=12, red=0.4, green=0.4, blue=0.4, alpha=1.0))
                g.add("node", kind, self._visible_params(kind, 0.0), image="plate")
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)

    def test_kinds_zero_mask_hides_the_shape(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Constant", dict(width=16, height=12, red=0.4, green=0.4, blue=0.4, alpha=1.0))
                g.add("matte", "Constant", dict(width=16, height=12, alpha=0.0))
                g.add("node", kind, self._visible_params(kind, 1.0), image="plate", mask="matte")
                plate = evaluator_pixels(g.doc, "plate")
                out = evaluator_pixels(g.doc, "node")
                np.testing.assert_allclose(out, plate)


class ImageCompositeTests(unittest.TestCase):
    """The shape is drawn *over* an optional wired input, Nuke's own Draw-node convention."""

    def test_rectangle_covers_the_background_where_it_is_opaque(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=10, height=10, red=0.1, green=0.2, blue=0.3, alpha=1.0))
        g.add("node", "Rectangle", dict(width=10, height=10, box_x=0.0, box_y=0.0, box_width=10.0,
                                        box_height=10.0, softness=0.0, red=1.0, green=0.0, blue=0.0,
                                        alpha=1.0), image="plate")
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(out[5, 5], [1.0, 0.0, 0.0, 1.0])

    def test_rectangle_shows_the_background_outside_its_own_box(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=10, height=10, red=0.1, green=0.2, blue=0.3, alpha=1.0))
        g.add("node", "Rectangle", dict(width=10, height=10, box_x=0.0, box_y=0.0, box_width=2.0,
                                        box_height=2.0, softness=0.0, red=1.0, green=0.0, blue=0.0,
                                        alpha=1.0), image="plate")
        plate = evaluator_pixels(g.doc, "plate")
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(out[8, 8], plate[8, 8])

    def test_mismatched_image_format_raises(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=10, height=10))
        g.add("node", "Rectangle", dict(width=20, height=20), image="plate")
        with self.assertRaises(ValueError):
            evaluator_pixels(g.doc, "node")


class TilePathParityTests(unittest.TestCase):
    """Every group-c2 kind renders byte-identical pixels on the evaluator and the tile path,
    including at tile seams (a canvas larger than one tile forces the tile executor to stitch)."""

    def _cases(self):
        return {
            "Ramp": dict(p0_x=0.0, p0_y=0.0, p1_x=47.0, p1_y=31.0, mix=1.0),
            "Radial": dict(box_x=5.0, box_y=5.0, box_width=20.0, box_height=15.0, softness=0.4, mix=1.0),
            "Rectangle": dict(box_x=5.0, box_y=5.0, box_width=20.0, box_height=15.0, softness=0.2, mix=1.0),
            "Noise": dict(size=12.0, octaves=3, seed=9, mix=1.0),
            "Text": dict(message="Tile seam", box_x=2.0, box_y=2.0, box_width=44.0, box_height=28.0, mix=1.0),
        }

    def test_draw_kinds_match_across_both_paths_with_a_wired_image_and_mask(self):
        for kind, params in self._cases().items():
            with self.subTest(kind=kind):
                self.assertIn(kind, SUPPORTED_TILED_KINDS)
                g = Graph()
                g.add("plate", "Checker", dict(width=48, height=32, size=8))
                g.add("matte", "Constant", dict(width=48, height=32, red=1, green=1, blue=1, alpha=0.5))
                g.add("node", kind, dict(params, width=48, height=32), image="plate", mask="matte")
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node")
                np.testing.assert_allclose(ti, ev, atol=1e-5)

    def test_draw_kinds_are_seamless_across_multiple_small_tiles(self):
        # A canvas well larger than a deliberately small tile edge forces several tiles to stitch;
        # identical output to the full-frame evaluator is the seam assertion.
        for kind, params in self._cases().items():
            with self.subTest(kind=kind):
                g = Graph()
                g.add("node", kind, dict(params, width=131, height=97))
                ev = evaluator_pixels(g.doc, "node")
                ti = tile_pixels(g.doc, "node", tile_edge=32)
                np.testing.assert_allclose(ti, ev, atol=1e-5)


class BypassTests(unittest.TestCase):
    """A disabled Draw node passes its wired image through, or a transparent frame when unwired."""

    def test_bypass_slot_is_the_optional_image_input(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                node = dict(type=kind, inputs={"image": "x", "mask": None})
                self.assertEqual(bypass_slot(node), "image")

    def test_bypassed_with_image_wired_equals_that_image_on_both_paths(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=32, height=24, size=8))
                g.add("node", kind, dict(width=32, height=24), image="plate")
                upstream = evaluator_pixels(g.doc, "plate")
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))

    def test_bypassed_with_nothing_wired_is_transparent_at_its_own_format(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("node", kind, dict(width=17, height=9))
                g.bypass("node")
                out = evaluator_pixels(g.doc, "node")
                self.assertEqual(out.shape, (9, 17, 4))
                self.assertTrue(np.all(out == 0.0))

    def test_draw_kinds_can_be_disabled_unlike_pure_generators(self):
        # Constant/Checker/Roto have nothing bypass_slot can name and stay refused; Draw nodes have
        # an optional image slot to name, so the same dispatcher command that refuses one accepts
        # the other.
        for kind in ("Constant", "Checker", "Roto"):
            with self.subTest(kind=kind):
                d = Dispatcher()
                d.execute(dict(op="create", id="n", type=kind))
                with self.assertRaises(ValueError):
                    d.execute(dict(op="disable", id="n", value=True))
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                d = Dispatcher()
                d.execute(dict(op="create", id="n", type=kind))
                d.execute(dict(op="disable", id="n", value=True))  # must not raise


class SpecCoverageTests(unittest.TestCase):
    def test_all_five_nodes_are_registered_with_mask_and_mix(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertEqual(SPECS[kind]["inputs"], [])
                self.assertIn("image", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mix", SPECS[kind]["params"])
                self.assertEqual(LIMITS["mix"], (0, 1))

    def test_choices_and_limits_cover_every_new_ranged_param(self):
        self.assertIn("justify", CHOICES)
        for name in ("p0_x", "p0_y", "p1_x", "p1_y", "color0_red", "color0_alpha", "color1_red",
                    "color1_alpha", "box_x", "box_y", "box_width", "box_height", "softness",
                    "z_slice", "octaves", "lacunarity", "seed", "font_size"):
            self.assertIn(name, LIMITS)

    def test_dispatcher_creates_every_new_node_with_valid_defaults(self):
        for kind in GROUP_C2:
            with self.subTest(kind=kind):
                d = Dispatcher()
                result = d.execute(dict(op="create", type=kind))
                self.assertIn("id", result["result"])


if __name__ == "__main__":
    unittest.main()
