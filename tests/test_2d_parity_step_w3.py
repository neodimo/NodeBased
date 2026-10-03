"""Lane L2 step W3: Blend's fringe, inject and mask channel, and ScreenKeyer's inside, outside and clean
plate inputs and its separate despill and alpha-bias colours. Pixel assertions on both paths."""
import unittest

import numpy as np

from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from tests.test_2d_parity_step_k1 import Graph, evaluator_pixels, tile_pixels


def px(r, g, b, a=1.0, shape=(4, 4)):
    out = np.zeros(shape + (4,), dtype=np.float32)
    out[...] = (r, g, b, a)
    return out


def blend(layers, mask=None, **params):
    slots = list(layers) + [None] * (16 - len(layers)) + [mask]
    return Evaluator._kernel("Blend", dict(SPECS["Blend"]["params"], **params), slots)


class BlendFringeTests(unittest.TestCase):
    def test_off_is_the_plain_weighted_average(self):
        a, b = px(1, 0, 0, 1), px(0, 0, 0.2, 0.2)
        np.testing.assert_allclose(blend([a, b])[0, 0], (0.5, 0.0, 0.1, 0.6), atol=1e-6)

    def test_fringe_at_a_soft_edge_leans_toward_the_thinner_inputs_colour(self):
        # A is opaque red; B is straight blue at 0.2 coverage (premultiplied 0, 0, 0.2, 0.2).
        a, b = px(1, 0, 0, 1), px(0, 0, 0.2, 0.2)
        plain, fringed = blend([a, b])[0, 0], blend([a, b], fringe=1)[0, 0]
        np.testing.assert_allclose(fringed, (0.3, 0.0, 0.3, 0.6), atol=1e-6)
        self.assertGreater(fringed[2], plain[2])      # the thin input's blue counts for more
        self.assertLess(fringed[0], plain[0])         # and the opaque input's red for less
        self.assertAlmostEqual(float(fringed[3]), float(plain[3]), places=6)   # coverage is unchanged

    def test_fringe_matches_plain_where_every_input_is_opaque_or_empty(self):
        for a, b in ((px(1, 0, 0, 1), px(0, 1, 0, 1)), (px(0, 0, 0, 0), px(0, 0, 0, 0))):
            np.testing.assert_allclose(blend([a, b], fringe=1), blend([a, b]), atol=1e-6)

    def test_fringe_works_without_normalize_and_with_weights(self):
        a, b = px(1, 0, 0, 1), px(0, 0, 0.2, 0.2)
        out = blend([a, b], fringe=1, normalize=0, weight0=0.5, weight1=2.0)[0, 0]
        np.testing.assert_allclose(out, (0.5 * 0.9, 0.0, 2.0 * 0.9, 0.9), atol=1e-5)   # straight sum (0.5, 0, 2) x alpha 0.9


class BlendMaskChannelAndInjectTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = px(1, 0, 0, 1), px(0, 0, 1, 1)
        self.mask = px(1.0, 0.25, 0.0, 0.5)    # red 1, green 0.25, blue 0, alpha 0.5

    def test_each_mask_channel_gates_by_its_own_value(self):
        mean = blend([self.a, self.b])[0, 0]
        for channel, value in (("alpha", 0.5), ("red", 1.0), ("green", 0.25), ("blue", 0.0)):
            with self.subTest(channel=channel):
                out = blend([self.a, self.b], mask=self.mask, mask_channel=channel)[0, 0]
                np.testing.assert_allclose(out, mean * value + self.a[0, 0] * (1 - value), atol=1e-6)

    def test_luminance_uses_rec709_weights(self):
        out = blend([self.a, self.b], mask=self.mask, mask_channel="luminance")[0, 0]
        luma = 0.2126 * 1.0 + 0.7152 * 0.25
        mean = blend([self.a, self.b])[0, 0]
        np.testing.assert_allclose(out, mean * luma + self.a[0, 0] * (1 - luma), atol=1e-6)

    def test_a_mask_channel_that_is_zero_masks_the_blend_away_entirely(self):
        np.testing.assert_allclose(blend([self.a, self.b], mask=self.mask, mask_channel="blue"), self.a, atol=1e-6)

    def test_inject_writes_the_chosen_channel_into_alpha(self):
        for channel, value in (("alpha", 0.5), ("red", 1.0), ("green", 0.25)):
            with self.subTest(channel=channel):
                out = blend([self.a, self.b], mask=self.mask, mask_channel=channel, inject=1)
                np.testing.assert_allclose(out[..., 3], value, atol=1e-6)
                plain = blend([self.a, self.b], mask=self.mask, mask_channel=channel)
                np.testing.assert_allclose(out[..., :3], plain[..., :3], atol=1e-6)
        self.assertTrue(np.allclose(blend([self.a, self.b], mask=self.mask)[..., 3], 1.0))

    def test_inject_ignores_mix_and_needs_a_wired_mask(self):
        out = blend([self.a, self.b], mask=self.mask, inject=1, mix=0.0)
        np.testing.assert_allclose(out[..., 3], 0.5, atol=1e-6)
        np.testing.assert_allclose(blend([self.a, self.b], inject=1), blend([self.a, self.b]), atol=1e-6)

    def test_registered_with_limits_and_choices(self):
        for name in ("fringe", "inject"):
            self.assertIn(name, LIMITS)
            self.assertIn(name, SPECS["Blend"]["params"])
        self.assertEqual(CHOICES["mask_channel"], ["alpha", "red", "green", "blue", "luminance"])
        self.assertEqual(SPECS["Blend"]["params"]["mask_channel"], "alpha")


class BlendTilePathTests(unittest.TestCase):
    def graph(self, width, height, **params):
        g = Graph()
        g.add("clear", "Constant", dict(width=width, height=height, red=0, green=0, blue=0, alpha=0.0))
        g.add("shape", "Rectangle", dict(width=width, height=height, box_x=width * 0.25, box_y=height * 0.2,
                                         box_width=width * 0.5, box_height=height * 0.6,
                                         red=1.0, green=0.2, blue=0.1), image="clear")
        g.add("soft", "Blur", dict(radius=4.0), image="shape")
        g.add("flat", "Constant", dict(width=width, height=height, red=0.1, green=0.3, blue=0.9, alpha=0.7))
        g.add("matte", "Constant", dict(width=width, height=height, red=0.8, green=0.3, blue=0.1, alpha=0.6))
        g.add("node", "Blend", params, in0="soft", in1="flat", mask="matte")
        return g

    def test_matches_the_evaluator_across_seams(self):
        cases = (dict(fringe=1), dict(inject=1), dict(mask_channel="red"), dict(mask_channel="luminance", inject=1),
                 dict(fringe=1, inject=1, mask_channel="green", mix=0.8, normalize=0),
                 dict(fringe=1, channels="rgb", weight0=2.0))
        for params in cases:
            for size in ((97, 71), (300, 200)):
                with self.subTest(params=params, size=size):
                    g = self.graph(*size, **params)
                    np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_graph_result_differs_with_each_option(self):
        base = evaluator_pixels(self.graph(97, 71).doc, "node")
        for params in (dict(fringe=1), dict(inject=1), dict(mask_channel="red")):
            with self.subTest(params=params):
                self.assertGreater(float(np.abs(evaluator_pixels(self.graph(97, 71, **params).doc, "node") - base).max()), 1e-3)

    def test_bypass_still_passes_the_first_wired_input(self):
        self.assertEqual(bypass_slot(dict(type="Blend", inputs=dict({f"in{i}": None for i in range(16)}, mask=None, in1="x"))), "in1")


if __name__ == "__main__":
    unittest.main()
