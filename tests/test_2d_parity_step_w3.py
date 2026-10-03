"""Lane L2 step W3: Blend's fringe, inject and mask channel, and ScreenKeyer's inside, outside and clean
plate inputs and its separate despill and alpha-bias colours. Pixel assertions on both paths."""
import unittest

import numpy as np

from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tiers import Region, input_regions
from tests.test_2d_parity_step_k1 import DISC, SCREEN, Graph, disc_frame, evaluator_pixels, screen_graph, tile_pixels


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


def key(image, inside=None, outside=None, clean=None, mask=None, **params):
    return Evaluator._kernel("ScreenKeyer", dict(SPECS["ScreenKeyer"]["params"], **params),
                             [image, inside, outside, clean, mask])


def matte(image, **kw):
    return key(image, keyer_view="screen_matte", **kw)[..., 3]


class ScreenKeyerGarbageMatteTests(unittest.TestCase):
    def setUp(self):
        self.image, self.coverage = disc_frame()
        self.base = matte(self.image)
        half = np.zeros(self.image.shape[:2] + (1,), dtype=np.float32)
        half[:, :32] = 1.0                         # the left half of the frame
        self.half = np.concatenate([half, half, half, half], axis=2)

    def test_inside_forces_foreground_where_it_is_set_and_leaves_the_rest(self):
        out = matte(self.image, inside=self.half)
        self.assertTrue(np.allclose(out[:, :32], 1.0))              # includes screen pixels, matte was 0
        self.assertLess(float(self.base[2, 2]), 1e-3)
        np.testing.assert_allclose(out[:, 32:], self.base[:, 32:], atol=1e-6)

    def test_outside_forces_background_where_it_is_set_and_leaves_the_rest(self):
        out = matte(self.image, outside=self.half)
        self.assertTrue(np.allclose(out[:, :32], 0.0))              # includes disc pixels, matte was 1
        self.assertGreater(float(self.base[32, 20]), 0.99)
        np.testing.assert_allclose(out[:, 32:], self.base[:, 32:], atol=1e-6)

    def test_a_soft_garbage_matte_scales_the_forcing_and_outside_wins_an_overlap(self):
        soft = np.full(self.image.shape, 0.4, dtype=np.float32)
        np.testing.assert_allclose(matte(self.image, outside=soft), self.base * 0.6, atol=1e-6)
        np.testing.assert_allclose(matte(self.image, inside=soft)[2, 2], 0.4, atol=1e-6)
        np.testing.assert_allclose(matte(self.image, inside=self.half, outside=self.half)[:, :32], 0.0, atol=1e-6)

    def test_forcing_shows_in_the_final_colour_and_the_status_view(self):
        final = key(self.image, inside=self.half)
        self.assertTrue(np.allclose(final[:, :32, 3], 1.0))
        status = key(self.image, outside=self.half, keyer_view="status")
        self.assertTrue(np.allclose(status[:, :32, 0], 0.0))

    def test_unwired_garbage_mattes_change_nothing(self):
        np.testing.assert_array_equal(key(self.image), Evaluator._kernel("ScreenKeyer", {}, [self.image]))


class ScreenKeyerCleanPlateTests(unittest.TestCase):
    def shaded(self):
        """A screen darkening to the right with a bright disc on it, and the empty screen itself."""
        ramp = np.linspace(1.0, 0.35, 64, dtype=np.float32)[None, :, None] * np.ones((64, 1, 1), np.float32)
        plate_rgb = np.concatenate([SCREEN[0] * ramp, SCREEN[1] * ramp, SCREEN[2] * ramp, np.ones_like(ramp)], axis=2)
        yy, xx = np.mgrid[0:64, 0:64]
        inside = (np.hypot(xx - 20, yy - 32) < 10)[..., None]
        disc = np.array([*DISC, 1.0], dtype=np.float32)
        return np.where(inside, disc, plate_rgb).astype(np.float32), plate_rgb.astype(np.float32), inside[..., 0]

    def test_a_clean_plate_keys_the_shaded_screen_the_flat_colour_cannot(self):
        image, plate, disc = self.shaded()
        without, with_plate = matte(image), matte(image, clean=plate)
        screen = ~disc
        self.assertGreater(float(without[screen].max()), 0.3)       # the dark right side leaks foreground
        self.assertLess(float(with_plate[screen].max()), 1e-3)      # the plate measures against its own shading
        self.assertGreater(float(with_plate[disc].min()), 0.99)

    def test_the_plate_is_the_reference_so_a_different_plate_gives_a_different_matte(self):
        image, plate, _ = self.shaded()
        self.assertGreater(float(np.abs(matte(image, clean=plate) - matte(image, clean=plate * 2.0)).max()), 0.4)   # a brighter reference: the screen reads half-keyed

    def test_pixels_where_the_plate_has_no_screen_fall_back_to_the_screen_colour(self):
        image, _, _ = self.shaded()
        grey = np.full(image.shape, 0.5, dtype=np.float32)
        grey[..., 3] = 1.0
        np.testing.assert_allclose(matte(image, clean=grey), matte(image), atol=1e-6)

    def test_a_clean_plate_equal_to_the_screen_colour_matches_no_plate(self):
        image, _ = disc_frame()
        flat = np.empty_like(image)
        flat[...] = (*SCREEN, 1.0)
        np.testing.assert_allclose(matte(image, clean=flat), matte(image), atol=1e-5)

    def test_a_clean_plate_of_the_wrong_size_is_refused(self):
        image, _ = disc_frame()
        with self.assertRaises(ValueError):
            key(image, clean=np.zeros((8, 8, 4), np.float32))


class ScreenKeyerBiasColourTests(unittest.TestCase):
    def setUp(self):
        # A green-spilled skin tone: its alpha and despill both depend on how red and blue are weighed.
        self.image = np.empty((2, 2, 4), dtype=np.float32)
        self.image[...] = (0.55, 0.62, 0.30, 1.0)

    def test_off_ignores_the_colours(self):
        base = key(self.image)
        np.testing.assert_array_equal(key(self.image, alpha_bias_red=1.0, despill_bias_blue=1.0), base)

    def test_the_alpha_bias_colour_moves_the_matte_and_not_the_despill(self):
        plain = key(self.image, keyer_view="intermediate")
        red_heavy = dict(bias_colours=1, alpha_bias_red=1.0, alpha_bias_green=0.5, alpha_bias_blue=0.0)
        blue_heavy = dict(bias_colours=1, alpha_bias_red=0.0, alpha_bias_green=0.5, alpha_bias_blue=1.0)
        a, b = matte(self.image, **red_heavy), matte(self.image, **blue_heavy)
        self.assertGreater(abs(float(a[0, 0] - b[0, 0])), 0.05)
        np.testing.assert_array_equal(key(self.image, keyer_view="intermediate", **red_heavy), plain)

    def test_the_despill_bias_colour_moves_the_colour_and_not_the_matte(self):
        base = key(self.image, keyer_view="intermediate")
        red_heavy = dict(bias_colours=1, despill_bias_red=1.0, despill_bias_blue=0.0)     # limit follows red
        blue_heavy = dict(bias_colours=1, despill_bias_red=0.0, despill_bias_blue=1.0)    # limit follows blue
        a = key(self.image, keyer_view="intermediate", **red_heavy)
        b = key(self.image, keyer_view="intermediate", **blue_heavy)
        self.assertAlmostEqual(float(a[0, 0, 1]), 0.55, places=5)
        self.assertAlmostEqual(float(b[0, 0, 1]), 0.30, places=5)
        np.testing.assert_array_equal(matte(self.image, **red_heavy), matte(self.image))
        np.testing.assert_array_equal(key(self.image, keyer_view="intermediate", bias_colours=1)[..., 1],
                                      base[..., 1])    # grey bias colours reproduce despill_bias 0.5

    def test_a_black_bias_colour_falls_back_to_the_scalar(self):
        black = dict(bias_colours=1, despill_bias_red=0.0, despill_bias_blue=0.0, alpha_bias_red=0.0,
                     alpha_bias_blue=0.0)
        np.testing.assert_allclose(key(self.image, despill_bias=0.25, screen_balance=0.25, **black),
                                   key(self.image, despill_bias=0.25, screen_balance=0.25), atol=1e-6)


class ScreenKeyerW3GraphTests(unittest.TestCase):
    def test_registration_and_slot_order(self):
        self.assertEqual(SPECS["ScreenKeyer"]["optional_inputs"], ["inside", "outside", "clean", "mask"])
        for name in SPECS["ScreenKeyer"]["params"]:
            with self.subTest(param=name):
                self.assertTrue(name in LIMITS or name in CHOICES or name == "mix", name)
        self.assertEqual(bypass_slot(dict(type="ScreenKeyer", inputs=dict(image="x", inside="y", outside=None,
                                                                          clean=None, mask=None))), "image")

    def test_region_rule_pads_the_image_garbage_mattes_and_plate_but_not_the_mask(self):
        regions = input_regions("ScreenKeyer", dict(screen_shrink=-2.0, screen_softness=3.0), Region(20, 20, 30, 30), 5)
        self.assertEqual([(r.x, r.width) for r in regions], [(15, 40)] * 4 + [(20, 30)])

    def graph(self, width, height, **params):
        g = Graph()
        plate = screen_graph(g, width, height)
        g.add("clearbase", "Constant", dict(width=width, height=height, red=0, green=0, blue=0, alpha=0.0))
        g.add("inside", "Rectangle", dict(width=width, height=height, box_x=2.0, box_y=2.0, box_width=width * 0.3,
                                          box_height=height * 0.3, red=1, green=1, blue=1), image="clearbase")
        g.add("outside", "Rectangle", dict(width=width, height=height, box_x=width * 0.6, box_y=height * 0.5,
                                           box_width=width * 0.3, box_height=height * 0.4, red=1, green=1, blue=1),
              image="clearbase")
        g.add("cleanplate", "Constant", dict(width=width, height=height, red=SCREEN[0] * 0.8, green=SCREEN[1] * 0.8,
                                             blue=SCREEN[2] * 0.8, alpha=1.0))
        g.add("matte", "Constant", dict(width=width, height=height, red=1, green=1, blue=1, alpha=0.6))
        g.add("node", "ScreenKeyer", params, image=plate, inside="inside", outside="outside", clean="cleanplate",
              mask="matte")
        return g

    def test_each_new_input_changes_the_graph_result(self):
        full = evaluator_pixels(self.graph(97, 71).doc, "node")
        for slot in ("inside", "outside", "clean"):
            with self.subTest(slot=slot):
                g = self.graph(97, 71)
                g.d.execute(dict(op="connect", id="node", input=slot, source=None))
                self.assertGreater(float(np.abs(evaluator_pixels(g.doc, "node") - full).max()), 1e-3)

    def test_tile_path_matches_the_evaluator_for_every_new_option_across_seams(self):
        cases = (dict(), dict(keyer_view="screen_matte"), dict(keyer_view="status"),
                 dict(screen_shrink=3.0, screen_softness=4.0, clip_black=0.1, clip_white=0.9, mix=0.8),
                 dict(screen_shrink=-2.0, bias_colours=1, alpha_bias_red=0.9, alpha_bias_blue=0.2,
                      despill_bias_red=0.2, despill_bias_blue=0.8))
        for params in cases:
            for size in ((97, 71), (300, 200)):
                with self.subTest(params=params, size=size):
                    g = self.graph(*size, **params)
                    np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_the_tile_path_matches_with_only_some_inputs_wired(self):
        for wired in (("inside",), ("outside",), ("clean",), ("inside", "clean")):
            with self.subTest(wired=wired):
                g = self.graph(200, 150, screen_softness=3.0)
                for slot in ("inside", "outside", "clean", "mask"):
                    if slot not in wired:
                        g.d.execute(dict(op="connect", id="node", input=slot, source=None))
                np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_mismatched_garbage_matte_format_is_refused_on_the_evaluator(self):
        g = self.graph(97, 71)
        g.add("small", "Constant", dict(width=40, height=30, alpha=1.0))
        g.d.execute(dict(op="connect", id="node", input="inside", source="small"))
        with self.assertRaises(ValueError):
            evaluator_pixels(g.doc, "node")


if __name__ == "__main__":
    unittest.main()
