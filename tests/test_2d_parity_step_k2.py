"""Lane L2 step K2: ScreenKeyer, the Keylight-style screen-difference keyer. Pixel assertions on the K1
synthetic green and blue screens, evaluator/tile-path parity including seams, mask + mix and bypass."""
import unittest

import numpy as np

from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS
from nodebased.tiers import Region, input_regions
from tests.test_2d_parity_step_k1 import (DISC, SCREEN, Graph, disc_frame, evaluator_pixels, screen_graph,
                                          tile_pixels)

BLUE_SCREEN = (0.1, 0.2, 0.8)
BLUE_KEY = dict(screen_red=0.1, screen_green=0.2, screen_blue=0.8)


def px(r, g, b, a=1.0):
    return np.array([[[r, g, b, a]]], dtype=np.float32)


class ScreenKeyerPixelTests(unittest.TestCase):
    def key(self, image, **params):
        return Evaluator._kernel("ScreenKeyer", params, [image])

    def matte(self, image, **params):
        return self.key(image, keyer_view="screen_matte", **params)[..., 3]

    def test_matte_is_zero_on_the_green_screen_and_one_on_the_disc(self):
        image, coverage = disc_frame()
        matte = self.matte(image)
        self.assertTrue(np.all(matte[coverage == 0] < 1e-4))
        self.assertTrue(np.all(matte[coverage == 1] > 1 - 1e-4))
        final = self.key(image)
        np.testing.assert_allclose(final[..., 3], matte)

    def test_matte_is_zero_on_the_blue_screen_and_one_on_the_disc(self):
        image, coverage = disc_frame(screen=BLUE_SCREEN)
        matte = self.matte(image, **BLUE_KEY)
        self.assertTrue(np.all(matte[coverage == 0] < 1e-4))
        self.assertTrue(np.all(matte[coverage == 1] > 1 - 1e-4))
        # A green key leaves a blue screen opaque.
        self.assertGreater(float(self.matte(image)[0, 0]), 0.99)

    def test_edge_ramps_monotonically_from_screen_to_disc(self):
        image, coverage = disc_frame(ramp=8.0)
        row = self.matte(image)[32, 32:]
        self.assertTrue(np.all(np.diff(row) <= 1e-6))
        self.assertTrue(np.any((row > 0.05) & (row < 0.95)))

    def test_screen_gain_removes_more_screen_monotonically(self):
        # A pixel a quarter of the way from grey to the screen: saturation 0.23, so the matte is
        # 1 - 0.23 * gain. Higher gain drives it down (a denser hole where the screen was).
        partial = px(0.5, 0.6, 0.4)
        mattes = [float(self.matte(partial, screen_gain=g)[0, 0]) for g in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)]
        self.assertAlmostEqual(mattes[0], 1.0, places=6)
        self.assertAlmostEqual(mattes[2], 1.0 - 0.15 / 0.65, places=5)
        self.assertTrue(all(b < a for a, b in zip(mattes[:5], mattes[1:6])), mattes)
        self.assertLess(mattes[-1], 1e-6)
        # A pure screen stays fully removed at any gain above 1 and the disc stays opaque at any gain.
        for gain in (1.0, 3.0):
            self.assertAlmostEqual(float(self.matte(px(*SCREEN), screen_gain=gain)[0, 0]), 0.0, places=5)
            self.assertAlmostEqual(float(self.matte(px(*DISC), screen_gain=gain)[0, 0]), 1.0, places=6)

    def test_screen_balance_moves_the_matte_between_the_other_two_channels(self):
        pixel = px(0.7, 0.6, 0.1)
        by_red = float(self.matte(pixel, screen_balance=0.0)[0, 0])       # green minus red
        by_blue = float(self.matte(pixel, screen_balance=1.0)[0, 0])      # green minus blue
        self.assertGreater(by_red, 0.9)
        self.assertLess(by_blue, 0.3)
        self.assertNotAlmostEqual(by_red, by_blue, places=2)

    def test_alpha_bias_pulls_ambiguous_pixels_toward_the_foreground(self):
        pixel = px(0.4, 0.7, 0.3)
        low = float(self.matte(pixel, alpha_bias=0.0)[0, 0])
        high = float(self.matte(pixel, alpha_bias=0.3)[0, 0])
        self.assertGreater(high, low)
        self.assertAlmostEqual(float(self.matte(px(*SCREEN), alpha_bias=1.0)[0, 0]), 1.0, places=5)

    def test_clip_black_and_white_clamp_the_matte_as_stated(self):
        partial = px(0.5, 0.6, 0.4)
        base = float(self.matte(partial)[0, 0])                       # 0.769...
        self.assertAlmostEqual(float(self.matte(partial, clip_black=base + 0.05)[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(self.matte(partial, clip_white=base - 0.05)[0, 0]), 1.0, places=6)
        # Between the two the matte is stretched to fill 0..1.
        stretched = float(self.matte(partial, clip_black=0.2, clip_white=0.9)[0, 0])
        self.assertAlmostEqual(stretched, (base - 0.2) / 0.7, places=5)
        # Over a soft edge everything at or below black is 0 and at or above white is 1.
        image, _ = disc_frame(ramp=8.0)
        raw = self.matte(image)
        clipped = self.matte(image, clip_black=0.3, clip_white=0.7)
        self.assertTrue(np.all(clipped[raw <= 0.3] == 0.0))
        self.assertTrue(np.all(clipped[raw >= 0.7] == 1.0))
        self.assertTrue(np.all(np.diff(clipped[32, 32:]) <= 1e-6))
        # Defaults change nothing.
        np.testing.assert_allclose(self.matte(image, clip_black=0.0, clip_white=1.0), raw)

    def test_clip_rollback_restores_part_of_what_the_clip_flattened(self):
        partial = px(0.5, 0.6, 0.4)
        raw = float(self.matte(partial)[0, 0])
        flat = float(self.matte(partial, clip_black=raw + 0.05)[0, 0])
        half = float(self.matte(partial, clip_black=raw + 0.05, clip_rollback=0.5)[0, 0])
        full = float(self.matte(partial, clip_black=raw + 0.05, clip_rollback=1.0)[0, 0])
        self.assertEqual(flat, 0.0)
        self.assertTrue(flat < half < full <= raw + 1e-6)
        # Pixels the clip left fully opaque or fully transparent are not touched.
        self.assertEqual(float(self.matte(px(*SCREEN), clip_black=0.1, clip_rollback=1.0)[0, 0]), 0.0)
        self.assertEqual(float(self.matte(px(*DISC), clip_white=0.9, clip_rollback=1.0)[0, 0]), 1.0)

    def test_status_view_marks_pixels_that_are_neither_zero_nor_one(self):
        image, _ = disc_frame(ramp=8.0)
        matte = self.matte(image)
        status = self.key(image, keyer_view="status")
        partial = (matte > 0.0) & (matte < 1.0)
        self.assertTrue(partial.any())
        np.testing.assert_allclose(status[partial][:, :3], 0.5)
        np.testing.assert_allclose(status[matte == 0][:, :3], 0.0)
        np.testing.assert_allclose(status[matte == 1][:, :3], 1.0)
        np.testing.assert_allclose(status[..., 3], 1.0)
        self.assertTrue(np.any(status[..., 0] == 0.0) and np.any(status[..., 0] == 1.0))

    def test_views_final_matte_and_intermediate(self):
        spilled = px(0.55, 0.75, 0.25, 0.9)
        matte = float(self.matte(spilled)[0, 0])
        self.assertTrue(0.0 < matte < 1.0)
        final = self.key(spilled)[0, 0]
        inter = self.key(spilled, keyer_view="intermediate")[0, 0]
        grey = self.key(spilled, keyer_view="screen_matte")[0, 0]
        np.testing.assert_allclose(grey, [matte] * 4, rtol=1e-6)
        self.assertAlmostEqual(float(inter[3]), 0.9, places=6)          # the input alpha, no matte applied
        np.testing.assert_allclose(final[:3], inter[:3] * matte, rtol=1e-6)
        self.assertAlmostEqual(float(final[3]), matte, places=6)

    def test_despill_leaves_neutral_grey_unchanged_and_removes_green_from_spill(self):
        for level in (0.0, 0.18, 0.5, 4.0):
            grey = px(level, level, level)
            np.testing.assert_allclose(self.key(grey, keyer_view="intermediate")[0, 0, :3], [level] * 3, atol=1e-7)
            np.testing.assert_allclose(self.key(grey)[0, 0, :3], [level] * 3, atol=1e-7)
        spilled = px(0.55, 0.75, 0.25)
        r, g, b = self.key(spilled, keyer_view="intermediate")[0, 0, :3]
        self.assertAlmostEqual(float(g), 0.4, places=6)                   # limited to (r + b) / 2
        self.assertAlmostEqual(float(r), 0.55, places=6)
        self.assertAlmostEqual(float(b), 0.25, places=6)
        # A foreground with no green excess is left alone.
        clean = px(*DISC)
        np.testing.assert_allclose(self.key(clean, keyer_view="intermediate")[0, 0, :3], DISC, atol=1e-7)

    def test_despill_bias_weights_the_two_other_channels(self):
        spilled = px(0.6, 0.9, 0.2)
        by_red = self.key(spilled, keyer_view="intermediate", despill_bias=0.0)[0, 0, 1]
        by_blue = self.key(spilled, keyer_view="intermediate", despill_bias=1.0)[0, 0, 1]
        self.assertAlmostEqual(float(by_red), 0.6, places=6)
        self.assertAlmostEqual(float(by_blue), 0.2, places=6)

    def test_bluescreen_despill_limits_blue(self):
        spilled = px(0.4, 0.3, 0.9)
        out = self.key(spilled, keyer_view="intermediate", **BLUE_KEY)[0, 0, :3]
        self.assertAlmostEqual(float(out[2]), 0.35, places=6)
        grey = px(0.3, 0.3, 0.3)
        np.testing.assert_allclose(self.key(grey, keyer_view="intermediate", **BLUE_KEY)[0, 0, :3], [0.3] * 3, atol=1e-7)

    def test_screen_grow_and_shrink_move_the_edge(self):
        image, _ = disc_frame(radius=16.0, ramp=1e-3)
        base = self.matte(image)
        grown = self.matte(image, screen_shrink=3.0)       # positive grows the screen: the disc gets smaller
        shrunk = self.matte(image, screen_shrink=-3.0)
        self.assertEqual(float(base[32, 32 + 15]), 1.0)
        self.assertEqual(float(grown[32, 32 + 15]), 0.0)
        self.assertEqual(float(grown[32, 32 + 11]), 1.0)
        self.assertEqual(float(shrunk[32, 32 + 18]), 1.0)
        self.assertTrue(np.all(grown <= base + 1e-6))
        self.assertTrue(np.all(shrunk >= base - 1e-6))
        np.testing.assert_allclose(self.matte(image, screen_shrink=0.3), base)

    def test_screen_softness_blurs_the_matte_without_moving_flat_areas(self):
        image, coverage = disc_frame(radius=16.0, ramp=1e-3)
        base = self.matte(image)
        soft = self.matte(image, screen_softness=4.0)
        self.assertLessEqual(int(((base > 0) & (base < 1)).sum()), 4)   # the four axis pixels sit half on the disc
        self.assertGreater(int(((soft > 0.01) & (soft < 0.99)).sum()), 20)
        self.assertAlmostEqual(float(soft[32, 32]), 1.0, places=5)
        self.assertAlmostEqual(float(soft[2, 2]), 0.0, places=5)
        self.assertTrue(np.all(np.diff(soft[32, 32:]) <= 1e-6))
        np.testing.assert_allclose(self.matte(image, screen_softness=0.3), base)

    def test_the_output_is_finite_and_keeps_the_frame_shape(self):
        image = np.random.default_rng(1).random((13, 17, 4)).astype(np.float32) * 3
        for view in CHOICES["keyer_view"]:
            with self.subTest(view=view):
                out = self.key(image, keyer_view=view, screen_shrink=2.0, screen_softness=2.0)
                self.assertEqual(out.shape, image.shape)
                self.assertTrue(np.isfinite(out).all())


class ScreenKeyerGraphTests(unittest.TestCase):
    def test_registered_with_limits_and_choices(self):
        self.assertIn("ScreenKeyer", SPECS)
        for name in SPECS["ScreenKeyer"]["params"]:
            with self.subTest(param=name):
                self.assertTrue(name in LIMITS or name in CHOICES or name == "mix", name)
        self.assertEqual(CHOICES["keyer_view"], ["final", "status", "screen_matte", "intermediate"])

    def test_evaluator_graph_end_to_end(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("key", "ScreenKeyer", {}, image=plate)
        out = evaluator_pixels(g.doc, "key")
        self.assertEqual(out.shape, (71, 97, 4))
        self.assertLess(float(out[2, 2, 3]), 1e-3)
        self.assertGreater(float(out[34, 50, 3]), 0.99)

    def test_mix_zero_is_identity_and_a_zero_mask_hides_the_effect(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("node", "ScreenKeyer", dict(mix=0.0), image=plate)
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, plate))
        g = Graph()
        plate = screen_graph(g)
        g.add("matte", "Constant", dict(width=97, height=71, alpha=0.0))
        g.add("node", "ScreenKeyer", {}, image=plate, mask="matte")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, plate))


class ScreenKeyerTilePathTests(unittest.TestCase):
    def test_kind_is_tiled_and_region_rule_pads_by_shrink_plus_softness(self):
        self.assertIn("ScreenKeyer", SUPPORTED_TILED_KINDS)
        regions = input_regions("ScreenKeyer", dict(screen_shrink=-2.0, screen_softness=3.0), Region(20, 20, 30, 30), 2)
        self.assertEqual((regions[0].x, regions[0].y, regions[0].width, regions[0].height), (15, 15, 40, 40))
        self.assertEqual((regions[1].x, regions[1].width), (20, 30))
        plain = input_regions("ScreenKeyer", {}, Region(20, 20, 30, 30), 2)
        self.assertEqual((plain[0].x, plain[0].width), (20, 30))

    def test_matches_the_evaluator_for_every_view_across_seams_with_a_mask(self):
        cases = (dict(),
                 dict(screen_shrink=3.0, screen_softness=4.0, clip_black=0.1, clip_white=0.9, clip_rollback=0.5),
                 dict(screen_shrink=-2.0, screen_gain=2.0, alpha_bias=0.1, screen_balance=0.3, despill_bias=0.7))
        for params in cases:
            for view in CHOICES["keyer_view"]:
                for width, height in ((97, 71), (300, 200)):
                    with self.subTest(params=params, view=view, size=(width, height)):
                        g = Graph()
                        plate = screen_graph(g, width, height)
                        g.add("matte", "Constant", dict(width=width, height=height, red=1, green=1, blue=1, alpha=0.6))
                        g.add("node", "ScreenKeyer", dict(params, keyer_view=view, mix=0.85), image=plate, mask="matte")
                        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_matches_on_a_bluescreen_without_a_mask(self):
        g = Graph()
        g.add("blue", "Constant", dict(width=200, height=150, red=BLUE_SCREEN[0], green=BLUE_SCREEN[1],
                                       blue=BLUE_SCREEN[2], alpha=1.0))
        g.add("box", "Rectangle", dict(width=200, height=150, box_x=60.0, box_y=40.0, box_width=70.0,
                                       box_height=50.0, red=DISC[0], green=DISC[1], blue=DISC[2]), image="blue")
        g.add("node", "ScreenKeyer", dict(BLUE_KEY, screen_softness=5.0, screen_shrink=2.0), image="box")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)


class ScreenKeyerBypassTests(unittest.TestCase):
    def test_bypass_slot_and_both_paths(self):
        self.assertEqual(bypass_slot(dict(type="ScreenKeyer", inputs={"image": "x", "mask": None})), "image")
        g = Graph()
        plate = screen_graph(g)
        g.add("node", "ScreenKeyer", {}, image=plate)
        upstream = evaluator_pixels(g.doc, plate)
        self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
        g.bypass("node")
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
        self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))


if __name__ == "__main__":
    unittest.main()
