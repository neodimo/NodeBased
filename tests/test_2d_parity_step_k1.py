"""Lane L2 step K1: ChromaKeyer, IBKColor, IBKGizmo. Pixel assertions on a synthetic green-screen frame
with a known foreground disc, evaluator/tile-path parity including seams, mask + mix and bypass. See
docs/PARITY_2D.md for the audit rows these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from nodebased.tiers import input_regions, Region

SCREEN = (0.1, 0.8, 0.2)
DISC = (0.8, 0.1, 0.1)


def disc_frame(size=64, radius=16.0, ramp=1.0, screen=SCREEN, disc=DISC):
    """A flat screen with a disc in the middle; `ramp` is the edge width in pixels (coverage goes
    0 to 1 across it, so `ramp=1` is a one-pixel antialiased edge). Returns pixels and coverage."""
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(xx - size / 2, yy - size / 2)
    coverage = np.clip((radius - dist) / ramp + 0.5, 0.0, 1.0)[..., None].astype(np.float32)
    screen_px = np.array([*screen, 1.0], dtype=np.float32)
    disc_px = np.array([*disc, 1.0], dtype=np.float32)
    return (coverage * disc_px + (1 - coverage) * screen_px).astype(np.float32), coverage[..., 0]


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


def tile_pixels(document, target):
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def screen_graph(g, width=97, height=71):
    """A green screen with a red box and a soft edge: `plate` is the frame under test."""
    g.add("green", "Constant", dict(width=width, height=height, red=SCREEN[0], green=SCREEN[1],
                                    blue=SCREEN[2], alpha=1.0))
    g.add("box", "Rectangle", dict(width=width, height=height, box_x=30.0, box_y=20.0, box_width=40.0,
                                   box_height=28.0, red=DISC[0], green=DISC[1], blue=DISC[2]), image="green")
    g.add("plate", "Blur", dict(radius=2.0), image="box")
    return "plate"


class ChromaKeyerPixelTests(unittest.TestCase):
    def key(self, image, **params):
        return Evaluator._kernel("ChromaKeyer", params, [image])

    def test_alpha_is_zero_on_the_screen_and_one_on_the_disc(self):
        image, coverage = disc_frame()
        out = self.key(image)
        self.assertTrue(np.all(out[coverage == 0][:, 3] < 1e-3))
        self.assertTrue(np.all(out[coverage == 1][:, 3] > 1 - 1e-3))

    def test_edge_ramps_monotonically_from_screen_to_disc(self):
        image, coverage = disc_frame(ramp=8.0)
        alpha = self.key(image)[..., 3]
        row = alpha[32, 32:]          # centre of the disc out through the edge to the screen
        self.assertTrue(np.all(np.diff(row) <= 1e-6), "alpha must not rise while moving out to the screen")
        edge = (coverage[32, 32:] > 0.05) & (coverage[32, 32:] < 0.95)
        self.assertGreater(int(edge.sum()), 3)
        self.assertTrue(np.any((row > 0.05) & (row < 0.95)), "the edge must carry intermediate alpha")

    def test_the_key_colour_moves_the_key(self):
        image, _ = disc_frame(screen=(0.1, 0.2, 0.8))       # a bluescreen
        green_key = self.key(image)[..., 3]
        blue_key = self.key(image, key_red=0.1, key_green=0.2, key_blue=0.8)[..., 3]
        self.assertGreater(float(green_key[0, 0]), 0.9, "a green key does not remove blue screen")
        self.assertLess(float(blue_key[0, 0]), 1e-3)

    def test_tolerance_and_softness_shape_the_ramp(self):
        grey_green = np.array([[[0.3, 0.6, 0.35, 1.0]]], dtype=np.float32)   # part way to neutral
        tight = float(self.key(grey_green, key_tolerance=0.0, key_softness=0.05)[0, 0, 3])
        loose = float(self.key(grey_green, key_tolerance=1.0, key_softness=0.5)[0, 0, 3])
        self.assertGreater(tight, 0.99)
        self.assertLess(loose, 0.01)

    def test_a_shadowed_screen_still_keys_out_and_luma_gain_brings_brightness_back(self):
        dark = np.array([[[0.02, 0.16, 0.04, 1.0]]], dtype=np.float32)     # the screen at a fifth of its light
        self.assertLess(float(self.key(dark, shadow_level=0.0)[0, 0, 3]), 1e-3)
        self.assertGreater(float(self.key(dark, shadow_level=0.0, luma_gain=2.0)[0, 0, 3]), 0.5)

    def test_shadow_and_highlight_levels_push_extremes_toward_opaque(self):
        black = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        self.assertGreater(float(self.key(black)[0, 0, 3]), 0.99)        # black is not the screen anyway
        near_black_screen = np.array([[[0.0003, 0.0024, 0.0006, 1.0]]], dtype=np.float32)
        self.assertLess(float(self.key(near_black_screen, shadow_level=0.0)[0, 0, 3]), 1e-3)
        self.assertGreater(float(self.key(near_black_screen, shadow_level=0.02)[0, 0, 3]), 0.9)
        bright_screen = np.array([[[0.5, 4.0, 1.0, 1.0]]], dtype=np.float32)
        self.assertLess(float(self.key(bright_screen)[0, 0, 3]), 1e-3)
        self.assertGreater(float(self.key(bright_screen, highlight_level=1.0)[0, 0, 3]), 0.99)

    def test_despill_removes_the_green_cast_from_a_spill_tinted_edge(self):
        spilled = np.array([[[0.55, 0.75, 0.25, 1.0]]], dtype=np.float32)   # a red-ish pixel with green on it
        on = self.key(spilled, despill=1)
        off = self.key(spilled, despill=0)
        np.testing.assert_allclose(off[0, 0, :3], spilled[0, 0, :3])
        r, g, b = on[0, 0, :3]
        self.assertLessEqual(float(g), (r + b) / 2 + 1e-6)
        self.assertAlmostEqual(float(r), 0.55, places=6)
        self.assertAlmostEqual(float(b), 0.25, places=6)
        # A pixel with no green excess is left alone.
        clean = np.array([[[0.8, 0.1, 0.1, 1.0]]], dtype=np.float32)
        np.testing.assert_allclose(self.key(clean, despill=1)[0, 0, :3], clean[0, 0, :3])

    def test_despill_bias_weights_red_against_blue(self):
        spilled = np.array([[[0.6, 0.9, 0.2, 1.0]]], dtype=np.float32)
        red_led = self.key(spilled, despill_bias=0.0)[0, 0, 1]
        blue_led = self.key(spilled, despill_bias=1.0)[0, 0, 1]
        self.assertAlmostEqual(float(red_led), 0.6, places=6)
        self.assertAlmostEqual(float(blue_led), 0.2, places=6)

    def test_despill_follows_a_bluescreen_key(self):
        spilled = np.array([[[0.4, 0.3, 0.9, 1.0]]], dtype=np.float32)
        out = self.key(spilled, key_red=0.1, key_green=0.2, key_blue=0.8)
        self.assertAlmostEqual(float(out[0, 0, 2]), (0.4 + 0.3) / 2, places=6)

    def test_premultiply_and_invert(self):
        edge = np.array([[[0.35, 0.55, 0.18, 1.0]]], dtype=np.float32)   # about a third disc, two thirds screen
        straight = self.key(edge, despill=0)
        multiplied = self.key(edge, despill=0, premultiply=1)
        alpha = float(straight[0, 0, 3])
        self.assertTrue(0.0 < alpha < 1.0)
        np.testing.assert_allclose(multiplied[0, 0, :3], straight[0, 0, :3] * alpha, rtol=1e-6)
        self.assertAlmostEqual(float(self.key(edge, invert=1)[0, 0, 3]), 1.0 - alpha, places=6)


class IBKColorTests(unittest.TestCase):
    def plate(self, image, **params):
        return Evaluator._kernel("IBKColor", params, [image])

    def test_a_hole_in_the_screen_fills_within_two_percent_of_the_screen_colour(self):
        image, coverage = disc_frame(radius=8.0)
        hole = coverage > 0
        plate = self.plate(image, fill_size=10, screen_erode=1.0)
        expected = np.array(SCREEN, dtype=np.float32)
        self.assertTrue(hole.any())
        error = np.abs(plate[hole][:, :3] - expected) / expected
        self.assertLess(float(error.max()), 0.02)
        self.assertTrue(np.all(plate[..., 3] == 1.0))
        # Known screen is left as it was.
        far = np.hypot(*np.mgrid[0:64, 0:64].astype(np.float32) - 32) > 20
        np.testing.assert_allclose(plate[far][:, :3], np.broadcast_to(expected, (int(far.sum()), 3)), atol=1e-6)

    def test_a_noisy_screen_fills_with_local_screen_colour_not_the_foreground(self):
        rng = np.random.default_rng(3)
        image, coverage = disc_frame(radius=6.0)
        image[..., 1] += (rng.random(image.shape[:2]) * 0.06).astype(np.float32)
        plate = self.plate(image, fill_size=12)
        hole = coverage > 0
        self.assertTrue(np.all(plate[hole][:, 1] > 0.75))
        self.assertTrue(np.all(plate[hole][:, 0] < 0.15))       # never the disc's red

    def test_fill_size_bounds_the_reach_and_patch_black_covers_the_rest(self):
        image, coverage = disc_frame(radius=14.0)
        short = self.plate(image, fill_size=3, screen_erode=0.0, patch_black=0)
        centre = short[32, 32, :3]
        np.testing.assert_allclose(centre, [0, 0, 0])
        patched = self.plate(image, fill_size=3, screen_erode=0.0, patch_black=1, darks=0.05)
        np.testing.assert_allclose(patched[32, 32, :3], [0.0, 0.05, 0.0], atol=1e-6)
        wide = self.plate(image, fill_size=20, screen_erode=1.0)
        np.testing.assert_allclose(wide[32, 32, :3], SCREEN, atol=1e-5)

    def test_erode_drops_the_contaminated_edge_from_the_known_screen(self):
        image, _ = disc_frame(radius=10.0)
        # One antialiased ring pixel keeps a little red; without erode it stays as "screen".
        image[32, 22] = (0.15, 0.78, 0.2, 1.0)
        kept = self.plate(image, screen_erode=0.0, fill_size=4)[32, 22, :3]
        replaced = self.plate(image, screen_erode=1.0, fill_size=4)[32, 22, :3]
        np.testing.assert_allclose(kept, [0.15, 0.78, 0.2], atol=1e-6)
        np.testing.assert_allclose(replaced, SCREEN, atol=1e-5)

    def test_darks_and_lights_clamp_the_screen_channel(self):
        image, _ = disc_frame()
        low = self.plate(image, darks=0.9, lights=1000.0)[0, 0]
        np.testing.assert_allclose(low[:3], np.array(SCREEN) * (0.9 / 0.8), rtol=1e-5)
        high = self.plate(image, darks=0.0, lights=0.4)[0, 0]
        np.testing.assert_allclose(high[:3], np.array(SCREEN) * (0.4 / 0.8), rtol=1e-5)

    def test_a_bluescreen_fills_with_blue(self):
        image, coverage = disc_frame(radius=6.0, screen=(0.1, 0.2, 0.8))
        plate = self.plate(image, screen_type="blue", fill_size=10)
        np.testing.assert_allclose(plate[coverage > 0][:, :3],
                                   np.broadcast_to(np.array((0.1, 0.2, 0.8)), (int((coverage > 0).sum()), 3)),
                                   atol=1e-5)


class IBKGizmoTests(unittest.TestCase):
    def key(self, fg, plate, bg=None, **params):
        return Evaluator._kernel("IBKGizmo", params, [fg, plate, bg, None])

    def perfect_plate(self, shape):
        plate = np.zeros(shape, dtype=np.float32)
        plate[...] = (*SCREEN, 1.0)
        return plate

    def test_a_perfect_plate_reproduces_the_disc_matte(self):
        image, coverage = disc_frame(ramp=1e-3)             # a hard-edged disc
        out = self.key(image, self.perfect_plate(image.shape))
        off_edge = coverage != 0.5                      # the four axis pixels sit exactly half on the disc
        np.testing.assert_allclose(out[..., 3][off_edge], (coverage > 0.5).astype(np.float32)[off_edge], atol=1e-4)
        # The screen subtracts to nothing and the disc is untouched.
        self.assertTrue(np.all(out[coverage == 0][:, :3] < 1e-5))
        self.assertTrue(np.all(out[coverage == 0][:, 3] == 0.0))
        np.testing.assert_allclose(out[coverage == 1][:, :3], image[coverage == 1][:, :3], atol=1e-6)

    def test_soft_edge_alpha_rises_monotonically_into_the_disc(self):
        image, _ = disc_frame(ramp=8.0)
        alpha = self.key(image, self.perfect_plate(image.shape))[..., 3]
        row = alpha[32, 32:]
        self.assertTrue(np.all(np.diff(row) <= 1e-6))
        self.assertTrue(np.any((row > 0.05) & (row < 0.95)))

    def test_screen_subtraction_off_premultiplies_the_foreground_instead(self):
        image, _ = disc_frame(ramp=8.0)
        plate = self.perfect_plate(image.shape)
        on = self.key(image, plate, screen_subtraction=1)
        off = self.key(image, plate, screen_subtraction=0)
        np.testing.assert_allclose(off[..., :3], image[..., :3] * off[..., 3:4], atol=1e-6)
        np.testing.assert_allclose(on[..., 3], off[..., 3])
        edge = (on[..., 3] > 0.05) & (on[..., 3] < 0.95)
        self.assertTrue(np.any(edge))
        # Subtracting the transparent share of the plate removes green the plain premult keeps.
        self.assertLess(float(on[edge][:, 1].max()), float(off[edge][:, 1].max()))

    def test_red_and_blue_green_weights_move_the_key(self):
        pixel = np.array([[[0.6, 0.7, 0.1, 1.0]]], dtype=np.float32)
        plate = self.perfect_plate(pixel.shape)
        even = float(self.key(pixel, plate)[0, 0, 3])
        red_only = float(self.key(pixel, plate, red_weight=1.0, blue_green_weight=0.0)[0, 0, 3])
        blue_only = float(self.key(pixel, plate, red_weight=0.0, blue_green_weight=1.0)[0, 0, 3])
        self.assertTrue(len({round(even, 4), round(red_only, 4), round(blue_only, 4)}) == 3)
        self.assertGreater(red_only, blue_only)          # red suppresses green more than blue here

    def test_luminance_match_follows_a_dimmer_foreground_screen(self):
        dim_screen = np.array([[[0.05, 0.4, 0.1, 1.0]]], dtype=np.float32)      # half-lit screen
        plate = self.perfect_plate(dim_screen.shape)
        unmatched = float(self.key(dim_screen, plate)[0, 0, 3])
        matched = float(self.key(dim_screen, plate, luminance_match=1)[0, 0, 3])
        self.assertGreater(unmatched, 0.3)
        self.assertLess(matched, 1e-3)

    def test_use_bg_luminance_takes_the_background_brightness(self):
        dim_screen = np.array([[[0.05, 0.4, 0.1, 1.0]]], dtype=np.float32)
        plate = self.perfect_plate(dim_screen.shape)
        bright_bg = np.array([[[2.0, 2.0, 2.0, 1.0]]], dtype=np.float32)
        from_fg = float(self.key(dim_screen, plate, bright_bg, luminance_match=1)[0, 0, 3])
        from_bg = float(self.key(dim_screen, plate, bright_bg, luminance_match=1, use_bg_luminance=1)[0, 0, 3])
        self.assertLess(from_fg, 1e-3)
        self.assertNotAlmostEqual(from_fg, from_bg, places=2)

    def test_iBKColor_into_iBKGizmo_keys_a_frame_end_to_end(self):
        image, coverage = disc_frame(radius=8.0, ramp=1e-3)
        plate = Evaluator._kernel("IBKColor", {"fill_size": 10}, [image])
        out = self.key(image, plate)
        off_edge = coverage != 0.5
        np.testing.assert_allclose(out[..., 3][off_edge], (coverage > 0.5).astype(np.float32)[off_edge], atol=1e-3)


class GraphTests(unittest.TestCase):
    def test_new_kinds_are_registered_with_limits_and_choices(self):
        for kind in ("ChromaKeyer", "IBKColor", "IBKGizmo"):
            self.assertIn(kind, SPECS)
            for name, value in SPECS[kind]["params"].items():
                with self.subTest(kind=kind, param=name):
                    self.assertTrue(name in LIMITS or name in CHOICES or name == "mix" or name == "invert",
                                    f"{name} has neither LIMITS nor CHOICES")
        self.assertEqual(CHOICES["screen_type"], ["green", "blue"])

    def test_evaluator_graph_end_to_end(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("key", "ChromaKeyer", {}, image=plate)
        g.add("clean", "IBKColor", dict(fill_size=12), c=plate)
        g.add("ibk", "IBKGizmo", {}, fg=plate, c="clean")
        for name in ("key", "clean", "ibk"):
            out = evaluator_pixels(g.doc, name)
            self.assertEqual(out.shape, (71, 97, 4))
            self.assertTrue(np.isfinite(out).all())
        ibk = evaluator_pixels(g.doc, "ibk")
        self.assertLess(float(ibk[2, 2, 3]), 1e-3)
        self.assertGreater(float(ibk[34, 50, 3]), 0.99)
        chroma = evaluator_pixels(g.doc, "key")
        self.assertLess(float(chroma[2, 2, 3]), 1e-3)
        self.assertGreater(float(chroma[34, 50, 3]), 0.99)

    def test_mix_zero_is_identity_and_a_zero_mask_hides_the_effect(self):
        for kind, inputs in (("ChromaKeyer", "image"), ("IBKColor", "c")):
            with self.subTest(kind=kind, gate="mix"):
                g = Graph()
                plate = screen_graph(g)
                g.add("node", kind, dict(mix=0.0), **{inputs: plate})
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, plate))
            with self.subTest(kind=kind, gate="mask"):
                g = Graph()
                plate = screen_graph(g)
                g.add("matte", "Constant", dict(width=97, height=71, alpha=0.0))
                g.add("node", kind, {}, mask="matte", **{inputs: plate})
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, plate))
        g = Graph()
        plate = screen_graph(g)
        g.add("clean", "IBKColor", {}, c=plate)
        g.add("node", "IBKGizmo", dict(mix=0.0), fg=plate, c="clean")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, plate))

    def test_a_mismatched_plate_format_raises(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("small", "Constant", dict(width=16, height=16, red=0.1, green=0.8, blue=0.2))
        g.add("node", "IBKGizmo", {}, fg=plate, c="small")
        with self.assertRaises(ValueError):
            evaluator_pixels(g.doc, "node")


class TilePathParityTests(unittest.TestCase):
    def test_kinds_are_tiled(self):
        for kind in ("ChromaKeyer", "IBKColor", "IBKGizmo"):
            self.assertIn(kind, SUPPORTED_TILED_KINDS)

    def test_ibk_color_region_rule_pads_by_erode_plus_fill(self):
        regions = input_regions("IBKColor", dict(fill_size=10, screen_erode=2.0), Region(20, 20, 30, 30), 2)
        self.assertEqual((regions[0].x, regions[0].y, regions[0].width, regions[0].height), (8, 8, 54, 54))
        self.assertEqual((regions[1].x, regions[1].width), (20, 30))         # the mask reads the output region only
        self.assertEqual(input_regions("ChromaKeyer", {}, Region(20, 20, 30, 30), 2)[0].width, 30)
        gizmo = input_regions("IBKGizmo", {}, Region(20, 20, 30, 30), 4)
        self.assertTrue(all(r.width == 30 for r in gizmo))

    def test_chroma_keyer_matches_across_both_paths_with_a_mask(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("matte", "Constant", dict(width=97, height=71, red=1, green=1, blue=1, alpha=0.6))
        g.add("node", "ChromaKeyer", dict(despill=1, premultiply=1, luma_gain=0.5, mix=0.8), image=plate, mask="matte")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_ibk_color_matches_across_both_paths_and_seams(self):
        cases = (dict(fill_size=6, screen_erode=1.0),
                 dict(fill_size=15, screen_erode=2.0, patch_black=0),
                 dict(fill_size=0, screen_erode=0.0),
                 dict(fill_size=9, screen_erode=1.0, darks=0.3, lights=0.7))
        for params in cases:
            for width, height in ((97, 71), (300, 200)):
                with self.subTest(params=params, size=(width, height)):
                    g = Graph()
                    plate = screen_graph(g, width, height)
                    g.add("matte", "Constant", dict(width=width, height=height, red=1, green=1, blue=1, alpha=0.5))
                    g.add("node", "IBKColor", dict(params, mix=0.9), c=plate, mask="matte")
                    np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_ibk_gizmo_matches_across_both_paths_with_bg_and_mask(self):
        g = Graph()
        plate = screen_graph(g, 200, 150)
        g.add("clean", "IBKColor", dict(fill_size=12), c=plate)
        g.add("bg", "Checker", dict(width=200, height=150, size=9))
        g.add("matte", "Constant", dict(width=200, height=150, red=1, green=1, blue=1, alpha=0.7))
        g.add("node", "IBKGizmo", dict(luminance_match=1, use_bg_luminance=1, mix=0.85),
              fg=plate, c="clean", bg="bg", mask="matte")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)
        g.add("plain", "IBKGizmo", dict(screen_type="blue", screen_subtraction=0), fg=plate, c="clean")
        np.testing.assert_allclose(tile_pixels(g.doc, "plain"), evaluator_pixels(g.doc, "plain"), atol=1e-6)


class BypassTests(unittest.TestCase):
    def test_bypass_slots(self):
        self.assertEqual(bypass_slot(dict(type="ChromaKeyer", inputs={"image": "x", "mask": None})), "image")
        self.assertEqual(bypass_slot(dict(type="IBKColor", inputs={"c": "x", "mask": None})), "c")
        self.assertEqual(bypass_slot(dict(type="IBKGizmo", inputs={"fg": "x", "c": "y", "bg": None, "mask": None})), "fg")

    def test_bypassed_nodes_pass_their_input_on_both_paths(self):
        for kind, slot in (("ChromaKeyer", "image"), ("IBKColor", "c")):
            with self.subTest(kind=kind):
                g = Graph()
                plate = screen_graph(g)
                g.add("node", kind, {}, **{slot: plate})
                upstream = evaluator_pixels(g.doc, plate)
                self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))

    def test_bypassed_gizmo_passes_fg_and_never_reads_the_plate(self):
        g = Graph()
        plate = screen_graph(g)
        g.add("clean", "IBKColor", {}, c=plate)
        g.add("node", "IBKGizmo", {}, fg=plate, c="clean")
        upstream = evaluator_pixels(g.doc, plate)
        self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
        g.bypass("node")
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
        self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))


if __name__ == "__main__":
    unittest.main()
