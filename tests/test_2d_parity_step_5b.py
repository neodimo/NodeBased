"""Step 5b: Grain, Posterize, SoftClip, HSVTool, AddMix, Blend, CopyRectangle. Pixel assertions on
constructed images, evaluator/tile-path parity across several tiles (with and without a mask), mix,
bypass and registration. See docs/PARITY_2D.md for the audit rows these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, tile_pixels, kernel

KINDS = ("Grain", "Posterize", "SoftClip", "HSVTool", "AddMix", "Blend", "CopyRectangle")


def grain_params(**over):
    p = dict(SPECS["Grain"]["params"])
    p.update(over)
    return p


class GrainTests(unittest.TestCase):
    def test_mean_is_preserved_and_the_amplitude_follows_intensity(self):
        flat = np.full((256, 256, 4), 0.4, np.float32)
        out = Evaluator._grain(flat, grain_params(red_size=1.0, green_size=1.0, blue_size=1.0,
                                                  red_intensity=0.1, green_intensity=0.1, blue_intensity=0.05), frame=1)
        for c, sigma in ((0, 0.1), (1, 0.1), (2, 0.05)):
            self.assertAlmostEqual(float(out[..., c].mean()), 0.4, delta=0.002)
            self.assertAlmostEqual(float(out[..., c].std()), sigma, delta=sigma * 0.05)
        np.testing.assert_array_equal(out[..., 3], flat[..., 3])
        # a larger size makes the grain coarser: neighbouring pixels agree more
        coarse = Evaluator._grain(flat, grain_params(red_size=8.0, red_intensity=0.1), frame=1)[..., 0]
        fine = Evaluator._grain(flat, grain_params(red_size=1.0, red_intensity=0.1), frame=1)[..., 0]
        self.assertGreater(float(np.corrcoef(coarse[:, :-1].ravel(), coarse[:, 1:].ravel())[0, 1]), 0.6)
        self.assertLess(abs(float(np.corrcoef(fine[:, :-1].ravel(), fine[:, 1:].ravel())[0, 1])), 0.05)

    def test_zero_intensity_is_the_identity(self):
        image = np.random.RandomState(0).rand(20, 20, 4).astype(np.float32)
        p = grain_params(red_intensity=0.0, green_intensity=0.0, blue_intensity=0.0)
        np.testing.assert_array_equal(Evaluator._grain(image, p, frame=3), image)

    def test_grain_moves_with_the_frame_and_repeats_for_one_seed_and_frame(self):
        flat = np.full((64, 64, 4), 0.5, np.float32)
        p = grain_params()
        f1, f1b, f2 = (Evaluator._grain(flat, p, frame=f) for f in (1, 1, 2))
        np.testing.assert_array_equal(f1, f1b)
        self.assertFalse(np.array_equal(f1, f2))
        self.assertFalse(np.array_equal(f1, Evaluator._grain(flat, grain_params(seed=7), frame=1)))
        self.assertAlmostEqual(float(f1[..., 0].mean()), float(f2[..., 0].mean()), delta=0.02)   # same statistics, new pattern
        # Nuke's "-frame" recipe: seed = -frame makes the folded seed constant
        a = Evaluator._grain(flat, grain_params(seed=-1), frame=1)
        b = Evaluator._grain(flat, grain_params(seed=-2), frame=2)
        np.testing.assert_array_equal(a, b)

    def test_pattern_is_tied_to_the_pixel_position(self):
        image = np.random.RandomState(4).rand(48, 48, 4).astype(np.float32)
        p = grain_params()
        whole = Evaluator._grain(image, p, frame=5)
        part = Evaluator._grain(image[10:30, 7:40], p, origin=(7, 10), frame=5)
        np.testing.assert_array_equal(whole[10:30, 7:40], part)

    def test_luminance_weighting_leaves_black_alone_and_black_sets_the_floor(self):
        image = np.zeros((32, 32, 4), np.float32)
        image[:, 16:, :3] = 1.0
        base = grain_params(red_size=1.0, green_size=1.0, blue_size=1.0, luminance_weighted=1, black=0.0)
        out = Evaluator._grain(image, base, frame=1)
        self.assertEqual(float(np.abs(out[:, :16] - image[:, :16]).max()), 0.0)
        self.assertGreater(float(np.abs(out[:, 16:] - image[:, 16:])[..., :3].std()), 0.01)
        floor = Evaluator._grain(image, dict(base, black=0.5), frame=1)
        self.assertGreater(float(np.abs(floor[:, :16] - image[:, :16]).max()), 0.001)


class PosterizeTests(unittest.TestCase):
    def _ramp(self):
        image = np.zeros((4, 256, 4), np.float32)
        image[..., :3] = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :, None]
        image[..., 3] = 0.77
        return image

    def test_two_colours_gives_exactly_two_levels(self):
        out = kernel("Posterize", self._ramp(), colors=2)
        self.assertEqual(sorted(np.unique(out[..., :3]).tolist()), [0.0, 1.0])
        self.assertEqual(float(out[0, 127, 0]), 0.0)
        self.assertEqual(float(out[0, 128, 0]), 1.0)
        np.testing.assert_array_equal(out[..., 3], self._ramp()[..., 3])   # alpha is outside "rgb"

    def test_four_colours_and_hdr_and_channels(self):
        out = kernel("Posterize", self._ramp(), colors=4)
        np.testing.assert_allclose(sorted(np.unique(out[..., :3]).tolist()), [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-6)
        hdr = np.full((2, 2, 4), (5.0, -1.0, 0.4, 1.0), np.float32)
        np.testing.assert_allclose(kernel("Posterize", hdr, colors=2)[0, 0], (1.0, 0.0, 0.0, 1.0))
        rgba = kernel("Posterize", self._ramp(), colors=2, channels="rgba")
        self.assertEqual(sorted(np.unique(rgba[..., 3]).tolist()), [1.0])


class SoftClipTests(unittest.TestCase):
    def _values(self, *values):
        image = np.zeros((1, len(values), 4), np.float32)
        image[..., :3] = np.array(values, np.float32)[None, :, None]
        image[..., 3] = 1.0
        return image

    def test_logarithmic_compress_leaves_low_values_and_compresses_high_ones(self):
        image = self._values(-1.0, 0.0, 0.5, 0.8, 1.0, 2.0, 4.0, 10.0)
        out = kernel("SoftClip", image, conversion="logarithmic compress", softclip_min=0.8, softclip_max=4.0)[0, :, 0]
        np.testing.assert_array_equal(out[:4], image[0, :4, 0])            # at or below min: untouched
        self.assertAlmostEqual(float(out[6]), 1.0, places=5)                      # max lands on 1.0
        self.assertTrue(np.all(np.diff(out) > 0), out.tolist())                   # monotone
        self.assertTrue(np.all(out[4:] < image[0, 4:, 0]))                        # compressed
        self.assertAlmostEqual(float(out[4]), 1.0, delta=0.2)
        # slope 1 at the join: the curve leaves softclip_min with unit slope
        near = kernel("SoftClip", self._values(0.8, 0.8001), conversion="logarithmic compress",
                      softclip_min=0.8, softclip_max=4.0)[0, :, 0]
        self.assertAlmostEqual(float((near[1] - near[0]) / 0.0001), 1.0, delta=0.01)

    def test_none_and_a_max_of_one_leave_the_image_alone(self):
        image = self._values(0.3, 1.5, 9.0)
        np.testing.assert_array_equal(kernel("SoftClip", image), image)
        np.testing.assert_array_equal(kernel("SoftClip", image, conversion="logarithmic compress"), image)

    def test_preserve_hue_and_saturation_scales_the_pixel_down(self):
        image = np.array([[[4.0, 2.0, 1.0, 1.0], [0.5, 0.25, 0.1, 1.0]]], np.float32)
        out = kernel("SoftClip", image, conversion="preserve hue and saturation", softclip_max=1.0)
        np.testing.assert_allclose(out[0, 0], (1.0, 0.5, 0.25, 1.0), atol=1e-6)
        np.testing.assert_array_equal(out[0, 1], image[0, 1])

    def test_preserve_hue_and_brightness_desaturates_at_constant_luma(self):
        image = np.array([[[3.0, 0.2, 0.1, 1.0], [0.5, 0.25, 0.1, 1.0]]], np.float32)
        out = kernel("SoftClip", image, conversion="preserve hue and brightness", softclip_max=1.0)
        luma = lambda px: 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]
        self.assertAlmostEqual(float(out[0, 0, :3].max()), 1.0, places=5)
        self.assertAlmostEqual(float(luma(out[0, 0])), float(luma(image[0, 0])), places=5)
        np.testing.assert_array_equal(out[0, 1], image[0, 1])


class HSVToolTests(unittest.TestCase):
    def _pixels(self, *colours):
        image = np.ones((1, len(colours), 4), np.float32)
        image[0, :, :3] = np.array(colours, np.float32)
        return image

    def test_default_is_the_identity(self):
        image = np.random.RandomState(3).rand(8, 8, 4).astype(np.float32) * 3
        np.testing.assert_array_equal(kernel("HSVTool", image), image)

    def test_rotating_hue_by_120_turns_red_into_green(self):
        out = kernel("HSVTool", self._pixels((1, 0, 0), (0, 1, 0), (0, 0, 1)), hue_rotation=120.0)
        np.testing.assert_allclose(out[0, :, :3], [(0, 1, 0), (0, 0, 1), (1, 0, 0)], atol=1e-6)
        np.testing.assert_allclose(kernel("HSVTool", self._pixels((0.5, 0.2, 0.1)), hue_rotation=360.0)[0, 0, :3],
                                   (0.5, 0.2, 0.1), atol=1e-6)

    def test_hue_range_limits_the_rotation_and_a_rolloff_softens_it(self):
        image = self._pixels((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0.5, 0))
        hard = kernel("HSVTool", image, hue_range_min=340.0, hue_range_max=20.0, hue_rotation=120.0)   # wraps through 0
        np.testing.assert_allclose(hard[0, 0, :3], (0, 1, 0), atol=1e-6)                                 # red inside
        np.testing.assert_array_equal(hard[0, 1:3], image[0, 1:3])                                       # green and blue outside
        self.assertFalse(np.allclose(hard[0, 3, :3], image[0, 3, :3]) is False and False)                # orange (30 deg): outside, unchanged
        np.testing.assert_array_equal(hard[0, 3], image[0, 3])
        soft = kernel("HSVTool", image, hue_range_min=340.0, hue_range_max=20.0, hue_rolloff=20.0, hue_rotation=120.0)
        np.testing.assert_allclose(soft[0, 0, :3], (0, 1, 0), atol=1e-6)
        # orange sits 10 degrees past the range, halfway down the 20 degree rolloff: rotated by 60
        np.testing.assert_allclose(soft[0, 3, :3], (0.5, 1.0, 0.0), atol=1e-5)

    def test_saturation_and_brightness_adjust_scale_or_force(self):
        image = self._pixels((0.8, 0.4, 0.4), (2.0, 1.0, 1.0))
        half = kernel("HSVTool", image, sat_adjust=-0.5)
        np.testing.assert_allclose(half[0, 0, :3], (0.8, 0.6, 0.6), atol=1e-6)         # S 0.5 -> 0.25, V kept
        forced = kernel("HSVTool", image, sat_adjust=0.0, set_saturation=1)
        np.testing.assert_allclose(forced[0, :, :3], [(0.8, 0.8, 0.8), (2.0, 2.0, 2.0)], atol=1e-6)
        bright = kernel("HSVTool", image, brt_adjust=1.0, brightness_range_max=1000.0)
        np.testing.assert_allclose(bright[0, 0, :3], (1.6, 0.8, 0.8), atol=1e-6)
        np.testing.assert_allclose(bright[0, 1, :3], (4.0, 2.0, 2.0), atol=1e-6)
        setv = kernel("HSVTool", image, brt_adjust=0.25, set_brightness=1, brightness_range_max=1000.0)
        np.testing.assert_allclose(setv[0, 0, :3], (0.25, 0.125, 0.125), atol=1e-6)

    def test_saturation_and_brightness_ranges_gate_the_pixels(self):
        image = self._pixels((0.5, 0.5, 0.5), (0.9, 0.3, 0.3))    # grey (S 0) and a saturated red (S 0.667)
        out = kernel("HSVTool", image, saturation_range_min=0.5, saturation_range_max=1.0, hue_rotation=120.0)
        np.testing.assert_array_equal(out[0, 0], image[0, 0])
        np.testing.assert_allclose(out[0, 1, :3], (0.3, 0.9, 0.3), atol=1e-6)
        dim = kernel("HSVTool", image, brightness_range_min=0.0, brightness_range_max=0.6, hue_rotation=120.0)
        np.testing.assert_allclose(dim[0, 0, :3], (0.5, 0.5, 0.5), atol=1e-6)   # grey has no hue to rotate
        np.testing.assert_array_equal(dim[0, 1], image[0, 1])                       # V 0.9 > 0.6: outside

    def test_output_alpha_writes_the_combined_weight(self):
        image = self._pixels((1, 0, 0), (0, 0, 1))
        out = kernel("HSVTool", image, hue_range_min=340.0, hue_range_max=20.0, output_alpha=1)
        np.testing.assert_allclose(out[0, :, 3], (1.0, 0.0))
        np.testing.assert_array_equal(out[0, :, :3], image[0, :, :3])
        np.testing.assert_array_equal(kernel("HSVTool", image, hue_rotation=90.0)[..., 3], image[..., 3])


class AddMixTests(unittest.TestCase):
    def _pair(self):
        rng = np.random.RandomState(6)
        a = rng.rand(16, 16, 4).astype(np.float32)          # deliberately not premultiplied
        b = rng.rand(16, 16, 4).astype(np.float32)
        return a, b

    def test_equals_merge_over_of_a_premultiplied_a(self):
        a, b = self._pair()
        expected = Evaluator._kernel("Merge", dict(SPECS["Merge"]["params"], operation="over"),
                                     [Evaluator._premultiply(a), b])
        got = Evaluator._kernel("AddMix", dict(SPECS["AddMix"]["params"]), [a, b])
        np.testing.assert_array_equal(got, expected)
        plain = Evaluator._kernel("Merge", dict(SPECS["Merge"]["params"], operation="over"), [a, b])
        self.assertFalse(np.allclose(got, plain))

    def test_mask_and_mix_gate_it_like_merge(self):
        a, b = self._pair()
        mask = np.zeros_like(a)
        mask[:, :8, 3] = 1.0
        got = Evaluator._kernel("AddMix", dict(SPECS["AddMix"]["params"], mix=0.5), [a, b, mask])
        expected = Evaluator._merge_gated("over", Evaluator._premultiply(a), b, 0.5, mask)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(got[:, 8:], b[:, 8:])


class BlendTests(unittest.TestCase):
    def _layers(self, count=3):
        rng = np.random.RandomState(8)
        return [rng.rand(12, 12, 4).astype(np.float32) for _ in range(count)]

    def _run(self, layers, mask=None, **params):
        slots = list(layers) + [None] * (8 - len(layers)) + [mask]
        return Evaluator._kernel("Blend", dict(SPECS["Blend"]["params"], **params), slots)

    def test_equal_weights_give_the_mean(self):
        layers = self._layers(3)
        np.testing.assert_allclose(self._run(layers), sum(layers) / 3, atol=1e-6)
        eight = self._layers(8)
        np.testing.assert_allclose(self._run(eight), sum(eight) / 8, atol=1e-6)

    def test_weights_normalise_or_sum(self):
        a, b, c = self._layers(3)
        got = self._run([a, b, c], weight0=1.0, weight1=3.0, weight2=0.0)
        np.testing.assert_allclose(got, (a + 3 * b) / 4, atol=1e-6)
        summed = self._run([a, b], weight0=0.5, weight1=0.25, normalize=0)
        np.testing.assert_allclose(summed, 0.5 * a + 0.25 * b, atol=1e-6)

    def test_gaps_are_skipped_channels_limit_it_and_mask_gates_it(self):
        a, _, c = self._layers(3)
        gap = Evaluator._kernel("Blend", dict(SPECS["Blend"]["params"]), [a, None, c] + [None] * 5 + [None])
        np.testing.assert_allclose(gap, (a + c) / 2, atol=1e-6)      # weight1 is unused: one wired input in its place
        rgb = self._run([a, c], channels="rgb")
        np.testing.assert_array_equal(rgb[..., 3], a[..., 3])
        mask = np.zeros_like(a)
        mask[:, :6, 3] = 1.0
        gated = self._run([a, c], mask=mask, mix=1.0)
        np.testing.assert_allclose(gated[:, :6], ((a + c) / 2)[:, :6], atol=1e-6)
        np.testing.assert_array_equal(gated[:, 6:], a[:, 6:])

    def test_bypass_passes_the_first_wired_input(self):
        wired = lambda **slots: dict(type="Blend", inputs=dict({f"in{i}": None for i in range(8)}, mask=None, **slots))
        self.assertEqual(bypass_slot(wired(in0="x", in1="y")), "in0")
        self.assertEqual(bypass_slot(wired(in1="y", in3="z")), "in1")
        self.assertEqual(bypass_slot(wired()), "in0")


class CopyRectangleTests(unittest.TestCase):
    def _pair(self):
        a = np.full((40, 60, 4), (1.0, 0.0, 0.0, 1.0), np.float32)
        b = np.full((40, 60, 4), (0.0, 0.0, 1.0, 1.0), np.float32)
        return a, b

    def _area(self, **over):
        return dict(SPECS["CopyRectangle"]["params"], area_x=10.0, area_y=5.0, area_r=30.0, area_t=25.0, **over)

    def test_copies_exactly_inside_and_leaves_the_outside(self):
        a, b = self._pair()
        out = Evaluator._copy_rectangle(a, b, self._area())
        np.testing.assert_array_equal(out[5:25, 10:30], a[5:25, 10:30])
        outside = np.ones(out.shape[:2], bool)
        outside[5:25, 10:30] = False
        np.testing.assert_array_equal(out[outside], b[outside])

    def test_channels_limit_the_copy(self):
        a, b = self._pair()
        out = Evaluator._copy_rectangle(a, b, self._area(channels="alpha"))
        np.testing.assert_array_equal(out[10, 15], (0.0, 0.0, 1.0, 1.0))
        a2 = a.copy()
        a2[..., 3] = 0.25
        np.testing.assert_array_equal(Evaluator._copy_rectangle(a2, b, self._area(channels="alpha"))[10, 15, 3], 0.25)

    def test_softness_fades_inward_from_the_edges(self):
        a, b = self._pair()
        out = Evaluator._copy_rectangle(a, b, self._area(softness=1.0))    # band = half the shorter side (10 px)
        red = out[..., 0]
        self.assertEqual(float(red[0, 0]), 0.0)                # outside
        self.assertLess(float(red[15, 10]), 0.1)               # on the left edge: nearly all B
        self.assertGreater(float(red[15, 20]), 0.94)           # at the centre column: nearly all A
        self.assertTrue(np.all(np.diff(red[15, 10:20]) >= 0))  # ramps up monotonically
        self.assertAlmostEqual(float(red[15, 14]), 4.5 / 10.0, places=5)

    def test_origin_shifts_the_box(self):
        a, b = self._pair()
        moved = Evaluator._copy_rectangle(a[:, 5:], b[:, 5:], self._area(), origin=(5, 0))
        full = Evaluator._copy_rectangle(a, b, self._area())
        np.testing.assert_array_equal(moved, full[:, 5:])


class GraphTests(unittest.TestCase):
    def _plate(self, g):
        g.add("plate", "Rectangle", dict(width=257, height=193, box_x=40, box_y=30, box_width=120, box_height=90, alpha=1.0,
                                         red=0.9, green=0.5, blue=0.2))
        g.add("bg", "Ramp", dict(width=257, height=193))
        g.add("third", "Checker", dict(width=257, height=193, size=16))
        g.add("matte", "Constant", dict(width=257, height=193, red=1, green=1, blue=1, alpha=0.5))

    CASES = (("Grain", dict(seed=9, red_size=1.5, green_size=6.0, blue_size=2.0, red_intensity=0.2, green_intensity=0.2,
                            blue_intensity=0.2, luminance_weighted=1, black=0.1), dict(image="plate")),
             ("Posterize", dict(colors=5), dict(image="plate")),
             ("SoftClip", dict(conversion="logarithmic compress", softclip_min=0.3, softclip_max=3.0), dict(image="bg")),
             ("SoftClip", dict(conversion="preserve hue and brightness", softclip_max=0.6), dict(image="plate")),
             ("HSVTool", dict(hue_range_min=0.0, hue_range_max=60.0, hue_rolloff=30.0, hue_rotation=90.0, sat_adjust=0.3), dict(image="plate")),
             ("AddMix", dict(), dict(A="plate", B="bg")),
             ("Blend", dict(weight0=1.0, weight1=2.0, weight2=0.5), dict(in0="plate", in1="bg", in2="third")),
             ("CopyRectangle", dict(area_x=20.0, area_y=17.0, area_r=201.0, area_t=141.0, softness=0.3), dict(A="plate", B="bg")))

    def test_tiles_match_the_evaluator_across_several_tiles_with_and_without_a_mask(self):
        for kind, params, wires in self.CASES:
            for masked in (False, True):
                with self.subTest(kind=kind, params=params, masked=masked):
                    g = Graph()
                    self._plate(g)
                    inputs = dict(wires, **({"mask": "matte"} if masked else {}))
                    g.add("node", kind, params, **inputs)
                    self.assertIn(kind, SUPPORTED_TILED_KINDS)
                    np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_grain_differs_between_frames_through_the_graph(self):
        g = Graph()
        self._plate(g)
        g.add("node", "Grain", dict(seed=3), image="bg")
        from nodebased.imaging import Evaluator as E
        one = E().evaluate(dict(g.doc, view="node"), frame=1)
        two = E().evaluate(dict(g.doc, view="node"), frame=2)
        again = E().evaluate(dict(g.doc, view="node"), frame=1)
        np.testing.assert_array_equal(one, again)
        self.assertFalse(np.array_equal(one, two))

    def test_mix_zero_and_bypass(self):
        cases = (("Grain", dict(image="plate"), dict(red_intensity=0.3)),
                 ("Posterize", dict(image="plate"), dict(colors=2)),
                 ("SoftClip", dict(image="plate"), dict(conversion="preserve hue and saturation", softclip_max=0.3)),
                 ("HSVTool", dict(image="plate"), dict(hue_rotation=90.0)),
                 ("AddMix", dict(A="bg", B="plate"), dict()),
                 ("Blend", dict(in0="plate", in1="bg"), dict()),
                 ("CopyRectangle", dict(A="bg", B="plate"), dict(area_x=0.0, area_y=0.0, area_r=100.0, area_t=100.0)))
        for kind, wires, params in cases:
            with self.subTest(kind=kind):
                g = Graph()
                self._plate(g)
                first = wires.get("image") or wires.get("B") or wires.get("in0")
                base = evaluator_pixels(g.doc, first)
                g.add("mixed", kind, dict(params, mix=0.0), **wires)
                g.add("node", kind, params, **wires)
                np.testing.assert_allclose(evaluator_pixels(g.doc, "mixed"), base, atol=1e-6)
                self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), base))
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), base))
                self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), base))

    def test_registration(self):
        for kind in KINDS:
            self.assertIn("mask", SPECS[kind]["optional_inputs"])
            for name, value in SPECS[kind]["params"].items():
                if name in LIMITS and isinstance(value, (int, float)):
                    lo, hi = LIMITS[name]
                    self.assertTrue(lo <= value <= hi, (kind, name, value))
        self.assertEqual(bypass_slot(dict(type="AddMix", inputs={"A": "a", "B": "b", "mask": None})), "B")
        self.assertEqual(bypass_slot(dict(type="CopyRectangle", inputs={"A": "a", "B": None, "mask": None})), "A")
        self.assertEqual(SPECS["Blend"]["inputs"] + SPECS["Blend"]["optional_inputs"][:-1],
                         [f"in{i}" for i in range(8)])
        for choice in CHOICES["conversion"]:
            self.assertIn(choice, ("none", "preserve hue and brightness", "preserve hue and saturation",
                                   "logarithmic compress"))


if __name__ == "__main__":
    unittest.main()
