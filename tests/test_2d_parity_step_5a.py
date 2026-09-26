"""Step 5a: EdgeBlur, EdgeExtend, LightWrap, Dither. Pixel assertions on constructed mattes,
evaluator/tile-path parity across several tiles (with a mask), mix, bypass and registration.
See docs/PARITY_2D.md for the audit rows these flip."""
import unittest

import numpy as np

from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, tile_pixels, kernel

KINDS = ("EdgeBlur", "EdgeExtend", "LightWrap", "Dither")


def square_on_checker(size=64, lo=20, hi=44):
    """Opaque square (premultiplied) with a fine checker inside, transparent black outside."""
    frame = np.zeros((size, size, 4), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((xx + yy) % 2) * 0.8 + 0.1).astype(np.float32)
    inside = (yy >= lo) & (yy < hi) & (xx >= lo) & (xx < hi)
    for c in range(3):
        frame[..., c] = np.where(inside, checker, 0.0)
    frame[..., 3] = inside
    return frame


class EdgeBlurTests(unittest.TestCase):
    def test_only_a_band_at_the_edge_changes_and_its_width_is_measured(self):
        frame = square_on_checker()
        out = kernel("EdgeBlur", frame, edgeblur_size=3.0, edge_mult=1.0)
        changed = np.abs(out - frame).max(axis=2) > 1e-6
        rows = np.nonzero(changed[:, 32])[0]
        # Column 32 crosses the square's top (row 20) and bottom (row 44) edges. Every changed
        # row lies within the 3 pixel band either side of one of them: rows 17-22 or 41-46.
        self.assertTrue(all(17 <= r <= 22 or 41 <= r <= 46 for r in rows.tolist()), rows.tolist())
        self.assertEqual(rows.tolist(), list(range(17, 23)) + list(range(41, 47)))   # band width 3 + 3 per edge
        self.assertEqual(int(changed[24:40, 24:40].sum()), 0, "interior detail beyond the band is untouched")
        # the same checker under a plain Blur is smeared right through the interior
        plain = kernel("Blur", frame, radius=3.0)
        self.assertGreater(float(np.abs(plain - frame)[30:34, 30:34].max()), 0.05)
        np.testing.assert_array_equal(out[28:36, 28:36], frame[28:36, 28:36])

    def test_edge_mult_zero_is_the_identity_and_a_wider_multiplier_widens_the_band(self):
        frame = square_on_checker()
        np.testing.assert_array_equal(kernel("EdgeBlur", frame, edge_mult=0.0), frame)
        narrow = (np.abs(kernel("EdgeBlur", frame, edgeblur_size=4.0, edge_mult=1.0) - frame).max(axis=2) > 1e-6).sum()
        wide = (np.abs(kernel("EdgeBlur", frame, edgeblur_size=4.0, edge_mult=2.0) - frame).max(axis=2) > 1e-6).sum()
        self.assertGreater(int(wide), int(narrow))

    def test_flat_input_is_untouched(self):
        flat = np.full((32, 32, 4), 0.5, np.float32)
        np.testing.assert_allclose(kernel("EdgeBlur", flat, edgeblur_size=5.0), flat, atol=1e-6)


class EdgeExtendTests(unittest.TestCase):
    def _matte(self):
        frame = np.zeros((9, 20, 4), np.float32)
        frame[:, :10] = (1.0, 0.0, 0.0, 1.0)              # opaque red
        frame[:, 10] = (0.05, 0.0, 0.0, 0.3)               # dark, half-transparent fringe: straight (0.167, 0, 0)
        return frame

    def test_dark_fringe_takes_the_edge_colour_and_it_extends_past_the_matte(self):
        out = kernel("EdgeExtend", self._matte(), extend_size=3.0)
        for x in (9, 10, 11, 12):
            np.testing.assert_allclose(out[4, x, :3], (1.0, 0.0, 0.0), atol=1e-6, err_msg=f"column {x}")
        # first fully transparent pixel carries the edge colour; alpha untouched everywhere
        np.testing.assert_allclose(out[4, 11, :3], (1.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_array_equal(out[..., 3], self._matte()[..., 3])
        # colour reaches exactly ceil(size) pixels past the last trusted pixel, no farther
        np.testing.assert_allclose(out[4, 12, :3], (1.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_array_equal(out[4, 14, :3], (0.0, 0.0, 0.0))

    def test_without_extension_the_fringe_is_dark(self):
        out = kernel("EdgeExtend", self._matte(), extend_size=0.0)
        self.assertAlmostEqual(float(out[4, 10, 0]), 0.05 / 0.3, places=5)
        np.testing.assert_array_equal(out[4, 11, :3], (0.0, 0.0, 0.0))


class LightWrapTests(unittest.TestCase):
    def _pair(self, size=64):
        fg = np.zeros((size, size, 4), np.float32)
        fg[16:48, 16:48] = (0.2, 0.2, 0.2, 1.0)
        bg = np.full((size, size, 4), (1.0, 0.6, 0.2, 1.0), np.float32)
        return fg, bg

    def test_light_lands_inside_the_edge_band_only(self):
        fg, bg = self._pair()
        out = Evaluator._kernel("LightWrap", dict(SPECS["LightWrap"]["params"], wrap_diffuse=6.0), [fg, bg])
        added = (out - fg)[..., :3].max(axis=2)
        self.assertGreater(float(added[32, 17]), 0.1)               # just inside the left edge
        self.assertAlmostEqual(float(added[32, 32]), 0.0, places=6)  # deep inside: nothing
        self.assertEqual(float(added[:, :16].max()), 0.0)            # outside the matte: nothing
        self.assertEqual(float(added[:16].max()), 0.0)
        touched = np.nonzero(added[32] > 1e-4)[0]
        self.assertTrue(all(16 <= x <= 22 or 41 <= x <= 47 for x in touched.tolist()), touched.tolist())
        np.testing.assert_array_equal(out[..., 3], fg[..., 3])

    def test_dark_background_wraps_nothing_and_constant_highlight_overrides_it(self):
        fg, _ = self._pair()
        black = np.zeros_like(fg)
        base = dict(SPECS["LightWrap"]["params"])
        np.testing.assert_array_equal(Evaluator._kernel("LightWrap", base, [fg, black]), fg)
        const = dict(base, use_constant_highlight=1, red=0.0, green=0.0, blue=1.0)
        out = Evaluator._kernel("LightWrap", const, [fg, black])
        self.assertGreater(float(out[32, 17, 2] - fg[32, 17, 2]), 0.1)
        self.assertEqual(float(out[32, 17, 0] - fg[32, 17, 0]), 0.0)

    def test_highlight_merge_operations_differ_and_intensity_zero_is_the_foreground(self):
        fg, bg = self._pair()
        base = dict(SPECS["LightWrap"]["params"])
        results = {op: Evaluator._kernel("LightWrap", dict(base, highlight_merge=op), [fg, bg]) for op in CHOICES["highlight_merge"]}
        self.assertEqual(len({r.tobytes() for r in results.values()}), 4)
        np.testing.assert_array_equal(Evaluator._kernel("LightWrap", dict(base, intensity=0.0), [fg, bg]), fg)


class DitherTests(unittest.TestCase):
    def test_mean_is_preserved_where_plain_quantising_is_not(self):
        flat = np.full((128, 128, 4), 0.3, np.float32)
        out = kernel("Dither", flat, bits=4, seed=5)
        self.assertAlmostEqual(float(out[..., :3].mean()), 0.3, delta=0.004)
        plain = np.round(0.3 * 15) / 15
        self.assertGreater(abs(plain - 0.3), 0.01)          # a plain 4-bit quantise is off by more
        levels = np.unique(np.round(out[..., 0] * 15, 4))
        self.assertTrue(np.allclose(levels, np.round(levels)), "values sit on the 4-bit grid")
        self.assertGreater(len(levels), 1)
        np.testing.assert_array_equal(out[..., 3], flat[..., 3])

    def test_same_seed_repeats_and_another_seed_differs(self):
        image = np.random.RandomState(1).rand(40, 40, 4).astype(np.float32)
        a, b = kernel("Dither", image, seed=3), kernel("Dither", image, seed=3)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, kernel("Dither", image, seed=4)))

    def test_noise_is_tied_to_the_pixel_position(self):
        image = np.random.RandomState(2).rand(48, 48, 4).astype(np.float32)
        whole = Evaluator._dither(image, SPECS["Dither"]["params"])
        part = Evaluator._dither(image[10:30, 7:40], SPECS["Dither"]["params"], origin=(7, 10))
        np.testing.assert_array_equal(whole[10:30, 7:40], part)


class GraphTests(unittest.TestCase):
    def _plate(self, g):
        g.add("plate", "Rectangle", dict(width=257, height=193, box_x=40, box_y=30, box_width=120, box_height=90, alpha=1.0,
                                         red=0.9, green=0.5, blue=0.2))
        g.add("bg", "Ramp", dict(width=257, height=193))
        g.add("matte", "Constant", dict(width=257, height=193, red=1, green=1, blue=1, alpha=0.5))

    def test_tiles_match_the_evaluator_across_several_tiles_with_a_mask(self):
        cases = (("EdgeBlur", dict(edgeblur_size=6.0, edge_mult=1.5), dict(image="plate")),
                 ("EdgeExtend", dict(extend_size=9.0, extend_threshold=0.3), dict(image="plate")),
                 ("Dither", dict(bits=5, seed=11, dither_amount=1.5), dict(image="plate")),
                 ("LightWrap", dict(wrap_diffuse=14.0, fgblur=3.0, bgblur=9.0, intensity=1.5), dict(fg="plate", bg="bg")),
                 ("LightWrap", dict(wrap_diffuse=5.0, highlight_merge="screen", use_constant_highlight=1), dict(fg="plate", bg="bg")))
        for kind, params, wires in cases:
            for masked in (False, True):
                with self.subTest(kind=kind, params=params, masked=masked):
                    g = Graph()
                    self._plate(g)
                    inputs = dict(wires, **({"mask": "matte"} if masked else {}))
                    g.add("node", kind, params, **inputs)
                    self.assertIn(kind, SUPPORTED_TILED_KINDS)
                    np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_mix_zero_and_bypass(self):
        for kind, wires, params in (("EdgeBlur", dict(image="plate"), dict(edgeblur_size=8.0)),
                                    ("EdgeExtend", dict(image="plate"), dict(extend_size=8.0)),
                                    ("Dither", dict(image="plate"), dict(bits=2)),
                                    ("LightWrap", dict(fg="plate", bg="bg"), dict(wrap_diffuse=12.0))):
            with self.subTest(kind=kind):
                g = Graph()
                self._plate(g)
                base = evaluator_pixels(g.doc, "plate")
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
        self.assertEqual(bypass_slot(dict(type="LightWrap", inputs={"fg": "a", "bg": "b", "mask": None})), "fg")
        self.assertEqual(bypass_slot(dict(type="LightWrap", inputs={"fg": None, "bg": "b", "mask": None})), "fg")
        for name in ("edgeblur_size", "edge_mult", "extend_size", "extend_threshold", "wrap_diffuse", "fgblur",
                     "bgblur", "bits", "dither_amount"):
            self.assertIn(name, LIMITS)


if __name__ == "__main__":
    unittest.main()
