"""ColorCorrect like Nuke's (properties overhaul, DiMo 10/18).

Known values follow the measured model of Nuke 17's ColorCorrect (andreageremia.it, "The
ColorCorrect Node in Nuke"): Rec.709 luminance 0.2125/0.7154/0.0721; shadows = 1 - smoothstep(0,
0.09, L), highlights = smoothstep(0.5, 1, L), midtones the rest; per range saturation, contrast,
gamma, gain, offset in that order, master woven in knob by knob; contrast pivots 0.18 (master,
midtones), 0.04 (shadows), 0.65 (highlights); values <= 0 scaled by what 1 becomes."""
import copy
import unittest

import numpy as np

from nodebased.core import SCHEMA_VERSION, SPECS, demo_document, upgrade_document, validate
from nodebased.imaging import Evaluator


def cc(rgb, **params):
    pixel = np.array([[[*rgb, 1.0]]], np.float32)
    return Evaluator._color_correct(pixel, params)[0, 0, :3].astype(np.float64)


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lum(rgb):
    return 0.2125 * rgb[0] + 0.7154 * rgb[1] + 0.0721 * rgb[2]


class ColorCorrectMathTests(unittest.TestCase):
    def test_defaults_pass_colour_through_exactly(self):
        for rgb in ((0.2, 0.3, 0.4), (0.01, 0.02, 0.03), (4.0, 2.0, 1.0), (-0.1, 0.5, 0.2)):
            np.testing.assert_array_equal(cc(rgb), np.float32(rgb))

    def test_master_runs_saturation_contrast_gamma_gain_offset_in_that_order(self):
        rgb = (0.3, 0.2, 0.1)
        L = lum(rgb)
        s, c, g, k, o = 1.3, 1.2, 1.4, 1.5, 0.05
        expected = []
        for value in rgb:
            x = L + s * (value - L)
            x = (x / 0.18) ** c * 0.18
            x = x ** (1 / g)
            expected.append(x * k + o)
        np.testing.assert_allclose(cc(rgb, saturation=s, contrast=c, gamma=g, gain=k, offset=o),
                                   expected, rtol=1e-5)

    def test_master_contrast_pivots_on_middle_grey(self):
        np.testing.assert_allclose(cc((0.18, 0.18, 0.18), contrast=2.0), (0.18,) * 3, rtol=1e-5)
        np.testing.assert_allclose(cc((0.36, 0.36, 0.36), contrast=2.0)[0], 0.72, rtol=1e-5)

    def test_gamma_is_a_true_power_above_one(self):
        np.testing.assert_allclose(cc((3.0, 3.0, 3.0), gamma=2.0), (3 ** 0.5,) * 3, rtol=1e-5)

    def test_values_at_or_below_zero_scale_by_what_one_becomes(self):
        out = cc((-0.1, 0.5, 0.5), contrast=1.6)
        self.assertAlmostEqual(out[0], -0.1 * (1 / 0.18) ** 0.6, places=5)

    def test_master_gain_comes_before_a_range_offset(self):
        rgb = (0.2, 0.2, 0.2)       # luminance 0.2: wholly midtones
        np.testing.assert_allclose(cc(rgb, gain=2.0, midtones_offset=0.1), (0.5,) * 3, rtol=1e-5)

    def test_ranges_blend_by_the_default_lookup_curves(self):
        for rgb in ((0.02, 0.03, 0.04), (0.6, 0.7, 0.8), (0.05, 0.05, 0.05)):
            L = lum(rgb)
            shadows = 1 - smoothstep(0, 0.09, L)
            highlights = smoothstep(0.5, 1, L)
            midtones = 1 - shadows - highlights
            out = cc(rgb, shadows_gain=2.0, highlights_offset=0.25)
            expected = [shadows * 2 * v + midtones * v + highlights * (v + 0.25) for v in rgb]
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_each_range_contrast_has_its_own_pivot(self):
        for zone, pivot, rgb in (("shadows", 0.04, (0.0, 0.0, 0.0)), ("midtones", 0.18, (0.18,) * 3),
                                 ("highlights", 0.65, (2.0,) * 3)):
            x = 0.01 if zone == "shadows" else rgb[0]
            out = cc((x, x, x), **{f"{zone}_contrast": 2.0})
            L = x
            weights = {"shadows": 1 - smoothstep(0, 0.09, L), "highlights": smoothstep(0.5, 1, L)}
            weights["midtones"] = 1 - weights["shadows"] - weights["highlights"]
            expected = weights[zone] * (x / pivot) ** 2 * pivot + (1 - weights[zone]) * x
            self.assertAlmostEqual(out[0], expected, places=5, msg=zone)

    def test_a_custom_shadow_curve_moves_the_range(self):
        curve = '{"interpolation":"linear","points":[[0,1],[0.4,0]]}'
        out = cc((0.2, 0.2, 0.2), shadows_gain=3.0, shadows_curve=curve)
        # luminance 0.2: half shadow by this curve, half midtone (highlight 0)
        np.testing.assert_allclose(out, (0.5 * 0.6 + 0.5 * 0.2,) * 3, rtol=1e-5)

    def test_unpremultiplies_before_correcting(self):
        pixel = np.array([[[0.1, 0.1, 0.1, 0.5]]], np.float32)
        out = Evaluator._color_correct(pixel, {"offset": 0.1})
        np.testing.assert_allclose(out[0, 0], (0.15, 0.15, 0.15, 0.5), rtol=1e-5)


class ColorCorrectMigrationTests(unittest.TestCase):
    """Schema 24 -> 25 maps the old lift/gamma/gain/saturation node onto the new master knobs."""

    def old_document(self, **params):
        base = {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0, "mix": 1.0}
        base.update(params)
        document = demo_document()
        document["version"] = 24       # the literal: the ColorCorrect step is v24 -> v25
        document["nodes"]["cc"] = {"type": "ColorCorrect", "name": "OldColorCorrect", "pos": [0, 0],
                                   "disabled": False, "inputs": {"image": None, "mask": None},
                                   "params": base}
        return document

    @staticmethod
    def old_formula(rgb, lift, gamma, gain, saturation):
        x = np.asarray(rgb, np.float64) * gain + lift * (1 - np.asarray(rgb, np.float64))
        x = np.sign(x) * np.abs(x) ** (1 / gamma)
        L = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
        return L + (x - L) * saturation

    def migrated(self, **params):
        document = upgrade_document(copy.deepcopy(self.old_document(**params)))
        validate(document)
        self.assertEqual(document["version"], SCHEMA_VERSION)
        migrated = document["nodes"]["cc"]["params"]
        self.assertNotIn("lift", migrated)
        self.assertEqual(set(migrated), set(SPECS["ColorCorrect"]["params"]))
        return migrated

    def test_migration_is_exact_when_gamma_is_one(self):
        old = dict(lift=0.1, gamma=1.0, gain=2.0, saturation=0.5)
        params = self.migrated(**old)
        rgb = (0.2, 0.3, 0.4)
        # the old saturation used 0.2126/0.7152/0.0722; the difference is in the fourth decimal
        np.testing.assert_allclose(cc(rgb, **params), self.old_formula(rgb, **old), atol=2e-4)

    def test_migration_is_exact_for_gain_and_gamma_without_lift(self):
        old = dict(lift=0.0, gamma=2.0, gain=4.0, saturation=1.0)
        params = self.migrated(**old)
        rgb = (0.2, 0.3, 0.4)
        np.testing.assert_allclose(cc(rgb, **params), self.old_formula(rgb, **old), rtol=1e-5)

    def test_default_old_node_stays_an_identity(self):
        params = self.migrated()
        self.assertEqual(params, SPECS["ColorCorrect"]["params"])


if __name__ == "__main__":
    unittest.main()
