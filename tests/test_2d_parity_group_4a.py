"""Step 4a: HueCorrect and ColorMatrix. Pixel assertions against hand-computed values,
evaluator/tile-path parity (including across several tiles), mask + mix, bypass and spec coverage.
See docs/PARITY_2D.md for the audit rows these flip."""
import unittest

import numpy as np

from nodebased.core import Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor

NEW_KINDS = ("HueCorrect", "ColorMatrix")
BANDS = ("red", "yellow", "green", "cyan", "blue", "magenta")


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


def hue_correct(image, **params):
    p = dict(SPECS["HueCorrect"]["params"])
    p.update(params)
    return Evaluator._kernel("HueCorrect", p, [image])


def color_matrix(image, rows=None, **params):
    p = dict(SPECS["ColorMatrix"]["params"])
    if rows is not None:
        for i in range(3):
            for j in range(3):
                p[f"matrix_{i}{j}"] = rows[i][j]
    p.update(params)
    return Evaluator._kernel("ColorMatrix", p, [image])


def px(*v):
    return np.array([[list(v)]], np.float32)


def noise_image(seed=3, h=16, w=24):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 4), dtype=np.float32) * 2.0 - 0.25


def hue_pixel(deg):
    """A fully saturated pixel at the given HSV hue."""
    x = 1.0 - abs((deg / 60.0) % 2.0 - 1.0)
    sector = int(deg // 60) % 6
    rgb = [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][sector]
    return px(*rgb, 1.0)


class HueCorrectTests(unittest.TestCase):
    def test_unit_multipliers_are_identity(self):
        image = noise_image()
        np.testing.assert_array_equal(hue_correct(image), image)

    def test_red_band_saturation_zero_desaturates_red_and_leaves_green(self):
        red = hue_correct(px(1.0, 0.0, 0.0, 1.0), sat_red=0.0)
        luma = 0.2126
        np.testing.assert_allclose(red[0, 0, :3], [luma] * 3, atol=1e-6)
        green = px(0.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(hue_correct(green, sat_red=0.0), green, atol=1e-6)

    def test_band_anchors_take_exactly_their_own_multiplier(self):
        for i, band in enumerate(BANDS):
            with self.subTest(band=band):
                pixel = hue_pixel(60.0 * i)
                out = hue_correct(pixel, **{f"lum_{band}": 0.5})
                np.testing.assert_allclose(out[0, 0, :3], pixel[0, 0, :3] * 0.5, atol=1e-6)

    def test_luminance_leaves_neutrals_alone(self):
        grey = px(0.4, 0.4, 0.4, 1.0)
        params = {f"lum_{b}": 3.0 for b in BANDS}
        np.testing.assert_allclose(hue_correct(grey, **params), grey, atol=1e-7)

    def test_interpolation_is_continuous_across_band_edges(self):
        # Saturation multiplier sweeps 0 (red) .. 1 elsewhere; walk the hue through the red anchor
        # and the wrap at 360 and require no jump larger than the step's own contribution.
        hues = np.arange(-30.0, 90.0, 0.25) % 360.0
        image = np.concatenate([hue_pixel(h) for h in hues], axis=1)
        out = hue_correct(image, sat_red=0.0, lum_yellow=2.0)
        steps = np.abs(np.diff(out[0, :, :3], axis=0)).max()
        self.assertLess(float(steps), 0.03)
        # ... and the anchor itself is reached: the pure-red pixel is fully desaturated.
        idx = int(np.argmin(np.abs(hues - 0.0)))
        np.testing.assert_allclose(out[0, idx, :3], [0.2126] * 3, atol=1e-6)

    def test_interpolation_midpoint_is_halfway(self):
        # Hue 30 sits midway between red and yellow: smoothstep(0.5) = 0.5.
        out = hue_correct(hue_pixel(30.0), lum_red=0.0, lum_yellow=1.0)
        np.testing.assert_allclose(out[0, 0, :3], hue_pixel(30.0)[0, 0, :3] * 0.5, atol=1e-6)

    def test_hue_shift_rotates_and_preserves_channel_average(self):
        out = hue_correct(px(1.0, 0.0, 0.0, 1.0), hue_shift=120.0)
        np.testing.assert_allclose(out[0, 0, :3], [0.0, 1.0, 0.0], atol=1e-6)
        image = noise_image()
        shifted = hue_correct(image, hue_shift=37.0)
        np.testing.assert_allclose(shifted[..., :3].mean(axis=2), image[..., :3].mean(axis=2), atol=1e-5)
        np.testing.assert_array_equal(shifted[..., 3], image[..., 3])

    def test_alpha_and_hdr_stay_finite(self):
        image = px(50.0, -3.0, 0.0, 0.25)
        out = hue_correct(image, sat_red=0.0, lum_blue=4.0, hue_shift=20.0)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[0, 0, 3]), 0.25)


class ColorMatrixTests(unittest.TestCase):
    def test_identity_is_identity(self):
        image = noise_image()
        np.testing.assert_array_equal(color_matrix(image), image)

    def test_permutation_swaps_channels_exactly(self):
        image = noise_image()
        out = color_matrix(image, rows=[[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(out[..., 0], image[..., 2])
        np.testing.assert_array_equal(out[..., 1], image[..., 0])
        np.testing.assert_array_equal(out[..., 2], image[..., 1])
        np.testing.assert_array_equal(out[..., 3], image[..., 3])

    def test_row_convention(self):
        out = color_matrix(px(1.0, 2.0, 3.0, 1.0), rows=[[1, 1, 1], [0, 2, 0], [0, 0, 0.5]])
        np.testing.assert_allclose(out[0, 0], [6.0, 4.0, 1.5, 1.0])

    def test_invert_restores_the_input(self):
        image = noise_image()
        rows = [[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.05, 0.25, 0.7]]
        forward = color_matrix(image, rows=rows)
        self.assertFalse(np.allclose(forward, image, atol=1e-3))
        back = color_matrix(forward, rows=rows, invert=1)
        np.testing.assert_allclose(back, image, atol=1e-5)

    def test_singular_matrix_with_invert_passes_input_through(self):
        image = noise_image()
        rows = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
        np.testing.assert_array_equal(color_matrix(image, rows=rows, invert=1), image)
        self.assertFalse(np.array_equal(color_matrix(image, rows=rows), image))


class MaskMixTests(unittest.TestCase):
    PARAMS = {"HueCorrect": dict(sat_red=0.0, lum_green=0.5, hue_shift=15.0),
              "ColorMatrix": dict(matrix_00=0.0, matrix_01=1.0, matrix_10=1.0, matrix_11=0.0)}

    def _graph(self, kind, mix=1.0, masked=False):
        g = Graph()
        g.add("plate", "Checker", dict(width=16, height=12, size=4))
        params = dict(self.PARAMS[kind], mix=mix)
        if masked:
            g.add("matte", "Constant", dict(width=16, height=12, alpha=0.0))
            g.add("node", kind, params, image="plate", mask="matte")
        else:
            g.add("node", kind, params, image="plate")
        return g

    def test_mix_zero_and_zero_mask_are_identity(self):
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                g = self._graph(kind, mix=0.0)
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate"))
                g = self._graph(kind, masked=True)
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate"))


class TilePathParityTests(unittest.TestCase):
    CASES = {"HueCorrect": dict(sat_red=0.2, sat_cyan=1.7, lum_yellow=0.6, lum_magenta=1.4,
                                hue_shift=25.0, mix=0.8),
             "ColorMatrix": dict(matrix_00=0.7, matrix_01=0.2, matrix_02=0.1, matrix_10=0.0,
                                 matrix_11=1.2, matrix_12=-0.2, matrix_20=0.3, matrix_21=0.3,
                                 matrix_22=0.4, invert=1, mix=0.9)}

    def test_both_nodes_are_tile_native(self):
        for kind in NEW_KINDS:
            self.assertIn(kind, SUPPORTED_TILED_KINDS)

    def test_tile_path_matches_the_evaluator_across_tiles_with_a_mask(self):
        for kind, params in self.CASES.items():
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=257, height=193, size=11))
                g.add("matte", "Constant", dict(width=257, height=193, red=1, green=1, blue=1, alpha=0.5))
                g.add("node", kind, params, image="plate", mask="matte")
                np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)


class BypassTests(unittest.TestCase):
    def test_bypassed_kind_equals_its_input_on_both_paths(self):
        visible = {"HueCorrect": dict(sat_red=0.0, sat_blue=0.0, sat_green=0.0, sat_yellow=0.0,
                                      sat_cyan=0.0, sat_magenta=0.0),
                   "ColorMatrix": dict(matrix_00=0.5, matrix_11=0.25)}
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=32, height=24, size=8))
                g.add("tint", "Grade", dict(exposure=-1.0), image="plate")
                g.add("node", kind, visible[kind], image="tint")
                upstream = evaluator_pixels(g.doc, "tint")
                self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                g.bypass("node")
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
                self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))

    def test_bypass_slot_is_the_single_image_input(self):
        for kind in NEW_KINDS:
            self.assertEqual(bypass_slot(dict(type=kind, inputs={"image": "x", "mask": None})), "image")


class SpecCoverageTests(unittest.TestCase):
    def test_registered_with_mask_and_mix(self):
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                self.assertIn("mask", SPECS[kind].get("optional_inputs", []))
                self.assertIn("mix", SPECS[kind]["params"])

    def test_limits_cover_every_knob(self):
        for kind in NEW_KINDS:
            for name in SPECS[kind]["params"]:
                self.assertIn(name, LIMITS, name)

    def test_defaults_are_identities(self):
        params = SPECS["ColorMatrix"]["params"]
        for i in range(3):
            for j in range(3):
                self.assertEqual(params[f"matrix_{i}{j}"], 1.0 if i == j else 0.0)
        self.assertTrue(all(SPECS["HueCorrect"]["params"][f"{k}_{b}"] == 1.0 for k in ("sat", "lum") for b in BANDS))

    def test_dispatcher_creates_every_new_node(self):
        for kind in NEW_KINDS:
            self.assertIn("id", Dispatcher().execute(dict(op="create", type=kind))["result"])


if __name__ == "__main__":
    unittest.main()
