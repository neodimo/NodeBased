"""Step 3a parts 2 and 3: Soften and Exposure. Pixel assertions against hand-computed values,
evaluator/tile-path parity (including across several tiles), mask + mix, bypass and spec coverage.
See docs/PARITY_2D.md for the audit rows these flip."""
import math
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor

NEW_KINDS = ("Soften", "Exposure")


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


def soften(image, **params):
    p = {"soften_size": 4.0, "channels": "rgba", "mix": 1.0}
    p.update(params)
    return Evaluator._kernel("Soften", p, [image])


def exposure(image, **params):
    p = dict(SPECS["Exposure"]["params"])
    p.update(params)
    return Evaluator._kernel("Exposure", p, [image])


def px(*v):
    return np.array([[list(v)]], np.float32)


class SoftenPixelTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((21, 21, 4), np.float32)
        self.frame[10, 10] = 1.0

    def test_single_pixel_gives_a_symmetric_kernel_that_keeps_the_energy(self):
        out = soften(self.frame, soften_size=3.0)
        for c in range(4):
            plane = out[..., c]
            self.assertAlmostEqual(float(plane.sum()), 1.0, places=5)
            np.testing.assert_allclose(plane, plane[::-1, :], atol=1e-7)
            np.testing.assert_allclose(plane, plane[:, ::-1], atol=1e-7)
            np.testing.assert_allclose(plane, plane.T, atol=1e-7)

    def test_kernel_is_gaussian_shaped_and_matches_hand_computed_weights(self):
        out = soften(self.frame, soften_size=3.0)
        # sigma = 1, half-width 3: 1-D weights exp(-x^2/2) / sum, the 2-D kernel is their outer product
        w = np.exp(-0.5 * np.arange(-3, 4) ** 2)
        w /= w.sum()
        np.testing.assert_allclose(out[7:14, 7:14, 0], np.outer(w, w), atol=1e-6)
        self.assertEqual(float(out[..., 0][:7].sum() + out[..., 0][14:].sum()), 0.0)
        # Falls off smoothly, unlike Blur's flat box.
        row = out[10, 10:14, 0]
        self.assertTrue((np.diff(row) < 0).all())

    def test_size_zero_and_below_cutoff_are_identity(self):
        frame = np.random.default_rng(1).random((9, 11, 4)).astype(np.float32)
        np.testing.assert_array_equal(soften(frame, soften_size=0.0), frame)
        np.testing.assert_array_equal(soften(frame, soften_size=0.4), frame)

    def test_flat_image_is_unchanged_including_at_the_borders(self):
        flat = np.full((8, 8, 4), 0.37, np.float32)
        np.testing.assert_allclose(soften(flat, soften_size=5.0), flat, atol=1e-6)

    def test_channels_rgb_leaves_alpha_alone(self):
        out = soften(self.frame, soften_size=3.0, channels="rgb")
        np.testing.assert_array_equal(out[..., 3], self.frame[..., 3])
        self.assertLess(float(out[10, 10, 0]), 1.0)


class ExposurePixelTests(unittest.TestCase):
    def test_one_stop_doubles_a_mid_grey(self):
        out = exposure(px(0.18, 0.18, 0.18, 1.0), red=1.0, green=1.0, blue=1.0, gang=0)
        np.testing.assert_allclose(out, px(0.36, 0.36, 0.36, 1.0), atol=1e-7)

    def test_minus_one_stop_halves_and_zero_is_identity(self):
        np.testing.assert_allclose(exposure(px(0.5, 0.5, 0.5, 1.0), red=-1.0, green=-1.0, blue=-1.0, gang=0),
                                   px(0.25, 0.25, 0.25, 1.0), atol=1e-7)
        image = px(0.3, 1.7, -0.2, 0.6)
        np.testing.assert_array_equal(exposure(image), image)

    def test_per_channel_exposure_and_gang(self):
        image = px(0.5, 0.5, 0.5, 1.0)
        np.testing.assert_allclose(exposure(image, red=1.0, green=0.0, blue=-1.0, gang=0),
                                   px(1.0, 0.5, 0.25, 1.0), atol=1e-7)
        # Ganged: the red slider drives all three channels, as Nuke moves the sliders together.
        np.testing.assert_allclose(exposure(image, red=1.0, green=-3.0, blue=5.0, gang=1),
                                   px(1.0, 1.0, 1.0, 1.0), atol=1e-7)

    def test_blackpoint_pixel_becomes_black_and_the_rest_scales_about_it(self):
        image = px(0.1, 0.5, 0.3, 1.0)
        out = exposure(image, blackpoint=0.1, red=1.0, green=1.0, blue=1.0, gang=0)
        # (in - 0.1) * 2: red at the black point is exactly 0, green 0.8, blue 0.4
        np.testing.assert_allclose(out, px(0.0, 0.8, 0.4, 1.0), atol=1e-7)

    def test_densities_mode_matches_the_hand_computed_value(self):
        # 0.6-gamma negative stock: gain = 10 ** (density / 0.6). density 0.3 -> 10 ** 0.5
        out = exposure(px(0.2, 0.2, 0.2, 1.0), exposure_mode="densities", red=0.3, green=0.3,
                       blue=0.6, gang=0)
        np.testing.assert_allclose(out[0, 0, :3], [0.2 * 10 ** 0.5, 0.2 * 10 ** 0.5, 0.2 * 10.0], rtol=1e-6)
        self.assertEqual(float(out[0, 0, 3]), 1.0)
        # One stop is log10(2) * 0.6 of density.
        stop = exposure(px(0.5, 0.5, 0.5, 1.0), exposure_mode="densities", red=0.6 * math.log10(2),
                        gang=1)
        np.testing.assert_allclose(stop[0, 0, :3], [1.0, 1.0, 1.0], rtol=1e-6)

    def test_channels_rgba_applies_the_red_gain_to_alpha_rgb_leaves_it(self):
        image = px(0.5, 0.5, 0.5, 0.5)
        self.assertEqual(float(exposure(image, red=1.0, gang=1)[0, 0, 3]), 0.5)
        self.assertEqual(float(exposure(image, red=1.0, gang=1, channels="rgba")[0, 0, 3]), 1.0)

    def test_hdr_and_negative_inputs_stay_finite(self):
        image = px(50.0, -3.0, 0.0, 1.0)
        self.assertTrue(np.isfinite(exposure(image, red=4.0, gang=1)).all())


class MaskMixTests(unittest.TestCase):
    def _params(self, kind, mix):
        return {"Soften": {"soften_size": 3.0, "mix": mix},
                "Exposure": {"red": 2.0, "gang": 1, "blackpoint": 0.05, "mix": mix}}[kind]

    def test_mix_zero_is_identity(self):
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=16, height=12, size=4))
                g.add("node", kind, self._params(kind, 0.0), image="plate")
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate"))

    def test_zero_mask_hides_the_effect(self):
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Constant", dict(red=0.4, green=0.4, blue=0.4, alpha=1.0))
                g.add("matte", "Constant", dict(alpha=0.0))
                g.add("node", kind, self._params(kind, 1.0), image="plate", mask="matte")
                np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate"))


class TilePathParityTests(unittest.TestCase):
    CASES = {"Soften": dict(soften_size=5.0, channels="rgba", mix=1.0),
             "Exposure": dict(red=1.0, green=0.5, blue=-0.5, gang=0, blackpoint=0.05, mix=0.8)}

    def test_both_nodes_are_tile_native(self):
        for kind in NEW_KINDS:
            self.assertIn(kind, SUPPORTED_TILED_KINDS)

    def test_tile_path_matches_the_evaluator_with_a_mask(self):
        for kind, params in self.CASES.items():
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=48, height=32, size=8))
                g.add("matte", "Constant", dict(width=48, height=32, red=1, green=1, blue=1, alpha=0.5))
                g.add("node", kind, params, image="plate", mask="matte")
                np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_soften_is_seamless_across_multiple_tiles(self):
        for size in (3.0, 6.0, 12.0):
            with self.subTest(size=size):
                g = Graph()
                g.add("plate", "Checker", dict(width=257, height=193, size=11))
                g.add("node", "Soften", dict(soften_size=size), image="plate")
                np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_exposure_matches_across_multiple_tiles(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=257, height=193, size=11))
        g.add("node", "Exposure", self.CASES["Exposure"], image="plate")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)


class BypassTests(unittest.TestCase):
    def test_bypassed_kind_equals_its_input_on_both_paths(self):
        visible = {"Soften": dict(soften_size=4.0), "Exposure": dict(red=2.0, gang=1)}
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                g = Graph()
                g.add("plate", "Checker", dict(width=32, height=24, size=8))
                g.add("half", "Grade", dict(exposure=-1.0), image="plate")
                g.add("node", kind, visible[kind], image="half")
                upstream = evaluator_pixels(g.doc, "half")
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

    def test_limits_and_choices(self):
        for name in ("soften_size", "blackpoint", "gang", "red", "green", "blue"):
            self.assertIn(name, LIMITS)
        self.assertEqual(CHOICES["exposure_mode"], ["stops", "densities"])
        self.assertEqual(SPECS["Exposure"]["params"]["exposure_mode"], "stops")

    def test_dispatcher_creates_every_new_node_and_validates_choices(self):
        for kind in NEW_KINDS:
            d = Dispatcher()
            self.assertIn("id", d.execute(dict(op="create", type=kind))["result"])
        d = Dispatcher()
        d.execute(dict(op="create", id="e", type="Exposure"))
        with self.assertRaises(Exception):
            d.execute(dict(op="set", id="e", param="exposure_mode", value="lights"))


if __name__ == "__main__":
    unittest.main()
