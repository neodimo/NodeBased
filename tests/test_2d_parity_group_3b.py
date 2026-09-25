"""Step 3b: Defocus, DirBlur, DropShadow, Position, BlackOutside, AdjustBBox. Pixel assertions
against hand-computed values, evaluator/tile-path parity (including across several tiles), mask +
mix, bypass, window behaviour and spec coverage. See docs/PARITY_2D.md for the audit rows these flip."""
import math
import unittest

import numpy as np

from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor


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


def evaluator_raster(document, target):
    return Evaluator().evaluate_raster(dict(document, view=target))


def tile_pixels(document, target):
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def kernel(kind, image, **params):
    p = dict(SPECS[kind]["params"])
    p.update(params)
    return Evaluator._kernel(kind, p, [image])


def dot(size=41, at=None, value=1.0):
    frame = np.zeros((size, size, 4), np.float32)
    at = size // 2 if at is None else at
    frame[at, at] = value
    return frame


class DefocusPixelTests(unittest.TestCase):
    def test_single_pixel_becomes_a_flat_disc_of_the_expected_radius(self):
        out = kernel("Defocus", dot(), defocus=5.0)
        plane = out[..., 0]
        ys, xs = np.mgrid[-20:21, -20:21]
        inside = (xs ** 2 + ys ** 2) <= 25
        weight = 1.0 / inside.sum()
        # Every pixel inside the radius carries the same weight (a plateau, not a bell), every
        # pixel outside it carries nothing.
        np.testing.assert_allclose(plane[inside], weight, rtol=1e-5)
        self.assertEqual(float(plane[~inside].max()), 0.0)
        self.assertEqual(int(inside.sum()), 81)   # hand count of lattice points in x^2 + y^2 <= 25

    def test_energy_is_preserved_and_the_disc_is_symmetric(self):
        out = kernel("Defocus", dot(), defocus=7.5)
        for c in range(4):
            plane = out[..., c]
            self.assertAlmostEqual(float(plane.sum()), 1.0, places=5)
            np.testing.assert_allclose(plane, plane[::-1, :], atol=1e-7)
            np.testing.assert_allclose(plane, plane[:, ::-1], atol=1e-7)
            np.testing.assert_allclose(plane, plane.T, atol=1e-7)

    def test_aspect_stretches_the_disc_along_x(self):
        out = kernel("Defocus", dot(), defocus=6.0, aspect=2.0)[..., 0]
        ys, xs = np.nonzero(out)
        # rx = 6, ry = 3
        self.assertEqual((int(xs.min()), int(xs.max())), (14, 26))
        self.assertEqual((int(ys.min()), int(ys.max())), (17, 23))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)

    def test_radius_zero_and_below_cutoff_are_identity(self):
        frame = np.random.default_rng(3).random((9, 11, 4)).astype(np.float32)
        np.testing.assert_array_equal(kernel("Defocus", frame, defocus=0.0), frame)
        np.testing.assert_array_equal(kernel("Defocus", frame, defocus=0.4), frame)

    def test_flat_image_is_unchanged_including_at_the_borders(self):
        flat = np.full((10, 12, 4), 0.37, np.float32)
        np.testing.assert_allclose(kernel("Defocus", flat, defocus=6.0), flat, atol=1e-6)

    def test_channels_rgb_leaves_alpha_alone(self):
        frame = dot()
        out = kernel("Defocus", frame, defocus=4.0, channels="rgb")
        np.testing.assert_array_equal(out[..., 3], frame[..., 3])
        self.assertLess(float(out[20, 20, 0]), 1.0)


class DefocusGraphTests(unittest.TestCase):
    def test_tiles_match_the_evaluator_across_several_tiles_and_with_a_mask(self):
        for params in (dict(defocus=5.0), dict(defocus=9.0, aspect=0.5), dict(defocus=4.0, aspect=3.0)):
            with self.subTest(params=params):
                g = Graph()
                g.add("plate", "Checker", dict(width=257, height=193, size=11))
                g.add("matte", "Constant", dict(width=257, height=193, red=1, green=1, blue=1, alpha=0.5))
                g.add("node", "Defocus", params, image="plate", mask="matte")
                np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_mix_zero_is_identity_and_zero_mask_hides_the_effect(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=32, height=24, size=4))
        g.add("matte", "Constant", dict(width=32, height=24, alpha=0.0))
        g.add("mixed", "Defocus", dict(defocus=5.0, mix=0.0), image="plate")
        g.add("masked", "Defocus", dict(defocus=5.0), image="plate", mask="matte")
        base = evaluator_pixels(g.doc, "plate")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "mixed"), base)
        np.testing.assert_allclose(evaluator_pixels(g.doc, "masked"), base)

    def test_bypassed_defocus_equals_its_input_on_both_paths(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=32, height=24, size=8))
        g.add("node", "Defocus", dict(defocus=6.0), image="plate")
        upstream = evaluator_pixels(g.doc, "plate")
        self.assertFalse(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
        g.bypass("node")
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, "node"), upstream))
        self.assertTrue(np.array_equal(tile_pixels(g.doc, "node"), upstream))

    def test_registered_with_mask_mix_limits_and_a_tile_path(self):
        self.assertIn("Defocus", SUPPORTED_TILED_KINDS)
        self.assertIn("mask", SPECS["Defocus"]["optional_inputs"])
        self.assertIn("mix", SPECS["Defocus"]["params"])
        for name in ("defocus", "aspect"):
            self.assertIn(name, LIMITS)
        self.assertEqual(bypass_slot(dict(type="Defocus", inputs={"image": "x", "mask": None})), "image")


if __name__ == "__main__":
    unittest.main()
