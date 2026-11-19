"""2D behaviour, lane 9 step X2: the vectorised defocus gather (nodebased/defocusgather.py) gives the picture the per-pixel loop gave.

ZDefocus and Bokeh are run through the per-pixel reference loop and through the grouped gather on small pictures with a depth ramp, for the disc, the
softened disc, the blades, a kernel image, the hollow filter, fill foreground off, spherical aberration and the gamma and bloom encodings; the two
agree to 1e-6 (the loop sums in float32, the gather in float64, so the loop is the rougher of the two on large filters: see the float64 test).
"""
from __future__ import annotations

import unittest

import numpy as np

from nodebased import bokehknobs, zdefocusknobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator

H, W = 14, 21


def texture(seed=2, h=H, w=W):
    pixels = np.random.default_rng(seed).random((h, w, 4)).astype(np.float32)
    pixels[..., 3] = 1.0
    return pixels


def ramp(lo=0.5, hi=6.0):
    return np.tile(np.linspace(lo, hi, W, dtype=np.float32), (H, 1))


def both(run):
    """(loop result, gather result) of the callable `run`."""
    try:
        Evaluator._zdefocus_reference = True
        old = np.asarray(run(), np.float64)
    finally:
        Evaluator._zdefocus_reference = False
    return old, np.asarray(run(), np.float64)


def zd(image, depth, kernel=None, **knobs_):
    params = {**SPECS["ZDefocus"]["params"], "shape": 1.0, "focal_plane": 3.0, "depth_of_field": 0.2, "size": 5.0, "max_size": 5.0, **knobs_}
    return lambda: Evaluator._zdefocus(image, depth, params, kernel)


def bk(image, depth, kernel=None, **knobs_):
    params = {**SPECS["Bokeh"]["params"], "focalPlane": 3.0, "focusRegionSize": 0.2, "max_kernelsize": 12.0, "multiplier": 8.0, **knobs_}
    planes = bokehknobs.planes_for(image, params, lambda name: None)
    return lambda: bokehknobs.render(image, depth, params, kernel, planes)


class GatherEqualsTheLoop(unittest.TestCase):
    def check(self, run, label):
        old, new = both(run)
        self.assertGreater(np.abs(old - texture()).max(), 1e-3, label + ": the case must blur something")
        # 1e-6 absolute plus 1e-6 relative: the bloom boost lifts values above 1, where the float32 loop's own rounding is a few 1e-6.
        np.testing.assert_allclose(new, old, atol=1e-6, rtol=1e-6, err_msg=label)

    def test_zdefocus_shapes_and_knobs(self):
        image, depth = texture(), ramp()
        cases = {
            "disc": {},
            "soft disc": {"shape": 0.3},
            "blades": {"bokeh_shape": "blades", "blade_count": 5, "blade_rotation": 20.0},
            "blades, rounded, hollow": {"bokeh_shape": "blades", "roundness": 0.4, "inner_size": 0.5, "inner_brightness": 0.5},
            "aspect": {"aspect": 2.0},
            "fill foreground off": {"fill_foreground": 0},
            "gamma and bloom": {"bloom_gamma": 1, "bloom": 1, "bloom_threshold": 0.5},
            "no layers snapping": {"autoLayerSpacing": 0, "nLayers": 3},
            "legacy": {"legacy_zdefocus": 1, "legacy_zdefocus_focus": 1, "focal_plane": 3.0},
            "direct": {"zdefocus_math": "direct", "size": 3.0},
        }
        for label, extra in cases.items():
            with self.subTest(label):
                self.check(zd(image, depth, **extra), label)

    def test_zdefocus_with_a_kernel_image(self):
        kernel = np.zeros((7, 7, 4), np.float32)
        kernel[..., 3] = np.random.default_rng(4).random((7, 7)).astype(np.float32)
        kernel[3, 3, 3] = 1.0
        for extra in ({}, {"use_input_channels": 0}):
            with self.subTest(extra):
                self.check(zd(texture(), ramp(), kernel, bokeh_shape="image", **extra), "kernel image")

    def test_bokeh_outputs_and_knobs(self):
        image, depth = texture(), ramp()
        cases = {
            "default": {},
            "blades": {"kernelType": "Aperture Blades", "kNumSides": 5, "kRotation": 15.0},
            "soft": {"kSoftness": 0.6},
            "spherical aberration": {"kSphAbb": 0.5},
            "bloom": {"bokeh_bloom": 1.0, "bloomThreshold": 0.4},
            "chromatic aberration": {"bokeh_bloom": 1.0, "bloomThreshold": 0.4, "kChrAbb": 0.6},
            "aspect": {"kAspectRatio": 1.7},
            "front only": {"focusOutput": "Front"},
        }
        for label, extra in cases.items():
            with self.subTest(label):
                self.check(bk(image, depth, **extra), label)

    def test_bokeh_with_a_kernel_image(self):
        kernel = np.zeros((9, 9, 4), np.float32)
        kernel[2:7, 2:7, :] = 1.0
        kernel[4, 4, :] = 3.0
        self.check(bk(texture(), ramp(), kernel, kernelType="Input"), "bokeh kernel input")

    def test_flat_depth_and_a_wholly_sharp_picture(self):
        image = texture()
        flat = np.full((H, W), 3.0, np.float32)
        old, new = both(zd(image, flat))
        np.testing.assert_allclose(new, old, atol=1e-6)
        old, new = both(zd(image, ramp(), size=0.4, max_size=0.4))
        np.testing.assert_allclose(new, old, atol=1e-6)

    def test_every_pixel_hidden_by_something_nearer_keeps_its_own_sample(self):
        # A hollow filter whose middle is dropped, around a pixel with only farther neighbours: the loop's "visible[radius, radius] = True" rule.
        image = texture()
        depth = np.full((H, W), 5.0, np.float32)
        depth[7, 10] = 0.2
        for extra in ({"inner_size": 0.9, "inner_brightness": 0.0}, {"fill_foreground": 0}):
            with self.subTest(extra):
                old, new = both(zd(image, depth, **extra))
                np.testing.assert_allclose(new, old, atol=1e-6)

    def test_a_wide_filter_is_closer_to_a_float64_sum_than_the_float32_loop_is(self):
        """On big filters the loop's float32 accumulation drifts (about 2.5e-5 at size 50 on a 1080p ramp); the gather sums in float64."""
        image, depth = texture(h=6, w=48), np.tile(np.linspace(0.5, 6.0, 48, dtype=np.float32), (6, 1))
        run = zd(image, depth, size=24.0, max_size=24.0)
        encode = zdefocusknobs.encode
        try:
            zdefocusknobs.encode = lambda i, p: np.asarray(encode(i, p), np.float64)
            Evaluator._zdefocus_reference = True
            truth = np.asarray(Evaluator._zdefocus(image.astype(np.float64), depth, {**SPECS["ZDefocus"]["params"], "shape": 1.0, "focal_plane": 3.0, "depth_of_field": 0.2, "size": 24.0, "max_size": 24.0}), np.float64)
        finally:
            zdefocusknobs.encode = encode
            Evaluator._zdefocus_reference = False
        old, new = both(run)
        self.assertLessEqual(float(np.abs(new - truth).max()), 1e-6)
        self.assertLessEqual(float(np.abs(new - truth).max()), float(np.abs(old - truth).max()) + 1e-9)

    def test_chunking_does_not_change_the_picture(self):
        from nodebased import defocusgather
        image, depth = texture(), ramp()
        reference = both(zd(image, depth))[1]
        chunk = defocusgather.CHUNK_TAPS
        try:
            defocusgather.CHUNK_TAPS = 50
            np.testing.assert_array_equal(np.asarray(zd(image, depth)(), np.float64), reference)
        finally:
            defocusgather.CHUNK_TAPS = chunk


class TilesStillEqualTheFullFrame(unittest.TestCase):
    def test_zdefocus_and_bokeh_tiles(self):
        from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels, tile_pixels
        for kind, params in (("ZDefocus", {"focal_plane": 0.5, "depth_of_field": 0.2, "size": 4.0, "max_size": 3.0}),
                             ("Bokeh", {"focalPlane": 0.5, "focusRegionSize": 0.1, "max_kernelsize": 6.0, "multiplier": 4.0})):
            g = Graph()
            g.add("plate", "Checker", dict(width=45, height=37, size=5))
            g.add("depth", "Ramp", dict(width=45, height=37))
            g.add("node", kind, params, image="plate", depth="depth")
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16):
                with self.subTest(kind=kind, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), full, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
