"""2D behaviour, lane 9 step K: the thirteen knobs that moved nothing, and Nuke's labels and order on the filter and transform panels.

* LightWrap: Disable luminance based wrap, Enable Glow, LumaTolerance, white and midtone bias act; graphs saved before keep the older pixels
  (legacy_lightwrap_luma).
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_behaviour_filter_f import disc_over_bright, wrap_of
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, kern, texture, tile_pixels


def light_wrap(fg, bg, **params):
    q = {**SPECS["LightWrap"]["params"], **params}
    return np.asarray(Evaluator._kernel("LightWrap", q, [fg, bg, None]), np.float64)


class LightWrapKnobs(unittest.TestCase):
    BASE = dict(intensity=1.0, wrap_diffuse=4.0, fgblur=1.0, bgblur=0.0)

    def test_disable_luminance_based_wrap_makes_the_wrap_uniform_whatever_the_background(self):
        fg, dim, alpha = disc_over_bright(bg=(0.2, 0.12, 0.04))
        _, bright, _ = disc_over_bright(bg=(1.0, 0.6, 0.2))
        rim = (alpha > 0)
        a, b = wrap_of(fg, dim, **self.BASE)[..., 0], wrap_of(fg, bright, **self.BASE)[..., 0]
        self.assertGreater(float(b[rim].max()) / float(a[rim].max()), 4.0)       # luminance based: a brighter background wraps more
        a, b = (wrap_of(fg, bg, disableluma=1, **self.BASE)[..., 0] for bg in (dim, bright))
        self.assertAlmostEqual(float(b[rim].max()) / float(a[rim].max()), 1.0, places=3)   # disabled: the same wrap

    def test_enable_glow_extends_the_wrap_outside_the_matte(self):
        fg, bg, alpha = disc_over_bright()
        plain = wrap_of(fg, bg, **self.BASE)[..., 0]
        glow = wrap_of(fg, bg, enableglow=1, **self.BASE)[..., 0]
        self.assertEqual(float(plain[alpha == 0].max()), 0.0)
        self.assertGreater(float(glow[alpha == 0].max()), 0.2)
        np.testing.assert_allclose(glow[alpha > 0], plain[alpha > 0], atol=1e-6)      # inside the matte nothing changes

    def test_luma_tolerance_fades_a_dark_background_out_of_the_wrap(self):
        fg, bg, alpha = disc_over_bright(bg=(0.2, 0.12, 0.04))
        base = wrap_of(fg, bg, **self.BASE)[..., 0].max()
        tol = wrap_of(fg, bg, LumaTolerance=1.0, **self.BASE)[..., 0].max()
        self.assertLess(float(tol), float(base) * 0.5)
        _, bright, _ = disc_over_bright(bg=(1.0, 1.0, 1.0))
        np.testing.assert_allclose(wrap_of(fg, bright, LumaTolerance=1.0, **self.BASE), wrap_of(fg, bright, **self.BASE), atol=1e-6)

    def test_gray_is_the_contrast_pivot_and_white_the_gamma_fixed_point(self):
        fg, bg, alpha = disc_over_bright(bg=(0.5, 0.5, 0.5))
        flat = wrap_of(fg, bg, **self.BASE)
        edge = flat[..., 0] > 0.05
        # a value equal to the pivot is unchanged by contrast: with gray at that value, contrast leaves the wrap alone there
        pinned = wrap_of(fg, bg, contrast=3.0, gray=0.5, **self.BASE)[..., 0]
        free = wrap_of(fg, bg, contrast=3.0, **self.BASE)[..., 0]
        peak = np.unravel_index(np.argmax(flat[..., 0]), flat[..., 0].shape)
        self.assertAlmostEqual(float(pinned[peak]), float(flat[..., 0][peak]), places=4)
        self.assertGreater(abs(float(free[peak]) - float(flat[..., 0][peak])), 0.05)
        # a value equal to white is unchanged by gamma
        held = wrap_of(fg, bg, gamma=2.0, white=0.5, **self.BASE)[..., 0]
        moved = wrap_of(fg, bg, gamma=2.0, **self.BASE)[..., 0]
        self.assertAlmostEqual(float(held[peak]), float(flat[..., 0][peak]), places=4)
        self.assertGreater(abs(float(moved[peak]) - float(flat[..., 0][peak])), 0.05)

    def test_each_knob_changes_the_pixels_of_a_new_node(self):
        fg, bg, alpha = disc_over_bright()
        base = light_wrap(fg, bg, **self.BASE)
        for knob, value in (("disableluma", 1), ("enableglow", 1), ("LumaTolerance", 2.0)):
            self.assertTrue(differs(base, light_wrap(fg, bg, **{**self.BASE, knob: value})), knob)
        self.assertTrue(differs(light_wrap(fg, bg, contrast=2.0, **self.BASE), light_wrap(fg, bg, contrast=2.0, gray=0.5, **self.BASE)))
        self.assertTrue(differs(light_wrap(fg, bg, gamma=2.0, **self.BASE), light_wrap(fg, bg, gamma=2.0, white=0.5, **self.BASE)))

    def test_graphs_saved_before_keep_their_pixels(self):
        fg, bg, alpha = disc_over_bright()
        saved = {k: v for k, v in SPECS["LightWrap"]["params"].items() if k != "legacy_lightwrap_luma"}
        old = light_wrap(fg, bg, **{**saved, "legacy_lightwrap_luma": 1, **self.BASE})
        for knob, value in (("disableluma", 1), ("enableglow", 1), ("LumaTolerance", 2.0), ("white", 0.5), ("gray", 0.5)):
            np.testing.assert_array_equal(light_wrap(fg, bg, **{**saved, "legacy_lightwrap_luma": 1, **self.BASE, knob: value}), old)
        g = Graph()
        g.add("a", "Checker", dict(width=45, height=37, size=5))
        g.add("b", "Checker", dict(width=45, height=37, size=7))
        g.add("n", "LightWrap", dict(enableglow=1), fg="a", bg="b")
        doc = copy.deepcopy(g.doc)
        doc["nodes"]["n"]["params"].pop("legacy_lightwrap_luma")
        loaded = upgrade_document(doc)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_lightwrap_luma"], 1)
        self.assertEqual(SPECS["LightWrap"]["params"]["legacy_lightwrap_luma"], 0)

    def test_the_tile_path_equals_the_evaluator_with_the_new_knobs(self):
        for masked in (False, True):
            g = Graph()
            g.add("fgplate", "Rectangle", dict(width=45, height=37, box_x=8.0, box_y=7.0, box_width=26.0, box_height=20.0, red=0.3, green=0.3, blue=0.3, alpha=1.0))
            g.add("bgplate", "Constant", dict(width=45, height=37, red=1.0, green=0.6, blue=0.2, alpha=1.0))
            inputs = {"fg": "fgplate", "bg": "bgplate"}
            if masked:
                g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
                inputs["mask"] = "matte"
            g.add("node", "LightWrap", dict(intensity=1.5, wrap_diffuse=5.0, fgblur=1.0, bgblur=2.0, enableglow=1, LumaTolerance=0.9,
                                            contrast=1.5, gamma=1.4, gray=0.4, white=0.8), **inputs)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(edge=edge, masked=masked):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)

run = kern


class QualityKnobs(unittest.TestCase):
    """Nuke's quality knobs on Soften, Laplacian and EdgeDetect scale the picture down once the filter is wider than them (the same reading
    Blur and Sharpen have): at the default size they change nothing, past it they do."""

    def test_the_quality_limit_acts_once_the_size_passes_it(self):
        pixels = texture()
        for kind, size_knob, quality_knob in (("Soften", "soften_size", "quality"), ("Laplacian", "laplacian_size", "quality"),
                                              ("EdgeDetect", "edge_blur_size", "edge_blur_quality")):
            with self.subTest(kind=kind):
                wide = {size_knob: 60.0}
                tight = run(kind, pixels, **wide, **{quality_knob: 5.0})
                loose = run(kind, pixels, **wide, **{quality_knob: 100.0})
                self.assertTrue(differs(tight, loose), kind)
                small = {size_knob: 2.0}
                np.testing.assert_array_equal(run(kind, pixels, **small, **{quality_knob: 5.0}), run(kind, pixels, **small, **{quality_knob: 100.0}))

    def test_the_tile_path_equals_the_evaluator_at_a_tight_quality(self):
        for kind, params in (("Soften", dict(soften_size=40.0, quality=4.0)), ("Laplacian", dict(laplacian_size=40.0, quality=4.0)),
                             ("EdgeDetect", dict(edge_blur_size=40.0, edge_blur_quality=4.0))):
            g = Graph()
            g.add("a", "Checker", dict(width=45, height=37, size=5))
            g.add("node", kind, params, image="a")
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(kind=kind, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)


class DirBlurKnobs(unittest.TestCase):
    """DirBlur's Quality, PixelAspect and Holdout Channel act on a node made now; a graph saved before keeps its pixels (legacy_dirblur_knobs)."""

    def test_quality_scales_the_blur_down(self):
        pixels = texture()
        high = kern("DirBlur", pixels, blur_type="linear", length=11.0, angle=20.0, Quality="high")
        normal = kern("DirBlur", pixels, blur_type="linear", length=11.0, angle=20.0, Quality="normal")
        low = kern("DirBlur", pixels, blur_type="linear", length=11.0, angle=20.0, Quality="low")
        self.assertTrue(differs(high, normal))
        self.assertTrue(differs(normal, low))
        # the scaled blur is the same blur, smoother: a flat picture stays flat and the mean holds
        flat = np.full((16, 20, 4), 0.4, np.float32)
        np.testing.assert_allclose(kern("DirBlur", flat, length=9.0, Quality="low"), flat, atol=1e-6)

    def test_pixel_aspect_changes_the_blurs_reach_across(self):
        pixels = texture()
        square = kern("DirBlur", pixels, blur_type="linear", length=9.0, angle=0.0)
        wide = kern("DirBlur", pixels, blur_type="linear", length=9.0, angle=0.0, PixelAspect=2.0)
        self.assertTrue(differs(square, wide))
        impulse = np.zeros((9, 41, 4), np.float32)
        impulse[4, 20] = 1.0
        spread = lambda a: int((kern("DirBlur", impulse, blur_type="linear", length=10.0, angle=0.0, PixelAspect=a)[4, :, 0] > 1e-6).sum())
        self.assertLess(spread(2.0), spread(1.0))                 # a pixel twice as wide: half as many of them under the blur
        radial = kern("DirBlur", pixels, blur_type="radial", angle=20.0, center_x=22.0, center_y=18.0)
        self.assertTrue(differs(radial, kern("DirBlur", pixels, blur_type="radial", angle=20.0, center_x=22.0, center_y=18.0, PixelAspect=2.0)))

    def test_holdout_keeps_the_held_out_pixels_out_of_the_zoom_average(self):
        pixels = np.zeros((21, 21, 4), np.float32)
        pixels[..., :3] = 0.2
        pixels[..., 3] = 0.0
        pixels[10, 14, :3] = 1.0
        pixels[10, 14, 3] = 1.0                                  # a bright pixel with a non-black holdout channel
        zoom = dict(blur_type="zoom", length=40.0, center_x=10.5, center_y=10.5)
        smeared = kern("DirBlur", pixels, **zoom)
        held = kern("DirBlur", pixels, holdout="on", **zoom)
        self.assertTrue(differs(smeared, held))
        self.assertGreater(float(smeared[10, 15, 0]), 0.21)       # without the holdout the bright pixel smears outward onto its neighbour
        np.testing.assert_allclose(held[10, 15, 0], 0.2, atol=1e-6)   # with it the neighbour averages only the dark pixels
        np.testing.assert_allclose(held[10, 14], pixels[10, 14], atol=1e-6)   # a held-out pixel keeps its own value
        linear = dict(blur_type="linear", length=9.0)
        np.testing.assert_array_equal(kern("DirBlur", pixels, holdout="on", **linear), kern("DirBlur", pixels, **linear))   # zoom only

    def test_a_graph_saved_before_keeps_its_pixels(self):
        pixels = texture()
        for knob in (dict(Quality="low"), dict(PixelAspect=2.0), dict(holdout="on", blur_type="zoom", center_x=20.0, center_y=18.0, length=20.0)):
            base = dict(blur_type="linear", length=9.0, legacy_dirblur_knobs=1)
            plain = kern("DirBlur", pixels, **{**base, **{k: v for k, v in knob.items() if k not in ("Quality", "PixelAspect", "holdout")}})
            np.testing.assert_array_equal(kern("DirBlur", pixels, **{**base, **knob}), plain)

    def test_the_tile_path_equals_the_evaluator_and_the_scaled_blurs_read_the_whole_frame(self):
        g = Graph()
        g.add("a", "Checker", dict(width=45, height=37, size=5))
        g.add("node", "DirBlur", dict(blur_type="linear", length=13.0), image="a")
        whole = evaluator_pixels(g.doc, "node")
        for edge in (7, 16, 32):
            with self.subTest(edge=edge):
                np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-5)
        # Quality's blocks and a non-square pixel sit on the whole picture, as zoom and radial do: the node is not tiled (the frame path computes them)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        for params in (dict(Quality="low"), dict(PixelAspect=2.0), dict(blur_type="zoom")):
            h = Graph()
            h.add("a", "Checker", dict(width=45, height=37, size=5))
            h.add("node", "DirBlur", dict(length=13.0, **params), image="a")
            self.assertFalse(executor.supports_tiled(dict(h.doc, view="node"), "node"), params)
            self.assertEqual(evaluator_pixels(h.doc, "node").shape[:2], (37, 45))


if __name__ == "__main__":
    unittest.main()
