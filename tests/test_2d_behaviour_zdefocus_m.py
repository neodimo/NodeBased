"""2D behaviour, lane 9 step M: ZDefocus gets Nuke's main tab (focal point, depth of field, blur inside, fill foreground, layers, outputs).

* focal point: a point inside the picture sets the focus plane to the depth read there, so the pixel at the point stays sharp;
* depth of field and blur inside: a band around the focus plane stays sharp (blur inside off) or nearly so (on); outside it the size and
  maximum set the blur;
* fill foreground: off, a blurred foreground fades into nothing at its edge; on, it keeps its colour;
* layers: more layers bring the result closer to the per-pixel result, auto spacing and the curve change which sizes exist;
* output: focal plane setup and layer setup colour the picture red / green / blue in front of / inside / behind the depth of field, show
  image lays them over the picture, filter shape setup draws the filter; tiles equal the full frame; a saved node keeps its picture.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, knobs, zdefocusknobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import differs, evaluator_pixels, old_document, tile_pixels, Graph


def zd(pixels, depth, **knobs_):
    params = {**SPECS["ZDefocus"]["params"], **knobs_}
    return np.asarray(Evaluator._zdefocus(np.asarray(pixels, np.float32), np.asarray(depth, np.float32), params), np.float64)


def texture(h=12, w=16, seed=3):
    pixels = np.random.default_rng(seed).random((h, w, 4)).astype(np.float32)
    pixels[..., 3] = 1.0
    return pixels


def ramp(h=12, w=16, lo=0.2, hi=4.0):
    return np.tile(np.linspace(lo, hi, w, dtype=np.float32), (h, 1))


def impulse(n=31):
    picture = np.zeros((n, n, 4), np.float32)
    picture[n // 2, n // 2] = 1.0
    return picture


def extent(plane, threshold=1e-9):
    ys, xs = np.nonzero(plane > threshold)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


BASE = dict(zdefocus_math="depth", focal_plane=2.0, depth_of_field=0.5, size=6.0, max_size=5.0)


class FocalPointTests(unittest.TestCase):
    def test_the_pixel_at_the_focal_point_stays_sharp(self):
        picture, depth = texture(), ramp()
        for fx, fy in ((12.0, 5.0), (3.0, 8.0)):
            with self.subTest(point=(fx, fy)):
                out = zd(picture, depth, **BASE, focal_x=fx, focal_y=fy)
                np.testing.assert_array_equal(out[int(fy), int(fx)], picture[int(fy), int(fx)])
                self.assertTrue(differs(out, zd(picture, depth, **BASE)))

    def test_the_point_takes_the_place_of_the_focus_plane(self):
        picture, depth = texture(), ramp()
        a = zd(picture, depth, **{**BASE, "focal_plane": 0.5}, focal_x=12.0, focal_y=5.0)
        b = zd(picture, depth, **{**BASE, "focal_plane": 3.5}, focal_x=12.0, focal_y=5.0)
        np.testing.assert_array_equal(a, b)

    def test_the_default_point_and_a_point_outside_the_picture_leave_the_focus_plane(self):
        picture, depth = texture(), ramp()
        reference = zd(picture, depth, **BASE)
        np.testing.assert_array_equal(zd(picture, depth, **BASE, focal_x=200.0, focal_y=200.0), reference)
        np.testing.assert_array_equal(zd(picture, depth, **BASE, focal_x=-4.0, focal_y=3.0), reference)


class DepthOfFieldTests(unittest.TestCase):
    def band(self, **knobs_):
        return zd(texture(), ramp(), **{**BASE, "depth_of_field": 1.5, **knobs_})

    def test_without_blur_inside_the_band_is_sharp(self):
        out, depth, picture = self.band(blur_dof=0), ramp(), texture()
        inside = np.abs(depth - 2.0) <= 0.75
        self.assertTrue(inside.any())
        np.testing.assert_array_equal(out[inside], picture[inside])
        self.assertTrue(differs(out[~inside], picture[~inside]))

    def test_blur_inside_softens_the_band_a_little_and_leaves_the_rest(self):
        on, off = self.band(blur_dof=1), self.band(blur_dof=0)
        inside = np.abs(ramp() - 2.0) <= 0.75
        self.assertTrue(differs(on[inside], off[inside]))
        np.testing.assert_array_equal(on[~inside], off[~inside])
        np.testing.assert_array_equal(zd(texture(), np.full((12, 16), 2.0), **BASE, blur_dof=1), texture())   # the focus plane itself stays sharp

    def test_a_depth_of_field_of_zero_has_no_band(self):
        np.testing.assert_array_equal(zd(texture(), ramp(), **{**BASE, "depth_of_field": 0.0, "blur_dof": 1}),
                                      zd(texture(), ramp(), **{**BASE, "depth_of_field": 0.0, "blur_dof": 0}))

    def test_outside_the_band_the_size_and_the_maximum_set_the_blur(self):
        flat = np.full((31, 31), 6.0)
        base = {**BASE, "focal_plane": 2.0, "depth_of_field": 1.0, "blur_dof": 0}
        sizes = {s: extent(zd(impulse(), flat, **{**base, "size": s, "max_size": 50.0})[..., 0])[0] for s in (2.0, 6.0, 12.0)}
        self.assertLess(sizes[2.0], sizes[6.0])
        self.assertLess(sizes[6.0], sizes[12.0])
        self.assertEqual(extent(zd(impulse(), flat, **{**base, "size": 100.0, "max_size": 3.0})[..., 0]), (7, 7))      # clipped at maximum
        # blur at infinite depth is `size`: far away the radius tends to it, and nearer than the plane it can exceed it
        far = extent(zd(impulse(), np.full((31, 31), 1e5), **{**base, "size": 5.0, "max_size": 50.0})[..., 0])[0]
        self.assertEqual(far, 11)
        near = extent(zd(impulse(), np.full((31, 31), 0.5), **{**base, "size": 5.0, "max_size": 50.0})[..., 0])[0]
        self.assertGreater(near, far)

    def test_the_far_blur_grows_with_distance_from_the_focus_plane(self):
        widths = [extent(zd(impulse(), np.full((31, 31), d), **{**BASE, "blur_dof": 0, "max_size": 50.0})[..., 0])[0] for d in (3.0, 5.0, 12.0)]
        self.assertEqual(widths, sorted(widths))
        self.assertLess(widths[0], widths[-1])


class FillForegroundTests(unittest.TestCase):
    def plate(self):
        picture = np.ones((12, 16, 4), np.float32)
        depth = np.full((12, 16), 3.0, np.float32)
        depth[:, :8] = 0.5                                         # the left half is in front of the focus plane
        return picture, depth

    def test_off_a_blurred_foreground_fades_at_its_edge_and_on_it_keeps_its_colour(self):
        picture, depth = self.plate()
        knob = {**BASE, "focal_plane": 2.0, "max_size": 3.0, "size": 6.0}
        on, off = zd(picture, depth, **knob, fill_foreground=1), zd(picture, depth, **knob, fill_foreground=0)
        np.testing.assert_allclose(on, picture, atol=1e-6)
        self.assertLess(off[5, 7, 0], 1.0 - 1e-3)                  # the foreground's last column
        self.assertLess(off[5, 7, 3], 1.0 - 1e-3)                  # and its alpha
        np.testing.assert_allclose(off[:, :3], 1.0, atol=1e-6)     # away from the edge nothing changes
        np.testing.assert_allclose(off[:, 8:], 1.0, atol=1e-6)     # the background is not a foreground

    def test_a_flat_depth_has_nothing_to_fill(self):
        picture = texture()
        flat = np.full((12, 16), 0.5)
        knob = {**BASE, "focal_plane": 2.0, "max_size": 3.0}
        np.testing.assert_array_equal(zd(picture, flat, **knob, fill_foreground=1), zd(picture, flat, **knob, fill_foreground=0))


class LayerTests(unittest.TestCase):
    KNOB = {**BASE, "max_size": 6.0, "size": 8.0, "blur_dof": 0, "autoLayerSpacing": 0, "layerCurve": 0.0}

    def error(self, layers):
        picture, depth = texture(), ramp(lo=0.3, hi=6.0)
        reference = zd(picture, depth, **self.KNOB, nLayers=256)
        return float(np.abs(zd(picture, depth, **self.KNOB, nLayers=layers) - reference).mean())

    def test_more_layers_come_closer_to_the_per_pixel_result(self):
        errors = [self.error(n) for n in (2, 4, 16, 64)]
        self.assertGreater(errors[0], errors[2])
        self.assertGreater(errors[1], errors[3])
        for earlier, later in zip(errors, errors[1:]):
            self.assertGreaterEqual(earlier + 1e-9, later)
        self.assertLess(errors[-1], 0.02)                          # stated tolerance of 64 layers against the per-pixel result

    def test_layer_sizes_follow_the_curve(self):
        flat = zdefocusknobs.layer_sizes({**self.KNOB, "nLayers": 5})
        np.testing.assert_allclose(flat, [0, 1.5, 3, 4.5, 6])
        near = zdefocusknobs.layer_sizes({**self.KNOB, "nLayers": 5, "layerCurve": 1.0})      # concentrated towards the focus
        far = zdefocusknobs.layer_sizes({**self.KNOB, "nLayers": 5, "layerCurve": -1.0})      # concentrated towards the maximum
        self.assertLess(near[1], flat[1])
        self.assertGreater(far[1], flat[1])
        self.assertEqual(len(zdefocusknobs.layer_sizes({**self.KNOB, "nLayers": 1000})), 256)

    def test_the_curve_and_the_number_act_on_the_picture(self):
        picture, depth = texture(), ramp(lo=0.3, hi=6.0)
        base = zd(picture, depth, **self.KNOB, nLayers=4)
        self.assertTrue(differs(base, zd(picture, depth, **self.KNOB, nLayers=2)))
        self.assertTrue(differs(base, zd(picture, depth, **{**self.KNOB, "layerCurve": 2.0}, nLayers=4)))

    def test_automatic_spacing_ignores_the_manual_controls(self):
        picture, depth = texture(), ramp(lo=0.3, hi=6.0)
        auto = {**self.KNOB, "autoLayerSpacing": 1}
        np.testing.assert_array_equal(zd(picture, depth, **{**auto, "nLayers": 2}), zd(picture, depth, **{**auto, "nLayers": 40, "layerCurve": 3.0}))
        self.assertTrue(differs(zd(picture, depth, **auto), zd(picture, depth, **{**self.KNOB, "nLayers": 2})))
        sizes = zdefocusknobs.layer_sizes({**auto, "max_size": 8.0})
        self.assertEqual(sizes[0], 0.0)
        self.assertEqual(sizes[-1], 8.0)
        self.assertLess(sizes[1] - sizes[0], sizes[-1] - sizes[-2])        # closer together near the focus plane

    def test_a_whole_pixel_blur_is_its_own_layer(self):
        flat = np.full((31, 31), 0.4)                               # direct: blur = 0.4 * 10 = 4
        self.assertEqual(extent(zd(impulse(), flat, zdefocus_math="direct", size=10.0, max_size=50.0)[..., 0]), (9, 9))


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.picture, self.depth = texture(), ramp(lo=0.2, hi=4.0)
        self.knob = {**BASE, "focal_plane": 2.0, "depth_of_field": 1.0}
        self.front, self.inside = self.depth < 1.5, np.abs(self.depth - 2.0) <= 0.5

    def test_focal_plane_setup_colours_front_inside_and_behind(self):
        out = zd(self.picture, self.depth, **self.knob, zdefocus_output="focal plane setup", show_image=0)
        behind = self.depth > 2.5
        self.assertTrue(self.front.any() and self.inside.any() and behind.any())
        np.testing.assert_array_equal(out[self.front][:, :3], np.tile([1, 0, 0], (int(self.front.sum()), 1)))
        np.testing.assert_array_equal(out[self.inside][:, :3], np.tile([0, 1, 0], (int(self.inside.sum()), 1)))
        np.testing.assert_array_equal(out[behind][:, :3], np.tile([0, 0, 1], (int(behind.sum()), 1)))
        np.testing.assert_array_equal(out[..., 3], 1.0)

    def test_with_a_depth_of_field_of_zero_nothing_is_green(self):
        out = zd(self.picture, self.depth, **{**self.knob, "depth_of_field": 0.0}, zdefocus_output="focal plane setup", show_image=0)
        self.assertEqual(float(out[..., 1][self.depth < 2.0].max()), 0.0)
        self.assertEqual(int(((out[..., 1] == 1) & (out[..., 0] == 0) & (out[..., 2] == 0)).sum()), 0)

    def test_show_image_lays_the_colours_over_the_picture(self):
        bare = zd(self.picture, self.depth, **self.knob, zdefocus_output="focal plane setup", show_image=0)
        over = zd(self.picture, self.depth, **self.knob, zdefocus_output="focal plane setup", show_image=1)
        np.testing.assert_allclose(over[..., :3], 0.5 * self.picture[..., :3] + 0.5 * bare[..., :3], atol=1e-6)
        np.testing.assert_array_equal(over[..., 3], self.picture[..., 3])

    def test_layer_setup_gives_pixels_of_one_layer_one_colour(self):
        knob = {**self.knob, "max_size": 6.0, "size": 8.0, "autoLayerSpacing": 0, "nLayers": 4, "layerCurve": 0.0}
        out = zd(self.picture, self.depth, **knob, zdefocus_output="layer setup", show_image=0)
        radii, layer = zdefocusknobs.snap_to_layers(zdefocusknobs.blur_sizes(self.depth, 2.0, 0.5, knob), knob)
        for index in np.unique(layer):
            colours = np.unique(out[layer == index][:, :3], axis=0)
            sides = np.unique(zdefocusknobs.side(self.depth, 2.0, 0.5)[layer == index])
            self.assertLessEqual(len(colours), len(sides))
        self.assertGreaterEqual(len(np.unique(out[..., :3].reshape(-1, 3), axis=0)), 3)
        self.assertTrue(differs(out, zd(self.picture, self.depth, **knob, zdefocus_output="focal plane setup", show_image=0)))

    def test_filter_shape_setup_draws_the_filter(self):
        out = zd(np.zeros((31, 31, 4), np.float32), np.full((31, 31), 3.0), **{**BASE, "max_size": 5.0}, zdefocus_output="filter shape setup")
        white = out[..., 0] == 1
        yy, xx = np.mgrid[-15:16, -15:16]
        np.testing.assert_array_equal(white, xx * xx + yy * yy <= 25)
        np.testing.assert_array_equal(out[..., 3], 1.0)
        blades = zd(np.zeros((31, 31, 4), np.float32), np.full((31, 31), 3.0), **{**BASE, "max_size": 5.0, "bokeh_shape": "blades", "blade_count": 3},
                    zdefocus_output="filter shape setup")
        self.assertTrue(differs(blades, out))

    def test_result_is_the_default_output(self):
        np.testing.assert_array_equal(zd(self.picture, self.depth, **self.knob), zd(self.picture, self.depth, **self.knob, zdefocus_output="result"))
        self.assertEqual(core.CHOICES["zdefocus_output"][0], "result")

    def test_every_new_knob_changes_pixels(self):
        picture, depth = texture(), ramp(lo=0.3, hi=6.0)
        base = {**BASE, "depth_of_field": 1.5, "max_size": 6.0, "size": 8.0, "autoLayerSpacing": 0, "nLayers": 6}
        reference = zd(picture, depth, **base)
        for name, value in (("zdefocus_output", "focal plane setup"), ("autoLayerSpacing", 1), ("nLayers", 3), ("layerCurve", 0.0)):
            with self.subTest(knob=name):
                self.assertTrue(differs(reference, zd(picture, depth, **{**base, name: value})), name)
        with self.subTest(knob="focal point"):
            self.assertTrue(differs(reference, zd(picture, depth, **base, focal_x=12.0, focal_y=4.0)))
        with self.subTest(knob="blur_dof"):
            fine = {**base, "autoLayerSpacing": 1}
            self.assertTrue(differs(zd(picture, depth, **fine), zd(picture, depth, **{**fine, "blur_dof": 0})))
        # show image acts with a setup output; fill foreground with a foreground that has an edge
        setup = {**base, "zdefocus_output": "focal plane setup"}
        self.assertTrue(differs(zd(picture, depth, **setup, show_image=1), zd(picture, depth, **setup, show_image=0)))
        plate = FillForegroundTests().plate()
        knob = {**BASE, "max_size": 3.0}
        self.assertTrue(differs(zd(*plate, **knob, fill_foreground=1), zd(*plate, **knob, fill_foreground=0)))


class FilterTests(unittest.TestCase):
    """The disc and bladed filters' own knobs, the catadioptric hole and the bloom and gamma controls."""
    FLAT = np.full((41, 41), 6.0)
    KNOB = {**BASE, "focal_plane": 2.0, "blur_dof": 0, "max_size": 8.0, "size": 8.0, "depth_of_field": 1.0}

    def blur(self, picture=None, **knobs_):
        return zd(impulse(41) if picture is None else picture, self.FLAT, **{**self.KNOB, **knobs_})

    def test_shape_dissolves_a_gaussian_into_the_disc(self):
        disc, soft, gauss = (self.blur(shape=v)[..., 0] for v in (1.0, 0.5, 0.0))
        self.assertEqual(extent(disc), extent(gauss))
        self.assertGreater(disc[20, 20 + 3], 0.0)
        self.assertTrue(differs(disc, soft) and differs(soft, gauss))
        # the middle of a Gaussian filter weighs more than its edge; the flat disc weighs them alike
        self.assertAlmostEqual(float(disc[20, 20 + 2] / disc[20, 20]), 1.0, places=6)
        self.assertGreater(float(gauss[20, 20] / gauss[20, 20 + 3]), 1.5)

    def test_aspect_squeezes_x_below_one_and_y_above_one(self):
        wide, narrow = self.blur(aspect=0.5)[..., 0], self.blur(aspect=2.0)[..., 0]
        square = extent(self.blur()[..., 0])
        self.assertEqual(extent(wide)[1], square[1])
        self.assertLess(extent(wide)[0], square[0])
        self.assertEqual(extent(narrow)[0], square[0])
        self.assertLess(extent(narrow)[1], square[1])

    def test_a_catadioptric_filter_is_hollow(self):
        flat, ring = self.blur(catadioptric=0)[..., 0], self.blur(catadioptric=1, catadioptric_size=0.5)[..., 0]
        self.assertGreater(float(flat[20, 20]), 0.0)
        self.assertEqual(float(ring[20, 20]), 0.0)                                  # the impulse's own pixel lies in the hole
        self.assertTrue(differs(ring, self.blur(catadioptric=1, catadioptric_size=0.2)[..., 0]))
        np.testing.assert_array_equal(self.blur(catadioptric_size=0.9), self.blur())   # the size acts only while catadioptric is on

    def test_bloom_boosts_highlights_over_the_threshold_before_the_blur(self):
        picture = impulse(41) * 0.9
        plain, boosted = self.blur(picture), self.blur(picture, bloom=1, bloom_threshold=0.8, bloom_gain=2.0)
        np.testing.assert_allclose(boosted[..., :3], plain[..., :3] * 2.0, rtol=1e-5, atol=1e-7)     # colour only: alpha is not a highlight
        np.testing.assert_array_equal(boosted[..., 3], plain[..., 3])
        below = self.blur(picture, bloom=1, bloom_threshold=0.95, bloom_gain=2.0)
        np.testing.assert_allclose(below, plain, atol=1e-7)
        self.assertTrue(differs(boosted, self.blur(picture, bloom=1, bloom_threshold=0.8, bloom_gain=3.0)))

    def test_gamma_correction_blurs_in_a_2_2_encoding(self):
        picture = np.zeros((41, 41, 4), np.float32)
        picture[:, 20:, :3] = 1.0
        picture[..., 3] = 1.0
        plain, gamma = self.blur(picture), self.blur(picture, bloom_gamma=1)
        # the edge pixel is half bright blurred as it is; blurred in the gamma encoding it is 0.5 ^ 2.2 darker
        self.assertTrue(differs(plain, gamma))
        self.assertLess(float(gamma[20, 19, 0]), float(plain[20, 19, 0]))
        np.testing.assert_allclose(self.blur(picture * 0 + 1.0, bloom_gamma=1), 1.0, atol=1e-5)

    def test_the_filter_knobs_leave_a_saved_node_alone(self):
        picture = impulse(41)
        reference = zd(picture, self.FLAT, **{**self.KNOB, "legacy_zdefocus_focus": 1})
        for name, value in (("shape", 0.0), ("aspect", 2.0), ("catadioptric", 1), ("bloom", 1), ("bloom_gamma", 1)):
            np.testing.assert_array_equal(zd(picture, self.FLAT, **{**self.KNOB, "legacy_zdefocus_focus": 1, name: value}), reference)

    def test_tiles_equal_the_full_frame_with_the_filter_knobs(self):
        extra = dict(shape=0.3, aspect=0.5, catadioptric=1, bloom=1, bloom_gamma=1)
        full = evaluator_pixels(TileAndLayoutTests().doc(False, **extra), "node")
        for edge in (7, 16, 32):
            np.testing.assert_allclose(tile_pixels(TileAndLayoutTests().doc(False, **extra), "node", edge), full, atol=1e-6)


class OlderDocumentTests(unittest.TestCase):
    NEW = ("legacy_zdefocus_focus", "zdefocus_output", "show_image", "fill_foreground", "focal_x", "focal_y", "blur_dof", "autoLayerSpacing",
           "nLayers", "layerCurve",
           "shape", "aspect", "catadioptric", "catadioptric_size", "bloom", "bloom_threshold", "bloom_gain", "bloom_gamma")

    def test_a_saved_node_keeps_the_step_c_picture(self):
        old = old_document("ZDefocus", dict(focal_plane=1.0, depth_of_field=1.0, max_size=6.0, legacy_zdefocus=0, zdefocus_math="depth", size=25.0),
                           self.NEW)["nodes"]["n"]["params"]
        self.assertEqual(old["legacy_zdefocus_focus"], 1)
        self.assertEqual(SPECS["ZDefocus"]["params"]["legacy_zdefocus_focus"], 0)
        for z, want in ((3.0, 6), (1.5, 3)):                      # min(max, |z - focus| / dof * max): the step C reading
            out = zd(impulse(), np.full((31, 31), z), **{k: old[k] for k in old if k not in ("channels", "mix")})
            self.assertEqual(extent(out[..., 0]), (2 * want + 1, 2 * want + 1))

    def test_the_new_knobs_do_nothing_to_a_saved_node(self):
        picture, depth = texture(), ramp(lo=0.3, hi=6.0)
        base = {**BASE, "legacy_zdefocus_focus": 1, "max_size": 6.0}
        reference = zd(picture, depth, **base)
        for name, value in (("zdefocus_output", "focal plane setup"), ("fill_foreground", 0), ("focal_x", 12.0), ("blur_dof", 0), ("autoLayerSpacing", 0),
                            ("nLayers", 2), ("layerCurve", 3.0)):
            np.testing.assert_array_equal(zd(picture, depth, **base, **{name: value}, focal_y=4.0), reference)

    def test_a_node_older_than_the_page_gets_the_legacy_flags(self):
        old = old_document("ZDefocus", dict(focal_plane=1.0, depth_of_field=1.0, max_size=6.0, depth_math="1/depth"),
                           ("legacy_zdefocus", "zdefocus_math", "size") + self.NEW)["nodes"]["n"]["params"]
        self.assertEqual((old["legacy_zdefocus"], old["legacy_zdefocus_focus"]), (1, 1))


class TileAndLayoutTests(unittest.TestCase):
    def doc(self, masked, **knobs_):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("depth", "Ramp", dict(width=45, height=37, p0_x=0, p0_y=0, p1_x=44, p1_y=0) if "p0_x" in SPECS["Ramp"]["params"] else dict(width=45, height=37))
        inputs = {"image": "plate", "depth": "depth"}
        if masked:
            g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
            inputs["mask"] = "matte"
        params = {"focal_plane": 0.5, "depth_of_field": 0.2, "size": 4.0, "max_size": 3.0, "autoLayerSpacing": 0, "nLayers": 5, **knobs_}
        g.add("node", "ZDefocus", params, **inputs)
        return g.doc

    def test_tiles_equal_the_full_frame(self):
        for masked in (False, True):
            for extra in ({}, {"fill_foreground": 0}, {"zdefocus_output": "layer setup"}, {"zdefocus_output": "focal plane setup", "show_image": 0}):
                full = evaluator_pixels(self.doc(masked, **extra), "node")
                for edge in (7, 16, 32):
                    with self.subTest(masked=masked, extra=extra, edge=edge):
                        np.testing.assert_allclose(tile_pixels(self.doc(masked, **extra), "node", edge), full, atol=1e-6)

    def test_a_focal_point_and_the_filter_shape_view_run_on_the_full_frame(self):
        for extra in ({"focal_x": 10.0, "focal_y": 5.0}, {"zdefocus_output": "filter shape setup"}):
            params = {**SPECS["ZDefocus"]["params"], **extra}
            self.assertTrue(zdefocusknobs.needs_full_frame(params), extra)
        self.assertFalse(zdefocusknobs.needs_full_frame(SPECS["ZDefocus"]["params"]))
        self.assertFalse(zdefocusknobs.needs_full_frame({**SPECS["ZDefocus"]["params"], "legacy_zdefocus_focus": 1, "focal_x": 10.0}))

    def test_the_main_tab_follows_the_guide_and_every_param_is_laid_out_once(self):
        shown = [g.label for g in knobs.KNOB_LAYOUT["ZDefocus"] if not g.params[0].startswith("legacy_") and g.params != ("mix",) and g.kind != "legacy"]
        self.assertEqual(shown[:15], ["Channels", "Depth channel", "Math", "Output", "Show image", "Fill foreground", "Focus plane (C)", "Focal point xy",
                                      "Depth of field", "Blur inside", "Size", "Maximum", "Automatic layer spacing", "Depth layers", "Layer curve"])
        params = [name for g in knobs.KNOB_LAYOUT["ZDefocus"] for name in g.params]
        for name in self.NEW_PARAMS:
            self.assertEqual(params.count(name), 1, name)
        for name in self.NEW_PARAMS:
            self.assertIn(name, SPECS["ZDefocus"]["params"])
            self.assertTrue(name in core.LIMITS or name in core.CHOICES, name)

    NEW_PARAMS = ("zdefocus_output", "show_image", "fill_foreground", "focal_x", "focal_y", "blur_dof", "autoLayerSpacing", "nLayers", "layerCurve",
                  "shape", "aspect", "catadioptric", "catadioptric_size", "bloom", "bloom_threshold", "bloom_gain", "bloom_gamma")


if __name__ == "__main__":
    unittest.main()
