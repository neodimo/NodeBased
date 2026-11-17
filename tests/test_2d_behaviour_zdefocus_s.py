"""2D behaviour, lane 9 step S: ZDefocus's bokeh-shape knobs.

The bladed filter (roundness, inner size, inner feather, inner brightness, its own aspect ratio) and the image filter (filter channel, use input
channels, filter bounds, image filter, clamp image filter, legacy resize mode) are tested on a bright point on a plane behind the focus plane: the
blurred point is the filter. The cases in tools/nuke_fidelity_cases_filter.py state the weights; these tests state the properties that need no weights
(direction of a rotation, the clamp, the smoothing and sharpening, the bounds) and that a node saved before the knobs keeps its picture bit for bit.
"""
import hashlib
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, knobs, zdefocusknobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, old_document, tile_pixels

N = 31
MID = N // 2


def point(n=N):
    picture = np.zeros((n, n, 4), np.float32)
    picture[n // 2, n // 2] = 1.0
    return picture


def spread(kernel=None, size=10.0, **knobs_):
    """A point on a plane at depth 3 (focus plane 1), blurred 6.67 pixels -> the layer of 7: the filter as a float64 picture."""
    params = {**SPECS["ZDefocus"]["params"], "shape": 1.0, "zdefocus_math": "depth", "focal_plane": 1.0, "depth_of_field": 0.0, "blur_dof": 0, "size": size,
              "max_size": 7.0, **knobs_}
    return np.asarray(Evaluator._zdefocus(point(), np.full((N, N), 3.0, np.float32), params, kernel), np.float64)


def view(plane):
    """The filter in a picture's terms: rows down, so [dy, dx] from the middle."""
    return plane[..., 0]


def filter_picture(h=12, w=16, block=None, at=(5, 7)):
    out = np.zeros((h, w, 4), np.float32)
    block = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) if block is None else np.asarray(block, np.float32)
    y, x = at
    out[y:y + block.shape[0], x:x + block.shape[1], 3] = block
    return out


class BladedFilterTests(unittest.TestCase):
    BLADED = dict(bokeh_shape="blades", blade_count=5)

    def blur(self, **k):
        return spread(**{**self.BLADED, **k})

    def test_the_new_bladed_defaults_are_the_guides(self):
        params = SPECS["ZDefocus"]["params"]
        self.assertEqual((params["roundness"], params["inner_size"], params["inner_feather"], params["inner_brightness"]),
                         (0.2, 0.8, 1.0, 0.8))

    def test_roundness_rounds_the_sides_between_the_polygon_and_the_disc(self):
        areas = [float((view(self.blur(roundness=r, inner_brightness=1.0)) > 1e-12).sum()) for r in (0.0, 0.4, 0.8, 1.0)]
        self.assertEqual(areas, sorted(areas))                                     # the sides bow out as roundness grows
        self.assertLess(areas[0], areas[-1])
        disc = spread(bokeh_shape="disc")
        np.testing.assert_allclose(self.blur(roundness=1.0, inner_brightness=1.0), disc, atol=1e-7)   # round all the way: the disc

    def test_blades_count_the_corners(self):
        # a flat-sided polygon of n blades fits inside the circle of the blur size by its corners: fewer blades, less area
        areas = [float((view(self.blur(blade_count=n, roundness=0.0, inner_brightness=1.0)) > 0).sum()) for n in (3, 4, 6, 12)]
        self.assertEqual(areas, sorted(areas))

    def test_a_positive_rotation_turns_the_filter_counter_clockwise(self):
        # three blades, roundness 0: a triangle with a flat side to the right (picture's +x) at rotation 0; 90 degrees counter-clockwise puts the
        # flat side at the top of the picture (rows above the middle), since the picture's up is counter-clockwise from the right
        def triangle(rotation):
            plane = view(self.blur(blade_count=3, roundness=0.0, inner_brightness=1.0, blade_rotation=rotation))
            offsets = np.arange(N) - MID
            dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
            return plane, dy, dx
        right, dy, dx = triangle(0.0)
        up, _, _ = triangle(90.0)
        columns = np.nonzero(right.max(axis=0) > 0)[0]
        self.assertGreater(float(right[:, columns.max()].sum()), float(right[:, columns.min()].sum()))      # flat side right: the right column is the longest
        rows = np.nonzero(up.max(axis=1) > 0)[0]
        self.assertGreater(float(up[rows.min(), :].sum()), float(up[rows.max(), :].sum()))                  # flat side up: the top row is the longest
        down = view(self.blur(blade_count=3, roundness=0.0, inner_brightness=1.0, blade_rotation=-90.0))
        rows = np.nonzero(down.max(axis=1) > 0)[0]
        self.assertGreater(float(down[rows.max(), :].sum()), float(down[rows.min(), :].sum()))              # and -90 puts it at the bottom

    def test_the_aspect_ratio_squeezes_the_bladed_filter_like_the_disc(self):
        base = self.blur(roundness=1.0, inner_brightness=1.0)
        wide, tall = view(self.blur(roundness=1.0, inner_brightness=1.0, aspect=0.5)), view(self.blur(roundness=1.0, inner_brightness=1.0, aspect=2.0))
        columns = lambda plane: int((plane.max(axis=0) > 0).sum())
        rows = lambda plane: int((plane.max(axis=1) > 0).sum())
        self.assertLess(columns(wide), columns(view(base)))                          # below 1 squeezes x
        self.assertEqual(rows(wide), rows(view(base)))
        self.assertLess(rows(tall), rows(view(base)))                                # above 1 squeezes y
        self.assertEqual(columns(tall), columns(view(base)))

    def test_each_inner_knob_changes_the_filter(self):
        base = self.blur()
        for name, value in (("roundness", 0.6), ("inner_size", 0.4), ("inner_feather", 0.2), ("inner_brightness", 0.3)):
            with self.subTest(knob=name):
                self.assertTrue(differs(self.blur(**{name: value}), base))

    def test_inner_brightness_one_or_inner_size_one_is_a_flat_filter(self):
        flat = self.blur(inner_brightness=1.0)
        plane = view(flat)
        np.testing.assert_allclose(plane[plane > 0], plane[plane > 0].max(), rtol=1e-6)
        np.testing.assert_allclose(self.blur(inner_size=1.0, inner_feather=0.5), flat, atol=1e-7)

    def test_the_inner_feather_moves_the_ramp_outward_above_half_and_inward_below(self):
        black = dict(inner_size=0.5, inner_brightness=0.0, roundness=1.0)
        outward, inward = view(self.blur(inner_feather=1.0, **black)), view(self.blur(inner_feather=0.0, **black))
        hard = view(self.blur(inner_feather=0.5, **black))
        edge = 7 * 0.5                                                                  # the inner polygon's edge, in pixels
        for row_col, plane in (("x", outward), ("x", inward), ("x", hard)):
            self.assertGreater(plane.sum(), 0)
        # just inside the edge: outward leaves it black like the hard edge, inward lifts it; just outside: outward is dimmer than the hard edge
        inside, outside = MID + int(edge) - 1, MID + int(math.ceil(edge)) + 1
        self.assertEqual(float(outward[MID, inside]), float(hard[MID, inside]))
        self.assertGreater(float(inward[MID, inside]), float(hard[MID, inside]))
        self.assertLess(float(outward[MID, outside]), float(hard[MID, outside]))
        rim = MID + 6                                                                   # beyond the edge both a hard and an inward edge weigh 1
        self.assertAlmostEqual(float(inward[MID, outside] / inward[MID, rim]), float(hard[MID, outside] / hard[MID, rim]), places=6)


RING = filter_picture(block=[[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])


class ImageFilterTests(unittest.TestCase):
    def blur(self, kernel=None, **k):
        kernel = filter_picture() if kernel is None else kernel
        return spread(kernel, bokeh_shape="image", **k)

    def test_a_bright_point_takes_the_shape_of_the_filter_as_drawn(self):
        # an asymmetric filter: a right triangle whose heavy side is the bottom-left. The highlight must show it that way up, not turned half way round.
        kernel = filter_picture(block=[[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        plane = view(self.blur(kernel, image_filter="impulse"))
        below, above = float(plane[MID + 1:, :].sum()), float(plane[:MID, :].sum())
        left, right = float(plane[:, :MID].sum()), float(plane[:, MID + 1:].sum())
        self.assertGreater(below, above)
        self.assertGreater(left, right)

    def test_impulse_carries_the_original_values_and_the_others_smooth_or_sharpen(self):
        kernel = filter_picture(block=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        impulse = view(self.blur(kernel, image_filter="impulse"))
        values = np.unique(np.round(impulse[impulse > 0], 9))
        self.assertEqual(len(values), 1)                                            # every original pixel weighs the same: one value
        for name in ("cubic", "keys", "simon", "rifman", "mitchell", "parzen", "lanczos4", "lanczos6", "sinc4", "bilinear"):
            with self.subTest(filter=name):
                out = view(self.blur(kernel, image_filter=name))
                self.assertAlmostEqual(float(out.sum()), 1.0, places=5)             # every filter is normalised
        shaped = filter_picture()
        impulse, keys, parzen = (view(self.blur(shaped, image_filter=name)) for name in ("impulse", "keys", "parzen"))
        self.assertTrue(differs(impulse, parzen) and differs(keys, parzen))
        # the sharpening filters (keys, simon, rifman) ring: a negative lobe beside a step inside the filter; parzen (a B-spline) does not
        for name in ("keys", "simon", "rifman"):
            self.assertLess(float(self.blur(RING, image_filter=name).min()), 0.0, name)
        self.assertGreaterEqual(float(self.blur(RING, image_filter="parzen").min()), 0.0)

    def test_clamp_image_filter_removes_the_halo_of_a_sharpening_filter(self):
        ringing = self.blur(RING, image_filter="rifman")
        clamped = self.blur(RING, image_filter="rifman", clamp_image_filter=1)
        self.assertLess(float(ringing.min()), 0.0)                                  # the halo: a negative weight beside the hole
        self.assertGreaterEqual(float(clamped.min()), 0.0)
        self.assertTrue(differs(clamped, ringing))
        np.testing.assert_array_equal(self.blur(RING, image_filter="impulse", clamp_image_filter=1), self.blur(RING, image_filter="impulse"))

    def test_the_filter_channel_and_use_input_channels(self):
        kernel = filter_picture()
        kernel[5:8, 7, 0] = 1.0                                                     # red: the left column of the block
        red, alpha = self.blur(kernel, image_filter="impulse", filter_channel="rgba.red"), self.blur(kernel, image_filter="impulse")
        self.assertTrue(differs(red, alpha))
        np.testing.assert_allclose(red[..., 0], red[..., 3])                       # one plane is read for every channel
        paired = self.blur(kernel, image_filter="impulse", use_input_channels=1)
        np.testing.assert_allclose(paired[..., 0], red[..., 0], atol=1e-7)          # red by the filter's red
        np.testing.assert_allclose(paired[..., 3], alpha[..., 3], atol=1e-7)        # alpha by its alpha
        self.assertTrue(differs(paired[..., 0], paired[..., 3]))
        # the filter channel is not read while use input channels is on
        np.testing.assert_array_equal(self.blur(kernel, image_filter="impulse", use_input_channels=1, filter_channel="rgba.red"), paired)

    def test_the_filter_is_the_box_round_its_pixels_and_format_bounds_use_the_whole_picture(self):
        corner, middle = filter_picture(at=(0, 0)), filter_picture(at=(5, 7))
        np.testing.assert_allclose(self.blur(corner, image_filter="impulse"), self.blur(middle, image_filter="impulse"), atol=1e-7)   # the box, not the position
        k = dict(image_filter="impulse", legacy_resize_mode=1)
        shape_a, shape_b = self.blur(corner, filter_bounds="shape", **k), self.blur(middle, filter_bounds="shape", **k)
        np.testing.assert_allclose(shape_a, shape_b, atol=1e-7)
        format_a, format_b = self.blur(corner, filter_bounds="format", **k), self.blur(middle, filter_bounds="format", **k)
        self.assertTrue(differs(format_a, format_b))                               # the whole picture: the offset of the block inside it counts
        self.assertTrue(differs(format_b, shape_b))

    def test_the_image_knobs_need_a_wired_filter_and_the_image_type(self):
        params = {**SPECS["ZDefocus"]["params"], "bokeh_shape": "image", "zdefocus_math": "depth"}
        with self.assertRaises(ValueError):
            Evaluator._zdefocus(point(), np.full((N, N), 3.0, np.float32), params, None)
        disc = spread(bokeh_shape="disc")
        np.testing.assert_array_equal(spread(filter_channel="rgba.red", use_input_channels=1, image_filter="rifman", clamp_image_filter=1), disc)


class SavedNodeTests(unittest.TestCase):
    """A node saved before these knobs keeps its picture bit for bit: the same picture a node of the step M release drew."""
    GOLDEN = {"blades": "ef46f61d227a80f2ffaa", "blades_hollow": "6f4d6ddbb7df90bbc074", "image": "3ed71c2e1d609ce9d43d"}

    def pictures(self, **extra):
        rng = np.random.default_rng(5)
        img = rng.random((12, 16, 4)).astype(np.float32)
        img[..., 3] = 1
        depth = np.tile(np.linspace(0.5, 4.0, 16, dtype=np.float32), (12, 1))
        kern = rng.random((5, 7, 4)).astype(np.float32)
        kern[..., 3] = np.where(kern[..., 3] > 0.4, kern[..., 3], 0)
        base = dict(SPECS["ZDefocus"]["params"], zdefocus_math="depth", focal_plane=2.0, depth_of_field=0.5, size=6.0, max_size=5.0, **extra)
        configs = {"blades": dict(bokeh_shape="blades", blade_count=5, blade_rotation=20.0, aspect=1.5),
                   "blades_hollow": dict(bokeh_shape="blades", blade_count=3, catadioptric=1), "image": dict(bokeh_shape="image")}
        out = {}
        for name, knobs_ in configs.items():
            p = dict(base, **knobs_)
            result = Evaluator._zdefocus(img, depth, p, kern if name == "image" else None)
            out[name] = hashlib.sha256(np.ascontiguousarray(result, np.float32).tobytes()).hexdigest()[:20]
        return out

    def test_the_legacy_filters_draw_the_old_pixels(self):
        self.assertEqual(self.pictures(legacy_zdefocus_filters=1), self.GOLDEN)

    def test_a_node_without_the_flag_is_a_saved_node(self):
        params = {**SPECS["ZDefocus"]["params"]}
        self.assertEqual(params["legacy_zdefocus_filters"], 0)
        reference = self.pictures(legacy_zdefocus_filters=1)
        self.assertEqual(reference, self.GOLDEN)
        self.assertNotEqual(self.pictures(legacy_zdefocus_filters=0), self.GOLDEN)        # a new node draws the page's filters

    def test_an_older_document_gets_the_legacy_flag_on_load(self):
        names = ("legacy_zdefocus_filters", "roundness", "inner_size", "inner_feather", "inner_brightness", "legacy_resize_mode",
                 "filter_channel", "use_input_channels", "filter_bounds", "image_filter", "clamp_image_filter")
        old = old_document("ZDefocus", dict(focal_plane=1.0, depth_of_field=1.0, max_size=6.0), names)["nodes"]["n"]["params"]
        self.assertEqual(old["legacy_zdefocus_filters"], 1)
        self.assertEqual((old["roundness"], old["inner_brightness"], old["image_filter"]), (0.2, 0.8, "cubic"))


class TileAndPanelTests(unittest.TestCase):
    def doc(self, **knobs_):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("depth", "Ramp", dict(width=45, height=37, p0_x=0, p0_y=0, p1_x=44, p1_y=0))
        params = {"focal_plane": 0.5, "depth_of_field": 0.2, "size": 4.0, "max_size": 3.0, "autoLayerSpacing": 0, "nLayers": 5,
                  "bokeh_shape": "blades", "blade_count": 5, **knobs_}
        g.add("node", "ZDefocus", params, image="plate", depth="depth")
        return g.doc

    def test_tiles_equal_the_full_frame_with_the_bladed_knobs(self):
        extra = dict(roundness=0.5, inner_size=0.6, inner_feather=0.7, inner_brightness=0.2, aspect=1.5, blade_rotation=30.0)
        full = evaluator_pixels(self.doc(**extra), "node")
        for edge in (7, 16, 32):
            np.testing.assert_allclose(tile_pixels(self.doc(**extra), "node", edge), full, atol=1e-6, err_msg=f"tile {edge}")

    def test_an_image_filter_runs_on_the_full_frame(self):
        self.assertTrue(zdefocusknobs.needs_full_frame({**SPECS["ZDefocus"]["params"], "bokeh_shape": "image"}))
        self.assertFalse(zdefocusknobs.needs_full_frame({**SPECS["ZDefocus"]["params"], "bokeh_shape": "blades"}))

    def test_the_panel_lists_the_new_knobs_in_the_guides_order(self):
        shown = [g.label for g in knobs.KNOB_LAYOUT["ZDefocus"]
                 if g.params[0] not in ("legacy_zdefocus", "legacy_zdefocus_focus", "legacy_zdefocus_filters") and g.params != ("mix",)]
        start = shown.index("Filter type")
        self.assertEqual(shown[start:start + 19],
                         ["Filter type", "Filter shape", "Aspect ratio", "Blades", "Roundness", "Rotation", "Inner size", "Inner feather",
                          "Inner brightness", "Catadioptric size", "Catadioptric", "Legacy resize mode", "Filter channel", "Use input channels",
                          "Filter bounds", "Image filter", "Clamp image filter", "Gamma correction", "Bloom"])
        params = [name for g in knobs.KNOB_LAYOUT["ZDefocus"] for name in g.params]
        for name in ("roundness", "inner_size", "inner_feather", "inner_brightness", "legacy_resize_mode", "filter_channel",
                     "use_input_channels", "filter_bounds", "image_filter", "clamp_image_filter", "legacy_zdefocus_filters"):
            self.assertEqual(params.count(name), 1, name)
            self.assertIn(name, SPECS["ZDefocus"]["params"])


if __name__ == "__main__":
    unittest.main()
