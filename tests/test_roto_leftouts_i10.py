"""Lane 9 step I10: the left-outs from the first Roto tools of 11/10.

1. B-Spline outlines keep sharp tips (a triple control point at every corner of the Bezier outline).
2. An island inside a hole is added back as an ordinary shape in front of the hole.
4. A pocket touching the subject's edge stays background and the status bar says so.
(3, strokes with their own transform, is in tests/test_roto_stick_i10.py.)
"""
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest

from nodebased import roto, rotooutline, shapes as shape_model
from tests import test_roto_edges_i1 as base
from tests.test_roto_outline_holes_i6 import BACKGROUND, SUBJECT, image_of, ring_inside, render_matte, disagreement
from tests.test_roto_outline_i5 import W, H, boundary_distances, compose, disc_inside, star_geometry, supersampled, rounded_rectangle_inside

ISLAND = (0.2, 0.7, 0.9)


def curve_error(points, inside, curve_type):
    """Distance from each vertex of the flattened outline to the true boundary: (mean, max)."""
    distance = boundary_distances(points, inside) if curve_type == "bezier" else bspline_distances(points, inside)
    return float(distance.mean()), float(distance.max())


def bspline_distances(controls, inside, factor=8):
    ys, xs = np.mgrid[0:H * factor, 0:W * factor].astype(np.float64)
    mask = inside((xs + 0.5) / factor, (ys + 0.5) / factor)
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    ey, ex = np.nonzero(edge)
    samples = np.stack([(ex + 0.5) / factor, (ey + 0.5) / factor], axis=1)
    vertices = np.array(roto.flatten_points(controls, closed=True, curve_type="bspline"))
    return np.sqrt(((vertices[:, None, :] - samples[None, :, :]) ** 2).sum(-1)).min(axis=1)


def island_picture(island_colour=SUBJECT):
    """A ring (radii 45 to 85) with a disc of its own in the middle (radius 20)."""
    ring = supersampled(W, H, lambda x, y: (np.hypot(x - 160, y - 140) <= 85) & (np.hypot(x - 160, y - 140) >= 45))
    island = supersampled(W, H, lambda x, y: np.hypot(x - 160, y - 140) <= 20)
    image = np.zeros((H, W, 4), np.float32)
    image[..., 3] = 1.0
    image[..., :3] = (np.array(BACKGROUND) * (1 - ring - island)[..., None] + np.array(SUBJECT) * ring[..., None]
                      + np.array(island_colour) * island[..., None])
    return image


def island_truth(x, y):
    r = np.hypot(x - 160, y - 140)
    return ((r <= 85) & (r >= 45)) | (r <= 20)


def render(shapes_points):
    """The matte of ``[(points, mode, curve_type), ...]`` drawn in that order."""
    stored = [{"name": f"s{i}", "mode": mode, "opacity": 1.0, "feather": 0.0, "feather_model": "edge", "closed": True,
               "points": points, "curve_type": curve, "color": [1.0, 1.0, 1.0, 1.0], "blend_mode": "over", "source": "color"}
              for i, (points, mode, curve) in enumerate(shapes_points)]
    return roto.rasterise(shape_model.resolve_shapes({"shapes": stored}, 1), W, H)[..., 3]


def c_ring(gap):
    def inside(x, y):
        r = np.hypot(x - 160, y - 140)
        return ((r <= 80) & (r >= 45)) & ~((np.abs(y - 140) <= gap / 2) & (x > 160))
    return inside


class SharpTipTests(unittest.TestCase):
    def star_points(self):
        inside, vertices = star_geometry()
        image = compose(supersampled(W, H, inside), (0.9, 0.7, 0.2), (0.15, 0.2, 0.3))
        return inside, vertices, rotooutline.outline_from_click(image, (160.0, 150.0))

    def test_the_star_corners_are_found(self):
        _, _, points = self.star_points()
        self.assertGreaterEqual(len(rotooutline.bezier_corners(points)), 10)

    def test_smooth_outlines_have_no_corners_and_do_not_change(self):
        for inside in (disc_inside(), rounded_rectangle_inside()):
            points = rotooutline.outline_from_click(compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1)), (160.0, 140.0))
            self.assertEqual(rotooutline.bezier_corners(points), [])
            self.assertEqual(rotooutline.bspline_controls(points), rotooutline.bspline_controls(points, sharp=False))

    def test_b_spline_star_is_as_accurate_as_the_bezier_outline(self):
        inside, vertices, points = self.star_points()
        bezier = curve_error(points, inside, "bezier")
        before = curve_error(rotooutline.bspline_controls(points, sharp=False), inside, "bspline")
        after = curve_error(rotooutline.bspline_controls(points), inside, "bspline")
        print(f"star outline error, mean/max px: bezier {bezier[0]:.2f}/{bezier[1]:.2f}, "
              f"b-spline before {before[0]:.2f}/{before[1]:.2f}, after {after[0]:.2f}/{after[1]:.2f}")
        # the Bezier outline's own bounds from the I5 tests: mean under 1.0 px, max under 3.0 px
        self.assertLess(after[0], 1.0)
        self.assertLess(after[1], 3.0)
        # and clearly better than the rounded one
        self.assertLess(after[0], 0.5 * before[0])
        self.assertLess(after[1], 0.6 * before[1])

    def test_b_spline_tips_land_on_the_star_tips(self):
        inside, vertices, points = self.star_points()
        curve = np.array(roto.flatten_points(rotooutline.bspline_controls(points), closed=True, curve_type="bspline"))
        for index, vertex in enumerate(vertices):
            nearest = float(np.hypot(*(curve - vertex).T).min())
            self.assertLess(nearest, 1.5, f"vertex {index} is {nearest:.2f} px from the curve")

    def test_a_square_keeps_four_sharp_corners_as_a_b_spline(self):
        inside = lambda x, y: (np.abs(x - 160) <= 60) & (np.abs(y - 140) <= 60)
        points = rotooutline.outline_from_click(compose(supersampled(W, H, inside), (0.9, 0.3, 0.3), (0.2, 0.5, 0.3)), (160.0, 140.0))
        controls = rotooutline.bspline_controls(points)
        curve = np.array(roto.flatten_points(controls, closed=True, curve_type="bspline"))
        for corner in ((100, 80), (220, 80), (220, 200), (100, 200)):
            self.assertLess(float(np.hypot(*(curve - corner).T).min()), 1.2)
        mean, maximum = curve_error(controls, inside, "bspline")
        self.assertLess(mean, 0.6)
        self.assertLess(maximum, 2.0)

    def test_the_curve_still_passes_through_the_smooth_points(self):
        # a disc with one corner-less outline: the interpolation property of the old solve is unchanged
        inside = disc_inside()
        points = rotooutline.outline_from_click(compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1)), (160.0, 140.0))
        controls = rotooutline.bspline_controls(points)
        radii = [math.hypot(x - 160.0, y - 140.0) for x, y in roto.flatten_points(controls, closed=True, curve_type="bspline")]
        self.assertAlmostEqual(float(np.mean(radii)), 70.0, delta=1.0)


class IslandTests(unittest.TestCase):
    def test_an_island_inside_a_hole_is_found_whatever_its_colour(self):
        for colour in (SUBJECT, ISLAND):
            outer, holes, islands, open_holes = rotooutline.outline_with_islands(island_picture(colour), (160.0 + 65.0, 140.0))
            self.assertEqual((len(holes), len(islands), open_holes), (1, 1, 0), f"colour {colour}")
            for points, radius in ((holes[0], 45.0), (islands[0], 20.0)):
                radii = [math.hypot(x - 160.0, y - 140.0) for x, y in roto.flatten_points(points, closed=True)]
                self.assertAlmostEqual(float(np.mean(radii)), radius, delta=1.0)

    def test_the_matte_with_the_island_matches_the_truth(self):
        outer, holes, islands, _ = rotooutline.outline_with_islands(island_picture(), (160.0 + 65.0, 140.0))
        matte = render([(outer, "union", "bezier")] + [(h, "subtract", "bezier") for h in holes] + [(i, "union", "bezier") for i in islands])
        near, truth = disagreement(matte, island_truth)
        self.assertLess(len(near), 0.06 * truth.sum() + 40)
        self.assertTrue(len(near) == 0 or near.max() <= 1.0, f"a pixel {near.max():.2f} px from the true edge disagrees")
        self.assertGreater(matte[140, 160], 0.99)       # the island is back
        self.assertLess(matte[140, 160 + 32], 0.01)     # the hole around it is still cut
        self.assertGreater(matte[140, 160 + 65], 0.99)  # the ring is full
        without = render([(outer, "union", "bezier")] + [(h, "subtract", "bezier") for h in holes])
        self.assertLess(without[140, 160], 0.01)        # what the island fixes

    def test_a_speck_in_a_hole_is_not_an_island(self):
        ring = supersampled(W, H, lambda x, y: (np.hypot(x - 160, y - 140) <= 85) & (np.hypot(x - 160, y - 140) >= 45))
        speck = supersampled(W, H, lambda x, y: np.hypot(x - 160, y - 140) <= 1.6)
        image = np.zeros((H, W, 4), np.float32)
        image[..., 3] = 1.0
        image[..., :3] = np.array(BACKGROUND) * (1 - ring - speck)[..., None] + np.array(SUBJECT) * (ring + speck)[..., None]
        outer, holes, islands, _ = rotooutline.outline_with_islands(image, (160.0 + 65.0, 140.0))
        self.assertEqual((len(holes), len(islands)), (1, 0))

    def test_the_island_size_limits_are_those_of_a_hole(self):
        raw = np.zeros((60, 60), bool)
        hole = np.zeros_like(raw)
        hole[10:50, 10:50] = True
        raw[20:22, 20:22] = True        # 4 cells: a speck
        raw[30:38, 30:38] = True        # 64 cells
        found = rotooutline.find_islands(raw, np.zeros_like(raw), hole)
        self.assertEqual([int(part.sum()) for part in found], [64])

    def test_a_ring_without_an_island_has_none(self):
        outer, holes, islands, _ = rotooutline.outline_with_islands(image_of(ring_inside()), (160.0 + 57.0, 140.0))
        self.assertEqual((len(holes), islands), (1, []))

    def test_outline_with_holes_keeps_its_two_values(self):
        outer, holes = rotooutline.outline_with_holes(island_picture(), (160.0 + 65.0, 140.0))
        self.assertEqual(len(holes), 1)


class TouchingHoleTests(unittest.TestCase):
    def test_a_pocket_behind_a_narrow_gap_is_counted_and_not_a_hole(self):
        for gap in (3, 6, 12):
            result = rotooutline.outline_with_islands(image_of(c_ring(gap)), (160.0 - 60.0, 140.0))
            self.assertEqual((len(result.holes), result.open_holes), (0, 1), f"gap {gap}")

    def test_a_wide_gap_is_just_a_c_shape(self):
        result = rotooutline.outline_with_islands(image_of(c_ring(40)), (160.0 - 60.0, 140.0))
        self.assertEqual((len(result.holes), result.open_holes), (0, 0))

    def test_closed_holes_and_plain_shapes_report_none(self):
        self.assertEqual(rotooutline.outline_with_islands(image_of(ring_inside()), (217.0, 140.0)).open_holes, 0)
        inside, _ = star_geometry()
        star = compose(supersampled(W, H, inside), (0.9, 0.7, 0.2), (0.15, 0.2, 0.3))
        self.assertEqual(rotooutline.outline_with_islands(star, (160.0, 150.0)).open_holes, 0)
        disc = image_of(disc_inside())
        self.assertEqual(rotooutline.outline_with_islands(disc, (160.0, 140.0)).open_holes, 0)

    def test_the_outline_itself_is_unchanged(self):
        # the pocket stays background: the outline is the C
        result = rotooutline.outline_with_islands(image_of(c_ring(6)), (160.0 - 60.0, 140.0))
        matte = render([(result.outer, "union", "bezier")])
        self.assertLess(matte[140, 160], 0.01)          # the middle of the C is empty
        self.assertGreater(matte[140, 160 - 62], 0.99)  # the ring is full


class _FakeRaster:
    def __init__(self, image):
        self.image = image

    def to_display(self):
        return self.image


class ViewerTests(unittest.TestCase):
    """The click through the real input path."""

    open = base.ViewerEdgeTests.open
    tearDown = base.ViewerEdgeTests.tearDown
    scene_pos = base.ViewerEdgeTests.scene_pos

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.open(base.disc_plate_document())
        self.viewer.select_roto_tool("Shapes", "Bezier")
        self.viewer.roto_toolbar.set_setting("r", "Bezier", "outline_click", True)
        self.messages = []
        bar = self.window.statusBar()
        original = bar.showMessage
        bar.showMessage = lambda text, *args: (self.messages.append(text), original(text, *args))[1]

    def said(self):
        """Every status message the Outline from click run sent."""
        return " | ".join(m for m in self.messages if m.startswith("Outline from click"))

    def shapes(self):
        return self.window.dispatcher.document["node_data"].get("r", {}).get("shapes", [])

    def click(self, picture, x, y):
        self.viewer.window.evaluator.evaluate_raster = lambda *a, **k: _FakeRaster(picture)
        QTest.mouseClick(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos=self.scene_pos(x, y))
        base.APP.processEvents()

    def picture(self, inside, colour=SUBJECT):
        return compose(supersampled(100, 100, inside), colour, BACKGROUND)

    def test_a_click_on_a_ring_with_an_island_adds_outline_hole_and_island_in_one_undo_step(self):
        r = lambda x, y: np.hypot(x - 50, y - 50)
        ring = self.picture(lambda x, y: ((r(x, y) <= 36) & (r(x, y) >= 20)) | (r(x, y) <= 8))
        undo = len(self.window.dispatcher.undo_stack)
        self.click(ring, 50 + 28, 50)
        shapes = self.shapes()
        self.assertEqual([s["mode"] for s in shapes], ["union", "subtract", "union"])
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        self.assertEqual(self.viewer.roto_selected_shape_index, 0)
        radii = [math.hypot(x - 50.0, y - 50.0) for x, y in roto.flatten_points(shapes[2]["points"], closed=True)]
        self.assertAlmostEqual(float(np.mean(radii)), 8.0, delta=1.0)
        self.assertIn("1 island", self.said())
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), [])
        self.window.command({"op": "redo"})
        self.assertEqual(len(self.shapes()), 3)

    def test_b_spline_islands_are_b_splines_with_sharp_tips(self):
        self.viewer.select_roto_tool("Shapes", "B-Spline")
        self.viewer.roto_toolbar.set_setting("r", "B-Spline", "outline_click", True)
        r = lambda x, y: np.hypot(x - 50, y - 50)
        self.click(self.picture(lambda x, y: ((r(x, y) <= 36) & (r(x, y) >= 20)) | (r(x, y) <= 8)), 50 + 28, 50)
        self.assertEqual([s.get("curve_type") for s in self.shapes()], ["bspline"] * 3)

    def test_the_status_bar_names_a_pocket_touching_the_edge(self):
        r = lambda x, y: np.hypot(x - 50, y - 50)
        c_shape = self.picture(lambda x, y: (r(x, y) <= 36) & (r(x, y) >= 20) & ~((np.abs(y - 50) <= 2) & (x > 50)))
        self.click(c_shape, 50 - 28, 50)
        self.assertEqual([s["mode"] for s in self.shapes()], ["union"])
        message = self.said()
        self.assertIn("touching the subject's edge", message)
        self.assertIn("subtract", message)

    def test_a_plain_outline_says_nothing_about_pockets(self):
        r = lambda x, y: np.hypot(x - 50, y - 50)
        self.click(self.picture(lambda x, y: r(x, y) <= 30), 50, 50)
        self.assertIn("Outline from click", self.said())
        self.assertNotIn("pocket", self.said())


if __name__ == "__main__":
    unittest.main()
