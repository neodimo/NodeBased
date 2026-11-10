"""Lane 2 step I6: Outline from click cuts the holes out of a subject.

The subject is outlined as one closed shape; every interior hole at least `MIN_HOLE_FRACTION` of its area (and
`MIN_HOLE_CELLS` working cells) becomes a closed shape set to subtract, placed right after it. Accuracy is measured on the
rendered matte against the true coverage: every pixel the two disagree on lies within 1 px of the true edge.
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
from tests.test_roto_outline_i5 import W, H, compose, supersampled
from tests.test_roto_edges_i1 import DISC_CENTRE

SUBJECT, BACKGROUND = (0.9, 0.55, 0.25), (0.1, 0.1, 0.1)
EDGE_TOLERANCE = 1.0


def ring_inside(centre=(160.0, 140.0), outer=80.0, inner=35.0):
    return lambda x, y: (np.hypot(x - centre[0], y - centre[1]) <= outer) & (np.hypot(x - centre[0], y - centre[1]) >= inner)


def letter_inside():
    """A letter-like block (a rounded rectangle) with two round holes, like an 8 or a B."""
    def inside(x, y):
        dx = np.maximum(np.abs(x - 160.0) - 55.0, 0)
        dy = np.maximum(np.abs(y - 140.0) - 85.0, 0)
        body = np.hypot(dx, dy) <= 25.0
        upper = np.hypot(x - 160.0, y - 108.0) <= 24.0
        lower = np.hypot(x - 160.0, y - 176.0) <= 20.0
        return body & ~upper & ~lower
    return inside


def speck_inside():
    """A disc with one real hole and a 3 px speck of background inside it."""
    def inside(x, y):
        disc = np.hypot(x - 160.0, y - 140.0) <= 85.0
        hole = np.hypot(x - 125.0, y - 140.0) <= 25.0
        speck = np.hypot(x - 200.0, y - 120.0) <= 1.6
        return disc & ~hole & ~speck
    return inside


def image_of(inside):
    return compose(supersampled(W, H, inside), SUBJECT, BACKGROUND)


def render_matte(outer, holes):
    """The matte of the outline and its holes drawn as the Roto does: the outline unioned, each hole subtracting."""
    def shape(points, mode):
        return {"name": "s", "mode": mode, "opacity": 1.0, "feather": 0.0, "feather_model": "edge", "closed": True,
                "points": points, "color": [1.0, 1.0, 1.0, 1.0], "blend_mode": "over", "source": "color"}
    stored = [shape(outer, "union")] + [shape(h, "subtract") for h in holes]
    resolved = shape_model.resolve_shapes({"shapes": stored}, 1)
    return roto.rasterise(resolved, W, H)[..., 3]


def centroid(points):
    return np.mean(np.array(roto.flatten_points(points, closed=True)), axis=0)


def disagreement(matte, inside, factor=8):
    """Distances from the pixels where `matte` (>0.5) and the true subject disagree to the true edge."""
    ys, xs = np.mgrid[0:H * factor, 0:W * factor].astype(np.float64)
    fine = inside((xs + 0.5) / factor, (ys + 0.5) / factor)
    truth = fine.reshape(H, factor, W, factor).mean(axis=(1, 3)) >= 0.5
    wrong = (matte >= 0.5) != truth
    edge = fine & ~(np.roll(fine, 1, 0) & np.roll(fine, -1, 0) & np.roll(fine, 1, 1) & np.roll(fine, -1, 1))
    ey, ex = np.nonzero(edge)
    samples = np.stack([(ex + 0.5) / factor, (ey + 0.5) / factor], axis=1)
    wy, wx = np.nonzero(wrong)
    if not len(wy):
        return np.zeros(0), truth
    pixels = np.stack([wx + 0.5, wy + 0.5], axis=1)
    near = np.array([np.sqrt(((samples - p) ** 2).sum(-1)).min() for p in pixels])
    return near, truth


class HoleDetectionTests(unittest.TestCase):
    def check_matte(self, inside, outer, holes):
        matte = render_matte(outer, holes)
        near, truth = disagreement(matte, inside)
        coverage = float(truth.mean())
        self.assertLess(len(near), 0.06 * truth.sum() + 40, f"{len(near)} wrong pixels of {int(truth.sum())}")
        self.assertTrue(len(near) == 0 or near.max() <= EDGE_TOLERANCE, f"a pixel {near.max():.2f} px from the true edge disagrees")
        return coverage

    def test_ring_has_one_hole_and_the_matte_matches(self):
        inside = ring_inside()
        outer, holes = rotooutline.outline_with_holes(image_of(inside), (160.0 + 57.0, 140.0))
        self.assertEqual(len(holes), 1)
        centre = centroid(holes[0])
        self.assertLess(float(np.hypot(*(centre - (160.0, 140.0)))), 1.5)
        radii = [math.hypot(x - 160.0, y - 140.0) for x, y in roto.flatten_points(holes[0], closed=True)]
        self.assertAlmostEqual(float(np.mean(radii)), 35.0, delta=1.0)
        self.check_matte(inside, outer, holes)
        # the hole is really cut: its centre is empty and the ring is full
        matte = render_matte(outer, holes)
        self.assertLess(matte[140, 160], 0.01)
        self.assertGreater(matte[140, 217], 0.99)

    def test_the_old_outline_still_fills_the_hole(self):
        inside = ring_inside()
        outline = rotooutline.outline_from_click(image_of(inside), (217.0, 140.0))
        radii = [math.hypot(x - 160.0, y - 140.0) for x, y in roto.flatten_points(outline, closed=True)]
        self.assertAlmostEqual(float(np.mean(radii)), 80.0, delta=1.0)

    def test_letter_like_shape_has_two_holes(self):
        inside = letter_inside()
        outer, holes = rotooutline.outline_with_holes(image_of(inside), (160.0, 140.0))
        self.assertEqual(len(holes), 2)
        centres = sorted(centroid(h)[1] for h in holes)
        self.assertAlmostEqual(centres[0], 108.0, delta=1.5)
        self.assertAlmostEqual(centres[1], 176.0, delta=1.5)
        # biggest hole first
        areas = [abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])) / 2.0)
                 for poly in ([tuple(q) for q in roto.flatten_points(h, closed=True)] for h in holes)]
        self.assertGreaterEqual(areas[0], areas[1])
        self.check_matte(inside, outer, holes)

    def test_a_tiny_speck_does_not_become_a_hole(self):
        inside = speck_inside()
        outer, holes = rotooutline.outline_with_holes(image_of(inside), (200.0, 180.0))
        self.assertEqual(len(holes), 1)
        centre = centroid(holes[0])
        self.assertLess(float(np.hypot(*(centre - (125.0, 140.0)))), 1.5)

    def test_a_subject_without_holes_gives_an_empty_list(self):
        inside = lambda x, y: np.hypot(x - 160.0, y - 140.0) <= 70.0
        outer, holes = rotooutline.outline_with_holes(image_of(inside), (160.0, 140.0))
        self.assertEqual(holes, [])
        self.assertGreaterEqual(len(outer), rotooutline.MIN_POINTS)

    def test_the_hole_size_limits_are_stated(self):
        self.assertEqual(rotooutline.MIN_HOLE_FRACTION, 0.01)
        self.assertEqual(rotooutline.MIN_HOLE_CELLS, 16)
        body = np.ones((40, 40), bool)
        silhouette = body.copy()
        body[10:12, 10:12] = False      # 4 cells
        body[20:30, 20:30] = False      # 100 cells
        found = rotooutline.find_holes(body, silhouette)
        self.assertEqual([int(part.sum()) for part in found], [100])


class ViewerHoleTests(unittest.TestCase):
    """The click through the real input path: the outline and its holes arrive in one command."""

    open = base.ViewerEdgeTests.open
    tearDown = base.ViewerEdgeTests.tearDown
    scene_pos = base.ViewerEdgeTests.scene_pos

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.open(base.disc_plate_document())
        self.viewer.select_roto_tool("Shapes", "Bezier")
        self.viewer.roto_toolbar.set_setting("r", "Bezier", "outline_click", True)

    def shapes(self):
        return self.window.dispatcher.document["node_data"].get("r", {}).get("shapes", [])

    def test_a_click_on_a_ring_adds_the_outline_and_its_hole_in_one_undo_step(self):
        # the plate is a disc; feed the outline a ring picture instead
        ring = compose(supersampled(100, 100, lambda x, y: (np.hypot(x - 50, y - 50) <= 28) & (np.hypot(x - 50, y - 50) >= 12)),
                       SUBJECT, BACKGROUND)
        self.viewer.window.evaluator.evaluate_raster = lambda *a, **k: _FakeRaster(ring)
        undo = len(self.window.dispatcher.undo_stack)
        QTest.mouseClick(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         pos=self.scene_pos(50 + 20, 50))
        base.APP.processEvents()
        shapes = self.shapes()
        self.assertEqual([s["mode"] for s in shapes], ["union", "subtract"])
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        self.assertEqual(self.viewer.roto_selected_shape_index, 0)
        radii = [math.hypot(x - 50.0, y - 50.0) for x, y in roto.flatten_points(shapes[1]["points"], closed=True)]
        self.assertAlmostEqual(float(np.mean(radii)), 12.0, delta=1.0)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), [])
        self.window.command({"op": "redo"})
        self.assertEqual(len(self.shapes()), 2)


class _FakeRaster:
    def __init__(self, image):
        self.image = image

    def to_display(self):
        return self.image


if __name__ == "__main__":
    unittest.main()
