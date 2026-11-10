"""Roto outline from one click (lane 9, step I5).

Accuracy is measured against the true boundary of synthetic subjects: the distance from each vertex of the
flattened outline to the nearest point of the true boundary (drawn at 8x oversampling). Thresholds held:

* disc, rounded rectangle, star, subject on a textured background: mean distance <= 1.0 px, max <= 3.0 px
* low contrast (0.08 step with sensor noise): mean <= 1.5 px, max <= 4.0 px
* 8 to 40 points for every subject; the star's five concave points and five tips are each within 2.5 px of an
  outline point.
"""
import math
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest

from nodebased import roto, rotoedges, rotooutline
from tests import test_roto_edges_i1 as base
from tests.test_roto_edges_i1 import DISC_CENTRE, edge_error

W, H = 320, 280
DISC_C, DISC_R = (160.0, 140.0), 70.0


def supersampled(width, height, inside, factor=4):
    ys, xs = np.mgrid[0:height * factor, 0:width * factor].astype(np.float64)
    hit = inside((xs + 0.5) / factor, (ys + 0.5) / factor).astype(np.float64)
    return hit.reshape(height, factor, width, factor).mean(axis=(1, 3))


def compose(coverage, subject, background):
    image = np.zeros(coverage.shape + (4,), np.float32)
    image[..., 3] = 1.0
    image[..., :3] = background * (1 - coverage[..., None]) + np.asarray(subject) * coverage[..., None]
    return image


def disc_inside(centre=DISC_C, radius=DISC_R):
    return lambda x, y: np.hypot(x - centre[0], y - centre[1]) <= radius


def rounded_rectangle_inside(centre=(160.0, 140.0), half_width=100.0, half_height=70.0, radius=25.0):
    def inside(x, y):
        dx = np.maximum(np.abs(x - centre[0]) - (half_width - radius), 0)
        dy = np.maximum(np.abs(y - centre[1]) - (half_height - radius), 0)
        return np.hypot(dx, dy) <= radius
    return inside


def star_geometry(centre=(160.0, 140.0), outer=90.0, inner=40.0, tips=5):
    vertices = []
    for i in range(2 * tips):
        angle = -math.pi / 2 + math.pi * i / tips
        radius = outer if i % 2 == 0 else inner
        vertices.append((centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle)))
    vertices = np.array(vertices)

    def inside(x, y):
        result = np.zeros(x.shape, bool)
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            result ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            j = i
        return result
    return inside, vertices


def boundary_distances(points, inside, width=W, height=H, factor=8):
    """Distance from each flattened-outline vertex to the nearest boundary sample of the true subject."""
    ys, xs = np.mgrid[0:height * factor, 0:width * factor].astype(np.float64)
    mask = inside((xs + 0.5) / factor, (ys + 0.5) / factor)
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    ey, ex = np.nonzero(edge)
    samples = np.stack([(ex + 0.5) / factor, (ey + 0.5) / factor], axis=1)
    vertices = np.array(roto.flatten_points(points, closed=True))
    return np.sqrt(((vertices[:, None, :] - samples[None, :, :]) ** 2).sum(-1)).min(axis=1)


def outline(image, seed, box=None):
    return rotooutline.outline_from_click(image, seed, box)


class AccuracyTests(unittest.TestCase):
    def check(self, points, inside, mean, maximum):
        self.assertGreaterEqual(len(points), rotooutline.MIN_POINTS)
        self.assertLessEqual(len(points), rotooutline.MAX_POINTS)
        distance = boundary_distances(points, inside)
        self.assertLess(distance.mean(), mean, f"mean {distance.mean():.2f}")
        self.assertLess(distance.max(), maximum, f"max {distance.max():.2f}")
        return distance

    def test_disc_is_within_one_pixel_mean(self):
        inside = disc_inside()
        image = compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1))
        points = outline(image, DISC_C)
        self.check(points, inside, 1.0, 3.0)
        radii = [math.hypot(x - DISC_C[0], y - DISC_C[1]) for x, y in roto.flatten_points(points, closed=True)]
        self.assertAlmostEqual(float(np.mean(radii)), DISC_R, delta=1.0)

    def test_disc_clicked_off_centre_gives_the_same_outline(self):
        inside = disc_inside()
        image = compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1))
        for seed in ((120.0, 120.0), (200.0, 170.0), (150.0, 100.0)):
            self.check(outline(image, seed), inside, 1.0, 3.0)

    def test_rounded_rectangle(self):
        inside = rounded_rectangle_inside()
        image = compose(supersampled(W, H, inside), (0.9, 0.3, 0.3), (0.2, 0.5, 0.3))
        points = self.check_points = outline(image, (160.0, 140.0))
        self.check(points, inside, 1.0, 3.0)

    def test_star_finds_the_concave_points(self):
        inside, vertices = star_geometry()
        image = compose(supersampled(W, H, inside), (0.9, 0.7, 0.2), (0.15, 0.2, 0.3))
        points = outline(image, (160.0, 150.0))
        self.check(points, inside, 1.0, 3.0)
        positions = np.array([[p["x"], p["y"]] for p in points])
        for index, vertex in enumerate(vertices):
            nearest = float(np.hypot(*(positions - vertex).T).min())
            kind = "tip" if index % 2 == 0 else "concave point"
            self.assertLess(nearest, 2.5, f"{kind} {index} at {vertex} is {nearest:.2f} px from the nearest outline point")

    def test_subject_on_a_textured_background(self):
        inside = disc_inside()
        noise = 0.35 + 0.15 * np.random.default_rng(3).standard_normal((H, W, 1)).astype(np.float32)
        coverage = supersampled(W, H, inside)
        image = np.zeros((H, W, 4), np.float32)
        image[..., 3] = 1.0
        image[..., :3] = np.clip(noise, 0, 1) * (1 - coverage[..., None]) + np.array([0.8, 0.45, 0.2]) * coverage[..., None]
        self.check(outline(image, DISC_C), inside, 1.0, 3.0)

    def test_low_contrast_subject(self):
        inside = disc_inside()
        image = compose(supersampled(W, H, inside), (0.5, 0.5, 0.5), (0.42, 0.42, 0.42))
        image[..., :3] += 0.004 * np.random.default_rng(3).standard_normal((H, W, 3)).astype(np.float32)
        self.check(outline(image, DISC_C), inside, 1.5, 4.0)

    def test_a_box_separates_a_subject_from_an_overlapping_twin(self):
        left, right = disc_inside((120.0, 140.0), 55.0), disc_inside((220.0, 140.0), 55.0)
        both = lambda x, y: left(x, y) | right(x, y)
        image = compose(supersampled(W, H, both), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1))
        alone = outline(image, (120.0, 140.0))
        # without a box the two overlapping discs are one subject
        self.assertGreater(max(x for x, _ in roto.flatten_points(alone, closed=True)), 260.0)
        boxed = outline(image, (120.0, 140.0), box=(55.0, 75.0, 170.0, 205.0))
        clipped = lambda x, y: left(x, y) & (x < 170.0)
        self.check(boxed, clipped, 1.5, 4.0)
        self.assertLess(max(x for x, _ in roto.flatten_points(boxed, closed=True)), 172.0)

    def test_a_click_outside_the_picture_or_on_nothing_raises(self):
        image = compose(supersampled(W, H, disc_inside()), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1))
        with self.assertRaises(rotooutline.OutlineError):
            outline(image, (-5.0, 10.0))

    def test_point_counts_follow_the_shape(self):
        inside = disc_inside()
        disc = outline(compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1)), DISC_C)
        inside, _ = star_geometry()
        star = outline(compose(supersampled(W, H, inside), (0.9, 0.7, 0.2), (0.15, 0.2, 0.3)), (160.0, 150.0))
        self.assertLessEqual(len(disc), 12)
        self.assertGreater(len(star), len(disc))

    def test_straight_runs_have_no_handles(self):
        inside = rounded_rectangle_inside()
        points = outline(compose(supersampled(W, H, inside), (0.9, 0.3, 0.3), (0.2, 0.5, 0.3)), (160.0, 140.0))
        flat = sum(1 for p in points if p["out_x"] == 0.0 and p["out_y"] == 0.0)
        self.assertGreaterEqual(flat, 2)

    def test_bspline_controls_pass_through_the_outline(self):
        inside = disc_inside()
        points = outline(compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1)), DISC_C)
        controls = rotooutline.bspline_controls(points)
        curve = roto.flatten_points(controls, closed=True, curve_type="bspline")
        radii = [math.hypot(x - DISC_C[0], y - DISC_C[1]) for x, y in curve]
        self.assertAlmostEqual(float(np.mean(radii)), DISC_R, delta=1.0)
        self.assertLess(float(np.std(radii)), 1.0)

    def test_the_input_image_is_not_modified(self):
        inside = disc_inside()
        image = compose(supersampled(W, H, inside), (0.9, 0.55, 0.25), (0.1, 0.1, 0.1))
        before = image.copy()
        outline(image, DISC_C)
        self.assertTrue(np.array_equal(image, before))


class TimingTests(unittest.TestCase):
    def test_1080p_frame_outlines_in_a_few_seconds(self):
        width, height = 1920, 1080
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        centre, radius = (700.0, 420.0), 110.0
        coverage = np.clip(radius - np.hypot(xs + 0.5 - centre[0], ys + 0.5 - centre[1]) + 0.5, 0, 1)
        rng = np.random.default_rng(1)
        background = (0.3 + 0.1 * np.sin(xs / 37.0)[..., None] * np.sin(ys / 29.0)[..., None]
                      + 0.02 * rng.standard_normal((height, width, 1)).astype(np.float32))
        image = np.zeros((height, width, 4), np.float32)
        image[..., 3] = 1.0
        image[..., :3] = background * (1 - coverage[..., None]) + np.array([0.85, 0.5, 0.2]) * coverage[..., None]
        started = time.perf_counter()
        points = outline(image, centre)
        elapsed = time.perf_counter() - started
        radii = [math.hypot(x - centre[0], y - centre[1]) for x, y in roto.flatten_points(points, closed=True)]
        print(f"\noutline from click, 1080p: {elapsed:.2f} s, {len(points)} points, "
              f"mean radius error {abs(np.mean(radii) - radius):.2f} px, max {max(abs(np.array(radii) - radius)):.2f} px")
        self.assertLess(max(abs(np.array(radii) - radius)), 2.5)
        # Wall-clock time is only meaningful on a quiet machine: under tools/run_tests.py --jobs N the interpreters
        # share the cores (the 11/10 full suite took 31 s for a frame that takes 0.9 s alone). The measured timing
        # is in docs/ROTO_TRACKING.md; here it is checked only when the test runs on its own.
        if float(os.environ.get("NODEBASED_TEST_TIMEOUT_SCALE") or 1) == 1:
            self.assertLess(elapsed, 6.0)


class ViewerOutlineTests(base.ViewerEdgeTests):
    """The click through the real input path: viewport mouse events into the Viewer."""

    def setUp(self):
        super().setUp()
        self.viewer.select_roto_tool("Shapes", "Bezier")

    def shapes(self):
        return self.window.dispatcher.document["node_data"].get("r", {}).get("shapes", [])

    def outline_on(self, on=True, tool="Bezier"):
        self.viewer.roto_toolbar.set_setting("r", tool, "outline_click", on)

    def press_release(self, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
        QTest.mouseClick(self.viewer.viewport(), Qt.MouseButton.LeftButton, modifiers, pos=self.scene_pos(x, y))
        APP_EVENTS()

    def test_a_click_inside_the_disc_adds_one_selected_shape_in_one_undo_step(self):
        self.outline_on()
        before = len(self.shapes())
        undo = len(self.window.dispatcher.undo_stack)
        self.press_release(*DISC_CENTRE)
        shapes = self.shapes()
        self.assertEqual(len(shapes), before + 1)
        shape = shapes[-1]
        self.assertTrue(shape["closed"])
        self.assertTrue(rotooutline.MIN_POINTS <= len(shape["points"]) <= rotooutline.MAX_POINTS)
        errors = [edge_error(x, y) for x, y in roto.flatten_points(shape["points"], closed=True)]
        self.assertLess(float(np.mean(errors)), 1.0, errors)
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        self.assertEqual(self.viewer.roto_selected_shape_index, len(shapes) - 1)
        self.assertEqual(self.viewer.roto_selected_points, {(len(shapes) - 1, i) for i in range(len(shape["points"]))})
        self.assertFalse(self.viewer.roto_drawing)

    def test_one_undo_removes_the_outline_and_redo_brings_it_back(self):
        self.outline_on()
        self.press_release(*DISC_CENTRE)
        self.assertEqual(len(self.shapes()), 1)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), [])
        self.window.command({"op": "redo"})
        self.assertEqual(len(self.shapes()), 1)

    def test_the_mode_is_off_by_default_and_a_click_then_starts_an_ordinary_shape(self):
        self.assertFalse(self.viewer._roto_outline_mode())
        self.press_release(*DISC_CENTRE)
        self.assertEqual(self.shapes(), [])
        self.assertTrue(self.viewer.roto_drawing)
        self.assertEqual(len(self.viewer.roto_draw_points), 1)

    def test_setting_is_stored_and_off_is_not_stored(self):
        self.outline_on()
        shared = self.window.dispatcher.document["node_data"]["r"]["tool_settings"]["__global__"]
        self.assertTrue(shared["outline_click"])
        self.outline_on(False)
        self.assertNotIn("tool_settings", self.window.dispatcher.document["node_data"].get("r", {}))

    def test_the_strip_offers_the_mode_for_bezier_and_bspline_only(self):
        strip = self.viewer.roto_toolbar
        for tool in ("Bezier", "B-Spline"):
            strip.choose("Shapes", tool)
            self.assertIn("outline_click", strip.settings_widgets, tool)
        for tool in ("Ellipse", "Cusped Bezier"):
            strip.choose("Shapes", tool)
            self.assertNotIn("outline_click", strip.settings_widgets, tool)

    def test_shift_drag_draws_a_box_and_the_next_click_uses_it(self):
        self.outline_on()
        undo = len(self.window.dispatcher.undo_stack)
        QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                         pos=self.scene_pos(15, 15))
        QTest.mouseMove(self.viewer.viewport(), self.scene_pos(85, 85))
        QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                           pos=self.scene_pos(85, 85))
        APP_EVENTS()
        self.assertIsNotNone(self.viewer.roto_outline_box)
        self.assertEqual(self.shapes(), [])
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo)
        self.press_release(*DISC_CENTRE)
        self.assertEqual(len(self.shapes()), 1)
        self.assertIsNone(self.viewer.roto_outline_box)
        errors = [edge_error(x, y) for x, y in roto.flatten_points(self.shapes()[-1]["points"], closed=True)]
        self.assertLess(float(np.mean(errors)), 1.0)

    def test_escape_cancels_a_pending_box_and_changes_nothing(self):
        self.outline_on()
        QTest.mousePress(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                         pos=self.scene_pos(15, 15))
        QTest.mouseMove(self.viewer.viewport(), self.scene_pos(85, 85))
        QTest.mouseRelease(self.viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                           pos=self.scene_pos(85, 85))
        self.assertIsNotNone(self.viewer.roto_outline_box)
        undo = len(self.window.dispatcher.undo_stack)
        self.viewer.setFocus()
        QTest.keyClick(self.viewer, Qt.Key.Key_Escape)
        self.assertIsNone(self.viewer.roto_outline_box)
        self.assertIsNone(self.viewer.roto_outline_drag)
        self.assertEqual(self.shapes(), [])
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo)

    def test_bspline_tool_makes_a_bspline_shape_that_follows_the_disc(self):
        self.viewer.select_roto_tool("Shapes", "B-Spline")
        self.outline_on(tool="B-Spline")
        self.press_release(*DISC_CENTRE)
        shape = self.shapes()[-1]
        self.assertEqual(shape["curve_type"], "bspline")
        curve = roto.flatten_points(shape["points"], closed=True, curve_type="bspline")
        errors = [edge_error(x, y) for x, y in curve]
        self.assertLess(float(np.mean(errors)), 1.0, errors)

    def test_a_click_with_no_picture_connected_leaves_the_document_alone(self):
        self.window.command({"op": "connect", "id": "r", "input": "bg", "source": None})
        self.outline_on()
        undo = len(self.window.dispatcher.undo_stack)
        self.press_release(*DISC_CENTRE)
        self.assertEqual(self.shapes(), [])
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo)


def APP_EVENTS():
    from tests.test_roto_ui import APP
    APP.processEvents()


# the inherited snap tests belong to test_roto_edges_i1; only this module's own tests run here
for _name in [n for n in dir(base.ViewerEdgeTests) if n.startswith("test_")]:
    setattr(ViewerOutlineTests, _name, None)
del _name


if __name__ == "__main__":
    unittest.main()
