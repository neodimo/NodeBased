"""Lane 2 step I11: Shapes from matte keys the frames that need it.

Keyed every Nth frame, a shape follows a subject whose path curves only roughly between keys; the in-betweens fit cannot
always close the gap. Each in-between frame is therefore measured against its own matte, and a frame whose outline sits
more than the tolerance (1 px by default) from the matte's edge is keyed from the matte too, with the same point
correspondence, and its neighbours are fitted again. No timing is asserted.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDoubleSpinBox

from nodebased import roto, rotomatte, shapes as shape_model
from nodebased.rotolist import ShapeListTree
from tests import test_roto_edges_i1 as base

WIDTH, HEIGHT = 640, 360
FRAMES = range(1, 13)
FACTOR = 8


def curved(frame):
    """The walkthrough's subject: it crosses the screen and speeds up and slows down vertically."""
    cx, cy = 150.0 + 28.0 * frame, 150.0 + 6.0 * frame - 0.4 * frame ** 2
    return cx, cy, 85.0 + frame, 55.0


def straight(frame):
    return 150.0 + 28.0 * frame, 150.0, 90.0, 55.0


def coverage(path, frame, factor=4):
    cx, cy, rx, ry = path(frame)
    ys, xs = np.mgrid[0:HEIGHT * factor, 0:WIDTH * factor].astype(np.float32)
    hit = ((((xs + 0.5) / factor - cx) / rx) ** 2 + (((ys + 0.5) / factor - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    return hit.reshape(HEIGHT, factor, WIDTH, factor).mean(axis=(1, 3))


def worst_edge_distance(shapes, path, frame):
    """The farthest, from the subject's true edge, of the pixels on which the drawn shapes and the true subject disagree."""
    cx, cy, rx, ry = path(frame)
    drawn = roto.rasterise(shape_model.resolve_shapes({"shapes": shapes}, frame), WIDTH, HEIGHT)[..., 3] >= 0.5
    fine = np.zeros((HEIGHT * FACTOR, WIDTH * FACTOR), dtype=bool)
    xs = (np.arange(WIDTH * FACTOR, dtype=np.float32) + 0.5) / FACTOR
    for y0 in range(0, HEIGHT * FACTOR, 512):
        ys = ((np.arange(y0, min(y0 + 512, HEIGHT * FACTOR), dtype=np.float32) + 0.5) / FACTOR)[:, None]
        fine[y0:y0 + 512] = ((xs[None, :] - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    truth = fine.reshape(HEIGHT, FACTOR, WIDTH, FACTOR).mean(axis=(1, 3)) >= 0.5
    wrong = drawn != truth
    if not wrong.any():
        return 0.0
    edge = fine & ~(np.roll(fine, 1, 0) & np.roll(fine, -1, 0) & np.roll(fine, 1, 1) & np.roll(fine, -1, 1))
    ey, ex = np.nonzero(edge)
    samples = np.stack([(ex + 0.5) / FACTOR, (ey + 0.5) / FACTOR], axis=1)
    wy, wx = np.nonzero(wrong)
    pixels = np.stack([wx + 0.5, wy + 0.5], axis=1)
    return float(max(np.sqrt(((samples - p) ** 2).sum(-1)).min() for p in pixels))


class AdaptiveKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matte = staticmethod(lambda frame: coverage(curved, frame))
        cls.plain, cls.plain_sequence = rotomatte.shapes_from_matte(cls.matte, 1, 12, step=4, tolerance=0)
        cls.adaptive, cls.adaptive_sequence = rotomatte.shapes_from_matte(cls.matte, 1, 12, step=4)

    def test_the_stated_default(self):
        self.assertEqual(rotomatte.DEFAULT_TOLERANCE, 1.0)

    def test_without_adaptive_keys_the_curved_path_strays(self):
        worst = max(worst_edge_distance(self.plain, curved, f) for f in FRAMES)
        print(f"every Nth frame only: worst disagreeing pixel {worst:.2f} px from the edge")
        self.assertGreater(worst, 1.0)
        self.assertEqual(self.plain_sequence.added_keys, [])
        self.assertEqual(self.plain_sequence.key_frames, [1, 5, 9, 12])

    def test_the_curved_path_is_within_a_pixel_on_every_frame(self):
        worst = max(worst_edge_distance(self.adaptive, curved, f) for f in FRAMES)
        print(f"adaptive keys {self.adaptive_sequence.added_keys}: worst disagreeing pixel {worst:.2f} px from the edge")
        self.assertLessEqual(worst, 1.0)

    def test_keys_were_added_and_named(self):
        added = self.adaptive_sequence.added_keys
        self.assertGreater(len(added), 0)
        for frame in added:
            self.assertNotIn(frame, (1, 5, 9, 12))
            self.assertIn(frame, self.adaptive_sequence.key_frames)
        self.assertEqual(len(self.adaptive), 1)

    def test_a_straight_path_adds_no_keys(self):
        shapes, sequence = rotomatte.shapes_from_matte(lambda f: coverage(straight, f), 1, 12, step=4)
        self.assertEqual(sequence.added_keys, [])
        self.assertEqual(sequence.key_frames, [1, 5, 9, 12])
        self.assertLessEqual(max(worst_edge_distance(shapes, straight, f) for f in FRAMES), 1.0)

    def test_a_tighter_tolerance_adds_more_keys(self):
        _, tight = rotomatte.shapes_from_matte(self.matte, 1, 12, step=4, tolerance=0.25)
        self.assertGreater(len(tight.added_keys), len(self.adaptive_sequence.added_keys))

    def test_a_looser_tolerance_adds_fewer_keys(self):
        _, loose = rotomatte.shapes_from_matte(self.matte, 1, 12, step=4, tolerance=3.0)
        self.assertLessEqual(len(loose.added_keys), len(self.adaptive_sequence.added_keys))

    def test_no_fill_keeps_only_the_regular_keys(self):
        _, sequence = rotomatte.shapes_from_matte(self.matte, 1, 12, step=4, fill=False)
        self.assertEqual(sequence.added_keys, [])
        self.assertEqual(sequence.key_frames, [1, 5, 9, 12])

    def test_points_stay_in_correspondence(self):
        counts = {len(self.adaptive[0]["points"])}
        self.assertEqual(len(counts), 1)
        keyed = {k["frame"] for p in self.adaptive[0]["points"] for k in p["x"]["curve"]["keys"] if isinstance(p["x"], dict)}
        self.assertTrue(set(self.adaptive_sequence.added_keys) <= keyed)


class OutlineErrorTests(unittest.TestCase):
    def square(self, left=10, right=30):
        image = np.zeros((40, 40), np.float32)
        image[10:30, left:right] = 1.0
        return image

    def test_an_exact_match_scores_zero(self):
        self.assertEqual(rotomatte.outline_error(self.square(), self.square(), 0.5, 4.0), 0.0)

    def test_a_shifted_square_scores_its_shift(self):
        self.assertAlmostEqual(rotomatte.outline_error(self.square(13, 33), self.square(), 0.5, 6.0), 3.0, delta=0.2)

    def test_a_half_pixel_shift_of_an_antialiased_edge_scores_half_a_pixel(self):
        matte = self.square()
        drawn = self.square()
        drawn[10:30, 29] = 1.0
        drawn[10:30, 30] = 0.5
        self.assertAlmostEqual(rotomatte.outline_error(drawn, matte, 0.5, 6.0), 0.5, delta=0.1)

    def test_a_region_under_the_minimum_area_is_left_out(self):
        matte = np.zeros((40, 40), np.float32)
        matte[5:8, 5:8] = 1.0
        matte[20:35, 20:35] = 1.0
        drawn = np.zeros_like(matte)
        drawn[20:35, 20:35] = 1.0
        self.assertGreater(rotomatte.outline_error(drawn, matte, 0.5, 4.0, min_area=1), 0.0)
        self.assertEqual(rotomatte.outline_error(drawn, matte, 0.5, 4.0, min_area=64), 0.0)


class _FakeRaster:
    def __init__(self, image):
        self.image = image

    def to_display(self):
        return self.image


class ViewerAdaptiveTests(unittest.TestCase):
    open = base.ViewerEdgeTests.open
    tearDown = base.ViewerEdgeTests.tearDown

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.open(base.disc_plate_document())
        self.viewer = self.window.viewer
        self.original = self.window.evaluator.evaluate_raster

    def shapes(self):
        return self.window.dispatcher.document["node_data"].get("r", {}).get("shapes", [])

    def run_command(self, **settings):
        def fake(doc, target=None, frame=None, **k):
            image = np.zeros((HEIGHT, WIDTH, 4), np.float32)
            image[..., 3] = coverage(curved, int(frame))
            return _FakeRaster(image)
        self.window.evaluator.evaluate_raster = fake
        try:
            return self.viewer.shapes_from_matte(**settings)
        finally:
            self.window.evaluator.evaluate_raster = self.original

    def test_added_keys_are_reported_and_the_whole_trace_is_one_undo_step(self):
        undo = len(self.window.dispatcher.undo_stack)
        self.assertTrue(self.run_command(first=1, last=12, step=4))
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        message = self.window.statusBar().currentMessage()
        self.assertIn("plus", message)
        self.assertIn("where the in-betweens strayed over 1 px", message)
        added = len(self.viewer.last_matte_sequence.added_keys)
        self.assertGreater(added, 0)
        self.assertIn(f"plus {added} key", message)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), [])
        self.window.command({"op": "redo"})
        self.assertEqual(len(self.shapes()), 1)

    def test_a_tolerance_of_zero_adds_no_keys_and_says_nothing_of_them(self):
        self.assertTrue(self.run_command(first=1, last=12, step=4, tolerance=0))
        self.assertNotIn("plus", self.window.statusBar().currentMessage())
        self.assertEqual(self.viewer.last_matte_sequence.added_keys, [])

    def test_the_dialog_has_a_tolerance_field(self):
        self.window.rebuild_properties_dock()
        base.APP.processEvents()
        lists = self.window.findChildren(ShapeListTree)
        if not lists:
            self.skipTest("no Roto list in this layout")
        dialog = lists[0].matte_dialog()
        spin = dialog.findChild(QDoubleSpinBox, "roto-list-matte-tolerance")
        self.assertIsNotNone(spin)
        self.assertEqual(spin.value(), rotomatte.DEFAULT_TOLERANCE)
        self.assertEqual(dialog.settings()["tolerance"], rotomatte.DEFAULT_TOLERANCE)
        spin.setValue(0.0)
        self.assertEqual(dialog.settings()["tolerance"], 0.0)
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
