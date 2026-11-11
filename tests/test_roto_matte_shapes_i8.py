"""Lane 2 step I8: Shapes from matte.

A matte (a keyer's output, a generated matte, a painted alpha) is traced into editable Roto shapes: one closed Bezier shape
per connected region, one subtracting shape per hole, on the 50 % level, with the feather set from the measured edge width.
Over a frame range the shapes are keyed on every Nth frame with their points kept in correspondence, regions that split or
vanish start or end a shape, and the frames between are filled with the in-betweens fit. Accuracy is measured on the
rendered matte: every pixel on which the shapes and the matte disagree lies within 1 px of the matte's edge.
"""
import copy
import json
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings

from nodebased import roto, rotomatte, rotooutline, shapes as shape_model
from nodebased.rotolist import ShapeListTree
from tests import test_roto_edges_i1 as base
from tests.test_roto_outline_holes_i6 import disagreement
from tests.test_roto_outline_i5 import W, H, supersampled

EDGE_TOLERANCE = 1.0


def hard_matte(inside):
    return supersampled(W, H, inside).astype(np.float32)


def soft_disc(centre, radius, ramp):
    """A disc whose edge rises linearly from 0 to 1 over `ramp` px, centred on the radius (50 % at `radius`)."""
    ys, xs = np.mgrid[0:H, 0:W] + 0.5
    return np.clip(0.5 + (radius - np.hypot(xs - centre[0], ys - centre[1])) / ramp, 0.0, 1.0).astype(np.float32)


def disc(cx, cy, r):
    return lambda x, y: np.hypot(x - cx, y - cy) <= r


def render(shapes, frame=1):
    """The alpha the Roto draws for stored `shapes` at `frame`."""
    return roto.rasterise(shape_model.resolve_shapes({"shapes": shapes}, frame), W, H)[..., 3]


def stored(matte_shapes, feather=None):
    """Static Roto shapes from `matte_outlines` results."""
    out = []
    for index, item in enumerate(matte_shapes):
        value = item.feather if feather is None else feather
        out.append({"name": f"s{index}", "mode": item.mode, "opacity": 1.0, "feather": value,
                    "feather_model": "box" if value > 0 else "edge", "closed": True, "points": item.points,
                    "color": [1.0, 1.0, 1.0, 1.0], "blend_mode": "over", "source": "color"})
    return out


def radii(points, centre):
    return [math.hypot(x - centre[0], y - centre[1]) for x, y in roto.flatten_points(points, closed=True)]


def check_edges(test, rendered, inside):
    near, truth = disagreement(rendered, inside)
    test.assertLess(len(near), 0.05 * truth.sum() + 40, f"{len(near)} wrong pixels of {int(truth.sum())}")
    test.assertTrue(len(near) == 0 or near.max() <= EDGE_TOLERANCE, f"a pixel {near.max():.2f} px from the true edge disagrees")
    return near


class TraceOneMatteTests(unittest.TestCase):
    def test_a_disc(self):
        found = rotooutline.matte_outlines(hard_matte(disc(160, 140, 70)))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].mode, "union")
        self.assertGreaterEqual(len(found[0].points), rotooutline.MIN_POINTS)
        self.assertLessEqual(len(found[0].points), rotooutline.MAX_POINTS)
        values = radii(found[0].points, (160, 140))
        self.assertAlmostEqual(float(np.mean(values)), 70.0, delta=0.3)
        self.assertLess(max(values) - min(values), 1.0)
        self.assertEqual(found[0].feather, 0.0)
        self.assertAlmostEqual(found[0].area, math.pi * 70 ** 2, delta=0.01 * math.pi * 70 ** 2)
        check_edges(self, render(stored(found)), disc(160, 140, 70))

    def test_a_matte_with_a_hole(self):
        inside = lambda x, y: (np.hypot(x - 160, y - 140) <= 80) & (np.hypot(x - 160, y - 140) >= 35)
        found = rotooutline.matte_outlines(hard_matte(inside))
        self.assertEqual([s.mode for s in found], ["union", "subtract"])
        self.assertAlmostEqual(float(np.mean(radii(found[0].points, (160, 140)))), 80.0, delta=0.4)
        self.assertAlmostEqual(float(np.mean(radii(found[1].points, (160, 140)))), 35.0, delta=0.4)
        rendered = render(stored(found))
        self.assertLess(rendered[140, 160], 0.01)           # the hole is cut
        self.assertGreater(rendered[140, 217], 0.99)        # the ring is full
        check_edges(self, rendered, inside)

    def test_a_small_hole_stays_filled_and_a_speck_is_left_out(self):
        def inside(x, y):
            return (disc(110, 140, 60)(x, y) & ~disc(110, 140, 2.5)(x, y)) | disc(260, 40, 3.0)(x, y)
        found = rotooutline.matte_outlines(hard_matte(inside), min_area=64)
        self.assertEqual([s.mode for s in found], ["union"])     # the 20 px hole and the 28 px speck are under 64 px
        self.assertAlmostEqual(float(np.mean(radii(found[0].points, (110, 140)))), 60.0, delta=0.4)
        both = rotooutline.matte_outlines(hard_matte(inside), min_area=10)
        self.assertEqual(sorted(s.mode for s in both), ["subtract", "union", "union"])

    def test_two_regions(self):
        inside = lambda x, y: disc(80, 140, 50)(x, y) | disc(230, 120, 35)(x, y)
        found = rotooutline.matte_outlines(hard_matte(inside))
        self.assertEqual([s.mode for s in found], ["union", "union"])
        self.assertEqual(len({s.region for s in found}), 2)
        self.assertGreater(found[0].area, found[1].area)            # biggest first
        check_edges(self, render(stored(found)), inside)

    def test_an_island_inside_a_hole_is_a_union_again(self):
        def inside(x, y):
            d = np.hypot(x - 160, y - 140)
            return ((d <= 110) & (d >= 70)) | (d <= 30)
        found = rotooutline.matte_outlines(hard_matte(inside))
        self.assertEqual([s.mode for s in found], ["union", "subtract", "union"])
        rendered = render(stored(found))
        self.assertGreater(rendered[140, 160], 0.99)        # the island
        self.assertLess(rendered[140, 160 + 50], 0.01)      # the gap
        check_edges(self, rendered, inside)

    def test_channel_and_threshold(self):
        image = np.zeros((H, W, 4), np.float32)
        image[..., 3] = 1.0
        image[..., 1] = soft_disc((160, 140), 50, 4.0)
        green = rotooutline.matte_channel(image, "green")
        self.assertEqual(rotooutline.matte_channel(image, "alpha").min(), 1.0)
        low = rotooutline.matte_outlines(green, threshold=0.2)[0]
        high = rotooutline.matte_outlines(green, threshold=0.8)[0]
        self.assertGreater(np.mean(radii(low.points, (160, 140))), np.mean(radii(high.points, (160, 140))) + 2.0)
        # 20 % and 80 % of a 4 px linear edge sit 1.2 px either side of the 50 % level
        self.assertAlmostEqual(float(np.mean(radii(low.points, (160, 140)))), 50 + 1.2, delta=0.4)
        self.assertAlmostEqual(float(np.mean(radii(high.points, (160, 140)))), 50 - 1.2, delta=0.4)
        with self.assertRaises(rotooutline.OutlineError):
            rotooutline.matte_outlines(np.zeros((H, W), np.float32))

    def test_soft_edge_sets_the_feather_and_renders_like_the_matte(self):
        for ramp in (6.0, 10.0, 16.0):
            with self.subTest(ramp=ramp):
                matte = soft_disc((160, 140), 70, ramp)
                found = rotooutline.matte_outlines(matte)
                self.assertEqual(len(found), 1)
                shape = found[0]
                # the box feather's window is 2 * feather + 1 wide: it rises over (almost) the matte's edge width
                self.assertAlmostEqual(2 * shape.feather + 1, ramp, delta=0.2 * ramp)
                self.assertAlmostEqual(float(np.mean(radii(shape.points, (160, 140)))), 70.0, delta=0.4)
                rendered = render(stored(found))
                # the rendered edge is as wide as the matte's: 10 % to 90 % along the radius
                row, matte_row = rendered[140, 160 + 40:160 + 100], matte[140, 160 + 40:160 + 100]
                def width(profile):
                    x = np.arange(len(profile))
                    return float(np.interp(0.1, profile[::-1], x[::-1]) - np.interp(0.9, profile[::-1], x[::-1]))
                self.assertAlmostEqual(width(row), width(matte_row), delta=0.2 * width(matte_row))
                self.assertLess(float(np.abs(rendered - matte).max()), 0.2)

    def test_the_edge_width_to_feather_rule(self):
        self.assertEqual(rotooutline.box_feather(0.0), 0.0)
        self.assertEqual(rotooutline.box_feather(1.2), 0.0)
        self.assertEqual(rotooutline.box_feather(5.0), 2.0)
        self.assertEqual(rotooutline.box_feather(11.0), 5.0)

    def test_region_labels_are_four_connected(self):
        mask = np.zeros((6, 8), bool)
        mask[1:3, 1:3] = True
        mask[4, 5] = mask[3, 6] = True          # diagonal neighbours stay apart
        labels, count = rotooutline.label_regions(mask)
        self.assertEqual(count, 3)
        self.assertEqual(int((labels == labels[1, 1]).sum()), 4)

    def test_polygon_mask_matches_a_disc(self):
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        polygon = np.stack([50 + 20 * np.cos(t), 40 + 20 * np.sin(t)], axis=1)
        mask = rotooutline.polygon_mask(polygon, 20, 10, 60, 60)
        self.assertAlmostEqual(float(mask.sum()), math.pi * 400, delta=12)
        self.assertTrue(mask[40 - 10, 50 - 20])


# -- a moving matte ---------------------------------------------------------------------------------

FRAMES = range(1, 13)


def ellipse(frame):
    cx, cy = 100 + 6 * frame + 0.1 * frame ** 2, 130 + 1.5 * frame - 0.2 * frame ** 2
    rx, ry = 62 + frame, 34
    return lambda x, y: ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


_MATTES = {}


def moving_matte(frame):
    if frame not in _MATTES:
        _MATTES[frame] = hard_matte(ellipse(frame))
    return _MATTES[frame]


class MovingMatteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shapes, cls.sequence = rotomatte.shapes_from_matte(moving_matte, 1, 12, step=4)

    def test_one_shape_keyed_every_fourth_frame(self):
        self.assertEqual(len(self.shapes), 1)
        self.assertEqual(self.sequence.key_frames, [1, 5, 9, 12])
        self.assertEqual(self.sequence.events, [])
        self.assertEqual(self.shapes[0]["lifetime"], {"type": "range", "start": 1, "end": 12})
        keys = sorted({k["frame"] for p in self.shapes[0]["points"] for f in ("x", "y")
                       for k in p[f]["curve"]["keys"] if isinstance(p[f], dict)})
        for k in (1, 5, 9, 12):
            self.assertIn(k, keys)
        self.assertGreaterEqual(len(self.shapes[0]["points"]), rotooutline.MIN_POINTS)
        self.assertLessEqual(len(self.shapes[0]["points"]), rotooutline.MAX_POINTS)

    def test_every_frame_matches_the_matte_within_a_pixel(self):
        worst = 0.0
        for frame in FRAMES:
            near = check_edges(self, render(self.shapes, frame), ellipse(frame))
            worst = max(worst, float(near.max()) if len(near) else 0.0)
        print(f"moving matte: worst disagreeing pixel {worst:.2f} px from the edge over {len(FRAMES)} frames")

    def test_point_zero_stays_on_the_same_side(self):
        angles = []
        for frame in FRAMES:
            resolved = shape_model.resolve_shapes({"shapes": self.shapes}, frame)[0]
            outline = np.array(roto.flatten_points(resolved["points"], closed=True))
            centre = outline.mean(axis=0)
            p0 = resolved["points"][0]
            angles.append(math.degrees(math.atan2(p0["y"] - centre[1], p0["x"] - centre[0])))
        spread = max(angles) - min(angles)
        self.assertLess(spread, 15.0, f"point 0 moved around the shape: {np.round(angles, 1)}")

    def test_points_do_not_slide_between_keys(self):
        """The same point sits at the same share of the way round the outline on each key frame."""
        track = self.sequence.tracks[0]
        shares = np.array([track.fractions])
        self.assertEqual(len(track.keys[1]), len(track.keys[5]))
        self.assertEqual(len(track.keys[5]), len(track.keys[12]))
        first = np.array(rotomatte._arc_fractions(track.keys[1], track.polygons[1]))
        last = np.array(rotomatte._arc_fractions(track.keys[12], track.polygons[12]))
        difference = np.abs(((last - first) + 0.5) % 1.0 - 0.5)
        self.assertLess(float(difference.max()), 0.08, "a point slid along the outline between the first and last key")
        del shares

    def test_the_fit_error_on_keys_is_small(self):
        for track in self.sequence.tracks:
            for frame, error in track.errors.items():
                self.assertLess(error, 1.0, f"frame {frame}")

    def test_the_fill_keeps_a_frame_the_interpolation_already_fits(self):
        plain, _ = rotomatte.shapes_from_matte(moving_matte, 1, 12, step=4, fill=False)
        self.assertEqual(plain, self.shapes)       # the interpolation is within 1 px, so no in-between gets a key


def accelerating(frame):
    cx, cy = 60 + 1.6 * (frame - 1) ** 2, 140 + 30 * math.sin((frame - 1) * 0.5)
    return lambda x, y: ((x - cx) / 45) ** 2 + ((y - cy) / 38) ** 2 <= 1.0


class FillTests(unittest.TestCase):
    """A matte that accelerates: the interpolation between keys misses by several pixels and the fill pulls it back."""

    @staticmethod
    def worst(shapes, frames):
        out = []
        for frame in frames:
            near, _ = disagreement(render(shapes, frame), accelerating(frame))
            out.append(float(near.max()) if len(near) else 0.0)
        return out

    def test_the_fill_pulls_the_in_betweens_onto_the_matte(self):
        matte = lambda frame: hard_matte(accelerating(frame))
        plain, _ = rotomatte.shapes_from_matte(matte, 1, 5, step=4, fill=False)
        filled, _ = rotomatte.shapes_from_matte(matte, 1, 5, step=4)
        before, after = self.worst(plain, (2, 3, 4)), self.worst(filled, (2, 3, 4))
        print(f"fill: worst disagreeing pixel {max(before):.1f} px without, {max(after):.2f} px with")
        self.assertGreater(max(before), 4.0)
        self.assertLessEqual(max(after), EDGE_TOLERANCE)

    def test_fast_motion_is_followed_frame_by_frame(self):
        """Between keys 5 and 9 the matte jumps further than its own width: no overlap, so the gap is walked frame by frame."""
        shapes, sequence = rotomatte.shapes_from_matte(lambda frame: hard_matte(accelerating(frame)), 1, 12, step=4)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(sequence.events, [])
        self.assertTrue({1, 5, 6, 7, 8, 9, 12} <= set(sequence.key_frames))
        for frame in range(1, 13):
            check_edges(self, render(shapes, frame), accelerating(frame))


class SplitTests(unittest.TestCase):
    """Two discs that touch for a while, then part: a region splits and a second shape starts."""

    @staticmethod
    def inside(frame):
        distance = 40 + 4 * (frame - 1)
        return lambda x, y: disc(160 - distance / 2, 140, 30)(x, y) | disc(160 + distance / 2, 140, 30)(x, y)

    def matte(self, frame):
        return hard_matte(self.inside(frame))

    def test_a_region_that_splits(self):
        shapes, sequence = rotomatte.shapes_from_matte(self.matte, 1, 12, step=4)
        self.assertEqual(len(shapes), 2)
        kinds = [(e.kind, e.frame) for e in sequence.events]
        self.assertEqual(kinds, [("split", 7)], "the discs part on frame 7 (their edges touch at 6)")
        self.assertIn("split off", str(sequence.events[0]))
        self.assertIn("frame 7", str(sequence.events[0]))
        lifetimes = sorted((s["lifetime"]["start"], s["lifetime"]["end"]) for s in shapes)
        self.assertEqual(lifetimes, [(1, 12), (7, 12)])
        # the new shape exists only from the frame it appeared on
        late = next(s for s in shapes if s["lifetime"]["start"] == 7)
        resolved = shape_model.resolve_shapes({"shapes": shapes}, 6)
        self.assertFalse(resolved[shapes.index(late)].get("visible", True))
        for frame in range(1, 13):
            check_edges(self, render(shapes, frame), self.inside(frame))

    def test_a_region_that_vanishes_and_one_that_appears(self):
        def inside(frame):
            def f(x, y):
                out = disc(90, 140, 45)(x, y)
                if frame <= 6:
                    out = out | disc(240, 140, 30)(x, y)           # vanishes after frame 6
                if frame >= 9:
                    out = out | disc(240, 60, 25)(x, y)            # appears on frame 9
                return out
            return f
        shapes, sequence = rotomatte.shapes_from_matte(lambda frame: hard_matte(inside(frame)), 1, 12, step=4)
        events = {(e.kind, e.frame) for e in sequence.events}
        self.assertEqual(events, {("vanished", 7), ("appeared", 9)})
        lifetimes = sorted((s["lifetime"]["start"], s["lifetime"]["end"]) for s in shapes)
        self.assertEqual(lifetimes, [(1, 6), (1, 12), (9, 12)])
        for frame in range(1, 13):
            check_edges(self, render(shapes, frame), inside(frame))

    def test_two_regions_that_merge(self):
        shapes, sequence = rotomatte.shapes_from_matte(lambda frame: self.matte(13 - frame), 1, 12, step=4)
        merges = [e for e in sequence.events if e.kind == "merged"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0].frame, 7)          # frame 7 shows the discs touching (the split test's frame 6)
        self.assertEqual(len(shapes), 2)
        self.assertEqual(sorted(s["lifetime"]["end"] for s in shapes), [6, 12])


class NamesAndDefaultsTests(unittest.TestCase):
    def test_names_avoid_existing_shapes(self):
        shapes, _ = rotomatte.shapes_from_matte(lambda frame: hard_matte(disc(160, 140, 60)), 1, 1, taken_names={"matte1", "matte2"})
        self.assertEqual([s["name"] for s in shapes], ["matte3"])

    def test_a_single_frame_is_static(self):
        shapes, sequence = rotomatte.shapes_from_matte(lambda frame: hard_matte(disc(160, 140, 60)), 3, 3)
        self.assertEqual(sequence.key_frames, [3])
        self.assertTrue(all(isinstance(p[f], float) for p in shapes[0]["points"] for f in ("x", "y")))
        self.assertEqual(shapes[0]["lifetime"], {"type": "range", "start": 3, "end": 3})

    def test_the_stated_defaults(self):
        self.assertEqual(rotomatte.DEFAULT_STEP, 4)
        self.assertEqual(rotooutline.MATTE_LEVEL, 0.5)
        self.assertEqual(rotooutline.MATTE_MIN_AREA, 64)


# -- through the viewer ---------------------------------------------------------------------------------

class _FakeRaster:
    def __init__(self, image):
        self.image = image

    def to_display(self):
        return self.image


def matte_image(frame, inside_for=ellipse):
    image = np.zeros((H, W, 4), np.float32)
    image[..., 3] = hard_matte(inside_for(frame))
    return image


class ViewerMatteTests(unittest.TestCase):
    open = base.ViewerEdgeTests.open
    tearDown = base.ViewerEdgeTests.tearDown

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.open(base.disc_plate_document())
        self.viewer = self.window.viewer
        self.original = self.window.evaluator.evaluate_raster

    def shapes(self):
        return self.window.dispatcher.document["node_data"].get("r", {}).get("shapes", [])

    def run_command(self, inside_for=ellipse, **settings):
        self.window.evaluator.evaluate_raster = lambda doc, target=None, frame=None, **k: _FakeRaster(
            matte_image(int(frame), inside_for))
        try:
            return self.viewer.shapes_from_matte(**settings)
        finally:
            self.window.evaluator.evaluate_raster = self.original

    def test_the_command_adds_the_shapes_in_one_undo_step(self):
        undo = len(self.window.dispatcher.undo_stack)
        self.assertTrue(self.run_command(first=1, last=12, step=4))
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        self.assertEqual([s["name"] for s in self.shapes()], ["matte1"])
        keys = {k["frame"] for p in self.shapes()[0]["points"] for k in p["x"]["curve"]["keys"]}
        self.assertTrue({1, 5, 9, 12} <= keys)
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), [])
        self.window.command({"op": "redo"})
        self.assertEqual(len(self.shapes()), 1)

    def test_a_split_is_named_in_the_status_bar(self):
        self.assertTrue(self.run_command(SplitTests.inside, first=1, last=12, step=4))
        message = self.window.statusBar().currentMessage()
        self.assertIn("split off matte1 at frame 7", message)
        self.assertEqual(len(self.shapes()), 2)

    def test_no_input_changes_nothing(self):
        self.window.dispatcher.execute({"op": "disconnect", "id": "r", "input": "bg"}) if False else None
        undo = len(self.window.dispatcher.undo_stack)
        before = copy.deepcopy(self.window.dispatcher.document)
        self.assertFalse(self.run_command(source="mask", first=1, last=3))
        self.assertIn("Mask input", self.window.statusBar().currentMessage())
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo)
        self.assertEqual(self.window.dispatcher.document, before)

    def test_an_empty_matte_reports_and_changes_nothing(self):
        self.assertFalse(self.run_command(lambda frame: (lambda x, y: np.zeros_like(x, bool)), first=1, last=3))
        self.assertIn("no region", self.window.statusBar().currentMessage())
        self.assertEqual(self.shapes(), [])

    def test_the_menu_and_dialog_change_nothing_until_run(self):
        self.window.rebuild_properties_dock()
        base.APP.processEvents()
        lists = self.window.findChildren(ShapeListTree)
        if not lists:
            self.skipTest("no Roto list in this layout")
        before = json.dumps(self.window.dispatcher.document, sort_keys=True)
        names = [a.objectName() for a in lists[0].build_menu().actions()]
        self.assertIn("roto-list-shapes-from-matte", names)
        dialog = lists[0].matte_dialog()
        settings = dialog.settings()
        self.assertEqual((settings["source"], settings["channel"], settings["threshold"], settings["min_area"], settings["step"]),
                         ("bg", "alpha", 0.5, 64, 4))
        self.assertEqual({d.objectName() for d in dialog.findChildren(type(dialog.findChild(object, "roto-list-matte-source")))
                          if d.objectName().startswith("roto-list-matte")} >= {"roto-list-matte-source", "roto-list-matte-channel"}, True)
        dialog.deleteLater()
        self.assertEqual(json.dumps(self.window.dispatcher.document, sort_keys=True), before)

    def test_saved_documents_still_render_the_same(self):
        """The command adds no parameter and no schema field: a Roto without the command's shapes renders as before."""
        evaluated = self.window.evaluator.evaluate_raster(self.window.dispatcher.document, target="r", frame=2)
        before = evaluated.to_display().copy()
        self.run_command(first=1, last=3, step=2)
        self.window.command({"op": "undo"})
        again = self.window.evaluator.evaluate_raster(self.window.dispatcher.document, target="r", frame=2).to_display()
        self.assertTrue(np.array_equal(before, again))


if __name__ == "__main__":
    unittest.main()
