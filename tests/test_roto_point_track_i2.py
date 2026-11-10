"""Roto point tracking that follows a deforming shape (lane 9, step I2)."""
import copy
import math
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import rotopointtrack as rp
from nodebased import shapes as shape_model

WIDTH, HEIGHT = 200, 160
CENTRE = (80.0, 80.0)
RADIUS = 40.0
FRAMES = list(range(2, 12))            # frame 1 is the reference
STRETCH_RATE = 0.07                    # the right half grows this much of its reference width per frame


def texture(u, v):
    return (0.5 + 0.12 * np.sin(0.61 * u + 0.2) + 0.12 * np.cos(0.47 * v + 1.1) + 0.10 * np.sin(0.33 * (u + v) + 0.7)
            + 0.09 * np.cos(0.52 * (u - 1.3 * v)) + 0.08 * np.sin(0.83 * u - 0.29 * v + 2.0)
            + 0.06 * np.cos(0.71 * (u + 0.4 * v)))


def disc_frame(frame, mode, width=WIDTH, height=HEIGHT, x0=0, y0=0):
    """A textured disc on a flat ground; texture and edge are analytic in the disc's own coordinates, so the truth is exact."""
    t = frame - 1
    ys, xs = np.mgrid[y0:y0 + height, x0:x0 + width].astype(np.float64)
    x, y = xs + 0.5, ys + 0.5
    if mode == "translate":
        u, v = x - 1.3 * t, y - 0.7 * t
    else:
        u, v = np.where(x > CENTRE[0], CENTRE[0] + (x - CENTRE[0]) / (1 + STRETCH_RATE * t), x), y
    coverage = np.clip(RADIUS - np.hypot(u - CENTRE[0], v - CENTRE[1]) + 0.5, 0.0, 1.0)
    return 0.2 * (1 - coverage) + coverage * texture(u, v)


def circle_points(count=24):
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        points.append({"x": CENTRE[0] + RADIUS * math.cos(a), "y": CENTRE[1] + RADIUS * math.sin(a),
                       "in_x": 5.0 * math.sin(a), "in_y": -5.0 * math.cos(a),
                       "out_x": -5.0 * math.sin(a), "out_y": 5.0 * math.cos(a),
                       "feather_x": 3.0 * math.cos(a), "feather_y": 3.0 * math.sin(a)})
    return points


def truth(point, frame, mode):
    t = frame - 1
    if mode == "translate":
        return point["x"] + 1.3 * t, point["y"] + 0.7 * t
    if point["x"] > CENTRE[0]:
        return CENTRE[0] + (point["x"] - CENTRE[0]) * (1 + STRETCH_RATE * t), point["y"]
    return point["x"], point["y"]


def source(mode, occlude=None):
    cache = {}

    def get(frame):
        if frame not in cache:
            image = disc_frame(frame, mode)
            if occlude is not None:
                rect = occlude(frame)
                if rect is not None:
                    x0, y0, x1, y1 = rect
                    image[y0:y1, x0:x1] = 0.5
            cache[frame] = image
        return cache[frame]
    return get


class Sequences:
    """The planar prediction of each sequence, computed once."""
    translate = stretch = None
    points = circle_points()

    @classmethod
    def prepare(cls):
        if cls.translate is None:
            cls.translate_source = source("translate")
            cls.stretch_source = source("stretch")
            cls.translate = rp.planar_matrices(cls.translate_source, cls.points, 1, FRAMES)
            cls.stretch = rp.planar_matrices(cls.stretch_source, cls.points, 1, FRAMES)


def planar_point(matrix, point):
    return rp._project(matrix, np.array([[point["x"], point["y"]]]))[0]


class PointTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Sequences.prepare()

    def test_a_translating_disc_is_followed_within_half_a_pixel_over_ten_frames(self):
        points = Sequences.points
        result = rp.track_points(Sequences.translate_source, points, 1, FRAMES, planar=Sequences.translate)
        self.assertEqual(sorted(result), FRAMES)
        worst = 0.0
        for frame in FRAMES:
            for i, point in enumerate(points):
                tx, ty = truth(point, frame, "translate")
                worst = max(worst, math.hypot(result[frame]["points"][i]["x"] - tx, result[frame]["points"][i]["y"] - ty))
            self.assertFalse(any(result[frame]["flagged"]), frame)
            self.assertGreater(min(result[frame]["confidence"]), 0.9)
        print(f"\ntranslate: worst point error {worst:.3f} px over {len(FRAMES)} frames")
        self.assertLess(worst, 0.5)

    def test_the_right_half_stretching_is_followed_where_the_planar_track_misses(self):
        points = Sequences.points
        result = rp.track_points(Sequences.stretch_source, points, 1, FRAMES, planar=Sequences.stretch)
        right = [i for i, p in enumerate(points) if p["x"] > CENTRE[0] + 5]
        own = planar = 0.0
        for frame in FRAMES:
            for i in right:
                tx, ty = truth(points[i], frame, "stretch")
                q = result[frame]["points"][i]
                own = max(own, math.hypot(q["x"] - tx, q["y"] - ty))
                px, py = planar_point(Sequences.stretch[frame], points[i])
                planar = max(planar, math.hypot(px - tx, py - ty))
        print(f"\nstretch: right-side point error {own:.3f} px; planar-only error {planar:.3f} px")
        self.assertLess(own, 1.0)
        self.assertGreater(planar, 3.0)

    def test_the_left_half_of_the_stretching_disc_stays_put(self):
        points = Sequences.points
        result = rp.track_points(Sequences.stretch_source, points, 1, FRAMES, planar=Sequences.stretch)
        left = [i for i, p in enumerate(points) if p["x"] < CENTRE[0] - 5]
        for frame in FRAMES:
            for i in left:
                q = result[frame]["points"][i]
                self.assertLess(math.hypot(q["x"] - points[i]["x"], q["y"] - points[i]["y"]), 1.0, (frame, i))

    def test_an_occluded_patch_loses_confidence_and_falls_back_to_the_planar_prediction(self):
        points = Sequences.points
        victim = 0                                       # the rightmost point
        first_hidden = 6

        def occluder(frame):
            if frame < first_hidden:
                return None
            x, y = truth(points[victim], frame, "translate")
            return (int(x) - 30, int(y) - 30, int(x) + 31, int(y) + 31)
        hidden = source("translate", occluder)
        result = rp.track_points(hidden, points, 1, FRAMES, planar=Sequences.translate)
        for frame in FRAMES:
            confidence, flagged = result[frame]["confidence"][victim], result[frame]["flagged"][victim]
            if frame < first_hidden:
                self.assertGreater(confidence, rp.THRESHOLD, frame)
                self.assertFalse(flagged, frame)
            else:
                self.assertLess(confidence, rp.THRESHOLD, frame)
                self.assertTrue(flagged, frame)
                fallback = planar_point(Sequences.translate[frame], points[victim])
                q = result[frame]["points"][victim]
                self.assertLess(math.hypot(q["x"] - fallback[0], q["y"] - fallback[1]), 1e-6, frame)
            # every other point is untouched by the occlusion
            others = [i for i in range(len(points)) if i != victim and math.hypot(
                points[i]["x"] - points[victim]["x"], points[i]["y"] - points[victim]["y"]) > 45]
            self.assertFalse(any(result[frame]["flagged"][i] for i in others), frame)

    def test_a_point_picks_up_its_patch_again_when_the_occluder_goes(self):
        points = Sequences.points
        victim = 0

        def occluder(frame):
            if 5 <= frame <= 7:
                x, y = truth(points[victim], frame, "translate")
                return (int(x) - 30, int(y) - 30, int(x) + 31, int(y) + 31)
            return None
        result = rp.track_points(source("translate", occluder), points, 1, FRAMES, planar=Sequences.translate)
        self.assertTrue(all(result[f]["flagged"][victim] for f in (5, 6, 7)))
        for frame in (8, 9, 10, 11):
            self.assertFalse(result[frame]["flagged"][victim], frame)
            tx, ty = truth(points[victim], frame, "translate")
            q = result[frame]["points"][victim]
            self.assertLess(math.hypot(q["x"] - tx, q["y"] - ty), 0.6, frame)

    def test_tangents_and_feather_offsets_move_with_their_point(self):
        points = Sequences.points
        result = rp.track_points(Sequences.translate_source, points, 1, FRAMES, planar=Sequences.translate)
        for i, point in enumerate(points):
            moved = result[11]["points"][i]
            for name in ("in_x", "in_y", "out_x", "out_y", "feather_x", "feather_y"):
                self.assertAlmostEqual(moved[name], point[name], delta=0.3, msg=(i, name))

    def test_offsets_turn_with_the_planar_rotation(self):
        points = circle_points(8)
        angle = math.radians(30)
        matrix = np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0, 0, 1.0]])
        flat = lambda frame: np.full((HEIGHT, WIDTH), 0.4)          # no texture: every point falls back to the planar map
        result = rp.track_points(flat, points, 1, [2], planar={2: matrix})
        for i, point in enumerate(points):
            moved = result[2]["points"][i]
            self.assertAlmostEqual(moved["out_x"], math.cos(angle) * point["out_x"] - math.sin(angle) * point["out_y"], places=3)
            self.assertAlmostEqual(moved["out_y"], math.sin(angle) * point["out_x"] + math.cos(angle) * point["out_y"], places=3)
            self.assertTrue(result[2]["flagged"][i])

    def test_progress_and_cancel(self):
        points = Sequences.points
        seen = []
        rp.track_points(Sequences.translate_source, points, 1, FRAMES[:3], planar=Sequences.translate,
                        progress=lambda frame, step, total: seen.append((frame, step, total)))
        self.assertEqual(seen, [(2, 1, 3), (3, 2, 3), (4, 3, 3)])

        class Stop:
            @staticmethod
            def is_set():
                return True
        from concurrent.futures import CancelledError
        with self.assertRaises(CancelledError):
            rp.track_points(Sequences.translate_source, points, 1, FRAMES, cancel=Stop())

    def test_a_backward_pass_follows_the_disc_to_earlier_frames(self):
        points = circle_points(24)
        reference = 11
        order = rp.frame_order(reference, 1, 11, "backward")[0]
        self.assertEqual(order, list(range(10, 0, -1)))
        # the same texture run in reverse: the reference is the last frame, the disc moves the other way
        shifted = [{**p, "x": p["x"] + 1.3 * 10, "y": p["y"] + 0.7 * 10} for p in points]
        result = rp.track_points(Sequences.translate_source, shifted, reference, order)
        for frame in order:
            for i, point in enumerate(points):
                tx, ty = truth(point, frame, "translate")
                q = result[frame]["points"][i]
                self.assertLess(math.hypot(q["x"] - tx, q["y"] - ty), 0.6, (frame, i))


class HelperTests(unittest.TestCase):
    def test_frame_order(self):
        self.assertEqual(rp.frame_order(5, 1, 8, "forward"), [[6, 7, 8]])
        self.assertEqual(rp.frame_order(5, 1, 8, "backward"), [[4, 3, 2, 1]])
        self.assertEqual(rp.frame_order(5, 3, 7, "both"), [[6, 7], [4, 3]])
        self.assertEqual(rp.frame_order(5, 5, 5, "forward"), [[]])
        with self.assertRaises(rp.PointTrackError):
            rp.frame_order(5, 1, 8, "sideways")
        with self.assertRaises(rp.PointTrackError):
            rp.frame_order(5, 8, 1)

    def test_too_few_points(self):
        with self.assertRaises(rp.PointTrackError):
            rp.track_points(lambda f: np.zeros((20, 20)), [{"x": 5.0, "y": 5.0}], 1, [2])

    def test_smoothing_pulls_toward_neighbours_and_leaves_unweighted_points_alone(self):
        offsets = [(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        smoothed = rp.smooth_offsets(offsets, [1, 1, 1, 0], True, 0.5)
        self.assertAlmostEqual(smoothed[1][0], 1.0)
        self.assertAlmostEqual(smoothed[0][0], 1.0)         # neighbours: point 1 (2.0) and point 3 (weight 0, ignored): half way to 2.0
        np.testing.assert_allclose(smoothed[3], offsets[3])
        np.testing.assert_allclose(rp.smooth_offsets(offsets, [1, 1, 1, 1], True, 0.0), offsets)
        # an open shape's ends have one neighbour
        open_shape = rp.smooth_offsets([(4.0, 0.0), (0.0, 0.0), (0.0, 0.0)], [1, 1, 1], False, 1.0)
        self.assertAlmostEqual(open_shape[0][0], 0.0)

    def test_keys_follow_the_autokey_rules(self):
        keyed = rp.key_scalar(10.0, {3: 11.0, 4: 12.0}, 2, auto_key=True)
        self.assertEqual([(k["frame"], k["value"]) for k in keyed["curve"]["keys"]], [(2, 10.0), (3, 11.0), (4, 12.0)])
        self.assertEqual(rp.key_scalar(10.0, {3: 11.0}, 2, auto_key=False), 10.0)
        again = rp.key_scalar(keyed, {4: 20.0, 5: 21.0}, 2, auto_key=False)
        self.assertEqual([(k["frame"], k["value"]) for k in again["curve"]["keys"]], [(2, 10.0), (3, 11.0), (4, 20.0), (5, 21.0)])
        self.assertEqual(keyed["curve"]["keys"][2]["value"], 12.0)          # the original is not mutated
        self.assertEqual(rp.key_scalar(7.0, {}, 1), 7.0)


class TimingTests(unittest.TestCase):
    def test_a_24_point_shape_at_1080p_tracks_in_well_under_a_second_a_frame(self):
        width, height = 1920, 1080
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
        shifted = {}

        def frame(f):
            if f not in shifted:
                u, v = xs + 0.5 - 1.1 * (f - 1), ys + 0.5 - 0.6 * (f - 1)
                shifted[f] = texture(u * 0.9, v * 0.9)
            return shifted[f]
        points = [{"x": 960 + 300 * math.cos(2 * math.pi * i / 24), "y": 540 + 300 * math.sin(2 * math.pi * i / 24)}
                  for i in range(24)]
        frame(1), frame(2), frame(3), frame(4)
        start = time.perf_counter()
        result = rp.track_points(frame, points, 1, [2, 3, 4])
        per_frame = (time.perf_counter() - start) / 3
        print(f"\n24 points at 1080p: {per_frame * 1000:.0f} ms per frame (frames pre-rendered)")
        self.assertFalse(any(any(result[f]["flagged"]) for f in (2, 3, 4)))
        self.assertLess(per_frame, 1.0)


# -- group 2: in the app ------------------------------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QSettings

from nodebased.app import Window
from nodebased.core import Dispatcher, empty_document
from nodebased.tileexec import TileExecutor
from tests.test_roto_ui import APP
from tests.waiting import wait_until


def roto_shape(name="disc", count=24, **extra):
    shape = {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
             "points": [{k: float(p[k]) for k in ("x", "y", "in_x", "in_y", "out_x", "out_y")} for p in circle_points(count)]}
    shape.update(extra)
    return shape


class WindowCase(unittest.TestCase):
    """A Roto node over a plate that the stretching-disc frames replace (the compositor call is patched, as the other tracking tests do)."""
    mode = "stretch"
    occlusions = None             # {frame: (x0, y0, x1, y1)} painted flat over the plate

    def make_window(self, shapes, last=11, current=1, node_type="Roto", autokey=True, view=False):
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "roto", "type": node_type})
        dispatcher.execute({"op": "connect", "id": "roto", "input": "bg" if node_type == "Roto" else "image", "source": "plate"})
        dispatcher.document["time"].update(first=1, last=last, current=current)
        if node_type == "Roto":
            dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": shapes})
        else:
            dispatcher.execute({"op": "set_paint_items", "id": "roto", "items": [{"kind": "shape", "blend": "over", "visible": True, **s} for s in shapes]})
        if view:
            dispatcher.execute({"op": "view", "id": "roto"})
        if not autokey:
            dispatcher.execute({"op": "set_roto_tool_settings", "id": "roto", "tool": "__global__", "settings": {"autokey": False}})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        return window

    def compose(self, _executor, _doc, _source, region, frame, **_kwargs):
        gray = disc_frame(frame, self.mode, region.width, region.height, region.x, region.y)
        rect = (self.occlusions or {}).get(frame)
        if rect is not None:
            x0, y0, x1, y1 = rect
            gray[max(0, y0 - region.y):max(0, y1 - region.y), max(0, x0 - region.x):max(0, x1 - region.x)] = 0.5
        rgba = np.stack(np.broadcast_arrays(gray, gray, gray, np.ones_like(gray)), axis=-1).astype(np.float32)
        return SimpleNamespace(pixels=rgba)

    def run_tracking(self, window, *args, **kwargs):
        with patch.object(TileExecutor, "compose_region", lambda *a, **k: self.compose(*a, **k)):
            self.assertTrue(window.track_roto_points("roto", *args, **kwargs))
            self.assertTrue(wait_until(lambda: window._points_future is None, timeout=120))

    def shapes(self, window):
        payload = window.dispatcher.document["node_data"]["roto"]
        return payload["shapes"] if "shapes" in payload else [i for i in payload["items"] if i.get("kind") == "shape"]

    def resolved(self, window, frame, index=0):
        payload = window.dispatcher.document["node_data"]["roto"]
        shapes = self.shapes(window)
        return shape_model.resolve_shapes({"shapes": shapes}, frame)[index]["points"]


class TrackPointsInTheAppTests(WindowCase):
    def test_keys_land_on_every_frame_and_follow_the_stretch(self):
        window = self.make_window([roto_shape()])
        window.viewer.roto_selected_shape_index = 0
        undo = len(window.dispatcher.undo_stack)
        self.run_tracking(window)
        self.assertEqual(len(window.dispatcher.undo_stack), undo + 1)                 # one undo step for the whole run
        points = circle_points(24)
        for frame in range(1, 12):
            resolved = self.resolved(window, frame)
            for i, point in enumerate(points):
                tx, ty = truth(point, frame, "stretch")
                limit = 1.0 if point["x"] > CENTRE[0] + 5 else 1.2
                if abs(point["x"] - CENTRE[0]) <= 5:
                    continue
                self.assertLess(math.hypot(resolved[i]["x"] - tx, resolved[i]["y"] - ty), limit, (frame, i))
        # every point carries a key on each tracked frame, plus the reference frame
        keys = self.shapes(window)[0]["points"][0]["x"]["curve"]["keys"]
        self.assertEqual([k["frame"] for k in keys], list(range(1, 12)))

    def test_the_app_matches_the_tracker_module_run_by_itself(self):
        window = self.make_window([roto_shape()])
        window.viewer.roto_selected_shape_index = 0
        self.run_tracking(window)
        Sequences.prepare()
        direct = rp.track_points(Sequences.stretch_source, Sequences.points, 1, FRAMES, planar=Sequences.stretch)
        worst = 0.0
        for frame in FRAMES:
            resolved = self.resolved(window, frame)
            for i in range(24):
                worst = max(worst, math.hypot(resolved[i]["x"] - direct[frame]["points"][i]["x"],
                                              resolved[i]["y"] - direct[frame]["points"][i]["y"]))
        print(f"\nin the app against the module run alone: worst difference {worst:.3f} px")
        self.assertLess(worst, 0.5)

    def test_one_undo_removes_every_key(self):
        window = self.make_window([roto_shape()])
        window.viewer.roto_selected_shape_index = 0
        before = copy.deepcopy(window.dispatcher.document["node_data"]["roto"])
        self.run_tracking(window)
        self.assertNotEqual(window.dispatcher.document["node_data"]["roto"], before)
        window.command({"op": "undo"})
        self.assertEqual(window.dispatcher.document["node_data"]["roto"], before)
        window.command({"op": "redo"})
        self.assertIn("curve", self.shapes(window)[0]["points"][0]["x"])

    def test_flags_appear_only_on_the_occluded_frames_and_are_named_in_the_status_bar(self):
        victim = 0
        self.mode = "translate"
        self.occlusions = {}
        for frame in (6, 7, 8):
            x, y = truth(circle_points(24)[victim], frame, "translate")
            self.occlusions[frame] = (int(x) - 30, int(y) - 30, int(x) + 31, int(y) + 31)
        window = self.make_window([roto_shape()])
        window.viewer.roto_selected_shape_index = 0
        self.run_tracking(window)
        for frame in range(1, 12):
            flagged = window.viewer.roto_flagged_points("roto", frame, 0, 24)
            self.assertEqual(victim in flagged, frame in (6, 7, 8), frame)
            far = {i for i, p in enumerate(circle_points(24)) if math.hypot(p["x"] - circle_points(24)[victim]["x"],
                                                                            p["y"] - circle_points(24)[victim]["y"]) > 45}
            self.assertFalse(flagged & far, frame)
        message = window.statusBar().currentMessage()
        self.assertIn("low-confidence", message)
        self.assertIn("disc point 1 (frames 6–8)", message)
        # the flags live in the viewer only: the document holds no trace of them
        self.assertNotIn("flag", repr(window.dispatcher.document["node_data"]["roto"]).lower())
        # and they follow the point count: an edited shape draws none
        self.assertEqual(window.viewer.roto_flagged_points("roto", 6, 0, 23), frozenset())

    def test_locked_shapes_are_skipped(self):
        window = self.make_window([roto_shape("free"), roto_shape("locked", locked=True)])
        locked_before = copy.deepcopy(self.shapes(window)[1])
        self.run_tracking(window, shape_indices=[0, 1])
        self.assertEqual(self.shapes(window)[1], locked_before)
        self.assertIn("curve", self.shapes(window)[0]["points"][0]["x"])
        window2 = self.make_window([roto_shape("locked", locked=True)])
        window2.viewer.roto_selected_shape_index = 0
        with patch.object(TileExecutor, "compose_region", lambda *a, **k: self.compose(*a, **k)):
            self.assertFalse(window2.track_roto_points("roto"))

    def test_autokey_off_leaves_a_static_shape_alone_but_updates_a_keyed_one(self):
        window = self.make_window([roto_shape()], autokey=False)
        before = copy.deepcopy(window.dispatcher.document["node_data"]["roto"])
        undo = len(window.dispatcher.undo_stack)
        self.run_tracking(window, shape_indices=[0])
        self.assertEqual(window.dispatcher.document["node_data"]["roto"], before)
        self.assertEqual(len(window.dispatcher.undo_stack), undo)
        self.assertIn("Autokey", window.statusBar().currentMessage())
        # key the shape at the reference frame, then the run writes its keys with Autokey still off
        shapes = copy.deepcopy(before["shapes"])
        for point in shapes[0]["points"]:
            for name in ("x", "y"):
                point[name] = {"value": point[name], "curve": {"interpolation": "smooth", "keys": [{"frame": 1, "value": point[name]}]}}
        window.command({"op": "set_shapes", "id": "roto", "shapes": shapes})
        self.run_tracking(window, shape_indices=[0])
        keys = self.shapes(window)[0]["points"][0]["x"]["curve"]["keys"]
        self.assertEqual([k["frame"] for k in keys], list(range(1, 12)))

    def test_backward_tracking_from_the_last_frame(self):
        self.mode = "translate"
        points = circle_points(24)
        moved = [{k: float(p[k]) + (1.3 * 10 if k == "x" else 0.7 * 10 if k == "y" else 0.0)
                  for k in ("x", "y", "in_x", "in_y", "out_x", "out_y")} for p in points]
        for m, p in zip(moved, points):
            m["in_x"], m["in_y"], m["out_x"], m["out_y"] = p["in_x"], p["in_y"], p["out_x"], p["out_y"]
        window = self.make_window([{**roto_shape(), "points": moved}], current=11)
        window.viewer.roto_selected_shape_index = 0
        self.run_tracking(window, "backward")
        for frame in (10, 6, 1):
            resolved = self.resolved(window, frame)
            for i, point in enumerate(points):
                tx, ty = truth(point, frame, "translate")
                self.assertLess(math.hypot(resolved[i]["x"] - tx, resolved[i]["y"] - ty), 0.7, (frame, i))

    def test_forward_only_leaves_earlier_frames_alone(self):
        window = self.make_window([roto_shape()], current=4)
        window.viewer.roto_selected_shape_index = 0
        self.mode = "translate"
        self.run_tracking(window, "forward")
        keys = [k["frame"] for k in self.shapes(window)[0]["points"][0]["x"]["curve"]["keys"]]
        self.assertEqual(keys, list(range(4, 12)))

    def test_a_roto_paint_shape_is_tracked_through_its_items(self):
        self.mode = "translate"
        window = self.make_window([roto_shape()], node_type="RotoPaint")
        item_index = 0
        window.viewer.paint_selected_item_index = item_index
        self.run_tracking(window)
        resolved = self.resolved(window, 11)
        for i, point in enumerate(circle_points(24)):
            tx, ty = truth(point, 11, "translate")
            self.assertLess(math.hypot(resolved[i]["x"] - tx, resolved[i]["y"] - ty), 0.7, i)

    def test_a_shape_needs_selecting_and_an_input(self):
        window = self.make_window([roto_shape()])
        window.viewer.roto_selected_shape_index = -1
        self.assertFalse(window.track_roto_points("roto"))
        window.viewer.roto_selected_shape_index = 0
        self.assertFalse(window.track_roto_points("missing"))


class CommandPlacementTests(WindowCase):
    def test_the_panel_has_track_points_beside_track_shape(self):
        window = self.make_window([roto_shape()])
        window.graph.items_by_id["roto"].setSelected(True)
        APP.processEvents()
        from nodebased import rotopanel
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QSpinBox
        root = QWidget(); layout = QVBoxLayout(root)
        rotopanel.add_tracking_actions(layout, window, "roto")
        self.assertIsNotNone(root.findChild(QPushButton, "roto-track-shape"))
        self.assertIsNotNone(root.findChild(QPushButton, "roto-track-points"))
        direction = root.findChild(QComboBox, "roto-track-points-direction")
        self.assertEqual([direction.itemData(i) for i in range(direction.count())], ["both", "forward", "backward"])
        self.assertEqual((root.findChild(QSpinBox, "roto-track-points-first").value(),
                          root.findChild(QSpinBox, "roto-track-points-last").value()), (1, 11))
        calls = []
        window.track_roto_points = lambda key, *a, **k: calls.append((key, a, k)) or True
        root.findChild(QComboBox, "roto-track-points-direction").setCurrentIndex(1)
        root.findChild(QSpinBox, "roto-track-points-last").setValue(8)
        root.findChild(QPushButton, "roto-track-points").click()
        self.assertEqual(calls, [("roto", ("forward", 1, 8), {})])

    def test_the_shape_list_menu_has_track_points(self):
        window = self.make_window([roto_shape("a"), roto_shape("b")], view=True)
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None, timeout=90))
        window.graph.items_by_id["roto"].setSelected(True)
        APP.processEvents()
        panel = window.build_node_panel("roto")
        window.set_properties_widget(panel)
        from PySide6.QtWidgets import QTreeWidget
        tree = panel.findChild(QTreeWidget, "roto-shape-list")
        tree.select_ids([("item", 1)])
        action = [a for a in tree.build_menu().actions() if a.text() == "Track points"]
        self.assertEqual(len(action), 1)
        self.assertTrue(action[0].isEnabled())
        calls = []
        window.track_roto_points = lambda key, direction, **k: calls.append((key, direction, k)) or True
        action[0].trigger()
        self.assertEqual(calls, [("roto", "both", {"shape_indices": [1]})])


if __name__ == "__main__":
    unittest.main()
