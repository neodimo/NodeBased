"""Reduce keys for Roto shapes and RotoPaint strokes (lane 9, step I9, past Nuke).

Every error here is measured independently of the module under test: the shape (or stroke) is resolved on each frame through
`shapes._resolve_shape` (the interpolation Roto draws with), flattened to its outline, and the reduced result is compared with the
original frame by frame. Key counts are printed for the docs. No test asserts elapsed time.
"""
import copy
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import rotokeyreduce as rk
from nodebased import rotopointtrack as rp
from nodebased import roto
from nodebased import shapes as shape_model

TOLERANCE = 0.5


def scalar(base, keys=None, interpolation="smooth"):
    if keys is None:
        return {"value": float(base)}
    return {"value": float(base), "curve": {"interpolation": interpolation,
                                            "keys": [{"frame": int(f), "value": float(v)} for f, v in keys]}}


def moving_shape(motion, frames, interpolation="smooth", corners=((0, 0), (40, 0), (40, 40), (0, 40)), name="shape", **extra):
    """A closed polygon whose corners move together by motion(frame) = (dx, dy), keyed on every frame."""
    points = []
    for cx, cy in corners:
        point = {name_: 0.0 for name_ in shape_model.POINT_FIELDS}
        point["x"] = scalar(cx, [(f, cx + motion(f)[0]) for f in frames], interpolation)
        point["y"] = scalar(cy, [(f, cy + motion(f)[1]) for f in frames], interpolation)
        points.append(point)
    return {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": points, **extra}


def rendered(item, frame):
    """The outline Roto draws on `frame` (curve vertices) and the feather handle positions, after the shape's transform."""
    resolved = shape_model._resolve_shape(item, frame, None, False)
    vertices = roto.flatten_points(resolved["points"], closed=resolved["closed"], curve_type=resolved.get("curve_type", "bezier"))
    handles = np.array([(p["x"] + p["feather_x"], p["y"] + p["feather_y"]) for p in resolved["points"]])
    return vertices, handles


def worst_error(original, reduced, frames):
    """Largest distance in px between the two outlines on any frame in `frames` (resampled by arc length if the vertex counts differ)."""
    worst = 0.0
    for frame in frames:
        (va, ha), (vb, hb) = rendered(original, frame), rendered(reduced, frame)
        if len(va) != len(vb):
            def resample(v):
                closed = np.vstack([v, v[:1]])
                along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
                s = np.linspace(0, along[-1], 400, endpoint=False)
                return np.stack([np.interp(s, along, closed[:, 0]), np.interp(s, along, closed[:, 1])], axis=1)
            va, vb = resample(va), resample(vb)
        worst = max(worst, float(np.hypot(*(va - vb).T).max()), float(np.hypot(*(ha - hb).T).max()))
    return worst


def key_frames(shape):
    return rk.key_frames_of(shape)


def curve_key_sets(shape):
    """The key frames of every animated scalar of the shape, as a list of tuples."""
    return [tuple(k["frame"] for k in v["curve"]["keys"]) for p in shape["points"] for v in p.values()
            if isinstance(v, dict) and v.get("curve")]


FRAMES = list(range(1, 41))


class ReduceEngineTests(unittest.TestCase):
    def test_linear_motion_keyed_every_frame_keeps_two_keys(self):
        for interpolation in ("linear", "smooth"):
            shape = moving_shape(lambda f: (3.0 * f, 2.0 * f), FRAMES, interpolation)
            result = rk.reduce_item(shape, TOLERANCE)
            self.assertEqual((result.before, result.after), (40, 2), interpolation)
            self.assertEqual(result.kept, [1, 40])
            self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_keeping_the_stored_interpolation_still_stays_within_tolerance(self):
        shape = moving_shape(lambda f: (3.0 * f, 2.0 * f), FRAMES, "smooth")
        result = rk.reduce_item(shape, TOLERANCE, interpolation="keep")
        self.assertLess(result.after, result.before)
        self.assertGreater(result.after, 2)          # a smooth key holds still at each end, so a straight run needs more keys
        self.assertTrue(all(c.get("curve", {}).get("interpolation") == "smooth" for p in result.item["points"] for c in p.values()
                            if isinstance(c, dict) and c.get("curve")))
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_ease_in_and_out_keeps_few_keys_and_every_frame_is_within_tolerance(self):
        def smoothstep(f):
            t = (f - 1) / 39.0
            return 120.0 * t * t * (3 - 2 * t), 60.0 * t * t * (3 - 2 * t)
        shape = moving_shape(smoothstep, FRAMES, "smooth")
        result = rk.reduce_item(shape, TOLERANCE)
        print(f"\nsmooth ease in and out: {result.before} keys -> {result.after} (interpolation {result.interpolation}), worst {result.max_error:.3f} px")
        self.assertEqual(result.kept, [1, 40])
        self.assertEqual(result.interpolation, "as stored")
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_a_cosine_ease_that_smooth_keys_do_not_draw_exactly_keeps_a_few_keys(self):
        def cosine(f):
            t = (f - 1) / 39.0
            return 120.0 * (0.5 - 0.5 * math.cos(math.pi * t)), 0.0
        shape = moving_shape(cosine, FRAMES, "smooth")
        result = rk.reduce_item(shape, TOLERANCE)
        print(f"\ncosine ease in and out: {result.before} keys -> {result.after} ({result.interpolation}), worst {result.max_error:.3f} px")
        self.assertLessEqual(result.after, 16)
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_a_stop_and_go_motion_keeps_the_stop(self):
        def stop_and_go(f):
            return 5.0 * min(f, 15) + 5.0 * max(0, f - 25), 0.0
        for interpolation in ("linear", "smooth"):
            shape = moving_shape(stop_and_go, FRAMES, interpolation)
            result = rk.reduce_item(shape, TOLERANCE)
            self.assertEqual(result.kept, [1, 15, 25, 40], interpolation)
            self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_a_direction_change_keeps_its_corner(self):
        def there_and_back(f):
            return 4.0 * (20 - abs(f - 20)), 0.0
        shape = moving_shape(there_and_back, FRAMES, "linear")
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertEqual(result.kept, [1, 20, 40])
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_all_points_of_a_shape_are_keyed_on_the_same_frames(self):
        # the corners move differently, so each one alone would keep different frames
        shape = moving_shape(lambda f: (0.0, 0.0), FRAMES)
        for index, point in enumerate(shape["points"]):
            wobble = [(f, point["x"]["value"] + (8.0 * math.sin(f / (3.0 + index)) if index else 0.5 * f)) for f in FRAMES]
            point["x"] = scalar(point["x"]["value"], wobble, "linear")
            point["y"] = scalar(point["y"]["value"], [(f, point["y"]["value"] + (index * f % 7) * 0.2) for f in FRAMES], "linear")
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertLess(result.after, result.before)
        sets = set(curve_key_sets(result.item))
        self.assertEqual(len(sets), 1, sets)
        self.assertEqual(sorted(sets.pop()), result.kept)
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_a_tighter_tolerance_keeps_more_keys_and_every_one_holds(self):
        def wavy(f):
            return 40.0 * math.sin(f / 6.0), 15.0 * math.cos(f / 4.0)
        shape = moving_shape(wavy, FRAMES, "smooth")
        counts = []
        for tolerance in (2.0, 0.5, 0.1):
            result = rk.reduce_item(shape, tolerance)
            counts.append(result.after)
            self.assertLessEqual(worst_error(shape, result.item, FRAMES), tolerance, tolerance)
        print(f"\nwavy motion, 40 frames, tolerance 2 / 0.5 / 0.1 px keeps {counts} keys")
        self.assertLess(counts[0], counts[1])
        self.assertLessEqual(counts[1], counts[2])

    def test_a_static_shape_is_untouched(self):
        static = {"name": "still", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
                  "points": [{n: float(v) for n, v in zip(shape_model.POINT_FIELDS, (x, y, 0, 0, 0, 0))}
                             for x, y in ((0, 0), (10, 0), (10, 10))]}
        result = rk.reduce_item(static, TOLERANCE)
        self.assertFalse(result.changed)
        self.assertEqual(result.item, static)
        self.assertEqual((result.before, result.after), (0, 0))
        out, summary = rk.reduce_items([static], [0], TOLERANCE)
        self.assertEqual(out, [static])
        self.assertEqual(summary["reduced"], 0)
        self.assertEqual(summary["examined"], 0)

    def test_a_shape_with_two_keys_or_already_minimal_is_untouched(self):
        shape = moving_shape(lambda f: (3.0 * f, 0.0), [1, 40], "linear")
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertFalse(result.changed)
        self.assertEqual(result.item, shape)
        bendy = moving_shape(lambda f: (50.0 * math.sin(f / 3.0), 0.0), range(1, 31), "linear")
        result = rk.reduce_item(bendy, 0.5)
        again = rk.reduce_item(result.item, 0.5)
        self.assertEqual(again.after, again.before)
        self.assertFalse(again.changed)

    def test_the_input_is_never_modified(self):
        shape = moving_shape(lambda f: (3.0 * f, 2.0 * f), FRAMES)
        before = copy.deepcopy(shape)
        rk.reduce_item(shape, TOLERANCE)
        self.assertEqual(shape, before)

    def test_a_frame_range_leaves_the_keys_outside_it(self):
        shape = moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear")
        result = rk.reduce_item(shape, TOLERANCE, first=11, last=30)
        self.assertEqual(result.kept, [*range(1, 11), *range(31, 41)])       # frames 11 to 30 may go; the motion is straight
        self.assertEqual(result.before - result.after, 20)
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_the_error_is_measured_on_the_rendered_outline_after_the_shape_transform(self):
        def wavy(f):
            return 6.0 * math.sin(f / 5.0), 0.0
        plain = moving_shape(wavy, FRAMES, "linear")
        scaled = copy.deepcopy(plain)
        scaled["transform"] = {"scale_x": 4.0, "scale_y": 4.0}
        a = rk.reduce_item(plain, TOLERANCE)
        b = rk.reduce_item(scaled, TOLERANCE)
        print(f"\nsame wobble, no transform {a.after} keys, drawn four times larger {b.after} keys")
        self.assertGreater(b.after, a.after)
        self.assertLessEqual(worst_error(scaled, b.item, FRAMES), TOLERANCE)

    def test_curved_spans_tangents_and_feather_offsets_are_part_of_the_outline(self):
        shape = moving_shape(lambda f: (0.0, 0.0), FRAMES, "linear", corners=((0, 0), (60, 0), (60, 60), (0, 60)))
        for index, point in enumerate(shape["points"]):
            point["out_x"] = scalar(10.0, [(f, 10.0 + (f - 1) * 0.5) for f in FRAMES], "linear")       # straight growth
            point["in_y"] = scalar(-10.0, [(f, -10.0 - 6.0 * math.sin(f / 4.0)) for f in FRAMES], "linear")
            point["feather_x"] = scalar(0.0, [(f, 0.3 * (f - 1)) for f in FRAMES], "linear")
            point["feather_y"] = 0.0
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertLess(result.after, result.before)
        self.assertGreater(result.after, 4)             # the wobbling tangent needs real keys
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_a_b_spline_shape_with_tension_is_reduced_within_tolerance(self):
        shape = moving_shape(lambda f: (2.0 * f, 25.0 * math.sin(f / 7.0)), FRAMES, "linear", curve_type="bspline",
                             corners=((0, 0), (50, 10), (60, 50), (10, 60), (-10, 30)))
        for index, point in enumerate(shape["points"]):
            point["tension"] = scalar(0.0, [(f, 0.8 * math.sin(f / 9.0)) for f in FRAMES], "linear")
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertLess(result.after, result.before)
        self.assertLessEqual(worst_error(shape, result.item, FRAMES), TOLERANCE)

    def test_an_open_spline_is_reduced_too(self):
        shape = moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear", closed=False)
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertEqual(result.kept, [1, 40])

    def test_sparse_hand_keys_that_are_all_needed_are_kept(self):
        shape = moving_shape(lambda f: (0.0, 0.0), [1, 10, 20, 30])
        for point, values in zip(shape["points"], ((0, 30, 0, 30), (0, 0, 30, 30), (0, 10, 0, 10), (0, 5, 5, 0))):
            point["x"] = scalar(0, list(zip([1, 10, 20, 30], values)), "linear")
        result = rk.reduce_item(shape, TOLERANCE)
        self.assertEqual(result.kept, [1, 10, 20, 30])
        self.assertFalse(result.changed)

    def test_bad_arguments_raise(self):
        shape = moving_shape(lambda f: (3.0 * f, 0.0), FRAMES)
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(rk.ReduceError):
                rk.reduce_item(shape, bad)
        with self.assertRaises(rk.ReduceError):
            rk.reduce_item(shape, 0.5, interpolation="cubic")

    def test_a_reduced_shape_still_validates(self):
        shape = moving_shape(lambda f: (3.0 * f, 2.0 * f), FRAMES)
        result = rk.reduce_item(shape, TOLERANCE)
        shape_model.validate_shape(result.item, "shapes[0]")


# -- strokes -----------------------------------------------------------------------------------------------------------------
def stuck_stroke(frames=range(1, 31), interpolation="smooth", scale_end=1.0):
    from tests.test_roto_paint import stroke
    base = [(30.0, 40.0), (60.0, 55.0), (90.0, 40.0), (120.0, 60.0)]
    points = []
    for x, y in base:
        points.append({"x": scalar(x, [(f, x + 2.0 * (f - 1)) for f in frames], interpolation),
                       "y": scalar(y, [(f, y + 1.0 * (f - 1)) for f in frames], interpolation), "pressure": 1.0})
    item = stroke(points=points)
    if scale_end != 1.0:
        n = len(list(frames))
        item["stick"] = {"scale": scalar(1.0, [(f, 1.0 + (scale_end - 1.0) * (f - 1) / (n - 1)) for f in frames], interpolation),
                         "rotation": 0.0}
    return item


def stroke_positions(item, frame):
    resolved = shape_model.resolve_paint_items({"items": [item]}, frame)[0]
    return np.array([(p["x"], p["y"]) for p in resolved["points"]]), float(resolved["brush"]["size"])


class StrokeReduceTests(unittest.TestCase):
    def test_a_stroke_sliding_in_a_straight_line_keeps_two_keys(self):
        item = stuck_stroke()
        result = rk.reduce_item(item, TOLERANCE)
        self.assertEqual((result.before, result.after), (30, 2))
        for frame in range(1, 31):
            a, _ = stroke_positions(item, frame)
            b, _ = stroke_positions(result.item, frame)
            self.assertLessEqual(float(np.hypot(*(a - b).T).max()), TOLERANCE)

    def test_the_stuck_scale_is_part_of_the_error_and_the_brush_follows_within_tolerance(self):
        item = stuck_stroke(scale_end=2.0)
        item["brush"] = {**item["brush"], "size": 20.0}
        result = rk.reduce_item(item, TOLERANCE)
        self.assertEqual(result.after, 2)
        for frame in range(1, 31):
            _, size_a = stroke_positions(item, frame)
            _, size_b = stroke_positions(result.item, frame)
            self.assertLessEqual(abs(size_a - size_b) / 2.0, TOLERANCE + 1e-6)
        # a scale that wobbles needs keys: the brush radius is measured in px
        wobble = copy.deepcopy(item)
        wobble["stick"]["scale"] = scalar(1.0, [(f, 1.0 + 0.4 * math.sin(f / 3.0)) for f in range(1, 31)], "linear")
        reduced = rk.reduce_item(wobble, TOLERANCE)
        self.assertGreater(reduced.after, 4)
        for frame in range(1, 31):
            _, size_a = stroke_positions(wobble, frame)
            _, size_b = stroke_positions(reduced.item, frame)
            self.assertLessEqual(abs(size_a - size_b) / 2.0, TOLERANCE + 1e-6)

    def test_a_stroke_that_never_stuck_is_untouched(self):
        from tests.test_roto_paint import stroke
        item = stroke(points=[{"x": 10.0, "y": 10.0, "pressure": 1.0}, {"x": 20.0, "y": 25.0, "pressure": 1.0}])
        out, summary = rk.reduce_items([item], [0], TOLERANCE)
        self.assertEqual(out, [item])
        self.assertEqual(summary["examined"], 0)


# -- Track points output on the I2 test plate ------------------------------------------------------------------------------------
def tracked_shape(mode, frames):
    """The I2 disc, Track points run over `frames` and written as keys the way the app writes them (Autokey on)."""
    from tests.test_roto_point_track_i2 import circle_points, source
    points = circle_points(24)
    get = source(mode)
    matrices = rp.planar_matrices(get, points, 1, list(frames))
    result = rp.track_points(get, points, 1, list(frames), planar=matrices)
    stored = []
    for index, point in enumerate(points):
        entry = {}
        for name in rp.POINT_FIELDS:
            updates = {frame: float(data["points"][index][name]) for frame, data in result.items()}
            entry[name] = rp.key_scalar(float(point.get(name, 0.0)), updates, 1, True)
        stored.append(entry)
    return {"name": "disc", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": stored}


class TrackedOutputTests(unittest.TestCase):
    def check(self, mode, frames, tolerance):
        shape = tracked_shape(mode, frames)
        span = [1, *frames]
        result = rk.reduce_item(shape, tolerance)
        error = worst_error(shape, result.item, range(1, max(frames) + 1))
        print(f"\nTrack points on the I2 plate ({mode}, {len(span)} keyed frames), tolerance {tolerance} px: "
              f"{result.before} keys -> {result.after} ({result.interpolation}), worst {error:.3f} px")
        self.assertLessEqual(error, tolerance + 1e-6)
        self.assertLess(result.after, result.before)
        self.assertEqual(len(set(curve_key_sets(result.item))), 1)
        return result

    def test_a_translating_disc_tracked_over_24_frames_reduces_within_tolerance_on_every_frame(self):
        result = self.check("translate", range(2, 26), TOLERANCE)
        self.assertLess(result.after, result.before * 0.6)

    def test_a_stretching_disc_tracked_over_ten_frames_reduces_within_tolerance_on_every_frame(self):
        self.check("stretch", range(2, 12), TOLERANCE)

    def test_a_looser_tolerance_keeps_no_more_keys(self):
        tight = self.check("stretch", range(2, 12), 0.25)
        loose = rk.reduce_item(tracked_shape("stretch", range(2, 12)), 2.0)
        self.assertLessEqual(loose.after, tight.after)


# -- the app command, undo, list menu, strip --------------------------------------------------------------------------------------
from PySide6.QtCore import QSettings

from nodebased.app import Window
from nodebased.core import Dispatcher, empty_document
from tests.test_roto_ui import APP
from tests.waiting import wait_until


class AppCase(unittest.TestCase):
    def make_window(self, entries, node_type="Roto", last=40, current=1, view=False):
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "roto", "type": node_type})
        dispatcher.execute({"op": "connect", "id": "roto", "input": "bg" if node_type == "Roto" else "image", "source": "plate"})
        dispatcher.document["time"].update(first=1, last=last, current=current)
        if node_type == "Roto":
            dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": entries})
        else:
            dispatcher.execute({"op": "set_paint_items", "id": "roto", "items": entries})
        if view:
            dispatcher.execute({"op": "view", "id": "roto"})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        return window

    def payload(self, window):
        return window.dispatcher.document["node_data"]["roto"]

    def status(self, window):
        return window.statusBar().currentMessage()


def paint_shape(shape):
    return {"kind": "shape", "blend": "over", "visible": True, **shape}


class ReduceKeysInTheAppTests(AppCase):
    def test_one_undo_step_restores_every_key_and_the_status_bar_counts_them(self):
        shape = moving_shape(lambda f: (3.0 * f, 2.0 * f), FRAMES, "linear")
        window = self.make_window([shape])
        window.viewer.roto_selected_shape_index = 0
        before = copy.deepcopy(self.payload(window))
        steps = len(window.dispatcher.undo_stack)
        self.assertTrue(window.reduce_roto_keys("roto"))
        self.assertEqual(len(window.dispatcher.undo_stack), steps + 1)
        self.assertEqual(key_frames(self.payload(window)["shapes"][0]), [1, 40])
        self.assertIn("kept 2 of 40 keys on 1 shape", self.status(window))
        window.command({"op": "undo"})
        self.assertEqual(self.payload(window), before)
        window.command({"op": "redo"})
        self.assertEqual(key_frames(self.payload(window)["shapes"][0]), [1, 40])

    def test_several_shapes_in_one_step_and_a_static_one_is_left_alone(self):
        keyed = moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear", name="a")
        other = moving_shape(lambda f: (0.0, 2.0 * f), FRAMES, "smooth", name="b")
        static = copy.deepcopy(keyed)
        static["name"] = "still"
        for point in static["points"]:
            point["x"], point["y"] = float(point["x"]["value"]), float(point["y"]["value"])
        window = self.make_window([keyed, other, static])
        steps = len(window.dispatcher.undo_stack)
        self.assertTrue(window.reduce_roto_keys("roto", item_indices=[0, 1, 2]))
        self.assertEqual(len(window.dispatcher.undo_stack), steps + 1)
        shapes = self.payload(window)["shapes"]
        self.assertEqual([key_frames(s) for s in shapes], [[1, 40], [1, 40], []])
        self.assertEqual(shapes[2], static)
        self.assertIn("kept 4 of 80 keys on 2 shapes", self.status(window))

    def test_a_static_selection_changes_nothing_and_adds_no_undo_step(self):
        static = {"name": "still", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
                  "points": [{n: float(v) for n, v in zip(shape_model.POINT_FIELDS, (x, y, 0, 0, 0, 0))}
                             for x, y in ((0, 0), (10, 0), (10, 10))]}
        window = self.make_window([static])
        window.viewer.roto_selected_shape_index = 0
        before, steps = copy.deepcopy(self.payload(window)), len(window.dispatcher.undo_stack)
        self.assertFalse(window.reduce_roto_keys("roto"))
        self.assertEqual(self.payload(window), before)
        self.assertEqual(len(window.dispatcher.undo_stack), steps)
        self.assertIn("no animated", self.status(window))

    def test_locked_shapes_are_left_alone(self):
        shape = moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear", locked=True)
        window = self.make_window([shape])
        before = copy.deepcopy(self.payload(window))
        self.assertFalse(window.reduce_roto_keys("roto", item_indices=[0]))
        self.assertEqual(self.payload(window), before)

    def test_the_tolerance_and_range_come_from_the_strip_settings_unless_given(self):
        def wavy(f):
            return 40.0 * math.sin(f / 6.0), 15.0 * math.cos(f / 4.0)
        shape = moving_shape(wavy, FRAMES, "smooth")
        counts = []
        for tolerance in (0.25, 3.0):
            window = self.make_window([shape])
            window.viewer.roto_selected_shape_index = 0
            window.command({"op": "set_roto_tool_settings", "id": "roto", "tool": "__global__",
                            "settings": {"reduce_tolerance": tolerance}})
            self.assertTrue(window.reduce_roto_keys("roto"))
            counts.append(len(key_frames(self.payload(window)["shapes"][0])))
            self.assertIn(f"within {tolerance:g} px", self.status(window))
        self.assertGreater(counts[0], counts[1])
        window = self.make_window([moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear")])
        window.viewer.roto_selected_shape_index = 0
        window.command({"op": "set_roto_tool_settings", "id": "roto", "tool": "__global__",
                        "settings": {"reduce_from": 11.0, "reduce_to": 30.0}})
        self.assertTrue(window.reduce_roto_keys("roto"))
        self.assertEqual(key_frames(self.payload(window)["shapes"][0]), [*range(1, 11), *range(31, 41)])

    def test_a_roto_paint_stroke_and_a_shape_item_are_reduced_through_the_items(self):
        stroke_item = stuck_stroke()
        shape_item = paint_shape(moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear"))
        window = self.make_window([shape_item, stroke_item], node_type="RotoPaint")
        window.viewer.paint_selected_item_index = 1
        steps = len(window.dispatcher.undo_stack)
        self.assertTrue(window.reduce_roto_keys("roto"))                       # only the selected stroke
        items = self.payload(window)["items"]
        self.assertEqual(len(key_frames(items[1])), 2)
        self.assertEqual(len(key_frames(items[0])), 40)
        self.assertIn("1 stroke", self.status(window))
        self.assertTrue(window.reduce_roto_keys("roto", item_indices=[0, 1]))
        self.assertEqual(key_frames(self.payload(window)["items"][0]), [1, 40])
        self.assertEqual(len(window.dispatcher.undo_stack), steps + 2)
        window.command({"op": "undo"}); window.command({"op": "undo"})
        self.assertEqual(len(key_frames(self.payload(window)["items"][1])), 30)

    def test_a_node_without_a_selection_says_so(self):
        window = self.make_window([moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear")])
        window.viewer.roto_selected_shape_index = -1
        steps = len(window.dispatcher.undo_stack)
        self.assertFalse(window.reduce_roto_keys("roto"))
        self.assertEqual(len(window.dispatcher.undo_stack), steps)


class CommandPlacementTests(AppCase):
    def test_the_select_tools_strip_has_the_reduce_keys_button_and_settings(self):
        from nodebased.roto_settings import GLOBAL_SETTINGS, TOOL_CONTROLS
        for tool in ("Select All", "Select Splines", "Select Points", "Select Feather Points"):
            labels = {c["label"]: c["widget_kind"] for c in TOOL_CONTROLS[tool]}
            self.assertEqual(labels.get("Reduce keys"), "button", tool)
            self.assertEqual(labels.get("Reduce tolerance"), "field", tool)
            self.assertEqual((labels.get("Reduce from"), labels.get("Reduce to")), ("field", "field"), tool)
        for tool in ("Bezier", "Brush", "Clone"):
            self.assertNotIn("Reduce keys", [c["label"] for c in TOOL_CONTROLS[tool]])
        self.assertEqual((GLOBAL_SETTINGS["reduce_tolerance"], GLOBAL_SETTINGS["reduce_from"], GLOBAL_SETTINGS["reduce_to"]), (0.5, 0.0, 0.0))

    def test_the_strip_button_runs_the_command_and_the_settings_are_stored_on_the_node(self):
        window = self.make_window([moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear")], view=True)
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None, timeout=90))
        window.graph.items_by_id["roto"].setSelected(True)
        APP.processEvents()
        viewer = window.viewer
        viewer.select_roto_tool("Select", "Select Points")
        toolbar = viewer.roto_toolbar
        toolbar.set_setting("roto", "Select Points", "reduce_tolerance", 1.5)
        self.assertEqual(self.payload(window)["tool_settings"]["__global__"]["reduce_tolerance"], 1.5)
        self.assertEqual(toolbar._tool_settings("roto", "Select Points")["reduce_tolerance"], 1.5)
        toolbar.set_setting("roto", "Select Points", "reduce_tolerance", 0.5)
        self.assertNotIn("reduce_tolerance", self.payload(window).get("tool_settings", {}).get("__global__", {}))
        for name in ("reduce_tolerance", "reduce_from", "reduce_to", "reduce_keys"):
            self.assertIn(name, toolbar.settings_widgets)
        viewer.roto_selected_shape_index = 0
        steps = len(window.dispatcher.undo_stack)
        toolbar.settings_widgets["reduce_keys"].click()
        self.assertEqual(key_frames(self.payload(window)["shapes"][0]), [1, 40])
        self.assertEqual(len(window.dispatcher.undo_stack), steps + 1)
        self.assertIn("Reduce keys: kept 2 of 40 keys", self.status(window))

    def test_the_shape_list_menu_has_reduce_keys_and_runs_it_on_the_selection(self):
        window = self.make_window([moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear", name="a"),
                                   moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear", name="b")], view=True)
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None, timeout=90))
        window.graph.items_by_id["roto"].setSelected(True)
        APP.processEvents()
        panel = window.build_node_panel("roto")
        window.set_properties_widget(panel)
        from PySide6.QtWidgets import QTreeWidget
        tree = panel.findChild(QTreeWidget, "roto-shape-list")
        tree.select_ids([("item", 1)])
        action = [a for a in tree.build_menu().actions() if a.objectName() == "roto-list-reduce-keys"]
        self.assertEqual(len(action), 1)
        self.assertTrue(action[0].isEnabled())
        calls = []
        window.reduce_roto_keys = lambda key, *a, **k: calls.append((key, a, k)) or True
        tree.cmd_reduce_keys(0.25, (3, 9))
        self.assertEqual(calls, [("roto", (0.25, 3, 9), {"item_indices": [1]})])
        tree.cmd_reduce_keys()
        self.assertEqual(calls[1], ("roto", (None, None, None), {"item_indices": [1]}))

    def test_the_menu_item_is_greyed_out_with_nothing_selected(self):
        window = self.make_window([moving_shape(lambda f: (3.0 * f, 0.0), FRAMES, "linear")], view=True)
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None, timeout=90))
        window.graph.items_by_id["roto"].setSelected(True)
        APP.processEvents()
        panel = window.build_node_panel("roto")
        window.set_properties_widget(panel)
        from PySide6.QtWidgets import QTreeWidget
        tree = panel.findChild(QTreeWidget, "roto-shape-list")
        tree.clearSelection()
        action = [a for a in tree.build_menu().actions() if a.objectName() == "roto-list-reduce-keys"]
        self.assertEqual(len(action), 1)
        self.assertFalse(action[0].isEnabled())


if __name__ == "__main__":
    unittest.main()
