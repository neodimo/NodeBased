"""Roto image-aware in-betweens (lane 2, step I3): the fill and the command."""
import copy
import math
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings

from nodebased import rotoedges, rotoinbetween, roto
from nodebased import shapes as shape_model
from nodebased.app import Window
from nodebased.core import Dispatcher, empty_document
from nodebased.rotolist import ShapeListTree
from tests.test_roto_edges_i1 import CENTRE, circle_points, disc_image
from tests.test_roto_ui import APP
from tests.waiting import wait_until

START, END = 1, 9
R0, R1 = 30.0, 50.0


def growth(frame, curve):
    """Disc radius at `frame`: `curve` maps progress 0..1 to growth 0..1 (linear or ease-in)."""
    t = (frame - START) / (END - START)
    return R0 + (R1 - R0) * curve(t)


def ease_in(t):
    return t * t


def linear(t):
    return t


def keyed_shape(interpolation="linear", count=16):
    """A circle keyed only at the first and last frame, with radius R0 then R1."""
    first = circle_points(count, R0)
    last = circle_points(count, R1)
    points = []
    for a, b in zip(first, last):
        point = {}
        for name in rotoedges.POINT_FIELDS:
            point[name] = {"value": a[name], "curve": {"interpolation": interpolation, "keys": [
                {"frame": START, "value": a[name]}, {"frame": END, "value": b[name]}]}}
        points.append(point)
    return {"name": "disc", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": points}


def resolver(shape):
    payload = {"shapes": [shape]}
    return lambda frame: shape_model.resolve_shapes(payload, frame)[0]


def field_provider(curve, size=(200, 180)):
    def field_for(frame, box):
        image = disc_image(size[0], size[1], CENTRE, growth(frame, curve))
        return rotoedges.field_from_image(image, rotoedges.aligned_box(*box, size[0], size[1]))
    return field_for


def rms_to_radius(points, radius):
    outline = roto.flatten_points(points, closed=True)
    d = [math.hypot(x - CENTRE[0], y - CENTRE[1]) - radius for x, y in outline]
    return math.sqrt(sum(v * v for v in d) / len(d))


def shape_at(shape, frame):
    return shape_model.resolve_shapes({"shapes": [shape]}, frame)[0]["points"]


class FillTests(unittest.TestCase):
    def test_nonlinear_growth_is_fitted_where_interpolation_misses(self):
        shape = keyed_shape()
        result = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END)
        self.assertEqual(result.examined, list(range(2, 9)))
        filled = copy.deepcopy(shape)
        rotoinbetween.write_keys(filled, result.keys, START, END)
        worst_plain, worst_fit = 0.0, 0.0
        for frame in range(2, 9):
            truth = growth(frame, ease_in)
            plain = rms_to_radius(shape_at(shape, frame), truth)
            fitted = rms_to_radius(shape_at(filled, frame), truth)
            self.assertLess(fitted, 0.75, (frame, plain, fitted))
            if 3 <= frame <= 7:
                self.assertGreater(plain, 2.0, (frame, plain))
            worst_plain, worst_fit = max(worst_plain, plain), max(worst_fit, fitted)
        self.assertGreater(worst_plain, 4.0)
        self.assertGreater(result.movement[5], 2.0)

    def test_keys_exist_only_between_the_ends_and_the_ends_are_untouched(self):
        shape = keyed_shape()
        result = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END)
        self.assertTrue(all(START < frame < END for frame in result.keys))
        filled = copy.deepcopy(shape)
        rotoinbetween.write_keys(filled, result.keys, START, END)
        for frame in (START, END):
            self.assertEqual(shape_at(filled, frame), shape_at(shape, frame))

    def test_linear_growth_gets_no_new_keys(self):
        shape = keyed_shape()
        result = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(linear), START, END)
        self.assertEqual(result.keys, {})
        self.assertEqual(result.key_count, 0)
        self.assertEqual(len(result.movement), 7)
        self.assertLess(max(result.movement.values()), 0.75)

    def test_threshold_decides_which_points_get_keys(self):
        shape = keyed_shape()
        loose = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, threshold=0.75)
        strict = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, threshold=4.0)
        never = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, threshold=1000.0)
        self.assertGreater(loose.key_count, strict.key_count)
        self.assertGreater(strict.key_count, 0)
        self.assertEqual(never.key_count, 0)
        self.assertEqual(loose.movement, strict.movement)       # the threshold changes keys, not the fit
        self.assertNotIn(2, strict.keys)                         # frame 2 misses by about 2 px only

    def test_step_fills_every_nth_frame(self):
        shape = keyed_shape()
        every = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, step=1)
        third = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, step=3)
        self.assertEqual(third.examined, [4, 7])
        self.assertEqual(sorted(third.keys), [4, 7])
        self.assertEqual(sorted(every.keys), list(range(2, 9)))
        self.assertEqual(rotoinbetween.frames_between(1, 9, 100), [])
        self.assertEqual(rotoinbetween.frames_between(1, 2), [])

    def test_frames_the_artist_already_keyed_are_left_alone(self):
        shape = keyed_shape()
        result = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, skip_frames=[5])
        self.assertNotIn(5, result.keys)
        self.assertEqual(result.skipped, [5])
        self.assertNotIn(5, result.movement)

    def test_frames_without_an_image_are_skipped(self):
        shape = keyed_shape()
        result = rotoinbetween.fill_inbetweens(resolver(shape), lambda frame, box: None, START, END)
        self.assertEqual(result.keys, {})
        self.assertEqual(result.skipped, list(range(2, 9)))

    def test_band_limits_how_far_a_point_travels(self):
        shape = keyed_shape()
        narrow = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, band=2.0)
        wide = rotoinbetween.fill_inbetweens(resolver(shape), field_provider(ease_in), START, END, band=8.0)
        self.assertLess(narrow.movement[5], wide.movement[5])
        self.assertLessEqual(narrow.movement[5], 2.6)

    def test_static_points_gain_keys_pinned_at_both_ends(self):
        static = {"name": "disc", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
                  "points": circle_points(16, R0)}
        shape = copy.deepcopy(static)
        keys = {5: {0: {**static["points"][0], "x": static["points"][0]["x"] + 3.0}}}
        rotoinbetween.write_keys(shape, keys, START, END)
        self.assertEqual(rotoinbetween.key_frames(shape), [START, 5, END])
        self.assertEqual(shape_at(shape, START)[0]["x"], static["points"][0]["x"])
        self.assertEqual(shape_at(shape, END)[0]["x"], static["points"][0]["x"])
        self.assertAlmostEqual(shape_at(shape, 5)[0]["x"], static["points"][0]["x"] + 3.0)
        self.assertEqual(shape["points"][1], static["points"][1])        # an unmoved point stays plain

    def test_surrounding_keys(self):
        shape = keyed_shape()
        self.assertEqual(rotoinbetween.surrounding_keys(shape, 4), (1, 9))
        self.assertEqual(rotoinbetween.surrounding_keys(shape, 1), (1, 9))
        self.assertIsNone(rotoinbetween.surrounding_keys(shape, 9))
        self.assertIsNone(rotoinbetween.surrounding_keys(shape, 0))
        self.assertIsNone(rotoinbetween.surrounding_keys({"points": circle_points(4, 10)}, 3))

    def test_time_per_inbetween_on_a_24_point_shape_at_1080p(self):
        size = (1920, 1080)
        centre, radius = (960.0, 540.0), 300.0
        first, last = circle_points(24, radius, centre), circle_points(24, radius + 24.0, centre)
        shape = {"name": "big", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": [
            {name: {"value": a[name], "curve": {"interpolation": "linear", "keys": [
                {"frame": START, "value": a[name]}, {"frame": END, "value": b[name]}]}}
             for name in rotoedges.POINT_FIELDS} for a, b in zip(first, last)]}

        images = {frame: disc_image(size[0], size[1], centre, radius + 3.0 * (frame - START))
                  for frame in range(START + 1, END)}

        def field_for(frame, box):
            return rotoedges.field_from_image(images[frame], rotoedges.aligned_box(*box, size[0], size[1]))

        started = time.perf_counter()
        result = rotoinbetween.fill_inbetweens(resolver(shape), field_for, START, END)
        elapsed = time.perf_counter() - started
        per_frame = elapsed / max(len(result.examined), 1)
        print(f"\nI3 time per in-between, 24 points at 1080p (field and fit; the plate is already rendered): {per_frame * 1000:.0f} ms")
        self.assertLess(per_frame, 2.0)


# -- the window: a Radial plate whose disc grows non-linearly over frames 1 to 9 ---------------------

PLATE_CENTRE = (50.0, 50.0)
PLATE_R0, PLATE_R1 = 15.0, 35.0
EDGE_BIAS = 25.08 / 25.0       # a hard Radial of box radius r has an area-equivalent edge radius of r * this


def plate_radius(frame):
    return PLATE_R0 + (PLATE_R1 - PLATE_R0) * ease_in((frame - START) / (END - START))


def growing_plate_document(shapes=None, current=5, layers=None):
    document = empty_document()
    document["nodes"] = {}
    document["view"] = None
    document["time"].update(first=START, last=END, current=current)
    dispatcher = Dispatcher(document)
    dispatcher.execute({"op": "create", "id": "plate", "type": "Radial"})
    for name, value in dict(width=100, height=100, softness=0.0).items():
        dispatcher.execute({"op": "set", "id": "plate", "param": name, "value": value})
    for frame in range(START, END + 1):
        radius = plate_radius(frame)
        for name, value in (("box_x", PLATE_CENTRE[0] - radius), ("box_y", PLATE_CENTRE[1] - radius),
                            ("box_width", 2 * radius), ("box_height", 2 * radius)):
            dispatcher.execute({"op": "set_key", "id": "plate", "param": name, "frame": frame, "value": value})
    dispatcher.execute({"op": "create", "id": "r", "type": "Roto", "pos": [0, 0]})
    dispatcher.execute({"op": "set", "id": "r", "param": "width", "value": 100})
    dispatcher.execute({"op": "set", "id": "r", "param": "height", "value": 100})
    dispatcher.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
    dispatcher.execute({"op": "view", "id": "r"})
    if shapes:
        command = {"op": "set_shapes", "id": "r", "shapes": shapes}
        if layers:
            command["layers"] = layers
        dispatcher.execute(command)
    return dispatcher.document


def plate_shape(name="disc", **extra):
    """A circle keyed at frames 1 and 9 only, on the plate's disc."""
    first = circle_points(16, PLATE_R0 * EDGE_BIAS, PLATE_CENTRE)
    last = circle_points(16, PLATE_R1 * EDGE_BIAS, PLATE_CENTRE)
    shape = {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": [
        {n: {"value": a[n], "curve": {"interpolation": "linear", "keys": [
            {"frame": START, "value": a[n]}, {"frame": END, "value": b[n]}]}}
         for n in rotoedges.POINT_FIELDS} for a, b in zip(first, last)]}
    shape.update(extra)
    return shape


class WindowTests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        rotoedges.CACHE.clear()
        self.window = None

    def tearDown(self):
        if self.window is not None:
            self.window.saved_document = self.window.dispatcher.document
            self.window.close()
            APP.processEvents()

    def open(self, shapes, current=5):
        self.window = Window(growing_plate_document(shapes, current))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        self.viewer = self.window.viewer
        self.viewer.select_roto_tool("Select", "Select Points")
        return self.viewer

    def shapes(self):
        return self.window.dispatcher.document["node_data"]["r"]["shapes"]

    def error_at(self, shape, frame):
        points = shape_model.resolve_shapes({"shapes": [shape]}, frame)[0]["points"]
        return rms_to_radius_about(points, PLATE_CENTRE, plate_radius(frame) * EDGE_BIAS)

    def test_fill_through_the_window_fits_every_middle_frame(self):
        self.open([plate_shape()])
        self.viewer.roto_selected_shape_index = 0
        before = [self.error_at(self.shapes()[0], f) for f in range(2, 9)]
        self.assertGreater(max(before), 1.5)
        undo = len(self.window.dispatcher.undo_stack)
        self.assertTrue(self.viewer.fill_roto_inbetweens())
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo + 1)
        after = [self.error_at(self.shapes()[0], f) for f in range(2, 9)]
        for frame, value in zip(range(2, 9), after):
            self.assertLess(value, 0.75, (frame, before, after))
        self.assertEqual(sorted(rotoinbetween.key_frames(self.shapes()[0])), list(range(1, 10)))
        message = self.window.statusBar().currentMessage()
        self.assertIn("Fill in-betweens: added", message)
        self.assertIn("keys on 7 frames", message)

    def test_strip_button_runs_the_fill_and_the_settings_are_read(self):
        self.open([plate_shape()])
        self.viewer.roto_selected_shape_index = 0
        toolbar = self.viewer.roto_toolbar
        toolbar.choose("Select", "Select Points")
        toolbar.set_setting("r", "Select Points", "fill_step", 4.0)
        toolbar.set_setting("r", "Select Points", "fill_threshold", 0.5)
        toolbar.set_setting("r", "Select Points", "fill_band", 10.0)
        toolbar.settings_widgets["fill_inbetweens"].click()
        APP.processEvents()
        self.assertEqual(rotoinbetween.key_frames(self.shapes()[0]), [1, 5, 9])
        self.assertLess(self.error_at(self.shapes()[0], 5), 0.75)

    def test_from_and_to_set_the_range_and_the_current_frame_is_not_needed(self):
        shape = plate_shape()
        self.open([shape], current=1)
        self.viewer.roto_selected_shape_index = 0
        toolbar = self.viewer.roto_toolbar
        toolbar.set_setting("r", "Select Points", "fill_from", 3.0)
        toolbar.set_setting("r", "Select Points", "fill_to", 7.0)
        self.assertTrue(self.viewer.fill_roto_inbetweens())
        self.assertEqual(rotoinbetween.key_frames(self.shapes()[0]), [1, 4, 5, 6, 9])

    def test_one_undo_removes_every_added_key(self):
        self.open([plate_shape("a"), plate_shape("b")])
        before = copy.deepcopy(self.shapes())
        self.assertTrue(self.viewer.fill_roto_inbetweens([0, 1]))
        self.assertNotEqual(self.shapes(), before)
        for shape in self.shapes():
            self.assertEqual(rotoinbetween.key_frames(shape), list(range(1, 10)))
        self.window.command({"op": "undo"})
        self.assertEqual(self.shapes(), before)

    def test_locked_shapes_and_shapes_in_locked_layers_are_untouched(self):
        locked = plate_shape("locked", locked=True)
        layered = plate_shape("layered", layer="L")
        free = plate_shape("free")
        document = growing_plate_document([locked, layered, free],
                                          layers=[{"name": "L", "visible": True, "locked": True}])
        self.window = Window(document)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        self.viewer = self.window.viewer
        before = copy.deepcopy(self.shapes())
        self.assertTrue(self.viewer.fill_roto_inbetweens([0, 1, 2]))
        after = self.shapes()
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1], before[1])
        self.assertNotEqual(after[2], before[2])
        self.window.command({"op": "undo"})
        self.assertFalse(self.viewer.fill_roto_inbetweens([0, 1]))
        self.assertEqual(self.shapes(), before)

    def test_a_shape_with_no_keys_around_the_frame_is_reported_not_changed(self):
        static = {"name": "plain", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
                  "points": circle_points(16, 20.0, PLATE_CENTRE)}
        self.open([static])
        before = copy.deepcopy(self.shapes())
        undo = len(self.window.dispatcher.undo_stack)
        self.assertFalse(self.viewer.fill_roto_inbetweens([0]))
        self.assertEqual(self.shapes(), before)
        self.assertEqual(len(self.window.dispatcher.undo_stack), undo)
        self.assertIn("keyed on both sides", self.window.statusBar().currentMessage())

    def test_an_existing_key_between_the_ends_is_kept(self):
        shape = plate_shape()
        for point in shape["points"]:
            for name in rotoedges.POINT_FIELDS:
                keys = point[name]["curve"]["keys"]
                keys.insert(1, {"frame": 5, "value": keys[0]["value"] + (keys[1]["value"] - keys[0]["value"]) * 0.5})
        self.open([shape], current=3)
        self.viewer.roto_selected_shape_index = 0
        before = {n: [dict(k) for k in shape["points"][0][n]["curve"]["keys"] if k["frame"] == 5] for n in ("x", "y")}
        self.assertTrue(self.viewer.fill_roto_inbetweens())
        keys = self.shapes()[0]["points"][0]["x"]["curve"]["keys"]
        self.assertEqual([k for k in keys if k["frame"] == 5], before["x"])
        self.assertEqual(sorted(k["frame"] for k in keys), [1, 2, 3, 4, 5, 9])      # filled between 1 and 5 only

    def test_list_menu_has_fill_in_betweens(self):
        self.open([plate_shape()])
        self.window.rebuild_properties_dock()
        APP.processEvents()
        lists = self.window.findChildren(ShapeListTree)
        if not lists:
            self.skipTest("no Roto list in this layout")
        names = [a.objectName() for a in lists[0].build_menu().actions()]
        self.assertIn("roto-list-fill-inbetweens", names)

    def test_old_documents_have_the_fill_settings_at_their_defaults(self):
        self.open([plate_shape()])
        settings = self.viewer.roto_toolbar._tool_settings("r", "Select Points")
        self.assertEqual((settings["fill_band"], settings["fill_threshold"], settings["fill_step"]),
                         (8.0, 0.75, 1.0))
        self.assertEqual(self.window.dispatcher.document["node_data"]["r"].get("tool_settings", {}), {})


def rms_to_radius_about(points, centre, radius):
    outline = roto.flatten_points(points, closed=True)
    d = [math.hypot(x - centre[0], y - centre[1]) - radius for x, y in outline]
    return math.sqrt(sum(v * v for v in d) / len(d))


if __name__ == "__main__":
    unittest.main()
