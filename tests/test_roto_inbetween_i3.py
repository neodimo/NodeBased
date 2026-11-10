"""Roto image-aware in-betweens (lane 2, step I3): the fill and the command."""
import copy
import math
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import rotoedges, rotoinbetween, roto
from nodebased import shapes as shape_model
from tests.test_roto_edges_i1 import CENTRE, circle_points, disc_image

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


if __name__ == "__main__":
    unittest.main()
