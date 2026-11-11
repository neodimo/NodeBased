"""Lane 9 step I10, left-out 3: strokes with a transform of their own stick to the plate, and shear reaches the clone offset.

A stroke's points are keyed in its own space and Roto draws them through its transform. The picture is followed where the stroke
is drawn, so every frame's answer goes back through the inverse of that frame's transform. These tests render the stroke with
the Roto code (`paintknobs.resolve_stroke`, `shapes.resolve_paint_items`) and check against the plate's true motion.
"""
import copy
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import opticalflow, paintknobs, rotostick
from nodebased import shapes as shape_model
from tests.test_roto_paint_stick_i7 import (AppCase, FrameCache, HEIGHT, REFERENCE, TEXTURE, WIDTH, curve_stroke, errors,
                                            long_stroke, paint_stroke, plate, truth)

TRANSFORMS = {
    "rotate and translate": {"rotate": 12.0, "translate_x": 9.0, "translate_y": -4.0},
    "scale": {"scale_x": 1.4, "scale_y": 0.8},
    "skew": {"skew_x": 15.0, "skew_y": -6.0},
    "skew the other way round": {"skew_x": 15.0, "skew_y": -6.0, "skew_order": "YX"},
    "stored centre": {"rotate": 20.0, "center_x": 150.0, "center_y": 90.0, "scale_x": 1.2, "scale_y": 1.2},
    "only the x centre stored": {"rotate": 25.0, "center_x": 140.0, "scale_x": 1.3, "skew_y": 8.0},
    "everything": {"rotate": 30.0, "scale_x": 1.5, "scale_y": 0.7, "skew_x": 10.0, "translate_x": 12.0, "translate_y": 7.0,
                   "center_x": 170.0, "center_y": 110.0},
}
MATRIX = [1.1, 0.2, 0.0, 6.0, -0.1, 0.9, 0.0, -3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def transformed_stroke(transform=None, matrix=None, count=16, **extra):
    item = paint_stroke([(p["x"], p["y"]) for p in curve_stroke(count)], **extra)
    if transform is not None:
        item["transform"] = dict(transform)
    if matrix is not None:
        item["extra_matrix"] = list(matrix)
    return item


def drawn(item, frame):
    """The points Roto draws for the stroke at `frame`."""
    resolved = paintknobs.resolve_stroke(paintknobs.resolve_stuck(item, frame) if paintknobs.sticks(item) else item, frame, {})
    return np.array([[p["x"], p["y"]] for p in resolved["points"]])


class InverseTests(unittest.TestCase):
    def test_the_inverse_gives_back_the_own_space_points_for_every_kind_of_transform(self):
        cases = dict(TRANSFORMS)
        for name, transform in cases.items():
            for matrix in (None, MATRIX):
                with self.subTest(transform=name, matrix=matrix is not None):
                    item = transformed_stroke(transform, matrix)
                    local = np.array([[p["x"], p["y"]] for p in item["points"]])
                    back = rotostick.unresolve_stroke_points(item, drawn(item, 5), 5)
                    self.assertLess(float(np.abs(back - local).max()), 1e-6)

    def test_the_matrix_alone_and_no_transform(self):
        item = transformed_stroke(matrix=MATRIX)
        local = np.array([[p["x"], p["y"]] for p in item["points"]])
        self.assertLess(float(np.abs(rotostick.unresolve_stroke_points(item, drawn(item, 1), 1) - local).max()), 1e-6)
        plain = transformed_stroke()
        points = [(1.0, 2.0), (3.0, 4.0)]
        self.assertEqual(rotostick.unresolve_stroke_points(plain, points, 1).tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(rotostick.stroke_brush_factor(plain, 1), 1.0)

    def test_an_animated_transform_is_inverted_frame_by_frame(self):
        transform = {"rotate": {"value": 0.0, "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 0.0}, {"frame": 9, "value": 40.0}]}}, "scale_x": 1.2}
        item = transformed_stroke(transform)
        local = np.array([[p["x"], p["y"]] for p in item["points"]])
        for frame in (1, 4, 9):
            back = rotostick.unresolve_stroke_points(item, drawn(item, frame), frame)
            self.assertLess(float(np.abs(back - local).max()), 1e-6, f"frame {frame}")

    def test_the_brush_factor_is_the_square_root_of_the_transform_area(self):
        item = transformed_stroke({"scale_x": 2.0, "scale_y": 0.5, "skew_x": 20.0}, MATRIX)
        factor = rotostick.stroke_brush_factor(item, 1)
        resolved = paintknobs.resolve_stroke(item, 1, {})
        self.assertAlmostEqual(factor, resolved["brush"]["size"] / item["brush"]["size"], places=9)


class FollowTests(unittest.TestCase):
    """The engine on a moving plate, for a stroke drawn through its own transform: the drawn stroke follows the true motion."""

    def follow(self, item, kind, last=10):
        rendered = drawn(item, REFERENCE)
        points = [{"x": float(x), "y": float(y)} for x, y in rendered]
        results = rotostick.stick_stroke(FrameCache(lambda f: plate(kind, f)), points, REFERENCE, list(range(2, last + 1)),
                                         brush_size=float(paintknobs.resolve_stroke(item, REFERENCE, {})["brush"]["size"]))
        keyed = rotostick.key_stroke(item, rotostick.localise_results(item, results, REFERENCE), REFERENCE)
        return rendered, keyed

    def check(self, item, kind, frame=10):
        rendered, keyed = self.follow(item, kind, frame)
        stroke = [{"x": float(x), "y": float(y)} for x, y in rendered]
        gx, gy = truth(kind, frame, np.array([p["x"] - .5 for p in stroke]), np.array([p["y"] - .5 for p in stroke]))
        got = shape_model.resolve_paint_items({"items": [keyed]}, frame)[0]["points"]
        got = np.array([[p["x"] - .5, p["y"] - .5] for p in got])
        err = np.hypot(got[:, 0] - gx, got[:, 1] - gy)
        return err, keyed

    def test_a_rotated_and_scaled_stroke_follows_a_translating_rotating_and_bending_plate(self):
        item = transformed_stroke({"rotate": 15.0, "scale_x": 1.2, "scale_y": 1.2})
        for kind in ("translate", "rotate", "bend"):
            with self.subTest(kind=kind):
                err, keyed = self.check(item, kind)
                print(f"\nstroke with a rotate and scale transform, {kind}, frame 10: mean {err.mean():.3f} px, max {err.max():.3f} px")
                self.assertLess(err.mean(), 0.6)
                self.assertLess(err.max(), 1.7)

    def test_skew_a_stored_centre_and_the_matrix_stick_too(self):
        cases = (("skew", transformed_stroke({"skew_x": 12.0, "translate_x": 5.0})),
                 ("stored centre", transformed_stroke({"rotate": 20.0, "center_x": 160.0, "center_y": 100.0})),
                 ("matrix", transformed_stroke(None, MATRIX)),
                 ("default centre, rotate and matrix", transformed_stroke({"rotate": 30.0}, MATRIX)))
        for name, item in cases:
            with self.subTest(case=name):
                err, _ = self.check(item, "rotate")
                print(f"\n{name}: mean {err.mean():.3f} px, max {err.max():.3f} px")
                self.assertLess(err.mean(), 0.6)
                self.assertLess(err.max(), 1.7)

    def test_every_frame_on_the_way_stays_close(self):
        item = transformed_stroke({"rotate": 25.0, "scale_x": 1.1, "scale_y": 0.9, "skew_x": 8.0})
        rendered, keyed = self.follow(item, "rotate")
        stroke = [{"x": float(x), "y": float(y)} for x, y in rendered]
        for frame in range(2, 11):
            gx, gy = truth("rotate", frame, np.array([p["x"] - .5 for p in stroke]), np.array([p["y"] - .5 for p in stroke]))
            got = np.array([[p["x"] - .5, p["y"] - .5] for p in shape_model.resolve_paint_items({"items": [keyed]}, frame)[0]["points"]])
            err = np.hypot(got[:, 0] - gx, got[:, 1] - gy)
            self.assertLess(err.mean(), 0.6, f"frame {frame}")

    def test_the_painted_frame_stays_exactly_as_painted(self):
        item = transformed_stroke({"rotate": 15.0, "scale_x": 1.2, "scale_y": 1.2})
        _, keyed = self.follow(item, "rotate")
        self.assertEqual(drawn(keyed, REFERENCE).tolist(), drawn(item, REFERENCE).tolist())
        self.assertEqual(paintknobs.resolve_stroke(paintknobs.resolve_stuck(keyed, REFERENCE), REFERENCE, {})["brush"],
                         paintknobs.resolve_stroke(item, REFERENCE, {})["brush"])

    def test_the_drawn_brush_follows_the_plates_scale_from_its_painted_size(self):
        item = transformed_stroke({"scale_x": {"value": 1.0, "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 1.5}, {"frame": 10, "value": 1.5}]}}, "scale_y": 1.5})
        item["brush"] = {**item["brush"], "size": 10.0}
        rendered, keyed = self.follow(item, "zoom")
        at = lambda frame: paintknobs.resolve_stroke(paintknobs.resolve_stuck(keyed, frame), frame, {})["brush"]["size"]
        self.assertAlmostEqual(at(1), 10.0 * 1.5, places=6)
        self.assertAlmostEqual(at(10) / at(1), 1.135, delta=0.03)

    def test_a_stroke_without_a_transform_is_untouched_by_the_new_step(self):
        item = transformed_stroke()
        results = {2: {"points": [(1.0, 2.0)], "scale": 1.1, "rotation": 0.0, "shear": 0.0, "confidence": 1.0, "fallback": False, "low": []}}
        self.assertIs(rotostick.localise_results(item, results, 1), results)


class ShearTests(unittest.TestCase):
    OFFSET = [20.0, 10.0]

    def sheared(self, h=0.14):
        """The painted-frame to current displacement of a picture sheared by h about the stroke."""
        yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
        field = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        field[..., 0] = h * (yy - HEIGHT / 2)
        return field

    def test_the_shear_of_the_picture_is_measured(self):
        points = np.array([[150.5, 100.5], [170.5, 120.5], [190.5, 110.5]])
        scale, rotation, shear = rotostick.local_motion_shear(self.sheared(0.14), points, 10.0)
        self.assertAlmostEqual(scale, 1.0, delta=0.005)
        self.assertAlmostEqual(shear, 0.07, delta=0.005)
        self.assertEqual(rotostick.local_motion(self.sheared(0.14), points, 10.0), (scale, rotation))
        pure = rotostick.local_motion_shear(np.zeros((HEIGHT, WIDTH, 2), np.float32), points, 10.0)
        self.assertEqual(pure, (1.0, 0.0, 0.0))

    def test_a_turning_and_scaling_picture_has_no_shear(self):
        yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
        a, s = 0.1, 1.1
        dx, dy = xx - 160.0, yy - 120.0
        field = np.stack([s * (math.cos(a) * dx - math.sin(a) * dy) - dx, s * (math.sin(a) * dx + math.cos(a) * dy) - dy], axis=-1).astype(np.float32)
        scale, rotation, shear = rotostick.local_motion_shear(field, np.array([[150.5, 100.5], [170.5, 120.5]]), 10.0)
        self.assertAlmostEqual(shear, 0.0, delta=0.005)
        self.assertAlmostEqual(rotation, a, delta=0.005)

    def stroke_with_offset(self, shear, h=None):
        item = paint_stroke([(20.5, 30.5), (30.5, 36.5)], source_offset=list(self.OFFSET))
        results = {2: {"points": [(20.5, 30.5), (30.5, 36.5)], "scale": 1.0, "rotation": 0.0, "shear": shear,
                       "confidence": 1.0, "fallback": False, "low": []}}
        return rotostick.key_stroke(item, results, 1)

    def test_shear_reaches_the_clone_offset(self):
        h = 0.14
        keyed = self.stroke_with_offset(0.0)
        # the plate's own map of the offset: J (offset), J = [[1, h], [0, 1]]
        want = (self.OFFSET[0] + h * self.OFFSET[1], self.OFFSET[1])
        # with the measured shear (and the turn the polar split reads from the same map)
        points = np.array([[150.5, 100.5], [170.5, 120.5], [190.5, 110.5]])
        scale, rotation, shear = rotostick.local_motion_shear(self.sheared(h), points, 10.0)
        item = paint_stroke([(20.5, 30.5), (30.5, 36.5)], source_offset=list(self.OFFSET))
        results = {2: {"points": [(20.5, 30.5), (30.5, 36.5)], "scale": scale, "rotation": rotation, "shear": shear,
                       "confidence": 1.0, "fallback": False, "low": []}}
        with_shear = paintknobs.resolve_stuck(rotostick.key_stroke(item, results, 1), 2)["source_offset"]
        results[2]["shear"] = 0.0
        without = paintknobs.resolve_stuck(rotostick.key_stroke(item, results, 1), 2)["source_offset"]
        error_with = math.hypot(with_shear[0] - want[0], with_shear[1] - want[1])
        error_without = math.hypot(without[0] - want[0], without[1] - want[1])
        print(f"\nclone offset {self.OFFSET}, the plate's map gives {want}: with shear {with_shear} (off by {error_with:.2f} px), "
              f"without {without} (off by {error_without:.2f} px)")
        self.assertLess(error_with, 0.25)
        self.assertGreater(error_without, 3.0 * error_with)

    def test_shear_is_keyed_only_when_the_picture_shears(self):
        plain = self.stroke_with_offset(0.0)
        self.assertNotIn("shear", plain["stick"])
        sheared = self.stroke_with_offset(0.05)
        self.assertIn("curve", sheared["stick"]["shear"])
        self.assertEqual(paintknobs.resolve_stuck(sheared, 1)["source_offset"], self.OFFSET)

    def test_the_document_accepts_stick_shear_and_rejects_nonsense(self):
        paintknobs.validate_stick({"scale": 1.0, "rotation": 0.0, "shear": 0.1}, "item")
        with self.assertRaises(ValueError):
            paintknobs.validate_stick({"shear": "wide"}, "item")
        with self.assertRaises(ValueError):
            paintknobs.validate_stick({"skewness": 0.1}, "item")

    def test_the_key_reducer_keeps_the_shear_within_its_tolerance(self):
        from nodebased import rotokeyreduce
        points = [(20.5, 30.5), (30.5, 36.5), (40.5, 30.5)]
        item = paint_stroke(points, source_offset=[40.0, 20.0])
        results = {f: {"points": [(x + 1.0 * (f - 1), y) for x, y in points], "scale": 1.0, "rotation": 0.0,
                       "shear": 0.002 * (f - 1) ** 2, "confidence": 1.0, "fallback": False, "low": []} for f in range(2, 21)}
        keyed = rotostick.key_stroke(item, results, 1)
        reduced = rotokeyreduce.reduce_item(keyed, tolerance=0.5).item
        shear_keys = len(reduced["stick"]["shear"]["curve"]["keys"])
        self.assertLess(shear_keys, 20)
        for frame in range(1, 21):
            a = paintknobs.resolve_stuck(keyed, frame)["source_offset"]
            b = paintknobs.resolve_stuck(reduced, frame)["source_offset"]
            self.assertLess(math.hypot(a[0] - b[0], a[1] - b[1]), 0.75, f"frame {frame}")


class StickInTheAppWithTransformTests(AppCase):
    def test_a_stroke_with_a_transform_is_keyed_and_follows_the_plate(self):
        item = long_stroke(transform={"rotate": 15.0, "scale_x": 1.2, "scale_y": 1.2, "translate_x": 4.0})
        window = self.make_window([item])
        window.viewer.paint_selected_item_index = 0
        before = copy.deepcopy(window.dispatcher.document["node_data"]["paint"])
        self.run_stick(window)
        keyed = self.items(window)[0]
        self.assertIn("curve", keyed["points"][0]["x"])
        self.assertEqual(keyed["transform"], item["transform"])                 # the transform itself is left as it was
        rendered = drawn(item, REFERENCE)
        stroke = [{"x": float(x), "y": float(y)} for x, y in rendered]
        gx, gy = truth("rotate", 11, np.array([p["x"] - .5 for p in stroke]), np.array([p["y"] - .5 for p in stroke]))
        got = shape_model.resolve_paint_items(window.dispatcher.document["node_data"]["paint"], 11)[0]["points"]
        got = np.array([[p["x"] - .5, p["y"] - .5] for p in got])
        err = np.hypot(got[:, 0] - gx, got[:, 1] - gy)
        print(f"\nin the app, stroke with a transform, frame 11: mean {err.mean():.3f} px, max {err.max():.3f} px")
        self.assertLess(err.mean(), 0.7)
        self.assertLess(err.max(), 1.9)
        message = window.statusBar().currentMessage()
        self.assertIn("Stick to plate: 1 stroke keyed over 10 frames", message)
        self.assertNotIn("left alone", message)
        window.command({"op": "undo"})
        self.assertEqual(window.dispatcher.document["node_data"]["paint"], before)

    def test_a_locked_stroke_is_still_left_alone_and_a_matrix_stroke_sticks(self):
        window = self.make_window([long_stroke(locked=True, transform={"rotate": 5.0}), long_stroke(extra_matrix=MATRIX)])
        window.viewer.paint_selected_item_index = 0
        self.assertFalse(window.stick_roto_strokes("paint", item_indices=[0]))
        self.assertIn("locked", window.statusBar().currentMessage())
        self.run_stick(window, item_indices=[1])
        self.assertIn("curve", self.items(window)[1]["points"][0]["x"])
        self.assertNotIsInstance(self.items(window)[0]["points"][0]["x"], dict)           # the locked one has no keys


if __name__ == "__main__":
    unittest.main()
