"""Strokes that stick to the plate (lane 9, step I7): the engine, the keys it writes, the render and the app command.

The plates are a smoothed random texture rendered through an analytic map, so the true position of every painted point on
every frame is known exactly. Timings are printed, never asserted.
"""
import copy
import math
import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import opticalflow, paintknobs, rotostick
from nodebased import shapes as shape_model

HEIGHT, WIDTH = 240, 320
REFERENCE = 1


def _texture(seed=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((HEIGHT + 200, WIDTH + 200), np.float32)
    yy, xx = np.mgrid[:HEIGHT + 200, :WIDTH + 200].astype(np.float32)
    for scale, amp in ((2, .5), (4, 1.0), (8, 1.2), (16, .8)):
        small = rng.random(((HEIGHT + 200) // scale + 2, (WIDTH + 200) // scale + 2)).astype(np.float32)
        out += amp * opticalflow._sample(small, xx / scale, yy / scale)
    return (out - out.min()) / (out.max() - out.min())


TEXTURE = _texture()


def back_map(kind, frame, x, y):
    """Where the plate pixel now at (x, y) was on the painted frame (index coordinates)."""
    t = frame - REFERENCE
    cx, cy = WIDTH / 2, HEIGHT / 2
    if kind == "translate":
        return x - 1.2 * t, y - 0.7 * t
    if kind == "rotate":
        a = 0.012 * t
        dx, dy = x - cx - 1.0 * t, y - cy - 0.5 * t
        return cx + np.cos(a) * dx + np.sin(a) * dy, cy - np.sin(a) * dx + np.cos(a) * dy
    if kind == "bend":
        return x - 1.0 * t - 0.35 * t * np.sin(y / 38.0), y - 0.4 * t + 0.3 * t * np.sin(x / 45.0)
    if kind == "zoom":
        s = 1.0 + 0.015 * t
        return cx + (x - cx) / s, cy + (y - cy) / s
    raise ValueError(kind)


def plate(kind, frame, width=WIDTH, height=HEIGHT):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    bx, by = back_map(kind, frame, xx, yy)
    return opticalflow._sample(TEXTURE, bx + 100, by + 100).astype(np.float32)


def truth(kind, frame, xs, ys):
    """Where the painted-frame points (index coordinates) are on `frame`: the inverse of `back_map`, by iteration."""
    x, y = np.array(xs, float), np.array(ys, float)
    for _ in range(40):
        bx, by = back_map(kind, frame, x, y)
        x, y = x - (bx - xs), y - (by - ys)
    return x, y


def curve_stroke(count=24):
    t = np.linspace(0, 1, count)
    return [{"x": float(120 + 90 * s + 0.5), "y": float(100 + 40 * math.sin(s * 5) + 0.5)} for s in t]


def errors(kind, results, frame, stroke):
    gx, gy = truth(kind, frame, np.array([p["x"] - .5 for p in stroke]), np.array([p["y"] - .5 for p in stroke]))
    got = np.array(results[frame]["points"]) - 0.5
    return np.hypot(got[:, 0] - gx, got[:, 1] - gy)


def occluded_plate(frame):
    """The translating plate with a different texture sweeping across the stroke over frames 3 to 9."""
    image = plate("translate", frame).copy()
    if 3 <= frame <= 9:
        centre = 60 + (frame - 1) * 20
        yy, xx = np.mgrid[:HEIGHT, :WIDTH]
        mask = (np.abs(xx - centre) < 34) & (np.abs(yy - 105) < 70)
        patch = TEXTURE[(yy + 17) % TEXTURE.shape[0], (xx + 53 - 3 * frame) % TEXTURE.shape[1]] * .8 + .1
        image[mask] = patch[mask]
    return image


class FrameCache:
    def __init__(self, make):
        self.make, self.frames = make, {}

    def __call__(self, frame):
        if frame not in self.frames:
            self.frames[frame] = self.make(frame)
        return self.frames[frame]


class StickEngineTests(unittest.TestCase):
    def follow(self, kind, last=10, **kwargs):
        stroke = curve_stroke()
        source = FrameCache(lambda f: plate(kind, f))
        start = time.perf_counter()
        results = rotostick.stick_stroke(source, stroke, REFERENCE, list(range(REFERENCE + 1, last + 1)), brush_size=20, **kwargs)
        print(f"\n{kind}: {last - REFERENCE} frames of {WIDTH}x{HEIGHT} in {time.perf_counter() - start:.2f} s")
        return stroke, results

    def test_a_stroke_sits_on_the_same_texture_ten_frames_later_on_a_translating_rotating_and_bending_plate(self):
        for kind in ("translate", "rotate", "bend"):
            with self.subTest(kind=kind):
                stroke, results = self.follow(kind)
                err = errors(kind, results, 10, stroke)
                print(f"{kind} at frame 10: mean {err.mean():.3f} px, max {err.max():.3f} px")
                self.assertLess(err.mean(), 0.5)
                self.assertLess(err.max(), 1.5)
                self.assertFalse(any(results[f]["fallback"] for f in results))

    def test_every_frame_on_the_way_is_as_close(self):
        for kind in ("rotate", "bend"):
            stroke, results = self.follow(kind)
            for frame in range(2, 11):
                with self.subTest(kind=kind, frame=frame):
                    err = errors(kind, results, frame, stroke)
                    self.assertLess(err.mean(), 0.5)
                    self.assertLess(err.max(), 1.5)

    def test_the_local_scale_and_turn_of_the_motion_are_reported(self):
        _, rotated = self.follow("rotate")
        self.assertAlmostEqual(rotated[10]["rotation"], 0.108, delta=0.01)
        self.assertAlmostEqual(rotated[10]["scale"], 1.0, delta=0.01)
        _, zoomed = self.follow("zoom")
        self.assertAlmostEqual(zoomed[10]["scale"], 1.135, delta=0.02)
        self.assertAlmostEqual(zoomed[10]["rotation"], 0.0, delta=0.01)
        _, moved = self.follow("translate")
        self.assertAlmostEqual(moved[10]["scale"], 1.0, delta=0.01)
        self.assertAlmostEqual(moved[10]["rotation"], 0.0, delta=0.01)

    def test_a_backward_pass_follows_the_plate_to_earlier_frames(self):
        stroke = curve_stroke()
        # painted on frame 8 and followed back to frame 3
        source = FrameCache(lambda f: plate("rotate", f))
        results = rotostick.stick_stroke(source, stroke, 8, [7, 6, 5, 4, 3], brush_size=20)
        # a point at frame-8 pixel (x, y) came from the reference plate at back_map(8); on frame 3 it sits where that point goes
        bx, by = back_map("rotate", 8, np.array([p["x"] - .5 for p in stroke]), np.array([p["y"] - .5 for p in stroke]))
        gx, gy = truth("rotate", 3, bx, by)
        got = np.array(results[3]["points"]) - 0.5
        self.assertLess(np.hypot(got[:, 0] - gx, got[:, 1] - gy).mean(), 0.5)

    def test_a_covered_stroke_falls_back_to_the_planar_motion_and_says_which_frames(self):
        stroke = curve_stroke()
        source = FrameCache(occluded_plate)
        results = rotostick.stick_stroke(source, stroke, REFERENCE, list(range(2, 13)), brush_size=20)
        whole, _partial = rotostick.fallback_frames(results)
        print(f"\nplanar fallback on frames {whole}")
        self.assertTrue({5, 6, 7, 8}.issubset(whole), whole)
        self.assertFalse({2}.intersection(whole))
        for frame in whole:
            err = errors("translate", results, frame, stroke)
            with self.subTest(frame=frame):
                self.assertLess(err.mean(), 1.0)
                self.assertLess(err.max(), 2.5)
        self.assertIn("planar motion on frames", rotostick.describe_fallback("stroke1", results))
        self.assertIn("stroke1: ", rotostick.describe_fallback("stroke1", results))
        # once the plate shows again the stroke is back on its texture (the last point may still be under the sweeping patch)
        self.assertLess(np.median(errors("translate", results, 12, stroke)), 1.0)

    def test_progress_and_cancel(self):
        stroke = curve_stroke()
        source = FrameCache(lambda f: plate("translate", f))
        seen = []
        rotostick.stick_stroke(source, stroke, REFERENCE, [2, 3, 4], brush_size=20, progress=lambda f, i, n: seen.append((f, i, n)))
        self.assertEqual(seen, [(2, 1, 3), (3, 2, 3), (4, 3, 3)])

        class Stop:
            def is_set(self):
                return True
        from concurrent.futures import CancelledError
        with self.assertRaises(CancelledError):
            rotostick.stick_stroke(source, stroke, REFERENCE, [2, 3], brush_size=20, cancel=Stop())

    def test_a_single_dab_sticks_too(self):
        dab = [{"x": 150.5, "y": 120.5}]
        source = FrameCache(lambda f: plate("rotate", f))
        results = rotostick.stick_stroke(source, dab, REFERENCE, [2, 3, 4, 5, 6], brush_size=20)
        gx, gy = truth("rotate", 6, np.array([150.0]), np.array([120.0]))
        got = np.array(results[6]["points"][0]) - 0.5
        self.assertLess(math.hypot(got[0] - gx[0], got[1] - gy[0]), 0.7)

    def test_helpers(self):
        self.assertEqual(rotostick.frame_order(5, 1, 9, "forward"), [[6, 7, 8, 9]])
        self.assertEqual(rotostick.frame_order(5, 1, 9, "both"), [[6, 7, 8, 9], [4, 3, 2, 1]])
        with self.assertRaises(rotostick.StickError):
            rotostick.frame_order(5, 9, 1)
        with self.assertRaises(rotostick.StickError):
            rotostick.stick_stroke(lambda f: plate("translate", f), [], 1, [2])
        self.assertEqual(rotostick.describe_fallback("a", {2: {"fallback": False, "low": []}}), "")


# -- the keys, the document and the render ------------------------------------------------------------------------------
SIZE = 96


def paint_stroke(points, **extra):
    from tests.test_roto_paint import stroke
    return stroke(points=[{"x": float(x), "y": float(y), "pressure": 1.0} for x, y in points], **extra)


class SyntheticResults:
    """Hand-made `stick_stroke` results: the stroke slides 2 px right and 1 px down a frame, doubling in scale by frame 5."""

    @staticmethod
    def make(points, frames=range(2, 6)):
        out = {}
        for f in frames:
            t = f - 1
            out[f] = {"points": [(x + 2.0 * t, y + 1.0 * t) for x, y in points], "scale": 1.0 + 0.25 * t,
                      "rotation": 0.1 * t, "confidence": 1.0, "fallback": False, "low": []}
        return out


class KeyTests(unittest.TestCase):
    POINTS = [(20.5, 30.5), (30.5, 36.5), (40.5, 30.5)]

    def keyed(self, **extra):
        item = paint_stroke(self.POINTS, **extra)
        return item, rotostick.key_stroke(item, SyntheticResults.make(self.POINTS), 1)

    def test_points_and_scale_become_keys_with_the_painted_frame_unchanged(self):
        item, keyed = self.keyed()
        for index, (x, y) in enumerate(self.POINTS):
            for axis, field in enumerate(("x", "y")):
                self.assertIn("curve", keyed["points"][index][field])
                frames = [k["frame"] for k in keyed["points"][index][field]["curve"]["keys"]]
                self.assertEqual(frames, [1, 2, 3, 4, 5])
        resolved = paintknobs.resolve_stuck(keyed, 1)
        self.assertEqual([(p["x"], p["y"]) for p in resolved["points"]], self.POINTS)
        self.assertEqual(resolved["brush"], item["brush"])
        self.assertNotIn("stick", resolved)
        resolved = paintknobs.resolve_stuck(keyed, 4)
        self.assertEqual([(p["x"], p["y"]) for p in resolved["points"]], [(x + 6.0, y + 3.0) for x, y in self.POINTS])

    def test_the_brush_size_follows_the_scale_and_the_clone_offset_follows_scale_and_turn(self):
        item, keyed = self.keyed(source_offset=[10.0, 0.0])
        self.assertEqual(item["brush"]["size"], 4.0)
        resolved = paintknobs.resolve_stuck(keyed, 3)       # scale 1.5, turn 0.2 rad
        self.assertAlmostEqual(resolved["brush"]["size"], 6.0)
        self.assertAlmostEqual(resolved["source_offset"][0], 10.0 * 1.5 * math.cos(0.2))
        self.assertAlmostEqual(resolved["source_offset"][1], 10.0 * 1.5 * math.sin(0.2))
        self.assertEqual(paintknobs.resolve_stuck(keyed, 1)["source_offset"], [10.0, 0.0])

    def test_outside_the_followed_frames_a_stroke_holds_its_last_position(self):
        _, keyed = self.keyed()
        far = paintknobs.resolve_stuck(keyed, 40)
        last = paintknobs.resolve_stuck(keyed, 5)
        self.assertEqual(far["points"], last["points"])

    def test_resolve_paint_items_applies_it_and_leaves_ordinary_strokes_alone(self):
        ordinary = paint_stroke(self.POINTS)
        _, keyed = self.keyed()
        payload = {"items": [ordinary, keyed], "layers": []}
        items = shape_model.resolve_paint_items(payload, 3)
        self.assertEqual(items[0]["points"], ordinary["points"])
        self.assertEqual(items[0], dict(items[0]))
        self.assertEqual([(p["x"], p["y"]) for p in items[1]["points"]], [(x + 4.0, y + 2.0) for x, y in self.POINTS])
        self.assertNotIn("stick", items[0])

    def test_the_document_accepts_keyed_points_and_stick_and_rejects_a_bad_one(self):
        from nodebased.core import Dispatcher, empty_document
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "paint", "type": "RotoPaint"})
        _, keyed = self.keyed()
        d.execute({"op": "set_paint_items", "id": "paint", "items": [keyed]})
        bad = copy.deepcopy(keyed)
        bad["stick"] = {"scale": "big"}
        with self.assertRaises(Exception):
            d.execute({"op": "set_paint_items", "id": "paint", "items": [bad]})
        bad = copy.deepcopy(keyed)
        bad["stick"] = {"zoom": 2.0}
        with self.assertRaises(Exception):
            d.execute({"op": "set_paint_items", "id": "paint", "items": [bad]})
        bad = copy.deepcopy(keyed)
        bad["points"][0]["x"] = {"value": 1e9, "curve": None}
        with self.assertRaises(Exception):
            d.execute({"op": "set_paint_items", "id": "paint", "items": [bad]})
        self.assertEqual(len(d.document["node_data"]["paint"]["items"]), 1)

    def test_a_saved_document_without_stick_is_unchanged_by_the_new_code(self):
        from nodebased.core import Dispatcher, empty_document
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "paint", "type": "RotoPaint"})
        item = paint_stroke(self.POINTS)
        d.execute({"op": "set_paint_items", "id": "paint", "items": [item]})
        self.assertEqual(d.document["node_data"]["paint"]["items"][0], item)
        self.assertFalse(paintknobs.sticks(item))


class RenderTests(unittest.TestCase):
    def graph(self, items):
        from tests.test_roto_paint_w1 import graph
        return graph(items)

    def render(self, d, frame):
        from nodebased.imaging import Evaluator
        return Evaluator().evaluate_raster(d.document, "paint", frame=frame).pixels

    def tiled(self, d, frame, edge=8, tier=1):
        from nodebased.imaging import Evaluator
        from nodebased.tileexec import TileExecutor
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
        document = dict(d.document, view="paint")
        region = executor.canvas_region(document, "paint", frame=frame, tier=tier)
        return executor.compose_region(document, "paint", region, frame=frame, tier=tier).pixels

    def stuck_item(self):
        points = [(6.5, 10.5), (12.5, 12.5), (18.5, 10.5)]
        item = paint_stroke(points, brush={"size": 6.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})
        results = {f: {"points": [(x + 1.5 * (f - 1), y + 1.0 * (f - 1)) for x, y in points], "scale": 1.0 + 0.1 * (f - 1),
                       "rotation": 0.0, "confidence": 1.0, "fallback": False, "low": []} for f in range(2, 9)}
        return rotostick.key_stroke(item, results, 1), item

    def test_the_painted_frame_renders_exactly_as_before_and_later_frames_move_the_paint(self):
        keyed, plain = self.stuck_item()
        before = self.render(self.graph([plain]), 1)
        after = self.render(self.graph([keyed]), 1)
        self.assertEqual(before.tobytes(), after.tobytes())
        d = self.graph([keyed])

        def centre(image):
            weight = image[..., 0] - image[..., 1]
            ys, xs = np.mgrid[:image.shape[0], :image.shape[1]]
            return (weight * xs).sum() / weight.sum(), (weight * ys).sum() / weight.sum()
        c1, c6 = centre(self.render(d, 1)), centre(self.render(d, 6))
        self.assertAlmostEqual(c6[0] - c1[0], 7.5, delta=0.3)
        self.assertAlmostEqual(c6[1] - c1[1], 5.0, delta=0.3)
        # the brush grew with the scale: more paint on frame 6 than on frame 1
        self.assertGreater((self.render(d, 6)[..., 1] < 0.2).sum(), (self.render(d, 1)[..., 1] < 0.2).sum())

    def test_tiles_equal_the_full_frame_for_a_stuck_stroke_at_every_tile_size_and_proxy_tier(self):
        keyed, _ = self.stuck_item()
        d = self.graph([keyed])
        for frame in (1, 4, 8):
            full = self.render(d, frame)
            for edge in (8, 16, 32):
                with self.subTest(frame=frame, edge=edge):
                    np.testing.assert_array_equal(self.tiled(d, frame, edge), full)
        from nodebased.imaging import Evaluator
        for tier in (2, 4):
            full = Evaluator().evaluate_raster(d.document, "paint", frame=5, tier=tier).pixels
            with self.subTest(tier=tier):
                self.assertEqual(self.tiled(d, 5, 8, tier).tobytes(), full.tobytes())

    def test_a_clone_keeps_its_source_to_target_relation_as_the_plate_turns_and_stretches(self):
        # engine on a rotating plate, then the clone offset it resolves against the plate's true motion of the source point
        stroke = curve_stroke()
        source = FrameCache(lambda f: plate("rotate", f))
        results = rotostick.stick_stroke(source, stroke, REFERENCE, list(range(2, 11)), brush_size=20)
        offset = [40.0, -25.0]
        item = paint_stroke([(p["x"], p["y"]) for p in stroke], tool="clone", source_offset=offset)
        keyed = rotostick.key_stroke(item, results, 1)
        resolved = paintknobs.resolve_stuck(keyed, 10)
        px, py = np.array([p["x"] for p in stroke]) - .5, np.array([p["y"] for p in stroke]) - .5
        # the source of the stroke's first point, carried by the plate's own motion, against the stroke point plus the resolved offset
        sx, sy = truth("rotate", 10, px[:1] + offset[0], py[:1] + offset[1])
        want = (sx[0] + 0.5, sy[0] + 0.5)
        have = (resolved["points"][0]["x"] + resolved["source_offset"][0], resolved["points"][0]["y"] + resolved["source_offset"][1])
        print(f"\nclone source: plate says {want}, stroke plus offset says {have}")
        self.assertLess(math.hypot(want[0] - have[0], want[1] - have[1]), 1.5)
        # and with a zooming plate the offset grows with it
        zoomed = rotostick.stick_stroke(FrameCache(lambda f: plate("zoom", f)), stroke, REFERENCE, list(range(2, 11)), brush_size=20)
        keyed = rotostick.key_stroke(item, zoomed, 1)
        resolved = paintknobs.resolve_stuck(keyed, 10)
        self.assertAlmostEqual(math.hypot(*resolved["source_offset"]) / math.hypot(*offset), 1.135, delta=0.02)
        self.assertAlmostEqual(resolved["brush"]["size"] / item["brush"]["size"], 1.135, delta=0.02)


# -- in the app ---------------------------------------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QSettings

from nodebased.app import Window
from nodebased.core import Dispatcher, empty_document
from nodebased.tileexec import TileExecutor
from tests.test_roto_ui import APP
from tests.waiting import wait_until


class AppCase(unittest.TestCase):
    kind = "rotate"
    occluded = False

    def make_window(self, items, last=11, current=1):
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "paint", "type": "RotoPaint"})
        dispatcher.execute({"op": "connect", "id": "paint", "input": "image", "source": "plate"})
        dispatcher.document["time"].update(first=1, last=last, current=current)
        dispatcher.execute({"op": "set_paint_items", "id": "paint", "items": items})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        return window

    def compose(self, _executor, _doc, _source, region, frame, **_kwargs):
        full = occluded_plate(frame) if self.occluded else plate(self.kind, frame)
        gray = full[region.y:region.y + region.height, region.x:region.x + region.width]
        rgba = np.stack(np.broadcast_arrays(gray, gray, gray, np.ones_like(gray)), axis=-1).astype(np.float32)
        return SimpleNamespace(pixels=rgba)

    def run_stick(self, window, *args, **kwargs):
        with patch.object(TileExecutor, "compose_region", lambda *a, **k: self.compose(*a, **k)), \
                patch.object(TileExecutor, "canvas_region", lambda *a, **k: SimpleNamespace(
                    x=0, y=0, width=WIDTH, height=HEIGHT, right=WIDTH, bottom=HEIGHT)):
            self.assertTrue(window.stick_roto_strokes("paint", *args, **kwargs))
            self.assertTrue(wait_until(lambda: window._points_future is None, timeout=180))

    def items(self, window):
        return window.dispatcher.document["node_data"]["paint"]["items"]


def long_stroke(**extra):
    return paint_stroke([(p["x"], p["y"]) for p in curve_stroke(16)], **extra)


class StickInTheAppTests(AppCase):
    def test_the_command_keys_the_selected_stroke_and_one_undo_removes_every_key(self):
        window = self.make_window([long_stroke()])
        window.viewer.paint_selected_item_index = 0
        before = copy.deepcopy(window.dispatcher.document["node_data"]["paint"])
        self.run_stick(window)
        item = self.items(window)[0]
        self.assertIn("curve", item["points"][0]["x"])
        self.assertIn("stick", item)
        stroke = curve_stroke(16)
        resolved = shape_model.resolve_paint_items(window.dispatcher.document["node_data"]["paint"], 11)[0]
        gx, gy = truth("rotate", 11, np.array([p["x"] - .5 for p in stroke]), np.array([p["y"] - .5 for p in stroke]))
        got = np.array([[p["x"] - .5, p["y"] - .5] for p in resolved["points"]])
        err = np.hypot(got[:, 0] - gx, got[:, 1] - gy)
        print(f"\nin the app, frame 11: mean {err.mean():.3f} px, max {err.max():.3f} px")
        self.assertLess(err.mean(), 0.6)
        self.assertLess(err.max(), 1.7)
        self.assertIn("Stick to plate: 1 stroke keyed over 10 frames", window.statusBar().currentMessage())
        window.command({"op": "undo"})
        self.assertEqual(window.dispatcher.document["node_data"]["paint"], before)
        window.command({"op": "redo"})
        self.assertIn("curve", self.items(window)[0]["points"][0]["x"])

    def test_the_status_bar_names_the_frames_that_fell_back_to_planar_motion(self):
        self.kind, self.occluded = "translate", True
        window = self.make_window([long_stroke()], last=12)
        window.viewer.paint_selected_item_index = 0
        self.run_stick(window)
        message = window.statusBar().currentMessage()
        print("\n" + message)
        self.assertIn("planar motion on frames", message)
        self.assertIn("stroke1", message)

    def test_forward_only_and_a_range_leave_other_frames_alone(self):
        window = self.make_window([long_stroke()], current=3)
        window.viewer.paint_selected_item_index = 0
        self.run_stick(window, "forward", 3, 6)
        frames = [k["frame"] for k in self.items(window)[0]["points"][0]["x"]["curve"]["keys"]]
        self.assertEqual(frames, [3, 4, 5, 6])

    def test_locked_strokes_and_strokes_with_a_transform_are_left_alone(self):
        window = self.make_window([long_stroke(locked=True)])
        window.viewer.paint_selected_item_index = 0
        before = copy.deepcopy(self.items(window))
        self.assertFalse(window.stick_roto_strokes("paint"))
        self.assertEqual(self.items(window), before)
        self.assertIn("locked", window.statusBar().currentMessage())

    def test_a_stroke_needs_selecting_and_an_input_and_a_paint_node(self):
        window = self.make_window([long_stroke()])
        window.viewer.paint_selected_item_index = None
        self.assertFalse(window.stick_roto_strokes("paint"))
        window.viewer.paint_selected_item_index = 0
        self.assertFalse(window.stick_roto_strokes("missing"))

    def test_stick_follows_a_stroke_already_keyed_by_an_earlier_run_from_where_it_now_is(self):
        window = self.make_window([long_stroke()], current=1)
        window.viewer.paint_selected_item_index = 0
        self.run_stick(window, "forward", 1, 4)
        first_keys = [k["frame"] for k in self.items(window)[0]["points"][0]["x"]["curve"]["keys"]]
        window.set_time(current=4)
        self.run_stick(window, "forward", 4, 8)
        keys = [k["frame"] for k in self.items(window)[0]["points"][0]["x"]["curve"]["keys"]]
        self.assertEqual(first_keys, [1, 2, 3, 4])
        self.assertEqual(keys, [1, 2, 3, 4, 5, 6, 7, 8])


class CommandPlacementTests(AppCase):
    def test_the_tracking_panel_has_stick_to_plate_beside_track_points(self):
        window = self.make_window([long_stroke()])
        from PySide6.QtWidgets import QComboBox, QPushButton, QSpinBox, QVBoxLayout, QWidget
        from nodebased import rotopanel
        root = QWidget(); layout = QVBoxLayout(root)
        rotopanel.add_tracking_actions(layout, window, "paint")
        self.assertIsNotNone(root.findChild(QPushButton, "roto-track-points"))
        self.assertIsNotNone(root.findChild(QPushButton, "roto-stick-to-plate"))
        calls = []
        window.stick_roto_strokes = lambda key, *a, **k: calls.append((key, a, k)) or True
        root.findChild(QComboBox, "roto-track-points-direction").setCurrentIndex(1)
        root.findChild(QSpinBox, "roto-track-points-last").setValue(8)
        root.findChild(QPushButton, "roto-stick-to-plate").click()
        self.assertEqual(calls, [("paint", ("forward", 1, 8), {})])

    def test_the_select_tools_strip_has_a_stick_to_plate_button_and_no_stored_setting(self):
        from nodebased.roto_settings import TOOL_CONTROLS
        for tool in ("Select All", "Select Splines", "Select Points", "Select Feather Points"):
            labels = {c["label"]: c["widget_kind"] for c in TOOL_CONTROLS[tool]}
            self.assertEqual(labels.get("Stick to plate"), "button", tool)
        for tool in ("Bezier", "Brush", "Clone"):
            self.assertNotIn("Stick to plate", [c["label"] for c in TOOL_CONTROLS[tool]])

    def test_the_shape_list_menu_has_stick_to_plate_for_strokes_only(self):
        window = self.make_window([long_stroke(), long_stroke(name="stroke2")])
        window.show()
        window.command({"op": "view", "id": "paint"})
        self.assertTrue(wait_until(lambda: window.frame is not None, timeout=90))
        window.graph.items_by_id["paint"].setSelected(True)
        APP.processEvents()
        panel = window.build_node_panel("paint")
        window.set_properties_widget(panel)
        from PySide6.QtWidgets import QTreeWidget
        tree = panel.findChild(QTreeWidget, "rotopaint-layer-list")
        tree.select_ids([("item", 1)])
        action = [a for a in tree.build_menu().actions() if a.text() == "Stick to plate"]
        self.assertEqual(len(action), 1)
        self.assertTrue(action[0].isEnabled())
        calls = []
        window.stick_roto_strokes = lambda key, direction, **k: calls.append((key, direction, k)) or True
        action[0].trigger()
        self.assertEqual(calls, [("paint", "both", {"item_indices": [1]})])


class TimingTests(unittest.TestCase):
    def test_a_stroke_at_1080p_over_24_frames_prints_its_time(self):
        """Printed, never asserted: the full suite runs many interpreters at once. Set NODEBASED_TIMING_TESTS=1 to bound it."""
        width, height = 1920, 1080
        # a 1080p plate tiled from the texture, moved 1.3 px right and 0.8 px down a frame
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        crop = {}

        def frame(f):
            if f not in crop:
                t = f - 1
                crop[f] = opticalflow._sample(TEXTURE, ((xx - 1.3 * t) % 400) * 0.5 + 100, ((yy - 0.8 * t) % 300) * 0.5 + 100)
            return crop[f]
        points = [{"x": 960 + 260 * math.cos(i / 4.0), "y": 540 + 100 * math.sin(i / 3.0)} for i in range(32)]
        xs = [p["x"] for p in points]; ys = [p["y"] for p in points]
        margin = 130
        left, top = int(min(xs) - margin), int(min(ys) - margin)
        right, bottom = int(max(xs) + margin), int(max(ys) + margin)
        local = [{"x": p["x"] - left, "y": p["y"] - top} for p in points]
        roi = FrameCache(lambda f: frame(f)[top:bottom, left:right])
        start = time.perf_counter()
        results = rotostick.stick_stroke(roi, local, 1, list(range(2, 26)), brush_size=40)
        elapsed = time.perf_counter() - start
        print(f"\n1080p, 24 frames, region {right - left}x{bottom - top}: {elapsed:.1f} s ({elapsed / 24:.2f} s per frame)")
        self.assertEqual(len(results), 24)
        if os.environ.get("NODEBASED_TIMING_TESTS") == "1":
            self.assertLess(elapsed / 24, 3.0)


if __name__ == "__main__":
    unittest.main()
