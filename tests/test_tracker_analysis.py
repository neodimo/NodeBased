"""Deterministic pixel Tracker analysis and atomic desktop wiring."""
import tests.isolation  # noqa: F401  (keep Qt settings out of the real user file)
import os
import threading
import time
import unittest
from concurrent.futures import Future

import numpy as np

from nodebased.core import Dispatcher, empty_document
from nodebased.tracker import AnalysisError, analyse, match_pattern


def texture(size=96, seed=7):
    rng = np.random.default_rng(seed)
    rgb = rng.random((size, size), dtype=np.float32)
    image = np.ones((size, size, 4), dtype=np.float32)
    image[..., :3] = rgb[..., None]
    return image


def moved(image, dx, dy):
    out = np.zeros_like(image)
    y0, y1 = max(0, dy), min(image.shape[0], image.shape[0] + dy)
    x0, x1 = max(0, dx), min(image.shape[1], image.shape[1] + dx)
    out[y0:y1, x0:x1] = image[y0-dy:y1-dy, x0-dx:x1-dx]
    return out


def warped_patch(pattern, angle_degrees, scale):
    height, width = pattern.shape
    cy, cx = (height - 1) / 2, (width - 1) / 2
    yy, xx = np.mgrid[:height, :width].astype(float)
    angle = np.deg2rad(angle_degrees)
    dx, dy = xx-cx, yy-cy
    sx = cx + (np.cos(angle)*dx + np.sin(angle)*dy) / scale
    sy = cy + (-np.sin(angle)*dx + np.cos(angle)*dy) / scale
    return _sample(pattern, sx, sy)


def _sample(image, xs, ys):
    xs = np.clip(xs, 0, image.shape[1] - 1.000001)
    ys = np.clip(ys, 0, image.shape[0] - 1.000001)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs-x0, ys-y0
    return (image[y0, x0]*(1-fx)*(1-fy) + image[y0, x0+1]*fx*(1-fy)
            + image[y0+1, x0]*(1-fx)*fy + image[y0+1, x0+1]*fx*fy)


def warped_plate(angle_degrees=0.0, scale=1.0):
    yy, xx = np.mgrid[:96, :96].astype(float)
    angle = np.deg2rad(angle_degrees)
    dx, dy = xx-48, yy-48
    sx = 48 + (np.cos(angle)*dx + np.sin(angle)*dy)/scale
    sy = 48 + (-np.sin(angle)*dx + np.cos(angle)*dy)/scale
    return (np.exp(-((sx-44)**2+(sy-45)**2)/18) + .6*np.exp(-((sx-53)**2+(sy-50)**2)/9)
            + .3*np.sin(sx*.4+sy*.17)).astype(np.float32)


class TrackerAnalysisTests(unittest.TestCase):
    def test_integer_peak_and_subpixel_refinement_are_bounded(self):
        first = texture()
        second = moved(first, 3, 2)
        x, y, score = match_pattern(first, second, (48.5, 48.5), 6, 10)
        self.assertGreater(score, 0.99)
        self.assertAlmostEqual(x, 51.5, delta=0.51)
        self.assertAlmostEqual(y, 50.5, delta=0.51)

    def test_translate_warp_preserves_existing_result_and_rotation_fit_uses_limits(self):
        from nodebased.tracker import _fit_pattern_warp
        full = texture()
        translated = moved(full, 2, -1)
        self.assertEqual(match_pattern(full, translated, (48.5, 48.5), 6, 8),
                         match_pattern(full, translated, (48.5, 48.5), 6, 8, warp="Translate"))
        yy, xx = np.mgrid[:25, :25]
        pattern = (np.exp(-((xx-8)**2+(yy-10)**2)/18) + .6*np.exp(-((xx-17)**2+(yy-15)**2)/9)
                   + .3*np.sin(xx*.4+yy*.17))
        plate = np.pad(warped_patch(pattern, 2.0, 1.0), 3, mode="edge")
        one = _fit_pattern_warp(pattern, plate, 15, 15, 12, "Translate/Rotate", 1, 0.01)
        fitted = _fit_pattern_warp(pattern, plate, 15, 15, 12, "Translate/Rotate", 100, 0.01)
        self.assertAlmostEqual(fitted[2], 2.0, delta=0.2)
        harder = np.pad(warped_patch(pattern, 8.0, 1.0), 3, mode="edge")
        one = _fit_pattern_warp(pattern, harder, 15, 15, 12, "Translate/Rotate", 1, 0.01)
        many = _fit_pattern_warp(pattern, harder, 15, 15, 12, "Translate/Rotate", 100, 0.01)
        loose = _fit_pattern_warp(pattern, harder, 15, 15, 12, "Translate/Rotate", 100, 1.0)
        self.assertGreater(abs(one[2] - 8.0), abs(many[2] - 8.0))
        self.assertGreater(abs(loose[2] - 8.0), abs(many[2] - 8.0))

    def test_scale_warp_recovers_synthetic_pattern_scale(self):
        from nodebased.tracker import _fit_pattern_warp
        yy, xx = np.mgrid[:25, :25]
        pattern = (np.exp(-((xx-8)**2+(yy-10)**2)/18) + .6*np.exp(-((xx-17)**2+(yy-15)**2)/9)
                   + .3*np.sin(xx*.4+yy*.17))
        plate = np.pad(warped_patch(pattern, 0.0, 1.03), 3, mode="edge")
        fitted = _fit_pattern_warp(pattern, plate, 15, 15, 12, "Translate/Rotate/Scale", 100, 0.01)
        self.assertAlmostEqual(fitted[3], 1.03, delta=0.005)

    def test_analysis_records_pattern_warp_for_single_track_transform(self):
        from nodebased.tracker import solve
        reference_values, current_values = warped_plate(), warped_plate(2.0, 1.03)
        reference = np.ones((96, 96, 4), np.float32); current = reference.copy()
        reference[..., :3] = reference_values[..., None]
        current[..., :3] = current_values[..., None]
        info = {}
        result = analyse({1: reference, 2: current}, 1, (48.5, 48.5), 12, 12,
                         last_frame=2, warp="Translate/Rotate/Scale", info=info)
        self.assertAlmostEqual(result[2][0], 48.5, delta=0.2)
        self.assertAlmostEqual(info["warps"][2]["rotate"], 2.0, delta=0.2)
        self.assertAlmostEqual(info["warps"][2]["scale"], 1.03, delta=0.005)
        keys = lambda values, initial: {"value": initial, "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": initial}, {"frame": 2, "value": values}]}}
        payload = {"tracks": [{"name": "one", "enabled": 1, "x": {"value": 48.5, "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 48.5}, {"frame": 2, "value": result[2][0]}]}},
            "y": 48.5, "warp_rotation": keys(info["warps"][2]["rotate"], 0.0),
            "warp_scale": keys(info["warps"][2]["scale"], 1.0)}]}
        solved = solve(payload, 2, {"reference_frame": 1, "mode": "match_move", "livelink_transform": 1,
                                   "apply_translate": 1, "apply_rotate": 1, "apply_scale": 1})
        self.assertAlmostEqual(solved["rotate"], 2.0, delta=0.2)
        self.assertAlmostEqual(solved["scale"], 1.03, delta=0.005)

    def test_five_frame_warp_tracks_rotation_and_scale(self):
        from nodebased.tracker import match_pattern
        from nodebased.trackerknobs import analysis_options
        frames = {frame: warped_plate(2.0*(frame-1), 1.0) for frame in range(1, 6)}
        info = {}
        rotations = analyse(frames, 1, (48.5, 48.5), 12, 20, last_frame=5,
                            warp="Translate/Rotate", info=info)
        self.assertAlmostEqual(info["warps"][5]["rotate"], 8.0, delta=0.2)
        fixed = match_pattern(frames[1], frames[5], (48.5, 48.5), 12, 20)
        fitted = match_pattern(frames[1], frames[5], (48.5, 48.5), 12, 20,
                               warp="Translate/Rotate", max_iter=100)
        self.assertGreater(fitted[2], fixed[2])
        scale_frames = {frame: warped_plate(0.0, 1.03**(frame-1)) for frame in range(1, 6)}
        scale_info = {}
        analyse(scale_frames, 1, (48.5, 48.5), 12, 20, last_frame=5,
                warp="Translate/Rotate/Scale", info=scale_info)
        self.assertAlmostEqual(scale_info["warps"][5]["scale"], 1.03**4, delta=0.005)
        self.assertEqual(analysis_options({"auto_regrab_pattern": 1})["grab"].behavior, "every frame")

    def test_auto_regrab_changes_track_when_pattern_appearance_evolves(self):
        from nodebased.trackerknobs import analysis_options
        frames = {frame: warped_plate(2.0*(frame-1), 1.0) for frame in range(1, 9)}
        fixed = analyse(frames, 1, (55.5, 48.5), 5, 16, last_frame=8)
        adaptive = analyse(frames, 1, (55.5, 48.5), 5, 16, last_frame=8,
                           **analysis_options({"auto_regrab_pattern": 1}))
        self.assertGreater(np.linalg.norm(np.subtract(fixed[8], adaptive[8])), 0.25)

    def test_negative_data_window_origin_is_preserved(self):
        from nodebased.raster import Raster
        from nodebased.tiers import Region
        first = Raster(texture(), Region(-20, -10, 96, 96))
        second = Raster(moved(first.pixels, 2, -1), Region(-20, -10, 96, 96))
        x, y, _ = match_pattern(first, second, (28.5, 38.5), 6, 10)
        self.assertAlmostEqual(x, 30.5, delta=0.51)
        self.assertAlmostEqual(y, 37.5, delta=0.51)

    def test_ordered_analysis_seeds_each_frame(self):
        first = texture()
        frames = {frame: moved(first, 2 * (frame - 1), frame - 1) for frame in range(1, 4)}
        result = analyse(frames, 1, (48.5, 48.5), 5, 6, last_frame=3)
        self.assertEqual(list(result), [1, 2, 3])
        self.assertAlmostEqual(result[3][0], 52.5, delta=0.51)
        self.assertAlmostEqual(result[3][1], 50.5, delta=0.51)

    def test_region_reads_match_full_frame_tracker_analysis(self):
        from nodebased.raster import Raster
        from nodebased.tiers import Region
        first = texture(128)
        frames = {frame: moved(first, 2 * (frame - 1), frame - 1) for frame in range(1, 4)}

        def roi_frame(frame, center, margin):
            image = frames[frame]
            x0 = max(0, int(np.floor(center[0] - margin - .5)))
            y0 = max(0, int(np.floor(center[1] - margin - .5)))
            x1 = min(image.shape[1], int(np.ceil(center[0] + margin + .5)) + 1)
            y1 = min(image.shape[0], int(np.ceil(center[1] + margin + .5)) + 1)
            return Raster(image[y0:y1, x0:x1], Region(x0, y0, x1-x0, y1-y0),
                          Region(0, 0, image.shape[1], image.shape[0]))

        full = analyse(frames, 1, (60.5, 60.5), 6, 12, last_frame=3)
        region = analyse(frames, 1, (60.5, 60.5), 6, 12, last_frame=3,
                         roi_frame=roi_frame)
        self.assertEqual(region, full)

    def test_backward_analysis_tracks_from_reference_to_inclusive_start(self):
        first = texture()
        frames = {frame: moved(first, 3 * (frame - 1), -2 * (frame - 1))
                  for frame in range(1, 5)}
        result = analyse(frames, 4, (57.5, 42.5), 5, 8, first_frame=1,
                         last_frame=4, direction="backward")
        self.assertEqual(list(result), [4, 3, 2, 1])
        self.assertAlmostEqual(result[1][0], 48.5, delta=0.51)
        self.assertAlmostEqual(result[1][1], 48.5, delta=0.51)

    def test_analysis_can_match_a_chosen_colour_channel(self):
        first = texture()
        first[..., :3] = 0
        first[..., 0] = texture(seed=19)[..., 0]
        second = moved(first, 2, 1)
        x, y, score = match_pattern(first, second, (48.5, 48.5), 5, 8, channels="red")
        self.assertGreater(score, 0.99)
        self.assertAlmostEqual(x, 50.5, delta=0.51)
        self.assertAlmostEqual(y, 49.5, delta=0.51)

    def test_rejects_texture_radii_and_bounds(self):
        flat = np.ones((32, 32, 4), np.float32)
        with self.assertRaisesRegex(AnalysisError, "insufficient texture"):
            match_pattern(flat, flat, (16.5, 16.5), 3, 5)
        with self.assertRaisesRegex(AnalysisError, "pattern_radius"):
            match_pattern(flat, flat, (16.5, 16.5), 0, 2)
        with self.assertRaisesRegex(AnalysisError, "out of bounds"):
            match_pattern(texture(20), texture(20), (1.5, 1.5), 4, 6)

    def test_rejects_an_occluded_or_unrelated_pattern(self):
        with self.assertRaisesRegex(AnalysisError, "no reliable match"):
            match_pattern(texture(seed=7), texture(seed=99), (48.5, 48.5), 6, 10)

    def test_set_tracks_is_atomic_and_undoable(self):
        document = empty_document()
        dispatcher = Dispatcher(document)
        dispatcher.execute({"op": "create", "id": "t", "type": "Tracker"})
        existing = [{"name": "keep", "enabled": 0, "x": 4.0, "y": 5.0}]
        dispatcher.execute({"op": "set_tracks", "id": "t", "tracks": existing})
        before = dispatcher.document
        with self.assertRaises(ValueError):
            dispatcher.execute({"op": "set_tracks", "id": "t", "tracks": [{"name": "bad"}]})
        self.assertEqual(dispatcher.document, before)
        updated = existing + [{"name": "new", "enabled": 1, "x": 8.0, "y": 9.0}]
        dispatcher.execute({"op": "set_tracks", "id": "t", "tracks": updated})
        dispatcher.execute({"op": "undo"})
        self.assertEqual(dispatcher.document["node_data"]["t"]["tracks"], existing)
        dispatcher.execute({"op": "redo"})
        self.assertEqual(dispatcher.document["node_data"]["t"]["tracks"], updated)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "offscreen UI test")
class TrackerUiTests(unittest.TestCase):
    def test_inspector_exposes_pick_and_analysis_controls(self):
        from PySide6.QtWidgets import QApplication, QPushButton
        from nodebased.app import Window
        app = QApplication.instance() or QApplication([])
        doc = empty_document()
        dispatcher = Dispatcher(doc)
        dispatcher.execute({"op": "create", "id": "c", "type": "Constant"})
        dispatcher.execute({"op": "create", "id": "t", "type": "Tracker"})
        dispatcher.execute({"op": "connect", "id": "t", "input": "image", "source": "c"})
        dispatcher.execute({"op": "view", "id": "t"})
        window = Window(dispatcher.document)
        window.graph.items_by_id["t"].setSelected(True)
        window.inspect("t")
        labels = [button.text() for button in window.properties.findChildren(QPushButton)]
        self.assertIn("Add track point at reference…", labels)
        self.assertIn("Analyze forward", labels)
        self.assertIn("Analyze backward", labels)
        before = window.dispatcher.document
        window.add_tracker_point((20.5, 20.5))
        self.assertEqual(window.dispatcher.document, before)
        window.close()
        window.executor.shutdown(wait=True, cancel_futures=True)
        app.processEvents()

    def test_completed_analysis_commits_to_original_tracker_after_selection_changes(self):
        from PySide6.QtWidgets import QApplication
        from nodebased.app import Window
        app = QApplication.instance() or QApplication([])
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "c", "type": "Constant"})
        for key in ("t", "other"):
            dispatcher.execute({"op": "create", "id": key, "type": "Tracker"})
            dispatcher.execute({"op": "connect", "id": key, "input": "image", "source": "c"})
        dispatcher.execute({"op": "view", "id": "t"})
        window = Window(dispatcher.document)
        window.graph.items_by_id["t"].setSelected(True)
        window.add_tracker_point((20.5, 20.5))
        seed = dict(window._tracker_seed)
        window._tracker_job = {"key": "t", "index": 0, "seed": seed,
                               "base_tracks": [], "reference_frame": 1}
        window._tracker_cancel = threading.Event()
        window._tracker_future = Future()
        window._tracker_future.set_result({1: (20.5, 20.5), 2: (22.5, 21.5)})
        window.graph.items_by_id["t"].setSelected(False)
        window.graph.items_by_id["other"].setSelected(True)
        window._poll_tracker_analysis()
        self.assertEqual(window.dispatcher.document["node_data"]["t"]["tracks"][0]["name"], "track1")
        self.assertNotIn("other", window.dispatcher.document.get("node_data", {}))
        window.saved_document = window.dispatcher.document
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
