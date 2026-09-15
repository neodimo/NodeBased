"""Deterministic pixel Tracker analysis and atomic desktop wiring."""
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


class TrackerAnalysisTests(unittest.TestCase):
    def test_integer_peak_and_subpixel_refinement_are_bounded(self):
        first = texture()
        second = moved(first, 3, 2)
        x, y, score = match_pattern(first, second, (48.5, 48.5), 6, 10)
        self.assertGreater(score, 0.99)
        self.assertAlmostEqual(x, 51.5, delta=0.51)
        self.assertAlmostEqual(y, 50.5, delta=0.51)

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
