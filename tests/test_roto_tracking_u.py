import unittest

import numpy as np


class RotoTrackingExportTests(unittest.TestCase):
    def test_cornerpin_exports_modes_and_link_setting(self):
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QPushButton
        from tests.test_roto_ui import APP
        from tests.test_roto import square
        from nodebased.animation import resolve_document
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "roto", "type": "Roto"})
        dispatcher.execute({"op": "connect", "id": "roto", "input": "bg", "source": "plate"})
        dispatcher.document["time"].update(first=1, last=2, current=1)
        dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": [square(10, 10, 30, 30)]})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        window.inspect("roto")
        self.assertIsNotNone(window.properties.widget().findChild(QPushButton, "roto-track-shape"))
        self.assertIsNotNone(window.properties.widget().findChild(QPushButton, "roto-create-tracking-export"))
        for i, (x, y) in enumerate(((10, 10), (30, 10), (30, 30), (10, 30)), 1):
            for axis, start, end in (("x", x, x+3), ("y", y, y)):
                window.command({"op": "set_curve", "id": "roto", "param": f"pt{i}_{axis}",
                                "curve": {"interpolation": "linear", "keys": [
                                    {"frame": 1, "value": start}, {"frame": 2, "value": end}]}})
        window.command({"op": "set", "id": "roto", "param": "export_menu", "value": "CornerPin2D (relative)"})
        self.assertTrue(window.export_roto_tracking("roto"))
        pin_id = next(k for k, n in window.dispatcher.document["nodes"].items() if n["type"] == "CornerPin")
        self.assertIn("to1_x", window.dispatcher.document["expressions"][pin_id])
        window.command({"op": "set", "id": "roto", "param": "of1_x", "value": 5.0})
        resolved_reference = resolve_document(window.dispatcher.document, 1)
        self.assertAlmostEqual(resolved_reference["nodes"][pin_id]["params"]["to1_x"], 15.0)
        resolved = resolve_document(window.dispatcher.document, 2)
        self.assertAlmostEqual(resolved["nodes"][pin_id]["params"]["to1_x"], 18.0)
        self.assertAlmostEqual(resolved["nodes"][pin_id]["params"]["from1_x"], 10.0)

        window.command({"op": "set", "id": "roto", "param": "link_output", "value": 0})
        window.command({"op": "set", "id": "roto", "param": "export_menu", "value": "CornerPin2D (stabilize)"})
        self.assertTrue(window.export_roto_tracking("roto"))
        pins = [(k, n) for k, n in window.dispatcher.document["nodes"].items() if n["type"] == "CornerPin"]
        stabilize_id, stabilize = pins[-1]
        resolved = resolve_document(window.dispatcher.document, 2)
        self.assertAlmostEqual(resolved["nodes"][stabilize_id]["params"]["from1_x"], 18.0)
        self.assertAlmostEqual(resolved["nodes"][stabilize_id]["params"]["to1_x"], 15.0)
        resolved_reference = resolve_document(window.dispatcher.document, 1)
        self.assertAlmostEqual(resolved_reference["nodes"][stabilize_id]["params"]["from1_x"], 15.0)
        window.command({"op": "set", "id": "roto", "param": "of1_x", "value": 15.0})
        resolved = resolve_document(window.dispatcher.document, 2)
        self.assertAlmostEqual(resolved["nodes"][stabilize_id]["params"]["from1_x"], 18.0)

        window.command({"op": "set", "id": "roto", "param": "export_menu", "value": "CornerPin2D (absolute)"})
        self.assertTrue(window.export_roto_tracking("roto"))
        pins = [(k, n) for k, n in window.dispatcher.document["nodes"].items() if n["type"] == "CornerPin"]
        absolute_id = pins[-1][0]
        resolved = resolve_document(window.dispatcher.document, 1)
        self.assertEqual(resolved["nodes"][absolute_id]["params"]["from2_x"], 960.0)
        self.assertEqual(resolved["nodes"][absolute_id]["params"]["from3_y"], 540.0)
        self.assertEqual(resolved["nodes"][absolute_id]["params"]["to1_x"], 25.0)
        resolved = resolve_document(window.dispatcher.document, 2)
        self.assertEqual(resolved["nodes"][absolute_id]["params"]["to1_x"], 28.0)

    def test_tracker_export_creates_four_animated_tracks(self):
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QSettings
        from tests.test_roto_ui import APP
        from tests.test_roto import square
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        from nodebased.shapes import resolve_tracks
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "roto", "type": "Roto"})
        dispatcher.execute({"op": "connect", "id": "roto", "input": "bg", "source": "plate"})
        dispatcher.document["time"].update(first=1, last=2, current=1)
        dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": [square(10, 10, 30, 30)]})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        for i, (x, y) in enumerate(((10, 10), (30, 10), (30, 30), (10, 30)), 1):
            for axis, start, end in (("x", x, x+3), ("y", y, y)):
                window.command({"op": "set_curve", "id": "roto", "param": f"pt{i}_{axis}",
                                "curve": {"interpolation": "linear", "keys": [
                                    {"frame": 1, "value": start}, {"frame": 2, "value": end}]}})
        window.command({"op": "set", "id": "roto", "param": "export_menu", "value": "Tracker"})
        self.assertTrue(window.export_roto_tracking("roto"))
        tracker_id = next(k for k, n in window.dispatcher.document["nodes"].items() if n["type"] == "Tracker")
        tracks = window.dispatcher.document["node_data"][tracker_id]["tracks"]
        self.assertEqual(len(tracks), 4)
        np.testing.assert_allclose([[p["x"], p["y"]] for p in resolve_tracks({"tracks": tracks}, 1)],
                                   [[10, 10], [30, 10], [30, 30], [10, 30]])
        np.testing.assert_allclose([[p["x"], p["y"]] for p in resolve_tracks({"tracks": tracks}, 2)],
                                   [[13, 10], [33, 10], [33, 30], [13, 30]])

    def test_track_selected_shape_passes_its_bounding_quad_to_planar_analysis(self):
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QSettings
        from tests.test_roto_ui import APP
        from tests.test_roto import square
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "roto", "type": "Roto"})
        dispatcher.execute({"op": "connect", "id": "roto", "input": "bg", "source": "plate"})
        dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": [square(18, 21, 45, 53)]})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        captured = {}
        window.analyse_planar = lambda key, **kwargs: captured.update(key=key, **kwargs) or True
        window.viewer.roto_selected_shape_index = 0
        self.assertTrue(window.track_roto_shape("roto"))
        self.assertEqual(captured["key"], "roto")
        np.testing.assert_array_equal(captured["corners_override"],
                                      [[18, 21], [45, 21], [45, 53], [18, 53]])

    def test_roto_analysis_settings_prepare_channels_filter_and_clamp(self):
        from nodebased import trackerknobs
        from nodebased.tracker import _analysis_pixels
        from nodebased.core import Dispatcher, empty_document
        defaults = trackerknobs.new_params("Roto")
        self.assertEqual(defaults["pretrack_filter"], "none")
        self.assertEqual(defaults["tracking_channels"], "rgb")
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "roto", "type": "Roto"})
        with self.assertRaisesRegex(ValueError, "pretrack_filter"):
            dispatcher.execute({"op": "set", "id": "roto", "param": "pretrack_filter", "value": "median"})
        image = np.array([[[-2.0, 2.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        unclamped, _ = _analysis_pixels(image, "red", "none", False)
        clamped, _ = _analysis_pixels(image, "red", "none", True)
        self.assertEqual(float(unclamped[0, 0]), -2.0)
        self.assertEqual(float(clamped[0, 0]), 0.0)
        filtered, _ = _analysis_pixels(image, "red", "adjust contrast", False)
        self.assertGreater(float(filtered.max()), float(filtered.min()))

    def test_selected_shape_tracks_three_pixels_per_frame_and_stores_corners(self):
        import tests.isolation  # noqa: F401
        from types import SimpleNamespace
        from unittest.mock import patch
        from PySide6.QtCore import QSettings
        from tests.test_roto_ui import APP
        from tests.test_roto import square
        from tests.waiting import wait_until
        from nodebased.app import Window
        from nodebased.animation import resolve_document
        from nodebased.core import Dispatcher, empty_document
        from nodebased.tileexec import TileExecutor
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "roto", "type": "Roto"})
        dispatcher.execute({"op": "connect", "id": "roto", "input": "bg", "source": "plate"})
        dispatcher.execute({"op": "set", "id": "roto", "param": "tracking_channels", "value": "red"})
        dispatcher.execute({"op": "set", "id": "roto", "param": "adjust_for_luminance_changes", "value": 1})
        dispatcher.execute({"op": "set", "id": "roto", "param": "pretrack_filter", "value": "none"})
        dispatcher.execute({"op": "set", "id": "roto", "param": "clamp_footage", "value": 0})
        dispatcher.document["time"].update(first=1, last=2, current=1)
        dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": [square(100, 100, 180, 180)]})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        window.viewer.roto_selected_shape_index = 0
        def compose(_executor, _doc, _source, region, frame, **_kwargs):
            x = np.arange(region.x, region.x + region.width, dtype=np.float32)[None, :] - 3*(frame-1)
            y = np.arange(region.y, region.y + region.height, dtype=np.float32)[:, None]
            signal = np.sin(x*.19) + np.cos(y*.17) + .7*np.sin((x+y)*.11) + .3*np.cos((2*x-y)*.07)
            gray = .5 + .15*signal
            if frame > 1:
                gray = gray*1.6 + .1
            rgba = np.stack(np.broadcast_arrays(gray, gray, gray, np.ones_like(gray)), axis=-1).astype(np.float32)
            return SimpleNamespace(pixels=rgba)
        with patch.object(TileExecutor, "compose_region", compose):
            self.assertTrue(window.track_roto_shape("roto"))
            self.assertTrue(wait_until(lambda: window._planar_future is None, timeout=10))
        resolved = resolve_document(window.dispatcher.document, 2)["nodes"]["roto"]["params"]
        self.assertAlmostEqual(resolved["pt1_x"], 103.0, delta=.25)
        self.assertAlmostEqual(resolved["pt2_x"], 183.0, delta=.25)
        self.assertAlmostEqual(resolved["pt1_y"], 100.0, delta=.25)
        reference = resolve_document(window.dispatcher.document, 1)["nodes"]["roto"]["params"]
        self.assertAlmostEqual(reference["pt1_x"], 100.0, delta=.01)
        self.assertAlmostEqual(reference["pt3_y"], 180.0, delta=.01)
