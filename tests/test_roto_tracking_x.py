import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class RotoPaintTrackingTests(unittest.TestCase):
    def test_tracking_channels_filter_and_footage_clamp_change_analysis_pixels(self):
        from nodebased.tracker import _analysis_pixels
        source = np.array([[[-2.0, 2.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        red, _ = _analysis_pixels(source, "red", "none", False)
        clamped, _ = _analysis_pixels(source, "red", "none", True)
        filtered, _ = _analysis_pixels(source, "red", "adjust contrast", False)
        self.assertEqual(float(red[0, 0]), -2.0)
        self.assertEqual(float(clamped[0, 0]), 0.0)
        self.assertGreater(float(filtered.max()), float(filtered.min()))

    def test_tracking_tab_defaults_and_older_document_migration(self):
        from nodebased.core import SPECS, empty_document, upgrade_document
        defaults = SPECS["RotoPaint"]["params"]
        self.assertEqual(defaults["pretrack_filter"], "none")
        self.assertEqual(defaults["tracking_channels"], "rgb")
        self.assertEqual(defaults["export_menu"], "CornerPin2D (relative)")
        old = empty_document()
        old["nodes"]["paint"] = {"id": "paint", "type": "RotoPaint", "name": "RotoPaint",
                                  "pos": [0, 0], "inputs": {"image": None, "input2": None, "mask": None},
                                  "params": {"mix": 1.0}, "disabled": False}
        upgraded = upgrade_document(copy.deepcopy(old))
        self.assertEqual(upgraded["nodes"]["paint"]["params"]["tracking_channels"], "rgb")
        self.assertEqual(upgraded["nodes"]["paint"]["params"]["link_output"], 1)

    def test_selected_stroke_tracks_bounding_quad_and_exports(self):
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QSettings
        from tests.test_roto_ui import APP
        from tests.test_roto_paint import stroke
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        from nodebased.animation import resolve_document

        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Constant"})
        dispatcher.execute({"op": "create", "id": "paint", "type": "RotoPaint"})
        dispatcher.execute({"op": "connect", "id": "paint", "input": "image", "source": "plate"})
        item = stroke(points=[{"x": x, "y": y, "pressure": 1.0}
                              for x, y in ((18, 21), (45, 21), (45, 53), (18, 53))])
        dispatcher.execute({"op": "set_paint_items", "id": "paint", "items": [item]})
        dispatcher.document["time"].update(first=1, last=2, current=1)
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        window.viewer.paint_selected_item_index = 0
        captured = {}
        window.analyse_planar = lambda key, **kwargs: captured.update(key=key, **kwargs) or True
        self.assertTrue(window.track_roto_shape("paint"))
        self.assertEqual(captured["key"], "paint")
        np.testing.assert_array_equal(captured["corners_override"],
                                      [[18, 21], [45, 21], [45, 53], [18, 53]])

        for index, (x, y) in enumerate(((18, 21), (45, 21), (45, 53), (18, 53)), 1):
            for axis, value in (("x", x), ("y", y)):
                window.command({"op": "set_curve", "id": "paint", "param": f"pt{index}_{axis}",
                                "curve": {"interpolation": "linear", "keys": [
                                    {"frame": 1, "value": value}, {"frame": 2, "value": value + (3 if axis == "x" else 0)}]}})
        for export_kind in ("CornerPin2D (relative)", "CornerPin2D (absolute)", "CornerPin2D (stabilize)", "Tracker"):
            window.command({"op": "set", "id": "paint", "param": "export_menu", "value": export_kind})
            self.assertTrue(window.export_roto_tracking("paint"))
            created = [(node_id, node) for node_id, node in window.dispatcher.document["nodes"].items()
                       if node["type"] in ("CornerPin", "Tracker")]
            self.assertEqual(created[-1][1]["type"], "Tracker" if export_kind == "Tracker" else "CornerPin")
            if export_kind.startswith("CornerPin"):
                pin_id, _pin = created[-1]
                value = resolve_document(window.dispatcher.document, 2)["nodes"][pin_id]["params"]["to1_x"]
                self.assertEqual(value, 18.0 if "stabilize" in export_kind else 21.0)
        window.command({"op": "set", "id": "paint", "param": "link_output", "value": 0})
        window.command({"op": "set", "id": "paint", "param": "export_menu", "value": "CornerPin2D (relative)"})
        self.assertTrue(window.export_roto_tracking("paint"))
        pin_id = [node_id for node_id, node in window.dispatcher.document["nodes"].items()
                  if node["type"] == "CornerPin"][-1]
        self.assertIn("to1_x", window.dispatcher.document["animation"]["curves"][pin_id])

    def test_selected_stroke_tracks_a_three_pixel_plate_translation(self):
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QSettings
        from tests.test_roto_ui import APP
        from tests.test_roto_paint import stroke
        from tests.waiting import wait_until
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        from nodebased.tileexec import TileExecutor
        from nodebased.animation import resolve_document

        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Constant",
                            "params": {"width": 256, "height": 256, "red": 0.5, "green": 0.5, "blue": 0.5}})
        dispatcher.execute({"op": "create", "id": "paint", "type": "RotoPaint"})
        dispatcher.execute({"op": "connect", "id": "paint", "input": "image", "source": "plate"})
        item = stroke(points=[{"x": x, "y": y, "pressure": 1.0}
                              for x, y in ((100, 100), (180, 100), (180, 180), (100, 180))])
        dispatcher.execute({"op": "set_paint_items", "id": "paint", "items": [item]})
        dispatcher.document["time"].update(first=1, last=2, current=1)
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        window.viewer.paint_selected_item_index = 0

        def compose(_executor, _doc, _source, region, frame, **_kwargs):
            x = np.arange(region.x, region.x + region.width, dtype=np.float32)[None, :] - 3*(frame-1)
            y = np.arange(region.y, region.y + region.height, dtype=np.float32)[:, None]
            signal = np.sin(x*.19) + np.cos(y*.17) + .7*np.sin((x+y)*.11) + .3*np.cos((2*x-y)*.07)
            gray = .5 + .15*signal
            rgba = np.stack(np.broadcast_arrays(gray, gray, gray, np.ones_like(gray)), axis=-1).astype(np.float32)
            return SimpleNamespace(pixels=rgba)

        with patch.object(TileExecutor, "compose_region", compose):
            self.assertTrue(window.track_roto_shape("paint"))
            self.assertTrue(wait_until(lambda: window._planar_future is None, timeout=10))
        moved = resolve_document(window.dispatcher.document, 2)["nodes"]["paint"]["params"]
        self.assertAlmostEqual(moved["pt1_x"], 103.0, delta=.25)
        self.assertAlmostEqual(moved["pt2_x"], 183.0, delta=.25)
        self.assertAlmostEqual(moved["pt1_y"], 100.0, delta=.25)


if __name__ == "__main__":
    unittest.main()
