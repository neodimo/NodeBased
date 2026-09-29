"""Acceptance checks for Tracker completion and the standalone Stabilize node."""
import os
import unittest
from threading import Event
from concurrent.futures import CancelledError, Future

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, empty_document, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.tracker import analyse, solve

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def textured_square():
    rng = np.random.default_rng(3)
    plate = np.zeros((220, 240, 4), np.float32)
    plate[..., 3] = 1
    texture = rng.random((32, 32), dtype=np.float32)
    plate[72:104, 72:104, :3] = texture[..., None]
    return plate


def fractional_translate(image, dx, dy=0.0):
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    sx, sy = xx-dx, yy-dy
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx-x0, sy-y0
    out = np.zeros_like(image)
    out[..., 3] = 1
    for oy, wy in ((0, 1-fy), (1, fy)):
        for ox, wx in ((0, 1-fx), (1, fx)):
            x, y = x0+ox, y0+oy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            out[..., :3][valid] += image[y[valid], x[valid], :3] * (wy*wx)[valid, None]
    return out


class TrackerAcceptanceTests(unittest.TestCase):
    def test_subpixel_square_recovers_three_point_two_five_pixels_forward_and_backward(self):
        base = textured_square()
        frames = {frame: fractional_translate(base, 3.25*(frame-1)) for frame in range(1, 31)}
        for direction, reference, point in (
                ("forward", 1, (88.5, 88.5)),
                ("backward", 30, (88.5+3.25*29, 88.5))):
            path = analyse(frames, reference, point, 8, 8, first_frame=1, last_frame=30,
                           direction=direction)
            for frame, (x, y) in path.items():
                with self.subTest(direction=direction, frame=frame):
                    self.assertAlmostEqual(x, 88.5+3.25*(frame-1), delta=0.1)
                    self.assertAlmostEqual(y, 88.5, delta=0.1)

    def test_cancellation_returns_no_partial_track_data(self):
        base = textured_square()
        frames = {frame: fractional_translate(base, 2.0*(frame-1)) for frame in range(1, 8)}
        cancel = Event()

        def progress(completed, total, frame, point):
            if completed == 2:
                cancel.set()

        with self.assertRaises(CancelledError):
            analyse(frames, 1, (88.5, 88.5), 8, 8, first_frame=1, last_frame=7,
                    cancel=cancel, progress=progress)

    def test_two_tracks_recover_rotation(self):
        angle = np.deg2rad(23.0)
        source = [(20.0, 30.0), (80.0, 50.0)]
        center = (50.0, 40.0)
        tracks = []
        for index, (x, y) in enumerate(source):
            ux, uy = x-center[0], y-center[1]
            tracks.append({"name": f"p{index}", "enabled": 1,
                           "x": {"value": x, "curve": {"interpolation": "constant", "keys": [
                               {"frame": 1, "value": x},
                               {"frame": 2, "value": center[0]+ux*np.cos(angle)-uy*np.sin(angle)}]}},
                           "y": {"value": y, "curve": {"interpolation": "constant", "keys": [
                               {"frame": 1, "value": y},
                               {"frame": 2, "value": center[1]+ux*np.sin(angle)+uy*np.cos(angle)}]}}})
        result = solve({"tracks": tracks}, 2, {"reference_frame": 1})
        self.assertAlmostEqual(result["rotate"], 23.0, delta=1e-6)

    def test_stabilize_node_keeps_integer_shaking_plate_still(self):
        dispatcher = Dispatcher(empty_document())
        for key, kind in (("plate", "Checker"), ("shake", "Transform"), ("stab", "Stabilize")):
            dispatcher.execute({"op": "create", "id": key, "type": kind,
                                "params": {"width": 64, "height": 64} if kind == "Checker" else {}})
        dispatcher.execute({"op": "connect", "id": "shake", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "connect", "id": "stab", "input": "image", "source": "shake"})
        dispatcher.execute({"op": "time", "first": 1, "last": 3, "current": 1})
        for frame, offset in ((1, 0), (2, 3), (3, -2)):
            dispatcher.execute({"op": "set_key", "id": "shake", "param": "translate_x",
                                "frame": frame, "value": offset})
        dispatcher.execute({"op": "set_tracks", "id": "stab", "tracks": [{
            "name": "anchor", "enabled": 1, "x": {"value": 32.5, "curve": {
                "interpolation": "constant", "keys": [
                    {"frame": 1, "value": 32.5}, {"frame": 2, "value": 35.5},
                    {"frame": 3, "value": 30.5}]}}, "y": 32.5}]})
        evaluator = Evaluator()
        frames = [evaluator.evaluate_raster(dispatcher.document, target="stab", frame=f).fit(
            evaluator.evaluate_raster(dispatcher.document, target="stab", frame=1).display)
                  for f in (1, 2, 3)]
        for pixels in frames[1:]:
            self.assertLess(float(np.mean(np.abs(pixels-frames[0]))), 0.01)

    def test_v15_tracker_payload_migrates_without_pixel_change(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker",
                            "params": {"width": 64, "height": 64}})
        dispatcher.execute({"op": "create", "id": "tracker", "type": "Tracker"})
        dispatcher.execute({"op": "connect", "id": "tracker", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "set_tracks", "id": "tracker", "tracks": [
            {"name": "anchor", "enabled": 1, "x": 32.5, "y": 32.5}]})
        old = dispatcher.document
        for name in ("smoothing", "pattern_radius", "search_radius", "adaptive_update", "tracking_channels"):
            old["nodes"]["tracker"]["params"].pop(name, None)
        old["node_data"]["tracker"]["tracks"][0].pop("error", None)
        old["version"] = SCHEMA_VERSION-1
        before = Evaluator().evaluate_raster(old, target="tracker", frame=1).pixels.copy()
        upgraded = upgrade_document(old)
        after = Evaluator().evaluate_raster(upgraded, target="tracker", frame=1).pixels
        np.testing.assert_array_equal(before, after)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "offscreen UI test")
class TrackerUiAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from nodebased.app import STYLE
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(STYLE)

    def setUp(self):
        from nodebased.app import Window
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker",
                            "params": {"width": 64, "height": 64}})
        dispatcher.execute({"op": "create", "id": "tracker", "type": "Tracker"})
        dispatcher.execute({"op": "connect", "id": "tracker", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "time", "first": 1, "last": 2, "current": 1})
        tracks = []
        for index, (x, y) in enumerate(((10., 10.), (54., 10.), (54., 54.), (10., 54.))):
            tracks.append({"name": f"p{index}", "enabled": 1,
                           "x": {"value": x, "curve": {"interpolation": "constant", "keys": [
                               {"frame": 1, "value": x}, {"frame": 2, "value": x+3}]}},
                           "y": {"value": y, "curve": {"interpolation": "constant", "keys": [
                               {"frame": 1, "value": y}, {"frame": 2, "value": y+2}]}}})
        dispatcher.execute({"op": "set_tracks", "id": "tracker", "tracks": tracks})
        self.window = Window(dispatcher.document)

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.executor.shutdown(wait=True, cancel_futures=True)
        self.app.processEvents()

    def test_cornerpin_and_transform_exports_carry_track_animation(self):
        self.assertTrue(self.window.export_tracker_cornerpin("tracker"))
        self.assertTrue(self.window.export_tracker_transform("tracker"))
        nodes = self.window.dispatcher.document["nodes"]
        pin_id, pin = next((key, node) for key, node in nodes.items() if node["type"] == "CornerPin")
        transform_id, transform = next((key, node) for key, node in nodes.items()
                                        if node["type"] == "Transform")
        self.assertEqual(pin["inputs"]["image"], "plate")
        self.assertEqual(transform["inputs"]["image"], "plate")
        curves = self.window.dispatcher.document["animation"]["curves"]
        self.assertAlmostEqual(curves[pin_id]["from1_x"]["keys"][0]["value"], 10.0)
        self.assertAlmostEqual(curves[pin_id]["to1_x"]["keys"][1]["value"], 13.0)
        self.assertAlmostEqual(curves[transform_id]["translate_x"]["keys"][1]["value"], 3.0)
        self.window.command({"op": "create", "id": "expected", "type": "Transform",
                             "params": {"translate_x": 3.0, "translate_y": 2.0}})
        self.window.command({"op": "connect", "id": "expected", "input": "image", "source": "plate"})
        evaluator = Evaluator()
        pin_image = evaluator.evaluate_raster(self.window.dispatcher.document, target=pin_id, frame=2)
        expected = evaluator.evaluate_raster(self.window.dispatcher.document, target="expected", frame=2)
        np.testing.assert_allclose(pin_image.fit(pin_image.display),
                                   expected.fit(pin_image.display), atol=1e-6)

    def test_viewer_drag_keys_track_and_resize_boxes(self):
        from PySide6.QtCore import QPointF
        self.window.command({"op": "view", "id": "tracker"})
        self.window.graph.items_by_id["tracker"].setSelected(True)
        viewer = self.window.viewer
        viewer.tracker_drag = {"kind": "point", "index": 0, "scene": QPointF(15.5, 16.5), "moved": True}
        self.assertTrue(viewer._commit_tracker_drag())
        point = self.window.dispatcher.document["node_data"]["tracker"]["tracks"][0]
        self.assertAlmostEqual(point["x"]["curve"]["keys"][0]["value"], 15.5)
        viewer.tracker_drag = {"kind": "pattern", "center": (15.5, 16.5),
                               "scene": QPointF(25.5, 26.5), "moved": True}
        self.assertTrue(viewer._commit_tracker_drag())
        self.assertEqual(self.window.dispatcher.document["nodes"]["tracker"]["params"]["pattern_radius"], 10)

    def test_cancelled_ui_analysis_preserves_existing_tracks(self):
        track = {"name": "existing", "enabled": 1, "x": 12.5, "y": 18.5}
        self.window.command({"op": "set_tracks", "id": "tracker", "tracks": [track]})
        before = self.window.dispatcher.document["node_data"]["tracker"]["tracks"]
        future = Future()
        future.set_exception(CancelledError())
        self.window._tracker_future = future
        self.window._tracker_cancel = Event()
        self.window._tracker_cancel.set()
        self.window._tracker_job = {"key": "tracker", "index": 1, "base_tracks": before,
                                   "seed": {"name": "new", "enabled": 1, "x": 20.5, "y": 20.5}}
        self.window._poll_tracker_analysis()
        self.assertEqual(self.window.dispatcher.document["node_data"]["tracker"]["tracks"], before)


if __name__ == "__main__":
    unittest.main()
