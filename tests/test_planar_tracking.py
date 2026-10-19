import unittest

import numpy as np

from nodebased.planar import fit_homography_ransac, project_points


class PlanarFitTests(unittest.TestCase):
    def test_projective_fit_recovers_card_corners_and_is_deterministic(self):
        source = np.array([[12, 15], [90, 12], [95, 78], [10, 82],
                           [30, 25], [65, 27], [70, 60], [25, 63]], dtype=float)
        known = np.array([[1.08, .04, 13], [-.025, .96, 7], [.0007, -.0004, 1.]])
        destination = project_points(known, source)
        first = fit_homography_ransac(source, destination, threshold=.01)
        second = fit_homography_ransac(source, destination, threshold=.01)
        np.testing.assert_allclose(project_points(first[0], source[:4]), destination[:4], atol=1e-7)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_allclose(first[0], second[0], atol=0)
        self.assertAlmostEqual(first[2], 0.0, places=8)

    def test_quarter_occlusion_outliers_do_not_pull_the_plane(self):
        rng = np.random.default_rng(41)
        source = rng.uniform([0, 0], [100, 80], (32, 2))
        known = np.array([[.93, -.08, 11], [.06, 1.04, -4], [-.0003, .0006, 1.]])
        destination = project_points(known, source)
        destination[:8] = rng.uniform(0, 100, (8, 2))
        matrix, inliers, rms = fit_homography_ransac(source, destination, threshold=.05)
        self.assertEqual(int(inliers.sum()), 24)
        self.assertLess(rms, 1e-7)
        corners = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=float)
        np.testing.assert_allclose(project_points(matrix, corners), project_points(known, corners), atol=1e-7)


if __name__ == "__main__":
    unittest.main()

class PlanarSequenceTests(unittest.TestCase):
    def test_analysis_cancellation_and_progress_are_observed_between_frames(self):
        import threading
        from concurrent.futures import CancelledError
        from nodebased.planar import track_region
        plate = np.random.default_rng(4).random((48, 48, 4), dtype=np.float32)
        plate[..., 3] = 1
        cancel = threading.Event()
        cancel.set()
        progress = []
        with self.assertRaises(CancelledError):
            track_region({1: plate, 2: plate.copy()},
                         np.array([[8, 8], [40, 8], [40, 40], [8, 40]], float), 1,
                         samples_per_axis=3, vector_detail=1, backend="cpu",
                         cancel=cancel, progress=lambda *args: progress.append(args))
        self.assertEqual(progress, [])

    def test_lucas_kanade_region_tracks_24_homography_frames(self):
        from nodebased.flow_nodes import warp_by_homography
        from nodebased.planar import track_region
        rng = np.random.default_rng(21)
        plate = rng.random((96, 112, 4), dtype=np.float32)
        plate[..., 3] = 1
        quad = np.array([[20, 18], [88, 16], [90, 76], [18, 78]], dtype=float)
        frames = {}
        expected = {}
        for frame in range(1, 25):
            dx, dy = .35*(frame-12), .18*(frame-12)
            matrix = np.array([[1, .001*(frame-12), dx], [-.001*(frame-12), 1, dy], [0, 0, 1]], dtype=float)
            frames[frame] = warp_by_homography(plate, matrix)
            expected[frame] = project_points(matrix, quad)
        result = track_region(frames, quad, 12, samples_per_axis=12, vector_detail=3,
                              backend="cpu", threshold=3.0)
        repeated = track_region(frames, quad, 12, samples_per_axis=12, vector_detail=3,
                                backend="cpu", threshold=3.0)
        for frame in range(1, 25):
            np.testing.assert_allclose(result[frame]["corners"], expected[frame], atol=1.0)
            np.testing.assert_array_equal(result[frame]["corners"], repeated[frame]["corners"])


class PlanarNodeTests(unittest.TestCase):
    def test_planar_worker_keeps_ui_responsive_and_cancellation_click_lands(self):
        import threading
        import time
        from unittest import mock
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QSettings, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QPushButton
        from tests.test_roto_ui import APP
        from tests.waiting import wait_until
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Constant",
                            "params": {"width": 64, "height": 64}})
        dispatcher.execute({"op": "create", "id": "plane", "type": "PlanarTracker"})
        dispatcher.execute({"op": "connect", "id": "plane", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "view", "id": "plane"})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document),
                                 window.close(), window.executor.shutdown(wait=True, cancel_futures=True),
                                 APP.processEvents()))
        window.show()
        window.inspect("plane")
        window.set_planar_region("plane", [(12, 12), (52, 12), (52, 52), (12, 52)])
        entered, release = threading.Event(), threading.Event()
        original = __import__("nodebased.tileexec", fromlist=["TileExecutor"]).TileExecutor.compose_region

        def slow_compose(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original(*args, **kwargs)

        with mock.patch("nodebased.app.TileExecutor.compose_region", side_effect=slow_compose):
            self.assertTrue(window.analyse_planar("plane", 1, 2))
            self.assertTrue(entered.wait(1), "analysis worker did not start")
            self.assertTrue(wait_until(lambda: "PlanarTracker analysis:" in window.statusBar().currentMessage(),
                                       timeout=1), "analysis progress was not visible")
            cancel_button = window.properties.widget().findChild(QPushButton, "planar-cancel-analysis")
            self.assertIsNotNone(cancel_button)
            started = time.perf_counter()
            QTest.mouseClick(cancel_button, Qt.MouseButton.LeftButton)
            click_ms = (time.perf_counter() - started) * 1000
            self.assertLess(click_ms, 100, f"cancel click took {click_ms:.1f} ms")
            self.assertTrue(window._planar_cancel.is_set())
            release.set()
            self.assertTrue(wait_until(lambda: window._planar_future is None, timeout=3))

    def test_node_is_registered_and_passes_image_identically_on_evaluator_and_tiles(self):
        from nodebased.core import Dispatcher, SPECS, empty_document
        from nodebased.imaging import Evaluator
        from nodebased.tileexec import TileExecutor
        self.assertIn("PlanarTracker", SPECS)
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Constant",
                           "params": {"width": 24, "height": 20, "red": .2, "green": .3,
                                      "blue": .4, "alpha": 1.0}})
        dispatcher.execute({"op": "create", "id": "plane", "type": "PlanarTracker"})
        dispatcher.execute({"op": "connect", "id": "plane", "input": "image", "source": "plate"})
        doc = dispatcher.document
        expected = Evaluator().evaluate_raster(doc, "plate", frame=1).pixels
        actual = Evaluator().evaluate_raster(doc, "plane", frame=1).pixels
        np.testing.assert_array_equal(actual, expected)
        executor = TileExecutor(evaluator=Evaluator())
        region = executor.canvas_region(doc, "plane", frame=1, tier=1)
        tiled = executor.compose_region(doc, "plane", region, frame=1, tier=1).pixels
        np.testing.assert_array_equal(tiled, expected)

    def test_planar_corner_tracks_use_the_v8_scalar_envelope(self):
        from nodebased.planar import resolve_corners, tracks_from_results
        results = {1: {"corners": [[1, 2], [8, 2], [8, 7], [1, 7]], "error": 0.0},
                   2: {"corners": [[2, 3], [9, 3], [9, 8], [2, 8]], "error": .1}}
        payload = {"tracks": tracks_from_results(results)}
        self.assertEqual(payload["tracks"][0]["x"]["curve"]["keys"][1], {"frame": 2, "value": 2.0})
        np.testing.assert_allclose(resolve_corners(payload, 2), results[2]["corners"])

    def test_roto_planar_follow_maps_shape_by_the_homography(self):
        from tests.test_roto import square
        from nodebased.planar import tracks_from_results
        from nodebased.shapes import resolve_roto_shapes
        reference = np.array([[10, 10], [90, 10], [90, 70], [10, 70]], dtype=float)
        matrix = np.array([[1.02, .04, 7], [-.03, .97, 4], [.0005, -.0002, 1.]])
        moved = project_points(matrix, reference)
        payload = {"tracks": tracks_from_results({1: {"corners": reference.tolist(), "error": 0},
                                                    2: {"corners": moved.tolist(), "error": 0}})}
        shape = square(30, 25, 55, 50, track_link={"tracker_id": "plane", "track_name": "corner1"},
                       track_follow="planar")
        document = {"nodes": {"plane": {"type": "PlanarTracker", "params": {"reference_frame": 1}}},
                    "node_data": {"plane": payload}}
        resolved = resolve_roto_shapes({"shapes": [shape]}, 2, document)[0]
        expected = project_points(matrix, np.array([[30, 25], [55, 25], [55, 50], [30, 50]]))
        np.testing.assert_allclose([[p["x"], p["y"]] for p in resolved["points"]], expected, atol=1e-6)


    def test_quarter_area_occlusion_motion_fixture(self):
        from nodebased.flow_nodes import warp_by_homography
        from nodebased.planar import track_region
        rng = np.random.default_rng(23)
        plate = rng.random((96, 112, 4), dtype=np.float32); plate[..., 3] = 1
        quad = np.array([[20, 18], [88, 16], [90, 76], [18, 78]], dtype=float)
        a = warp_by_homography(plate, np.eye(3))
        h = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1.]])
        b = warp_by_homography(plate, h); b[16:46, 55:91, :] = 0
        result = track_region({1:a, 2:b}, quad, 1, samples_per_axis=24, vector_detail=1, backend="cpu", threshold=0.5)
        np.testing.assert_allclose(result[2]["corners"], project_points(h, quad), atol=1.0)
        self.assertGreaterEqual(result[2]["features"], 4)

    def test_viewer_picks_quad_and_exports_cornerpin(self):
        import tests.isolation  # noqa: F401
        from PySide6.QtCore import QPointF, QSettings, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QPushButton
        from tests.test_roto_ui import APP
        from tests.waiting import wait_until
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "create", "id": "plane", "type": "PlanarTracker"})
        dispatcher.execute({"op": "connect", "id": "plane", "input": "image", "source": "plate"})
        dispatcher.execute({"op": "view", "id": "plane"})
        window = Window(dispatcher.document)
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 window.executor.shutdown(wait=True, cancel_futures=True), APP.processEvents()))
        window.show(); window.graph.items_by_id["plane"].setSelected(True); window.inspect("plane")
        self.assertTrue(wait_until(lambda: window.properties.widget().findChild(
            QPushButton, "planar-pick-region") is not None, timeout=2))
        self.assertTrue(window.begin_planar_pick("plane"))
        # A click lands on a whole pixel, so the picked corner is the scene point under that
        # pixel; at any zoom other than 1 it is a fraction off the requested one.
        picked = []
        for point in ((30, 30), (90, 30), (90, 80), (30, 80)):
            pixel = window.viewer.mapFromScene(QPointF(*point))
            picked.append(window.viewer._tracker_data_point(window.viewer.mapToScene(pixel)))
            QTest.mouseClick(window.viewer.viewport(), Qt.MouseButton.LeftButton, pos=pixel)
        corners = window.dispatcher.document["node_data"]["plane"]["tracks"]
        self.assertEqual(len(corners), 4)
        self.assertEqual(corners[0]["name"], "corner1")
        window.dispatcher.document["time"]["last"] = 2
        keyed = []
        for track in corners:
            moved = dict(track)
            for axis in ("x", "y"):
                value = float(track[axis])
                delta = 7.0 if axis == "x" else -3.0
                moved[axis] = {"value": value, "curve": {"interpolation": "linear", "keys": [
                    {"frame": 1, "value": value}, {"frame": 2, "value": value + delta}]}}
            keyed.append(moved)
        window.command({"op": "set_tracks", "id": "plane", "tracks": keyed})
        self.assertTrue(window.export_tracker_cornerpin("plane"))
        pins = [(ident, n) for ident, n in window.dispatcher.document["nodes"].items()
                if n["type"] == "CornerPin"]
        self.assertEqual(len(pins), 1)
        pin_id, pin = pins[0]
        np.testing.assert_allclose([[pin["params"][f"to{i}_x"], pin["params"][f"to{i}_y"]]
                                    for i in range(1, 5)], picked)
        for i, (x, y) in enumerate(picked, start=1):
            curves = window.dispatcher.document["animation"]["curves"][pin_id]
            px = curves[f"to{i}_x"]["keys"]
            py = curves[f"to{i}_y"]["keys"]
            np.testing.assert_allclose((px[1]["value"], py[1]["value"]), (x + 7.0, y - 3.0))
