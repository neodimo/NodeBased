import unittest
import numpy as np

from nodebased import scene3d, track3d


class Track3DMathTests(unittest.TestCase):
    def camera_at(self, x):
        return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(x, 0, 0)),
                              scene3d.Vec3(x, 0, -1), 50.0)

    def test_known_point_triangulates_across_a_camera_move(self):
        width, height = 640, 480
        point = np.array((0.35, -0.2, -4.0))
        cameras = [self.camera_at(-1.0), self.camera_at(0.0), self.camera_at(1.0)]
        pixels = [scene3d.project(camera, width, height, point[None, :])[0][0] for camera in cameras]
        recovered, residuals = track3d.triangulate(cameras, pixels, width, height)
        np.testing.assert_allclose(recovered, point, atol=1e-3)
        self.assertLess(max(residuals), 0.01)

    def test_noisy_track_recovers_within_one_centimeter(self):
        width, height = 640, 480
        point = np.array((0.35, -0.2, -4.0))
        cameras = [self.camera_at(-1.0), self.camera_at(0.0), self.camera_at(1.0)]
        pixels = [scene3d.project(camera, width, height, point[None, :])[0][0] for camera in cameras]
        noisy = [xy + np.array((0.5, -0.5)) for xy in pixels]
        recovered, _ = track3d.triangulate(cameras, noisy, width, height)
        self.assertLess(np.linalg.norm(recovered - point), 0.01)

    def test_parallel_rays_have_a_named_error(self):
        camera = self.camera_at(0.0)
        with self.assertRaisesRegex(ValueError, "too_parallel_rays"):
            track3d.triangulate([camera, camera], [(320, 240), (320, 240)], 640, 480)

    def test_projection_marks_behind_and_offscreen_without_clamping(self):
        camera = self.camera_at(0.0)
        front = track3d.reconcile(camera, (0.0, 0.0, -4.0), 640, 480)
        self.assertAlmostEqual(front["x"], 320.0, places=4)
        self.assertAlmostEqual(front["y"], 240.0, places=4)
        self.assertFalse(front["behind_camera"])
        behind = track3d.reconcile(camera, (0.0, 0.0, 2.0), 640, 480)
        self.assertTrue(behind["behind_camera"])
        offscreen = track3d.reconcile(camera, (100.0, 0.0, -4.0), 640, 480)
        self.assertTrue(offscreen["off_screen"])
        self.assertGreater(offscreen["x"], 640)

    def test_typed_nodes_declare_bypass_and_full_frame_execution(self):
        from nodebased.core import OUTPUT_TYPES, bypass_slot
        from nodebased.tiles import SUPPORTED_TILED_KINDS
        self.assertEqual(OUTPUT_TYPES["PointsTo3D"], "scene")
        self.assertEqual(OUTPUT_TYPES["Reconcile3D"], "track")
        self.assertEqual(bypass_slot({"type": "PointsTo3D", "inputs": {"object": "obj"}}), "object")
        self.assertEqual(bypass_slot({"type": "Reconcile3D", "inputs": {"point": "point"}}), "point")
        self.assertNotIn("PointsTo3D", SUPPORTED_TILED_KINDS)
        self.assertNotIn("Reconcile3D", SUPPORTED_TILED_KINDS)


if __name__ == "__main__":
    unittest.main()

class Track3DNodeTests(unittest.TestCase):
    def test_nodes_round_trip_camera_motion_track(self):
        from nodebased.core import Dispatcher
        from nodebased.imaging import Evaluator

        doc = Dispatcher().document
        dispatch = Dispatcher(doc)
        dispatch.execute({"op": "create", "id": "cam", "type": "Camera3D", "name": "Camera"})
        dispatch.execute({"op": "create", "id": "tri", "type": "PointsTo3D", "name": "PointsTo3D",
                          "params": {"image_width": 640, "image_height": 480,
                                     "frame0": 1, "frame1": 2, "frame2": 3}})
        dispatch.execute({"op": "create", "id": "rec", "type": "Reconcile3D", "name": "Reconcile3D",
                          "params": {"image_width": 640, "image_height": 480,
                                     "point_x": 0.35, "point_y": -0.2, "point_z": 0.0,
                                     "frame_start": 1, "frame_end": 4}})
        dispatch.execute({"op": "connect", "id": "tri", "input": "camera", "source": "cam"})
        dispatch.execute({"op": "connect", "id": "rec", "input": "camera", "source": "cam"})
        for f, x in ((1, -1.0), (2, 0.0), (3, 1.0)):
            dispatch.execute({"op": "set_key", "id": "cam", "param": "tx", "frame": f, "value": x})
            dispatch.execute({"op": "set_key", "id": "cam", "param": "target_x", "frame": f, "value": x})
        dispatch.execute({"op": "set_key", "id": "cam", "param": "tz", "frame": 3, "value": 5.0})
        dispatch.execute({"op": "set_key", "id": "cam", "param": "target_z", "frame": 3, "value": 0.0})
        dispatch.execute({"op": "set_key", "id": "cam", "param": "tz", "frame": 4, "value": -5.0})
        dispatch.execute({"op": "set_key", "id": "cam", "param": "target_z", "frame": 4, "value": -10.0})
        world = np.array((0.35, -0.2, 0.0))
        pixels = []
        for f, x in ((1, -1.0), (2, 0.0), (3, 1.0)):
            p = dict(dispatch.document["nodes"]["cam"]["params"], tx=x, target_x=x)
            pixels.append(scene3d.project(scene3d.camera_from_node({"params": p}), 640, 480, world[None, :])[0][0])
        for i, (x, y) in enumerate(pixels):
            dispatch.execute({"op": "set", "id": "tri", "param": f"track_x{i}", "value": float(x)})
            dispatch.execute({"op": "set", "id": "tri", "param": f"track_y{i}", "value": float(y)})
        dispatch.execute({"op": "connect", "id": "rec", "input": "point", "source": "tri"})
        evaluator = Evaluator()
        evaluator.evaluate_raster(dispatch.document, "tri", frame=1, typed=True)
        np.testing.assert_allclose(evaluator.track3d_diagnostics["tri"]["position"], world, atol=1e-3)
        self.assertLess(max(evaluator.track3d_diagnostics["tri"]["residuals"]), 0.01)
        result = evaluator.evaluate_raster(dispatch.document, "rec", frame=1, typed=True)
        self.assertEqual(result["kind"], "TrackerTrack")
        self.assertEqual(len(result["track"]["x"]["curve"]["keys"]), 4)
        from nodebased.shapes import validate_payload
        validate_payload("Tracker", {"tracks": [result["track"]]}, "track result")
        for actual, expected in zip(result["points"][:3], pixels):
            np.testing.assert_allclose((actual["x"], actual["y"]), expected, atol=1e-3)
        self.assertTrue(all(not p["behind_camera"] for p in result["points"][:3]))
        self.assertTrue(result["points"][3]["behind_camera"])
