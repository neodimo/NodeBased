from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

from nodebased import scene3d
from nodebased.conditioning_verify import main, verify_conditioning
from nodebased.scene_state import write_scene_state


class ConditioningVerificationTests(unittest.TestCase):
    def _shot(self, root):
        vertices = np.array(((-1, -1, 0), (1, -1, 0), (0, 1, 0)), np.float32)
        geometry = scene3d.Geometry(vertices=vertices, triangles=np.array(((0, 1, 2),), np.int32),
                                    name="hero", color=(1, .2, .1, 1))
        camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 5)))
        id_pass = np.zeros((24, 24, 3), np.float32)
        id_pass[6:18, 6:18, 0] = 1
        beauty = np.zeros((24, 24, 4), np.float32)
        beauty[6:18, 6:18] = (1, .2, .1, 1)
        path = root / "shot.scene.json"
        write_scene_state(path, {1: {"scene": scene3d.Scene(geometries=(geometry,)),
                                      "camera": camera, "resolution": (24, 24),
                                      "passes": {"object_id": id_pass, "beauty": beauty}}},
                          resolution=(24, 24))
        camera_row = {"world_transform": camera.transform.matrix().tolist(),
                      "vertical_fov_degrees": camera.fov}
        return path, id_pass, beauty, camera_row

    def test_known_render_passes_camera_object_and_lighting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ids, beauty, camera = self._shot(root)
            # The ID must match the manifest's name hash; obtain it from the file to keep the
            # test independent of a hard-coded hash spelling.
            from nodebased.scene_state import read_scene_state
            object_id = read_scene_state(state)["frames"][0]["objects"][0]["id"]
            report = verify_conditioning(state, {1: {
                "camera": camera, "object_ids": {object_id: ids[..., 0] == 1},
                "beauty": beauty, "dominant_light_direction": [0, -1, 0],
                "next_beauty": beauty, "motion_reference": np.zeros((24, 24, 2), np.float32),
                "depth_landmarks": {"near": {"expected": 1, "observed": 1},
                                    "far": {"expected": 2, "observed": 2}},
                "reference_light_direction": [0, -1, 0],
                "shadow_direction": [1, 0, 0], "reference_shadow_direction": [1, 0, 0],
            }}, root / "verify")
            bindings = report["frames"][0]["bindings"]
            self.assertTrue(bindings["camera"]["pass"])
            self.assertTrue(bindings["objects"][0]["pass"])
            self.assertTrue(bindings["lighting"]["pass"])
            self.assertFalse(report["intensity_checked"])
            self.assertEqual(report["score_card"]["verdict"], "PASS")
            self.assertIn("Lighting intensity is not checked", (root / "verify.txt").read_text())

    def test_two_degree_camera_nudge_fails_only_camera_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, _ids, _beauty, camera = self._shot(root)
            angle = np.radians(2)
            camera["world_transform"] = (np.array(camera["world_transform"]) @ np.array((
                (np.cos(angle), 0, np.sin(angle), 0), (0, 1, 0, 0),
                (-np.sin(angle), 0, np.cos(angle), 0), (0, 0, 0, 1)))).tolist()
            report = verify_conditioning(state, {1: {"camera": camera}}, root / "nudged")
            measured = report["frames"][0]["bindings"]["camera"]
            self.assertAlmostEqual(measured["rotation_error_degrees"], 2.0, places=3)
            self.assertFalse(measured["pass"])

    def test_object_id_centroid_is_reprojected_and_three_pixel_shift_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ids, _beauty, camera = self._shot(root)
            from nodebased.scene_state import read_scene_state
            object_id = read_scene_state(state)["frames"][0]["objects"][0]["id"]
            shifted = np.roll(ids[..., 0] == 1, 3, axis=1)
            shifted_beauty = np.roll(_beauty, 3, axis=1)
            report = verify_conditioning(state, {1: {
                "camera": camera, "object_ids": {object_id: shifted}, "beauty": _beauty,
                "next_beauty": shifted_beauty,
                "motion_reference": np.zeros((24, 24, 2), np.float32),
                "object_lock_states": {object_id: "loosened"},
            }}, root / "object-shift")
            binding = report["frames"][0]["bindings"]["objects"][0]
            self.assertAlmostEqual(binding["error_pixels"], 3.0)
            self.assertFalse(binding["pass"])
            self.assertEqual(binding["lock_state"], "loosened")
            self.assertFalse(report["frames"][0]["bindings"]["motion"]["pass"])

    def test_depth_inverted_landmarks_fail_depth_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, _ids, beauty, _camera = self._shot(root)
            report = verify_conditioning(state, {1: {
                "beauty": beauty,
                "depth_landmarks": {"near": {"expected": 1, "observed": 3},
                                    "middle": {"expected": 2, "observed": 2},
                                    "far": {"expected": 3, "observed": 1}},
            }}, root / "depth-inverted")
            depth = report["frames"][0]["bindings"]["depth"]
            self.assertEqual(depth["rank_correlation"], -1.0)
            self.assertFalse(depth["pass"])

    def test_tracker_solves_camera_from_rendered_non_coplanar_landmarks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            centers = ((-2, -1, 0), (2, -1, .5), (-1, 2, -1), (1, 1, 1.5),
                       (0, -2, 2), (-2, 1, 1), (2, 2, -1.5), (0, 0, -.5))
            geometries = []
            for index, (x, y, z) in enumerate(centers):
                vertices = np.array(((x - .35, y - .25, z), (x + .35, y - .25, z),
                                     (x, y + .35, z)), np.float32)
                geometries.append(scene3d.Geometry(vertices=vertices,
                    triangles=np.array(((0, 1, 2),), np.int32),
                    color=(.2 + index * .08, .8 - index * .07, .3 + index * .05, 1),
                    name=f"mark{index}"))
            scene = scene3d.Scene(geometries=tuple(geometries))
            base = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 10)),
                                  fov=40)
            theta = np.radians(2)
            moved = scene3d.Camera(transform=base.transform,
                target=scene3d.Vec3(10 * np.sin(theta), 0, 10 - 10 * np.cos(theta)), fov=40)
            images = {1: scene3d.render(scene, base, 128, 128),
                      2: scene3d.render(scene, moved, 128, 128)}
            state_path = root / "tracked.scene.json"
            write_scene_state(state_path, {
                1: {"scene": scene, "camera": base, "resolution": (128, 128)},
                2: {"scene": scene, "camera": base, "resolution": (128, 128)},
            }, first_frame=1, last_frame=2, resolution=(128, 128))
            report = verify_conditioning(state_path,
                {frame: {"beauty": image} for frame, image in images.items()}, root / "tracked")
            frames = {row["frame"]: row["bindings"]["camera"] for row in report["frames"]}
            self.assertTrue(frames[1]["pass"], frames[1])
            self.assertFalse(frames[2]["pass"], frames[2])
            self.assertAlmostEqual(frames[2]["rotation_error_degrees"], 2.0, delta=.3)
            self.assertGreaterEqual(frames[2]["tracker_landmarks"], 6)
            self.assertTrue(any(item["measurement_source"] == "tracker"
                                for item in report["frames"][1]["bindings"]["objects"]))

    def test_twenty_degree_key_light_rotation_fails_lighting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, _ids, _beauty, camera = self._shot(root)
            angle = np.radians(20)
            report = verify_conditioning(state, {1: {
                "camera": camera, "dominant_light_direction": [np.sin(angle), -np.cos(angle), 0],
                "reference_light_direction": [0, -1, 0],
            }}, root / "light")
            binding = report["frames"][0]["bindings"]["lighting"]
            self.assertAlmostEqual(binding["light_direction_error_degrees"], 20.0, places=4)
            self.assertFalse(binding["pass"])

    def test_rendered_twenty_degree_key_rotation_is_measured_from_the_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            floor = scene3d.Geometry(
                vertices=np.array(((-3, -1, -3), (3, -1, 3), (3, -1, -3),
                                   (-3, -1, -3), (-3, -1, 3), (3, -1, 3)), np.float32),
                triangles=np.array(((0, 1, 2), (3, 4, 5)), np.int32),
                color=(.8, .8, .8, 1), name="floor")
            sphere = scene3d._sphere(1.4, 32, (.8, .8, .8, 1), scene3d.Transform3D(
                position=scene3d.Vec3(0, .2, 0)))
            light = scene3d.Light(kind="Directional", intensity=1.2,
                position=scene3d.Vec3(4, 6, 4), target=scene3d.Vec3(), shadows=True)
            scene = scene3d.Scene(geometries=(floor, sphere), lights=(light,))
            camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 2, 8)),
                                    target=scene3d.Vec3(0, -.2, 0), fov=45)
            angle = np.radians(20)
            vector = light.target.array() - light.position.array()
            rotation = np.array(((np.cos(angle), 0, np.sin(angle)), (0, 1, 0),
                                 (-np.sin(angle), 0, np.cos(angle))))
            altered = replace(scene, lights=(replace(light,
                target=scene3d.Vec3(*(light.position.array() + rotation @ vector))),))
            plate = scene3d.render(altered, camera, 96, 96, output="rgba", mode="raster")
            state_path = root / "light.scene.json"
            write_scene_state(state_path, {1: {"scene": scene, "camera": camera,
                "resolution": (96, 96)}}, resolution=(96, 96))
            report = verify_conditioning(state_path, {1: {"beauty": plate}}, root / "light-auto")
            binding = report["frames"][0]["bindings"]["lighting"]
            self.assertGreater(binding["light_direction_error_degrees"], 5)
            self.assertIsNotNone(binding.get("shadow_direction_error_degrees"))
            self.assertFalse(binding["pass"])
            self.assertFalse(report["intensity_checked"])

    def test_cli_reads_json_npz_and_writes_both_shot_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ids, beauty, camera = self._shot(root)
            observation_path = root / "observations.json"
            observation_path.write_text(json.dumps({"1": {
                "camera": camera, "beauty": {"array": "beauty"},
                "id_pass": {"array": "ids"}}}), encoding="utf-8")
            np.savez_compressed(root / "observations.npz", beauty=beauty, ids=ids)
            self.assertEqual(main([str(state), str(observation_path), str(root / "cli-report")]), 0)
            self.assertTrue((root / "cli-report.json").is_file())
            self.assertTrue((root / "cli-report.txt").is_file())


if __name__ == "__main__":
    unittest.main()
