import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import scene3d
from nodebased.conditioning_verify import verify_conditioning
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
                "reference_light_direction": [0, -1, 0],
                "shadow_direction": [1, 0, 0], "reference_shadow_direction": [1, 0, 0],
            }}, root / "verify")
            bindings = report["frames"][0]["bindings"]
            self.assertTrue(bindings["camera"]["pass"])
            self.assertTrue(bindings["objects"][0]["pass"])
            self.assertTrue(bindings["lighting"]["pass"])
            self.assertFalse(report["intensity_checked"])
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
            self.assertAlmostEqual(measured["rotation_error_degrees"], 2.0, places=4)
            self.assertFalse(measured["pass"])

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


if __name__ == "__main__":
    unittest.main()
