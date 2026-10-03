import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased import scene3d
from nodebased.conditioned_read import read_conditioned_sequence
from nodebased.conditioning_verify import verify_conditioning
from nodebased.control_bundle import write_control_bundle
from nodebased.generative import ProviderDescription, generate
from nodebased.scene_state import read_scene_state, write_scene_state
from tests.test_conditioning_verify import ConditioningVerificationTests


class GenerativeProviderTests(unittest.TestCase):
    def _bundle(self, root):
        camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 5)))
        state = root / "shot.scene.json"
        write_scene_state(state, {1: {"scene": scene3d.Scene(), "camera": camera, "resolution": (8, 8)},
                                  2: {"scene": scene3d.Scene(), "camera": camera, "resolution": (8, 8)}},
                          resolution=(8, 8))
        plate = np.zeros((8, 8, 4), np.float32)
        plate[..., 0] = np.arange(8, dtype=np.float32)[None, :] + 1
        plate[..., 3] = 1
        samples = {}
        for frame in (1, 2):
            samples[frame] = {"beauty": plate.copy(), "depth": np.ones((8, 8), np.float32),
                              "normals": np.zeros((8, 8, 3), np.float32),
                              "motion_forward": np.zeros((8, 8, 2), np.float32),
                              "motion_backward": np.zeros((8, 8, 2), np.float32),
                              "object_ids": np.zeros((8, 8), np.float32)}
        samples[1]["motion_forward"][..., 0] = 1
        manifest = write_control_bundle(state, root / "controls", samples)
        return state, manifest, plate

    def _read(self, path):
        image = oiio.ImageInput.open(str(path))
        self.assertIsNotNone(image)
        try:
            spec = image.spec()
            return np.asarray(image.read_image(oiio.FLOAT), dtype=np.float32).reshape(spec.height, spec.width, 4)
        finally:
            image.close()

    def test_standins_and_conditioned_read_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, plate = self._bundle(root)
            reproject = generate(bundle, state, root / "reproject.####.exr", "reproject")
            null = generate(bundle, state, root / "null.####.exr", "null")
            self.assertEqual(reproject["honoured"], ["depth", "motion", "camera_pose"])
            self.assertEqual(null["honoured"], [])
            projected = self._read(root / "reproject.0001.exr")
            self.assertEqual(float(projected[3, 4, 0]), 4.0)
            self.assertEqual(float(projected[3, 0, 0]), 0.0)
            np.testing.assert_array_equal(self._read(root / "null.0001.exr"), plate)
            conditioned = read_conditioned_sequence(str(root / "reproject.####.exr"), bundle, state)
            self.assertEqual(len(conditioned), 2)
            self.assertEqual(conditioned[0].layers["motion_forward"].shape, (8, 8, 2))

    def test_descriptor_missing_motion_marks_capability_and_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, _plate = self._bundle(root)
            desc = json.loads((Path(__file__).parents[1] / "nodebased/providers/reproject.json").read_text())
            desc["accepts"]["motion"]["accepted"] = False
            desc["returns"]["honoured"] = ["depth", "camera_pose"]
            path = root / "depth-only.json"
            path.write_text(json.dumps(desc))
            loaded = ProviderDescription.load(path)
            self.assertIn("Motion: ignored", loaded.panel_lines())
            report = verify_conditioning(state, {1: {"motion_conditioned": False},
                                                 2: {"motion_conditioned": False}}, root / "score")
            motion = report["score_card"]["bindings"]["motion"]
            self.assertEqual(motion["status"], "not conditioned")
            self.assertIsNone(motion["pass"])

    def test_generated_sequence_round_trips_and_scores_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, ids, beauty, camera = ConditioningVerificationTests()._shot(root)
            samples = {1: {"beauty": beauty, "depth": np.where(ids[..., 0] > 0, 5, 0).astype(np.float32),
                           "normals": np.zeros((24, 24, 3), np.float32),
                           "motion_forward": np.zeros((24, 24, 2), np.float32),
                           "motion_backward": np.zeros((24, 24, 2), np.float32),
                           "object_ids": np.zeros((24, 24), np.float32)}}
            bundle = write_control_bundle(state, root / "controls", samples)
            generate(bundle, state, root / "reproject.####.exr", "reproject")
            conditioned = read_conditioned_sequence(str(root / "reproject.####.exr"), bundle, state)[0]
            object_id = read_scene_state(state)["frames"][0]["objects"][0]["id"]
            report = verify_conditioning(state, {1: {
                "camera": camera, "object_ids": {object_id: ids[..., 0] > 0},
                "beauty": conditioned.beauty, "dominant_light_direction": [0, -1, 0],
                "next_beauty": conditioned.beauty,
                "motion_reference": np.zeros((24, 24, 2), np.float32),
                "depth_landmarks": {"near": {"expected": 1, "observed": 1},
                                    "far": {"expected": 2, "observed": 2}},
                "reference_light_direction": [0, -1, 0],
                "shadow_direction": [1, 0, 0], "reference_shadow_direction": [1, 0, 0],
            }}, root / "verify")
            self.assertEqual(report["score_card"]["verdict"], "PASS")
            self.assertTrue(all(item["pass"] for item in report["score_card"]["bindings"].values()))

    def test_refuses_unsupported_supplied_text_with_provider_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, _plate = self._bundle(root)
            with self.assertRaisesRegex(ValueError, "Provider reproject cannot honour controls: text"):
                generate(bundle, state, root / "out.####.exr", "reproject", text="blue sky")


if __name__ == "__main__":
    unittest.main()
