import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased import scene3d
from nodebased.conditioned_read import _resize_center_cover, read_conditioned_sequence
from nodebased.conditioning_verify import _depth_measurement, _rank_correlation
from nodebased.control_bundle import write_control_bundle
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.scene_state import write_scene_state


class ConditionedReadTests(unittest.TestCase):
    def _bundle(self, root):
        camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 5)))
        state = root / "shot.scene.json"
        write_scene_state(state, {1: {"scene": scene3d.Scene(), "camera": camera,
                                      "resolution": (8, 8)}}, resolution=(8, 8))
        rgba = np.zeros((8, 8, 4), np.float32)
        rgba[..., 3] = 1
        samples = {1: {"beauty": rgba, "depth": np.zeros((8, 8), np.float32),
                       "normals": np.zeros((8, 8, 3), np.float32),
                       "motion_forward": np.zeros((8, 8, 2), np.float32),
                       "motion_backward": np.zeros((8, 8, 2), np.float32),
                       "object_ids": np.zeros((8, 8), np.float32)}}
        manifest = write_control_bundle(state, root / "controls", samples)
        return state, manifest

    def _write_generated(self, path, color_space="ACEScg"):
        pixels = np.ones((4, 8, 4), np.float32)
        spec = oiio.ImageSpec(8, 4, 4, oiio.FLOAT)
        spec.channelnames = ["R", "G", "B", "A"]
        spec.attribute("oiio:ColorSpace", color_space)
        output = oiio.ImageOutput.create(str(path))
        self.assertTrue(output.open(str(path), spec))
        try:
            self.assertTrue(output.write_image(pixels))
        finally:
            output.close()

    def test_conditioned_sequence_aligns_and_exposes_bundle_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, manifest = self._bundle(root)
            self._write_generated(root / "generated.0001.exr")
            result = read_conditioned_sequence(str(root / "generated.####.exr"), manifest, state)[0]
            self.assertEqual(result.beauty.shape, (8, 8, 4))
            self.assertEqual(set(result.layers), {"beauty", "depth", "normals", "motion_forward",
                                                  "motion_backward", "object_ids"})
            self.assertEqual(result.layer_metadata["generated_beauty"]["resize_rule"],
                             "centre-crop-to-fill, then bilinear resize")
            graph = Dispatcher()
            graph.execute({"op": "create", "id": "read", "type": "ConditionedRead",
                           "params": {"path": str(root / "generated.####.exr"),
                                      "manifest": str(manifest), "scene_state": str(state)}})
            raster = Evaluator().evaluate_raster(graph.document, "read", frame=1)
            self.assertEqual(raster.pixels.shape, (8, 8, 4))
            self.assertIn("motion_forward", raster.layers)

    def test_wrong_generated_colour_space_tag_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, manifest = self._bundle(root)
            self._write_generated(root / "generated.0001.exr", "sRGB")
            with self.assertRaisesRegex(ValueError, "colour-space tag"):
                read_conditioned_sequence(str(root / "generated.####.exr"), manifest, state)

    def test_center_cover_returns_shot_dimensions(self):
        image = np.zeros((4, 8, 4), np.float32)
        image[:, 2:6] = 1
        result = _resize_center_cover(image, 4, 4)
        self.assertEqual(result.shape, (4, 4, 4))
        np.testing.assert_allclose(result, 1)

    def test_depth_rank_detects_inverted_landmarks(self):
        score = _depth_measurement({"depth_landmarks": {
            "near": {"expected": 1, "observed": 3},
            "middle": {"expected": 2, "observed": 2},
            "far": {"expected": 3, "observed": 1},
        }}, type("T", (), {"depth_rank_correlation": .8})())
        self.assertEqual(score["rank_correlation"], -1.0)
        self.assertFalse(score["pass"])

    def test_depth_rank_accepts_same_order(self):
        self.assertEqual(_rank_correlation([1, 2, 3], [4, 5, 6]), 1.0)


if __name__ == "__main__":
    unittest.main()
