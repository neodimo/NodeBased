"""Temporal coherence (stage 8 of the end-to-end sprint): the warp, the flicker measure, the report's temporal line."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nodebased import scene3d, temporal
from nodebased.artifacts import ArtifactStore
from nodebased.conditioning_verify import temporal_line
from nodebased.control_bundle import write_control_bundle
from nodebased.generative import generate
from nodebased.loops import verify_task
from nodebased.scene_state import write_scene_state


def texture(height, width, seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((height, width, 3)).astype(np.float32)


class WarpTests(unittest.TestCase):
    def test_whole_pixel_translation_is_exact(self):
        image = texture(20, 30)
        motion = np.zeros((20, 30, 2), np.float32)
        motion[..., 0], motion[..., 1] = -3.0, 2.0  # each pixel of frame N was 3 px left and 2 px lower on N-1
        warped, inside = temporal.warp(image, motion)
        expected = np.zeros_like(image)
        expected[:18, 3:] = image[2:, :27]
        np.testing.assert_array_equal(warped, expected)
        self.assertTrue(inside[:18, 3:].all())
        self.assertFalse(inside[18:].any() or inside[:, :3].any())

    def test_fractional_translation_is_exact_on_a_linear_ramp(self):
        height, width = 8, 16
        ramp = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) * 0.5
        motion = np.zeros((height, width, 2), np.float32)
        motion[..., 0] = -1.25
        warped, inside = temporal.warp(ramp, motion)
        expected = (np.arange(width, dtype=np.float32) - 1.25) * 0.5
        np.testing.assert_allclose(warped[:, 2:], np.tile(expected[2:], (height, 1)), atol=1e-6)
        self.assertFalse(inside[:, :1].any())

    def test_mask_drops_changed_ids_and_occlusions(self):
        h, w = 10, 10
        motion = np.zeros((h, w, 2), np.float32)
        depth = np.full((h, w), 2.0, np.float32)
        ids = np.zeros((h, w), np.float32)
        previous_ids, previous_depth = ids.copy(), depth.copy()
        previous_ids[2, 2] = 5.0          # an object left this pixel
        previous_depth[6, 6] = 1.0        # a nearer surface hid this pixel on the frame before
        depth[8, 8] = 0.0                 # empty now, a surface before
        mask = temporal.valid_mask(motion, depth, ids, previous_depth, previous_ids)
        self.assertFalse(mask[2, 2] or mask[6, 6] or mask[8, 8])
        self.assertEqual(int(mask.sum()), h * w - 3)


class FlickerTests(unittest.TestCase):
    def controls(self, frames, h, w, shift):
        motion = np.zeros((h, w, 2), np.float32)
        motion[..., 0] = -shift
        return {f: {"motion_backward": motion, "depth": np.full((h, w), 3.0, np.float32),
                    "ids": np.zeros((h, w), np.float32)} for f in frames}

    def test_zero_when_each_frame_is_the_warped_copy_of_the_one_before(self):
        frames, h, w = (1, 2, 3, 4), 16, 24
        controls = self.controls(frames, h, w, 1.5)
        generated = {1: texture(h, w)}
        for before, after in zip(frames, frames[1:]):
            generated[after] = temporal.warp(generated[before], controls[after]["motion_backward"])[0]
        result = temporal.flicker(list(frames), generated, controls)
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["p95"], 0.0)
        self.assertTrue(result["pass"])
        self.assertEqual(result["pairs_measured"], 3)
        self.assertGreater(result["compared_fraction"], 0.8)

    def test_independent_noise_frames_fail(self):
        frames, h, w = (1, 2, 3), 16, 24
        controls = self.controls(frames, h, w, 0.0)
        generated = {f: texture(h, w, seed=f) for f in frames}
        result = temporal.flicker(list(frames), generated, controls)
        self.assertGreater(result["mean"], temporal.THRESHOLD_MEAN)
        self.assertFalse(result["pass"])
        self.assertIn("FAIL", temporal_line(result))


def make_moving_shot(root, frames=(1, 2, 3), width=40, height=24, shift=2):
    """A plate whose texture slides `shift` pixels right per frame, with the matching exported motion."""
    camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 5)))
    state, scene_id = write_scene_state(
        root / "shot.scene.json",
        {f: {"scene": scene3d.Scene(), "camera": camera, "resolution": (width, height)} for f in frames},
        resolution=(width, height), return_artifact_id=True)
    wide = texture(height, width + shift * len(frames))
    samples = {}
    for index, f in enumerate(frames):
        offset = shift * (len(frames) - index)
        beauty = np.ones((height, width, 4), np.float32)
        beauty[..., :3] = wide[:, offset:offset + width] * 0.8
        forward = np.zeros((height, width, 2), np.float32); forward[..., 0] = shift
        backward = np.zeros((height, width, 2), np.float32); backward[..., 0] = -shift
        samples[f] = {"beauty": beauty, "depth": np.full((height, width), 4.0, np.float32),
                      "normals": np.tile(np.array([0, 0, 1], np.float32), (height, width, 1)),
                      "motion_forward": forward, "motion_backward": backward,
                      "object_ids": np.zeros((height, width), np.float32)}
    bundle, bundle_id = write_control_bundle(state, root / "controls", samples, return_artifact_id=True)
    return state, scene_id, bundle, bundle_id


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(self.root / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ArtifactStore()

    def test_report_carries_the_temporal_line_and_the_verdict_ignores_it(self):
        state, scene_id, bundle, bundle_id = make_moving_shot(self.root)
        generated = generate(bundle, state, self.root / "out" / "out.####.exr", "null")["artifact_id"]
        result = verify_task({"scene_state_id": scene_id, "control_bundle_id": bundle_id,
                              "_artifact_root": str(self.store.root)}, [generated], lambda *_: None, threading.Event())
        report = json.loads(self.store.get(result["artifact_id"]))
        t = report["temporal"]
        self.assertEqual(t["pairs_measured"], 2)
        self.assertLess(t["mean"], 0.01)  # the plate slides exactly with its vectors: nothing flickers
        self.assertTrue(t["pass"])
        self.assertNotIn("temporal", report["score_card"]["bindings"])

    def test_text_report_has_the_temporal_line(self):
        state, scene_id, bundle, bundle_id = make_moving_shot(self.root)
        from nodebased.conditioning_verify import verify_conditioning
        from nodebased.control_bundle import read_control_bundle
        controls = read_control_bundle(bundle, state)
        observations = {c.frame: {"beauty": c.beauty.values, "not_conditioned": ["camera", "depth", "motion",
                                                                                  "objects"]} for c in controls}
        measured = temporal.flicker([c.frame for c in controls], {c.frame: c.beauty.values for c in controls},
                                    {c.frame: {"motion_backward": c.motion_backward.values, "depth": c.depth.values,
                                               "ids": c.object_ids.values} for c in controls})
        verify_conditioning(state, observations, self.root / "report" / "verification.json", temporal=measured)
        text = (self.root / "report" / "verification.txt").read_text()
        line = [row for row in text.splitlines() if row.startswith("Temporal:")]
        self.assertEqual(len(line), 1)
        self.assertIn("flicker mean 0.00", line[0])
        self.assertIn("threshold mean 4, 95th percentile 12: PASS", line[0])


if __name__ == "__main__":
    unittest.main()
