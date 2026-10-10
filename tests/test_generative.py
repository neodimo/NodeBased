import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased import scene3d
from nodebased.conditioned_read import read_conditioned_sequence
from nodebased.conditioning_verify import verify_conditioning
from nodebased.artifacts import ArtifactStore
from nodebased.control_bundle import write_control_bundle
from nodebased.core import Dispatcher
from nodebased.generative import ProviderDescription, generate
from nodebased.imaging import Evaluator
from nodebased.scene_state import read_scene_state, write_scene_state
from nodebased.workers import Job, Worker
from tests.test_conditioning_verify import ConditioningVerificationTests
from tests.test_scene_state import _shot as plan7_shot


class GenerativeProviderTests(unittest.TestCase):
    def test_plan7_scene_has_artifacts_at_every_link(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples = {f: {"scene": plan7_shot(f)[0], "camera": plan7_shot(f)[1], "resolution": (32, 24)}
                       for f in (1, 2, 3)}
            state, scene_id = write_scene_state(root / "plan7.scene.json", samples,
                                                 resolution=(32, 24), return_artifact_id=True)
            planes = {"beauty": np.ones((24, 32, 4), np.float32), "depth": np.zeros((24, 32), np.float32),
                      "normals": np.zeros((24, 32, 3), np.float32),
                      "motion_forward": np.zeros((24, 32, 2), np.float32),
                      "motion_backward": np.zeros((24, 32, 2), np.float32),
                      "object_ids": np.zeros((24, 32), np.float32)}
            bundle, bundle_id = write_control_bundle(state, root / "controls", {f: planes for f in (1, 2, 3)},
                                                     return_artifact_id=True)
            generated = generate(bundle, state, root / "plan7.####.exr", "null")
            self.assertTrue(generated["artifact_id"])
            self.assertEqual(len(read_conditioned_sequence(generated["artifact_id"], bundle, state)), 3)
            report = verify_conditioning(state, {f: {} for f in (1, 2, 3)}, root / "plan7-report",
                                         artifact_ids=[generated["artifact_id"], bundle_id])
            rows = ArtifactStore().provenance(report["artifact_id"])
            kinds = [row["kind"] for row in rows]
            self.assertEqual(kinds, ["scene_state", "control_bundle", "provider_description",
                                     "generated_sequence", "verification_report"])
            self.assertEqual(rows[0]["id"], scene_id)
            self.assertIn("verification_report", subprocess.check_output(
                [sys.executable, "-m", "nodebased.artifacts", "provenance", report["artifact_id"]],
                text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}))

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
            self.assertEqual(reproject["honoured"], ["depth", "motion", "seed"])
            self.assertEqual(null["honoured"], ["seed"])
            # Frame 1 has nothing before it and is copied; frame 2 is frame 1's beauty carried one pixel
            # right by frame 1's forward motion, so each output frame lines up with its own camera.
            first = self._read(root / "reproject.0001.exr")
            np.testing.assert_array_equal(first, plate)
            projected = self._read(root / "reproject.0002.exr")
            self.assertEqual(float(projected[3, 4, 0]), 4.0)
            self.assertEqual(float(projected[3, 0, 0]), 0.0)
            np.testing.assert_array_equal(self._read(root / "null.0001.exr"), plate)
            conditioned = read_conditioned_sequence(str(root / "reproject.####.exr"), bundle, state)
            self.assertEqual(len(conditioned), 2)
            self.assertEqual(conditioned[0].layers["motion_forward"].shape, (8, 8, 2))

    def test_the_same_frames_written_later_get_the_same_artifact_id(self):
        # Generated frames are content-addressed; a header timestamp must not change their bytes.
        import time
        from nodebased.generative import _write_frame
        frame = np.zeros((4, 4, 4), np.float32)
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.exr", Path(folder) / "b.exr"
            _write_frame(first, frame, "ACEScg")
            time.sleep(1.1)
            _write_frame(second, frame, "ACEScg")
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_worker_reproject_matches_in_process_artifact_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, _plate = self._bundle(root)
            inline = generate(bundle, state, root / "same.%04d.exr", "reproject")
            sidecar = bundle.with_suffix(".artifact.json")
            bundle_id = json.loads(sidecar.read_text())["artifact_id"]
            worker = Worker(Job("reproject", bundle_id, {"manifest_path": str(bundle),
                           "scene_state_path": str(state), "output_pattern": str(root / "same.%04d.exr")}))
            isolated = worker.run()
            self.assertEqual(isolated["type"], "result")
            self.assertEqual(isolated["artifact_id"], inline["artifact_id"])

    def test_descriptor_missing_motion_marks_capability_and_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, _plate = self._bundle(root)
            desc = json.loads((Path(__file__).parents[1] / "nodebased/providers/reproject.json").read_text())
            desc["accepts"]["motion"]["accepted"] = False
            desc["returns"]["honoured"] = ["depth", "seed"]
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
            generated = generate(bundle, state, root / "sequence.####.exr", "null")
            conditioned = read_conditioned_sequence(generated["artifact_id"], bundle, state)[0]
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
            }}, root / "verify", artifact_ids=[generated["artifact_id"],
                json.loads(bundle.with_suffix(".artifact.json").read_text())["artifact_id"]])
            self.assertEqual(report["score_card"]["verdict"], "PASS")
            self.assertTrue(all(item["pass"] for item in report["score_card"]["bindings"].values()))
            chain = ArtifactStore().provenance(report["artifact_id"])
            self.assertEqual([row["kind"] for row in chain],
                             ["scene_state", "control_bundle", "provider_description", "generated_sequence", "verification_report"])
            self.assertEqual(len({row["id"] for row in chain}), 5)


    def test_unhonoured_text_runs_the_shot_and_is_recorded_as_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, _plate = self._bundle(root)
            plain = generate(bundle, state, root / "plain.####.exr", "reproject")
            briefed = generate(bundle, state, root / "briefed.####.exr", "reproject", text="blue sky")
            self.assertEqual(plain["ignored"], [])
            self.assertEqual(briefed["ignored"], ["text"])
            self.assertEqual(briefed["honoured"], plain["honoured"])
            manifest = self._generation(briefed["artifact_id"])
            self.assertEqual(manifest["ignored_controls"], ["text"])
            self.assertEqual(manifest["options"]["text"], "blue sky")
            row = ArtifactStore().provenance(briefed["artifact_id"])[-1]
            self.assertEqual(row["provenance"]["ignored_controls"], ["text"])
            # the text is part of what was asked, so it is part of the artifact identity
            self.assertNotEqual(briefed["artifact_id"], plain["artifact_id"])
            # a provider that honours a control records nothing as ignored for it
            self.assertEqual(generate(bundle, state, root / "seeded.####.exr", "reproject", seed=3)["ignored"], [])

    def _generation(self, artifact_id):
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(ArtifactStore().get(artifact_id))) as archive:
            return json.loads(archive.read("generation.json"))

    def test_inputs_the_provider_cannot_take_are_still_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state, bundle, _plate = self._bundle(root)
            desc = json.loads((Path(__file__).parents[1] / "nodebased/providers/reproject.json").read_text())
            desc["limits"]["max_frames"] = 1
            path = root / "one-frame.json"
            path.write_text(json.dumps(desc))
            with self.assertRaisesRegex(ValueError, "does not support 2 frames"):
                generate(bundle, state, root / "out.####.exr", "reproject", description=path, text="blue sky")
            desc["limits"]["max_frames"] = 10
            desc["limits"]["max_resolution"] = [2, 2]
            path.write_text(json.dumps(desc))
            with self.assertRaisesRegex(ValueError, "resolution limit"):
                generate(bundle, state, root / "out.####.exr", "reproject", description=path)
            with self.assertRaisesRegex(ValueError, "cannot honour controls: sampler_settings"):
                generate(bundle, state, root / "out.####.exr", "reproject", sampler_settings={"steps": 4})

    def test_worker_runs_640_by_360_frames_without_the_caller_limiting_threads(self):
        """The 2 GiB address-space cap used to fail above about 300x180 on a many-core machine until the
        caller limited numpy's thread pools; the worker now does that, and sizes its cap from the job."""
        from nodebased.workers import DEFAULT_MEMORY_BYTES, job_memory_bytes
        width, height = 640, 360
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 5)))
            state = root / "big.scene.json"
            frames = (1, 2, 3)
            write_scene_state(state, {f: {"scene": scene3d.Scene(), "camera": camera,
                                          "resolution": (width, height)} for f in frames},
                              resolution=(width, height))
            rng = np.random.default_rng(3)
            planes = {"beauty": rng.random((height, width, 4), dtype=np.float32),
                      "depth": np.full((height, width), 5, np.float32),
                      "normals": np.zeros((height, width, 3), np.float32),
                      "motion_forward": np.ones((height, width, 2), np.float32),
                      "motion_backward": np.zeros((height, width, 2), np.float32),
                      "object_ids": np.zeros((height, width), np.float32)}
            bundle, bundle_id = write_control_bundle(state, root / "controls", {f: planes for f in frames},
                                                     return_artifact_id=True)
            options = {"manifest_path": str(bundle), "scene_state_path": str(state),
                       "output_pattern": str(root / "big.%04d.exr")}
            self.assertEqual(job_memory_bytes(options), DEFAULT_MEMORY_BYTES)
            result = Worker(Job("reproject", bundle_id, options)).run()
            self.assertEqual(result["type"], "result", result)
            for number in frames:
                frame = self._read(root / f"big.{number:04d}.exr")
                self.assertEqual(frame.shape[:2], (height, width))
                self.assertTrue(np.isfinite(frame).all())

    def test_worker_cap_grows_with_the_job_and_never_drops_below_the_default(self):
        from nodebased.workers import DEFAULT_MEMORY_BYTES, job_memory_bytes
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"frames": [{"pixel_size": [3840, 2160]}] * 40}))
            options = {"manifest_path": str(manifest)}
            self.assertGreater(job_memory_bytes(options), DEFAULT_MEMORY_BYTES)
            self.assertGreaterEqual(job_memory_bytes(options), 3840 * 2160 * 40 * 13 * 4)
            manifest.write_text(json.dumps({"frames": [{"pixel_size": [16, 9]}]}))
            self.assertEqual(job_memory_bytes(options), DEFAULT_MEMORY_BYTES)
            self.assertEqual(job_memory_bytes({"manifest_path": str(Path(folder) / "missing.json")}),
                             DEFAULT_MEMORY_BYTES)

    def test_generate_is_a_safe_plate_tap_and_names_missing_plate(self):
        graph = Dispatcher()
        graph.execute({"op": "create", "id": "plate", "type": "Constant",
                       "params": {"width": 2, "height": 2, "red": 0.25, "green": 0.5,
                                  "blue": 0.75, "alpha": 1.0}})
        graph.execute({"op": "create", "id": "generate", "type": "Generate"})
        graph.execute({"op": "connect", "id": "generate", "input": "image", "source": "plate"})
        result = Evaluator().evaluate_raster(graph.document, "generate")
        np.testing.assert_array_equal(result.pixels,
                                      Evaluator().evaluate_raster(graph.document, "plate").pixels)
        unwired = Dispatcher()
        unwired.execute({"op": "create", "id": "generate", "type": "Generate"})
        with self.assertRaisesRegex(ValueError, "Generate: connect a source plate image"):
            Evaluator().evaluate_raster(unwired.document, "generate")


if __name__ == "__main__":
    unittest.main()
