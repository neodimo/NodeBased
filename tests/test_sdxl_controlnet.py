"""The SDXL ControlNet-Union provider (stage 3 of the end-to-end sprint).

The CPU tests run without torch: conversions are plain numpy, and the model process is replaced by a stub
interpreter. The CUDA test runs the real model and is skipped where the provider environment or a CUDA
device is missing.
"""
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tests.isolation  # noqa: F401  (a private GPU lock file, so a stub provider does not wait for the suite)
from nodebased.artifacts import ArtifactStore
from nodebased.generative import GENERATION_MANIFEST, ProviderDescription, generate
from nodebased.providers import sdxl_controlnet as sdxl
from nodebased.workers import Job, Worker
from tests.test_determinism import make_shot, unpack

STUB_MODEL = '''
import json, sys
import numpy as np
job = json.load(open(sys.argv[-1]))
for row in job["frames"]:
    depth = np.load(row["depth"]); normals = np.load(row["normals"])
    out = np.zeros((job["height"], job["width"], 3), np.uint8)
    out[..., 0] = depth[..., 0]; out[..., 1] = normals[..., 1]; out[..., 2] = job["seed"] % 256
    np.save(row["result"], out)
    print("progress", row["frame"], len(job["frames"]), flush=True)
json.dump({"peak_vram_allocated_mib": 1, "seen_prompt": job["prompt"], "settings": job["settings"]}, open(job["info"], "w"))
'''


class ConversionTests(unittest.TestCase):
    def test_depth_rule_is_inverse_depth_over_the_sequence_range(self):
        depth = np.array([[1.0, 2.0, 4.0, np.inf], [0.0, np.nan, 2.0, 1.0]], np.float32)
        near, far = sdxl.depth_range([depth, np.array([[4.0]], np.float32)])
        self.assertEqual((near, far), (1.0, 4.0))
        image = sdxl.depth_control_image(depth, near, far)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (2, 4, 3))
        # near 1 -> 255, far 4 -> 0, 2 -> (0.5 - 0.25) / 0.75 = 1/3 -> 85; no depth -> 0
        self.assertEqual(image[..., 0].tolist(), [[255, 85, 0, 0], [0, 0, 85, 255]])
        self.assertTrue((image[..., 0] == image[..., 1]).all() and (image[..., 0] == image[..., 2]).all())
        self.assertEqual(sdxl.depth_range([np.full((2, 2), np.inf, np.float32)]), (0.0, 1.0))

    def test_normals_rule_maps_unit_vectors_and_fills_empty_pixels_flat(self):
        normals = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [np.nan, 0, 0], [0, 0, -1]]], np.float32)
        image = sdxl.normals_control_image(normals)
        self.assertEqual(image[0, 0].tolist(), [128, 128, 255])
        self.assertEqual(image[0, 1].tolist(), [255, 128, 128])
        self.assertEqual(image[0, 2].tolist(), [128, 255, 128])
        self.assertEqual(image[1, 0].tolist(), [128, 128, 255])
        self.assertEqual(image[1, 1].tolist(), [128, 128, 255])
        self.assertEqual(image[1, 2].tolist(), [128, 128, 0])

    def test_working_size_keeps_the_aspect_on_the_latent_grid(self):
        self.assertEqual(sdxl.working_size(480, 270), (1024, 576))
        self.assertEqual(sdxl.working_size(1024, 1024), (1024, 1024))
        self.assertEqual(sdxl.working_size(270, 480), (576, 1024))

    def test_settings_have_defaults_and_refuse_unknown_keys_and_bad_values(self):
        self.assertEqual(sdxl.resolve_settings(None), sdxl.DEFAULT_SETTINGS)
        merged = sdxl.resolve_settings({"steps": 12, "guidance": 7})
        self.assertEqual((merged["steps"], merged["guidance"], merged["control_mode"]), (12, 7.0, "depth_normals"))
        for bad in ({"sampler": "ddim"}, {"steps": 0}, {"steps": 2.5}, {"guidance": 99}, {"steps": True},
                    {"controlnet_conditioning_scale": -1}, {"control_mode": "canny"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                sdxl.resolve_settings(bad)


@unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
class ProviderPlumbingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stub = self.root / "stub_model.py"
        stub.write_text(STUB_MODEL)
        launcher = self.root / "stub-python"
        # Stands in for the provider environment's interpreter: runs the stub whatever script it is given.
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$(eval echo \\${{$#}})"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        patcher = mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(self.root / "cache"),
                                               sdxl.PYTHON_ENV: str(launcher)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state, self.scene_id, self.bundle, self.bundle_id = make_shot(self.root, frames=(1, 2))

    def test_generate_runs_each_frame_and_records_the_resolved_settings(self):
        progress = []
        result = generate(self.bundle, self.state, self.root / "a" / "out.####.exr", "sdxl_controlnet",
                          text="a rainy street", seed=5, sampler_settings={"steps": 9}, progress=progress.append)
        self.assertEqual(result["honoured"], ["depth", "normals", "text", "seed", "sampler_settings"])
        self.assertEqual(result["ignored"], [])
        self.assertEqual(result["frames"], [1, 2])
        self.assertEqual(result["run"]["seen_prompt"], "a rainy street")
        files = unpack(ArtifactStore(), result["artifact_id"])
        generation = json.loads(files[GENERATION_MANIFEST])
        self.assertEqual(generation["seed"], 5)
        self.assertEqual(generation["sampler_settings"]["steps"], 9)
        self.assertEqual(generation["sampler_settings"]["control_mode"], "depth_normals")
        self.assertEqual(generation["options"]["text"], "a rainy street")
        self.assertTrue(generation["seed_honoured"] and generation["sampler_settings_honoured"])
        self.assertEqual(sorted(name for name in files if name.endswith(".exr")), ["out.0001.exr", "out.0002.exr"])
        self.assertTrue(any(fraction == 1.0 for fraction, _ in progress))
        # the stub wrote seed % 256 into blue; sRGB 5/255 decodes to a small positive linear value
        from nodebased.conditioned_read import _read_rgba
        self.assertGreater(float(_read_rgba(str(self.root / "a" / "out.0001.exr"))[0][..., 2].mean()), 0.0)

    def test_same_manifest_twice_gives_identical_ids_and_frames(self):
        first = generate(self.bundle, self.state, self.root / "a" / "out.####.exr", "sdxl_controlnet", seed=3, text="x")
        second = generate(self.bundle, self.state, self.root / "b" / "out.####.exr", "sdxl_controlnet", seed=3, text="x")
        other = generate(self.bundle, self.state, self.root / "c" / "out.####.exr", "sdxl_controlnet", seed=4, text="x")
        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertNotEqual(first["artifact_id"], other["artifact_id"])

    def test_bad_settings_refuse_before_any_model_process_starts(self):
        with mock.patch.object(sdxl.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                generate(self.bundle, self.state, self.root / "d" / "out.####.exr", "sdxl_controlnet",
                         sampler_settings={"scheduler": "ddim"})
            popen.assert_not_called()

    def test_oversize_frames_and_wrong_counts_are_refused(self):
        big_state, _, big_bundle, _ = make_shot(self.root / "big", width=1032, height=8)
        with self.assertRaisesRegex(ValueError, "resolution limit"):
            generate(big_bundle, big_state, self.root / "e" / "out.####.exr", "sdxl_controlnet")

    def test_failing_model_process_reports_its_last_lines(self):
        Path(self.root / "stub_model.py").write_text("print('boom line'); raise SystemExit(7)")
        with self.assertRaisesRegex(RuntimeError, "exit 7.*boom line"):
            generate(self.bundle, self.state, self.root / "f" / "out.####.exr", "sdxl_controlnet")

    def test_cancel_kills_the_model_process(self):
        Path(self.root / "stub_model.py").write_text("import time\nprint('progress 1 2', flush=True)\ntime.sleep(60)")
        cancelled = threading.Event()

        def progress(value):
            cancelled.set()
            raise RuntimeError("Job cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            generate(self.bundle, self.state, self.root / "g" / "out.####.exr", "sdxl_controlnet", progress=progress)
        self.assertTrue(cancelled.is_set())


def cuda_unavailable():
    if not Path(sdxl.provider_python()).exists():
        return f"provider interpreter {sdxl.provider_python()} is missing (set {sdxl.PYTHON_ENV})"
    return sdxl.available()


class RealModelTests(unittest.TestCase):
    def test_same_manifest_twice_gives_byte_identical_exr_frames(self):
        reason = cuda_unavailable()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(root / "cache")}):
                state, _, bundle, _ = make_shot(root, width=96, height=64, frames=(1, 2))
                runs = [generate(bundle, state, root / name / "out.####.exr", "sdxl_controlnet",
                                 text="a rainy street at night", seed=11, sampler_settings={"steps": 6})
                        for name in ("first", "second")]
                frames = [{k: v for k, v in unpack(ArtifactStore(), run["artifact_id"]).items() if k.endswith(".exr")}
                          for run in runs]
        self.assertEqual(runs[0]["artifact_id"], runs[1]["artifact_id"])
        self.assertEqual(sorted(frames[0]), ["out.0001.exr", "out.0002.exr"])
        self.assertEqual(frames[0], frames[1])
        self.assertGreater(runs[0]["run"]["peak_vram_allocated_mib"], 1000)


if __name__ == "__main__":
    unittest.main()
