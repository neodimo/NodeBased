"""SVD capability and worker wiring, exercised without model weights."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased.generative import ProviderDescription, generate
from nodebased.providers import svd_image_video as svd
from nodebased.workers import Job, Worker
from tests.test_determinism import make_shot

STUB = '''
import json, sys, numpy as np
job=json.load(open(sys.argv[-1]))
assert job['frames']==2 and job['settings']['fps']==8
for i in range(job['frames']):
    array=np.full((job['height'],job['width'],3),20+i,dtype=np.uint8)
    np.save(job['result_dir']+f'/result_{i:04d}.npy',array)
    print(f"progress {i+1} {job['frames']}",flush=True)
json.dump({'peak_vram_allocated_mib':1},open(job['info'],'w'))
'''


class SVDTests(unittest.TestCase):
    def test_old_demo_document_loads_with_reference_image_default(self):
        from nodebased.core import load_document
        path = Path(__file__).resolve().parents[1] / "examples/demo_shot/demo_shot.nbcomp"
        doc = load_document(path)
        self.assertEqual(doc["nodes"]["generate"]["params"]["reference_image"], "")

    def test_description_shows_supported_and_ignored_controls(self):
        desc = ProviderDescription.load("svd_image_video")
        self.assertTrue(desc.local_gpu)
        self.assertEqual(desc.returns["honoured"], ["reference_frames", "seed", "sampler_settings"])
        self.assertEqual(desc.control_report()["camera_pose"], "ignored")
        self.assertEqual(desc.control_report()["depth"], "ignored")
        self.assertIn("Reference Frames: used", desc.panel_lines())
        self.assertTrue(Worker(Job("svd_image_video", "bundle")).local_gpu)

    def test_model_entrypoint_loads_without_repository_package_context(self):
        result = subprocess.run([sys.executable, "-I", str(Path(svd.__file__))],
                                capture_output=True, text=True)
        self.assertIn("usage: svd_image_video.py", result.stderr)
        self.assertNotIn("ImportError", result.stderr)

    def test_settings_refuse_unknown_and_bad_values(self):
        self.assertEqual(svd.resolve_settings({"fps": 8})["fps"], 8)
        for bad in ({"foo": 1}, {"fps": 0}, {"steps": True}, {"noise_aug_strength": -1}):
            with self.assertRaises(ValueError):
                svd.resolve_settings(bad)

    @unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
    def test_request_building_generation_and_worker_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2))
            (root / "stub.py").write_text(STUB)
            launcher = root / "stub-python"
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "stub.py"}" "$(eval echo \\${{$#}})"\n')
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
            import OpenImageIO as oiio
            ref = root / "reference.png"
            output = oiio.ImageOutput.create(str(ref))
            output.open(str(ref), oiio.ImageSpec(16, 12, 3, oiio.UINT8))
            output.write_image(np.full((12, 16, 3), 128, np.uint8))
            output.close()
            env = {"NODEBASED_CACHE": str(root / "cache"), svd.PYTHON_ENV: str(launcher),
                   svd.MODEL_ENV: str(root)}
            with mock.patch.dict(os.environ, env):
                result = generate(bundle, state, root / "out" / "frame.####.exr", "svd_image_video",
                                  reference_frames=[str(ref)], text="ignored", seed=3,
                                  sampler_settings={"fps": 8})
                self.assertEqual(result["frames"], [1, 2])
                self.assertEqual(result["ignored"], ["text"])
                self.assertEqual(result["run"]["peak_vram_allocated_mib"], 1)
                self.assertEqual(result["sampler_settings"]["fps"], 8)
                options = {"manifest_path": str(bundle), "scene_state_path": str(state),
                           "output_pattern": str(root / "worker" / "frame.####.exr"),
                           "reference_frames": [str(ref)], "sampler_settings": {"fps": 8}}
                worker_result = Worker(Job("svd_image_video", "bundle", options)).run()
                self.assertEqual(worker_result["result"]["frames"], [1, 2])

    @unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
    def test_cancel_terminates_model_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2))
            (root / "stub.py").write_text("import time\nprint('progress 1 2',flush=True)\ntime.sleep(60)\n")
            launcher = root / "stub-python"
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "stub.py"}" "$(eval echo \\${{$#}})"\n')
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
            with mock.patch.dict(os.environ, {svd.PYTHON_ENV: str(launcher), svd.MODEL_ENV: str(root)}):
                def progress(_):
                    raise RuntimeError("Job cancelled")
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    generate(bundle, state, root / "out" / "frame.####.exr", "svd_image_video",
                             sampler_settings={"fps": 8}, progress=progress)

@unittest.skipUnless(os.environ.get("NODEBASED_PROVIDER_TEST") == "1", "set NODEBASED_PROVIDER_TEST=1 for real SVD generation")
class RealSVDTest(unittest.TestCase):
    def test_two_frame_generation_on_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2))
            result = generate(bundle, state, root / "out" / "frame.####.exr", "svd_image_video",
                              seed=3, sampler_settings={"steps": 2})
            self.assertEqual(len(result["outputs"]), 2)
            self.assertGreater(result["run"]["peak_vram_allocated_mib"], 0)
