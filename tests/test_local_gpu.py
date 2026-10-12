"""A provider that drives the GPU declares `local_gpu` in its description; its worker runs without the address-space cap."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nodebased.generative import ProviderDescription
from nodebased.workers import Job, Worker


class DescriptionTests(unittest.TestCase):
    def test_description_declares_what_it_honours(self):
        desc = ProviderDescription.load("sdxl_controlnet")
        self.assertEqual((desc.name, desc.version, desc.locality, desc.local_gpu), ("sdxl_controlnet", "1.2", "local", True))
        self.assertEqual(desc.max_resolution, (1024, 1024))
        self.assertEqual(desc.max_frames, 240)
        self.assertEqual(desc.color_spaces, ("ACEScg",))
        report = desc.control_report()
        for used in ("depth", "normals", "motion", "ids", "text", "seed", "sampler_settings"):
            self.assertEqual(report[used], "used", used)
        for ignored in ("camera_pose", "reference_frames"):
            self.assertEqual(report[ignored], "ignored", ignored)
        self.assertIn("motion is honoured", desc.accepts["motion"]["role"])
        self.assertFalse(ProviderDescription.load("reproject").local_gpu)


@unittest.skipIf(sys.platform == "win32", "RLIMIT_AS is the POSIX cap")
class WorkerCapTests(unittest.TestCase):
    """A GPU subprocess needs CUDA's address space, so the worker's address-space cap must not apply to it."""

    def popen_kwargs(self, provider):
        seen = {}

        def fake_popen(*args, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop after Popen")
        with tempfile.TemporaryDirectory() as temp, mock.patch("nodebased.workers.subprocess.Popen", fake_popen):
            with self.assertRaisesRegex(RuntimeError, "stop after Popen"):
                Worker(Job(provider, "bundle", {"output_pattern": str(Path(temp) / "o.%04d.exr")})).run()
        return seen

    def test_local_gpu_provider_runs_without_the_address_space_cap(self):
        self.assertTrue(Worker(Job("sdxl_controlnet", "bundle")).local_gpu)
        self.assertIsNone(self.popen_kwargs("sdxl_controlnet")["preexec_fn"])

    def test_stand_ins_and_entry_points_keep_the_cap(self):
        for provider in ("reproject", "null", "tests.worker_fixtures:succeeds"):
            self.assertFalse(Worker(Job(provider, "bundle")).local_gpu, provider)
            self.assertIsNotNone(self.popen_kwargs(provider)["preexec_fn"], provider)


if __name__ == "__main__":
    unittest.main()
