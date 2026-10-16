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
    assert np.load(row["motion_backward"]).shape == (job["height"], job["width"], 2)
    assert np.load(row["z"]).shape == np.load(row["ids"]).shape == (job["height"], job["width"])
    out = np.zeros((job["height"], job["width"], 3), np.uint8)
    out[..., 0] = depth[..., 0]; out[..., 1] = normals[..., 1]; out[..., 2] = job["seed"] % 256
    np.save(row["result"], out)
    print("progress", row["frame"], len(job["frames"]), flush=True)
if job.get("preview_dir"):
    for row in job["frames"]:
        for step in (1, 2):
            path = job["preview_dir"] + "/f%04d_s%d.npy" % (row["frame"], step)
            np.save(path, np.zeros((2, 2, 3), np.uint8))
            print("preview " + json.dumps({"frame": row["frame"], "stage": "text2img", "step": step, "of": 2, "path": path}), flush=True)
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
        self.assertEqual(sdxl.resolve_settings(None)["denoise_strength"], 0.5)
        merged = sdxl.resolve_settings({"steps": 12, "guidance": 7})
        self.assertEqual((merged["steps"], merged["guidance"], merged["control_mode"]), (12, 7.0, "depth_normals"))
        for bad in ({"sampler": "ddim"}, {"steps": 0}, {"steps": 2.5}, {"guidance": 99}, {"steps": True},
                    {"controlnet_conditioning_scale": -1}, {"control_mode": "canny"}, {"denoise_strength": 0},
                    {"denoise_strength": 1.5}):
            with self.assertRaises(ValueError, msg=str(bad)):
                sdxl.resolve_settings(bad)


class FastPreviewSettingsTests(unittest.TestCase):
    """`fast_preview` (spike A): the SDXL-Lightning 4-step path is an opt-in setting; the default is untouched."""

    def test_off_by_default_and_not_recorded_when_false(self):
        self.assertEqual(sdxl.resolve_settings({"fast_preview": False}), sdxl.DEFAULT_SETTINGS)
        self.assertNotIn("fast_preview", sdxl.resolve_settings(None))

    def test_on_pins_four_steps_and_guidance_zero(self):
        settings = sdxl.resolve_settings({"fast_preview": True})
        self.assertEqual((settings["steps"], settings["guidance"], settings["fast_preview"]), (4, 0.0, True))
        self.assertEqual(settings["denoise_strength"], 0.5)   # two of the four trained steps for a guided frame
        self.assertEqual(sdxl.resolve_settings({"fast_preview": True, "steps": 4, "guidance": 0}), settings)

    def test_on_refuses_another_schedule_and_a_strength_with_no_step(self):
        for bad in ({"fast_preview": True, "steps": 8}, {"fast_preview": True, "guidance": 5.0},
                    {"fast_preview": True, "denoise_strength": 0.1}, {"fast_preview": "yes"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                sdxl.resolve_settings(bad)
        self.assertEqual(sdxl.resolve_settings({"fast_preview": True, "denoise_strength": 1.0})["denoise_strength"], 1.0)


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
        self.assertEqual(result["honoured"], ["depth", "normals", "motion", "ids", "text", "seed", "sampler_settings"])
        self.assertEqual(result["ignored"], [])
        self.assertEqual(result["frames"], [1, 2])
        self.assertEqual(result["run"]["seen_prompt"], "a rainy street")
        files = unpack(ArtifactStore(), result["artifact_id"])
        generation = json.loads(files[GENERATION_MANIFEST])
        self.assertEqual(generation["seed"], 5)
        self.assertEqual(generation["sampler_settings"]["steps"], 9)
        self.assertEqual(generation["sampler_settings"]["control_mode"], "depth_normals")
        self.assertEqual(generation["sampler_settings"]["denoise_strength"], 0.5)
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
        # the denoising strength is part of the result's identity like the seed
        weaker = generate(self.bundle, self.state, self.root / "s" / "out.####.exr", "sdxl_controlnet", seed=3,
                          text="x", sampler_settings={"denoise_strength": 0.3})
        self.assertNotEqual(first["artifact_id"], weaker["artifact_id"])

    def test_previews_reach_the_callback_and_never_change_the_frames(self):
        events = []
        controls = __import__("nodebased.generative", fromlist=["x"]).read_control_bundle(self.bundle, self.state)
        plain, _ = sdxl.run_frames(controls, text="x", seed=2, sampler_settings=None, workdir=self.root / "w1")
        shown, _ = sdxl.run_frames(controls, text="x", seed=2, sampler_settings=None, workdir=self.root / "w2",
                                   preview=events.append)
        self.assertEqual([(e["frame"], e["step"]) for e in events], [(1, 1), (1, 2), (2, 1), (2, 2)])
        self.assertTrue(all("received" in e for e in events))
        for a, b in zip(plain, shown):
            np.testing.assert_array_equal(a, b)
        with mock.patch.dict(os.environ, {sdxl.PREVIEW_DIR_ENV: str(self.root / "p")}):
            sdxl.run_frames(controls, text="x", seed=2, sampler_settings=None, workdir=self.root / "w3")
        self.assertEqual(len((self.root / "p" / "previews.jsonl").read_text().splitlines()), 4)

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


class GuideImageTests(unittest.TestCase):
    """The model side's reprojection step, run here with numpy (it needs no torch)."""

    def rows(self, root, motion_x=0.0, ids_after=0.0):
        h, w = 9, 16
        rows = []
        for frame, ids in ((1, 0.0), (2, ids_after)):
            motion = np.zeros((h, w, 2), np.float32); motion[..., 0] = motion_x if frame == 2 else 0.0
            identity = np.zeros((h, w), np.float32); identity[:3, :3] = ids
            paths = {key: str(root / f"{frame}.{key}.npy") for key in ("motion_backward", "z", "ids")}
            np.save(paths["motion_backward"], motion)
            np.save(paths["z"], np.full((h, w), 2.0, np.float32))
            np.save(paths["ids"], identity)
            rows.append({"frame": frame, **paths})
        return rows

    def test_still_frame_is_carried_exactly_and_a_changed_object_is_a_hole(self):
        temporal = sdxl._load_temporal()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before, after = self.rows(root)
            picture = np.random.default_rng(1).integers(0, 255, (18, 32, 3)).astype(np.uint8)
            warped, mask = sdxl.guide_image(temporal, picture, after, before, 32, 18)
            np.testing.assert_array_equal(warped, picture.astype(np.float32))
            self.assertTrue(mask.all())
            before, after = self.rows(root, ids_after=7.0)
            _, mask = sdxl.guide_image(temporal, picture, after, before, 32, 18)
            self.assertFalse(mask[:6, :6].any())
            self.assertTrue(mask[6:, 6:].all())

    def test_motion_is_rescaled_to_the_working_size(self):
        temporal = sdxl._load_temporal()
        with tempfile.TemporaryDirectory() as temp:
            before, after = self.rows(Path(temp), motion_x=-1.0)  # one plate pixel is two working pixels
            picture = np.random.default_rng(2).integers(0, 255, (18, 32, 3)).astype(np.uint8)
            warped, mask = sdxl.guide_image(temporal, picture, after, before, 32, 18)
            np.testing.assert_array_equal(warped[:, 2:], picture[:, :30].astype(np.float32))
            self.assertFalse(mask[:, :2].any())


def cuda_unavailable():
    if not Path(sdxl.provider_python()).exists():
        return f"provider interpreter {sdxl.provider_python()} is missing (set {sdxl.PYTHON_ENV})"
    return sdxl.available()


def lightning_missing():
    try:
        return "" if (sdxl._cached_snapshot(sdxl.LIGHTNING_REPO) / sdxl.LIGHTNING_LORA).exists() else "the Lightning LoRA is not cached"
    except OSError:
        return "the Lightning LoRA is not cached"


class RealModelTests(unittest.TestCase):
    def test_fast_preview_previews_on_or_off_give_byte_identical_frames(self):
        """Spike A: the previews decode through their own VAE copy and draw no random numbers, so the frames match."""
        reason = cuda_unavailable() or lightning_missing()
        if reason:
            self.skipTest(reason)
        from nodebased.generative import read_control_bundle
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(root / "cache")}):
                state, _, bundle, _ = make_shot(root, width=96, height=64, frames=(1, 2))
                controls = read_control_bundle(bundle, state)
                settings = {"fast_preview": True}
                plain, info = sdxl.run_frames(controls, text="a stone wall", seed=3, sampler_settings=settings, workdir=root / "a")
                events = []
                shown, _ = sdxl.run_frames(controls, text="a stone wall", seed=3, sampler_settings=settings,
                                           workdir=root / "b", preview=events.append)
        self.assertTrue(info["fast_preview"])
        for a, b in zip(plain, shown):
            np.testing.assert_array_equal(a, b)
        # frame 1: four text-to-image steps; frame 2: the fill pass (four) and the image-to-image tail (the last two)
        self.assertEqual([(e["stage"], e["step"], e["of"]) for e in events if e["frame"] == 1], [("text2img", i, 4) for i in (1, 2, 3, 4)])
        self.assertEqual([(e["stage"], e["step"], e["of"]) for e in events if e["frame"] == 2 and e["stage"] == "guided"],
                         [("guided", 1, 2), ("guided", 2, 2)])

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
        self.assertEqual(runs[0]["run"]["denoise_strength"], 0.5)  # frame 2 went through image-to-image

    def test_image_to_image_frame_is_pinned_by_seed_and_strength(self):
        """Frame 2 starts from frame 1 warped: the same seed and strength give the same bytes; another strength
        leaves frame 1 (text-to-image as before) alone and changes frame 2 and the id."""
        reason = cuda_unavailable()
        if reason:
            self.skipTest(reason)
        from nodebased.conditioned_read import _read_rgba
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(root / "cache")}):
                state, _, bundle, _ = make_shot(root, width=96, height=64, frames=(1, 2))
                runs = [generate(bundle, state, root / name / "out.####.exr", "sdxl_controlnet", text="a stone wall",
                                 seed=4, sampler_settings={"steps": 6, "denoise_strength": strength})
                        for name, strength in (("a", 0.4), ("b", 0.4), ("c", 0.8))]
                frames = [{k: v for k, v in unpack(ArtifactStore(), run["artifact_id"]).items() if k.endswith(".exr")}
                          for run in runs]
                pixels = {(name, frame): _read_rgba(str(root / name / f"out.{frame:04d}.exr"))[0]
                          for name in ("a", "c") for frame in (1, 2)}
        self.assertEqual(runs[0]["artifact_id"], runs[1]["artifact_id"])
        self.assertEqual(frames[0], frames[1])
        self.assertNotEqual(runs[0]["artifact_id"], runs[2]["artifact_id"])
        np.testing.assert_array_equal(pixels["a", 1], pixels["c", 1])
        self.assertFalse(np.array_equal(pixels["a", 2], pixels["c", 2]))


if __name__ == "__main__":
    unittest.main()
