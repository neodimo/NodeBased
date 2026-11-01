"""CameraCtrl provider: camera conversion, capability card and worker wiring, exercised without model weights."""
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import scene3d
from nodebased.generative import ProviderDescription, generate
from nodebased.providers import cameractrl_svd as ctrl
from nodebased.workers import Job, Worker
from tests.test_determinism import make_shot

STUB = '''
import json, sys, numpy as np
job=json.load(open(sys.argv[-1]))
assert job['frames']==2 and job['settings']['fps']==8 and len(job['poses'])==2
assert all(len(row)==19 for row in job['poses'])
assert job['generation_size']==[576,320]
json.dump(job['poses'],open(job['result_dir']+'/poses_seen.json','w'))
for i in range(job['frames']):
    array=np.full((job['height'],job['width'],3),20+i,dtype=np.uint8)
    np.save(job['result_dir']+f'/result_{i:04d}.npy',array)
    print(f"step {i+1} 2",flush=True)
    print(f"progress {i+1} {job['frames']}",flush=True)
json.dump({'peak_vram_allocated_mib':1},open(job['info'],'w'))
'''


def camera(position=(0, 0, 0), target=(0, 0, -1), fov=45.0, roll=0.0):
    return scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(*position)),
                          target=scene3d.Vec3(*target), fov=fov, roll=roll)


def w2c_of(row):
    return np.array(row[7:], np.float64).reshape(3, 4)


def relative_poses(rows):
    """CameraCtrl's inference.get_relative_pose with zero_first_frame_scale, reimplemented from the pinned commit."""
    w2cs, c2ws = [], []
    for row in rows:
        w2c = np.eye(4)
        w2c[:3, :] = w2c_of(row)
        w2cs.append(w2c)
        c2ws.append(np.linalg.inv(w2c))
    abs2rel = np.eye(4) @ w2cs[0]
    return [np.eye(4)] + [abs2rel @ c2w for c2w in c2ws[1:]]


class PoseRowTests(unittest.TestCase):
    def test_identity_camera_flips_y_and_z_into_opencv(self):
        (row,) = ctrl.pose_rows([camera()], 576, 320)
        self.assertEqual(len(row), 19)
        np.testing.assert_allclose(w2c_of(row), [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]], atol=1e-6)
        self.assertEqual(row[5:7], [0.0, 0.0])
        self.assertEqual(row[3:5], [0.5, 0.5])

    def test_camera_moved_along_world_x_translates_the_world_the_other_way(self):
        (row,) = ctrl.pose_rows([camera(position=(1, 0, 0), target=(1, 0, -1))], 576, 320)
        np.testing.assert_allclose(w2c_of(row)[:, 3], [-1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(w2c_of(row)[:, :3], np.diag([1, -1, -1]), atol=1e-6)

    def test_camera_behind_the_origin_has_positive_depth_for_points_ahead(self):
        # a camera at z=5 looking at the origin sees the origin 5 units in front of it (OpenCV +Z)
        (row,) = ctrl.pose_rows([camera(position=(0, 0, 5), target=(0, 0, 0))], 576, 320)
        point = w2c_of(row) @ np.array([0, 0, 0, 1.0])
        np.testing.assert_allclose(point, [0, 0, 5], atol=1e-6)

    def test_quarter_turn_to_the_left_is_a_pure_yaw_after_the_relative_pose_step(self):
        rows = ctrl.pose_rows([camera(), camera(target=(-1, 0, 0))], 576, 320)
        first, second = relative_poses(rows)
        np.testing.assert_allclose(first, np.eye(4), atol=1e-9)
        # the camera now looks along the first camera's -X; its right axis is the first camera's +Z
        np.testing.assert_allclose(second[:3, :3], [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-6)
        np.testing.assert_allclose(second[:3, 3], 0, atol=1e-6)
        angle = math.degrees(math.atan2(second[0, 2], second[0, 0]))
        self.assertAlmostEqual(abs(angle), 90.0, places=4)

    def test_dolly_is_translation_along_the_cameras_own_forward_axis(self):
        rows = ctrl.pose_rows([camera(position=(2, 1, 5), target=(2, 1, 4)),
                               camera(position=(2, 1, 4), target=(2, 1, 3))], 576, 320)
        second = relative_poses(rows)[1]
        np.testing.assert_allclose(second[:3, :3], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(second[:3, 3], [0, 0, 1], atol=1e-6)  # +Z is forward in OpenCV

    def test_intrinsics_follow_the_vertical_field_of_view(self):
        (row,) = ctrl.pose_rows([camera(fov=90.0)], 576, 320)
        self.assertAlmostEqual(row[2], 0.5, places=9)
        self.assertAlmostEqual(row[1], 0.5 * 320 / 576, places=9)
        (narrow,) = ctrl.pose_rows([camera(fov=40.0)], 576, 320)
        self.assertAlmostEqual(narrow[2], 0.5 / math.tan(math.radians(20.0)), places=9)

    def test_rows_are_json_serialisable_floats_and_reject_non_cameras(self):
        rows = ctrl.pose_rows([camera(), camera(position=(0, 0, -1), target=(0, 0, -2))], 576, 320)
        self.assertEqual(json.loads(json.dumps(rows)), rows)
        with self.assertRaisesRegex(ValueError, "Camera on every frame"):
            ctrl.pose_rows([camera(), None], 576, 320)


class CameraCtrlProviderTests(unittest.TestCase):
    def test_description_accepts_the_camera_and_reports_the_rest_ignored(self):
        desc = ProviderDescription.load("cameractrl_svd")
        self.assertTrue(desc.local_gpu)
        self.assertEqual(desc.max_frames, 25)
        self.assertEqual(list(desc.max_resolution), [576, 320])
        report = desc.control_report()
        self.assertEqual(report["camera_pose"], "used")
        for name in ("depth", "normals", "motion", "ids", "text"):
            self.assertEqual(report[name], "ignored", name)
        for name in ("reference_frames", "seed", "sampler_settings"):
            self.assertEqual(report[name], "used", name)
        self.assertEqual(desc.returns["honoured"], ["camera_pose", "reference_frames", "seed", "sampler_settings"])
        self.assertIn("Camera Pose: used", desc.panel_lines())
        self.assertTrue(Worker(Job("cameractrl_svd", "bundle")).local_gpu)

    def test_model_entrypoint_loads_without_repository_package_context(self):
        result = subprocess.run([sys.executable, "-I", str(Path(ctrl.__file__))], capture_output=True, text=True)
        self.assertIn("usage: cameractrl_svd.py", result.stderr)
        self.assertNotIn("ImportError", result.stderr)

    def test_settings_match_step_a_and_refuse_unknown_and_bad_values(self):
        self.assertEqual(ctrl.resolve_settings({"fps": 8})["fps"], 8)
        for bad in ({"foo": 1}, {"fps": 0}, {"steps": True}, {"noise_aug_strength": -1}):
            with self.assertRaises(ValueError):
                ctrl.resolve_settings(bad)

    def test_too_many_frames_and_a_missing_environment_are_refused_before_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2))
            with mock.patch.dict(os.environ, {ctrl.MODEL_ENV: str(root / "none")}):
                with self.assertRaisesRegex(RuntimeError, ctrl.MODEL_ENV):
                    generate(bundle, state, root / "out" / "frame.####.exr", "cameractrl_svd")
            with mock.patch.dict(os.environ, {ctrl.VARIANT_ENV: "svd"}):
                controls = [mock.Mock(frame=i) for i in range(15)]
                with self.assertRaisesRegex(ValueError, "at most 14"):
                    ctrl.run_frames(controls, cameras=[], reference_frames=None, seed=0, sampler_settings={},
                                    workdir=root / "work")
            with mock.patch.dict(os.environ, {ctrl.VARIANT_ENV: "nope"}):
                with self.assertRaisesRegex(ValueError, ctrl.VARIANT_ENV):
                    ctrl.variant()

    def test_cameras_come_from_the_scene_state_in_bundle_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2))
            from nodebased.control_bundle import read_control_bundle
            controls = read_control_bundle(bundle, state)
            cameras = ctrl.cameras_for(controls, state)
            self.assertEqual(len(cameras), 2)
            self.assertTrue(all(isinstance(c, scene3d.Camera) for c in cameras))

    @unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
    def test_request_building_generation_and_worker_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2))
            (root / "stub.py").write_text(STUB)
            launcher = root / "stub-python"
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{root / "stub.py"}" "$(eval echo \\${{$#}})"\n')
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
            (root / "src" / "cameractrl").mkdir(parents=True)
            env = {"NODEBASED_CACHE": str(root / "cache"), ctrl.PYTHON_ENV: str(launcher),
                   ctrl.MODEL_ENV: str(root), ctrl.SOURCE_ENV: str(root / "src")}
            seen = []
            with mock.patch.dict(os.environ, env):
                result = generate(bundle, state, root / "out" / "frame.####.exr", "cameractrl_svd",
                                  text="ignored", seed=3, sampler_settings={"fps": 8},
                                  progress=lambda value: seen.append(value))
                self.assertEqual(result["frames"], [1, 2])
                self.assertEqual(result["ignored"], ["text"])
                self.assertEqual(result["honoured"].count("camera_pose"), 1)
                self.assertEqual(result["run"]["peak_vram_allocated_mib"], 1)
                self.assertEqual(result["run"]["checkpoint"], "CameraCtrl_svdxt.ckpt")
                self.assertTrue(any("denoising step" in message for _, message in seen))
                options = {"manifest_path": str(bundle), "scene_state_path": str(state),
                           "output_pattern": str(root / "worker" / "frame.####.exr"),
                           "sampler_settings": {"fps": 8}}
                worker_result = Worker(Job("cameractrl_svd", "bundle", options)).run()
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
            (root / "src" / "cameractrl").mkdir(parents=True)
            env = {ctrl.PYTHON_ENV: str(launcher), ctrl.MODEL_ENV: str(root), ctrl.SOURCE_ENV: str(root / "src")}
            with mock.patch.dict(os.environ, env):
                def progress(_):
                    raise RuntimeError("Job cancelled")
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    generate(bundle, state, root / "out" / "frame.####.exr", "cameractrl_svd",
                             sampler_settings={"fps": 8}, progress=progress)


@unittest.skipUnless(os.environ.get("NODEBASED_PROVIDER_TEST") == "1",
                     "set NODEBASED_PROVIDER_TEST=1 for real CameraCtrl generation")
class RealCameraCtrlTest(unittest.TestCase):
    def test_four_frame_generation_on_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state, _, bundle, _ = make_shot(root, frames=(1, 2, 3, 4))
            result = generate(bundle, state, root / "out" / "frame.####.exr", "cameractrl_svd",
                              seed=3, sampler_settings={"steps": 2})
            self.assertEqual(len(result["outputs"]), 4)
            self.assertGreater(result["run"]["peak_vram_allocated_mib"], 0)


if __name__ == "__main__":
    unittest.main()
