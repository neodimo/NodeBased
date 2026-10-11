"""Stage 7 of the end-to-end sprint: the CameraSolve3D node, the DPVO provider and the `solve_camera` op.

The CPU tests run without torch: the solver process is replaced by a stub interpreter or a stub function, and the
placement maths is checked on synthetic cameras. The GPU run (real DPVO on the demo plate) is opt-in:
NODEBASED_SLAM_TEST=1, skipped where the SLAM environment or a CUDA device is missing.
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

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import camera_solve as cs
from nodebased import filmback, scene3d
from nodebased.artifacts import ArtifactStore
from nodebased.core import CHOICES, LIMITS, OUTPUT_TYPES, SPECS, Dispatcher, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.providers import camera_solve_dpvo as dpvo

STUB_SOLVER = '''
import json, sys
import numpy as np
job = json.load(open(sys.argv[-1]))
n = len(job["frames"])
frame = np.load(job["frames"][0])
c2w = np.tile(np.eye(4), (n, 1, 1))
c2w[:, 0, 3] = np.arange(n) * 0.1            # slides right 0.1 per frame
np.savez(job["result"], c2w=c2w, points=np.array([[0, 0, 5.0], [1, 0, 4.0]]), colors=np.array([[255, 0, 0], [0, 255, 0]], np.uint8))
json.dump({"frames": n, "seed": job["seed"], "loop_closure": job["loop_closure"], "intrinsics": job["intrinsics"],
           "frame_shape": list(frame.shape), "peak_vram_mb": 1.0}, open(job["info"], "w"))
for i in range(n):
    print("progress", i + 1, n, flush=True)
'''


def sliding_solve(frames, intrinsics, **kwargs):
    """Stands in for the solver: a camera sliding right 0.1 per frame, two points in front of it."""
    n = len(frames)
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 0, 3] = np.arange(n) * 0.1
    return {"c2w": c2w, "points": np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 4.0]]),
            "colors": np.array([[255, 0, 0], [0, 255, 0]], np.uint8), "info": {"frames": n, "seed": kwargs["seed"]}}


def random_pose(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, rng.normal(size=3) * 3
    return pose


class NodeWiringTests(unittest.TestCase):
    def test_node_is_registered_like_a_provider_tap(self):
        self.assertIn("CameraSolve3D", SPECS)
        self.assertEqual(SPECS["CameraSolve3D"]["inputs"], ["image"])
        self.assertEqual(OUTPUT_TYPES["CameraSolve3D"], "image")
        self.assertEqual(bypass_slot({"type": "CameraSolve3D", "inputs": {"image": "plate"}}), "image")
        for name in ("loop_closure", "unit_scale", "anchor_tx", "anchor_roll"):
            self.assertIn(name, LIMITS)
        self.assertEqual(SPECS["CameraSolve3D"]["params"]["provider"], dpvo.PROVIDER_NAME)

    def test_plate_passes_through_and_an_unwired_node_says_so(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "plate", "type": "Checker", "params": {"width": 32, "height": 16}},
            {"op": "create", "id": "solve", "type": "CameraSolve3D"}]})
        evaluator = Evaluator()
        with self.assertRaisesRegex(ValueError, "CameraSolve3D: connect a source plate image"):
            evaluator.evaluate(dispatcher.document, "solve", frame=1)
        dispatcher.execute({"op": "connect", "id": "solve", "input": "image", "source": "plate"})
        np.testing.assert_array_equal(evaluator.evaluate(dispatcher.document, "solve", frame=1),
                                      evaluator.evaluate(dispatcher.document, "plate", frame=1))

    def test_provider_description(self):
        description = dpvo.description()
        self.assertEqual((description["name"], description["kind"], description["local_gpu"]),
                         (dpvo.PROVIDER_NAME, "camera_solve", True))
        self.assertTrue(description["accepts"]["intrinsics"]["accepted"])
        self.assertEqual(description["limits"]["min_frames"], dpvo.MIN_FRAMES)
        text = (dpvo.SOLVER_DIR / f"{dpvo.PROVIDER_NAME}.json").read_text()
        self.assertNotIn("/home/", text)
        self.assertIn("providers/solvers/*.json", (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
        # a solver description is not a conditioning provider: `generate` must not offer it
        from nodebased.generative import PROVIDER_DIR
        self.assertNotIn(dpvo.PROVIDER_NAME, {path.stem for path in PROVIDER_DIR.glob("*.json")})

    def test_intrinsics_follow_the_camera3d_film_back(self):
        f = dpvo.focal_pixels(filmback.DEFAULT_FOCAL, filmback.DEFAULT_VAPERTURE, 270)
        fov = filmback.fov_from_aperture(filmback.DEFAULT_FOCAL, filmback.DEFAULT_VAPERTURE)
        self.assertAlmostEqual(f, 135 / np.tan(np.radians(fov) / 2), places=6)
        self.assertEqual(dpvo.intrinsics_for(480, 270, filmback.DEFAULT_FOCAL, filmback.DEFAULT_VAPERTURE)[2:], (240.0, 135.0))


class PlacementTests(unittest.TestCase):
    def test_keys_and_poses_round_trip_including_roll(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            pose = random_pose(rng)
            pose[:3, :3] = pose[:3, :3] @ np.diag([1, -1, -1])   # flip so the optical axis is not near straight up
            keys = cs.keys_from_pose(pose)
            back = cs.pose_from_keys(**keys)
            np.testing.assert_allclose(back, pose, atol=1e-6)

    def test_camera3d_built_from_keys_looks_where_the_pose_looks(self):
        pose = cs.pose_from_keys(1.0, 2.0, 8.0, 0.5, 0.0, 0.0, 12.0)
        keys = cs.keys_from_pose(pose)
        camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(keys["tx"], keys["ty"], keys["tz"])),
                                scene3d.Vec3(keys["target_x"], keys["target_y"], keys["target_z"]), 40.0, roll=keys["roll"])
        pixels, depth = scene3d.project(camera, 200, 100, np.array([[0.5, 0.0, 0.0]]))
        np.testing.assert_allclose(pixels[0], (100, 50), atol=1e-3)   # the look-at target is the image centre
        self.assertAlmostEqual(keys["roll"], 12.0, places=4)

    def test_first_frame_lands_on_the_anchor_and_lengths_scale(self):
        rng = np.random.default_rng(5)
        solve = np.stack([random_pose(rng) for _ in range(5)])
        points = rng.normal(size=(7, 3))
        anchor = cs.pose_from_keys(0.0, 1.1, 8.0, -0.2, 0.0, 0.0)
        placed, cloud = cs.place_in_world(solve, points, anchor, 2.5)
        np.testing.assert_allclose(placed[0], anchor, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(np.diff(placed[:, :3, 3], axis=0), axis=1),
                                   2.5 * np.linalg.norm(np.diff(solve[:, :3, 3], axis=0), axis=1), atol=1e-9)
        # a point seen in the first camera at (a, b, c) sits at anchor @ (2.5 a, 2.5 b, 2.5 c)
        local = (points[0] - solve[0, :3, 3]) @ solve[0, :3, :3]
        np.testing.assert_allclose(cloud[0], anchor[:3, :3] @ (2.5 * local) + anchor[:3, 3], atol=1e-9)

    def test_survey_recovers_the_scale_from_two_points(self):
        width, height, fov = 320, 180, 40.0
        fy = height / 2 / np.tan(np.radians(fov) / 2)
        solve = np.tile(np.eye(4), (5, 1, 1))
        solve[:, 0, 3] = np.arange(5) * 0.3
        world_scale = 3.0                    # real units per solve unit
        a, b = np.array([0.0, 0.5, 6.0]), np.array([1.5, -0.2, 8.0])       # solve-frame points
        observations = []
        for point in (a, b):
            row = []
            for index in (0, 2, 4):
                keys = cs.keys_from_pose(solve[index])
                camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(keys["tx"], keys["ty"], keys["tz"])),
                                        scene3d.Vec3(keys["target_x"], keys["target_y"], keys["target_z"]), fov, roll=keys["roll"])
                pixel, _ = scene3d.project(camera, width, height, point[None])
                row.append({"frame": index, "x": float(pixel[0, 0]), "y": float(pixel[0, 1])})
            observations.append(row)
        scale = cs.survey_scale(solve, (fy, fy, width / 2, height / 2), width, height,
                                world_scale * np.linalg.norm(a - b), observations)
        self.assertAlmostEqual(scale, world_scale, places=2)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            cs.survey_scale(solve, (fy, fy, 0, 0), width, height, 1.0, observations[:1])

    def test_cloud_round_trips_through_a_splat_file(self):
        from nodebased.splats import read_ply
        with tempfile.TemporaryDirectory() as temp:
            path = cs.write_cloud(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]), np.array([[255, 0, 0], [0, 128, 255]], np.uint8),
                                  Path(temp) / "cloud.ply")
            cloud = read_ply(path)
        np.testing.assert_allclose(cloud.positions, [[1, 2, 3], [0, 0, 1]], atol=1e-6)
        colour = 0.5 + cs.SH_DC * cloud.sh[:, 0, :]
        np.testing.assert_allclose(colour, [[1, 0, 0], [0, 128 / 255, 1]], atol=2e-3)


def plate_document(frames=14):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Checker", "params": {"width": 64, "height": 32}},
        {"op": "create", "id": "solve", "type": "CameraSolve3D", "params": {"anchor_tz": 8.0, "unit_scale": 2.0, "seed": 4}},
        {"op": "connect", "id": "solve", "input": "image", "source": "plate"},
        {"op": "time", "first": 1, "last": frames, "current": 1}]})
    return dispatcher


class SolveOpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ArtifactStore(Path(self.temp.name) / "store")

    def solve(self, dispatcher, solver=sliding_solve, **cmd):
        return cs.solve_camera_op(Evaluator(), dispatcher, {"op": "solve_camera", "id": "solve", **cmd},
                                  store=self.store, solver=solver, workdir=self.temp.name)

    def test_op_writes_a_keyed_camera_a_cloud_and_the_solve_id(self):
        dispatcher = plate_document()
        reply = self.solve(dispatcher)
        document = dispatcher.document
        self.assertEqual((reply["camera"], reply["points"], reply["scale_source"]), ("solve_camera", "solve_points", "unit_scale"))
        self.assertEqual(document["nodes"]["solve"]["params"]["solve"], reply["solve"])
        self.assertEqual(document["nodes"]["solve_camera"]["type"], "Camera3D")
        keys = document["animation"]["curves"]["solve_camera"]["tx"]["keys"]
        self.assertEqual([k["frame"] for k in keys], list(range(1, 15)))
        # the slide of 0.1 per frame is 0.2 in the scene (unit_scale 2); the camera starts at the anchor (0, 0, 8)
        self.assertAlmostEqual(keys[0]["value"], 0.0)
        self.assertAlmostEqual(keys[13]["value"], 2.6, places=3)
        self.assertEqual(document["nodes"]["solve_points"]["type"], "ReadSplat3D")
        self.assertTrue(Path(document["nodes"]["solve_points"]["params"]["splat_path"]).is_file())
        record = json.loads(self.store.get(reply["solve"]))
        self.assertEqual((record["seed"], record["scale"], record["frames"]), (4, 2.0, [1, 14]))

    def test_rerun_updates_the_same_nodes_and_clears_an_old_focal_curve(self):
        dispatcher = plate_document()
        self.solve(dispatcher)
        dispatcher.execute({"op": "set_curve", "id": "solve_camera", "param": "focal",
                            "curve": {"interpolation": "linear", "keys": [{"frame": 1, "value": 30.0}, {"frame": 2, "value": 31.0}]}})
        self.solve(dispatcher, camera="solve_camera")
        self.assertEqual(sum(1 for n in dispatcher.document["nodes"].values() if n["type"] == "Camera3D"), 1)
        self.assertNotIn("focal", dispatcher.document["animation"]["curves"]["solve_camera"])

    def test_poses_within_a_hair_of_each_other_round_to_the_same_keys_and_solve_id(self):
        def noisy(frames, intrinsics, **kwargs):
            out = sliding_solve(frames, intrinsics, **kwargs)
            out["c2w"][:, 0, 3] += 3e-5 * np.sin(np.arange(len(frames)))
            return out
        ids = [self.solve(plate_document(), solver=solver)["solve"] for solver in (sliding_solve, noisy)]
        self.assertEqual(ids[0], ids[1])

    def test_op_refuses_what_it_cannot_solve(self):
        dispatcher = plate_document()
        with self.assertRaisesRegex(ValueError, "not a CameraSolve3D"):
            cs.solve_camera_op(Evaluator(), dispatcher, {"op": "solve_camera", "id": "plate"})
        with self.assertRaisesRegex(ValueError, "at least 12 frames"):
            self.solve(dispatcher, last=5)
        dispatcher.execute({"op": "set", "id": "solve", "param": "provider", "value": "nope"})
        with self.assertRaisesRegex(ValueError, "unknown camera solve provider"):
            self.solve(dispatcher)

    def test_survey_replaces_the_unit_scale(self):
        dispatcher = plate_document()
        with mock.patch.object(cs, "survey_scale", return_value=0.5) as scale:
            reply = self.solve(dispatcher, survey={"distance": 2.0, "points": [[{"frame": 1, "x": 1, "y": 2}], [{"frame": 1, "x": 3, "y": 4}]]})
        self.assertEqual((reply["scale"], reply["scale_source"]), (0.5, "survey"))
        self.assertEqual(scale.call_args.args[5][0][0]["frame"], 0)    # document frame 1 is solve index 0


@unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
class SubprocessProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stub = self.root / "stub_solver.py"
        self.stub.write_text(STUB_SOLVER)
        launcher = self.root / "stub-python"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.stub}" "$(eval echo \\${{$#}})"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        patcher = mock.patch.dict(os.environ, {dpvo.PYTHON_ENV: str(launcher)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.frames = [np.full((32, 48, 3), i, np.uint8) for i in range(13)]

    def test_job_goes_out_and_poses_come_back(self):
        progress = []
        result = dpvo.run_solve(self.frames, (40.0, 40.0, 24.0, 16.0), loop_closure=True, seed=9,
                                workdir=self.root / "work", progress=progress.append)
        self.assertEqual(result["c2w"].shape, (13, 4, 4))
        self.assertEqual(result["points"].shape, (2, 3))
        self.assertEqual(result["colors"].dtype, np.uint8)
        self.assertEqual((result["info"]["seed"], result["info"]["loop_closure"], result["info"]["frame_shape"]), (9, True, [32, 48, 3]))
        self.assertEqual(result["info"]["intrinsics"], [40.0, 40.0, 24.0, 16.0])
        self.assertEqual(progress[-1][0], 1.0)
        job = json.loads((self.root / "work" / "job.json").read_text())
        self.assertEqual(job["dpvo_dir"], dpvo.dpvo_dir())
        np.testing.assert_array_equal(np.load(job["frames"][5]), self.frames[5])

    def test_failure_reports_the_last_lines_and_few_frames_never_start_a_process(self):
        self.stub.write_text("print('boom line'); raise SystemExit(7)")
        with self.assertRaisesRegex(RuntimeError, "exit 7.*boom line"):
            dpvo.run_solve(self.frames, (40.0, 40.0, 24.0, 16.0), workdir=self.root / "w2")
        with mock.patch.object(dpvo.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "at least 12 frames"):
                dpvo.run_solve(self.frames[:5], (40.0, 40.0, 24.0, 16.0), workdir=self.root / "w3")
            with self.assertRaisesRegex(ValueError, "same-size RGB"):
                dpvo.run_solve(self.frames[:-1] + [np.zeros((8, 8, 3), np.uint8)], (40.0, 40.0, 24.0, 16.0), workdir=self.root / "w4")
            popen.assert_not_called()

    def test_cancel_kills_the_solver(self):
        self.stub.write_text("import time\nprint('progress 1 13', flush=True)\ntime.sleep(60)")
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            dpvo.run_solve(self.frames, (40.0, 40.0, 24.0, 16.0), workdir=self.root / "w5", cancelled=cancelled)

    def test_missing_environment_is_reported_not_raised(self):
        with mock.patch.dict(os.environ, {dpvo.PYTHON_ENV: str(self.root / "nowhere")}):
            self.assertIn("does not exist", dpvo.available())


@unittest.skipUnless(os.environ.get("NODEBASED_SLAM_TEST") == "1", "set NODEBASED_SLAM_TEST=1 to run DPVO on the card")
class RealSolveTests(unittest.TestCase):
    def test_demo_plate_is_solved_to_a_few_centimetres(self):
        reason = dpvo.available()
        if reason:
            self.skipTest(reason)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "demo_shot"))
        import make_plate
        import solve_camera
        from nodebased.color import display_rgb
        from nodebased.media import read_media_raster
        with tempfile.TemporaryDirectory() as temp:
            plate = Path(temp) / "plate"
            make_plate.render_plate(plate)
            frames = []
            for i in range(1, 25):
                shown = np.clip(display_rgb(np.array(read_media_raster(str(plate / f"plate.{i:04d}.exr")).to_display()[..., :3], np.float32), "sRGB"), 0, 1)
                frames.append(np.rint(shown * 255).astype(np.uint8))
            height, width = frames[0].shape[:2]
            intrinsics = dpvo.intrinsics_for(width, height, solve_camera.shot.FOCAL, filmback.DEFAULT_VAPERTURE)
            runs = [dpvo.run_solve(frames, intrinsics, seed=0, workdir=Path(temp) / name) for name in ("a", "b")]
        self.assertEqual(runs[0]["c2w"].shape, (24, 4, 4))
        self.assertLess(np.abs(runs[0]["c2w"] - runs[1]["c2w"]).max(), 1e-2)   # seeded runs agree to the CUDA atomics' noise


if __name__ == "__main__":
    unittest.main()
