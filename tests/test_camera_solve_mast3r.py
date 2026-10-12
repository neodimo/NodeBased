"""Stage 9 of the end-to-end sprint: MASt3R-SLAM as the second camera solver behind CameraSolve3D.

The CPU tests run without torch (stub solver function or stub interpreter). The GPU run (real MASt3R-SLAM on the
demo plate at its smallest supported size, 512 x 288) is opt-in: NODEBASED_SLAM_TEST=1, skipped where the
MASt3R environment or a CUDA device is missing.
"""
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import camera_solve as cs
from nodebased import filmback
from nodebased.artifacts import ArtifactStore
from nodebased.imaging import Evaluator
from nodebased.providers import camera_solve_dpvo as dpvo
from nodebased.providers import camera_solve_mast3r as mast3r
from tests.test_camera_solve import plate_document, sliding_solve

SOLVED_FOCAL_PX = 40.0       # what the stub solver "recovers" on the 64 x 32 checker plate

STUB_SOLVER = '''
import json, sys
import numpy as np
job = json.load(open(sys.argv[-1]))
n = len(job["frames"])
c2w = np.tile(np.eye(4), (n, 1, 1))
c2w[:, 0, 3] = np.arange(n) * 0.1
np.savez(job["result"], c2w=c2w, points=np.array([[0, 0, 5.0]]), colors=np.array([[255, 0, 0]], np.uint8))
json.dump({"frames": n, "seed": job["seed"], "loop_closure": job["loop_closure"], "focal_px": job.get("focal", 123.5),
           "mast3r_dir": job["mast3r_dir"]}, open(job["info"], "w"))
for i in range(n):
    print("progress", i + 1, n, flush=True)
'''


def lens_solve(frames, intrinsics, **kwargs):
    """Stands in for MASt3R-SLAM: the sliding camera, plus a recovered lens."""
    out = sliding_solve(frames, intrinsics, **kwargs)
    height, width = frames[0].shape[:2]
    out["intrinsics"] = (SOLVED_FOCAL_PX, SOLVED_FOCAL_PX, width / 2.0, height / 2.0)
    return out


def pinhole_pointmap(width, height, focal, rng, outliers=0.0):
    """A camera-frame point map of random depths seen by a pinhole of `focal` pixels, centre principal point."""
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    z = rng.uniform(2.0, 9.0, size=(height, width))
    x = (u - (width - 1) / 2.0) / focal * z
    y = (v - (height - 1) / 2.0) / focal * z
    points = np.stack([x, y, z], -1).reshape(-1, 3)
    bad = rng.random(len(points)) < outliers
    points[bad] *= rng.uniform(0.5, 1.5, size=(bad.sum(), 3))
    return points


class SolverChoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ArtifactStore(Path(self.temp.name) / "store")

    def solve(self, dispatcher, provider, solver, **cmd):
        dispatcher.execute({"op": "set", "id": "solve", "param": "provider", "value": provider})
        return cs.solve_camera_op(Evaluator(), dispatcher, {"op": "solve_camera", "id": "solve", **cmd},
                                  store=self.store, solver=solver, workdir=self.temp.name)

    def test_both_solvers_are_offered_and_dpvo_stays_the_default(self):
        self.assertEqual(set(cs.SOLVERS), {dpvo.PROVIDER_NAME, mast3r.PROVIDER_NAME})
        from nodebased.core import SPECS
        self.assertEqual(SPECS["CameraSolve3D"]["params"]["provider"], dpvo.PROVIDER_NAME)

    def test_mast3r_solve_reports_itself_and_keys_the_recovered_lens(self):
        dispatcher = plate_document()
        reply = self.solve(dispatcher, mast3r.PROVIDER_NAME, lens_solve)
        self.assertEqual((reply["provider"], reply["focal_source"]), (mast3r.PROVIDER_NAME, "solved"))
        # 40 px on a 32 px tall plate through the node's 18.672 mm film back
        expected = round(SOLVED_FOCAL_PX * filmback.DEFAULT_VAPERTURE / 32, 3)
        self.assertAlmostEqual(reply["focal"], expected)
        camera = dispatcher.document["nodes"]["solve_camera"]["params"]
        self.assertAlmostEqual(camera["focal"], expected)
        self.assertEqual(camera["vaperture"], filmback.DEFAULT_VAPERTURE)
        record = json.loads(self.store.get(reply["solve"]))
        self.assertEqual((record["provider"], record["focal"], record["focal_source"]), (mast3r.PROVIDER_NAME, expected, "solved"))
        self.assertAlmostEqual(record["intrinsics_px"][1], SOLVED_FOCAL_PX, places=2)

    def test_the_node_focal_is_ignored_by_mast3r_and_used_by_dpvo(self):
        dispatcher = plate_document()
        dispatcher.execute({"op": "set", "id": "solve", "param": "focal", "value": 85.0})
        dpvo_reply = self.solve(dispatcher, dpvo.PROVIDER_NAME, sliding_solve)
        self.assertEqual((dpvo_reply["provider"], dpvo_reply["focal"], dpvo_reply["focal_source"]), (dpvo.PROVIDER_NAME, 85.0, "node"))
        self.assertNotIn("focal_source", json.loads(self.store.get(dpvo_reply["solve"])))   # stage 7 records unchanged
        mast3r_reply = self.solve(dispatcher, mast3r.PROVIDER_NAME, lens_solve)
        self.assertNotEqual(mast3r_reply["focal"], 85.0)
        self.assertEqual(dispatcher.document["nodes"]["solve_camera"]["params"]["focal"], mast3r_reply["focal"])

    def test_the_solve_identity_covers_the_solver_choice(self):
        def same_poses_no_lens_change(frames, intrinsics, **kwargs):
            out = sliding_solve(frames, intrinsics, **kwargs)
            out["intrinsics"] = intrinsics       # MASt3R "recovers" exactly the node's lens: only the solver differs
            return out
        ids = {provider: self.solve(plate_document(), provider, solver)["solve"]
               for provider, solver in ((dpvo.PROVIDER_NAME, sliding_solve), (mast3r.PROVIDER_NAME, same_poses_no_lens_change))}
        self.assertNotEqual(ids[dpvo.PROVIDER_NAME], ids[mast3r.PROVIDER_NAME])
        again = self.solve(plate_document(), mast3r.PROVIDER_NAME, same_poses_no_lens_change)["solve"]
        self.assertEqual(again, ids[mast3r.PROVIDER_NAME])

    def test_each_solver_keeps_its_own_frame_minimum(self):
        dispatcher = plate_document(frames=10)
        with self.assertRaisesRegex(ValueError, "at least 12 frames"):
            self.solve(dispatcher, dpvo.PROVIDER_NAME, sliding_solve)
        self.assertEqual(self.solve(dispatcher, mast3r.PROVIDER_NAME, lens_solve)["frames"], [1, 10])
        with self.assertRaisesRegex(ValueError, "at least 8 frames"):
            self.solve(dispatcher, mast3r.PROVIDER_NAME, lens_solve, last=5)
        with self.assertRaisesRegex(ValueError, "camera_solve_dpvo, camera_solve_mast3r"):
            self.solve(dispatcher, "camera_solve_droid", sliding_solve)

    def test_survey_uses_the_recovered_lens(self):
        with mock.patch.object(cs, "survey_scale", return_value=0.5) as scale:
            self.solve(plate_document(), mast3r.PROVIDER_NAME, lens_solve,
                       survey={"distance": 2.0, "points": [[{"frame": 1, "x": 1, "y": 2}], [{"frame": 1, "x": 3, "y": 4}]]})
        self.assertAlmostEqual(scale.call_args.args[1][1], SOLVED_FOCAL_PX, places=2)


class LensFitTests(unittest.TestCase):
    def test_focal_of_a_pinhole_pointmap_is_recovered(self):
        rng = np.random.default_rng(1)
        for focal in (250.0, 400.0, 700.0):
            points = pinhole_pointmap(128, 72, focal, rng, outliers=0.2)
            self.assertAlmostEqual(mast3r.focal_from_pointmap(points, np.full(len(points), 3.0), 128, 72) / focal, 1.0, places=3)

    def test_unconfident_or_behind_pixels_give_no_focal(self):
        rng = np.random.default_rng(2)
        points = pinhole_pointmap(32, 16, 100.0, rng)
        self.assertTrue(np.isnan(mast3r.focal_from_pointmap(points, np.ones(len(points)), 32, 16)))
        points[:, 2] *= -1
        self.assertTrue(np.isnan(mast3r.focal_from_pointmap(points, np.full(len(points), 3.0), 32, 16)))

    def test_working_size_is_upstreams_512_frame(self):
        self.assertEqual(mast3r.working_size(960, 540), (512, 288, 1.875))
        self.assertEqual(mast3r.working_size(1920, 1080), (512, 288, 3.75))
        self.assertEqual(mast3r.working_size(512, 288), (512, 288, 1.0))
        self.assertEqual(mast3r.working_size(480, 270), (512, 288, 0.9375))

    def test_provider_description(self):
        description = mast3r.description()
        self.assertEqual((description["name"], description["kind"], description["local_gpu"]),
                         (mast3r.PROVIDER_NAME, "camera_solve", True))
        self.assertFalse(description["accepts"]["intrinsics"]["accepted"])
        self.assertEqual(description["limits"]["min_frames"], mast3r.MIN_FRAMES)
        self.assertNotIn("/home/", (mast3r.SOLVER_DIR / f"{mast3r.PROVIDER_NAME}.json").read_text())
        from nodebased.generative import PROVIDER_DIR
        self.assertNotIn(mast3r.PROVIDER_NAME, {path.stem for path in PROVIDER_DIR.glob("*.json")})


def dolly_segment(rng, focal=350.0, width=512, height=288, points=500, frames=16, pose_noise=0.0):
    """A keyframe and the frames tracked against it on a short dolly with a slow pan, matches to a whole pixel
    (as MASt3R gives them) with a tenth of them wrong. Returns (segment, true (R, t) per frame)."""
    centre = np.array([(width - 1) / 2, (height - 1) / 2])
    pixels = np.column_stack([rng.uniform(0, width, points), rng.uniform(0, height, points)])
    depth = rng.uniform(3, 12, points)
    world = np.column_stack([(pixels - centre) / focal * depth[:, None], depth])
    rows, truth = [], []
    for j in range(1, frames + 1):
        a = j / frames
        R = mast3r._rotation(np.array([[0.01 * np.sin(j), 0.06 * a, 0.004 * np.cos(j)]]))[0]
        t = -R @ np.array([0.4 * a, 0.02 * np.sin(j), -1.4 * a])
        seen = world @ R.T + t
        observed = focal * seen[:, :2] / seen[:, 2:3] + centre
        inside = np.flatnonzero((observed[:, 0] >= 0) & (observed[:, 0] < width) & (observed[:, 1] >= 0) & (observed[:, 1] < height))
        observed = np.rint(observed[inside])
        wrong = rng.random(len(inside)) < 0.1
        observed[wrong] += rng.normal(0, 20, (wrong.sum(), 2))
        R0 = R @ mast3r._rotation(rng.normal(0, pose_noise * 0.1, (1, 3)))[0]
        rows.append({"frame": j, "R": R0, "t": t + rng.normal(0, pose_noise, 3), "index": inside, "observed": observed})
        truth.append((R, t))
    return {"pixels": pixels, "inv_depth": 1 / (depth * rng.uniform(0.95, 1.05, points)), "frames": rows}, truth


class BundleAdjustTests(unittest.TestCase):
    def test_free_focal_is_found_from_exact_geometry(self):
        segment, _truth = dolly_segment(np.random.default_rng(0))
        for start in (1.12, 0.9):
            out = mast3r.bundle_adjust([segment], 350.0 * start, 512, 288, free_focal=True)
            self.assertAlmostEqual(out["focal"] / 350.0, 1.0, places=2)
            self.assertLess(out["rms_pixels"], 0.6)          # whole-pixel matches: about 0.4 px

    def test_held_focal_stays_and_the_poses_come_back(self):
        segment, truth = dolly_segment(np.random.default_rng(1), pose_noise=0.03)
        before = np.mean([np.linalg.norm(f["t"] - t) for f, (_R, t) in zip(segment["frames"], truth)])
        out = mast3r.bundle_adjust([segment], 350.0, 512, 288)
        self.assertEqual(out["focal"], 350.0)
        after = np.mean([np.linalg.norm(-R.T @ t - (-Rt.T @ tt)) for (R, t), (Rt, tt) in zip(out["frames"][0], truth)])
        self.assertLess(after, before / 5)
        self.assertEqual(len(out["frames"][0]), len(segment["frames"]))

    def test_nothing_to_adjust(self):
        out = mast3r.bundle_adjust([], 300.0, 512, 288)
        self.assertEqual((out["focal"], out["frames"], out["observations"]), (300.0, [], 0))

    def test_sim3_data_to_rotation(self):
        half = np.radians(30) / 2          # 30 degrees about z, lietorch order [t, qx qy qz qw, s]
        t, R, scale = mast3r._sim3([1, 2, 3, 0, 0, np.sin(half), np.cos(half), 2.0])
        np.testing.assert_allclose(R, [[np.cos(2 * half), -np.sin(2 * half), 0], [np.sin(2 * half), np.cos(2 * half), 0], [0, 0, 1]], atol=1e-12)
        self.assertEqual((list(t), scale), ([1, 2, 3], 2.0))
        np.testing.assert_allclose(mast3r._rotation(np.array([[0, 0, 2 * half]]))[0], R, atol=1e-12)


@unittest.skipIf(sys.platform == "win32", "the stub interpreter is a shell script")
class SubprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stub = self.root / "stub_solver.py"
        self.stub.write_text(STUB_SOLVER)
        launcher = self.root / "stub-python"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.stub}" "$(eval echo \\${{$#}})"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        patcher = mock.patch.dict(os.environ, {mast3r.PYTHON_ENV: str(launcher), mast3r.DIR_ENV: str(self.root / "m3")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.frames = [np.full((36, 64, 3), i, np.uint8) for i in range(9)]

    def test_job_goes_out_and_poses_and_the_lens_come_back(self):
        progress = []
        result = mast3r.run_solve(self.frames, None, loop_closure=True, seed=3, workdir=self.root / "work",
                                  progress=progress.append)
        self.assertEqual(result["c2w"].shape, (9, 4, 4))
        self.assertEqual(result["intrinsics"], (123.5, 123.5, 32.0, 18.0))
        self.assertEqual((result["info"]["seed"], result["info"]["loop_closure"], result["info"]["mast3r_dir"]),
                         (3, True, str(self.root / "m3")))
        self.assertEqual(progress[-1][0], 1.0)

    def test_no_lens_is_a_failure(self):
        self.stub.write_text(STUB_SOLVER.replace('job.get("focal", 123.5)', 'float("nan")'))
        with self.assertRaisesRegex(RuntimeError, "recovered no focal length"):
            mast3r.run_solve(self.frames, None, workdir=self.root / "w2")

    def test_missing_environment_is_reported_not_raised(self):
        with mock.patch.dict(os.environ, {mast3r.PYTHON_ENV: str(self.root / "nowhere")}):
            self.assertIn("does not exist", mast3r.available())


@unittest.skipUnless(os.environ.get("NODEBASED_SLAM_TEST") == "1", "set NODEBASED_SLAM_TEST=1 to run MASt3R-SLAM on the card")
class RealSolveTests(unittest.TestCase):
    FOCAL_TOLERANCE = 0.05     # the recovered focal within 5% of the true one at 512 x 288 (measured: see the sprint doc)

    def test_demo_plate_lens_is_recovered_at_the_smallest_size(self):
        reason = mast3r.available()
        if reason:
            self.skipTest(reason)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "demo_shot"))
        import make_plate
        import shot
        from nodebased.color import display_rgb
        from nodebased.media import read_media_raster
        with tempfile.TemporaryDirectory() as temp:
            plate = Path(temp) / "plate"
            make_plate.render_plate(plate, size=(512, 288))
            frames = []
            for i in range(1, shot.FRAMES + 1):
                shown = np.clip(display_rgb(np.array(read_media_raster(str(plate / f"plate.{i:04d}.exr")).to_display()[..., :3], np.float32), "sRGB"), 0, 1)
                frames.append(np.rint(shown * 255).astype(np.uint8))
            result = mast3r.run_solve(frames, None, seed=0, workdir=Path(temp) / "run")
        true_focal = dpvo.focal_pixels(shot.FOCAL, filmback.DEFAULT_VAPERTURE, 288)
        self.assertEqual(result["c2w"].shape, (shot.FRAMES, 4, 4))
        self.assertLess(abs(result["intrinsics"][1] / true_focal - 1), self.FOCAL_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
