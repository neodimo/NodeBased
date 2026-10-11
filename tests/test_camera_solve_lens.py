"""Rendering 13 step L2: the camera solve recovers the lens, and survey points fix the scale.

All CPU. The solver is a stub that hands over what DPVO hands over (poses, points and the network's tracked
positions) built from a known camera, lens and scene, so the truth is known. The GPU solve of the demo plates is
a recorded run (examples/demo_shot/README.md), not a test.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import camera_solve as cs
from nodebased import filmback, lensmodel, lensrefine as lr, scene3d
from nodebased.artifacts import ArtifactStore
from nodebased.core import CHOICES, SCHEMA_VERSION, SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.providers import camera_solve_dpvo as dpvo

WIDTH, HEIGHT = 1920, 1080
FOCAL_PX = 0.78 * WIDTH            # the true lens of the synthetic shot: about a 30 mm lens on a 36 mm wide back
K1 = -0.04


def synthetic_shot(width=WIDTH, height=HEIGHT, focal=FOCAL_PX, k1=K1, k2=0.0, frames=20, points=450, noise=0.15, seed=1):
    """A dolly and pan past random points, seen through a lens. Returns truth poses (c2w), truth points and the
    tracked observations (each point is followed for 8 to 13 frames, as the network does)."""
    rng = np.random.default_rng(seed)
    c2w = np.tile(np.eye(4), (frames, 1, 1))
    for i in range(frames):
        t = i / (frames - 1)
        yaw, pitch, roll = np.radians(-6 + 12 * t), np.radians(2 * np.sin(2 * np.pi * t)), np.radians(np.sin(3 * t))
        cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        c2w[i, :3, :3] = ry @ rx @ rz
        c2w[i, :3, 3] = [-1.5 + 3.0 * t, 0.1 * np.sin(5 * t), 0.5 * t]
    cloud = np.column_stack((rng.uniform(-6, 6, points), rng.uniform(-2.5, 3, points), rng.uniform(4, 16, points)))
    rotation, translation = lr.world_to_camera(c2w)
    frame, point, xy, weight = [], [], [], []
    for p in range(points):
        birth, life = rng.integers(0, frames - 4), rng.integers(8, 14)
        for i in range(birth, min(frames, birth + life)):
            camera = rotation[i] @ cloud[p] + translation[i]
            if camera[2] < 1:
                continue
            pixel = lr.camera_pixels(camera[None], focal, k1, k2, width, height)[0] + rng.normal(0, noise, 2)
            if 5 < pixel[0] < width - 5 and 5 < pixel[1] < height - 5:
                frame.append(i); point.append(p); xy.append(pixel); weight.append(rng.uniform(0.4, 1.0))
    tracks = {"frame": np.array(frame), "point": np.array(point), "xy": np.array(xy), "weight": np.array(weight)}
    return c2w, cloud, tracks


def slam_start(c2w, cloud, rng, rotation_deg=0.3, translation=0.01, points=0.05):
    """What the solver hands over: close to the truth but not on it, and consistent only with its assumed lens."""
    rotation, trans = lr.world_to_camera(c2w)
    rotation = lr._exp_rotation(rng.normal(0, np.radians(rotation_deg), (len(c2w), 3))) @ rotation
    trans = trans + rng.normal(0, translation, trans.shape)
    return lr.camera_to_world(rotation, trans), cloud + rng.normal(0, points, cloud.shape)


class LensModelAgreementTests(unittest.TestCase):
    def test_camera_pixels_are_the_lens_model_of_the_ideal_pixel(self):
        """A recovered k1 can be typed into Camera3D.lens_k1 because the refinement uses the same lens."""
        rng = np.random.default_rng(2)
        camera = np.column_stack((rng.uniform(-3, 3, 50), rng.uniform(-2, 2, 50), rng.uniform(3, 12, 50)))
        ideal = lr.camera_pixels(camera, FOCAL_PX, 0.0, 0.0, WIDTH, HEIGHT)
        lens = lensmodel.Lens(k1=-0.05, k2=0.01)
        expected = lensmodel.distort_pixels(lens, ideal[:, 0], ideal[:, 1], WIDTH, HEIGHT)
        np.testing.assert_allclose(lr.camera_pixels(camera, FOCAL_PX, -0.05, 0.01, WIDTH, HEIGHT),
                                   np.column_stack(expected), atol=1e-6)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth, cls.cloud, cls.tracks = synthetic_shot()
        cls.start, cls.start_cloud = slam_start(cls.truth, cls.cloud, np.random.default_rng(4))

    def estimate(self, **kwargs):
        return lr.estimate_lens(self.start, self.start_cloud, self.tracks, WIDTH, HEIGHT, FOCAL_PX * 1.15, **kwargs)

    def test_focal_length_is_recovered_within_one_percent_at_1920_wide(self):
        result = self.estimate(estimate_focal=True, distortion="k1")
        self.assertLess(abs(result["focal_px"] / FOCAL_PX - 1.0), 0.01)
        self.assertLess(result["rms_px_after"], 0.5)
        self.assertLess(result["rms_px_after"], result["rms_px_before"])

    def test_k1_is_recovered_within_ten_percent_of_a_known_value(self):
        result = self.estimate(estimate_focal=True, distortion="k1")
        self.assertLess(abs(result["k1"] / K1 - 1.0), 0.10)
        self.assertIsNotNone(result["k1_sigma"])
        self.assertLess(abs(result["k1"] - K1), 5 * result["k1_sigma"] + 1e-3)

    def test_a_typed_focal_is_held_while_k1_is_estimated(self):
        true_focal = lr.estimate_lens(self.start, self.start_cloud, self.tracks, WIDTH, HEIGHT, FOCAL_PX,
                                      estimate_focal=False, distortion="k1")
        self.assertEqual(true_focal["focal_px"], FOCAL_PX)
        self.assertLess(abs(true_focal["k1"] / K1 - 1.0), 0.10)

    def test_nothing_to_estimate_returns_none(self):
        self.assertIsNone(lr.estimate_lens(self.start, self.start_cloud, self.tracks, WIDTH, HEIGHT, FOCAL_PX,
                                           estimate_focal=False, distortion="none"))

    def test_an_undistorted_plate_gives_k1_near_zero(self):
        truth, cloud, tracks = synthetic_shot(k1=0.0, seed=3)
        start, start_cloud = slam_start(truth, cloud, np.random.default_rng(6))
        result = lr.estimate_lens(start, start_cloud, tracks, WIDTH, HEIGHT, FOCAL_PX * 0.9, estimate_focal=True, distortion="k1")
        self.assertLess(abs(result["focal_px"] / FOCAL_PX - 1.0), 0.01)
        self.assertLess(abs(result["k1"]), 0.004)

    def test_the_refinement_is_deterministic(self):
        a, b = self.estimate(estimate_focal=True, distortion="k1"), self.estimate(estimate_focal=True, distortion="k1")
        self.assertEqual((a["focal_px"], a["k1"]), (b["focal_px"], b["k1"]))
        np.testing.assert_array_equal(a["poses"], b["poses"])

    def test_the_path_keeps_the_length_the_solver_gave_it(self):
        result = self.estimate(estimate_focal=True, distortion="k1")
        length = lambda poses: np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()
        self.assertAlmostEqual(length(result["poses"]), length(self.start), places=6)
        np.testing.assert_allclose(result["poses"][0], self.start[0], atol=1e-6)    # the first pose is the gauge

    def test_frames_without_tracks_follow_their_neighbour(self):
        # drop every observation on frame 10: it is no keyframe, so it keeps its motion relative to frame 9
        keep = self.tracks["frame"] != 10
        tracks = {name: value[keep] for name, value in self.tracks.items()}
        result = lr.estimate_lens(self.start, self.start_cloud, tracks, WIDTH, HEIGHT, FOCAL_PX * 1.15,
                                  estimate_focal=True, distortion="k1")
        before = np.linalg.inv(self.start[9]) @ self.start[10]
        after = np.linalg.inv(result["poses"][9]) @ result["poses"][10]
        np.testing.assert_allclose(after, before, atol=1e-6)

    def test_too_little_data_is_refused_in_words(self):
        tracks = {name: value[:30] for name, value in self.tracks.items()}
        with self.assertRaisesRegex(ValueError, "lens estimation needs tracked points on three frames or more"):
            lr.estimate_lens(self.start, self.start_cloud, tracks, WIDTH, HEIGHT, FOCAL_PX, estimate_focal=True)

    def test_unknown_distortion_is_refused(self):
        with self.assertRaisesRegex(ValueError, "distortion must be one of"):
            lr.distortion_terms("k3")

    def test_selection_is_by_count_and_spacing(self):
        keep, frames, points = lr.select_observations(self.tracks["frame"], self.tracks["point"], self.tracks["xy"],
                                                      self.tracks["weight"], len(self.cloud), len(self.truth),
                                                      max_frames=6, max_points=50)
        self.assertEqual(len(frames), 6)
        self.assertEqual((frames[0], frames[-1]), (0, len(self.truth) - 1))
        self.assertLessEqual(len(points), 50)
        self.assertTrue(np.isin(self.tracks["frame"][keep], frames).all())


class SurveyTests(unittest.TestCase):
    def survey(self, **kwargs):
        return {"picks": kwargs.get("picks", []), "distances": kwargs.get("distances", [])}

    def pick(self, name, position=None, frame=1):
        return {"name": name, "frame": frame, "x": 10.0, "y": 20.0, "position": position}

    def test_parsing_validates_in_words(self):
        self.assertEqual(cs.parse_survey(""), {"picks": [], "distances": []})
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            cs.parse_survey("{nope")
        with self.assertRaisesRegex(ValueError, "unique"):
            cs.parse_survey({"picks": [self.pick("A"), self.pick("A")]})
        with self.assertRaisesRegex(ValueError, "three numbers"):
            cs.parse_survey({"picks": [self.pick("A", [1, 2])]})
        with self.assertRaisesRegex(ValueError, "two different picks"):
            cs.parse_survey({"picks": [self.pick("A")], "distances": [{"a": "A", "b": "A", "distance": 1}]})
        with self.assertRaisesRegex(ValueError, "positive"):
            cs.parse_survey({"picks": [self.pick("A"), self.pick("B")], "distances": [{"a": "A", "b": "B", "distance": 0}]})

    def test_fewer_picks_than_needed_leave_the_solve_up_to_scale_and_say_so(self):
        for picks, distances in (([], []), ([self.pick("A")], []), ([self.pick("A"), self.pick("B")], []),
                                 ([self.pick("A", [0, 0, 0]), self.pick("B")], [])):
            plan = cs.survey_plan(self.survey(picks=picks, distances=distances))
            self.assertEqual(plan["mode"], "up_to_scale", picks)
            self.assertIn("Up to scale", plan["message"])
            self.assertIn("a distance between two picks", plan["message"])
        self.assertIn("no picks yet", cs.survey_plan(self.survey())["message"])
        self.assertIn("1 pick)", cs.survey_plan(self.survey(picks=[self.pick("A")]))["message"])

    def test_a_distance_or_two_positions_fix_the_scale_and_three_positions_the_placement(self):
        a, b = self.pick("A"), self.pick("B")
        plan = cs.survey_plan(self.survey(picks=[a, b], distances=[{"a": "A", "b": "B", "distance": 2.0}]))
        self.assertEqual((plan["mode"], plan["pairs"]), ("scale", [("A", "B", 2.0)]))
        plan = cs.survey_plan(self.survey(picks=[self.pick("A", [0, 0, 0]), self.pick("B", [3, 4, 0])]))
        self.assertEqual((plan["mode"], plan["pairs"]), ("scale", [("A", "B", 5.0)]))
        plan = cs.survey_plan(self.survey(picks=[self.pick("A", [0, 0, 0]), self.pick("B", [3, 4, 0]), self.pick("C", [0, 0, 2])]))
        self.assertEqual(plan["mode"], "similarity")
        self.assertIn("Aligned to 3 surveyed points", plan["message"])
        line = cs.survey_plan(self.survey(picks=[self.pick(n, [i, 2 * i, 3 * i]) for i, n in enumerate("ABC")]))
        self.assertEqual(line["mode"], "scale")            # three points on one line cannot fix an orientation
        self.assertIn("lie on one line", line["message"])

    def test_three_known_points_return_the_truth_scale_rotation_and_translation(self):
        rng = np.random.default_rng(8)
        solve = rng.normal(size=(3, 3)) * 2
        angle = np.radians(37)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        scale, translation = 3.7, np.array([1.0, -2.0, 0.5])
        world = scale * solve @ rotation.T + translation
        got_scale, got_rotation, got_translation = cs.similarity_transform(solve, world)
        self.assertAlmostEqual(got_scale, scale, places=9)
        np.testing.assert_allclose(got_rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(got_translation, translation, atol=1e-9)
        # a mirrored survey is never answered with a reflection
        _s, mirrored, _t = cs.similarity_transform(solve, world * [1, 1, -1])
        self.assertAlmostEqual(np.linalg.det(mirrored), 1.0)

    def test_alignment_resolves_a_plan_against_triangulated_picks(self):
        found = {"A": np.array([0.0, 0, 0]), "B": np.array([1.0, 0, 0]), "C": np.array([0.0, 2, 0])}
        positions = {"A": [10.0, 0, 0], "B": [10.0, 3, 0], "C": [10.0, 0, -6]}
        plan = cs.survey_plan(self.survey(picks=[self.pick(n, positions[n]) for n in "ABC"]))
        aligned = cs.survey_alignment(plan, found)
        self.assertAlmostEqual(aligned["scale"], 3.0, places=9)
        self.assertLess(aligned["residual"], 1e-9)
        plan = cs.survey_plan(self.survey(picks=[self.pick("A"), self.pick("B")], distances=[{"a": "A", "b": "B", "distance": 4.0}]))
        self.assertAlmostEqual(cs.survey_alignment(plan, found)["scale"], 4.0, places=9)
        with self.assertRaisesRegex(ValueError, "up to scale"):
            cs.survey_alignment(cs.survey_plan(self.survey()), found)

    def test_the_survey_triangulates_through_the_estimated_lens(self):
        """Picks are plate pixels, so the lens is undone before the rays are intersected."""
        c2w = np.tile(np.eye(4), (5, 1, 1))
        c2w[:, 0, 3] = np.arange(5) * 0.3
        fov = cs.fov_from_intrinsics(HEIGHT, FOCAL_PX)
        truth = np.array([0.4, -0.3, 6.0])
        rows = []
        for index in (0, 2, 4):
            camera = cs.solve_camera_object(c2w[index], fov, lens=(K1, 0.0))
            pixel, _ = scene3d.project_lens(camera, WIDTH, HEIGHT, truth[None])
            rows.append({"frame": index, "x": float(pixel[0, 0]), "y": float(pixel[0, 1])})
        found = cs.triangulate_points(c2w, (FOCAL_PX, FOCAL_PX, WIDTH / 2, HEIGHT / 2), WIDTH, HEIGHT,
                                      [rows, rows[:2]], lens=(K1, 0.0))
        np.testing.assert_allclose(found[0], truth, atol=2e-3)


def plate_document(frames=24, size=(96, 54), **solve_params):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Checker", "params": {"width": size[0], "height": size[1]}},
        {"op": "create", "id": "solve", "type": "CameraSolve3D", "params": {"anchor_tz": 8.0, **solve_params}},
        {"op": "connect", "id": "solve", "input": "image", "source": "plate"},
        {"op": "time", "first": 1, "last": frames, "current": 1}]})
    return dispatcher


class NodeAndOpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ArtifactStore(Path(self.temp.name) / "store")

    def stub(self, c2w, cloud, tracks, size=(96, 54)):
        """A solver that returns a prepared solve, with the lens the node typed left in its poses."""
        def solver(frames, intrinsics, **kwargs):
            return {"c2w": c2w, "points": cloud, "colors": np.full((len(cloud), 3), 200, np.uint8), "tracks": tracks,
                    "info": {"frames": len(frames), "seed": kwargs["seed"]}}
        return solver

    def run_op(self, dispatcher, solver, **cmd):
        return cs.solve_camera_op(Evaluator(), dispatcher, {"op": "solve_camera", "id": "solve", **cmd},
                                  store=self.store, solver=solver, workdir=self.temp.name)

    def test_node_registers_the_lens_and_survey_knobs_and_old_documents_load(self):
        params = SPECS["CameraSolve3D"]["params"]
        self.assertEqual((params["solve_focal"], params["solve_distortion"], params["survey"]), ("known", "none", ""))
        self.assertEqual(CHOICES["solve_focal"], ["known", "estimate"])
        self.assertEqual(CHOICES["solve_distortion"], ["none", "k1", "k1 k2"])
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "solve", "type": "CameraSolve3D"})
        old = dispatcher.document
        for name in ("solve_focal", "solve_distortion", "survey"):
            old["nodes"]["solve"]["params"].pop(name)
        old["version"] = 22
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["nodes"]["solve"]["params"]["solve_distortion"], "none")
        validate(upgraded)

    def test_known_lens_leaves_the_solve_as_it_was(self):
        truth, cloud, tracks = synthetic_shot(frames=24)
        reply = self.run_op(plate_document(), self.stub(truth, cloud, tracks))
        self.assertIsNone(reply["lens"])
        self.assertEqual(reply["alignment"]["mode"], "up_to_scale")
        record = json.loads(self.store.get(reply["solve"]))
        self.assertNotIn("lens", record)
        self.assertIn("Up to scale", record["alignment"]["message"])

    def test_estimating_the_lens_writes_the_focal_and_k1_to_the_camera_and_the_solve(self):
        truth, cloud, tracks = synthetic_shot(frames=24)
        start, start_cloud = slam_start(truth, cloud, np.random.default_rng(4))
        typed_mm = FOCAL_PX * 1.15 * filmback.DEFAULT_VAPERTURE / HEIGHT
        dispatcher = plate_document(solve_focal="estimate", solve_distortion="k1", focal=typed_mm)
        # the plate the stub "solves" is 96 x 54; the stub's tracks are at 1920 x 1080, so give it the bigger plate
        dispatcher.execute({"op": "set", "id": "plate", "param": "width", "value": WIDTH})
        dispatcher.execute({"op": "set", "id": "plate", "param": "height", "value": HEIGHT})
        dispatcher.execute({"op": "set", "id": "solve", "param": "unit_scale", "value": 2.0})
        reply = self.run_op(dispatcher, self.stub(start, start_cloud, tracks))
        lens = reply["lens"]
        true_mm = FOCAL_PX * filmback.DEFAULT_VAPERTURE / HEIGHT
        self.assertLess(abs(lens["focal_mm"] / true_mm - 1.0), 0.01)
        self.assertLess(abs(lens["k1"] / K1 - 1.0), 0.10)
        self.assertGreater(lens["rms_px_before"], lens["rms_px_after"])
        camera = dispatcher.document["nodes"][reply["camera"]]["params"]
        self.assertEqual((camera["focal"], camera["lens_k1"]), (lens["focal_mm"], lens["k1"]))
        record = json.loads(self.store.get(reply["solve"]))
        self.assertEqual(record["lens"], lens)
        self.assertEqual(record["lens"]["focal_typed_mm"], round(typed_mm, 4))
        self.assertAlmostEqual(record["intrinsics_px"][0], lens["focal_px"], places=2)
        self.assertEqual(dispatcher.document["nodes"]["solve_points"]["type"], "ReadSplat3D")
        # the panel says it in words
        text = "\n".join(cs.panel_lines(dispatcher.document["nodes"]["solve"]["params"], record))
        self.assertIn(f"focal {lens['focal_mm']:.3f} mm estimated (typed {typed_mm:.3f} mm", text)
        self.assertIn("k1 ", text)
        self.assertIn("Reprojection error", text)
        self.assertIn("Up to scale", text)
        # a rerun with a known lens clears the distortion again
        dispatcher.execute({"op": "set", "id": "solve", "param": "solve_distortion", "value": "none"})
        dispatcher.execute({"op": "set", "id": "solve", "param": "solve_focal", "value": "known"})
        self.run_op(dispatcher, self.stub(start, start_cloud, tracks))
        self.assertEqual(dispatcher.document["nodes"][reply["camera"]]["params"]["lens_k1"], 0.0)

    def test_a_solver_without_tracks_cannot_estimate_a_lens(self):
        truth, cloud, _tracks = synthetic_shot(frames=24)
        dispatcher = plate_document(solve_focal="estimate")
        solver = lambda frames, intrinsics, **kw: {"c2w": truth, "points": cloud, "colors": np.zeros((len(cloud), 3), np.uint8)}
        with self.assertRaisesRegex(ValueError, "no tracked points"):
            self.run_op(dispatcher, solver)

    def test_the_survey_places_the_solve_in_the_world_and_a_short_survey_says_up_to_scale(self):
        truth, cloud, tracks = synthetic_shot(frames=24, points=60)
        solve_points = {"A": np.array([0.0, 0.0, 6.0]), "B": np.array([1.0, 0.0, 6.0]), "C": np.array([0.0, 1.0, 7.0])}
        scale, angle, shift = 2.5, np.radians(20), np.array([5.0, 1.0, -3.0])
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        world = {name: scale * rotation @ p + shift for name, p in solve_points.items()}
        picks = [{"name": n, "frame": 1, "x": 5.0, "y": 5.0, "position": [float(v) for v in world[n]]} for n in "ABC"]
        dispatcher = plate_document(survey=json.dumps({"picks": picks, "distances": []}))

        def fake_track(frames, pick, first, last):
            fov = cs.fov_from_intrinsics(54, dpvo.focal_pixels(filmback.DEFAULT_FOCAL, filmback.DEFAULT_VAPERTURE, 54))
            rows = []
            for index in (0, 6, 12, 18):
                camera = cs.solve_camera_object(truth[index], fov)
                pixel, _ = scene3d.project(camera, 96, 54, solve_points[pick["name"]][None])
                rows.append({"frame": index, "x": float(pixel[0, 0]), "y": float(pixel[0, 1])})
            return rows
        solver = self.stub(truth, cloud, tracks)
        with mock.patch.object(cs, "track_pick", fake_track):
            reply = self.run_op(dispatcher, solver)
        self.assertEqual(reply["alignment"]["mode"], "similarity")
        self.assertAlmostEqual(reply["scale"], scale, places=2)
        self.assertLess(reply["alignment"]["residual"], 0.02)
        keys = dispatcher.document["animation"]["curves"]["solve_camera"]["tx"]["keys"]
        expected = scale * rotation @ truth[0, :3, 3] + shift
        self.assertAlmostEqual(keys[0]["value"], expected[0], places=2)       # the survey, not the anchor, places frame 1
        # one pick on its own leaves the solve up to scale
        short = plate_document(survey=json.dumps({"picks": picks[:1], "distances": []}))
        reply = self.run_op(short, solver)
        self.assertEqual((reply["alignment"]["mode"], reply["scale_source"]), ("up_to_scale", "unit_scale"))
        self.assertIn("Up to scale (1 pick)", reply["alignment"]["message"])


class TrackedPickTests(unittest.TestCase):
    def test_a_pick_is_followed_through_the_plate(self):
        """A bright dot that drifts two pixels a frame is found on every frame, so the pick has observations to triangulate."""
        frames = []
        for i in range(8):
            image = np.zeros((60, 80, 3), np.uint8)
            rng = np.random.default_rng(1)
            image[:] = rng.integers(0, 40, (60, 80, 1), dtype=np.uint8)
            image[28:33, 28 + 2 * i:33 + 2 * i] = 255
            image[30, 25 + 2 * i:36 + 2 * i] = 0
            frames.append(image)
        rows = cs.track_pick(frames, {"name": "A", "frame": 1, "x": 30.5, "y": 30.5}, 1, 8)
        self.assertGreaterEqual(len(rows), 6)
        by_frame = {row["frame"]: row for row in rows}
        self.assertAlmostEqual(by_frame[7]["x"] - by_frame[0]["x"], 14.0, delta=1.0)
        with self.assertRaisesRegex(ValueError, "outside the solve"):
            cs.track_pick(frames, {"name": "A", "frame": 20, "x": 30.0, "y": 30.0}, 1, 8)


if __name__ == "__main__":
    unittest.main()
