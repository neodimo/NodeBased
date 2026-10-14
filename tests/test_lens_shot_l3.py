"""Rendering 13 step L3: the distorted plate through the demo shot, end to end (examples/demo_shot, `--plate distorted`).

Everything runs on the CPU at 192 by 108 and four frames; the full-size numbers are in the example's README. The graph under
test is built with the true camera and lens; the committed `demo_shot_lens.nbcomp` holds the camera the landmark solve
recovered (LandmarkSolveTests), and the DPV-SLAM solve, which needs the card, is kept apart."""
import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import lensmodel
from nodebased.core import SPECS
from nodebased.imaging import Evaluator

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo_shot"
sys.path.insert(0, str(DEMO))
import compare_runs  # noqa: E402
import lens_check  # noqa: E402
import make_plate  # noqa: E402
import run_by_hand  # noqa: E402
import shot  # noqa: E402
import solve_camera  # noqa: E402

SIZE, FRAMES = (192, 108), 4
LENS = shot.lens_of(shot.DISTORTED_K1, shot.DISTORTED_K2)


def truth_keys(frames=shot.FRAMES):
    return {f: {**shot.camera_at(f), "roll": 0.0, "focal": shot.FOCAL} for f in range(1, frames + 1)}


def lens_document(plate_folder, lens=LENS):
    """The lens shot graph with the true camera and lens, its Read node on `plate_folder`."""
    margins = shot.overscan_margins(lens, SIZE)
    document = shot.build_lens_graph(SIZE, truth_keys(), lens, margins)
    document["nodes"]["plate"]["params"]["path"] = shot.plate_pattern(plate_folder)
    return document, margins


class LensShotBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.plate = cls.root / "plate_distorted"
        cls.hashes = make_plate.render_plate(cls.plate, SIZE, FRAMES, k1=shot.DISTORTED_K1, k2=shot.DISTORTED_K2)
        cls.document, cls.margins = lens_document(cls.plate)
        cls.graph = cls.root / "demo_shot_lens.nbcomp"
        shot.save_lens_graph(cls.graph, SIZE, truth_keys(), LENS, cls.margins, plate_folder=cls.plate)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()


class DistortedPlateTests(LensShotBase):
    def test_the_plate_is_repeatable_and_has_both_radial_terms(self):
        again = make_plate.render_plate(self.root / "again", SIZE, FRAMES, k1=shot.DISTORTED_K1, k2=shot.DISTORTED_K2)
        self.assertEqual(again, self.hashes)
        k1_only = make_plate.render_plate(self.root / "k1", SIZE, 1, k1=shot.DISTORTED_K1)
        self.assertNotEqual(k1_only[1], self.hashes[1])            # the second term bends the plate too

    def test_the_committed_hashes_describe_the_distorted_plate(self):
        recorded = json.loads((DEMO / "plate_distorted.sha256.json").read_text())
        self.assertEqual((recorded["size"], recorded["k1"], recorded["k2"]),
                         (list(shot.DISTORTED_SIZE), shot.DISTORTED_K1, shot.DISTORTED_K2))
        self.assertEqual(sorted(map(int, recorded["frames"])), list(range(1, shot.FRAMES + 1)))
        self.assertEqual(make_plate.check({1: "0" * 64}, recorded["frames"]), [1])

    def test_the_committed_pieces_are_there_and_hold_no_frames_or_home_paths(self):
        names = {path.name for path in DEMO.iterdir() if path.is_file()}
        for name in ("demo_shot_lens.nbcomp", "solve_lens.json", "plate_distorted.sha256.json", "lens_check.py"):
            self.assertIn(name, names)
        self.assertFalse([n for n in names if n.endswith(".exr")])
        for name in ("demo_shot_lens.nbcomp", "solve_lens.json"):
            self.assertNotIn("/home/", (DEMO / name).read_text(encoding="utf-8"))


class RoundTripTests(LensShotBase):
    def test_undistort_then_redistort_the_plate_is_the_identity_away_from_the_edge(self):
        evaluator = Evaluator()
        scratch = json.loads(json.dumps(self.document))
        scratch["nodes"]["_back"] = {"type": "Redistort", "name": "back", "pos": [0, 0], "disabled": False,
                                     "params": dict(scratch["nodes"]["redistort"]["params"]),
                                     "inputs": {"image": "undistort", "camera": "camera"}}
        plate = np.asarray(evaluator.evaluate_raster(scratch, "plate", frame=2).to_display(), np.float32)
        back = np.asarray(evaluator.evaluate_raster(scratch, "_back", frame=2).to_display(), np.float32)
        band = 6
        error = np.abs(back - plate)[band:-band, band:-band, :3]
        # two bilinear resamples soften the plate's sharp edges; at this small size they are a larger share of the picture
        self.assertLess(float(error.mean()), 0.01)
        self.assertLess(float(np.percentile(error, 99)), 0.15)

    def test_the_overscan_render_redistorted_is_the_distorted_plate(self):
        """Feed the shot's own overscan render where the generated picture goes: the graph's output is the plate."""
        evaluator = Evaluator()
        scratch = json.loads(json.dumps(self.document))
        scratch["nodes"]["fit"]["inputs"]["image"] = "render"
        undistort = evaluator.evaluate_raster(scratch, "undistort", frame=3)
        render = evaluator.evaluate_raster(scratch, "render", frame=3)
        self.assertEqual((render.display.width, render.display.height),
                         (SIZE[0] + 2 * self.margins[0], SIZE[1] + 2 * self.margins[1]))
        # the render covers what Undistort grew to, so Redistort reads real pixels everywhere
        self.assertTrue(undistort.data.x >= -self.margins[0] and undistort.data.y >= -self.margins[1])
        out = np.asarray(evaluator.evaluate_raster(scratch, "redistort", frame=3).to_display(), np.float32)[..., :3]
        plate = lens_check.read_display(self.plate / "plate.0003.exr")
        self.assertEqual(out.shape, plate.shape)
        error = np.abs(out - plate)[4:-4, 4:-4]
        self.assertLess(float(error.mean()), 0.003)
        self.assertGreater(float(plate[0, 0].sum() + plate[-1, -1].sum()), 0.1)      # the corners hold the set, not black

    def test_the_overscan_camera_is_the_same_physical_lens_on_a_larger_film_back(self):
        film, scale = shot.overscan_camera_params(LENS, SIZE, self.margins)
        self.assertGreater(scale, 1.0)
        plate_lens, big_lens = lensmodel.from_params(LENS), lensmodel.from_params(
            {k: v for k, v in film.items() if k.startswith("lens_")})
        big = (SIZE[0] + 2 * self.margins[0], SIZE[1] + 2 * self.margins[1])
        # an ideal pixel `offset` from the centre lands at the same offset from the centre on both formats
        offsets = np.array([[10.0, 5.0], [70.0, 40.0], [-90.0, 52.0], [30.0, -48.0]])
        a = lensmodel.distort_pixels(plate_lens, offsets[:, 0] + SIZE[0] / 2, offsets[:, 1] + SIZE[1] / 2, *SIZE)
        b = lensmodel.distort_pixels(big_lens, offsets[:, 0] + big[0] / 2, offsets[:, 1] + big[1] / 2, *big)
        np.testing.assert_allclose(np.array(a[0]) - SIZE[0] / 2, np.array(b[0]) - big[0] / 2, atol=1e-6)
        np.testing.assert_allclose(np.array(a[1]) - SIZE[1] / 2, np.array(b[1]) - big[1] / 2, atol=1e-6)


class ExportCarriesTheLensTests(LensShotBase):
    def test_the_scene_state_has_the_lens_and_the_overscan_resolution(self):
        from nodebased.geoexport import export_obj
        document = json.loads(json.dumps(self.document))
        target = self.root / "export" / "shot.scene.json"
        document["nodes"]["export"]["params"]["geo_write_path"] = str(target)
        export_obj(document, "export", range(1, 3), str(target), Evaluator())
        written = json.loads(target.read_text())
        frame = written["frames"][0]
        lens = frame["camera_data"]["lens"]
        film, scale = shot.overscan_camera_params(LENS, SIZE, self.margins)
        self.assertTrue(lens["distorted"])
        self.assertAlmostEqual(lens["k1"], film["lens_k1"], places=9)
        self.assertAlmostEqual(lens["k2"], film["lens_k2"], places=9)
        self.assertEqual(frame["resolution"], [SIZE[0] + 2 * self.margins[0], SIZE[1] + 2 * self.margins[1]])
        # the plate's own lens is recoverable from the export: scale back by the film-back ratio
        self.assertAlmostEqual(lens["k1"] / scale ** 2, shot.DISTORTED_K1, places=9)
        self.assertAlmostEqual(lens["k2"] / scale ** 4, shot.DISTORTED_K2, places=9)


class ShotRunTests(LensShotBase):
    def test_hand_run_gives_a_passing_verdict_and_a_round_trip_error_of_its_own(self):
        summary = run_by_hand.run(self.plate, self.root / "out-hand", self.root / "store-hand", self.graph,
                                  frames=FRAMES, log=lambda *_: None)
        self.assertEqual(summary["verdict"], "PASS")
        block = summary["lens_round_trip"]
        self.assertIn("plate_identity", block)
        self.assertIn("result_vs_plate", block)
        self.assertEqual(block["judged"], ["plate_identity", "result_vs_plate"])
        self.assertEqual(block["limits"], lens_check.LIMITS)
        # the committed limits are for the shot at 640 by 360; this one is under a third as wide and its edges are softer
        small = {"plate_identity": {"mean": 0.012, "p99": 0.15}, "result_vs_plate": {"mean": 0.012, "p99": 0.15}}
        judged = lens_check.measure(self.graph, self.plate, self.root / "out-hand", FRAMES, limits=small)
        self.assertTrue(judged["pass"], judged)
        self.assertFalse(lens_check.measure(self.graph, self.plate, self.root / "out-hand", FRAMES,
                                            limits={"plate_identity": {"mean": 1e-9, "p99": 1e-9},
                                                    "result_vs_plate": {"mean": 1.0, "p99": 1.0}})["pass"])
        # a model that restyles the picture is not judged on the pixel row
        model = lens_check.measure(self.graph, self.plate, self.root / "out-hand", FRAMES, judged_result=False)
        self.assertEqual(model["judged"], ["plate_identity"])

    def test_hand_and_agent_runs_match_on_the_distorted_plate(self):
        logged = []
        hand, agent, rows = compare_runs.run(self.plate, self.root / "compare", self.graph, frames=FRAMES, log=logged.append)
        labels = [label for label, *_ in rows]
        self.assertEqual(labels, ["SceneState", "ControlBundle", "Generated sequence", "Verification report", "Verdict",
                                  "Determinism", "Written output", "Lens round trip"])
        for label, _kind, a, b, same in rows:
            self.assertTrue(same, f"{label}: hand {a} agent {b}")
        for key in ("scene_state", "control_bundle", "generated_sequence", "report"):
            self.assertRegex(agent[key], r"^[0-9a-f]{64}$")
        order = list(dict.fromkeys(agent["tool_sequence"]))
        self.assertEqual(order, ["knowledge", "describe", "inspect", "view", "edit", "export_conditioning", "generate",
                                 "verify", "artifacts"])
        self.assertEqual(agent["verdict"], "PASS")
        self.assertTrue(any(line.startswith("lens: k1") for line in logged))


class GraphTests(LensShotBase):
    def test_the_graph_has_the_round_trip_in_order_and_the_solve_with_two_survey_picks(self):
        nodes = self.document["nodes"]
        def wired(key):
            return {slot: source for slot, source in nodes[key]["inputs"].items() if source}
        self.assertEqual(wired("undistort"), {"image": "plate", "camera": "camera"})
        self.assertEqual(nodes["generate"]["inputs"]["image"], "undistort")
        self.assertEqual(wired("merge"), {"A": "grade", "B": "undistort"})
        self.assertEqual(wired("redistort"), {"image": "merge", "camera": "camera"})
        self.assertEqual(nodes["write"]["inputs"]["image"], "redistort")
        self.assertEqual(nodes["export"]["inputs"]["camera"], "camera_os")
        self.assertEqual(nodes["render"]["inputs"]["camera"], "camera_os")
        params = nodes["solve"]["params"]
        self.assertEqual((params["solve_focal"], params["solve_distortion"]), ("estimate", "k1 k2"))
        for kind in ("Undistort", "Redistort", "Reformat", "CameraSolve3D"):
            self.assertIn(kind, SPECS)
        picks, distance = shot.pick_survey(shot.DISTORTED_SIZE, LENS)
        survey = json.loads(shot.survey_json(picks, distance))
        self.assertEqual(len(survey["picks"]), 2)
        self.assertEqual(len(survey["distances"]), 1)
        self.assertGreater(survey["distances"][0]["distance"], 3.0)

    def test_the_plate_argument_picks_the_shot(self):
        self.assertEqual(shot.resolve_plate("distorted")[2:], (True,))
        self.assertEqual(shot.resolve_plate("distorted")[0], shot.LENS_GRAPH)
        self.assertEqual(shot.resolve_plate(None)[2:], (False,))
        self.assertEqual(shot.resolve_plate("/some/folder")[1], Path("/some/folder"))
        self.assertTrue(shot.is_lens_graph(self.document))
        self.assertFalse(shot.is_lens_graph(json.loads((DEMO / "demo_shot.nbcomp").read_text())))


class LandmarkSolveTests(unittest.TestCase):
    """The camera of the distorted plate comes from the set's surveyed landmarks and one lens shared by every frame: the
    reprojection of a solve with no survey is flat across k1 on this move (README), the landmarks' known positions are not."""
    SIZE, FRAMES = (320, 180), 12

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        plate = Path(cls.folder.name) / "plate"
        make_plate.render_plate(plate, cls.SIZE, cls.FRAMES, k1=shot.DISTORTED_K1, k2=shot.DISTORTED_K2)
        cls.frames = solve_camera.read_plate(plate, cls.FRAMES)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def solve(self, typed_scale=shot.SOLVE_TYPED_FOCAL_SCALE):
        return solve_camera.solve_lens_landmarks(self.frames, *self.SIZE, LENS, shot.FOCAL * typed_scale, log=lambda *_: None)

    def test_the_lens_and_the_camera_come_out_of_a_typed_focal_length_that_is_20_percent_long(self):
        cameras, block, rows = self.solve()
        self.assertLess(abs(block["focal_mm"] / shot.FOCAL - 1.0), 0.03)
        self.assertLess(abs(block["k1"] - shot.DISTORTED_K1), 0.005)
        self.assertLess(abs(block["k2"] - shot.DISTORTED_K2), 0.003)
        self.assertLess(block["rms_px"], 1.0)
        self.assertEqual(sorted(cameras), list(range(1, self.FRAMES + 1)))
        self.assertLess(max(r["position_error_m"] for r in rows), 0.06)
        self.assertLess(max(r["rotation_error_degrees"] for r in rows), 0.5)
        self.assertAlmostEqual(cameras[1].lens_k1, block["k1"], places=5)
        self.assertAlmostEqual(cameras[1].lens_k2, block["k2"], places=5)

    def test_a_short_typed_focal_length_converges_to_the_same_lens_and_a_rerun_gives_the_same_numbers(self):
        _cameras, long_block, _rows = self.solve(1.2)
        _cameras, short_block, _rows = self.solve(0.85)
        self.assertLess(abs(short_block["focal_mm"] / long_block["focal_mm"] - 1.0), 0.01)
        self.assertLess(abs(short_block["k1"] - long_block["k1"]), 0.003)
        self.assertEqual(self.solve()[1], self.solve()[1])

    def test_the_committed_graph_carries_the_solved_camera_and_lens_not_the_rendering_camera(self):
        record = json.loads((DEMO / "solve_lens.json").read_text())
        self.assertEqual(record["solver"], "landmarks")
        self.assertLess(abs(record["errors"]["focal_percent"]), 3.0)
        self.assertLess(abs(record["errors"]["k1"]), 0.005)
        self.assertLess(record["errors"]["position_max_m"], 0.06)
        camera = json.loads((DEMO / "demo_shot_lens.nbcomp").read_text())["nodes"]["camera"]["params"]
        self.assertEqual((camera["lens_k1"], camera["lens_k2"]), (record["recovered"]["k1"], record["recovered"]["k2"]))
        self.assertNotEqual(camera["lens_k1"], shot.DISTORTED_K1)           # recovered from the plate, not copied from the renderer


class CardSolveTests(unittest.TestCase):
    """The DPV-SLAM solve of the full-size distorted plate needs the whole graphics card. It skips when the card or the
    solver is missing, or when another job (a suite run, a model) holds the machine's GPU lock for more than 10 seconds."""

    @classmethod
    def setUpClass(cls):
        from nodebased import gpulock
        from nodebased.providers import camera_solve_dpvo as dpvo
        reason = dpvo.available()
        if reason:
            raise unittest.SkipTest(reason)
        cls.stack = contextlib.ExitStack()
        saved = os.environ.get(gpulock.PATH_ENV)
        os.environ[gpulock.PATH_ENV] = gpulock.DEFAULT_PATH      # tests.isolation points the lock at a private file
        def restore():
            if saved is None:
                os.environ.pop(gpulock.PATH_ENV, None)
            else:
                os.environ[gpulock.PATH_ENV] = saved
        cls.stack.callback(restore)
        try:
            cls.stack.enter_context(gpulock.gpu_job(timeout=10))
        except TimeoutError:
            cls.stack.close()
            raise unittest.SkipTest("the GPU lock is held by another job")
        cls.folder = tempfile.TemporaryDirectory()
        cls.stack.callback(cls.folder.cleanup)

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def test_the_solve_recovers_the_focal_length_and_the_camera_path_of_the_distorted_plate(self):
        import solve_camera
        plate = Path(self.folder.name) / "plate_distorted"
        make_plate.render_plate(plate, shot.DISTORTED_SIZE, shot.FRAMES, k1=shot.DISTORTED_K1, k2=shot.DISTORTED_K2)
        document, reply, _picks = solve_camera.solve_distorted(plate, Path(self.folder.name) / "cache", log=lambda *_: None)
        camera = document["nodes"]["camera"]["params"]
        # the measured errors are in the example's README: a solve with no survey finds the camera path and the scale but not
        # k1 (its reprojection is flat across k1 on this move), so only the focal length is held to a bound
        self.assertLess(abs(camera["focal"] / shot.FOCAL - 1.0), 0.15)
        self.assertTrue(np.isfinite([camera["lens_k1"], camera["lens_k2"]]).all())
        self.assertEqual(reply["scale_source"], "survey")


if __name__ == "__main__":
    unittest.main()
