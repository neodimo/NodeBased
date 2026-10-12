"""Stage 9 of the end-to-end sprint: the verifier's survey check (reprojection of surveyed points through the
exported cameras), on its own and through the agent's `verify` op."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import filmback, scene3d
from nodebased.conditioning_verify import Tolerances, survey_line, survey_reprojection

ROOT = Path(__file__).resolve().parent.parent


def camera(tz=7.0, fov=None):
    fov = fov or filmback.fov_from_aperture(filmback.DEFAULT_FOCAL, filmback.DEFAULT_VAPERTURE)
    return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0.0, 0.0, tz)), scene3d.Vec3(0.0, 0.0, 0.0), fov)


def tracks(cam, width, height, world, frames, shift=(0.0, 0.0)):
    pixels, _ = scene3d.project(cam, width, height, np.asarray(world, np.float64))
    return {"points": [{"name": f"p{i}", "world": list(map(float, point)),
                        "track": [{"frame": f, "x": float(pixels[i, 0] + shift[0]), "y": float(pixels[i, 1] + shift[1])}
                                  for f in frames]} for i, point in enumerate(world)]}


WORLD = [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.4, -0.3, -0.5)]


class SurveyReprojectionTests(unittest.TestCase):
    def state(self, cam, width=320, height=180, frames=(1, 2)):
        return {"frames": [{"frame": f, "camera": cam, "resolution": (width, height)} for f in frames]}

    def test_exact_tracks_pass_and_a_shift_is_measured_in_pixels(self):
        cam = camera()
        exact = survey_reprojection(self.state(cam), tracks(cam, 320, 180, WORLD, (1, 2)))
        self.assertLess(exact["rms_pixels"], 1e-3)
        self.assertEqual((exact["points"], exact["observations"], exact["pass"]), (3, 6, True))
        self.assertAlmostEqual(exact["threshold_rms_pixels"], 0.9)        # 0.5% of 180
        self.assertAlmostEqual(exact["threshold_max_pixels"], 2.7)
        shifted = survey_reprojection(self.state(cam), tracks(cam, 320, 180, WORLD, (1, 2), shift=(3.0, 4.0)))
        self.assertAlmostEqual(shifted["rms_pixels"], 5.0, places=3)
        self.assertFalse(shifted["pass"])
        self.assertIn("RMS 5.00 px", survey_line(shifted))
        self.assertTrue(survey_line(shifted).endswith("FAIL"))

    def test_a_wrong_lens_fails_the_survey(self):
        truth = camera()
        survey = tracks(truth, 320, 180, WORLD, (1,))
        wrong = camera(fov=truth.fov * 1.15)
        report = survey_reprojection(self.state(wrong, frames=(1,)), survey)
        self.assertFalse(report["pass"])
        self.assertGreater(report["rms_pixels"], 3.0)

    def test_the_bar_scales_with_the_plate_and_unprojectable_points_fail(self):
        cam = camera()
        big = survey_reprojection(self.state(cam, 1920, 1080), tracks(cam, 1920, 1080, WORLD, (1, 2), shift=(3.0, 0.0)))
        self.assertTrue(big["pass"])                      # 3 px of a 1080-line plate is within 0.5%
        behind = tracks(cam, 320, 180, WORLD, (1,))
        behind["points"].append({"name": "behind", "world": [0.0, 0.0, 9.0], "track": [{"frame": 1, "x": 0, "y": 0}]})
        behind["points"].append({"name": "late", "world": [0.0, 0.0, 0.0], "track": [{"frame": 99, "x": 0, "y": 0}]})
        report = survey_reprojection(self.state(cam, frames=(1,)), behind)
        self.assertEqual((report["missing"], report["pass"]), (2, False))
        self.assertIn("2 not projected", survey_line(report))
        self.assertFalse(survey_reprojection(self.state(cam), {"points": []})["pass"])

    def test_custom_tolerance(self):
        cam = camera()
        report = survey_reprojection(self.state(cam), tracks(cam, 320, 180, WORLD, (1, 2), shift=(1.0, 0.0)),
                                     Tolerances(survey_rms_fraction=0.01, survey_max_fraction=0.02))
        self.assertTrue(report["pass"])


class VerifyOpSurveyTests(unittest.TestCase):
    def test_verify_op_scores_the_survey_and_it_counts_in_the_verdict(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "NODEBASED_CACHE": cache.name, "PYTHONPATH": str(ROOT)}
        process = subprocess.Popen([sys.executable, "-m", "nodebased.agent"], cwd=ROOT, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(process.kill)

        def send(request, ok=True):
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
            self.assertEqual(reply["ok"], ok, reply)
            return reply.get("result") if ok else reply["error"]
        width, height = 32, 24
        for command in (
                {"op": "create", "id": "cube", "type": "Cube3D", "params": {"cube_size": 1}},
                {"op": "create", "id": "scene", "type": "Scene3D"},
                {"op": "create", "id": "camera", "type": "Camera3D", "params": {"tz": 7}},
                {"op": "create", "id": "render", "type": "Render3D", "params": {"width": width, "height": height, "samples": 1}},
                {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
                {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
                {"op": "connect", "id": "render", "input": "camera", "source": "camera"}):
            send(command)
        ids = send({"op": "export_conditioning", "id": "render", "first": 1, "last": 1})
        generated = send({"op": "generate", "scene_state": ids["scene_state"], "bundle": ids["control_bundle"],
                          "provider": "reproject"})["generated_sequence"]
        brief = {"op": "verify", "scene_state": ids["scene_state"], "bundle": ids["control_bundle"],
                 "generated_sequence": generated}
        plain = send(brief)
        self.assertNotIn("survey", plain)
        self.assertNotIn("survey", plain["score_card"]["bindings"])
        good = send({**brief, "survey": tracks(camera(), width, height, WORLD, (1,))})
        self.assertTrue(good["survey"]["pass"], good["survey"])
        self.assertTrue(good["score_card"]["bindings"]["survey"]["pass"])
        self.assertEqual(good["verdict"], plain["verdict"])
        self.assertNotEqual(good["report"], plain["report"])
        bad = send({**brief, "survey": tracks(camera(), width, height, WORLD, (1,), shift=(4.0, 0.0))})
        self.assertFalse(bad["score_card"]["bindings"]["survey"]["pass"])
        self.assertEqual(bad["verdict"], "FAIL")
        self.assertIn("survey must be", send({**brief, "survey": [1, 2]}, ok=False))
        process.stdin.close()
        process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
