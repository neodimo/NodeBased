"""tools/laptop_scene_check.py: the one command DiMo runs on the Windows laptop and the A6000 machine (Lane 4, 10/23, step B part 4).
The summary and the verdict are tested on canned run files; one real run on the software renderer proves the command end to end."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import tests.isolation  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import laptop_scene_check as check  # noqa: E402


def result(scene="cube", adapter="NVIDIA GeForce RTX 4070 Laptop GPU", between=24.0, worst=46.0, tiers=None):
    return {"scene": scene, "adapter": adapter, "proxy": "on", "fps": 24.0, "interval_ms": {"p95": 44.0},
            "tier_pictures": tiers or {"1": 140},
            "first_seconds": {"cold": {"fps_between_pictures": between, "interval_max_ms": worst, "frames_skipped": 0,
                                       "first_picture": {"open_to_first_picture_s": 0.58}},
                              "warm": {"interval_max_ms": worst}}}


class Rows(unittest.TestCase):
    def test_a_run_becomes_the_numbers_to_paste_back(self):
        row = check.row_of(result(tiers={"1": 90, "2": 10}))
        self.assertEqual((row["first_picture_s"], row["between_fps"], row["worst_ms"], row["reduced_percent"]), (0.58, 24.0, 46.0, 10))

    def test_a_run_that_produced_nothing_is_an_error_row(self):
        self.assertEqual(check.row_of({"scene": "mixed", "error": "no result in 900 s"})["error"], "no result in 900 s")

    def test_a_scene_holds_with_no_gap_over_the_gate_warm_and_after_a_rest(self):
        good, rested = check.row_of(result()), check.row_of(result(worst=53.9))
        self.assertEqual(check.verdict(good, rested), "OK")

    def test_a_gap_after_a_rest_is_named(self):
        text = check.verdict(check.row_of(result()), check.row_of(result(worst=154.0)))
        self.assertIn("154.0 ms gap after a rest", text)

    def test_a_slow_scene_is_named(self):
        self.assertIn("19.5 pictures a second", check.verdict(check.row_of(result(between=19.5)), check.row_of(result())))

    def test_a_missing_run_is_no_result(self):
        self.assertEqual(check.verdict({"error": "x"}, check.row_of(result())), "no result")


class Summary(unittest.TestCase):
    info = {"platform": "Windows-11", "when": "2026-10-24 09:00 AM",
            "adapters": [{"name": "Intel(R) Iris(R) Xe Graphics", "type": "integrated", "backend": "D3D12"},
                         {"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "type": "discrete", "backend": "D3D12"}],
            "chosen": {"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "type": "discrete",
                       "reason": "Automatic: a discrete graphics card is preferred over Intel(R) Iris(R) Xe Graphics"}}

    def test_the_app_runs_its_own_choice_then_the_other_kind_of_card(self):
        self.assertEqual(check.adapters_to_run(self.info), ["NVIDIA GeForce RTX 4070 Laptop GPU", "Intel(R) Iris(R) Xe Graphics"])
        only = dict(self.info, adapters=self.info["adapters"][1:])
        self.assertEqual(check.adapters_to_run(only), ["NVIDIA GeForce RTX 4070 Laptop GPU"])

    def test_the_text_names_the_cards_the_choice_and_each_scene(self):
        warm, rested = check.row_of(result()), check.row_of(result(worst=61.0))
        report = {"machine": self.info, "runs": [{"adapter": "NVIDIA GeForce RTX 4070 Laptop GPU", "size": "1920x1080", "scenes": [
            {"scene": "cube", "warm": warm, "rested": rested, "verdict": check.verdict(warm, rested)}]}]}
        text = check.summary_text(report)
        self.assertIn("card: Intel(R) Iris(R) Xe Graphics (integrated, D3D12)", text)
        self.assertIn("The app chooses NVIDIA GeForce RTX 4070 Laptop GPU: Automatic: a discrete graphics card is preferred", text)
        self.assertIn("cube: first picture 0.58 s, 24.0 fps, gap 46.0 ms, after a rest 61.0 ms", text)
        self.assertIn("61.0 ms gap after a rest", text)
        self.assertLess(len(text.splitlines()), 40)       # short enough to paste into a message


class EndToEnd(unittest.TestCase):
    def test_the_command_runs_a_scene_and_writes_the_result_files(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
            done = subprocess.run([sys.executable, str(ROOT / "tools" / "laptop_scene_check.py"), "--self-check", "--out", folder],
                                  env=env, capture_output=True, text=True, timeout=300)
            self.assertIn("Paste back", done.stdout, done.stdout[-600:] + done.stderr[-600:])
            report = json.loads((pathlib.Path(folder) / "result.json").read_text())
            scene = report["runs"][0]["scenes"][0]
            self.assertEqual(scene["scene"], "cube")
            self.assertNotIn("error", scene["warm"], scene)
            self.assertGreater(scene["warm"]["between_fps"], 5)
            text = (pathlib.Path(folder) / "result.txt").read_text()
            self.assertIn("cube: first picture", text)


if __name__ == "__main__":
    unittest.main()
