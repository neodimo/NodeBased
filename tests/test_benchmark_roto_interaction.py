"""The Roto interaction benchmark tool runs headless on a tiny shot and reports every figure it promises.

No timing is asserted (the suite runs many interpreters at once): the test checks that the actions ran, that their events were
posted and handled, and that a point drag ends in exactly one undo step.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class RotoInteractionBenchmarkTests(unittest.TestCase):
    def test_self_check_runs_and_reports_the_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "out.json"
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
            run = subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_roto_interaction.py"), "--self-check",
                                  "--output", str(output)], env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            result = json.loads(output.read_text())
        actions = result["cases"]["light"]["actions"]
        self.assertEqual(set(actions), {"tool_switch_keys", "drag_point", "paint_stroke"})
        drag = actions["drag_point"]
        self.assertEqual(drag["moves_posted"], 60)
        self.assertGreater(drag["moves_handled"], 0)
        self.assertEqual(drag["undo_steps_added"], 1)            # one drag, one undo step
        for key in ("move_handler_ms", "move_overlay_ms", "picture_age_ms"):
            self.assertIsNotNone(drag[key]["median"], key)
        self.assertEqual(actions["paint_stroke"]["undo_steps_added"], 1)


if __name__ == "__main__":
    unittest.main()
