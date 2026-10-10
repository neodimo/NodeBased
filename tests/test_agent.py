"""The headless JSON-lines dispatcher runs the generative pipeline ops that need no window."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HeadlessPipelineTests(unittest.TestCase):
    def run_agent(self, requests):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "NODEBASED_CACHE": cache.name,
               "PYTHONPATH": str(ROOT)}
        done = subprocess.run([sys.executable, "-m", "nodebased.agent"], cwd=ROOT, env=env,
                              input="".join(json.dumps(r) + "\n" for r in requests),
                              capture_output=True, text=True, timeout=240)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        return [json.loads(line) for line in done.stdout.splitlines()]

    def test_export_loop_artifacts_and_workspace_refusal(self):
        scene = [
            {"op": "create", "id": "cube", "type": "Cube3D", "params": {"cube_size": 1}},
            {"op": "create", "id": "scene", "type": "Scene3D"},
            {"op": "create", "id": "camera", "type": "Camera3D", "params": {"tz": 7}},
            {"op": "create", "id": "render", "type": "Render3D",
             "params": {"width": 16, "height": 12, "samples": 1}},
            {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
            {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "render", "input": "camera", "source": "camera"}]
        chain = scene + [
            {"op": "export_conditioning", "id": "render", "first": 1, "last": 1},
            {"op": "workspace", "tab": "3D"},
            {"op": "artifacts", "action": "list", "kind": "scene_state"}]
        replies = self.run_agent(chain)
        self.assertFalse(replies[-2]["ok"])
        self.assertIn("desktop window", replies[-2]["error"])
        listed = replies[-1]["result"]["artifacts"]
        self.assertEqual([row["id"] for row in listed], [replies[-3]["result"]["scene_state"]])

    def test_generate_verify_loop_run_to_completion(self):
        """Headless has no queue panel, so the ids are fed back by a two-stage script."""
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "NODEBASED_CACHE": cache.name,
               "PYTHONPATH": str(ROOT)}
        process = subprocess.Popen([sys.executable, "-m", "nodebased.agent"], cwd=ROOT, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(process.kill)

        def send(request):
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
            self.assertTrue(reply["ok"], reply)
            return reply["result"]
        for command in (
                {"op": "create", "id": "cube", "type": "Cube3D", "params": {"cube_size": 1}},
                {"op": "create", "id": "scene", "type": "Scene3D"},
                {"op": "create", "id": "camera", "type": "Camera3D", "params": {"tz": 7}},
                {"op": "create", "id": "render", "type": "Render3D",
                 "params": {"width": 16, "height": 12, "samples": 1}},
                {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
                {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
                {"op": "connect", "id": "render", "input": "camera", "source": "camera"}):
            send(command)
        ids = send({"op": "export_conditioning", "id": "render", "first": 1, "last": 1})
        generated = send({"op": "generate", "scene_state": ids["scene_state"],
                          "bundle": ids["control_bundle"], "provider": "null"})["generated_sequence"]
        send({"op": "verify", "scene_state": ids["scene_state"], "bundle": ids["control_bundle"],
              "generated_sequence": generated})
        status = send({"op": "loop", "action": "start", "scene_state": ids["scene_state"],
                       "bundle": ids["control_bundle"], "provider": "null", "max_attempts": 2,
                       "spend_cap": 0.02, "estimate": 0.01, "unit": "credits (estimate)"})
        self.assertTrue(status["finished"])
        self.assertEqual(len(status["attempts"]), 2)
        chain = send({"op": "artifacts", "action": "provenance", "id": generated})["chain"]
        self.assertTrue({ids["scene_state"], ids["control_bundle"]} <= {row["id"] for row in chain})
        process.stdin.close()
        process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
