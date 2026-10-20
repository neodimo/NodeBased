"""The field-kit laptop presentation check: its self-check runs here and its result record reads back.
Nothing in this module measures a GPU or claims a Windows number."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "laptop_present_check.py"


def load():
    spec = importlib.util.spec_from_file_location("laptop_present_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaptopCheckTests(unittest.TestCase):
    def test_self_check_runs_on_this_machine_and_says_it_measured_nothing(self):
        done = subprocess.run([sys.executable, "-I", str(TOOL), "--self-check"], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("self-check passed", done.stdout)
        self.assertIn("no adapter, backend or cadence was measured", done.stdout)

    def test_measuring_is_refused_off_windows(self):
        if sys.platform == "win32":
            self.skipTest("this is the Windows machine")
        done = subprocess.run([sys.executable, "-I", str(TOOL), "--seconds", "10"], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Windows measurement only", done.stderr)

    def test_the_matrix_runs_dx12_first_and_vulkan_as_the_comparator(self):
        matrix = load().MATRIX
        self.assertEqual(matrix[0], ("Dx12", "high-performance"))
        self.assertEqual(matrix[1], ("Vulkan", "high-performance"))

    def test_result_record_carries_adapter_backend_and_cadence_and_reads_back(self):
        tool = load()
        meta = {"status": "complete", "platform": "Windows-11-test", "requested_backend": "Vulkan",
                "requested_preference": "high-performance", "selected_backend": "Vulkan",
                "selected_adapter": {"device": "Test GPU", "adapter_type": "IntegratedGPU", "backend_type": "Vulkan",
                                     "vendor": "x", "driver": "1.2", "description": "d"},
                "selection_matches_requested_class": False,
                "summary": {"retained": {"frames_after_warmup": 50,
                                         "focused_same_size_present_return_interval_ms": tool.distribution([16.0, 17.0, 33.0]),
                                         "input_proxy_latest_ms": tool.distribution([])}}}
        result = tool.build_result(meta)
        self.assertEqual(result["selected_adapter"]["device"], "Test GPU")
        self.assertEqual(result["selected_backend"], "Vulkan")
        self.assertFalse(result["selection_matches_requested_class"])      # a mismatch is recorded, not hidden
        self.assertTrue(result["measured_on_windows"])
        self.assertEqual(result["cadence"]["retained"]["interval_p50_ms"], 17.0)
        self.assertAlmostEqual(result["cadence"]["retained"]["fps_at_p50"], 1000 / 17.0)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "result.json").write_text(json.dumps({"cases": [result]}), encoding="utf-8")
            back = json.loads((Path(folder) / "result.json").read_text(encoding="utf-8"))["cases"][0]
            self.assertEqual(back, json.loads(json.dumps(result)))
            self.assertEqual(tool.read_back(folder), 0)

    def test_a_failed_run_still_yields_a_record(self):
        result = load().build_result({"status": "failed", "platform": "p", "requested_backend": "Dx12"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["cadence"], {})
        self.assertFalse(result["measured_on_windows"])


if __name__ == "__main__":
    unittest.main()
