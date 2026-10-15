"""The field test kit (tools/field_report.py, tools/make_field_bundle.py, docs/FIELD_TEST.md, docs/FIELD_CLAUDE.md).

The end-to-end test runs the real command with --quick on this machine; on a machine with no GPU the GPU steps are recorded as
failed, which still has to produce the report, so the assertions are about the report's shape and not about every step passing.
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import field_report  # noqa: E402
import make_field_bundle  # noqa: E402

STATUSES = {"ok", "failed", "timeout", "crashed", "skipped"}


def load(out):
    return json.loads((Path(out) / "field-report.json").read_text(encoding="utf-8"))


class FieldReportEndToEnd(unittest.TestCase):
    def test_quick_run_writes_report_json_logs_and_zip_in_under_three_minutes(self):
        with tempfile.TemporaryDirectory(prefix="field-report-test-") as tmp:
            out = Path(tmp) / "report"
            started = time.monotonic()
            result = subprocess.run([sys.executable, str(ROOT / "tools" / "field_report.py"), "--quick", "--out", str(out),
                                     "--label", "test machine"], cwd=ROOT, capture_output=True, text=True, timeout=175)
            self.assertLess(time.monotonic() - started, 180)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
            report = load(out)
            self.assertEqual(report["label"], "test machine")
            self.assertTrue(report["quick"])
            names = [s["name"] for s in report["steps"]]
            self.assertEqual(names, [name for name, _, _ in field_report.STEPS])
            for step in report["steps"]:
                self.assertIn(step["status"], STATUSES, step["name"])
            self.assertEqual(next(s for s in report["steps"] if s["name"] == "sdxl")["status"], "skipped")
            self.assertTrue(report["machine"]["nodebased_version"])
            markdown = (out / "field-report.md").read_text(encoding="utf-8")
            for heading in ("## Summary", "## Steps", "## Machine", "## Observations"):
                self.assertIn(heading, markdown)
            self.assertTrue(any((out / "logs").glob("*.log")))
            archive = Path(tmp) / "report.zip"
            self.assertTrue(archive.exists())
            with zipfile.ZipFile(archive) as z:
                members = z.namelist()
            self.assertIn("report/field-report.md", members)
            self.assertIn("report/field-report.json", members)
            self.assertFalse(any("/work/" in m for m in members))


class FieldReportSurvivesFailures(unittest.TestCase):
    def test_a_benchmark_that_raises_is_recorded_and_the_run_goes_on(self):
        with tempfile.TemporaryDirectory(prefix="field-report-test-") as tmp:
            out = Path(tmp) / "report"
            with mock.patch.object(field_report, "step_lens", side_effect=RuntimeError("injected benchmark crash")):
                code = field_report.main(["--quick", "--only", "bench_lens,bench_pathtrace", "--out", str(out)])
            self.assertEqual(code, 0)
            steps = {s["name"]: s for s in load(out)["steps"]}
            self.assertEqual(steps["bench_lens"]["status"], "crashed")
            self.assertIn("injected benchmark crash", steps["bench_lens"]["reason"])
            self.assertIn("RuntimeError", steps["bench_lens"]["output_tail"])
            self.assertIn(steps["bench_pathtrace"]["status"], STATUSES - {"crashed", "skipped"})   # it still ran after the crash
            self.assertEqual(steps["bench_4k"]["status"], "skipped")                                  # left out by --only
            markdown = (out / "field-report.md").read_text(encoding="utf-8")
            self.assertIn("did not complete cleanly", markdown)
            self.assertIn("injected benchmark crash", markdown)
            self.assertTrue((Path(tmp) / "report.zip").exists())

    def test_a_command_that_times_out_or_cannot_start_is_a_result(self):
        with tempfile.TemporaryDirectory(prefix="field-report-test-") as tmp:
            log = Path(tmp) / "x.log"
            slow = field_report.run_command([sys.executable, "-c", "import time; print('start', flush=True); time.sleep(60)"], log, timeout=2)
            self.assertTrue(slow["timed_out"])
            self.assertIn("start", slow["output"])
            missing = field_report.run_command(["no-such-program-field-report"], Path(tmp) / "y.log", timeout=5)
            self.assertIsNone(missing["returncode"])
            self.assertIn("could not run", missing["output"])

    def test_rezip_puts_the_observations_into_the_report(self):
        with tempfile.TemporaryDirectory(prefix="field-report-test-") as tmp:
            out = Path(tmp) / "report"
            field_report.main(["--quick", "--only", "bench_pathtrace", "--skip", "bench_pathtrace", "--out", str(out)])
            self.assertIn("Fill this in", (out / "field-report.md").read_text(encoding="utf-8"))
            (out / "observations.md").write_text("- Display: 4K at 150%, nothing wrong\n", encoding="utf-8")
            self.assertEqual(field_report.main(["--out", str(out), "--rezip"]), 0)
            self.assertIn("4K at 150%", (out / "field-report.md").read_text(encoding="utf-8"))
            with zipfile.ZipFile(Path(tmp) / "report.zip") as z:
                self.assertIn("4K at 150%", z.read("report/field-report.md").decode("utf-8"))

    def test_gpu_test_modules_are_found_by_what_they_import(self):
        chosen, tokens = field_report.gpu_test_modules()
        self.assertIn("test_gpulens", chosen)
        self.assertIn("test_3d_gpu_rt", chosen)
        self.assertNotIn("test_knowledge", chosen)
        self.assertIn("wgpu", tokens)
        for module in field_report.QUICK_TEST_MODULES:
            self.assertTrue((ROOT / "tests" / f"{module}.py").exists(), module)

    def test_every_headline_baseline_names_a_source(self):
        for key, (value, unit, source) in field_report.BASELINES.items():
            self.assertGreater(value, 0, key)
            self.assertTrue(source.startswith("docs/"), key)
            self.assertTrue((ROOT / source.split(" ")[0]).exists(), source)


class FieldBundle(unittest.TestCase):
    def test_requirements_include_both_extras_and_the_pip_flags_target_windows(self):
        requirements = make_field_bundle.project_requirements()
        self.assertTrue(any(r.startswith("wgpu") for r in requirements))
        self.assertTrue(any(r.startswith("usd-core") for r in requirements))
        self.assertTrue(any(r.startswith("PySide6") for r in requirements))
        command = make_field_bundle.pip_download_command(Path("w"), requirements, "win_amd64", "3.12")
        for flag in ("--only-binary=:all:", "win_amd64", "3.12", "cp312"):
            self.assertIn(flag, command)

    def test_dry_run_writes_nothing_and_manifest_verifies_a_copy(self):
        with tempfile.TemporaryDirectory(prefix="field-bundle-test-") as tmp:
            out = Path(tmp) / "bundle"
            self.assertEqual(make_field_bundle.main(["--out", str(out), "--dry-run"]), 0)
            self.assertFalse(out.exists())
            out.mkdir()
            (out / "a.whl").write_bytes(b"wheel")
            (out / "sub").mkdir()
            (out / "sub" / "b.bin").write_bytes(b"weights")
            entries, total = make_field_bundle.build_manifest(out)
            self.assertEqual(total, 12)
            self.assertEqual({e["path"] for e in entries}, {"a.whl", "sub/b.bin"})
            self.assertEqual(make_field_bundle.verify(out), 0)
            (out / "sub" / "b.bin").write_bytes(b"changed")
            self.assertEqual(make_field_bundle.verify(out), 1)


class RunTestsAcceptsSeveralPatterns(unittest.TestCase):
    def test_modules_unions_the_patterns(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import run_tests
        found = run_tests.modules(["test_gpulock.py", "test_gpumemory.py", "test_gpulock.py"])
        self.assertEqual(found, ["tests.test_gpulock", "tests.test_gpumemory"])
        self.assertEqual(run_tests.modules("test_gpulock.py"), ["tests.test_gpulock"])


if __name__ == "__main__":
    unittest.main()
