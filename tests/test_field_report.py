"""The field test kit (tools/field_report.py, tools/make_field_bundle.py, docs/FIELD_TEST.md, docs/FIELD_CLAUDE.md).

The end-to-end test runs the real command with --quick on this machine; on a machine with no GPU the GPU steps are recorded as
failed, which still has to produce the report, so the assertions are about the report's shape and not about every step passing.
"""
import json
import os
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
    def test_quick_machine_run_writes_report_json_logs_and_zip_in_under_three_minutes(self):
        with tempfile.TemporaryDirectory(prefix="field-report-test-") as tmp:
            out = Path(tmp) / "report"
            started = time.monotonic()
            # Exercise the real report/archive path without timing every hardware benchmark
            # inside the release suite. A software Windows adapter can make those benchmarks
            # exceed the smoke test deadline and leave an active child holding the log open.
            result = subprocess.run([sys.executable, str(ROOT / "tools" / "field_report.py"), "--quick", "--only", "machine", "--out", str(out),
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

    def test_the_render_mode_step_runs_the_progression_benchmark_and_keeps_its_numbers(self):
        record = {"adapter": "wgpu adapter: test card", "backend": "gpu", "scenes": {"furnished": {
            "first_look_seconds": {"cold": 0.9, "warm_median": 0.52, "warm_worst": 0.55},
            "settled_seconds": {"cold": 9.0, "warm_median": 8.1},
            "abandon_ms": {"n": 6, "median": 12.0, "p95": 31.5, "worst": 31.5},
            "edit_to_first_look_ms": {"n": 6, "median": 580.0, "p95": 620.0, "worst": 620.0},
            "first_look_traced": [640, 360], "orbit_paint_ms": {"median": 12.3, "worst": 13.6}}}}
        seen = []

        def fake_run(argv, log, timeout, **more):
            seen.append([str(a) for a in argv])
            Path(argv[argv.index("--field") + 1]).write_text(json.dumps(record), encoding="utf-8")
            return dict(returncode=0, timed_out=False, seconds=1.0, output="", output_tail="")
        with tempfile.TemporaryDirectory(prefix="field-report-test-") as tmp:
            ctx = field_report.Context(Path(tmp) / "out", "test", True, False)
            ctx.work.mkdir(parents=True)
            step = field_report.Step("bench_render_mode", "Render mode")
            with mock.patch.object(field_report, "run_command", fake_run):
                field_report.step_render_mode(ctx, step)
        self.assertEqual(step.status, "ok")
        self.assertIn("benchmark_progression.py", seen[0][1])
        self.assertIn("--quick", seen[0])
        keys = {h["key"]: h["value"] for h in step.headlines}
        self.assertEqual(keys["rendermode.furnished.first_look"], 520)
        self.assertEqual(keys["rendermode.furnished.abandon_p95"], 31.5)
        self.assertEqual(keys["rendermode.furnished.edit_first_look_p95"], 620.0)
        self.assertEqual(keys["rendermode.furnished.settled"], 8.1)
        self.assertEqual(keys["rendermode.furnished.first_look_pixels"], 640 * 360)
        self.assertEqual(keys["rendermode.furnished.orbit_paint_worst"], 13.6)
        self.assertEqual(step.data["adapter"], "wgpu adapter: test card")
        self.assertIn("bench_render_mode", [name for name, _, _ in field_report.STEPS])

    def test_live_picture_field_step_records_adapter_backend_and_observations(self):
        record = {"adapter_info": {"name": "Example GPU", "backend": "Vulkan"},
                  "results": [{"series": "roto_drag", "first_picture_s": 0.21,
                               "settle_s": 1.2},
                              {"series": "tier4_playback", "fps": 22.5}]}
        seen = []

        def fake_run(argv, log, timeout, **more):
            seen.append([str(a) for a in argv])
            Path(argv[argv.index("--field") + 1]).write_text(json.dumps(record), encoding="utf-8")
            return dict(returncode=0, timed_out=False, seconds=1.0, output="", output_tail="")
        with tempfile.TemporaryDirectory(prefix="field-live-picture-test-") as tmp:
            ctx = field_report.Context(Path(tmp) / "out", "Field laptop", True, False)
            ctx.work.mkdir(parents=True)
            step = field_report.Step("bench_live_picture", "Live picture")
            with mock.patch.object(field_report, "run_command", fake_run):
                field_report.step_live_picture(ctx, step)
        self.assertEqual(step.status, "ok")
        self.assertIn("live_picture_bench.py", seen[0][1])
        self.assertIn("--field", seen[0])
        self.assertIn("--automatic-playback", seen[0])
        self.assertIn("--warm-playback-range", seen[0])
        keys = {h["key"]: h["value"] for h in step.headlines}
        self.assertEqual(keys["livepicture.adapter"], "Example GPU")
        self.assertEqual(keys["livepicture.roto_drag.first_picture_s"], 0.21)
        self.assertEqual(keys["livepicture.tier4_playback.fps"], 22.5)
        self.assertIn("bench_live_picture", [name for name, _, _ in field_report.STEPS])

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


class FieldBundleEverything(unittest.TestCase):
    """--everything without a network: the downloads are replaced by fakes, the manifest and README are real."""

    LISTING = '<a href="3.12.9/">3.12.9/</a> <a href="3.12.10/">3.12.10/</a> <a href="3.12.14/">3.12.14/</a> <a href="3.13.1/">3.13.1/</a>'

    def test_python_installer_is_the_newest_patch_that_has_a_windows_installer(self):
        have = {f"{make_field_bundle.PYTHON_FTP}3.12.10/python-3.12.10-amd64.exe", f"{make_field_bundle.PYTHON_FTP}3.12.9/python-3.12.9-amd64.exe"}
        version, url = make_field_bundle.latest_python_installer("3.12", self.LISTING, lambda u: u in have)
        self.assertEqual(version, "3.12.10")
        self.assertTrue(url.endswith("/3.12.10/python-3.12.10-amd64.exe"))
        with self.assertRaises(SystemExit):
            make_field_bundle.latest_python_installer("3.12", self.LISTING, lambda u: False)

    def test_installers_and_release_land_in_the_manifest_with_hashes(self):
        with tempfile.TemporaryDirectory(prefix="field-bundle-test-") as tmp:
            out = Path(tmp)
            fetch = lambda url: ("bytes of " + url).encode()
            parts = make_field_bundle.fetch_installers(out, "3.12", False, fetch=fetch, listing=self.LISTING, exists=lambda u: "3.12.10" in u)
            self.assertEqual([p["name"] for p in parts], ["python", "vc_redist"])
            self.assertEqual(parts[0]["version"], "3.12.10")
            self.assertEqual(parts[0]["sha256"], make_field_bundle.sha256(out / "installers" / "python-3.12.10-amd64.exe"))
            self.assertEqual(parts[1]["url"], make_field_bundle.VC_REDIST_URL)

            seen = []

            def fake_gh(command):
                seen.append(command)
                self.assertEqual(command[:4], ["gh", "release", "download", "v0.37.0"])
                folder = Path(command[command.index("--dir") + 1])
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "NodeBased-0.37.0-windows-x64-portable.zip").write_bytes(b"zip")
                good = make_field_bundle.hashlib.sha256(b"zip").hexdigest()
                (folder / "SHA256SUMS").write_text(f"{good}  NodeBased-0.37.0-windows-x64-portable.zip\n{'0' * 64}  NodeBased-0.37.0-windows-x64-setup.exe\n")
                return 0
            stale = out / "app" / "NodeBased-0.37.0-windows-x64-setup.exe"
            stale.parent.mkdir(); stale.write_bytes(b"left over from an earlier bundle")
            app, problems = make_field_bundle.fetch_release(out, "0.37.0", False, gh=fake_gh)
            self.assertIn("*windows*portable*", seen[0])
            self.assertEqual([a["path"] for a in app], ["app/NodeBased-0.37.0-windows-x64-portable.zip"])
            self.assertEqual([a["matches_release_sums"] for a in app], [True])
            self.assertEqual(problems, [])
            self.assertFalse(stale.exists())
            self.assertFalse(list((out / "app").glob("*setup*")))

            names = ["cpython-3.12.14+20261013-x86_64-pc-windows-msvc-install_only.tar.gz", "cpython-3.12.15+20261013-x86_64-pc-windows-msvc-install_only.tar.gz",
                     "cpython-3.12.15+20261013-x86_64-pc-windows-msvc-install_only_stripped.tar.gz", "cpython-3.12.15+20261013-aarch64-pc-windows-msvc-install_only.tar.gz",
                     "cpython-3.13.16+20261013-x86_64-pc-windows-msvc-install_only.tar.gz"]
            self.assertEqual(make_field_bundle.standalone_asset(names, "3.12"), (names[1], "3.12.15"))
            with self.assertRaises(SystemExit):
                make_field_bundle.standalone_asset(names[2:4], "3.12")
            archive = b"tar bytes"
            sums = f"{make_field_bundle.hashlib.sha256(archive).hexdigest()}  {names[1]}\n"
            fetch_standalone = lambda url: sums.encode() if url.endswith("SHA256SUMS") else archive
            release = lambda args: {"tagName": "20261013", "assets": [{"name": n} for n in names]}
            standalone = make_field_bundle.fetch_standalone(out, "3.12", False, gh_json=release, fetch=fetch_standalone)
            self.assertEqual(standalone[0]["path"], "python/" + names[1])
            self.assertEqual((standalone[0]["name"], standalone[0]["version"], standalone[0]["release"]), ("python-standalone", "3.12.15", "20261013"))
            self.assertIn("20261013/cpython-3.12.15%2B20261013-x86_64", standalone[0]["url"])
            self.assertTrue(standalone[0]["matches_release_sums"])
            with self.assertRaises(SystemExit):
                make_field_bundle.fetch_standalone(out, "3.12", False, gh_json=release, fetch=lambda url: b"0" * 64 + b"  " + names[1].encode() if url.endswith("SHA256SUMS") else archive)
            parts = parts + standalone

            missing = ["wheels/: examplepkg has no win_amd64 cp312 wheel (pip download failed)"]
            make_field_bundle.write_readme(out, "0.37.0", "abc1234", "nodebased-abc1234.zip", "win_amd64", "3.12", parts + app, missing, [])
            entries, total = make_field_bundle.build_manifest(out, parts + app, missing, problems)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["missing"], missing)
            self.assertEqual({c["name"] for c in manifest["components"]}, {"python", "python-standalone", "vc_redist", "app"})
            listed = {e["path"] for e in entries}
            self.assertTrue({"BUNDLE-README.txt", "verify-bundle.ps1", "installers/vc_redist.x64.exe", "python/" + names[1]} <= listed)
            self.assertFalse([p for p in listed if "setup" in p])
            readme = (out / "BUNDLE-README.txt").read_text(encoding="utf-8")
            self.assertTrue(readme.startswith("No admin rights needed: everything installs under your user folder.\n"))
            for needle in ("561.17", "560.76", "docs.nvidia.com", "vc_redist", "OPTIONAL", "nvidia-smi", "--no-index", "DPVO", "examplepkg",
                           "HF_HUB_CACHE", "verify-bundle.ps1", "nodebased-abc1234.zip", "$env:USERPROFILE\\nb", "InstallAllUsers=0 PrependPath=1 Include_test=0",
                           "tar -xzf", "skipped", "the app's packages carry their own runtime"):
                self.assertIn(needle, readme)
            self.assertNotIn("C:\\nb", readme)
            self.assertNotIn("Add python.exe to PATH", readme)
            self.assertNotIn("setup exe next to", readme)
            self.assertIn("Get-FileHash", (out / "verify-bundle.ps1").read_text(encoding="utf-8"))
            self.assertEqual(make_field_bundle.verify(out), 0)

    def test_runtime_dll_report_reads_the_wheels(self):
        with tempfile.TemporaryDirectory(prefix="field-bundle-test-") as tmp:
            out = Path(tmp)
            (out / "wheels").mkdir()
            (out / "wheels-provider").mkdir()
            for path, members in (("wheels/shiboken6-6.12.0-cp310-abi3-win_amd64.whl", ["shiboken6/msvcp140.dll", "shiboken6/vcruntime140_1.dll", "shiboken6/x.pyd"]),
                                  ("wheels-provider/torch-2.14.1+cu126-cp312-cp312-win_amd64.whl", ["torch/lib/libiomp5md.dll"]),
                                  ("wheels/numpy-2.5.4-cp312-cp312-win_amd64.whl", ["numpy/msvcp140.dll"])):
                with make_field_bundle.zipfile.ZipFile(out / path, "w") as archive:
                    for member in members:
                        archive.writestr(member, b"x")
            report = make_field_bundle.runtime_dll_report(out)
            self.assertIn("carry their own runtime: yes (shiboken6); no (torch)", report)
            self.assertIn("msvcp140.dll, vcruntime140_1.dll", report)
            self.assertNotIn("numpy", report)

    def test_reuse_keeps_only_folders_that_still_match_the_manifest(self):
        with tempfile.TemporaryDirectory(prefix="field-bundle-test-") as tmp:
            out = Path(tmp)
            (out / "wheels").mkdir()
            (out / "wheels" / "a.whl").write_bytes(b"wheel")
            make_field_bundle.build_manifest(out)
            manifest = make_field_bundle.previous_manifest(out)
            self.assertTrue(make_field_bundle.folder_is_intact(out, manifest, "wheels"))
            self.assertFalse(make_field_bundle.folder_is_intact(out, manifest, "weights"))
            (out / "wheels" / "a.whl").write_bytes(b"other")
            self.assertFalse(make_field_bundle.folder_is_intact(out, manifest, "wheels"))
            self.assertEqual(make_field_bundle.previous_manifest(out / "nowhere"), {})

    def test_a_requirement_without_a_windows_wheel_is_named_not_dropped(self):
        calls = []

        def fake_pip(command, dry_run):
            calls.append(command)
            names = command[command.index("--only-binary=:all:") + 1:]
            return 1 if "sourceonly" in names else 0
        self.assertEqual(make_field_bundle.download_wheels(Path("w"), ["good", "sourceonly"], "win_amd64", "3.12", False, call=fake_pip), ["sourceonly"])
        self.assertEqual(make_field_bundle.download_wheels(Path("w"), ["good"], "win_amd64", "3.12", False, call=fake_pip), [])

    def test_everything_dry_run_lists_the_downloads_and_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="field-bundle-test-") as tmp:
            out = Path(tmp) / "bundle"
            # an empty weights cache, as on the CI runners: the dry run lists the weights as absent and still succeeds
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(Path(tmp) / "no-hub")}):
                self.assertEqual(make_field_bundle.main(["--everything", "--out", str(out), "--dry-run"]), 0)
            self.assertFalse(out.exists())


class RunTestsAcceptsSeveralPatterns(unittest.TestCase):
    def test_modules_unions_the_patterns(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import run_tests
        found = run_tests.modules(["test_gpulock.py", "test_gpumemory.py", "test_gpulock.py"])
        self.assertEqual(found, ["tests.test_gpulock", "tests.test_gpumemory"])
        self.assertEqual(run_tests.modules("test_gpulock.py"), ["tests.test_gpulock"])


if __name__ == "__main__":
    unittest.main()
