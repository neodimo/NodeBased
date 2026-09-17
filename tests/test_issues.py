import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PySide6.QtCore import QSettings

from nodebased import __version__
from nodebased import issues


class IssueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings_dir = tempfile.TemporaryDirectory()
        cls.old_format = QSettings.defaultFormat()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, cls.settings_dir.name)
        QSettings("NodeBased", "NodeBased").clear()

    @classmethod
    def tearDownClass(cls):
        QSettings("NodeBased", "NodeBased").clear()
        QSettings.setDefaultFormat(cls.old_format)
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()

    def test_settings_default_and_persist(self):
        self.assertTrue(issues.issue_filing_enabled())
        issues.set_issue_filing_enabled(False)
        self.assertFalse(issues.issue_filing_enabled())
        self.assertFalse(QSettings("NodeBased", "NodeBased").value(
            "agent/file_issue_enabled", True))

    def test_disabled_does_not_call_external_tools(self):
        issues.set_issue_filing_enabled(False)
        with mock.patch("nodebased.issues.subprocess.run") as run, \
                mock.patch("nodebased.issues.urllib.request.urlopen") as urlopen:
            with self.assertRaises(RuntimeError):
                issues.file_issue("title", "body")
        run.assert_not_called()
        urlopen.assert_not_called()

    def test_footer(self):
        footer = issues.build_footer("agent-7")
        self.assertIn("NodeBased " + __version__, footer)
        self.assertIn(issues.platform.system(), footer)
        self.assertIn(issues.platform.release(), footer)
        self.assertIn("agent-7", footer)
        self.assertIn("filed by agent", footer)

    def test_gh_happy_path_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "nested" / "issues.jsonl"
            result = mock.Mock(returncode=0, stdout="https://github.com/neodimo/NodeBased/issues/42\n")
            with mock.patch("nodebased.issues.shutil.which", return_value="/usr/bin/gh"), \
                    mock.patch("nodebased.issues.subprocess.run", return_value=result) as run:
                self.assertEqual(issues.file_issue("Bug", "Details", ["bug"], "agent", log_path=log),
                                 {"url": "https://github.com/neodimo/NodeBased/issues/42", "number": 42})
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][0:4], ["/usr/bin/gh", "issue", "create", "--repo"])
            entry = json.loads(log.read_text().strip())
            self.assertEqual(entry["title"], "Bug")
            self.assertEqual(entry["number"], 42)
            self.assertEqual(entry["agent_name"], "agent")

    def test_rest_fallback(self):
        class Response:
            status = 201
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps({"html_url": "https://github.com/neodimo/NodeBased/issues/9", "number": 9}).encode()
        captured = {}
        def fake_urlopen(request, timeout=None):
            captured.update(request=request, timeout=timeout)
            return Response()
        with tempfile.NamedTemporaryFile() as log, \
                mock.patch("nodebased.issues.shutil.which", return_value=None), \
                mock.patch.dict(os.environ, {"GITHUB_TOKEN": "secret"}, clear=True), \
                mock.patch("nodebased.issues.urllib.request.urlopen", fake_urlopen):
            result = issues.file_issue("Bug", "Body", log_path=log.name)
            entries = issues.read_issue_log(log.name)
        request = captured["request"]
        self.assertEqual(request.full_url, "https://api.github.com/repos/neodimo/NodeBased/issues")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(json.loads(request.data), {"title": "Bug", "body": mock.ANY, "labels": []})
        self.assertEqual(result["number"], 9)
        self.assertEqual(len(entries), 1)

    def test_no_token_does_not_call_api(self):
        with mock.patch("nodebased.issues.shutil.which", return_value=None), \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("nodebased.issues.subprocess.run", return_value=mock.Mock(returncode=1, stdout="")) as run, \
                mock.patch("nodebased.issues.urllib.request.urlopen") as urlopen:
            with self.assertRaisesRegex(RuntimeError, "No GitHub token"):
                issues.file_issue("Bug", "Body")
        run.assert_not_called()
        urlopen.assert_not_called()

    def test_log_reading_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.jsonl"
            self.assertEqual(issues.read_issue_log(path), [])
            path.write_text('{"title": "one"}\n{"title": "two"}\n')
            self.assertEqual(issues.read_issue_log(path), [{"title": "one"}, {"title": "two"}])
        with self.assertRaises(ValueError):
            issues.file_issue("", "body")


if __name__ == "__main__":
    unittest.main()
