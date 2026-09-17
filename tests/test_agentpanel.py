import os
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtWidgets import QApplication

from nodebased import issues
from nodebased.agentpanel import AgentPanel, PtyTerminal, claude_launch_command, codex_launch_command


class AgentPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings_dir = tempfile.TemporaryDirectory()
        cls.old_xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = cls.settings_dir.name
        cls.old_format = QSettings.defaultFormat()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, cls.settings_dir.name)
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        QSettings("NodeBased", "NodeBased").clear()
        QSettings.setDefaultFormat(cls.old_format)
        if cls.old_xdg_config_home is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = cls.old_xdg_config_home
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()

    def test_launch_commands(self):
        claude = claude_launch_command("sock/\"x", "agent\"x")
        self.assertEqual(claude[0:2], ["claude", "--mcp-config"])
        self.assertIn("--append-system-prompt", claude)
        self.assertTrue(os.path.exists(claude[2]))
        codex = codex_launch_command("sock/\"x", "agent\"x")
        self.assertEqual(codex[0], "codex")
        self.assertEqual(codex[1], "-c")
        self.assertIn("mcp_servers.nodebased.command=\"python\"", codex)
        self.assertIn("sock/\\\"x", codex[4])
        self.assertIn("agent\\\"x", codex[4])

    def test_settings_and_defaults(self):
        panel = AgentPanel()
        self.assertFalse(panel.disk_checkbox.isChecked())
        self.assertTrue(panel.issue_checkbox.isChecked())
        panel.disk_checkbox.setChecked(True)
        self.assertTrue(QSettings("NodeBased", "NodeBased").value("agent/allow_disk_ops"))
        with mock.patch("nodebased.agentpanel.issues.set_issue_filing_enabled") as setter:
            panel2 = AgentPanel()
            panel2.issue_checkbox.setChecked(False)
            setter.assert_called_once_with(False)

    def test_activity_log(self):
        panel = AgentPanel()
        panel.log_tool_call("knowledge", {"topic": "overview"}, "ok")
        self.assertIn("knowledge", panel.activity_log.item(0).text())
        self.assertIn("overview", panel.activity_log.item(0).text())

    def test_missing_cli_is_inert(self):
        panel = AgentPanel()
        panel.set_endpoint("endpoint", "agent")
        with mock.patch("nodebased.agentpanel.shutil.which", return_value=None), \
                mock.patch("nodebased.agentpanel.PtyTerminal") as terminal:
            panel.start_claude()
            panel.start_codex()
        terminal.assert_not_called()
        self.assertIn("not installed", panel.activity_log.item(0).text())
        self.assertIn("not installed", panel.activity_log.item(1).text())

    @unittest.skipUnless(sys.platform != "win32", "POSIX PTY test")
    def test_pty_output_close_and_resize(self):
        terminal = PtyTerminal(["/bin/printf", "hello-pty"])
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "hello-pty" not in terminal.output.toPlainText():
            QCoreApplication.processEvents()
            time.sleep(0.01)
        self.assertIn("hello-pty", terminal.output.toPlainText())
        terminal.resize(400, 200)
        terminal.close()
        terminal.close()


if __name__ == "__main__":
    unittest.main()
