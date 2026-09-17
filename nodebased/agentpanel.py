"""The optional interactive terminal and agent controls used by NodeBased."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pyte
from PySide6.QtCore import QEvent, QSettings, QSocketNotifier, QTimer, Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from . import issues


_OVERVIEW_PROMPT = "Call the knowledge tool with topic 'overview' before doing anything else in this NodeBased session."


def claude_launch_command(endpoint: str, agent_name: str) -> list[str]:
    """Create Claude Code's MCP config and return its interactive command."""
    config = {
        "mcpServers": {
            "nodebased": {
                "command": "python",
                "args": ["-m", "nodebased.mcpserver", "--endpoint", endpoint,
                         "--agent-name", agent_name],
            }
        }
    }
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="nodebased-mcp-",
                                          encoding="utf-8", delete=False)
    try:
        json.dump(config, handle)
    finally:
        handle.close()
    return ["claude", "--mcp-config", handle.name, "--append-system-prompt", _OVERVIEW_PROMPT]


def _toml_string(value: str) -> str:
    # JSON string syntax is also a valid TOML basic string, including escaped quotes.
    return json.dumps(value, ensure_ascii=False)


def codex_launch_command(endpoint: str, agent_name: str) -> list[str]:
    args = ["-m", "nodebased.mcpserver", "--endpoint", endpoint, "--agent-name", agent_name]
    return [
        "codex", "-c", f"mcp_servers.nodebased.command={_toml_string('python')}",
        "-c", f"mcp_servers.nodebased.args={json.dumps(args, ensure_ascii=False)}",
        _OVERVIEW_PROMPT,
    ]


class PtyTerminal(QWidget):
    """A small read-only terminal display connected to a real child PTY."""

    finished = Signal()

    def __init__(self, command=None, cwd=None, env=None, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._master_fd = None
        self._process = None
        self._notifier = None
        self._closed = False
        self.screen = pyte.Screen(*self._screen_size())
        self.stream = pyte.ByteStream(self.screen)
        self.output = QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.output.setFont(QFont("Monospace"))
        self.output.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.output)

        if sys.platform == "win32":
            self._init_windows()
        else:
            self._init_posix(command, cwd, env)

    def _screen_size(self):
        font_metrics = self.fontMetrics()
        cols = max(1, self.width() // max(1, font_metrics.horizontalAdvance("M")))
        rows = max(1, self.height() // max(1, font_metrics.lineSpacing()))
        return cols, rows

    def _init_windows(self):
        try:
            import winpty  # noqa: F401  # Deliberately lazy and optional.
        except ImportError:
            self.output.setPlainText("Windows PTY terminal requires the optional pywinpty package")
        else:
            self.output.setPlainText("Windows PTY terminal support is unavailable in this build")

    def _init_posix(self, command, cwd, env):
        import fcntl
        import pty
        import termios

        self._fcntl = fcntl
        self._termios = termios
        self._master_fd, slave_fd = pty.openpty()
        os.set_blocking(self._master_fd, False)
        if command is None:
            command = [os.environ.get("SHELL") or "/bin/bash"]
        elif isinstance(command, str):
            command = [command]
        try:
            self._process = subprocess.Popen(
                list(command), stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                start_new_session=True, cwd=cwd, env=env,
            )
        finally:
            os.close(slave_fd)
        self._set_winsize()
        self._notifier = QSocketNotifier(self._master_fd, QSocketNotifier.Type.Read, self)
        self._notifier.activated.connect(self._read_available)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._check_process)
        self._poll_timer.start()

    def _set_winsize(self):
        if self._master_fd is None or not hasattr(self, "_termios"):
            return
        cols, rows = self._screen_size()
        try:
            self.screen.resize(rows, cols)
            size = struct.pack("HHHH", rows, cols, 0, 0)
            self._fcntl.ioctl(self._master_fd, self._termios.TIOCSWINSZ, size)
        except (OSError, ValueError):
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._set_winsize()
        self._render()

    def _render(self):
        self.output.setPlainText("\n".join(self.screen.display))
        scrollbar = self.output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _read_available(self, *_args):
        if self._master_fd is None:
            return
        while True:
            try:
                data = os.read(self._master_fd, 65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                self._child_finished()
                break
            if not data:
                self._child_finished()
                break
            self.stream.feed(data)
            self._render()

    def _check_process(self):
        if self._process is not None and self._process.poll() is not None:
            self._child_finished()

    def _child_finished(self):
        if self._process is not None and self._process.poll() is None:
            return
        if self._notifier is not None:
            self._notifier.setEnabled(False)
        if getattr(self, "_poll_timer", None) is not None:
            self._poll_timer.stop()
        if self._process is not None:
            self.finished.emit()
            self._process = None

    def send_text(self, text: str):
        if self._master_fd is None or self._closed:
            return
        try:
            os.write(self._master_fd, text.encode())
        except OSError:
            pass

    def _key_bytes(self, event: QKeyEvent):
        keys = {
            Qt.Key.Key_Return: b"\r", Qt.Key.Key_Enter: b"\r",
            Qt.Key.Key_Backspace: b"\x7f", Qt.Key.Key_Tab: b"\t",
            Qt.Key.Key_Up: b"\x1b[A", Qt.Key.Key_Down: b"\x1b[B",
            Qt.Key.Key_Right: b"\x1b[C", Qt.Key.Key_Left: b"\x1b[D",
        }
        return keys.get(event.key(), event.text().encode())

    def keyPressEvent(self, event):
        data = self._key_bytes(event)
        if data:
            try:
                os.write(self._master_fd, data)
            except (OSError, TypeError):
                pass
            event.accept()
        else:
            event.ignore()

    def eventFilter(self, watched, event):
        if watched is self.output and event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(watched, event)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._notifier is not None:
            self._notifier.setEnabled(False)
        if getattr(self, "_poll_timer", None) is not None:
            self._poll_timer.stop()
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=0.3)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    process.kill()
                    process.wait(timeout=0.3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None


class AgentPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.endpoint = None
        self.agent_name = "nodebased-agent"
        self.terminal = None
        self._claude_config_paths = []
        self.claude_button = QPushButton("Start Claude Code")
        self.codex_button = QPushButton("Start Codex")
        self.claude_button.clicked.connect(self.start_claude)
        self.codex_button.clicked.connect(self.start_codex)
        self.activity_log = QListWidget()
        self.disk_checkbox = QCheckBox("Allow save / load / render from the agent")
        self.issue_checkbox = QCheckBox("Allow the agent to file GitHub issues")
        settings = QSettings("NodeBased", "NodeBased")
        self.disk_checkbox.setChecked(self._setting_bool(settings, "agent/allow_disk_ops", False))
        self.issue_checkbox.setChecked(issues.issue_filing_enabled())
        self.disk_checkbox.toggled.connect(self._set_disk_ops)
        self.issue_checkbox.toggled.connect(issues.set_issue_filing_enabled)
        self.terminal_container = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(self.claude_button)
        buttons.addWidget(self.codex_button)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addLayout(self.terminal_container)
        layout.addWidget(QLabel("Activity"))
        layout.addWidget(self.activity_log)
        layout.addWidget(self.disk_checkbox)
        layout.addWidget(self.issue_checkbox)

    @staticmethod
    def _setting_bool(settings, key, default):
        value = settings.value(key, default)
        return value not in (False, "false", "0", 0)

    def _set_disk_ops(self, enabled):
        settings = QSettings("NodeBased", "NodeBased")
        settings.setValue("agent/allow_disk_ops", bool(enabled))
        settings.sync()

    def set_endpoint(self, endpoint, agent_name):
        self.endpoint, self.agent_name = endpoint, agent_name

    def log_tool_call(self, name, arguments, result_summary):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.activity_log.addItem(f"{stamp} | {name}({arguments}) -> {result_summary}")
        self.activity_log.scrollToBottom()

    def _start(self, cli):
        if not self.endpoint:
            self.activity_log.addItem("Agent terminal unavailable: endpoint is not set")
            return
        if not shutil.which(cli):
            self.activity_log.addItem(f"{cli} is not installed; cannot start agent terminal")
            return
        self._replace_terminal()
        command = claude_launch_command(self.endpoint, self.agent_name) if cli == "claude" else codex_launch_command(self.endpoint, self.agent_name)
        if cli == "claude":
            self._claude_config_paths.append(command[3])
        self.terminal = PtyTerminal(command=command, cwd=os.getcwd(), parent=self)
        self.terminal_container.addWidget(self.terminal)
        QTimer.singleShot(1500, lambda: self.terminal.send_text(_OVERVIEW_PROMPT + "\n") if self.terminal else None)

    def _replace_terminal(self):
        if self.terminal is not None:
            self.terminal.close()
            self.terminal.deleteLater()
            self.terminal = None

    def start_claude(self):
        self._start("claude")

    def start_codex(self):
        self._start("codex")

    def closeEvent(self, event):
        self._replace_terminal()
        super().closeEvent(event)
