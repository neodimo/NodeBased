"""A headless MCP client and session for scripted agents.

`McpClient` speaks JSON-RPC over stdio to any MCP server command (the same protocol Claude Code or
Codex use with `python -m nodebased.mcpserver`). `HeadlessSession` starts the whole attached-agent
setup without a display: an offscreen NodeBased window that opens the `--agent` bridge on a document,
and the stdio MCP server pointed at that bridge. Everything an agent does then goes through the
window's `agent_command`, the one path a human edit and an attached CLI share.

The "agent" scripts built on this (examples/demo_shot/run_by_agent.py, the stage 1 test) have no
model in the loop: they are fixed plans, so a run tests the tools and the contract, not a model.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class ToolError(RuntimeError):
    """A tool call came back with isError set."""


class McpClient:
    """Newline-delimited JSON-RPC client for one MCP server subprocess."""

    def __init__(self, command, env=None, cwd=None):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                        env=env, cwd=cwd)
        self._next_id = 0
        self.log = []  # (tool, arguments) in call order: the tool sequence the plan chose

    def request(self, method, params=None):
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed its output")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']['message']}")
        return reply["result"]

    def notify(self, method, params=None):
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}) + "\n")
        self.process.stdin.flush()

    def initialize(self):
        result = self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                             "clientInfo": {"name": "nodebased-scripted-agent", "version": "1"}})
        self.notify("notifications/initialized")
        return result

    def tools(self):
        return [tool["name"] for tool in self.request("tools/list")["tools"]]

    def call_text(self, name, **arguments):
        """Call a tool; return its text, or raise ToolError."""
        self.log.append((name, arguments))
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = result["content"][0]["text"]
        if result["isError"]:
            raise ToolError(f"{name}: {text}")
        return text

    def call(self, name, **arguments):
        """Call a tool and parse its JSON reply (`knowledge` replies are text: use call_text)."""
        return json.loads(self.call_text(name, **arguments))

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()
        self.process.stdout.close()


class HeadlessSession:
    """An offscreen NodeBased window plus its MCP server, torn down on exit.

    `cache` becomes NODEBASED_CACHE (the artifact store); `disk_ops` sets the "Allow save / load /
    render from the agent" setting in a private settings home, so the user's real settings are
    untouched."""

    def __init__(self, project, cache, disk_ops=True, timeout=120, attach_client=True):
        self.project, self.cache, self.timeout = Path(project), Path(cache), timeout
        self.home = tempfile.TemporaryDirectory(prefix="nodebased-agent-home-")
        settings = Path(self.home.name) / ".config" / "NodeBased"
        settings.mkdir(parents=True)
        (settings / "NodeBased.conf").write_text(
            f"[agent]\nallow_disk_ops={'true' if disk_ops else 'false'}\n", encoding="utf-8")
        self.endpoint = "nodebased-headless-" + uuid.uuid4().hex
        self.env = {**os.environ, "HOME": self.home.name, "XDG_CONFIG_HOME": str(Path(self.home.name) / ".config"),
                    "NODEBASED_AGENT_DISK_OPS": "true" if disk_ops else "false",
                    "NODEBASED_CACHE": str(self.cache), "QT_QPA_PLATFORM": "offscreen",
                    "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        self.window = subprocess.Popen([sys.executable, "-m", "nodebased", str(self.project),
                                        "--agent", self.endpoint], env=self.env, cwd=str(REPO),
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.client = None
        if attach_client:  # False: another client (a real Claude Code session) attaches to the endpoint itself
            self.client = McpClient([sys.executable, "-m", "nodebased.mcpserver", "--endpoint", self.endpoint,
                                     "--agent-name", "scripted-agent"], env=self.env, cwd=str(REPO))
            self.client.initialize()
            self._wait_for_window()

    def _wait_for_window(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.client.call("inspect")
                self.client.log.clear()  # the start-up poll is not part of the plan's tool sequence
                return
            except Exception as error:  # the window is still starting: its endpoint is not open yet
                if self.window.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"NodeBased window did not come up: {error}")
                time.sleep(0.5)

    def __enter__(self):
        return self.client

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.client:
            self.client.close()
        self.window.terminate()
        try:
            self.window.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.window.kill()
        self.home.cleanup()
