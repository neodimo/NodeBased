import json
import os
import tempfile
import unittest
import uuid
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QSettings

from nodebased import agent, issues, knowledge
from nodebased.core import Dispatcher
from nodebased.mcpserver import MCPServer


class MCPServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory()
        cls.old_format = QSettings.defaultFormat()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, cls.settings_dir.name)

    @classmethod
    def tearDownClass(cls):
        QSettings("NodeBased", "NodeBased").clear()
        QSettings.setDefaultFormat(cls.old_format)
        cls.settings_dir.cleanup()

    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.endpoint = "nodebased-mcp-test-" + uuid.uuid4().hex
        self.dispatcher = Dispatcher()
        self.dispatcher.execute({"op": "create", "id": "c", "type": "Constant"})
        self.calls = []

        def dispatch(command):
            self.calls.append(command)
            if command["op"] == "errors":
                return {"errors": [], "current_frame": 0}
            if command["op"] == "reference_context":
                return {"revision": self.dispatcher.revision, "prompt": command["prompt"], "artifacts": []}
            return self.dispatcher.execute(command)

        self.bridge = agent.LocalBridge(self.endpoint, dispatch)
        self.server = MCPServer(self.endpoint, "test-agent")

    def tearDown(self):
        if self.server.connection:
            self.server.connection.close()
        self.bridge.close()
        self.app.processEvents()

    def call(self, name, arguments=None, request_id=1):
        return self.server.dispatch({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                                     "params": {"name": name, "arguments": arguments or {}}})

    def test_initialize_and_notification(self):
        result = self.server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": {"protocolVersion": "2024-11-05"}})
        self.assertEqual(result["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(result["result"]["serverInfo"]["name"], "nodebased-mcp")
        self.assertEqual(self.server.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}), None)

    def test_tools_and_resources(self):
        result = self.server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual({tool["name"] for tool in result["result"]["tools"]},
                         {"describe", "inspect", "edit", "undo", "redo", "view", "errors",
                          "reference_context", "knowledge", "file_issue"})
        resources = self.server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
        self.assertEqual(len(resources["result"]["resources"]), len(knowledge.TOPICS))
        uri = "knowledge://overview"
        read = self.server.dispatch({"jsonrpc": "2.0", "id": 3, "method": "resources/read",
                                     "params": {"uri": uri}})
        self.assertEqual(read["result"]["contents"][0]["text"], knowledge.topic("overview"))

    def test_bridge_tools_proxy(self):
        for name, arguments, op in (
                ("describe", {}, "describe"), ("inspect", {}, "inspect"),
                ("undo", {}, "undo"), ("redo", {}, "redo"), ("view", {"id": "c"}, "view"),
                ("errors", {"since": 3}, "errors"),
                ("reference_context", {"prompt": "p", "directory": "/tmp"}, "reference_context")):
            self.calls.clear()
            result = self.call(name, arguments)
            self.assertFalse(result["result"]["isError"], name)
            self.assertEqual(self.calls[-1]["op"], op)

    def test_edit_validates_and_auto_inspects(self):
        self.calls.clear()
        result = self.call("edit", {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.4}]})
        self.assertFalse(result["result"]["isError"])
        self.assertEqual([call["op"] for call in self.calls], ["describe", "inspect", "batch"])
        self.assertEqual(self.calls[-1]["if_revision"], 1)

    def test_disk_ops_are_gated(self):
        self.calls.clear()
        result = self.call("edit", {"commands": [{"op": "save", "path": "/tmp/a.nbcomp"}]})
        self.assertTrue(result["result"]["isError"])
        self.assertNotIn("batch", [call["op"] for call in self.calls])
        QSettings("NodeBased", "NodeBased").setValue("agent/allow_disk_ops", True)
        result = self.call("edit", {"commands": [{"op": "save", "path": "/tmp/a.nbcomp"}]})
        self.assertFalse(result["result"]["isError"])
        # A lone save/load/render command is sent as its own top-level bridge request, never
        # nested inside "batch" (the real Dispatcher only accepts graph-editing ops in a batch).
        self.assertEqual(self.calls[-1]["op"], "save")

    def test_knowledge_without_bridge_and_live_state(self):
        offline = MCPServer("no-listener-" + uuid.uuid4().hex)
        result = offline.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": "knowledge", "arguments": {"topic": "overview"}}})
        self.assertFalse(result["result"]["isError"])
        result = self.call("knowledge", {"topic": "live_state"})
        self.assertFalse(result["result"]["isError"])
        self.assertIn("c", json.loads(result["result"]["content"][0]["text"])["nodes"])

    def test_file_issue_and_disabled_issue_filing(self):
        with mock.patch("nodebased.mcpserver.issues.file_issue", return_value={"number": 7}) as file_issue:
            result = self.call("file_issue", {"title": "T", "body": "B", "labels": ["bug"]})
            self.assertEqual(json.loads(result["result"]["content"][0]["text"]), {"number": 7})
            file_issue.assert_called_once()
        QSettings("NodeBased", "NodeBased").setValue("agent/file_issue_enabled", False)
        with mock.patch("nodebased.mcpserver.issues.file_issue") as file_issue:
            result = self.call("file_issue", {"title": "T", "body": "B"})
            self.assertTrue(result["result"]["isError"])
            file_issue.assert_not_called()

    def test_unknown_and_unreachable_errors(self):
        result = self.server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                       "params": {"name": "missing", "arguments": {}}})
        self.assertIn("Unknown tool", result["error"]["message"])
        result = self.server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "resources/read",
                                       "params": {"uri": "knowledge://missing"}})
        self.assertIn("Unknown resource", result["error"]["message"])
        offline = MCPServer("no-listener-" + uuid.uuid4().hex)
        result = offline.dispatch({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "inspect", "arguments": {}}})
        self.assertTrue(result["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
