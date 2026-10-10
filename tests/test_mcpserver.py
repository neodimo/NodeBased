import io
import json
import zipfile
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
                          "reference_context", "knowledge", "file_issue", "export_conditioning",
                          "generate", "verify", "loop", "artifacts", "workspace"})
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
        # The allowed save really writes the document, so the path has to exist on every
        # platform: /tmp is not a directory on Windows.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "a.nbcomp")
        result = self.call("edit", {"commands": [{"op": "save", "path": path}]})
        self.assertTrue(result["result"]["isError"])
        self.assertNotIn("batch", [call["op"] for call in self.calls])
        QSettings("NodeBased", "NodeBased").setValue("agent/allow_disk_ops", True)
        result = self.call("edit", {"commands": [{"op": "save", "path": path}]})
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


class MCPPipelineTests(unittest.TestCase):
    """A scripted MCP client delivers the smoke demo from GENERATIVE_CONDITIONING.md through the
    tools only, against a real (offscreen) window so every call crosses Window.agent_command."""

    @classmethod
    def setUpClass(cls):
        from tests.test_desktop import APP, release_window, wait_until
        cls.app, cls.release_window, cls.wait_until = APP, staticmethod(release_window), staticmethod(wait_until)

    def setUp(self):
        from nodebased.app import Window
        QSettings("NodeBased", "NodeBased").clear()
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)
        patcher = mock.patch.dict(os.environ, {"NODEBASED_CACHE": self.cache.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.endpoint = "nodebased-mcp-pipeline-" + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(self.wait_until(lambda: self.window.frame is not None))
        self.server = MCPServer(self.endpoint, "test-agent")
        self.addCleanup(self.teardown_window)
        self.build_scene()

    def teardown_window(self):
        if self.server.connection:
            self.server.connection.close()
        self.window.saved_document = self.window.dispatcher.document
        self.window.queue_panel.runner.shutdown(wait=True)
        self.window.close()
        self.app.processEvents()
        self.release_window(self)

    def call(self, name, arguments=None):
        reply = self.server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                      "params": {"name": name, "arguments": arguments or {}}})
        return reply["result"]

    def ok(self, name, arguments=None):
        result = self.call(name, arguments)
        self.assertFalse(result["isError"], result["content"][0]["text"])
        return json.loads(result["content"][0]["text"])

    def build_scene(self):
        self.ok("edit", {"commands": [
            {"op": "create", "id": "cube", "type": "Cube3D", "params": {"cube_size": 1}},
            {"op": "create", "id": "scene", "type": "Scene3D"},
            {"op": "create", "id": "camera", "type": "Camera3D", "params": {"tz": 7}},
            {"op": "create", "id": "render", "type": "Render3D",
             "params": {"width": 160, "height": 120, "samples": 1}},
            {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
            {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "render", "input": "camera", "source": "camera"}]})

    def wait_loop(self, loop_id):
        status = {}
        def finished():
            nonlocal status
            status = self.ok("loop", {"action": "status", "loop": loop_id})
            return status["finished"]
        self.assertTrue(self.wait_until(finished), status)
        return status

    def export(self):
        exported = self.ok("export_conditioning", {"id": "render", "first": 1, "last": 1})
        return exported["scene_state"], exported["control_bundle"]

    def test_smoke_demo_through_tools_only(self):
        scene_id, bundle_id = self.export()
        started = self.ok("loop", {"action": "start", "scene_state": scene_id, "bundle": bundle_id,
                                   "provider": "null", "max_attempts": 2, "spend_cap": 0.02,
                                   "estimate": 0.01, "unit": "credits (estimate)"})
        status = self.wait_loop(started["loop"])
        # The unconditioned null baseline carries beauty only: the checks it can answer pass, the
        # depth, motion and normals checks are labelled "not conditioned", and the first attempt
        # passes, so the loop stops after one attempt and one estimate.
        self.assertEqual(status["stop_reason"], "pass")
        self.assertEqual([a["state"] for a in status["attempts"]], ["pass"])
        self.assertEqual([a["seed"] for a in status["attempts"]], [0])
        self.assertAlmostEqual(status["cumulative_estimate"], 0.01)
        self.assertEqual(status["controls"]["seed"], "used")
        self.assertEqual(status["controls"]["sampler_settings"], "ignored")
        generated = status["attempts"][-1]["generated_sequence"]
        used = self.ok("loop", {"action": "use_result", "loop": started["loop"]})
        self.assertEqual(used["path"], generated)
        self.ok("edit", {"commands": [{"op": "create", "id": "result", "type": "ConditionedRead",
                                       "params": used["node"]["params"]}]})
        document = self.ok("inspect")["document"]
        self.assertEqual(document["nodes"]["result"]["params"]["path"], generated)
        chain = self.ok("artifacts", {"action": "provenance", "id": generated})["chain"]
        ids = [row["id"] for row in chain]
        self.assertIn(scene_id, ids)
        self.assertIn(bundle_id, ids)
        listed = self.ok("artifacts", {"action": "list", "kind": "generated_sequence"})["artifacts"]
        self.assertIn(generated, [row["id"] for row in listed])
        self.ok("view", {"id": "result"})

    def test_same_brief_twice_is_byte_identical_and_only_the_seed_changes_the_id(self):
        """Stage 2 determinism proof through the tools: the same brief gives the same artifact id and
        the same frame bytes; changing only the seed changes the id. The stand-ins ignore the
        seed's value, so the frames stay identical; the id differs because the hashed generation
        manifest records the seed."""
        scene_id, bundle_id = self.export()
        for provider in ("null", "reproject"):
            brief = {"scene_state": scene_id, "bundle": bundle_id, "provider": provider, "seed": 1234}
            first, second = self.ok("generate", brief), self.ok("generate", brief)
            self.assertEqual(first["generated_sequence"], second["generated_sequence"])
            self.assertEqual((first["seed"], first["version"]), (1234, "1.0"))
            self.assertEqual(first["controls"]["seed"], "used")
            self.assertEqual(first["controls"]["sampler_settings"], "ignored")
            other = self.ok("generate", {**brief, "seed": 1235})
            self.assertNotEqual(first["generated_sequence"], other["generated_sequence"])
            store = self.window.queue_panel.queue.store
            def frames(result):
                with zipfile.ZipFile(io.BytesIO(store.get(result["generated_sequence"]))) as archive:
                    return {n: archive.read(n) for n in archive.namelist() if n.endswith(".exr")}
            self.assertEqual(frames(first), frames(second))
            self.assertEqual(frames(first), frames(other))
            chain = self.ok("artifacts", {"action": "provenance", "id": first["generated_sequence"]})["chain"]
            self.assertEqual((chain[-1]["provenance"]["seed"], chain[-1]["provenance"]["provider"]),
                             (1234, provider))
            verdict = self.ok("verify", {"scene_state": scene_id, "bundle": bundle_id,
                                         "generated_sequence": first["generated_sequence"]})
            self.assertEqual(verdict["determinism"], {"provider": provider, "version": "1.0", "seed": 1234,
                                                      "sampler_settings": {}, "seed_honoured": True})
        refused = self.call("generate", {"scene_state": scene_id, "bundle": bundle_id, "provider": "null",
                                         "sampler_settings": {"steps": 4}})
        self.assertTrue(refused["isError"])
        self.assertIn("sampler_settings", refused["content"][0]["text"])
        self.assertTrue(self.call("generate", {"scene_state": scene_id, "bundle": bundle_id,
                                               "provider": "null", "seed": -1})["isError"])

    def test_loop_reports_seed_per_attempt(self):
        scene_id, bundle_id = self.export()
        started = self.ok("loop", {"action": "start", "scene_state": scene_id, "bundle": bundle_id,
                                   "provider": "null", "max_attempts": 1, "spend_cap": 1, "estimate": 0.5,
                                   "unit": "credits (estimate)", "seed": 77, "vary_seed": True})
        status = self.wait_loop(started["loop"])
        self.assertEqual((status["seed"], status["vary_seed"]), (77, True))
        self.assertEqual([a["seed"] for a in status["attempts"]], [77])

    def test_generate_and_verify_are_synchronous(self):
        scene_id, bundle_id = self.export()
        generated = self.ok("generate", {"scene_state": scene_id, "bundle": bundle_id,
                                         "provider": "null"})["generated_sequence"]
        verdict = self.ok("verify", {"scene_state": scene_id, "bundle": bundle_id,
                                     "generated_sequence": generated})
        self.assertIn(verdict["verdict"], ("PASS", "FAIL"))
        self.assertIn("bindings", verdict["score_card"])
        self.assertEqual(self.ok("artifacts", {"action": "provenance", "id": verdict["report"]})["chain"][-1]["id"],
                         verdict["report"])

    def test_loop_spend_cap_and_cancel(self):
        scene_id, bundle_id = self.export()
        args = {"action": "start", "scene_state": scene_id, "bundle": bundle_id, "provider": "null",
                "max_attempts": 2, "spend_cap": 0.005, "estimate": 0.01, "unit": "credits (estimate)"}
        status = self.wait_loop(self.ok("loop", args)["loop"])
        self.assertEqual(status["stop_reason"], "budget_exhausted")
        self.assertEqual(status["attempts"], [])
        self.assertTrue(self.call("loop", {"action": "cancel"})["isError"])

    def test_output_pattern_needs_disk_ops(self):
        scene_id, bundle_id = self.export()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        arguments = {"scene_state": scene_id, "bundle": bundle_id, "provider": "null",
                     "output_pattern": os.path.join(folder.name, "out.####.exr")}
        refused = self.call("generate", arguments)
        self.assertTrue(refused["isError"])
        self.assertIn("allow_disk_ops", refused["content"][0]["text"])
        self.assertEqual(os.listdir(folder.name), [])
        QSettings("NodeBased", "NodeBased").setValue("agent/allow_disk_ops", True)
        self.ok("generate", arguments)
        self.assertTrue(any(name.endswith(".exr") for name in os.listdir(folder.name)))

    def test_bad_arguments_are_errors(self):
        self.assertTrue(self.call("generate", {"scene_state": "x", "bundle": "y", "provider": "null"})["isError"])
        scene_id, bundle_id = self.export()
        self.assertTrue(self.call("generate", {"scene_state": scene_id, "bundle": bundle_id,
                                               "provider": "../../etc/passwd"})["isError"])
        self.assertTrue(self.call("export_conditioning", {"id": "missing"})["isError"])
        self.assertTrue(self.call("workspace", {"tab": "Nope"})["isError"])

    def test_workspace_switches_tab(self):
        self.assertEqual(self.ok("workspace", {"tab": "3D"})["tab"], "3D")
        self.assertEqual(self.window.workspace_tab, "3D")
        self.assertEqual(self.ok("workspace", {"tab": "Composite"})["tab"], "Composite")

    def test_every_new_tool_shows_in_the_activity_log(self):
        scene_id, bundle_id = self.export()
        generated = self.ok("generate", {"scene_state": scene_id, "bundle": bundle_id,
                                         "provider": "null"})["generated_sequence"]
        self.ok("verify", {"scene_state": scene_id, "bundle": bundle_id, "generated_sequence": generated})
        self.ok("loop", {"action": "status"})
        self.ok("artifacts", {"action": "list"})
        self.ok("workspace", {"tab": "Render"})
        log = self.window.agent_panel.activity_log
        lines = [log.item(i).text() for i in range(log.count())]
        for name in ("export_conditioning", "generate", "verify", "loop", "artifacts", "workspace"):
            self.assertTrue(any(f"| {name}(" in line and line.endswith("-> ok") for line in lines), name)
        self.assertTrue(self.call("generate", {"scene_state": "bad", "bundle": bundle_id,
                                               "provider": "null"})["isError"])
        self.assertTrue(log.item(log.count() - 1).text().rsplit("-> ", 1)[1].startswith("error:"))


if __name__ == "__main__":
    unittest.main()
