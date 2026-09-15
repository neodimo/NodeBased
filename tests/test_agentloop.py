import copy
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased.core import Dispatcher
from nodebased import agentloop as al


def live_describe():
    return Dispatcher().execute({"op": "describe"})


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.describe = live_describe()
        self.dispatcher = Dispatcher()
        self.dispatcher.execute({"op": "create", "id": "c", "type": "Constant"})
        self.document = self.dispatcher.document

    def test_disallowed_ops_are_rejected(self):
        for op in ("save", "load", "render", "undo", "redo", "errors", "describe", "inspect",
                   "reference_context", "batch", "set_key", "set_shapes", "set_expression", "bogus"):
            proposal = {"commands": [{"op": op}], "done": False, "rationale": "x"}
            with self.subTest(op=op):
                with self.assertRaises(al.ProtocolError):
                    al.validate_proposal(proposal, self.describe, self.document)

    def test_non_object_commands_are_rejected(self):
        for garbage in ("nope", 1, None, ["op", "set"]):
            proposal = {"commands": [garbage], "done": False, "rationale": "x"}
            with self.subTest(garbage=garbage):
                with self.assertRaises(al.ProtocolError):
                    al.validate_proposal(proposal, self.describe, self.document)

    def test_valid_create_set_connect_batch_is_accepted(self):
        proposal = {"commands": [
            {"op": "create", "id": "v", "type": "Viewer"},
            {"op": "connect", "id": "v", "input": "image", "source": "c"},
            {"op": "set", "id": "c", "param": "red", "value": 0.5},
            {"op": "view", "id": "v"},
        ], "done": False, "rationale": "wire a viewer"}
        commands = al.validate_proposal(proposal, self.describe, self.document)
        self.assertEqual(len(commands), 4)

    def test_create_references_within_same_batch(self):
        proposal = {"commands": [
            {"op": "create", "id": "g", "type": "Grade"},
            {"op": "connect", "id": "g", "input": "image", "source": "c"},
        ], "done": False, "rationale": "grade the constant"}
        al.validate_proposal(proposal, self.describe, self.document)

    def test_unknown_node_type_is_rejected(self):
        proposal = {"commands": [{"op": "create", "type": "Nonexistent"}], "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "unknown node type"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_unknown_param_is_rejected(self):
        proposal = {"commands": [{"op": "set", "id": "c", "param": "not_a_param", "value": 1}],
                    "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "no parameter"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_unknown_node_id_is_rejected(self):
        proposal = {"commands": [{"op": "set", "id": "missing", "param": "red", "value": 1}],
                    "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "unknown node id"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_numeric_out_of_range_is_rejected(self):
        proposal = {"commands": [{"op": "set", "id": "c", "param": "alpha", "value": 5}],
                    "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "outside"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_invalid_choice_is_rejected(self):
        self.dispatcher.execute({"op": "create", "id": "r", "type": "Read"})
        proposal = {"commands": [{"op": "set", "id": "r", "param": "colorspace", "value": "Not A Space"}],
                    "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "not one of"):
            al.validate_proposal(proposal, self.describe, self.dispatcher.document)

    def test_disabling_a_source_node_is_rejected(self):
        proposal = {"commands": [{"op": "disable", "id": "c", "value": True}], "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "cannot be disabled"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_if_revision_on_member_command_is_rejected(self):
        proposal = {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.1, "if_revision": 0}],
                    "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "if_revision belongs on the outer batch"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_missing_done_or_bad_rationale_is_rejected(self):
        with self.assertRaisesRegex(al.ProtocolError, "'done'"):
            al.validate_proposal({"commands": [], "rationale": "x"}, self.describe, self.document)
        with self.assertRaisesRegex(al.ProtocolError, "'rationale'"):
            al.validate_proposal({"commands": [], "done": True, "rationale": 5}, self.describe, self.document)

    def test_command_cap_is_enforced(self):
        commands = [{"op": "set", "id": "c", "param": "red", "value": 0.1}] * (al.MAX_COMMANDS_PER_BATCH + 1)
        proposal = {"commands": commands, "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "command-cap|commands, exceeding"):
            al.validate_proposal(proposal, self.describe, self.document)

    def test_time_op_validates_fields_and_ranges(self):
        proposal = {"commands": [{"op": "time", "current": 5}], "done": False, "rationale": "x"}
        al.validate_proposal(proposal, self.describe, self.document)
        proposal = {"commands": [{"op": "time", "bogus": 1}], "done": False, "rationale": "x"}
        with self.assertRaisesRegex(al.ProtocolError, "unknown field"):
            al.validate_proposal(proposal, self.describe, self.document)


class JsonParsingTests(unittest.TestCase):
    def test_plain_json(self):
        data = al._parse_json_response('{"commands": [], "done": true, "rationale": "ok"}')
        self.assertEqual(data, {"commands": [], "done": True, "rationale": "ok"})

    def test_fenced_json_with_language_tag(self):
        text = 'Sure, here it is:\n```json\n{"commands": [], "done": true, "rationale": "ok"}\n```\nThanks.'
        data = al._parse_json_response(text)
        self.assertEqual(data["done"], True)

    def test_fenced_json_without_language_tag(self):
        text = '```\n{"commands": [], "done": false, "rationale": "still working"}\n```'
        data = al._parse_json_response(text)
        self.assertEqual(data["rationale"], "still working")

    def test_garbage_raises_protocol_error(self):
        with self.assertRaises(al.ProtocolError):
            al._parse_json_response("this is not json at all")

    def test_oversized_response_raises_protocol_error(self):
        text = "x" * (al.MAX_PROVIDER_RESPONSE_CHARS + 1)
        with self.assertRaisesRegex(al.ProtocolError, "exceeding the"):
            al._parse_json_response(text)


class AnthropicProviderTests(unittest.TestCase):
    def test_missing_api_key_raises(self):
        with self.assertRaises(ValueError):
            al.AnthropicProvider(env={})

    def test_reads_key_only_from_env(self):
        provider = al.AnthropicProvider(env={"ANTHROPIC_API_KEY": "sk-test-123"})
        self.assertEqual(provider.api_key, "sk-test-123")
        self.assertEqual(provider.model, al.AnthropicProvider.DEFAULT_MODEL)

    def test_build_request_includes_schema_images_and_history(self):
        provider = al.AnthropicProvider(env={"ANTHROPIC_API_KEY": "sk-test-123"}, model="claude-sonnet-5")
        describe = live_describe()
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "capture-01.png"
            raw_bytes = b"\x89PNG\r\n\x1a\nfakepixels"
            image_path.write_bytes(raw_bytes)
            context = al.ProposalContext(
                prompt="match this still life", describe=describe,
                document=Dispatcher().document, revision=0,
                images=[{"id": "v", "name": "Viewer", "roles": ["view"], "path": str(image_path)}],
                history=[{"index": 1, "rationale": "prior step"}])
            request = provider.build_request(context)
        self.assertEqual(request["model"], "claude-sonnet-5")
        self.assertIn("create", request["system"])
        content = request["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("match this still life", content[0]["text"])
        self.assertIn("prior step", content[0]["text"])
        image_blocks = [b for b in content if b["type"] == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/png")
        decoded = __import__("base64").b64decode(image_blocks[0]["source"]["data"])
        self.assertEqual(decoded, raw_bytes)

    def test_propose_sends_request_and_parses_response(self):
        provider = al.AnthropicProvider(env={"ANTHROPIC_API_KEY": "sk-test-123"})
        describe = live_describe()
        context = al.ProposalContext(prompt="p", describe=describe, document=Dispatcher().document, revision=0)

        response_body = json.dumps({
            "content": [{"type": "text",
                         "text": '```json\n{"commands": [{"op": "set", "id": "c", "param": "red", '
                                 '"value": 0.4}], "done": false, "rationale": "brighten"}\n```'}]}).encode()

        class FakeResponse:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return response_body

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["request"] = req
            captured["timeout"] = timeout
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            result = provider.propose(context)

        self.assertEqual(result, {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.4}],
                                   "done": False, "rationale": "brighten"})
        sent_request = captured["request"]
        self.assertEqual(sent_request.get_header("X-api-key"), "sk-test-123")
        self.assertEqual(sent_request.get_header("Anthropic-version"), al.AnthropicProvider.API_VERSION)
        self.assertEqual(captured["timeout"], provider.timeout)

    def test_propose_raises_on_http_error(self):
        provider = al.AnthropicProvider(env={"ANTHROPIC_API_KEY": "sk-test-123"})
        describe = live_describe()
        context = al.ProposalContext(prompt="p", describe=describe, document=Dispatcher().document, revision=0)

        import urllib.error
        import io

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(al.AnthropicProvider.API_URL, 429, "rate limited",
                                          hdrs=None, fp=io.BytesIO(b"slow down"))

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(al.ProtocolError, "429"):
                provider.propose(context)

    def test_propose_raises_on_malformed_response(self):
        provider = al.AnthropicProvider(env={"ANTHROPIC_API_KEY": "sk-test-123"})
        describe = live_describe()
        context = al.ProposalContext(prompt="p", describe=describe, document=Dispatcher().document, revision=0)
        response_body = json.dumps({"content": [{"type": "text", "text": "not json"}]}).encode()

        class FakeResponse:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return response_body

        with mock.patch("urllib.request.urlopen", lambda req, timeout=None: FakeResponse()):
            with self.assertRaises(al.ProtocolError):
                provider.propose(context)


class ScriptedProviderTests(unittest.TestCase):
    def test_replays_proposals_then_reports_done(self):
        provider = al.ScriptedProvider([
            {"commands": [{"op": "set", "id": "c", "param": "red", "value": 1}], "done": False, "rationale": "a"},
            {"commands": [], "done": True, "rationale": "b"},
        ])
        ctx = al.ProposalContext(prompt="p", describe={}, document={}, revision=0)
        first = provider.propose(ctx)
        second = provider.propose(ctx)
        third = provider.propose(ctx)
        self.assertEqual(first["rationale"], "a")
        self.assertEqual(second["rationale"], "b")
        self.assertEqual(third, {"commands": [], "done": True, "rationale": "scripted sequence exhausted"})


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "offscreen UI test")
class AgentLoopIntegrationTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from nodebased.app import Window
        from nodebased.core import empty_document
        self.app = QApplication.instance() or QApplication([])
        self.endpoint = "nodebased-agentloop-test-" + uuid.uuid4().hex
        document = empty_document()
        document["nodes"] = {
            "c": {"type": "Constant", "name": "Constant", "pos": [0, 0], "disabled": False,
                  "inputs": {}, "params": {"width": 4, "height": 3, "red": 0.1, "green": 0.1,
                                            "blue": 0.1, "alpha": 1.0}},
            "v": {"type": "Viewer", "name": "Viewer", "pos": [200, 0], "disabled": False,
                  "inputs": {"image": "c"}, "params": {}},
        }
        document["view"] = "v"
        self.window = Window(document, self.endpoint)
        self.window.show()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.executor.shutdown(wait=True, cancel_futures=True)
        self.app.processEvents()

    def _make_loop(self, provider, **kwargs):
        connection = al.Connection(self.endpoint)
        loop = al.AgentLoop(connection, provider, "make it brighter and add a grade", auto_confirm=True,
                             report=lambda record: None, emit=lambda message: None, **kwargs)
        return connection, loop

    def test_two_iterations_create_and_adjust_with_one_undo_step_each(self):
        provider = al.ScriptedProvider([
            {"commands": [
                {"op": "create", "id": "g", "type": "Grade"},
                {"op": "connect", "id": "g", "input": "image", "source": "c"},
                {"op": "connect", "id": "v", "input": "image", "source": "g"},
            ], "done": False, "rationale": "insert a grade"},
            {"commands": [{"op": "set", "id": "g", "param": "exposure", "value": 1.2}],
             "done": True, "rationale": "brighten via the grade"},
        ])
        connection, loop = self._make_loop(provider, iterations=3)
        try:
            history = loop.run()
        finally:
            connection.close()

        self.assertEqual(len(history), 2)
        self.assertTrue(all(record["applied"] for record in history))
        document = self.window.dispatcher.document
        self.assertIn("g", document["nodes"])
        self.assertEqual(document["nodes"]["v"]["inputs"]["image"], "g")
        self.assertEqual(document["nodes"]["g"]["params"]["exposure"], 1.2)
        self.assertTrue(history[1]["done"])

        after_both = copy.deepcopy(document)
        self.window.command({"op": "undo"}, render=False)
        after_one_undo = self.window.dispatcher.document
        self.assertNotEqual(after_one_undo, after_both)
        self.assertEqual(after_one_undo["nodes"]["g"]["params"]["exposure"], 0.0)
        self.assertIn("g", after_one_undo["nodes"])

        self.window.command({"op": "undo"}, render=False)
        after_two_undos = self.window.dispatcher.document
        self.assertNotIn("g", after_two_undos["nodes"])
        self.assertEqual(after_two_undos["nodes"]["v"]["inputs"]["image"], "c")

    def test_stale_revision_is_retried_exactly_once_then_succeeds(self):
        window = self.window

        class RaceProvider(al.Provider):
            """Mimics a concurrent human edit landing between this loop's inspect and its batch."""
            def __init__(self):
                self.calls = 0

            def propose(self, context):
                self.calls += 1
                if self.calls == 1:
                    # Mutate the live document out of band so the loop's first batch attempt
                    # (built from the revision it just inspected) is stale by the time it applies.
                    window.dispatcher.execute({"op": "rename", "id": "c", "name": "Renamed Externally"})
                    return {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.9}],
                            "done": False, "rationale": "brighten (first attempt)"}
                return {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.9}],
                        "done": True, "rationale": "brighten (retry)"}

        provider = RaceProvider()
        connection, loop = self._make_loop(provider, iterations=3)
        try:
            history = loop.run()
        finally:
            connection.close()

        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["applied"])
        self.assertEqual(window.dispatcher.document["nodes"]["c"]["params"]["red"], 0.9)
        self.assertEqual(window.dispatcher.document["nodes"]["c"]["name"], "Renamed Externally")

    def test_stale_revision_that_fails_twice_stops_with_clear_error(self):
        window = self.window

        class AlwaysRacingProvider(al.Provider):
            """Every call mutates the document again, so the retried batch is stale too."""
            def __init__(self):
                self.calls = 0

            def propose(self, context):
                self.calls += 1
                window.dispatcher.execute({"op": "rename", "id": "c", "name": f"Renamed {self.calls}"})
                return {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.9}],
                        "done": False, "rationale": "brighten"}

        provider = AlwaysRacingProvider()
        connection, loop = self._make_loop(provider, iterations=3)
        try:
            with self.assertRaises(al.StaleRevisionError):
                loop.run()
        finally:
            connection.close()
        self.assertEqual(provider.calls, 2)

    def test_dry_run_applies_nothing(self):
        before = copy.deepcopy(self.window.dispatcher.document)
        before_revision = self.window.dispatcher.revision
        provider = al.ScriptedProvider([
            {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.99}],
             "done": False, "rationale": "would brighten"},
        ])
        connection, loop = self._make_loop(provider, iterations=2, dry_run=True)
        try:
            history = loop.run()
        finally:
            connection.close()
        self.assertTrue(all(not record["applied"] for record in history))
        self.assertTrue(all(record["dry_run"] for record in history))
        self.assertEqual(self.window.dispatcher.document, before)
        self.assertEqual(self.window.dispatcher.revision, before_revision)

    def test_decline_stops_the_loop_without_applying(self):
        provider = al.ScriptedProvider([
            {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.99}],
             "done": False, "rationale": "would brighten"},
            {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.5}],
             "done": False, "rationale": "second step, never reached"},
        ])
        before = copy.deepcopy(self.window.dispatcher.document)
        connection = al.Connection(self.endpoint)
        loop = al.AgentLoop(connection, provider, "brighten it", iterations=3,
                             confirm=lambda record: False,
                             report=lambda record: None, emit=lambda message: None)
        try:
            history = loop.run()
        finally:
            connection.close()
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["declined"])
        self.assertFalse(history[0]["applied"])
        self.assertEqual(self.window.dispatcher.document, before)

    def test_iteration_cap_is_respected(self):
        provider = al.ScriptedProvider([
            {"commands": [{"op": "set", "id": "c", "param": "red", "value": v}],
             "done": False, "rationale": f"step {v}"}
            for v in (0.2, 0.3, 0.4, 0.5, 0.6)
        ])
        connection, loop = self._make_loop(provider, iterations=2)
        try:
            history = loop.run()
        finally:
            connection.close()
        self.assertEqual(len(history), 2)
        self.assertEqual(self.window.dispatcher.document["nodes"]["c"]["params"]["red"], 0.3)

    def test_cli_dry_run_end_to_end_applies_nothing(self):
        before = copy.deepcopy(self.window.dispatcher.document)
        with tempfile.TemporaryDirectory() as folder:
            script_path = Path(folder) / "script.json"
            script_path.write_text(json.dumps([
                {"commands": [{"op": "set", "id": "c", "param": "red", "value": 0.77}],
                 "done": True, "rationale": "cli dry run"},
            ]), encoding="utf-8")
            exit_code = al.main(["--connect", self.endpoint, "--prompt", "brighten it",
                                 "--provider", "scripted", "--script", str(script_path),
                                 "--dry-run", "--iterations", "1"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.window.dispatcher.document, before)


if __name__ == "__main__":
    unittest.main()
