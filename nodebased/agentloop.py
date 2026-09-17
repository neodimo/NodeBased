"""External client that closes the v10 reference-image loop.

Connects to a running GUI's ``--agent`` local endpoint (the same QLocalSocket transport
``nodebased.agent`` uses), captures reference-context PNGs, asks a provider to propose graph
edits, validates them, and applies them as one guarded atomic ``batch`` per iteration. NodeBased
itself makes no model or network calls; only the provider objects in this module (concretely
``AnthropicProvider``) may reach the network, and only when this client runs them.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication
from PySide6.QtNetwork import QLocalSocket

MAX_REQUEST = 1024 * 1024  # mirrors nodebased.agent.MAX_REQUEST; the shared transport cap.
DEFAULT_ITERATIONS = 3
MAX_ITERATIONS = 10
MAX_COMMANDS_PER_BATCH = 64
MAX_IMAGES = 8
MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PROVIDER_RESPONSE_CHARS = 200_000

# Only edit operations that mutate the graph a reference-matching loop should ever need. Every
# other protocol op (save/load/render/undo/redo/errors/describe/inspect/reference_context/batch
# and the animation/expression/shape/track family) is refused before a batch is ever sent, even
# though the server would reject most of them from inside a batch anyway.
ALLOWED_EDIT_OPS = frozenset({
    "create", "set", "connect", "move", "rename", "label", "thumbnail", "disable", "delete",
    "reference", "view", "time",
})


class ProtocolError(RuntimeError):
    """A malformed provider response, a validation failure, or a rejected batch."""


class StaleRevisionError(ProtocolError):
    """The server rejected a batch because the document changed since the last inspect."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class Connection:
    """Synchronous newline-JSON client over the ``--agent`` QLocalSocket transport."""

    def __init__(self, name, timeout_ms=30000):
        self.socket = QLocalSocket()
        self.socket.connectToServer(name)
        if not self.socket.waitForConnected(5000):
            raise ConnectionError(f"Cannot connect to agent endpoint {name!r}: {self.socket.errorString()}")
        self.timeout_ms = timeout_ms
        self._buffer = bytearray()

    def request(self, cmd):
        payload = json.dumps(cmd).encode() + b"\n"
        if len(payload) > MAX_REQUEST:
            raise ProtocolError("Request exceeds 1 MiB")
        self.socket.write(payload)
        self.socket.flush()
        deadline = time.monotonic() + self.timeout_ms / 1000
        # A plain blocking waitForReadyRead() only watches this socket's own file descriptor; in
        # a real --connect process that is enough because the GUI process pumps its own event
        # loop independently. In an offscreen test the GUI's LocalBridge server lives in this
        # same thread, so its newConnection/readyRead signals only fire when something here
        # drives the event queue -- hence the short poll interleaved with processEvents().
        while b"\n" not in self._buffer:
            QCoreApplication.processEvents()
            if not self.socket.bytesAvailable() and not self.socket.waitForReadyRead(20):
                if time.monotonic() >= deadline:
                    raise ConnectionError("Agent response timed out or disconnected")
                continue
            self._buffer.extend(bytes(self.socket.readAll()))
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        response = json.loads(line)
        if not response.get("ok"):
            raise ProtocolError(str(response.get("error", "agent request failed")))
        return response["result"]

    def close(self):
        self.socket.disconnectFromServer()


# ---------------------------------------------------------------------------
# Client-side validation
# ---------------------------------------------------------------------------

def _validate_range(name, value, limits, index):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError(f"command {index}: {name!r} must be numeric")
    bounds = limits.get(name)
    if bounds is not None and not (bounds[0] <= value <= bounds[1]):
        raise ProtocolError(f"command {index}: {name}={value} is outside [{bounds[0]}, {bounds[1]}]")


def _validate_param_value(name, value, limits, choices, index):
    if name in choices:
        if not isinstance(value, str) or value not in choices[name]:
            raise ProtocolError(f"command {index}: {name}={value!r} is not one of {choices[name]}")
        return
    if name in limits:
        _validate_range(name, value, limits, index)


def _validate_pos(cmd, index):
    pos = cmd.get("pos")
    if pos is None:
        return
    if not isinstance(pos, list) or len(pos) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pos):
        raise ProtocolError(f"command {index}: pos must be [x, y]")


def _validate_command(cmd, describe, known_types, index):
    """known_types maps node id -> node type for every id that exists before this command, and is
    updated in place so later commands in the same batch may reference nodes created earlier in it."""
    if not isinstance(cmd, dict):
        raise ProtocolError(f"command {index}: must be a JSON object")
    op = cmd.get("op")
    if op not in ALLOWED_EDIT_OPS:
        raise ProtocolError(
            f"command {index}: op {op!r} is not permitted; allowed ops are {sorted(ALLOWED_EDIT_OPS)}")
    if "if_revision" in cmd:
        raise ProtocolError(f"command {index}: if_revision belongs on the outer batch, not a member command")

    nodes_spec = describe.get("nodes", {})
    limits = describe.get("limits", {})
    choices = describe.get("choices", {})

    if op == "create":
        kind = cmd.get("type")
        if not isinstance(kind, str) or kind not in nodes_spec:
            raise ProtocolError(f"command {index}: unknown node type {kind!r}")
        params = cmd.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ProtocolError(f"command {index}: params must be an object")
        allowed_params = nodes_spec[kind]["params"]
        for name, value in params.items():
            if name not in allowed_params:
                raise ProtocolError(f"command {index}: {kind} has no parameter {name!r}")
            _validate_param_value(name, value, limits, choices, index)
        _validate_pos(cmd, index)
        node_id = cmd.get("id")
        if node_id is not None:
            if not isinstance(node_id, str) or not node_id:
                raise ProtocolError(f"command {index}: id must be a non-empty string")
            if node_id in known_types:
                raise ProtocolError(f"command {index}: id {node_id!r} already exists")
            known_types[node_id] = kind
        return

    if op == "view":
        node_id = cmd.get("id")
        if node_id is not None and (not isinstance(node_id, str) or node_id not in known_types):
            raise ProtocolError(f"command {index}: view references unknown node id {node_id!r}")
        return

    if op == "time":
        fields = {"first", "last", "current", "fps"}
        unknown = set(cmd) - fields - {"op"}
        if unknown:
            raise ProtocolError(f"command {index}: time has unknown field(s) {sorted(unknown)}")
        if not (set(cmd) & fields):
            raise ProtocolError(f"command {index}: time requires at least one of {sorted(fields)}")
        time_limits = describe.get("time_limits", {})
        for name in fields & set(cmd):
            _validate_range(name, cmd[name], time_limits, index)
        return

    node_id = cmd.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise ProtocolError(f"command {index}: id must be a non-empty string")
    if node_id not in known_types:
        raise ProtocolError(f"command {index}: unknown node id {node_id!r}")
    kind = known_types[node_id]

    if op == "set":
        name = cmd.get("param")
        if name not in nodes_spec.get(kind, {}).get("params", {}):
            raise ProtocolError(f"command {index}: {kind} has no parameter {name!r}")
        if "value" not in cmd:
            raise ProtocolError(f"command {index}: set requires 'value'")
        _validate_param_value(name, cmd["value"], limits, choices, index)
    elif op == "connect":
        slot = cmd.get("input")
        inputs = list(nodes_spec.get(kind, {}).get("inputs", [])) + \
            list(nodes_spec.get(kind, {}).get("optional_inputs", []))
        if slot not in inputs:
            raise ProtocolError(f"command {index}: {kind} has no input {slot!r}")
        source = cmd.get("source")
        if source is not None and (not isinstance(source, str) or source not in known_types):
            raise ProtocolError(f"command {index}: connect source {source!r} is not a known node id")
    elif op == "move":
        pos = cmd.get("pos")
        if not isinstance(pos, list) or len(pos) != 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in pos):
            raise ProtocolError(f"command {index}: pos must be [x, y]")
    elif op == "rename":
        name = cmd.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError(f"command {index}: rename requires a non-empty 'name'")
    elif op == "disable":
        value = cmd.get("value")
        if type(value) is not bool:
            raise ProtocolError(f"command {index}: disable requires a boolean 'value'")
        if not nodes_spec.get(kind, {}).get("inputs"):
            raise ProtocolError(f"command {index}: {kind} is a source node and cannot be disabled")
    elif op == "delete":
        del known_types[node_id]
    elif op == "reference":
        value = cmd.get("value")
        if type(value) is not bool:
            raise ProtocolError(f"command {index}: reference requires a boolean 'value'")


def validate_proposal(proposal, describe, document):
    """Validate a raw provider proposal and return its (already-validated) commands list."""
    if not isinstance(proposal, dict):
        raise ProtocolError("provider proposal must be a JSON object")
    commands = proposal.get("commands")
    if not isinstance(commands, list):
        raise ProtocolError("provider proposal 'commands' must be a list")
    if len(commands) > MAX_COMMANDS_PER_BATCH:
        raise ProtocolError(f"provider proposal has {len(commands)} commands, exceeding the "
                             f"{MAX_COMMANDS_PER_BATCH}-command cap")
    if "done" not in proposal or type(proposal["done"]) is not bool:
        raise ProtocolError("provider proposal 'done' must be a boolean")
    rationale = proposal.get("rationale", "")
    if not isinstance(rationale, str):
        raise ProtocolError("provider proposal 'rationale' must be a string")
    known_types = {node_id: node["type"] for node_id, node in document.get("nodes", {}).items()}
    for index, cmd in enumerate(commands):
        _validate_command(cmd, describe, known_types, index)
    return commands


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

@dataclass
class ProposalContext:
    """Everything a provider needs to propose the next batch of edits."""
    prompt: str
    describe: dict
    document: dict
    revision: int
    images: list = field(default_factory=list)
    history: list = field(default_factory=list)


class Provider(ABC):
    """Proposes the next batch of graph edits given the current reference-loop context."""

    @abstractmethod
    def propose(self, context: ProposalContext) -> dict:
        """Return ``{"commands": [...], "done": bool, "rationale": str}``."""


class ScriptedProvider(Provider):
    """Deterministic provider for tests and demos: replays a fixed sequence of proposals."""

    def __init__(self, proposals):
        self._proposals = list(proposals)
        self._index = 0

    def propose(self, context):
        if self._index >= len(self._proposals):
            return {"commands": [], "done": True, "rationale": "scripted sequence exhausted"}
        proposal = self._proposals[self._index]
        self._index += 1
        return proposal


def _read_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _parse_json_response(text):
    text = text.strip()
    if len(text) > MAX_PROVIDER_RESPONSE_CHARS:
        raise ProtocolError(
            f"provider response is {len(text)} characters, exceeding the {MAX_PROVIDER_RESPONSE_CHARS} cap")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1).strip() if fence else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"could not parse provider response as JSON: {error}") from error


_RULES_TEXT = (
    "You are proposing edits to a NodeBased compositing graph so its viewed/referenced output "
    "matches a reference image described by the prompt. Respond with strict JSON only, no prose "
    "outside the JSON object, in exactly this shape:\n"
    '{"commands": [ ... ], "done": <boolean>, "rationale": "<string>"}\n'
    "Each entry in \"commands\" is one edit operation object with an \"op\" field. Only these ops "
    "are permitted: " + ", ".join(sorted(ALLOWED_EDIT_OPS)) + ". Do not propose save, load, render, "
    "undo, redo, errors, describe, inspect, reference_context, batch, or any node_data/animation/"
    "expression operation. Give an explicit \"id\" on \"create\" when a later command in the same "
    "batch must reference that node. Node types, parameters, numeric ranges, and choice lists are "
    "given below. Propose the smallest batch of edits that makes real progress toward the "
    "reference. Set \"done\" to true only once the graph already matches well enough that no "
    "further edits are needed; \"commands\" may be empty in that case."
)


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API using only the stdlib (``urllib``), no new dependency."""

    DEFAULT_MODEL = "claude-sonnet-5"
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model=None, api_key=None, timeout=60, max_tokens=4096, env=None):
        import os
        self.model = model or self.DEFAULT_MODEL
        # The key is read only from the environment (or an explicit override for tests) and is
        # never logged, written to disk, or included in any report the loop emits.
        self.api_key = api_key or (env or os.environ).get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def propose(self, context):
        request = self.build_request(context)
        body = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(self.API_URL, data=body, method="POST", headers={
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise ProtocolError(f"Anthropic API error {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProtocolError(f"Anthropic API request failed: {error.reason}") from error
        return self.parse_response(payload)

    def build_request(self, context: ProposalContext):
        schema = {"nodes": context.describe.get("nodes"), "limits": context.describe.get("limits"),
                  "choices": context.describe.get("choices"), "time_limits": context.describe.get("time_limits"),
                  "allowed_operations": sorted(ALLOWED_EDIT_OPS)}
        system = _RULES_TEXT + "\n\nOperation schema:\n" + json.dumps(schema)
        payload = {"prompt": context.prompt, "revision": context.revision,
                   "document": context.document, "history": context.history}
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": "Reference images follow this message, in capture order. Match the graph to "
                    "the prompt using the current document snapshot and prior iteration history.\n"
                    + json.dumps(payload),
        }]
        for image in context.images:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": _read_base64(image["path"])}})
            content.append({"type": "text",
                             "text": f"^ {image.get('name')} ({', '.join(image.get('roles', []))})"})
        return {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": content}]}

    def parse_response(self, payload):
        text = "".join(block.get("text", "") for block in payload.get("content", [])
                        if block.get("type") == "text")
        data = _parse_json_response(text)
        if not isinstance(data, dict):
            raise ProtocolError("Anthropic response JSON was not an object")
        commands = data.get("commands")
        done = data.get("done")
        rationale = data.get("rationale", "")
        if not isinstance(commands, list):
            raise ProtocolError("Anthropic response 'commands' must be a list")
        if type(done) is not bool:
            raise ProtocolError("Anthropic response 'done' must be a boolean")
        if not isinstance(rationale, str):
            raise ProtocolError("Anthropic response 'rationale' must be a string")
        return {"commands": commands, "done": done, "rationale": rationale}


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def _default_confirm(record):
    print(json.dumps({"commands": record["commands"], "rationale": record["rationale"]}, indent=2))
    try:
        answer = input("Apply this batch? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _default_report(record):
    if record.get("dry_run"):
        status = "dry-run"
    elif record["applied"]:
        status = "applied"
    elif record.get("declined"):
        status = "declined"
    else:
        status = "no-op"
    print(f"[iteration {record['index']}] revision {record['revision_before']} -> "
          f"{record['revision_after']} ({status})")
    if record["rationale"]:
        print(f"  rationale: {record['rationale']}")
    print(f"  commands: {json.dumps(record['commands'])}")
    if record["applied"]:
        print("  this iteration is exactly one undo step in the GUI")
    for label in ("before_capture", "after_capture"):
        artifacts = record.get(label)
        if artifacts:
            print(f"  {label.replace('_', ' ')}: " + ", ".join(a["path"] for a in artifacts))


def _default_emit(message):
    print(message)


class AgentLoop:
    """Drives the describe -> inspect -> capture -> propose -> validate -> batch cycle."""

    def __init__(self, connection: Connection, provider: Provider, prompt, *,
                 iterations=DEFAULT_ITERATIONS, dry_run=False, auto_confirm=False,
                 keep_captures=False, capture_parent=None, confirm=None, report=None, emit=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not (1 <= iterations <= MAX_ITERATIONS):
            raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
        self.connection = connection
        self.provider = provider
        self.prompt = prompt
        self.iterations = iterations
        self.dry_run = dry_run
        self.auto_confirm = auto_confirm
        self.keep_captures = keep_captures
        self.capture_parent = capture_parent
        self._confirm = confirm or _default_confirm
        self._report = report or _default_report
        self._emit = emit or _default_emit
        self._captures_root = None

    def run(self):
        history = []
        self._captures_root = Path(tempfile.mkdtemp(prefix="nodebased-agent-loop-",
                                                       dir=self.capture_parent))
        try:
            describe = self.connection.request({"op": "describe"})
            for index in range(1, self.iterations + 1):
                record = self._run_iteration(index, describe, history)
                history.append(record)
                if record["done"] or not record["commands"] or record.get("declined"):
                    break
            return history
        except KeyboardInterrupt:
            self._emit("nodebased-agent-loop: interrupted; stopping cleanly")
            return history
        finally:
            if self.keep_captures:
                self._emit(f"captures retained at {self._captures_root}")
            else:
                shutil.rmtree(self._captures_root, ignore_errors=True)

    def _capture(self):
        result = self.connection.request({
            "op": "reference_context", "prompt": self.prompt, "directory": str(self._captures_root)})
        artifacts = result["artifacts"]
        if len(artifacts) > MAX_IMAGES:
            raise ProtocolError(
                f"reference_context returned {len(artifacts)} artifacts, exceeding the {MAX_IMAGES} cap")
        total_bytes = sum(Path(a["path"]).stat().st_size for a in artifacts)
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ProtocolError(f"reference captures total {total_bytes} bytes, exceeding the "
                                 f"{MAX_TOTAL_IMAGE_BYTES} cap")
        return result

    def _apply(self, commands, revision):
        try:
            return self.connection.request({"op": "batch", "if_revision": revision, "commands": commands})
        except ProtocolError as error:
            if "stale" in str(error):
                raise StaleRevisionError(str(error)) from error
            raise

    def _propose(self, describe, history):
        inspected = self.connection.request({"op": "inspect"})
        revision = inspected["revision"]
        document = inspected["document"]
        capture = self._capture()
        context = ProposalContext(prompt=self.prompt, describe=describe, document=document,
                                   revision=revision, images=capture["artifacts"], history=history)
        proposal = self.provider.propose(context)
        commands = validate_proposal(proposal, describe, document)
        return revision, commands, proposal, capture["artifacts"]

    def _run_iteration(self, index, describe, history):
        revision, commands, proposal, before_artifacts = self._propose(describe, history)
        record = {"index": index, "revision_before": revision, "revision_after": revision,
                   "commands": commands, "rationale": proposal.get("rationale", ""),
                   "done": bool(proposal.get("done")), "applied": False, "declined": False,
                   "dry_run": self.dry_run, "before_capture": before_artifacts, "after_capture": None}
        if not commands:
            self._report(record)
            return record
        if self.dry_run:
            self._report(record)
            return record
        if not self.auto_confirm and not self._confirm(record):
            record["declined"] = True
            self._report(record)
            return record
        try:
            applied = self._apply(commands, revision)
        except StaleRevisionError:
            # Exactly one retry on staleness: re-inspect, re-capture, re-propose, re-validate, and
            # send the retried batch against the fresh revision. Any failure from here (staleness
            # again, validation, or an application error) is fatal for this run -- the loop never
            # retries a failed batch blindly.
            self._emit(f"[iteration {index}] revision went stale; re-inspecting and retrying once")
            revision, commands, proposal, before_artifacts = self._propose(describe, history)
            record.update(revision_before=revision, commands=commands,
                           rationale=proposal.get("rationale", ""), done=bool(proposal.get("done")),
                           before_capture=before_artifacts)
            if not commands:
                self._report(record)
                return record
            if not self.auto_confirm and not self._confirm(record):
                record["declined"] = True
                self._report(record)
                return record
            applied = self._apply(commands, revision)
        record["revision_after"] = applied["revision"]
        record["applied"] = True
        after = self._capture()
        record["after_capture"] = after["artifacts"]
        self._report(record)
        return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="nodebased-agent-loop",
        description="Close the reference-image loop against a running NodeBased --agent endpoint")
    parser.add_argument("--connect", required=True, metavar="LOCAL_NAME",
                         help="the running GUI's --agent local endpoint name")
    parser.add_argument("--prompt", required=True, help="what the graph should be built/adjusted to match")
    parser.add_argument("--provider", choices=("anthropic", "scripted"), default="anthropic")
    parser.add_argument("--model", default=None, help="Anthropic model id (default claude-sonnet-5)")
    parser.add_argument("--script", metavar="PATH",
                         help="JSON file of proposals for --provider scripted "
                              '(a list of {"commands": [...], "done": bool, "rationale": str})')
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                         help=f"max iterations (default {DEFAULT_ITERATIONS}, max {MAX_ITERATIONS})")
    parser.add_argument("--dry-run", action="store_true", help="print the proposed batch and apply nothing")
    parser.add_argument("--yes", action="store_true", help="apply every proposed batch without confirmation")
    parser.add_argument("--keep-captures", action="store_true",
                         help="do not delete the temporary capture directory on exit")
    parser.add_argument("--capture-dir", default=None,
                         help="parent directory for the capture temp directory (default: system temp)")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (1 <= args.iterations <= MAX_ITERATIONS):
        parser.error(f"--iterations must be between 1 and {MAX_ITERATIONS}")
    if args.provider == "scripted" and not args.script:
        parser.error("--provider scripted requires --script PATH")

    QCoreApplication.instance() or QCoreApplication([])
    try:
        connection = Connection(args.connect)
    except ConnectionError as error:
        print(f"nodebased-agent-loop: {error}", file=sys.stderr)
        return 1

    if args.provider == "scripted":
        proposals = json.loads(Path(args.script).read_text(encoding="utf-8"))
        provider: Provider = ScriptedProvider(proposals)
    else:
        try:
            provider = AnthropicProvider(model=args.model)
        except ValueError as error:
            print(f"nodebased-agent-loop: {error}", file=sys.stderr)
            connection.close()
            return 1

    loop = AgentLoop(connection, provider, args.prompt, iterations=args.iterations,
                      dry_run=args.dry_run, auto_confirm=args.yes, keep_captures=args.keep_captures,
                      capture_parent=args.capture_dir)
    try:
        loop.run()
    except (ProtocolError, ConnectionError) as error:
        print(f"nodebased-agent-loop: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
