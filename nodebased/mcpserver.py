"""A small stdio MCP server for a running NodeBased GUI."""
from __future__ import annotations

import argparse
import json
import re
import sys

from PySide6.QtCore import QCoreApplication, QSettings

import nodebased
from . import agentloop, issues, knowledge


DEFAULT_PROTOCOL_VERSION = "2024-11-05"
DISK_OPS = frozenset({"save", "load", "render"})


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


TOOLS = [
    {"name": "describe", "description": "Return the node and parameter schema.",
     "inputSchema": _schema()},
    {"name": "inspect", "description": "Return the live document and revision.",
     "inputSchema": _schema()},
    {"name": "edit", "description": "Validate and apply one atomic edit batch. If if_revision is omitted, the current revision is inspected automatically.",
     "inputSchema": _schema({"commands": {"type": "array", "items": {"type": "object"}},
                             "if_revision": {"type": "integer"}}, ("commands",))},
    {"name": "undo", "description": "Undo the latest document change.", "inputSchema": _schema()},
    {"name": "redo", "description": "Redo the latest undone document change.", "inputSchema": _schema()},
    {"name": "view", "description": "Set the live viewed node, or clear it with null.",
     "inputSchema": _schema({"id": {"type": ["string", "null"]}}, ("id",))},
    {"name": "errors", "description": "Return live render errors since a Unix timestamp.",
     "inputSchema": _schema({"since": {"type": "number"}})},
    {"name": "reference_context", "description": "Capture reference images for a prompt.",
     "inputSchema": _schema({"prompt": {"type": "string"}, "directory": {"type": "string"},
                             "include_view": {"type": "boolean"}}, ("prompt", "directory"))},
    {"name": "knowledge", "description": "Read a NodeBased knowledge topic.",
     "inputSchema": _schema({"topic": {"type": "string", "enum": list(knowledge.TOPICS)}}, ("topic",))},
    {"name": "file_issue", "description": "File a NodeBased issue.",
     "inputSchema": _schema({"title": {"type": "string"}, "body": {"type": "string"},
                             "labels": {"type": "array", "items": {"type": "string"}}},
                             ("title", "body"))},
]


def _json_error(code, message, request_id=None):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class MCPServer:
    def __init__(self, endpoint, agent_name=None):
        self.endpoint = endpoint
        self.agent_name = agent_name
        self.connection = None
        self.initialized = False
        self._app = QCoreApplication.instance() or QCoreApplication([])

    def _bridge(self):
        if self.connection is None:
            self.connection = agentloop.Connection(self.endpoint)
        return self.connection

    def _request_bridge(self, command):
        try:
            return self._bridge().request(command)
        except Exception:
            if self.connection is not None:
                try:
                    self.connection.close()
                finally:
                    self.connection = None
            raise

    @staticmethod
    def _disk_ops_allowed():
        value = QSettings("NodeBased", "NodeBased").value("agent/allow_disk_ops", False)
        return value not in (False, "false", "0", 0)

    def _edit(self, arguments):
        # Real batch members are graph-editing ops only (see Dispatcher._edit in core.py); save,
        # load and render are top-level ops the Dispatcher never accepts inside a "batch". A
        # disk op here is therefore sent as its own bridge request, after the validated batch of
        # the remaining edit ops, rather than folded into the batch itself.
        commands = arguments.get("commands")
        if not isinstance(commands, list):
            raise ValueError("edit commands must be an array")
        if any(not isinstance(cmd, dict) for cmd in commands):
            raise agentloop.ProtocolError("edit commands must be objects")
        disk_commands = [cmd for cmd in commands if cmd.get("op") in DISK_OPS]
        if disk_commands and not self._disk_ops_allowed():
            raise agentloop.ProtocolError("save/load/render commands are disabled by agent/allow_disk_ops")
        for index, command in enumerate(disk_commands):
            if set(command) - {"op", "path", "id", "frame", "bits", "compression"}:
                raise agentloop.ProtocolError(f"command {index}: invalid {command.get('op')} command")
            if not isinstance(command.get("path"), str) or not command["path"]:
                raise agentloop.ProtocolError(f"command {index}: {command['op']} requires a path")
        describe = self._request_bridge({"op": "describe"})
        inspected = self._request_bridge({"op": "inspect"})
        edit_commands = [cmd for cmd in commands if cmd.get("op") not in DISK_OPS]
        proposal = {"commands": edit_commands, "done": True, "rationale": "mcp edit"}
        agentloop.validate_proposal(proposal, describe, inspected.get("document", inspected))
        revision = arguments.get("if_revision", inspected.get("revision"))
        results = []
        if edit_commands:
            results.append(self._request_bridge({"op": "batch", "if_revision": revision,
                                                  "commands": edit_commands}))
        for command in disk_commands:
            results.append(self._request_bridge(command))
        return results[0] if len(results) == 1 else {"results": results}

    def _live_state(self):
        inspected = self._request_bridge({"op": "inspect"})
        errors = self._request_bridge({"op": "errors"})
        document = inspected.get("document", inspected)
        nodes = document.get("nodes", {})
        state_nodes = {key: {"type": value.get("type"), "connections": value.get("inputs", {})}
                       for key, value in nodes.items() if isinstance(value, dict)}
        return json.dumps({"nodes": state_nodes, "render_errors": errors}, sort_keys=True, indent=2)

    def _tool(self, name, arguments):
        arguments = arguments or {}
        if name == "describe":
            return self._request_bridge({"op": "describe"})
        if name == "inspect":
            return self._request_bridge({"op": "inspect"})
        if name == "edit":
            return self._edit(arguments)
        if name in {"undo", "redo"}:
            return self._request_bridge({"op": name})
        if name == "view":
            return self._request_bridge({"op": "view", "id": arguments.get("id")})
        if name == "errors":
            return self._request_bridge({"op": "errors", "since": arguments.get("since", 0)})
        if name == "reference_context":
            return self._request_bridge({"op": "reference_context", "prompt": arguments.get("prompt"),
                                         "directory": arguments.get("directory"),
                                         "include_view": arguments.get("include_view", True)})
        if name == "knowledge":
            topic = arguments.get("topic")
            if topic == "live_state":
                return self._live_state()
            return knowledge.topic(topic)
        if name == "file_issue":
            if not issues.issue_filing_enabled():
                raise RuntimeError("Issue filing is disabled")
            return issues.file_issue(arguments.get("title"), arguments.get("body"),
                                     arguments.get("labels"), agent_name=self.agent_name)
        raise ValueError(f"Unknown tool: {name!r}")

    def dispatch(self, request):
        if isinstance(request, dict) and "id" not in request:
            if request.get("method") == "notifications/initialized":
                self.initialized = True
            return None
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or "method" not in request:
            return _json_error(-32600, "Invalid Request", request.get("id") if isinstance(request, dict) else None)
        request_id = request.get("id")
        method = request["method"]
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return _json_error(-32602, "params must be an object", request_id)
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested == DEFAULT_PROTOCOL_VERSION else DEFAULT_PROTOCOL_VERSION
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": version, "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "nodebased-mcp", "version": nodebased.__version__}}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
        if method == "resources/list":
            resources = [{"uri": f"knowledge://{topic}", "name": topic, "mimeType": "text/plain"}
                         for topic in knowledge.TOPICS]
            return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": resources}}
        if method == "resources/read":
            uri = params.get("uri", "")
            prefix = "knowledge://"
            topic = uri[len(prefix):] if uri.startswith(prefix) else None
            if topic not in knowledge.TOPICS:
                return _json_error(-32602, f"Unknown resource URI: {uri!r}", request_id)
            text = self._live_state() if topic == "live_state" else knowledge.topic(topic)
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}}
        if method == "tools/call":
            name = params.get("name")
            if name not in {tool["name"] for tool in TOOLS}:
                return _json_error(-32602, f"Unknown tool: {name!r}", request_id)
            try:
                result = self._tool(name, params.get("arguments") or {})
                # knowledge/live_state already return plain text (live_state is pre-serialized
                # JSON text); every other tool returns a JSON-able structure to encode here.
                text = result if isinstance(result, str) else json.dumps(result, sort_keys=True, allow_nan=False)
                is_error = False
            except Exception as error:
                text = str(error)
                is_error = True
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "content": [{"type": "text", "text": text}], "isError": is_error}}
        return _json_error(-32601, f"Method not found: {method}", request_id)

    handle_request = dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description="NodeBased stdio MCP server")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--agent-name")
    args = parser.parse_args(argv)
    server = MCPServer(args.endpoint, args.agent_name)
    for line in sys.stdin:
        if not line:
            break
        try:
            request = json.loads(line)
            response = server.dispatch(request)
        except (ValueError, UnicodeError, json.JSONDecodeError) as error:
            request_id = None
            match = re.search(r'"id"\s*:\s*([^,}\s]+)', line)
            if match:
                try:
                    request_id = json.loads(match.group(1))
                except (ValueError, json.JSONDecodeError):
                    pass
            response = _json_error(-32700, f"Parse error: {error}", request_id)
        if response is not None:
            sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
