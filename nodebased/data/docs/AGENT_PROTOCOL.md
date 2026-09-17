# Agent protocol v1

Transport: newline-delimited UTF-8 JSON. `request_id` is an optional correlation
value; replies are `{id, ok, result}` or `{id, ok:false, error}`. GUI local endpoint
accepts at most 1 MiB buffered input. Keep one request in flight per client.

Commands use `op` and these fields:

Mutating requests may include `if_revision` (integer). The command is accepted only when it
equals the current Dispatcher revision; malformed or stale guards fail before save/load,
undo/redo, or edit mutation. A guarded `batch` is one atomic edit and one undo slot.

- `describe`: node schemas, numeric ranges and operation names.
- `inspect`: complete document snapshot and monotonically increasing revision.
- `create`: `type`, optional `id`, `name`, `params`, `pos:[x,y]`.
- `set`: `id`, `param`, `value`.
- `connect`: destination `id`, `input`, `source` ID (or null to disconnect).
- `move`: `id`, `pos:[x,y]`.
- `rename`: `id`, `name`.
- `disable`: `id`, `value` boolean; bypasses first input; generators cannot bypass.
- `delete`: `id`; disconnects consumers and clears view if necessary.
- `reference`: `id`, `value` boolean; append/remove an ordered agent-reference tag. The tag is
  idempotent, undoable, and removed atomically when its node is deleted.
- `view`: `id` or null.
- `time`: any subset of `first`, `last`, `current`, and `fps`. The resulting
  range is validated atomically; use `describe` to discover limits and current
  values.
- `batch`: `commands` array of edit operations, one atomic undo unit. Use explicit
  IDs when subsequent edits reference a node created in the same batch.
- `undo`, `redo`: shared document history, limited to 100 undo snapshots.
- `save`, `load`: `path` to `.nbcomp` JSON. GUI load requires saved current state.
- `render`: headless only, `path` ending in `.png` or `.exr`, optional node `id`
  and integer `frame`; defaults to the document view and current frame. No
  implicit external model calls or network operations.
- `errors`: GUI local endpoint only. Optional `since` (Unix timestamp, default
  `0`) returns only failures newer than it, so polling doesn't re-report the
  same entries. Reaches into the live async render pipeline rather than the
  document, which is the only way an attached agent can see a render failure
  that a human hasn't reported — including one on a read-ahead frame that
  never became "current" and so never reached the status bar. Result:
  `errors` (list of `{frame, generation, target, message, timestamp}`, oldest
  first, capped at the 200 most recent), `current_frame`, `displayed_frame`
  (the last generation actually shown), `generation` (the newest requested),
  `playing`, and `status` (the current status-bar text).
- `reference_context`: GUI local endpoint only. Requires a non-empty prompt and an existing
  output `directory`; `include_view` defaults to true. It synchronously captures up to eight
  distinct PNG display previews in current-frame order: viewed node first, then tagged
  references, deduplicating a viewed reference and reporting both roles. Each artifact reports
  node id/name, roles, absolute path, width, and height. The response includes `revision`,
  `current_frame`, and the prompt. It uses the live view/exposure/channel/background controls,
  does not mutate the document, and makes no model or network call.

`describe` currently lists the common document operations; the headless-only
render extension, and the GUI-only `errors` op, are described here instead.
Unknown ops/types/parameters, invalid ranges, missing nodes and cycles fail
atomically. Commands are serialized by the GUI thread. There is no revision
precondition or collaborative merge in v1: agents should inspect immediately
before editing. Node outputs are immutable image DAG values; AI loops belong
to the future task execution domain, not graph cycles.

## Reference loop client

`nodebased.agentloop` (console script `nodebased-agent-loop`) is an external client that
consumes `reference_context` and proposes `batch` edits to close the "match this reference
image" loop. It is a separate process that connects to a running GUI's `--agent` endpoint over
the same QLocalSocket transport as `nodebased.agent --connect`; NodeBased itself still makes no
model or network calls, and the GUI has no idea the far end of the socket is an AI loop rather
than a human-driven script.

Each iteration: `inspect` (once per iteration, to get the current revision and document) →
`reference_context` with the user's prompt into a fresh temp directory → hand the prompt,
`describe` schema, document snapshot, captured images, and prior-iteration history to a
`Provider` → validate the returned commands client-side → send one guarded
`{"op": "batch", "if_revision": <the inspected revision>, "commands": [...]}` → capture again to
record the result. `describe` itself is only fetched once per run. The provider's `done: true`
flag, an empty `commands` list, or the user declining a confirmation prompt ends the loop; so
does reaching the iteration cap.

Validation (client-side, ahead of the server's own atomic validation, purely for a clearer
error than a rejected batch would give):

- Only `create`, `set`, `connect`, `move`, `rename`, `disable`, `delete`, `reference`, `view`,
  and `time` are accepted. `save`, `load`, `render`, `undo`, `redo`, `errors`, `describe`,
  `inspect`, `reference_context`, nested `batch`, and every animation/expression/shape/track op
  are refused before anything is sent.
- Every command must be a well-formed JSON object with a known `op`.
- `create` node types and every `set`/`create` parameter name are checked against `describe`;
  numeric parameters are checked against `describe`'s `limits`, and choice parameters against
  `describe`'s `choices`, where a bound exists.
- Node ids referenced by `set`/`connect`/`move`/`rename`/`disable`/`delete`/`reference`/`view`
  must already exist in the inspected document or have been created earlier in the same batch
  (an explicit `id` on `create` is required to reference it later in that batch).
- `disable` on a node with no inputs (a source/generator) is refused, matching the server rule.
- A member command may not carry its own `if_revision`; that guard belongs on the outer batch.

Hard caps: iterations default to 3 and cap at 10; a proposed batch caps at 64 commands; a
capture round caps at 8 images (matching `reference_context`'s own cap) and a bounded total
image byte count. A provider's raw response text is also capped before it is even handed to a
JSON parser. On a stale-revision rejection (the document changed since the loop's `inspect`),
the loop re-inspects, re-captures, and re-asks the provider exactly once; a second failure of any
kind stops the run with a clear error. The loop never retries a failed batch blindly.

`ScriptedProvider` replays a fixed list of `{"commands": [...], "done": bool, "rationale": str}`
proposals and takes no network access; it backs the test suite and `--provider scripted --script
FILE.json` demos. `AnthropicProvider` calls the Anthropic Messages API with the stdlib `urllib`
only (no new dependency), sending the reference PNGs as base64 image blocks and the `describe`
schema plus the validation rules above as the system prompt. It requires strict JSON output and
parses a response defensively, including one fenced ```json code block. The API key is read only
from the `ANTHROPIC_API_KEY` environment variable and is never logged or written to disk.

`--dry-run` prints every proposed batch and applies nothing. The default mode prints the batch
and asks for confirmation before applying. `--yes` applies every batch without asking. Every
applied iteration is exactly one atomic `batch`, so it is exactly one undo step in the GUI.
Ctrl-C stops the loop cleanly between steps. Capture temp directories are removed on exit unless
`--keep-captures` is given.

```sh
python -m nodebased --agent nodebased-local &
nodebased-agent-loop --connect nodebased-local --prompt "match the reference lighting" --yes
```

## MCP server

`nodebased.mcpserver` (console script `nodebased-mcp`) exposes this same LocalBridge transport as
a stdio MCP server (JSON-RPC 2.0, newline-delimited), so a general-purpose coding agent (Claude
Code, Codex) can drive a running GUI directly instead of speaking the raw wire protocol. It
implements `initialize`, `notifications/initialized`, `tools/list`, `tools/call`, `resources/list`
and `resources/read`. It is a thin proxy: NodeBased itself still makes no model or network calls,
and every mutating tool call still goes through the same Dispatcher validation as a human edit.

Run it against a GUI already started with `--agent <name>` (the Agent panel, see
`docs/AGENT_PANEL.md`, starts one automatically):

```sh
python -m nodebased --agent nodebased-local &
python -m nodebased.mcpserver --endpoint nodebased-local --agent-name my-agent
```

Tools:

- `describe`, `inspect` — proxy `describe`/`inspect` unchanged.
- `edit` — `{commands, if_revision}`. Commands are validated the same way `agentloop` validates a
  proposal (only `create`/`set`/`connect`/`move`/`rename`/`label`/`thumbnail`/`disable`/`delete`/
  `reference`/`view`/`time` are accepted); those are sent as one atomic `batch`. If `if_revision`
  is omitted, the tool inspects first and uses the current revision, so a caller doesn't need a
  separate round trip. A `save`/`load`/`render` command is refused unless the `agent/allow_disk_ops`
  QSettings flag is on (off by default; the Agent panel exposes it as a checkbox) — when allowed,
  each disk command is sent as its own top-level bridge request after the batch, since the real
  Dispatcher only accepts graph-editing ops inside a `batch`.
- `undo`, `redo`, `view`, `errors`, `reference_context` — proxy the matching op unchanged.
- `knowledge` — `{topic}`, one of the topics in `nodebased.knowledge.TOPICS` (`overview`, `nodes`,
  `protocol`, `colour`, `playback`, `animation`, `roto_tracking`, `time_model`, `version`,
  `limits`) plus `live_state`. Every topic except `live_state` works even without a reachable GUI
  endpoint (it reads the shipped docs/spec data directly); `live_state` calls `inspect`/`errors`
  on the bridge and summarizes the current nodes and render errors as JSON.
- `file_issue` — `{title, body, labels}`. Files a GitHub issue on `neodimo/NodeBased` (`gh` CLI if
  present, else the REST API with a `GITHUB_TOKEN`/`gh auth token` token), tags the body with a
  version/OS/agent footer, and appends an entry to a local log. Refused when the "Allow the agent
  to file GitHub issues" setting (on by default) is off.

Every knowledge topic is also exposed as an MCP resource at `knowledge://<topic>`.

A tool failure (validation error, unreachable bridge, disabled issue filing) comes back as an MCP
tool result with `isError: true` and the error text in `content`, not a JSON-RPC-level error —
only a malformed request or an unknown method/tool/resource uses a JSON-RPC error object.
