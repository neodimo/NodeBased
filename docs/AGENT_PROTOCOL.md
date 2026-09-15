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
