# Agent protocol v1

Transport: newline-delimited UTF-8 JSON. `request_id` is an optional correlation
value; replies are `{id, ok, result}` or `{id, ok:false, error}`. GUI local endpoint
accepts at most 1 MiB buffered input. Keep one request in flight per client.

Commands use `op` and these fields:

- `describe`: node schemas, numeric ranges and operation names.
- `inspect`: complete document snapshot and monotonically increasing revision.
- `create`: `type`, optional `id`, `name`, `params`, `pos:[x,y]`.
- `set`: `id`, `param`, `value`.
- `connect`: destination `id`, `input`, `source` ID (or null to disconnect).
- `move`: `id`, `pos:[x,y]`.
- `rename`: `id`, `name`.
- `disable`: `id`, `value` boolean; bypasses first input; generators cannot bypass.
- `delete`: `id`; disconnects consumers and clears view if necessary.
- `view`: `id` or null.
- `batch`: `commands` array of edit operations, one atomic undo unit. Use explicit
  IDs when subsequent edits reference a node created in the same batch.
- `undo`, `redo`: shared document history, limited to 100 undo snapshots.
- `save`, `load`: `path` to `.nbcomp` JSON. GUI load requires saved current state.
- `render`: headless only, `path` ending in `.png`, optional node `id`; defaults to
  document view. No implicit external model calls or network operations.

`describe` currently lists the common document operations; the headless-only
render extension is described here. Unknown ops/types/parameters, invalid ranges,
missing nodes and cycles fail atomically. Commands are serialized by the GUI
thread. There is no revision precondition or collaborative merge in v1: agents
should inspect immediately before editing. Node outputs are immutable image DAG
values; AI loops belong to the future task execution domain, not graph cycles.
