# Architecture / ADR 0001

## Decision

Use Qt 6/PySide6 for the initial native docking/graphics shell, and a small pure
Python command/document layer with NumPy reference kernels. This gets a real
cross-platform desktop vertical slice running without choosing a GPU stack from
marketing claims. This is reversible: the image evaluator is independent of UI
widgets and has a narrow `evaluate(document, node_id)` boundary.

Do not use Electron, a browser graph, or a full game engine for the first shell.
Qt is not itself proof of snappiness. Measure and keep heavyweight I/O/model
imports out of startup. Licensing/distribution obligations for Qt and future
codecs must be documented before packaged releases; no third-party compositor
source is copied into this repository.

## Boundaries

- `core.py`: schema, validation, undoable atomic commands, serializable document.
- `imaging.py`: immutable premultiplied linear float32 RGBA frames, kernels and
  bounded LRU. Decode/encode uses Qt image codecs initially; **no EXR or OCIO yet**.
- `app.py`: Qt graph, inspector and viewer. Evaluates document snapshots off the
  GUI thread; generation stamps prevent stale frames replacing newer requests.
- `agent.py`: JSON-lines headless command interface and a local-socket client.
  GUI uses the same Dispatcher and undo stack; no hidden agent-only edit path.

Disk project saves are atomic. File content changes invalidate Read cache entries
using path/size/mtime_ns; content-addressed media hashing belongs in M1. Cache keys
include node type/parameters/upstream keys, not editor coordinates. Retention is
bounded but working-set allocation is not yet tiled or strictly memory-budgeted.
Future scheduler must account for in-flight frames, GPU residency and decode
buffers; do not advertise this LRU as state-of-the-art production caching.

## Future typed execution domains

Keep pure image/geometry/scene DAG evaluation separate from side-effectful
AI/task execution. A loop is a structured task node with explicit iteration
state, max iterations, budgets, stop condition and cancellation; not a cyclic
image connection. Every model artifact carries provider/version, capabilities,
weights revision, seed, conditioning hashes, settings and hardware/runtime
provenance. Provider extensions are namespaced, discoverable and preserved.
Unsupported requested conditioning is an error or an explicit artist choice.

An agent can discover operations, inspect state, apply an atomic batch and undo.
The optional GUI endpoint is an opt-in OS-user-restricted local socket, never a
network listener. It has the launching user's filesystem authority. Credentials
and remote inference are out of scope for M0. Future remote tools require scoped
sessions, explicit permission boundaries, previews and operation audit history.

## Prior art / research provenance

- Natron: https://natrongithub.github.io/ (user-supplied reference, not evaluated here).
- Comp: https://github.com/ukmsz/Comp-releases (repository metadata read 2026-09-09;
  no runtime comparison performed).
- Griptape Nodes: https://github.com/griptape-ai/griptape-nodes (user's workflow
  reference; verify current API/licensing before integrating).
- Existing project research read at inception:
  `../houdini-dcc-diffusion/docs/PRIOR-ART.md` in Omid's workspace. Its
  deterministic-conditioning investigation highlights GEN3C and TrajectoryCrafter.
  These are leads to validate, not newly benchmarked integrations.

No claim that these are all competitors, or that any platform has no alternatives.
