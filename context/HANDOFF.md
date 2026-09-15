# Handoff checkpoint

## Reference loop client handoff — 2026-09-14

Implemented `nodebased/agentloop.py` (console script `nodebased-agent-loop`) on
`v016/agent-loop`, based on `main` `6b3eee1`. It is an external client, not a NodeBased
change: connects to a running GUI's `--agent` endpoint, loops `inspect` -> `reference_context`
-> `Provider.propose` -> client-side validation -> one guarded `batch` -> `reference_context`
again, with `describe` fetched once per run. `ScriptedProvider` is deterministic for tests/demos;
`AnthropicProvider` uses stdlib `urllib` only, reads `ANTHROPIC_API_KEY` from the environment
only, and never logs or writes it to disk. Client-side validation restricts commands to
`create/set/connect/move/rename/disable/delete/reference/view/time`, checked against `describe`'s
node/param/limit/choice schema, with hard caps on iterations, commands per batch, images, image
bytes, and provider-response size. A stale-revision rejection gets exactly one
re-inspect/re-capture/re-propose retry, then a clear error; the loop never retries blindly.
`--dry-run`/interactive-confirm/`--yes` gate application, and each applied iteration is exactly
one GUI undo step (one atomic `batch`). NodeBased itself still makes no model or network call.
Along the way, fixed an offscreen-test-only deadlock: a same-thread GUI-plus-client integration
test needs `Connection.request()` to interleave short `waitForReadyRead` polls with
`QCoreApplication.processEvents()`, since a plain blocking wait never pumps the Qt event queue
the GUI-side `LocalBridge` needs to answer. Focused `tests.test_agentloop` passed **33 tests in
1.454s**; full offscreen discovery passed **535 tests in 146.752s**. `git diff --check` passed.
Unverified: a live Anthropic API call (only `urllib.urlopen` is mocked) and native
(non-offscreen) desktop interaction with the CLI.

## Reference bridge handoff — 2026-09-14

The bounded v1 bridge is implemented on `openclaw/reference-bridge`: schema v10 reference tags,
atomic `reference` edits with delete cleanup and undo/redo, exact revision preconditions, inspector
checkbox wiring, and GUI-local `reference_context` PNG capture. The context path evaluates through
the existing float32 premultiplied evaluator and display conversion, caps at eight captures, and
writes every response into a fresh collision-resistant child directory. There is no model/network
call and no TileKey/tiled change. Canonical focused coverage passed **24 tests in 0.439s** and full
offscreen discovery passed **502 tests in 147.834s**. Native-display QA remains unverified.

## Tracker pixel analysis handoff — 2026-09-14

Implemented from exact base `5958490d50c5315ac7308ac5fe85614b61be0198`: `tracker.analyse` performs
deterministic zero-mean NCC with bounded windows and sub-pixel parabola refinement, then the Tracker
UI picks a reference point and analyzes forward. The sole document mutation after success is one
validated atomic/undoable `set_tracks`; failure and cancel preserve the prior document. Parent
review pinned completion to the original Tracker, rejects concurrent track edits, made cancellation
win even if clicked just after computation, and rejects weak/occluded matches. Focused Tracker/Roto/UI
verification passed **56 tests in 1.036s**; full offscreen discovery passed **492 tests in 146.035s**.
Exact-head Desktop conformance run `34926562383` passed **492/492** on Ubuntu
(180.515s) and Windows (310.711s) at `2f5c85b`. Native-display interaction remains pending.
No `.claude/`, tag, release, schema, TileKey, or tiled claim was changed.

Updated 2026-09-14 at exact commit
`52bb039f20b6871179ba02081a520bf0ddd8257e`.

## Shipped state

- `v0.15.0` is tagged at `eaf99f4ca9e771705e17c7bd1137e9e75a7c2913`.
- The follow-up Roto UI commit is pushed on `main` and `origin/main`; it is
  intentionally not a new tag or release.
- Product version is `0.15.0`; current document schema is v10. Roto/Tracker payloads
  use the v8 `node_data` step, expressions use the v9 section, and agent reference
  tags use the v10 `references` section.
- Roto drawing and point dragging are implemented in `nodebased/app.py` and
  covered by `tests/test_roto_ui.py`. Gestures route through `set_shapes`, with
  validation, undo, save, agent semantics, and animated-point key retention.
- `docs/ROTO_TRACKING.md` describes the post-release Roto UI and bounded pixel Tracker
  analysis. `docs/RELEASE_NOTES.md` remains scoped to the exact `v0.15.0` tag and does
  not claim either follow-up shipped in that release.

## Verification and limits

The closeout focused on `tests.test_roto` plus `tests.test_roto_ui`: **48 tests
passed in 0.844s**. The full offscreen discovery suite then passed **484 tests
in 146.851s**. Exact-head Desktop conformance run `34924508493` passed Ubuntu
and Windows at `52bb039`. Real-X-server EXR QA also passed: decoder 5/5; 512px
playback 24.0 fps with a 0.08s longest gap; 1600px + Blur 18.2 fps with a 0.34s
gap. Native-display QA, real pointer feel, installer shell integration, GPU
behavior, and native Tracker pointer interaction remain unverified.

## Branch hygiene

As of 2026-09-14 the only branch is `main`, and no extra worktrees remain. Historical
roto/tracker experiments are preserved as `archive/*` tags on origin (see
`context/state.md`). They are provenance only, not merge instructions.

## Next owner

This historical Roto checkpoint is superseded by the Tracker and reference-bridge
handoffs above. Use their newest TASKLOG entries and focused tests for current work.
