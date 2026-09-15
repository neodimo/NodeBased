# Handoff checkpoint

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
- Product version is `0.15.0`; document schema is v9. Roto/Tracker payloads
  use the v8 `node_data` step, while expressions use the v9 section.
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

`spike/roto-tracker`, `openclaw/nodebased-roto2`, and local
`roto/rebase-onto-main` were inspected and retained as historical/divergent
branches. They are not follow-up merge instructions. No branch, tag, release,
push, or `.claude/` content was changed by the closeout work.

## Next owner

Use `docs/ROTO_TRACKING.md` for the feature contract and
`tests/test_roto_ui.py` for the deterministic UI contract. Tracker analysis
needs a separately scoped implementation and deterministic pixel fixtures.
