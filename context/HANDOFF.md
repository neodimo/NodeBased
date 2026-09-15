# Handoff checkpoint

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
- `docs/ROTO_TRACKING.md` describes the post-release UI and keeps image-based
  tracker analysis explicitly future work. `docs/RELEASE_NOTES.md` remains scoped
  to the exact `v0.15.0` tag and does not claim this follow-up shipped in that release.

## Verification and limits

The closeout focused on `tests.test_roto` plus `tests.test_roto_ui`: **48 tests
passed in 0.844s**. The full offscreen discovery suite then passed **484 tests
in 146.851s**. Exact-head Desktop conformance run `34924508493` passed Ubuntu
and Windows at `52bb039`. Real-X-server EXR QA also passed: decoder 5/5; 512px
playback 24.0 fps with a 0.08s longest gap; 1600px + Blur 18.2 fps with a 0.34s
gap. Native-display QA, real pointer feel, installer shell integration, GPU
behavior, and pixel-based tracker analysis remain unverified or unimplemented.

## Branch hygiene

`spike/roto-tracker`, `openclaw/nodebased-roto2`, and local
`roto/rebase-onto-main` were inspected and retained as historical/divergent
branches. They are not follow-up merge instructions. No branch, tag, release,
push, or `.claude/` content was changed by the closeout work.

## Next owner

Use `docs/ROTO_TRACKING.md` for the feature contract and
`tests/test_roto_ui.py` for the deterministic UI contract. Tracker analysis
needs a separately scoped implementation and deterministic pixel fixtures.
