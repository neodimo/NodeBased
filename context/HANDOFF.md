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
- `docs/RELEASE_NOTES.md` and `docs/ROTO_TRACKING.md` describe the shipped UI
  and keep image-based tracker analysis explicitly future work.

## Verification and limits

The closeout focused on `tests.test_roto` plus `tests.test_roto_ui`: **48 tests
passed in 0.844s**. The full offscreen discovery suite then passed **484 tests
in 146.851s**. Native-display QA, real pointer feel, installer shell
integration, GPU behavior, and pixel-based tracker analysis remain unverified
or unimplemented as stated above.

Exact-head Xvfb playback evidence also passed: decoder frames 1, 3, 7, 11, 12;
512px/no blur displayed 12/12 at 24.0 fps with a 0.08s maximum gap; 1600px/Blur
displayed 12/12 at 18.2 fps with a 0.34s maximum gap. Both verdicts were
`playing`. The evidence is local scratch under `scratch/closeout-exrqa` and is
not committed; it does not constitute native-display QA.

## Branch hygiene

`spike/roto-tracker`, `openclaw/nodebased-roto2`, and local
`roto/rebase-onto-main` were inspected and retained as historical/divergent
branches. They are not follow-up merge instructions. No branch, tag, release,
push, or `.claude/` content was changed by the closeout work.

## Next owner

Use `docs/ROTO_TRACKING.md` for the feature contract and
`tests/test_roto_ui.py` for the deterministic UI contract. Tracker analysis
needs a separately scoped implementation and deterministic pixel fixtures.
