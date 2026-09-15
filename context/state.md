# Current state — 2026-09-14

## Tracker pixel analysis implementation

The exact-base Tracker follow-up is implemented locally and remains unpushed/unreleased. The pure
analysis API is `nodebased.tracker.analyse`; the desktop inspector picks a reference point and
analyzes forward asynchronously. It commits only after success through one `set_tracks` command,
so cancel/error leaves the document unchanged. Float32 premultiplied scene-linear pixels, negative
data-window origins, proxy/full display mapping, and the existing solve are preserved. Tracker and
Roto remain on the reference full-frame path; no TileKey or schema change was made.

## Closeout at exact main commit

The repository was audited at `52bb039f20b6871179ba02081a520bf0ddd8257e`
(`main`, `origin/main`). This is the artist-facing Roto UI follow-up on the
tagged `v0.15.0` release (`eaf99f4ca9e771705e17c7bd1137e9e75a7c2913`). The
release tag remains at its intended release commit; the follow-up is pushed on
`main` and has not been tagged or released separately.

`__version__` is `0.15.0`; `SCHEMA_VERSION` is **9**. Schema v9 includes the
expression section and numeric-knob formula UI shipped in the release. Schema
v8 is the `node_data` payload step for Roto/Tracker. The post-release Roto UI adds a
foreground overlay for drawing closed polygons and dragging existing points;
edits use the validated, undoable `set_shapes` command and preserve animated
point envelopes.

## Verification

- Focused: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest
  tests.test_roto tests.test_roto_ui` — **48 tests passed in 0.844s**.
- Full: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s
  tests` — **484 tests passed in 146.851s**.
- Exact-head Desktop conformance run `34924508493` passed on Ubuntu and Windows
  at `52bb039` (3m43s and 5m52s respectively).
- Real-X-server EXR QA passed its decoder check at frames 1, 3, 7, 11, and 12.
  Playback showed all 12 frames in order: 512px/no-blur sustained 24.0 fps with
  a 0.08s longest update gap; 1600px/Blur sustained 18.2 fps with a 0.34s gap.
- Native-display visual QA, real input-device feel, and GPU/display claims
  remain unverified by the offscreen suite.

## Remaining blockers

- `nodebased.tracker.analyse` does not exist. Tracker positions are authored
  through `set_tracks`; no implementation currently derives tracks from image
  pixels. Building that analysis is future work and was deliberately excluded
  from this closeout.
- Roto has no transform handles or track markers, no open/stroked splines,
  planar tracking, or ROI-limited/tiered execution. The evaluator path remains
  full-frame for these node kinds.
- Native hardware and installer-shell QA still require a real desktop.
- GPU acceleration for the CPU ACES display transform remains unstarted.

## Branch and TODO audit

The current refs were checked directly. `spike/roto-tracker` (`da01882`) and
`openclaw/nodebased-roto2` (`8d9d584`) are divergent historical experiments,
not merge candidates: they are respectively 75/2 and 73/4 commits behind/ahead
of `52bb039` when measured in each direction. Local `roto/rebase-onto-main`
(`fa0756a`) is also an intermediate worktree branch, 7/3 commits behind/ahead.
They are retained for provenance; no branch deletion is part of this closeout.

The tracked product and test sources contain no unresolved `TODO` or `FIXME`
items. The remaining future-work language is intentional scope documentation,
principally tracker analysis and unverified native/GPU behavior.

The unrelated untracked `.claude/` tree was left untouched and is excluded from
the closeout commit.

## Next owner

The next implementation owner can use `docs/ROTO_TRACKING.md` as the contract
and `tests/test_roto_ui.py` as the deterministic interaction contract. Any
pixel-analysis tracker work should begin as a separately scoped feature with
new deterministic image fixtures and a schema/command review.
