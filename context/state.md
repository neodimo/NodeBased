# Current state — 2026-09-09

## Delivery

M0 native 2D foundation implemented and pushed to `neodimo/NodeBased` main.
The requested full DCC remains **partial**. Product contract: `docs/VISION.md`.
Architecture: `docs/ARCHITECTURE.md`. Launch: `uv run nodebased` or README setup.

## Verified code revision

`4bc49ae7fe77dfe7ef88088485acbef4a2145892`

- Linux local offscreen: 20/20 tests pass.
- GitHub Ubuntu + Windows: both jobs passed the 20-test suite at this revision.
  https://github.com/neodimo/NodeBased/actions/runs/34324355434
- Headless CLI subprocess: create/view/render/save passed.
- Tests exercise actual local-agent edits/undo, Qt keyboard/properties and mouse
  wiring, stale preview rejection, export snapshot stability, graph transactions,
  persistence, cache invalidation and numerical image behavior.
- Actual shell screenshot inspected and attached in #nodebased:
  `artifacts/desktop.png` (deliberate ignored local test output). Both CI jobs
  upload their offscreen screenshots as workflow artifacts.
- Earlier Ubuntu failure: missing `libEGL.so.1`; fixed with explicit runtime
  dependencies. Initial Windows job already passed before that fix.

This state update and final TASKLOG entry are documentation-only after the tested
code revision; final handoff commit intentionally skips a redundant CI run.

## Remaining / next owner

Gonzo: next engineering milestone is M1 in `docs/VISION.md`: production color/media
(EXR/OCIO, channels/sequences), tile/ROI scheduling and measured performance.
Native display/GPU interaction, packaged installers and production memory/latency
are unverified. No 3D, procedural geometry, AI inference/loops or controlled-video
execution yet. These remain explicit milestones rather than implied features.
Omid: the runnable artifact is this repo (`uv run nodebased`); assess the native
viewer/graph/properties ergonomics to steer the next pass. No approval is pending.

## Active release work — update button + portable edition

User explicitly requested a release with GameStore-like updating, then a portable
Windows edition. Implementation and packaging prepared. 31 local tests pass.
Pending: real Windows installer/portable and Linux AppImage builds; publish only
when package tests pass. `packaging/build.py` includes installed/reinstalled and
actual portable helper/restart smoke tests. Release workflow can publish on an
explicit manual dispatch. No user approval is pending.
