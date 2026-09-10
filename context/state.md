# Current state — 2026-09-10

## Released

**v0.8.0 is the latest published stable release:**
<https://github.com/neodimo/NodeBased/releases/tag/v0.8.0>. It adds wall-clock
forward playback, a bounded three-frame read-ahead queue, cancellation and
exact stale-frame guards, dropped-frame accounting, and undo-safe transport.
Release details and verification evidence are recorded in `TASKLOG.md`.

## In development

The time contract and future editorial join are in `docs/TIME_MODEL.md`.
Playback acceptance criteria are fixed in `docs/PLAYBACK.md`; implementation
evidence is in `tests/test_playback.py` and `tests/test_desktop.py`.

## Boundaries

This remains a full-frame CPU reference compositor. There is no proxy/tile
scheduler, animation, editorial clip/track model, audio,
roto/tracking, 3D, or model execution yet. The time boundary is intentionally
compatible with future clips, retimes, tracks, and nested compositions without
claiming those features already exist.

## Next owner

Omid owns interactive playback QA on a real sequence. M3 owns an isolated
parameter-animation foundation on
`m3/animation-curves`; Gonzo must review it before merging.
