# Current state — 2026-09-09

## Released

**v0.6.5 is the latest published stable release:**
<https://github.com/neodimo/NodeBased/releases/tag/v0.6.5>. It ships the
selectable/drag-compatible Dot fixes and the final folded-green-N application
icon. Release details and verification evidence are recorded in `TASKLOG.md`.

## In development

The v0.7.0 candidate establishes time as a first-class schema and evaluation
concept: composition frame range/current frame/fps, frame-threaded evaluation,
padded image sequences, time-selective cache invalidation, a minimal timeline
strip, and frame-aware agent commands. The contract and future editorial join
are in `docs/TIME_MODEL.md`; implementation evidence is in `tests/test_time.py`.

## Boundaries

This remains a full-frame CPU reference compositor. There is no playback,
read-ahead/proxy/tile scheduler, animation, editorial clip/track model, audio,
roto/tracking, 3D, or model execution yet. The time boundary is intentionally
compatible with future clips, retimes, tracks, and nested compositions without
claiming those features already exist.

## Next owner

Gonzo owns exact-HEAD validation and the v0.7.0 Windows/Linux release. Omid owns
interactive QA of sequence scrubbing and timeline ergonomics after packages are
published. The next engineering slice should build playback/read-ahead and
proxy-aware scheduling before expanding into the clip/layer editorial UI.
