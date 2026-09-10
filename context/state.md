# Current state — 2026-09-09

## Released

**v0.7.0 is the latest published stable release:**
<https://github.com/neodimo/NodeBased/releases/tag/v0.7.0>. It ships schema-v5
composition time, padded image sequences, time-aware caching, the minimal
timeline strip, frame-aware agent control, and the NSIS icon correction.
Release details and verification evidence are recorded in `TASKLOG.md`.

## In development

The time contract and future editorial join are in `docs/TIME_MODEL.md`;
implementation evidence is in `tests/test_time.py`.

## Boundaries

This remains a full-frame CPU reference compositor. There is no playback,
read-ahead/proxy/tile scheduler, animation, editorial clip/track model, audio,
roto/tracking, 3D, or model execution yet. The time boundary is intentionally
compatible with future clips, retimes, tracks, and nested compositions without
claiming those features already exist.

## Next owner

Omid owns interactive QA of sequence scrubbing and timeline ergonomics. Gonzo
owns the next engineering slice: playback/read-ahead and proxy-aware scheduling
before expanding into the clip/layer editorial UI.
