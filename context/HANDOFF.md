# Handoff checkpoint

Written so any agent — a different session, a different model provider — can
pick this up cold. Update this file whenever state changes meaningfully:
after a commit, after a test-count change, before a long/detached run, and
before ending a turn on unfinished work. Do not rely on chat history to carry
this; chat history is not guaranteed to be in context for whoever resumes.

## Exact repo state (verify before trusting this file — it can go stale)

```
cd /home/omid/.openclaw/workspace/projects/nodebased
git branch --show-current   # feat/tile-artifact-engine
git log --oneline -1        # c8193e4 WIP: adaptive cache, disk spill, proxy tiers, and FPS controls
git status --short
```

As of 2026-09-10 13:32 PDT, uncommitted on top of `c8193e4`:
- `nodebased/tileexec.py`, `nodebased/tiles.py` — tile executor from a detached
  run that was interrupted by a provider rate limit. Uncommitted, not yet
  reviewed by Gonzo for the correctness gates below.
- `tests/test_tileexec.py`, `tests/test_tiles.py` — its 40 tests.
- `tests/test_proxy.py` (modified) — fixed a broken assertion that measured
  RGBA std where the constant alpha channel drowns out the RGB signal being
  tested; now measures RGB std. Unrelated to the tile executor.

**Verified fact, checked directly, not inherited from a prior report:**
`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests` is
**271/271 green** including all of the above uncommitted files, as of this
checkpoint.

## What is proven vs. not

Proven (green tests + direct measurement this session):
- Adaptive memory cache sized from installed RAM (`nodebased/cachetier.py`),
  replacing the old fixed 256 MiB that produced zero cache hits at 4K.
- Disk spill tier (clause C4), survives a process restart — measured directly.
- Proxy tiers 1/2/4 actually execute in `Evaluator.evaluate()` (not just
  declared in `tiers.py`) — sources decimate, pixel-unit params scale,
  digest folds in tier so tiers never cross-satisfy each other.
- FPS playback control, default 24, undoable, presets, re-anchors mid-play.
- Benchmark numbers for hd/2k/4k/8k with the new cache — in `TASKLOG.md`.

Not yet proven — do not report these as done:
- The uncommitted tile executor (`tileexec.py`/`tiles.py`) has NOT been
  reviewed against the acceptance gate in `docs/EVALUATION_TIERS.md`
  (golden-image ROI==crop equality, disk round-trip, export purity). A
  Codex subagent review already found 3 real correctness blockers in an
  earlier draft (optional masks missing from digests, same-named-generator
  collisions, tiled Merge accepting mismatched formats) — check whether
  those are fixed in the current uncommitted files before trusting them.
- It is tile-cached full-frame composition with some tile-local kernel
  execution, not true ROI-driven viewer scheduling or source-level region
  reads. Do not describe it as "tile-native" without re-verifying that
  characterization against the current code.
- Nothing on this branch is merged to `main`.

## Immediate next action

Review `nodebased/tileexec.py` / `nodebased/tiles.py` against
`docs/EVALUATION_TIERS.md`'s gate, confirm the 3 previously-found blockers
are actually fixed (don't take a prior report's word for it — check the
current file), then decide: commit as reviewed WIP, or send back for fixes.

## Why this file exists

2026-09-10: Omid pointed out the 5-hour credit window was maxing out in
under an hour, driven by parallel detached/subagent lanes each re-deriving
full context from zero memory of each other, plus a model comparison
bake-off interleaved with real implementation work. He asked for written
checkpoints, updated often, specifically so a switch to a different model
provider mid-task (e.g. an OpenAI model, if Anthropic credit runs out) can
resume from paper rather than from chat memory. This file is that paper.
