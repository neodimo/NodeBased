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
git log --oneline -1        # 28417a3 Review and fix the tile executor: 4 correctness bugs found, all fixed
git status --short          # should be clean
```

**Verified fact, checked directly, not inherited from a prior report:**
`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests` is
**275/275 green** as of commit `28417a3`.

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
- The tile executor (`tileexec.py`/`tiles.py`) had 4 correctness bugs, each
  reproduced against the pre-fix code, fixed, and confirmed byte-exact
  against the reference evaluator (not just "no exception"): mask edits not
  invalidating the tile cache, unrenamed same-kind generators colliding,
  tiled Merge silently accepting mismatched canvas sizes, and Blur+mask
  raising on every call. See commit `28417a3` for the reproduction and fix
  of each. Golden-tested against the reference evaluator across a
  Checker→Blur(masked)→Merge chain at tiers 1/2/4 on a multi-tile canvas —
  byte-exact.

Not yet proven — do not report these as done:
- It is tile-cached full-frame composition with some tile-local kernel
  execution, not true ROI-driven viewer scheduling or source-level region
  reads. Do not describe it as "tile-native" without re-verifying that
  characterization against the current code.
- No performance/capability gate run yet: measured 4K cold/warm behavior
  for the tile executor specifically (as opposed to the full-frame
  evaluator, which is already measured), and confirmation that unsupported
  nodes (Transform, Crop — deliberately excluded, see `SUPPORTED_TILED_KINDS`
  in `tiles.py`) fall back explicitly rather than making false speed claims.
- No viewer/app integration — `TileExecutor` is not wired into `app.py`'s
  preview path yet; the desktop app still renders through the full-frame
  `Evaluator` only.
- Nothing on this branch is merged to `main`.

## Immediate next action

Either (a) wire `TileExecutor` into the app's preview path as an opt-in
fast path behind `supports_tiled()`, falling back to the existing
full-frame `Evaluator` otherwise, or (b) run the performance/capability
gate first (4K cold/warm numbers, explicit fallback verification) to
decide whether (a) is worth doing yet. Recommend (b) first — don't wire a
performance optimization into the UI before measuring that it's actually
faster than the full-frame path plus the new adaptive cache.

## Why this file exists

2026-09-10: Omid pointed out the 5-hour credit window was maxing out in
under an hour, driven by parallel detached/subagent lanes each re-deriving
full context from zero memory of each other, plus a model comparison
bake-off interleaved with real implementation work. He asked for written
checkpoints, updated often, specifically so a switch to a different model
provider mid-task (e.g. an OpenAI model, if Anthropic credit runs out) can
resume from paper rather than from chat memory. This file is that paper.
