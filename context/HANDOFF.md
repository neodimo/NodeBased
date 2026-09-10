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
git log --oneline -1        # 2f5105c Document a real EXR overscan bug found while scoping tile-executor wiring
git status --short          # bounding-box implementation is intentionally uncommitted
```

**Verified fact, checked directly, not inherited from a prior report:**
`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests` is
**288/288 green** with the uncommitted bounding-box implementation.

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
- No viewer/app integration — `TileExecutor` is not wired into `app.py`'s
  preview path yet; the desktop app still renders through the full-frame
  `Evaluator` only.
- Nothing on this branch is merged to `main`.

**Now measured** (2026-09-10, on a supported-kinds-only graph — Checker/
Constant/Grade/Blur/Merge/Viewer, no Transform/Crop — cold/warm/edit
timings, disk tier off for both to isolate the compute difference):

```
hd   1920x1080: full  cold= 459.5ms warm=0.089ms edit= 305.8ms
                tile  cold= 330.4ms warm=3.969ms edit= 184.3ms
2k   2048x1152: full  cold= 342.3ms warm=0.058ms edit= 330.7ms
                tile  cold= 229.9ms warm=4.091ms edit= 204.3ms
4k   3840x2160: full  cold=1396.6ms warm=0.064ms edit=1359.3ms
                tile  cold= 936.8ms warm=29.266ms edit= 762.7ms
```

Tile executor is faster cold and on the interaction-edit case at every
resolution measured (roughly 1.4-1.8x at 4K), likely because per-tile
box-blur passes on ~256px chunks are cheaper than one cumsum pass over
the full array. It is much slower on the warm (unchanged) case — 4-29ms
of tile-reassembly/lookup overhead vs. the full-frame evaluator's single
dict lookup at <0.1ms — worth knowing since a static, unedited viewport
is a common state, not just an edge case.

Also confirmed: a graph containing an unsupported kind (Transform) falls
back explicitly (`TileResult.tiled == False`,
`full_frame_fallbacks == 1`) rather than silently mis-rendering or
lying about coverage.

Caveat: `compose()` always renders the *whole* canvas tile-by-tile: there
is no viewport-limited/partial-region call yet. The measured win above is
purely from smaller per-tile compute (cache-friendlier array sizes), not
from "only recompute what's visible" — that's still unbuilt. The real
demand-driven-viewport win this architecture is *for* has not been
measured because it doesn't exist yet.

## Immediate next action

**Current action:** EXR data-window support is implemented but uncommitted.
`Raster`/`evaluate_raster()` retain overscan through core evaluator geometry;
`evaluate()` remains display-window compatible. The next prerequisite to
viewer integration remains per-node data-window ROI requests in the tile
executor plus a 4K requested-region benchmark. Read `docs/BOUNDING_BOX.md`
and run `tests/test_boundingbox.py` before continuing.

## Why this file exists

2026-09-10: Omid pointed out the 5-hour credit window was maxing out in
under an hour, driven by parallel detached/subagent lanes each re-deriving
full context from zero memory of each other, plus a model comparison
bake-off interleaved with real implementation work. He asked for written
checkpoints, updated often, specifically so a switch to a different model
provider mid-task (e.g. an OpenAI model, if Anthropic credit runs out) can
resume from paper rather than from chat memory. This file is that paper.
