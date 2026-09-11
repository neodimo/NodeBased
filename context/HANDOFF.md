# Handoff checkpoint

Written so any agent — a different session, a different model provider — can
pick this up cold. Update this file whenever state changes meaningfully:
after a commit, after a test-count change, before a long/detached run, and
before ending a turn on unfinished work. Do not rely on chat history to carry
this; chat history is not guaranteed to be in context for whoever resumes.

## Exact repo state (verify before trusting this file — it can go stale)

```
cd /home/omid/.openclaw/workspace/projects/nodebased
git log --oneline -1   # e16a01a Merge animation curves (schema v6) onto the tile engine
git status --short     # clean
gh release view v0.9.1 # 4 assets, published 2026-09-10T22:52:56Z (latest release)
```

**Verified directly on 2026-09-10, not inherited from a prior report:**
`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
→ **Ran 357 tests in 50.1s / OK**. `SCHEMA_VERSION = 6`.

`main` is ahead of the v0.9.1 tag: animation and the pure-black viewer
background are merged but unreleased.

## The v0.9 release gate is closed

The gate was: (1) source-level bounded Reads, (2) viewer requests routed
through the tile executor, (3) a 4K viewport benchmark. All three landed,
merged at `17e2a1f`, and shipped. Do not re-open this as if it were pending —
that mistake cost a full status cycle earlier today.

v0.9.0's three workflows failed on Windows disk-cache init (no profile env
vars in the sanitized release-test environment). v0.9.1 fixes it with a
`TEMP`/process-directory fallback plus a regression test, and all its
workflows are green. Ship v0.9.1, not v0.9.0.

## What is proven vs. not

Proven (green tests + direct measurement):
- Adaptive memory cache sized from installed RAM; disk spill tier survives a
  process restart.
- Proxy tiers 1/2/4 execute for real in `Evaluator.evaluate()`; digest folds
  in tier so tiers never cross-satisfy.
- Tile executor is byte-exact against the reference evaluator on a
  Checker→Blur(masked)→Merge chain at tiers 1/2/4 across a multi-tile canvas.
- EXR data windows survive evaluation, including negative origins; a real
  overscan tile request at (-8,-8) returns the rendered margin exactly.
- Viewer requests its visible scene rectangle only; export still re-renders a
  complete full-resolution frame.
- 4K numbers in `docs/BENCHMARKS-v0.9-4k.md`: 312 ms vs 2404 ms cold TTFP,
  209 ms vs 2053 ms edit p50 for a 1920×1080 viewport on a 4K canvas.

Not proven — do not report these as done:
- Tiled/mip **source** I/O. Scanline formats may decode whole compressed rows.
- Any GPU or native-display claim. All desktop verification is offscreen.
- Warm-case tile performance: the tile path is *slower* than full-frame when
  nothing changed (reassembly overhead vs. one dict lookup). A static viewport
  is a common state, not an edge case.
- Nothing on `spike/roto-tracker` or `arch/representation-core` is merged or
  release-validated.
- Animated playback has no native-display QA, and animated Transform/Crop take
  the full-frame fallback rather than the tile path.

## Done since the last checkpoint

`m3/animation-curves` is **merged** (`e16a01a`). The rebase dropped 8 of its 12
commits as already-upstream and hit one conflict in `imaging.py`, resolved so
curve resolution runs before `tiers.scale_params`.

It also exposed a real defect worth remembering as a pattern: **a feature built
before the tile engine will read `node["params"]` and quietly bypass anything
the engine layers on top.** Animated params rendered correctly through the
reference evaluator and froze through tiles. The fix bakes curves once at the
`TileExecutor` API boundary (`animation.resolve_document`). Do not trust a
feature branch that only proves itself against `Evaluator`.

## Immediate next action

Assess `spike/roto-tracker` (`da01882`, declares schema v7). It is now unblocked
— it assumed v6 had landed, and v6 is on `main`. Sequence:

1. Rebase onto `main` and see what survives.
2. Expect the same class of gap animation just hit: the spike also predates the
   tile engine, so check its ROI/proxy rules against `tileexec`, not only
   against the reference evaluator. A passing reference-evaluator test proves
   nothing about the viewer's actual path.
3. Decide what is promotable from a spike versus what gets rewritten.

Also open: `main` carries unreleased work, so a v0.10 release decision is
available whenever Omid wants it.

Housekeeping available now: `feat/tile-artifact-engine` and
`m3/animation-curves` are both merged and can be deleted locally and on origin.

## Why this file exists

2026-09-10: Omid pointed out the 5-hour credit window was maxing out in
under an hour, driven by parallel detached/subagent lanes each re-deriving
full context from zero memory of each other, plus a model comparison
bake-off interleaved with real implementation work. He asked for written
checkpoints, updated often, specifically so a switch to a different model
provider mid-task can resume from paper rather than from chat memory. This
file is that paper.

Second failure mode recorded the same day: after v0.9.1 was already published,
status answers still described the release as blocked, because they were
reading the chat log instead of `git log`, `gh run list`, and
`gh release view`. Check the repo and the remote first; the chat is the least
reliable source in the room.
