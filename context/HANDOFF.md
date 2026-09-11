# Handoff checkpoint

Written so any agent — a different session, a different model provider — can
pick this up cold. Update this file whenever state changes meaningfully:
after a commit, after a test-count change, before a long/detached run, and
before ending a turn on unfinished work. Do not rely on chat history to carry
this; chat history is not guaranteed to be in context for whoever resumes.

## Exact repo state (verify before trusting this file — it can go stale)

```
cd /home/omid/.openclaw/workspace/projects/nodebased
git log --oneline -1   # 6f44631 Release NodeBased v0.9.1
git status --short     # clean
gh release view v0.9.1 # 4 assets, published 2026-09-10T22:52:56Z
```

**Verified directly on 2026-09-10, not inherited from a prior report:**
`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
→ **Ran 294 tests in 52.4s / OK**.

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
- Nothing on `m3/animation-curves`, `spike/roto-tracker`, or
  `arch/representation-core` is merged or release-validated.

## Immediate next action

Review and land `m3/animation-curves` (`37d99ea`, schema v6, 12 ahead / 23
behind `main`). Sequence:

1. Rebase it onto `main` — the entire tile engine landed underneath it.
2. Confirm the v5→v6 document upgrade path and that animated params survive
   tile evaluation and proxy tiers (the animation work predates both).
3. Full suite green at the rebased HEAD before merge.
4. Only then touch `spike/roto-tracker`; it assumes v6 and declares v7, so it
   collides if animation has not landed first.

Housekeeping available now: `feat/tile-artifact-engine` is merged and can be
deleted locally and on origin.

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
