# v0.33 — M2 gate: startup, time to first pixel and interaction latency

docs/VISION.md's M2 gate asks for "measured cold/warm startup, time-to-first-pixel, p50/p95
interaction latency, 4K/8K memory ceilings and throughput". `docs/M1_GATE.md`'s build-benchmarks
section (`docs/BENCHMARKS-v0.33-4k.md`) already covers a full-frame vs. tiled-viewport cold/warm
and edit p50/p95 on one synthetic graph; this page adds the pieces that were still missing:
process startup, time to first pixel on a real `Read` -> `Viewer` graph, and p50/p95 latency of
six representative interactive edits. v0.9 never measured any of these, so there is no prior
number to put them next to -- unlike `docs/BENCHMARKS-v0.33-4k.md`'s table, this one stands
alone.

Measured 2026-09-30 at about 10:25 AM PDT on commit `53531b4` (this table's own commit is the
next one on this branch), Linux 7.2.4-ogc3.1.fc44.x86_64, Python 3.12.13, NumPy 2.5.3, on this
workstation (not a CI runner; the Windows CI runner's own numbers land as the
`benchmark-4k-viewport-windows`/`-linux` artifacts described under "CI" below, not in this static
table, for the same reason `docs/BENCHMARKS-v0.33-4k.md` keeps its own CI numbers out of its
table -- a shared, contended runner is a trend line, not this workstation's figure). Script:
`tools/benchmark_4k_viewport.py` (extended for this gate; the M1-era full-frame/tiled-viewport
method is unchanged).

## Process startup

A fresh subprocess importing `nodebased.core`/`nodebased.imaging` and constructing an empty
`Dispatcher()` document, timed end-to-end (interpreter launch included) from outside the
subprocess, run twice in a row. "Cold" and "warm" here are the OS's own page/dentry cache for the
interpreter and the package's files -- the first launch may still be paying for disk I/O the
second no longer has to -- not the evaluator's in-memory result cache (a new process has none).
The desktop GUI's own Qt/PySide6 import cost is out of scope: that is a GUI-startup number, not
this data-layer-readiness one.

| | cold | warm |
| --- | ---: | ---: |
| process startup to an empty graph | 177.2 ms | 172.7 ms |

## Time to first pixel, Read -> Viewer

A real EXR written to a temp file at each size and read back through `Read` -> `Viewer` -- the
actual node a document opens with, not a synthetic generator like `Checker`.

| size | cold TTFP |
| --- | ---: |
| 1080p (1920x1080) | 33.7 ms |
| 4K (3840x2160) | 136.9 ms |

## Representative edit latency, p50/p95, 10 samples each

Six interactive edits, each a genuine cache miss by construction (the same "drag a slider"
construction `docs/BENCHMARKS-v0.33-4k.md`'s own edit loop uses): `Grade.exposure`,
`Transform.translate_x`, `Merge.mix`, `Blur.radius`, a Roto shape's point dragged (`set_shapes`
with a moved point), and a Tracker frame step (four keyframed, non-constant tracks, so the solved
transform genuinely differs at every sampled frame rather than mostly cache-hitting across the
step).

| edit | 1080p p50 | 1080p p95 | 4K p50 | 4K p95 |
| --- | ---: | ---: | ---: | ---: |
| Grade | 33.6 ms | 41.7 ms | 133.6 ms | 158.5 ms |
| Transform | 208.4 ms | 217.5 ms | 829.5 ms | 884.8 ms |
| Merge | 12.7 ms | 26.7 ms | 47.5 ms | 103.1 ms |
| Blur | 177.5 ms | 183.2 ms | 697.9 ms | 725.9 ms |
| Roto shape drag | 14.9 ms | 16.2 ms | 45.1 ms | 47.1 ms |
| Tracker frame step | 246.9 ms | 250.9 ms | 981.3 ms | 1009.8 ms |

**Reading this table straight:** every one of these is the full-frame reference-evaluator path
(`Evaluator.evaluate_raster`), the same path `docs/M1_GATE.md`'s cancellation section found
already over its own 100 ms cancel budget for Grade/ColorCorrect/Saturation alone (fixed for
those three in this same step, see `docs/M1_GATE.md`'s "Resolved for these three kinds" note).
Transform, Blur and Tracker here are markedly slower still (Transform and Blur are not
tile-native at all -- `docs/M1_GATE.md`'s stated exclusion, same as Mirror -- so a real interactive
session showing either at 4K goes through this exact full-frame cost, not a tiled one). This
table does not claim any of these hit the M1 gate's 100 ms interactivity contract; it only
measures what the current full-frame cost is, which is what the gate asks for and what the tests
below turn into a budget. Closing the gap for Transform/Blur/Tracker the way this step closed it
for Grade/ColorCorrect/Saturation is unstarted work.

## Gates

`tests/test_m2_latency_gate.py` turns two of the 1080p numbers above into slow tests (skipped on a
software/CPU adapter the same way the project's other slow gates are, even though nothing here
touches the GPU -- consistency with how a slow, machine-timing-sensitive test is marked
elsewhere): time to first pixel at 1080p, and `Grade`'s edit p95 at 1080p -- `Grade.exposure` is
the one canonical "drag a slider" edit every benchmark in this doc and in
`docs/BENCHMARKS-v0.33-4k.md` already uses, and the one this same step's cancel-granularity fix
(`docs/M1_GATE.md`'s "Resolved for these three kinds" note) actually touched. Both budgets are the
measurement above plus 30 percent headroom:

| gate | measured | budget (+30%) |
| --- | ---: | ---: |
| Read -> Viewer TTFP, 1080p | 33.7 ms | 43.8 ms |
| Grade edit p95, 1080p | 41.7 ms | 54.2 ms |

Transform/Blur/Merge/Roto/Tracker at 1080p are not gated here: their own p95 numbers above are
the current state, not yet a target anyone has committed to, and gating a test on them now would
either be a budget so loose it catches nothing or a red test on every run until that work
happens. Needs Gonzo: decide whether to commit a budget (and the work to hit it) for those five.

## 4K/8K memory ceilings and throughput

docs/VISION.md's M2 gate's other still-open half. Method: `tools/benchmark_memory_throughput.py`
builds the ten-node graph the step brief names -- Read, Grade, Transform, Merge, Blur, Tracker,
ColorCorrect (with Roto feeding its `mask` input -- Roto has no `image` input of its own, a
shape-source matte generator like Nuke's, so it cannot sit in the main chain), Reformat, Write --
reading a short (four-frame) real EXR sequence, and plays it back through the full-frame reference
evaluator (`Evaluator.evaluate_raster`) and through the tile-path executor (`TileExecutor.
compose_region`, a full-canvas region). Transform, Reformat and Tracker are not tile-native (the
same precedented exclusion `docs/M1_GATE.md` already states for Transform/Crop/Mirror), so this
particular graph's tile path falls back to the full-frame evaluator for the whole chain at every
frame (`tileexec.py`'s documented, expected fallback, not a bug in this benchmark) -- the tile and
full-frame numbers below are consequently close to each other; a graph built entirely from tile-
native kinds would show the usual tile-path advantage `docs/BENCHMARKS-v0.33-4k.md` measures.
"Cold"/"warm" here means the on-disk result tier (`cachetier.DiskCache`): cold is a fresh,
empty on-disk store (every frame a genuine miss); warm reuses that now-populated store with a
fresh in-memory cache, so a hit comes from disk, not from the process's own memory.

Measured 2026-09-30 at about 2:45 PM PDT on commit `035b9ba`, same workstation as the rest of this
doc. Peak memory is `resource.getrusage(...).ru_maxrss`, the process's whole-run high-water mark
-- it only ever grows, so the 8K rows below include whatever the 4K rows already allocated and
never returned to the OS; read each row as "peak memory of the run up to and including this
measurement", not an isolated per-resolution figure. A single fully isolated subprocess per row
would remove that overlap at the cost of four fresh Python/NumPy/OpenImageIO startups per point;
not done here, named as a limitation instead.

| resolution | path | disk cache | fps | peak memory |
| --- | --- | --- | ---: | ---: |
| 4K (3840x2160) | full-frame | cold | 0.30 | 5812.1 MB |
| 4K (3840x2160) | full-frame | warm | 0.30 | 5813.6 MB |
| 4K (3840x2160) | tile | cold | 0.30 | 5817.2 MB |
| 4K (3840x2160) | tile | warm | 0.30 | 5818.4 MB |
| 8K (7680x4320) | full-frame | cold | 0.07 | 14254.8 MB |
| 8K (7680x4320) | full-frame | warm | 0.09 | 14255.2 MB |
| 8K (7680x4320) | tile | cold | 0.07 | 14257.0 MB |
| 8K (7680x4320) | tile | warm | 0.10 | 14258.1 MB |

Reading this straight: at these sizes, a ten-node chain dominated by full-frame fallbacks (four of
the ten kinds are not tile-native) is slow in absolute terms -- well under one frame per second at
both sizes -- and the *default* evaluator/tile-cache budgets (each independently sized as a
fraction of this machine's physical RAM, `cachetier.default_memory_bytes()`) together hold multiple
full-resolution results, which is why 8K peak memory here is over 14 GB, not the 4 GB this gate
asks for. That gap is exactly what the memory ceiling below closes -- it is a separate, explicitly
combined budget, not the default this table measures.

### Memory ceiling

`cachetier.SharedMemoryBudget` ties the evaluator's raster cache and the tile executor's tile
cache to one combined byte ceiling. It is opt-in, constructed by passing `TileExecutor(memory_
budget=...)` instead of a separate `cache`/`evaluator`; the two are mutually exclusive, since an
explicit `cache` or `evaluator` would sit outside the shared accounting. `cachetier.
default_combined_memory_bytes()` (4096 MiB, overridable with `NODEBASED_COMBINED_CACHE_MB`) is the
ceiling this gate's own test passes when it wants the real default rather than a size tuned for a
fast test. Each cache still evicts its own least-recently-used entries exactly as it always did;
the only change is the headroom either one is offered: `ceiling_for(name)` returns the combined
total minus every *other* registered cache's current bytes, so the sum of both can never exceed
the ceiling no matter which side is under memory pressure. This is deliberately not one merged LRU
across both caches' key spaces (they do not share one); it is the weaker property the gate's
wording actually asks for -- the two together stay under budget, each evicting by its own least
recent use. **Not covered yet:** the desktop app (`nodebased/app.py`) still constructs its
`TileExecutor` without `memory_budget`, so it keeps today's two independent budgets; wiring the
app itself onto a combined ceiling is a separate, not-yet-made decision (a UI setting, most
likely) that touches startup behaviour well beyond this step's scope. Needs Gonzo: decide whether
and when to make the app opt in.

`tests/test_memory_ceiling_gate.py` drives four real 7680x4320 (8K) frames of the same ten-node
graph through a `TileExecutor` whose combined budget is 1.3x one 8K frame's own size (~658 MB --
smaller than that cannot be held to at all, since `Evaluator._store`'s existing contract keeps an
oversized *single* result resident rather than refusing it, the fix for the 8K cache going
silently inert the M1-era work already made; see `nodebased/imaging.py`'s `_store` docstring).
After every frame the test asserts the combined cache bytes stay within the budget plus 10
percent, and after the four-frame playback it recomputes frame 1 (long evicted by then) and
asserts the result is pixel-identical (`np.testing.assert_array_equal`) to what was rendered live
-- eviction costs a recompute, it must never corrupt one.

### Throughput

`tests/test_m2_throughput_gate.py` turns the 4K tile-path warm number above into a slow,
GPU-adapter-gated test (same skip convention as `tests/test_m2_latency_gate.py`): a floor, not a
ceiling, so headroom makes the budget *looser* by subtracting rather than adding it.

| gate | measured | budget (-30%) |
| --- | ---: | ---: |
| 4K tile-path playback fps | 0.29 fps | 0.20 fps |

**Optimisation pass.** Profiling a single 4K full-frame evaluation of this graph
(`cProfile`, sorted by cumulative time) found `Evaluator._resample`'s bilinear filter dominating:
2.19 of 3.39 total seconds, almost all of it in the four per-corner `fetch(xi, yi)` closures the
old code ran independently even though `x0`/`x1`/`y0`/`y1` are each shared by two of the four
corners. Rewritten to clip and validate each axis value once and reuse it across the two corners
that need it (nodebased/imaging.py's `_resample`): measured on a standalone 3840x2160 array
(`np.random.default_rng(0)`, `sx`/`sy` offset by a constant so every sample is a genuine
off-integer bilinear fetch), two runs each, before and after the change:

| variant | before | after |
| --- | ---: | ---: |
| bilinear, `clamp=False` | 0.768 / 0.776 s | 0.739 / 0.741 s |
| bilinear, `clamp=True` | 0.581 / 0.581 s | 0.555 / 0.556 s |

Output checksums (`result.sum()`) matched exactly before and after on both variants -- the
rewrite changes only how many times clip/validity are computed, never the pixels. A modest win
(roughly 4-5 percent): the four gathers themselves (`src[yc, xc]`, fancy indexing over the full
array) are the bulk of the remaining cost and are not reducible without changing which pixels get
read, which this pass does not touch. **Not covered yet:** Transform/Reformat/Tracker's shared
`_filtered_pixels`/`_transform` path and `ColorCorrect`'s own kernel are each a further chunk of
the same profile and got no pass this step; `tests/test_transform_resample_perf.py` (gated behind
`NB_PERF=1`) is the existing local perf check for this function and still passes unchanged.

## CI

`tools/benchmark_4k_viewport.py --skip-m2` runs as an optional, non-blocking step on both the
Windows and Linux CI runners in `.github/workflows/checks.yml`, uploading the M1-era JSON table
as `benchmark-4k-viewport-windows`/`benchmark-4k-viewport-linux`. `docs/M1_GATE.md`'s "not covered
yet" note about a Linux CI run of the H2 benchmark is closed by the Linux half of this step. The
M2 additions (`--skip-m2` off) are not wired into CI: a fresh-subprocess startup measurement and a
temp-file `Read` on a shared, contended runner would be noise, not a trend line worth uploading.
