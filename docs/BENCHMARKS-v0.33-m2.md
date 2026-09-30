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

## CI

`tools/benchmark_4k_viewport.py --skip-m2` runs as an optional, non-blocking step on both the
Windows and Linux CI runners in `.github/workflows/checks.yml`, uploading the M1-era JSON table
as `benchmark-4k-viewport-windows`/`benchmark-4k-viewport-linux`. `docs/M1_GATE.md`'s "not covered
yet" note about a Linux CI run of the H2 benchmark is closed by the Linux half of this step. The
M2 additions (`--skip-m2` off) are not wired into CI: a fresh-subprocess startup measurement and a
temp-file `Read` on a shared, contended runner would be noise, not a trend line worth uploading.
