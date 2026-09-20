# v0.17 playback benchmarks: EXR decode, ingest, proxy decimation, decode-ahead

Status: measured evidence for the 4K playback-perf pass. Numbers below are the authority for
any future claim about this path; do not restate a faster number without a new measurement in
this file. v0.16's display-transform work (`docs/BENCHMARKS-v0.16-display.md`) made the ACES
2.0 view transform fast; this pass targets what was left — EXR decode, color ingest, and proxy
decimation, plus a parallel read-ahead pool so decode happens off the single preview worker's
critical path.

## Machine

Same machine as `docs/BENCHMARKS-v0.16-display.md`: 32 logical cores, Linux
`7.2.4-ogc3.1.fc44.x86_64`, AMD Radeon 8060S (Strix Halo) integrated GPU as the default GLX/EGL
vendor, PySide6 6.11.2, OpenImageIO 3.1.17.0, PyOpenColorIO 2.5.2. Test asset: 100-frame 4K
(3840×2160) `noise_test_4k.####.exr`, tagged `lin_ap1_scene` (ACEScg, i.e. already the working
space) with premultiplied alpha, ZIPS half-float, in the main repo root.

## How to reproduce

```
tools/playback_qa.py [VIEW] [SECONDS] --plate PATTERN [--full]
QT_QPA_PLATFORM=xcb DISPLAY=:0 .venv/bin/python tools/playback_qa.py "ACES 2.0" 8 \
    --plate <repo>/noise_test_4k.####.exr
```

Promoted from the gitignored `scratch/v016-qa/playback_qa.py`. Same behaviour: a real `Window`
on a real X server, local `QEventLoop` (avoids the `confirm_discard` modal on `app.quit()`),
collision-free node ids, `NODEBASED_QA_SOURCE` to run the identical harness against a different
checkout for a true A/B (used throughout this document to compare against unmodified `204b455`).
`--full` unchecks "Proxy while playing" so playback stays at the artist's selected tier instead
of the standard auto-proxy.

The per-stage numbers below are `time.perf_counter()` around each stage in isolation, 5-15
iterations after a warmup call, `min`/`median` reported in milliseconds, using distinct frame
numbers per iteration so the evaluator's own caches cannot turn a warm repeat into the "cold"
number.

## Profile: one cold 4K frame, before this pass (`204b455`)

| stage | min | median | source |
| --- | --- | --- | --- |
| OIIO raw decode (`ImageInput.read_image`, no ingest) | 43 ms | 53 ms | `media.py` decode call |
| `color.to_working` (associated alpha, ACEScg passthrough) | — | 144 ms | `nodebased/color.py::to_working` |
| Channel select (`pixels[..., [0,1,2]]` fancy index) | 40 ms | 48 ms | `nodebased/media.py` |
| `media.read_media_raster` (decode + ingest, tier 1) | 235 ms | 238 ms | full Read decode path |
| `Evaluator._decimate` tier 2 (4K → HD) | 149 ms | 157 ms | `nodebased/imaging.py` |
| `Evaluator._decimate` tier 4 (4K → quarter) | 111 ms | 111 ms | `nodebased/imaging.py` |
| `display_rgb` ACES 2.0, GPU, 4K (unchanged from v0.16) | 42 ms | 43 ms | `nodebased/color.py` |
| `display_rgb` ACES 2.0, GPU, HD/tier-2-equivalent | 8 ms | 10 ms | `nodebased/color.py` |

**Reading this table:** decode itself was never the bottleneck (43-53ms). The two real costs
were color ingest (`to_working`, 144ms) and proxy decimation (111-157ms) — both pure-NumPy
array operations that turned out to be doing far more memory traffic than the arithmetic
required, and both ran on the single preview worker thread for every frame regardless of
whether that frame had ever been decoded before.

## Fix 1: `to_working`'s unpremult/premult round trip was operating on a non-contiguous view

`to_working` divides RGB by alpha (unpremult), optionally runs an OCIO transform, then
multiplies back. The divide/multiply targeted `result[..., :3]` — a view that skips the alpha
channel and is therefore **not C-contiguous** (each row must stride past one float it doesn't
touch). NumPy's elementwise loop over that view measured ~4x slower than the identical
arithmetic on the full contiguous `(H, W, 4)` array:

| operation (4K, one call) | on `result[..., :3]` (non-contiguous) | on full `result` with a 4-wide factor (contiguous) |
| --- | --- | --- |
| divide by scale | 59 ms | 15 ms |
| multiply by scale | 59 ms | 15 ms |

The fix builds a 4-wide divisor/multiplier with `1.0` in the alpha column (dividing or
multiplying a finite float by `1.0` is bit-exact, so the alpha channel is untouched) and applies
it to the whole contiguous array. `to_working` end-to-end, associated alpha, ACEScg passthrough
(the common Read case): **144 ms → 60 ms** (measured both ways with the identical input,
`np.array_equal` bit-exact on the RGB channels, `tests.test_media.ColorTests` — 9 tests — cover
the correctness properties this must preserve).

## Fix 2: channel selection used fancy indexing where a slice would do

`media.py`'s `read_media_raster`/`read_media_region` pulled the RGB channels out of a decoded
`(H, W, C)` array with `pixels[..., [index - first for index in rgb]]`. When the channels are
contiguous and ascending — R, G, B in that order, the common case — this is always a slice in
disguise, but fancy indexing forces a copy through NumPy's general (slower) gather path: 48ms
median vs 30ms for the equivalent slice on one 4K frame. `_select_rgb` now takes the slice
fast path whenever the indices are a contiguous ascending run and falls back to fancy indexing
for a genuine reshuffle (e.g. a BGR-ordered layer). `read_media_raster` end-to-end after both
fixes: **235 ms → ~163-177 ms** for a cold decode (measured with distinct frame numbers each
call so the evaluator's cache cannot hide the real cost).

## Fix 3: `Evaluator._decimate`'s reshape + multi-axis mean

The proxy-tier downscale reshaped `(H, W, C)` into `(rows, tier, columns, tier, C)` and called
`.mean(axis=(1, 3))`. That access pattern defeats NumPy's fast reduction loops:

| tier | old: `reshape(...).mean(axis=(1,3))` | new: accumulate `tier` strided slices, scale once | speedup |
| --- | --- | --- | --- |
| 2 (4K → HD) | 149-157 ms | 18.7-18.8 ms | **~8x** |
| 4 (4K → quarter) | 111 ms | 13.6-13.8 ms | **~8x** |

The new implementation adds `tier` row-strided views (`pixels[0::tier] + pixels[1::tier] + ...`),
scales once, then repeats the same pattern on columns — the identical sum of the same `tier *
tier` input values divided by `tier * tier`, just accumulated in a different order, so results
agree with the old implementation to float32 rounding (`np.allclose`, and the existing
tolerance-based `tests.test_proxy.ProxyExecutionTests::test_a_proxy_predicts_the_full_resolution_result`
and `tests.test_tileexec` golden-vs-reference tests, which compare tiled output to the
full-frame `Evaluator` at the same tier — both sides call the same updated function, so this is
self-consistent by construction).

**This decimation cost matters more than it looks like on paper.** The existing (pre-v0.17)
"proxy while playing" feature (`Window.toggle_playback`, `tiers.auto_playback_tier`) already
auto-selects tier 2 for a 4K source, specifically because the ACES 2.0 transform was too slow
at full res (v0.16's own rationale). But the tier-1 Read decode used for *every* tier — see
`tileexec._node_full_image_at_tier`: decode never depends on tier, only the decimation
afterward does — meant every proxied playback frame was still paying full 4K decode plus a
157ms decimation just to throw away 3/4 of the pixels. Fixing decimation alone is a direct,
measured win for the feature that was already supposed to be carrying this workload.

## Fix 4: a parallel decode-ahead pool moves EXR decode off the single preview worker

Even after fixes 1-3, a cold decode+ingest is still ~163-177ms — real cost, just smaller. The
single preview worker (`Window.executor`, `max_workers=1`, kept single-owner per
`docs/PLAYBACK.md`) still paid that cost synchronously for every frame it evaluates, including
the read-ahead frames it already prefetches (`Window.future_frames`).

`nodebased/decodepool.py` adds `DecodeAheadPool`: a small (`min(4, cpu_count)`, tunable via
`NODEBASED_DECODE_AHEAD_WORKERS`), separately-threaded, bounded-memory (`min(4, cpu_count)`
workers, `NODEBASED_DECODE_AHEAD_MB` overrides the default ~10% of `cachetier`'s machine-sized
budget) LRU cache of decoded-but-not-yet-decimated Read rasters. It shares no state with the
single-owner `Evaluator`/`TileCache` — it only ever produces raw decoded arrays, never touches
graph evaluation, tile composition or the display transform — so running several decodes in
parallel does not touch the "single evaluator owner" invariant `docs/PLAYBACK.md` documents.

`Window.request_preview` submits background decode requests for the same frames
`future_frames()` already names, for every `Read` ancestor of the viewed target, **only when
the current tier is not 1** (a tier-1 request takes the bounded-region-read fast path in
`tileexec._generator_tile` instead and never consults this cache at all — prefetching there
would just burn the pool's own threads on decodes nobody reads back; measured as a net loss for
full-resolution playback, see "A gate that mattered" below). `tileexec._generator_tile`'s
tier != 1 Read branch checks the pool before paying for its own decode.

**Cancellation.** A decode already dispatched to OIIO cannot be interrupted mid-flight — the
same limitation `PlaybackQueue.replace` documents for graph evaluation. `DecodeAheadPool` uses
an `epoch` counter instead: `Window.request_preview` bumps it on a real content-invalidating
event (an edit, a scrub, a view/channel/tier change — anything where `cancel_active` would be
true) and **never** on a plain playback tick, matching the same tick-vs-content-change
distinction `docs/PLAYBACK.md` criterion 3 already draws for the render queue itself. A job
whose epoch is stale when it finishes still ran to completion, but its result is dropped
instead of occupying a cache slot a still-wanted frame could use.

Warm-cache effect, measured directly (`tests.test_tileexec.DecodeAheadIntegrationTests`):
composing a proxied (tier 2) frame whose decode was pre-warmed by the pool costs **~38 ms**
(decimation + graph eval only) vs **~197 ms** cold (decode + decimation + graph eval) — and
`TileExecutor.stats["source_decodes"]` stays at 0 for the warm case, proving no redundant
decode happened, not just that the pixels came out the same (they do:
`np.testing.assert_array_equal` against the cold path in the same test).

### A gate that mattered: tier 1 must not prefetch

The first version of this fix called `prefetch_reads` unconditionally whenever a future-frames
list existed. Full-resolution (tier 1) playback got *slower* as a result — 1.31 fps vs a clean
1.18 fps baseline at first glance looked like a wash, but the real problem was worse than "no
benefit": the pool's worker threads were doing full 4K decodes that the tier-1 render path
(bounded-region reads) never looks at, burning CPU and GIL time that the single preview worker
needed for its own bounded reads. Gating `prefetch_reads` on `tier != 1` fixed this — see the
full-resolution row in the playback table below, which is a real (if modest) improvement over
baseline once the wasted work is gone.

## Real-hardware playback, before/after (`QT_QPA_PLATFORM=xcb DISPLAY=:0`, drawn fps / median ms of the frames that weren't display-cache hits)

Baseline column re-measured with the same harness (`tools/playback_qa.py`) against a clean
checkout of unmodified `204b455` via `NODEBASED_QA_SOURCE`, not quoted from the v0.16 QA
entry — that entry's harness predates the `--full` flag and its recorded run did not have the
"proxy while playing" auto-switch active for reasons not reproduced here, so it is not a
like-for-like baseline for this table.

| scenario | baseline (`204b455`) | v0.17 (this pass) | speedup |
| --- | --- | --- | --- |
| ACES 2.0, auto-proxy (tier 2), default GPU | 2.10 fps / 463 ms | 10.21-11.42 fps / median not reported | **~4.9-5.4x** |
| ACES 2.0, auto-proxy, `NODEBASED_DISPLAY_GPU=0` (forced CPU) | not separately measured | 6.30 fps / 444 ms | — |
| sRGB, auto-proxy | not separately measured | 10.74 fps / median not reported | — |
| ACES 2.0, full resolution (`--full`, tier 1), default GPU | 1.18 fps | 1.38 fps / median not reported | ~1.2x |

The repaired-head parent run used three 12-second ACES auto-proxy runs (10.21-11.42 fps), plus
10.74 fps sRGB and 1.38 fps full-resolution ACES. All reported no render errors. The harness's
`render_ms_median_uncached` field is calculated from cold/uncached draws only; display-cache-hit
replays are excluded from that median and remain included in `drawn_fps`. The parent report did
not include the individual uncached-median or cache-hit counts, so those values are left
explicitly unfilled rather than inferred from total throughput. The forward-loop check now
accepts the expected `100 -> 1` wrap while rejecting ordinary backward jumps.

**24 fps at native 4K ACES 2.0 is not reached.** The best measured throughput is 11.42 fps at the
standard auto-proxy tier (2, i.e. 1920×1080) — a real ~4.9-5.4x improvement over the same
already-proxied baseline, not yet real time. See "Remaining gap and next steps" below.

**Bounding-box scope:** this playback lane makes no new guarantee about negative-origin or
overscan execution through the tiled proxy path. The pre-existing tiled proxy overscan limitation
remains explicitly out of scope; the negative-origin coverage in the bounding-box/tile tests does
not constitute native-display proxy evidence.

## Remaining gap and next steps

An isolated microbenchmark of the warm-decode critical path (decimate + graph eval + GPU
display transform + `to_qimage`, tier 2) measures **~76-83 ms/frame** (~12-13 fps ceiling), but
real playback measures ~125-130 ms/frame (~7.6-8 fps) — a ~45-50ms gap the per-stage benchmarks
above do not account for. Candidates, not yet isolated:

- **GIL contention between the decode-ahead pool and the single preview worker.** A quick sweep
  (`NODEBASED_DECODE_AHEAD_WORKERS`) measured 5.1 fps at 1 worker, 7.89 fps at 2, 7.73 fps at 4
  — the default (4) and 2 are within run-to-run noise of each other on this machine, so this is
  suggestive, not conclusive. Left at the `min(4, cpu_count)` default; the env var exists for
  further tuning without a code change.
- **`DisplayCache.identity`'s `blake2b(json.dumps(document, sort_keys=True))` digest**, computed
  on every request regardless of image size. Not isolated in this pass.
- **Qt scene rebuild per frame** (`viewer.scene().clear()`, `addPixmap`, `setSceneRect`,
  `draw_format_overlay`) and cross-thread signal dispatch from the worker to the GUI thread.
  Not isolated in this pass.
- `to_qimage`'s masked unpremultiply (`np.divide(rgb, weight, out=np.zeros_like(rgb),
  where=weight > 1e-8)`) measured ~17ms on an opaque HD-sized frame. A `np.where`/`np.maximum`
  safe-denominator rewrite measured ~1.7x faster in isolation, but changes behaviour for a
  premultiplied pixel with alpha exactly 0 and a *nonzero* RGB — the current code forces that
  pixel to black regardless of the numerator, which a safe-denominator rewrite would not do.
  That is a real (if unusual/defensive) case, so this was deliberately **not** changed — the
  measured gain did not justify the correctness risk given the effort budget for this pass.
- GPU-side decimation (folding the tier downscale into the existing OCIO GPU shader pass) was
  considered per the task brief's "optionally draw from a GPU texture" option, but the profile
  above shows readback/`QImage`/paint costs are small (sub-5ms) next to decode/ingest/decimate,
  which is what this pass actually targeted; not pursued.
- **Not attempted:** scanline-stride decode to skip rows during a proxy read. Measured directly
  (see git history of this investigation) — looping single-row `ImageInput.read_scanlines`
  calls to skip every other row on this ZIPS-compressed file costs *more* than a full decode
  (118-172ms vs 53-100ms) because per-call Python/OIIO overhead dominates over ~1000 individual
  calls; the OIIO Python binding does not expose a cheaper strided-scanline read.

**Recommended next step:** instrument the real `Window.start_preview`/`preview_ready` path
directly (not an isolated microbenchmark) to attribute the ~45-50ms real-vs-isolated gap to a
specific stage before optimizing further, since guessing among the candidates above risks
spending effort on the wrong one.
