# M1 gate evidence

docs/VISION.md's M1 gate ("production 2D substrate") reads: "HDR/alpha/channel golden images,
hostile-media tests, cache correctness, interactive cancellation, exact build benchmarks on
Windows + Linux." This page tracks what evidence exists for each part and what is still missing.
It does not replace VISION.md's gate statement; it is the checklist against it.

## Golden images and hostile media

Covered by `tests/test_golden_2d.py` (references in `tests/data/golden/`, shared graph builders
in `tests/golden_2d_cases.py`) and `tests/test_hostile_media_2d.py`.

### Golden-image regression suite

Seventeen small graphs (24x16 canvas or smaller, so every `.npy` reference stays well under the
256 KB budget) lock the evaluator's pixel output for:

- HDR values above 1 and negative values through Grade.
- Merge's `over`, `plus` and `multiply` operations.
- Premult and Unpremult.
- Shuffle and ShuffleCopy of a named layer (`normals`/`depth`/`motion`) read off a small
  multichannel EXR fixture generated on the fly.
- Transform with each of its three filters (`nearest`, `bilinear`, `cubic`).
- Crop with a crop box larger than the source format (the data window is intersected down to
  the source, never grown past it) and one smaller than it.
- Reformat `to_box` both larger and smaller than the source format.
- A multichannel EXR Read, written straight back out (`write_exr` with `raster_layer_arrays`),
  then read again: the round trip a Read -> Write chain performs on every render, including its
  named `depth` layer.

Each case is checked against its reference on the full-frame evaluator path (`Evaluator.
evaluate_raster`) and, for every node kind on `tileexec.SUPPORTED_TILED_KINDS` (Grade, Merge,
Premult, Unpremult, Shuffle, ShuffleCopy, Read), on the tile path too (`TileExecutor.
compose_region`), with a stated per-channel absolute tolerance. Transform, Crop and Reformat are
**not** tile-native (the same, already-precedented exclusion as Mirror -- see `docs/PARITY_2D.md`
and `tileexec.SUPPORTED_TILED_KINDS`): those three cases are checked on the full-frame path only,
and a test asserts they are still correctly absent from `SUPPORTED_TILED_KINDS` rather than
silently regressing into an untested tile path. On any mismatch the failure names the worst
pixel: its index, the value the evaluator produced, the value the reference holds, and the
tolerance it exceeded by.

`tests/data/golden/regenerate.py` rebuilds every reference from the same case builders the test
suite imports, so the two can never drift apart, and refuses to overwrite an existing reference
without `--force`, so a graph's behaviour cannot change silently: review the pixel diff before
regenerating, and say why in the commit message when a reference legitimately moves.

**Not covered yet:** CornerPin, STMap/IDistort/VectorBlur, the warp/paint nodes, and 3D-fed
multichannel passes (Cryptomatte, relight bundles) have no golden references here; they have
their own pixel-assertion tests elsewhere (`tests/test_2d_parity_step_5c.py` and the 3D suites)
but not a locked golden image. OCIO/ACES colour-managed golden images (as opposed to the
scene-linear working-space values these graphs check) are also not covered.

### Hostile media

`tests/test_hostile_media_2d.py` runs every case under a 5-second wall-clock budget and asserts
resident memory does not grow by more than 1 GB (`resource.getrusage(...).ru_maxrss` before and
after), even for the case built to make a reader over-allocate.

- **Truncated / zero-byte / header-corrupt EXR, PNG, JPEG and TIFF:** a valid tiny file per
  format, truncated to a quarter of its length, replaced with zero bytes, or with its first 16
  header bytes bit-flipped. All raise a catchable `ValueError` (or, for a still-unreadable PNG,
  the underlying `libpng` reports its own read error to stderr before OpenImageIO turns it into
  the same catchable `ValueError` -- noisy, not a crash).
- **Wrong extension:** content that is not any known image format, saved under each of the four
  extensions, raises a catchable error. Real image bytes saved under a mismatched *image*
  extension (an EXR's bytes named `.png`) are a **documented fallback, not guaranteed to fail**:
  OpenImageIO's `ImageInput.open` is documented to fall back from the extension-selected plugin
  to content (magic-number) sniffing, so this can come back as a successful, correct read of the
  original image. The test accepts either outcome and, when the fallback fires, checks the
  recovered pixels equal the source's.
- **Unsupported extension** (for example `.mov`) is rejected up front with the exact supported
  set named in the message, before any content is touched.
- **1x1 image:** reads cleanly in all four formats.
- **Header claims 100,000 x 100,000:** an EXR whose display window (`full_width`/`full_height`)
  claims 100,000 x 100,000 while its actual pixel data is 4x4 is rejected by `media.
  read_media_raster`'s existing `Full-frame reader dimensions must be between 1 and 8192` check
  *before* any pixel buffer is allocated -- the guard that keeps this case's peak memory flat.
- **Sequence with a missing frame:** the default `missing="error"` policy names the missing frame
  number and the file path it looked for. The `hold` policy (nearest existing frame) and `black`
  policy (a zero raster at the requested frame's own size) are exercised as the documented
  fallbacks for the same gap.
- **Sequence with a frame of a different size:** reading the odd frame on its own works (each
  frame decodes independently; nothing assumes uniform sequence dimensions). Wiring that frame
  against a normally sized one into a `Merge` raises `Merge inputs must have matching formats in
  M0` -- a named, catchable error, not a silent resample or a crash.
- **NaN/Inf pixels:** an EXR holding a NaN pixel and +/-Inf pixels reads back with those exact
  values (`half_safe` only clamps *finite* overflow, established behaviour -- see `media.
  half_safe` and `tests/test_media.py`'s own overflow test). Running that image through a Grade
  node does not crash or hang either.

**Known limitation found, not fixed this step:** an unwired mask/mix blend computes `filtered *
gate + source * (1.0 - gate)`; at `gate == 1.0` the second term is `inf * 0.0`, which is `nan` in
IEEE float arithmetic even though the blend is meant to be a pure pass-through of `filtered`. So
an Inf pixel run through Grade (or any other masked/mixed node) comes back NaN instead of Inf.
This is silent numeric corruption, not a crash or a hang, so it is outside this step's "fix any
case that crashes or hangs" bar; `tests/test_hostile_media_2d.py`'s
`test_nan_pixels_flow_through_a_grade_node_without_crashing` documents it and only asserts the
no-crash/no-hang half. Needs Gonzo: decide whether to special-case `gate == 1.0` in the shared
mask/mix blend (touches every masked/mixed 2D kernel, not just Grade) as its own step.

## Cache correctness

Covered by `tests/test_cache_correctness.py`, which walks every 2D (image-output) `SPECS` kind,
wires a minimal graph with plain `Constant` sources, and for each declared param asserts the
full-frame cache key (`Evaluator.evaluate_raster`'s digest) and the tile-path cache key
(`tileexec._compute_node_digests`) both change and force a fresh cache miss (a genuine recompute,
never a stale hit), then asserts an unchanged graph is served from cache (a hit) on both paths.
142 of 165 kinds are covered directly; the other 23 need state a single-node graph cannot
fabricate (a real file, a linked Tracker, a multichannel layer, or are graph containers) and are
named with a reason in the test's `STRUCTURAL_EXCLUSIONS`, the same stated-exclusion convention
`tileexec.py` already uses for Transform/Crop/Mirror's own tile-path exclusion. A further 16
individual params on 8 otherwise-covered kinds are similarly named in `PARAM_SKIPS`.

Three findings came out of the sweep, each verified against the code rather than assumed:
`OCIODisplay.look` called a `PyOpenColorIO` method that does not exist on OCIO 2.5 and crashed on
first use (no prior test exercised it) — fixed in `nodebased/ocio_nodes.py` by routing through
`LegacyViewingPipeline`, with a regression test added to `tests/test_ocio_nodes.py`.
`TimeDissolve.which` is only read under `ease='animation curve'` with a curve set on it; every
other ease computes it from `in`/`out`/frame, so the stored value being cache-inert there is
correct, not a bug — read the digest formula to confirm this rather than assumed it.
`MotionBlur`/`MotionBlur2D`/`MotionBlur3D`/`Kronos`/`OFlow`/`VectorGenerator`/`TimeWarp`
deliberately evaluate fractional sub-frames outside the memory cache on every call
(`Evaluator.evaluate_raster`'s own docstring calls this the "ephemeral" contract); their
outer-node cache-key correctness is still asserted, only their repeat-call recompute count is
exempted, with that reasoning named in the test's `FRACTIONAL_SAMPLING_KINDS`.

`tests/test_cachetier.py` and `tests/test_cacheinspector.py` remain the disk-tier and
inspector-panel coverage they always were; this section is the per-param cache-KEY coverage the
gate's wording asks for, which neither of those audited.

## Interactive cancellation

Covered by `tests/test_interactive_cancellation.py`: on a 3840x2160, eight-node graph, a cancel
issued mid-evaluation stops work within 100 ms and leaves no result cached under the target's
digest, on both `Evaluator.evaluate_raster` (checked once per node) and `TileExecutor.compose`
(checked once per 256px tile); the next, uncancelled look is pixel-correct on both paths.
Synchronises with the in-flight worker thread by waiting on real progress counters
(`Evaluator.misses` / `TileExecutor.cache.misses`) rather than a fixed sleep, so a fast graph
finishing before the cancel lands cannot make the test pass for the wrong reason.

**Important qualifier, found while building this test, not assumed going in:**
`evaluate_raster`'s cancel check is once per node, so a single node whose own kernel exceeds
100 ms cannot be interrupted inside itself. Measured at 4K: `Grade` ~140 ms, `Saturation`
~160 ms, `ColorCorrect` ~290 ms — all already over budget on their own. The full-frame test graph
therefore deliberately uses cheap passthrough/format kinds (`Dot`, `Crop`, `Mirror`,
`ChannelShuffle`); the 100 ms full-frame contract holds for graphs shaped like that one, not
unconditionally for every node kind. The tile test graph carries `Grade`/`Blur` on purpose to
show it is the 256px tile granularity that buys interactivity back for heavier kernels, not a
cheap kernel choice. **Needs Gonzo:** decide whether the M1 gate's "interactive cancellation"
claim should be scoped to the tile path (already unconditional) plus a named exception list for
full-frame, or whether `evaluate_raster` needs a finer-grained (e.g. row-chunked) cancel check
inside expensive kernels to make the full-frame path's guarantee unconditional too.

**Resolved for these three kinds (M2 gate, docs/BENCHMARKS-v0.33-m2.md):** `Grade`,
`ColorCorrect` and `Saturation` each now check `cancel` between 256-row bands
(`imaging.py`'s `_ROW_CHUNKED_MASK_MIX_KINDS`) rather than once for the whole kernel call,
proven pixel-identical to the unchunked result and asserted under the 100 ms budget by
`tests/test_interactive_cancellation.py`'s `FullFrameRowChunkedKindCancellationTests`, each kind
run alone as the only node in its graph. Every other MASK_MIX_KINDS member whose own kernel
might exceed 100 ms at 4K is still unmeasured and still only checked once per node.

## Build benchmarks

`docs/BENCHMARKS-v0.33-4k.md` reruns `docs/BENCHMARKS-v0.9-4k.md`'s method (full 3840x2160
reference evaluator vs. a centered 1920x1080 `TileExecutor` viewport; cold TTFP, warm, edit
p50/p95) on the current build, plus peak process memory, which v0.9 did not measure.
`tools/benchmark_4k_viewport.py` is the script, and it now also runs as an optional,
non-blocking step of the Windows job in `.github/workflows/checks.yml`, uploading its JSON table
as the `benchmark-4k-viewport-windows` artifact labelled with the runner's own hostname so it
cannot be mistaken for a fixed workstation's numbers.

**Not covered yet:** this is one Linux workstation run plus a Windows CI trend line, not the
gate's "Windows + Linux exact build benchmarks" as a release-blocking pair measured the same way
on both. A Linux CI run of the same script (mirroring the Windows step) and a decision on what
regression threshold, if any, turns either into a gate rather than a trend line are still open.

## M2 gate

docs/VISION.md's M2 gate ("native/GPU performance") reads: "measured cold/warm startup,
time-to-first-pixel, p50/p95 interaction latency, 4K/8K memory ceilings and throughput." Full
detail and every number live in `docs/BENCHMARKS-v0.33-m2.md`; this table is the same
supported/partial/missing summary this page's M1 section already keeps, for M2.

| part | status | evidence |
| --- | --- | --- |
| Cold/warm process startup | covered | `docs/BENCHMARKS-v0.33-m2.md`'s "Process startup" table |
| Time to first pixel (Read -> Viewer) | covered, gated at 1080p | `docs/BENCHMARKS-v0.33-m2.md`'s "Time to first pixel" table, `tests/test_m2_latency_gate.py` |
| p50/p95 interaction latency | covered (six edits), gated for Grade at 1080p | `docs/BENCHMARKS-v0.33-m2.md`'s "Representative edit latency" table, `tests/test_m2_latency_gate.py` |
| 4K/8K memory ceiling | covered | `docs/BENCHMARKS-v0.33-m2.md`'s "Memory ceiling" section, `tests/test_memory_ceiling_gate.py`, `cachetier.SharedMemoryBudget` |
| 4K/8K throughput | covered (4K tile-path fps), gated | `docs/BENCHMARKS-v0.33-m2.md`'s "Throughput" section, `tests/test_m2_throughput_gate.py` |

**Not covered yet:**

- Transform/Blur/Merge/Roto/Tracker's own p95 latency at 1080p are measured but not gated (named
  in `docs/BENCHMARKS-v0.33-m2.md`'s "Gates" section already).
- The 4 GiB ceiling covers the desktop app's evaluator raster cache and tile cache together.
  It does not bound process RSS: in-flight images, source decode, display and GPU caches, and
  Python/NumPy allocations are outside it. A whole-process 8K memory ceiling remains open.
- 8K throughput has no gated fps floor (only the measured table); at well under 0.1 fps on this
  graph, a meaningful gate needs the optimisation work below to land first, not just a budget
  number.
- Only one node (`Evaluator._resample`'s bilinear filter) in the ten-node throughput graph has
  had an optimisation pass. Transform/Reformat/Tracker's shared `_filtered_pixels`/`_transform`
  path and `ColorCorrect`'s own kernel are named in `docs/BENCHMARKS-v0.33-m2.md`'s "Throughput"
  section as the next places the same profile points at.
