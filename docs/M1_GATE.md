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

## Cache correctness, interactive cancellation, build benchmarks

Not covered by this step. Cache correctness has partial coverage in `tests/test_cachetier.py`
and `tests/test_cacheinspector.py` (not audited against the M1 gate's wording here); interactive
cancellation has `nodebased/cancellation.py` and scattered tests; Windows + Linux exact build
benchmarks do not exist yet. These are open work for a later M1-gate step.
