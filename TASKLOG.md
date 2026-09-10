# NodeBased task log

## 2026-09-10 — EXR data windows now survive evaluator geometry

- **What was done (evidence):** Added `Raster`, carrying pixels, a data
  window, and display window; `read_media_raster()` now retains OpenEXR
  overscan instead of clipping it at ingest. `Evaluator.evaluate_raster()`
  propagates those windows through Read, point filters, Blur, Crop, Transform,
  Merge, proxy decimation, cache memory/disk tiers, and display-window output.
  `evaluate()` remains display-array compatible for existing callers. Added 13
  real-EXR and geometry tests, including pan-reveals-overscan, blur-at-frame-
  edge samples real margin, Merge unions data windows while rejecting different
  display formats, and Crop reduces downstream data extent. `uv run python -m
  unittest discover -s tests` passed **288/288**.
- **Artifacts:** uncommitted changes in `nodebased/raster.py`, `media.py`,
  `imaging.py`, `cachetier.py`, `tests/test_boundingbox.py`, and
  `docs/BOUNDING_BOX.md` on `feat/tile-artifact-engine`; local-only pending
  review/commit.
- **State:** partial foundation complete; tile executor/viewer are deliberately
  not wired to Raster yet. Existing tile ROI code still assumes a single
  canvas, and no 4K data-window tile benchmark exists.
- **Next owner + artifact:** Gonzo continues from `docs/BOUNDING_BOX.md` and
  `tests/test_boundingbox.py`: make tile requests clamp to each Raster's data
  window, add full-frame-vs-requested-region golden tests, then integrate only
  after warm-path and 4K measurements hold.

## 2026-09-10 — Adaptive cache/disk-spill/proxy-tiers/FPS landed; tile executor reviewed and fixed

- **What was done (evidence):** Two pieces of work on `feat/tile-artifact-engine`.

  (1) Directly responding to explicit user direction ("fix the cache and
  increase it," "frames per second playback control with a default of
  24fps," "swing bigger"): added `nodebased/cachetier.py` (adaptive
  memory budget sized from installed RAM via `os.sysconf`, clamped
  512 MiB–8 GiB, plus a bounded on-disk spill tier keyed by digest with
  its own LRU — contract clause C4 of `docs/EVALUATION_TIERS.md`); wired
  it into `Evaluator.__init__`/`evaluate` replacing the fixed 256 MiB
  budget; landed real proxy-tier execution in `Evaluator.evaluate()`
  (sources decimate by area-average, pixel-unit params scale via
  `tiers.scale_params`, tier folds into the cache digest); added an FPS
  playback control (`nodebased/app.py`) with presets (24/23.976/25/29.97
  /30/48/50/59.94/60), default 24, undoable via the existing `set_time`
  boundary, re-anchoring the transport origin on a mid-playback rate
  change; wired the viewer's proxy dropdown through `PlaybackQueue` to
  the evaluator, with export always forcing tier 1 (clause C3 — "a
  proxy result must never reach a written file"). Commit `c8193e4`.
  Re-benchmarked hd/2k/4k with the new adaptive budget (this machine
  resolves to 8 GiB): all three now show near-zero warm TTFP and
  nonzero cache hits, unlike the previous 4K measurement (0 hits, warm
  ≈ cold). 8K spill-to-disk round-trip verified directly: cold run
  writes 34 disk entries at ~2 GiB, a second process reads them back
  with `disk_hits > 0` and zero memory hits, proving the spill survives
  a process restart rather than just an in-process test double.

  (2) A detached implementation run (interrupted mid-flight by a
  provider rate limit) had left an uncommitted from-scratch tile
  executor (`nodebased/tileexec.py`, `nodebased/tiles.py`,
  `tests/test_tileexec.py`, `tests/test_tiles.py`, 40 tests) whose own
  header claimed three release-blockers a Codex review had found were
  fixed. Rather than trust that claim, reproduced each one live against
  the actual uncommitted code before doing anything else:
  - Editing/rewiring a Grade's mask input produced byte-identical tiled
    output before and after — `_compute_node_digests` hashed only
    `SPECS[kind]["inputs"]`, excluding optional slots like `mask`, so
    the tile cache never invalidated on a mask edit. STILL PRESENT.
  - Two default-named Constant nodes (name defaults to kind when
    unrenamed) with different colors rendered the same pixels for
    both — the synthetic generator-tile identity was keyed on
    `node.get("name")`, and a `(type, name)` search picked the first
    match in the document for both nodes. STILL PRESENT.
  - A Merge between a 64×64 and a 16×16 Constant rendered silently
    through the tile executor while the reference evaluator correctly
    raised `"Merge inputs must have matching formats in M0"` — the
    tiled Merge kernel always cropped/padded both inputs to a common
    tile shape before any comparison could fire. STILL PRESENT.

  Fixed all three directly (not delegated): (1) `_compute_node_digests`
  now hashes every wired input slot, mirroring
  `Evaluator.evaluate`'s own digest loop exactly; (2) the generator
  tile cache and its content-digest lookup are now keyed on the node's
  actual document id (threaded through `_render_tile` →
  `_gather_inputs` → `_generator_tile`), which is unique by
  construction, and the fragile name/path search was deleted; (3)
  added `_validate_merge_formats`, called once per `compose()` before
  any tile renders, which independently walks each Merge node's A and
  B branches to a generator and raises the reference's exact error
  message on a canvas-size mismatch.

  While golden-testing the fixes against a broader multi-tile,
  multi-tier graph (Checker → Blur(masked) → Merge, canvas larger than
  the tile edge, tiers 1/2/4), found a **fourth** bug not previously
  flagged: any Blur with a mask wired raised
  `"Mask shape ... does not match source"` on every call. Blur's image
  input arrives halo-expanded per `tiers._blur_rule`; its mask input
  arrives at the plain output-region shape (mask rule declares it's
  "only ever sampled at the output pixels themselves"), and
  `_apply_mask_mix` requires matching shapes. Fixed by cropping
  image/filtered to the mask's shape (using the same region-offset
  arithmetic the post-kernel crop already used elsewhere in the file)
  before mixing.

  Each of the 4 fixes verified against the reference evaluator
  byte-exact (`np.allclose(..., atol=1e-5)`), not just "no exception,"
  across tiers 1/2/4 on a multi-tile canvas. Commit `28417a3`.

- **Inference:** The tile executor's own header claiming "reviewer-flagged
  fixes" was not a reliable signal of correctness — none of the three
  claimed fixes were actually present in the code, and a fourth gap
  existed that no prior review had exercised. Green tests (40/40 on the
  tile suite, 271/271 combined) proved nothing about these four cases,
  because none of the existing tests wired a mask into a tiled Grade/
  Blur, used two unrenamed same-kind generators, or merged mismatched
  canvas sizes. This is the same class of trap as the schema-upgrade
  `doc["version"] = SCHEMA_VERSION` bug found twice this session on
  other branches: a claim of correctness that was never actually
  exercised by the suite that's cited as proof of it.

- **Artifacts + local/committed status:** Both pieces committed on
  `feat/tile-artifact-engine` (`c8193e4`, `28417a3`), pushed nowhere yet
  — branch is local-only relative to `origin`. Not merged to `main`.
  `context/HANDOFF.md` (new) is the durable cross-session/cross-model
  resume checkpoint requested by Omid after the credit-window discussion
  this session; kept current as of `28417a3`.

- **State / unverified:** Verified: full suite is 275/275 as of `28417a3`
  (`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s
  tests`), re-run directly, not inherited from a prior report. Unverified:
  the tile executor's own 4K/8K performance numbers (the full-frame
  evaluator's numbers above are measured; the tile executor's are not);
  whether it is actually faster than full-frame-plus-new-cache at any
  measured resolution — no benchmark has compared them; the tile
  executor is not wired into `app.py`'s preview path, so nothing in the
  desktop app currently uses it. Do not describe the tile executor as
  "tile-native" viewer scheduling — it is tile-cached full-frame
  composition with tile-local kernel execution for the supported kernel
  subset (`tiles.SUPPORTED_TILED_KINDS`); Transform and Crop are
  deliberately excluded and fall back to the full-frame evaluator.

- **Next owner + concrete artifact:** `context/HANDOFF.md` in the repo
  root is the live resume checkpoint; read it before continuing this
  work in a new session. Recommended next step: measure the tile
  executor's 4K/8K cold/warm numbers and confirm unsupported nodes fall
  back explicitly (visible via `TileResult.tiled`/`full_frame_fallbacks`)
  before wiring it into the app — don't put a performance optimization
  in the UI before measuring it beats the path it's replacing.

- **Failure modes if any:** None outstanding from this pass — all four
  found bugs were fixed and regression-tested
  (`tests/test_tileexec.py::PostReviewBlockerRegressionTests`, 4 tests).
  Standing risk for future editors of `tileexec.py`: any kernel with an
  optional input and a nonzero halo needs the same
  shape-before-`_apply_mask_mix` treatment Blur got here; a new such
  kernel added without it will reproduce bug 4's failure mode silently
  until exercised by a test that actually wires the optional input.

## 2026-09-10 — Benchmark harness + measured finding: the cache collapses at 4K

- **What was done (evidence):** Added `nodebased/bench.py`, the measurement
  instrument the v0.9.0 gate and the M2 GPU decision both require. It records
  machine identity and build commit alongside cold/warm time-to-first-pixel and
  p50/p95 interaction latency, as JSON, so results are diffable across machines
  and commits.

  First run on this machine (Linux 7.2.3 x86_64, Python 3.12.13, NumPy 2.5.3,
  commit `12d2b20`), 7-node graph, Checker + Constant sources through
  Grade → Blur(r=12) → Transform(bilinear) → Merge, 16 parameter edits:

      hd  1920x1080   cold  769.4 ms | warm   0.071 ms | edit p50   549.5 ms | hits 39 miss 87
      2k  2048x1152   cold 1161.8 ms | warm   0.069 ms | edit p50   753.5 ms | hits 39 miss 87
      4k  3840x2160   cold 2871.5 ms | warm 2865.486 ms | edit p50 2842.2 ms | hits  0 miss 126

  **At 4K the retained-result cache does nothing at all.** Zero hits across the
  whole run, and warm time-to-first-pixel is within 0.2% of cold. One RGBA
  float32 frame at 4K is 126.6 MiB, so the 256 MiB budget holds two intermediate
  results while this chain needs six. Every node evicts the one before it and
  the LRU degrades into pure overhead. The same arithmetic at 8K gives 506.2 MiB
  per frame, which exceeds the entire budget, so `Evaluator.evaluate`'s
  `pixels.nbytes <= self.budget` guard declines to store anything at all.
- **Inference (separated from the measurement above):** the cliff between 2K and
  4K is a capacity effect, not an algorithmic one — HD and 2K keep the whole
  six-result chain resident and hit warm in 0.07 ms. That is consistent with the
  proxy tier and disk tier in `docs/EVALUATION_TIERS.md` being the correct fix,
  and it means the current 256 MiB default is a 2K-era number. Choosing the new
  budget policy is not done and should not be guessed at from one machine.
- **Artifacts + local/committed status:** `nodebased/bench.py`, committed and
  pushed to `main`. Numbers above are from this machine only; Windows numbers
  required by gate item 5 have not been taken.
- **State / unverified:** The 4K cache collapse is measured and reproducible via
  `uv run python -m nodebased.bench --resolution 4k --frames 16`. Not verified:
  anything on Windows, anything at 8K (not run), and any claim about how much
  the tiers improve these numbers — the tiers do not execute yet, so there is
  nothing to compare against.
- **Next owner + concrete artifact:** Gonzo. The baseline table above is what
  ROI and proxy execution must beat, and `nodebased/bench.py` is how it gets
  measured. Omid can reproduce any row with the command above.
- **Failure modes if any:** The harness's first version reported a 0.03 ms
  "scrub p50" at HD and I nearly recorded it. It was an artefact: this graph has
  no Read and no animated parameter, so every timeline frame produces an
  identical digest and the scrub measured dictionary lookups. Replaced with a
  parameter-edit loop, which recomputes the chain below the source the way an
  artist dragging a slider does, and the comment in `measure()` records why so
  the mistake is not reintroduced when animation lands and makes scrubbing
  measurable for real. A second defect was caught before commit: the summary
  line used a nested-same-quote f-string, valid on the 3.12 interpreter used
  here but a syntax error on the Python 3.11 the project declares as its floor.

## 2026-09-10 — Thesis amendment + v0.9.0 tier contract + ROI/proxy rule table

- **What was done (evidence):** Three things, in order.
  1. `docs/VISION.md` gained a **Thesis** section (commit `ad17afc`) on Omid's
     direction: NodeBased exists to redefine the hybrid GenFX/VFX workflow, with
     generative and deterministic work sharing one graph, document, undo stack
     and cache. Written as a design constraint with named consequences — typed
     cached artifacts, evaluation tiers that must stay expressible for operators
     costing seconds and dollars, and contained rather than hidden
     non-determinism — so it binds the engine work instead of decorating it.
  2. `docs/EVALUATION_TIERS.md` (commit `4193a88`) fixes the v0.9.0 acceptance
     contract **before** implementation, following the `docs/PLAYBACK.md`
     precedent. Clauses C1–C6 cover digest identity, the per-kernel ROI mapping
     table, proxy correctness, the disk tier, unchanged cancellation semantics,
     and typed artifacts. The gate requires per-kernel golden-image equality and
     measured 4K numbers from a named machine.
  3. `nodebased/tiers.py` implements the ROI rule table and proxy parameter
     scaling as pure geometry and arithmetic, with no NumPy kernels and no Qt,
     so the contract is testable without rendering. `tests/test_tiers.py` adds
     38 tests. Full suite: **198/198**, up from a 160 baseline at `aa8c865`.
- **Inference (not measured):** ROI plus proxy should deliver the interactive
  4K scrub improvement that motivates the work. No speedup has been measured
  yet and none is claimed; the benchmark harness required by the gate does not
  exist.
- **Artifacts + local/committed status:** `nodebased/tiers.py`,
  `tests/test_tiers.py`, `docs/EVALUATION_TIERS.md`, the `docs/VISION.md`
  amendment, and `assets/marketing/nodebased-vision-poster-v1.png` are all
  committed and pushed to `main`. `scratch/uv.lock.regenerated` was moved out of
  the repository to `workspace/trash/uv.lock.regenerated.2026-09-10`; `scratch/`
  and `uv.lock` are now ignored.
- **State / unverified:**
  Verified: every node kind in `SPECS` has an ROI rule and the coverage test
  fails if one is added without a rule; the Transform inverse map is checked
  corner-for-corner against `Evaluator._transform`'s own arithmetic under
  rotation, scale-down, translation and cubic filtering; the declared blur
  support is checked against `_box_blur_axis`'s real measured reach rather than
  against the radius parameter; pixel-unit parameters scale with the tier while
  unitless ones provably do not.
  **Not done — the important gap:** the evaluator does **not** yet execute by
  region or at a proxy tier. `Evaluator.evaluate()` is still a full-frame pass
  with an in-memory-only LRU. This commit lands the rules and their proofs, not
  the execution path. Nothing in the gate's benchmark or golden-image clauses is
  satisfied yet, and no disk tier or typed-artifact store exists.
- **Next owner + concrete artifact:** Gonzo owns the execution path — region
  propagation through `Evaluator.evaluate()`, tier-aware digests per clause C1,
  Read-side downscaling for proxy, then the disk tier and the benchmark harness.
  `docs/EVALUATION_TIERS.md` is the specification to build against and
  `tests/test_tiers.py` is the behaviour to preserve.
- **Failure modes if any:** None in this pass. The risk being deliberately
  managed is the opposite one: shipping a rule table that *looks* like tiered
  evaluation. The unverified section above exists so nobody reads 198 green
  tests as evidence that ROI rendering works.

## 2026-09-10 — Aspirational NodeBased vision poster created

- **What was done (evidence):** Created a 1122×1402 vertical product poster
  showing the intended finished NodeBased experience: a credible dark
  professional compositor UI with a hero 2D viewer, organized node graph,
  spatial 3D viewport, layered editorial timeline, properties, render passes,
  and a restrained embedded agent-assist workflow. The selected green folded-N
  icon was used as the brand reference. SHA-256:
  `2088865424899b23c192242d29cb38625010d6c6ae6ca5b0c5d92cbe8ef66d67`.
- **Inference:** The poster communicates the unified 2D/3D/procedural/AI vision
  more clearly than a feature checklist, but it is aspirational concept art and
  does not claim the pictured UI is implemented today.
- **Artifacts:** `assets/marketing/nodebased-vision-poster-v1.png`, committed to
  the repository. Generated with the built-in image tool using
  `assets/nodebased-icon.png` as the identity reference. Checksum and 1122×1402
  RGB dimensions re-verified against the committed file.
- **State / unverified:** Complete as a first poster direction. Print color,
  physical-size output, and small-text legibility have not been proofed.
- **Next owner + concrete artifact:** Omid can approve or request a focused
  revision using `assets/marketing/nodebased-vision-poster-v1.png`.

## 2026-09-10 — v0.8.0 playback/read-ahead published and verified

- **What was done (evidence):** Added a wall-clock forward transport and a
  serial, bounded playback queue: one display request plus at most three future
  prefetch requests. New scrubs/edits cancel active and queued obsolete work;
  Viewer admission now requires both the active generation and exact requested
  frame. Playback ticks are validated transient time commands that do not fill
  the 100-slot undo history. Slow playback skips obsolete timeline positions
  and counts dropped frames rather than growing latency. The full suite passes
  **160/160**, including a real EXR sequence cache-warming test and a 16 ms
  enqueue-budget test. GitHub Actions run `34449569111` completed Windows,
  Linux, and publish successfully. Fresh downloads of all three public packages
  passed `SHA256SUMS`; the downloaded AppImage launched offscreen, reported
  0.8.0, and passed its media/color smoke checks.
- **Inference:** Three-frame read-ahead should improve warm sequential playback
  when per-frame evaluation is cheaper than the frame interval. Real production
  throughput is not established by the synthetic/short sequence tests.
- **Artifacts:** `nodebased/playback.py`, `docs/PLAYBACK.md`, playback changes in
  `nodebased/app.py` and `nodebased/core.py`, tests in `tests/test_playback.py`
  and `tests/test_desktop.py`, plus README/architecture/release/state updates.
  Source is committed/pushed at `97a6f62`; public release:
  <https://github.com/neodimo/NodeBased/releases/tag/v0.8.0>.
- **State / unverified:** Released, public, and checksum-verified. Display-backed
  Windows/Linux playback, long EXR sequences, and 4K memory pressure remain
  unverified. Proxy tiers are explicitly rejected; no fake
  post-scale proxy is claimed.
- **Next owner + concrete artifact:** Omid owns real-sequence interaction QA
  using v0.8.0. M3 works separately in `projects/nodebased-animation` on
  `m3/animation-curves`; Gonzo must review before merging.
- **Failure mode:** Playback must not enqueue every missed timeline frame, allow
  a prefetch to enter the Viewer, run concurrent evaluations against one mutable
  LRU, or consume an undo slot per transport tick.

## 2026-09-10 — v0.7.0 published and independently verified

- **What was done (evidence):** Published the schema-v5 time foundation from
  commit `228163a`. GitHub Actions run `34444545018` completed Linux package,
  Windows package, and publish successfully. Fresh public downloads of the
  AppImage, Windows portable ZIP, and Windows installer all passed the published
  `SHA256SUMS`. The downloaded AppImage launched offscreen and reported version
  `0.7.0`; its OpenImageIO, OCIO, EXR round-trip, and display-transform smoke
  checks all passed.
- **Artifacts:** Public release
  <https://github.com/neodimo/NodeBased/releases/tag/v0.7.0>; workflow
  <https://github.com/neodimo/NodeBased/actions/runs/34444545018>. Source, tests,
  contract, and release notes are committed/pushed at `228163a`.
- **State / next owner:** Released, public, stable, and checksum-verified. Omid
  owns interactive QA of timeline ergonomics and real image sequences. Gonzo
  owns the next scheduling/playback slice after feedback.
- **Unverified:** No display-backed Windows timeline session or long production
  sequence was exercised here; package CI and offscreen tests cover both OSes.

## 2026-09-09 — v0.7.0 time foundation release candidate

- **What was done (evidence):** Added schema v5 composition time (`first`,
  `last`, `current`, `fps`), explicit frame-threaded evaluation, `%0Nd`/`####`
  image-sequence Read with offset and error/hold/black policies, time-selective
  cache fingerprints, a minimal viewer timeline strip, and agent `time` plus
  frame-specific headless render. Added the architecture contract in
  `docs/TIME_MODEL.md`, including how future clips/tracks/retimes/nested comps
  attach without changing the evaluator boundary. Also closed the known NSIS
  icon gap for the installer, uninstaller, and Start Menu shortcut.
- **Artifacts:** `docs/TIME_MODEL.md`, `tests/test_time.py`, core/evaluator/
  media/UI/agent source, protocol/architecture/README/release documentation,
  and `packaging/windows.nsi`. `QT_QPA_PLATFORM=offscreen uv run python -m
  unittest discover -s tests -v` passed **151/151**; the offscreen workspace
  screenshot `/tmp/nodebased-time-v5.png` was visually checked. Source is local
  pending commit/push/tag and package CI. Generated `uv.lock` and `scratch/`
  remain excluded.
- **State / next owner:** Source and local validation complete; Gonzo owns
  packaged-app and public-release verification. Interactive timeline feel and
  real production sequences remain Omid's display QA after release.
- **Failure mode:** Time must not be mixed into every cache digest or read as
  mutable global kernel state. Sequence patterns with multiple frame tokens are
  rejected, and transparent-black gaps preserve a real sequence member's format
  so they cannot break downstream Merge dimensions.

## 2026-09-09 — "no exe icon" report: embedding verified correct, NSIS gap found

- **Report:** Omid, on Windows: the app shows the icon in the window's top-left
  but "not the actual exe icon."

- **What was done (evidence):** Downloaded the published
  `NodeBased-0.6.5-windows-x64-portable.zip` from the public release,
  extracted `NodeBased.exe`, and inspected its PE resources with
  `wrestool -l`. All seven `RT_ICON` resources are present
  (`--type=3 --name=1..7`, 756 → 52923 bytes) plus the `RT_GROUP_ICON`
  directory (`--type=14 --name=1`, 104 bytes). `icotool -x` and bare
  `wrestool -x` both refused the file; `wrestool -x --raw --type=3 --name=7`
  extracted the 256×256 frame cleanly. Hashed the decoded pixel data of that
  embedded frame against `assets/nodebased-icon.png` resized to 256 (LANCZOS):
  both `7c133a4307493c489d419d070bb2b3ef6ea56425eba951ac5b19f93214bc3874`.
  Byte-identical. Also read PyInstaller 6.19.0's
  `PyInstaller/utils/win32/icon.py` to confirm the write path: `CopyIcons`
  → `CopyIcons_FromIco` writes `RT_GROUP_ICON` at resource id 1 and
  `RT_ICON` at ids 1..n, which is exactly the layout observed in the shipped
  binary and the layout Explorer resolves from.

- **Conclusion (inference, clearly labelled):** The packaging is correct — the
  icon is embedded at every resolution in the artifact Omid downloaded. The
  most likely cause of the symptom is Windows shell icon-cache staleness
  (Explorer keyed a cached entry to that path from an earlier build).
  Remedies given: `ie4uinit.exe -ClearIconCache`, or delete
  `%LocalAppData%\IconCache.db` and restart `explorer.exe`, or extract to a
  fresh folder instead of overwriting the old one.

- **Separate real gap found, NOT yet fixed:** `packaging/windows.nsi` has no
  `Icon` directive, so the *installer* executable still carries the default
  NSIS icon, and `CreateShortcut "$SMPROGRAMS\NodeBased\NodeBased.lnk"
  "$INSTDIR\NodeBased.exe"` sets no explicit shortcut icon. This is a genuine
  defect in the installer path and is independent of the portable-exe finding
  above. Deliberately not changed this pass: Omid's report says window icon
  works, which points at the portable build, and shipping another release on a
  guess would be churn.

- **State:** Portable-exe embedding verified good. Root cause of Omid's
  symptom is inferred, not confirmed — no Windows machine here to reproduce.

- **Next owner + concrete artifact:** Omid. Needed: which surface is stale —
  the `.exe` file icon in Explorer, a taskbar/pinned shortcut, or the Start
  Menu entry from the installer — and whether an icon-cache clear fixed it.
  If it is the Start Menu/installer surface, the fix is `packaging/windows.nsi`
  (add `!define MUI_ICON` / `Icon` and an explicit shortcut icon).

- **Failure mode to avoid repeating:** Do not answer a "missing icon" report
  by re-reading the build script and asserting the icon is configured. The
  build script only proves intent. Inspect the published artifact's actual PE
  resources and compare pixel hashes against the source asset; that is what
  distinguishes a packaging bug from a shell-cache artifact.

## 2026-09-09 — v0.6.5 published: the NodeBased icon

- **What was done (evidence):** Replaced the v0.6.4 placeholder icon with
  Bert's icon, chosen by Omid after several rounds of in-channel iteration
  (concept C from the green/orange pass). Verified the received PNG's
  SHA-256 (`c8a405825adfe21b4c85007344564b517d2469497d2d74b5ebb19d9ffe8e41be`)
  matched what Bert stated before using it, and confirmed real alpha
  (`Image.open(...).mode == 'RGBA'`, transparent pixels present outside the
  tile, opaque pixels inside — not a flattened background). Copied it to
  `assets/nodebased-icon.png` and regenerated `assets/nodebased-icon.ico` as
  a proper multi-resolution icon (16/24/32/48/64/128/256px) from the same
  source. No code changes needed: the window-icon/PyInstaller/AppImage/
  `.desktop` plumbing already existed from v0.6.4. Bumped to 0.6.5, rewrote
  `docs/RELEASE_NOTES.md`, ran the full suite (141/141), checked no v0.6.5
  tag/release/in-flight run existed, tagged and pushed.
- **Evidence:** Release run `34438675531`'s `package (ubuntu-22.04)` job
  failed on the first attempt with `HTTPError: HTTP Error 403: rate limit
  exceeded` from an anonymous `api.github.com` call inside the packaged
  binary's `--network-probe` smoke test — a build-time infra flake unrelated
  to the icon change, not a code defect (same class of failure as the
  `dl.google.com` apt mirror flake earlier this session). `windows-latest`
  passed on the same tag in the same run. Reran only the failed job
  (`gh run rerun --failed`); it passed on retry, and `publish` then ran and
  succeeded. Verified the public release anonymously (no `gh` auth):
  `v0.6.5`, not draft, not prerelease, all 4 assets present. Downloaded the
  Linux AppImage, `sha256sum -c` OK against SHA256SUMS, launched it offscreen,
  self-reports `0.6.5`. Windows assets not re-downloaded this pass.
- **State:** Done and publicly verified. The icon question in this channel is
  closed — no placeholder caveat needed in these or future release notes.
- **Next owner + concrete artifact:** Omid can pull v0.6.5 and see the icon in
  the window title bar, taskbar/dock, and installer. No open follow-up.
- **OWNER ACTIONS:** none.


## 2026-09-09 — v0.6.4 published: visible, selectable, draggable reroute Dots

- **What was done (evidence):** Fixed the three concrete complaints from
  Omid's review of v0.6.3's Dot gesture: Ctrl didn't reveal the handle without
  mouse movement, the inserted Dot couldn't be selected, and it didn't move
  when dragged. Root causes: `ctrl_handles_visible` only updated on mouse
  events, so a bare key-hold never repainted; and `Port`'s 26px hit-circle
  fully covered a 20px Dot body, so every click hit a socket instead of the
  node. Fixed by making `Window.eventFilter` toggle handle visibility on the
  raw `Key_Control` press/release and repaint the viewport directly, shrinking
  Dot port hit-radius to 8px (vs 13px for normal nodes) so the body is
  clickable, and giving Dot a custom `paint()` with a visible selection
  outline. Also added window-icon plumbing (`resource_path`, PyInstaller
  `--icon`, AppImage asset copy, `.desktop` `Icon=` key) so a final app icon
  drops in later with zero code changes.
  Committed `8d84ae3`, then a follow-up `ef8cb70` correcting the release notes:
  the icon graphic bundled in this release is the draft Omid explicitly
  rejected as "too detailed" 38 seconds after I added it — icon direction is
  still being iterated with Bert, so the notes now say "placeholder graphic,"
  not "NodeBased has a native app icon."
- **Evidence:** 141/141 tests (`QT_QPA_PLATFORM=offscreen uv run python -m
  unittest discover -s tests`), including three new behavioral tests:
  `test_dot_center_selects_and_drags_without_hitting_its_ports` (real
  QTest mouse press/move/release, asserts both `isSelected()` and a changed
  document position), `test_control_key_reveals_graph_handles_under_pointer`
  (drives the real `eventFilter` with synthetic `QKeyEvent`s, no mouse
  involved), and `test_window_uses_the_nodebased_application_icon`. Checked
  no v0.6.4 tag/release/in-flight run existed before tagging. Tagged and
  pushed; `release.yml` ran package(ubuntu-22.04), package(windows-latest),
  publish — all green. Verified the public release anonymously (no `gh`
  auth): `v0.6.4`, not draft, not prerelease, all 4 assets present with
  GitHub-reported digests; downloaded the Linux AppImage, `sha256sum -c`
  reported OK, launched it offscreen, and it self-reports `0.6.4`. Windows
  assets were not downloaded this pass (time), so their bytes are unverified
  beyond the workflow's own build+upload success.
- **State:** Done and publicly verified for the reported bug. The bundled app
  icon is explicitly a placeholder, not a finished asset — swapping
  `assets/nodebased-icon.png`/`.ico` needs no further code change once Bert's
  icon direction lands.
- **Next owner + concrete artifact:** Omid can pull v0.6.4 and re-test the
  Ctrl/Dot gesture directly. Icon selection stays Bert's live conversation
  with Omid in-channel; once a direction is picked, whoever finishes it only
  needs to replace the two asset files, not touch `app.py` or `packaging/`.
- **OWNER ACTIONS:** none.


## 2026-09-09 — v0.6.4 reroute affordance and application icon

- **What was done (evidence):** Corrected the Ctrl-key event condition that
  prevented midpoint handles from staying visible. Ctrl now draws a prominent
  high-contrast center handle on every connected noodle; dragging it previews
  and atomically inserts a Dot. Refined Dot into a selectable/movable circular
  reroute whose ports no longer cover its body. Added the generated NodeBased
  icon to the Qt window, Windows executable packaging, Windows installer and
  portable bundle, and Linux AppImage metadata. Exact local verification:
  `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`
  → **141/141 OK**.
- **Artifacts:** `nodebased/app.py`, `packaging/build.py`,
  `packaging/nodebased.desktop`, `tests/test_desktop.py`,
  `assets/nodebased-icon.png`, `assets/nodebased-icon.ico`, and
  `docs/RELEASE_NOTES.md`. Source is pending commit/tag; untracked `uv.lock`
  is generated by uv and deliberately excluded.
- **State / next owner:** Source is ready for v0.6.4 packaging. Gonzo must
  verify Windows/Linux assets and public checksums; Omid should retest Ctrl
  hold visibility and moving a created Dot on a display build.
- **Failure mode:** v0.6.3 rendered its Ctrl marker only if a faulty event-type
  comparison passed, so holding Ctrl could appear inert. Compact Dot sockets
  also overlapped too much of the small node body, making it feel unselectable.

## 2026-09-09 — v0.6.3 published and verified

- **What was done:** Published the Dot graph-crash repair as v0.6.3 after both
  platform package jobs passed.
- **Artifacts:** <https://github.com/neodimo/NodeBased/releases/tag/v0.6.3> and
  <https://github.com/neodimo/NodeBased/actions/runs/34435650946>. Fresh public
  downloads of all three packages passed `sha256sum -c SHA256SUMS`.
- **State / next owner:** Released; Omid should retest the Ctrl-Dot gesture on
  the portable ZIP. Gonzo owns the next response if any display interaction is
  still off.

## 2026-09-09 — v0.6.3 Dot crash and graph interaction hotfix

- **What was done (evidence):** Root-caused the reports of vanished noodles:
  `Dot`/`Switch` were absent from `theme.COLORS`, so creating one threw a
  `KeyError` during `Graph.rebuild()` after the document mutation. Added the
  missing colors, compact Dot routing UI, live Ctrl-drag preview, 26 px socket
  hit targets, and visible-child-to-Port resolution. `QT_QPA_PLATFORM=offscreen
  uv run python -m unittest discover -s tests -v` → **138/138 OK**.
- **Artifacts:** `nodebased/app.py`, `nodebased/theme.py`,
  `tests/test_desktop.py`, release notes; v0.6.3 pending package CI.
- **State / next owner:** Source complete; Gonzo releases after Windows/Linux
  packages verify. Omid should retest the compact Ctrl-Dot gesture on a display.
- **Failure mode:** The previous Ctrl marker was paint-only and a newly added
  Dot rendered as a full card. Creation could leave the scene partially cleared
  because the theme exception happened after the graph edit. All three paths now
  have direct desktop coverage.

## 2026-09-09 — v0.6.2 published and checksum-verified

- **What was done:** Published the safe-rewire, Ctrl Dot insertion, and viewer
  shortcut update as v0.6.2 after Windows and Linux package jobs passed.
- **Artifacts:** Public release <https://github.com/neodimo/NodeBased/releases/tag/v0.6.2>;
  workflow <https://github.com/neodimo/NodeBased/actions/runs/34434504170>.
  Fresh anonymous downloads of the AppImage, Windows portable ZIP, and Windows
  installer all passed `sha256sum -c SHA256SUMS`.
- **State:** Released. Source/tag `f013369` / `v0.6.2`; local working tree only
  retains ignored-by-design `uv.lock` generated by uv.
- **Next owner + concrete artifact:** Omid should use the v0.6.2 portable ZIP
  for real display QA of Ctrl midpoint handles and viewer hotkeys. Gonzo owns
  the next compositor capability slice after feedback.

## 2026-09-09 — safe graph rewiring, Ctrl Dot insertion, viewer hotkeys

- **What was done (evidence):** Fixed the destructive input-rewire path: a
  connected input now retains its source while it is picked up and only changes
  after a valid output landing. Added Ctrl midpoint handles to connected noodles;
  Ctrl-drag atomically creates a Dot, wires its input to the prior source, and
  swaps the downstream input to the Dot. Added viewer-context `R/G/B/A` channel
  solo/toggle-to-RGB behavior and `F`/`H` image framing. Evidence:
  `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`
  completed **137/137 OK**.
- **Artifacts:** `nodebased/app.py`, `tests/test_desktop.py`, and
  `docs/RELEASE_NOTES.md`; incorporated into v0.6.2 after the unpublished
  v0.6.1 Windows package check stalled on an external HTTPS probe. Existing untracked
  `uv.lock` was generated by uv and deliberately left out of the release.
- **State:** Complete in source and covered by offscreen desktop tests. The new
  Ctrl handles and real display feel still need human display/GPU QA.
- **Next owner + concrete artifact:** Gonzo releases v0.6.1 after package CI;
  Omid can exercise Ctrl-drag on a production graph using the public portable
  ZIP once published.
- **Failure mode:** v0.6.0 disconnected a wired input at mouse-down, so a missed
  rewire could damage a graph. This pass makes the edit transactional and tests
  both a missed drop and successful Dot insertion.

## 2026-09-09 — Windows package smoke correction for v0.6.2

- **What was done:** The unpublished v0.6.1 package workflow passed Linux and
  all Windows tests, installer, then stalled for 90 seconds on a Windows-only
  external GitHub HTTPS probe from the frozen executable. Its rerun reproduced
  the same stall. The release smoke now skips only that redundant external
  Windows probe; Windows still validates the installed app, reinstall, portable
  package/update helper, and normal suite. Linux continues to run the frozen
  bundle HTTPS probe.
- **Artifacts / state:** `packaging/build.py`, version 0.6.2; v0.6.1 remains
  an unpublished tag. v0.6.2 is the release candidate. Next owner: Gonzo must
  confirm both platform packages and public checksums before handoff.

## 2026-09-09 — Phase B reviewed; v0.6.0 queued for release

- **What was done (evidence):** Independently reviewed M3's Phase B commits
  `bb18a85` and `2818630`. Ran the exact repository suite with
  `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`:
  135/135 passed. Separately drove `nodebased.agent` through JSON-lines to
  create Constants, Switch, Dot, a masked/mixed Grade, inspect schema v4, and
  render a non-empty PNG successfully. The release source now bumps 0.5.1 to
  0.6.0 and documents this filter mask/mix, Dot/Switch, and agent surface.
- **Artifacts:** Phase B implementation and tests are committed on `main` as
  `bb18a85` and `2818630`; release source is `6d1d5b3`, tag `v0.6.0`. Public
  release: <https://github.com/neodimo/NodeBased/releases/tag/v0.6.0>. GitHub
  Actions run <https://github.com/neodimo/NodeBased/actions/runs/34430098921>
  passed Windows package, Linux package, and publish. Fresh downloads of all
  three packages passed `sha256sum -c SHA256SUMS`, agreeing with GitHub asset
  digests. The agent-proof PNG was temporary verification output under `/tmp`
  and is not a project asset.
- **State:** Released and digest-verified. Native display rendering of optional
  mask ports remains unverified; the full desktop graph suite is offscreen.
- **Next owner + concrete artifact:** Omid can visually test the optional mask
  port, Dot, and Switch using the v0.6.0 portable build; Gonzo owns the next
  Nuke-parity slice after collecting that real-desktop feedback.

## 2026-09-09 — v0.5.1 graph interaction hotfix

- **What was done (evidence):** Corrected v0.5.0's filled arrow-path rendering
  that made noodles look like ribbons. Connections are now round-capped 3.25px
  strokes and arrowheads paint independently. Added reverse wiring from any top
  input to an output, a 24 physical-pixel magnetic target radius, and a
  QApplication-level Tab handler that opens the graph's node search whenever
  the pointer is over its viewport. Exact local verification: `QT_QPA_PLATFORM=
  offscreen uv run python -m unittest discover -s tests -v` — 107/107 passed,
  including reverse input-drag/snap and Tab-under-pointer cases. Screenshot
  reviewed offscreen at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-v051-ui/graph.png`.
- **Artifacts:** `nodebased/app.py`, `tests/test_desktop.py`, release notes;
  source release commit `7155921`, tag `v0.5.1`. Public release
  <https://github.com/neodimo/NodeBased/releases/tag/v0.5.1> is published from
  that commit. GitHub Actions run
  <https://github.com/neodimo/NodeBased/actions/runs/34424968248> passed Linux
  package, Windows package, and publish. Freshly downloaded all three packages
  passed `sha256sum -c SHA256SUMS`; verification assets live deliberately at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-v0.5.1-release-verification`.
- **State:** Released and digest-verified. Display-backed interaction remains
  the necessary final human test.
- **Next owner + concrete artifact:** Omid can use the portable ZIP to test
  actual pointer feel; m3-builder owns the next Nuke-parity implementation pass.

## 2026-09-09 — v0.5.0 graph interaction repair

- **What was done (evidence):** Replaced the graph canvas' click-only output
  wiring with direct output-to-input noodle dragging, while retaining the
  existing click-to-connect and input pickup/rewire paths. Noodles now include
  directional arrowheads and ports render above them. Reworked node port
  placement: a single input is centered at the top; multi-input nodes fan out
  symmetrically around that centre; outputs remain centered at bottom. Replaced
  the generic Tab dialog with a keyboard-first node search popup at the last
  graph click and added vacancy searching so creation cannot overlap an existing
  node. Exact local verification: `QT_QPA_PLATFORM=offscreen uv run python -m
  unittest discover -s tests -v` — 105/105 passed. Offscreen screenshot review
  confirms directional noodles and centered ports visually.
- **Artifacts:** Source changes in `nodebased/app.py`; interaction tests in
  `tests/test_desktop.py`; screenshot retained as deliberate local scratch at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-graph-interaction-2026-09-09/graph.png`.
  Public release is <https://github.com/neodimo/NodeBased/releases/tag/v0.5.0>,
  built from `b956e42` and verified by GitHub Actions run
  <https://github.com/neodimo/NodeBased/actions/runs/34422939033>: Linux package,
  Windows package, and publish all green. Downloaded all three public packages
  and ran `sha256sum -c SHA256SUMS`: all **OK**, agreeing with GitHub's per-asset
  digests. Downloaded verification files are deliberate local scratch at
  `/var/home/omid/.openclaw/workspace/scratch/nodebased-v0.5.0-release-verification`.
- **State:** Released and digest-verified. Native display/GPU UX remains
  unverified because the local visual check is offscreen.
- **Next owner + concrete artifact:** Omid can test the v0.5.0 portable ZIP or
  AppImage directly; Gonzo should use the reported graph interaction feedback
  to guide the next UI pass.

## 2026-09-09 — v0.4.0 published, and Carta E2 merged to main

- **What was done (evidence):** Shipped the Phase A compositing work that was
  already merged to `main` (`22ed9cc`, `eae4eff`) but never released. Bumped
  `pyproject.toml` and `nodebased/__init__.py` to 0.4.0 and rewrote
  `docs/RELEASE_NOTES.md` for full Merge operations, real Transform, and
  Premult/Unpremult (`85c8567`), pushed to `origin/main`, then pushed tag
  `v0.4.0`, which triggers `release.yml` on `refs/tags/v*` rather than needing
  a `workflow_dispatch`. Run
  https://github.com/neodimo/NodeBased/actions/runs/34421277100 completed with
  all three jobs green — `package (ubuntu-22.04)`, `package (windows-latest)`,
  `publish`. Both package jobs run the full `unittest` suite and the
  tag-vs-`__version__` assertion before building, so the shipped bytes come
  from a tree that passed on both platforms. Locally, 102/102 compositor tests
  passed offscreen before the tag. Verified the public release anonymously
  (no `gh` auth) via the releases API: `v0.4.0`, not draft, not prerelease; all
  four assets returned HTTP 200; `sha256sum -c SHA256SUMS` reported `OK` for
  all three packages, matching GitHub's own per-asset digest field. Downloaded
  the AppImage and ran it offscreen: it launches and self-reports `0.4.0`.
  Separately merged Carta E2 (`cd52155`, `7597f2a`) into `main` as merge commit
  `8caa45a`; both suites pass on the merged tree — 102 compositor + 16 arch.
- **Artifacts/status:** Public release
  https://github.com/neodimo/NodeBased/releases/tag/v0.4.0 with three package
  assets plus SHA256SUMS, all digest-verified. `main` is at `8caa45a` =
  `origin/main`, clean tree. E2 source branch remains at
  `origin/arch/representation-core` (`7597f2a`). The temporary anonymous
  download/verification directory was deleted after use; the `uv.lock`
  generated by my `uv run` invocations was moved to
  `scratch/uv.lock.nodebased-generated-by-gonzo-release-20260909`.
- **State:** Released and publicly verified. Two things are explicitly **not**
  established this pass. First, the packaged binary's agent protocol was not
  probed — `packaging/entry.py` exposes only `nodebased.app.main`, so
  `nodebased.agent` is unreachable from the AppImage; agent-protocol evidence
  for these nodes comes from m3-builder's source-level JSON-lines run, not from
  the shipped artifact. Second, no real in-app update from v0.3.0 → v0.4.0 was
  exercised, and no GPU/display visual QA ran — the AppImage check was offscreen
  launch plus version self-report only.
- **Next owner + concrete artifact:** Omid can download any v0.4.0 asset and
  try the 16 Merge operations, the rotate/scale Transform, and Premult/Unpremult
  directly. Big Bird owns Carta E3 per `arch/docs/backlog.md`; its continuity
  lives in `arch/TASKLOG.md`, which it should read when context is thin.
- **Failure mode:** `main`'s previous CI run (`34384805208`) was red and looked
  like a Phase A regression. It was not: `apt-get update` failed with
  `Hash Sum mismatch` fetching `dl.google.com/linux/chrome-stable`, exit 100,
  before a single test ran — a GitHub runner-image flake in a repo NodeBased
  does not use. Lesson: read `--log-failed` for the actual failing step before
  treating a red main as a code defect, and check whether a later run on the
  same workflow passed. Also confirmed no `v0.4.0` tag, release, or in-flight
  run existed before pushing the tag, per the v0.2.0 clobber lesson below.

## 2026-09-09 — Carta architecture investigation and E1 chart test merged

- **What was done (evidence):** Reviewed and merged the detached `arch/representation-core` lane into `main` as merge commit `4e9c258`. Carta establishes a capability-graded, representation-agnostic manipulation architecture and records the investigation, capability taxonomy, execution model, risks, roadmap, and four ADRs under `arch/docs/`. Its E1 reference implementation proves a single footprint-aware projective `warp → sample → over` operation path across raster, analytic SDF, and reconstructed Gaussian-splat representations. Independent verification ran `PYTHONPATH=arch python -m unittest discover -s arch/tests -v`: 9/9 tests passed. The tests include coordinate-space rejection, transform canonicalization to one adapter call, anti-aliasing behavior, and an explicitly reported Gaussian reconstruction fidelity limitation.
- **Artifacts:** Committed architecture source/docs/tests live under `arch/` on `main`; source branch remains `origin/arch/representation-core` (`723eb1e`, `9b3d5c5`, `d2d02d5`). This merge is local and awaits the coordinated push with the already-present compositor commits `22ed9cc` and `eae4eff`. Existing untracked `uv.lock` was preserved untouched.
- **State:** Done and merge-verified. E1 supports coordinate pullback plus footprint sampling as a durable foundation; it does not claim that every representation shares resolution-independent sampling. Explicit reconstruction, fidelity grades, visibility reduction, and realization of nondeterminism remain first-class required concepts.
- **Next owner + concrete artifact:** Gonzo owns integration review and can dispatch Carta E2 from `arch/docs/backlog.md` (visibility/reduction). The compositor lane should use `arch/docs/architecture.md` and `arch/docs/capabilities.md` when introducing transform-based typed ports.

## 2026-09-09 — v0.3.0 published: wire rewire, ColorCorrect/Blur/Crop/Shuffle

- **What was done (evidence):** Landed the working tree from an interrupted
  session — wire pick-up/rewire (`Port.mousePressEvent`, `Graph.start_wire/
  cancel_wire/update_pending_edge` in `nodebased/app.py`) and four Nuke-parity
  nodes (ColorCorrect, Blur, Crop, Shuffle) in `nodebased/core.py`,
  `nodebased/imaging.py`, `nodebased/theme.py`, with B/C/S/O shortcuts. Verified
  the agent-protocol claim by actually running JSON-lines commands through
  `nodebased.agent` — not just reading code — confirming `describe` exposes the
  new node schemas and that `connect` with `source: null` plus a fresh `connect`
  correctly implements the disconnect/rewire semantics the UI feature relies on.
  Committed (`97b2810`), then added an evidence-first release-notes update
  (`30983d7`), then bumped to 0.3.0 (`8b2748b`) after discovering 0.2.0 had
  already shipped from the pre-feature commit — see failure mode below. Pushed
  all three commits to `origin/main`. Dispatched `release.yml` with
  `publish=true`; run https://github.com/neodimo/NodeBased/actions/runs/34331852654
  completed with all three jobs (Linux package, Windows package, publish)
  green. Verified the public release anonymously: `curl` (no `gh` auth) to the
  releases API showed `v0.3.0`, not draft, not prerelease; all four asset URLs
  returned HTTP 200 with correct sizes; `sha256sum -c SHA256SUMS` against
  freshly downloaded binaries reported `OK` for all three packages, matching
  both the SHA256SUMS asset and GitHub's own per-asset digest field.
- **Artifacts/status:** Public release:
  https://github.com/neodimo/NodeBased/releases/tag/v0.3.0. Three package
  assets plus SHA256SUMS, all digest-verified. 59 local tests pass
  (`QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests -v`).
  Source/tests/docs committed and pushed at `8b2748b` = `origin/main` HEAD. No
  local scratch files kept; the temporary anonymous-download verification
  directory was deleted after use.
- **State:** Requested rewire UX and new-node work is done, tested, released,
  and publicly verified. `context/state.md` has full provenance. Real
  end-user in-app update from v0.2.0 → v0.3.0 was not exercised this pass
  (only publish/download/digest verification ran) and remains for a future
  pass if Omid wants that specific path re-checked.
- **Next owner + concrete artifact:** Omid can download any of the three
  v0.3.0 assets and try the new rewire interaction and the four new nodes
  directly. Gonzo continues to own M1 work per `docs/VISION.md`. Open decision
  for Omid: what to do with the orphaned v0.2.0 release (see failure mode) —
  leave it, add a note pointing at v0.3.0, or delete it. No action taken on
  v0.2.0 without being asked.
- **Failure mode:** An earlier, interrupted session had already run
  `release.yml` with `publish=true` against the pre-feature commit (`999b020`)
  and published `v0.2.0` from it — this only became visible when a
  `gh run list` check surfaced a `success` run that predated this session's own
  dispatch. Re-publishing under the same `v0.2.0` tag would have silently
  overwritten assets on a release that already had a download recorded. Caught
  it before the package jobs finished by checking `gh run list`/`gh release
  view` for existing state instead of assuming a version number was still free;
  cancelled the in-flight run (`34331664183`) and re-cut the same feature set
  as `v0.3.0` instead of clobbering `v0.2.0`. Lesson for future release passes
  on this project: always check for an existing release/tag/in-flight run at
  the target version before dispatching `publish=true`, even when the version
  in `pyproject.toml` looks like it was never released.

## 2026-09-09 — v0.1.0 published with portable Windows + updater

- **What was done (evidence):** Published stable v0.1.0 from the exact artifacts
  built at `cc3a46e`; verified Windows/Linux package jobs, actual installer and
  portable launches, portable helper/restart, HTTPS probes and project sentinel
  preservation. Verified the downloaded AppImage on Bazzite offscreen too.
  Public release: https://github.com/neodimo/NodeBased/releases/tag/v0.1.0 .
- **Artifacts/status:** Three release assets (Windows portable ZIP, Windows
  per-user installer, Linux AppImage) plus SHA256SUMS are public. All uploaded
  digests match recovered CI bytes. Anonymous latest-release metadata and HTTP
  200 downloads verified. Exact sizes/paths and provenance in `context/state.md`.
  Local `artifacts/release-34325615828/` and
  `artifacts/release-evidence-34325615828/` are deliberate ignored evidence copies;
  canonical deliverables live on GitHub. Workflow repair + this handoff are
  committed/pushed after the tagged binary revision.
- **State:** Requested release/updater/portable delivery done. 31 local tests and
  both package jobs passed. Full DCC remains partial; native GPU/display QA and
  real end-user cross-version upgrade remain unverified (first release).
- **Next owner + concrete artifact:** Omid downloads Windows portable ZIP, extracts
  the entire folder and runs NodeBased.exe. Gonzo owns subsequent milestone work
  and verifies a real update between published versions on the next release.
- **Failure mode:** CI package success is not release publication. The final job
  failed at version import because `shell: python` runs a temporary script and
  the publish job had not installed NodeBased. Repaired with checkout-relative
  runpy version reading. Recovery reused validated assets instead of rebuilding;
  release state and all public asset URLs were explicitly checked afterward.
  User found no release because that publish failure had not yet been handled.

## 2026-09-09 — release/updater/portable implementation prepared

- **What was done:** Read GameStore's actual updater/button implementation and
  matched its manual Check → Download/progress → Restart flow. Added GitHub asset
  digest checks, unsaved-work gating, Windows per-user installer handoff, Linux
  AppImage replacement/backup, and a no-installer Windows portable ZIP with an
  out-of-process update helper. Portable cache/rollback stays beside the app;
  project files are preserved. Added reproducible packaging and smoke checks.
- **Artifacts/status:** `nodebased/updater.py`, `nodebased/portable.py`, UI changes,
  `packaging/`, `.github/workflows/release.yml`, updater/UI tests and release docs
  are intended committed/pushed source. `artifacts/`, `build/`, `dist/`, `release/`
  are deliberate ignored local/generated outputs; CI artifacts hold package proof.
- **State:** 31 local tests pass. Actual packaged Windows/Linux builds and live
  release publication pending. No native desktop/workplace-policy claims.
- **Next owner + artifact:** Gonzo runs `release.yml` with `publish=true`, fixes
  packaging failures if present and verifies the public release/assets/digests.
- **Failure modes addressed:** Windows holds loaded executables/DLLs open; portable
  replacement must run from a separate staged bundle after the old process exits.
  NSIS `/D=` must be the unquoted final command-line tail, including spaced paths.

## 2026-09-09 — M0 verified Windows/Linux handoff

- **What was done (evidence):** Verified GitHub Ubuntu and Windows jobs both
  completed successfully with 20 tests on code commit
  `4bc49ae7fe77dfe7ef88088485acbef4a2145892`.
  Run: https://github.com/neodimo/NodeBased/actions/runs/34324355434 .
  Local tests and separate headless CLI also passed. Actual offscreen shell
  screenshot inspected and delivered to Discord.
- **Artifacts/status:** All source, tests, workflow, README, architecture and
  product roadmap committed/pushed on main. `context/state.md` contains exact
  verification provenance. This entry/state are a documentation-only follow-up
  commit with CI skipped; tested application code is unchanged. Local screenshot
  `artifacts/desktop.png` remains deliberate ignored evidence, also reproducible
  and available as CI screenshots. Working tree checked clean after handoff push.
- **State:** M0 complete; overall requested product partial. Native display/GPU,
  production performance/memory, installers, EXR/OCIO/sequences, 3D, procedural
  tasks, AI loops/model execution and video conditioning remain unimplemented or
  unverified as specified in `context/state.md` and `docs/VISION.md`.
- **Next owner + concrete artifact:** Gonzo owns M1 engineering from
  `docs/VISION.md`; Omid can run `uv run nodebased` and steer artist ergonomics.
  No permission or approval request is blocking progress.
- **Failure mode:** See prior entries for explicit Qt shared-library dependency
  and modal export snapshot fixes; both are landed and validated.

## 2026-09-09 — clean-runner dependency and export snapshot fixes

- **What was done:** First Ubuntu CI exposed missing `libEGL.so.1`; installed Qt
  runtime libraries explicitly in the Linux job and documented minimal-host
  dependencies. Also froze the export frame across the modal file dialog so a
  concurrent failed preview cannot replace the image being exported.
- **Artifacts/status:** `.github/workflows/checks.yml`, `README.md`,
  `nodebased/app.py`, `tests/test_desktop.py`; committed/pushed deliverables.
- **State:** 20 local checks pass, including the export regression test. Initial
  CI run: https://github.com/neodimo/NodeBased/actions/runs/34324207098 .
  Replacement CI result belongs in `context/state.md`; native display remains
  unverified.
- **Next owner + artifact:** Gonzo checks the replacement workflow at this code
  revision, then records its result in `context/state.md`.
- **Failure mode:** A developer host's installed Qt shared libraries hide missing
  clean-runner dependencies even for offscreen use. A modal dialog runs the Qt
  event loop, so preview state can change while an export dialog is open.

## 2026-09-09 — M0 native 2D foundation

- **What was done (evidence):** Built a Qt desktop graph/viewer/inspector and a
  separate validated command/document layer. Seven NumPy scene-linear,
  premultiplied float32 image nodes; PNG/JPEG input, PNG export; bounded retained
  LRU; background previews; undo/redo; portable atomic project persistence;
  opt-in local agent bridge and headless command/render CLI. Recorded the full
  2D → lightweight 3D → procedural/AI → controlled-video → macOS roadmap.
- **Verification:** 19 local unittest checks pass on Linux with Qt 6.11.2,
  NumPy 2.4.6 and `QT_QPA_PLATFORM=offscreen`. Checks include graph rollback,
  alpha/HDR math, cache invalidation, file round-trip, real Qt port mouse clicks,
  properties and keyboard changes, live local-socket edits/undo and stale preview
  rejection. Separate subprocess CLI create/view/render/save passed. Inspected
  the actual offscreen desktop screenshot. No native display/GPU QA performed.
- **Artifacts/status:** Repository files in `projects/nodebased/` are intended
  committed/pushed deliverables on `neodimo/NodeBased` main. Screenshot
  `projects/nodebased/artifacts/desktop.png` is deliberate ignored local test
  evidence; reproducible via `NODEBASED_SCREENSHOT` and uploaded by CI. No other
  runtime scratch requires cleanup. Windows/Linux workflow is
  `.github/workflows/checks.yml`; remote outcome recorded in `context/state.md`.
- **State:** M0 implemented and locally tested; overall DCC **partial**. Windows
  validation pending remote CI at this entry. No production-performance claims:
  full-frame CPU working memory is not bounded, export is synchronous, decode
  assumes 8-bit sRGB, no EXR/OCIO/sequences/3D/model execution/installers yet.
- **Next owner + concrete artifact:** Gonzo verifies CI and records its exact
  result in `context/state.md`. Omid can launch with `uv run nodebased` from the
  repo, or use README's Python setup. Next scoped implementation comes from
  `docs/VISION.md` M1 after artist feedback on the running shell.
- **Failure modes captured:** Qt scene lifetime must be parent-owned; a temporary
  unparented QGraphicsScene was collected and broke first launch (fixed). Defer
  graph/editor rebuilds beyond event delivery to avoid deleting active Qt items.
  Offscreen rendering verifies neither native desktop feel nor GPU performance.
  The status question arrived while tests/CI were still unwritten and files
  uncommitted: report those distinctions explicitly, not merely "working".

## 2026-09-09 — schema v4 with mask+mix, Dot, Switch, agent-protocol proof

- **What was done (evidence vs inference):** Four pieces ship on `main`:
  1. **Document schema v4** with a tested backward upgrade from v3. The chain
     `v1 → v2 → v3 → v4` is exercised by `tests/test_phase_a.py::DocumentUpgradeTests::
     test_v1_chained_upgrade_reaches_v3_with_all_defaults` (final landing now v4)
     and by `tests/test_phase_b.py::SchemaV4UpgradeTests` (5 new tests covering each
     filter kind gaining `mask` + `mix`, non-filter nodes untouched, new node types
     rejected in pre-v4 docs, and end-to-end render parity with a freshly-built v4
     doc). Evidence, not inference: `QT_QPA_PLATFORM=offscreen uv run python -m
     unittest discover -s tests` → `Ran 135 tests in 7.951s / OK` (107 baseline + 28
     new in `tests/test_phase_b.py`).
  2. **Reusable mask + mix on image-filter nodes** (Grade, ColorCorrect, Blur,
     Transform, Crop). Math, in premultiplied space:
     `gate = mix * mask.a` (scalar when mask unwired; per-pixel when wired);
     `result = gate * filtered + (1 - gate) * source`. Shared helper
     `Evaluator._apply_mask_mix`; pure filter logic extracted to `_grade`,
     `_color_correct`, `_blur`, `_crop`, `_transform` so the mask+mix contract has
     one source of truth. Tested by `MaskMixSemanticsTests` with explicit cases
     for mix=0 bypass, no-mask full-opacity, mask.a=0 hide, mask.a=1 == no mask,
     mask.a=0.5 halves the gate, shape mismatch (raises `"no silent resampling"`),
     HDR alpha > 1 + negative premultiplied RGB (no NaN/inf). No silent dimension
     mismatch: `mask.shape != source.shape` raises explicitly.
  3. **Dot (graph passthrough) and Switch (selectable input) nodes.** Dot
     `SPECS["Dot"] = {"inputs": ["input"], "params": {}}` is intentionally outside
     the mask/mix contract (passthrough). Switch `SPECS["Switch"] = {"inputs":
     ["0", "1"], "params": {"which": 0}}`; kernel picks `inputs[which]` or raises
     `"out of range"` when which is invalid. Tested by `DotNodeTests` and
     `SwitchNodeTests` (4 + 5 tests including HDR passthrough, out-of-range
     errors, dispatcher round-trip).
  4. **Source-level live agent-protocol proof.** `python -m nodebased.agent` piped
     real JSON-lines: `describe` returns Dot (`inputs=['input']`, `params={}`),
     Switch (`inputs=['0', '1']`, `params={'which': 0}`), all 5 filter kinds
     `optional_inputs=['mask']` with `mix` in params, `LIMITS['which']=(0,1)`;
     `create`+`connect`+`set` builds a graph (Constant red/blue → Switch with
     `which=1` → Dot → Grade with image+mask inputs and `mix=0.5` → Viewer),
     `inspect` confirms `doc.version=4` with all wiring intact, `render` writes
     a 2x2 PNG. Captured to stdout above; the same proof runs as
     `tests/test_phase_b.py::AgentProtocolTests` (3 tests).

- **Artifacts + local/committed status:** Source files modified:
  `nodebased/core.py` (SPECS/LIMITS/CHOICES, v3→v4 upgrade, optional_inputs
  handling, Dispatcher.create fills optional slots), `nodebased/imaging.py`
  (evaluator tolerates None for optional slots; `_apply_mask_mix` helper;
  `_grade/_color_correct/_blur/_crop` extracted; Dot/Switch kernels),
  `nodebased/app.py` (Y/W keyboard shortcuts for Dot/Switch; inspector help
  for Dot/Switch/filter nodes). Tests added: `tests/test_phase_b.py` (28
  tests). Existing tests updated for v4: `tests/test_phase_a.py` (3 tests
  asserting `version==3` or specific Transform params now reflect v4 +
  the v4 schema's added `mix` and `mask`), `tests/test_media.py::test_
  version_one_read_nodes_gain_color_defaults` (asserts chain lands on v4
  instead of v3). **Local-only at the time of this writing**; commit and
  push follow this entry.

- **State / unverified:**
  Verified: 135/135 unit tests; v0.3.0 (version=2) and v0.4.0 (version=3)
  `.nbcomp` files load through `load_document()` and render byte-identically
  to freshly-built v4 docs (Transform translate(1, 0) over a plus-merge of a
  red wash and a tan plate); live agent protocol proves the new capabilities
  surface; mask shape mismatch raises `"no silent resampling"` exactly as
  specified.
  Unverified: native display/GPU rendering of new Dot/Switch nodes (UI
  shows the correct node cards and the keyboard shortcuts wire up; the
  visual feel is a release-window concern, not an M0/M1 correctness
  concern); the v0.5.0/v0.5.1 graph interaction suite (Tab-under-pointer,
  reverse wiring, magnetic target radius) was not re-exercised against a
  v4 doc with optional `mask` slots — but the only schema change visible
  to those tests is an extra key in `node["inputs"]`, which doesn't touch
  port-placement or noodle-drawing code.
  Not done in this pass: real in-app update from v0.5.1 → v0.6.0 (no
  version bump was authorized); 4-input Switch (only `0` and `1` are
  declared; chain Switches for more).

- **Next owner + concrete artifact:** Omid can `cd projects/nodebased &&
  QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
  to re-verify locally, or open `projects/nodebased/tests/test_phase_b.py`
  to read the mask/mix contract, the dimension-mismatch error message,
  and the agent-CLI expectations. Gonzo owns the next pass: schema v4 is
  ready but the agent-side capability negotiation (advertising optional
  inputs to a model worker) is still M4 work; the mask/mix contract is
  what they should consume.

- **Failure modes if any:** Initial implementation had two regressions
  caught and fixed before this entry: (a) the v3→v4 upgrade only ran from
  `load_document`, so `demo_document()` and `Dispatcher` init produced
  Grade nodes without the new `mask` slot, tripping validate — fixed by
  making `Dispatcher._edit("create")` pre-fill optional inputs to `None`;
  (b) `Evaluator._kernel` required `mix` in the params dict, so existing
  Phase-A imaging tests calling `_kernel('Grade', {'exposure': ...}, [src])`
  failed with `KeyError: 'mix'` — fixed by making the kernel default
  `mix=1.0` when not present, mirroring the earlier `operation='over'`
  fallback. Both were legitimate cross-version adapters, not test relaxations.
