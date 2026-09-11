# Current state — 2026-09-10 23:42 PDT (re-verified directly)

## Current unreleased head

Implementation is pushed through checkpoint `7697b0a` (`258e8da` on top of `9761660`).
Exact-tree verification: **370/370 tests OK** locally in 75.158s, `git diff --check` clean;
Desktop conformance run `34571040018` passed on Ubuntu and Windows at that exact head.

- Processing is scene-linear **ACEScg float32**, premultiplied RGBA, through evaluator,
  tiles, and caches. Source inputs convert into ACEScg on ingest. Tagged EXRs are honored;
  untagged EXRs fall back to Linear Rec.709.
- Viewer transforms now operate on straight color and re-associate alpha afterward. New
  projects default to ACES 2.0 SDR 100-nit Rec.709. EXR output is tagged ACEScg; PNG output
  converts from ACEScg through OCIO.
- Schema **v7** adds saved project settings. `Edit -> Project settings...` (`S`) exposes the
  bundled config/working-space/display contract and lets the artist set the default view and
  black/checker background through undoable Dispatcher edits.
- The slow-playback CI test now delays the tile executor actually used by the viewer and is
  independent of hosted-runner raster speed.
- No release has been cut; **v0.10.0 remains latest**. Native display and the user's original
  EXR plate remain unverified.

The roto spike's proposed schema v7 now conflicts with main and must become v8 before it can
land.

## Released

**v0.10.0 is the latest published stable release:**
<https://github.com/neodimo/NodeBased/releases/tag/v0.10.0>, published
2026-09-11 01:37 UTC. Both tag workflows — Build release packages and Desktop
conformance — completed **success**. Assets verified with
`gh release view v0.10.0`: `NodeBased-0.10.0-linux-x86_64.AppImage` (105 MB),
`NodeBased-0.10.0-windows-x64-portable.zip` (76 MB),
`NodeBased-0.10.0-windows-x64-setup.exe` (52 MB), `SHA256SUMS`. The AppImage
was downloaded and checked against the published sums (`sha256sum -c` → OK);
the two Windows artifacts are listed in `SHA256SUMS` but were not independently
downloaded and re-hashed.

v0.9.1 was the prior stable release. v0.9.0 was tagged first and its three
workflows **failed** on a Windows cache-root bug: disk-cache init assumed
profile environment variables that a sanitized Windows release-test environment
does not provide. v0.9.1 is the repair (fallback to `TEMP` or the process
directory) plus a regression test in `tests/test_cachetier.py`. v0.9.0 exists as
a tag/release but is not a recommended download.

## Unreleased on `main` since v0.10.0

- `a8e8ce7 Fix playback stalling instead of dropping frames`. DiMo reported an
  EXR sequence freezing on whatever frame play was pressed on while the timeline
  kept running; scrubbing the same sequence was fine. Cause was the transport,
  not animation or tiles: every playback tick cancelled the render in flight,
  and the display gate additionally required a finished frame to still be the
  playhead, so any frame costing more than one frame interval could satisfy
  neither. Measured offscreen: 512px (~7 ms/frame) displayed 12 of 12; 1600px
  with a blur (~570 ms/frame) displayed **0**. Reproduces identically at the
  `v0.9.1` tag, so it was latent from the first transport slice rather than a
  v0.10.0 regression. Fix: a tick replaces the queue without cancelling active
  work, the display gate accepts a result newer than what is on screen while
  playing, and a finished render kicks the preview timer. Pinned by
  `tests/test_desktop.py::SlowPlaybackTests`; reverting either half returns the
  viewer to zero displayed frames.

  **Correction, same evening — the original claim above was overstated.** The
  "0 displayed" and "stalls identically at v0.9.1" figures came from an offscreen
  harness that injected `time.sleep` into `Evaluator.evaluate`, which is not the
  tile path the viewer uses. Re-measured on generated linear float32 EXR
  sequences under a real X server (`tests/manual/qa_exr_playback.py`, frame
  identity decoded from displayed pixels, decoder itself verified against known
  scrub positions):

  | build | 512² no blur | 1600² + blur | 3840×2160 + blur |
  | --- | --- | --- | --- |
  | v0.8.0 | — | — | **0 frames in 12 s** |
  | v0.9.1 | — | — | 0.3 fps, 3 distinct |
  | v0.10.0 pre-fix | 23.8 fps, 12/12 | 0.7 fps, 4 distinct | 0.3 fps, 4 distinct |
  | fixed `a8e8ce7` | 24.0 fps, 12/12 | 4.2 fps, 11 distinct | 2.0 fps, 10 distinct |

  The fix is real: ~6.7× at 4K, ~5.7× at 1600², fast content unchanged. But a
  **total** stall reproduces only at v0.8.0, before the tile engine. v0.9.1 and
  v0.10.0 measure the same as each other, so **DiMo's "it played in the previous
  version and freezes now" is not explained by anything measured.** That symptom
  remains undiagnosed; the open question is which build he considers "previous"
  and what his graph/resolution is.

## Repo facts, checked not inherited

- `main` = `a8e8ce7`, clean tree, pushed. The `v0.10.0` tag stays at `44acff4`.
- `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
  → **Ran 362 tests / OK** at `a8e8ce7` (357 at the v0.10.0 release commit).
- `SCHEMA_VERSION = 6` on `main`.
- `feat/tile-artifact-engine` and `m3/animation-curves` are deleted locally and
  on origin. Remote heads are exactly `main`, `spike/roto-tracker`,
  `arch/representation-core`.

## What v0.9.x delivers

Bounded tile evaluation in 256px tiles with halo-aware Blur; EXR data windows
preserved through Read/evaluation/tile requests including negative overscan
origins; full-resolution bounded Read acquisition by data-window coordinates;
the viewer requests only its visible scene rectangle through `TileExecutor`.
Export always re-renders a complete full-resolution frame, so a viewport crop
cannot escape into a deliverable. Unsupported node kinds (e.g. Transform,
Crop) take an explicit full-frame fallback with telemetry — they do not
silently mis-render.

Measured evidence is in `docs/BENCHMARKS-v0.9-4k.md` (commit `54c5151`):
at 4K, a centered 1920×1080 viewport request is 312 ms cold TTFP vs 2404 ms
for the full-frame evaluator, and 209 ms vs 2053 ms on grade-edit p50 —
7.7× and 9.8× on that supported graph. CPU-only; no GPU claim.

## What v0.10.0 adds over v0.9.1

- **Animation curves, schema v6** (`m3/animation-curves`, merged `e16a01a`).
  Per-node per-param curves with endpoint hold, Dispatcher `set_key` /
  `delete_key` / `clear_curve` ops with atomic undo/redo, bounded playback, and
  agent-CLI animation ops. Curves are an evaluation-time overlay; stored
  `node["params"]` are never mutated. See `docs/ANIMATION.md`.
- **Animation resolves before the tile executor reads params.** The rebase
  exposed a real defect: `tileexec` reads `node["params"]` at a dozen sites and
  had no knowledge of curves, so an animated parameter rendered correctly
  through the reference evaluator and **froze at its base value through tiles**
  — the viewer's default path since v0.9. `animation.resolve_document` bakes
  curves once at the `TileExecutor` API boundary. Confirmed by stubbing the fix
  out: 8 assertions fail, including identical tile output at frames 1 and 10 of
  an exposure ramp.
- **Viewer transparency shows pure black, not a checkerboard** (`47a7462`).
  The checker tinted every pixel it showed through, so partially transparent
  areas read brighter than the graph produced them. Frames are premultiplied,
  so black is a true no-op. Still available as
  `to_qimage(background="checker")`.

## Boundaries — still true, do not overstate

- Tiled/mip **source** I/O is unbuilt. Scanline-coded formats may still decode
  whole compressed rows.
- The tile path is slower than the full-frame evaluator on the **warm,
  unedited** case (tile reassembly/lookup overhead vs. one dict hit).
- Animated `Transform`/`Crop` animate correctly but take the full-frame
  fallback; those kinds are not in `SUPPORTED_TILED_KINDS`, so animation there
  gets no tile benefit.
- No editorial clip/track model, audio, roto/tracking, 3D, or model execution
  on `main`.
- No native display/GPU QA has been performed on the tile viewer or on animated
  playback; all desktop verification is offscreen.

## Open lanes

| branch | head | state |
| --- | --- | --- |
| `spike/roto-tracker` | `da01882` | WIP schema v7 shapes/tracks, ROI + proxy rules. **Now unblocked** — it assumed v6 had landed, and v6 is on `main`. Still a spike, not a merge candidate. |
| `arch/representation-core` | `7597f2a` | E2 visibility-reduction experiment. |

The schema-ordering hazard is cleared: `main` is 6, so the spike's v7 no longer
collides. Any new schema work starts from 7 and must not assume the spike's
shape.

## Color pipeline — audit 2026-09-10, DiMo named it a next priority

DiMo asked to "get the linear 32-bit working space and ACES Rec.709 viewing
space color pipeline right." Most of the plumbing is already there and correct;
the defects are at the display end. Audited at `a8e8ce7`:

**Already correct.** Working space is linear Rec.709 (`color.WORKING`) over the
built-in OCIO ACES config `ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5`. Every
buffer in the chain is float32 RGBA — `raster.py`, `tiles.py`, `tileexec.py`,
`cachetier.ARTIFACT_DTYPE`. `media.py` auto-detects `.exr` as linear Rec.709 and
premultiplied, everything else as sRGB/straight, and converts on ingest. The
ACES Rec.709 view exists: `'ACES 2.0'` → `ACES 2.0 - SDR 100 nits (Rec.709)` on
`sRGB - Display`.

**Defect 1 — the view transform is applied to premultiplied RGB.**
`imaging.to_qimage` passes `frame[..., :3]` straight to `color.display_rgb`
(imaging.py:102 and :110) without unpremultiplying. The sRGB OETF and the ACES
tone curve are both nonlinear, so every partially transparent pixel is wrong.
Measured at `a8e8ce7` on one pixel of linear 0.18 grey at alpha 0.5:

| view | viewer shows | correct over black |
| --- | --- | --- |
| sRGB | 85 | 59 |
| ACES 2.0 | 56 | 45 |

`write_png` (imaging.py:118-120) already does it the right way — unpremultiply,
encode, re-associate — so **the viewer and the PNG export disagree on any
semi-transparent pixel.** That divergence is the proof; it is not a judgement
call about preference.

**Defect 2 — the default view is sRGB.** `VIEWS` is ordered
`('sRGB', 'ACES 2.0', 'Linear')` and the combo takes index 0 (app.py:712-713),
so a fresh session is not viewing through ACES.

**Note, not a defect.** The `'Linear'` view returns working-space values with no
encoding before the 8-bit quantize in `to_qimage`. That is a legitimate debug
view; it should not be mistaken for a display transform.

**Cost note.** `display_rgb` runs on the full frame on the UI thread at every
display. It is now on the playback hot path that `a8e8ce7` just opened up.

## Next owner

Gonzo owns two lanes, ordering **pending DiMo's call**:

1. Color pipeline — fix defect 1 first (it is contained, measurable, and has a
   negative control ready: the export path). Expect display-golden churn in the
   test suite.
2. `spike/roto-tracker` assessment: rebase onto `main`, decide what is
   promotable, and expect the same class of gap animation just hit — the spike
   also predates the tile engine, so its ROI/proxy rules need checking against
   `tileexec` rather than only against the reference evaluator.

Omid owns real-hardware QA: the v0.10.0 AppImage/installer, interactive viewport
feel, and — now specifically — **confirming the `a8e8ce7` playback fix against
the actual EXR sequence that froze.** Offscreen tests show frames reaching the
viewer under a deliberately slow render; they cannot confirm what he saw.
