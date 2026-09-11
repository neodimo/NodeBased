# Current state — 2026-09-10 (late, re-verified directly)

## Released

**v0.9.1 is the latest published stable release:**
<https://github.com/neodimo/NodeBased/releases/tag/v0.9.1>, published
2026-09-10 22:52 UTC by the release workflow. Verified with
`gh release view v0.9.1`: assets are `NodeBased-0.9.1-linux-x86_64.AppImage`,
`NodeBased-0.9.1-windows-x64-portable.zip`,
`NodeBased-0.9.1-windows-x64-setup.exe`, and `SHA256SUMS`. All three tag
workflows completed **success**.

v0.9.0 was tagged first and its three workflows **failed** on a Windows
cache-root bug: disk-cache init assumed profile environment variables that a
sanitized Windows release-test environment does not provide. v0.9.1 is the
repair (fallback to `TEMP` or the process directory) plus a regression test in
`tests/test_cachetier.py`. v0.9.0 exists as a tag/release but is not the
recommended download.

**`main` now carries unreleased work past v0.9.1** — animation and the viewer
background change below. The next release is not yet cut.

## Repo facts, checked not inherited

- `main` = `e16a01a Merge animation curves (schema v6) onto the tile engine`,
  clean tree.
- `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
  → **Ran 357 tests / OK** on that exact commit.
- `SCHEMA_VERSION = 6` on `main`.
- Merged and deletable: `feat/tile-artifact-engine`, `m3/animation-curves`.

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

## Landed on `main` since v0.9.1 (unreleased)

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

## Next owner

Gonzo owns the `spike/roto-tracker` assessment: rebase onto `main`, decide what
is promotable, and expect the same class of gap animation just hit — the spike
also predates the tile engine, so its ROI/proxy rules need checking against
`tileexec` rather than only against the reference evaluator.

Omid owns real-hardware QA of the v0.9.1 AppImage/installer and interactive
viewport feel — that is the one thing no offscreen test covers.
