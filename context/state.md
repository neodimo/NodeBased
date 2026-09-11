# Current state — 2026-09-10 (end of day, re-verified directly)

## Released

**v0.9.1 is the latest published stable release:**
<https://github.com/neodimo/NodeBased/releases/tag/v0.9.1>, published
2026-09-10 22:52 UTC by the release workflow. Verified directly with
`gh release view v0.9.1`: assets are `NodeBased-0.9.1-linux-x86_64.AppImage`,
`NodeBased-0.9.1-windows-x64-portable.zip`,
`NodeBased-0.9.1-windows-x64-setup.exe`, and `SHA256SUMS`. All three tag
workflows (Desktop conformance on `main` and on `v0.9.1`, Build release
packages) completed **success**.

v0.9.0 was tagged first and its three workflows **failed** on a Windows
cache-root bug: disk-cache init assumed profile environment variables that a
sanitized Windows release-test environment does not provide. v0.9.1 is the
repair (fallback to `TEMP` or the process directory) plus a regression test in
`tests/test_cachetier.py`. v0.9.0 exists as a tag/release but is not the
recommended download.

## Repo facts, checked not inherited (2026-09-10 16:5x PDT)

- `main` = `6f44631 Release NodeBased v0.9.1`, clean tree, in sync with
  `origin/main`.
- `QT_QPA_PLATFORM=offscreen uv run python -m unittest discover -s tests`
  → **Ran 294 tests / OK** on that exact commit.
- `SCHEMA_VERSION = 5` on `main`.
- `feat/tile-artifact-engine` is merged (`17e2a1f`) and finished; the branch
  still exists at `26fc06e` and can be deleted.

## What v0.9.x actually delivers

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

## Boundaries — still true, do not overstate

- Tiled/mip **source** I/O is unbuilt. Scanline-coded formats may still decode
  whole compressed rows.
- The tile path is slower than the full-frame evaluator on the **warm,
  unedited** case (tile reassembly/lookup overhead vs. one dict hit).
- No animation, editorial clip/track model, audio, roto/tracking, 3D, or model
  execution on `main`.
- No native display/GPU QA has been performed on the tile viewer; all desktop
  verification is offscreen.

## Open lanes (all unmerged, none blocking a release)

| branch | head | state |
| --- | --- | --- |
| `m3/animation-curves` | `37d99ea` | schema v6 + animation; 12 ahead / 23 behind `main`. **Awaiting Gonzo review before merge.** |
| `spike/roto-tracker` | `da01882` | WIP schema v7 shapes/tracks, ROI + proxy rules. Spike, not a merge candidate. |
| `arch/representation-core` | `7597f2a` | E2 visibility-reduction experiment. |

**Schema ordering matters:** `main`=5, `m3`=6, `roto spike`=7. The roto spike
assumed v6 landed. Animation must merge and rebase to `main` first, or v7 will
collide.

## Next owner

Gonzo owns the `m3/animation-curves` review: rebase onto `main` (23 commits of
tile engine underneath it), confirm the v5→v6 upgrade path, re-run the full
suite, then merge. Omid owns real-hardware QA of the v0.9.1 AppImage/installer
and interactive viewport feel — that is the one thing no offscreen test covers.
