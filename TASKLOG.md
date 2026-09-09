# NodeBased task log

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
