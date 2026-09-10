# NodeBased task log

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
