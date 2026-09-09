# Current state — 2026-09-09

## Released

**v0.3.0 is published and is the current latest**, public, stable (neither draft
nor prerelease): https://github.com/neodimo/NodeBased/releases/tag/v0.3.0
Built application revision: `8b2748b` (`target_commitish` on the release object,
confirmed via `gh release view`).

Adds, over 0.2.0: wire pick-up/rewire (click a wired input to unhook its edge into
a pending connection and drop it on a different input) and four Nuke-parity nodes
— ColorCorrect, Blur, Crop, Shuffle. Full detail in `docs/RELEASE_NOTES.md`.

- Windows portable ZIP: `NodeBased-0.3.0-windows-x64-portable.zip` (73,244,737 bytes).
- Windows per-user installer: `NodeBased-0.3.0-windows-x64-setup.exe` (49,974,784 bytes).
- Linux AppImage: `NodeBased-0.3.0-linux-x86_64.AppImage` (102,738,424 bytes).
- SHA256SUMS: all three package digests verified equal across three independent
  sources — the SHA256SUMS asset content, a fresh anonymous `sha256sum` of each
  downloaded file, and GitHub's own per-asset `digest` field from the API.

**v0.2.0 also exists** at https://github.com/neodimo/NodeBased/releases/tag/v0.2.0,
built from `999b02084df612592f26e949732d7352fb202d7a` (EXR/OCIO/charcoal theme,
no rewire/new nodes). It was published by an earlier, interrupted session before
this one resumed; its Windows portable ZIP shows 1 download. Left untouched rather
than edited/deleted without being asked — flagging for Omid's awareness only.

## Evidence

- 59 local tests passed (`QT_QPA_PLATFORM=offscreen uv run python -m unittest
  discover -s tests -v`), including new rewire/ColorCorrect/Blur/Crop/Shuffle
  coverage, both before and after the version bump commit.
- Verified the agent-protocol claim by actually running commands through
  `nodebased.agent` (the real JSON-lines dispatch path the socket bridge also
  uses), not by code inspection: `describe` returned ColorCorrect/Blur/Crop/
  Shuffle with correct params/limits/choices straight from SPECS/LIMITS/CHOICES;
  `connect` with `source: null` correctly cleared an existing input, and the
  freed source was then wired to a different node's input — the exact
  disconnect/rewire sequence the new UI feature depends on.
- Release workflow run (`workflow_dispatch`, `publish=true`) on commit `8b2748b`:
  https://github.com/neodimo/NodeBased/actions/runs/34331852654 — Linux package,
  Windows package, and publish jobs all succeeded.
- A first `publish=true` dispatch (run `34331664183`) was cancelled deliberately
  before the package jobs finished, because it targeted `v0.2.0`, which was
  already live from the older commit; publishing again under the same tag would
  have silently replaced assets on a release someone had already downloaded from.
  Fixed by bumping to `0.3.0` (`nodebased/__init__.py`, `pyproject.toml`) before
  re-dispatching.
- Public anonymous verification, no `gh` auth: `curl` to
  `api.github.com/repos/neodimo/NodeBased/releases/latest` returned `tag_name:
  v0.3.0`, `draft: false`, `prerelease: false`; all four asset URLs
  (`releases/download/v0.3.0/...`) returned HTTP 200 with byte counts matching
  the release metadata; `sha256sum -c SHA256SUMS` against the anonymously
  downloaded binaries reported `OK` for all three packages.

## Artifact disposition

All application/source/tests/package recipes/docs committed and pushed
(`8b2748b`, `HEAD` of `main`). No local scratch verification files were kept —
the anonymous download/hash-check directory (`/tmp/nb_release_verify`) was
deleted after the digest comparison completed. Canonical binaries are the public
v0.3.0 release assets and this run's CI artifacts.

## Remaining / next owner

Gonzo owns M1 production 2D work from `docs/VISION.md`: sequences/timeline,
ROI/tile scheduler, caching and measured native/GPU performance. The full DCC
remains partial: no 3D, procedural geometry or AI generation/loops yet.
Omid can download the v0.3.0 portable ZIP/installer/AppImage and evaluate the
new rewire UX and the four new nodes directly.
Native display/GPU performance and workplace application-policy acceptance remain
unverified; binaries are still unsigned. Real end-user cross-version update
(v0.2.0 → v0.3.0 via the in-app updater) is untested this pass — only the
publish/download/digest path was verified. Omid may also want a decision on
whether the orphaned v0.2.0 release should stay, get an editorial note pointing
at v0.3.0, or be removed; no action taken on it without being asked.
