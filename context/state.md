# Current state — 2026-09-09

## Released

**v0.1.0 is published**, public, stable (neither draft nor prerelease):
https://github.com/neodimo/NodeBased/releases/tag/v0.1.0

Built application revision: `cc3a46e` (see tag for full commit).

- Windows portable ZIP: `NodeBased-0.1.0-windows-x64-portable.zip` (61,648,268 bytes).
  Extract whole folder; run NodeBased.exe. No installation/admin request/registry
  install; cache and updater backups under its `portable-data` directory.
- Windows per-user installer: `NodeBased-0.1.0-windows-x64-setup.exe` (42,328,106 bytes).
- Linux AppImage: `NodeBased-0.1.0-linux-x86_64.AppImage` (90,323,448 bytes).
- SHA256SUMS: all package digests match GitHub's uploaded asset digests.

Manual Check → Download/progress → Restart updater follows GameStore's UX.
Unsaved-project prompt can cancel restart. ZIP uses an out-of-process helper;
installer is per-user; AppImage replaces original with a retained backup.
Bundled CA trust and package-level HTTPS probes support standalone networking.

## Evidence

- 31 local tests passed; Windows/Linux package jobs passed:
  https://github.com/neodimo/NodeBased/actions/runs/34325615828
- Actual Windows installation/reinstallation/portable launches and portable helper
  replacement + restart passed, preserving a sentinel project. Packaged HTTPS
  probes passed. Linux AppImage launch + HTTPS probe passed on Ubuntu 22.04.
- Downloaded exact AppImage also rendered the demo offscreen on the Bazzite host.
- Public anonymous latest-release API and all three download URLs verified.
  Updater asset selection works for installer/portable/AppImage; current 0.1.0
  correctly sees no newer version. Real cross-version end-user upgrade remains
  unverified until a subsequent release exists.
- Published from already-tested CI artifacts after final publish job failed.
  Cause: Python runner executes a temp script, not a repository module; importing
  nodebased in the uninstalled publishing job failed. Workflow now uses checkout-
  relative `runpy.run_path` for version lookup, locally checked. No binaries were
  rebuilt or changed during publication recovery; workflow fix is post-release.

## Artifact disposition

All application/source/tests/package recipes/docs committed and pushed.
Recovered release files and JSON/screenshot evidence are deliberate ignored local
copies under `artifacts/release-34325615828/` and
`artifacts/release-evidence-34325615828/`; canonical binaries are public release
assets and CI artifacts. No scratch cleanup handoff is pending.

## Remaining / next owner

Gonzo owns M1 production 2D work from `docs/VISION.md`: EXR/OCIO/channels/sequences,
ROI/tile scheduler, caching and measured native/GPU performance. The full DCC
remains partial: no 3D, procedural geometry or AI generation/loops yet.
Omid can download the portable ZIP and evaluate real desktop ergonomics.
Native display/GPU performance and workplace application-policy acceptance remain
unverified; early binaries are unsigned. No approval request is blocking work.
