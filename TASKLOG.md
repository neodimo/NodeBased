## 2026-10-04 — 0.34.0 AppImage pin repair retagged; fresh CI running (Gonzo, 10:13 AM PDT)

- **What was done:** The prior `9872bbc` tag's Windows package job passed. Linux passed product tests and PyInstaller but the AppImage builder checksum guard stopped it. The reviewed upstream asset and independent local download agree on SHA-256 `95cbe7cce9717fce90c484e34052ee7c7f1d7635b33c12525b4776826a7d29b6`; release commit `b216429` updates that pin. Its exact-HEAD suite passed 4,286 tests (3 skipped). Force-with-lease moved annotated `v0.34.0` to `b216429`, and merge `14f2011` brought the repair into `main` and was pushed. Fresh tag package and conformance workflows started.
- **Artifacts:** Pushed tag `v0.34.0` and main merge `14f2011`; fresh package run `37219590616`, conformance run `37219590689`. Deliberate local scratch: `scratch/nb-lanes/run/release-v0.34.0-fix3`, `scratch/nb-lanes/run/release-v0.34.0-b216429.log`, `scratch/nb-lanes/run/appimagetool-20261004.AppImage`, and `scratch/nb-lanes/run/handoff-034-appimagetool-pin-1004.md`. The release tracker is `scratch/nb-lanes/auto/state.json`.
- **State:** Partial. Fresh Linux/Windows tag CI, installer assets, and publication are unverified. Fluids (lane 6) and 2D parity B (lane 8) GPT workers are held for today's spent Codex slice; no new worker was started.
- **Next owner + concrete artifact:** Gonzo's `nodebased-034-followup` wake checks runs `37219590616` and `37219590689`, then runs `python3 scratch/nb-lanes/auto/tick.py tick`; verify published release assets and their hashes before closing `status/nodebased.json` and disabling the wake.
- **Failure mode:** The upstream `/continuous/` AppImage builder asset changed under a stable URL. Its checksum guard correctly stopped packaging; a PyInstaller success does not prove the AppImage exists.

## 2026-10-04 — 0.34.0 AppImage builder repin and third retag (Gonzo, 10:12 AM PDT)

- **What was done:** The `9872bbc` tagged Linux package run passed tests and PyInstaller, then its checksum guard rejected a changed upstream `appimagetool` binary. The old Windows package job passed; Linux and Windows desktop conformance passed. GitHub's asset metadata and an independent local download agreed on the replacement binary's SHA-256 and 15,092,216-byte size. The isolated `b216429` pin repair passed its exact-HEAD suite: 4,286 tests, 3 skipped, exit 0. `v0.34.0` was moved to `b216429` with force-with-lease. Fresh tag package run `37219590616` and desktop run `37219590689` started. The tag commit was merged into `main` at `14f2011`, so the tag is now an ancestor of main.
- **Artifacts:** The tag, merge, and this note are committed/pushed. Deliberate local scratch: `scratch/nb-lanes/run/release-v0.34.0-fix3` at `b216429`, `scratch/nb-lanes/run/release-v0.34.0-b216429.log`, `scratch/nb-lanes/run/034-linux-9872bbc-job.log`, `scratch/nb-lanes/run/appimagetool-20261004.AppImage`, and `scratch/nb-lanes/run/handoff-034-appimagetool-pin-1004.md`.
- **State:** Partial. Fresh tagged Linux/Windows package and desktop workflows, actual installer assets, and publication remain unverified. No release is published yet.
- **Next owner + concrete artifact:** Gonzo checks tagged runs `37219590616` and `37219590689`, then runs `python3 scratch/nb-lanes/auto/tick.py tick` and verifies the published release's Linux AppImage and Windows installer/portable assets before announcing completion.
- **Failure mode:** The upstream `/continuous/` AppImage builder URL is mutable. The pinned checksum correctly stopped a changed binary; repinning requires verifying GitHub asset metadata against downloaded bytes. A passing PyInstaller stage does not prove an AppImage exists.

## 2026-10-04 — 0.34.0 second retag, CI pending (Gonzo, 8:15 AM PDT)

- **What was done:** Verified `9872bbc` exact-HEAD suite: 4,286 tests, 3 skipped, exit 0. The prior Windows package job passed. Retagged `v0.34.0` to `9872bbc` with force-with-lease, then cherry-picked the CI-only rate-limit fix onto main as `6aa70cd` after Lane 8 L2 integrated. Fresh tag workflows started.
- **Artifacts:** Tag `v0.34.0` pushed; main fix `6aa70cd` and this handoff note pushed to `main`. Deliberate local scratch: `scratch/nb-lanes/run/release-v0.34.0-fix2`, `scratch/nb-lanes/run/release-v0.34.0-9872bbc.log`, and `scratch/nb-lanes/run/handoff-034-probe-rate-limit-1004.md`.
- **State:** Partial. Fresh tag CI, installer assets, and publication unverified.
- **Next owner + concrete artifact:** Gonzo checks fresh tag runs from `gh run list`, inspects release assets, then runs `scratch/nb-lanes/auto/tick.py` to publish only after both platforms are green.
- **Failure mode:** Anonymous GitHub API quota broke the packaged-app HTTPS smoke; the ephemeral Actions token is scoped to that CI probe and is not bundled into the AppImage.

## 2026-10-04 — continuous mode merge: Lane 8 (2D parity B, GPT-6 Luna) step L2 of 2: inspect, stop and choose results from a controlled loop (integrator tick)

- **What was done:** `main` `7ccf754` -> `9bd2f95` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `9bd2f95`: Ran 4303 tests in 3813.978 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.
