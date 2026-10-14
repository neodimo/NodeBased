## 2026-10-14 — Lane 8 Q1 of 1: the tiled Roto path is faster than the full-frame one (DONE)

- **What was done:** Profiled the prior tiled path at tiers 1 and 8 before changes. The tier-1 525 ms penalty came chiefly from 40 additional tile matte rasterizations; at tier 8 the extra 42 ms included digest-time full matte evaluation plus tile rendering. Broad Roto requests now use the reference path. Tile digests hash resolved shape data, and flattened shape geometry is reused across tiles. Quarter-frame and 512-pixel ROI timings are recorded in `docs/ROTO_TRACKING.md` and its bundled copy.
- **Artifacts:** `docs/ROTO_TRACKING.md`, `nodebased/data/docs/ROTO_TRACKING.md`, `nodebased/roto.py`, `nodebased/tileexec.py`, `tests/test_roto.py`; commits `368d337`, `a73a367`, `d584209`, local branch only, not pushed. Profiles and five-run samples are in deliberate scratch at `scratch/nb-lanes/run/profile-L8-baseline.txt` and `scratch/nb-lanes/run/bench-L8-Q1.json`.
- **State:** Done. Full-frame medians are within 1% of the evaluator at both tiers; tile/reference pixels match. 159 targeted tests passed. Windows and real-display review are unverified.
- **Next owner + artifact:** Gonzo reviews and integrates `openclaw/nb-2d-parity-b` after lane checks; use the committed Roto tracking doc and test logs listed above.
- **Failure mode:** The previous path rerasterized the matte once per tile and rendered it again to compute the tile digest; broad Roto requests paid both costs.

## 2026-10-14 — continuous mode merge: Lane 4 (Rendering, Claude Sonnet 5.5) step L3 of 3: the distorted plate round trip (finish 2: the card measurements) (finish 1), Lane 8 (2D parity B, GPT-6 Luna) step S3 of 3: Roto, RotoPaint and the trackers render only what the viewer asks for (finish 1) (integrator tick)

- **What was done:** `main` `37e60325` -> `0ecbfae1` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `0ecbfae1`: Ran 5337 tests in 19272.464 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

