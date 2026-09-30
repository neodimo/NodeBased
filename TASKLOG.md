## 2026-09-29 — Lane 8 G1: named layers on the tile path

- **What was done (evidence):** Tile cache artifacts now carry cropped named arrays. Multichannel Read and ReadBundle, plus Render3D source rasters, feed layer-aware Shuffle, ShuffleCopy, Remove, ZSlice, ZDefocus and ZMerge tile kernels. Read proxy decimation retains named arrays. Docs and bundled docs copies were revised; old tests that asserted layered Shuffle fallback now assert tile support.
- **Artifacts:** implementation/tests/docs in this worktree; intended for the lane commit. No scratch artifact retained.
- **State:** complete; targeted parity, tier, ROI, cache, bypass and registry/panel checks pass. Real display QA and Windows verification remain unverified.
- **Next owner + artifact:** Gonzo/integrator can pick up branch `openclaw/nb-2d-parity-b` and review the G1 commit; `tests/test_tile_layers.py` is the focused end-to-end proof.
- **Failure mode corrected:** the old docs and tile test treated all named layers as full-frame-only; they were stale once tile artifacts gained layer arrays.

## 2026-09-29 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5) step E1 of 2: TimeWarp, and TimeBlur and TimeEcho on the tile path with cached fractional samples, Lane 4 (Rendering, Claude Sonnet 5) step X2 of 2: denoiser controls, depth of field and motion blur in the viewport, instances drawn in the viewport (integrator tick)

- **What was done:** `main` `65dc1c1` -> `5f2fd1d` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `5f2fd1d`: Ran 3749 tests in 2401.279 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

