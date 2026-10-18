## 2026-10-18 — Lane 2 VP1 of 2: opt-in retained-texture 2D viewer slice (PARTIAL: switch stays off by default; real-display gate owed)

- **What was done — evidence:** Added `nodebased/directviewer.py` (ledger rules, retained-texture presenter, `QRenderWidget(present_method="screen")` surface), a Window menu item "Direct picture (experimental GPU)" (also `NODEBASED_DIRECT_VIEWER=1`) opening a floating dock, `docs/DIRECT_VIEWER.md`, `tests/test_directviewer.py` (21 tests), `tools/viewer_direct_bench.py`, `tools/viewer_direct_functional.py`. Pixel tests render into an offscreen target and pass on RTX 3080 Ti, Radeon 8060S (RADV) and llvmpipe. The functional walk (changing checker, wheel/middle-drag, resize, dock/undock/refloat, hide/reopen, close and fresh widget) passed on NVIDIA and llvmpipe at pixel ratio 1.0 and 1.5 on a private Xvfb display. Targeted rerun of navigation, viewer-strip and 35 desktop tests: OK. Default Viewer path is unchanged.
- **Evidence limits:** Xvfb has no GPU presentation or DRI3: the AMD adapter cannot present to it (it is covered by the offscreen pixel tests only), NVIDIA presents through a software copy at ~16 per second, llvmpipe at the 120/s cap. Those cadence numbers describe Xvfb, not a screen, and the default Viewer's numbers there are Qt raster to an X backing store, so no faster/slower verdict exists. Present return is not screen response.
- **Inference:** Pan/zoom costs no conversion or upload (counters show 0 for both); a new picture costs one RGBA conversion and one texture write. Whether that beats the Viewer on a real screen is unmeasured.
- **Found:** `context.get_preferred_format(adapter)` answers for an adapter whose `configure` then aborts the process in native code ("Surface does not support the adapter's queue family", Radeon on Xvfb). Fixed by choosing the adapter through `request_adapter_sync(canvas=context)`. This was found by the run itself (commit 72a64ae1).
- **Process note:** one 400x300 probe window from `compat.py` was opened on display :0 by a loop mistake, for a second or two; every other run used the private Xvfb :61, which was killed at the end.
- **Artifacts (committed in this worktree, not pushed):** the three feature commits plus this note; `benchmarks/viewer/direct-vp1-*-2026-10-18.json`; run logs under `scratch/nb-lanes/run/` (tests-L2*.log, vp1/).
- **State:** PARTIAL. Core contracts hold in tests; the switch stays off. Left for Gonzo: real-screen cadence, vsync, input latency and the A/B against the Viewer on the live display; AMD on a real X display; docking and non-unit scale on that display; Windows; native Wayland; real device loss; Roto/Tracker/A-B/ROI hit order and overlays (the direct surface has none); full colour/alpha ownership (it shows the Viewer's display-transformed 8-bit picture).
- **Next owner + artifact:** Gonzo integrates and owns the VP2 parity/Windows gate, using `tools/viewer_direct_bench.py` and `tools/viewer_direct_functional.py` on the real display.
- **Failure mode:** none open in the default path; a start or draw failure hides the dock and leaves the Viewer in charge.

## 2026-10-18 — continuous mode merge: Lane 3 (Research, GPT-6 Astra) step V2 of 1: prove a direct presentation path for the viewer (integrator tick)

- **What was done:** `main` `dadbab0e` -> `2e64b221` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `2e64b221`: Ran 5489 tests in 20643.253 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

