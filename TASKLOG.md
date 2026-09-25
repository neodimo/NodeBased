## 2026-09-25 — Lane 2, plan "2D Viewer parity", step V1 of 3: viewer inputs 1-9, A/B, compare modes, wipe

- **What was done (evidence):** `core.py` gains `viewer_input` and `viewer_compare` and the optional
  `settings.viewer` keys `inputs`/`active`/`b`/`compare` (absent while default, so old documents are
  byte-identical); new `nodebased/compare.py` (display-only combine, align, wipe geometry); `app.py`
  gains the input strip, `Alt+1-9` for B, the wipe (centre and rotation handles, Ctrl-click and
  `Shift+W` reset), B rendering inside the same display request, and the readout naming the buffer.
  `tests/test_viewer_compare.py` (22 tests) plus 451 targeted tests including all of `test_desktop`
  pass on the branch.
- **Not done / limits:** wipe position and angle are not saved in the document; `Write` and export
  use A whatever the mode; a compare costs about twice a single view; no visual QA on a real display.
- **Artifacts:** code committed on `openclaw/nb-2d-parity`; docs in `docs/PLAYBACK.md` (section
  "Viewer inputs and the A/B compare"), `docs/AGENT_PROTOCOL.md`, `docs/PARITY_2D.md`, bundled copies identical.
- **Next:** V2 of 3, owner Lane 2 (brief `scratch/nb-lanes/auto/briefs/`).
- **Failure mode to avoid:** a frozen `FrameRequest` cannot carry B's frame; the worker hands it to
  `preview_ready` through `Window.compare_results` keyed by (generation, frame).

## 2026-09-25 — continuous mode merge: Lane 4 (rendering) step D of 4: GPU shadows on relit splats and shadow catching (integrator tick)

## 2026-09-25 — continuous mode merge: Lane 4 (rendering) step E of 4: particles on the GPU and in the 3D viewport, Spot lights in the viewport (integrator tick)

- **What was done:** `main` `25166e1` -> `f03b499` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `f03b499`: Ran 1988 tests in 1416.659 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

