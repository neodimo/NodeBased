## 2026-10-18 — Lane 4 (Rendering, Claude Sonnet 5.5) H1 of 2: a denoised 1080p first look within one second (DONE; A6000, laptop and Windows unmeasured)

- **What was done (evidence).** Render mode (`progressiverender.step_interactive`, used by `viewport3d._paint_progressive`) now shows a
  denoised picture of the full output size first: stage one traces 480 x 270 at 4 samples, guides from one shared prepared scene
  (`gpupathtrace.reusing_scene`), threaded float32 `ptdenoise.denoise_strips`, bilinear enlarge, background; stage two 960 x 540 at
  16 samples; then the unchanged doubling steps. States carry `preview=True`, `denoised=True` and the traced size; the viewport note
  reads "denoised preview from 480x270". Stage one sizes itself from the measured cost per traced pixel (`first_stage_pixels`).
  `ptdenoise.denoise` is about twice as fast with identical output (verified bit for bit against the old code on a synthetic frame).
  Measured (`tools/benchmark_first_look.py`, JSON in `benchmarks/first_look/`, 1920 x 1080, call to 8-bit RGBA, warm = camera nudged
  per run): RTX 3080 Ti furnished room cold 0.61 s, warm median 0.47 s, worst 0.51 s; room plus 40,000 splats and smoke cold 0.97 s,
  warm 0.78 s, worst 0.83 s. Radeon 8060S furnished 0.64 / 0.50 s, mixed 0.70 / 0.75 s. llvmpipe furnished cold 1.16 s, warm 0.76 s.
  Before: `pathtrace.render(output="denoise")` at 1920 x 1080, 4 samples, 10.1 s (furnished) and 19.4 s (mixed) on the RTX 3080 Ti.
  Targeted tests: `tests/test_3d_first_look.py` (25 tests: delivery, size, nonempty, finite, cleaner than raw, ladder, reset,
  cancellation, budget, viewport, scene-reuse counts, final render equal with and without reuse) plus nine neighbouring modules,
  159 tests OK on the RTX 3080 Ti; the first-look and progressive modules also OK on the Radeon and llvmpipe through force-adapter.py.
  Commits on `openclaw/nb-3d-astra-lane`: `55821821`, `b5370652` and this note's commit; local, not pushed.
- **Inference / caveats.** The numbers were taken while the integrator's full suite held a shared GPU lock (never exclusive: the suite
  started a second 6 h run and starved the exclusive wait); one CPU core was busy and the card read idle in 58 of 80 samples of an
  earlier run, so contamination is judged small, not zero. The one-second target is met warm on all three adapters measured and cold
  on the two GPUs; llvmpipe's cold first call (1.16 s) misses it. The mixed scene's first look is 240 x 135 growing to about 273 x 154,
  not 480 x 270: its 350 ms scene build (redone on every camera move) is the floor. Image quality was judged by display-space error
  against a 256-sample reference at the same resolution (0.0035 against 0.0115 raw on the furnished room), not by eye on a real display.
- **Not done / unverified.** The 48 GB A6000 and the hybrid Iris Xe + RTX 4070 laptop were not reachable; Windows not run; no real-display
  QA of the label or the hand-over from the denoised preview to the raw 32-sample step (which can look noisier than the preview).
  The denoiser is still on the processor (0.27 s of the 0.47 s warm); stage two (1.25 s on the RTX) is slower than a first look.
- **Next owner.** Gonzo: merge the lane branch after the suite. Lane 4 H2 (continuous refinement): a GPU a-trous denoiser so every pass
  is denoised and stage two fits the second, then A6000 and laptop numbers with `tools/benchmark_first_look.py --adapter ...`.
  Artifact: `docs/3D_FOUNDATION.md`, "Lane 4 step notes", step H1.

## 2026-10-18 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice (integrator tick)

## 2026-10-19 — continuous mode merge: Lane 2 (2D parity, Claude Sonnet 5.5) VP1 of 2: opt-in retained-texture 2D viewer vertical slice (finish 1) (integrator tick)

- **What was done:** `main` `5ca57a81` -> `638121a5` plus this docs commit. Evidence and per-lane commit
  list in the dated `context/state.md` section. Full suite at `638121a5`: Ran 5522 tests in 20876.618 s, OK (skipped=1), exit 0.
- **Not done:** visual QA; CI not read; no Windows run.

