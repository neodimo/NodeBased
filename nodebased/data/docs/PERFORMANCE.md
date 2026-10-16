# Performance

Frame-rate facts and the settings that decide them. Measure first; every brief carries a frame-rate budget.

## Adapter choice

NodeBased renders its 3D views on one graphics adapter, chosen once at startup and fixed until the next start (the
device state is global).

**How it chooses.** The app lists every adapter wgpu can see and ranks them: discrete GPU first, then integrated, then
software; a tie goes to the adapter with the larger buffer limits. The startup log prints the full list (name, type,
backend, driver) with the choice marked. The old request for "high performance" is only the fallback when the list is
empty. On a hybrid laptop (Intel Iris Xe plus an RTX 4070) that request came back with the Intel adapter, and the top
bar read "Intel · 339 ms" a frame; ranking by type takes the RTX.

**How the artist overrides it.**
- Settings → Graphics adapter: Automatic, or one adapter by name. Takes effect after restarting NodeBased (the Settings
  dialog says so, and the status bar repeats it after a change).
- `NODEBASED_ADAPTER=<part of the name>` (case-insensitive) beats the setting, for scripted runs and field reports.
  A name that matches nothing falls back to Automatic and logs a warning.
- The top bar's adapter readout shows the adapter in use.

**The slow-adapter warning.** When the app runs on an integrated or software adapter while a discrete one exists on the
machine, the readout turns amber and reads "<adapter>: a faster adapter is available, Settings → Graphics adapter". Its
tooltip adds the Windows route for drivers that hide the discrete card from applications: Settings → System → Display →
Graphics, add NodeBased, set it to High performance.

`tools/field_report.py` lists the adapters in its machine section and marks the one the app chose (and prints the warning
when it applies).

### Measured at 1920 x 1080

`python tools/benchmark_adapter_choice.py` runs each case once per adapter under the GPU lock. "Empty 3D scene" is the GPU
renderer drawing a scene with no objects; "Checker to Viewer" is a Checker graph played in the real window (offscreen
platform).

| Adapter | Type | Empty 3D scene | Checker to Viewer | 3000 lit cards, 4 samples |
| --- | --- | ---: | ---: | ---: |
| NVIDIA GeForce RTX 3080 Ti | discrete | 163 fps (6.0 ms) | 10.3 fps | not measured: the GPU lock stayed held by the test suite for 30 minutes |
| AMD Radeon 8060S Graphics | integrated | 165 fps (6.0 ms) | 10.8 fps | not measured, same reason |
| llvmpipe | software | 162 fps (6.1 ms) | 10.4 fps | 0.47 fps (2118 ms) |

Reading the numbers: the empty scene and the Checker playback land on the same figure on all three adapters. An empty
frame costs a fixed 6 ms of setup and readback, and the Checker graph is evaluated on the processor and drawn by the
window's own display context, which the "Graphics adapter" setting does not change. They show that the choice costs
nothing where the card is idle. The gain from the right card appears with geometry: the 3000-card scene takes 2.1 s a
frame on llvmpipe. Run the benchmark on the laptop to get the NVIDIA and Intel figures for that row.

Laptop (DiMo, 10/15): Windows, 2880 x 1800 at 175 percent scaling, Intel Iris Xe plus RTX 4070 mobile.

| Machine and adapter | Frame time |
| --- | ---: |
| Laptop, Intel Iris Xe (what the app chose before this change) | 339 ms (about 3 fps) |
| Laptop, RTX 4070 mobile (after this change) | to be filled in from the next field report |
