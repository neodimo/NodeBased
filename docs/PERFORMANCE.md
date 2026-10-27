# Performance

Frame-rate facts and the settings that decide them. Measure first; every brief carries a frame-rate budget.

## Adapter choice

NodeBased renders its 3D views on one graphics adapter, chosen once at startup and fixed until the next start (the
device state is global).

**How it chooses.** The app lists every adapter wgpu can see and uses a startup heuristic: discrete GPU first, then
integrated, then software; a tie goes to the adapter with the larger buffer limits. This is not a speed measurement:
the fastest card depends on the workload and connection. The startup log prints the full list (name, type,
backend, driver) with the choice marked. The old request for "high performance" is only the fallback when the list is
empty. On a hybrid laptop (Intel Iris Xe plus an RTX 4070) that request came back with the Intel adapter, and the top
bar read "Intel · 339 ms" a frame; ranking by type takes the RTX.

**How the artist overrides it.**
- Settings → Graphics adapter: Automatic, or one adapter by name. Takes effect after restarting NodeBased (the Settings
  dialog says so, and the status bar repeats it after a change).
- `NODEBASED_ADAPTER=<part of the name>` (case-insensitive) beats the setting, for scripted runs and field reports.
  A name that matches nothing falls back to Automatic and logs a warning.
- The top bar's adapter readout shows the adapter in use.

**The adapter suggestion.** When the app runs on an integrated or software adapter while a discrete one exists on the
machine, the readout turns amber and reads "<adapter>: a discrete adapter is available, Settings → Graphics adapter". Its
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
| NVIDIA GeForce RTX 3080 Ti | discrete | 163 fps (6.0 ms) | 10.3 fps | 0.36 fps (2798 ms median) |
| AMD Radeon 8060S Graphics | integrated | 165 fps (6.0 ms) | 10.8 fps | 0.65 fps (1543 ms median) |
| llvmpipe | software | 162 fps (6.1 ms) | 10.4 fps | 0.47 fps (2118 ms) |

Reading the numbers: the empty scene and the Checker playback land on the same figure on all three adapters. An empty
frame costs a fixed 6 ms of setup and readback, and the Checker graph is evaluated on the processor and drawn by the
window's own display context, which the "Graphics adapter" setting does not change. They show that the choice costs
nothing where the card is idle. Differences appear with geometry: the 3000-card scene takes 2.1 s a
frame on llvmpipe. In two runs of the heavy card scene, this workstation's Radeon was about 1.8× faster than its RTX
3080 Ti eGPU. This is one workload, not a general card ranking; Automatic still prefers discrete GPUs, so an artist on
this workstation can select Radeon in Settings for this scene. Run the benchmark on the laptop to get the NVIDIA and
Intel figures for that machine before claiming a laptop gain.

Laptop (DiMo, 10/15): Windows, 2880 x 1800 at 175 percent scaling, Intel Iris Xe plus RTX 4070 mobile.

| Machine and adapter | Frame time |
| --- | ---: |
| Laptop, Intel Iris Xe (what the app chose before this change) | 339 ms (about 3 fps) |
| Laptop, RTX 4070 mobile (after this change) | to be filled in from the next field report |

## Playback

Targets:

- A comp whose picture does not change (an empty 3D scene, a constant Checker, a Read of one still)
  plays at the comp's rate: the cost per tick is a paint of about 1 ms.
- Animated content at 1080p spends under 16 ms per frame on the display path, excluding the render
  itself. Met in the opt-in Direct picture (about 2.7 ms, see "Direct viewer display transform"); not
  met yet in the default Viewer, where `to_qimage` is still about 41 ms per frame (see "Still slow").
- The time row shows the frames per second playback achieves, and turns warm below the comp's rate.
- The realistic moving 1080p comp targets 24 fps uncached on the RTX 3080 Ti; 16 fps is the
  intermediate gate. The latest accepted-picture first pass is 5.451 fps, so both gates remain open. The
  laptop stand-in targets at least 12 fps; its latest result is 3.473 fps. At 4K, the target is
  24 fps on an automatic reduced tier and a full-resolution still within one second after stop;
  the RTX result is pending.
- A 3D comp under an animated camera plays at the comp's rate on the RTX 3080 Ti at full resolution (1920 x 1080): the cube, the
  furnished room and, since 10/23, the mixed mesh and splat scene, which refines to its full-quality picture within 2 s of
  stopping (see "Lane 4 step notes", 3D playback at full size and the mixed scene at full size).

What changed, with the measurement from before and after:

| Case (1080p, 24 fps comp) | Before | After |
|---|---:|---:|
| Checker, constant | 9.8 fps | 24.0 fps (timer-bound) |
| Empty 3D scene, static camera | 11.6 fps | 24.0 fps (timer-bound) |
| Checker with an animated size | 9.7 fps | 11.0 fps |
| Empty 3D scene, animated camera (GPU) | 3.5 fps | 11.9 fps |
| Cube, animated camera (GPU) | 1.4 fps | 2.0 fps |
| Render3D of an empty scene, one frame (CPU or GPU) | 190 to 198 ms | 8 ms |
| Render3D of a cube, one frame, CPU / GPU | 773 / 569 ms | 618 / 411 ms |
| Process CPU while a checker plays | 177 percent | 20 percent |
| Graph repaints at idle | 743 per second, 21 percent of a core | none, 0 percent |

### Static frames convert once

`playback.is_time_dependent(document, target)` walks the target and everything wired into it. The
picture can change with the frame only when an upstream knob has an animation curve of more than
one key, any expression drives a knob, or a node kind is not on the `STATIC_KINDS` list (Read
sequences, simulation caches, retimes, particles and so on all stay time-dependent). The list is an
allow-list on purpose: a kind that is missing costs speed, a kind listed by mistake freezes
playback on one picture. Add a kind only after reading that it never looks at the frame number.

For a static target `DisplayCache.key` uses one shared key (`STATIC_FRAME`) instead of the frame
number, the frame is stored at full precision, and the last converted picture is kept (keyed on
the document, region and display setup), so showing it again costs no compose and no conversion.
The timeline band reports every frame of a static comp as cached.

### Frames reach the display without a half-float round trip

The worker used to round every composed frame to half-float for the display cache (`put`, about
15 ms) and copy it back to float32 when the frame was shown (`get`, about 8 ms). Now `put(lazy=True)`
keeps the producer's float32 array, `get(copy=False)` hands it back, and `compact()` rounds the
entries to half-float on the worker when nothing else is waiting (one frame at a time; a new
request stops it) or when the memory budget needs the room. Callers of `get(copy=False)` must not
write into the array.

### Supersample resolve

`scene3d.box_resolve` adds the strided views and divides once, as `imaging._decimate` does, in
place of `reshape(...).mean(axis=(1, 3))` (148 ms of an empty 1080p frame, the same finding as
`BENCHMARKS-v0.17-playback.md`). A scene with no geometry, splats, particles, volumes, instances,
environments or camera-visible lights renders at one sample, since there is nothing to antialias.

### Idle repaint loop

The graph's minimap is a translucent child of the graph viewport. It used to be repainted from the
graph's `paintEvent`, which its own repaint triggered again: about 700 paints a second at idle and
3,000 or more during playback. It now repaints on scroll, zoom, resize, selection, node moves and
rebuilds (`GraphCorner.refresh_minimap`), never from a paint event.

### Honest readout

The real time light measures the frames put on screen over the last second (`Window.achieved_fps`).
It reads "real time · 24 fps" while the rate holds and "11 of 24 fps" in the warm colour once the
rate falls below 95 percent of the comp's. It makes no claim in the first half second of playback.

### Direct viewer display transform

With the Direct picture dock up (Window menu, or `NODEBASED_DIRECT_VIEWER=1`), a playback frame no
longer goes through `to_qimage`. The worker hands the window the scene-linear float frame and its
display settings (`carddisplay.LazyPicture`); the dock uploads it once into a float texture and its
fragment shader does gain, channel select, unpremultiply, gamma, OCIO's own view-transform shader,
the checker, the clipping warning and the 8-bit quantise. The card agrees with `to_qimage` within
1/255 on every channel (tests/test_carddisplay.py) on the RTX 3080 Ti, the Radeon 8060S and
llvmpipe. The Viewer holds its last picture while the dock plays and repaints when playback stops;
the 8-bit picture is built only when something asks for it. Details in docs/DIRECT_VIEWER.md.

Measured 2026-10-18 on a private Xvfb display (xcb, no DRI3), 24 fps comps at 1080p unless the row
says 4K, 10 s of the app's own playback, NVIDIA RTX 3080 Ti presenting, with the full test suite running beside it (so
the absolute rates are lower than the table above). Summary in
`benchmarks/viewer/direct-display-transform-xvfb-2026-10-18.json`:

| Case | Before (Viewer) | After (Direct picture) |
|---|---:|---:|
| Display path per frame on the UI thread, animated checker | 45.2 ms (`to_qimage`) | 2.7 ms (hand-over 0.2, upload 1.9, draw 0.6) |
| Animated checker | 11.6 fps | 19.0 fps |
| Animated empty 3D scene (GPU) | 8.3 fps | 15.6 fps |
| 4K animated checker, proxy 1/2 while playing | 10.3 fps (`to_qimage` 47.6 ms) | 19.3 fps (display path 2.8 ms) |
| 4K animated checker, full resolution | 2.9 fps (`to_qimage` 190 ms) | 3.7 fps (bound by the 128 MB float upload) |

On the card, the display shader costs 0.4 ms per 960 x 540 draw on the RTX 3080 Ti, 0.3 ms on the
Radeon 8060S and 2.4 ms on llvmpipe. OCIO's lookup tables sit in a storage buffer: inlined as
constant arrays they made the ACES 2.0 draw 17 ms, because naga copies a constant array into
function memory for every dynamic index. The float frame's copy to the card is asynchronous but
not free: 12 ms at 1080p and 48 ms at 4K on the RTX 3080 Ti (about 2.7 GB/s over PCIe with the suite
on the card; a persistent mapped staging ring measured the same), 1.9 ms on the Radeon's shared
memory.

What still holds the animated checker below 24 fps:

- Presenting to Xvfb costs about 21 ms per frame on the UI thread, with vsync on or off (VP1 saw
  44 to 52 ms at full size): with no DRI3 the swapchain image is copied out by the processor. That
  is a property of this display server; presentation on a real screen is not measured yet.
- On the worker, `DisplayCache.put` compacts the oldest waiting frame to half-float under budget
  pressure on every put (37 percent of worker samples), on top of the 21 ms compose. That is profile
  item 2, untouched here.
- At 4K full resolution the frame is 128 MB of float32 per upload; keeping the render on the card
  (the later L4 step: same device as Render3D) removes it.

### Direct picture playback (VP2a, 10/19)

The Direct picture's uploads no longer run on the Qt thread: a filler thread copies each picture into one
of three reused staging buffers and the draw only records the GPU copy; 4K float frames go up as half
floats. At 4K the dock's largest draw on the RTX 3080 Ti went from 262 ms to under 6 ms, and a 4K picture
costs the Qt thread about 1 ms (VP1: 25 ms of `write_texture`). Graph press, drag and timeline scrub stay
under 31 ms at the 95th percentile while the dock plays, at 1080p and 4K, on the RTX and llvmpipe (private
Xvfb display). Numbers per adapter, the frame-order check and what was not measured are in
docs/DIRECT_VIEWER.md, "Playback (VP2a)".

### Lane 8: realistic comp playback evaluation (10/22)

The uncached fixture in `tests/fixtures/playback_comp.nbcomp` measures evaluation through the
finished float frame handed to the display. Timings below use 2 ms worker-stack samples, so node
costs are estimates; the paired JSON profiles preserve the timeline and sampling evidence.

| Comp | Before: top five nodes (ms/frame) | After: top five nodes (ms/frame) |
|---|---|---|
| 1080p / RTX 3080 Ti | Transform 183.6; Blur 91.2; ColorCorrect 76.4; Read plate 31.5; Read background 29.8 | Transform 67.7; Grade 27.6; Merge 18.8; Blur 18.8; Roto 11.0 |
| 4K / llvmpipe | Transform 667.2; ColorCorrect 324.0; Blur 311.0; Read background 118.8; Read plate 117.2 | Transform 499.3; ColorCorrect 286.7; Merge 131.7; Blur 122.3; Grade 119.7 |

| Prior uncached measurements (historical; original runs included post-stop drain frames) | Rate |
|---|---:|
| 1080p / RTX 3080 Ti, corrected exact-window run | 3.7 fps (37 frames / 10 s) |
| 1080p / RTX 3080 Ti, earlier baseline → finish 1 | 1.5 → 4.5 fps (not gate-valid) |
| 1080p / llvmpipe | 1.6 → 2.4 fps (historical) |
| 1080p / integrated laptop stand-in (four CPUs, 2× Qt scale) | 2.3 fps (historical) |
| 4K / llvmpipe | 0.4 → 0.5 fps (historical) |

Finish-2 follow-up (10-second uncached runs): the final RTX 3080 Ti run displayed 48 unique fresh
frames at 5.451 fps; its overall 7.3 fps includes later cache repeats. The fresh llvmpipe and
four-CPU / 2×-scale laptop runs reached 5.215 and 3.422 fps. The RTX result remains below the
16 fps intermediate gate; the laptop result remains below 12 fps. The 4K RTX run enabled automatic
proxying, showed tiers 1, 2, 4 and 8 in the time row, and reached 2.895 fps; its full-resolution
still arrived 1,032 ms after stop. The 4K llvmpipe run reached 2.512 fps and returned the full
still in 1,417 ms. Both remain below the 24 fps / one-second RTX gates. JSON evidence is in the
matching `L8-finish-*1022.json` files under the lane run directory. The benchmark now counts a
picture only after the Viewer accepts that request generation; this corrected the earlier
7.277-fps first-pass estimate, which counted completed handoffs without checking presentation.

| Finish-2 follow-up (uncached first pass) | Rate | Other gate evidence |
|---|---:|---|
| 1080p / RTX 3080 Ti | 5.451 fps | 16 fps intermediate: open |
| 1080p / llvmpipe | 5.215 fps | Diagnostic only |
| 1080p / laptop stand-in (four CPUs, 2× scale) | 3.422 fps | 12 fps target: open |
| 4K / RTX 3080 Ti with automatic proxy | 2.895 fps | Tiers 1, 2, 4 and 8; full still after 1,032 ms |
| 4K / llvmpipe with automatic proxy | 2.512 fps | Tiers 1, 2, 4 and 8; full still after 1,417 ms |

For the unreachable 48 GB A6000 workstation, run `python tools/benchmark_2d_playback.py --size
1080p --field` with generated plate/background sequences and repeat at `--size 4k`. The compact
field output reports `adapter`, `resolution`, `playback_proxy_enabled`, `playback_tiers`,
`achieved_fps`, `first_pass_fps`, `post_stop_full_resolution_ms` and `status_line`. The 1080p run
should show full-tier fresh playback against 24 fps; the 4K run should show a reduced tier in the
time row, comp-rate playback, and a full-resolution still within 1,000 ms. No A6000 measurement is
claimed here.

Cached playback stayed near its prior cadence: 1080p llvmpipe was 12.3 fps before and 11.7–11.8
fps after, with about 84 ms median frame intervals in both runs. 4K llvmpipe reached 4.7 fps on a
cached 16-frame loop (frames 33–48), the window retained by the default cache budget at UHD.
The earlier animated-size Checker Viewer check reached 23.79 fps on the RTX 3080 Ti, below its
24 fps gate. The 24 fps transport was rounding each repeating timer interval to 42 ms (23.81 Hz).
Playback now uses phase-locked precise one-shot intervals. The corrected fixed-window check counted
720 displayed pictures in 30 seconds from the first live picture (24.00 fps); timer and display
interval medians were 41.659 ms and 41.715 ms. Its evidence is
`benchmarks/2d-playback/checker-after-timer-1022.json`. The earlier baseline is
`benchmarks/2d-playback/finish1-checker-1022-animated-rtx.json`. The measurement uses a precise
30-second cutoff anchored at the first live picture; the prior coarse sample timer could stop
early and the boundary picture was excluded.

The finish-1 4.5 fps result was overstated: the original benchmark kept counting frames delivered during worker drain after playback stopped. `tools/benchmark_2d_playback.py` now anchors a precise single-shot sample window, counts only displayed frames inside it, and samples the worker only within that interval. Its corrected 10-second RTX run displayed 37 frames (3.7 fps), with a 286.765 ms median frame interval and 298.573 ms p95. The 16 fps gate remains unmet. The corrected profile's estimated node costs average Transform 67.7 ms, Grade 27.6 ms, Merge 18.8 ms, Blur 18.8 ms and Roto 11.0 ms per evaluated frame; Transform's fractional animated offset still takes the full-frame resampling path. Evidence is `benchmarks/2d-playback/finish2-corrected-after-1022-1080p-rtx3080ti.json`. Earlier 4.5 fps profiles are retained as historical artifacts, not valid gate measurements.

The measured full-resolution changes cut small-radius Blur with shifted-window accumulation,
use separable cubic sampling for translation-only Transform, and bound Read-ahead to one upcoming
full-resolution frame. Read-ahead is skipped when the future display frame is already cached.
Playback also skips synchronous disk spills for evicted transient evaluation frames and restores
the normal disk-cache policy when stopped. This finish-1 run showed steady 288 ms median frame
intervals, with no disk-tier write samples, and 4.5 fps over ten seconds. Animated Transform remains
a full-frame fallback because its canvas-global coordinates do not yet have a tile kernel. The 16 fps
uncached comp gate remains open; the animated Checker gate now passes on the RTX. Finish-1 profiles are
`benchmarks/2d-playback/finish1-after-1022-1080p-rtx3080ti.{json,txt}`; the earlier profile files under
`benchmarks/2d-playback/` contain the per-stage timeline and Qt/worker sampling evidence; read
decode events include future frames warmed by the reader pool.

The earlier 288.432 ms profile reconciles approximately as follows; it is retained as a diagnostic profile, while the corrected finish-2 run is the gate measurement. The node estimate is the
mean of the 2 ms worker-stack samples; the two wall-clock values are medians from the same run, so
the residual is explicit and the components are approximate.

| Per displayed frame | Time | Basis |
|---|---:|---|
| Node evaluation | 109.69 ms | Sum of per-node worker-stack estimates |
| Other worker-side work | 107.37 ms | Residual within the 260.587 ms worker-to-UI interval, after node estimate and display conversion |
| Float-to-display conversion | 43.53 ms | Median `to_qimage` call |
| UI/event-loop interval outside the worker | 27.85 ms | Difference between frame interval and worker-to-UI interval |
| **Total** | **288.44 ms** | Reconciles to the measured 288.432 ms median |

The worker-side residual includes input decode/waiting, format and buffer work, and evaluator
orchestration. The profile recorded 90 plate decode calls averaging 44.96 ms each over 44 evaluated
frames (about two reads per frame); these calls can overlap with read-ahead, so that service time is
an attribution inside the residual rather than extra wall time. Cache insertion itself was 0.013 ms
median, with no disk-tier writes. The current capture cannot separate buffer-copy cost from time
waiting on decoded reads or Python scheduling/GIL contention; those remain unverified contributors.
The reconciled artifact is `benchmarks/2d-playback/finish1-critical-path-1022.json`.

For an exact integer-pixel translation, Transform now copies the overlapping source window
instead of running the cubic sampler. At 1080p on CPU NumPy, the isolated translation path fell
from 78.06 ms to 1.91 ms (nine-run medians) with pixel agreement within 1e-7 across nearest,
bilinear and cubic filters; see `benchmarks/2d-playback/transform-integer-offset-1022.json`.
The realistic comp animates by fractional pixels, so this optimization does not improve the
corrected 3.7 fps result recorded before the fractional-translation follow-up below.

The follow-up replaces four full-frame gathers for fractional horizontal Transform with
contiguous tap slices. Its isolated 1080p CPU median fell from 51.4 ms to 20.6 ms (2.5×),
with exact pixel agreement. On the RTX 3080 Ti, the 10-second real-window run displayed 118
pictures (11.8 fps); the first pass through 48 unique, unevaluated display frames was 7.06 fps.
Later repeats hit the display cache and are reported separately so they cannot hide the cold-pass
rate. Cached playback reached 18.4 fps. The 16 fps fresh-frame gate remains open. The new profile
estimates Transform at 21.3 ms per frame, followed by Grade 10.9 ms, Merge 8.2 ms, Blur 7.9 ms
and Roto 7.8 ms. Display conversion remained 40.7 ms median; plate reads were 39.3 ms median per
read and may overlap through read-ahead. See
`benchmarks/2d-playback/finish2-transform-horizontal-1022.json` and
`benchmarks/2d-playback/finish2-cached-transform-horizontal-1022.json`.

### Still slow

- In the default Viewer, `to_qimage` is about 41 ms per 1080p frame and 170 ms at 4K (ten
  full-frame float32 temporaries and a float readback from the integrated GPU). The Direct picture
  does it on the card during playback (above); the Viewer itself, stills and compare pictures
  still pay it.
- The real-window profile showed one `work` worker job for each displayed frame; startup and
  shutdown jobs are separate. The earlier claim of two worker jobs per frame does not match the
  observed playback path.
- The CPU rasteriser takes 620 ms for a cube; the GPU path waits on the card.

## Graph input latency

A click on a node paints its selection highlight before anything else happens, and letting go of a dragged node does not rebuild the Properties panel.

What changed:

- **Highlight first.** The graph's selection change only schedules the Properties panel and the keyed-frame band on a zero-delay timer. The press handler returns in about a millisecond and the highlight paints in the same frame.
- **No rebuild for a move.** A command that only moves nodes (a drag release, an agent `move`, a batch of moves) leaves the Properties panel alone. When the dragged items already sit where the document now puts them, the graph is not rebuilt either.
- **Panel reuse.** The last 16 panels are kept per node and shown again when nothing in the document, the playhead or the theme has changed since. An explicit `inspect` always builds a fresh panel.
- **Lazy material preview.** The material sphere on geometry nodes renders after the panel is on screen.
- **Narrower window filter.** The window's event filter is installed only on focusable panel widgets, the only ones that receive the focus events it reads.

Method: the real application on a private X display (Xvfb, 5120x2880), real `xdotool` presses and drags, 2400x1300 window, minimap on. "Highlight painted" is the end of the graph's own paint after the press event arrives. "Release" is the time `commit_moves` takes after the mouse is let go. 12 presses on distinct visible nodes (then the same 12 again for the second selection), 6 drags of 60 moves of 2 px. Medians, with p95 in brackets, in milliseconds.

| | 18-node demo, before | 18-node demo, after | 205-node synthetic, before | 205-node synthetic, after |
|---|---:|---:|---:|---:|
| Press handler | 117 (292) | 0.4 (0.6) | 43 (50) | 1.0 (1.1) |
| Press to highlight painted | 130 (307) | 11.9 (14.8) | 72 (76) | 23.9 (26.4) |
| Same, selecting the node a second time | 141 (333) | 11.5 (12.7) | 65 (77) | 24.0 (25.4) |
| Panel build after the press (first selection) | in the press | 129 (207) | in the press | 36 (44) |
| Panel swap, second selection of the same node | in the press | 46 (80) | in the press | 15 (16) |
| Release (`commit_moves`) | 154 (327) | 2.8 (3.5) | 80 (94) | 4.0 (4.2) |

The remaining press-to-highlight time on the large graph is the graph's own repaint (about 20 ms for 44 visible nodes at this pane size), which the press no longer adds to.

## Graph at display rate

Dragging a node, panning and clicking in a graph of hundreds of nodes now costs about the same as in a small one.

What changed:

- **Local wire routing.** A moved node re-routes only its own wires. A wire keeps its path until one of its ends, their facing or the wire mode changes, so the rest of the graph is not marked for repainting on every mouse move.
- **Cached node bodies.** A node's body (fill, inner glow, outline and icon) is drawn once per look and screen scale into a picture at full device resolution and shared by every node with the same family, shape and size. It is redrawn when the selection, the level of detail or the zoom changes. Names and details stay live text, so they are sharp at 175 percent.
- **Cached wires.** A placed wire is kept as a picture on screen and redrawn when it is re-routed, restyled or the zoom changes. A click or a pan no longer redraws every wire.
- **Plain wires while moving.** During a node drag, a pan or a wheel zoom the wires are drawn without smoothing; they are redrawn smooth when it ends.
- **Level of detail.** Below 40 percent zoom a node is a flat rounded card without name, details, icon or socket labels, and wires drop their glow and arrowhead.
- **OpenGL graph, off.** Settings, "Node graph drawing" (or `NODEBASED_GRAPH_OPENGL=1`) draws the graph through OpenGL. It stays off: see below.

Method: this workstation, the real application on a private X display (Xvfb, so the software path), real `xdotool` input, 2400x1300 window, graph pane 1875x396 with 44 nodes in view at 100 percent zoom. Synthetic graphs of 205 and 505 items (a Checker and Grade, Blur, ColorCorrect and Invert nodes, 20 to a row); the 18-node demo shot laid out in a grid. Drags are 80 moves of 2 px about 8 ms apart, so about 120 moves a second is the most the input can ask for. "Pan zoomed out" is a pan at 30 percent zoom, where most of the graph is in view. Medians in milliseconds unless stated. These numbers are from this workstation; the hybrid laptop has not been measured.

| | demo, before | demo, after | 205 nodes, before | 205 nodes, after | 500 nodes, before | 500 nodes, after |
|---|---:|---:|---:|---:|---:|---:|
| Press to highlight painted | 15.4 | 4.4 | 25.4 | 8.3 | 26.6 | 9.4 |
| Drag, repaints per second | 82 | 120 | 31 | 119 | 19 | 118 |
| Drag, mouse move to painted | 10.0 | 2.6 | 30.8 | 2.4 | 48.5 | 2.4 |
| Drag, graph paint | 8.1 | 1.2 | 20.5 | 1.0 | 23.4 | 1.0 |
| Drag, move handler | 0.44 | 0.14 | 8.4 | 0.14 | 22.5 | 0.15 |
| Pan at 100 percent, graph paint | 3.4 | 1.0 | 10.2 | 2.7 | 10.1 | 2.9 |
| Pan zoomed out, graph paint | 1.8 | 1.4 | 14.6 | 11.4 | 37.4 | 27.2 |
| Wheel zoom, graph paint | 3.7 | 3.9 | 4.5 | 5.7 | 4.9 | 5.7 |
| Idle, process CPU | 0.4 % | 0.3 % | 0.3 % | 0.3 % | 0.4 % | 0.4 % |

At 175 percent (`QT_SCALE_FACTOR=1.75`, 1400x780 window): 205 nodes drag at 41 repaints a second before and 120 after (paint 11.4 to 1.6 ms), 500 nodes at 24 before and 104 after; press to highlight 14 to 23 ms before, 10 to 17 after (the first press of a session also draws the selected look once).

**OpenGL graph.** With a `QOpenGLWidget` as the graph's viewport, the only OpenGL this private display offers is the software renderer (llvmpipe), and there it is slower everywhere: 205 nodes drag at 52 repaints a second (raster 119), pan paint 9.8 ms (2.7), wheel zoom 11 ms (5.7), idle CPU 4.7 percent (0.3). Three graph tests also fail with it on: two because a repaint request on the OpenGL surface does not paint at once, and one about a drag release whose cause is not yet known. The NVIDIA and Radeon cards could not be measured: their OpenGL needs the workstation's real display. Turning it on also changed how wgpu starts: wgpu's OpenGL adapter probe aborts (EGL BadAccess) once a Qt OpenGL context exists on the same thread, so the graph lists the wgpu adapters before it creates its surface and releases its context after every paint. The setting stays off until it has been measured on the cards and the laptop.

Still slow: a pan with most of a 500-node graph in view (27 ms a paint: mostly Qt drawing a few hundred items and sockets, plus about 3 ms for the minimap, which redraws every node on each scroll), and wheel zoom, where every zoom step redraws the cached pictures at the new scale.

## Lane 8 step notes

### R6-1: the realistic comp with Grade, Transform, Blur and Merge resident on the GPU (10/25, step A of 3)

Research spike R6-1 (`briefs/proposed/R6-1-resident-four-node-spike.md`, plan in [GPU 2D plan](GPU_2D_PLAN.md)). Step A
delivers the baseline and harness, the opt-in island and a first pixel check. The full pixel gate set, the
`tools/benchmark_gpu2d.py` modes and the go/no-go are steps B and C; nothing here is a gate pass.

**What exists.** `nodebased/gpu2d.py` runs Grade, Transform, Blur and Merge as float32 compute passes on one wgpu device.
`Evaluator.gpu2d` is `None` by default (every node on the CPU kernels, nothing changes); setting it to a `GPU2D` makes a
Grade, Transform or Blur whose readers are all island nodes a deferred, pixel-less node, and the island's exit node (the
Merge in the realistic comp) runs the whole chain on the device and reads back once for the CPU ColorCorrect. Each node has
one small parameter-mapping function (`map_grade`, `map_blur`, `map_transform`, `map_merge`), so a new Nuke knob is a change to
that mapping plus a CPU fallback. Anything the device cannot run (Blur radius above 4, Transform motion blur, a Merge
operation without a kernel, an unwired Merge input, mismatched display windows, a window over the byte budget, non-finite
pixels, a lost device) is a counted, reported CPU boundary that evaluates the node on the existing CPU kernels; every
case where the CPU raises an error is left to the CPU so the error stays the CPU's. `tools/benchmark_2d_playback.py --backend gpu2d`
selects it (`cpu` is the default); `--frame-range N` extends the fixture to N frames, `--warmup-frames` runs sacrificial frames
before a GPU run, and each result file carries an `environment` record (revision, fixture and variant hashes, hardware, drivers,
command).

**Counters** (`GPU2D.counters`, per island in the benchmark result): uploads and bytes, downloads and bytes, intermediate
readbacks (must be 0), submissions, dispatches, buffer allocations, pool hits, live and peak bytes, cold shader setup, and the
time spent uploading, encoding, waiting for the passes (`gpu_ms`) and reading back. `gpu_ms` is submit until a 16-byte flags
readback returns, which happens after every pass has finished (this wgpu build's queue work-done callback fails with a ctype
error, so enqueue time is not used and no GPU timestamps are taken yet). Masks cross the bus as their alpha plane only.

**Baseline: CPU nodes, this revision** (`benchmarks/2d-gpu/R6-1/baseline-cpu-1025-*.json`; uncached, disk cache off,
pipelined playback, 1080p full resolution, private Xvfb with software OpenGL, first pass through the fresh frames only;
fixture `tests/fixtures/playback_comp.nbcomp` sha256 `697f157a...bca8f`; sources from
`tools/benchmark_2d_playback.py --generate-inputs $NB_RUN/plates --size 1080p --frames 240`, 480 files, hashes in
`inputs-1080p-240.sha256`, list digest `ee66e89a...dae6e4`; revision `6c1fd46f`):

| Run | Passes | Fresh fps | Interval mean (ms) | p50 / p95 / p99 (ms) | Node evaluation median (ms) | First picture (ms) |
|---|---:|---|---:|---|---|---:|
| 48 frames, RTX 3080 Ti | 3 | 6.06, 6.25, 6.77 | 148.7 to 159.4 | 148 to 159 / 163 to 179 / 165 to 189 | 141.6 to 151.6 | 202 to 245 |
| 240 frames, RTX 3080 Ti | 3 | 9.01, 9.01, 9.03 | 110.6 to 110.9 | 102 / 155 to 157 / 166 to 167 | 95.6 to 95.9 | 204 to 216 |
| 48 frames, Radeon 8060S | 1 | 6.77 | 148.9 | 149 / 161 / 164 | 141.5 | 199 |
| 48 frames, llvmpipe | 1 | 6.68 | 150.6 | 150 / 165 / 175 | 143.4 | 204 |

All 48 and all 240 fresh frames were accepted in every pass. **The 240-frame variant is easier than the 48-frame comp.** The
curves hold their last key's value after frame 48 (endpoint hold), so from frame 49 the Transform is a whole-pixel translate of
18 and takes the CPU's copy shortcut; only the plates differ. Node evaluation falls from about 143 ms to about 96 ms and the fresh
rate rises to 9 fps for that reason, not because the CPU got faster. The sustained gate is therefore not comparable with the 48-frame
numbers until the extension animates the translation too (a Gonzo decision; the brief allows only curve extrapolation).

**First pixel check** (`tools/check_gpu2d_fixture.py`, evidence `pixels-fixture-1080p-*-1025.json`): 11 fixture frames (1, 2, 6, 12,
18, 24, 30, 36, 42, 47, 48), the Merge output and the ColorCorrect output, GPU against the CPU at full 1080p; the gate is
`abs(gpu - cpu) <= 2e-5 + 2e-5 * abs(cpu)` on every finite channel.

| Adapter | Max abs error | Mean | p99 | Worst gate ratio (limit 1) |
|---|---:|---:|---:|---:|
| RTX 3080 Ti | 2.4e-7 | 1.0e-9 | 6e-8 | 0.0079 |
| Radeon 8060S | 3.0e-7 | 1.0e-9 | 6e-8 | 0.0093 |
| llvmpipe | 2.4e-7 | 1.0e-9 | 6e-8 | 0.0079 |

Every frame: data and display windows equal to the CPU's, one download, zero intermediate readbacks, four nodes on the device,
no CPU boundary. `tests/test_gpu2d.py` (38 tests: mapping, windows, fallbacks without a device; Grade, Blur, Transform, Merge
kernels, the chain, counters, non-finite input, over-budget, device loss and cancellation with one) passes on all three adapters.

**First island playback, preliminary** (one or two passes of the 48-frame run per adapter, not gate runs;
`island-smoke-1025-summary.json`): the island itself works end to end through the real window.

| Adapter | Fresh fps (CPU nodes, same adapter) | Island total, median (ms) | Passes (ms) | Upload (ms) | Readback (ms) | Node evaluation median (ms) |
|---|---|---:|---:|---:|---:|---:|
| Radeon 8060S | 13.3 and 12.9 (6.77) | 14.1 and 13.5 | 5.5 and 5.7 | 5.1 and 4.7 | 2.5 and 2.5 | 68.9 and 69.9 (141.5) |
| RTX 3080 Ti | 5.0 and 5.4 (6.06 to 6.77) | 161 and 94 | 12.0 and 3.0 | 105 and 75 | 43 and 14 | 207 and 155 (141.6 to 151.6) |

The island is not the slow part on either card; the passes take 3 to 12 ms. On the integrated Radeon, which shares memory with the
processor, the whole island takes 14 ms and playback doubles. On this workstation's RTX 3080 Ti the picture is the opposite:
**host to device transfer dominates.** `benchmarks/2d-gpu/R6-1/transfer-rates-1025.json`: one 33 MB frame uploads at 1.0 GB/s (33 ms)
and reads back at 2.9 GB/s (11.6 ms) on the RTX, against 19 and 23 GB/s on the Radeon; the card sits behind a PCI link currently at
5 GT/s by 2 lanes (maximum 16 GT/s by 16), which is this machine's external enclosure, not what a desktop slot or the Windows laptop
will do. The island sends the plate, the background and the Roto matte's alpha plane (66 MB per frame) and reads back 33 MB, so the
transfers alone are 100 ms or more here, which is why the RTX is slower than the CPU nodes in this setup. Whether the RTX row can
pass its 20 ms (four nodes) and 30 ms (uploads, nodes and readback) gates is therefore a statement about the link, to be decided on a
desktop-class RTX and on the RTX 4070 laptop (unverified; Gonzo owns those runs).

**Findings recorded for steps B and C.**

- Cubic Transform next to a very bright pixel (a 50.0 pixel among values near 0.4) differed from the CPU by up to 2.5e-5 in step A.
  Resolved in step B (below): the gate now holds with a worst ratio of 0.006 on all three adapters.
- CPU behaviour reproduced on purpose: `Transform` with `clamp` on and a whole-pixel vertical map zero-pads the rows the source does
  not cover (`_resample_cubic_separable` crops and pads rows), where the general path replicates the edge. The device copies that.
  Needs Lane 2 to say whether the CPU path is intended.
- Blur is validated for radii 1 to 4 (the CPU's own float32 shifted sum); wider radii use float64 cumulative sums on the CPU and stay
  on the CPU until the float32 accumulation is measured (step B).
- Remaining after the island: the node stage still holds the Read decodes and the Roto matte (33.5 and 25.0 ms of the Radeon's 74.1 ms serial
  graph, measured in step C; the CPU ColorCorrect is an identity at the fixture's values and costs 0.07 ms) and the 8-bit display
  conversion; closing 24 fps needs those as well.

Rerun: `python tools/benchmark_2d_playback.py --plate-pattern "$NB_RUN/plates/plate.%04d.exr" --background-pattern "$NB_RUN/plates/background.%04d.exr" --size 1080p --cache-mode uncached --display :N --backend gpu2d --seconds 12 --output run.json`
on a private X display, then `python tools/summarize_2d_playback.py run.json`; the pixel check is
`python tools/check_gpu2d_fixture.py --inputs "$NB_RUN/plates" --size 1080p --frames 1 24 48 --adapter default`; tests with
`python /path/force-adapter.py integrated|cpu tests.test_gpu2d` for the other two adapters.

### R6-1: the pixel and failure gates (10/25, step B of 3)

Step B turns the step A check into the full R6-1 part 3 gate set: `tests/test_gpu2d.py` (62 tests) passes on the RTX 3080 Ti, the
Radeon 8060S and llvmpipe (software), with every finite channel inside `abs(gpu - cpu) <= 2e-5 + 2e-5 * abs(cpu)`. The tolerance
was not widened; two kernel defects were found and fixed instead. The A6000, the RTX 4070 laptop and Windows D3D12 were not available:
those rows are **unverified**, and Gonzo owns them.

**What the tests cover** (all with the CPU nodes as truth, through `Evaluator._windowed_kernel`, so windows, mask and mix are the real ones):
negative and HDR values with 6.0 and 50.0 spikes; alpha exactly 0 and 1 including colour at alpha 0; masks and mix 0, partial and 1 on
every node; odd, 1-pixel and 65 by 1 windows at data origins from -1000 to 4000; border impulses; fractional translations, rotation,
non-uniform scale, pivot, skew, invert, clamp and black-outside for nearest, bilinear and cubic; every Merge operation, both input orders;
Blur radii 0.2 to 4; a seeded sweep of 400 random parameter sets per node; non-finite inputs and results; and the realistic comp frame by
frame (a full 48-frame animation walked every eighth frame, the extrapolated frames 60 to 240, a 480 by 270 frame and a proxy tier),
each with one download, no intermediate readback and no CPU boundary.

**Pixel results** (`benchmarks/2d-gpu/R6-1/kernel-gates-*-1025.json`; the node rows are the worst over every comparison of that node,
"Fixture" is the 23 realistic-comp comparisons; the pass limit for the gate ratio is 1):

| Adapter | Node | Comparisons | Max abs error | Mean | p99 | Worst gate ratio |
|---|---|---:|---:|---:|---:|---:|
| RTX 3080 Ti | Grade | 154 | 7.6e-6 | 2.4e-9 | 4.8e-7 | 0.0058 |
| RTX 3080 Ti | Blur | 229 | 3.8e-6 | 4.1e-10 | 1.1e-7 | 0.0047 |
| RTX 3080 Ti | Transform | 723 | 1.9e-6 | 7.6e-10 | 2.4e-7 | 0.0060 |
| RTX 3080 Ti | Merge | 405 | 1.2e-4 | 2.0e-9 | 9.5e-7 | 0.0082 |
| RTX 3080 Ti | Fixture | 23 | 6.0e-8 | 5.6e-10 | 3.0e-8 | 0.0019 |
| Radeon 8060S | Grade / Blur / Transform / Merge | 154 / 229 / 723 / 405 | 7.6e-6 / 3.8e-6 / 4.8e-7 / 1.2e-4 | 3.7e-9 / 7.4e-10 / 4.5e-11 / 1.9e-9 | 4.8e-7 / 1.2e-7 / 6.0e-8 / 9.5e-7 | 0.0074 / 0.0056 / 0.0043 / 0.0078 |
| Radeon 8060S | Fixture | 23 | 6.0e-8 | 1.5e-9 | 3.0e-8 | 0.0020 |
| llvmpipe | Grade / Blur / Transform / Merge | 154 / 229 / 723 / 405 | 0 / 0 / 2.8e-14 / 1.2e-7 | 0 / 0 / 2.2e-20 / 4.7e-12 | 0 / 0 / 0 / 3.0e-8 | 0 / 0 / 0 / 0.0032 |
| llvmpipe | Fixture | 23 | 0 | 0 | 0 | 0 |

The Merge maximum is large in absolute terms only because the test images hold values near 50. At full 1080p the realistic comp
(`pixels-stepB-48-*` for frames 1 to 48, `pixels-stepB-240-*` for the extrapolated frames 49 to 240, the Merge and ColorCorrect outputs, 34
comparisons per adapter) differs by at most 6e-8 on the RTX and the Radeon (gate ratio 0.002) and by exactly 0 on llvmpipe, with one
download, zero intermediate readbacks and no CPU boundary on every frame. A further 6,400 random node parameter sets per adapter
(four seeds of 1,600, outside the committed sweep) all passed; the worst gate ratio was 0.010.

**Two kernel defects found and fixed.**

- *The sample fraction.* The Transform shader built a source coordinate from two separately rounded float32 pieces, so the fraction could
  be off by about 1e-7; next to a 50.0 pixel the cubic filter turned that into 4e-5, past the gate. The tables now carry each coordinate
  as an exact integer part plus a 48-bit fixed-point fraction in two 24-bit limbs, the shader sums them in integers and rounds once, as
  the CPU's float64 coordinate does when it casts its fraction.
- *The CPU's two cubic paths.* The translation-only (separable) path builds its Catmull-Rom weights in float64 and casts once; the
  general path builds them in float32. The device now takes the separable path's weights from a table the host computes with the CPU's
  own expression, and evaluates the general path's float32 weights with every intermediate behind an opaque bit-mask
  (`rounded`, a mask of all ones read from the parameters) so that no driver can fuse a multiply into the next add or reorder the
  polynomial. Without the mask the Radeon and llvmpipe reorder the weight polynomial (8,478 of 24,001 test weights differed by up to
  1e-6) and the fused tap sums moved the result by a further factor of ten on the RTX; with it the weights match the CPU bit for bit on all three.

**Explicit CPU boundaries, named** (each counted as a reported boundary, never guessed): Blur radius above 4 (the CPU switches to float64
cumulative sums; a float32 port cannot match them inside the gate); Transform motion blur; Merge operations without a kernel
(anything outside the 19 in `MERGE_OPS`, such as overlay); a Merge with an unwired input or mismatched display windows; a window larger than a
quarter of the byte budget or than the device's storage binding; non-finite input pixels, and non-finite results (the whole frame is
recomputed on the CPU, which gives the CPU's exact values); a lost device, after which the island stays off until reopened.

**Failure behaviour, checked with a fake device (no hardware) and with the real ones:** no device (open returns None with the reason, the
benchmark raises instead of silently running on the CPU, the evaluator default stays CPU); unsupported parameter or mode (a CPU boundary);
over budget (refused at the first allocation, nothing submitted, nothing leaked); cancellation (before any device work, and between
passes of a half-built island: no submission, the next frame runs on the same device); device loss (a failed submission or allocation raises
`DeviceLost`, every resident image becomes stale, the evaluator reruns that frame on the CPU and serves the following frames from the
CPU); stale generation (a resident source cached under an earlier device epoch is never reused and is re-uploaded; a cancelled frame
leaves no cached result, so the same evaluator renders the next generation correctly).

Rerun: `python /path/force-adapter.py integrated tests.test_gpu2d` (and `cpu`, or no wrapper for the default card); set `NB_GPU2D_STATS=file.json`
to write the table above. The 1080p check is `python tools/check_gpu2d_fixture.py --inputs "$NB_RUN/plates" --size 1080p --frames 1 2 6 12 18 24 30 36 42 47 48 --adapter default`
(add `--last 240` and frames 49 to 240 for the extrapolated range). Regenerated EXR sources carry a capture date, so their file hashes
change on each generation; the pixels do not.

### R6-1: measurements and the go/no-go (10/25, step C of 3)

Step C measures the island against every R6-1 gate with the new `tools/benchmark_gpu2d.py` (commands `kernels`, `graph`, `playback`, `gates`,
`table`, `resummarize`, `make-inputs`; tests in `tests/test_benchmark_gpu2d.py`) and records the decision. Evidence is in
`benchmarks/2d-gpu/R6-1/` under the `-1025` names: `kernels-*`, `graph-*`, `playback-*`, `gates-*`, `kernels-summary-1025.json` and
`transfer-rates-island-pattern-1025.json`. The A6000, the RTX 4070 laptop with its Iris Xe and Windows D3D12 were not available: those rows
are **unverified**, and Gonzo owns them (field-kit commands at the end of this section).

**Decision.** The island is a go as a design and a no-go as a speed-up on this workstation's RTX 3080 Ti.

- Pixels, residency and execution time pass everywhere they were measured: 402 million channels per GPU mode agree with the CPU chain within
  the gate (worst gate ratio 0.0020, 0 violations), 1 download and 0 intermediate readbacks per frame, and the four nodes take 14.5 ms (RTX
  3080 Ti, p95, with that frame's sources just uploaded) or 4.3 ms (sources already on the card) against 78 to 85 ms on the CPU kernels.
- The RTX 3080 Ti row fails every gate that includes the bus: uploads, the four nodes and the readback take 165.7 ms (p95; gate 30 ms), the
  node stage gets slower than the CPU (213 ms against 142 ms), playback runs at 4.9 fps against 6.7 fps on the CPU nodes. This card sits behind
  a PCI link negotiated at generation 4 with 2 lanes, and moving the 75 MB of sources up and the 33.5 MB result back costs 85 ms on it before
  any node runs (below). The 30 ms gate cannot be met on that link whatever the island does.
- On the integrated Radeon 8060S, where host and card share memory, the same island takes 11.0 ms in total (p95), the node stage gets 2.06 times
  faster on the 48-frame clip, and playback doubles (13.1 fps against 6.7). It still misses 16 fps and 24 fps. **The stage left is the sources:
  the two Read decodes take 33.5 ms and the Roto matte 25.0 ms of the 74.1 ms serial graph, the island 13.6 ms.**

**Method.** Revision `499a0d2a` plus the tool (committed as `c188cd5b`; its only changes to existing code are two extra timing counters in
`gpu2d.py` and a clock offset in the playback harness), fixture `tests/fixtures/playback_comp.nbcomp` unchanged (sha256 `697f157a...`), sources
regenerated with `tools/benchmark_2d_playback.py --generate-inputs $NB_RUN/plates --size 1080p --frames 240` (480 half-float EXR files on disk).
AMD Ryzen AI MAX+ 395, wgpu 0.32.0, NVIDIA driver 615.78.08, RADV, llvmpipe (LLVM 22.1.8). Every GPU run took `/tmp/nb-gpu.lock`, so the integrator's
suites did not overlap them. Timing is bounded completion, never enqueue time: `gpu_ms` runs from the queue submit until a 16-byte flags readback
returns after the last pass; **no GPU timestamp queries**, because the shared device (`gpu3d._state`) is created without the `timestamp-query`
feature and this wgpu build's queue work-done callback fails. Three passes per tier, each a fresh process. A `kernels` pass decodes the
48 animated frames' three sources once into RAM (4.8 GB, excluded from every kernel number), runs two sacrificial frames in every mode, checks
pixels of both GPU modes on all 48 frames, then times four modes over the same frames and parameters: the CPU kernels (proven bit-identical
to `Evaluator` on frames 1 to 3), one GPU run per node (every node uploads its inputs and reads its result back), the island with every source
uploaded for that frame (**fresh**: the transfer-inclusive number) and the island with the sources already resident (**warm**: the first run of
a frame uploads, two timed runs follow). `exec_ms` is host staging plus encoding plus the passes to completion, with the source uploads and the
terminal readback taken out. `playback` passes are the real window through `tools/benchmark_2d_playback.py`: uncached (disk cache off), private
Xvfb with software OpenGL, decode-ahead on and timed, two sacrificial frames before each island run, first lap only (pictures counted when the
Viewer accepts the request generation, every frame of the lap shown, no frame skipped or repeated). Stage records exist for fresh evaluations
only: 47 of the 48 pictures of the 48-frame lap (frame 1 is on screen when play starts) and 240 of 240 for the long clip. The "dropped" counts
in the raw files cover the whole 12 or 60 second window, later laps included, and never count as progress. The 240-frame clip extends the
curves with their last key, so from frame 49 on the Transform is a whole-pixel translate that the CPU copies without resampling: the CPU is
faster there (98 ms node stage against 142 ms) and the island's relative gain shrinks.

**The four nodes alone, predecoded, 1080p, 48 animated frames, three passes pooled** (milliseconds; `kernels-summary-1025.json`):

| Tier | CPU kernels p50 / p95 | One run per node p50 / p95 | Island, sources uploaded that frame p50 / p95 | Island, sources resident p50 / p95 |
|---|---:|---:|---:|---:|
| RTX 3080 Ti (1 GiB cap) | 78.3 / 85.2 | 234.7 / 452.4 | 163.9 / 165.7 | 16.3 / 19.3 |
| Radeon 8060S (512 MiB cap) | 78.8 / 83.6 | 21.7 / 24.1 | 9.7 / 11.0 | 5.4 / 6.0 |
| Radeon 8060S, four CPUs, 2x scale stand-in (512 MiB) | 78.3 / 83.7 | 21.7 / 24.4 | 9.9 / 11.3 | 5.5 / 6.0 |
| llvmpipe, software wgpu, 12 frames, one pass (1 GiB) | 79.3 / 84.5 | 57.5 / 76.9 | 33.5 / 37.9 | 20.4 / 22.2 |
| llvmpipe, 480 x 270, 128 MiB cap | 3.7 / 4.0 | 6.0 / 8.1 | 3.4 / 4.0 | 2.1 / 2.7 |

Execution only (`exec_ms`, p50 / p95; per-pass p95 in brackets):

| Tier | Sources uploaded that frame | Sources resident |
|---|---|---|
| RTX 3080 Ti | 13.6 / 14.5 (14.0, 14.4, 14.6) | 3.2 / 4.3 (4.1, 9.4, 4.3) |
| Radeon 8060S | 4.6 / 5.2 (4.8, 5.2, 5.4) | 3.8 / 4.2 (4.2, 4.0, 4.4) |
| llvmpipe, 1080p | 26.7 / 31.2 | 18.0 / 19.7 |

The CPU chain's medians are Grade 11.4 ms, Transform 24.2, Blur 20.2, Merge 21.1 (graph run); the island's cold shader setup is 29.0 ms
(RTX), 28.5 ms (Radeon) and 39 ms (llvmpipe), measured separately and left out of every pass. Counters per frame, identical on every tier: the island
uploads 3 sources plus the Transform's table (74.7 MB), downloads once (33.5 MB), submits twice, dispatches 9 passes (6 when the sources are
resident, which skips the three input checks), allocates 0 to 2 buffers after warm-up (6 to 9 pool hits), 0 intermediate readbacks, no CPU
boundary; peak 457 MiB of the RTX's 1 GiB cap and 313 MiB of the Radeon's 512 MiB. The per-node round trip moves 175 MB up and 133 MB down for
each frame (4 downloads): residency alone is a 2.2 times gain on the Radeon (21.7 to 9.7 ms).

**Where the RTX's 164 ms goes** (median of the island with fresh sources, pass 2 and 3; pass 1 in brackets): staging 0.7 ms, uploading 74.7 MB
104.7 ms (75.8), encoding 0.3, the passes to completion 12.1 (3.3), reading back 45.3 ms (13.4), of which the device-to-host copy and map take 43.8
(11.9) and the host copy 1.4. Raw rates from `xfer2.py`/`xfer3.py` on this link (`transfer-rates-island-pattern-1025.json`): 75 MB up in 73.7 ms
(1.01 GB/s), 33.5 MB back in 11.8 ms (2.8 GB/s), and the Radeon does the same in 3.6 and 1.9 ms (20.6 and 17.8 GB/s). Three upload paths
(`write_buffer`, a mapped staging buffer plus a copy, and the same in 8 MiB pieces) all take 73.6 ms: the upload call is not the cause. The rate
moves between passes of identical code (pass 1 above) because the card's link renegotiates with load (`nvidia-smi` reads gen 4, 2 lanes of 16
while busy and gen 1 at rest). The best case measured, 93 ms in total, is three times the 30 ms gate. For scale (a calculation from the link width): 2 lanes of generation 4 carry at most 3.9 GB/s, so even a perfect copy path would need 28 ms for the transfers alone; and with
the 14.5 ms of execution the gate asks for about 7 GB/s averaged over both directions.

**Playback through the real window, uncached, first lap, three passes per row** (`playback-*-1025-*.timeline.json.gz`, summaries beside them; ranges
cover the three passes; `Node stage` is the median of the node-evaluation stage in the timeline):

| Row | Fresh fps | Interval p50 / p95 / p99 (ms) | Node stage (ms) | First picture (ms) | Lap time (s) |
|---|---|---|---|---|---|
| RTX 3080 Ti, 48 frames, CPU nodes | 6.64 to 6.75 | 149 to 151 / 161 to 164 / 166 to 169 | 139.4 to 145.2 | 201 to 222 | 7.1 to 7.2 |
| RTX 3080 Ti, 48 frames, island | 4.66 to 4.95 | 222 to 227 / 230 to 354 / 384 to 395 | 210.8 to 217.7 | 185 to 195 | 9.7 to 10.3 |
| RTX 3080 Ti, 240 frames, CPU nodes | 8.78 to 8.95 | 103 to 105 / 154 to 159 / 163 to 169 | 97.3 to 98.3 | 199 to 235 | 26.9 to 27.4 |
| RTX 3080 Ti, 240 frames, island | 5.14 to 5.36 | 183 to 186 / 254 to 276 / 322 to 368 | 175.3 to 177.5 | 189 to 199 | 44.9 to 46.8 |
| Radeon 8060S, 48 frames, CPU nodes | 6.65 to 6.73 | 148 to 150 / 165 to 166 / 166 to 173 | 140.9 to 143.6 | 201 to 216 | 7.1 to 7.2 |
| Radeon 8060S, 48 frames, island | 12.97 to 13.49 | 74 to 75 / 78 to 86 / 79 to 91 | 67.7 to 71.1 | 109 to 115 | 3.5 to 3.7 |
| Radeon 8060S, 240 frames, CPU nodes | 8.85 to 8.93 | 103 to 104 / 155 to 157 / 163 to 164 | 97.1 to 97.9 | 202 to 206 | 26.9 to 27.2 |
| Radeon 8060S, 240 frames, island | 16.90 to 17.43 | 55 to 57 / 79 to 80 / 82 to 85 | 51.9 to 53.6 | 112 to 122 | 13.8 to 14.2 |
| Radeon stand-in (4 CPUs, 2x scale), 48 frames, CPU nodes | 4.72 to 4.81 | 209 to 212 / 222 to 234 / 224 to 236 | 205.1 to 209.8 | 253 to 260 | 9.9 to 10.1 |
| Radeon stand-in, 48 frames, island | 7.11 to 7.26 | 138 to 141 / 148 to 154 / 158 to 164 | 134.0 to 135.8 | 171 to 174 | 6.6 to 6.7 |
| llvmpipe, 48 frames, CPU nodes (numpy) | 6.59 to 6.71 | 147 to 153 / 164 to 168 / 165 to 171 | 142.5 to 145.9 | 198 to 220 | 7.1 to 7.2 |
| llvmpipe, 48 frames, island (software wgpu) | 9.90 to 10.36 | 94 to 98 / 112 to 116 / 119 to 129 | 87.1 to 92.2 | 165 to 176 | 4.6 to 4.8 |

**What the node stage holds under the island** (median pass; each island joined to the accepted frame whose evaluation contained it, 47 or 240
frames per pass; milliseconds):

| Row | Node stage | Island total | Upload | Passes + encode | Readback | Left outside the island |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3080 Ti, 48 frames | 210.8 | 161.0 | 104.9 | 12.4 | 42.8 | 56.6 |
| RTX 3080 Ti, 240 frames | 175.3 | 151.1 | 92.4 | 11.9 | 45.8 | 25.1 |
| Radeon 8060S, 48 frames | 71.1 | 14.2 | 5.2 | 5.8 | 2.4 | 55.8 |
| Radeon 8060S, 240 frames | 52.2 | 11.5 | 3.6 | 4.8 | 2.3 | 40.7 |
| Radeon stand-in, 48 frames | 135.8 | 13.5 | 4.5 | 5.8 | 2.6 | 122.4 |

The serial graph without a window (`graph-*-1025.json`, 48 frames, medians by node kind) names that remainder: CPU nodes 136.0 ms in total (Read 33.3,
Roto 23.9, Grade 11.4, Transform 24.2, Blur 20.2, Merge 21.1, ColorCorrect 0.06); with the island on the Radeon 74.1 ms (Read 33.5, Roto 25.0, island
13.6, staging 0.6, ColorCorrect 0.07); with the island on the RTX 221.7 ms (Read 33.1, Roto 24.9, island 161.6). ColorCorrect at this fixture's
values is an identity that costs 0.06 ms, so moving it to the card would gain nothing here. Display conversion (18 to 19 ms per frame, 37 ms on the
stand-in) overlaps the next evaluation in the pipeline and lifts the interval only on the stand-in.

**Go/no-go against every R6-1 gate.** Pass, fail or unverified, with the number:

| Gate | RTX 3080 Ti | RTX 4070 laptop | A6000 | Radeon 8060S (reference) |
|---|---|---|---|---|
| Pixels within 2e-5 + 2e-5 abs(cpu) | pass: worst ratio 0.0020, 0 violations, 48 frames x 2 modes | unverified | unverified | pass: 0.0020, 0 violations |
| Zero readbacks between the kernels, four nodes resident | pass: 0 readbacks, 1 download, no CPU boundary | unverified | unverified | pass |
| Allocations within the cap | pass: peak 457 of 1024 MiB | unverified | unverified | pass: 313 of 512 MiB |
| Four-node resident execution p95 <= 20 ms | pass: 14.5 ms (sources just uploaded), 4.3 ms (resident) | unverified | unverified | pass: 5.2 ms, 4.2 ms |
| Uploads + execution + terminal readback p95 <= 30 ms | **fail: 165.7 ms** (best pass p50 93.4) | unverified | unverified | pass: 11.0 ms |
| Node-stage median at least 2x better than the same machine's CPU run | **fail: 0.67x** (141.6 to 212.8 ms); 240 frames 0.56x | unverified | unverified | pass 2.06x (143.5 to 69.6); 240 frames fail 1.86x |
| 16 fps fresh playback, interval p95 <= 62.5 ms (laptop: 12 fps, 83.33 ms) | **fail: 4.9 fps**, p95 230 to 354 ms; 240 frames 5.3 fps | unverified | unverified | fail: 13.1 fps, p95 78 to 86 ms; 240 frames 17.3 fps, p95 79 to 80 ms (p95 fails) |
| 24 fps final, interval p95 <= 41.67 ms | **fail** | unverified | unverified | fail |

llvmpipe has no speed gate: its pixel and failure rows pass (480 x 270 at the 128 MiB cap, 1080p at 1 GiB, the 62 step B tests), and the CPU default and
fallback paths are the ones the CPU rows ran and the step B failure tests cover. The Radeon column applies the RTX thresholds for reference; R6-1 sets no
Radeon speed gate. **The spike is PARTIAL: the RTX row is measured and fails its transfer and playback gates, the laptop and A6000 rows are unmeasured.**

**What is left on the 24 fps path (measured stage and an estimate).** On the Radeon the island is 14 ms of a 74 ms frame; the two Read decodes
and the Roto matte, run one after the other on the CPU, are 59 ms. They are independent, so run on three threads they would take about 25 ms (the
Roto), and 25 + 14 = 39 ms would sit under the 41.7 ms frame (an estimate from these medians, unmeasured; it also needs the decode pool and
the conversion stage to keep their overlap). On the RTX 3080 Ti behind a 2-lane link the transfers (85 ms or more) rule out 24 fps and
16 fps; the same card on a wider link is the open question.

**Other findings.** (1) Residency matters: one GPU run per node costs 2.2 times the island on the Radeon and 1.4 times on the RTX, and a
frame's sources must cross the bus once. (2) The result readback is 33.5 MB because CPU ColorCorrect and the display still need float pixels;
a GPU ColorCorrect with the display conversion on the card would remove it (not measured). (3) The decode pool does not hide the Read decodes
under the island: the Radeon's node stage in the window (70 ms) is close to the serial sum of its parts (74 ms). (4) Step A's estimate that the CPU ColorCorrect took part of the
remainder was wrong; the remainder is Read plus Roto. (5) The numbers at the stand-in's four CPUs show the same island time (13.5 ms) and a
node stage dominated by the CPU sources (122 ms outside the island).

**Field-kit commands for the unverified rows (owner: Gonzo).** On the RTX 4070 laptop, A6000 and the Windows D3D12 adapters, with the plates
generated as above and `NB_RUN` set: `python tools/benchmark_gpu2d.py kernels --inputs "$NB_RUN/plates" --adapter default --budget-mb 512 --label rtx4070laptop --output k1.json`
(three times, three fresh processes; 1024 for the A6000; log the adapter, VRAM and the PCI link first), then `python tools/benchmark_gpu2d.py playback --inputs "$NB_RUN/plates" --display :N --backend gpu2d --label rtx4070laptop --frame-range 48 --passes 3 --out-dir out`
and the same with `--backend cpu` and `--frame-range 240`, and finally `python tools/benchmark_gpu2d.py gates --tier rtx4070laptop --kernels k1.json k2.json k3.json --playback out/summary-gpu2d-48-rtx4070laptop.json --cpu-playback out/summary-cpu-48-rtx4070laptop.json`.
Windows rows need the real display route recorded by hand (render and present devices, scaling, power state); no Radeon or Xvfb run can stand in
for them.

Rerun the measurements here: `NB_RUN=... python tools/benchmark_gpu2d.py kernels --inputs "$NB_RUN/plates" --adapter default --budget-mb 1024 --label rtx3080ti --output k.json`
(use `--adapter integrated --budget-mb 512` for the Radeon, `--adapter cpu --budget-mb 128` with `make-inputs` sources at 480 x 270 for the
llvmpipe row, and `taskset -c 0-3` with `QT_SCALE_FACTOR=2` through `playback --cpus 0-3 --env QT_SCALE_FACTOR=2` for the stand-in);
`python tools/benchmark_gpu2d.py graph --inputs "$NB_RUN/plates" --backend gpu2d --label rtx3080ti --output g.json`; `python tools/benchmark_gpu2d.py table --kernels k*.json`;
tests with `python -m unittest tests.test_gpu2d tests.test_benchmark_gpu2d` (and the same through
`run/force-adapter.py integrated|cpu`): 68 tests pass on the RTX 3080 Ti, the Radeon 8060S and llvmpipe.

### R6-1: the independent sources of one frame on worker threads (10/26, source-pool step A of 1)

Step C found the Radeon's 74 ms graph frame made of two Read decodes (33.5 ms), the Roto matte (25.0 ms) and the island (13.6 ms), computed one
after the other. This step computes the independent sources side by side and measures it. Evidence is in `benchmarks/2d-gpu/R6-1/` under the `-1026`
names (`graph-*`, `playback-*`, `source-pool-summary-1026.json`, `source-pool-environment-1026.json`, `playback-summary-*-staticroto-1026.json`).

**Result.** The graph evaluated without a window gets much faster; the real playback window gets the same or a little faster, and the Radeon's
island row does not move. Measured, three fresh passes per row (details below):

- Whole graph, no window, 48 frames, p50 of the frame: Radeon island 74.0 to 74.4 ms to 42.3 to 42.7 ms; CPU kernels 136.0 to 146.8 ms to 102.3 to
  103.4 ms (RTX 3080 Ti: 136.5 to 137.2 ms to 101.3 to 102.4 ms).
- Real window, CPU nodes, 48 frames: 6.66 to 6.69 fps become 6.99 to 7.58 fps on the Radeon box (node stage 142.8 to 143.5 ms become 124.9 to 136.3 ms), and 6.75 fps becomes
  7.70 to 7.75 fps with the RTX selected (node stage 138.8 to 142.0 ms become 120.3 to 122.4 ms).
- Real window, Radeon island, 48 frames: 12.89 to 13.25 fps serial and 12.20 to 12.87 fps with the pool (interval p95 81.5 to 84.8 ms and 83.5 to
  90.0 ms), and 240 frames 17.26 to 17.62 fps and 17.03 to 17.16 fps (p95 77.1 to 79.5 ms and 80.6 to 86.0 ms).
- **The Radeon does not reach 24 fps on the realistic comp and does not pass the 16 fps gate (interval p95 62.5 ms) in any row.** The 240-frame
  clip averages 17 fps, as in step C, with a p95 of 80 ms or more.

**What was built.**

- `nodebased/sourcepool.py`: `SourcePool`, a few daemon threads that run `SourceTask`s. A task is claimed exactly once, by a worker or by the thread
  that joins it; the joining thread runs any task no worker has started, so no work waits in a queue and a pool that has shut down (or has no workers)
  hands nothing to a stopped executor ("cannot schedule new futures after shutdown" is the message `concurrent.futures` raises, which Windows CI showed in the desktop tests on 10/25;
  this pool does not use it). Workers never wait for another task and no lock is held while a source computes, so the pool has no wait cycle with the
  decode-ahead pool, the GPU lock or the evaluator's cache lock. Memory in flight is bounded by the worker count and a byte budget (10% of the cache
  memory, at least 256 MiB, `NODEBASED_SOURCE_POOL_MB`; each source is assumed to hold the size it produced last time, 64 MiB before that); a worker that would pass
  the budget leaves the task to the joining thread.
- Workers: none at three CPUs or fewer, one at exactly four (the process affinity decides; the later Roto read-ahead step turned it on there), otherwise up to three
  (`CPUs - 3`, at most 3). `NODEBASED_SOURCE_WORKERS` sets the count outright (0 turns the pool off). `Window` installs the pool on its evaluator and
  shuts it down at close; a bare `Evaluator()` has none and walks serially as before.
- `Evaluator.evaluate_raster` starts every source with no connected input (Read, Roto, Noise, Ramp, Radial, Rectangle, Grid, Checker; not Text, which paints with Qt)
  when a walk has two or more of them, each as a nested `evaluate_raster` of that one node. The nested call takes the same cache key, disk tier,
  decimation and error as the walk would, so the cache keys are unchanged. A source is joined by the first node that reads its value or digest, so
  a Grade, Transform and Blur run while a Roto computes, and any source nobody read is joined after the walk. When a node fails, the earliest failing source in
  graph order is the error raised, which is the serial walk's order. Cancellation reaches the nested calls through the same event; when the call
  ends, by any route, unstarted tasks are dropped and running ones are waited for, so no source work outlives the call.
- The retained results (`_store`, the hit and miss counters) and the two Roto matte caches take a lock; the disk spill of an evicted result runs outside it.
- `tools/benchmark_gpu2d.py graph --source-workers N` (default 0, the step C method). Playback passes choose the count with
  `--env NODEBASED_SOURCE_WORKERS=N`.

**Tests** (`tests/test_sourcepool.py`, 22 tests, 0.9 s). Pooled pixels and digests equal the serial evaluator's bit for bit on every node, frame and tier 1 and 2; the cache
keys, hit and miss counters, bytes and Roto matte counters equal the serial walk's; two Reads decode at the same moment (a two-party barrier inside the decode); a
Grade-side kernel runs while a slow Roto computes; one or both Reads missing give the serial error text for every target; a failure earlier in the walk wins over a
failing source; cancelling mid-walk raises `Cancelled` and leaves nothing in flight or queued; no source work remains running after a failing call; a shut-down pool
and a one-source byte budget still give the serial pixels; a disabled or connected node is not launched; the `Window` installs and closes the pool.

**Whole graph, no window** (`benchmark_gpu2d.py graph`, Radeon and RTX boxes at 32 CPUs, three workers; the stand-in is `taskset -c 0-3` with two workers forced;
frame p50 / p95 in milliseconds, range over three passes):

| Row | Serial (0 workers) | Source pool |
|---|---|---|
| CPU kernels, Radeon box | 136.0 to 146.8 / 141.1 to 244.4 | 102.3 to 103.4 / 108.8 to 110.9 |
| CPU kernels, RTX box | 136.5 to 137.2 / 142.9 to 143.9 | 101.3 to 102.4 / 108.4 to 109.8 |
| Island on the Radeon | 74.0 to 74.4 / 80.6 to 85.3 | 42.3 to 42.7 / 46.7 to 48.7 |
| Island on the RTX 3080 Ti | 221.9 to 223.0 / 237.2 to 243.4 | 130.0 to 194.0 / 204.6 to 206.6 (the link renegotiates between passes) |
| Stand-in, CPU kernels | 150.1 to 150.8 / 155.4 to 156.4 | 121.1 to 122.2 / 128.2 to 130.4 |
| Stand-in, island on the Radeon | 89.0 to 90.2 / 91.2 to 92.2 | 60.3 to 60.7 / 63.1 to 64.8 |

The per-node rows in the graph files add busy time over threads: with the pool the Radeon island frame reads Read 42.3 to 43.6 ms, Roto 29.7 to 30.2 ms and the
island 11.5 to 11.7 ms, while the frame took 42.3 to 42.7 ms. Running the sources together slows each of them (Read 33.5 to 42.5 ms, Roto 25.0 to 30.0 ms) and the
chain beside them (Grade 11.4 to 12.5 ms), so the gain is smaller than the sum. The step C estimate for the island frame was 39 ms; the measurement is 42.5 ms.

**Real window** (`benchmark_gpu2d.py playback`, uncached, first lap, three passes per row; ranges cover the passes; "Step C" is the 10/25 result on the same fixture,
"Serial" is today's run with `NODEBASED_SOURCE_WORKERS=0`, "Pool" has three workers except on the stand-in; interval in milliseconds, node stage is the median
node-evaluation stage in the timeline):

| Row | Step C fps / p95 / node stage | Serial fps / p95 / node stage | Pool fps / p95 / node stage |
|---|---|---|---|
| Radeon, 48 frames, CPU nodes | 6.65 to 6.73 / 164.8 to 166.1 / 140.9 to 143.6 | 6.66 to 6.69 / 162.3 to 170.5 / 142.8 to 143.5 | 6.99 to 7.58 / 151.4 to 164.4 / 124.9 to 136.3 |
| Radeon, 48 frames, island | 12.97 to 13.49 / 78.3 to 86.5 / 67.7 to 71.1 | 12.89 to 13.25 / 81.5 to 84.8 / 67.6 to 70.8 | 12.20 to 12.87 / 83.5 to 90.0 / 69.9 to 72.9 |
| Radeon, 240 frames, CPU nodes | 8.85 to 8.93 / 154.8 to 157.3 / 97.1 to 97.9 | 8.80 to 8.91 / 155.8 to 160.4 / 96.4 to 97.7 | 8.83 to 9.21 / 136.4 to 152.5 / 97.4 to 98.5 |
| Radeon, 240 frames, island | 16.90 to 17.43 / 78.7 to 79.9 / 51.9 to 53.6 | 17.26 to 17.62 / 77.1 to 79.5 / 51.7 to 53.0 | 17.03 to 17.16 / 80.6 to 86.0 / 53.5 to 55.1 |
| RTX 3080 Ti, 48 frames, CPU nodes | 6.64 to 6.75 / 161.0 to 163.6 / 139.4 to 145.2 | 6.75 / 161.3 to 164.5 / 138.8 to 142.0 | 7.70 to 7.75 / 138.9 to 143.5 / 120.3 to 122.4 |
| RTX 3080 Ti, 48 frames, island | 4.66 to 4.95 / 229.8 to 354.4 / 210.8 to 217.7 | 4.77 to 5.15 / 240.3 to 278.1 / 207.8 to 217.8 | 4.64 to 4.87 / 241.9 to 398.5 / 212.1 to 217.0 |
| Stand-in (4 CPUs, 2x scale), 48 frames, CPU nodes | 4.72 to 4.81 / 222.0 to 233.5 / 205.1 to 209.8 | default, no workers: 4.75 to 4.81 / 221.8 to 229.4 / 206.5 to 209.9 | 2 workers forced: 5.21 to 5.65 / 199.3 to 215.1 / 175.8 to 188.1 |
| Stand-in, 48 frames, island | 7.11 to 7.26 / 147.7 to 154.3 / 134.0 to 135.8 | default, no workers: 7.17 to 7.31 / 147.6 to 155.2 / 132.3 to 135.9 | 2 workers forced: 8.82 to 9.04 / 129.3 to 140.8 / 107.8 to 110.1 |

**Why the window gains less than the graph.** What the windows show (these are measurements; the causes are not established):

- The preview worker's call stacks (the harness's sampling profile) in the Radeon island pass: with serial sources 718 of its 1243 samples (58%) are inside the Roto matte; with
  the pool 650 of 1242 (52%) are the worker waiting in the join for a source and 107 are it running one itself. The Roto is the longest source and the island cannot
  start before it, so the frame takes about the Roto plus the island plus the rest, with or without a second thread.
- A variant of the fixture whose Roto shape does not move (the matte is computed once and cached; `playback-summary-gpu2d-48-radeon8060s-staticroto-1026.json`) plays at 17.7 to
  18.2 fps with the pool and 17.9 to 18.0 fps without, interval p95 65.7 to 72.2 ms and 67.9 to 73.3 ms, node stage 47.0 to 54.4 ms and 48.0 to 52.6 ms. A Roto that costs nothing would
  therefore lift the Radeon island row from 13 to about 18 fps, which still misses the 62.5 ms p95 gate and 24 fps. The window sits near 50 ms of node stage with the matte cached.
- Two decode-ahead workers (`NODEBASED_DECODE_AHEAD_WORKERS=2`) instead of one, with and without the pool, leave the Radeon island row at 12.4 to 12.6 fps (pool) and 13.0 to 13.1 fps
  (serial); node stage 71.4 to 71.9 ms and 68.3 to 70.3 ms.
- Where the decode-ahead pool is off (four CPUs or fewer) the pool does help the window (stand-in rows above: +0.5 to +0.8 fps on the CPU nodes and +1.5 to +1.9 fps on the island, node stage about 20 to 30 ms shorter in both).
  Gonzo answered yes on 10/26; the Roto read-ahead step runs one worker there.

**Do the Radeon rows reach 16 and 24 fps?** No. Fresh playback of the 48-frame lap runs at 12.2 to 12.9 fps with the pool (12.9 to 13.3 fps serial), the 240-frame clip at 17.0 to 17.2 fps with a p95 of 80.6 to 86.0 ms
(the 62.5 ms gate fails, as in step C), and no row comes near 24 fps (41.67 ms). The island frame in the serial graph reached 42.5 ms, 0.8 ms over the 24 fps interval, and the window adds the
display conversion, the hand-off and the other threads on top of it.

**Left out.** RTX 4070 laptop, A6000 and Windows rows stay unverified (Gonzo owns them); the llvmpipe playback rows were not repeated; the RTX 240-frame rows were not repeated (the 48-frame island row is bound by the PCI
link whatever the sources do). Plate files were regenerated with the unchanged generator; EXR headers carry a creation time, so their file hashes differ from `inputs-1080p-240.sha256`.
An early targeted run of `tests.test_gpu2d` and `tests.test_benchmark_gpu2d` (28 s) ran on the RTX 3080 Ti without `/tmp/nb-gpu.lock`; every measurement and the final GPU test runs took it.

**Needs Gonzo.** (Both questions below were answered yes on 10/26 and are built in the Roto read-ahead step that follows.) (1) Decide whether to turn two source workers on at four CPUs (the stand-in gains 0.5 to 1.9 fps; the decode-ahead pool, which an earlier test found to slow that size down, is
off there). (2) The longest source, the animated Roto matte, decides the Radeon island frame; computing the next frame's matte while this frame's island runs (a read-ahead like the decode-ahead pool, using
the source pool) would remove up to the 13 to 18 fps difference measured above, and it needs the app's next-frames list; this step did not build it. (3) The remaining 50 ms of node stage with a free Roto is unexplained.

Rerun: `python tools/benchmark_gpu2d.py graph --inputs "$NB_RUN/plates" --backend gpu2d --adapter integrated --budget-mb 512 --label radeon8060s --source-workers 3 --output g.json`
(`--source-workers 0` for the serial walk) and `python tools/benchmark_gpu2d.py playback ... --env NODEBASED_SOURCE_WORKERS=3`; tests: `python -m unittest tests.test_sourcepool`.

### R6-1: the next frame's Roto matte drawn while this frame is shown, and a source worker at four CPUs (10/26, Roto read-ahead step B of 2)

Step A left the animated Roto matte as the longest source and the thing the Radeon island frame waited for (12 to 13 fps; 18 fps with the matte already drawn). This step
draws the matte of the next frame or two while the current frame is evaluated, converted and shown, and switches the source workers on at four CPUs. Evidence is in
`benchmarks/2d-gpu/R6-1/` under the `-1026b` names (`playback-*`, `playback-summary-*`, `m2-latency-standin-*`, `interaction-standin-*`, `roto-ahead-summary-1026b.json`,
`roto-ahead-environment-1026b.json`, `roto-ahead-gate-stats-1026b.json`). In the file names `ahead` is the first version (the read-ahead on every playback request), `adaptive` the final
code, `noahead` the same code with `NODEBASED_ROTO_AHEAD=0` (the step A configuration), `standin-w0|w1|w2` the stand-in with that many source workers, `standin-default` its default
(one worker), `smoke-ahead1|2|3` a short experiment with the lead set to 1, 2 and 3 frames, and `da2` two decode-ahead workers.

**Result.** The Radeon island row gains 20 to 30 percent where the Roto decides the frame, the RTX island rows gain 5 to 20 percent, the CPU-node rows are unchanged, and the
four-CPU stand-in plays 1.0 to 1.4 fps faster. **The Radeon island comes close to 16 fps on average and does not pass the 16 fps gate (interval p95 62.5 ms), and does not come near 24 fps.**

- Real window, Radeon island, 48 frames: 12.3 fps (control today, 12.2 to 12.9 in step A) becomes 15.2 to 15.9 fps; the node stage 71.5 to 74.0 ms becomes 49.5 to 53.1 ms; interval p95
  87.8 to 89.9 ms becomes 74.8 to 86.2 ms. 240 frames: 17.1 fps becomes 17.0 to 18.0 fps and p95 81.5 to 82.5 ms becomes 70.4 to 75.3 ms (the Roto holds its last key after frame 48, so
  only the first 48 frames of the clip draw a new matte). The earlier always-on version measured 15.0 to 16.2 fps (p95 73.5 to 79.9 ms) and 18.0 to 18.4 fps (p95 68.4 to 70.6 ms).
- Real window, RTX 3080 Ti island: 4.6 to 5.0 fps becomes 5.5 to 5.8 fps (48 frames) and 5.1 to 5.2 becomes 5.4 to 5.6 (240 frames, the rows step A left out). The card's link is still the limit.
- CPU nodes (Radeon box and RTX box, 48 and 240 frames): 7.4 to 7.6 and 9.2 to 9.3 fps without the read-ahead, 7.5 to 7.6 and 9.2 fps with it. The gate below keeps it off there.
- Laptop stand-in (four CPUs, 2x scale): CPU nodes 4.74 to 4.77 fps become 5.67 to 5.83 fps (node stage 205 to 208 ms become 168 to 170 ms); the Radeon island 6.96 to 7.24 fps becomes 8.00 to 8.57 fps
  (node stage 134 to 142 ms become 114 to 119 ms). These gains come from the one source worker; the read-ahead stays off on the stand-in (see the gate).
- Interaction on the stand-in (the evaluator, 24 samples per interaction, three passes): a Grade knob drag is unchanged (87.1 to 87.3 ms, one worker 86.7 to 87.5 ms); a scrub to an uncomputed frame falls from 148 ms to 109 to 112 ms;
  a Roto point drag with the Reads cached rises from 48.1 to 48.4 ms to 49.8 to 50.5 ms (about 1.5 ms, 3 percent). The M2 edit-latency benchmark (`tools/benchmark_4k_viewport.py --source-workers N`, three passes
  each at zero, one and two workers on `taskset -c 0-3`) moves no median outside the pass-to-pass spread (4K Grade 75.1 to 77.4 ms and 74.1 to 78.2 ms; 4K Merge 40.2 to 41.0 and 40.3 to 42.6; 1080p Roto shape drag
  11.3 to 12.0 and 10.7 to 11.9); single passes show p95 outliers with the pool on and off alike (4K Tracker frame step p95 857 to 866 ms off, 854 to 1044 ms with one worker).

**What was built.**

- `nodebased/rotoahead.py`: `RotoAhead` queues, for the Roto nodes with no input under the viewed node, the matte of the next frames in playback order (`NODEBASED_ROTO_AHEAD`, default 2 frames, 0 turns it off). It does
  nothing in a graph that remaps time, for a Roto with a wired input, for a view zoomed to under 80 percent of the picture (the tiles draw only that part), or on the request that starts playback or follows an edit; the window
  schedules it on playback ticks. A Roto whose shapes carry no animation (`payload_animates`, a read-only function over the stored shapes) is read ahead once per document state. The lead is also capped by the bytes in
  flight (`NODEBASED_ROTO_AHEAD_MB`, default 256, counted with the size each Roto produced last).
- `Evaluator.read_ahead_node` evaluates one such Roto for a frame on a pool worker through the ordinary nested `evaluate_raster`, so the matte enters the cache under the key the walk computes and nothing else is stored.
  While it runs it announces its digest; a walk that reaches the same digest waits for it instead of drawing the matte again (`ahead_joins`, cancellable), and a walk that reaches a matte nobody has started draws it itself.
  Its errors stay in its task: the walk that needs the frame draws the Roto and raises the same error on the same node.
- `SourcePool.submit_ahead` is a second, lower queue: workers take it only when no source of the frame in hand is waiting, one stream at a time (`ahead_limit`), and a task that does not fit the byte budget is dropped instead of waiting.
  Nobody joins an ahead task. The window cancels the lead (`RotoAhead.cancel`: queued tasks dropped at once, a running one stops before it stores) when playback stops or turns, the playhead is moved by hand or the document is edited
  (every request that cancels the preview in flight), and when it closes, before the pool shuts down and is waited for.
- The gate. The walk records how long it waited at the join of a Roto it had to draw (`Evaluator.roto_is_critical`, a moving average over frames where the Roto took 5 ms or more); the read-ahead runs while that wait is 4 ms or more, or unmeasured.
  Where the chain beside the Roto is the long part of the frame (the CPU kernels, the stand-in), the matte is ready when the Merge asks for it, and the always-on version measured 0.5 to 4 percent slower there (Radeon CPU nodes, 48 frames: 6.99 to 7.51 fps against 7.42 to 7.60) as the
  drawing took memory bandwidth from the chain. One pass each of the final code (`roto-ahead-gate-stats-1026b.json`): Radeon island 520 matte tasks, 73 drawn and 447 already cached, no walk waited for one in flight, measured wait 15.5 ms;
  Radeon CPU nodes, stand-in island and stand-in CPU nodes 0 tasks, measured wait 0.0 ms.
- Workers at four CPUs: `sourcepool.FOUR_CPU_WORKERS = 1`; none at three CPUs or fewer; `NODEBASED_SOURCE_WORKERS` still sets the count outright. Measured on the stand-in with the same code and the read-ahead off: the CPU nodes play at 5.79 to 5.92 fps with one worker and 5.41 to 5.69
  with two, the Radeon island at 8.57 to 8.68 and 9.13 to 9.23. One worker is the cap because the CPU kernels are what the shipped app runs (the island is opt-in) and a second worker costs them 0.1 to 0.5 fps; the island would take the second worker (+0.5 fps).
- `tools/benchmark_gpu2d.py interaction` (knob drag, Roto point drag, scrub of the realistic comp through the evaluator) and `tools/benchmark_4k_viewport.py --source-workers N` give the latency comparison; `tools/benchmark_2d_playback.py` records the pool and read-ahead counters in its pass file.

**Tests** (`tests/test_rotoahead.py`, 29 tests, 4 s). A matte read ahead equals the walk's bit for bit at tiers 1 and 2 (shapes scaled onto the small canvas so the mattes have pixels) and leaves the cache holding exactly the entries the walk makes;
a Roto that raises gives the serial error text on the same node; a walk that reaches a matte in flight waits and the matte is drawn once; the wait can be cancelled; a cancel drops the queued mattes and a running one stores nothing, with nothing left in flight;
an edit never serves the old matte; the same frame is not queued twice; a held Roto is read ahead once; the lead follows the frame count and the byte cap; nothing is queued without workers or after shutdown and a pool shut down with work queued raises nothing
("cannot schedule new futures after shutdown" cannot occur, the pool does not use `concurrent.futures`); a playing walk, the read-ahead and the source pool run together without deadlock; the gate measures a Roto the walk waits for as critical and one beside a longer chain as ready;
the island frames on the card equal the island frames without the read-ahead (RTX 3080 Ti); the window installs the read-ahead and closing cancels it. `tests/test_sourcepool.py` follows the new four-CPU rule.

**Measured table** (`benchmark_gpu2d.py playback`, uncached, first lap, three fresh passes per row, ranges over the passes: fps / interval p95 in ms / node stage in ms; "Step A" is the earlier 10/26 pool run, "Control" is today's run with `NODEBASED_ROTO_AHEAD=0`, which
is the step A configuration, "Final" is the code of this step; the 240-frame RTX rows have no step A entry and show step C, 10/25):

| Row | Step A | Control | Final |
|---|---|---|---|
| Radeon, 48 frames, island | 12.20 to 12.87 / 83.5 to 90.0 / 69.9 to 72.9 | 12.26 to 12.39 / 87.8 to 89.9 / 71.5 to 74.0 | 15.17 to 15.92 / 74.8 to 86.2 / 49.5 to 53.1 |
| Radeon, 240 frames, island | 17.03 to 17.16 / 80.6 to 86.0 / 53.5 to 55.1 | 17.06 to 17.16 / 81.5 to 82.5 / 53.6 to 54.5 | 17.01 to 17.98 / 70.4 to 75.3 / 46.7 to 50.4 |
| Radeon, 48 frames, CPU nodes | 6.99 to 7.58 / 151.4 to 164.4 / 124.9 to 136.3 | 7.42 to 7.60 / 143.6 to 151.0 / 122.6 to 124.0 | 7.58 to 7.62 / 144.4 to 150.5 / 121.6 to 123.6 |
| Radeon, 240 frames, CPU nodes | 8.83 to 9.21 / 136.4 to 152.5 / 97.4 to 98.5 | 9.19 to 9.32 / 137.1 to 138.4 / 95.8 to 96.4 | 9.22 to 9.26 / 138.1 to 138.8 / 95.6 to 96.5 |
| RTX 3080 Ti, 48 frames, island | 4.64 to 4.87 / 241.9 to 398.5 / 212.1 to 217.0 | 4.58 to 5.02 / 238.4 to 361.4 / 210.9 to 218.5 | 5.46 to 5.80 / 209.0 to 218.8 / 179.6 to 181.9 |
| RTX 3080 Ti, 240 frames, island | step C: 5.14 to 5.36 / 254.4 to 276.0 / 175.3 to 177.5 | 5.05 to 5.19 / 267.7 to 289.5 / 176.2 to 177.4 | 5.37 to 5.64 / 250.5 to 272.0 / 173.3 to 175.7 |
| RTX 3080 Ti, 48 frames, CPU nodes | 7.70 to 7.75 / 138.9 to 143.5 / 120.3 to 122.4 | 7.51 to 7.61 / 145.2 to 152.4 / 123.0 to 125.1 | 7.47 to 7.50 / 147.7 to 153.4 / 123.9 to 124.6 |
| RTX 3080 Ti, 240 frames, CPU nodes | step C: 8.78 to 8.95 / 154.2 to 159.1 / 97.3 to 98.3 | 9.20 to 9.30 / 137.4 to 140.6 / 95.3 to 96.4 | 9.15 to 9.19 / 139.3 to 141.3 / 96.3 to 97.0 |
| Stand-in (4 CPUs, 2x scale), 48 frames, island | no workers: 7.17 to 7.31 / 147.6 to 155.2 / 132.3 to 135.9 | no pool: 6.96 to 7.24 / 147.7 to 151.7 / 133.6 to 141.8; one worker, read-ahead off: 8.57 to 8.68 / 129.5 to 135.5 / 110.6 to 113.9 | one worker (default): 8.00 to 8.57 / 133.6 to 145.3 / 113.7 to 119.0 |
| Stand-in, 48 frames, CPU nodes | no workers: 4.75 to 4.81 / 221.8 to 229.4 / 206.5 to 209.9 | no pool: 4.74 to 4.77 / 226.6 to 228.7 / 205.3 to 208.3; one worker, read-ahead off: 5.79 to 5.92 / 188.4 to 196.2 / 166.9 to 171.0 | one worker (default): 5.67 to 5.83 / 194.6 to 211.6 / 167.8 to 169.8 |

Other rows (stand-in, 48 frames, two workers): CPU nodes 5.41 to 5.69 fps (read-ahead off) and 5.66 to 5.72 (always on); island 9.13 to 9.23 and 9.34 to 9.57. One worker with the always-on read-ahead: CPU nodes 5.88 to 5.91, island 8.96 to 9.08. The stand-in shares its four cores with the
other lanes' processes, which are not pinned, so its ranges move from session to session by about 0.5 fps; compare the rows measured in one session (the control, one-worker and two-worker rows were).

Lead and decode-ahead experiments on the Radeon island, 48 frames, the always-on version, two passes each: a lead of 1, 2 and 3 frames gives 16.3 to 16.6, 16.2 to 16.8 and 16.7 to 16.9 fps (the read-ahead off, one pass: 11.6), so two is kept. With two decode-ahead workers instead of the default
(`NODEBASED_DECODE_AHEAD_WORKERS=2`) and the read-ahead on, three passes give 16.7 to 17.0 fps, interval p95 64.9 to 68.5 ms and node stage 44.6 to 46.3 ms: with the matte ready, decoding the Reads is the next thing the frame waits for. That setting is not made default here.

**Do the Radeon rows reach 16 and 24 fps?** On average the 48-frame lap runs at 15.2 to 15.9 fps (the always-on version 15.0 to 16.2, the short lead experiment 16.2 to 16.9) and the 240-frame clip at 17.0 to 18.0 fps. **The 16 fps gate, an interval p95 of 62.5 ms, is not passed in any row** (74.8 to 86.2 ms and 70.4 to 75.3 ms;
64.9 to 68.5 ms with two decode-ahead workers). **No row comes near 24 fps** (41.67 ms; the node stage with the matte ready is 47 to 53 ms).

**Left out.** RTX 4070 laptop, A6000 and Windows rows stay unverified (Gonzo owns them); the llvmpipe playback rows and the stand-in 240-frame rows were not repeated; the first-version (`ahead`) rows were measured before the gate and before the tick-only scheduling and are kept as evidence of the always-on version.
The GPU islands are opt-in in the app, so the CPU-node rows are what a default install plays. Four short targeted runs (`tests.test_benchmark_gpu2d` twice and the island class of `tests.test_rotoahead` twice, 1 to 30 seconds each) ran on the RTX 3080 Ti without `/tmp/nb-gpu.lock`, a repeat of the step A slip; every measurement and
every later GPU test run took the lock. Plates were regenerated with the unchanged generator (EXR headers carry a creation time, so their file hashes differ from `inputs-1080p-240.sha256`).

**Needs Gonzo.** (1) Whether two decode-ahead workers become the default where the Roto read-ahead is on (+0.7 to +1.0 fps and a p95 near the 16 fps gate on the Radeon island; an earlier test found a decoder slows the four-CPU stand-in down, which this does not touch). (2) The remaining 47 to 53 ms of node stage with the matte ready.
(3) Whether the island becomes the default playback backend, since the read-ahead only pays there.

Rerun: `python tools/benchmark_gpu2d.py playback --inputs "$NB_RUN/plates" --display :183 --backend gpu2d --force-adapter integrated --label radeon8060s --frame-range 48 --passes 3 --budget-mb 512 --out-dir "$NB_RUN/pbb"` (`--env NODEBASED_ROTO_AHEAD=0` for the control,
`--cpus 0-3 --env QT_SCALE_FACTOR=2` for the stand-in), `python tools/benchmark_gpu2d.py interaction --inputs "$NB_RUN/plates" --source-workers 1 --output i.json` and `taskset -c 0-3 python tools/benchmark_4k_viewport.py --source-workers 1 --json m2.json`; tests: `python -m unittest tests.test_rotoahead tests.test_sourcepool`.

### R6-1: where the last 50 ms of the Radeon frame go, and the worker and GPU-node settings (10/26, step C of 3)

Step B left the Radeon island at 15.2 to 15.9 fps with 47 to 53 ms of node stage even with the Roto matte ready. This step traces one played frame stage by stage, adds the two settings
DiMo is to decide (decode and source workers; GPU nodes as the playback backend), each with today's behaviour as its default and a one-line flip, and measures both. Evidence is in
`benchmarks/2d-gpu/R6-1/` under the `-1026c` names: `stages-*` (the stage tables, with the per-pass `*.stages.json.gz` and `*.timeline.json.gz` behind them), `summary-*`, `leaf-samples-*` (the
line-level sampler), `ab-read-path-transform-tables-*` and `join-ab-*` (the interleaved A/B passes), `matrix-*` (the worker and GPU-node matrix, with `matrix-load-log-1026c.txt`), `lead-*`,
`interaction-*` and `environment-1026c.json`. Every timing row is three fresh passes, rows interleaved where two configurations are compared, and a pass counts only when no integrator test suite
was running at either end of it (the integrator's full suite ran over two of this step's measurement sessions, at 9:45 AM and 10:03 AM; the passes it touched were discarded and repeated).

**Result.**

- *The three largest stages of the 50.5 ms node stage* are the plate Read on the evaluating thread (14.2 ms), the island run (15.6 ms) and the CPU ColorCorrect (7.7 ms); Transform staging (5.7 ms) is next. The table and what would remove each follow.
- *The frame is paced by the decode pipeline.* One decode thread decodes two 22 ms frames per 62 ms and the walk meets a plate it has to wait for or decode itself; making the walk's own lookups 7 ms faster moved 8.8 ms into an inline decode and the frame rate did not change.
- *Kept: the walk waits for a decode that is already running* (`DecodeAheadPool.get(join=True)`) instead of decoding the same plate a second time. Radeon island, one decode worker, three interleaved passes: 16.44 to 16.49 fps and interval p95 67.5 to 77.9 ms become
  16.56 to 16.93 fps and 63.8 to 67.0 ms; stand-in island with one decode worker 9.93 to 10.27 fps and p95 124 to 131 ms become 10.47 to 10.74 fps and 107 to 115 ms; CPU nodes and two decode workers unchanged. Turn it off with `NODEBASED_DECODE_JOIN=0`.
- *Taken back: a memo of each Read's directory resolution and the kept Transform axis tables.* Both worked as built (the Read's lookups fell from 12.4 to 5.3 ms) and the frame rate did not move; see below.
- *The harness's own stack sampler costs 0.7 fps* and was on in every earlier table; the Radeon island plays 16.1 to 16.7 fps with it off.
- *Decode workers:* one decode worker at four CPUs (today none) is a clear gain on the stand-in (CPU nodes +6.5 percent, island +16 percent); a second decode worker adds 0.2 fps and 7 ms of p95 on the Radeon island and 0.2 fps on the stand-in's island, costs 0.1 fps on the stand-in's CPU nodes and
  changes nothing on the RTX. Two source workers help the stand-in's island (+7 percent) and cost its CPU nodes 3 percent.
- *GPU nodes:* a large gain wherever the integrated card is the one used (Radeon 16.2 to 16.9 fps against 7.4 to 7.8; knob drag 88 ms to 7 ms), a loss on the RTX 3080 Ti on its USB4 link (5.6 to 5.9 fps against 7.8, scrub 102 to 150 to 161 ms). The island opens on the adapter that
  Settings → Graphics adapter names, and "Automatic" picked the discrete card, so a default-on island would have landed on the RTX here (step D: Automatic now picks the card that measures fastest for the island, see below).
- The 16 fps gate (interval p95 62.5 ms) is still not passed in any configuration (best single pass 63.4 ms) and 24 fps is not near (16.2 to 17.0 fps).

**The stage table.** One played frame of the realistic comp on the Radeon island (48 frames, first lap, three fresh passes, the Roto read-ahead of step B on, the harness's stack sampler off; `tools/benchmark_gpu2d.py playback --stages --no-sampler`, then
`tools/benchmark_gpu2d.py stages`). Every number is exclusive milliseconds per fresh frame on the thread that evaluates it (a stage's own time without the stages nested in it), averaged over the frames of a pass and then over the passes; the range is over the passes.

| Stage on the evaluating thread | ms | range |
|---|---|---|
| **The plate Read** (first source, run by the walk): file lookups and cache handling | 11.3 | 10.2 to 12.2 |
|   the same, decode run inline because the pool had not finished it | 2.9 | 0.9 to 5.4 |
| **The island run**: upload to the card | 5.4 | 5.4 to 5.5 |
|   kernels on the card (submit to done) | 5.1 | 5.0 to 5.1 |
|   download: host copy into the picture | 2.6 | 2.5 to 2.6 |
|   download: copy and map | 0.8 | 0.7 to 0.8 |
|   host work: parameter blob, submission, buffers, encode | 1.8 | 1.6 to 2.0 |
| **ColorCorrect** on the CPU kernels (the viewed node) | 7.7 | 7.5 to 7.9 |
| Transform staging (plan and tables on the CPU) | 5.7 | 5.0 to 6.2 |
| Wait at the Roto join (the matte was drawn ahead) | 1.2 | 1.0 to 1.5 |
| Walk between nodes and the other staging | 0.8 | 0.8 to 0.8 |
| *Evaluator walk, total* | 45.2 | 43.9 to 46.8 |
| *Node stage* (the window's evaluation start to end; 5.3 ms of it is the window's own work around the walk) | 50.5 | 49.3 to 51.7 |

The island run splits by the island's own counters (`upload_ms`, `encode_ms`, `gpu_ms`, `readback_map_ms`, `readback_host_ms`). In the same node stage the other threads were busy: the decode-ahead thread decoded for 43.5 ms (two 22 ms decodes per frame on one thread), the source
workers drew mattes for 49.6 ms (27.9 ms for the frame in hand, 21.7 ms ahead) and evaluated the second Read for 10.5 ms. The same table with the stack sampler on (`stages-radeon8060s-island-with-sampler-1026c.json`) has a node stage of 53.0 ms (51.1 to 54.2).

**The three largest stages, and what would remove each.**

1. *The plate Read on the evaluating thread, 14.2 ms.* The walk evaluates the first source itself (the pool takes the others), and the frame comes from the decode pool only when that has finished it: in the first of the sampler-on passes the walk decoded the plate inline on 9 of 47 frames (18 to 30 ms each).
   The other 11 ms are the walk's file-system lookups of the Read (41 system calls per Read, 33 of them `lstat` from three `Path.resolve` calls), which take 19 µs alone and 11 ms while the other threads run. What removes it: a second decode thread or a longer lead (the setting below and `NODEBASED_DECODE_AHEAD_FRAMES`;
   on the Radeon island, three interleaved passes each, a lead of 2 with one worker gives 16.64 to 16.91 fps and a lead of 1 with two workers 16.53 to 16.99, against 16.22 to 16.52 today, and the two together no more (16.34 to 16.73; `lead-radeon8060s-island-1026c.json`)), and the join above. Making the lookups cheaper does not help on its own.
2. *The island run, 15.6 ms.* Upload 5.4 ms (the plate, the background and the matte's alpha plane: 74 MB per frame, about 14 GB/s into the integrated card's memory), kernels 5.1 ms, download 3.3 ms (33 MB back, 2.6 ms of it the host copy out of the mapped buffer into the picture), 1.8 ms of host work.
   What would remove it: uploading the sources as half floats (half the bytes; the colour ingest after decode produces float32, so the ingest would have to move to the card too) and keeping the finished frame on the card for the display transform instead of downloading it.
3. *ColorCorrect on the CPU kernels, 7.7 ms.* The viewed node is not an island node, so the island's result comes back to the host and a full-frame float kernel runs on it. What would remove it: a ColorCorrect kernel in the island (a pointwise kernel), which also lets the frame stay on the card for the display transform, the same step as the download above.

Transform staging, 5.7 ms, is the next stage. It takes 0.57 ms when nothing else runs and 5 to 6 ms while the source workers draw mattes.

**What else runs costs the evaluating thread about 20 ms.** The same frame with the source pool off (`NODEBASED_SOURCE_WORKERS=0`, the Roto drawn inline, one pass, `stages-radeon8060s-island-serial-sources-1026c.json`, 13.7 fps): Transform staging 0.6 ms, the Read's lookups below 0.5 ms, ColorCorrect 5.1 ms, upload 4.7 ms, download copy 2.1 ms,
and 43.9 ms for the Roto inline. A line-level sampler of the evaluating thread (`--leaf-sampler`, `leaf-samples-*`) puts all of the Read's 11 ms in `os.lstat` and `stat` calls. An isolated test (a thread drawing mattes beside a thread making the same calls) did not reproduce the slowdown, and changing the interpreter's thread switch
interval to 1 ms and 0.5 ms gave 16.5 to 16.8 and 16.6 to 16.9 fps against 16.1 to 16.7, inside the pass spread, so the mechanism is not isolated here. A cheaper matte rasteriser (44 ms for a 1080p matte on one core today) would take the competition away; that is Roto's area.

**The harness's stack sampler costs 0.7 fps.** `benchmark_2d_playback.py` samples every thread's stack every 2 ms, which takes the interpreter lock each time, and every earlier table of this document was measured with it on. The Radeon island with `--no-sampler` plays 16.1 to 16.7 fps (node stage 49.3 to 51.7 ms, p95 67.7 to 79.3 ms) against 15.3 to 16.1 fps (51.1 to 54.2 ms) with it, three passes each. The tables of steps A and B stay comparable
with each other; the tables of this step say which they are.

**Two changes tried and taken back** (`ab-read-path-transform-tables-radeon8060s-1026c.json`, commit `51882387` and its revert). (a) A memo of each Read's directory resolution (trusted while `stat` of the directory names the same device and inode; 9 tests, results identical to `Path.resolve` for plain files, links, retargeted links, missing files, relative paths and `..`) took the Read's lookups from 12.4 to 5.3 ms.
(b) The Transform plan's axis tables, kept for the axis a frame did not move (5 tests, bit-identical plans, GPU gates unchanged on three adapters). Interleaved passes of the Radeon island, plain then changed, three each: 16.36 to 16.45 fps and a 51.9 ms node stage plain, 15.73 to 16.29 fps and 54.1 ms changed. The 7 ms the lookups gave back went to an inline decode of the same plate (2.8 to 11.6 ms): the walk reaches the Read sooner and finds the decode thread
still busy. The pipeline is paced by the decode pool, and the join above is the change that addresses it.

**The settings.** Three rows in Settings, in the Interface list under Max properties panels: *2D nodes* (the check box "Compute 2D nodes on the graphics card"), *Decode workers* and *Source workers* (Automatic, Off, 1, 2, 3, and 4 for decode). They are kept in the machine's preferences (`playback/gpu_nodes`, `playback/decode_workers`, `playback/source_workers`), never in the project. The defaults are three constants at the top of
`nodebased/playbackprefs.py`; flipping one is that one line, and an artist who has chosen a value keeps it:

| Constant | Today | Meaning |
|---|---|---|
| `DEFAULT_GPU_NODES` | `False` | Grade, Transform, Blur and Merge run on the CPU kernels |
| `DEFAULT_DECODE_WORKERS` | `None` | automatic: one decode thread, none at four CPUs or fewer |
| `DEFAULT_SOURCE_WORKERS` | `None` | automatic: none at three CPUs or fewer, one at four, up to three beyond (step B) |

The environment overrides win over the saved choice, for the benchmarks, and Settings says so under the row when one is set: `NODEBASED_GPU_NODES=0|1`, `NODEBASED_DECODE_AHEAD_WORKERS=N` (0 now turns the pool off; it used to read as "use the default"), `NODEBASED_SOURCE_WORKERS=N`, `NODEBASED_DECODE_AHEAD_FRAMES=N` (how many frames ahead the decode pool is asked for: 1 at full size, 3 on a proxy tier) and `NODEBASED_DECODE_JOIN=0`.

**A changed setting takes effect at once; nothing needs a restart.** The worker counts resize the two pools while the window runs (`DecodeAheadPool.resize`, `SourcePool.resize`: new threads start, surplus threads retire as each finishes the job it is on, a task nobody takes stays with the thread that joins it). The GPU nodes are installed or removed on the preview worker between two frames (`Window.apply_gpu_nodes`), so a frame in flight never sees the backend change under it; the evaluator's cache and the display and tile caches are cleared at the switch so no picture mixes both, and the frame is drawn again.
The evaluator's disk tier is left alone: a result of either backend has the same digest and the two agree to within the gate (`abs(gpu - cpu) <= 2e-5 + 2e-5 * abs(cpu)`).

**What the fallback does when the card refuses the island.** (1) No adapter, no wgpu, or a card that fails to compile the island's shaders: the CPU nodes stay in use, the status bar says "GPU nodes unavailable (reason); the CPU nodes stay in use", the Settings row repeats the reason, and the check box stays as chosen (switching it off clears the refusal, switching it on tries again). (2) The island refuses one node or graph (an unsupported Transform mode, a window larger than the card's budget): that node runs on its CPU kernel and the island carries on with the others.
(3) A device lost while playing: the frame in hand reruns whole on the CPU nodes (same pixels), then the window drops the island with the message "the graphics device was lost"; switch the setting off and on to try again. (4) A frame with non-finite pixels is recomputed whole on the CPU, as before. The island opens on the adapter Settings → Graphics adapter names; with "Automatic" it opens on the card that measures fastest for the island (step D; step C opened the discrete card first).

**Decode and source workers: measured** (`playback --no-sampler`, 48 frames, first lap, uncached, three clean passes per row, rows interleaved; fps / interval p95 in ms / node stage p50 in ms; ranges over the passes; `NODEBASED_DECODE_AHEAD_WORKERS` and `NODEBASED_SOURCE_WORKERS` set the rows, as the settings do). Today's defaults: Radeon and RTX box one decode worker and three source workers; the stand-in no decode worker and one source worker.

| Radeon 8060S (integrated) | GPU nodes (island): fps / interval p95 ms / node stage ms | CPU nodes: fps / interval p95 ms / node stage ms |
|---|---|---|
| today's default | 16.20 to 16.62 / 69.0 to 81.2 / 48.6 to 51.7 | 7.42 to 7.81 / 140.8 to 146.1 / 122.1 to 129.8 |
| 2 decode workers | 16.24 to 16.91 / 64.2 to 68.4 / 50.0 to 50.2 | 7.57 to 7.74 / 139.0 to 169.3 / 123.3 to 124.9 |

| RTX 3080 Ti (USB4 link) | GPU nodes (island): fps / interval p95 ms / node stage ms | CPU nodes: fps / interval p95 ms / node stage ms |
|---|---|---|
| today's default | 5.62 to 5.93 / 206.3 to 209.6 / 179.3 to 185.1 | 7.74 to 7.78 / 140.5 to 147.8 / 122.7 to 125.7 |
| 2 decode workers | 5.51 to 5.87 / 214.1 to 229.8 / 179.8 to 193.3 | 7.75 to 7.85 / 139.2 to 144.3 / 120.9 to 123.5 |

| Four-CPU stand-in (Radeon, 4 CPUs, 2x scale) | GPU nodes (island): fps / interval p95 ms / node stage ms | CPU nodes: fps / interval p95 ms / node stage ms |
|---|---|---|
| today's default | 8.71 to 8.80 / 128.7 to 131.8 / 109.7 to 110.7 | 5.89 to 5.94 / 188.3 to 195.3 / 165.0 to 165.9 |
| 1 decode worker | 10.11 to 10.27 / 125.8 to 129.2 / 60.7 to 80.7 | 6.28 to 6.33 / 173.7 to 177.3 / 152.1 to 154.7 |
| 2 decode workers | 10.36 to 10.48 / 107.7 to 112.1 / 78.5 to 86.9 | 6.17 to 6.18 / 181.9 to 184.7 / 156.5 to 158.6 |
| 2 source workers | 9.29 to 9.40 / 128.4 to 133.7 / 100.9 to 104.5 | 5.70 to 5.76 / 198.6 to 202.0 / 168.4 to 170.8 |
| 1 decode + 2 source workers | 9.99 to 10.05 / 128.3 to 130.1 / 71.3 to 84.8 | 6.27 to 6.30 / 169.1 to 176.9 / 150.9 to 155.2 |


On the Radeon and the RTX box the second decode worker changes the frame rate by at most 0.3 fps (inside the spread everywhere but the Radeon island's p95, 64 to 68 ms against 69 to 81 ms). On the stand-in a first decode worker lifts the island from 8.7 to 10.1 to 10.3 fps (the node stage from 110 to 61 to 81 ms) and the CPU nodes from 5.9 to 6.3 fps; a second adds 0.2 fps to the island (p95 108 to 112 ms against 126 to 129) and takes 0.1 fps
from the CPU nodes; a second source worker helps the island (9.3 to 9.4 fps) and hurts the CPU nodes (5.7 fps).

*Latency, on the stand-in and the other machines* (`tools/benchmark_gpu2d.py interaction`, the evaluator to ColorCorrect without a window, 24 samples per interaction, three passes; median ms, range over the passes). The decode pool works only while playing (`Window.request_preview` asks it for frames only when `self.playing`), so the decode setting cannot change these three; the source workers can.

| Interaction | Stand-in CPU nodes: 0 / 1 (today) / 2 source workers | Stand-in GPU nodes: 0 / 1 / 2 | Radeon CPU / GPU nodes | RTX CPU / GPU nodes |
|---|---|---|---|---|
| Grade knob drag, sources cached | 87.9-88.7 / 87.5-89.1 / 87.0-88.2 | 6.6-6.8 / 6.8-7.5 / 6.8-6.9 | 86.9-88.2 / 6.4-7.0 | 87.2-88.0 / 17.7-18.2 |
| Roto point drag, Reads cached | 47.1-48.3 / 49.7-50.4 / 49.2-50.5 | 34.4-34.5 / 35.4-36.0 / 35.8-36.1 | 48.3-49.9 / 35.9-36.7 | 48.7-49.2 / 52.9-53.4 |
| Scrub to a frame nobody has computed | 149.4-150.9 / 114.6-117.3 / 120.2-121.0 | 89.2-89.7 / 70.7-71.1 / 60.7-61.5 | 102.4-103.7 / 42.0-43.6 | 101.6-103.9 / 150.3-161.0 |

A second source worker costs the stand-in's CPU nodes 3 to 5 percent on a scrub and gains its island 14 percent; the knob drag does not move.

**GPU nodes as the default: what the measurements say.**

- *Gains:* every machine whose island runs on a card with fast access to memory. Radeon 8060S: 16.2 to 16.9 fps against 7.4 to 7.8 on the CPU nodes, a knob drag of 6.4 to 7.0 ms against 87 ms, a Roto point drag of 36 ms against 49 ms, a scrub of 42 to 44 ms against 102 ms. Four-CPU stand-in: 8.7 to 10.5 fps against 5.7 to 6.3, a knob drag of 7 ms against 88 ms, a scrub of 61 to 89 ms against 115 to 150 ms.
- *Loses:* the RTX 3080 Ti on its USB4 link, which was the adapter "Automatic" picked on this machine (step D: it no longer does): 5.5 to 5.9 fps against 7.7 to 7.9 on the CPU nodes, a scrub of 150 to 161 ms against 102 ms and a Roto point drag of 53 ms against 49 ms; only the knob drag wins (18 ms against 87 ms). The RTX 4070 laptop, the A6000 and Windows are unmeasured (Gonzo owns them).
- *CPU-node machines do not change:* with the setting off the code path is today's (step B measured the always-on read-ahead up to 4 percent slower on CPU nodes; the gated read-ahead of that step leaves them at 7.4 to 7.8 fps), and a card that refuses leaves them there (see the fallback above).
- *What a default-on island would need first:* a choice of adapter for the island that is not "the discrete card first" (a probe of the upload and download speed would do), since on a machine like this one the setting would send the island to the slower card (step D builds it); and the unmeasured machines.

**Tests.** `tests/test_playbackprefs.py` (24 tests, 3 s): the defaults are today's and flip with one line; automatic counts equal the old CPU rules at 2, 3, 4, 6 and 16 CPUs; the environment wins; both settings round-trip through the saved preferences (including Off against Automatic); the Settings dialog shows and returns each choice; both pools grow and shrink while they live (three tasks at once need three workers; a shrunk pool retires idle workers; a pool of none runs the work on the joiner; a shut-down pool ignores a resize); a running window takes new counts without a restart and the
environment fixes a count and says so; a card that refuses leaves the CPU nodes with a message; turning the GPU nodes on and off in a running window (on the software adapter) goes through the island, matches the CPU nodes within the gate on every frame and leaves the evaluator as before when turned off again; a choice changed while the card opens is applied afterwards; a card that loses its device is dropped with a message and the frame stays right;
every worker count (0, 1, 2, 3) gives pixels identical to the serial evaluator. `tests/test_decodepool.py` (4 new): a running decode is joined and runs once, a queued job is not waited for, cancelling and a seek stop the wait, shutdown wakes a joiner. Neighbouring modules (`test_sourcepool`, `test_rotoahead`, `test_tileexec`, `test_playback_pipeline`, `test_perf_playback`, `test_benchmark_gpu2d`, `test_desktop` decode and settings tests) pass, 185 tests together.
`tests.test_gpu2d`, `tests.test_playbackprefs`, `tests.test_rotoahead` and `tests.test_decodepool` pass on the default adapter (RTX 3080 Ti), the integrated Radeon and llvmpipe (134 tests each); every GPU run took `/tmp/nb-gpu.lock`.

**Left out.** RTX 4070 laptop, A6000 and Windows rows stay unverified (Gonzo owns them). The tile executor's own Read path does not use the join; the realistic comp takes the full-frame evaluator (47 full-frame evaluations and no tile composes in a pass). The 240-frame rows were not repeated. The decode setting was not measured on a machine with a card on another bus than these three.

**Needs Gonzo.** (1) The defaults: a first decode worker at four CPUs (`DEFAULT_DECODE_WORKERS = 1`, or the automatic rule) is the largest measured gain on the stand-in; the second decode worker adds little. (2) GPU nodes as the default: gains on the integrated card, loses on the RTX over USB4; the island needs its own adapter choice first. (3) The remaining stages: a cheaper matte rasteriser (Roto's area), ColorCorrect and the display transform on the card, half-float uploads.

Rerun: the stage table is `python tools/benchmark_gpu2d.py playback --inputs "$NB_RUN/plates" --display :183 --backend gpu2d --force-adapter integrated --label radeon8060s-1026c --frame-range 48 --passes 3 --budget-mb 512 --stages --no-sampler --out-dir "$NB_RUN/pb"` and then `python tools/benchmark_gpu2d.py stages --dir "$NB_RUN/pb" --prefix gpu2d-48-radeon8060s-1026c --output stages.json`;
the matrix rows add `--env NODEBASED_DECODE_AHEAD_WORKERS=2` and the like (`--cpus 0-3 --env QT_SCALE_FACTOR=2` for the stand-in); interaction is `python tools/benchmark_gpu2d.py interaction --inputs "$NB_RUN/plates" --source-workers 1 --backend gpu2d --adapter integrated --output i.json`; tests `python -m unittest tests.test_playbackprefs tests.test_decodepool`.

### R6-1: Automatic picks the card the island is faster on, and the Read's file checks (10/26, step D of 4)

Step C found the two things that decide whether "GPU nodes by default" can be offered: "Automatic" put the island on the RTX 3080 Ti, where it runs a quarter slower than the CPU nodes, while the integrated Radeon doubles them; and the plate Read spent its
lookup time in `lstat`/`stat` calls. This step makes Automatic choose by measurement and cuts the Read's calls. Evidence is in `benchmarks/2d-gpu/R6-1/` under the `-1026d` names: `automatic-playback-*` (the Automatic rows), `island-probe-*` (the measurements Automatic keeps),
`island-open-cost-*`, `automatic-discrete-only-*`, `interaction-automatic-*`, `syscalls-per-read-*`, `read-lookups-*`, `read-lookups-cpu-nodes-*`, `read-leaf-samples-*`, `environment-*` and `load-log-*`. Every timing row is three fresh passes with the rows interleaved, and every pass ran under `/tmp/nb-gpu.lock` with no integrator suite
running (`load-log-1026d.txt`: 40 passes, `other_suites=0` in each).

**Result.**

- *Automatic picks the Radeon on this machine, and nothing else changed for an artist who names a card.* Realistic comp, 48 frames, first lap, island on: 17.26 to 17.37 fps where Automatic's old choice (the RTX) gave 5.69 to 6.11, against 7.95 to 8.04 on the CPU nodes. On the four-CPU stand-in: 9.00 to 9.04 fps against 3.86 to 3.99 and 6.06 to 6.08 on the CPU nodes.
- *A machine whose only card is the slow one keeps the CPU nodes.* With the Radeon hidden from Automatic the RTX measures 89 ms against 32 ms for the same two nodes on the CPU, no card passes, the island does not open and Settings says why (`automatic-discrete-only-1026d.json`).
- *The Read's lookups went from 27 system calls to 3 per evaluation* (41 to 5 for a played frame, which also asks for the decode pool's key). The plate Read stage fell from 10.2 to 7.9 ms on the Radeon island and the frame rate did not move (16.91 to 16.93 fps, 16.95 to 17.10 after): the 2.3 ms went to the next stage, as in step C.
  On the CPU nodes, where the walk is the critical path, the change is a small gain: 7.86 to 7.94 fps becomes 7.97 to 8.02, and on the stand-in 5.95 to 6.00 becomes 6.05 to 6.07.
- The 16 fps gate (interval p95 62.5 ms) is not passed in any configuration yet (best pass 63.4 ms in step C, 64.5 ms here); 24 fps is not near.

**How Automatic chooses** (`nodebased/islandadapter.py`). The probe is one island frame in the pattern of the realistic comp, 1920 x 1080 in float: a plate, a background and a matte are uploaded, Grade and Merge run, the result is downloaded; a warming run, then five timed runs, the median of the whole run. The same two nodes run on the CPU kernels beside it.
A card is eligible when its median is below the CPU nodes' median; the fastest eligible card wins, ties go to the discrete card, then to the name. When no card is eligible the CPU nodes stay in use and the reason is shown. Software adapters are not candidates.

| Measured on this machine (median of five, ms) | Whole probe frame | Upload (75 MB) | Kernels | Download (33 MB) |
|---|---|---|---|---|
| RTX 3080 Ti (USB4, PCIe gen 4 x2) | 88.8 | 74.2 | 1.2 | 12.8 |
| Radeon 8060S (integrated) | 8.2 | 4.3 | 2.7 | 1.5 |
| The same two nodes on the CPU kernels (32 CPUs / the four-CPU stand-in) | 32.0 / 32.6 | | | |

The probe's ratio to the CPU nodes against the realistic comp's node stage: Radeon 0.25 against 0.40, RTX 2.8 against 1.54 (island node stage over CPU-nodes node stage). Two points put the break-even at a probe ratio of about 1.6, so a threshold of 1.0 (the card must beat the CPU nodes on the probe) leaves room for the comp's other costs.

- *Cached per card and driver.* The record is kept in the machine's preferences (`playback/island_probe`, one JSON text) under the adapter's identity (vendor, device, driver version, adapter type, backend); the CPU figure is kept under the processor count. A new card, a new driver or a new probe version measures again; nothing else does.
- *Never during playback, never while the card is busy.* While a clip plays, or while another job holds `/tmp/nb-gpu.lock` (taken without waiting, held for the measurement only), the island does not open: the CPU nodes stay in use, the status bar says "GPU nodes waiting: ...", and the window asks again when playback stops and every 20 seconds after a busy lock. A stored measurement is used at once, during playback too.
- *An explicit adapter is never overridden.* A card named in Settings → Graphics adapter or by `NODEBASED_ADAPTER` is used as named and is not measured. (That setting also names the 3D adapter; Automatic is the empty choice.)
- *Settings shows the card and the reason.* With the island on, the 2D nodes row reads for example "Running on AMD Radeon 8060S Graphics (RADV STRIX_HALO). Automatic: measured 8 ms for a frame's upload, kernels and download, against 32 ms on the CPU nodes; NVIDIA GeForce RTX 3080 Ti took 89 ms." When no card passes it reads "No graphics card was faster than the CPU nodes in the measurement (...); the CPU nodes are in use."
- *Cost.* The first Automatic open on this machine took 1.7 s in total (both cards opened, the CPU nodes and both cards timed, off the Qt thread), later opens 0 ms (`island-open-cost-1026d.json`).
- *Switches.* `NODEBASED_ISLAND_PROBE=0` skips the measurement and keeps the old ranking (discrete first); `NODEBASED_ISLAND_PROBE_FILE=<path>` keeps the measurements in a file instead of the preferences (the benchmarks use it). `GPU2D.open("automatic")` is the call the app makes; the benchmarks take `--island-adapter automatic` (playback) and `--adapter automatic` (interaction).
  A benchmark that runs under `flock /tmp/nb-gpu.lock` must also set `NODEBASED_GPU_LOCK_HELD=1`, or the measurement finds the lock busy and waits.

**Automatic before and after** (`playback --no-sampler`, 48 frames, first lap, uncached, 1080p; fps / interval p95 ms / node stage p50 ms, range over three interleaved passes).

| | Island, Automatic before (the discrete card first) | Island, Automatic now | CPU nodes |
|---|---|---|---|
| This machine | RTX 3080 Ti: 5.69 to 6.11 / 218.0 to 256.2 / 179.0 to 191.6 | Radeon 8060S: 17.26 to 17.37 / 64.5 to 66.7 / 47.7 to 50.4 | 7.95 to 8.04 / 137.4 to 139.7 / 119.8 to 122.1 |
| Four-CPU stand-in (4 CPUs, 2x scale) | RTX 3080 Ti: 3.86 to 3.99 / 425.4 to 431.8 / 257.2 to 257.7 | Radeon 8060S: 9.00 to 9.04 / 118.5 to 119.5 / 105.8 to 107.4 | 6.06 to 6.08 / 178.9 to 183.8 / 161.3 to 161.8 |

*Latency* (`interaction`, evaluator to ColorCorrect, 24 samples, source workers 1, median ms, three passes): Automatic (the Radeon) against the CPU nodes: Grade knob drag 6.9 to 7.0 against 87.6 to 89.3, Roto point drag 35.3 to 36.3 against 48.3 to 49.3, scrub to a frame nobody has computed 50.4 to 50.6 against 99.9 to 100.5.
Automatic and `--adapter integrated` give the same numbers in the same session (scrub 50.6 to 51.2 and 50.5 to 50.7). The same Radeon rows at the step C commit and at this step, interleaved: scrub 51.3 to 52.0 and 50.3 to 50.6; step C's own table said 42 to 44 ms for the Radeon scrub, a figure this session's passes do not reproduce at either commit.

**The Read's file checks.** One Read evaluation of the plate made 27 system calls (22 `lstat`, 5 `stat`); `syscalls-per-read-1026d.json` lists the sites. 22 came from two `Path.resolve()` calls that walked the directory chain of a file whose directories do not change: one in the Read's cache key (`imaging.py`), one in `localize.wanted`,
which resolved every Read's path although no auto-localize folder was named, so nothing could match. The other five were `is_file`/`stat` calls on the same file. What changed (`media.file_identity`, `media.read_identity`):

- `localize.wanted` returns before resolving when no auto-localize folder is named (11 calls).
- The cache key, the decode pool's key and the tile path's key ask `read_identity` for existence, size, modified time and real path together: an `lstat` of the file and a `stat` of its directory. The directory's real path is kept while its device, inode, modified time and changed time are the same, so a retargeted link, a replaced or renamed directory and a new file in the directory all resolve again.
  The file itself is never cached: its size and time are read for every evaluation, which is what makes a replaced file re-key the Read (tests: a file replaced during the session is read again without Reload; a directory link retargeted mid-session is followed; Reload still reads a file replaced with the same size and time; a symlinked frame, a relative or dotted path, a gap and a missing file take the old full lookup and agree with it).
- `NODEBASED_READ_LOOKUP=0` goes back to `Path.resolve()` and `Path.stat()` for every call.

| Radeon island, three interleaved passes, exclusive ms per fresh frame | `NODEBASED_READ_LOOKUP=0` | Now |
|---|---|---|
| Plate Read on the evaluating thread (includes the wait for the decode) | 9.7 to 10.4 | 7.0 to 9.2 |
| of it, file lookups by the 1 ms line sampler (two passes) | 3.2 | 0.9 to 1.1 |
| of it, waiting for the decode pool (same sampler) | 2.0 to 2.1 | 2.2 to 3.4 |
| Transform staging | 4.5 to 5.8 | 8.8 to 11.5 |
| Node stage | 46.2 to 47.8 | 49.1 to 51.4 |
| Frame rate / interval p95 | 16.91 to 16.93 fps / 62.2 to 64.0 ms | 16.95 to 17.10 fps / 64.1 to 81.7 ms |

The Read stage fell by about 2.3 ms and Transform staging grew by about 4.7 ms, as in step C, where the 7 ms given back by shorter lookups went to an inline decode: the evaluating thread, the decode thread and two matte workers all hold the interpreter lock, a shorter Read puts the walk at Transform staging while the matte workers draw, and the frame time follows the sum of their Python work. The Roto matte (44 ms for a 1080p matte on one core) is the largest piece of that sum;
a faster matte rasteriser is the next step, after Lane 2's RotoPaint work merges (`nodebased/shapes.py` and `nodebased/paint.py` were not touched here).

**GPU nodes as the default, with the new Automatic: gains and losses.**

- *Gains:* every machine on which one card passes the probe. This machine: 17.3 fps against 8.0 on the CPU nodes, a knob drag of 7 ms against 88 ms, a Roto point drag of 36 ms against 49 ms, a scrub of 50 ms against 100 ms. Four-CPU stand-in: 9.0 against 6.1 fps. The island's pixels agree with the CPU nodes on the RTX 3080 Ti, the Radeon and llvmpipe (`tests.test_islandadapter`: the probe chain on each adapter).
- *The loss step C found is gone:* the RTX 3080 Ti on its USB4 link (5.7 to 6.1 fps against 8.0) is not chosen when the Radeon is there, and the island does not open on it when it is the only card.
- *Losses that remain:* (1) a card named in Settings is used as named, so naming the RTX still costs the quarter (the Settings tooltip says Automatic measures; the note under the row shows the card in use). (2) The first Automatic open costs 1.7 s once per card and driver, and the CPU nodes run until it is done; a measurement that cannot run (playback, a busy card) waits. (3) The probe is two nodes at 1080p; a card that wins narrowly on it could lose on a longer chain, which the threshold of 1.0 (break-even near 1.6) is chosen to avoid, from two machines' worth of evidence. (4) A machine where Windows hides the discrete card from the application offers Automatic only the integrated one, as before.
- *CPU-node machines do not change:* with the setting off the code path is today's, and a machine where no card passes stays on the CPU nodes with a stated reason. The Read's change is a 1 to 2 percent gain there.
- *Unmeasured:* the RTX 4070 laptop, the A6000 and Windows (Gonzo owns them); the thresholds and the probe size are constants in `islandadapter.py`.
- *The default is still `DEFAULT_GPU_NODES = False`;* that is DiMo's call (asked 10/26 9:02 AM).

**Tests.** `tests/test_islandadapter.py` (30 tests): the fastest passing card wins whatever order the cards are listed in; a discrete card with a fast link beats the integrated one; ties go to the discrete card; a card slower than the CPU nodes, or none at all, keeps the CPU nodes with the reason; software adapters and a card listed again through OpenGL are not candidates; a new driver, a new probe version or the processor count measures again and nothing else does;
a named adapter is never measured; nothing is measured while a clip plays or while the real lock file is held by another descriptor, and a stored measurement is used during playback; a failed probe leaves the other card in charge and is not stored; the measurement runs under the lock; `NODEBASED_ISLAND_PROBE=0` keeps the old ranking; the stored measurements survive a new store object; `GPU2D.open("automatic")` reports a deferral and a refusal differently and carries the reason on the backend;
the probe chain gives the same pixels on every adapter this machine has (within 2e-5 of the CPU kernels). `tests/test_playbackprefs.py` (4 new): a deferred answer is not asked for again in a loop and is retried when playback stops, stopping playback asks again, a refusal reads "No graphics card was faster ...", the Settings row names the card and the reason.
`tests/test_readlookup.py` (26): identity equals `Path.resolve()` and `Path.stat()` for plain files, linked directories, linked files, retargeted links at two depths, relative and dotted paths, missing files and gaps; two calls when warm; the directory is resolved once for a played sequence and again when it changes; a replaced file and a retargeted link are read again; Reload still works; the tile path's key agrees; `localize.wanted` does not look the file up without a folder.
The modules above and `tests.test_gpu2d`, `tests.test_rotoahead`, `tests.test_decodepool` and `tests.test_playbackprefs` pass on the default adapter (RTX 3080 Ti), the integrated Radeon and llvmpipe (194 tests each); every GPU run took `/tmp/nb-gpu.lock`. The neighbouring Read, cache, time and tile modules pass on the processor.

**Needs Gonzo.** (1) The defaults, now that Automatic cannot send the island to the slower card on this machine: `DEFAULT_GPU_NODES` (the gains and losses above), `DEFAULT_DECODE_WORKERS`. (2) A faster matte rasteriser, after Lane 2's RotoPaint work merges: the frame time is the sum of the threads' Python work, and the matte is the largest piece. (3) The unmeasured machines.

Rerun: Automatic against the old choice is `python tools/benchmark_gpu2d.py playback --inputs "$NB_RUN/plates" --display :183 --backend gpu2d --island-adapter automatic --label auto --frame-range 48 --passes 3 --budget-mb 512 --no-sampler --out-dir "$NB_RUN/pb"` (`--island-adapter default` for the old choice, `--backend cpu` for the CPU nodes; set `NODEBASED_GPU_LOCK_HELD=1` and `NODEBASED_ISLAND_PROBE_FILE=$NB_RUN/island-probe.json` inside the lock);
the Read rows add `--env NODEBASED_READ_LOOKUP=0`; interaction is `python tools/benchmark_gpu2d.py interaction --inputs "$NB_RUN/plates" --source-workers 1 --backend gpu2d --adapter automatic --output i.json`; tests `python -m unittest tests.test_islandadapter tests.test_readlookup tests.test_playbackprefs`.

### R6-1: a faster Roto and RotoPaint matte drawer (10/27, step E of 5)

Step D left the Radeon island at 17.3 fps and named the Roto matte as what stood between it and 24 fps: about 44 ms for a 1080p matte on one core, drawn on workers that compete with the thread that evaluates the frame. This step replaces the drawer. Evidence is in
`benchmarks/2d-gpu/R6-1/` under the `-1027e` names: `matte-*` (the matte alone), `drawer-playback-*` (the realistic comp, old drawer against new), `stages-radeon8060s-legacy-*` and `-fast-*` (stage tables), `free-matte-*` and `decode-workers-*` (what is left once the matte is out of the way),
`environment-*` and `load-log-*`. Every timing row is three passes, rows interleaved; every pass of the comp ran under `/tmp/nb-gpu.lock` with no integrator suite running (`load-log-1027e.txt`: all 30 passes of the main table read `other_suites=0`; the stage, ceiling and decode rows have their own logs in `load-log-1027e-stages-ceiling-decode.txt`; a first set of the ceiling row and of the decode row, taken while or just after an integrator suite ran,
was discarded and repeated on a quiet machine).

**Result.**

- *The matte is 4 to 290 times faster and agrees with the old drawer to 1.2e-7.* A 1080p matte of the realistic comp's shape takes 4.1 to 4.9 ms against 23.3 ms; a 50-point curve with a box feather 10.2 to 10.9 ms against 108 to 109; the same with motion blur 21 ms against 446 to 447; an open stroke 27 ms against 3.7 seconds (4K: 75 ms against 21.7 seconds).
  The largest pixel difference to the old drawer is 6e-8 for fills (1.2e-7 for two shapes) and exactly 0 for strokes and the signed-distance feather; an 8-bit step is 3.9e-3.
- *The Radeon island plays 17.77 to 17.95 fps against 17.49 to 17.61 with the old drawer in the same session* (interval p95 65.9 to 68.4 ms against 63.4 to 75.8; node stage 43.2 to 46.3 ms against 47.6 to 51.2). The stage table: node stage 52.1 ms (50.7 to 54.2) becomes 47.4 ms (47.2 to 47.8), and the Roto work on the source workers falls from 52.5 ms per frame to 24.4 ms.
  Every other row moves a little in the same direction (table below).
- **The Radeon does not reach 24 fps.** With the matte out of the way altogether (a Roto that never changes, so the matte costs nothing after frame 1) the same comp plays 19.25 to 19.34 fps. Better matte drawing cannot close the rest of the gap, on the CPU or on the card, so the optional matte on the card was not built.
- *Not changed:* Roto's and RotoPaint's knobs, the document format, the cache keys (the digest of a Roto does not depend on the drawer; a test asserts it) and every other node. The old drawer stays behind a setting for one release.

**What changed** (`nodebased/rotofast.py`, dispatched from `nodebased/roto.py`).

- *Closed shapes.* The old drawer sorted every edge against every sub-scanline of a 16-row band, a dense ``(sub-scanlines x edges)`` problem for the whole height of the shape. The new one makes one crossing per (edge, sub-scanline) pair the edge really covers, sorts them once, takes the non-zero winding as one running sum
  and folds the eight sub-scanlines of a row into that row as they are accumulated. The sample positions (8 per row at `(k + 0.5) / 8`), the winding rule and the horizontal clip are the old drawer's, and the half-open test of which sub-scanlines an edge covers repeats the old comparison, so coverage agrees to float rounding.
  Tall shapes are accumulated in bands of at most 2 million cells so a 4K shape does not allocate hundreds of megabytes.
- *Open strokes.* Every segment was measured against every pixel of the patch (a 10-point stroke is about 150 segments). A segment now touches its own bounding box widened by the stroke's reach (half the width, the largest feather, and the half pixel), the same reach for every segment, so a pixel that can be lit still meets its nearest segment and the result is identical. The
  per-vertex arrays are built with array arithmetic.
- *Signed-distance feather.* The per-segment normals and offsets are one array operation per span instead of a Python loop, each segment's distance field is evaluated from two one-dimensional axes on its own box, and the loop only records which segment won each pixel; the feather radius is read once for every pixel from its winner. The tie between two segments that share a vertex is
  settled by the same float32 rounding the old drawer had, so the picture is unchanged.
- *Compositing.* The matte and the colour are written into one output array (no concatenation and no second copy of 33 MB at 1080p), the first shape on an empty canvas is a plain write, and a shape of full opacity is not scaled by one.
- *The interpreter lock.* Everything heavy is a large `numpy` call (ufuncs, `cumsum`, `argsort`, `copyto`), which runs without the lock. Measured with `tools/benchmark_roto_matte.py --lock-check` (`lock-check-1027e.json`): a pure-Python counting loop on the main thread runs at 92 to 99 percent of its solo speed beside a thread drawing the new matte over and over and at 95 percent beside one
  drawing the old matte, so the old drawer released the lock for almost all of its time too, and the lock was not what the matte cost the frame (the guess in step D). What differs is how long the drawing runs: 21 to 22 drawings fit in two seconds against 2. Four threads drawing the same shapes at once get the single-thread picture (test).
- *A shape with a non-finite point* keeps the old drawer's behaviour (the old kernels answer for it).
- *The setting.* Settings → Interface → *Roto drawer*: Fast (default) or Previous; `NODEBASED_ROTO_DRAWER=fast|legacy` wins over the saved choice and Settings says so under the row; `roto.set_drawer("legacy")` is the call. A change takes effect at once and drops the cached mattes of the other drawer.
  `DEFAULT_ROTO_DRAWER` in `nodebased/playbackprefs.py` is the one-line default. The previous drawer is `roto.coverage_legacy` and the unchanged bodies of `stroke_coverage` and `edge_feather`; remove it after one release.

**The matte alone** (`tools/benchmark_roto_matte.py`, `roto.rasterise` on resolved shapes, no evaluator and no window; median draw time of seven drawings per pass, range over three interleaved passes; 32 CPUs idle; `matte-1027e.json`).

| 1080p | Previous drawer ms | Fast drawer ms | Faster by | Largest pixel difference |
|---|---|---|---|---|
| 6-point polygon, no feather | 22.0 to 23.3 | 2.0 to 2.2 | 10.8x | 6.0e-08 |
| 6-point polygon, feather 18 (box): the realistic comp's shape | 23.3 to 23.6 | 4.1 to 4.9 | 5.1x | 6.0e-08 |
| 50-point curve, feather 18 (box) | 107.9 to 109.2 | 10.2 to 10.9 | 10.3x | 6.0e-08 |
| 50-point curve, feather 18, motion blur (4 samples, moving) | 445.7 to 447.2 | 20.9 to 21.2 | 21.2x | 6.0e-08 |
| 6-point polygon, feather 18, motion blur | 52.3 to 54.7 | 7.9 to 8.0 | 6.7x | 6.0e-08 |
| 50-point curve, signed-distance feather 18, per-point offsets | 148.6 to 150.4 | 31.7 to 34.4 | 4.6x | 0 |
| open 10-point stroke, width 30, feather 8 | 3722.8 to 3765.9 | 27.2 to 27.4 | 137x | 0 |
| polygon with feather 10 and a subtracted 50-point blob | 117.1 to 117.3 | 14.2 to 14.3 | 8.2x | 1.2e-07 |

| 4K | Previous drawer ms | Fast drawer ms | Faster by | Largest pixel difference |
|---|---|---|---|---|
| 6-point polygon, no feather | 77.0 to 78.0 | 6.4 to 6.5 | 12.0x | 6.0e-08 |
| 6-point polygon, feather 18 (box) | 91.0 to 93.0 | 15.0 | 6.1x | 6.0e-08 |
| 50-point curve, feather 18 (box) | 422.3 to 426.6 | 45.6 to 45.8 | 9.3x | 6.0e-08 |
| 50-point curve, feather 18, motion blur (4 samples, moving) | 1710.2 to 1711.7 | 91.4 to 91.5 | 18.7x | 6.0e-08 |
| 6-point polygon, feather 18, motion blur | 184.4 to 187.6 | 28.5 to 28.6 | 6.5x | 6.0e-08 |
| 50-point curve, signed-distance feather 18, per-point offsets | 568.4 to 573.7 | 131.8 to 132.3 | 4.3x | 0 |
| open 10-point stroke, width 30, feather 8 | 21745.8 to 21782.4 | 74.4 to 75.3 | 290x | 0 |
| polygon with feather 10 and a subtracted 50-point blob | 457.3 to 505.1 | 70.2 to 77.6 | 6.6x | 6.0e-08 |

The signed-distance feather is the slowest fast case: it measures every pixel near the outline against each of the outline's roughly 1000 segments and does that in float64 to keep the old drawer's tie-breaks, so it gained 4 to 5 times against 10 to 290 times elsewhere.
Inside the played comp the Roto node takes 24 ms on the source worker (the drawer is 4 ms of it; the node alone on an idle core takes 7.7 ms, the rest is the digest, the feather and the other threads' competition for the interpreter lock and memory).

**The realistic comp, old drawer against new** (`playback --no-sampler`, 48 frames, first lap, uncached, 1080p; fps / interval p95 ms / node stage p50 ms; range over three interleaved rounds; `NODEBASED_ROTO_DRAWER=legacy` against `fast`; `drawer-playback-1027e.json`).

| | Previous drawer | Fast drawer | Step D (10/26, old drawer) |
|---|---|---|---|
| Radeon 8060S island (Automatic) | 17.49 to 17.61 / 63.4 to 75.8 / 47.6 to 51.2 | 17.77 to 17.95 / 65.9 to 68.4 / 43.2 to 46.3 | 17.26 to 17.37 / 64.5 to 66.7 / 47.7 to 50.4 |
| RTX 3080 Ti island (named; Automatic does not choose it here) | 5.83 to 5.87 / 225.2 to 238.9 / 184.4 to 188.1 | 6.02 to 6.22 / 222.8 to 260.8 / 153.1 to 177.2 | 5.69 to 6.11 / 218.0 to 256.2 / 179.0 to 191.6 |
| Four-CPU stand-in, island (Automatic: the Radeon) | 9.07 to 9.19 / 119.8 to 122.8 / 105.1 to 106.4 | 9.41 to 9.47 / 114.8 to 120.9 / 101.7 to 103.4 | 9.00 to 9.04 / 118.5 to 119.5 / 105.8 to 107.4 |
| CPU nodes | 8.13 to 8.21 / 132.7 to 134.6 / 117.3 to 119.6 | 8.20 to 8.31 / 131.0 to 134.3 / 116.5 to 116.6 | 7.95 to 8.04 / 137.4 to 139.7 / 119.8 to 122.1 |
| Four-CPU stand-in, CPU nodes | 6.06 to 6.12 / 177.8 to 180.9 / 160.2 to 162.4 | 6.19 to 6.21 / 176.3 to 184.3 / 157.2 to 160.0 | 6.06 to 6.08 / 178.9 to 183.8 / 161.3 to 161.8 |

The gain is 1 to 5 percent on every row (Radeon island +0.3 fps, RTX +0.3, stand-in island +0.3, CPU nodes +0.1, stand-in CPU nodes +0.1). Neither drawer passes the interval p95 gate of 62.5 ms on all passes (the Radeon island sits at 64 to 68 ms in both). The matte had already been taken off the critical path by the read-ahead of step B; what the faster drawer removes is
the competition it caused, which shows in the next table.

**Where the Radeon frame goes now** (`playback --stages --no-sampler`, three interleaved rounds each; mean exclusive ms per fresh frame on the evaluating thread, range over the rounds; `stages-radeon8060s-*-1027e.json`).

| Stage on the evaluating thread | Previous drawer | Fast drawer |
|---|---|---|
| Plate Read (includes the wait for the decode pool) | 9.4 (8.1 to 11.7) | 14.3 (12.0 to 15.6) |
| ColorCorrect on the CPU kernels | 8.4 (7.5 to 9.5) | 6.9 (6.8 to 7.1) |
| Transform staging | 9.0 (7.0 to 10.4) | 5.1 (4.1 to 6.9) |
| Island upload | 5.8 (5.6 to 6.0) | 5.6 (5.4 to 5.8) |
| Island kernels on the card | 5.2 (4.9 to 5.5) | 4.8 (4.6 to 4.9) |
| Island download (copy and map) | 3.3 | 3.0 |
| Island host work and encode | 2.1 | 1.1 |
| Wait at the Roto join | 1.8 (1.2 to 2.8) | 0.6 (0.5 to 0.7) |
| *Node stage* | 52.1 (50.7 to 54.2) | 47.4 (47.2 to 47.8) |
| *Other threads, busy ms per frame:* decode pool | 48.2 | 45.3 |
| *Roto on the source workers (drawn ahead and drawn for the frame in hand)* | 33.7 + 18.8 | 24.4 + 0 |

The Roto work on the source workers halves and the Transform staging and the join lose 5 ms between them. The plate Read stage grows by 5 ms for the reason step C found: the walk now arrives at the Read sooner and waits for the decode pool, which is busy 45 ms of every 55 ms interval (two 22 ms EXR decodes per frame on one thread).
The frame is paced by decoding and by the evaluating thread's own chain (Read, Transform staging, upload, kernels, download, ColorCorrect: 47 ms); 24 fps needs 41.7 ms.

**Why 24 fps is out of reach for the matte alone.**

- A comp with a static Roto, three interleaved passes beside the fast-drawer rows (Radeon island, the same session): 19.25 to 19.34 fps (interval p50 42.0 ms, p95 53.1 to 53.4 ms) against 17.58 to 17.70 with the animated Roto and the fast drawer (`free-matte-ceiling-1027e.json`). A matte costing nothing gains 1.7 fps, so a matte on the card, which could at best remove that cost, cannot reach 24 fps.
  It also needs the shape tables and the feather on the card, a second implementation to keep within tolerance of the first, and a CPU fallback; the measurement says it is not worth building for the frame rate.
- Two decode workers with the fast drawer (Radeon island, three interleaved rounds, `decode-workers-1027e.json`): 17.59 to 18.06 fps and interval p95 63.4 to 66.3 ms against 17.50 to 17.81 fps and 65.1 to 68.7 ms with one. The second decode worker adds about 0.2 fps, as it did in step C.
- What stands between the Radeon and 24 fps is the chain the evaluating thread runs for every frame: the plate decode (22 ms, twice per frame), the host-side staging and the 74 MB upload (5.6 ms), the CPU ColorCorrect (6.9 ms) and the download (3.0 ms). Moving ColorCorrect into the island and keeping the finished frame on the card for the display transform removes the last two,
  and half-float uploads halve the first; those are the next steps, none of them the matte.

**Tests.** `tests/test_rotofast.py` (35 tests, 1.3 s): the coverage kernel against the old one over random polygons (self-intersecting included), random curves with handles, holes, clipped, empty, degenerate, single-pixel shapes, vertices on sub-scanline boundaries and axis-aligned boxes with analytic values (quarter pixel in, half a row in); whole mattes under both drawers for plain and curved shapes, all three
falloffs and the falloff power, signed-distance feather with per-point offsets, inverted shapes and an inverted node, every blend mode with opacity and colour, subtract before and after painting, open strokes with every cap pair and feather, motion blur with a shutter, the legacy three-sample blur and the global blur, shape transforms and extra matrices (all within 1e-6); regions equal the full frame with each drawer
(six regions including one-pixel strips and corners); the tile executor equals the evaluator with each drawer; the cache digest of a Roto is the same under both drawers and a second frame with the same shapes is a cache hit; a set cancel stops the Roto before any matte is drawn or stored and leaves the evaluator working; a shape with only the original fields, the realistic comp's own Roto at frames 1, 12 and 48 at 1920 x 1080, and RotoPaint shape items (whole frame and a region) draw the same; four threads drawing at once give identical pictures; the
band size changes nothing; a point that is not a number behaves as before; each drawer never enters the other's kernels. `tests/test_playbackprefs.py` (4 new, 32 in all): the default and the environment, the saved choice round-trips, the Settings row, a running window switches and drops the old mattes.
Lane 2's step F and H pixel tests (`tests.test_2d_fidelity_f_roto`, `tests.test_2d_fidelity_h_rotopaint`, unchanged) and the Roto, RotoPaint, tile, planar-tracking, golden, cache, knowledge and benchmark-tool modules pass (485 tests with the two new ones); the modules that open the Settings dialog pass (324 tests). The drawer changes no shader and no buffer packing, so the card test modules were not rerun; the played passes ran on the Radeon and the RTX.

**Left out.** The RTX 4070 laptop, the A6000 and Windows rows stay unverified (Gonzo owns them). The 240-frame rows were not repeated. The signed-distance feather is the slowest of the new kernels (4 to 5 times faster than before) and a candidate for a float32 pass if a tolerance beyond 1e-6 is acceptable. The legacy drawer is to be removed after one release.

**Needs Gonzo.** (1) The defaults from the earlier steps are unchanged: `DEFAULT_GPU_NODES`, `DEFAULT_DECODE_WORKERS`. (2) The next 24 fps work is the ColorCorrect kernel in the island with the frame kept on the card, half-float uploads and the decode pipeline (above). (3) Remove `coverage_legacy` and the setting after one release. (4) The unmeasured machines.

Rerun: the matte alone is `python tools/benchmark_roto_matte.py --output matte.json` (`--sizes 1080p --cases simple feathered50 --passes 3 --reps 15`); the comp rows are the step D command with `--env NODEBASED_ROTO_DRAWER=legacy` or `=fast` (and `NODEBASED_GPU_LOCK_HELD=1` under the lock); the stage tables add `--stages`; tests `python -m unittest tests.test_rotofast tests.test_playbackprefs tests.test_2d_fidelity_f_roto tests.test_2d_fidelity_h_rotopaint`.

## Lane 2 step notes

### Playback pipeline: the next frame evaluates while this one is converted and shown (10/22)

Before, one worker evaluated a played frame, built its 8-bit picture, handed it to the Qt thread and only then
started the next frame, so a frame cost the sum of those. Now the worker puts the finished float frame on a
conversion stage (`playback.ConversionStage`: one thread, with a display context of its own because the ACES
transform runs on a thread-bound OpenGL context) and starts the next evaluation as soon as the stage has taken the
frame. The stage keeps one frame waiting; the evaluator waits for that slot rather than racing ahead, so a cheap
frame (a display-cache hit) is never evaluated only to be replaced. A frame that is replaced or discarded (playback
stopped, an edit, a seek) is counted and added to the `dropped` figure in the time row; pictures are shown in
request order and a request is shown at most once. After a stop, the picture catches up with the playhead, which
runs one frame ahead of the picture while a frame converts.

With the Direct picture on, the worker already hands the window the float frame and the card does the view
transform, so no 8-bit picture is built (counted: zero `to_qimage` calls in the 10-second Direct runs below). That
path is unchanged. Letting the worker start the next frame before the Direct picture's draw on the Qt thread
segfaulted the Xvfb benchmark (cause not found; the draw would then overlap the worker's graphics-card work on a
display with no DRI3), so that overlap is not shipped.

`Window.pipeline_playback = False` restores the one-thread order; the benchmark option `--no-pipeline` does the same.

**Where a fresh frame's time goes.** `tools/benchmark_2d_playback.py` records a mark per request at each stage
(`nodebased/playback_trace.py`), uncached realistic comp at 1080p, disk cache off, a picture counted only after the
Viewer accepts its request generation. Mean over the 48 fresh frames, RTX 3080 Ti (two runs each agree within 1 ms
per stage; shown: run 1), milliseconds:

| Stage | One thread (before) | Pipelined (after) |
|---|---:|---:|
| Request to worker, and setup | 0.6 | 0.6 |
| Node evaluation | 137.0 | 150.1 |
| Waiting for conversion (cache put, stage pickup) | 0.1 | 0.4 |
| Display conversion (`to_qimage`) | 40.4 | 55.2 |
| Hand-off to the Qt queue and delivery | 0.4 | 0.5 |
| Viewer present | 2.7 | 3.9 |
| **One frame, request to shown** | **181.1** | **210.7** |
| Transport wait before the request | 2.5 | 0.0 |
| Overlap with the previous picture | 0.0 | 56.9 |
| **Frame interval = frame - overlap + wait** | **183.6** | **153.8** |
| Interval measured separately (benchmark's Viewer-accept stamps, first pass) | 183.9 | 157.3 |

The parts add up to the measured interval within 0.05 to 3.2 percent in all twelve runs (`accounting` in each
timeline file). One frame now takes longer from request to picture (conversion and evaluation share memory
bandwidth and the interpreter, which cost evaluation 13 ms and conversion 15 ms), and the frame interval is
shorter because 57 ms of each frame's tail overlaps the next evaluation. Node evaluation is now 98 percent of the
interval: the remaining 24 fps gap is the node and read cost, which Lane 8 owns.

**Gates, uncached realistic comp, 1080p, 10-second runs, first pass through the 48 fresh frames.**

| Adapter | One thread | Pipelined | Gain | Handoff p50 (built picture to Viewer accepted) | Whole run, 10 s |
|---|---:|---:|---:|---:|---:|
| RTX 3080 Ti (two runs) | 5.32 and 5.34 fps | 6.50 and 6.48 fps | +22 percent | 4.3 ms | 6.9 to 10.2 fps |
| Radeon 8060S (`integrated`) | 5.52 fps | 6.43 fps | +17 percent | 4.3 ms | 7.5 to 9.9 fps |
| llvmpipe (`cpu`) | 5.55 fps | 6.60 fps | +19 percent | 3.3 ms | 7.7 to 11.0 fps |
| Laptop stand-in (`integrated`, four CPUs, 2x scale) | 3.86 fps | 4.18 fps | +8 percent | 3.3 ms | 3.8 to 4.1 fps |
| RTX 3080 Ti, Direct picture on Xvfb | 4.75 fps | 4.82 fps | none | 2.8 ms | 4.6 to 4.8 fps |

- Gate met on the RTX: the frame interval is no longer the sum of evaluation and display (154 ms against 205 ms for
  evaluation plus conversion), the hand-off is 4.3 ms median (limit 10 ms), and the fresh rate rose 22 percent.
- Not met: 24 fps. Evaluation alone takes 137 to 150 ms a frame, so one frame is bounded near 6.5 to 7 fps whatever
  the display does. The conversion no longer costs anything on the interval; what remains is Lane 8's.
- The laptop stand-in gains little: four CPUs run evaluation (232 ms) and the CPU conversion (69 ms) at once.
  It stays under its 12 fps target.
- The Direct picture on Xvfb is slower than the Viewer path because presenting to Xvfb copies each frame on the
  Qt thread (about 21 ms, see "Direct viewer display transform" above), and the pipeline has nothing to overlap there. A real screen is not
  measured.
- Estimate for the 48 GB A6000 workstation (not measured; it is unavailable): playback there is bounded the same
  way, so with evaluation near 140 to 150 ms on a CPU like this one, expect about 6 to 7 fps fresh, with conversion
  hidden. A faster CPU raises it in proportion; the card makes no difference to this comp's nodes.
- One conversion-stage drop is counted in each pipelined run: the frame waiting when playback stopped.

Evidence: `benchmarks/2d-playback/pipeline-1022-{serial,pipe}-{rtx-1,rtx-2,radeon,llvmpipe,laptop,direct-rtx}.timeline.json`
(per-frame marks, stage means, overlap and the accounting check); regenerate one from a raw benchmark result with
`tools/timeline_2d_playback.py`. The runs used private Qt settings (earlier runs wrote the real settings file and left
the Direct picture dock open for the next run, which changed what was measured).

## Lane 4 step notes

### 3D playback profile and fixes (10/22)

Measured with `tools/benchmark_3d_playback.py` (the real window on a private Xvfb display, 1920 x 1080, 24 fps comp, 10 s
of playback, camera animated) and `tools/profile_3d_playback.py` (stage timeline plus sampling profile of the Qt thread
and the worker). Results are under `benchmarks/3d-playback/` (`before/<adapter>/`, `after/<adapter>/`, `repeat/`,
`profile-1022-<scene>.{json,txt}` and `-after`). The full suite ran on the card during some runs; the benchmark flags
that in `gpu_wait_ms.shared_card_suspected`.

Where the time went, RTX 3080 Ti, per frame (median, milliseconds):

| Scene | Top five costs before | Bound by |
|---|---|---|
| Cube (frame 514) | resolve at 3840 x 2160 then down to 1080p 328; convert and emit 46; GPU render and wait 42; render encode 41; readback copy 38 | Samples: playback drew, read back and averaged the node's 2 samples at 3840 x 2160 for a 1080p comp |
| Furnished room (frame 1163) | GPU render and wait 415; resolve 320; scene prepare 209; render encode 78; convert and emit 49 | The card, then Python scene rebuild (shadow data and vertex packing every frame) |
| Cube after (frame 89) | convert and emit 27; display cache put 17; GPU render and wait 12; node bookkeeping 9; scene evaluation 3 | Display conversion on the worker |
| Furnished room after (frame 237) | GPU render and wait 122; convert and emit 47; render encode 19; node bookkeeping 15; scene upload 11 | The card |

The CPU rasteriser was chosen when a Render3D node's backend was set to CPU, even with a card present. The empty scene
(only the camera animating) re-rendered every frame although it draws nothing.

Fixes, each with a test in `tests/test_3d_playback_cost.py` that fails without it:

- Playing requests draw at one sample; stopping re-renders at the node's own samples (resolve 328 to 0.4 ms).
- A Render3D set to the CPU backend plays on the GPU when a card exists.
- An empty scene in front of a Render3D makes the picture static whatever the camera does.
- The readback staging buffer is kept between renders and copied once; no disk spill for frames evicted while playing.
- The packed shadow triangles and their BVH are reused while only the camera moves; vertex packing runs in NumPy
  (137 to 4 ms); same-material triangles are one draw.
- The display conversion touches only the covered rectangle.
- A landed played frame asks for the next one at once instead of waiting for the next tick.

Frames per second, before and after (the last column is the comp's rate):

| Case | RTX 3080 Ti | Radeon 8060S | llvmpipe | Laptop stand-in | Gate |
|---|---:|---:|---:|---:|---|
| Empty scene, animated camera | 9.1 to 24.0 | 10.4 to 24.0 | 9.6 to 23.9 | 9.3 to 24.0 | 24 |
| Cube, animated camera | 2.0 to 12.4 | 4.7 to 15.5 | 3.9 to 14.4 | 4.2 to 13.4 | 12 on the RTX |
| Furnished room, animated camera | 1.0 to 4.6 | 1.4 to 4.9 | not run | 1.4 to 4.5 | none |
| Render3D of the cube into a Grade | 1.9 to 9.3 | 4.0 to 11.5 | not run | not run | none |
| Mixed mesh and splat scene | 0.4 to 1.0 | not run | not run | not run | none |
| One Render3D frame of the cube (default 2 samples) | 411 to 88 ms | | | | under 150 ms |

The laptop stand-in is `run/force-adapter.py integrated` with `taskset -c 0-3` and `QT_SCALE_FACTOR=2` on this
workstation's Radeon; it is not the Windows laptop. The cube result of 12.4 fps sits 0.4 above the gate and measured
10.7 to 12.4 fps over three runs, 14.3 in a 2 s run, with the suite sharing the card.

What the A6000 run should show: run `python tools/benchmark_3d_playback.py --scene empty,cube,furnished,mixed,grade`
on the machine and read `adapter`, `fps`, `interval_ms` (p50, p95), `stage_p50_ms` and `gpu_wait_ms` from each JSON, plus a separate
`--once` run for `render_once_ms`. If the card is the limit the furnished room's `GPU render and wait` falls; if it stays
near the figures above, the Python stages (convert and emit, node bookkeeping) are the bound. No number is claimed here.

Costs not reached, ranked: (1) convert and emit, 27 to 47 ms a frame (float to display picture on the worker);
(2) display cache put, 17 ms on the cube; (3) the furnished room's 122 ms GPU render and 19 ms render encode; (4) node
bookkeeping after the render, 9 to 15 ms; (5) the mixed mesh and splat scene at 1.0 fps, not profiled; (6) readback
overlapping the next render (double-buffered staging), which the 2 ms readback no longer justifies.

### 3D playback at the comp's rate (10/22, step B)

Measured with `tools/benchmark_3d_playback.py`, 1920 x 1080, 24 fps comp, 8 s of playback in the real window, camera
animated. Results are under `benchmarks/3d-playback/final/<adapter>-proxy-on|off/`. "Proxy on" is the app as shipped
("Proxy while playing" checked: a frame that misses the comp's rate steps the picture down one resolution tier and the
automatic tier climbs back after a run of frames with room to spare); "off" plays every picture at full size, the raw cost.
The full test suite did not share the card during these runs, but other lanes' GPU jobs queue on the same lock.

What changed since the profile above:

- Shadow rays go through the BVH from 300 triangles; a scene whose camera moves keeps its uploaded buffers, bind groups and
  draw lists (furnished room render call 107 to 17 ms).
- An opaque frame converts for display without the divide and multiply by alpha; the display card makes the 8-bit
  picture and reads back 6 MB instead of 25 MB of float colour.
- A full display cache rounds old frames on a background thread; one ray buffer serves all bands; shape nodes remember their
  geometry; splat clouds remember their transformed copy; meshes with splats play through the raster shader and a splat layer.
- The automatic proxy tier ignores the cold first frames of a play and of each new tier. Stopping draws the frame at the
  node's own settings, and the benchmark times it (`settle_s`).

| Case (frames per second shown) | RTX 3080 Ti, proxy on | RTX 3080 Ti, proxy off | Radeon 8060S, proxy on | llvmpipe, proxy on | Laptop stand-in, proxy on | Gate |
|---|---:|---:|---:|---:|---:|---|
| Cube, animated camera | 24.1 | 23.9 | 24.0 | 23.9 | 24.0 | 24 (RTX), 24 (laptop) |
| Furnished room | 24.1 | 21.4 | 24.1 | not run | 24.1 | 24 (RTX), 12 (laptop) |
| Mixed mesh and splat scene | 23.9 | 7.3 | 24.0 | not run | not run | 24 at a reduced tier |

Tier mix with proxy on (pictures at full, half, quarter, eighth size): cube on the RTX 152, 41, 0, 0; furnished room 180,
13, 0, 0; mixed scene 3, 8, 181, 0. After stopping, the mixed scene's full picture is on screen in 2.3 s on the RTX (gate
3 s) and 0.8 s on the Radeon. The time row read "real time" in every proxy-on run. The furnished room at full resolution
and the mixed scene at full resolution do not hold 24 fps on their own (21.4 and 7.3 fps): the shipped app reaches the
rate by dropping the picture a tier for part of the run, and the tier shown is named in the time row. The cube on the
RTX steps to half size for about one picture in five although it runs at 23.9 fps at full size, because the automatic tier
reads anything under 24 as a miss (fixed in step A, "3D playback at full size" below).

The laptop stand-in is the Radeon with `taskset -c 0-3` and `QT_SCALE_FACTOR=2`; it is not the Windows laptop with the
Iris Xe and the RTX 4070. The A6000 and the laptop itself are not measured here.

Where the furnished room's full-size time goes now (RTX, per frame, milliseconds): GPU render and wait 14.5; display
picture 20 (the ACES 2.0 shader alone takes 6.9 ms on this card, 10 ms with upload and readback when the card is idle, and
about 20 ms when it shares the card with the ray render). That conversion ran on Mesa's software OpenGL, not on the card
(see step A below, which removes it).

### 3D playback at full size (10/23, step A)

Gate: the cube and the furnished room play at 24 fps at full size (1920 x 1080, animated camera, "Proxy while playing" off) on the
RTX 3080 Ti. Measured with `tools/benchmark_3d_playback.py --stages` on the private Xvfb display, 9 s of playback, 24 fps comp; results are
under `benchmarks/3d-playback/full-size-1023/<adapter>-proxy-off|on/` (the `rtx-proxy-off-card-picture-off` row is the same build with
`NODEBASED_CARD_PICTURE=0`, the old conversion).

What the 10/22 numbers were measuring. The "display GPU" of these runs is not the card: Xvfb has no hardware OpenGL, so the display
conversion ran on Mesa llvmpipe, the processor (the ACES 2.0 shader alone took 5.7 ms there, 12.6 ms for the whole conversion of a
frame). The conversion already ran beside the next evaluation (Lane 2's pipeline serves 3D playback), but both then fought for the
processor: with the old conversion the furnished room's evaluation took 23.7 ms, its conversion 26.5 ms and a picture's latency from
request to screen 61 ms, over the 41.7 ms frame time. The automatic tier reads that latency, which is what made the cube step down.

| RTX 3080 Ti, per frame, ms (mean) | Furnished, old conversion | Furnished, card picture | Cube, old | Cube, card picture |
|---|---:|---:|---:|---:|
| Node evaluation (includes the card's picture pass) | 23.7 | 20.9 | 16.1 | 18.4 |
| Display conversion | 26.5 | 0.5 | 21.8 | 0.6 |
| Request to screen | 61.0 | 27.6 | 46.5 | 25.9 |
| Overlap with the previous picture | 20.0 | 0.1 | 6.3 | 0.9 |
| Interval p95 | 59.9 | 43.7 | 51.8 | 44.5 |

Part 1, overlap. The pipeline already overlaps conversion with the next evaluation (rows above, "overlap"). What it could not hide was a
conversion that cost as much as the evaluation and competed for the same processor. The conversion is gone from the stage instead:

- `nodebased/gpupicture.py`: the render card makes the 8-bit picture of a played frame in the frame's own command submission, one
  more pass after the render that reads the frame texture and writes the picture (unpremultiply, the view's OCIO shader, associate again,
  quantise: `to_qimage`'s order). It is read back through the same buffer as the float frame (8 MB beside 33 MB). The float frame is
  untouched. `gpu3d.render` registers the picture against the array it returns; `to_qimage` asks the registry first, by identity, and
  builds the picture itself when nothing matches. The window arms it for played frames whose display is the plain case (exposure 0, RGB,
  gamma 1, no checker or clipping warning, no compare buffer). Not made: a frame with a background colour or supersampling, a visible
  environment or area light (something is drawn over the card's frame), a software adapter, a card that cannot build the
  shader, or a scene with splats on the old route (the splat layer is now drawn on the card, see the mixed scene at full size below); the
  old conversion runs for those. `NODEBASED_CARD_PICTURE=0` turns it off.
- Agreement with the processor: at most 1 of 255 on every channel, on the RTX 3080 Ti, the Radeon 8060S and llvmpipe (`tests/test_3d_card_picture.py`),
  and 0 or 1 level between the picture on screen and the processor's picture of the same float frame in every benchmark run
  (`picture_check` in the JSON).
- A scene with splats got 5 times slower in its splat layer (mixed scene 7.3 to 2.8 fps) when the picture's 8 MB of readback
  memory existed beside it. Found on 10/23 (mixed scene lane): a MAP_READ buffer of about 24 MB or more kept alive makes every other
  readback on the RTX 3080 Ti, each created and destroyed per call, take about 3 times as long (see the mixed scene at full size below).
  At the time such frames were given no picture; that has changed.

Part 2, ACES 2.0 on its own (`tools/benchmark_card_picture.py`, the card's share from submit to the picture mapped, 1920 x 1080,
the float frame already on the card; `benchmarks/3d-playback/full-size-1023/card-picture-pass.jsonl`): RTX 3080 Ti 3.4 ms with ACES 2.0,
3.0 with sRGB and 3.5 with no transform (run-to-run noise about 0.5 ms; the 8 MB picture's trip over this machine's USB4 link is nearly all of
it, so the shader is under half a millisecond); Radeon 8060S 0.38, 0.34 and 0.34 ms; llvmpipe 7.2, 2.1 and 1.5 ms (5.7 ms for ACES 2.0 on the
processor, which is why a software adapter keeps the old path). Copying the picture out of the mapped buffer adds 0.2 to 0.6 ms on the worker.
A baked 3D LUT or a shorter transform would save under half a millisecond on the RTX and nothing measurable on the Radeon, and would put
the agreement with the processor (1/255, `tests/test_carddisplay.py`) at risk on saturated colour, so the shader is unchanged. The gain
was moving the conversion to the card, not making it cheaper.

Part 3, no flapping. The automatic tier steps down when the middle picture of the last 12 took longer than the frame time (6 pictures at
least), or at once when 3 pictures in a row took over twice the frame time; one slow picture no longer does. A climb back that is
undone doubles the run of pictures with room to spare the next climb needs (12, 24, ... 96). Tests: a cube-like series with one
picture in five over the frame time stays at full size through 240 pictures (10 s); a severe series drops after 3 pictures.
`tests/test_3d_playback_rate.py` and `tests/test_desktop.py` were changed for the new rule (their old form dropped a tier on one
picture).

Part 4, the gates (frames per second shown; interval p95 in ms in brackets; every row 1920 x 1080):

| Case | RTX proxy off | RTX proxy on | Radeon off / on | Laptop stand-in off / on | llvmpipe off / on | Gate |
|---|---:|---:|---:|---:|---:|---|
| Cube | 23.98 (44) | 23.87 (44), all pictures full size | 24.02 / 24.06 | 23.97 / 24.05 | 23.99 / 23.94 | 24 at full size |
| Furnished room | 23.95 (44) | 23.96 (44), all full size | 24.07 / 24.02 | 24.01 / 23.98 | 2.97 / 18.5 (mostly 1/8) | 24 at full size |
| Mixed mesh and splat scene | 7.77 (135) | 23.93 at 1/4 size (5 full, 7 half, 205 quarter) | 10.42 / 23.99 | 9.83 / 23.93 | not run / 9.23 | recorded; reduced tier refines in under 3 s |

After stopping, the mixed scene's full picture is on screen in 2.24 s on the RTX (1.92 s with proxy off), 0.87 s on the Radeon and 0.78 s on the laptop
stand-in (gate 3 s). The furnished room on llvmpipe did not refine within the benchmark's 30 s (a software full-quality frame is
slower than that); its rate is a software figure and no gate. The old conversion on the same day, same card: cube 23.89 (p95 51.8), furnished 23.86 (p95 59.9) at the
comp's rate, 36.3 and 29.3 fps at a 60 fps comp; with the card picture 39.9 and 39.7 fps. At the comp's rate both builds show about 24 fps when
the card is not shared; the difference is the headroom (1.10 times for the cube, 1.35 for the furnished room) and the jitter, and on 10/22 a run with the card shared showed 21.4 fps for the old
build. The laptop stand-in is the Radeon with `taskset -c 0-3` and `QT_SCALE_FACTOR=2` on this workstation, not the Windows laptop with the Iris Xe and the RTX 4070.

The A6000 (estimate, not measured): the cube's 11 ms of "GPU render and wait" on this machine matches the 33 MB float readback over the
USB4 link (about 2.7 GB/s, 12 ms) more than any drawing (an inference from the link speed; the two were not timed apart). On a card in a PCIe 4.0 x16 slot that readback is a tenth of that, so the furnished room's render plus
picture pass should take about 7 to 8 ms and the cube's about 3 ms, with the presentation on the Qt thread the bound (the 60 fps headroom rows
stop near 40 fps on Xvfb, where presenting a picture costs about 20 ms of the Qt thread). Run
`python tools/benchmark_3d_playback.py --scene cube,furnished,mixed --stages` there and read `fps`, `interval_ms`, `frame_timeline` and `picture_check`.

Not done at the time: the mixed scene at full size (7.8 fps on the RTX) was bound by the splat layer, which was not on the card's
picture path; that is the next section. The Windows laptop is not measured.

### Mixed mesh and splat scene at full size (10/23, step A of the mixed lane)

Gate: the mixed scene (the furnished room, a 20,000-splat relit shell, one shadowing point light; `tools/benchmark_3d_playback.py --scene
mixed`) plays at 24 fps at 1920 x 1080 under the animated camera with "Proxy while playing" off on the RTX 3080 Ti, shows its full node
settings within 3 s of the stop, and the cube and the furnished room stay at 24 fps. Results: `benchmarks/3d-playback/mixed-1023/<adapter>-proxy-off|on/`,
the timeline in `benchmarks/3d-playback/mixed-1023/stages-rtx.json` (`tools/profile_3d_mixed_stages.py`, both routes). The benchmark's
mixed scene has no smoke; a volume together with splats is CPU-only (`gpu3d.render` raises Unsupported), so the timeline has no smoke row.

Part 1, where a 89 ms frame went (RTX 3080 Ti, 1080p, 1 sample, medians over 30 frames, processor route = `gpu3d.SPLATS_ON_CARD = False`):

| Stage, ms | Processor route | Card route |
|---|---:|---:|
| Mesh render (colour pass, plus a second depth pass for the splats' occlusion) with readback | 30.9 | 21.1 for the whole call |
| Splat sort (back to front, on the processor, 20,000 splats) | 1.1 | 0.9 |
| Splat colours (relit on the processor) | 3.2 | 3.0 |
| Splat layer: depth texture upload (8 MB), projection, raster, 33 MB readback | 34.9 (includes the sort and colours) | drawn in the mesh frame's submission |
| Compositing the splat layer over the meshes (NumPy on the processor) and the rest | 23.5 | 0 |
| Readback of the finished float frame | included above | 1.8 |
| Display picture (conversion stage) | 57.9 | 0.6 (made by the card) |
| Whole render call / frame as the pipeline sees it | 89.3 / 147 | 21.4 / 22 |

The splat sort was never the cost (1 ms at 20,000 splats); the layer left the card as a 33 MB float picture of its own, the mesh depth went
up to the card again as an 8 MB texture, the composite ran on the processor and the processor then converted the frame to 8 bits.

Part 2, the layer on the card's picture path. `gpusplat.prepare_on_card` projects the splats in a compute pass and draws them over the mesh
pass's colour target, back to front with premultiplied over blending (over is associative, so blending onto the meshes is the layer
composited over them). The splat shader reads the mesh pass's own depth buffer (a `depth32float` texture also bound for sampling) and
turns it back into the camera-forward distance, `near * far / (far - d * (far - near))`, where the raster shader wrote
`far * (z - near) / ((far - near) * z)`; nothing was drawn where it reads 1. No depth pass, no readback of the layer, no upload of the depth.
The meshes are submitted first and the processor prepares the layer (order, colours, buffers: 4.9 ms) while the card draws them; the layer
and the copies out follow in a second submission over the same frame. The display picture is made in that submission, as for any played
frame. `SplatCloud.normals()` is kept (a cloud is immutable) and was 1.9 ms of every relit frame. Used for opaque meshes with splats,
one band, a plain background, no volume, every adapter; translucent meshes with splats still go to the ray tracer's layer merge and a
shadowing light with splats outside playback still goes to the ray tracer (unchanged). `gpu3d.SPLATS_ON_CARD = False` returns the layer to the processor.

The cause of the 10/22 "5 times slower with a picture buffer" finding: any MAP_READ buffer of about 24 MB or more kept alive on the
RTX 3080 Ti makes the other readbacks (each creates, maps and destroys its own buffer) take about 3 times as long. It showed again as the full-quality picture
after a stop (the ray tracer reads back in 16 bands) taking 5.6 s instead of 1.9 s once a played frame's 25 MB readback buffer
(16.6 MB of half-float colour and the 8 MB picture) stayed alive. `gpu3d._release_staging` lets go of that buffer before a ray-traced render; a played
frame allocates it once again.

Part 3, the gates (frames per second shown; interval p95 in ms in brackets; every row 1920 x 1080, 10 s, Xvfb with software OpenGL):

| Case | RTX proxy off | RTX proxy on | Radeon off | Laptop stand-in off | llvmpipe off / on | Gate |
|---|---:|---:|---:|---:|---:|---|
| Cube | 23.93 (45) | 23.85 (44) | 23.87 (44) | 23.85 (45) | 23.90 (50) / 23.92 (49) | 24 at full size |
| Furnished room | 23.93 (45) | 23.90 (44) | 23.88 (44) | 23.79 (46) | 2.98 / 20.6 (mostly 1/8) | 24 at full size |
| Mixed mesh and splat scene | 23.99, 23.69, 23.69 (45 to 47) | 23.78 (45), all pictures full size | 23.82 (44) | 23.88 (45) | 2.39 / 11.0 (mostly 1/8) | 24 at full size |

Before: mixed 7.77 on the RTX (p95 135), 10.42 on the Radeon, 9.83 on the laptop stand-in, 9.23 on llvmpipe with the proxy on
(`benchmarks/3d-playback/full-size-1023`). The mixed scene's three RTX runs scatter between 23.7 and 24.0 (one to three of 240
pictures late per run) where the cube and the furnished room read 23.9; its node evaluation is 25 ms against 21 ms for the furnished room. The full-quality picture
is on screen 1.8 s after a stop on the RTX (gate 3 s), 0.9 s on the Radeon and 0.8 s on the laptop stand-in. The furnished room on llvmpipe is a software
figure and no gate. Picture on screen against the processor's picture of the same float frame: 0 or 1 level of 255 in every run (`picture_check`).
The laptop stand-in is the Radeon with `taskset -c 0-3` and `QT_SCALE_FACTOR=2` on this workstation, not the Windows laptop with the Iris Xe and the RTX 4070 (not measured).

The A6000 (estimate, not measured): the RTX's 19 ms of "GPU render and wait" holds roughly 9 ms of the 25 MB of half-float frame and picture crossing this
machine's USB4 link (about 2.7 GB/s, an inference from the link speed) and the mesh and splat drawing. On a card in a PCIe 4.0 x16 slot the transfer is about a tenth of that, so the
mixed frame's evaluation should be about 12 to 15 ms and the Qt thread's presentation the bound, as for the other scenes. Run
`python tools/benchmark_3d_playback.py --scene cube,furnished,mixed --stages` there.

Not done in this section (smoke and glass with splats are on the card since the 10/23 gaps step, below): the
mixed scene's 2-sample full-quality frame still reads back the layer through the ray tracer, the Windows laptop.

### Smoke and glass with splats on the card (10/23, step A of the gaps step)

Scene: `--scene mixedglass` of `tools/benchmark_3d_playback.py` is the mixed scene plus a 1.2 m smoke puff and a translucent glass sphere
(24 rows, about 2,300 triangles). Before this step smoke together with splats raised Unsupported in `gpu3d.render` (the CPU drew the
frame, 1.3 s at 1080p) and a translucent mesh with splats took the ray tracer's layer merge. Now the splat layer is drawn on the card
between the mesh passes and the smoke march, and around the glass in two parts ordered by a second depth buffer of the nearest
translucent surface; the smoke with meshes plays from resident buffers.

Where the card frame goes (RTX 3080 Ti, 1080p, 1 sample, 30 frames, `tools/profile_3d_mixed_stages.py --scene mixedglass`,
`benchmarks/3d-playback/gaps-1023/stages-rtx-mixedglass-card.json`): 32 ms for the whole render call. Without the smoke 21.7 ms, without
the splats 23.5 ms, without both 17.8 ms (before the last two changes below). Three changes took the played frame from 44 ms to 40.7 ms:
the glass depth pass draws one run per material (2,300 draws became 1); the translucent triangles are written far to near into a buffer
of their own, so the colour pass is one draw per material (it was a draw call per triangle, 14 ms of Python a frame); the ray count of the
volume work estimate is 6,000 (was 20,000), and `Plume3D`'s analytic plume is cached read-only per resolution and seed.

Gates, 1080p, "Proxy while playing" off, 10 s of playback (`benchmarks/3d-playback/gaps-1023/<adapter>-proxy-off|glass-proxy-off/`):

| Scene | RTX 3080 Ti | Radeon 8060S | Laptop stand-in (Radeon, 4 CPUs, 2x scale) | llvmpipe |
|---|---:|---:|---:|---:|
| cube | 23.86 fps | 23.9 | 23.9 | 23.93 |
| furnished room | 23.93 | 23.92 | 23.98 | 2.98 |
| mixed | 23.79 | 23.79 | 23.99 | 2.38 |
| mixedglass (new) | 23.2, 24.56 once the first tier settles | 23.74 | 23.64 | 0.59 (earlier build) |

The frame interval of the new scene is 40.8 ms on the RTX (budget 41.7 ms); the fps column includes the first frames of the run. The cube,
furnished and mixed rows on the Radeon and laptop rows were measured at 10:30 on the build before the glass changes (those scenes have no
translucent mesh and no smoke, so the changes do not touch them). The A6000 estimate (not measured): an estimate from the RTX's 26 ms card time: a
workstation card with the same or more shader throughput holds the 41.7 ms budget; the airgapped machine has to run it to confirm. Render mode settles the mixed scene in 6.3 s cold and 6.0 s
warm on the RTX (target under 10 s; Radeon 8.9 s, laptop stand-in 8.8 s; `benchmarks/progression/*-settle-1023.json`), was 24 s.

### The first second of a 3D scene (10/23, step A of the first-second step)

Question: does a 3D scene hold its rate from the first second, and how fast is the first picture? `tools/benchmark_3d_playback.py
--first-seconds 3 --replay` records, per scene (cube, furnished room, mixed, mixedglass), the time from opening the scene to the first
picture, every frame interval of the first 3 s of playback and every pipeline, shader module, buffer, texture and queue upload the card
was asked for in that time (`BuildTrace`; the product code is not touched). Cold is a fresh process; warm is the same process after a rewind
(`--replay`). `--play-at first` presses play the moment the first picture is up (`--play-delay S` waits S seconds first); the default
presses after the scene's own pictures are done. Numbers: `benchmarks/3d-playback/firstsec-1023/<run>/<scene>.json` (`rtx-before` is
main at ccc32cb3, the others the build of this step).

What the timelines showed, largest first:

- **Smoke, splats and glass: 7.1 s to the first picture.** A progressive preview draws a 1/8 draft first. With smoke that shades relit
  splats the exact picture is the processor's (`gpu3d.render` raises `Unsupported` outside playback), 7 s even at 240 x 135. The exact
  picture of that scene takes about 100 s at 1080p on the processor, so the playback tool waited for it, and pressing play while it ran
  showed nothing: the 100 s render could not be cancelled.
- **Pressing play did not cancel the render running ahead of it.** `TileExecutor.compose_region` did not hand its cancel event to the
  evaluator on the direct Render3D path, and `PlaybackQueue` set only the newest taken job's event, not the one still running ahead of it on
  the one worker (the app clears `busy` before that worker is free). After a stop, a rewind and a press of play the first played frame came
  when the stopped frame's render finished.
- **Before the first picture** (every scene): the first render pipeline 114 ms, its shader module 14 ms, a 38 ms texture upload, and for
  the mixed scenes 370 to 735 buffer creations (about 110 ms): 0.5 to 0.7 s from open to first picture in all.
- **In the first played frames:** the display pass of the card picture (2 shader modules, 1 pipeline, 8 ms) was built by the first played frame.
- **A frame slot lost in the middle of playback (finish pass).** The smoke, splats and glass card frame fills 40 of the 41.7 ms, so the
  moment a played frame finishes decides whether its successor is requested at once. `_next_playback_interval_ms` moves the transport's
  deadline on when a wakeup arrives within 0.5 ms of it, but `playback_tick` counted elapsed frames by truncation: a wakeup 0.06 ms short of
  the frame boundary (ticks log, `ticks*.json` in the lane run directory: elapsed 10.999 frames) moved the deadline and left the playhead,
  and the next request waited a whole slot, an 82 to 85 ms gap. Nine of twelve cold 3 s runs of the scene showed one or more of them
  (23.0 to 23.7 fps between pictures when one was there); none of sixteen runs after the change did. The other three scenes finish with more room before
  their deadline, so a wakeup rarely lands in that window.
- **The card's idle state.** After about 3.5 s without work the RTX drops to its idle clock (210 MHz, P8) and its link to the processor
  to PCIe generation 1 of 4. With play pressed after 4 s of rest every scene showed one to three intervals of 60 to 154 ms (cube 101, furnished
  128, mixed 89, smoke and glass 154), and the smoke, splats and glass scene met it in practice: its exact picture is 100 s of processor work,
  the card idles meanwhile, and a press of play after it read 20.8 to 22.9 fps with a 159 to 164 ms interval and 2 to 3 frames the playhead
  skipped. The Radeon and the laptop stand-in do not show it. Step A measured the shader clock (a compute probe 14 ms at the idle clock, 3 ms
  after 335 ms of load), found no cheap way to hold it, and left a choice about cost to Gonzo; step B's finish pass found that the link, not the
  shader clock, is most of the cost, and what it does about it is written under "The first frames after the card idles" in the step B section.

What changed:

- A coarse tier of a progressive preview that is not the artist's selected tier is a *draft*: `playback.first_look_document` marks a
  splat-with-meshes Render3D with the playing route's `pt_seed`, so the draft is drawn on the card (raster meshes, splat layer, smoke,
  glass; no shadows between splats and meshes) and never shares a cache entry with the exact picture. The selected tier's request carries the
  document unchanged, so the settled picture is what it was.
- `PlaybackQueue` keeps every job it has taken and not retired and cancels all of them when a request replaces the queue; the worker
  retires a job when it is done. `compose_region` passes its cancel event to the evaluator on all of its full-frame paths.
- A picture before playback builds the card picture's display pass (`gpupicture.warm`, armed through `arm_warm` for stopped display frames),
  so the first played frame creates no shader module or pipeline.
- `Window.playback_tick` counts a wakeup within `PLAYBACK_TICK_SLACK` (0.5 ms, the same slack `_next_playback_interval_ms` moves its
  deadline on from) as on the frame boundary, so a deadline that moved on always advances the playhead. The first-seconds report also lists
  the frames the playhead skipped (`frames_skipped`).

First picture and first 3 s of playback at 1080p ("Proxy while playing" off, play pressed at the first picture, cold app; "between" is the
rate between the first and last picture of the 3 s, "worst" the longest interval in them; budget 41.7 ms, twice that 83.3 ms; runs of
the finish pass in `benchmarks/3d-playback/firstsec-1023/*-fix-*`, 10/23):

| Adapter | cube | furnished | mixed | mixedglass |
|---|---|---|---|---|
| RTX 3080 Ti, first picture, before the step | 0.56 s | 0.53 s | 0.70 s | 7.11 s |
| RTX 3080 Ti, first picture, now | 0.58 s | 0.60 s | 0.67 s | 0.70 s |
| RTX, first 3 s, card warm: between / worst | 24.0 / 45 ms | 24.2 / 46 ms | 24.1 / 48 ms | 24.0 / 53 ms |
| RTX, play pressed after the scene settled: between / worst | 24.0 / 47 ms | 24.0 / 45 ms | 24.1 / 54 ms | 23.9 / 47 ms, first picture 230 ms after the press |
| RTX, play pressed after the card idled 4 s: between / worst | 24.8 / 101 ms | 25.3 / 128 ms | 24.7 / 89 ms | 23.0 / 154 ms |
| Radeon 8060S, first picture; between / worst | 0.51 s; 24.0 / 45 ms | 0.56 s; 24.1 / 45 ms | 0.59 s; 24.1 / 49 ms | 0.60 s; 24.0 / 46 ms |
| Laptop stand-in (Radeon, 4 CPUs, 2x scale), first picture; between / worst | 0.52 s; 24.0 / 46 ms | 0.56 s; 24.1 / 46 ms | 0.60 s; 24.1 / 47 ms | 0.61 s; 24.1 / 52 ms |
| llvmpipe, first picture; between / worst | 0.48 s; 24.4 / 50 ms | 0.54 s; 3.2 / 357 ms | 0.66 s; 2.5 / 435 ms | 0.98 s; under 2 pictures in 3 s |

Before the step the mixedglass run on the RTX never started playing inside the tool's two-minute wait (`rtx-before/mixedglass.json`: no
picture shown in 6 s of playback). The first played picture comes 25 to 60 ms after the press when the card is warm (before: 87 to 173 ms for
the three scenes that played). Warm replays (rewind after the exact picture settled, play again) on the RTX: worst interval 45, 47, 50 and
164 ms for the four scenes (the last after 100 s of processor work with the card idle); the Radeon 44, 47, 45 and 48 ms; the laptop stand-in
46, 49, 55 and 52 ms.

Held on the RTX: first picture under 1 s for all four scenes (0.58 to 0.70 s, cold); no pipeline or shader built by the first played frames
(the tests below); 24.0 fps or better between pictures in the first 3 s for all four scenes with no interval over 54 ms and no frame
skipped, when play is pressed at the first picture or after the scene has settled. The Radeon and the laptop stand-in hold the same, for
all four scenes, cold and warm. llvmpipe plays the cube at rate and the other scenes at the software renderer's own speed, as before.
Not held: with the RTX idle for 4 s before the press the first played frames wait on the card's idle state (the bullet above), one
to three intervals of 60 to 154 ms in every scene; see the step B section for the half-size ramp that shortens it.

Tests (`tests/test_first_second.py`; each fails without its change): the draft documents and the window's progressive tiers; a draft of
smoke, splats and glass is drawn on the card (the processor rasteriser is not called) and the exact picture still is the processor's; the
direct Render3D path stops on a set cancel event; the queue cancels every running job; the worker retires its jobs; a picture before
playback builds the display pass so the first played frames create no shader or pipeline; the played frames of smoke, splats and glass
rebuild no pipeline the first picture built. `PlaybackTickKeepsItsSlot` (finish pass): a tick 0.06 ms before a frame boundary advances the playhead, one 2 ms before waits, and the deadline and the tick share the slack constant. Run on the RTX, the Radeon and llvmpipe with the splat, volume and glass agreement tests
(`tests-L4-agree-*.log` in the lane run directory), each run under `flock -w 1800 /tmp/nb-gpu.lock` with a timeout.

### The four 3D scenes on the integrated tier, the adapter choice and the card's idle clock (10/23, step B of the laptop step)

Question: the Windows laptop (Iris Xe and an RTX 4070, hybrid) is the machine where the app felt sluggish. It has not been measured; the laptop
stand-in is the Radeon 8060S on `taskset -c 0-3` with `QT_SCALE_FACTOR=2` on this workstation. This section re-measures the four scenes (cube,
furnished room, mixed room with 20,000 splats, smoke with splats and glass) at 1080p on the RTX, the Radeon and the stand-in with the playback
tool of step A (`--first-seconds 3 --replay --play-at first`, "Proxy while playing" off, so the rows are the raw cost), names the largest cost on
the integrated tier, and reports what was and was not done about the card's idle clock. Numbers: `benchmarks/3d-playback/laptop-1023/{rtx,radeon,laptop}-{before,after}/<scene>.json`
(`before` is main at c1e55032, `after` this step); the stage timelines of the Radeon are in `stages/`.

First picture and first 3 s of playback after this step ("between" is the rate between the first and last picture, "worst" the longest interval
in the first 3 s cold and warm; budget 41.7 ms):

| Adapter | cube | furnished | mixed | mixedglass |
|---|---|---|---|---|
| RTX 3080 Ti | 0.57 s; 24.0 fps; worst 45 / 47 ms | 0.58 s; 24.2 fps; worst 45 / 45 ms | 0.64 s; 24.1 fps; worst 46 / 45 ms | 0.71 s; 24.0 fps; worst 54 / 96 ms |
| Radeon 8060S | 0.52 s; 24.1 fps; worst 45 / 46 ms | 0.54 s; 24.3 fps; worst 45 / 46 ms | 0.59 s; 24.2 fps; worst 50 / 45 ms | 0.60 s; 24.2 fps; worst 49 / 52 ms |
| Laptop stand-in | 0.52 s; 24.1 fps; worst 45 / 46 ms | 0.54 s; 24.2 fps; worst 45 / 50 ms | 0.59 s; 24.2 fps; worst 44 / 47 ms | 0.60 s; 24.1 fps; worst 55 / 50 ms |
| Laptop stand-in, whole 6 s run | 24.02 fps, p95 44.0 ms | 24.02 fps, p95 44.0 ms | 23.99 fps, p95 44.4 ms | 24.00 fps, p95 48.6 ms |

The RTX mixedglass warm 96 ms is the replay after the scene's 100 s processor picture, with the card idle for that time (the idle clock below). The
stand-in's mixedglass 55 ms is one interval above the 54 ms limit in the cold run; the slowest frames of that scene are the ones whose wait on the
card is 31 to 34 ms against a median of 14 ms (`submit to gpu_done` in `slowest_worker_frames`; the Radeon also draws the display, `shared_card_suspected`).

**The largest cost on the integrated tier.** Stage medians of a played frame on the stand-in, before the step, in ms (the render worker's frame;
the Qt thread's hand-over comes on top):

| Scene | whole frame | scene evaluation | GPU render and wait | readback copy | hand-over to the display |
|---|---|---|---|---|---|
| cube | 10.6 (25 percent of 41.7) | 1.2 | 1.6 | 2.8 | 3.1 |
| furnished | 19.2 (46 percent) | 1.7 | 9.4 | 3.0 | 3.0 |
| mixed | 19.4 (47 percent) | 2.0 | 9.6 | 2.7 | 3.0 |
| mixedglass | 25.9 (62 percent) | 2.1 | 14.3 | 2.7 | 3.0 |

The cube and the furnished room already held 24 fps on the stand-in, cold and warm, before this step (and so did the mixed scenes); the frame leaves
38 to 75 percent of the budget free. The cost all four scenes share, and the largest of the cube, is the readback copy: one NumPy copy of the 33 MB
float picture out of the mapped card buffer took 2.7 to 3.1 ms on the RTX, the Radeon and the stand-in alike. Copying plain memory of that size takes
1.2 ms in one thread and 0.75 ms in two, so the mapping limits it. `gpu3d._copy_band` now copies a band of 4 MB or more in slices of rows on up to four
threads (half the processors the process may use, `READBACK_PARTS`; NumPy lets go of the GIL for a copy this size). Readback copy, stage median before
and after: stand-in 2.79, 2.99, 2.73, 2.73 ms became 2.37, 2.35, 2.22, 2.20 ms; Radeon 2.70, 2.94, 2.80, 2.81 became 2.08, 2.21, 2.14, 2.15; RTX 2.77, 2.89, 2.78,
3.14 became 2.24, 2.25, 2.18, 2.72. The whole worker frame fell by 0 to 1.0 ms (stand-in 10.57, 19.22, 19.43, 25.86 became 10.53, 18.88, 19.32, 24.86), which is
the gain; the rates did not change because they were at 24.0 already. The next cost for the three heavier scenes is the card's render (9.4 to 14.3 ms on
the Radeon) and, on the Qt thread, the 3.0 to 3.5 ms hand-over; neither was touched. The Iris Xe is several times slower than the Radeon 8060S, so on the
laptop the render would be the cost that decides, and "Proxy while playing" is what keeps its rate; nothing here measures the laptop.
Test: `tests/test_readback_copy.py` (same rows in the same places with padded rows, short bands and a single row; big bands split and small ones not).

**Adapter choice on a hybrid machine.** `gpu3d.rank_adapters` puts a discrete card first, then integrated, virtual and software ones, and the same card listed
again through OpenGL after its Vulkan entry. `gpu3d.adapter_choice()` now says why the card in use was chosen: "Automatic: a discrete graphics card is preferred over
Intel(R) Iris(R) Xe Graphics", "chosen in Settings → Graphics adapter", "chosen in the NODEBASED_ADAPTER setting", "'x' matches no adapter on this machine, so the best one was chosen automatically", or,
when only the integrated card is listed, "Automatic: the only graphics card this application can see; a discrete card is not listed" (Windows hides the discrete card from an
application it has not been told to give it to; the amber pill and `ADAPTER_WARNING_HELP` already describe the per-application Windows setting). The reason is the last line of the
top bar pill's tooltip and the note under Settings → Graphics adapter. Tests (`tests/test_graphics_adapter.py`, class `HybridLaptop`): a faked list of an integrated card, a
software renderer and a discrete card opens only the discrete card; the reason names the card and what it was preferred over; the same card listed through OpenGL is not another card; a named card is opened
and the reason says who asked; a list with the integrated card alone says the discrete card is not listed; the pill and the Settings page show the reason. This machine's own list is the same
shape (RTX, Radeon, llvmpipe, the RTX again through OpenGL) and the app chooses the RTX.

**The laptop check.** `python tools/laptop_scene_check.py` (field test guide, "3D scene check") runs the four scenes at 1080p as the application ships (Proxy while playing on) on the card the
application chooses and then on the other kind of card, each warm and with play pressed 4 s after the first picture, and writes `result.txt` (under 40 lines, to paste back) and `result.json`.
Run here on the stand-in (`benchmarks/3d-playback/laptop-1023/scene-check-standin/result.txt`): the Radeon holds all four scenes with no interval over 54 ms warm or after the rest (45 to 52 ms);
on the RTX with four processors the cube, furnished room and mixed scene show 61, 90 and 91 ms after the rest, and the smoke, splats and glass scene plays 74 percent of its pictures at a lower resolution
because the proxy steps down on the four processors. `tests/test_laptop_scene_check.py` tests the summary and verdict on canned runs and runs the command once on the software renderer.

**The first frames after the card idles (finish pass): the cube holds the gate, the other three scenes are shorter but not under it.** The first pass of this step
tried to bring the RTX's idle clock up with a burst of compute and lower-resolution frames, and measured no gain (`idle-start-base.json`, `idle-start-ramp-attempt.json`, the probes
in `clock-ramp-probes/`). The finish pass found the actual cost. After a rest of 3.5 s the card's link to the processor drops to PCIe generation 1 (`nvidia-smi` shows
generation 1 of 4 at idle; this workstation's RTX is also on a x2 USB4 link), and the readback of a 33 MB float picture takes 84 ms instead of 12; the first two or three full-size played
frames are that readback. The link climbs back by itself, in stages, after about 0.65 s of traffic that keeps it busy (`clock-ramp-probes/link_probe.py` and
`link_probe2.py`, log `link_probes_1024.log`): 33 MB copies back to back 84 ms, then 12 ms after 0.85 s; 8 MB copies every 42 ms (53 percent of the link) 22 ms, then 11.6
after 0.6 s (a middle stage); 16 MB every 42 ms (all of it) 42 ms, then 6 after 0.7 s (the top); 2 and 4 MB every 42 ms nothing. A compute burst raises the shader clock but not the link,
which is why the first pass's burst did nothing for the frames. In a played frame the stage after a half-busy link costs 51 ms per 33 MB for three frames until the driver sees the link full,
and the top stage steps down again about 1.5 s after the copies that raised it stop.

What it does now (`nodebased/cardclock.py`, `gpu3d.start_clock_ramp`, `Window._start_clock_ramp`): a play pressed with "Proxy while playing" on, the artist's tier at Full, a Render3D in the comp and a
discrete card the app has not used for 3 s draws its first frames at half size (`CLOCK_RAMP_TIER`: 8 MB, 22 ms on the idle link, on time). Each of those frames is followed by a copy to the processor, sized to fill 80 percent of the
rest of the frame's time at the speed the frame's own readback showed (at most 16 MB, in buffers the card keeps: destroying one stalled every thread on the device for 87 to 135 ms), so the link is full
and climbs in the 0.6 to 0.8 s it takes. When a frame's readback has become fast (the last two under 0.35 of the first three, after 0.35 s, or at 1.5 s whatever it showed), the last few frames at the lower size are held a little longer than the one before
(the first frame at the chosen size arrives longer after its start than a half-size frame does: three times what the half size took, plus the time it spends packing its scene again; each hold at most 10 ms more than the one before, 60 ms at most), the chosen
tier is put back without a new request (a request cancels the frame in flight and cost 135 ms), and the extra copies taper off over 2 s. Nothing is held during a rest and a warm card is left alone; the extra traffic is the first 1 to 3.5 s of a play after a rest.
Integrated and software adapters never take this path, so the Radeon and the stand-in play as before.

Play pressed 4 s after the first picture, "Proxy while playing" on, 1080p, first 3 s, cold app, longest interval in ms (`idle-ramp-1024/rtx-ramp.json`, `idle-start-base.json` is the same measurement before):

| Adapter, scene | before: worst of 5 runs | after: runs, longest interval of each |
|---|---|---|
| RTX cube | 66 to 114 | 8 runs: 49 to 54 (7 under 54) |
| RTX furnished | 53 to 89 | 5 runs: 59 to 73 |
| RTX mixed | 89 to 94 | 5 runs: 70 to 85 |
| RTX smoke, splats and glass | 94 to 203 | 3 runs: 76 to 78 |
| Radeon, all four scenes | 45 to 52 | 12 runs: 44 to 52 and one 94 (furnished, 1.6 s in, a single hiccup that the ramp cannot cause: this card takes no ramp) |
| Laptop stand-in, all four scenes | 45 to 55 | 12 runs: 45 to 52 |

The cube holds the gate (no interval over 54 ms; one run in eight reaches 54.4). The other three RTX scenes still show one interval at the switch (59 to 85 ms) and, for the smoke scene, four of 70 to 76 ms at the start (0.43.2; the next section is what changed).

### One packed scene for both sizes (step C, 10/26/2026)

Where the switch frame went (`benchmarks/3d-playback/ffshare-1025/timeline-before.txt`, made by `switch_timeline.py` from the raw trace; the RTX, 10/25): the furnished room spent 18 ms packing its shadow triangles and 10 ms building its resident scene again, the mixed scene 5 and 8 ms plus 11.5 ms allocating the full-size readback buffer, the smoke, splats and glass scene the same plus the volume, and the cube repacked nothing.
Two causes: the half-size and full-size frames hold equal arrays that are different objects (a tier is part of each node's digest, so the Evaluator makes the scene again per tier) and the card's cache keys told read-only arrays apart by identity; and the full-size readback buffer was made by the first full-size frame.
What changed: read-only arrays are keyed by their content (hashed once per array object and remembered, bounded at 128 MB; a read-only view of an array that can still change is hashed every time), so the full-size frame reuses the shadow triangles and the resident scene the half-size frames made and an edit still repacks; a ramp frame makes the readback buffer the chosen size will need, so the switch frame reuses it; the switch frames are now 23 ms (furnished), 26 ms (mixed), 41 ms (smoke, splats and glass) and 19 ms (cube).
The smoke scene waits 40 to 47 ms on the card for each of its first four half-size frames while the shader clock is down, so the first frames are drawn at a quarter of the chosen size when the first half-size frame kept the card busy 85 percent of the frame time or arrived over 52 ms after the one before (the furnished room's first frame waits 24 to 41 ms on the idle clock and does not qualify); a quarter-size wait that has fallen to two thirds of the first, or 0.8 s, brings back half size.

Same measurement, 8 runs of each scene on each adapter, play pressed 4 s after the first picture (`benchmarks/3d-playback/ffshare-1025/gate/`, `summarize_gate.py`; the run directory also holds the raw traces), longest interval of each run in ms:

| Adapter, scene | 0.43.2 | step C: 8 runs, range (runs over 54) |
|---|---|---|
| RTX cube | 49 to 54 | 49.3 to 59.7 (5) |
| RTX furnished | 59 to 73 | 49.7 to 67.0 (3) |
| RTX mixed | 70 to 85 | 54.3 to 63.9 (8) |
| RTX smoke, splats and glass | 76 to 78 | 51.3 to 59.7 (3) |
| Radeon cube | 44 to 52 | 44.4 to 50.1 (0) |
| Radeon furnished | 44 to 52 | 46.1 to 72.8 (1, a single 72.8 ms hiccup, 0.7 s in; this card takes no ramp) |
| Radeon mixed | 44 to 52 | 46.3 to 54.7 (1, 54.7) |
| Radeon smoke, splats and glass | 44 to 52 | 44.9 to 50.4 (0) |
| Laptop stand-in, four scenes | 45 to 52 | 44.6 to 52.0 (0 of 32) |

The gate (no interval over 54 ms in any scene on the three adapters) holds on the laptop stand-in and, but for two single intervals, on the Radeon. It does not hold on the RTX: the switch interval is gone (the long intervals after the first 0.4 s are 54 to 60 ms, a few ms over, with one of 66.7 ms in a furnished run; the scene that had 70 to 85 ms is at 54 to 64), and what remains over 54 ms is 12 intervals in the first 0.4 s after the play (59.7 ms between the first two pictures of a cube run: `submit to gpu_done` is 13 ms for its first frame and 26 ms for its second) and 14 later; the known limit below says what the traces show.
Known limit: on the RTX 3080 Ti the pictures of the first 3 s of a play after a rest can arrive 54 to 67 ms apart (all four scenes; 26 intervals over 54 in the 32 runs above, 12 of them in the first 0.4 s and 14 later, median 56.7 ms). The finish pass (10/26) timed the trace of a 59.7 ms cube run and found that the card's clock is part of it and the rest is the app's own scheduling:

- The first half-size frame of a rested cube takes 13 ms on the card and is on screen 26 ms after its tick; the second takes 26 ms on the card (the shader clock is still down: 13 to 26 ms for the same work) and is on screen 40 ms after its tick. The interval between the two pictures is the frame time plus that difference, 60 ms.
- From then on the frames take 29 ms from the worker taking the request up to the end of the readback, so a picture is on screen 40 to 45 ms after its tick, about a frame time. A frame that finishes just before the next tick asks for the next frame before its own picture is shown (`preview_ready`, so the worker can render while the Qt thread paints), and that frame's Python work (7 to 8 ms of scene preparation and encoding) runs on the worker thread while the Qt thread is putting the finished picture on screen: the hand-over takes 12 ms instead of 5 (read in a run with the pacing described below, where every frame of the ramp took 29 ms, so the frames finished right at the tick) and the gaps alternate 52 and 28 ms (that cube run, 0.1 to 0.4 s: 47, 37, 44, 52, 28, 47, 51, 30, 53, 27). The same alternation is in the committed runs, with no pacing (mixed, run 3: 63.9, 28.4, 32.2, 50.8, 32.1, 51.2), and its peaks are the 54 to 60 ms intervals; the Python-lock explanation is read from those traces and not isolated by an experiment.

Three changes were tried on the RTX and not kept (`benchmarks/3d-playback/ffshare-1025/pacing-experiment/`: the summaries of each, and `pacing.patch`, the pacing with its tests): holding each frame of the ramp until it has taken 70 percent of the frame time from the render's start (4 runs of each scene: the cube 52.0 to 55.6 ms and 1 over 54, but the furnished room 4 of 4 over and the mixed scene 4 of 4 over, since the held frame left less time for the worker before the next tick), the same counted from the worker taking the request up (the cube 53.1 to 54.4, 2 over; the mixed scene 71.1 at worst), 50 percent of the frame time (the cube 44.1 to 51.3 and none over, the furnished room 1 of 4 over, the mixed scene 2 of 4, the smoke, splats and glass scene 4 of 4), and that with the next frame asked for after the picture is shown instead of before (the cube 1 of 4, the furnished room 2, the mixed scene 2, the smoke scene 4 of 4: no better). The 50 percent pacing over the full 8 runs per scene: the cube 4 over 54 (5 without it), the furnished room 5 (3), the mixed scene 6 (8) and the smoke, splats and glass scene 8 (3): the smoke scene's frames fill the frame time on the card and the pacing only moves them, so it was taken out again. What remains: the frames of the ramp take about a frame time on the RTX, so the beat between the scheduler and the picture hand-over (ask for the next frame after the picture is on screen only when the worker would not be left waiting) and a second frame that waits 26 ms on the idle clock decide whether an interval passes 54; a compute load that raises the clock without lengthening the played frames is the other open way (the first try lengthened them).

Tests of the ramp (`tests/test_card_clock.py`, each fails without the part it names): the link-up decision on canned frame times (a middle stage is not up, a fast frame before the minimum time is not enough, a card that never speeds up counts as up at 1.5 s); the extra copy fills the rest of the frame's time at the readback's speed, is never over 16 MB, is not sent for a frame that fills its time, keeps its buffers for the next ramp and ends quietly on a failing card; the holds grow in steps and count the scene's packing time and the cap; the copies taper off and stop; a ramp exists only for a discrete card that has rested; the window draws half size on a rested card, leaves a warm card, "Proxy while playing" off, an artist's own tier and a comp without a 3D render alone, puts the tier back without a new request, leaves the lowered frames out of the automatic tier, and a stop gives the tier back. Tests of this step run on the RTX, the Radeon and llvmpipe with the splat, volume and glass agreement tests, each under `flock -w 1800 /tmp/nb-gpu.lock` with a timeout (`tests-L4c-agree-*.log` in the lane run directory).
