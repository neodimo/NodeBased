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
