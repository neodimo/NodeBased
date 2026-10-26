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
- Remaining after the island: the node stage still holds the Read decodes, the Roto matte and the CPU ColorCorrect (about 55 ms of the
  Radeon's 69 ms), and the 8-bit display conversion; closing 24 fps needs those as well.

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

The cube holds the gate (no interval over 54 ms; one run in eight reaches 54.4). The other three RTX scenes still show one interval at the switch (59 to 85 ms) and, for the smoke scene, four of 70 to 76 ms at the start.
Cause of the switch interval: the first full-size frame packs and uploads the scene for the playing document again (12 ms for the furnished room, 40 ms and more for the mixed scene with its 20,000 splats),
because the scene the half-size frames drew is not the one it draws; the hold covers the time the readback and the render take at the larger size but not that, and a hold that covered it would make the frames longer than the
frame time. Cause of the smoke scene's start: its frames are bound by the card's shader clock (a half-size frame is 52 ms on the idle clock, 14 ms on the raised one), which only a compute load raises, and a compute load beside the played frames lengthened them
(first pass). Left: pack the playing document's scene at the chosen size while the half-size frames run (the evaluator caches a scene per tier, so this wants the scene evaluation shared between tiers), and for the smoke scene a lower first tier than half.

Tests of the ramp (`tests/test_card_clock.py`, each fails without the part it names): the link-up decision on canned frame times (a middle stage is not up, a fast frame before the minimum time is not enough, a card that never speeds up counts as up at 1.5 s); the extra copy fills the rest of the frame's time at the readback's speed, is never over 16 MB, is not sent for a frame that fills its time, keeps its buffers for the next ramp and ends quietly on a failing card; the holds grow in steps and count the scene's packing time and the cap; the copies taper off and stop; a ramp exists only for a discrete card that has rested; the window draws half size on a rested card, leaves a warm card, "Proxy while playing" off, an artist's own tier and a comp without a 3D render alone, puts the tier back without a new request, leaves the lowered frames out of the automatic tier, and a stop gives the tier back. Tests of this step run on the RTX, the Radeon and llvmpipe with the splat, volume and glass agreement tests, each under `flock -w 1800 /tmp/nb-gpu.lock` with a timeout (`tests-L4c-agree-*.log` in the lane run directory).
