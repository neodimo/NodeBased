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
| 1080p / RTX 3080 Ti | Transform 183.6; Blur 91.2; ColorCorrect 76.4; Read plate 31.5; Read background 29.8 | Transform 46.5; Grade 18.8; Blur 13.4; Merge 13.2; Roto 12.2 |
| 4K / llvmpipe | Transform 667.2; ColorCorrect 324.0; Blur 311.0; Read background 118.8; Read plate 117.2 | Transform 499.3; ColorCorrect 286.7; Merge 131.7; Blur 122.3; Grade 119.7 |

| Uncached run | Before → after |
|---|---:|
| 1080p / RTX 3080 Ti | 1.5 → 4.5 fps (finish 1; earlier after run 2.4) |
| 1080p / llvmpipe | 1.6 → 2.4 fps |
| 1080p / integrated laptop stand-in (four CPUs, 2× Qt scale) | — → 2.3 fps |
| 4K / llvmpipe | 0.4 → 0.5 fps |

Cached playback stayed near its prior cadence: 1080p llvmpipe was 12.3 fps before and 11.7–11.8
fps after, with about 84 ms median frame intervals in both runs. 4K llvmpipe reached 4.7 fps on a
cached 16-frame loop (frames 33–48), the window retained by the default cache budget at UHD.
The animated-size Checker Viewer check reached 23.79 fps on the RTX 3080 Ti, just below its 24 fps
gate. The finish-1 tool measures delivered pictures in a fixed window after the first live picture;
its result is `benchmarks/2d-playback/finish1-checker-1022-animated-rtx.json`.

The measured full-resolution changes cut small-radius Blur with shifted-window accumulation,
use separable cubic sampling for translation-only Transform, and bound Read-ahead to one upcoming
full-resolution frame. Read-ahead is skipped when the future display frame is already cached.
Playback also skips synchronous disk spills for evicted transient evaluation frames and restores
the normal disk-cache policy when stopped. This finish-1 run showed steady 288 ms median frame
intervals, with no disk-tier write samples, and 4.5 fps over ten seconds. Animated Transform remains
a full-frame fallback because its canvas-global coordinates do not yet have a tile kernel. The 16 fps
uncached comp gate and the 24 fps Checker gate remain open. Finish-1 profiles are
`benchmarks/2d-playback/finish1-after-1022-1080p-rtx3080ti.{json,txt}`; the earlier profile files under
`benchmarks/2d-playback/` contain the per-stage timeline and Qt/worker sampling evidence; read
decode events include future frames warmed by the reader pool.

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
