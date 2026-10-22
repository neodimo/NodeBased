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

| Comp / adapter | Before: fps | Top five node costs (estimated ms/frame) | Read decode (ms/displayed frame) | Display conversion (ms/displayed frame) |
|---|---:|---|---:|---:|
| 1080p / RTX 3080 Ti | 1.5 | Transform 183.6; Blur 91.2; ColorCorrect 76.4; Read plate 31.5; Read background 29.8 | 72.5 | 49.1 |
| 4K / llvmpipe | 0.4 | Transform 667.2; ColorCorrect 324.0; Blur 311.0; Read background 118.8; Read plate 117.2 | 348.5 | 216.1 |

Both graphs fall back to full-frame evaluation because animated Transform has no tile kernel and
depends on canvas-global coordinates. The 1080p run also measured 1.6 fps on llvmpipe. See
`benchmarks/2d-playback/profile-1022-1080p.txt` and `profile-1022-4k.txt` for the complete ranked
costs; the JSON files carry the per-stage timeline and Qt/worker sampling profiles. Before/after
measurements for the fixes are recorded in the same table as they land.

### Still slow

- In the default Viewer, `to_qimage` is about 41 ms per 1080p frame and 170 ms at 4K (ten
  full-frame float32 temporaries and a float readback from the integrated GPU). The Direct picture
  does it on the card during playback (above); the Viewer itself, stills and compare pictures
  still pay it.
- Each displayed frame is still two worker jobs with a signal to the UI thread between them.
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
