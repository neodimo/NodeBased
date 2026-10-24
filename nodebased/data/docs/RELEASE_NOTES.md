# NodeBased 0.43.0-preview.2

Preview build of 0.43.0: the packaged app was checked to install and launch on Linux and Windows; the full test run happens in the 0.43.0 release that follows.

## What changed since 0.42.1

- **Smoke, splats and glass in one scene on the graphics card.** A scene mixing a smoke puff, Gaussian splats and a glass
  sphere now plays from resident buffers at about 34 ms a frame at 1080p on the RTX 3080 Ti (it took 1.3 s). The splat layer
  is drawn between the mesh passes and the smoke march, and around glass in two parts ordered by the nearest see-through
  surface, so splats seen through glass and smoke in front of splats both come out right. The mixed mesh-and-splat scene
  draws the splats in the mesh frame's own submission and reads its depth.
- **Glass draws faster.** See-through meshes draw from a far-to-near vertex buffer, one draw per material instead of one per
  triangle. The glass scene plays at 40.7 ms a frame at 1080p on the RTX (was 44).
- **Render mode settles in a quarter of the time.** The mixed scene settles in 6 s (was 24): path tracer stacks are sized to
  the scene and smoke shadows meshes through the ray tracer's tree.
- **First picture in under a second.** The coarse steps of a progressive preview now draw splats with meshes on the card. The
  smoke, splats and glass scene shows its first picture in 0.64 s at 1080p on the RTX (was 7.1 s); all four test scenes come
  up in 0.58 to 0.70 s.
- **Playback at full rate from the first second.** All four test scenes play at 24 frames a second from the first picture,
  with no gap over 54 ms, on the RTX, the Radeon and the laptop stand-in. Pressing play while a slow still render is running
  cancels it and starts playback at once (it used to wait for that render, up to about 100 s). The first played frame builds
  no new shader or pipeline. A playback clock slip that dropped a whole frame in roughly every third 3 s of playback is fixed.
- **Fluids: whitewater refinement stays inside a card-memory budget.** Full-detail whitewater refines under a live memory
  budget and retries with less neighbour work instead of running out; scrubbing reuses the whitewater setup, and steam
  sampling stays at the preview size while you scrub.
- The 3D scene benchmark tools gained the smoke, splats and glass scene, and preview builds can now be published as
  pre-releases (`vX.Y.Z-preview.N`).

## Known limits

- After the RTX has idled about 4 seconds, its power-saving clock slows the first two played frames: one gap of 89 to 154 ms
  at the start of playback in every scene. Steady playback afterwards holds 24 fps.
- All timings come from this workstation (RTX 3080 Ti, Radeon, a four-core laptop stand-in, software renderer). The laptop,
  the A6000 and Windows have not been measured.
- Fluids: the full-detail Hot pour has not yet completed every stage under memory limits; steam and the laptop and A6000
  memory checks are still open.
- Uncached 2D playback is still well short of 24 fps; moving the nodes to the graphics card is the next decision.
- The Direct picture is still off by default.

## Moved to 0.44.0

- Deciding how to handle the idle-clock gap in the first frames (step down a resolution for those frames, or accept it).
- Fluids: steam and the laptop and A6000 memory checks for full-detail refinement.
- The GPU compositing spike for 2D (research write-up published, implementation not started).

# NodeBased 0.42.1: 3D comps at 24 fps, the Direct picture as a full viewer, Fluids that re-simulate in seconds, and faster 2D playback

## What changed since 0.41.0

0.42.0 was tagged but never published: its package build failed tests that needed a real graphics card on the cloud build machine. Those tests now skip cleanly there, and everything from 0.42.0 is in this release.

- **3D playback at the comp's rate.** Under an animated camera at 1080p, the cube and the furnished room play at 24 fps at full
  size on the RTX 3080 Ti, the Radeon and a four-core laptop stand-in (the furnished room was 2 to 12 fps). The card now makes
  the display picture itself, so converting a frame for the screen takes under 1 ms where it took 26 ms. The heavy mesh-and-splat
  scene plays at a reduced size and refines to the full picture about 2 s after you stop. The time row names the size shown
  and says "real time" only when the rate holds.
- **Render mode settles in seconds.** Render mode stops sampling where the picture is already clean. The furnished room
  settles in 3 s (was 8) and the mixed mesh-and-splat scene in 24 s (was 82). The mixed scene's first look arrives in about
  0.6 s.
- **The Direct picture does everything the Viewer does on this workstation.** Edit handles, Roto and RotoPaint drawing, paint
  strokes, Tracker and planar picking, and every pick tool (Cryptomatte, Focus, ZDefocus, Survey, PointsTo3D) behave exactly as
  in the Viewer, and tests prove it. All the Viewer's zoom keys work. Paused frames, edits and refinement go to the card as
  floats. The A/B compare modes and the wipe line run on the card. Handles and the pixel readout follow playback. The dock is
  saved with the workspace layout and can be switched to be the main viewer, and opening the 3D view puts it beside the
  picture. Zoomed out, the Direct picture averages pixels where it used to skip them. It stays off by default.
- **Faster uncached 2D playback.** The next frame computes while the current one is shown. Source images are read and
  colour-converted ahead of time, and EXR colour conversion is about 5 times faster. A realistic 1080p comp plays fresh frames
  at about 5.7 fps on the RTX (was 3.5). A 4K comp drops to a lower tier while it plays and returns to full resolution about
  a second after you stop. Stopping playback no longer blanks the Viewer.
- **Live pictures while you drag.** Roto, RotoPaint, Tracker and Transform drags show a picture within a third of a second,
  scrubbing shows a first picture in a quarter second, and paint strokes only reprocess what they touch.
- **Fluids at interactive speed.** The full-HD Hot pour settles in about 7 s on the RTX and 6 s on the laptop stand-in (was
  68 s), with soft steam, the pour reaching the cup and a soft table shadow. After an edit to the pour, the cup or the steam,
  the first re-simulated frame appears in under 0.2 s and the timeline fills as frames bake. A new edit cancels the old bake
  within 25 ms. Look-only knobs never re-bake, a change keyed at a frame re-bakes from that frame, and every edit shows a
  low-res version first, then sharpens. Older caches convert once without re-simulating.

## Known limits

- All timings come from this workstation. The laptop, the A6000 and Windows have not been measured.
- Uncached 2D playback is still well short of 24 fps. The nodes themselves now take about 170 ms of every frame, and moving
  them to the graphics card is the next decision.
- The mixed mesh-and-splat scene plays at full size at about 8 fps. At the comp's rate it uses a reduced size.
- The Direct picture is not the default viewer yet: it needs the laptop and A6000 runs, and it has no native Wayland or HDR
  surface. Zoomed out, the Viewer still skips pixels, so the two look different at 1:4 and 1:8.
- Fluids: the production-size timings for edit re-simulation have not been measured.

# NodeBased 0.42.0: 3D comps at 24 fps, the Direct picture as a full viewer, Fluids that re-simulate in seconds, and faster 2D playback

## What changed since 0.41.0

- **3D playback at the comp's rate.** Under an animated camera at 1080p, the cube and the furnished room play at 24 fps at full
  size on the RTX 3080 Ti, the Radeon and a four-core laptop stand-in (the furnished room was 2 to 12 fps). The card now makes
  the display picture itself, so converting a frame for the screen takes under 1 ms where it took 26 ms. The heavy mesh-and-splat
  scene plays at a reduced size and refines to the full picture about 2 s after you stop. The time row names the size shown
  and says "real time" only when the rate holds.
- **Render mode settles in seconds.** Render mode stops sampling where the picture is already clean. The furnished room
  settles in 3 s (was 8) and the mixed mesh-and-splat scene in 24 s (was 82). The mixed scene's first look arrives in about
  0.6 s.
- **The Direct picture does everything the Viewer does on this workstation.** Edit handles, Roto and RotoPaint drawing, paint
  strokes, Tracker and planar picking, and every pick tool (Cryptomatte, Focus, ZDefocus, Survey, PointsTo3D) behave exactly as
  in the Viewer, and tests prove it. All the Viewer's zoom keys work. Paused frames, edits and refinement go to the card as
  floats. The A/B compare modes and the wipe line run on the card. Handles and the pixel readout follow playback. The dock is
  saved with the workspace layout and can be switched to be the main viewer, and opening the 3D view puts it beside the
  picture. Zoomed out, the Direct picture averages pixels where it used to skip them. It stays off by default.
- **Faster uncached 2D playback.** The next frame computes while the current one is shown. Source images are read and
  colour-converted ahead of time, and EXR colour conversion is about 5 times faster. A realistic 1080p comp plays fresh frames
  at about 5.7 fps on the RTX (was 3.5). A 4K comp drops to a lower tier while it plays and returns to full resolution about
  a second after you stop. Stopping playback no longer blanks the Viewer.
- **Live pictures while you drag.** Roto, RotoPaint, Tracker and Transform drags show a picture within a third of a second,
  scrubbing shows a first picture in a quarter second, and paint strokes only reprocess what they touch.
- **Fluids at interactive speed.** The full-HD Hot pour settles in about 7 s on the RTX and 6 s on the laptop stand-in (was
  68 s), with soft steam, the pour reaching the cup and a soft table shadow. After an edit to the pour, the cup or the steam,
  the first re-simulated frame appears in under 0.2 s and the timeline fills as frames bake. A new edit cancels the old bake
  within 25 ms. Look-only knobs never re-bake, a change keyed at a frame re-bakes from that frame, and every edit shows a
  low-res version first, then sharpens. Older caches convert once without re-simulating.

## Known limits

- All timings come from this workstation. The laptop, the A6000 and Windows have not been measured.
- Uncached 2D playback is still well short of 24 fps. The nodes themselves now take about 170 ms of every frame, and moving
  them to the graphics card is the next decision.
- The mixed mesh-and-splat scene plays at full size at about 8 fps. At the comp's rate it uses a reduced size.
- The Direct picture is not the default viewer yet: it needs the laptop and A6000 runs, and it has no native Wayland or HDR
  surface. Zoomed out, the Viewer still skips pixels, so the two look different at 1:4 and 1:8.
- Fluids: the production-size timings for edit re-simulation have not been measured.

# NodeBased 0.41.0 — a rebuilt Properties panel, easy wire disconnect, a faster node graph, and live pictures while you edit

## What changed since 0.40.0

- **Properties panel overhaul.** Every number knob has a visible slider again, and fields take the width their numbers
  need. Each node has one header row with its own collapse. ColorCorrect now works like Nuke's: master, shadows, midtones
  and highlights, with Ranges, and old scripts convert automatically. Every colour knob opens a linear float colour picker
  in place of the 0-255 one.
- **Easy wire disconnect.** Drag an input wire off onto empty canvas, hover a wire and press Delete, or Ctrl+Alt-drag a
  knife line across wires. The tooltip and the ? overlay list all three.
- **The node graph at display rate.** A moved node re-routes only its own wires. Node bodies and wires are cached as sharp
  pictures, wires stay plain while you drag, pan or zoom, and below 40 percent zoom nodes become flat cards. On this
  workstation dragging in a 500-node graph went from 19 to 120 repaints a second (205 nodes: 31 to 120), the
  press-to-highlight time went from 25 to 8 ms and a pan repaint from 10 to 2.7 ms. OpenGL graph drawing is available as a
  setting and is off by default.
- **Viewer colour on the graphics card.** The Direct picture runs the display transform, OCIO and ACES included, on the
  card. An ACES draw went from 17 ms to 0.4 ms.
- **Direct picture playback and 4K.** Play a comp with the Direct picture open and the graph, timeline and Viewer keep
  answering within 31 ms, at 1080p and 4K. A 4K frame used to freeze the window for up to a quarter second and now costs it
  about 1 ms. A picture too big for the card is named in the status bar, closing the dock frees its memory, and the cache
  inspector shows what it holds.
- **A live picture while you work.** Roto, RotoPaint and Tracker controls show progressive pictures while you drag them,
  scrubbing refines progressively, and playback steps down one proxy level when rendering misses the frame budget.
  PlanarTracker analysis shows progress and can be cancelled.
- **Hot pour shot.** The corrected fluid shot has a visible stream and pool, its holdout matches the plate, and a solver
  failure on a stale grid is fixed.
- **Fixes.** A layout loop that could crash the Properties panel with some fonts, and the "cannot schedule new futures"
  errors from a cache tidy-up after the window had closed.

## Known limits

- The graph, playback and live-picture timings come from this workstation on a private display. The laptop, the A6000 and
  Windows were not measured, and panning with most of a 500-node graph in view (27 ms) and wheel zoom are still slow.
  OpenGL graph drawing was slower on the software renderer, and three graph tests fail with it on, so leave it off.
- Live pictures while editing: the first-picture and settle times and the playback frame rates are still to be measured on
  the 12 GB card and the software renderer.
- Hot pour: the corrected 8/32/128-sample timings, the 1080p progressive first look and settle times, and the 64-cell
  Radeon preview rate are still to be measured.
- Direct picture: 4K playback on the RTX reached the window in order for 171 of 241 frames, limited by its external link.
  It has not been checked on a real screen, on Windows or on the Radeon.

# NodeBased 0.40.0 — a denoised first look in half a second, and an experimental GPU picture for the 2D viewer

0.39.0 was tagged but never published (its Linux package build failed on two viewer tests that need a display); everything
from it ships here.

## What changed since 0.38.1

- **A denoised first look within a second.** Press P in the 3D viewport and a clean, denoised full-size picture shows up
  first, labelled "denoised preview from 480x270", then the render refines as before. On the RTX 3080 Ti the first picture
  takes 0.47 s for a furnished room and 0.78 s for the same room with 40,000 splats and smoke; the Radeon 8060S takes 0.50 s
  and 0.75 s. Before, a denoised 1080p render took 10 s and 19 s. The first stage is sized from measured cost and shares
  the scene with the guide passes.
- **Direct picture (experimental).** Window menu → "Direct picture (experimental GPU)", or start with
  `NODEBASED_DIRECT_VIEWER=1`, opens a floating dock that uploads each new picture to the graphics card once, then pans,
  zooms and resizes it there without redoing the colour conversion. It is off by default; the existing Viewer is unchanged
  and stays the default. `docs/DIRECT_VIEWER.md` describes it.
- **Adapter choice for the dock.** The direct picture picks its display adapter through wgpu's surface filter, so it lands on
  a card that can actually present to the window. A bench tool compares it against the current viewer.
- **Lost graphics device.** If the card resets or the driver restarts, the direct picture gets a new device and rebuilds the
  picture from the copy it keeps in memory, so the dock comes back without reopening it.
- **Native Wayland refused cleanly.** On a native Wayland session the dock now says it needs `QT_QPA_PLATFORM=xcb` and the
  Viewer carries on, where before the app could crash.
- **Windows laptop research.** `docs/WINDOWS_HYBRID_PRESENTATION.md` maps how a picture reaches the screen on a hybrid
  Intel + NVIDIA laptop and recommends DirectX 12 as the first Windows path for the direct picture, with a probe for the field kit.
- **Tests.** Drawing, pan, zoom and stale-picture rules are covered by automated tests on the RTX 3080 Ti, the Radeon and the
  software renderer, plus a walk through changing pictures, pan, zoom, resize, docking, reopen and 1.5x display scaling.

## Known limits

- First look: the software renderer's very first call takes 1.16 s; the A6000, the laptop and Windows were not measured;
  the denoiser still runs on the processor, and the raw step right after the preview can look noisier than the preview.
- The direct picture has no Roto, Tracker, A/B or region overlays; use the default Viewer for those. It is untested on
  Windows. Device-loss recovery is tested by destroying the device in software; a physical card reset has not been tried.

## Coming next (0.41.0)

- The Properties panel overhaul (sliders everywhere, a linear float colour picker, ColorCorrect like Nuke's), easy wire
  disconnect, the node graph at display rate with hundreds of nodes, and direct-picture playback and 4K.

# NodeBased 0.39.0 — an experimental GPU picture for the 2D viewer

## What changed since 0.38.1

- **Direct picture (experimental).** Window menu → "Direct picture (experimental GPU)", or start with
  `NODEBASED_DIRECT_VIEWER=1`, opens a floating dock that uploads each new picture to the graphics card once, then pans,
  zooms and resizes it there without redoing the colour conversion. It is off by default; the existing Viewer is unchanged
  and stays the default. `docs/DIRECT_VIEWER.md` describes it.
- **Adapter choice for the dock.** The direct picture picks its display adapter through wgpu's surface filter, so it lands on
  a card that can actually present to the window. A bench tool compares it against the current viewer.
- **Lost graphics device.** If the card resets or the driver restarts, the direct picture gets a new device and rebuilds the
  picture from the copy it keeps in memory, so the dock comes back without reopening it.
- **Native Wayland refused cleanly.** On a native Wayland session the dock now says it needs `QT_QPA_PLATFORM=xcb` and the
  Viewer carries on, where before the app could crash.
- **Windows laptop research.** `docs/WINDOWS_HYBRID_PRESENTATION.md` maps how a picture reaches the screen on a hybrid
  Intel + NVIDIA laptop and recommends DirectX 12 as the first Windows path for the direct picture, with a probe for the field kit.
- **Tests.** Drawing, pan, zoom and stale-picture rules are covered by automated tests on the RTX 3080 Ti, the Radeon and the
  software renderer, plus a walk through changing pictures, pan, zoom, resize, docking, reopen and 1.5x display scaling.

## Known limits

- The direct picture has no Roto, Tracker, A/B or region overlays; use the default Viewer for those.
- It is untested on Windows. Device-loss recovery is tested by destroying the device in software; a physical card reset has
  not been tried.
- Screen delay is not measured. Windows is tested by the automated suite on the CI runner only.

## Moved to 0.40.0

- The parity gate (VP2): overlays on the direct surface and the decision whether it can become the default Viewer.
- A Windows run of the direct picture on the hybrid laptop.

# NodeBased 0.38.1 — faster playback and graph clicks, a graphics adapter choice, and the Windows package repaired

## What changed since 0.38.0

0.38.0 was tagged but never published: its Windows package failed two tests. 0.38.1 carries everything listed under 0.38.0
below, plus:

- **Playback.** Static frames convert once and reach the display without a half-float round trip; Render3D skips
  supersampling an empty scene. At 1080p and 24 fps a constant Checker now plays at the full 24 fps (was 9.8), an empty 3D scene
  with an animated camera on the GPU at 11.9 fps (was 3.5), and one Render3D frame of an empty scene takes 8 ms (was about 190).
  The time row shows the frame rate playback really achieves, in the warm colour when it falls behind.
- **Graph input.** A click paints its selection highlight before the Properties panel is built, and letting go of a dragged node
  no longer rebuilds the panel or the graph. On the 18-node demo, press to highlight went from 130 ms to 12 ms and a drag release
  from 154 ms to 3 ms. Panels are reused on re-selection, the material preview renders after the panel shows, and the minimap
  repaints only when something changed.
- **Graphics adapter.** NodeBased ranks discrete over integrated over software cards when it picks an adapter, with a choice in
  Settings, a `NODEBASED_ADAPTER` override, a warning in the top bar when it runs on a slow adapter, and the chosen adapter in the
  field report. `tools/benchmark_adapter_choice.py` times each card. On this workstation the integrated Radeon beat the RTX
  3080 Ti eGPU on a heavy scene, so the app prefers a discrete card without promising it is the fastest.
- **Fixes.** The Properties panel cache is scoped to the graph level, so entering a group shows the right panel. The field-report
  smoke test no longer runs the hardware benchmarks, and the GPU-against-CPU lens test uses the cross-renderer tolerance; these were
  the two Windows failures.

## Known limits

- **Laptop:** the Intel Iris Xe against RTX 4070 comparison is not measured yet; the adapter choice there is untested on the real
  machine. Windows is tested by the automated suite on the CI runner only.
- **Playback:** the display conversion still costs about 41 ms a frame at 1080p, so animated content stays under real time.

# NodeBased 0.38.0 — the lens on the graphics card and in the 3D viewport and renderers, a camera and lens recovered from the surveyed plate, Roto rendering only the tiles the viewer asks for, and a field test kit for the A6000 work machine

## What changed since 0.37.0

- **Lens.**
  - `Undistort` and `Redistort` run on the graphics card: a wgpu compute warp behind a `lens_backend` knob (auto, cpu, gpu),
    checked against the CPU warp on three adapters, with `tools/benchmark_lens_warp.py` timing both at 1080p and 4K. Documents
    saved before the knob load with the automatic backend (schema 24).
  - The 3D viewport looks through the camera's lens: the overscan picture bent by Redistort on the card, a Lens switch on the
    viewer strip and the L key, overlays drawn through the lens. Frame-time benchmark on three adapters.
  - `Render3D` has a `through_lens` setting: the raster and path tracer backends render the camera's lens, matched against
    render + Redistort on three adapters, including a literal GPU-raster against CPU-raster test. Old documents load with it off.
- **Camera tracking.** The demo shot's distorted plate now gets its camera and lens from the surveyed landmarks (focal length 0.08%
  off, k1 within 0.002), and the shot passes the stand-in verdict with that camera. The DPV-SLAM solve keeps its own files.
- **Roto and the viewer.** `Roto` renders viewer tiles and analysis regions directly, and tiling is bypassed for broad requests. The
  second part of rendering only what the viewer asks for.
- **Fluids.** A camera-matched Hot pour shot plate in the examples, resized for the comp, with a test over the plate composite
  switches and rendered stills at frames 60 and 120 in the docs.
- **Field test kit.** `tools/field_report.py` records a machine, runs the GPU tests and benchmarks, and writes one report folder plus
  a zip; `tools/make_field_bundle.py --everything` builds an offline stick (standalone Python, vc_redist, portable Windows app,
  verify script) for a no-admin machine, with per-user paths and a runtime DLL report. `docs/FIELD_TEST.md` is the walkthrough and
  `docs/FIELD_CLAUDE.md` the file to paste to Claude Code on the work machine. Nothing is uploaded by the kit. Tested end to end,
  with an injected crash and a rezip.
- **Research.** The radar re-ranked for speed on the 12 GB workstation and the 48 GB A6000: few-step SDXL previews, NeuFlow v2,
  then materials generated once and rendered live, with a proposed SDXL experiment across both cards.
- **Tests.** The host-phase profile tolerates a loaded box (medians per phase against the frame median); `run_tests.py` takes
  several patterns.

## Known limits

- **Windows:** tested by the automated suite on the CI runner only. The field kit has been exercised on the Linux workstation; its
  first run on the A6000 machine is still ahead.
- **Lens:** the GPU warp and the lens in `Render3D` and the viewport agree with the CPU path within the documented limits on three
  adapters; a filmed plate has not been through the solve yet. MASt3R-SLAM still recovers the focal long, so DPV-SLAM stays the
  default solver.
- **Roto and trackers:** `Tracker` and `Stabilize` ROI rendering and the ROI patch shown during refinement are in a merge candidate
  that may land with this release or the next; full-frame refinement still takes about four seconds.
- **Fluids:** the 64-cell Hot pour bake on the 3080 Ti is measured (35 s for 120 frames); the full sample timings, 1080p progressive
  first look and Radeon preview/stop are not yet.
- **Planar tracking:** four corner clicks, CPU analysis over full frames, no real-display review.
- **New look:** starting a wire uses 20 px; a body drop onto a card with no free input is dropped; J while playing forwards stops and
  steps back.
- **2D:** the Keylight row stays partial. **Rigid bodies:** no fracture.

## Moved to 0.39.0

- The field report from the A6000 work machine, and whatever it turns up.
- The rest of rendering only what the viewer asks for: tracker and proxy kernels with timings, and a faster full-frame refinement.
- DiMo's filmed plate through the camera solve.
- The few-step SDXL preview experiment from the radar.
- The 2D-to-3D pipe (ImageToSplat, ImageToMesh, WorldSculpt), on hold.

# NodeBased 0.37.0 — a PlanarTracker node that drives CornerPin and Roto, MASt3R-SLAM as a second camera solver that needs no lens, a survey check on every exported camera, the distorted plate through the whole shot with View undistorted, and a fluid stop that settles within a second

## What changed since 0.36.0

- **2D.**
  - A `PlanarTracker` node: pick a four-corner plane in the viewer, track it across the frames, see its keyed quad
    and fit error, export the motion as an animated `CornerPin`, or make a `Roto` shape follow the plane. Tested on
    synthetic projective motion, with recovery from a quarter-area occlusion and the same keys on every run.
  - `Roto` can render an exact crop of its image: shapes whose bounds miss the requested region are skipped. The
    first part of rendering only what the viewer asks for.
- **Camera tracking.**
  - MASt3R-SLAM as a second solver in `CameraSolve3D` (no focal length handed in): per-frame poses from its
    keyframes, the focal length fitted to the keyframes' point maps, then the poses refined by a bundle adjustment
    of its own matches. Measured next to DPV-SLAM at 960x540 and 1920x1080: it recovers the focal 12 to 15% long
    and runs at 10 frames a second against DPV-SLAM's 46, in 3.2 GB, with bit-identical repeat runs. On the
    demo shot its camera fails the new survey check, so DPV-SLAM stays the default.
  - `verify` has a survey check: surveyed world points are projected through the exported cameras and compared
    with their tracks on the plate (RMS within 0.5% and worst within 1.5% of the plate height), scored in the
    verdict when a survey is given.
  - The demo shot has a stand-in for a filmed plate (a wide-angle Brown-Conrady lens and handheld shake) and a
    camera solve benchmark.
- **Rendering and the viewer.**
  - The distorted plate round trip: the demo shot runs with `--plate distorted` (plate, `Undistort`, conditioning
    and comp, `Redistort`, Write) by hand and by an agent to identical ids and pixels, with the lens round-trip
    error reported beside the verdict.
  - The viewer offers View undistorted / View plate for a graph with an `Undistort` node, showing the whole
    overscan frame with the plate's format outlined inside it.
- **Fluids.** Stopping the Hot pour preview reuses the baked liquid and whitewater caches: the half-resolution
  surface arrives in 12 to 312 ms and full detail within a quarter second at frames 1, 60 and 120 on both
  Radeon targets, inside the one-second target.
- **Research.** The research radar's last part: a sourced survey of compositing, keying, roto and simulation
  beyond what NodeBased does, with ranked experiment proposals.

## Known limits

- **Windows:** tested by the automated suite on the CI runner only. The GPU lock and the model providers (SDXL,
  DPV-SLAM, MASt3R-SLAM) have only been run on Linux.
- **Lens:** CPU only; `Render3D` and the viewport still draw through the ideal camera. On the distorted plate
  DPV-SLAM finds the camera path (5 cm) but not the lens (focal 10% off, k1 0.001 against -0.03), and the shot with
  that solved camera fails its verdict, so the committed demo keeps the rendering camera. No filmed plate yet.
- **Planar tracking:** picking is four corner clicks and the analysis runs on the CPU over full frames; no
  real-display review yet.
- **Roto and trackers:** only the Roto crop request honours the region; the tile executor still renders a full
  Roto matte and tracker analyses read full frames.
- **New look:** starting a wire uses 20 px; a body drop onto a card with no free input is dropped; J while playing
  forwards stops and steps back.
- **2D:** the Keylight row stays partial. **Rigid bodies:** no fracture.

## Moved to 0.38.0

- The rest of rendering only what the viewer asks for: tiles, trackers and proxy kernels, with timings.
- A lens estimate that recovers k1 on the distorted plate, then the solved camera in the default demo graph; the
  lens in `Render3D` and the viewport.
- DiMo's filmed plate through the camera solve.
- The 2D-to-3D pipe (ImageToSplat, ImageToMesh, WorldSculpt), on hold.

# NodeBased 0.36.0 — the new look lands (node column, viewer strip, timeline, Properties, transport), Roto draws and feathers in the viewer, a lens on the camera that the solve can now recover, shadow catchers and holdouts, SDXL as the first real model provider with a generated shot that stops flickering, and a camera solved from the plate

## What changed since 0.35.0

- **The new look (stages three to five, and the punch lists from DiMo's reviews).**
  - A left column of node families with SVG icons for the thirteen families and 146 common nodes, drawn in any
    colour, and a floating node panel; the window's minimum width follows the top bar. A free node dragged onto a
    wire is spliced in: the wire lights up under the card, the drop goes into the main input and the old
    destination (mask kept), one undo.
  - The viewer strip and corner readouts, a slim time row and timeline track, slim dock title bars, and a graph
    "?" overlay in place of the cheat-line. Panels are locked by default so docked panels have no title bars
    (Workspace → Lock panels brings them back). Simulate and Render have their own dock layouts, saved and
    restored per tab; an output preview panel; the top bar search lists node types, menu commands and settings in
    one list and Enter runs the first result. Views stay framed on resize until the artist zooms; the graph
    frames its nodes above the corner's button row.
  - Properties restyled: compact sections, colour wheels centred at neutral, number fields and sliders from the
    theme, and knob rows sized by the number they show so a colour's four fields and an xyz row stay on one line.
    The panel is empty with nothing selected.
  - A status bar: cache, frame, when the project was last saved, version, and a state word at the left (Ready,
    Rendering, Playing, Saving). The viewer no longer draws a dotted frame and size label around the canvas. The
    family panel animates on a shared motion clock with a motion preference.
  - The viewer surround is pure black in every theme, 2D and 3D. A full transport: start, previous key, step,
    play backwards, stop, play, next key, end and a loop mode (once, loop, ping-pong), with Nuke's keys (J, K, L,
    Alt+Left/Right); the strip and the timeline are one quiet playback unit with 28 px controls on an 8 px rhythm
    and a thin ruler with key ticks and the cache line. The accent is an amber like Nuke's selection colour
    (#e0a24a); the mint and the muted teal stay as named choices.
  - Wiring feel: a dragged wire snaps to the nearest compatible socket within 40 screen pixels, or to the card
    under the pointer, at every zoom; the socket and card light up and the status bar names the target. Pressing
    beside a socket, on its label or on a card's bottom band starts a wire; sockets grow as the pointer nears.
- **2D.**
  - `Roto` draws and edits bezier curves in the viewer (drawing begins on press, tolerances scale with zoom), moves
    several points at once, and has a transform box: move, scale and rotate the shape. Open splines can be
    stroked instead of filled.
  - Feather grows outward from the shape's edge, inward with a negative value, with a feather handle per point,
    animated offsets, a reset for the selected points and an undo per gesture. Older projects keep their previous
    feather rendering (schema 22) and the panel offers the model switch. A 1080p feather of 50 renders in
    about 0.11 s.
- **Rendering.**
  - Per-object switches on every mesh node: Cast shadows, Receive shadows, Visible to camera, Holdout and Shadow
    catcher, honoured by the CPU and GPU raster, ray-traced and path-traced modes (the instanced GPU tracer falls
    back to the CPU). A holdout is cut out of the picture and the data passes. A shadow catcher renders black with
    the shadows the scene's other objects throw on it as alpha; depth sees the floor, the object-id pass and
    Cryptomatte do not. Docs sections on shadow catcher floors and putting CG onto a plate, and
    `examples/shadow_catcher`. The viewport's shadow pass honours Cast shadows.
  - A lens on the camera: Brown-Conrady distortion (radial, tangential, principal point, squeeze) on `Camera3D`
    and imported cameras (schema 21; older files load with no lens). `Undistort` straightens a plate with overscan
    so nothing is cut off and `Redistort` puts the lens back; `Project3D`, the tracker rays and the SceneState
    export read the lens.
  - The GPU path tracer does less work outside the card: scene buffers, accumulator, bind groups and readback
    buffers are kept between frames, several passes go in one submission with one uniform write, volume grids are
    packed with one copy, and the image is finished on the card for callers that skip the noise maps. The CPU path
    tracer samples from the occlusion-scaled diffuse colour so it matches the GPU (one scene from 55 to 106 dB).
    The full portable matrix is measured on the RTX 3080 Ti, the Radeon 8060S and the software renderer, and
    `tools/profile_host_phases.py` measures the host side of a frame by phase.
- **Camera tracking.** A `CameraSolve3D` node and `solve_camera` command: DPV-SLAM runs on the graphics card as a
  subprocess provider, with anchor and scale controls, and writes a keyed `Camera3D` and a point cloud into the
  document. On the demo plate at 960x540 it runs at 45 frames per second in half a gigabyte and lands within 2 cm
  of the true camera; DROID-SLAM was measured as the reference.
  - The solve recovers the lens and takes its scale from a survey. With no lens given, `CameraSolve3D` fits the
    focal length and k1/k2 by a bundle adjustment over the solver's own tracked points, and reports the median
    error and inlier fraction before and after. Survey points picked in the viewer are followed through the
    plate and the solve is aligned to their measured distance by scale or similarity (schema 23). The panel shows
    the last solve's lens and scale under Solve result. On synthetic tracks with known truth the focal length is
    within 1 percent at 1920 wide and k1 within 10 percent; the demo shot gains a k1 lens plate for the benchmark.
  - A deterministic robust homography fit and a dense-flow planar follow (`nodebased/planar.py`): a selected
    quadrilateral is tracked with Lucas-Kanade flow and a projective fit with outlier rejection, recovering the
    corners within a pixel over 24 synthetic frames. Library only in this release; the node and viewer picking
    follow.
- **Generative conditioning.** The first real model provider: SDXL ControlNet-Union, with control images from the
  depth and normals passes, running in its own interpreter as a subprocess, one frame at a time with a fixed seed,
  deterministic on the RTX 3080 Ti; model CPU offload by default so the pixels never depend on how much of the
  card other programs hold. MCP tools (`export_conditioning`, `generate`, `verify`, `loop`, `artifacts`,
  `workspace`) for an attached agent; `verify` scores the plate against its own SceneState; `Generate` carries the
  seed and sampler settings in the artifact identity. The demo shot runs end to end: a rendered plate, a camera
  solve from it, the shot graph, a headless hand run, a scripted agent run through MCP and a real Claude Code run,
  all with the same verdict. A `ConditionedRead` of a generated sequence now reads the right frame (every frame
  rendered as frame 1 before). The Agent panel and the Conditioning Queue wear the new look.
  - The generated shot stops flickering. `verify` measures temporal flicker (each frame warped into the next
    with the exported motion, occlusion and ID masked) and `sdxl_controlnet` 1.2 guides every frame from a
    warped canvas of frame 1 with holes filled from the frame's own text-to-image pass. On the demo shot the
    flicker drops from 15.4 to 2.6 display steps mean (48.3 to 7.7 at the 95th percentile), a PASS; the shot runs
    twice by hand and once by the agent to identical result ids. Chaining image-to-image from the previous frame
    (1.1) was tried first and compounded the model's contrast gain until the picture broke up.
- **Fluids.** Cached fluid playback: frames are packed, shared legacy caches are kept, budgets are isolated per
  simulation, full runs are retained (the Hot pour steam cache included) and eviction during a disk read is
  tolerated. While scrubbing, the preview draws reduced detail (48³ 8-bit volumes, bounded liquid and whitewater
  points, liquid points before the full surface) and reports a production miss; reduced frames persist. Decoded
  preview frames sit in a ring buffer, procedural geometry tracks are reused, and a stopped preview refines
  progressively. The production Hot pour preview at 240 x 135 plays at about 6.5 ms a frame on the Radeon
  (23 ms on the software renderer), well inside the 12 frames per second target.
- **Infrastructure.** GPU jobs (the SDXL and camera-solve launchers, the demo script) take an exclusive lock on the
  shared GPU lock file, so a model run waits for a running suite and a suite waits for a run. Every window a test
  opens is closed again.

## Known limits

- **Windows:** tested by the automated suite on the CI runner only. The GPU lock and the model providers (SDXL,
  DPV-SLAM) have only been run on Linux.
- **Lens:** CPU only; `Render3D` and the viewport still draw through the ideal camera. The camera solve is given
  the lens or recovers it from its own tracks; the recovered lens inherits DPV-SLAM's millimetre of run-to-run
  drift, so two solves give slightly different focal lengths and ids after the solve change. The lens benchmark
  on the real DPVO solves (960 and 1920 wide, plain and k1 plates) is queued behind the GPU lock and not yet
  recorded. Use plates 960 pixels wide or more. No filmed plate yet.
- **Fluids:** the preview plays inside its budget, but stopping it enters whitewater reconstruction and does not
  settle to half resolution within a second; the stop timings at frames 1, 60 and 120 are not recorded.
- **Planar tracking:** library and tests only; no PlanarTracker node, region picking, CornerPin export or Roto
  follow yet, and one-pixel recovery under quarter-area occlusion is unproven.
- **New look:** starting a wire uses 20 px (40 would turn every press on a title into a wire); a body drop onto a
  card with no free input is dropped; J while playing forwards stops and steps back.
- **Roto:** the feather handles have no real-display visual review yet. **2D:** the Keylight row stays partial.
  **Rigid bodies:** no fracture.

## Moved to 0.37.0

- The PlanarTracker node with viewer picking, a tracked plane that drives CornerPin and Roto, and the rest of
  2D parity B.
- Lens, part three: the lens in `Render3D` and the viewport, and the measured DPVO lens benchmark.
- A fluid stop that reuses the baked liquid and whitewater artifacts and settles within a second.
- Stage 9 of the end-to-end sprint: a second live camera solver measured against the plate, from the research
  radar's tracking and reconstruction survey.
- The last stages of the new look, and DiMo's filmed plate through the camera solve.
- The 2D-to-3D pipe (ImageToSplat, ImageToMesh, WorldSculpt), on hold.

# NodeBased 0.35.0 — the first of the new look, a Roto node that animates and follows trackers, a graph that behaves like Nuke's, volumes in the path tracer, adaptive sampling that pays off on the GPU, and liquids that stay on the card

## What changed since 0.34.0

- **The new look (first two of seven stages).** A darker theme with new type and rounded controls; a new
  top bar with the project and shot, workspace tabs, one search box, a GPU status light and Check for updates.
  The node graph draws each node tinted in its family's colour with a coloured outline, wires take the colour of
  the node they come from, mask inputs sit on the right side of a node, a Curved / Right angle switch sets the
  wire style, and the minimap shows the families. The left node column, the viewer strip, the timeline,
  Properties and motion follow in the next releases.
- **2D.**
  - New `Roto` node: an editable shape list, feather and opacity per shape, motion blur from animated shapes,
    keys and auto-key on each shape, keys shown on the timeline, in the viewer and in Properties, shapes linked
    to named tracker paths, and an optional background plate with the shape's coverage in its alpha.
  - The graph works like Nuke's: a node made from a shortcut or Tab search is wired to the selection; new nodes
    land below the selection (or at the centre of the view with nothing selected) and the graph pans to show
    them; F frames the selection. S makes a Shuffle (Settings moved to Ctrl+,).
  - Animated knobs say so: a tinted field, a diamond on key frames, a key button, I sets a key, previous and
    next key in the knob menu. Timeline key marks name the node and knobs keyed on the hovered frame.
  - Numeric fields show their full value and accept every typed digit (out-of-range values clamp, exponents
    allowed). The time row reads Frame, In and Out like a compositor's and refuses an inverted range.
  - Write renders report frame N of M with elapsed time and Cancel, a completion message with Show in folder,
    and stop on a failing frame with its number.
  - Merge with nothing in A passes B through unchanged. New point tracking nodes (2D and 3D), `PointsTo3D` and
    `Reconcile3D`. `HSVTool` range masks and alpha outputs, HueCorrect's saturation threshold, tile-native
    `Encryptomatte` masks, ScreenKeyer's remaining matte and edge controls; keyframed analysis regions are
    measured on every frame of a range.
  - Workspace: an artist-sized default layout with the Node Graph and NODES side by side, cramped saved layouts
    repaired; the 3D viewport button opens the viewer in front. RotoPaint puts its paint controls first.
- **Rendering.**
  - Smoke and other volumes in the path tracers (CPU and GPU), with volume motion blur across `Render3D`'s
    shutter and every data pass (normals, position, ids, Cryptomatte, motion, depth) for volumes.
  - Adaptive sampling never takes more samples per pixel than "Path samples", and is now faster than fixed 64
    samples on all nine pairs of graphics card and benchmark scene (Radeon 8060S, RTX 3080 Ti and the software
    renderer); a portable benchmark (`tools/benchmark_portable_render.py`) measures any adapter.
  - Smoke renders fill the GPU: on the RTX 3080 Ti a smoke frame takes 59 ms instead of 143 ms, on the Radeon
    61 ms instead of 113 ms.
  - Adaptive sampling now saves time on the GPU: on an RTX 3080 Ti it beats fixed 64 samples' quality in less
    time on all three benchmark scenes (one scene: 47.8 dB in 141 ms against 46.8 dB in 230 ms).
  - Light linking on the GPU without a CPU fallback for splat and instance sets, and now also on liquid glints,
    lit particle sets (whitewater included) and smoke, in every renderer.
  - Transparent meshes mixed with Gaussian splats on the GPU; depth, position and object_id passes for scenes
    with splats; a faster layered splat resolve. Sparse volumes are read straight from their tiles in the
    viewport, the GPU raster and the GPU path tracer. A mixed-scene example project.
- **Fluids.**
  - The FLIP liquid stays on the GPU between substeps (particles, grid, pressure), surface tension and
    adaptive domains included; face-field extrapolation runs on the card.
  - Long bakes keep the card's memory bounded; colliders that run past the domain stay solid to the edge and no
    longer widen an adaptive domain; steam rises off a liquid surface and collides with it; whitewater emits at
    fine resolutions and its neighbour search is about ten times faster.
  - A new Hot pour scene and preset: tea poured into a glass on a table, with steam.
  - Simulation caches are written as one packed file per frame (older caches still load), and Hot pour's
    full-size bake, stills and playback are measured on the Radeon and the software renderer.
- **Generative conditioning.** A persistent job queue with a Conditioning Queue panel and LAN workers; bounded
  generate-and-verify loops with saved candidates, hard limits, and controls to inspect, stop and choose a
  result. Artifact writes are safe under concurrency and on Windows.

## Known limits

- **Generative conditioning:** still no real model provider; only the stand-ins run.
- **Fluids:** cached playback of the full-size Hot pour scene is slow (about 0.09 frames per second on the
  Radeon): its frames outgrow the cache and are worked out again. Real-time 128³ liquids are not reached yet.
- **New look:** on Windows the top bar needs more than the 800-pixel minimum window width; a window squeezed
  to its minimum can still cut it off.
- **2D:** the Keylight row stays partial. **Rigid bodies:** no fracture.
- **Windows:** tested by the automated suite on the CI runner only.

## Moved to 0.36.0

- Interactive Hot pour playback, the rest of the new look, and the remaining 2D parity and Roto polish found in
  the hands-on pass of October 7.
- A first real model provider for generative conditioning (DiMo's choice).
- The 2D-to-3D pipe (ImageToSplat, ImageToMesh, WorldSculpt), on hold.

# NodeBased 0.34.0 — one viewer for 2D and 3D, adaptive sampling and light groups, liquids an artist can scrub, several fluids in one scene, and a generative conditioning chain from scene export to scored result

## What changed since 0.33.0

- **One viewer.** The 2D viewer and the 3D viewport now share one panel; Tab switches between them. The
  default layout keeps at least 35% of the window for the Node Graph.
- **Rendering.**
  - The GPU ray-traced viewport mode draws `pbr` materials with their texture maps, the HDRI `Environment`
    and area lights itself, from one packed material table and texture atlas, with no CPU fallback.
  - Adaptive sampling with a noise target in the path tracer (CPU reference and GPU): each pixel takes at
    least 16 samples and stops once it is quiet, and the render ends early when every pixel has. The
    progressive viewport render shows the pass and the share of converged pixels. `Render3D` gains
    Denoise final (the viewport's denoiser on the final render) and a `beauty_raw` layer with the noisy
    beauty. A GPU bug that made multi-sample passes about four times too bright is fixed.
  - Light groups and light linking. Name a group on any light and `Render3D` writes one layer per group
    (`light.key`, `light.fill`, ...) that adds up to the render, in all three modes. Meshes, splat sets and
    instance sets can use only some lights or all but some; an excluded light neither lights the object nor
    is shadowed by it. The new `LightMixer` node gives each group its own gain and colour, from a
    `Render3D` or an EXR; with every gain at 1 the picture is unchanged to the last bit.
  - Shadows for dense instancing: 100,000 copies inside four shadow-casting lights' views run at 38 fps at
    1080p on an RTX 3080 Ti (6.8 before) and 29.5 on an AMD Radeon 8060S. The GPU picks the copies each
    light draws and uses simpler shapes for far ones.
  - A camera written to Alembic reads back within a tenth of a pixel across an animated range, now checked
    with the same conventions as the USD export.
- **2D.**
  - RotoPaint: brush hardness gives a real soft edge, an optional mask and mix, the tile path, and B, E and
    C pick brush, eraser and clone in the viewer.
  - Curve editor: Ctrl-drag breaks a key's tangent, a number row for position and slopes, Reset curve, and
    double-click adds a key without changing the curve's shape. HueCorrect's red, green and blue curves
    work (they did nothing before) and its suppress curves act separately. CurveTool's region has viewer
    handles; MinColor and CurveTool results update on release as one undo step; regions can be keyframed.
  - `Blend` gains fringe, inject and a per-channel mask choice. The screen keyer takes inside and outside
    mattes, a clean plate as its screen reference, and separate despill and alpha-bias colours. The 2D
    parity table was re-audited against the code (118 supported, 3 partial, 8 missing, 5 not applicable).
- **Fluids.**
  - Domains that follow the fluid: smoke, fire and liquid bounds can grow and shrink with the simulation;
    checkpoints, the viewport, `Render3D` and VDB export keep each frame's box. Off by default for
    existing and new solvers; the Explosion and Dam Break presets turn it on.
  - Liquids an artist can scrub: GPU particle transfers on active tiles, surface tension that rounds drops
    and breaks thin streams, open walls per face for liquid and smoke with an escaped-volume count, landed
    spray that returns to the pool, and foam, spray and bubbles that fade by lifetime.
  - Rigid bodies write the liquid's push into the FLIP checkpoint, so later frames replay with it, and
    repeating a frame does not add the impulse twice.
  - Several fluids in one scene, each with its own cache; a liquid surface can collide with smoke; each
    fluid exports to its own named VDB at a chosen resolution (mass-preserving downsampling). Sparse caches
    play back without being expanded: at 256³, 9 MiB resident instead of 64 MiB.
- **Generative conditioning (new, CPU, no model yet).**
  - `WriteGeo3D` exports a versioned per-frame scene state (camera, objects, simulations, lights,
    environment, render passes) and a control bundle of beauty, depth, normals, forward and backward
    motion and Cryptomatte IDs, with units and colour tags.
  - `VerifyConditioning` solves the camera from tracked landmarks, checks objects against ID passes or
    Tracker points and compares light direction, shadows and colour balance against a fresh render; it
    writes a per-shot score card with one pass/fail verdict.
  - `ConditionedRead` brings generated frames back into the comp aligned to the shot with their layers.
    `Generate` is a cacheable plate-through node with a provider contract that lists which controls a
    provider honours; providers run in isolated, memory-capped worker processes with progress, cancel and
    logs, and the previous result stays until the new one is done. A stand-in depth-and-motion reprojection
    provider proves the chain end to end.
  - Every bundle, generated sequence and report is kept by content in an artifact store with its
    provenance.

## Known limits

- **Generative conditioning:** no real model provider is connected yet; only the stand-ins run. Lighting
  intensity is not checked by the verifier. Not driven by hand on a real display yet.
- **Liquid speed:** a 128³ dam-break substep takes about 1.6 s on the GPU, well above the 40 ms goal; the
  GPU pressure solve alone is 200 ms.
- **Sparse fluid caches** play through the CPU tile path; the GPU still uploads them dense.
- **Light linking:** linked splat and instance sets fall back to the CPU in the GPU raster and ray-traced
  modes; liquid glints, particles and smoke ignore links.
- **Adaptive sampling** saves time on the software adapter but not on the two tested GPUs at 1280x720,
  for 2 to 6 dB less quality than fixed 64 samples.
- **2D:** the Keylight row stays partial (core matte, pre-blur, edge colour correction); HueCorrect's
  saturation threshold is not built; a keyframed analysis region is measured only at the frame where it was
  released.
- **Rigid bodies:** no fracture. **Houdini** reading of the resized VDBs is unverified (Blender checked).
- **Windows:** tested by the automated suite on the CI runner only; one `BurnIn` check is still skipped
  there.

## Moved to 0.35.0

- The conditioning job queue with the Conditioning Queue panel and LAN workers (finished on its lane, not
  yet merged), and a first real model provider (DiMo's choice).
- The 128³ liquid speed target and a sparse GPU upload for fluid caches.
- A hands-on QA pass on a real display, then the next 2D parity and Rendering plans (both lanes parked until
  Monday night).

# NodeBased 0.33.0 — the first release driven by hand: first-hour fixes from a real-display session, textures, HDRI and area lights in the GPU preview, lights the camera can see, the 2D toolset complete, and measured quality gates

## What changed since 0.32.0

- **Driven by hand before shipping.** For the first time the app was used on a real display, as an artist
  would, before a release (9/30, and again on 10/1 after the fixes). That session found 21 problems in the
  first hour of use; all of them are fixed in this release:
  - The mouse wheel over the Properties panel scrolls the panel. It never changes a dropdown, slider or
    number it happens to pass over; a value changes only once that field has focus.
  - Tab search lists an exact name first, then names that start with what you typed, then the rest; the
    Up and Down keys move the highlight.
  - Every knob has a readable label (no more `Tracker_Id` or `tracking_channels`).
  - A new project is 100 frames long, and loading an image sequence into an untouched project stretches the
    range to fit.
  - Save and Open start in the last folder you used. The image browser hides dot-folders unless you tick
    Show hidden, and accepts a pasted file path.
  - The default layout gives the image viewer, the node graph and the node list each a usable share of the
    window. The 3D viewport button brings the 3D view forward. The viewer toolbar wraps instead of cutting
    controls off; its reset buttons have icons. The node list names its categories.
  - Roto starts drawing on the first click in the viewer. Picking a Tracker point jumps to the reference
    frame, prompts on the picture itself and draws the point at once.
  - The 2D viewer pans freely in every direction at any zoom.
- **Two stability fixes from the 10/1 integration run.** The dock-resize animation could deadlock the whole
  app against the GPU display thread, and closing a viewer window while its preview worker was still winding
  down could abort the process; both are fixed (the docks resize without animation now).
- **The fast GPU preview catches up with the final render.** `pbr` materials with all five texture maps
  (base colour, metallic-roughness, normal, emissive, occlusion), lighting from an HDRI `Environment`, and
  Rect, Disc and Sphere area lights with their shadows now draw in the GPU raster viewport, matching the
  CPU reference. The GPU path tracer draws texture maps too.
- **Lights and the sky seen by the camera.** Turn on Visible to camera on a Rect, Disc or Sphere light or an
  HDRI Environment and the camera sees it: the light at its real brightness, the HDRI as the background.
  On for new lights, off for existing documents, so old renders do not change. Works in both path tracers,
  the raster preview and the ray-traced mode, which also reads all five texture maps now.
- **The viewport shows more of the final picture.** Denoiser controls on `Render3D` (strength, colour,
  normal and depth sensitivity, iterations, temporal reuse); depth of field and motion blur in the
  viewport's progressive Render mode; `Instance3D` copies drawn in the viewport with per-copy tint and
  picking; shadows from up to four lights at once, from splats and from instanced copies.
- **The 2D toolset is complete.** Every Nuke 2D node that does not need a proprietary algorithm or a
  scripting runtime is now built. New this release: `TimeWarp` (lookup curve, frame blending), `TimeBlur`
  and `TimeEcho` on the tile path, `Profile` (time and cache hits of the graph above it, with CSV export),
  `TVIScale`, `Precomp` (another NodeBased project as a node, with Reload), `Assert` (a pass-through check
  that stops the graph with a named message), `CrossTalk` fringe and unpremult, `CurveTool` AutoCrop by
  colour, `DustBust` patch fill for specks on moving backgrounds, and `Inpaint` on the tile path.
- **Faster tracking and paint on the GPU.** `SmartVector` builds its range on the GPU (1080p over 20
  frames: 28.7 s against 70.6 s on the CPU); `VectorDistort` and `VectorCornerPin` warp there too.
  `Inpaint` gains a patch-based fill and a GPU temporal fill (53.9 s against 115.4 s). `GridWarpTracker`
  follows local motion between tracks and ignores occluded points.
- **Layers keep the fast path.** Depth, motion and Cryptomatte layers now travel through the tile
  evaluator, so graphs like Render3D → ZDefocus → ZMerge keep the region of interest and proxy tiers.
- **Quality gates, measured.** Checked automatically from now on:
  - Exact output of seventeen reference images (HDR, alpha, channels, filters, data windows, an EXR round
    trip); broken, empty, mislabelled and size-lying files fail cleanly without a crash or hang.
  - Every 2D node's cache notices every setting change; cancelling stops work within a tenth of a second.
  - Startup, time to first pixel and edit latency at 1080p and 4K, published in
    `docs/BENCHMARKS-v0.33-m2.md`; 4K and 8K memory stays under a shared budget.
  - 3D cameras line up with their rendered pixels frame by frame, also after a USD export and re-import
    (USD camera export is new); the tracker recovers the same paths; every GPU renderer reports its
    graphics memory and stays under a budget on three heavy scenes.

## Known limits

- **GPU ray-traced viewport mode** still draws `pbr` materials, HDRI lighting and area lights by falling
  back to the CPU; its data layout needs a redesign first (next).
- **Shadows on 100,000 instanced copies** hold 30 fps when most copies fall outside the lights' view;
  every copy inside four lights' view at once is slower.
- **Camera round trip through Alembic** is not checked: NodeBased reads Alembic cameras but does not write
  Alembic.
- **Rigid bodies:** the liquid feels a body's push in the returned scene, but that impulse does not change
  later upstream FLIP checkpoints. No fracture.
- **2D:** `Blend`'s fringe, inject and per-channel mask choices are not built.
- **Windows:** tested by the automated suite on the CI runner only; nobody has driven the app on a real
  Windows display or GPU. One `BurnIn` check is still skipped there, and since the 2D and 3D views
  moved into one panel two viewer-look checks read a black picture on the Windows runner and are skipped
  there; whether a Windows user would see a black viewer is unknown until someone drives the app on a Windows
  display. Windows fonts make the node list's
  fixed chrome taller, so on a 1280x720 window the default layout leaves the 2D viewer slightly shorter
  than the node list, and at 1440x920 the Node Graph gets a little under its usual quarter of the column.
- **Hands-on testing** so far covers Linux with an RTX 3080 Ti at 1440x920; the GPU work also ran on an
  AMD Radeon 8060S and the software fallback in the automated suite.

# NodeBased 0.32.0 — per-node OCIO colour management, light effects, textures and lit particles in the path tracer, a viewport close to the final render, combustion, fluid presets and GPU smoke tools, sparse up-res at 512³, OFlow retiming, tracked warps and a rigid-body solver

## What changed since 0.31.0

- **Per-node colour management.** `OCIOColorspace`, `OCIODisplay`, `OCIOLookTransform`,
  `OCIOFileTransform` and `OCIOLogConvert` convert inside the graph, choosing colourspaces, displays,
  views and looks from the active OCIO config (the document config by default, or an override per
  node). `Colorspace` takes Nuke-style primaries, transfer and white point. The ACEScg working space and
  the viewer's own display choice are unchanged.
- **Light effects.** `Flare` (drag and animate it in the Viewer, or link it to a `Tracker` point),
  `Glint`, `Sparkles`, `GodRays`, radial rays, lens artifacts and `ScannedGrain` with grain matched
  across the three channels.
- **Textures in the path tracer.** Metallic-roughness, normal, occlusion and emissive maps are sampled
  per pixel on meshes. glTF files with real material data bring all of it in automatically, and a USD
  file's bound `UsdPreviewSurface` maps come in the same way. A plain-coloured file renders exactly as
  before.
- **Particles with materials, lit and shadowed.** Give particles a real material and let colour,
  opacity, size or glow follow each particle's age or speed. They are lit by the scene's lights like a
  matching mesh, take shadows and throw them. Particles now appear in the depth, position and object-id
  layers, `Cryptomatte` isolates a single particle, the CPU and GPU path tracers and the GPU viewport
  renderer draw them, an emissive particle brightens a nearby wall, and whitewater's foam, spray and
  bubbles each get their own default look.
- **A viewport close to the final render.** Relit splats carry their own de-lit albedo and roughness,
  pbr meshes and splats are lit with GGX and the dome, a look-dev HDRI background toggle, a shadow map
  from one key light, a progressive path-traced Render mode, and a material-ball preview in geometry
  node panels.
- **Splats and smoke path trace on the GPU on AMD too.** The wrong colours 0.31.0 saw on AMD came from the
  card's shader compiler under working-memory pressure on one path; fewer threads per group on that path
  fixed it and nothing else changed.
- **Combustion.** A fuel puff burns into heat and smoke, keeps unburnt fuel, expands from gas release,
  renders through the blackbody look and exports fuel and flame to VDB. Reference render in
  `docs/images/fluid_combustion.png`.
- **Fluid presets, shelf tools and knob presets.** Nine fluid presets with thumbnails (fire, smoke, dust
  impact, dam break, pour, viscous drip, whitewater splash and two rigid-body scenes) in the shared
  preset browser; smoke, liquid, collider and fire setups from the selected geometry in the NODES dock or
  the radial menu; named knob snapshots on every Fluids node, loaded as one undoable change.
- **GPU smoke tools.** Shape controls (disturbance, shredding, turbulence, confinement, field-limited
  dissipation) in the resident GPU solver; GPU-guided up-res at 2× and 4× (96³→192³: 0.24 s on the GPU
  against 0.70 s on the CPU); simulation stats for the selected fluid in the 3D viewport; VDB files that
  hold density, temperature, fuel and velocity together, read back by Blender with matching values.
- **Liquids on the GPU.** Temperature-aware viscosity (96³ dam break: 1.16 s CPU, 0.60 s GPU) and
  whitewater potentials and motion (200,000 particles: 7.59 s CPU, 2.98 s GPU) run on the GPU; auto
  prefers an available GPU.
- **Big grids.** Smoke up-res on compact active tiles on the CPU and GPU: at 512³ the sparse GPU up-res
  took 0.65 s against 1.86 s dense and stored 67 MiB of density against 512 MiB. The path tracer skips
  empty smoke cells with coarse majorants (a warmed 256³ plume: 1.20 → 0.40 ms per sample, image change
  below baseline noise).
- **GPU collisions and deforming colliders.** 200,000-particle self-collision under 40 ms after warm-up,
  checked as a settled pile against the CPU solver; deforming meshes push particles and smoke through
  their surface motion.
- **Rigid bodies.** Box, sphere, convex-mesh and compound bodies with gravity, density-derived mass,
  friction, restitution, torque and sleeping; bodies float on FLIP liquid, and moving bodies stir smoke
  as animated colliders. A deterministic CPU solver, with cached frames advanced incrementally.
- **Optical flow, continued.** `OFlow` retimes within an input range by speed or frame,
  `VectorToMotion` turns `SmartVector` layers into `VectorBlur` and `MotionBlur` input, and `MotionBlur`
  blurs a still beauty from camera motion over depth. Depth and vector sampling in the motion blur
  nodes match the reference.
- **Tracked warps and `LevelSet`.** `GridWarp` driven by selected `Tracker` points holds its grid through
  untracked frames; `GridWarpTracker` fits Tracker motion globally and blends inverse-distance local residuals, or advects each point by `SmartVector` layers. Both drives reject occluded and forward/backward-inconsistent samples and hold each point’s last valid motion; `LevelSet` measures,
  grows and shrinks mattes.
- **2D, the partial rows.** Tangent handles and four interpolations in the shared curve editor (and
  `CrossTalk`, `HueCorrect` and `ColorLookup` curves now actually bend the picture); a second wireable
  output on `ShuffleCopy`; viewer handles for `Sampler` and `MinColor`; `CopyBBox`; Flame and Lustre
  `.3dl` LUTs in `Vectorfield`; `CurveTool` min-luma, max-luma and exposure-difference analysis;
  `HSVTool` colour replacement; `DustBust` scans a frame range and lists suspected specks to accept or
  reject; `RotoPaint` brushes follow their size and direction, dodge and burn have a strength, and a
  Layers list edits any stroke or shape after the fact; `MatchGrade` keeps working with its reference
  disconnected once baked; `Erode` (filter) gains triangle and quadratic kernels and a real fractional
  falloff; `ContactSheet` takes 32 clips with row and column order, centred padding and a frame-range
  mode; `Blend` takes sixteen inputs and its tile-path mask slot is fixed; `Encryptomatte` writes
  ranked Cryptomatte layers from up to eight named mattes, read straight back by `Cryptomatte`.
- **Fixes.** Hovering a noodle raised an error after the output-name change. The top toolbar keeps
  Check for updates visible at 1100 pixels and collapses less-used items into a More menu.
- **Under the hood.** UI tests share event-pumping waits, so the suite no longer starves the render
  thread: the desktop module runs in 65 s instead of 349 s, the full suite in about 36 minutes.

## Known limits

- **Toolbar on Windows at 800 pixels.** With every collapsible item already in the More menu the fixed
  items are still wider than an 800-pixel window on the Windows test runner, so Check for updates moves
  into Qt's own overflow. Seen in the automated tests only; not yet looked at on a real Windows display.
- **GPU path tracing of splats and smoke** runs on NVIDIA and AMD; on Microsoft's software driver (the
  Windows fallback) it stays on the CPU, untested for lack of a Windows machine.
- **Path tracer:** texture maps are in the path tracer only; the fast GPU renderer draws none of them.
  The denoiser has no knobs yet, and depth of field and motion blur are not in the viewport.
- **Viewport:** shadow maps cover up to four shadow-casting lights at once (brightest first; beyond
  that the dimmest are dropped, noted on the status line), and opaque meshes, splats and Instance3D
  copies all cast. 100,000 Instance3D copies under all four shadowed lights at once measured ~10.6 fps
  on an RTX 3080 Ti, short of the 30 fps target (10,000 copies holds it; the cost scales close to
  linearly with instance count, so it is real per-instance shadow-pass work, not a fixed overhead).
- **Rigid bodies:** the liquid feels a body's push in the returned scene, but that impulse does not
  change later upstream FLIP checkpoints. No fracture.
- **Fluids:** render consumers still expand sparse tiles to dense arrays; a deforming collider uses the
  mean surface velocity within a large triangle; Houdini compatibility of the VDB files is unverified.
- **2D:** `Blend`'s fringe, inject and per-channel mask choices are not built, and its input count is a
  fixed sixteen where Nuke is unlimited. `ContactSheet`'s smaller gaps are written in the parity docs.
- **Windows:** one `BurnIn` check is still skipped there. The combustion documentation check now reads
  as UTF-8 on Windows; that was a test-only failure.
- **Verified by automated tests only**, on Linux (offscreen Qt and an RTX 3080 Ti; the fluid and particle
  GPU work also on an AMD Radeon 8060S and llvmpipe). Nobody has driven the new nodes, effects, presets or
  the rigid bodies on a real display, and none of the GPU work has run on a real Windows GPU.

## Moved to 0.33

Denoiser controls, depth of field and motion blur in the viewport, and instances drawn in the viewport;
`TimeWarp`, and `TimeBlur` and `TimeEcho` on the tile path; named layers on the tile path (`Shuffle`,
`ZDefocus`, `ZMerge` tiled); persistent two-way rigid-body and liquid coupling; textures in the fast GPU
renderer; GPU path tracing of splats and smoke on the Windows software driver.

# NodeBased 0.31.0 — splats, smoke and fire in the GPU path tracer with a denoiser, depth of field and motion blur, dockable panels and workspaces, tracking and Stabilize, optical flow retiming, warps, whitewater and a render-ready liquid mesh

## What changed since 0.30.0

- **Splats, smoke and fire in the path tracer, on the GPU.** `Render3D`'s `pathtrace` mode now renders
  Gaussian splats, smoke and fire together with meshes. Splats are lit by every light and the dome, cast
  shadows, bounce light and show in mirrors and through glass. Smoke is lit and shadows meshes; fire
  lights the meshes, splats and smoke near it. A room with 40,000 splats and a smoke puff costs about
  half a second per sample at 1080p on an RTX 3080 Ti. The GPU version runs on NVIDIA cards only for now;
  see Known limits.
- **A denoiser.** A new `denoise` output filters the path-traced picture, and the EXR can hold the raw
  beauty beside albedo, normals, depth and the filtered image. At 16 samples it cuts the error about
  2.1 times with splats and smoke in the picture and leaves the average within 0.1 percent.
- **Depth of field.** `Camera3D` gains an f-stop, a focus distance and a bokeh shape (round, or 3 to 16
  blades, rotated or squeezed), in the path tracer and the ray-traced mode. Imported Alembic and USD
  cameras bring their own values. Clicking in the 3D viewer sets the focus distance from the surface
  under the cursor.
- **Motion blur in `Render3D`.** Moving objects, moving cameras, animated Alembic and USD meshes, liquid
  surfaces, instances and particles blur across a shutter. Multichannel and relight renders blur too:
  colour and lighting layers smear, depth and normals stay sharp from the middle of the exposure. A new
  `motion` output gives per-pixel vectors for `VectorBlur`.
- **Dockable panels and workspaces.** The viewer, graph, properties and other panels can be dragged,
  floated and resized, and the window shrinks to 800 × 500. Named workspaces save and restore a layout,
  a two-monitor preset is included, and Ctrl+Space maximises the panel under the pointer.
- **Tracking and `Stabilize`.** Place and drag up to 16 track points with their pattern and search
  boxes, analyse forward or backward with confidence-coloured paths, key or clear points over a frame
  range, stabilise the plate, and export an animated `Transform` or `CornerPin`.
- **Optical flow.** `VectorGenerator` writes forward and backward motion vectors on the CPU or the GPU
  (both directions at 1080p: 4.34 s on the CPU, 1.34 s on the GPU). `Kronos` retimes with motion
  interpolation or frame blending, and `MotionBlur` adds blur from the footage's own motion.
- **`SmartVector`, `VectorDistort`, `VectorCornerPin` and flow-guided `Inpaint`.** Cache motion layers
  over a frame range, carry paint and corner pins along that motion, and fill a matte from nearby frames
  before falling back to diffusion.
- **`SplineWarp` and `GridWarp`.** Draw and edit paired curves or a grid in the viewer, animate the
  points, warp with mix and mask, and send a `SplineWarp` result to `STMap`.
- **Liquids.** Viscosity, including temperature-driven viscosity (cold liquid thickens up to ten times),
  and a narrow-band mode that keeps particles only near the surface. Animated colliders stir liquids.
  `FluidWhitewater3D` emits and caches foam, spray and bubbles from trapped air, wave crests and kinetic
  energy (Ihmsen et al. 2012), each with its own motion and lifespan.
- **A render-ready liquid mesh.** The liquid surface is smoothed across neighbouring frames, can carry
  up to four times the detail, keeps thin splash sheets closed, carries vertex motion into `Render3D`
  for motion blur, and exports OBJ and VDB sequences.
- **Preset browser and particle inspection.** Browse shipped and user presets with search, category
  filters and thumbnails from the NODES dock or the radial menu, save a selection as a preset, and click
  a particle in the viewport to read its id, age, velocity, speed, size and colour.
- **A faster cubic filter.** `Transform`'s cubic filter is 4.5 times faster at 960 × 540 and 4.7 times
  at 1920 × 1080 with black outside (3.2 and 3.3 times with clamp), with identical pixels. A rotated,
  1.5× scaled 960 × 540 image redraws in 0.16 s, down from 0.91 s.

## Corrections to the 0.30.0 notes

- 0.30.0 said cubic was "about ten times slower" than nearest, at about 1 second a frame. Most of the
  6-second redraw measured then came from the test harness, which starved the render thread while it
  waited. The real redraw was 0.91 s.

## Known limits

- **GPU path tracing of splats and smoke is NVIDIA only for now.** On an AMD integrated GPU and on
  Microsoft's software driver the GPU version gave wrong pictures in the automated tests, so on any adapter
  that is not NVIDIA a scene with splats, smoke or fire is path traced on the CPU (correct, and much slower).
  Meshes alone still path trace on any GPU. The cause is not found yet.
- **Path tracer:** no textures, no caustics, and lights are not seen directly by the camera. The
  denoiser has no knobs yet. Large smoke boxes are slow.
- **Depth of field and motion blur** are in the path tracer and the ray-traced mode; the viewport does
  not show them yet.
- **Optical flow and warps:** `SmartVector` and `Inpaint` select CPU or GPU flow with `flow_backend`;
  `VectorDistort` and `VectorCornerPin` warp on GPU when driven by SmartVector's GPU path. `Inpaint`
  offers diffusion and confidence/structure-priority patch-based spatial fill. OFlow, VectorToMotion and GridWarpTracker are implemented; MotionBlur3D still lacks its depth-vector path.
- **Liquids:** the viscosity solve runs on the CPU (it works with CPU or GPU pressure).
- **Cubic filter:** with clamp the speed-up is 3.2 to 3.3 times, short of the 4 times target. Nearest
  and bilinear are unchanged.
- **Panels:** the two-monitor preset and floating panels have been tested offscreen only. Placement on
  a real second monitor is unchecked.
- **Toolbar on Windows:** the top toolbar has grown, and on the Windows test runner its contents are wider
  than a 1440-pixel window, so Check for updates (the last item) moves into the toolbar's overflow menu.
  Seen in the automated tests only; not yet looked at on a real Windows display.
- **Windows:** one `BurnIn` check is still skipped on Windows (frames 1 and 2 burned in identically on
  the CI runner), so the burned-in frame number may not update there.
- **Verified by automated tests only**, on Linux (offscreen Qt and an RTX 3080 Ti). Nobody has driven
  the new panels, tracking, warps, preset browser or particle tooltip on a real display, and the GPU
  work has not run on a real Windows GPU.

## Moved to 0.32

A viewport close to the final render, then lit and shadowed particles and instances with materials;
light effects (`Flare`, `Glint`, `Sparkles`, `GodRays`, `ScannedGrain`); per-node colour management
(the OCIO nodes); combustion, then fluid presets and one-click shelf setups; and the 2D-to-3D pipe (on
hold).

# NodeBased 0.30.0 — a GPU path tracer, PBR materials and area lights, the node toolbar and radial menu, groups, glowing fire and refractive water, and dozens of new 2D nodes

## What changed since 0.29.0

- **A path tracer, on the GPU.** `Render3D`'s Mode gains `pathtrace`: soft shadows, colour bleeding,
  mirror and glass, lit by area lights, the usual lights and an HDRI sky with a sun. Backend `gpu` runs
  it on the card (a furnished full-HD room costs about 43 ms per sample on an RTX 3080 Ti), `cpu` on the
  reference. It refines in passes, stops on a time limit or when a tile is clean, cancels between passes,
  and writes diffuse and specular indirect light as their own outputs. `Instance3D` copies are traced
  without being flattened.
- **Physically based materials and area lights.** A mesh's Material knob gains `pbr` (Metallic,
  Roughness, Specular), the same shading relit splats use, so a mesh and a splat under one light match.
  `Light3D` gains Rect, Disc and Sphere types with real size (bigger means softer shadows), fixed power or
  plain glow, one or two sided, coloured directly or by Kelvin.
- **Fire that glows and smoke that scatters.** Flames emit blackbody light from the temperature
  channel (with an optional kelvin ramp) and light the smoke around them; smoke gains backlit
  anisotropy, a multiple-scattering approximation and Pyro-style look controls, on the CPU and GPU.
- **Water that refracts.** A Liquid material on meshes and `FluidSurface3D`: refraction and Fresnel
  reflection with total internal reflection, Beer-Lambert absorption, thin sheets and light glints, on
  the CPU ray tracer and the GPU ray tracer (within 1e-5 of each other), with a raster-mode approximation.
  `ParticleRender3D` gains a foam representation (`foam_density`, `spray_size`) composited with the water.
- **Pyro production pass.** `FluidCollide3D` gains `Animated`: colliders that move during the solve and
  push smoke or liquid aside, on every solver (it replaces Velocity from motion; old files still load).
  `FluidSolver3D` gains Shape controls: disturbance, shredding, turbulence with size and pulse,
  confinement, and dissipation limited to a field range. `FluidUpres3D` adds factor-2 or factor-4 detail
  over a coarse cache, advanced through cached frames with fuel carried and mass held steady.
- **`WriteVDB3D`.** Smoke, fire and liquid caches go out as sparse OpenVDB files for Houdini and Blender,
  one frame or a range, with an overwrite guard.
- **Particles that collide.** `ParticleBounce3D` and `ParticleEmitter3D` gain animated geometry (moving
  colliders and emitters, with Inherit velocity), and the new `ParticleCollide3D` makes particles bump
  and pile up with bounce, friction and a settle control.
- **`Instance3D` and `Shrinkwrap3D`.** Copy a mesh (or up to eight variants) onto every particle or point,
  sized, oriented and tinted per point, sharing memory, on the CPU renderers and the GPU ray tracer.
  `Shrinkwrap3D` fits a UV'd proxy (built for you, or your own) tightly onto any mesh.
- **Cryptomatte out of `Render3D`.** A Cryptomatte switch writes CryptoObject, CryptoMaterial and
  CryptoAsset layers that the `Cryptomatte` node reads, one id per instance.
- **Groups.** `Group`, `Input` and `Output` nodes nest graphs; Ctrl+G groups the selection, Ctrl+Shift+G
  ungroups, double-click enters, and a Root > Group breadcrumb bar leads back out.
- **The NODES toolbar.** A left-side dock lists every node in Nuke-style categories with one-line
  descriptions; search by name or by what a node does, click or drag to add. Favourites, the last ten
  nodes added, Ctrl+F and arrow-key search, a compact mode, and "What is this?" on any node. Tab search
  shows categories too.
- **The radial menu.** Hold Q over the graph for a ring of up to eight commands that changes with the
  selection; flick and release, or tap to pin it open. Write your own commands (graph edits or a small
  Python script) in Preferences > Radial commands. The ring learns what you pick and gives it a fixed
  slot; right-click locks a slice; Preferences > Radial settings changes the key and the flick distance.
- **Presets and shelf tools.** Eight particle presets (sparks, debris burst, dust puff, rain, snow,
  leaves in wind, fountain, magic trail) on a generic preset format, and shelf commands that turn a
  selected 3D node into an emitter or collider, scatter onto it, or add forces.
- **Artist tools for simulations.** A slice viewer (any field, a movable cut plane, velocity arrows, the
  value under the cursor) and a cache inspector (every cached frame with stats, wipe from a frame,
  re-solve a range, open the folder), both following the selected node.
- **A cleaner properties panel.** One shared layout with an editable header and view, reference, help,
  revert and close icons. `Transform` gains skew, per-axis scale, invert, edge modes, shutter motion
  blur, per-knob animation and cubic filtering.
- **Image metadata.** `Read` fills it and `Write` writes it; `ViewMetaData`, `ModifyMetaData`,
  `CopyMetaData`, `CompareMetaData`, `AddTimeCode` (with drop-frame) and `BurnIn`.
- **Many new 2D nodes.** Filters: `Convolve`, 5x5 and 7x7 `Matrix`, `EdgeDetect`, `Emboss`, `BumpBoss`,
  filter `Erode`, `Bilateral`, `Denoise`, and `ZDefocus` with disc, polygon or image bokeh. Colour:
  `Log2Lin`, `PLogLin`, `CrossTalk`, `Toe`, `Expression`, `Histogram`, `Sampler`, `MinColor`, `HistEQ`,
  `MatchGrade`, plus editable curves for `HueCorrect`, `ColorLookup` and `CrossTalk` and a new
  `ShuffleCopy`. Depth and layers: `ZMerge`, `ZSlice`, `Remove`. Time: `TimeBlur`, `TimeEcho`,
  `TimeDissolve`. LUTs: `Vectorfield` reads .cube files and `GenerateLUT` exports one; `Tile`. Also
  `MotionBlur2D`, `MotionBlur3D`, `CurveTool` and `ContactSheet`.
- **RotoPaint and DustBust.** Pressure-aware paint, clone and reveal strokes in the viewer, in the same
  layers as Roto shapes, that can follow a Tracker; DustBust makes single-frame clone dabs. Old Roto
  projects keep their look. The tracker gains backward tracking, channel choice and up to 16 points.

## Known limits

- **Path tracer:** no textures, splats, particles or volumes yet, lights are not seen directly by the
  camera, no caustics, and no denoiser. The HDRI sky sampling lives only in `pathtrace`; other modes keep
  the old sky.
- **Materials and lights:** PBR has no textures or clearcoat and is not read from glTF or USD; PBR and
  the new area lights render on the CPU (the GPU raster path falls back). Splats, smoke and 2D-comp
  lighting skip area lights.
- **Fluids:** `FluidUpres3D` runs on the CPU only, with no GPU path and no timing on the 3080 Ti yet; the
  fully GPU-resident solver refuses the Shape controls. `WriteVDB3D` writes one volume or liquid per
  file and is unchecked in Houdini; Blender read the file's layout but not its smoke values.
- **Particles and instances:** particle-to-particle collision is CPU only (about 700 ms a frame for
  20,000 packed particles); a deforming (not just moving) collider is not fully tracked. `Instance3D`
  copies are not drawn in the 3D viewport, and a GPU scene mixing instances with other objects falls back
  to the CPU. `Render3D` Cryptomatte runs on the CPU reference and leaves out volumes.
- **UI:** presets can only be placed through Python (no preset browser or Save selection as preset);
  clicking a particle in the viewport shows no tooltip yet; the sim stats overlay is not in the live
  viewport. The radial menu's "Describe it" is not built.
- **2D:** curve points have no tangent handles; `ShuffleCopy`'s second result is an `out2` layer, not a
  second output. `MinColor` and `Sampler` have no viewer handles. `ZDefocus`, the depth, time and
  temporal nodes use the full-frame evaluator. `Denoise` is not Nuke's algorithm. No `.3dl` LUTs, no
  automatic speck detection in DustBust, no `ScannedGrain` or `CopyBBox`. On Windows, one BurnIn check fails unexplained: frames 1 and 2 burned in identically on the CI runner, so the burned-in frame number may not update there; that check is skipped on Windows while it is investigated. `Transform` now defaults to the cubic filter (as Nuke does; older scripts keep nearest), and cubic is currently about ten times slower to render than nearest on the CPU (about 1 second a frame against 0.1 for a 960×540 image); set the filter to nearest for speed while a faster cubic is on the way.
- **Verified by automated tests only**, on Linux (offscreen Qt and an RTX 3080 Ti; 3223 tests on the
  last green merge). Nobody has driven the path tracer, radial menu, toolbar or RotoPaint on a real
  display, and the GPU work has not run on a real Windows GPU.

## Moved to 0.31

Dockable, floatable panels and saved workspaces, then the rest of tracking (draggable boxes, a
`Stabilize` node); splats, volumes and liquids in the path tracer with a denoiser, then depth of field,
motion blur and the rest of the production-look plan; Liquids 2 (viscosity, narrow-band FLIP,
whitewater, a render-ready mesh), then combustion; optical flow nodes (`VectorGenerator`, `Kronos`,
automatic motion blur; only the solver foundation is in) and the `SplineWarp` and `GridWarp` nodes (only
the warp-field foundation is in); and the 2D-to-3D pipe (on hold).

# NodeBased 0.29.0 — smoke, fire and liquids you can simulate and render, GPU volumes, keyers and Cryptomatte, multichannel EXR

## What changed since 0.28.0

- **Volumes are a scene member.** A `Volume` sits in a 3D scene next to meshes, splats and particles;
  `Plume3D` makes a procedural one. `Render3D` draws volumes with a lit CPU reference raymarch (lights,
  shadows inside the smoke, correct hiding by meshes in front of or inside it) and writes volume control
  passes: `volume_density`, `volume_motion`, `volume_temperature`, `volume_vorticity`, depth, and a new
  `volume_id` that numbers each volume for masking.
- **`ReadVDB3D` on an in-house OpenVDB reader.** Point it at a `.vdb` from Blender (or a numbered
  sequence such as `smoke.0001.vdb`) and pick which grids feed density, temperature and velocity from a
  list filled from the file; frame offset, voxel scale and a normal transform block. The reader handles
  float, half and Vec3f grids, zip and Blosc-LZ4 buffers and streamed files, with no OpenVDB dependency.
- **A 3D smoke and fire solver with Houdini Pyro's vocabulary.** `FluidSource3D`, `FluidForce3D`,
  `FluidCollide3D`, `FluidSolver3D` and `FluidCache3D`: a MAC-grid solver with semi-Lagrangian and
  MacCormack advection, buoyancy, vorticity, fire that burns fuel and expands, per-axis open boundaries
  and voxelised colliders that carry their motion. Forces and sources are in world units. Every solved
  frame is a checkpoint on disk, so scrubbing back never re-solves and a project resumes where it stopped.
- **The whole smoke solve can run on the graphics card.** `FluidSolver3D`'s pressure knob gains
  `resident` and `resident_sparse` (auto picks resident from one million cells when it fits the card):
  GPU multigrid pressure, GPU advection and active 8-cubed sparse tiles that grow and retire with the
  flow. On an RTX 3080 Ti a 128-cubed step takes 15 to 20 ms and a 256-cubed step 80 to 145 ms, so 100
  frames is seconds of card time where the CPU took about 2.4 hours. Results match the CPU solver and
  re-solve bit-identically.
- **FLIP liquids.** A liquid `FluidSource3D` into `FluidLiquidSolver3D` gives a particle liquid (FLIP/PIC
  blend, free surface, colliders, 3 to 12 particles per cell kept by thinning and refilling) that the
  existing particle renderers draw and `ParticleCache3D` caches. `FluidSurface3D` builds a closed,
  smoothed water mesh with outward normals for the mesh renderers; `FluidFoam3D` picks out splash
  droplets. A dam break keeps its volume and settles flat, and results repeat exactly.
- **Volumes on the GPU, in renders and the viewport.** `Render3D` on the GPU (or auto) draws smoke that
  matches the CPU reference, lit by every light including spot cones, with shadow rays and depth
  compositing against meshes. A 1080p frame that took the CPU minutes takes 0.03 to 0.11 s. The 3D
  viewport shows volumes live, coarse while you orbit and finer with the V toggle.
- **Volume motion blur and shadows between smoke and scene.** A shutter smears smoke along its own
  velocity field on the CPU and GPU. Meshes and casting splats shadow the smoke; the smoke shadows
  meshes, relit splats and the shadow catcher, with `shadow_density` scaling both ways. The volume
  control passes now also run on the GPU.
- **Multichannel EXR out of `Render3D`.** A `multichannel` output writes every pass (beauty, data
  passes, volume passes) into one EXR through `Write`, and `Read` brings it back layer by layer.
- **Splat relighting, properly.** `Delight` estimates the light a capture was photographed under and
  divides it out into albedo, roughness, cleaner normals and a crease term (off by default; see Known
  limits). De-lit splats shade with Cook-Torrance GGX under every light and a new `Environment` light
  type (wire a panorama, turn or blur it), take shadows from meshes and splats, and reflect nearby
  meshes. `Indirect samples` adds ray-traced occlusion and one bounce of coloured light, with a denoiser
  that touches only the bounce and reflections and quality presets from preview to final. A benchmark
  (`tools/benchmark_relight.py`) scores relighting against scenes with known answers.
- **Keyers.** `ChromaKeyer` (key colour, tolerance, softness, luminance gain, shadow and highlight
  handling, despill with bias), the IBK pair (`IBKColor` builds a clean screen plate, `IBKGizmo` keys
  against it) and `ScreenKeyer`, a Keylight-style screen-difference keyer with screen gain and balance,
  clip black/white/rollback, shrink/grow, softness and four views. All take mask and mix and match on
  the tiled path.
- **`Cryptomatte`.** Pick objects from a renderer's Cryptomatte EXR by name or raw id, or click them in
  the viewer, and get their matte with soft edges intact; views for the matte alone and every object in
  its own colour. Files that carry only Cryptomatte layers now read.
- **Fourth and fifth 2D parity passes.** `EdgeBlur`, `EdgeExtend`, `LightWrap`, `Dither`, `Grain`,
  `Posterize`, `SoftClip`, `HSVTool`, `AddMix`, `Blend` (up to eight inputs), `CopyRectangle`, `Matrix`
  (3x3, optional normalise) and `Laplacian`, all with mask and mix on the tiled path. `Shuffle` picks any
  named layer; `STMap`, `IDistort` and `VectorBlur` warp or blur by uv maps and motion-vector layers.
- **Conditioning bundles.** `Write`'s bundle option puts a per-frame JSON manifest beside each EXR
  (layers with units and direction, frame, camera, a document fingerprint); `ReadBundle` brings a model's
  output back at the same frame and refuses a frame or size mismatch.
- **`Backdrop` and `PostageStamp`.** A tinted, labelled, resizable box behind nodes whose title drag
  carries the nodes inside (created around the selection), and a node that shows a live thumbnail of its
  input on the graph, with a switch that hides its input wire.

## Known limits

- **Fluids:** fire renders as smoke with a temperature channel only (flames do not emit light yet); FLIP
  has no surface tension and its particle work stays on the CPU (a 128-cubed step takes about 2.3 s with
  the GPU pressure solve); no refraction for the water mesh. Sparse tiles only pay while the flow covers
  under half the box. `ReadVDB3D` has not been tried on a file written by Houdini itself and refuses
  level sets, camera-space and integer grids. Reading a full 256-cubed frame back is the slowest part of
  a GPU bake.
- **Volumes on the GPU:** renders that mix smoke with splats or use the ray tracer fall back to the CPU;
  the mesh shadow on smoke is hard-edged.
- **Splat relighting:** `Delight` misses its target on the sphere-on-ground scene (the cast shadow reads
  as paint) and stays off by default; indirect rays run on the CPU (a 9,000-splat frame takes about 23 s
  at final); no sky shadows from the environment and splats do not reflect splats; nothing measured on a
  real capture.
- **2D:** no `ScannedGrain`, `Toe`, `CopyBBox`, Convolve, 5x5 or 7x7 `Matrix`, `EdgeDetect` or `Emboss`
  yet. Keyers lack IBKGizmo's autolevels and clip knobs, Keylight's clean-plate and core-matte inputs,
  and Primatte/Ultimatte entirely; thresholds and grain strength are a reading of Nuke, unchecked on
  real footage. `Cryptomatte` has no Encryptomatte, sidecar manifests or half-float files, and
  `Render3D` does not write Cryptomatte layers yet. `Shuffle`, `STMap`, `IDistort` and `VectorBlur`
  render whole-image only, and their layer fields are text boxes.
- **Verified by automated tests only**, on Linux (offscreen Qt and an RTX 3080 Ti; 2654 tests on the
  last green merge). Nobody has watched the fluid nodes, GPU volumes, keyers or relit splats on a real
  display, and the GPU work has not run on a real Windows GPU.
- **Still true from 0.28.0:** particle turbulence does not move over time and particle colliders do not
  move; `WriteSplat3D` drops relight and shadow settings; the Viewer's wipe position is not saved; the
  format registry has no editor; gizmo rings point along world axes; the 3D viewport shows splats as
  opaque discs; `ReadGLTF3D` loads meshes only.

## Moved to 0.30

Fire that glows (blackbody emission, fire light on the smoke), multiple scattering, backlit anisotropy
and Pyro look presets in `Render3D` (built on a lane branch, not merged yet); the group UI (grouping
selected nodes, entering a group, the breadcrumb bar); the remaining 2D filters and a colour pass; Cryptomatte layers out of
`Render3D`; the next fluids plan (VDB out, a Pyro production pass, Liquids 2), which waits on DiMo's
choice of order; and the 2D-to-3D pipe (on hold).

# NodeBased 0.28.0 — a particle system, Spot lights and GPU shadows everywhere, a Nuke-style Viewer, WriteSplat3D

## What changed since 0.27.0

- **A particle system, eight nodes deep.** `ParticleEmitter3D` runs a deterministic solver (an emitter
  model with sub-frame births, seeded per substep, so the same document always produces the same
  particles) and `ParticleCache3D` writes every frame to the disk-backed simulation cache, so scrubbing
  never re-solves; the cache is keyed by a run identity that changes whenever anything upstream would
  change the motion. Forces chain on the stream: `ParticleGravity3D`, `ParticleDrag3D`, `ParticleWind3D`
  and `ParticleTurbulence3D`, each with a seeded probability and a frame window. `ParticleBounce3D`
  sweeps each substep's segment against a collider's BVH so nothing tunnels at any timestep, with
  restitution, Coulomb friction and kill-on-contact. `ParticleRender3D` chooses how they draw: points,
  spheres or camera-facing cards with an optional sprite image and a size scale, on its own node so
  drawing choices never change the run identity. Budgets and bypass follow the SIMULATION rules.
- **Particles render on the GPU and show in the 3D viewport.** `Render3D`'s CPU path draws them as
  size-scaled discs, spheres or cards; the GPU raster path draws the same three as one instanced pass
  over the mesh pass and matches the CPU render, and `auto` now picks the GPU for particle scenes. The
  editor viewport draws particles on both its GPU and CPU paths.
- **`Spot` lights actually light like spots.** The 0.27 cone angle, penumbra, cone falloff and distance
  falloff now apply on the CPU reference, both GPU paths and relit splats; the editor viewport draws the
  cone and its falloff. `Directional` and `Point` with no falloff render byte-identically to before.
- **Shadow bias, blur and samples per light.** `Light3D` gains `shadow_bias` (the old scene-scaled
  epsilon as the default), `shadow_blur` (the light's half-angle in degrees) and `shadow_samples`
  (jittered rays seeded by a hash of the point, so renders are reproducible and chunk independent).
  Mesh shadows in both CPU modes, the splat shadow paths, and the GPU raster and ray-traced shaders all
  use them; old documents load with the defaults and render byte-identically.
- **GPU shadows on relit splats and shadow catching.** Meshes and splats shadowing relit splats, and the
  caught shadow of `Catch shadows`, are traced on the GPU through the ray tracer's BVHs by a new
  visibility entry point that shares the CPU reference's cache, candidate rule, soft-shadow jitter and
  bias. Raster-mode splat scenes with a shadowed light route through the ray-traced GPU path. The two
  "CPU-only" clauses from 0.26 are gone, replaced by GPU-versus-CPU parity tests.
- **`Keep specular`.** Splat relighting and shadow catching can keep the capture's own specular instead
  of flattening it away; see Known limits for what is still missing.
- **A normals pass for splats.** A `normals_blend` pass (alpha-weighted splat normals), camera-facing
  orientation, and a `Smooth normals` knob on `ReadSplat3D` that averages over the k nearest splats;
  relighting and the normals pass both use it.
- **`WriteSplat3D`.** Writes a scene's splats to a 3D Gaussian Splatting PLY with the scene transforms
  baked in, with an overwrite guard and a tested round trip through `ReadSplat3D`. Like `WriteGeo3D` it
  writes only on the button and passes its input through.
- **`HueCorrect` and `ColorMatrix`.** Six hue anchors with smoothstep interpolation and a `hue_shift`;
  a 3x3 colour matrix with invert. Both take mask and mix and run on the tile path.
- **`TimeClip`, `FrameRange` and `AppendClip`.** Range mapping with hold, loop, bounce or black outside
  the range, and head-to-tail clips with a dissolve. A bypassed time node now digests correctly.
- **A document-wide format registry, `Grid` and `NoOp`.** Named formats live in `settings.formats`
  (seeded with the built-in `Reformat` list, added by an additive upgrade); `Reformat` resolves against
  the registry first and its node-local list second, and a new `format` document op (set, rename,
  delete) keeps every `Reformat` naming the entry in step, in one undo unit. `Grid` draws a grid by
  spacing or line counts with offset, line width, colour, optional image and mask. `NoOp` is a
  passthrough with a note knob.
- **The Viewer grows up: inputs, A/B and wipe.** Keys 1 to 9 connect the selected node to a viewer
  input; A and B buffers with compare modes A only, B only, wipe, over, under, minus and difference.
  B renders inside the same display request as A so both share the frame, and the wipe is clipped from
  the already-evaluated B picture, so dragging it evaluates nothing.
- **Viewer gain, gamma, zebra and display.** Display-only gain and gamma (each with a reset), a zebra
  toggle that stripes pixels above 1.0 and below 0.0, and a Display menu offering the project view or
  any view the ACES config provides, including Raw. All of it is saved as viewer state in the project;
  the pixel readout keeps showing the real scene-linear numbers.
- **Region of interest, proxy and format masks.** An ROI box that clips the tile request, a saved proxy
  tier stepping down to 1/8 with an on-screen badge, and display-only format masks from 1.33 to 2.40 as
  lines, half or full.
- **Fluids groundwork.** A survey of OpenVDB, Mantaflow, PhiFlow and Taichi for packageability on both
  platforms, a 2D MAC-grid smoke solver spike with CG projection and a `wgpu` red-black SOR pressure
  solve with a parity test, measured per-substep costs, and a written verdict: build our own solver on
  top of a volume scene member that is built first, with import-only as the fallback. Nothing user-facing
  ships from this yet; it is the plan the fluid nodes will follow.

## Known limits

- **Particles:** turbulence does not move over time; colliders do not move; particles do not keep their
  size off a surface; no mesh instancing. Particles with the GPU ray tracer, or in the same scene as
  splats, still render on the CPU. The viewport draws at most 250,000 particles per set.
- **`Keep specular` is partial:** the relight bundle still refuses splats, so there are no per-light
  specular layers for them, and splats have no shiny material, so no highlight is computed from the
  lights. **Splat normals** cannot yet feed the relight bundle or the 2D `Relight` node for the same
  reason, and the blend pass has no GPU version.
- **Splat shadows on the GPU** leave per-splat shading and the shadow cache on the CPU; transparent
  meshes together with splats, the data passes and the splats output stay CPU-only.
- **`WriteSplat3D` drops** relight, shadow, specular and smoothing settings, per-splat visibility and
  colour space, and writes nothing that is not a splat (listed in `docs/3D_FOUNDATION.md`).
- **Viewer:** the wipe position is not saved; export uses A; the ROI is ignored on nodes that cannot be
  evaluated in tiles (they show the whole frame); the "Format" mask is the frame's own shape. The
  toolbar is getting crowded and nobody has looked at it on a real display.
- **2D:** the document has no frame range per branch, so `FrameRange` works by remapping frames; the
  format registry has no editor in the app yet (formats are edited through the document op); `Grid` has
  only straight, hard-edged lines.
- **Verified by automated tests only**, on Linux (offscreen Qt and an RTX 3080 Ti; 2080 tests on the
  release base). Nobody has driven particles, the Spot cone, the new Viewer or the splat writer on a real
  display, on Linux or Windows.
- **Still true from 0.27.0:** gizmo rings and cubes point along world axes; the GPU work has not run on a
  real Windows GPU; the 3D viewport shows splats as opaque discs; `ReadGLTF3D` loads meshes only;
  `Reformat`, `CornerPin` and the picture-area nodes render whole-image only; `MergeGeo3D` keeps only
  the first input's colour, texture and material.

## Moved to 0.29

Multichannel EXR out of `Render3D` (written on a lane branch, waiting on a green full suite), the fourth
2D parity pass (`EdgeBlur`, `EdgeExtend`, `LightWrap`, `Dither`, `Grain`, `Posterize`, `SoftClip`,
`HSVTool`, `AddMix`, `Blend`, `CopyRectangle`; the first four are on the same lane branch), a layer-aware
`Shuffle`, a Formats editor in project settings, `ZDefocus` and a depth channel, `ChromaKeyer`, per-light
specular for splats in the relight bundle, the volume scene member and the first fluid nodes, and
`ImageToSplat` through the runtime manager with the rest of the 2D-to-3D integration (on hold).

# NodeBased 0.27.0 — Nuke's full Merge set, a real Reformat, camera and light tooling in 3D

## What changed since 0.26.0

- **`Merge` and `ChannelMerge` have all 30 of Nuke's operations.** The eleven that were missing land
  here: `matte`, `disjoint-over`, `conjoint-over`, `copy`, `exclusion`, `geometric`, `overlay`,
  `hard-light`, `soft-light`, `color-dodge` and `color-burn`, each checked against Foundry's documented
  formula on hand-worked pixels, with every division guarded so zero alpha and HDR values never produce
  a broken number. The tiled path matches the whole-image path for every operation.
- **`Reformat` with a real format model.** Named formats (`HD_1080`, `HD_720`, `UHD_4K`, `2K_DCP`,
  `Square_1K` or a custom width, height and pixel aspect), Nuke's `type` (to format, scale, to box) and
  `resize type` (none, width, height, fit, fill, distort), centre, flip, flop, turn, filter choice and
  preserve-bounding-box. It is the first node that changes the output display window itself. The
  format list is stored on the node; a document-wide shared format registry is still to come.
- **`CornerPin`.** Four "from" and four "to" points, forward or inverse, a projective warp sampled with
  the existing filters; pins equal to their sources are an exact identity.
- **Time nodes.** `TimeOffset`, `FrameHold` (Nuke's always-hold default) and a simplified `Retime`
  (nearest frame, no blending) evaluate their input at a remapped frame, and a still stays one cache
  entry across every frame it is asked for.
- **`Soften` and `Exposure`.** `Soften` is a Gaussian-shaped blur distinct from `Blur`'s box filter,
  seamless across tiles. `Exposure` is Nuke's standalone node: stops or densities, a black point that
  stays put, red, green and blue adjustments that can be ganged, plus mask and mix.
- **`Defocus`, `DirBlur` and `DropShadow`.** `Defocus` is a disc-shaped blur with an aspect knob (no
  depth yet, so no `ZDefocus`). `DirBlur` blurs along a direction, spins around a centre or zooms from
  one. `DropShadow` puts an offset, softened, tinted copy of the alpha under the input. All three take
  mask and mix; `Defocus` and linear `DirBlur` are seamless across tiles.
- **`Position`, `BlackOutside` and `AdjustBBox`.** Whole-pixel moves, a one-pixel black border around
  the picture area so filters stop smearing the edge, and growing or shrinking the picture area,
  optionally clipped to the frame. Nuke's bounding-box utilities, on the whole-image path.
- **`MergeGeo3D`, `Normals3D` and `DisplaceGeo3D`.** Merge up to eight shapes into one solid, each keeping
  its own position; recompute (area weighted and welded across seams and poles), flip or unify a shape's
  normals; push a surface in or out along its normals using an image, with the normals recomputed after.
  All three feed the other geometry nodes, export and the scene node.
- **Camera and light handles in the 3D viewport.** `Camera3D` and `Light3D` are pickable through a
  small marker and each gets a position handle and a target handle, so a camera or light can be aimed by
  dragging. One undo step per drag, Escape cancels.
- **`Camera3D` has a film back.** Focal length and horizontal and vertical apertures in millimetres,
  Nuke's model, with the vertical and horizontal field of view shown as read-only readouts. Defaults
  reproduce the old 45 degrees exactly, old documents are converted on load, and cameras read from
  Alembic, USD and glTF keep the lens the file describes.
- **`Light3D` can be a `Spot`.** Cone angle, penumbra, cone falloff and a distance falloff choice (none,
  linear, quadratic, cubic), with one pure attenuation function and tests. No renderer applies the cone
  yet; see Known limits.
- **Rows and columns, `Cylinder3D`, matrix readouts.** `Card3D` and `Sphere3D` take Nuke's `rows` and
  `columns` (a subdivided plane is a `Card3D` with more of both; the sphere's poles no longer carry
  degenerate triangles). `Cylinder3D` is a new primitive with radius, height, rows, columns and open or
  closed caps. Every node that carries a transform shows its read-only local and world matrices in the
  properties panel, following `Scene3D` and `Axis3D` parenting.

## Known limits

- **A `Spot` light lights like a `Directional` light** on every renderer, CPU and GPU: the model and
  knobs ship, the renderer wiring is written up as a request in `docs/3D_FOUNDATION.md` for the
  rendering lane.
- **`Reformat` and `CornerPin` render on the whole-image path only**, never tiled, like `Transform`,
  `Crop` and `Mirror`. `Reformat`'s formats live on each node; two nodes do not share a named format yet.
- **`MergeGeo3D` keeps only the first input's colour, texture and material**, and has eight slots
  rather than unlimited inputs. **Flipping normals does not change a render** because the CPU renderer
  lights both sides. Spinning and zoom `DirBlur` and the three picture-area nodes use the whole-image
  path.
- **`Retime`** samples the nearest frame with no blending. **`Exposure`** has no Lights or Cineon modes
  and no colourspace or unpremult options; Nuke's `mode` knob is called `exposure_mode` here.
- **Verified by automated tests only**, on Linux (offscreen Qt and an RTX 3080 Ti). Nobody has driven
  the new handles, film back or nodes on a real display, on Linux or Windows.
- **Still true from 0.26.0:** gizmo rings and cubes point along world axes; relit splats under a shadowed
  light, the shadow catcher and every data pass of a splat scene stay on the CPU; the GPU work has not run
  on a real Windows GPU; the 3D viewport shows splats as opaque discs; `ReadGLTF3D` loads meshes only.

## Moved to 0.28

Spot cone rendering, `ZDefocus` and a depth channel, `ChromaKeyer`, a shared format registry, particle nodes,
the fluids solver spike, `ImageToSplat` through the runtime manager, `WriteSplat3D`, shadow offset and
blur controls, kept specular, multichannel EXR, and the rest of the 2D-to-3D integration.

# NodeBased 0.26.0 — handles in both viewers, 24 more 2D nodes, glTF and Relight

## What changed since 0.25.0

- **Move, rotate and scale objects in the 3D viewport.** Click an object to select it: its node is
  selected in the graph, its properties open, and its bounds are outlined in the frame. A click on
  empty space clears; a drag still orbits or pans and never picks. With an object selected, `W` shows
  the move gizmo (three arrows and three planes), `E` the rotate gizmo (three rings) and `R` the
  scale gizmo (three axis cubes plus a centre cube for uniform scale). `Q` switches to pivot mode,
  where the arrows move the pivot instead of the object. Each drag writes the node's own transform
  knobs, shows a numeric readout, is one undo step, and keys animated knobs. Escape cancels. Nothing
  in the scene is re-evaluated while dragging.
- **A 2D Transform handle in the Viewer.** When a `Transform` is selected and the Viewer shows it or
  anything downstream of it, the Viewer draws its box, a dashed rotate ring, corner and edge scale
  handles and a round pivot marker. Drag the box to translate, the ring to rotate, a corner or edge
  to scale uniformly, and Ctrl-drag the pivot to move it with translation compensated so the image
  stays still. One undo step per drag; roto and tracker clicks keep priority.
- **24 more 2D nodes, all towards Nuke parity.** Each has an evaluator kernel and a tile-path kernel
  asserted identical, bypass, and pixel-asserting tests, with mask and mix wherever Nuke has them:
  - Colour and channels: `Invert`, `Clamp`, `Multiply`, `Add`, `Gamma`, `Saturation`, plus the
    two-input `Dissolve`, `Keymix`, `Copy` and `ChannelMerge`, which share `Merge`'s mask, mix and
    bypass-passes-B behaviour.
  - Filters: `Erode`, `Dilate`, `Median`, `Sharpen`, `Glow` and `Mirror`.
  - Draw: `Ramp`, `Radial`, `Rectangle`, `Noise` (repeatable by seed) and `Text`. Each can stand on
    its own or be composited over an input.
  - Keyers: `Keyer` (luminance, a single channel or saturation, with Nuke's four-point range),
    `HueKeyer` (a hue range with softness and a saturation window) and `Difference` (a matte from
    what changed between two inputs).
  - `Merge` gains `average`, `from` and `hypot`, completing Nuke's 19 operations.
  `docs/PARITY_2D.md` is the audit of Nuke 17's 2D toolbar that drives this work; 37 of its rows now
  read supported, up from 14 when the audit was written.
- **`ReadGLTF3D`.** Loads glTF 2.0 `.glb` and `.gltf` meshes with an in-house reader: triangles,
  strips and fans, normals, UVs, baked node transforms, quantized attributes and texture transforms.
  Base colour factors and textures become the surface colour and texture; alpha is honoured for
  `BLEND` and `MASK` materials.
- **Relight.** `Render3D` has a `relight` output carrying per-light diffuse and specular response
  layers from one raster evaluation, and a new 2D `Relight` node consumes it with ambient colour,
  diffuse, specular and mix knobs and up to eight `Light3D` inputs, so lights can be rebalanced
  without re-rendering the scene. CPU only.
- **`Axis3D` and `TransformGeo3D`.** Nuke's `Axis` (a pure parenting transform that accepts geometry,
  a light or a scene) and `TransformGeometry` (bakes the transform into the vertices, normals through
  the inverse transpose). Both share the transform knob block every geometry node already carries.
- **Groundwork that ships without a node yet.** `nodebased/simcache.py`, a deterministic memory plus
  disk cache for frame-by-frame simulations, with `docs/SIMULATION.md` (particles are next).
  `nodebased/runtimes.py`, an on-demand `uv` runtime manager pinned to the SHARP image-to-splat
  recipe, checksum-verified, cancellable, and never importing torch into the app. `docs/FLUIDS_SPIKE.md`
  is a sourced read of the fluids ecosystem.

## Known limits

- **Gizmo rings and cubes point along world X, Y and Z**, whichever way the object is rotated. Objects
  under a `TransformGeo3D` take the new value on release and do not visibly move mid-drag. Picking is
  the CPU bounds test whichever backend painted the frame, the outline is an axis-aligned bounds box,
  and `ReadAlembic3D`, `ReadUSD3D`, `ReadGLTF3D`, `ReadSplat3D` and `Project3D` are not pick targets.
  `Camera3D` and `Light3D` have no handles yet.
- **The gizmos and the new nodes were verified by automated tests** on Linux (offscreen Qt and an
  RTX 3080 Ti). Nobody has yet driven them on a real display, on Linux or Windows.
- **`Mirror` renders on the whole-image path only**, never tiled. `Retime`-style time nodes,
  `ChromaKeyer`, `Reformat` with a real format model and `CornerPin` are not in this release.
- **`ReadGLTF3D`** does not load points, lines, cameras, lights, skins, morph targets, animations,
  vertex colours, or metallic, roughness, normal, occlusion and emissive maps. Draco, meshopt, KTX2 and
  sparse accessors are refused by name.
- **`Relight`** is CPU only and needs the `relight` output re-rendered whenever the scene changes.
- **Still true from 0.25.0:** relit splats under a shadowed light, the shadow catcher, transparent or
  projected meshes mixed with splats and every data pass of a splat scene stay on the CPU; none of the
  GPU work has run on a real Windows GPU; the 3D viewport shows splats as opaque discs.

## Moved to 0.27

`TimeOffset`, `FrameHold` and `Retime` (written and tested on the 2D parity lane, merge pending at
tag time), `Camera3D` and `Light3D` handles, `ChromaKeyer`, `Reformat`, `CornerPin`, particle nodes,
the fluids solver spike, `ImageToSplat` through the new runtime manager, `WriteSplat3D`, better splat
normals, shadow offset and blur controls, kept specular, sphere rows and columns, the matrix readout,
multichannel EXR, and the rest of the 2D-to-3D integration.

# NodeBased 0.25.0 — splats and ray tracing move to the GPU, and bypass works

## Fixed in this release

- **Bypassing (disabling) a node now works.** In 0.24.0, bypassing a `Merge` put the text `'grade'`
  in the viewer, and bypassing a `Grade` changed nothing on screen. The first was a crash whose whole
  message was the name of a node; the second was the viewer's tiled path running the node it was
  meant to skip. Both evaluation paths now share one rule, and a bypassed node never runs.
  **Behaviour change:** a bypassed `Merge` passes **B**, its background, as Nuke does (A when B is
  not connected). A saved script that contains a bypassed `Merge` will look different.

  ![Bypassing a Merge, then a Grade; the viewer follows each toggle](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/01-bypass.gif)
  [Full-quality clip (MP4, real time)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/01-bypass.mp4)
- **Splat shadows appear on Windows.** On a Direct3D 12 backend, a light aimed straight down an
  axis cast no splat shadow at all in the GPU ray tracer: the surface came back at its plain
  unshadowed colour. Caught by the release packaging gate before any tag, so no released build ever
  had it. GPU test logs now also name the adapter, its backend and whether the colour target is
  float32 or half float, so a parity number can be read with the hardware that produced it.
- **Internal errors say what they are.** If NodeBased itself fails while evaluating, the viewer reads
  "Internal error while evaluating (KeyError: 'grade'). This is a NodeBased bug", and no longer a bare
  word.

## What changed since 0.24.0

- **Gaussian splats render on the GPU.** With `Backend` on `auto` or `gpu`, `Render3D` draws splats
  with wgpu for the beauty image in `raster` mode when every mesh is opaque: baked colours, or relit
  splats whose lights cast no shadows. 200,000 splats at 1920 x 1080 take 530 ms cold and 174 ms warm
  on an RTX 3080 Ti (reviewer's re-measurement); 0.24.0 took about 12 s on the CPU. Output agrees with
  the CPU render within 3e-3 in the tests and within 1e-6 in the reviewer's own comparisons. Anything
  outside that scope renders on the CPU under `auto` and is refused with the reason under `gpu`.

  ![200,000 splats: one camera move on the CPU, then an orbit on the GPU](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/02-splats-cpu-vs-gpu.gif)
  [Full clip, real time (MP4, 75 s): 25 s per camera move on the CPU, 360 to 530 ms on the GPU at 1280 x 720](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/02-splats-cpu-vs-gpu.mp4)
- **A ray-traced mode on the GPU.** `Mode` `raytrace` now runs on the GPU for mesh scenes: textures,
  lights, specular, emission, shadows through transparent surfaces, up to 64 surfaces per ray, and
  every data pass (`depth`, `normals`, `position`, `uv`, `object_id`, `albedo`, `diffuse`, `specular`,
  `emission`). A lit, shadowed 40,002-triangle scene at 1920 x 1080 takes 1.5 s on an RTX 3080 Ti
  (reviewer) against roughly two minutes on the CPU tracer (extrapolated from 320 x 180, so the ratio
  is rough). Passes agree with the CPU tracer within 1.4e-5; object ids are exact.

  ![GPU ray-traced beauty with the albedo, diffuse, specular, normals and depth passes](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/03-gpu-raytrace-passes-sheet.png)
- **Splats cast shadows in the GPU tracer.** Splats shadow meshes on the GPU in `raytrace` mode, and
  the visible splats are composited over the traced image. 200,000 splats casting and visible over a
  floor at 1920 x 1080: about 3 s cold, 2.3 s warm (reviewer). The 3.4-million-splat capture renders
  this way in 33 to 45 s and is not refused (lane's figure; not compared against the CPU).
- **Shadows caught on a capture without relighting it.** `ReadSplat3D` has a `Catch shadows` slider.
  CG meshes between a shadowed light and the capture darken the capture's own colours, so an object
  dropped into a scan grounds itself while the scan keeps its look; `Relight` can stay at 0. At 0 the
  render is byte-identical to before and traces nothing.

  ![Catch shadows swept from 0 to 0.85: the cube grounds itself on the capture](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/04-catch-shadows.gif)
  [Time-lapse (MP4): render waits play at 40x and say so on screen; the pictures hold in real time](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/04-catch-shadows-timelapse.mp4)
  · [the raw take, real time (MP4, 13 min)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/04-catch-shadows.mp4).
  Each slider step re-renders the capture for about 131 s on the CPU at 960 x 540; switching the
  slider off and on again afterwards hits the shadow cache and takes about 125 ms.
  Capture: Nelson Ghost Town by Paolo Tosolini (superspl.at), CC BY 4.0.
- **Cast shadows on or off per capture.** An environment capture's sky shell or walls otherwise block
  every light in the scene. With `Cast shadows` off the capture still renders, still receives and
  catches shadows, costs nothing in the shadow budget and builds no splat BVH.

  ![Cast shadows on: the sky shell blocks the sun. Off: the scene is sun lit](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/06-cast-shadows-pair.png)
- **Splat shadows are cached.** Visibility per splat is kept between renders, keyed by the casters,
  the mesh occluders and the light's geometry. Changing colour, intensity, ambient, `Relight` or the
  camera traces nothing. A 200,000-splat shell at 640 x 360: 36.4 s the first time, 5.3 s after.
  Moving a light still costs a full trace.
- **Large splat frames show progress and are no longer refused in the app.** A slow CPU splat frame
  shows a progress bar in the status bar with a percentage and the time left, a newer edit cancels it
  between tiles, and single-image export gets the same bar. The 3.4M-splat capture at 1920 x 1080,
  refused in 0.24.0, renders in about 95 s on the CPU. The estimate is pessimistic before 5% and
  within about 10 s from half way. The agent CLI and batch renders still refuse over-budget frames.

  ![The progress bar and time left during a 1080p render of the capture, played at 10x](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/05-progress-1080p.gif)
  [Time-lapse at 10x (MP4)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/05-progress-1080p-timelapse.mp4) · [real time (MP4, 3.5 min)](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/05-progress-1080p.mp4).
  The clip's scene is the capture plus a cube and a shadowed sun, and takes 208 s; the 95 s above is
  the capture alone. Capture: Nelson Ghost Town by Paolo Tosolini (superspl.at), CC BY 4.0.
- **HD frames with heavy shadows no longer refused on the GPU.** The GPU mesh renderer splits a frame
  into bands when one submission would be too much work, with cancellation between bands. 90,002
  triangles with a shadowed light at 1920 x 1080: 3.5 s. Output is bit-identical to a single
  submission. 0.24.0 refused such frames above about 19,000 triangles.

## Known limits

- **Still CPU-only for splats:** relit splats under a shadowed light, the shadow catcher, transparent
  or projected meshes mixed with splats, and every data pass of a scene that contains splats. `auto`
  falls back; `gpu` names the reason.
- **The GPU ray tracer needs 8 storage buffers per shader stage.** Adapters with fewer fall back to
  the CPU. It does not yet trace projected geometry.
- **The real capture as a shadow caster is heavy:** 33 to 45 s a frame on an RTX 3080 Ti. Turn
  `Cast shadows` off on environment captures.
- **Caught shadows** come from meshes only, are sampled at splat centres (large soft splats blur the
  edge), and do not show in the 3D viewport.
- **Progress** appears for viewer frames and single-image export. Thumbnails and sequence writes are
  no longer refused either, but show no bar inside a frame. GPU renders report no progress.
- **Bypass latency:** in the test window (960 x 540, offscreen) a toggle takes about 300 ms the
  first time and 170 ms after; evaluation is a cache lookup, and most of the rest is the CPU display
  conversion every viewer update pays. In the recorded clip above, on a real display with a small
  graph, each toggle reached a finished frame in 62 to 89 ms. The conversion cost is unchanged in
  this release.
- **Windows:** the GPU tests run in the release build on CI's software adapter (WARP on Direct3D 12,
  half-float colour) and pass there; none of the GPU work in this release has run on a real Windows
  GPU.
- **The 3D viewport still shows splats as opaque discs.** Relit, blended splats in the viewport are
  not in this release.

## Moved to 0.26

Relight passes and a 2D Relight node, better splat normals, shadow offset and blur controls, kept
specular, `WriteSplat3D`, sphere rows and columns, the matrix readout, multichannel EXR, particles,
volumes, a glTF reader, and the 2D-to-3D integration (SHARP image to splat, Pixal3D image to mesh,
WorldSculpt splat to meshes).

## How the media was made

Every still and clip above comes from the real app or a real render at commit `fa996cd` on an RTX
3080 Ti, made by `tools/release_media_stills.py`, `tools/release_media_clips.py` and
`tools/release_media_timelapse.py`. The step timings behind each clip are in
[manifest-clips.json](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/manifest-clips.json) and [manifest-stills.json](https://github.com/neodimo/NodeBased/releases/download/v0.25.0/manifest-stills.json).
Time-lapses speed up render waits only and say so on screen.

# NodeBased 0.24.0 — ray-traced mode, Gaussian splats and a faster 3D viewport

## What changed since 0.23.0

- **A ray-traced render mode.** `Render3D` has a `Mode` knob: `raster` (the default, and what
  every existing document uses) or `raytrace`, a CPU-only alternative that finds visibility with
  one ray per sub-sample through a BVH and shares the rasterizer's shading, so both modes agree
  on every output. CPU shadows now use the same BVH. On the GPU backend, shadow rays traverse a
  BVH on adapters that support it.
- **Gaussian splats.** `ReadSplat3D` loads 3D Gaussian Splatting `.ply` captures (SH degrees 0 to
  3, `as_authored` or `colmap` orientation) with Nuke-style transform knobs, and `Render3D`
  renders them on the CPU with an EWA splat rasterizer. Splats and meshes are ordered per pixel,
  in both render modes, against opaque and transparent geometry.
- **Splats in the passes.** `depth`, `normals`, `position`, `uv` and `object_id` include splats
  (first hit where accumulated splat opacity reaches 0.5), and a new `splats` output gives the
  splats' premultiplied contribution on its own.
- **Splat relighting.** `ReadSplat3D` has a `Relight` slider: 0 keeps the captured colours
  exactly, 1 re-lights each splat from the scene's `Light3D` nodes and the render's ambient,
  using the splat's flattest axis as its normal. With `Shadows` on a light, meshes and other
  splats shadow relit splats, and splats cast shadows onto meshes.
- **Real captures render.** The reference 3DGS Jacobian clamp removes the full-frame veil that
  large off-screen splats caused on a 3.4-million-splat capture, splats closer than 0.2 view
  units are culled as the 3DGS reference does, and memory for mixed mesh and splat frames stays
  near 150 MiB at 1080p through banded buffers (it was about 0.9 GiB).
- **A splat work budget that matches the real cost.** Renders are refused up front by counted
  tile work (splats per 16 x 16 tile), with an estimate in seconds and remedies in the message,
  instead of by bounding-box pairs. The 3.4M-splat capture is accepted at 640 x 360 and at
  1280 x 720.
- **An interactive GPU viewport.** The 3D viewport has its own wgpu renderer:
  meshes upload once, each object is one draw call, and orbiting only rewrites a camera matrix
  (about 1 ms per frame for a 64-segment sphere at 960 x 600 on an RTX 3080 Ti, against about
  400 ms on the CPU reference). Without a usable adapter it falls back to the CPU renderer and
  says so.
- **Splats in the viewport, as a layout proxy.** Each splat is drawn as a small opaque disc in
  its base colour, up to 1,000,000 per cloud, following `Relight`, `Opacity` and `Scale`. The
  3.4M-splat capture orbits at about 3 to 4 ms per frame on an RTX 3080 Ti, and F frames the
  bulk of a cloud rather than its outliers.
- **3D panels follow Nuke.** XYZ rows with a keyframe diamond per axis, typed
  fields for sizes and open-ended magnitudes, sliders only for bounded values. 3D nodes have
  rounded ends in the graph; `Scene3D`, `Light3D` and cameras are circles.
- **One transform block everywhere.** `Card3D`, `Cube3D`,
  `Sphere3D`, `ReadGeo3D` and `Scene3D` gain uniform scale, rotation order and pivot, in the same
  order as `ReadSplat3D`. Older documents load unchanged and render the same.
- **Navigation.** Alt + left drag pans the node graph and the viewer, Alt + scroll zooms, and
  touchpad zoom is proportional.

See [3D in NodeBased](3D_FOUNDATION.md) for conventions and details.

## Known limits

- **Splats render on the CPU only.** The GPU backend does not render splats (`auto` uses the
  CPU). Expect seconds to minutes per frame: 200,000 splats at 1080p take about 12 s, the
  3.4M-splat capture at 640 x 360 about 50 s. Large renders are refused up front rather than
  left to run, and there is no progress bar inside a frame. With the default budget the
  3.4M-splat capture is refused at 1920 x 1080 (estimated 142 s); 1280 x 720 is the largest
  standard size it renders at (estimated 105 s).
- **Splat shadows are slow, CPU-only, and a batch/reference feature.** One shadowed light on a
  200,000-splat shell (relit) took 36.1 s at 640 x 360 (down from 78-86 s before an exact BVH
  speed-up; 5.3 s without shadows) and 44.1 s at 1920 x 1080 (12.1 s without shadows). A mesh
  floor receiving shadows from the same 200,000 splats at 320 x 180 took 5.7 s against 4.0 s without.
  Shadow cost depends on how many overlapping splats each ray crosses, which the up-front render
  budget does not see, so a dense capture can take far longer than these figures. Surfaces made of
  overlapping splats keep a residual self-shadowing of about 3%; shadows darker than 0.1% count as
  fully dark; each splat's shadow is a closest-approach approximation of a Gaussian's density, not a
  volume integral.
- **Relighting is an approximation.** The capture's own lighting stays baked into the colours,
  so an unevenly lit capture is double-lit; normals are guessed from splat shape; specular and
  view-dependent colour are not re-lit; `albedo`, `diffuse`, `specular` and `emission` ignore
  splats.
- **The viewport's splats are not a preview of the render.** Opaque
  discs, no blending, no view-dependent colour, no splat shadows; clouds above 1,000,000 splats
  are strided. Without a GPU the viewport marks splat centres as points.
- **The ray-traced mode is CPU only**, hard shadows only, and no reflections, refraction or
  global illumination. At most 16 mesh surfaces may lie in front of a ray's terminating surface
  when transparent meshes and splats mix.
- The viewport shows no shadows, sorts transparency per object and shades at most 16 lights.
- Not in this release: a GPU splat path, GPU ray tracing, particles, volumes. These are staged
  in [3D_ROADMAP.md](3D_ROADMAP.md).
- Windows behaviour is proven by CI only; nobody has driven this release by hand on Windows. The
  GPU viewport and GPU renderer have run there only in the release workflow's tests, on a software
  adapter, never on a real Windows GPU.

# NodeBased 0.23.0 — GPU rendering, USD, Alembic, shadows, materials and AOVs

## What changed since 0.22.0

- **An optional GPU backend.** `Render3D` has a `Backend` knob: `cpu` (the default, and what
  every existing document uses), `gpu` (wgpu) or `auto`. `auto` falls back to the CPU renderer
  when no capable adapter is present or a GPU render fails; `gpu` reports the reason instead.
  Adapters that cannot render to `rgba32float` (downlevel GL/GLES class) count as unavailable.
  The CPU rasterizer remains the reference and the GPU path is tested against it.
- **USD import and export.** `ReadUSD3D` loads `.usd/.usda/.usdc/.usdz` stages as scenes (meshes
  with UVs and normals, world transforms, up axis and `metersPerUnit` honoured), `ReadUSDCamera3D`
  loads a camera, and `WriteGeo3D` writes USD when the path has a USD extension. Release builds
  bundle `usd-core`; from source, install `nodebased[usd]`.
- **Alembic import.** `ReadAlembic3D` and `ReadAlembicCamera3D` read Ogawa `.abc` files with an
  in-house, read-only reader: polygon meshes, transforms and cameras, with time interpolation.
  No extra package is needed.
- **Camera projection and OBJ export.** `Project3D` projects an image through a camera onto any
  geometry, perspective-correct, with `Outside`, `Backfaces` and an approximate depth-map
  `Occlusion` option. `WriteGeo3D` exports OBJ.
- **Shadows.** `Light3D` has a `Shadows` switch: hard shadows on the CPU renderer and on the GPU
  backend, which is parity-tested against the CPU result.
- **Materials.** Blinn-Phong specular and emission on both backends.
- **AOVs.** `Render3D`'s `Output` adds `albedo`, `diffuse`, `specular`, `emission`, `position`,
  `uv` and `object_id` alongside `rgba`, `depth` and `normals`, on CPU and GPU.
- **Rasterizer fix.** A top-left fill rule covers shared triangle edges exactly once, and the
  GPU backend matches it.

See [3D in NodeBased](3D_FOUNDATION.md) for conventions and details.

## Known limits

- **Alembic is partial.** Verified against Blender-exported files only. Polygon meshes,
  transforms and cameras; no curves, points, subdivision surfaces or materials. Every mesh is
  decoded again on each evaluation, so heavy archives are slow.
- **USD is partial.** Meshes and cameras only: no USD materials, lights, point instancers,
  curves or volumes. The stage is opened twice per evaluation.
- **GPU shadows are brute force** (every shadow ray tests every triangle) under a per-adapter
  work budget; a render above the budget is refused up front. A submitted GPU job cannot be
  cancelled. The 3D viewport does not show shadows.
- **One AOV pass per `Render3D` node.** Several passes mean several nodes; there is no
  multichannel EXR output.
- Materials stop at Blinn-Phong: no metalness, GGX, reflections, transmission or texture maps.
- Not in this release: ray tracing, Gaussian splats, particles, fluids. These are staged in
  [3D_ROADMAP.md](3D_ROADMAP.md).
- Windows behaviour is proven by CI only; nobody has driven this release by hand on Windows.

# NodeBased 0.22.0 — a 3D scene graph and viewport

## What changed since 0.21.1

- **3D nodes that render into the comp.** `Card3D`, `Cube3D`, `Sphere3D`, `ReadGeo3D` (Wavefront
  OBJ), `Light3D`, `Camera3D`, `Scene3D` and `Render3D`. `Render3D` produces the same float32,
  scene-linear, premultiplied RGBA as every other node, so Grade, Merge and Write follow it.
  Ports are typed: a mismatched connection is refused and the document is left untouched.
- **Textured cards for 2.5D.** Connect any image to a geometry node's `image` input. Textures
  are perspective-correct, mip-mapped, respect alpha and stay premultiplied. Upstream 2D edits
  re-render the 3D scene.
- **Lights.** Scenes are unlit until they contain a `Light3D` (directional or point); then
  surfaces are Lambert-shaded with an `ambient` term on `Render3D`.
- **Scene hierarchy.** `Scene3D` has its own transform and accepts geometry, lights and other
  scenes; nested scenes inherit their parent's transform.
- **Render controls.** Supersampled antialiasing (1–4), transparent background by default, and
  `rgba`, `depth` or `normals` output. Near-plane clipping, depth-sorted transparency,
  cancellation, proxy tiers and result caching all apply. Every numeric 3D parameter animates
  and takes expressions.
- **A 3D viewport.** Toolbar → 3D viewport. Orbit, pan, dolly, **F** to frame the scene, **C**
  to look through the authored camera. It shows the scene your `Render3D` will see — textures,
  animation and disabled nodes included — with a depth-tested grid, axes, camera frustum and
  light markers. Navigation never edits the authored camera. The dock is saved with the workspace.
- **Type-aware node creation.** A new node wires itself to the selection only where the types
  fit: select a plate and create a `Card3D` to texture it; select a card and create a `Scene3D`
  to collect it.

See [3D in NodeBased](3D_FOUNDATION.md) for conventions and details.

## Known limits

- The renderer is a CPU reference rasterizer: correct, tested against analytic answers, and
  not fast. It refuses scenes above 250,000 triangles. There is no GPU scene backend yet.
- No shadows, specular, materials, motion blur, depth of field or deep output. `depth` and
  `normals` are image passes.
- Not in this release: camera projection, USD/Alembic/FBX, ray tracing, Gaussian splats,
  particles, fluids, and in-viewport transform handles. These are staged in
  [3D_ROADMAP.md](3D_ROADMAP.md); Nuke 17.1 parity is a long-range target, not a claim.
- Interpenetrating transparent surfaces can sort incorrectly.

# NodeBased 0.21.1 — a properties panel that fits, and a workspace that remembers

## What changed since 0.21.0

- **The properties panel reflows to its dock.** Panels no longer paint wider than the dock and
  clip their right edge. Form labels wrap above their field when a row runs out of room, notes
  word-wrap, numeric fields share the width that is left, and long button and check box labels
  shorten (full text in the tooltip). Stacked panels follow the dock as it is widened or
  narrowed, and sideways scrolling is gone.
- **The window reopens the way you left it.** Window size and placement, dock layout (including
  the Agent dock) and the viewer/node-graph divider are saved on close and restored on launch.
  An unusable saved layout falls back to the default instead of half-applying.
- **Workspace → Default workspace.** A new menu puts the window, panels and dividers back where a
  fresh install has them, in place.
- **A bolder app icon.** The icon now fills its whole tile, with no border, so it reads at
  taskbar and dock sizes.

## Known limits

- On Wayland, applications cannot set their own window position, so only the window size and
  layout are restored there; the compositor chooses where the window appears.

# NodeBased 0.21.0 — direct GPU viewport rendering and packaged GPU verification

## What changed since 0.20.0

- **Direct GPU viewport rendering.** Scene-linear float32 frames stay float32 through
  computation and are transformed directly into the active viewport framebuffer.
  Channel isolation, exposure, alpha, checkerboard compositing, OCIO views, and
  top-down image orientation are covered by offscreen-GL tests.
- **Packaged Linux GPU smoke coverage.** The release workflow launches the real AppImage
  under Xvfb/GLX and requires the packaged application to report the GPU display path.
- **Cache precision remains explicit.** Full-resolution scene-linear cache entries remain
  isolated from display output; half-float storage is only permitted where the cache
  correctness gates prove its range and round-trip behavior.

## Fixes

- Release GPU resources while their owning OpenGL context is current, avoiding deferred
  texture destruction after the context has gone away.

# NodeBased 0.20.0 — an agent in the app, Nuke knobs and a pixel readout

## What changed since 0.19.0

- **An Agent panel with a real terminal.** View → Agent (or the Agent menu) opens a docked
  terminal that starts Claude Code or Codex in the project directory. The agent talks to the
  running application, so it acts on the comp you are looking at rather than a description of
  it.
- **The application explains itself to the agent.** A knowledge module ships its own docs
  inside the package: architecture, the time model, playback, colour management, roto and
  tracking, the agent protocol and the release history. The agent reads topics on demand
  instead of guessing.
- **`nodebased-mcp`.** A stdio MCP server exposes the live document over the local bridge:
  describe, inspect, edit, undo, redo, view, errors, reference context, knowledge and issue
  filing. Save, load and render stay behind a switch that is off by default.
- **The agent can file GitHub issues.** Reports go through the `gh` CLI, fall back to the REST
  API with a token, and are recorded in a local log. The switch is in the Agent panel.
- **Nuke's knob types.** Parameters render as float sliders with a soft-range ruler,
  XY pairs, colour knobs with a swatch and picker, checkboxes, dropdowns and file fields,
  each with an animation button. Values remain scene-linear floats around 0–1, never 0–255.
- **Expressions on any knob.** Press `=` with the cursor in a knob, or right-click and choose
  Enter expression. Expression-driven knobs take their own colour, and Clear expression puts
  the plain value back.
- **Stacked properties panels.** Double-click pins a node's properties below the current one,
  Nuke-style, with a per-panel close and a clear-all. The stack holds five panels by default;
  Preferences and the dock's spin box change the cap.
- **A User tab on every node,** alongside the node's own tab and the Node tab from 0.19.0,
  with Revert and Close.
- **Pixel values under the pointer.** The lower right of the viewer reads out full-resolution
  coordinates with Nuke's bottom-left origin, raw float RGBA and a colour swatch. It follows
  the pointer while you draw roto shapes too.
- **Hotkeys match Nuke.** Node creation keys were checked against Nuke's defaults and
  reassigned, and select-all, copy, cut, paste and duplicate were added. The viewer gains
  Ctrl+= / Ctrl+- zoom, Ctrl+1 for 1:1, and J/K/L shuttle.
- **Help → Keyboard shortcuts** lists every binding from the single table the menus use, so
  the dialog cannot drift from the application.

## Fixes

- The pixel readout was dead in the running application even though its tests passed: a second
  `mouseMoveEvent` on the viewer shadowed the readout's handler. The handlers are merged and
  the tests now hover through the real event path.
- Three tests made platform assumptions that only failed on Windows: a stored boolean read as
  raw Ini text, a log file opened twice, and a `/tmp` path used as a directory.

## Known limits

- The 0.17.0 playback limits are unchanged: native 24 fps at 4K ACES 2.0 is still not reached,
  and performance evidence remains from one Linux/X11 machine.
- The Agent panel has been driven by hand on Linux/X11 only: Claude Code starts and reaches its
  trust prompt. Signing in and running a full agent session through the panel, on any platform,
  is still unproven.
- The MCP server has been exercised against a running application with the window hidden, not
  through a full agent session.

# NodeBased 0.19.0 — Node tab, postage stamps and a quieter viewer

## What changed since 0.18.0

- **Merge reads like Nuke's.** `B` is the top-centre trunk input, `A` joins from the left, and
  Merge gains an optional `mask` on the right. Where the mask is zero the result is `B`. Older
  comps open with the mask unwired and render identically.
- **New nodes land underneath the selection.** Adding a node while one is selected wires it in
  and places it directly below, stacking down the same column on repeated adds.
- **Scrubbing reuses what playback cached at full resolution.** The display cache is keyed by
  frame and region and is checked before tiles are composed, and a cached whole frame also
  serves a zoomed-in crop.
- **Postage stamps.** Read, Constant and Checker nodes show a thumbnail of their output by
  default; any other node can turn one on. Stamps render only while the viewer is idle and
  never delay a frame. Settings → Interface can switch them all off.
- **A Node tab in the properties panel.** Each node has a second tab with a label (drawn on
  the node in place of its type), an Enabled switch and the postage-stamp switch.
- **The viewer only re-renders when its picture can change.** Moving, renaming or labelling a
  node, or editing a branch the viewer is not showing, no longer triggers a render.
- **Dots win the click.** Clicking a Dot selects it even zoomed out, and Ctrl-clicking a Dot no
  longer starts a noodle insert underneath it.
- **Cleaner cards and an accent colour.** The thick coloured bar on the left of each node is
  gone; the family colour stays in the outline. Settings → Interface adds an accent colour
  (presets or custom) on top of any theme, stored per machine.
- **Tidier example comp.** The demo's `A` branch sits to the left of the Merge, so noodles no
  longer cross.

## Compatibility

- Projects are now document version 12. Older projects upgrade on open; 0.18.0 and earlier
  cannot open a project saved by 0.19.0.

## Known limits

- The 0.17.0 playback limits are unchanged: native 24 fps at 4K ACES 2.0 is still not reached,
  and performance evidence remains from one Linux/X11 machine.
- The Node tab does not yet offer Nuke's tile colour, font or hide-input controls.

# NodeBased 0.18.0 — Nuke-familiar graph, properties panel and export

## What changed since 0.17.0

- **Node titles and disabled state read at a glance.** Titles are centred and larger, and a
  disabled or bypassed node dims and carries a large X across it instead of announcing itself
  only in small subtitle text. A disabled Merge still shows its `B` input, so the wiring stays
  visible while the node is off.
- **Input ports sit where a compositor expects them.** `A` is on the left edge and `mask` on
  the right, with ordinary inputs still centred along the top. Dot control points draw above
  noodles so a Dot still looks like a dot where connections cross it.
- **Properties-panel context menus stay open.** The knob and curve menus were parented to the
  panel, and the rebuild triggered by opening one destroyed the menu mid-show — the menu
  appeared to flash and vanish. The window owns them now.
- **The viewer no longer resizes the layout.** Comp format and playback status text fed back
  into widget size hints, so a 4K format or a long status line could redistribute the splitter
  during playback. Panel sizes now change only when the artist drags them. The status label
  elides for painting and keeps the full string for tooltips and the agent bridge.
- **Viewer nodes behave like Nuke's.** Viewing a node rewires every Viewer to it as one undo
  step, with cycle and self-view guards, and the viewer connection draws faint, dashed and
  arrowless so it does not read as a real input of the node it inspects.
- **Write nodes do the real exporting.** A Write renders its own upstream tree at full
  resolution, validates its parameters against the chosen file type, refuses an unpadded path
  for a frame range rather than overwriting one file every frame, passes its input through
  unchanged, and can be bypassed like any other node.
- **Read opens a sequence-aware browser.** Numbered frames batch into a single padded entry,
  grouping is on by default and can be switched off to list every frame, padding width and
  extension separate two sequences in the same folder, a lone numbered file stays a still, and
  a hole in a range is reported rather than quietly hidden.
- **Chrome and theming.** Check for updates is right-justified at the trailing edge of the
  toolbar, and the settings dialog offers theme colours, which persist as a preference and are
  never written into the document. An unknown stored theme falls back instead of failing.

## Known limits

- The 0.17.0 playback limits are unchanged: native 24 fps at 4K ACES 2.0 is still not reached,
  and performance evidence remains from one Linux/X11 machine.
- Theme choices cover the application palette; per-node colour overrides are not exposed.
- The sequence browser reads directory listings rather than image headers, so it groups by file
  naming and does not validate that batched frames share a format.

# NodeBased 0.17.0 — faster 4K playback ingest and parallel decode-ahead

## What changed since 0.16.0

- **Proxy playback is substantially faster on the measured 4K sequence.** The standard
  auto-proxy ACES 2.0 path improved from 2.10 fps to 10.21–11.42 fps on the development
  machine, a measured **4.9–5.4x** gain. The reproducible harness, stage timings, and exact
  hardware scope are recorded in `docs/BENCHMARKS-v0.17-playback.md`.
- **Decode and ingest do less memory work.** ACEScg premultiply handling now broadcasts a
  one-channel alpha factor instead of constructing full RGBA-sized factor arrays; ordinary
  RGB channel layouts take a slice fast path; and proxy decimation uses strided accumulation
  rather than a slow multi-axis reshape reduction. The tier-2 and tier-4 decimation stages
  measured about 8x faster in isolation while retaining the existing pixel tolerances.
- **Bounded parallel decode-ahead.** During proxy playback, a dedicated worker pool warms
  decoded Read frames ahead of the single-owner preview evaluator. The cache is memory-bounded,
  suppresses duplicate requests, drops stale work after seeks or edits, and never runs for
  full-resolution tier-1 playback where measurements showed it was counterproductive.
- **Source replacement and shutdown are safe.** Decode-cache identity includes the resolved
  file path, size, and modification time, including the reference-frame identity used by
  `missing=black`, so replacing media at the same path cannot return stale pixels. Window close
  cancels queued decode work and remains prompt even if a native decode is already in flight.
- **Playback QA is repeatable.** `tools/playback_qa.py` exercises a real desktop window and
  reports throughput and ordering; its ordering check accepts the expected end-to-start loop
  while still rejecting an ordinary backward jump.

## Known limits

- Native 24 fps at 4K ACES 2.0 is not reached. The best measured result is 11.42 fps at the
  normal tier-2 auto-proxy resolution; full-resolution playback improved only modestly, from
  1.18 fps to 1.38 fps.
- Performance evidence is from one Linux/X11 machine and one 100-frame 4K EXR sequence.
  Windows is covered by the conformance suite, not by equivalent real-hardware playback
  measurements. Decode-worker tuning, display-cache hashing, Qt scene rebuilds, and QImage
  conversion remain candidates for the next performance pass.
- Negative-origin/overscan behavior through the pre-existing tiled proxy path remains outside
  this release's playback-performance scope.

# NodeBased 0.16.0 — fast display transform, agent reference loop, roto drawing and pixel tracking

## What changed since 0.15.0

- **The ACES 2.0 view transform is no longer the playback bottleneck.** OCIO's CPU processor
  releases the GIL, so the display transform now runs across a thread pool. The output is
  bit-identical to the single-threaded path. Measured medians: 4K ACES 2.0 dropped from
  2170 ms to 176 ms per frame, and 4K sRGB from 129 ms to 20 ms. When an OpenGL 4 context is
  available, ACES 2.0 runs as an OCIO-generated GLSL shader instead. That path measured 47 ms
  at 4K and 5 ms at HD on the development machine's integrated GPU, within 1 8-bit code value
  of the CPU reference. sRGB stays on the CPU path, because GPU upload and readback made it
  slower. `NODEBASED_DISPLAY_GPU=0` forces the CPU path, and the viewer status shows which
  backend is active. Method and full numbers: `docs/BENCHMARKS-v0.16-display.md`.
- **Reference-image agent loop.** Nodes can be tagged `Reference for agent`, and the GUI endpoint
  captures the viewed and tagged images (schema v10, `reference_context`). The new
  `nodebased-agent-loop` client sends those captures and your prompt to a vision model
  (Anthropic by default, key from `ANTHROPIC_API_KEY`). It validates the proposed edits against
  the node schema and applies each iteration as one revision-guarded batch, which is one undo
  step. `--dry-run` applies nothing, and by default every batch asks for confirmation.
  NodeBased itself still makes no model or network calls.
- **Roto shapes can be drawn and edited in the viewer.** Draw closed shapes and drag existing
  points directly on the image. Edits are validated, undoable, and preserve animated points.
- **Tracker analyses real pixels.** Pick a point, and the Tracker follows it forward with
  normalised cross-correlation and sub-pixel refinement. Weak matches are reported as possible
  occlusion instead of writing bad motion. Results commit as one undoable edit, and cancelling
  leaves the document unchanged.

## Known limits

- The GPU display path was benchmarked on one Linux machine only. CI runners have no GPU and
  exercise the CPU fallback. Other GPUs, drivers, and Windows GL contexts are unverified. A
  display-less eGPU used through PRIME offload measured slower than the integrated GPU there.
- The agent loop's live model call was not exercised in automated tests.
- Tracking is forward-only point tracking. Roto has no transform handles, track linking, or
  ROI-limited evaluation yet.

# NodeBased 0.15.0 — expressions, keyframes, timeline, and roto foundation

## What changed since 0.14.0

- **Expressions are available in numeric knobs.** Set, edit, and clear formulas directly
  from the inspector. Expressions support `frame` and references to other knobs, resolve
  deterministically through the renderer, reject dependency cycles with their full path,
  and participate in atomic undo/redo.
- **Animated controls are visible and usable.** Numeric knobs expose keyframe controls;
  formula-driven controls show their resolved value and reject conflicting base-value or
  keyframe edits.
- **Timeline readability and cache feedback.** Adaptive tick marks/frame labels, blue
  keyframe underlines, and orange cached-frame underlines make temporal state visible.
- **Roto, Tracker, and ChannelShuffle groundwork.** Schema v8 added deterministic CPU
  roto mattes and channel routing. Interactive shape drawing and image-based tracker
  solving remain future work.
- **Expression-error feedback is stable.** Rejected formula edits stay visible in the
  inspector even if an asynchronous preview status update occurs.

# NodeBased 0.14.0 — playback correctness fixes and live agent diagnostics

## What changed since 0.13.0

- **Fixed: the display cache never actually hit during playback.** Its key
  hashed the whole document, including the live playhead position.
  Read-ahead builds every prefetch request from one document snapshot taken
  while the playhead is still on the current frame, so a frame warmed
  several frames ahead got hashed with the *old* playhead value — a
  guaranteed miss once that frame actually became current. Every read-ahead
  result was being thrown away. Fixed by normalizing the playhead to the
  frame actually being evaluated before hashing.
- **Measured: the ACES 2.0 view transform costs roughly 10x what sRGB
  costs on identical pixels**, and that cost scales with resolution (HD
  ~550ms vs ~56ms; 4K ~2.2s vs ~170ms). This is inherent to OCIO's built-in
  ACES 2.0 implementation, not a caching bug, and is the real reason native
  4K/ACES playback is still not real-time — moving the transform off the
  CPU is real, scoped, unstarted future work.
- **Nuke-style sequential playback fallback.** When rendering can't sustain
  real time, the transport used to keep following the wall clock anyway,
  so it could show frames wildly out of order — including jumping
  backward — once a slow render finally finished. Playback now degrades to
  strictly in-order, slower-than-real-time frames instead, matching Nuke's
  own fallback behavior. Verified against a real 4K sequence: before, 9
  frames displayed in 20s in the order 62, 14, 64, 17, 68, 18, 69, 22, 72;
  after, 9 frames in 25s as 2, 3, 4, 5, 6, 7, 8, 9, 10.
- **Live render-error visibility for an attached agent.** A new `errors`
  operation on the local agent bridge surfaces every evaluation failure —
  including ones on frames that never reached the screen — instead of
  requiring a human to read the status bar. See `docs/AGENT_PROTOCOL.md`.
- **Fixed: GitHub Release pages showed the entire changelog, not just the
  version being published.** The release workflow now extracts only the
  current version's section from the release notes.

# NodeBased 0.13.0 — branch-insert node placement and viewer format guides

## What changed since 0.12.0

- **Adding a node with a selection wires it into that node's branch.**
  Tab-search or a node hotkey used to always drop the new node at the last
  click position, unconnected. If a node is selected and the new node type
  has an input, it now lands near the selection, connects from its output,
  and takes over any existing downstream connection the selected node had —
  the same splice used by the existing Ctrl-drag-a-noodle-midpoint gesture —
  so it's inserted inline in the branch rather than forking a dead end.
  Generators with no input (Read/Constant/Checker) are unaffected and keep
  click-position placement.
- **Viewer format guides, Nuke-style.** A dotted outline now traces the
  display window at any zoom level, and a resolution readout ("3840 x
  2160") sits just outside its bottom-right corner. Implemented as a
  `drawForeground` paint rather than scene items, specifically so it can
  never perturb `itemsBoundingRect()` — the measurement the 0.12.0
  stuck-corner regression test depends on staying exact.

# NodeBased 0.12.0 — playback caching and proxy-resolution playback

## What changed since 0.11.0

- **EXR writes default to 16-bit half at ZIPS (1-scanline) compression.**
  Applies to export and the agent's `render` op alike; both bit depth and
  compression remain explicitly overridable. Half/ZIPS is now verified to
  match the tile path exactly against the reference evaluator on real 4K
  source (max abs delta 0.0, zero fallbacks).
- **Display-ready frame cache.** Looping playback, scrubbing back onto a
  frame already shown, or returning a paused parameter to a value it held
  before is now a cache hit instead of paying the ACES 2.0 view transform
  again — measured 3.14s cold vs 0.08s on an identical repeat at 4K. Keyed
  on the whole document plus frame/tier/view/exposure/channel/background, so
  a graph edit is correctly a miss and undoing it is correctly a hit again.
- **Proxy-resolution playback.** Sources above HD now play back at an
  automatically chosen lower tier — the same technique every NLE and
  compositor uses for this — and restore full quality the instant playback
  stops. Never overrides a tier chosen manually. Cuts a cold 4K frame from
  ~3.1s to ~1.1s. Read-ahead now also warms the transformed display cache,
  not just the raw composite, ahead of the playhead.
- **Fixed: viewer stuck zoomed into a stale corner.** Reconnecting the
  viewer to a differently sized source used to clamp its first frame to
  whatever fraction of the old image's viewport happened to overlap the new
  canvas, so the image could get stuck zoomed into a small corner with no
  way to recenter it. The first frame of a resized source now always
  requests the complete canvas.
- **Fixed: a schema-upgrade step could silently skip its own successor.**
  One upgrade step stamped the generic `SCHEMA_VERSION` constant instead of
  the literal version it actually produced — harmless while that step
  happened to be the last one, but the next schema version added would have
  landed a document tagged current while missing its newest section. Fixed
  and guarded by a test that checks no upgrade step makes this mistake.

## Known scope

Native 4K playback at full (untier'd) quality is not real-time — the ACES
2.0 display transform's CPU cost is the remaining bottleneck for that
specific case. Proxy-resolution playback above closes most of the practical
gap; moving the transform itself off the CPU is unscoped follow-up work.

# NodeBased 0.11.0 — ACEScg color pipeline and project settings

- **Linear ACEScg float32 working space.** Read inputs convert from their tagged
  or selected source space into ACEScg before graph evaluation. Untagged EXRs
  fall back to Linear Rec.709 instead of being assumed to already match the
  working space. EXR output is tagged ACEScg; PNG output converts from ACEScg
  through OCIO rather than applying only an sRGB transfer curve.
- **Correct premultiplied display.** Viewer transforms now unpremultiply,
  transform straight color, then re-associate alpha. Semi-transparent pixels no
  longer disagree between the viewer and PNG output.
- **ACES 2.0 Rec.709 default view.** New projects use the ACES 2.0 SDR 100-nit
  Rec.709 view on the sRGB display.
- **Project settings, schema v7.** `Edit → Project settings…` (`S`) exposes the
  bundled OCIO config, ACEScg working space, display, saved default view, and
  black/checker viewer background. Settings edits are validated, atomic,
  undoable, saved in the project, and available through the Dispatcher protocol.
- **Deterministic slow-playback coverage.** The transport regression now delays
  the tile executor actually used by the viewer and renders a tiny graph, so
  hosted-runner raster speed cannot decide whether the test passes.

# NodeBased 0.10.0 — animation curves

## What changed since 0.9.1

- **Animation curves (document schema v6).** Any numeric node parameter can hold
  a curve of keys with `constant` or `linear` interpolation. Curves are an
  evaluation-time overlay: a node's stored parameters are never mutated, so a
  graph with no curves renders byte-identically to a v5 document.
- **Endpoint hold.** Before the first key and after the last, a curve holds that
  key's value, matching Nuke's extrapolation.
- **Atomic curve editing.** `set_key`, `delete_key` and `clear_curve` are
  Dispatcher operations with undo/redo, and deleting a node drops its curves in
  the same transaction, so a document can never reference a curve on a node that
  is gone. The agent CLI exposes the same operations.
- **Animation reaches the tile viewport.** Curves resolve at the tile executor's
  API boundary, so animated parameters render identically through the tile path
  and the reference evaluator, at every proxy tier.
- **Transparency displays over black.** The viewer composites transparent and
  partially transparent regions over pure black instead of a checkerboard. The
  checker tinted every pixel it showed through, so soft alpha edges displayed
  brighter than the graph produced them.

## Upgrade

Documents at schema v1 through v5 upgrade in place on open, chained through to
v6, gaining an empty animation section. No stored parameter changes, so an
upgraded comp renders exactly as it did before.

## Acceptance evidence

357 tests pass offscreen on the release commit. Animated parameters are pinned
against the reference evaluator at frames 1/5/10 across proxy tiers 1/2/4, and
the frames are asserted to differ, so a parameter frozen at its base value
cannot pass by matching an equally frozen reference.

## Known scope

Animated `Transform` and `Crop` render correctly but take the full-frame
fallback rather than the tile path; those kernels are not tile-supported yet.
Desktop verification is offscreen, so interactive playback feel on real
hardware is unverified.

# NodeBased 0.9.1 — Windows cache-root repair

## What changed since 0.9.0

- **Windows release-test repair.** Disk-cache initialization now survives
  sanitized Windows environments with no profile variables, using `TEMP` or
  the process directory as a safe fallback. A regression test covers it.

# NodeBased 0.9.0 — tile-native viewport and EXR data windows

## What changed since 0.8.0

- **Bounded tile evaluation.** Supported graphs evaluate in 256px tiles, with
  halo-aware Blur and explicit fallback telemetry for unsupported kernels.
- **EXR data windows survive.** Read/evaluation/tile requests preserve offsets
  and overscan beyond the display frame, including negative origins.
- **Bounded source Reads.** Full-resolution tile requests acquire the requested
  source region instead of decoding then slicing a full image.
- **Visible viewport scheduling.** The Viewer requests its visible data-window
  rectangle; panning requests newly exposed pixels. Export remains a complete
  full-resolution reference render.
- **Cache/runtime foundation.** Adaptive 8GiB-max RAM cache, persistent disk
  spill, full/half/quarter proxy tiers, and 24fps delivery controls.

## Acceptance evidence

At 4K on the recorded Linux CPU host, a centered 1920×1080 tile viewport on a
supported graph measured 312.232ms cold and 208.855ms grade-edit p50, versus
2404.175ms and 2052.728ms for the full reference frame. See
`docs/BENCHMARKS-v0.9-4k.md` for scope and machine details.

# NodeBased 0.8.0 — bounded playback and read-ahead

## What changed since 0.7.0

- **Forward playback.** Space or the transport button plays the inclusive comp
  range at document FPS and loops at the end.
- **Bounded read-ahead.** One display request is prioritized ahead of at most
  three future frames. Evaluation stays serial so one worker owns the existing
  LRU cache.
- **Deadline behavior.** The playhead derives from elapsed wall time. Slow
  frames are counted and skipped instead of building a latency spiral.
- **Cancellation and display safety.** Scrubs, edits, channel/view changes and
  stop cancel obsolete work. A result must match both the active generation and
  current frame before it may enter the Viewer; prefetch results only warm cache.
- **Undo-safe transport.** Playback updates the persisted playhead through the
  validated command boundary without consuming the 100-slot artist undo stack.
- **Explicit proxy boundary.** This release only schedules full-quality frames.
  Unimplemented proxy tiers are rejected rather than faking a proxy by scaling
  an already fully-evaluated Viewer image.

## Acceptance evidence

The playback contract in `docs/PLAYBACK.md` fixes a 16 ms enqueue budget, a
three-frame queue cap, cooperative cancellation, stale-frame exclusion, and
wall-clock deadline behavior. Automated coverage exercises the queue, real EXR
sequence cache warming, UI transport, undo preservation and wrong-frame rejection.

# NodeBased 0.7.0 — time foundation and image sequences

## What changed since 0.6.5

- **Time is now first-class.** Schema v5 stores a composition frame range,
  current frame, and frame rate. The evaluator receives an explicit timeline
  frame, leaving a clean boundary for future clips, retimes, tracks, nested
  timelines, and animated parameters.
- **Image-sequence Read.** Read accepts printf (`plate.%04d.exr`) and hash
  (`plate.####.exr`) patterns, a source-frame offset, and explicit missing-frame
  behavior: error, hold nearest, or transparent black at the sequence format.
- **Time-aware caching.** A sequence frame changes the Read fingerprint and its
  dependent branch. Static sources and unrelated branches retain their cached
  pixels while the playhead moves.
- **Minimal timeline strip.** First/current/last controls and scrubbing sit under
  the Viewer. Left/Right step; Home/End jump to the range boundaries.
- **Agent time control.** `describe` advertises time/sequence capabilities,
  `time` edits the range atomically, and headless `render` accepts a frame.
- **Windows installer identity.** The NSIS installer/uninstaller and Start Menu
  shortcut now explicitly use the NodeBased icon.

## Validation

Schema migration, sequence resolution, frame offsets, missing-frame policies,
static/temporal cache behavior, timeline UI/undo synchronization, and a real
frame-specific agent render are covered by the full automated suite.

# NodeBased 0.6.5 — the NodeBased icon

## What changed since 0.6.4

- **NodeBased has its app icon.** A folded green ribbon forming an `N`, on a
  dark rounded tile, with real transparent alpha outside the tile. Used by the
  desktop window, Windows executable/installer, and Linux AppImage.

# NodeBased 0.6.4 — visible reroute handles

## What changed since 0.6.3

- **Ctrl now visibly exposes every noodle's reroute handle.** Each connected
  noodle gets a high-contrast circular midpoint marker while Ctrl is held. The
  marker responds immediately to the global Ctrl key event, even when Qt focus
  is elsewhere in the workspace.
- **Dots behave as reroutes.** A Dot is a compact circular node with a visible
  selected outline. Its body is selectable and draggable after creation; its
  sockets remain at the top and bottom rather than covering the center.
- **App icon plumbing landed, placeholder graphic.** The desktop window,
  Windows executable/installer, and Linux AppImage now load an icon file from
  `assets/`, so a final mark drops in with no code change. The mark bundled in
  this release is an early draft that was rejected for being too detailed;
  design is still in progress and this asset will be replaced.

## Validation

141 automated tests cover Ctrl-handle state, Dot selection and movement,
transactional noodle insertion, source asset loading, and the existing graph,
compositor, media, agent, updater, and packaging behavior.

# NodeBased 0.6.3 — Dot crash repair and reliable graph sockets

## What changed since 0.6.2

- **Fixed the Dot/Switch graph crash.** Their missing theme colors caused a
  scene rebuild exception after creation, leaving a partially cleared graph and
  apparently vanished noodles. Both nodes now render safely.
- **Dots are compact reroute points.** Ctrl-dragging a noodle midpoint shows a
  live Dot preview that follows the cursor and only changes the graph on drop.
- **Larger socket hit targets.** Visible ports remain small, but their grab area
  is 26 px; the graph recognizes the visible socket child as its port target so
  active wires cannot be accidentally dropped on it.

## Validation

138 automated tests cover Dot creation without clearing graph edges, live Ctrl
preview placement, atomic insertion, symmetric wiring, and the full compositor
and packaging suite.

# NodeBased 0.6.2 — safe noodles and viewer controls

## What changed since 0.6.0

- **Safe input rewiring.** Picking up an existing input preserves its original
  connection until a valid output is dropped. Esc or a drop on empty canvas
  leaves the comp intact; a completed gesture replaces only that input.
- **Ctrl-drag a noodle midpoint to insert a Dot.** Holding Ctrl reveals a center
  handle on every connected noodle. Drag one to place a Dot that is atomically
  spliced into the connection, so it cannot leave a half-disconnected graph.
- **Viewer hotkeys:** `R`, `G`, `B`, and `A` solo their channel; pressing the
  currently soloed channel again returns to RGB. `F` fits the image; `H` is the
  matching home/frame alias for the single-image 2D viewer.

## Validation

137 automated tests cover direct/reverse wiring, transactional input rewire,
Ctrl midpoint Dot insertion, channel toggles, and framing keys, in addition to
the compositor, media, schema, agent, and updater suites.

# NodeBased 0.6.0 — masks, mix, graph routing and selection

## What changed since 0.5.1

- **Reusable mask + mix controls** now apply to Grade, ColorCorrect, Blur,
  Transform and Crop. Their optional `mask` input gates the processed result by
  mask alpha, and `mix` blends it back with the source. Existing projects retain
  the exact prior behavior: an absent mask and `mix: 1.0` are the defaults.
- **Dot** adds a zero-cost graph-routing passthrough. **Switch** selects between
  two image inputs with its `which` control; both are available to keyboard node
  search and to the agent interface.
- **Document schema v4** upgrades v3 projects by adding optional filter-mask
  slots and default mix values. v1 through v3 projects continue through the
  tested upgrade chain.
- Agents can discover Dot, Switch, filter mask ports and mix parameters through
  `describe`, and build/render schema-v4 graphs through the local JSON-lines
  interface.

## Validation

135 automated tests cover schema migration, premultiplied mask/mix behavior,
Dot/Switch evaluation, desktop graph behavior, and agent graph construction.
Native display validation of the new optional mask ports remains a human QA item.

# NodeBased 0.5.1 — graph interaction hotfix

## What changed since 0.5.0

- Restores normal **round-capped noodles** at a more readable 3.25 px stroke.
  Arrowheads are painted separately, preventing Qt from filling the curve into
  a ribbon.
- Inputs now wire in **either direction**: drag an output to an input, or drag
  an empty/connected top input to an output. A 24-screen-pixel magnetic target
  makes ports much less finicky at any graph zoom.
- **Tab is pointer-contextual.** When the mouse is over the node graph, Tab is
  captured before Qt focus traversal and opens node search even if a dock or
  control currently owns keyboard focus.

# NodeBased 0.5.0 — graph interaction repair

## What changed since 0.4.0

- **Direct noodle drag wiring.** Drag from any output port directly onto an
  input port to connect. The connection preview follows the pointer and the
  destination resolves on mouse release, so wires no longer depend on a fragile
  click sequence. Existing click-to-connect and click-a-connected-input-to-pick-
  up-and-rewire remain available.
- **Legible directional connections.** Saved and pending noodles now terminate
  in small arrowheads that point into their input. Ports stay above noodles in
  the drawing order, making them easy to grab.
- **Centered top inputs.** A single input is exactly at a node's top centre;
  multiple inputs fan symmetrically around that centre. Outputs stay centered
  at the bottom.
- **Nuke-style Tab search.** Tab opens a keyboard-first popup search at the
  last graph click. Enter creates the selected node there. The placement code
  finds the nearest vacant slot, so a new node never lands on top of an
  existing node. Single-key creation still works.

The interaction coverage now includes direct drag-to-connect, input-centering,
non-overlapping placement, and Tab-search filtering, alongside the existing
rewire/disconnect tests.

# NodeBased 0.4.0 — Full Merge operations, real Transform, Premult/Unpremult

Fourth desktop release. Where 0.3.0 added new nodes, this one makes the
compositing core itself honest: the operations an artist actually reaches for
now behave the way they do in Nuke.

## What changed since 0.3.0

- **All 16 Nuke-style Merge operations.** Merge gains an `operation` choice:
  over, under, plus, minus, multiply, screen, max, min, difference, divide,
  mask, stencil, in, out, atop, xor. Inputs are premultiplied, A is the
  foreground and B is the background, matching Nuke's wiring convention.
  `divide` guards against zero. The 0.3.0 `over`/mix behaviour is preserved
  byte-identical, so existing comps render exactly as before.
- **Transform is a real transform.** Float `translate_x`/`translate_y`,
  `rotate` in degrees, `scale`, and a `center`, with a filter choice of
  nearest, bilinear or cubic. It is inverse-mapped and sub-pixel filtered, and
  does not wrap at the edges. Defaults reduce to identity, so 0.3.0 projects
  render unchanged.
- **Premult and Unpremult** are new single-input nodes. Unpremult leaves RGB
  untouched wherever alpha is 0, so it never emits NaN or inf into the rest of
  the graph.
- **Document schema v3, with a tested upgrade path.** Opening a v2 project
  converts old integer Transform `x`/`y` into float `translate_x`/
  `translate_y`, defaults rotate/scale/center/filter to identity, and gives
  Merge `operation: "over"` where it is missing. The v1 → v2 → v3 chain is
  covered by tests. The .nbcomp compatibility promise holds: projects saved by
  0.1.0, 0.2.0 and 0.3.0 all still open.
- The new node types, parameters and choices reach `agent.describe` and the
  inspector automatically, from the same SPECS/LIMITS/CHOICES tables the
  desktop UI reads. No separate agent-protocol update was needed.

## What changed in 0.3.0

- **Wire pick-up/rewire.** Clicking an already-connected input port unhooks its
  wire into a pending connection from the same source, which can then be dropped
  on a different input, or on empty space to disconnect. Real unhook/rehook, not
  delete-and-recreate.
- **Four new Nuke-parity nodes**, with keyboard shortcuts and inspector controls:
  **ColorCorrect** (lift/gamma/gain/saturation, sign-safe under fractional gamma
  on negative HDR values), **Blur** (separable box blur, O(n) via a cumulative
  sum, edge padding avoids darkened borders), **Crop** (masks to a rectangle
  without resizing the canvas), and **Shuffle** (remaps output channels from any
  input channel or a constant 0/1).
- All new node types and the null-source disconnect/rewire semantics are exposed
  automatically to the agent protocol's `describe` operation, driven by the same
  SPECS/LIMITS/CHOICES tables the desktop UI reads.
- Projects saved by 0.2.0 load unchanged; the graph format did not change.

## What changed since 0.1.0 (also in 0.2.0)

- **EXR read and write** through OpenImageIO. Float RGBA in, zip-compressed float
  RGBA out. HDR and negative values survive a round trip exactly.
- **Read node color controls.** Input space (Auto, sRGB, Linear Rec.709, ACEScg,
  ACES2065-1, Raw), alpha interpretation (Auto, Straight, Premultiplied), EXR layer
  selection and EXR subimage/part selection. Auto reads EXR as linear/premultiplied
  and PNG/JPEG/TIFF as sRGB/straight.
- **OpenColorIO display views.** sRGB, ACES 2.0 SDR (Rec.709) and Linear, using the
  built-in `cg-config-v4.0.0_aces-v2.0_ocio-v2.5` ACES config. No external OCIO
  config file or `OCIO` environment variable is required, in packages or from source.
- **TIFF input** alongside EXR, PNG and JPEG.
- **Export image** replaces Export PNG and offers float EXR or 8-bit sRGB PNG. The
  headless agent `render` operation follows the same rule by file extension.
- **Darker theme.** Neutral charcoal surfaces with teal, amber, lavender and
  soft-blue node accents, focus rings on inputs and a distinct update button.
- Data windows are composited onto the display window, so cropped EXRs keep their
  position instead of shifting to the origin.
- Projects saved by 0.1.0 load unchanged; Read nodes gain the new controls at their
  Auto defaults.

## Downloads

- **Windows portable ZIP:** extract the entire `NodeBased` folder to a writable
  location and run `NodeBased.exe`. No installer or administrator access required.
  Keep `_internal` and `portable.marker` beside the executable. Update cache and
  rollback files stay under its `portable-data` folder. No registry installation.
- **Windows installer:** per-user installation; no elevation requested.
- **Linux AppImage:** mark executable and run. No installation or Python setup.
  Requires an x86_64 Linux desktop with glibc 2.35+ and the relevant graphics
  libraries. If FUSE is unavailable, run with `APPIMAGE_EXTRACT_AND_RUN=1`.

These early binaries are unsigned. Workplace application-control policies still
apply, including to portable programs.

## Update button

Same explicit flow as GameStore: **Check for updates → Download v… → progress →
Restart to update**. There are no automatic downloads or surprise restarts.
Unsaved work can be saved, discarded or used to cancel the restart. Downloads use
GitHub's published asset SHA-256 digests and are rechecked before installation.
Windows portable builds update in place using a separate helper; installer builds
run the per-user installer. AppImages replace their original file atomically.
A previous portable bundle/AppImage is kept for manual rollback. Keep the app in
a writable location for updates. Source checkouts use Git/pip instead.

## Scope and known limits

Still an M0-class prototype, not a production compositor. Color management runs on
the CPU per preview: on the development machine a 960 × 540 frame took ~58 ms
through the sRGB view and ~202 ms through ACES 2.0, so display transforms are not
yet interactive at high resolution and need a GPU/LUT path. Deep EXR is rejected
rather than flattened. Reads are full-frame with an 8192-per-axis cap. Embedded ICC
profiles are ignored; set Input space explicitly for non-sRGB LDR sources.
Sequences, timeline, 3D and AI generation remain roadmap work. macOS arm64 comes
later. Offscreen tests and packaged launch checks do not establish native GPU or
display performance, or workplace-policy acceptance.
