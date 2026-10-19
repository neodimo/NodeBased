# Direct picture (experimental GPU viewer slice)

Status: VP1 of 2, opt-in, default off. The Viewer (a QGraphicsView fed by `Window._show_image`) is unchanged and remains the default, the fallback, the export source and the pixel-sampling source.

## What it is

Window menu, "Direct picture (experimental GPU)", or start with `NODEBASED_DIRECT_VIEWER=1`. It opens a floating dock holding a `QRenderWidget(present_method="screen")` (`nodebased/directviewer.py`). Every new preview picture the Viewer receives is also offered to this widget. The newest ready picture is converted to RGBA once and written once into a retained wgpu texture; wheel zoom, middle-drag or Alt-drag pan and double-click fit change a small uniform buffer and redraw, with no `to_qimage` and no pixmap rebuild. A format outline and an orange corner mark are drawn in a second GPU pass. The dock title says it is experimental and has no overlays. The dock is not part of the saved workspace.

## Contracts

- **Ownership.** `FrameLedger` holds the rules; `RetainedPresenter` is the only code that creates or destroys GPU objects. A same-size picture is written into the existing texture; a size change makes a new texture and the old one is held for 3 presents before `destroy()`. Pending pictures are bounded to the newest one; older pending pictures are dropped before conversion.
- **Revision and epoch.** Offers carry the source generation and the device epoch. An older generation or an older epoch is rejected. Equal generations are accepted (playback).
- **Surface textures.** `get_current_texture()` is called inside one draw and nothing from it survives the draw.
- **Pacing.** Input handlers change view state and call `request_draw()`; rendercanvas schedules the draw from Qt callbacks (capped at 120 per second). Nothing spins or waits for a surface.
- **Device.** One device, requested from the adapter wgpu reports as presentable to this window (`request_adapter_sync(canvas=...)`). `get_preferred_format` alone is not enough: on a display server without DRI3 it answers for an adapter whose `configure` aborts the process in native code. `NODEBASED_DIRECT_ADAPTER=discrete|integrated|cpu` states a wish; when the window cannot be presented to by that kind, the presenter uses the one that can and records the reason in `selection_note`.
- **Device loss.** wgpu only logs a lost device (the promise API raises `NotImplementedError`), so `LossWatch` listens to its log for "device lost" and a failing draw counts as a loss too. `RetainedPresenter.recover()` bumps the epoch, destroys every old resource and the old device, makes a new device from the same adapter, configures the surface again and queues the retained CPU picture (or a newer pending one) for upload. The widget repeats the failed draw once on the new device. Three recoveries within 30 seconds make it give up.
- **Failure.** Any start failure, or a draw that still fails after one recovery, hides the dock, unchecks the menu item and shows a status message; the Viewer carries on. Native Wayland is refused at the switch (see gaps) before any surface exists.

## Known gaps (VP1)

- **Overlays and hit order.** No Roto, Tracker, Paint, A/B wipe, region-of-interest, pixel readout or input strip on the direct surface; they stay on the Viewer and are not drawn into the GPU picture (an A/B wipe or ROI box shows on the Viewer only). Hit order is kept by separation: the direct surface handles only wheel zoom, middle-drag or Alt-drag pan and double-click fit, accepts every other mouse event and forwards nothing, so a Roto or Tracker click is only ever taken by the Viewer (tests/test_directviewer.py, `HitOrderTests`). Overlay parity on one surface is the VP2 gate.
- **Colour and alpha.** The picture shown is the Viewer's already display-transformed, background-composited 8-bit RGB. The GPU draws those bytes; it does not run the OCIO view, exposure, gamma, zebra or channel selection itself. Surface formats other than 8-bit (HDR, wide gamut) are not handled; sRGB-format surfaces re-encode, which the shader undoes once (tested within one code value). Captured from the composited desktop on the AMD, NVIDIA and llvmpipe adapters, the checker's two colours arrived as exactly (20, 60, 180) and (200, 120, 40).
- **Sampling** is nearest at every zoom, like the Viewer; no mip or filter for strong minification. With the desktop scaled 1.75x, Xwayland stretches the window by the compositor, so on this workstation the glass shows the picture resampled; the numbers above are before that stretch.
- **Native Wayland.** The installed rendercanvas (2.7.2) builds only X11 surface ids on Linux; under `QT_QPA_PLATFORM=wayland` the Qt window id is not an X window and the process died with an X `BadWindow` error. The switch now raises "native Wayland is not supported" (`unsupported_platform`) and the Viewer carries on. Run NodeBased with `QT_QPA_PLATFORM=xcb` (Xwayland) for the direct picture; this is what every real-screen number here used. A native Wayland surface needs a rendercanvas/wgpu change, which is outside VP1.
- **Windows** (D3D12/Vulkan backend choice, DWM) and macOS: not run; this workstation cannot.
- **Device loss.** Exercised by destroying the device under the presenter (wgpu then reports "Parent device is lost" on the next call) and by a logged loss message, on the RTX 3080 Ti, the Radeon and llvmpipe; recovery rebuilt the picture from the retained CPU image. A physical event (eGPU unplugged, driver reset) was not provoked.
- **Latency.** Present return is when the application call returns, not screen response. Nothing here measures input-to-photon time. rendercanvas caps drawing at 120 per second, so a wheel event waits up to 8.3 ms for its draw (measured median 8.3 ms to present return) on top of the compositor.
- **Hidden or occluded dock** still receives offers; the newest-only rule bounds the cost to one conversion per drawn frame.

## Measurements

Tools: `tools/viewer_direct_bench.py` (cadence, conversion and upload counts, process CPU), `tools/viewer_direct_functional.py` (dock walk), `tools/viewer_direct_screen.py` (what is on the glass). Results: `benchmarks/viewer/direct-vp1-*` (private Xvfb display, 10/18 first pass) and `benchmarks/viewer/direct-vp1f-*` (this workstation's real display, KDE Plasma on Wayland with Xwayland `:0`, 2194x1234 logical at 120 Hz, desktop scale 1.75, 10/18 finish pass).

**On the glass.** `viewer_direct_screen.py` shows a changing checker, takes a Spectacle screenshot of the composited desktop and measures the checker in it. On all three adapters, at pixel ratio 1.0 and 1.5: the picture is visible; each new picture reaches the screen with the right top-left colour (blue, orange, blue); the cell size matches the view (within 2.5 px); five wheel steps grew the cells 1.15^5 within 0.12; a 60 px middle-drag moved the grid 105 px against 104.75 expected (capture pixels per widget pixel 1.7458, the desktop scale); resizing to half and back followed the widget; closing and reopening showed the picture again. Spectacle captures every window the compositor shows; X captures do not see a screen-presented surface.

**Cadence and cost, 1920x1080 pictures, Viewer and direct widget the same size** (the Viewer pinned to a 1920x1080 viewport, 240 pictures and 240 wheel steps each):

| | Viewer (default) | Direct, Radeon | Direct, RTX 3080 Ti | Direct, llvmpipe |
|---|---|---|---|---|
| pictures per second | 154 | 121 | 107 | 121 |
| CPU per picture | 4.7 ms | 2.8 ms | 9.4 ms | 61 ms |
| CPU per wheel step | 3.6 ms | 1.4 ms | 2.2 ms | 56 ms |
| picture interval p50 / p95 | no pacing | 8.3 / 8.9 ms | 9.3 / 9.5 ms | 8.2 / 8.9 ms |
| wheel to present return p50 | 1.2 ms (synchronous raster) | 8.3 ms | 8.3 ms | 8.3 ms |

Reading it: the Viewer's repaint is a synchronous Qt raster into the X backing store, so its 154 per second and 1.2 ms say nothing about what reaches the glass; the direct widget is paced by rendercanvas at its 120 per second cap on this 120 Hz display. What the comparison does support: on the display's own Radeon the direct path uses about 40 percent less CPU per new picture and about 60 percent less per pan or zoom step, converts and uploads each picture once and never again on a view change (counters in the JSON), and shows the picture exactly. The eGPU draws correctly but pays for crossing USB4 (9.4 ms of CPU per picture), and llvmpipe is a fallback that costs far more than the Viewer. The 8.3 ms wheel response is the 120 per second cap, not the work. Whether the artist feels it as faster is not measured here.

The first-pass Xvfb numbers (no GPU presentation) stay in the repository and describe that display server, not a screen.
