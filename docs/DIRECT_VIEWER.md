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
- **Failure.** Any start or draw failure hides the dock, unchecks the menu item and shows a status message; the Viewer carries on.

## Known gaps (VP1)

- No Roto, Tracker, Paint, A/B wipe, region-of-interest, pixel readout or input strip on the direct surface; those stay on the Viewer. Hit order across overlays is therefore not exercised.
- Colour and alpha: the picture shown is the Viewer's already display-transformed, background-composited 8-bit RGB. The GPU draws those bytes; it does not run the OCIO view, exposure, gamma, zebra or channel selection itself. Surface formats other than 8-bit (HDR, wide gamut) are not handled; sRGB-format surfaces re-encode, which the shader undoes once (tested within one code value).
- Sampling is nearest at every zoom, like the Viewer; no mip or filter for strong minification.
- Windows (D3D12/Vulkan backend choice, DWM), native Wayland (Qt gives no display id; the xcb platform was used), macOS: not run.
- Device loss: the installed `GPUDevice._get_lost_async` raises `NotImplementedError`, so loss is only noticed when a draw raises; the epoch rules are unit-tested, the recovery path is not exercised on real loss.
- Real-screen cadence, vsync behaviour, tearing and input-to-photon latency are not measured. Present return is when the application call returns, not screen response.
- A hidden or occluded dock still receives offers; the newest-only rule bounds the cost to one conversion per drawn frame.

## Measurements

`tools/viewer_direct_bench.py` and `tools/viewer_direct_functional.py`; results in `benchmarks/viewer/direct-vp1-*-2026-10-18.json`. All of them were taken on a private Xvfb display (no GPU presentation, no DRI3), so the cadence numbers describe that display server, not a screen: llvmpipe presents in about 8 ms intervals (the 120 per second cap), NVIDIA presents through a software copy path at about 16 per second, and the AMD adapter cannot present to that display at all. The Viewer's own figures there are Qt raster to an X backing store. None of these decide whether the direct path is faster on a real screen; that comparison is left to the real-display pass.
