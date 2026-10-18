# Direct picture (experimental GPU viewer slice)

Status: VP1 of 2, opt-in, default off. The Viewer (a QGraphicsView fed by `Window._show_image`) is unchanged and remains the default, the fallback, the export source and the pixel-sampling source.

## What it is

Window menu, "Direct picture (experimental GPU)", or start with `NODEBASED_DIRECT_VIEWER=1`. It opens a floating dock holding a `QRenderWidget(present_method="screen")` (`nodebased/directviewer.py`). Every new preview picture the Viewer receives is also offered to this widget. The newest ready picture is converted to RGBA once and written once into a retained wgpu texture; wheel zoom, middle-drag or Alt-drag pan and double-click fit change a small uniform buffer and redraw, with no `to_qimage` and no pixmap rebuild. A format outline and an orange corner mark are drawn in a second GPU pass.

During playback the dock takes the scene-linear float frame instead (`submit_linear`), and the card runs the whole display transform; see "Display transform on the card" below. The dock title says it is experimental and has no overlays. The dock is not part of the saved workspace.

## Contracts

- **Ownership.** `FrameLedger` holds the rules; `RetainedPresenter` is the only code that creates or destroys GPU objects. A same-size picture is written into the existing texture; a size change makes a new texture and the old one is destroyed after the draw that replaced it is submitted. WebGPU's `destroy()` frees memory only once the queue has completed the submissions that used it, so this is a GPU-completion rule, not a count of presents (VP1 held it for 3 presents). Closing waits for the queue to go idle (`wait_idle`, a 4-byte mapped copy: wgpu-py 0.32's `on_submitted_work_done` cannot be called against wgpu-native 29). Pending pictures are bounded to the newest one; older pending pictures are dropped before conversion.
- **Revision and epoch.** Offers carry the source generation and the device epoch. An older generation or an older epoch is rejected. Equal generations are accepted (playback).
- **Surface textures.** `get_current_texture()` is called inside one draw and nothing from it survives the draw.
- **Pacing.** Input handlers change view state and call `request_draw()`; rendercanvas schedules the draw from Qt callbacks (capped at 120 per second). Nothing spins or waits for a surface.
- **Device.** One device, requested from the adapter wgpu reports as presentable to this window (`request_adapter_sync(canvas=...)`). `get_preferred_format` alone is not enough: on a display server without DRI3 it answers for an adapter whose `configure` aborts the process in native code. `NODEBASED_DIRECT_ADAPTER=discrete|integrated|cpu` states a wish. The backend order is D3D12 first on Windows (W1, docs/WINDOWS_HYBRID_PRESENTATION.md), then wgpu's own choice; elsewhere wgpu's choice. `NODEBASED_DIRECT_BACKEND=d3d12|vulkan|auto` overrides it, for the Vulkan comparison. `presenter.selection` records the requested kind and backends, the selected device and backend and every request that failed; `selection_note` says what was selected when it differs from the wish, without claiming that adapters never asked for could not present.
- **Failure.** Any start or draw failure hides the dock, unchecks the menu item and shows a status message; the Viewer carries on.

## Known gaps (VP1)

- No Roto, Tracker, Paint, A/B wipe, region-of-interest, pixel readout or input strip on the direct surface; those stay on the Viewer. Hit order across overlays is therefore not exercised.
- Colour and alpha: outside playback the picture shown is the Viewer's already display-transformed, background-composited 8-bit RGB. During playback the card runs the view transform itself (below). Surface formats other than 8-bit (HDR, wide gamut) are not handled; sRGB-format surfaces re-encode, which the shader undoes once (tested within one code value).
- While the direct picture plays, the Viewer holds its last picture; the refresh when playback stops repaints it. A B buffer (wipe, combined compare) keeps the processor path.
- Sampling is nearest at every zoom, like the Viewer; no mip or filter for strong minification.
- Windows (D3D12/Vulkan backend choice, DWM), native Wayland (Qt gives no display id; the xcb platform was used), macOS: not run.
- Device loss: the installed `GPUDevice._get_lost_async` raises `NotImplementedError`, so loss is only noticed when a draw raises; the epoch rules are unit-tested, the recovery path is not exercised on real loss.
- Real-screen cadence, vsync behaviour, tearing and input-to-photon latency are not measured. Present return is when the application call returns, not screen response.
- A hidden or occluded dock still receives offers; the newest-only rule bounds the cost to one conversion per drawn frame.

## Display transform on the card

`nodebased/carddisplay.py` builds `to_qimage`'s whole chain as one GLSL 4.50 fragment shader that wgpu compiles through naga: gain, channel select (RGB, R, G, B, A), unpremultiply, gamma, the view transform, re-associate, the checker background, the clipping warning and the 8-bit quantise (the unorm target rounds). The view transform is OCIO's own GPU shader for the view, the same text gpudisplay gives OpenGL; OCIO's lookup data (for ACES 2.0 the reach and gamut-cusp tables, nearest-sampled at texel centres, and a 363-entry hue array) goes into one read-only storage buffer per view; inlined as constant arrays it made the ACES draw 17 ms instead of 0.4 ms, because naga copies a constant array for every dynamic index. A view needing a 2D or 3D lookup, or a filtered table, raises `Unsupported`, and that frame goes through `to_qimage` as before.

The frame is uploaded once into an `rgba32float` texture (32 MB at 1080p, about 1.9 ms); settled half-float cache frames go up as `rgba16float` as they are. A float16 conversion on the processor costs 11 ms at 1080p, more than the upload it would save.

`tests/test_carddisplay.py` holds the card to `to_qimage` within 1/255 per channel over linear ramps, HDR highlights to 200, slightly negative values, alpha edges and premultiplied noise, for every channel mode, the sRGB, ACES 2.0 and Linear views, the checker, exposure, gamma and the clipping warning. Checked on the RTX 3080 Ti, the Radeon 8060S (force-adapter.py integrated) and llvmpipe (cpu). The largest difference seen is one code value.

During playback `Window` hands the dock a `carddisplay.LazyPicture` (the float frame plus the display settings) instead of `to_qimage`'s result. The 8-bit picture is built only if something asks for it: an open output preview, or the card refusing the view.

## Measurements

`tools/viewer_direct_bench.py` and `tools/viewer_direct_functional.py`; results in `benchmarks/viewer/direct-vp1-*-2026-10-18.json`. All of them were taken on a private Xvfb display (no GPU presentation, no DRI3), so the cadence numbers describe that display server, not a screen: llvmpipe presents in about 8 ms intervals (the 120 per second cap), NVIDIA presents through a software copy path at about 16 per second, and the AMD adapter cannot present to that display at all. The Viewer's own figures there are Qt raster to an X backing store. None of these decide whether the direct path is faster on a real screen; that comparison is left to the real-display pass.
