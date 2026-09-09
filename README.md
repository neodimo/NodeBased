# NodeBased

An artist-first native compositing DCC foundation, with a shared command layer for
humans and agents. Windows + Linux first; macOS arm64 is a later delivery target.

**Current stage: M0 executable prototype.** This is not Nuke/Houdini parity, a
production compositor, or an AI video-generation product yet.

## Download and update

Get the [latest release](https://github.com/neodimo/NodeBased/releases/latest):
Windows **portable ZIP**, per-user Windows installer, or Linux AppImage. Python
is not required for these packages. For portable Windows, extract the whole folder
and run `NodeBased.exe`; keep `_internal` and `portable.marker` beside it. Portable
updates/cache live under `portable-data`, with no registry installation. Keep
projects outside the application folder when practical.

The toolbar button follows GameStore's flow: **Check for updates → Download →
Restart to update**, with download progress and an unsaved-project prompt.
Downloads are SHA-256 verified. No automatic download/install. The first release
reports "Up to date" until a newer stable release exists. Portable apps still
need to comply with workplace application-control policy.

Linux: `chmod +x NodeBased-*.AppImage`, then launch the AppImage. If FUSE is absent,
use `APPIMAGE_EXTRACT_AND_RUN=1 ./NodeBased-0.2.0-linux-x86_64.AppImage`.
Builds require glibc 2.35+; macOS packages are deferred. Early binaries are unsigned.
See [release notes](docs/RELEASE_NOTES.md).

## Run from source

Python 3.11+ is required. From this repository:

```sh
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux: `source .venv/bin/activate`

Then:

```sh
python -m pip install -e .
python -m nodebased
```

Or with [uv](https://docs.astral.sh/uv/): `uv run nodebased`.
Linux needs a working Qt desktop platform (Wayland or X11). Tests can use Qt's
`offscreen` platform without a display. On minimal Ubuntu/Debian installs, Qt
also needs system libraries: `sudo apt-get install libegl1 libopengl0 libgl1 libxkbcommon0`.
An interactive X11 session may additionally need the distribution's Qt xcb
plugin dependencies. Packaged releases bundle Python and Qt; source installs use these dependencies.

## What works

- Native Qt viewer, node graph and dockable properties.
- Read EXR/PNG/JPEG/TIFF through OpenImageIO, with per-node input color space,
  alpha interpretation, EXR layer and subimage/part selection. Constant, Checker,
  Grade, integer Transform, premultiplied A-over-B Merge and Viewer nodes.
  Internal images are scene-linear Rec.709 float32 RGBA.
- Channel inspection, display exposure, sRGB / ACES 2.0 / Linear display views,
  fit/1:1, pan and zoom.
- Wire/disconnect nodes, edit parameters, select/move/delete, bypass, undo/redo.
- Atomic `.nbcomp` project saves with relative media paths; float EXR and 8-bit
  sRGB PNG export.
- Background preview with stale-result rejection and between-node cancellation.
- A bounded 256 MiB retained-result LRU and dependency-based invalidation.
- Optional live agent connection, plus headless graph editing/rendering.

## Controls

Focus the graph for node shortcuts. Tab opens node creation; R/G/M/T create
Read/Grade/Merge/Transform. Select a node and press 1 to view it, D to bypass,
F to frame the graph, Delete to remove. Middle-drag pans; wheel zooms.
Click an output port, then an input port to wire. Right-click an input disconnects.
New processing nodes connect their first input to the selected node.
Ctrl+Z / Ctrl+Shift+Z undo/redo; Ctrl+O opens; Ctrl+S saves; Ctrl+E exports.

Grade exposure, multiply and offset affect premultiplied RGB while preserving
alpha. Viewer exposure, channel and display view affect display only; exports are
independent of them.

## Color

The working space is scene-linear Rec.709, premultiplied float32. Color management
uses OpenColorIO's built-in `cg-config-v4.0.0_aces-v2.0_ocio-v2.5` ACES config, so
no external OCIO config file or `OCIO` environment variable is needed.

Each Read node carries **Input space** (`Auto`, sRGB, Linear Rec.709, ACEScg,
ACES2065-1, Raw) and **Alpha** (`Auto`, Straight, Premultiplied). `Auto` reads EXR
as linear/premultiplied and PNG/JPEG/TIFF as sRGB/straight. Straight sources are
premultiplied on ingest; premultiplied sources are unpremultiplied before the color
transform and restored afterwards, so the transform never sees alpha-weighted values.
**Layer** selects a named EXR layer (`diffuse`, `Z`, …) or leaves the root RGBA;
a single-channel selection is expanded to luminance. **Subimage** selects an EXR
part. The data window is composited onto the display window, so crops keep their
position instead of shifting.

Viewer display views are sRGB, ACES 2.0 SDR (Rec.709) and Linear. EXR export writes
zip-compressed float RGBA in the working space; PNG export unpremultiplies, converts
to sRGB and clamps to 8-bit. Embedded ICC profiles are not honored — set Input space
explicitly for non-sRGB LDR sources.

## Agent control

No model or credentials are required. Opt in to an OS-user-local endpoint:

```sh
python -m nodebased --agent nodebased-local
```

In another terminal:

```sh
python -m nodebased.agent --connect nodebased-local
```

Send one JSON object per line:

```json
{"op":"describe"}
{"op":"inspect","request_id":"inspect-1"}
{"op":"set","id":"grade","param":"exposure","value":1.5}
{"op":"undo"}
```

The demo uses stable IDs (`plate`, `grade`, `wash`, `merge`, `viewer`). Inspect other
projects to discover their IDs. The live endpoint shares the GUI's undo stack.
It is local, not an HTTP server, and has the launching OS user's file permissions.
`load` refuses to replace unsaved GUI changes. Do not expose it through a network
relay. Stale endpoint names are not automatically deleted.

For headless usage, omit `--connect`. It starts with an empty document; alternatively
supply `--project path.nbcomp`. Example input:

```json
{"op":"create","type":"Constant","id":"c","params":{"width":64,"height":64,"red":0.8}}
{"op":"view","id":"c"}
{"op":"save","path":"example.nbcomp"}
{"op":"render","path":"example.exr"}
```

`render` writes float EXR for `.exr` paths and 8-bit sRGB PNG otherwise, and is
headless-only in M0. The GUI exports from its current validated preview.
See [agent protocol](docs/AGENT_PROTOCOL.md) for operation shapes.

## Validation

```sh
python -m unittest discover -s tests -v
```

The desktop tests default to Qt offscreen. Set `NODEBASED_SCREENSHOT` to an output
PNG path to capture the tested shell. Windows/Linux CI executes the same suite;
passing offscreen checks does not establish interactive desktop/GPU performance.

## Known limits / next work

Full-frame CPU reference implementation: no tiles/ROI, disk cache, GPU evaluation,
sequences/timeline, animation, roto, tracking, 3D or model execution yet. Color
management is CPU-side per preview: on this development machine a 960 × 540 frame
took ~58 ms through the sRGB view and ~202 ms through the ACES 2.0 view, so display
transforms are not yet interactive at high resolution and need a GPU/LUT path.
Deep EXR is rejected rather than flattened, and the reader is full-frame with an
8192-per-axis cap and a 512 MiB channel-span limit. Retained cache bytes are bounded;
peak working memory is not. Cancellation is between nodes; decode/individual
kernels/export are not interruptible. Export runs on the GUI thread. Merge requires
matching dimensions; Transform is integer translation with fixed bounds. UI layout
is not persisted.

The full product contract and phased acceptance gates are in
[VISION](docs/VISION.md); engineering boundaries are in
[ARCHITECTURE](docs/ARCHITECTURE.md). Project handoff lives in [TASKLOG](TASKLOG.md)
and [current state](context/state.md).
