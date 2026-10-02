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
  alpha interpretation, EXR layer and subimage/part selection. Padded image
  sequences use `%04d` or `####` patterns with offset/hold/black controls.
  Constant, Checker, Grade, filtered Transform, 16-operation Merge, Premult,
  Unpremult, Dot, Switch and Viewer nodes.
  Internal images are scene-linear Rec.709 float32 RGBA.
- Channel inspection, display exposure, sRGB / ACES 2.0 / Linear display views,
  fit/1:1, pan and zoom.
- A composition frame range/current frame, timeline scrubbing and frame-aware
  evaluation/cache keys; agents can render a requested frame. Forward playback
  follows the document FPS with a bounded three-frame read-ahead queue, cooperative
  cancellation, dropped-frame accounting, and exact stale-frame rejection.
- Wire/disconnect nodes, edit parameters, select/move/delete, bypass, undo/redo.
- Atomic `.nbcomp` project saves with relative media paths; half-float ZIPS EXR
  (32-bit float on request) and 8-bit sRGB PNG export.
- Background preview with stale-result rejection and between-node cancellation.
- A bounded 256 MiB retained-result LRU and dependency-based invalidation.
- Optional live agent connection, plus headless graph editing/rendering.
- 3D scene graph: textured cards, cube, sphere and OBJ geometry, lights, nested scenes and a
  camera render into the ordinary float32 RGBA graph, with a navigable 3D viewport. The renderer
  is a CPU reference rasterizer. See [3D in NodeBased](docs/3D_FOUNDATION.md).

## Controls

The 2D viewer and the 3D viewport share one VIEWER panel, as in Nuke: with the pointer over the
panel, Tab switches between them. The 2D/3D buttons in the viewer toolbar do the same and show
which view is on screen, and Toolbar → 3D viewport switches the panel to 3D. Each view keeps its
own zoom, camera and selection across switches. Settings → Viewer can make viewing a 3D node
switch to 3D by itself (off by default).

Focus the graph for node shortcuts. Tab opens node creation; R/G/M/T create
Read/Grade/Merge/Transform. Select a node and press 1 to view it, D to bypass,
F to frame the graph, Delete to remove. Middle-drag or Alt+left-drag pans; the wheel or a touchpad scroll zooms, with or without Alt.
Click an output port, then an input port to wire. Right-click an input disconnects.
New processing nodes connect their first input to the selected node.
Ctrl+Z / Ctrl+Shift+Z undo/redo; Ctrl+O opens; Ctrl+S saves; Ctrl+E exports.
Space or the timeline transport button toggles forward playback. Left/Right step
the timeline and Home/End jump to its first/last frame.

Grade exposure, multiply and offset affect premultiplied RGB while preserving
alpha. Viewer exposure, channel and display view affect display only; exports are
independent of them.

### Adding nodes

The NODES dock (left side by default, Workspace → Show Nodes panel) lists every node kind by
category, or search across names and one-line descriptions with the box at the top. Click a row
to add it centred in the visible graph, or drag it to drop it at an exact spot. Ctrl+F jumps
straight to the search box from anywhere; type, then Down/Up move through the results and Return
adds the selected one (or the first, if you have not moved). Tab still opens the same search as a
popup at the mouse, for staying on the graph without reaching for the dock.

Right-click a row for "Add to Favourites" (or "Remove from Favourites") and "What is this?",
which opens the node's row in `docs/PARITY_2D.md` or `docs/3D_FOUNDATION.md`. Favourites and the
Recent category (your last ten node kinds added) sit pinned above the regular categories and
persist across restarts, the same way the theme does. Right-clicking a node on the graph itself
offers "What is this?" too. On a small screen, the Compact checkbox shrinks the category column
to icons only; hover it to see the names again.

## Color

The working space is scene-linear ACEScg, premultiplied float32. The default color
pipeline uses OpenColorIO's built-in `cg-config-v4.0.0_aces-v2.0_ocio-v2.5` ACES
config, so no external config or `OCIO` environment variable is needed by default.
Per-node OCIO transforms can use a document config or a node-specific config path.

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
16-bit half RGBA in the working space, ZIPS-compressed (deflate at one scanline per
block, so a reader can decode a single row without inflating its neighbours). The
export dialog offers 32-bit float as a second EXR filter for data passes that need
the full mantissa; finite values above half's 65504 ceiling are clamped to it rather
than silently turned into `inf`. PNG export unpremultiplies, converts
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
{"op":"time","first":1001,"last":1100,"current":1001,"fps":24}
{"op":"render","path":"example.exr","frame":1001}
```

`render` writes half/ZIPS EXR for `.exr` paths and 8-bit sRGB PNG otherwise; add
`"bits": "float"` or `"compression": "piz"` to override, and the response echoes
what was actually written. Unsupported values are rejected instead of substituted. It is
headless-only in M0. The GUI exports from its current validated preview.
See [agent protocol](docs/AGENT_PROTOCOL.md) for operation shapes.

### Reference loop client

`nodebased-agent-loop` is a separate client process that closes the "build/adjust this graph to
match a reference image" loop: it captures `reference_context` previews, asks a provider to
propose edits, and applies them as guarded `batch` commands. NodeBased itself still makes no
model or network calls.

```sh
python -m nodebased --agent nodebased-local &
ANTHROPIC_API_KEY=sk-... nodebased-agent-loop --connect nodebased-local \
    --prompt "match the reference lighting" --yes
```

Add `--dry-run` to preview proposed batches without applying them, or `--provider scripted
--script proposals.json` to drive it with a canned, network-free sequence. See
[agent protocol](docs/AGENT_PROTOCOL.md#reference-loop-client) for the validation rules and caps.

## Validation

```sh
python -m unittest discover -s tests -v
```

The desktop tests default to Qt offscreen. Set `NODEBASED_SCREENSHOT` to an output
PNG path to capture the tested shell. Windows/Linux CI executes the same suite;
passing offscreen checks does not establish interactive desktop/GPU performance.

## Known limits / next work

Full-frame CPU reference implementation: no tiles/ROI, disk cache, GPU scene evaluation,
animation, editorial clips/tracks, model execution, or full 3D parity. The bounded 3D
foundation is CPU/reference only: no textured cards, USD/Hydra, lights/materials/AOVs,
ray tracing, splats, particles, fluids, or deep output. Color
management is CPU-side per preview: on this development machine a 960 × 540 frame
took ~58 ms through the sRGB view and ~202 ms through the ACES 2.0 view, so display
transforms are not yet interactive at high resolution and need a GPU/LUT path.
Image sequences, a minimal timeline, forward playback and bounded read-ahead are
present, but there is no proxy system, audio, retiming, or clip/layer editorial
model yet. Playback uses serial full-resolution CPU evaluation; missed deadlines
skip obsolete positions. Deep EXR is
rejected rather than flattened, and the reader is full-frame with an
8192-per-axis cap and a 512 MiB channel-span limit. Retained cache bytes are bounded;
peak working memory is not. Cancellation is between nodes; decode/individual
kernels/export are not interruptible. Export runs on the GUI thread. Merge requires
matching dimensions; Transform is integer translation with fixed bounds. UI layout
is not persisted.

The full product contract and phased acceptance gates are in
[VISION](docs/VISION.md); engineering boundaries are in
[ARCHITECTURE](docs/ARCHITECTURE.md). Project handoff lives in [TASKLOG](TASKLOG.md)
and [current state](context/state.md).
