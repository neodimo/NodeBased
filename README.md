# NodeBased

An artist-first native compositing DCC foundation, with a shared command layer for
humans and agents. Windows + Linux first; macOS arm64 is a later delivery target.

**Current stage: M0 executable prototype.** This is not Nuke/Houdini parity, a
production compositor, or an AI video-generation product yet.

## Run

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
`offscreen` platform without a display. There are no standalone installers yet.

## What works

- Native Qt viewer, node graph and dockable properties.
- Read PNG/JPEG; Constant, Checker, Grade, integer Transform, premultiplied
  A-over-B Merge and Viewer nodes. Internal images are scene-linear float32 RGBA.
- Channel inspection, display exposure, fit/1:1, pan and zoom.
- Wire/disconnect nodes, edit parameters, select/move/delete, bypass, undo/redo.
- Atomic `.nbcomp` project saves with relative media paths; PNG export.
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
alpha. Viewer exposure and channels affect display only. PNG export unpremultiplies,
converts linear RGB to sRGB and clamps to 8-bit. PNG/JPEG import assumes sRGB;
embedded ICC profiles and higher bit depths are not preserved in M0.

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
{"op":"render","path":"example.png"}
```

`render` is headless-only in M0. The GUI exports from its current validated preview.
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
EXR, OCIO, sequences/timeline, animation, roto, tracking, 3D or model execution yet.
Retained cache bytes are bounded; peak working memory is not. Source dimensions
are capped at 8192 each, but large images can still exhaust memory. Cancellation
is between nodes; decode/individual kernels/export are not interruptible. PNG
export currently runs on the GUI thread. Merge requires matching dimensions;
Transform is integer translation with fixed bounds. UI layout is not persisted.

The full product contract and phased acceptance gates are in
[VISION](docs/VISION.md); engineering boundaries are in
[ARCHITECTURE](docs/ARCHITECTURE.md). Project handoff lives in [TASKLOG](TASKLOG.md)
and [current state](context/state.md).
