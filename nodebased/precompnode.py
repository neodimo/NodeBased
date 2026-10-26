"""Nuke's Precomp knobs of fidelity step H: `use_output`, `reading` (Nuke's "read file for output"), `missing_frames` (Nuke's
`on_error`), `reload_write`, and the Open and Render buttons.

A Precomp sources its picture from another saved document (`file`). Which node of that document it takes the picture from:

* `use_output` off (the default of a new node): the document's own view node, as Nuke's Precomp takes the script's output node.
  On: the node `output_node` names. A document saved before this knob existed keeps its output node: `upgrade_use_output` turns
  the switch on for a Precomp that names one.
* `reading` on: the picture is the file the referenced document's Write node wrote (the Write the output node is, or else the
  first enabled Write in render order), read back at the file number that Write used for the frame, as a Write's own "read file"
  does (nodebased/writenode.py), instead of the document being evaluated.
* `missing_frames`: what a frame that file does not have becomes: `error`, `black`, `checkerboard`, or `read input`, which
  evaluates the referenced document instead, as if `reading` were off for that frame.
* `reload_write`: a counter the Reload button beside Render bumps, so a changed or replaced file is read again (it is part of the
  node's cache key). `reload` is the counter of the Reload button beside the file.

The buttons: Open opens the referenced document in the window (`Window.open_precomp`); Render writes the referenced document's Write
node over its frame range (`render`, below, which the window's button runs), and Reload bumps the counters.
"""
from __future__ import annotations

from pathlib import Path

from . import pathcontext, writenode

PARAMS = {"use_output": 0, "reading": 0, "missing_frames": "error", "reload_write": 0}
MISSING_FRAMES = ("error", "black", "checkerboard", "read input")
KINDS = ("Precomp",)


def new_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def upgrade_use_output(params):
    """A Precomp saved before `use_output` existed took the output node it named: leave that on."""
    if "use_output" not in params:
        params["use_output"] = 1 if params.get("output_node") else 0


# ------------------------------------------------------------------------------------------------ the document
def load(params):
    """The referenced document and its path. Raises the errors a Precomp has always raised."""
    from .core import load_document
    path = params.get("file", "")
    if not path:
        raise ValueError("Precomp: choose a referenced document")
    try:
        return load_document(path), path
    except OSError:
        raise ValueError(f"Precomp: file not found: {path}") from None
    except ValueError as error:
        raise ValueError(f"Precomp: {path}: {error}") from None


def output_id(params, document, path):
    """The node of the referenced document the picture comes from."""
    chosen = (params.get("output_node") or "") if int(params.get("use_output", 0)) else ""
    found = chosen or document.get("view")
    if not found or found not in document["nodes"]:
        raise ValueError(f"Precomp: no output node {found!r} in {path}")
    return found


def reference_frame(params, frame):
    return (frame or 1) + int(params.get("frame_offset", 0))


# -------------------------------------------------------------------------------------------------- read file
def write_node(document, out_id, path):
    """The Write node `reading` and Render work with."""
    if document["nodes"][out_id]["type"] == "Write":
        return out_id
    order = writenode.render_order(document)
    if not order:
        raise ValueError(f"Precomp: there is no enabled Write node in {path}")
    return order[0]


def file_params(document, key):
    """A Write's parameters with its path filled in from its own context keys, and the document's range for the file number."""
    params = dict(document["nodes"][key]["params"])
    params["path"] = pathcontext.substitute(params.get("path", ""), params).strip()
    params["_first"], params["_last"] = int(document["time"]["first"]), int(document["time"]["last"])
    return params


def read_file(params, document, out_id, path, ref_frame):
    """The Raster of the referenced Write's file for `ref_frame`, or None when `missing_frames` is `read input` and the file is not
    there. Raises when there is nothing to read."""
    from .media import resolve_source_path
    key = write_node(document, out_id, path)
    write = file_params(document, key)
    if not write["path"]:
        raise ValueError(f"Precomp: the Write node {key!r} in {path} has no file to read")
    policy = str(params.get("missing_frames", "error"))
    if policy == "read input":
        number = writenode.read_back_number(write, ref_frame)
        try:
            resolve_source_path(write["path"], number, "error")
        except ValueError:
            return None
        write["on_error"] = "error"
    else:
        write["on_error"] = policy
    return writenode.read_back(write, ref_frame)


# ----------------------------------------------------------------------------------------------------- render
def render(params, evaluator, cancel=None, progress=None):
    """Write the referenced document's Write node(s) over its frame range: the Write the output node is, else every enabled Write in
    render order. Returns the file paths written. `progress(done, total)` is called after each frame."""
    from .cancellation import Cancelled
    from .media import is_sequence, sequence_path
    document, path = load(params)
    out_id = output_id(params, document, path)
    keys = [out_id] if document["nodes"][out_id]["type"] == "Write" else writenode.render_order(document)
    if not keys:
        raise ValueError(f"Precomp: there is no enabled Write node in {path} to render")
    plan = []
    for key in keys:
        write = dict(document["nodes"][key]["params"])
        writenode.check_views(write)
        target, file_type, bits = writenode.target_of(write)
        frames = writenode.frames_to_render(write, document["time"], False)
        if len(frames) > 1 and not is_sequence(target):
            raise ValueError(f"Precomp: the Write {key!r} in {path} writes the single file {Path(target).name!r} for "
                             f"{len(frames)} frames; use a padded pattern such as render.%04d.exr")
        plan.append((key, write, target, file_type, bits, frames))
    total = sum(len(item[5]) for item in plan)
    written = []
    for key, write, target, file_type, bits, frames in plan:
        if int(write.get("create_directories", 1)):
            Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            raster = evaluator.evaluate_raster(document, key, cancel=cancel, frame=frame, tier=1)
            number = writenode.file_frame_number(write, frame, frames[0])
            name = sequence_path(target, number) if is_sequence(target) else target
            writenode.write_raster(raster, name, file_type, bits, write)
            written.append(name)
            if progress is not None:
                progress(len(written), total)
    return written
