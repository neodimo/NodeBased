"""Nuke's Write-node knobs that change what is written and how: `channels`, the output transform (`colorspace`, `raw`),
`premultiplied`, `views`, the frame number written into the file name (`frame_mode` / `frame_expr`), `create_directories`,
the limit to a frame range (`use_limit` with `first` / `last`), the EXR `exr_compression` and, since fidelity step M, the
per-format options of Nuke's Write: TIFF, JPEG and DPX files (nodebased/writeformats.py) and the EXR options `autocrop`,
`metadata`, `interleave`, `first part`, `standard layer name format`, `write full layer names`, `truncate channel names` and the
DWAA / DWAB `compression level`.

Each default writes what Write always wrote: every channel and layer, EXR in the working space or PNG through the sRGB
display view, the document's own frame numbers, missing directories created, no range limit. `active(params)` is False for
those, and `write_raster` then hands the frame to the unchanged `write_exr` / `write_png`, so an old document writes the
same bytes.

The window (nodebased/app.py `render_write`) and the tests share this module, so a render in the interface and a render in a
test run the same code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import pathcontext, writeformats

# Fidelity step M: the EXR options of Nuke's Write. Every default writes the EXR that Write always wrote. `metadata` defaults
# to "all metadata except input/*", which is what Write always wrote; Nuke's own default, "default metadata", leaves out the
# exr/* keys as well. `standardlayernameformat` orders layer, view and channel in a name, and with the one view NodeBased has
# the two orders coincide, so it is stored and shown but changes no byte.
EXR_DEFAULTS = {"exr_autocrop": 0, "exr_metadata": "all metadata except input/*", "exr_interleave": "channels, layers and views",
                "exr_first_part": "", "exr_standard_layer_name_format": 0, "exr_write_full_layer_names": 0,
                "exr_truncate_channel_names": 0, "exr_dw_compression_level": 45.0}
METADATA_CHOICES = ("no metadata", "default metadata", "default metadata and exr/*", "all metadata except input/*", "all metadata")
INTERLEAVE_CHOICES = ("channels, layers and views", "channels and layers", "channels")
EXR_PARAMS = ("bit_depth", "exr_compression", *EXR_DEFAULTS)

DEFAULTS = {"channels": "all", "colorspace": "Auto", "premultiplied": 0, "raw": 0, "views": "main", "frame_mode": "expression",
            "frame_expr": "", "create_directories": 1, "first": 1, "last": 1, "use_limit": 0, "exr_compression": "zips",
            # A proxy destination is selected only while the document's viewer is at a proxy tier;
            # Read all lines requests a complete decode when a Write reads its own file back.
            "proxy": "", "read_all_lines": 0,
            # Fidelity step F: read the file back (`reading`, `on_error`, `reload`), the order Render all goes in
            # (`render_order`), and the context variables of the path (nodebased/pathcontext.py).
            "reading": 0, "on_error": "error", "reload": 0, "render_order": 1, **pathcontext.PARAMS,
            **EXR_DEFAULTS, **writeformats.DEFAULTS}
# These change what a Write shows or when it renders, not the bytes it writes.
NOT_BYTES = ("create_directories", "first", "last", "use_limit", "frame_mode", "frame_expr", "views", "proxy", "read_all_lines",
             "reading", "on_error", "reload", "render_order", *pathcontext.PARAMS,
             # TIFF, JPEG and DPX files are written by their own writer, never by the EXR / PNG path `active` decides; the EXR
             # options are read by `exr_options` instead.
             *writeformats.DEFAULTS, *EXR_DEFAULTS)
KEYS = tuple(DEFAULTS)
VIEW_CHOICES = ("main",)
# What each output transform is called inside an EXR's `oiio:ColorSpace` tag, so Read's Auto finds its way back.
EXR_TAGS = {"ACEScg": "ACEScg", "ACES2065-1": "aces2065-1", "Linear Rec.709": "lin_rec709_scene", "sRGB": "srgb", "Raw": "raw"}


def extras(params):
    return {key: params.get(key, default) for key, default in DEFAULTS.items()}


def active(params):
    """True when any Write knob here differs from the value that leaves Write as it was. The EXR options (`exr_options`) and
    the TIFF, JPEG and DPX knobs are not part of this question."""
    return any(params.get(key, default) != default for key, default in DEFAULTS.items() if key not in NOT_BYTES)


# --------------------------------------------------------------------------------------------------- frames
def target_of(params, proxy=False):
    """A Write's (path, file type, bit depth), or ValueError with the reason it cannot render: the path with its context keys filled
    in, the format taken from the extension for `Auto` or the extension made to agree with an explicit choice."""
    raw_path = params.get("proxy") if proxy and str(params.get("proxy", "")).strip() else params["path"]
    path = pathcontext.substitute(raw_path, params).strip()      # %{key} filled from the node's key / value pairs
    if not path:
        raise ValueError("Set an output path on the Write node first")
    chosen = params["file_type"]
    suffix = Path(path).suffix.lower().lstrip(".")
    named = writeformats.format_of_suffix(suffix)
    if chosen == "Auto":
        if named is None:
            raise ValueError(f"Cannot infer a format from {Path(path).name!r}; choose exr, png, tiff, jpeg or dpx, "
                             "or use that extension")
        chosen = named
    elif chosen not in writeformats.EXTENSIONS:
        raise ValueError(f"Write: unknown file type {chosen!r}")
    elif named != chosen:
        # Honour the explicit choice and make the filename agree with it, rather than writing PNG bytes into a file called .exr.
        path = str(Path(path).with_suffix("." + writeformats.EXTENSIONS[chosen][0]))
    return path, chosen, params["bit_depth"]


def frames_to_render(params, time, single):
    """The timeline frames a render covers: the current one, or the document range, limited to [first, last] when
    `use_limit` is on."""
    frames = [int(time["current"])] if single else list(range(int(time["first"]), int(time["last"]) + 1))
    if int(params.get("use_limit", 0)):
        low, high = sorted((int(params.get("first", 1)), int(params.get("last", 1))))
        frames = [frame for frame in frames if low <= frame <= high]
    return frames


def file_frame_number(params, frame, first_rendered):
    """The frame number written into the file name for timeline `frame`.

    expression: the value of the `frame` expression at that frame (blank: the frame itself); start at N: the first rendered
    frame is numbered N; offset: the frame plus N."""
    from .readknobs import _evaluate_frame
    mode = str(params.get("frame_mode", "expression"))
    text = str(params.get("frame_expr", "")).strip()
    if mode == "expression":
        value = _evaluate_frame(text, frame)
    elif mode == "start at":
        value = (_evaluate_frame(text, frame) if text else first_rendered) + (frame - first_rendered)
    elif mode == "offset":
        value = frame + (_evaluate_frame(text, frame) if text else 0.0)
    else:
        raise ValueError(f"Write: unknown frame mode {mode!r}")
    return int(round(value))


# ------------------------------------------------------------------------------------------------ read file
def first_rendered(params, time):
    frames = frames_to_render(params, time, False)
    return frames[0] if frames else int(time["first"])


def read_back_number(params, frame):
    """The number of the file a Write that is reading shows at timeline `frame`: the number it wrote into the name for that frame.
    The document's range arrives in `_first` and `_last` (the evaluator adds them)."""
    time = {"first": params.get("_first", frame), "last": params.get("_last", frame), "current": frame}
    return file_frame_number(params, frame, first_rendered(params, time))


def read_back(params, frame):
    """The Raster of the file a Write with `reading` on points at, for timeline `frame`: the same reader as Read, at the file number
    the Write wrote for that frame, with `on_error` (error, hold, black, checkerboard) for a frame that is not there."""
    from . import imaging
    path = str(params.get("path", "") or "").strip()
    if not path:
        raise ValueError("Write: Read file needs a file path")      # the evaluator passes the input on instead when none is set
    number = read_back_number(params, frame)
    space = "Raw" if int(params.get("raw", 0)) else str(params.get("colorspace", "Auto"))
    alpha_mode = "Premultiplied" if int(params.get("premultiplied", 0)) else "Auto"
    return imaging.read_image_raster(path, colorspace=space, alpha_mode=alpha_mode, frame=number,
                                     missing=str(params.get("on_error", "error")))


def render_order(document):
    """The ids of the document's Write nodes in the order Render all renders them: lowest `render_order` first, ties in the order
    the nodes were made."""
    writes = [(int(node["params"].get("render_order", 1)), index, key)
              for index, (key, node) in enumerate(document["nodes"].items())
              if node["type"] == "Write" and not node.get("disabled")]
    return [key for _order, _index, key in sorted(writes)]


def check_views(params):
    if str(params.get("views", "main")) not in VIEW_CHOICES:
        raise ValueError("Write: only the main view exists; stereo and multiview output is not supported")


# ------------------------------------------------------------------------------------------------- channels
def channel_plan(text):
    """(beauty channel letters in R, G, B, A order, wanted layer names or None for all layers) for a `channels` value."""
    from .ops2d_chrome import parse_channels
    word = str(text).strip().lower()
    if word == "all":
        return "RGBA", None
    chosen = parse_channels(text)
    letters = "".join(letter for letter, on in zip("RGBA", chosen.rgba) if on)
    return letters, tuple(chosen.layers)


def select_layers(layers, wanted):
    """The raster layers a `channels` value keeps: all of them for 'all', else those named (`depth` or `depth.Z`)."""
    if wanted is None:
        return layers
    bases = {name.split(".")[0] if name.split(".")[0] in layers else name for name in wanted}
    return {name: array for name, array in layers.items() if name in bases}


# -------------------------------------------------------------------------------------- output transform
def output_space(params, file_type):
    """The space the file holds: 'Raw' (no conversion), a named space, or 'Auto' resolved per file type: the working space
    for EXR and for a 32-bit float TIFF, sRGB for every other file."""
    if int(params.get("raw", 0)):
        return "Raw"
    space = str(params.get("colorspace", "Auto"))
    if space == "Auto":
        float_tiff = file_type == "tiff" and params.get("tiff_datatype", writeformats.DEFAULTS["tiff_datatype"]) == "32 bit float"
        return "ACEScg" if file_type == "exr" or float_tiff else "sRGB"
    return space


def to_output(rgba, space, premultiplied, straight_file):
    """Working-space premultiplied `rgba` converted into `space`.

    The transform runs on straight colour (colour divided by alpha, multiplied back after) unless `premultiplied` says to
    transform the colour as it is; `straight_file` says whether the file stores straight colour (PNG) or associated (EXR)."""
    from .color import INPUT_SPACES, WORKING, processor
    out = np.array(rgba, dtype=np.float32, copy=True, order="C")
    alpha = out[..., 3:4]
    use_alpha = not premultiplied and bool(np.any(alpha != 1.0))
    safe = np.where(np.abs(alpha) > 1e-8, alpha, 1).astype(np.float32)
    if use_alpha:
        out[..., :3] /= safe
    if space not in ("Raw",) and INPUT_SPACES.get(space) != WORKING:
        processor(WORKING, INPUT_SPACES[space]).applyRGBA(out)
    if use_alpha and not straight_file:
        out[..., :3] *= safe                      # an associated file keeps premultiplied colour
    return out


# ----------------------------------------------------------------------------------------------- writing
def write_raster(raster, target, file_type, bits, params):
    """Write one evaluated Raster to `target` as the Write node's knobs say. Returns the number of channels in the beauty."""
    from . import imaging, media
    check_views(params)
    pixels = raster.to_display()
    if file_type in writeformats.STILL_FORMATS:
        letters, _wanted = channel_plan(params.get("channels", "all"))
        writeformats.write_still(target, pixels, file_type, params, letters, output_space(params, file_type),
                                 bool(int(params.get("premultiplied", 0))))
        return len(letters.replace("A", "")) + (1 if "A" in letters and file_type != "jpeg" else 0)
    layers = media.raster_layer_arrays(raster)
    options = exr_options(params, raster.meta) if file_type == "exr" else None
    if not active(params) and options is None:
        if file_type == "exr":
            media.write_exr(target, pixels, bits=bits, layers=layers, metadata=raster.meta,
                            compression=str(params.get("exr_compression", "zips")))
        else:
            imaging.write_png(target, pixels)
        return 4
    letters, wanted = channel_plan(params.get("channels", "all"))
    space = output_space(params, file_type)
    premultiplied = bool(int(params.get("premultiplied", 0)))
    if file_type == "exr":
        converted = pixels if space in ("ACEScg", "Raw") else to_output(pixels, space, premultiplied, straight_file=False)
        media.write_exr(target, converted, bits=bits, layers=select_layers(layers, wanted), metadata=raster.meta,
                        compression=str(params.get("exr_compression", "zips")), beauty=letters,
                        colorspace_tag=EXR_TAGS[space], options=options)
    else:
        write_png_channels(target, pixels, letters, space, premultiplied)
    return len(letters)


# ---------------------------------------------------------------------------------------------- EXR options
def exr_attributes(meta, choice):
    """(header attributes, write the typed timecode) for a `metadata` choice of the EXR options.

    The stream's metadata keys are `input/*` (where the pixels came from), `exr/*` (header attributes an EXR Read found), the
    `timecode`, and any other `a/b` key a metadata node or another tool made. `no metadata` writes none of them; `default
    metadata` the timecode and the other keys; `default metadata and exr/*` those and the exr/* keys; `all metadata except
    input/*` is the same set (NodeBased has no other prefix), and `all metadata` adds the input/* keys, under their own names.
    A Cryptomatte manifest (`cryptomatte/...`) is part of the layers it describes and is written whatever the choice."""
    from .metadata import TIMECODE_KEY, key_to_header
    if choice not in METADATA_CHOICES:
        raise ValueError(f"Write: metadata must be one of {', '.join(METADATA_CHOICES)}, got {choice!r}")
    attributes = {}
    for key, value in (meta or {}).items():
        if key.startswith("cryptomatte/"):
            wanted = True
        elif key.startswith("input/"):
            wanted = choice == "all metadata"
        elif key.startswith("exr/"):
            wanted = choice in ("default metadata and exr/*", "all metadata except input/*", "all metadata")
        else:
            wanted = choice != "no metadata"
        if not wanted:
            continue
        name = key_to_header(key) or (key if key.startswith("input/") and ":" not in key else None)
        if name is not None:
            attributes[name] = str(value)
    return attributes, choice != "no metadata" and TIMECODE_KEY in (meta or {})


def exr_options(params, meta):
    """The `options` for `media.write_exr` that the Write's EXR knobs ask for, or None when every one is at its default (the
    EXR is then written by the unchanged call)."""
    levels = {key: params.get(key, default) for key, default in EXR_DEFAULTS.items()}
    compression = str(params.get("exr_compression", "zips"))
    dwa = compression in ("dwaa", "dwab")
    if all(levels[key] == EXR_DEFAULTS[key] for key in ("exr_autocrop", "exr_metadata", "exr_interleave", "exr_first_part",
                                                        "exr_write_full_layer_names", "exr_truncate_channel_names")) and not dwa:
        return None
    interleave = levels["exr_interleave"]
    if interleave not in INTERLEAVE_CHOICES:
        raise ValueError(f"Write: interleave must be one of {', '.join(INTERLEAVE_CHOICES)}, got {interleave!r}")
    attributes, timecode = exr_attributes(meta, levels["exr_metadata"])
    options = {"autocrop": bool(int(levels["exr_autocrop"])), "attributes": attributes, "timecode": timecode,
               "parts": "layers" if interleave == "channels" else "single",
               "full_layer_names": bool(int(levels["exr_write_full_layer_names"])),
               "truncate_names": bool(int(levels["exr_truncate_channel_names"]))}
    if interleave == "channels" and str(params.get("channels", "all")).strip().lower() == "all":
        options["first_part"] = str(levels["exr_first_part"]).strip()
    if dwa:
        options["dw_level"] = float(levels["exr_dw_compression_level"])
    return options


# ------------------------------------------------------------------------------------- what Properties shows
def panel_format(params):
    """The file format the Write is set to write: its `file_type`, or for Auto the one its path's extension names. An Auto Write
    with no recognised extension shows the EXR knobs, as every Write did before TIFF, JPEG and DPX."""
    chosen = params.get("file_type", "Auto")
    if chosen in writeformats.EXTENSIONS:
        return chosen
    suffix = Path(str(params.get("path", "") or "")).suffix
    return writeformats.format_of_suffix(suffix) or "exr"


def hidden_params(params):
    """The params Properties leaves out because they belong to a file format the Write is not set to write."""
    shown = panel_format(params)
    hidden = set()
    if shown != "exr":
        hidden.update(EXR_PARAMS)
    for name, keys in writeformats.PARAMS.items():
        if name != shown:
            hidden.update(keys)
    return hidden


def straight_output(frame, space, premultiplied):
    """(colour, alpha) of a premultiplied working-space `frame` as an output file holds them: colour converted into `space`
    (straight, unless `premultiplied` asks for the colour as it is), alpha clamped to 0..1. Colour is not clamped here."""
    from .color import display_rgb
    alpha = np.clip(frame[..., 3:4], 0, 1)
    if space == "sRGB" and not premultiplied:
        straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
        return np.array(display_rgb(straight, "sRGB"), dtype=np.float32), alpha
    return np.array(to_output(frame, space, premultiplied, straight_file=True)[..., :3], dtype=np.float32), alpha


def write_png_channels(path, frame, letters, space, premultiplied):
    """8-bit PNG of the selected colour channels (unselected colour reads 0), with alpha only when A is selected."""
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QSaveFile, QIODevice
    colour, alpha = straight_output(frame, space, premultiplied)
    colour = np.clip(colour, 0, 1)
    for index, letter in enumerate("RGB"):
        if letter not in letters:
            colour[..., index] = 0.0
    with_alpha = "A" in letters
    if with_alpha:
        rgba8 = (np.concatenate((colour, alpha), axis=2) * 255 + 0.5).astype(np.uint8)
        fmt = QImage.Format.Format_RGBA8888
        data = np.ascontiguousarray(rgba8)
    else:
        data = np.ascontiguousarray((colour * 255 + 0.5).astype(np.uint8))
        fmt = QImage.Format.Format_RGB888
    image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], fmt).copy()
    target = QSaveFile(str(path))
    if not target.open(QIODevice.OpenModeFlag.WriteOnly):
        raise ValueError(target.errorString())
    if not image.save(target, "PNG") or not target.commit():
        raise ValueError(f"Cannot export PNG: {target.errorString()}")
