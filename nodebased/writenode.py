"""Nuke's Write-node knobs that change what is written and how: `channels`, the output transform (`colorspace`, `raw`),
`premultiplied`, `views`, the frame number written into the file name (`frame_mode` / `frame_expr`), `create_directories`,
the limit to a frame range (`use_limit` with `first` / `last`) and the EXR `exr_compression`.

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

DEFAULTS = {"channels": "all", "colorspace": "Auto", "premultiplied": 0, "raw": 0, "views": "main", "frame_mode": "expression",
            "frame_expr": "", "create_directories": 1, "first": 1, "last": 1, "use_limit": 0, "exr_compression": "zips"}
KEYS = tuple(DEFAULTS)
VIEW_CHOICES = ("main",)
# What each output transform is called inside an EXR's `oiio:ColorSpace` tag, so Read's Auto finds its way back.
EXR_TAGS = {"ACEScg": "ACEScg", "ACES2065-1": "aces2065-1", "Linear Rec.709": "lin_rec709_scene", "sRGB": "srgb", "Raw": "raw"}


def extras(params):
    return {key: params.get(key, default) for key, default in DEFAULTS.items()}


def active(params):
    """True when any Write knob here differs from the value that leaves Write as it was."""
    return any(params.get(key, default) != default for key, default in DEFAULTS.items()
               if key not in ("create_directories", "first", "last", "use_limit", "frame_mode", "frame_expr", "views"))


# --------------------------------------------------------------------------------------------------- frames
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
    """The space the file holds: 'Raw' (no conversion), a named space, or 'Auto' resolved per file type."""
    if int(params.get("raw", 0)):
        return "Raw"
    space = str(params.get("colorspace", "Auto"))
    if space == "Auto":
        return "ACEScg" if file_type == "exr" else "sRGB"
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
    layers = media.raster_layer_arrays(raster)
    if not active(params):
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
                        colorspace_tag=EXR_TAGS[space])
    else:
        write_png_channels(target, pixels, letters, space, premultiplied)
    return len(letters)


def write_png_channels(path, frame, letters, space, premultiplied):
    """8-bit PNG of the selected colour channels (unselected colour reads 0), with alpha only when A is selected."""
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QSaveFile, QIODevice
    from .color import display_rgb
    alpha = np.clip(frame[..., 3:4], 0, 1)
    if space == "sRGB" and not premultiplied:
        straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
        colour = np.clip(display_rgb(straight, "sRGB"), 0, 1)
    else:
        colour = np.clip(to_output(frame, space, premultiplied, straight_file=True)[..., :3], 0, 1)
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
