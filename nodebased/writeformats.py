"""The still-image formats Write adds beside EXR and PNG: TIFF, JPEG and DPX, with Nuke's per-format options.

OpenImageIO writes the files. What this module owns is what goes into them:

* the choices and defaults of each format's knobs, under Nuke's spelling (`datatype`, `compression`, `quality`,
  `sub-sampling`, `fill`, `big endian`, `transfer`; the Write node keeps one param per format so the names never clash);
* the colour a file holds: the Write's output transform and premultiplied handling (nodebased/writenode.py
  `to_output`) give straight colour for TIFF and DPX, colour over black for JPEG, which has no alpha;
* the quantising: integer files are clamped to 0..1 first (so overbright and negative values cannot wrap) and then
  rounded to the nearest code value by OpenImageIO. A 32-bit float TIFF is written as it is, unclamped.

The files are written to a temporary name and renamed into place, as EXR is, so a failed write never leaves half a frame.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# Format name -> file name extensions it answers to; the first is the one an explicit choice writes.
EXTENSIONS = {"exr": ("exr",), "png": ("png",), "tiff": ("tif", "tiff"), "jpeg": ("jpg", "jpeg"), "dpx": ("dpx",)}
FORMATS = tuple(EXTENSIONS)
STILL_FORMATS = ("tiff", "jpeg", "dpx")

TIFF_DATATYPES = ("8 bit", "16 bit", "32 bit float")
TIFF_COMPRESSIONS = ("none", "PackBits", "LZW", "Deflate")
_TIFF_OIIO = {"none": "none", "PackBits": "packbits", "LZW": "lzw", "Deflate": "zip"}
JPEG_SUB_SAMPLINGS = ("4:1:1", "4:2:2", "4:4:4")
DPX_DATATYPES = ("8 bit", "10 bit", "12 bit", "16 bit")
# The Transfer header names OpenImageIO's DPX writer accepts (a name it does not know is stored as "Undefined").
DPX_TRANSFERS = ("(auto detect)", "User defined", "Printing density", "Linear", "Logarithmic", "Unspecified video",
                 "SMPTE 274M", "ITU-R 709-4", "ITU-R 601-5 system B or G", "ITU-R 601-5 system M", "NTSC composite video",
                 "PAL composite video", "Z depth linear", "Z depth homogeneous", "ADX")
# What "(auto detect)" writes for each output transform, the way Nuke sets the header from the LUT it used.
_AUTO_TRANSFER = {"ACEScg": "Linear", "ACES2065-1": "Linear", "Linear Rec.709": "Linear", "sRGB": "Unspecified video",
                  "Raw": "User defined"}

DEFAULTS = {
    "tiff_datatype": "8 bit", "tiff_compression": "Deflate",
    "jpeg_quality": 0.75, "jpeg_sub_sampling": "4:1:1",
    "dpx_datatype": "10 bit", "dpx_fill": 0, "dpx_big_endian": 1, "dpx_transfer": "(auto detect)",
}
CHOICES = {"tiff_datatype": TIFF_DATATYPES, "tiff_compression": TIFF_COMPRESSIONS, "jpeg_sub_sampling": JPEG_SUB_SAMPLINGS,
           "dpx_datatype": DPX_DATATYPES, "dpx_transfer": DPX_TRANSFERS}
# Which params each format shows in Properties; the EXR ones are listed in nodebased/writenode.py.
PARAMS = {"tiff": ("tiff_datatype", "tiff_compression"), "jpeg": ("jpeg_quality", "jpeg_sub_sampling"),
          "dpx": ("dpx_datatype", "dpx_fill", "dpx_big_endian", "dpx_transfer")}


def format_of_suffix(suffix):
    """The format name a file extension (with or without the dot, any case) belongs to, or None."""
    word = str(suffix).lower().lstrip(".")
    return next((name for name, suffixes in EXTENSIONS.items() if word in suffixes), None)


def bits_of(text):
    """The bit count in a datatype choice such as '10 bit' or '32 bit float'."""
    return int(str(text).split()[0])


def describe(file_type, params, bits="half"):
    """A short plain description of what a Write of `file_type` writes, for the status bar."""
    if file_type == "exr":
        return f"exr · {bits} float"
    if file_type == "tiff":
        return f"tiff · {params.get('tiff_datatype', DEFAULTS['tiff_datatype'])} · {params.get('tiff_compression', DEFAULTS['tiff_compression'])}"
    if file_type == "jpeg":
        return f"jpeg · quality {float(params.get('jpeg_quality', DEFAULTS['jpeg_quality'])):g}"
    if file_type == "dpx":
        return f"dpx · {params.get('dpx_datatype', DEFAULTS['dpx_datatype'])}"
    return f"{file_type} · 8-bit sRGB"


def _check_choice(params, key):
    value = params.get(key, DEFAULTS[key])
    if value not in CHOICES[key]:
        raise ValueError(f"Write: {key} must be one of {', '.join(CHOICES[key])}, got {value!r}")
    return value


def prepare(frame, letters, space, premultiplied, file_type, datatype):
    """The array a still file holds: float32, straight colour (JPEG: colour over black) with alpha last when `A` is chosen,
    converted by the output transform `space`. Integer datatypes are clamped to 0..1; 32-bit float is left as it is."""
    from .writenode import straight_output
    colour, alpha = straight_output(frame, space, premultiplied)
    if file_type == "jpeg" and not premultiplied:
        colour = colour * alpha                    # no alpha channel in the file: colour over black, as the viewer shows it
    for index, letter in enumerate("RGB"):
        if letter not in letters:
            colour[..., index] = 0.0
    with_alpha = "A" in letters and file_type != "jpeg"
    out = np.concatenate((colour, alpha), axis=2) if with_alpha else colour
    if datatype != "32 bit float":
        out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out, dtype=np.float32)


def write_still(path, frame, file_type, params, letters, space, premultiplied):
    """Write the evaluated `frame` (HxWx4, premultiplied working space) to `path` as TIFF, JPEG or DPX."""
    import OpenImageIO as oiio
    path = Path(path).expanduser().resolve()
    suffix_format = format_of_suffix(path.suffix)
    if suffix_format != file_type:
        raise ValueError(f"{file_type.upper()} output path must end in .{EXTENSIONS[file_type][0]}")
    if file_type == "tiff":
        datatype = _check_choice(params, "tiff_datatype")
        compression = _TIFF_OIIO[_check_choice(params, "tiff_compression")]
    elif file_type == "dpx":
        datatype = _check_choice(params, "dpx_datatype")
        transfer = _check_choice(params, "dpx_transfer")
    else:
        datatype = "8 bit"
        subsampling = _check_choice(params, "jpeg_sub_sampling")
        quality = float(params.get("jpeg_quality", DEFAULTS["jpeg_quality"]))
    pixels = prepare(frame, letters, space, premultiplied, file_type, datatype)
    bits = bits_of(datatype)
    storage = "float" if datatype == "32 bit float" else ("uint8" if bits == 8 else "uint16")
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, oiio.TypeDesc(storage))
    spec.channelnames = list("RGBA"[:channels])
    if channels == 4:
        spec.alpha_channel = 3
    if file_type == "tiff":
        spec.attribute("compression", compression)
    elif file_type == "jpeg":
        spec.attribute("CompressionQuality", int(round(max(0.0, min(1.0, quality)) * 100)))
        spec.attribute("jpeg:subsampling", subsampling)
    else:
        spec.attribute("oiio:BitsPerSample", bits)
        spec.attribute("oiio:Endian", "big" if int(params.get("dpx_big_endian", 1)) else "little")
        spec.attribute("dpx:Packing", "Packed" if int(params.get("dpx_fill", 0)) else "Filled, method A")
        spec.attribute("dpx:Transfer", _AUTO_TRANSFER.get(space, "User defined") if transfer == "(auto detect)" else transfer)
    fd, temporary = tempfile.mkstemp(prefix="." + path.stem, suffix=path.suffix, dir=path.parent)
    os.close(fd)
    writer = None
    try:
        writer = oiio.ImageOutput.create(temporary)
        if writer is None or not writer.open(temporary, spec):
            raise ValueError(f"Cannot open {file_type.upper()} output: " + oiio.geterror())
        if not writer.write_image(pixels):
            raise ValueError(f"Cannot write {file_type.upper()}: " + writer.geterror())
        if not writer.close():
            raise ValueError(f"Cannot finalize {file_type.upper()}: " + writer.geterror())
        writer = None
        os.replace(temporary, path)
    finally:
        if writer:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
