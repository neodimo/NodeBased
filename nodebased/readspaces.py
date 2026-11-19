"""The project's default colour spaces by file type: Root's int8LUT, int16LUT, logLUT and floatLUT (Reference Guide, Root, Color tab).

A Read whose colorspace is left at Auto takes the colour space its file's kind has in the project settings
(`settings.project`: `int8_lut`, `int16_lut`, `log_lut`, `float_lut`). The kinds, as Nuke sorts them:

- log: a Cineon or DPX file (`.cin`, `.dpx`), read with `log_lut` (Cineon by default);
- float: an OpenEXR file, read with `float_lut` (linear Rec.709 by default; Nuke's `linear`);
- 16-bit: any other file with more than eight bits per channel, read with `int16_lut` (sRGB by default);
- 8-bit: every other file, read with `int8_lut` (sRGB by default).

A file that names its own colour space other than sRGB (an `oiio:ColorSpace` tag NodeBased knows, such as ACEScg) keeps it, as before;
an sRGB tag decides nothing, because OpenImageIO puts it on almost every PNG and JPEG it opens. NodeBased has always read a
floating-point file that is not an EXR (a float TIFF) as sRGB like any other non-EXR picture; it stays in the 16-bit class so a
document that holds it keeps its pixels.

How the choice reaches the decoder: the evaluator and the tile executor replace a Read's `Auto` by `Auto|<int8>|<int16>|<log>|<float>`
(`token`) when the project's four spaces are not the ones Read always used, so the cache key follows the project, and
`nodebased/media.py` resolves that string with `resolve`. A project that holds the old spaces leaves the string `Auto`, byte for byte
what a document saved before this step sent.
"""
from . import projectsettings as _project

FIELDS = ("int8_lut", "int16_lut", "log_lut", "float_lut")
# What Auto always meant: sRGB for any picture, linear Rec.709 for an EXR (and sRGB for a log file, which Read took as any other picture).
LEGACY = ("sRGB", "sRGB", "sRGB", "Linear Rec.709")
LOG_EXTENSIONS = (".cin", ".dpx")
PREFIX = "Auto|"


def token(document):
    """The colorspace string a Read at Auto carries for this document: plain `Auto` when the project keeps the old spaces."""
    fields = _project.project(document)
    values = tuple(fields[name] for name in FIELDS)
    return "Auto" if values == LEGACY else PREFIX + "|".join(values)


def parse(colorspace):
    """The four spaces in an `Auto|...` string as {field: space}, or None for a plain colorspace (Auto included)."""
    if not isinstance(colorspace, str) or not colorspace.startswith(PREFIX):
        return None
    values = colorspace[len(PREFIX):].split("|")
    return dict(zip(FIELDS, values)) if len(values) == len(FIELDS) else None


def file_class(ext, spec):
    """'log', 'float', 'int16' or 'int8' for one decoded file (`spec` is its OpenImageIO ImageSpec)."""
    if ext in LOG_EXTENSIONS:
        return "log"
    if ext == ".exr":
        return "float"
    return "int8" if str(spec.format) in ("uint8", "int8") else "int16"


def resolve(ext, spec, spaces):
    """The colour space of a file at Auto: the project's space for its kind."""
    return spaces[file_class(ext, spec) + "_lut"]
