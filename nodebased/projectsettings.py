"""The project settings Nuke keeps on the Root node that a NodeBased document can hold: `settings.project`.

The group is optional. A document that never set a field has no `project` key, and one whose fields are all at
their defaults is stored without it, so old and untouched documents keep their exact serialized form (the
`formats` registry follows the same rule). Each field acts:

- `name`, `label`: the project's name and comment, shown in the window title and the Project Settings dialog.
- `project_directory`: the folder a Read's relative path is resolved against when the path is set (Nuke's
  [value root.project_directory]); an absolute path is left alone, and with no directory a relative path
  resolves against the working directory as before.
- `lock_connections`: no node's input can be rewired while it is on (Nuke's lock all connections).
- `lock_range`: loading a sequence leaves the project frame range where it is (Nuke's lock range).
- `script_directory`: the folder the document lives in; a Read's relative path resolves against it when the project
  directory is empty (Nuke's [python nuke.script_directory()] is the usual project directory expression).
- `format`: the project format, a name in `settings.formats`. A Constant, a Draw generator or a Reformat created without a
  size of its own takes that window; with no project format each node keeps its own default size. It is also the full
  size the proxy format is measured against.
- `proxy_type`, `proxy_format`: with the type on `format` the viewer's proxy tier is the one nearest to the project
  format's width over the proxy format's width (`proxy_tier`), where the type `scale` keeps the tier the artist picked.
- `read_proxy_files`: when a Read with a proxy file reads it (Nuke's `proxySetting`): never; if larger (the proxy file
  is at least the size the picture needs); if nearest (the file nearer that size wins); always. The default is `always`,
  which is how Read has always behaved with a proxy file.
"""
import math
from pathlib import Path

DEFAULTS = {"name": "", "label": "", "project_directory": "", "lock_connections": False, "lock_range": False,
            "script_directory": "", "format": "", "proxy_type": "scale", "proxy_format": "", "read_proxy_files": "always"}
TEXT_FIELDS = ("name", "label", "project_directory", "script_directory", "format", "proxy_format")
FLAG_FIELDS = ("lock_connections", "lock_range")
CHOICE_FIELDS = {"proxy_type": ("scale", "format"), "read_proxy_files": ("never", "if larger", "if nearest", "always")}
# The full size a proxy format is measured against when the project has no format of its own.
FALLBACK_FORMAT = "HD_1080"


def project(document):
    """The project fields with the defaults filled in for any the document does not hold."""
    return {**DEFAULTS, **(document.get("settings", {}).get("project") or {})}


def validate(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError(f"settings.project fields must be among {sorted(DEFAULTS)}")
    for name, value in values.items():
        if name in TEXT_FIELDS and not isinstance(value, str):
            raise ValueError(f"settings.project.{name} must be a string")
        if name in FLAG_FIELDS and type(value) is not bool:
            raise ValueError(f"settings.project.{name} must be true or false")
        if name in CHOICE_FIELDS and value not in CHOICE_FIELDS[name]:
            raise ValueError(f"settings.project.{name} must be one of {list(CHOICE_FIELDS[name])}")


def merge(settings, changes):
    """Apply `changes` to settings["project"]; the group is dropped again when every field is back at its default."""
    values = {**DEFAULTS, **settings.get("project", {}), **changes}
    validate(values)
    if values == DEFAULTS:
        settings.pop("project", None)
    else:
        settings["project"] = values
    return values


def resolve_media_path(document, value):
    """A Read's path as stored: absolute, a relative one taken from the project directory when one is set."""
    path = Path(value).expanduser()
    fields = project(document)
    directory = fields["project_directory"] or fields["script_directory"]
    if directory and not path.is_absolute():
        path = Path(directory).expanduser() / path
    return str(path.resolve())


def format_window(document, name):
    """(width, height, pixel_aspect) of the registry format `name`, or None for a name the registry does not hold."""
    from .core import document_formats
    entry = document_formats(document).get(name)
    return None if entry is None else (entry["width"], entry["height"], entry["pixel_aspect"])


def proxy_tier(document, tier):
    """The viewer tier the artist's pick `tier` stands for. With the proxy type on `scale` (or no proxy format named) it is
    the pick; on `format` any proxy pick means the proxy format, whose width against the project format's width gives the
    tier nearest in log scale (a proxy format that is not clearly smaller than the project format leaves the picture at
    full size)."""
    from .core import VIEWER_PROXY_TIERS
    fields = project(document)
    if tier == 1 or fields["proxy_type"] != "format":
        return tier
    proxy = format_window(document, fields["proxy_format"])
    full = format_window(document, fields["format"] or FALLBACK_FORMAT)
    if proxy is None or full is None:
        return tier
    ratio = full[0] / proxy[0]
    if ratio < 2 ** 0.5:
        return 1
    return min((t for t in VIEWER_PROXY_TIERS if t > 1), key=lambda t: abs(math.log2(t) - math.log2(ratio)))
